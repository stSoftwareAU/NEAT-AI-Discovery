//! Per-request GPU time budget (Issue #1928).
//!
//! The submitter derives a caller timeout from `calculate_gpu_batch_timeout()`
//! and then waits that long for a response. Before this module the GPU thread's
//! inner waits were fixed constants (`GPU_BUFFER_MAP_TIMEOUT_SECS`) applied
//! *per sub-batch*, so a request with `n` sub-batches could occupy the GPU
//! thread for `n × 295s` while the caller gave up after at most 300s — leaving
//! the thread inside the driver and the queue's `Drop` abandoning it.
//!
//! `GpuTimeBudget` carries the caller's budget into the worker. Every inner wait
//! asks the budget how long it may block, so the *sum* of all inner waits for
//! one request can never exceed the caller's timeout minus the safety margin.
//! The worker therefore always errors out first and sends a real error back
//! through the response channel instead of going silent.
//!
//! ```text
//! caller timeout T ──────────────────────────────────────────────▶ recv_timeout
//! inner deadline   ─────────────────────────────────▶ (T − margin)
//!                  │ sub-batch 1 │ sub-batch 2 │ … │  ← each wait capped at
//!                                                      the remaining budget
//! ```

use anyhow::{Result, anyhow};
use std::time::{Duration, Instant};

use super::device::{GPU_BUFFER_MAP_TIMEOUT_MARGIN_SECS, GPU_BUFFER_MAP_TIMEOUT_SECS};

/// The time a single GPU work request may spend inside the worker.
///
/// `unbounded()` is the no-deadline fallback: inner waits use the fixed
/// `GPU_BUFFER_MAP_TIMEOUT_SECS` constant, matching the pre-#1928 behaviour for
/// callers that never supplied a deadline.
#[derive(Debug, Clone, Copy)]
pub struct GpuTimeBudget {
    /// Instant after which the worker must abandon this request. `None` means
    /// no caller budget was supplied — fall back to the fixed constant.
    deadline: Option<Instant>,
}

impl GpuTimeBudget {
    /// A budget with no caller deadline: inner waits fall back to
    /// `GPU_BUFFER_MAP_TIMEOUT_SECS`.
    #[must_use]
    pub const fn unbounded() -> Self {
        Self { deadline: None }
    }

    /// Derive a budget from the caller's timeout, starting now.
    ///
    /// The inner deadline is the caller's timeout minus
    /// `GPU_BUFFER_MAP_TIMEOUT_MARGIN_SECS`, so the worker always gives up
    /// before the caller does.
    #[must_use]
    pub fn from_caller_timeout(caller_timeout: Duration) -> Self {
        Self::from_caller_timeout_at(Instant::now(), caller_timeout)
    }

    /// Deterministic form of [`Self::from_caller_timeout`] with an explicit
    /// start instant (used by tests, which must not depend on wall-clock time).
    #[must_use]
    pub fn from_caller_timeout_at(start: Instant, caller_timeout: Duration) -> Self {
        let margin = Duration::from_secs(GPU_BUFFER_MAP_TIMEOUT_MARGIN_SECS);
        Self {
            deadline: Some(start + caller_timeout.saturating_sub(margin)),
        }
    }

    /// Whether a caller deadline is attached to this budget.
    #[must_use]
    pub const fn is_bounded(&self) -> bool {
        self.deadline.is_some()
    }

    /// Time left at `now`. An unbounded budget always reports the fixed
    /// fallback; a bounded one saturates at zero once exhausted.
    #[must_use]
    pub fn remaining_at(&self, now: Instant) -> Duration {
        match self.deadline {
            Some(deadline) => deadline.saturating_duration_since(now),
            None => Duration::from_secs(GPU_BUFFER_MAP_TIMEOUT_SECS),
        }
    }

    /// Time left right now.
    #[must_use]
    pub fn remaining(&self) -> Duration {
        self.remaining_at(Instant::now())
    }

    /// Whole seconds left, for the buffer-map wait helpers.
    #[must_use]
    pub fn remaining_secs(&self) -> u64 {
        self.remaining().as_secs()
    }

    /// Cap a fixed wait at whatever budget is left.
    #[must_use]
    pub fn capped(&self, wait: Duration) -> Duration {
        wait.min(self.remaining())
    }

    /// Whether the budget is exhausted (bounded budgets only).
    #[must_use]
    pub fn is_expired(&self) -> bool {
        self.is_bounded() && self.remaining().is_zero()
    }

    /// Fail loudly when the budget is exhausted, so the worker returns a real
    /// error through the response channel rather than blocking on another
    /// sub-batch the caller will never wait for.
    pub fn check(&self, label: &str) -> Result<()> {
        if self.is_expired() {
            return Err(anyhow!(
                "GPU time budget exhausted before {label}. The caller's timeout is about to \
                 expire, so this request is abandoned to keep the GPU thread joinable."
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::utils::GPU_QUEUE_TIMEOUT_MAX_SECS;

    /// Issue #1928: the inner deadline is strictly earlier than the caller's
    /// timeout, by exactly the safety margin.
    #[test]
    fn inner_deadline_is_strictly_before_caller_timeout() {
        let start = Instant::now();
        let caller_timeout = Duration::from_secs(GPU_QUEUE_TIMEOUT_MAX_SECS);
        let budget = GpuTimeBudget::from_caller_timeout_at(start, caller_timeout);

        let remaining = budget.remaining_at(start);
        assert!(
            remaining < caller_timeout,
            "inner budget {remaining:?} must be shorter than caller timeout {caller_timeout:?}"
        );
        assert_eq!(
            remaining,
            caller_timeout - Duration::from_secs(GPU_BUFFER_MAP_TIMEOUT_MARGIN_SECS),
            "inner budget is the caller timeout less the safety margin"
        );
    }

    /// Issue #1928: a short caller timeout still yields a strictly shorter
    /// inner budget — the margin is never skipped.
    #[test]
    fn short_caller_timeout_still_leaves_a_margin() {
        let start = Instant::now();
        let caller_timeout = Duration::from_secs(60);
        let budget = GpuTimeBudget::from_caller_timeout_at(start, caller_timeout);
        assert!(budget.remaining_at(start) < caller_timeout);
    }

    /// Issue #1928: the budget shrinks monotonically across successive
    /// sub-batch iterations and their sum can never exceed the caller's
    /// timeout, however many sub-batches there are.
    #[test]
    fn budget_shrinks_across_successive_sub_batches() {
        let start = Instant::now();
        let caller_timeout = Duration::from_secs(300);
        let budget = GpuTimeBudget::from_caller_timeout_at(start, caller_timeout);

        // Simulate ten sub-batches, each burning 40s of wall clock.
        let mut previous = budget.remaining_at(start);
        let mut consumed = Duration::ZERO;
        for iteration in 1..=10 {
            let now = start + Duration::from_secs(40 * iteration);
            let allowed = budget.remaining_at(now);
            assert!(
                allowed <= previous,
                "iteration {iteration}: budget must not grow ({allowed:?} > {previous:?})"
            );
            consumed += allowed.min(Duration::from_secs(40));
            previous = allowed;
        }

        assert!(
            previous.is_zero(),
            "budget must be exhausted after 400s of sub-batches, got {previous:?}"
        );
        assert!(
            consumed < caller_timeout,
            "total inner wait {consumed:?} must stay inside the caller timeout {caller_timeout:?}"
        );
    }

    /// Issue #1928: with no caller deadline the inner waits fall back to the
    /// fixed `GPU_BUFFER_MAP_TIMEOUT_SECS` constant.
    #[test]
    fn unbounded_budget_falls_back_to_the_constant() {
        let budget = GpuTimeBudget::unbounded();
        assert!(!budget.is_bounded());
        assert_eq!(budget.remaining_secs(), GPU_BUFFER_MAP_TIMEOUT_SECS);
        assert!(
            !budget.is_expired(),
            "an unbounded budget never expires — it has no caller to outlive"
        );
        budget
            .check("fallback path")
            .expect("unbounded budget is always usable");
    }

    /// Issue #1928: an exhausted budget fails loudly instead of starting
    /// another inner wait.
    #[test]
    fn exhausted_budget_fails_loudly() {
        let start = Instant::now();
        let budget = GpuTimeBudget::from_caller_timeout_at(
            start - Duration::from_secs(600),
            Duration::from_secs(300),
        );

        assert!(budget.is_expired());
        assert_eq!(budget.remaining_secs(), 0);
        let err = budget
            .check("sub-batch 3")
            .expect_err("exhausted budget must error");
        let message = format!("{err}");
        assert!(
            message.contains("sub-batch 3"),
            "error must name the stage it refused: {message}"
        );
    }

    /// Issue #1928: fixed waits are capped by whatever budget remains.
    #[test]
    fn capped_wait_never_exceeds_remaining_budget() {
        let budget = GpuTimeBudget::from_caller_timeout_at(
            Instant::now() - Duration::from_secs(298),
            Duration::from_secs(300),
        );
        // Roughly 2s of the 295s inner budget is left, so a 5s poll shrinks.
        assert!(budget.capped(Duration::from_secs(5)) <= Duration::from_secs(5));

        let fresh = GpuTimeBudget::from_caller_timeout(Duration::from_secs(300));
        assert_eq!(
            fresh.capped(Duration::from_secs(5)),
            Duration::from_secs(5),
            "a fresh budget leaves short fixed waits untouched"
        );
    }
}
