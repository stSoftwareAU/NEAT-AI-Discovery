//! Wall-clock budget enforcement for focus ranking (Issue #1375).
//!
//! Focus ranking previously had **no wall-clock bound**: in the GRQ-13 incident
//! it ran for over an hour and blew the whole 3h discovery budget. The per-chunk
//! Rust FFI analysis already enforces a budget; this gives focus ranking the same
//! safety net so a pathological run degrades gracefully.
//!
//! [`FocusRankingDeadline`] is checked between ranking passes and inside the
//! per-neuron loops. On exceed it returns a structured
//! [`DiscoveryError::Timeout`], which the FFI surfaces as a retryable `timeout`
//! error so the TypeScript caller routes the abort into its existing local
//! fallback path.

use crate::ffi_types::DiscoveryError;
use anyhow::Result;
use std::time::{Duration, Instant};

/// Wall-clock deadline for a single focus-ranking invocation (Issue #1375).
///
/// Anchored at the ranking start instant plus the configured budget. When the
/// budget is disabled (`NEAT_AI_DISCOVERY_FOCUS_RANKING_BUDGET_MS=0`) the
/// deadline is `None` and [`Self::check`] is a no-op, preserving the previous
/// unbounded behaviour for callers that explicitly opt out.
pub(super) struct FocusRankingDeadline {
    /// Absolute monotonic deadline, or `None` when the bound is disabled.
    deadline: Option<Instant>,
    /// Configured budget in milliseconds (for the structured timeout error and
    /// log lines). `0` when disabled.
    budget_ms: u64,
}

impl FocusRankingDeadline {
    /// Build from the configured budget (env-driven), anchored at `start`.
    pub(super) fn from_config(start: Instant) -> Self {
        Self::from_budget_ms(start, crate::config::focus_ranking_budget_ms())
    }

    /// Build from an explicit optional budget, anchored at `start`.
    ///
    /// `None` disables the bound; `Some(ms)` sets a deadline `ms` milliseconds
    /// after `start`.
    pub(super) fn from_budget_ms(start: Instant, budget_ms: Option<u64>) -> Self {
        match budget_ms {
            Some(ms) => Self {
                deadline: start.checked_add(Duration::from_millis(ms)),
                budget_ms: ms,
            },
            None => Self {
                deadline: None,
                budget_ms: 0,
            },
        }
    }

    /// Return `Err(DiscoveryError::Timeout)` when the wall-clock budget has been
    /// exceeded, otherwise `Ok(())`.
    ///
    /// `phase` names the ranking pass for the structured warning log so operators
    /// can see where the abort fired. The check is a single monotonic-clock
    /// comparison, so calling it between passes and inside per-neuron loops adds
    /// no measurable overhead to fast runs.
    pub(super) fn check(&self, phase: &str) -> Result<()> {
        if let Some(deadline) = self.deadline
            && Instant::now() >= deadline
        {
            tracing::warn!(
                target: "neat_ai_discovery::focus::ranking",
                phase,
                budget_ms = self.budget_ms,
                "focus ranking exceeded wall-clock budget — aborting for graceful fallback (Issue #1375)",
            );
            return Err(DiscoveryError::Timeout {
                deadline_ms: self.budget_ms,
            }
            .into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi_types::DiscoveryErrorKind;

    #[test]
    fn disabled_budget_never_aborts() {
        let deadline = FocusRankingDeadline::from_budget_ms(Instant::now(), None);
        // Even well after construction, a disabled bound is a no-op.
        assert!(deadline.check("any_phase").is_ok());
    }

    #[test]
    fn future_deadline_does_not_abort() {
        let deadline = FocusRankingDeadline::from_budget_ms(Instant::now(), Some(60_000));
        assert!(deadline.check("phase").is_ok());
    }

    #[test]
    fn elapsed_budget_aborts_with_timeout_error() {
        // Anchor the deadline in the past so the budget is already exceeded.
        let start = Instant::now() - Duration::from_millis(500);
        let deadline = FocusRankingDeadline::from_budget_ms(start, Some(100));
        let err = deadline
            .check("verify_records")
            .expect_err("an exceeded budget must abort");
        let typed = err
            .downcast_ref::<DiscoveryError>()
            .expect("abort must be a typed DiscoveryError");
        assert_eq!(typed.error_kind(), DiscoveryErrorKind::Timeout);
        assert!(
            typed.error_kind().is_retryable(),
            "budget aborts are retryable so the caller falls back gracefully"
        );
    }
}
