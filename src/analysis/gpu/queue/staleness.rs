//! Stale GPU work request detection (Issue #1929).
//!
//! A submitter that times out simply drops its `response_rx`; the GPU thread
//! has no other signal that the caller has gone. Without a check the worker
//! dequeues that request minutes later, spends a full evaluation on it, and —
//! on the device-lost path — even re-initialises the `GpuAnalyzer` and retries
//! it `NEAT_AI_DISCOVERY_GPU_RETRY_LIMIT` times. All of that work is discarded
//! by the `send()` at the end, while live submitters queue behind it on a
//! bounded channel.
//!
//! Two conditions make a dequeued request stale:
//!
//! 1. **No live receiver** — the submitter dropped the [`CallerGuard`] it holds
//!    beside its response receiver, so nothing can ever observe the result.
//! 2. **Budget already expired** — the per-request `GpuTimeBudget` (Issue
//!    #1928) ran out while the request sat in the queue, so the caller is about
//!    to give up even though it is still waiting.
//!
//! ```text
//! dequeue ──▶ stale_reason() ──▶ None            ──▶ execute on GPU
//!                             ├─▶ ReceiverGone   ──▶ drop, count, no analysis
//!                             └─▶ BudgetExpired  ──▶ error to caller, count
//! ```
//!
//! `crossbeam_channel::Sender` exposes no receiver count, so liveness is
//! carried explicitly: the submitter keeps a `CallerGuard` for exactly as long
//! as it will accept a result, and the request carries the paired weak
//! [`CallerLiveness`] handle.

use std::sync::{Arc, Weak};

use super::GpuWorkRequest;
use crate::analysis::gpu::budget::GpuTimeBudget;

/// The submitter's proof that it is still waiting for a result.
///
/// Held beside the response receiver — usually as a local that lives until the
/// blocking `recv_timeout()` returns, or inside the `GpuFuture` — so it is
/// dropped at exactly the moment the caller stops listening.
#[derive(Debug)]
pub(crate) struct CallerGuard(#[allow(dead_code)] Arc<()>);

/// Worker-side view of a submitter's liveness.
#[derive(Debug, Clone)]
pub(crate) struct CallerLiveness(Weak<()>);

impl CallerLiveness {
    /// Whether the submitter still holds its [`CallerGuard`].
    pub(crate) fn is_live(&self) -> bool {
        self.0.strong_count() > 0
    }
}

/// Create a matched guard/liveness pair.
pub(crate) fn caller_liveness_pair() -> (CallerGuard, CallerLiveness) {
    let guard = Arc::new(());
    let liveness = CallerLiveness(Arc::downgrade(&guard));
    (CallerGuard(guard), liveness)
}

/// Why a dequeued request was skipped without invoking the analyser.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StaleReason {
    /// The caller dropped its response receiver — it timed out or was cancelled.
    ReceiverGone,
    /// The request's own time budget expired while it waited in the queue.
    BudgetExpired,
}

impl StaleReason {
    /// Short label for structured logs.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::ReceiverGone => "receiver_gone",
            Self::BudgetExpired => "budget_expired",
        }
    }
}

/// Whether a caller is still waiting for this request's result.
///
/// `Shutdown` carries no response channel and is always considered live so the
/// loop never mistakes it for abandoned work.
pub(crate) fn has_live_receiver(request: &GpuWorkRequest) -> bool {
    match request {
        GpuWorkRequest::HelpfulBatch { liveness, .. }
        | GpuWorkRequest::HarmfulBatch { liveness, .. }
        | GpuWorkRequest::ReluEval { liveness, .. }
        | GpuWorkRequest::ActivationEval { liveness, .. }
        | GpuWorkRequest::ActivationBatchEval { liveness, .. } => liveness.is_live(),
        GpuWorkRequest::Shutdown => true,
    }
}

/// The time budget attached to a request, if it carries one.
fn request_budget(request: &GpuWorkRequest) -> Option<GpuTimeBudget> {
    match request {
        GpuWorkRequest::HelpfulBatch { budget, .. }
        | GpuWorkRequest::HarmfulBatch { budget, .. }
        | GpuWorkRequest::ReluEval { budget, .. }
        | GpuWorkRequest::ActivationEval { budget, .. }
        | GpuWorkRequest::ActivationBatchEval { budget, .. } => Some(*budget),
        GpuWorkRequest::Shutdown => None,
    }
}

/// Classify a dequeued request. `None` means it is still worth executing.
///
/// A dead receiver is checked first: when nobody can receive the result there
/// is no point reporting a budget error either.
pub(crate) fn stale_reason(request: &GpuWorkRequest) -> Option<StaleReason> {
    if !has_live_receiver(request) {
        return Some(StaleReason::ReceiverGone);
    }
    if request_budget(request).is_some_and(|budget| budget.is_expired()) {
        return Some(StaleReason::BudgetExpired);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::samples::{HarmfulStats, ReluStats};
    use anyhow::Result;
    use crossbeam_channel::bounded;
    use std::time::{Duration, Instant};

    /// Build a `HelpfulBatch` request with the given budget, returning the
    /// caller guard so the test controls when the submitter "goes away".
    fn helpful_request(budget: GpuTimeBudget) -> (GpuWorkRequest, CallerGuard) {
        let (tx, rx) = bounded(1);
        let (guard, liveness) = caller_liveness_pair();
        // The receiver is irrelevant to staleness — the guard is the signal.
        drop(rx);
        (
            GpuWorkRequest::HelpfulBatch {
                samples: vec![],
                response_tx: tx,
                budget,
                liveness,
            },
            guard,
        )
    }

    /// Build a request of each non-helpful variant with the given budget and
    /// liveness, so the per-variant tests stay short.
    fn other_variants(budget: GpuTimeBudget, liveness: &CallerLiveness) -> Vec<GpuWorkRequest> {
        let (harmful_tx, _harmful_rx) = bounded::<Result<Vec<HarmfulStats>>>(1);
        let (relu_tx, _relu_rx) = bounded::<Result<(ReluStats, ReluStats, f32)>>(1);
        let (act_tx, _act_rx) = bounded::<Result<(f32, f32, f32, u32)>>(1);
        let (batch_tx, _batch_rx) = bounded::<Result<Vec<(f32, f32, f32, u32)>>>(1);
        vec![
            GpuWorkRequest::HarmfulBatch {
                samples_with_weights: vec![],
                response_tx: harmful_tx,
                budget,
                liveness: liveness.clone(),
            },
            GpuWorkRequest::ReluEval {
                samples: vec![],
                threshold: 0.0,
                response_tx: relu_tx,
                budget,
                liveness: liveness.clone(),
            },
            GpuWorkRequest::ActivationEval {
                samples: vec![],
                activation_type: 0,
                orientation: 1.0,
                scale: 1.0,
                response_tx: act_tx,
                budget,
                liveness: liveness.clone(),
            },
            GpuWorkRequest::ActivationBatchEval {
                samples: vec![],
                activation_configs: vec![],
                response_tx: batch_tx,
                budget,
                liveness: liveness.clone(),
            },
        ]
    }

    /// A budget whose deadline is already in the past.
    fn expired_budget() -> GpuTimeBudget {
        let start = Instant::now() - Duration::from_secs(600);
        GpuTimeBudget::from_caller_timeout_at(start, Duration::from_secs(60))
    }

    #[test]
    fn live_caller_with_unbounded_budget_is_not_stale() {
        let (request, _guard) = helpful_request(GpuTimeBudget::unbounded());
        assert!(has_live_receiver(&request));
        assert_eq!(stale_reason(&request), None);
    }

    #[test]
    fn dropped_caller_guard_is_stale() {
        let (request, guard) = helpful_request(GpuTimeBudget::unbounded());
        drop(guard); // the submitter timed out and stopped waiting
        assert!(!has_live_receiver(&request));
        assert_eq!(stale_reason(&request), Some(StaleReason::ReceiverGone));
    }

    #[test]
    fn expired_budget_is_stale_even_with_live_caller() {
        let (request, _guard) = helpful_request(expired_budget());
        assert!(has_live_receiver(&request));
        assert_eq!(stale_reason(&request), Some(StaleReason::BudgetExpired));
    }

    #[test]
    fn unexpired_budget_with_live_caller_is_not_stale() {
        let budget = GpuTimeBudget::from_caller_timeout(Duration::from_secs(600));
        let (request, _guard) = helpful_request(budget);
        assert_eq!(stale_reason(&request), None);
    }

    #[test]
    fn dropped_caller_wins_over_expired_budget() {
        let (request, guard) = helpful_request(expired_budget());
        drop(guard);
        assert_eq!(stale_reason(&request), Some(StaleReason::ReceiverGone));
    }

    #[test]
    fn shutdown_is_never_stale() {
        assert!(has_live_receiver(&GpuWorkRequest::Shutdown));
        assert_eq!(stale_reason(&GpuWorkRequest::Shutdown), None);
        assert!(request_budget(&GpuWorkRequest::Shutdown).is_none());
    }

    #[test]
    fn dropped_caller_detected_for_every_variant() {
        let (guard, liveness) = caller_liveness_pair();
        let requests = other_variants(GpuTimeBudget::unbounded(), &liveness);
        for request in &requests {
            assert_eq!(stale_reason(request), None, "live caller is not stale");
        }
        drop(guard);
        for request in &requests {
            assert_eq!(
                stale_reason(request),
                Some(StaleReason::ReceiverGone),
                "every variant must notice the submitter has gone"
            );
        }
    }

    #[test]
    fn expired_budget_detected_for_every_variant() {
        let (_guard, liveness) = caller_liveness_pair();
        for request in &other_variants(expired_budget(), &liveness) {
            assert_eq!(stale_reason(request), Some(StaleReason::BudgetExpired));
        }
    }

    #[test]
    fn stale_reason_labels_are_distinct() {
        assert_eq!(StaleReason::ReceiverGone.as_str(), "receiver_gone");
        assert_eq!(StaleReason::BudgetExpired.as_str(), "budget_expired");
    }
}
