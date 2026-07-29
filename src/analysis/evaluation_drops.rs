//! Per-batch accounting for the evaluation-loop drop sites (Issue #1798).
//!
//! Three per-candidate `continue` sites in the evaluation loops dropped
//! candidates without incrementing anything:
//!
//! | Site | Condition | Reason |
//! | --- | --- | --- |
//! | `analysis::neuron::evaluation` | no samples built for the source | [`REJECTION_NO_SAMPLES`] |
//! | `analysis::neuron::evaluation` | source variance discount collapsed to zero | [`REJECTION_ZERO_SOURCE_VARIANCE`] |
//! | `analysis::synapse::target_analysis::evaluation` | no samples in the GPU work item | [`REJECTION_NO_SAMPLES`] |
//!
//! Candidates dropped there never reach the accept gate, so they biased
//! [`crate::analysis::candidate_starvation::classify`] toward
//! `ProposalRichOverRejected` — and thus toward suppressing the widening
//! bypass — exactly when generation was the real bottleneck.
//!
//! # Design
//!
//! The drop sites sit in a per-candidate loop shared across rayon workers, so
//! the counters are plain relaxed atomic increments: no lock, no allocation,
//! no per-candidate `String`. The aggregate is folded into the metadata
//! rejection breakdown once per surface by [`fold_evaluation_drops`], the same
//! pattern [`crate::analysis::within_batch_failures::fold_within_batch_skips`]
//! uses.
//!
//! The guard predicates live here alongside the counters so counting cannot
//! drift from the condition it accounts for: a drop site calls
//! [`EvaluationDropCounters::drop_for_empty_samples`] /
//! [`EvaluationDropCounters::drop_for_zero_source_variance`] and `continue`s on
//! `true`. Drop *behaviour* is unchanged — the predicates are exactly the
//! conditions they replaced.

use std::sync::atomic::{AtomicU32, Ordering};

use crate::analysis::diagnostics::RejectionBreakdown;
use crate::analysis::diagnostics::rejection_reasons::{
    REJECTION_NO_SAMPLES, REJECTION_ZERO_SOURCE_VARIANCE,
};
use crate::analysis::samples::EPSILON;

/// Per-batch counters for the evaluation-loop drop sites.
///
/// Shared across rayon workers via `Arc`. Each orchestration call creates a
/// fresh instance; it does not persist across batches.
#[derive(Debug, Default)]
pub struct EvaluationDropCounters {
    no_samples: AtomicU32,
    zero_source_variance: AtomicU32,
}

impl EvaluationDropCounters {
    /// Create a zeroed set of counters.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns `true` when `samples` is empty, counting the drop as
    /// [`REJECTION_NO_SAMPLES`].
    ///
    /// Behaviour is identical to the bare `samples.is_empty()` guard this
    /// replaces — the caller still `continue`s on `true`.
    #[inline]
    pub fn drop_for_empty_samples<T>(&self, samples: &[T]) -> bool {
        let empty = samples.is_empty();
        if empty {
            self.no_samples.fetch_add(1, Ordering::Relaxed);
        }
        empty
    }

    /// Returns `true` when the source variance discount has collapsed to
    /// (near) zero — a constant source carrying no signal — counting the drop
    /// as [`REJECTION_ZERO_SOURCE_VARIANCE`].
    ///
    /// Behaviour is identical to the bare `discount <= EPSILON` guard this
    /// replaces.
    #[inline]
    pub fn drop_for_zero_source_variance(&self, source_variance_discount: f32) -> bool {
        let zero = source_variance_discount <= EPSILON;
        if zero {
            self.zero_source_variance.fetch_add(1, Ordering::Relaxed);
        }
        zero
    }

    /// Candidates dropped because no samples were available.
    #[must_use]
    pub fn no_samples(&self) -> u32 {
        self.no_samples.load(Ordering::Relaxed)
    }

    /// Candidates dropped because the source carried no variance.
    #[must_use]
    pub fn zero_source_variance(&self) -> u32 {
        self.zero_source_variance.load(Ordering::Relaxed)
    }
}

/// Fold this batch's evaluation-loop drops into `breakdown` (Issue #1798).
///
/// Aggregate, not per-candidate: one call per surface (neuron / synapse) keeps
/// the evaluation hot loop allocation-free. Each orchestration call owns a
/// distinct counter set, so calling this once per surface cannot double count.
/// Returns the total number of drops folded in.
pub fn fold_evaluation_drops(
    counters: &EvaluationDropCounters,
    breakdown: &mut RejectionBreakdown,
) -> u32 {
    let no_samples = counters.no_samples();
    let zero_variance = counters.zero_source_variance();
    breakdown.record_many_u32(REJECTION_NO_SAMPLES, no_samples);
    breakdown.record_many_u32(REJECTION_ZERO_SOURCE_VARIANCE, zero_variance);
    no_samples.saturating_add(zero_variance)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::samples::{HelpfulSample, compute_source_variance_discount};

    fn sample(activation: f32) -> HelpfulSample {
        HelpfulSample {
            activation,
            avg_error: 0.1,
            target_value: None,
            target_activation: None,
        }
    }

    #[test]
    fn new_counters_start_at_zero() {
        let counters = EvaluationDropCounters::new();
        assert_eq!(counters.no_samples(), 0);
        assert_eq!(counters.zero_source_variance(), 0);
    }

    #[test]
    fn empty_samples_drop_is_counted() {
        let counters = EvaluationDropCounters::new();
        let empty: [HelpfulSample; 0] = [];
        assert!(counters.drop_for_empty_samples(&empty));
        assert_eq!(counters.no_samples(), 1);
    }

    #[test]
    fn non_empty_samples_are_not_counted() {
        let counters = EvaluationDropCounters::new();
        let populated = [sample(0.0), sample(1.0)];
        assert!(!counters.drop_for_empty_samples(&populated));
        assert_eq!(counters.no_samples(), 0);
    }

    /// The zero-variance guard must agree with the real discount function on
    /// a genuinely constant source.
    #[test]
    fn constant_source_drop_is_counted() {
        let counters = EvaluationDropCounters::new();
        let constant = [sample(0.5), sample(0.5), sample(0.5)];
        let discount = compute_source_variance_discount(&constant);
        assert!(counters.drop_for_zero_source_variance(discount));
        assert_eq!(counters.zero_source_variance(), 1);
    }

    /// A source with real variance is not dropped and not counted.
    #[test]
    fn varying_source_is_not_counted() {
        let counters = EvaluationDropCounters::new();
        let varying = [sample(-1.0), sample(0.0), sample(1.0)];
        let discount = compute_source_variance_discount(&varying);
        assert!(discount > EPSILON, "test fixture must carry real variance");
        assert!(!counters.drop_for_zero_source_variance(discount));
        assert_eq!(counters.zero_source_variance(), 0);
    }

    #[test]
    fn fold_records_both_reasons() {
        let counters = EvaluationDropCounters::new();
        let empty: [HelpfulSample; 0] = [];
        let constant = [sample(2.0), sample(2.0)];
        for _ in 0..3 {
            counters.drop_for_empty_samples(&empty);
        }
        for _ in 0..2 {
            counters.drop_for_zero_source_variance(compute_source_variance_discount(&constant));
        }

        let mut breakdown = RejectionBreakdown::new();
        let folded = fold_evaluation_drops(&counters, &mut breakdown);

        assert_eq!(folded, 5);
        assert_eq!(breakdown.counts().get(REJECTION_NO_SAMPLES), Some(&3));
        assert_eq!(
            breakdown.counts().get(REJECTION_ZERO_SOURCE_VARIANCE),
            Some(&2)
        );
    }

    /// A clean batch records nothing — the reason keys are absent rather than
    /// present-and-zero.
    #[test]
    fn fold_records_nothing_when_no_drops() {
        let counters = EvaluationDropCounters::new();
        let mut breakdown = RejectionBreakdown::new();
        assert_eq!(fold_evaluation_drops(&counters, &mut breakdown), 0);
        assert!(breakdown.is_empty());
    }
}
