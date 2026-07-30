//! Within-batch target-failure short-circuit (Issue #1164).
//!
//! When a candidate targeting neuron T fails post-evaluation in the current
//! discovery batch, the within-batch tracker records the failure and lets the
//! next candidate evaluation short-circuit if the target's failure count has
//! reached the configured threshold. Complements the cross-batch cooldown
//! (Issue #1130) by closing the within-batch gap: a single batch could
//! otherwise emit several failing candidates for the same target before the
//! cross-batch cooldown engaged.
//!
//! # Design
//!
//! - Keyed by `target_neuron_uuid`.
//! - Tracks the within-batch failure count per target and the total number of
//!   skip events for diagnostics.
//! - A target is short-circuited once its within-batch failure count reaches
//!   [`WithinBatchFailureTracker::failure_limit`]. Default is `1` so the very
//!   next same-target candidate is skipped after the first failure.
//! - The threshold is env-var overridable via
//!   `NEAT_AI_DISCOVERY_BATCH_TARGET_FAILURE_LIMIT`.
//!
//! # Lifetime
//!
//! Each orchestration call creates a fresh tracker; it does not persist
//! across batches. Cross-batch cooldown is the responsibility of the global
//! [`crate::analysis::target_failure_tracker::TargetFailureTracker`].

use std::collections::HashMap;
use std::sync::Mutex;

use crate::analysis::constants::WITHIN_BATCH_TARGET_FAILURE_LIMIT;
use crate::analysis::diagnostics::RejectionBreakdown;
use crate::analysis::diagnostics::rejection_reasons::REJECTION_WITHIN_BATCH_TARGET_SHORT_CIRCUIT;
use crate::config::within_batch_target_failure_limit_env;

/// Per-batch tracker keyed by target UUID.
///
/// Designed to be shared across rayon worker threads via `Arc`. Internal
/// state is guarded by a `Mutex`; the critical sections are short
/// (`HashMap` insert/lookup) so contention is negligible compared to the GPU
/// evaluation work the tracker is gating.
#[derive(Debug)]
pub struct WithinBatchFailureTracker {
    state: Mutex<State>,
    failure_limit: u32,
}

#[derive(Debug, Default)]
struct State {
    failures: HashMap<String, u32>,
    skip_count: u32,
}

impl WithinBatchFailureTracker {
    /// Create a tracker using the env-var override or the compiled default.
    #[must_use]
    pub fn new() -> Self {
        Self::with_threshold(
            within_batch_target_failure_limit_env().unwrap_or(WITHIN_BATCH_TARGET_FAILURE_LIMIT),
        )
    }

    /// Create a tracker with an explicit threshold (for tests).
    #[must_use]
    pub fn with_threshold(failure_limit: u32) -> Self {
        Self {
            state: Mutex::new(State::default()),
            failure_limit: failure_limit.max(1),
        }
    }

    /// Returns the configured failure threshold.
    #[must_use]
    pub fn failure_limit(&self) -> u32 {
        self.failure_limit
    }

    /// Record a within-batch failure for `target_uuid`.
    pub fn record_failure(&self, target_uuid: &str) {
        let mut guard = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let entry = guard.failures.entry(target_uuid.to_string()).or_insert(0);
        *entry = entry.saturating_add(1);
    }

    /// Returns `true` when `target_uuid` should be skipped because its
    /// within-batch failure count has reached the threshold.
    #[must_use]
    pub fn should_skip(&self, target_uuid: &str) -> bool {
        let guard = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.failures.get(target_uuid).copied().unwrap_or(0) >= self.failure_limit
    }

    /// Increment the diagnostic skip counter; called when a candidate is
    /// short-circuited by [`Self::should_skip`].
    pub fn record_skip(&self) {
        let mut guard = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.skip_count = guard.skip_count.saturating_add(1);
    }

    /// Returns the cumulative number of candidates skipped during this batch.
    #[must_use]
    pub fn skip_count(&self) -> u32 {
        let guard = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.skip_count
    }

    /// Returns the within-batch failure count recorded for `target_uuid`.
    #[must_use]
    pub fn failure_count(&self, target_uuid: &str) -> u32 {
        let guard = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.failures.get(target_uuid).copied().unwrap_or(0)
    }

    /// Returns the number of distinct targets that have at least one
    /// within-batch failure.
    #[must_use]
    pub fn failed_target_count(&self) -> usize {
        let guard = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.failures.len()
    }
}

impl Default for WithinBatchFailureTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Fold this batch's short-circuit skips into `breakdown` as
/// [`REJECTION_WITHIN_BATCH_TARGET_SHORT_CIRCUIT`] rejections (Issue #1796).
///
/// Without this the short-circuit was a silent drop: the suppressed candidates
/// incremented no rejection counter, so
/// [`crate::analysis::candidate_starvation::classify`] — which reads only the
/// breakdown — could not see them.
///
/// Aggregate, not per-candidate: one call per surface (neuron / synapse) keeps
/// the evaluation hot loop allocation-free. Each orchestration call owns a
/// distinct tracker, so calling this once per surface cannot double count.
/// Returns the folded count.
pub fn fold_within_batch_skips(
    tracker: &WithinBatchFailureTracker,
    breakdown: &mut RejectionBreakdown,
) -> u32 {
    let skips = tracker.skip_count();
    breakdown.record_many_u32(REJECTION_WITHIN_BATCH_TARGET_SHORT_CIRCUIT, skips);
    skips
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_tracker_defaults_match_constant() {
        let tracker = WithinBatchFailureTracker::with_threshold(1);
        assert_eq!(tracker.failure_limit(), 1);
        assert_eq!(tracker.skip_count(), 0);
        assert_eq!(tracker.failed_target_count(), 0);
        assert!(!tracker.should_skip("any"));
    }

    /// First failure for a target → subsequent same-target candidates are
    /// short-circuited (default threshold 1).
    #[test]
    fn first_failure_short_circuits_subsequent_same_target_candidates() {
        let tracker = WithinBatchFailureTracker::with_threshold(1);
        assert!(!tracker.should_skip("T"));

        tracker.record_failure("T");
        assert!(tracker.should_skip("T"));
        assert_eq!(tracker.failure_count("T"), 1);
        assert_eq!(tracker.failed_target_count(), 1);
    }

    /// A success for a target does NOT add it to the failure set, so other
    /// candidates targeting T proceed normally.
    #[test]
    fn success_does_not_add_target_to_set() {
        let tracker = WithinBatchFailureTracker::with_threshold(1);

        // No failure recorded for "T" — should not be skipped.
        assert!(!tracker.should_skip("T"));
        assert_eq!(tracker.failed_target_count(), 0);
    }

    /// Threshold > 1 — only short-circuit after N within-batch failures.
    #[test]
    fn threshold_gt_one_requires_n_failures_before_skip() {
        let tracker = WithinBatchFailureTracker::with_threshold(3);

        tracker.record_failure("T");
        assert!(!tracker.should_skip("T"));
        tracker.record_failure("T");
        assert!(!tracker.should_skip("T"));

        tracker.record_failure("T");
        assert!(
            tracker.should_skip("T"),
            "skip should engage at the threshold"
        );
    }

    #[test]
    fn record_skip_increments_skip_count() {
        let tracker = WithinBatchFailureTracker::with_threshold(1);
        tracker.record_skip();
        tracker.record_skip();
        assert_eq!(tracker.skip_count(), 2);
    }

    #[test]
    fn unrelated_target_not_affected_by_failures() {
        let tracker = WithinBatchFailureTracker::with_threshold(1);
        tracker.record_failure("A");
        assert!(tracker.should_skip("A"));
        assert!(!tracker.should_skip("B"));
    }

    #[test]
    fn explicit_threshold_zero_is_clamped_to_one() {
        // A threshold of zero would be nonsensical (skip even before any
        // failure). The constructor clamps to 1.
        let tracker = WithinBatchFailureTracker::with_threshold(0);
        assert_eq!(tracker.failure_limit(), 1);
        assert!(!tracker.should_skip("T"));
        tracker.record_failure("T");
        assert!(tracker.should_skip("T"));
    }
}
