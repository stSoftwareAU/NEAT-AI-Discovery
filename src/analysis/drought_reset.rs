//! Operator escape hatch — forced one-shot reset after a configurable drought
//! (Issue #1205).
//!
//! When the `NEAT_AI_DISCOVERY_DROUGHT_RESET_AFTER_EPOCHS` environment variable
//! is set and the rolling outcome log reports at least that many consecutive
//! trailing failures, the orchestrator invokes
//! [`maybe_perform_drought_reset`] to flush the suppression layers that may be
//! blocking new candidates:
//!
//! - The [`CandidateOutcomeCache`] drops all failed entries (successes and
//!   per-source-type statistics are preserved).
//! - The [`TargetFailureTracker`] drops all targets that are currently in
//!   cooldown (below-threshold tracking is preserved).
//!
//! Both structures carry a `tombstone_reset_epoch` so the same drought streak
//! triggers at most one reset. A subsequent successful outcome (recorded via
//! `CandidateOutcomeCache::record` or `TargetFailureTracker::record_success`)
//! clears the tombstone and re-arms the lever for the next future drought.

use super::candidate_cache::CandidateOutcomeCache;
use super::target_failure_tracker::TargetFailureTracker;

/// Outcome of a single drought-reset invocation (Issue #1205).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DroughtResetOutcome {
    /// Number of failed candidate-cache entries removed (0 when no cache was
    /// supplied).
    pub candidate_cache_failed_cleared: usize,
    /// Number of target-failure-tracker entries dropped from active cooldown
    /// (0 when no tracker was supplied).
    pub target_cooldown_cleared: usize,
    /// The epoch at which the reset fired — recorded in both tombstones.
    pub reset_epoch: u64,
    /// Configured `drought_reset_after_epochs` value that armed this reset.
    pub drought_reset_after: u32,
}

/// Perform the one-shot drought reset when the trigger conditions hold.
///
/// Returns `Some(outcome)` exactly when the reset fired. Returns `None` when:
/// - The lever is disabled (`drought_reset_after == 0`).
/// - The consecutive-failure streak has not yet crossed `drought_reset_after`.
/// - EITHER structure already carries a tombstone for the current streak
///   (a reset has fired and no success has re-armed the lever).
///
/// The orchestrator also clears tombstones on either structure when
/// `consecutive_failures == 0` so a successful pass re-arms the lever even if
/// the per-candidate success path was not exercised (e.g. the cache was not
/// passed to the orchestrator).
pub fn maybe_perform_drought_reset(
    cache: Option<&mut CandidateOutcomeCache>,
    tracker: Option<&mut TargetFailureTracker>,
    consecutive_failures: u32,
    drought_reset_after: u32,
    current_epoch: u64,
) -> Option<DroughtResetOutcome> {
    // Lever disabled or threshold not reached.
    if drought_reset_after == 0 || consecutive_failures < drought_reset_after {
        return None;
    }

    // One-shot semantics: if either structure has already tombstoned the
    // current streak, do nothing. A successful pass clears the tombstone
    // before this point, so a fresh drought naturally re-fires.
    let cache_tombstoned = cache
        .as_ref()
        .is_some_and(|c| c.drought_reset_tombstone().is_some());
    let tracker_tombstoned = tracker
        .as_ref()
        .is_some_and(|t| t.drought_reset_tombstone().is_some());
    if cache_tombstoned || tracker_tombstoned {
        return None;
    }

    let candidate_cache_failed_cleared = match cache {
        Some(c) => c.clear_failed_entries(current_epoch),
        None => 0,
    };
    let target_cooldown_cleared = match tracker {
        Some(t) => t.clear_cooldown_entries(current_epoch),
        None => 0,
    };

    tracing::warn!(
        reset_name = "drought_escape_hatch",
        candidate_cache_failed_cleared,
        target_cooldown_cleared,
        consecutive_failures,
        drought_reset_after,
        current_epoch,
        env_var = "NEAT_AI_DISCOVERY_DROUGHT_RESET_AFTER_EPOCHS",
        "Issue #1205: drought escape hatch fired — cleared {} failed candidate cache entries \
         and {} active target cooldowns after {} consecutive empty passes",
        candidate_cache_failed_cleared,
        target_cooldown_cleared,
        consecutive_failures,
    );

    Some(DroughtResetOutcome {
        candidate_cache_failed_cleared,
        target_cooldown_cleared,
        reset_epoch: current_epoch,
        drought_reset_after,
    })
}

/// Re-arm the lever by clearing the drought-reset tombstone on both
/// structures (Issue #1205). The orchestrator calls this when the rolling
/// outcome log reports a fresh success (`consecutive_failures == 0`).
pub fn rearm_drought_reset(
    cache: Option<&mut CandidateOutcomeCache>,
    tracker: Option<&mut TargetFailureTracker>,
) {
    if let Some(c) = cache {
        c.clear_drought_reset_tombstone();
    }
    if let Some(t) = tracker {
        t.clear_drought_reset_tombstone();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn populated_cache() -> CandidateOutcomeCache {
        let mut cache = CandidateOutcomeCache::new();
        cache.record("s1", "t1", "addSynapse", false, 0);
        cache.record("s2", "t2", "addSynapse", false, 0);
        cache.record("s3", "t3", "addSynapse", true, 0);
        cache
    }

    fn populated_tracker() -> TargetFailureTracker {
        let mut tracker = TargetFailureTracker::with_thresholds(2, 100);
        // Two targets in cooldown.
        tracker.record_failure("A", 0);
        tracker.record_failure("A", 1);
        tracker.record_failure("B", 0);
        tracker.record_failure("B", 1);
        // One target with only one failure -> below threshold, retained.
        tracker.record_failure("C", 0);
        tracker
    }

    #[test]
    fn disabled_lever_never_fires() {
        let mut cache = populated_cache();
        let mut tracker = populated_tracker();
        let out = maybe_perform_drought_reset(Some(&mut cache), Some(&mut tracker), 100, 0, 5);
        assert!(out.is_none());
        assert_eq!(cache.len(), 3); // unchanged
        assert_eq!(tracker.len(), 3);
    }

    #[test]
    fn below_threshold_does_not_fire() {
        let mut cache = populated_cache();
        let mut tracker = populated_tracker();
        let out = maybe_perform_drought_reset(Some(&mut cache), Some(&mut tracker), 9, 10, 5);
        assert!(out.is_none());
    }

    #[test]
    fn at_threshold_fires_once_and_tombstones() {
        let mut cache = populated_cache();
        let mut tracker = populated_tracker();

        let out = maybe_perform_drought_reset(Some(&mut cache), Some(&mut tracker), 10, 10, 5)
            .expect("fires at threshold");
        assert_eq!(out.candidate_cache_failed_cleared, 2);
        assert_eq!(out.target_cooldown_cleared, 2);
        assert_eq!(out.reset_epoch, 5);

        // Successful entry preserved, failed entries gone.
        assert_eq!(cache.len(), 1);
        // Below-threshold tracker entry preserved, cooldown entries gone.
        assert_eq!(tracker.len(), 1);
        assert!(tracker.state("C").is_some());

        // Tombstones set on both.
        assert_eq!(cache.drought_reset_tombstone(), Some(5));
        assert_eq!(tracker.drought_reset_tombstone(), Some(5));
    }

    #[test]
    fn second_call_within_streak_is_noop() {
        let mut cache = populated_cache();
        let mut tracker = populated_tracker();

        let _first = maybe_perform_drought_reset(Some(&mut cache), Some(&mut tracker), 10, 10, 5)
            .expect("first fires");
        // Cache now has only successful entries — re-add a fresh failure to
        // prove the second call leaves it untouched.
        cache.record("s4", "t4", "addSynapse", false, 6);
        tracker.record_failure("D", 6);
        tracker.record_failure("D", 7);

        let second = maybe_perform_drought_reset(Some(&mut cache), Some(&mut tracker), 11, 10, 7);
        assert!(second.is_none(), "second invocation must not re-fire");

        // The freshly-added failure is preserved.
        assert!(
            cache
                .get_outcome("s4", "t4", "addSynapse")
                .is_some_and(|o| !o.succeeded)
        );
        assert!(tracker.state("D").is_some());
    }

    #[test]
    fn rearm_after_success_allows_next_reset() {
        let mut cache = populated_cache();
        let mut tracker = populated_tracker();

        // First fire.
        let _ = maybe_perform_drought_reset(Some(&mut cache), Some(&mut tracker), 10, 10, 5)
            .expect("first fires");

        // Re-arm via successful outcomes on both surfaces.
        cache.record("good-src", "good-tgt", "addSynapse", true, 6);
        tracker.record_success("good-tgt", 6);
        assert!(cache.drought_reset_tombstone().is_none());
        assert!(tracker.drought_reset_tombstone().is_none());

        // Set up a second drought.
        cache.record("s5", "t5", "addSynapse", false, 7);
        tracker.record_failure("E", 7);
        tracker.record_failure("E", 8);

        let second = maybe_perform_drought_reset(Some(&mut cache), Some(&mut tracker), 10, 10, 9);
        assert!(second.is_some(), "re-armed lever fires again");
    }

    #[test]
    fn rearm_via_helper_when_outcome_path_unused() {
        let mut cache = populated_cache();
        let mut tracker = populated_tracker();
        let _ = maybe_perform_drought_reset(Some(&mut cache), Some(&mut tracker), 10, 10, 5);
        rearm_drought_reset(Some(&mut cache), Some(&mut tracker));
        assert!(cache.drought_reset_tombstone().is_none());
        assert!(tracker.drought_reset_tombstone().is_none());
    }

    /// Issue #1790: the orchestrator sources `current_epoch` from the global
    /// tracker's internal counter (`orchestration.rs`). Now that the counter
    /// actually advances, the reset must fire at that non-zero epoch and stamp
    /// the tombstone with it — not with `0`.
    #[test]
    fn reset_at_advanced_internal_epoch_stamps_non_zero_tombstone() {
        let mut tracker = TargetFailureTracker::with_thresholds(2, 100);
        // Three discovery passes' worth of epoch advance, then two failures
        // recorded through the internal-counter path.
        tracker.advance_epoch();
        tracker.advance_epoch();
        tracker.advance_epoch();
        tracker.record_failure_now("A");
        tracker.record_failure_now("A");
        assert!(tracker.is_in_cooldown_now("A"));

        // Mirrors `orchestration.rs`: epoch_for_reset = guard.current_epoch().
        let epoch_for_reset = tracker.current_epoch();
        assert_eq!(epoch_for_reset, 3, "the internal counter must have moved");

        let out = maybe_perform_drought_reset(None, Some(&mut tracker), 10, 10, epoch_for_reset)
            .expect("fires at the advanced epoch");

        assert_eq!(out.reset_epoch, 3);
        assert_eq!(out.target_cooldown_cleared, 1);
        assert!(!tracker.is_in_cooldown_now("A"));
        assert_eq!(
            tracker.drought_reset_tombstone(),
            Some(3),
            "tombstone must carry the current non-zero epoch"
        );
    }

    #[test]
    fn fires_without_cache_supplied() {
        let mut tracker = populated_tracker();
        let out = maybe_perform_drought_reset(None, Some(&mut tracker), 10, 10, 5)
            .expect("fires with tracker-only");
        assert_eq!(out.candidate_cache_failed_cleared, 0);
        assert_eq!(out.target_cooldown_cleared, 2);
    }
}
