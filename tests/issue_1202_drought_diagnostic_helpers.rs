//! Unit tests for the drought-diagnostic helper functions (Issue #1202).
//!
//! Covers `TargetFailureTracker::active_cooldown_count`. The drought emission
//! helper itself has its own tests inside the module.
//!
//! Issue #1792: this file also covered `CandidateOutcomeCache::suppressed_count`.
//! That cache was never constructed outside tests, so the counter it fed was
//! structurally always `0` in production; the cache and those tests were
//! deleted.

use neat_ai_discovery::analysis::target_failure_tracker::TargetFailureTracker;

// =============================================================================
// TargetFailureTracker::active_cooldown_count
// =============================================================================

#[test]
fn active_cooldown_count_empty_tracker_is_zero() {
    let tracker = TargetFailureTracker::with_thresholds(3, 10);
    assert_eq!(tracker.active_cooldown_count(0), 0);
    assert_eq!(tracker.active_cooldown_count(1_000), 0);
}

#[test]
fn active_cooldown_count_below_threshold_not_in_cooldown() {
    let mut tracker = TargetFailureTracker::with_thresholds(3, 10);
    tracker.record_failure("A", 0);
    tracker.record_failure("A", 1);
    // Only 2 failures — below the 3-failure threshold.
    assert_eq!(tracker.active_cooldown_count(2), 0);
}

#[test]
fn active_cooldown_count_at_threshold_within_window() {
    let mut tracker = TargetFailureTracker::with_thresholds(3, 10);
    tracker.record_failure("A", 0);
    tracker.record_failure("A", 1);
    tracker.record_failure("A", 2);
    assert_eq!(tracker.active_cooldown_count(2), 1);
    assert_eq!(tracker.active_cooldown_count(11), 1);
    // Past the cooldown window.
    assert_eq!(tracker.active_cooldown_count(12), 0);
}

#[test]
fn active_cooldown_count_mix_of_expired_active_and_below_threshold() {
    let mut tracker = TargetFailureTracker::with_thresholds(3, 10);

    // Target A: 3 failures at epochs 0..=2 — cooldown until epoch 12.
    tracker.record_failure("A", 0);
    tracker.record_failure("A", 1);
    tracker.record_failure("A", 2);

    // Target B: 3 failures at epochs 5..=7 — cooldown until epoch 17.
    tracker.record_failure("B", 5);
    tracker.record_failure("B", 6);
    tracker.record_failure("B", 7);

    // Target C: only 2 failures — below threshold.
    tracker.record_failure("C", 0);
    tracker.record_failure("C", 1);

    // Target D: 5 failures, then a success — cooldown cleared.
    for epoch in 0..5 {
        tracker.record_failure("D", epoch);
    }
    tracker.record_success("D", 6);

    // At epoch 8: A and B in cooldown, C below threshold, D cleared.
    assert_eq!(tracker.active_cooldown_count(8), 2);
    // At epoch 13: A's window has elapsed, B still active.
    assert_eq!(tracker.active_cooldown_count(13), 1);
    // At epoch 20: both A and B have elapsed.
    assert_eq!(tracker.active_cooldown_count(20), 0);
}
