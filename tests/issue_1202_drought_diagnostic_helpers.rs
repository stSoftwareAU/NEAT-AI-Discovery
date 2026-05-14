//! Unit tests for the drought-diagnostic helper functions (Issue #1202).
//!
//! Covers `CandidateOutcomeCache::suppressed_count` and
//! `TargetFailureTracker::active_cooldown_count`. The drought emission helper
//! itself has its own tests inside the module.

use neat_ai_discovery::analysis::candidate_cache::{
    CandidateOutcomeCache, DEFAULT_STALENESS_WINDOW,
};
use neat_ai_discovery::analysis::target_failure_tracker::TargetFailureTracker;

// =============================================================================
// CandidateOutcomeCache::suppressed_count
// =============================================================================

#[test]
fn suppressed_count_empty_cache_is_zero() {
    let cache = CandidateOutcomeCache::new();
    assert_eq!(cache.suppressed_count(0), 0);
    assert_eq!(cache.suppressed_count(1_000_000), 0);
}

#[test]
fn suppressed_count_only_failures_within_window_are_counted() {
    let mut cache = CandidateOutcomeCache::new();
    // 3 failures at epoch 0, default staleness window is 100.
    cache.record("s1", "t1", "addSynapse", false, 0);
    cache.record("s2", "t2", "addSynapse", false, 0);
    cache.record("s3", "t3", "addSynapse", false, 0);
    // 1 success — never counts.
    cache.record("s4", "t4", "addSynapse", true, 0);

    assert_eq!(cache.suppressed_count(0), 3);
    // Halfway through the window — still suppressed.
    assert_eq!(cache.suppressed_count(50), 3);
    // At the window boundary — suppression releases (epoch + window not <).
    assert_eq!(cache.suppressed_count(DEFAULT_STALENESS_WINDOW), 0);
    assert_eq!(cache.suppressed_count(DEFAULT_STALENESS_WINDOW + 1), 0);
}

#[test]
fn suppressed_count_mix_of_expired_and_active() {
    let mut cache = CandidateOutcomeCache::with_staleness_window(50);
    // Failure at epoch 0 — expires at epoch 50.
    cache.record("s-old", "t1", "addSynapse", false, 0);
    // Failure at epoch 40 — expires at epoch 90.
    cache.record("s-mid", "t2", "addSynapse", false, 40);
    // Failure at epoch 70 — expires at epoch 120.
    cache.record("s-new", "t3", "addSynapse", false, 70);
    // Success — never counted.
    cache.record("s-ok", "t4", "addSynapse", true, 0);

    // At epoch 60: s-old expired (60 >= 0+50), s-mid still active (60 < 40+50),
    // s-new not yet recorded… wait, it's recorded but failure_epoch = 70, so
    // current_epoch < epoch + window → 60 < 70 + 50 → true, so it counts.
    assert_eq!(cache.suppressed_count(60), 2);

    // At epoch 100: s-old expired, s-mid expired (100 >= 90), s-new still
    // active (100 < 120).
    assert_eq!(cache.suppressed_count(100), 1);

    // At epoch 200: everything expired.
    assert_eq!(cache.suppressed_count(200), 0);
}

#[test]
fn suppressed_count_ignores_successes_within_window() {
    let mut cache = CandidateOutcomeCache::with_staleness_window(50);
    cache.record("s1", "t1", "addSynapse", true, 0);
    cache.record("s2", "t2", "addSynapse", true, 10);
    cache.record("s3", "t3", "addSynapse", true, 20);
    assert_eq!(cache.suppressed_count(25), 0);
}

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
