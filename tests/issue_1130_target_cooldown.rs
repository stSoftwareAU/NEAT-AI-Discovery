//! Integration tests for target-neuron cooldown after repeated discovery
//! failures (Issue #1130).
//!
//! Verifies that the `TargetFailureTracker` produced by
//! `candidate_cache`/`target_failure_tracker` correctly drops focus targets
//! that have accumulated consecutive failures within the cooldown window, so
//! preparation layers no longer waste the discovery budget probing dead
//! targets like `neuron-1063112866`.

use neat_ai_discovery::analysis::constants::{
    TARGET_COOLDOWN_CONSECUTIVE_FAILURES, TARGET_COOLDOWN_EPOCHS,
};
use neat_ai_discovery::analysis::target_failure_tracker::{
    TargetFailureTracker, filter_cooldown_targets,
};

/// Issue #1130: three consecutive failures on the same target cause it to be
/// skipped on the next "discovery run" within the cooldown window. Other
/// targets in the focus list remain.
#[test]
fn three_consecutive_failures_skip_target_in_cooldown_window() {
    let mut tracker = TargetFailureTracker::with_thresholds(3, 10);

    // Simulate the production failure pattern: 3 failures on the same target
    // across epochs 0..=2.
    for epoch in 0..3u64 {
        tracker.record_failure("neuron-1063112866", epoch);
    }

    // Next "discovery run" at epoch 3 — well within the 10-epoch window.
    let mut focus = vec![
        "neuron-1063112866".to_string(),
        "neuron-healthy".to_string(),
    ];
    let skipped = filter_cooldown_targets(&mut focus, &tracker, 3);

    assert_eq!(skipped, 1, "exactly one target should be skipped");
    assert_eq!(
        focus,
        vec!["neuron-healthy".to_string()],
        "only the cooldown target should be removed"
    );
}

/// Issue #1130: a target with fewer than the threshold number of failures is
/// NOT in cooldown.
#[test]
fn fewer_than_threshold_failures_do_not_trigger_cooldown() {
    let mut tracker = TargetFailureTracker::with_thresholds(3, 10);
    tracker.record_failure("T", 0);
    tracker.record_failure("T", 1);

    let mut focus = vec!["T".to_string()];
    let skipped = filter_cooldown_targets(&mut focus, &tracker, 2);

    assert_eq!(skipped, 0);
    assert_eq!(focus, vec!["T".to_string()]);
}

/// Issue #1130: after the cooldown window elapses, a previously-in-cooldown
/// target becomes eligible again.
#[test]
fn target_released_after_cooldown_window() {
    let mut tracker = TargetFailureTracker::with_thresholds(3, 10);
    for epoch in 0..3u64 {
        tracker.record_failure("T", epoch);
    }

    // At epoch 12 (2 + 10) the target is released.
    let mut focus = vec!["T".to_string()];
    let skipped = filter_cooldown_targets(&mut focus, &tracker, 12);
    assert_eq!(skipped, 0);
    assert_eq!(focus, vec!["T".to_string()]);
}

/// Issue #1130: a successful improvement resets the consecutive-failure
/// counter immediately and clears cooldown.
#[test]
fn success_clears_cooldown_immediately() {
    let mut tracker = TargetFailureTracker::with_thresholds(3, 10);
    for epoch in 0..3u64 {
        tracker.record_failure("T", epoch);
    }
    assert!(tracker.is_in_cooldown("T", 3));

    tracker.record_success("T", 3);

    // Still within the failure window but the counter has been reset.
    let mut focus = vec!["T".to_string()];
    let skipped = filter_cooldown_targets(&mut focus, &tracker, 3);
    assert_eq!(skipped, 0);
    assert_eq!(focus, vec!["T".to_string()]);
}

/// Issue #1130: compiled defaults match the documented proposal.
#[test]
fn default_thresholds_match_proposal() {
    assert_eq!(
        TARGET_COOLDOWN_CONSECUTIVE_FAILURES, 3,
        "proposed default is 3 consecutive failures"
    );
    assert_eq!(
        TARGET_COOLDOWN_EPOCHS, 10,
        "proposed default is a 10-epoch cooldown window"
    );
}

/// Issue #1130: the cooldown mirrors the production failure signature — 17 of
/// 18 entries targeting the same neuron. Once the 3rd consecutive failure
/// lands, the remaining 14+ attempts on that target are skipped.
#[test]
fn production_failure_pattern_saves_budget_after_third_failure() {
    let mut tracker = TargetFailureTracker::with_thresholds(3, 10);
    let hot_target = "neuron-1063112866";

    // Simulate 18 failure attempts on the same target across epochs 0..=17.
    let mut rejected_after_cooldown = 0u32;
    for epoch in 0..18u64 {
        let mut focus = vec![hot_target.to_string()];
        let skipped = filter_cooldown_targets(&mut focus, &tracker, epoch);
        if skipped > 0 {
            rejected_after_cooldown += 1;
            // The preparation layer would not submit this target for evaluation;
            // in the real pipeline this saves GPU budget.
            continue;
        }
        // Otherwise the run "ran" and failed.
        tracker.record_failure(hot_target, epoch);
    }

    // 3 failures to trigger cooldown at epoch 2, then epochs 3..=12 are
    // skipped (10 runs). At epoch 13 the window reopens — the run fails and
    // triggers cooldown again (streak is still at 4+). Exact count depends on
    // the window but must be substantially greater than zero and less than
    // the total number of attempts.
    assert!(
        rejected_after_cooldown > 0,
        "cooldown should save at least one attempt"
    );
    assert!(
        rejected_after_cooldown < 18,
        "cooldown should not skip every attempt"
    );
}
