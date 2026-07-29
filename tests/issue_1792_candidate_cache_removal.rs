//! Issue #1792 — `CandidateOutcomeCache` was never constructed outside tests,
//! so every production reference passed `None`.
//!
//! Decision (B): delete it. These tests are the regression guard that the
//! always-`None` parameter and the structurally-always-`0` reporting fields do
//! not come back:
//!
//! 1. The drought reset takes no cache argument and still clears target
//!    cooldowns — the one thing it actually does.
//! 2. `DroughtResetOutcome` carries no cache-clearing field (exhaustive
//!    destructuring fails to compile if one is re-added).
//! 3. The emitted `droughtDiagnostic` JSON carries no `candidateCacheSize` /
//!    `candidateCacheSuppressedCount` keys that could only ever report `0`.
//! 4. `DroughtInputs` has no cache field (exhaustive construction).

use neat_ai_discovery::analysis::diagnostics::RejectionBreakdown;
use neat_ai_discovery::analysis::discovery_mode::DiscoveryMode;
use neat_ai_discovery::analysis::drought_diagnostic::{DroughtInputs, emit_drought_diagnostic};
use neat_ai_discovery::analysis::drought_reset::{
    DroughtResetOutcome, maybe_perform_drought_reset, rearm_drought_reset,
};
use neat_ai_discovery::analysis::target_failure_tracker::TargetFailureTracker;

/// Two targets in cooldown plus one below-threshold target.
fn populated_tracker() -> TargetFailureTracker {
    let mut tracker = TargetFailureTracker::with_thresholds(2, 100);
    tracker.record_failure("A", 0);
    tracker.record_failure("A", 1);
    tracker.record_failure("B", 0);
    tracker.record_failure("B", 1);
    tracker.record_failure("C", 0);
    tracker
}

/// The reset fires through the cache-free signature and clears the cooldowns.
/// Exhaustive destructuring of the outcome is the compile-time guard that no
/// cache-clearing field survives.
#[test]
fn drought_reset_has_no_cache_parameter_and_still_clears_cooldowns() {
    let mut tracker = populated_tracker();

    let outcome =
        maybe_perform_drought_reset(Some(&mut tracker), 10, 10, 5).expect("fires at threshold");

    // Exhaustive — no `..`. Re-adding `candidate_cache_failed_cleared` breaks
    // this pattern and fails the build.
    let DroughtResetOutcome {
        target_cooldown_cleared,
        reset_epoch,
        drought_reset_after,
    } = outcome;

    assert_eq!(target_cooldown_cleared, 2, "both cooldown targets cleared");
    assert_eq!(reset_epoch, 5);
    assert_eq!(drought_reset_after, 10);

    // The below-threshold target survives; the tombstone is stamped.
    assert_eq!(tracker.len(), 1);
    assert!(tracker.state("C").is_some());
    assert_eq!(tracker.drought_reset_tombstone(), Some(5));
}

/// Re-arming takes only the tracker and clears its tombstone.
#[test]
fn rearm_takes_tracker_only() {
    let mut tracker = populated_tracker();
    let _ = maybe_perform_drought_reset(Some(&mut tracker), 10, 10, 5).expect("fires");
    assert!(tracker.drought_reset_tombstone().is_some());

    rearm_drought_reset(Some(&mut tracker));
    assert!(tracker.drought_reset_tombstone().is_none());
}

/// The serialised diagnostic must not carry candidate-cache counters. Before
/// the removal these keys were emitted on every drought and were structurally
/// always `0`, which made the diagnostic read as "the cache is empty" rather
/// than "there is no cache".
#[test]
fn drought_diagnostic_json_has_no_candidate_cache_keys() {
    let mut tracker = TargetFailureTracker::with_thresholds(2, 100);
    tracker.record_failure("tgt-x", 0);
    tracker.record_failure("tgt-x", 1);

    let mut breakdown = RejectionBreakdown::new();
    breakdown.record_many("below_threshold", 3);

    // Exhaustive construction — re-adding a `candidate_cache` field fails here.
    let inputs = DroughtInputs {
        consecutive_failures: 6,
        rolling_success_rate: 0.0,
        discovery_mode: DiscoveryMode::Conservative,
        target_tracker: Some(&tracker),
        current_epoch: 1,
        target_cooldown_skipped: 0,
        rejection_breakdown: &breakdown,
        candidates_returned: 1,
    };

    let diagnostic = emit_drought_diagnostic(&inputs, 5).expect("emits at threshold");
    let json = serde_json::to_value(&diagnostic).expect("diagnostic serialises");
    let object = json.as_object().expect("diagnostic is a JSON object");

    assert!(
        !object.contains_key("candidateCacheSize"),
        "removed field re-appeared: {object:?}"
    );
    assert!(
        !object.contains_key("candidateCacheSuppressedCount"),
        "removed field re-appeared: {object:?}"
    );

    // The signals that are genuinely wired are still reported.
    assert_eq!(object["targetCooldownActiveCount"], 1);
    assert_eq!(object["consecutiveFailures"], 6);
}

/// The deleted module must not be reachable from the public API surface.
/// `analysis::candidate_cache` no longer exists, so the only thing left to
/// assert behaviourally is that the drought reset reports nothing about it.
#[test]
fn reset_without_tracker_reports_only_cooldown_work() {
    let outcome = maybe_perform_drought_reset(None, 10, 10, 5).expect("fires with no tracker");
    assert_eq!(outcome.target_cooldown_cleared, 0);
    assert_eq!(outcome.reset_epoch, 5);
}
