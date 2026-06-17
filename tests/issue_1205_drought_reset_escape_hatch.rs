//! Issue #1205 — operator escape hatch: forced cache & cooldown reset after a
//! configurable drought duration.
//!
//! Integration tests for the one-shot drought reset that operates on the
//! `CandidateOutcomeCache` and `TargetFailureTracker`. Mirrors the acceptance
//! scenario in the issue body:
//!
//! - configure the env var to 10
//! - drive 12 forced-empty passes
//! - assert the reset fires exactly once at pass 10 and not again at pass 11/12
//! - verify the cache's source-type statistics survive the reset.

use neat_ai_discovery::analysis::candidate_cache::CandidateOutcomeCache;
use neat_ai_discovery::analysis::drought_reset::maybe_perform_drought_reset;
use neat_ai_discovery::analysis::target_failure_tracker::TargetFailureTracker;

/// Twelve forced-empty passes with `drought_reset_after = 10` fire exactly
/// one reset, at pass 10, and leave subsequent passes untouched.
#[test]
fn twelve_empty_passes_with_threshold_ten_fires_exactly_once() {
    let mut cache = CandidateOutcomeCache::new();
    cache.record_with_source_type("s1", "t1", "addSynapse", false, 0, "hidden");
    cache.record_with_source_type("s2", "t2", "addSynapse", false, 0, "input");
    cache.record_with_source_type("s3", "t3", "addSynapse", true, 0, "input");

    let mut tracker = TargetFailureTracker::with_thresholds(2, 100);
    tracker.record_failure("A", 0);
    tracker.record_failure("A", 1);
    tracker.record_failure("B", 0);
    tracker.record_failure("B", 1);

    let drought_reset_after: u32 = 10;
    let mut fire_count = 0_u32;
    let mut fire_at_pass: Vec<u32> = Vec::new();
    let mut cleared_failed_at_fire: usize = 0;
    let mut cleared_cooldown_at_fire: usize = 0;

    // Pass index simulates the orchestrator's `consecutive_failures` counter.
    for pass in 1..=12_u32 {
        let outcome = maybe_perform_drought_reset(
            Some(&mut cache),
            Some(&mut tracker),
            pass,
            drought_reset_after,
            u64::from(pass),
        );
        if let Some(o) = outcome {
            fire_count += 1;
            fire_at_pass.push(pass);
            cleared_failed_at_fire = o.candidate_cache_failed_cleared;
            cleared_cooldown_at_fire = o.target_cooldown_cleared;
        }
    }

    assert_eq!(
        fire_count, 1,
        "reset must fire exactly once across the streak"
    );
    assert_eq!(fire_at_pass, vec![10], "reset must fire at pass 10 only");
    assert_eq!(
        cleared_failed_at_fire, 2,
        "both failed cache entries cleared"
    );
    assert_eq!(cleared_cooldown_at_fire, 2, "both cooldown targets cleared");

    // The successful entry survived.
    assert_eq!(cache.len(), 1);
    assert!(
        cache
            .get_outcome("s3", "t3", "addSynapse")
            .is_some_and(|o| o.succeeded)
    );

    // Source-type stats preserved across the reset (institutional memory).
    let stats_input = cache.source_type_stats("input");
    assert_eq!(stats_input.attempts, 2);
    assert_eq!(stats_input.successes, 1);
    let stats_hidden = cache.source_type_stats("hidden");
    assert_eq!(stats_hidden.attempts, 1);
    assert_eq!(stats_hidden.successes, 0);

    // Tombstones remain set across the streak.
    assert_eq!(cache.drought_reset_tombstone(), Some(10));
    assert_eq!(tracker.drought_reset_tombstone(), Some(10));
}

/// After a successful pass clears the tombstone, a fresh drought re-arms the
/// lever and the reset can fire a second time.
#[test]
fn successful_pass_rearms_lever_for_next_drought() {
    let mut cache = CandidateOutcomeCache::new();
    cache.record("s1", "t1", "addSynapse", false, 0);
    let mut tracker = TargetFailureTracker::with_thresholds(2, 100);
    tracker.record_failure("A", 0);
    tracker.record_failure("A", 1);

    // First drought streak fires the reset.
    let first = maybe_perform_drought_reset(Some(&mut cache), Some(&mut tracker), 10, 10, 10)
        .expect("first reset fires");
    assert_eq!(first.candidate_cache_failed_cleared, 1);
    assert_eq!(first.target_cooldown_cleared, 1);

    // Successful outcome re-arms the lever on both surfaces.
    cache.record("good-src", "good-tgt", "addSynapse", true, 11);
    tracker.record_success("good-tgt", 11);
    assert!(cache.drought_reset_tombstone().is_none());
    assert!(tracker.drought_reset_tombstone().is_none());

    // Second drought streak.
    cache.record("s2", "t2", "addSynapse", false, 12);
    tracker.record_failure("B", 12);
    tracker.record_failure("B", 13);
    let second = maybe_perform_drought_reset(Some(&mut cache), Some(&mut tracker), 10, 10, 14)
        .expect("second reset fires after re-arm");
    assert_eq!(second.candidate_cache_failed_cleared, 1);
    assert_eq!(second.target_cooldown_cleared, 1);
}

/// When the operator explicitly opts out (env var `0`), the lever is disabled
/// and never fires — `drought_reset_after_epochs()` returns `None` and the
/// helper is not called. This test exercises the disabled-path directly via
/// `drought_reset_after = 0`, matching how a disabled env var is forwarded by
/// the orchestrator. (Since Issue #1422 the lever is armed by default when
/// unset; `0` is the deliberate opt-out.)
#[test]
fn lever_disabled_when_threshold_zero() {
    let mut cache = CandidateOutcomeCache::new();
    cache.record("s1", "t1", "addSynapse", false, 0);
    let mut tracker = TargetFailureTracker::with_thresholds(2, 100);
    tracker.record_failure("A", 0);
    tracker.record_failure("A", 1);

    for pass in 1..=50_u32 {
        let outcome = maybe_perform_drought_reset(
            Some(&mut cache),
            Some(&mut tracker),
            pass,
            0,
            u64::from(pass),
        );
        assert!(outcome.is_none(), "disabled lever must never fire");
    }
    assert_eq!(cache.len(), 1, "cache untouched when lever disabled");
    assert_eq!(tracker.len(), 1, "tracker untouched when lever disabled");
}
