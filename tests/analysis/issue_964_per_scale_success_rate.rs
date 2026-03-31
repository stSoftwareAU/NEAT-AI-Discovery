//! Tests for Issue #964: Per-scale success rate tracking for weight variant selection.
//!
//! The `ScaleOutcomeTracker` records outcomes keyed by `(module_name, weight_scale_tier)`
//! and computes Bayesian success rates per scale tier. This enables biasing future
//! variant generation toward historically successful scales.
//!
//! ## Key Behaviours Verified
//!
//! - Recording per-scale outcomes (success/failure)
//! - Computing Bayesian success rates per (module, scale) combination
//! - Providing per-scale boost factors clamped to [0.5, 2.0]
//! - Decay factor prevents stale data dominance
//! - Serialisation/deserialisation for persistence
//! - Applying per-scale boosts to variant `expected_multiplier` values
//! - Extracting scale tier from candidate comments

use neat_ai_discovery::analysis::scale_outcomes::{
    ScaleOutcomeTracker, ScaleStats, apply_scale_boosts_to_candidates,
};

// =============================================================================
// Basic Recording and Lookup
// =============================================================================

#[test]
fn empty_tracker_has_no_stats() {
    let tracker = ScaleOutcomeTracker::new();
    assert!(tracker.is_empty());
    let stats = tracker.stats("saturation detection", "Conservative");
    assert_eq!(stats.attempts, 0);
    assert_eq!(stats.successes, 0);
}

#[test]
fn record_outcome_and_retrieve_stats() {
    let mut tracker = ScaleOutcomeTracker::new();
    tracker.record("saturation detection", "Conservative", true);
    tracker.record("saturation detection", "Conservative", true);
    tracker.record("saturation detection", "Conservative", false);

    let stats = tracker.stats("saturation detection", "Conservative");
    assert_eq!(stats.attempts, 3);
    assert_eq!(stats.successes, 2);
}

#[test]
fn different_scales_tracked_independently() {
    let mut tracker = ScaleOutcomeTracker::new();
    tracker.record("saturation detection", "Conservative", true);
    tracker.record("saturation detection", "Micro-Nudge", false);
    tracker.record("saturation detection", "Micro-Nudge", false);

    let conservative = tracker.stats("saturation detection", "Conservative");
    assert_eq!(conservative.attempts, 1);
    assert_eq!(conservative.successes, 1);

    let micro = tracker.stats("saturation detection", "Micro-Nudge");
    assert_eq!(micro.attempts, 2);
    assert_eq!(micro.successes, 0);
}

#[test]
fn different_modules_tracked_independently() {
    let mut tracker = ScaleOutcomeTracker::new();
    tracker.record("saturation detection", "Conservative", true);
    tracker.record("dead neuron detection", "Conservative", false);

    let sat = tracker.stats("saturation detection", "Conservative");
    assert_eq!(sat.successes, 1);

    let dead = tracker.stats("dead neuron detection", "Conservative");
    assert_eq!(dead.successes, 0);
}

// =============================================================================
// Bayesian Success Rate
// =============================================================================

#[test]
fn bayesian_success_rate_with_no_data_returns_prior() {
    let stats = ScaleStats::default();
    let rate = stats.success_rate();
    assert!(
        (rate - 0.5).abs() < f64::EPSILON,
        "Empty stats should return prior 0.5, got {rate}"
    );
}

#[test]
fn bayesian_success_rate_converges_to_raw_rate() {
    let stats = ScaleStats {
        attempts: 1000,
        successes: 450,
    };
    let rate = stats.success_rate();
    assert!(
        (rate - 0.45).abs() < 0.01,
        "With 1000 samples at 45% raw rate, Bayesian rate should be ~0.45, got {rate}"
    );
}

#[test]
fn bayesian_rate_never_exactly_zero_or_one() {
    let all_fail = ScaleStats {
        attempts: 100,
        successes: 0,
    };
    assert!(all_fail.success_rate() > 0.0);

    let all_succeed = ScaleStats {
        attempts: 100,
        successes: 100,
    };
    assert!(all_succeed.success_rate() < 1.0);
}

// =============================================================================
// Per-Scale Boost Factor
// =============================================================================

#[test]
fn boost_neutral_when_insufficient_data() {
    let mut tracker = ScaleOutcomeTracker::new();
    // Record fewer than MIN_BOOST_SAMPLES outcomes
    for _ in 0..5 {
        tracker.record("module-a", "Conservative", true);
    }
    let boost = tracker.scale_boost("module-a", "Conservative");
    assert!(
        (boost - 1.0).abs() < f64::EPSILON,
        "Insufficient data should give neutral boost 1.0, got {boost}"
    );
}

#[test]
fn boost_reflects_scale_success_rate() {
    let mut tracker = ScaleOutcomeTracker::new();

    // High-success scale: 80% (8 of 10)
    for i in 0..10 {
        tracker.record("module-a", "Conservative", i < 8);
    }

    // Low-success scale: 20% (2 of 10)
    for i in 0..10 {
        tracker.record("module-a", "Micro-Nudge", i < 2);
    }

    let high_boost = tracker.scale_boost("module-a", "Conservative");
    let low_boost = tracker.scale_boost("module-a", "Micro-Nudge");

    assert!(
        high_boost > low_boost,
        "High-success boost ({high_boost}) should be > low-success boost ({low_boost})"
    );
    assert!(
        high_boost > 1.0,
        "80% success should give boost > 1.0, got {high_boost}"
    );
    assert!(
        low_boost < 1.0,
        "20% success should give boost < 1.0, got {low_boost}"
    );
}

#[test]
fn boost_clamped_within_bounds() {
    let mut tracker = ScaleOutcomeTracker::new();

    for _ in 0..20 {
        tracker.record("module-a", "Conservative", true);
    }
    for _ in 0..20 {
        tracker.record("module-a", "Micro-Nudge", false);
    }

    let perfect = tracker.scale_boost("module-a", "Conservative");
    let failure = tracker.scale_boost("module-a", "Micro-Nudge");

    assert!(
        perfect <= 2.0,
        "Perfect boost should be clamped to <= 2.0, got {perfect}"
    );
    assert!(
        failure >= 0.5,
        "Failure boost should be clamped to >= 0.5, got {failure}"
    );
}

#[test]
fn unknown_module_or_scale_gets_neutral_boost() {
    let tracker = ScaleOutcomeTracker::new();
    let boost = tracker.scale_boost("nonexistent", "Conservative");
    assert!(
        (boost - 1.0).abs() < f64::EPSILON,
        "Unknown module/scale should get neutral boost 1.0, got {boost}"
    );
}

// =============================================================================
// Decay Factor
// =============================================================================

#[test]
fn decay_reduces_counts_proportionally() {
    let mut tracker = ScaleOutcomeTracker::new();
    for _ in 0..20 {
        tracker.record("module-a", "Conservative", true);
    }
    for _ in 0..10 {
        tracker.record("module-a", "Conservative", false);
    }

    let before = tracker.stats("module-a", "Conservative");
    assert_eq!(before.attempts, 30);
    assert_eq!(before.successes, 20);

    tracker.apply_decay(0.5);

    let after = tracker.stats("module-a", "Conservative");
    // 30 * 0.5 = 15, 20 * 0.5 = 10
    assert_eq!(after.attempts, 15);
    assert_eq!(after.successes, 10);
}

#[test]
fn decay_preserves_success_rate_approximately() {
    let mut tracker = ScaleOutcomeTracker::new();
    for _ in 0..100 {
        tracker.record("module-a", "Conservative", true);
    }
    for _ in 0..100 {
        tracker.record("module-a", "Conservative", false);
    }

    let rate_before = tracker.stats("module-a", "Conservative").success_rate();
    tracker.apply_decay(0.5);
    let rate_after = tracker.stats("module-a", "Conservative").success_rate();

    assert!(
        (rate_before - rate_after).abs() < 0.05,
        "Decay should approximately preserve success rate: before={rate_before}, after={rate_after}"
    );
}

#[test]
fn decay_clamps_factor_to_valid_range() {
    let mut tracker = ScaleOutcomeTracker::new();
    for _ in 0..10 {
        tracker.record("module-a", "Conservative", true);
    }

    // Factor > 1.0 should be clamped to 1.0 (no change)
    tracker.apply_decay(2.0);
    let stats = tracker.stats("module-a", "Conservative");
    assert_eq!(stats.attempts, 10);

    // Factor < 0.0 should be clamped to 0.0 (full reset)
    tracker.apply_decay(-1.0);
    let stats = tracker.stats("module-a", "Conservative");
    assert_eq!(stats.attempts, 0);
}

#[test]
fn decay_ensures_successes_never_exceed_attempts() {
    let mut tracker = ScaleOutcomeTracker::new();
    // 3 successes out of 5 attempts — after decay with rounding, successes
    // could potentially exceed attempts without the safety clamp.
    for _ in 0..3 {
        tracker.record("module-a", "Conservative", true);
    }
    for _ in 0..2 {
        tracker.record("module-a", "Conservative", false);
    }

    tracker.apply_decay(0.3);
    let stats = tracker.stats("module-a", "Conservative");
    assert!(
        stats.successes <= stats.attempts,
        "Successes ({}) should never exceed attempts ({})",
        stats.successes,
        stats.attempts,
    );
}

// =============================================================================
// Serialisation Round-Trip
// =============================================================================

#[test]
fn serialisation_round_trip() {
    let mut tracker = ScaleOutcomeTracker::new();
    tracker.record("saturation detection", "Conservative", true);
    tracker.record("saturation detection", "Conservative", false);
    tracker.record("saturation detection", "Micro-Nudge", true);
    tracker.record("dead neuron detection", "Whisper", false);

    let json = serde_json::to_string(&tracker).expect("Serialisation should succeed");
    let deserialized: ScaleOutcomeTracker =
        serde_json::from_str(&json).expect("Deserialisation should succeed");

    assert_eq!(tracker, deserialized);

    let sat_cons = deserialized.stats("saturation detection", "Conservative");
    assert_eq!(sat_cons.attempts, 2);
    assert_eq!(sat_cons.successes, 1);

    let sat_micro = deserialized.stats("saturation detection", "Micro-Nudge");
    assert_eq!(sat_micro.attempts, 1);
    assert_eq!(sat_micro.successes, 1);

    let dead_whisper = deserialized.stats("dead neuron detection", "Whisper");
    assert_eq!(dead_whisper.attempts, 1);
    assert_eq!(dead_whisper.successes, 0);
}

// =============================================================================
// Scale Tier Extraction from Candidate Comments
// =============================================================================

#[test]
fn extract_scale_tier_from_variant_comments() {
    use neat_ai_discovery::analysis::scale_outcomes::extract_scale_tier;

    assert_eq!(
        extract_scale_tier("Conservative variant (clamped incoming/bias, outgoing scaled)"),
        Some("Conservative")
    );
    assert_eq!(
        extract_scale_tier("Gentle Nudge variant (tight outgoing, bias tamed)"),
        Some("Gentle Nudge")
    );
    assert_eq!(
        extract_scale_tier(
            "Micro-Nudge variant (ultra-conservative outgoing, tight incoming/bias)"
        ),
        Some("Micro-Nudge")
    );
    assert_eq!(
        extract_scale_tier(
            "Feather-Touch variant (near-equilibrium outgoing, minimal perturbation)"
        ),
        Some("Feather-Touch")
    );
    assert_eq!(
        extract_scale_tier("Whisper variant (minimal outgoing, near-zero perturbation)"),
        Some("Whisper")
    );
    assert_eq!(extract_scale_tier("some random comment"), None);
}

// =============================================================================
// Apply Scale Boosts to Coordinated Structural Candidates
// =============================================================================

#[test]
fn apply_scale_boosts_modifies_expected_multiplier() {
    let mut tracker = ScaleOutcomeTracker::new();

    // Conservative has high success rate
    for _ in 0..20 {
        tracker.record("test module", "Conservative", true);
    }
    // Micro-Nudge has low success rate
    for _ in 0..20 {
        tracker.record("test module", "Micro-Nudge", false);
    }

    let mut candidates = vec![
        make_candidate(
            1.0,
            "test module: Conservative variant (weight scaled to 0.5\u{00d7})",
        ),
        make_candidate(
            1.0,
            "test module: Micro-Nudge variant (weight scaled to 0.1\u{00d7})",
        ),
    ];

    apply_scale_boosts_to_candidates(&mut candidates, &tracker);

    // Conservative should be boosted (gain > 1.0)
    assert!(
        candidates[0].expected_creature_score_gain > 1.0,
        "Conservative should be boosted, got {}",
        candidates[0].expected_creature_score_gain
    );

    // Micro-Nudge should be penalised (gain < 1.0)
    assert!(
        candidates[1].expected_creature_score_gain < 1.0,
        "Micro-Nudge should be penalised, got {}",
        candidates[1].expected_creature_score_gain
    );
}

#[test]
fn apply_scale_boosts_no_effect_on_empty_tracker() {
    let tracker = ScaleOutcomeTracker::new();
    let mut candidates = vec![make_candidate(
        1.0,
        "test module: Conservative variant (weight scaled to 0.5\u{00d7})",
    )];

    apply_scale_boosts_to_candidates(&mut candidates, &tracker);

    assert!(
        (candidates[0].expected_creature_score_gain - 1.0).abs() < f32::EPSILON,
        "Empty tracker should not modify candidates, got {}",
        candidates[0].expected_creature_score_gain
    );
}

#[test]
fn apply_scale_boosts_ignores_candidates_without_scale_tier() {
    let mut tracker = ScaleOutcomeTracker::new();
    for _ in 0..20 {
        tracker.record("test module", "Conservative", true);
    }

    let mut candidates = vec![make_candidate(
        1.0,
        "test module: some detection without variant tier",
    )];

    apply_scale_boosts_to_candidates(&mut candidates, &tracker);

    assert!(
        (candidates[0].expected_creature_score_gain - 1.0).abs() < f32::EPSILON,
        "Candidates without scale tier should not be modified, got {}",
        candidates[0].expected_creature_score_gain
    );
}

#[test]
fn module_count_tracks_distinct_modules() {
    let mut tracker = ScaleOutcomeTracker::new();
    tracker.record("module-a", "Conservative", true);
    tracker.record("module-a", "Micro-Nudge", false);
    tracker.record("module-b", "Conservative", true);

    assert_eq!(tracker.module_count(), 2);
}

#[test]
fn all_stats_returns_nested_structure() {
    let mut tracker = ScaleOutcomeTracker::new();
    tracker.record("module-a", "Conservative", true);
    tracker.record("module-a", "Micro-Nudge", false);

    let all = tracker.all_stats();
    assert!(all.contains_key("module-a"));
    let module_a = &all["module-a"];
    assert!(module_a.contains_key("Conservative"));
    assert!(module_a.contains_key("Micro-Nudge"));
}

// =============================================================================
// Test Helpers
// =============================================================================

fn make_candidate(
    gain: f32,
    comment: &str,
) -> neat_ai_discovery::CoordinatedStructuralCandidateJson {
    neat_ai_discovery::CoordinatedStructuralCandidateJson {
        operations: vec![],
        expected_creature_score_gain: gain,
        comment: Some(comment.to_string()),
    }
}
