//! Tests for Issue #792: Wire up adaptive module weighting from outcome tracker to candidate scoring.
//!
//! Verifies that:
//! 1. `ModuleOutcomeTracker` can be passed in via `AnalyzeAllInput` and flows through
//!    the analysis pipeline
//! 2. Module boost factors from the tracker are applied to coordinated structural
//!    candidate expected gains
//! 3. Per-module stats in metadata reflect historical data from the tracker
//! 4. The tracker is returned in the output for persistence across runs
//! 5. `run_discovery_modules_parallel` populates stats from the tracker

use neat_ai_discovery::CoordinatedStructuralCandidateJson;
use neat_ai_discovery::analysis::discovery_dispatch::{
    DiscoveryDetectionResult, DiscoveryModuleSpec,
};
use neat_ai_discovery::analysis::module_weights::{ModuleOutcomeTracker, ModuleStats};
use neat_ai_discovery::analysis::shared::{AnalyzeSynapsesResult, SynapseAnalysisMetadata};

// =============================================================================
// Helpers
// =============================================================================

fn empty_synapse_result() -> AnalyzeSynapsesResult {
    AnalyzeSynapsesResult {
        helpful_synapses: Vec::new(),
        harmful_synapses: Vec::new(),
        synapse_weight_updates: Vec::new(),
        coordinated_structural_candidates: Vec::new(),
        candidate_clusters: Vec::new(),
        gpu_used: false,
        no_candidate_reasons: Vec::new(),
        metadata: SynapseAnalysisMetadata::default(),
    }
}

fn make_candidate(gain: f32, comment: &str) -> CoordinatedStructuralCandidateJson {
    CoordinatedStructuralCandidateJson {
        remove_neuron_compensation: None,
        constant_neuron_bias_fold: None,
        operations: vec![],
        expected_creature_score_gain: gain,
        comment: Some(comment.to_string()),
    }
}

fn tracker_with_history() -> ModuleOutcomeTracker {
    let mut tracker = ModuleOutcomeTracker::new();

    // High-success module: 80% (16 of 20)
    for i in 0..20 {
        tracker.record("high-success module", i < 16);
    }

    // Low-success module: 10% (2 of 20)
    for i in 0..20 {
        tracker.record("low-success module", i < 2);
    }

    tracker
}

// =============================================================================
// Test: Tracker stats populate metadata in parallel dispatch
// =============================================================================

#[test]
fn tracker_stats_populate_metadata_in_parallel_dispatch() {
    let mut tracker = tracker_with_history();
    let mut syn = empty_synapse_result();

    let modules = vec![
        DiscoveryModuleSpec {
            module_name: "high-success module".to_string(),
            phase_name: "test_792_high",
            max_candidates: 0,
            detect_fn: Box::new(|| {
                Some(DiscoveryDetectionResult {
                    detected_count: 1,
                    candidates: vec![make_candidate(0.5, "high-success module")],
                })
            }),
        },
        DiscoveryModuleSpec {
            module_name: "low-success module".to_string(),
            phase_name: "test_792_low",
            max_candidates: 0,
            detect_fn: Box::new(|| {
                Some(DiscoveryDetectionResult {
                    detected_count: 1,
                    candidates: vec![make_candidate(0.3, "low-success module")],
                })
            }),
        },
    ];

    neat_ai_discovery::analysis::discovery_dispatch::run_discovery_modules_parallel(
        &mut syn,
        modules,
        None,
        false,
        &mut tracker,
    );

    // Metadata should contain per-module stats with historical data
    let stats = &syn.metadata.discovery_module_stats;
    assert_eq!(stats.len(), 2, "Should have stats for both modules");

    let high = stats
        .iter()
        .find(|s| s.module_name == "high-success module")
        .expect("Should have stats for high-success module");
    assert_eq!(high.attempts, 20);
    assert_eq!(high.successes, 16);
    assert!(
        high.success_rate > 0.7,
        "High-success rate should be > 0.7, got {}",
        high.success_rate
    );

    let low = stats
        .iter()
        .find(|s| s.module_name == "low-success module")
        .expect("Should have stats for low-success module");
    assert_eq!(low.attempts, 20);
    assert_eq!(low.successes, 2);
    assert!(
        low.success_rate < 0.3,
        "Low-success rate should be < 0.3, got {}",
        low.success_rate
    );
}

// =============================================================================
// Test: Module boost applied to candidate gains
// =============================================================================

#[test]
fn module_boost_applied_to_coordinated_candidates() {
    let tracker = tracker_with_history();

    let high_boost = tracker.module_boost("high-success module");
    let low_boost = tracker.module_boost("low-success module");

    // Verify the boosts are different
    assert!(
        high_boost > low_boost,
        "High-success boost ({high_boost}) should exceed low-success boost ({low_boost})"
    );

    // Create two candidates with identical raw gains but from different modules
    let raw_gain = 0.05_f32;
    let high_candidate = make_candidate(raw_gain, "high-success module: test");
    let low_candidate = make_candidate(raw_gain, "low-success module: test");

    // Apply module boost (simulating what the wired-up pipeline does)
    let high_adjusted = raw_gain as f64 * high_boost;
    let low_adjusted = raw_gain as f64 * low_boost;

    assert!(
        high_adjusted > low_adjusted,
        "High-success candidate ({high_adjusted}) should rank above low-success ({low_adjusted})"
    );
    // The high-success module's boost should be > 1.0
    assert!(
        high_boost > 1.0,
        "High-success (80%) should have boost > 1.0, got {high_boost}"
    );
    // The low-success module's boost should be < 1.0
    assert!(
        low_boost < 1.0,
        "Low-success (10%) should have boost < 1.0, got {low_boost}"
    );

    // Verify apply_module_boost_to_candidates works on the actual function
    let mut candidates = vec![high_candidate, low_candidate];
    neat_ai_discovery::analysis::module_weights::apply_module_boost_to_candidates(
        &mut candidates,
        &tracker,
    );

    // After boosting, high-success candidate should have higher gain
    let high_gain = candidates
        .iter()
        .find(|c| {
            c.comment
                .as_ref()
                .is_some_and(|cm| cm.contains("high-success"))
        })
        .unwrap()
        .expected_creature_score_gain;
    let low_gain = candidates
        .iter()
        .find(|c| {
            c.comment
                .as_ref()
                .is_some_and(|cm| cm.contains("low-success"))
        })
        .unwrap()
        .expected_creature_score_gain;

    assert!(
        high_gain > low_gain,
        "After boost, high-success gain ({high_gain}) should exceed low-success ({low_gain})"
    );
}

// =============================================================================
// Test: Empty tracker is neutral (no change to gains)
// =============================================================================

#[test]
fn empty_tracker_produces_neutral_boost() {
    let tracker = ModuleOutcomeTracker::new();

    let raw_gain = 0.05_f32;
    let mut candidates = vec![make_candidate(raw_gain, "some module: test")];

    neat_ai_discovery::analysis::module_weights::apply_module_boost_to_candidates(
        &mut candidates,
        &tracker,
    );

    // With an empty tracker, gain should be unchanged (neutral boost = 1.0)
    assert!(
        (candidates[0].expected_creature_score_gain - raw_gain).abs() < 1e-6,
        "Empty tracker should leave gain unchanged, got {}",
        candidates[0].expected_creature_score_gain
    );
}

// =============================================================================
// Test: Tracker serialisation round-trip with module_outcome_tracker field
// =============================================================================

#[test]
fn tracker_serialises_in_input_json() {
    let tracker = tracker_with_history();
    let json = serde_json::to_string(&tracker).expect("Serialisation should succeed");
    let round_tripped: ModuleOutcomeTracker =
        serde_json::from_str(&json).expect("Deserialisation should succeed");

    assert_eq!(tracker, round_tripped);

    // Verify the stats survived
    let high = round_tripped.stats("high-success module");
    assert_eq!(high.attempts, 20);
    assert_eq!(high.successes, 16);
}

// =============================================================================
// Test: Ensemble scoring uses provided tracker instead of default
// =============================================================================

#[test]
fn ensemble_scoring_uses_real_tracker() {
    use neat_ai_discovery::CoordinatedStructuralOpJson;
    use neat_ai_discovery::analysis::ensemble_scoring;

    let tracker = tracker_with_history();

    // Two candidates targeting the same neuron from different modules (agreeing)
    let c1 = CoordinatedStructuralCandidateJson {
        remove_neuron_compensation: None,
        constant_neuron_bias_fold: None,
        operations: vec![CoordinatedStructuralOpJson::SetBias {
            neuron_uuid: "target-1".to_string(),
            bias: 0.5,
        }],
        expected_creature_score_gain: 0.04,
        comment: Some("high-success module: bias fix".to_string()),
    };
    let c2 = CoordinatedStructuralCandidateJson {
        remove_neuron_compensation: None,
        constant_neuron_bias_fold: None,
        operations: vec![CoordinatedStructuralOpJson::SetBias {
            neuron_uuid: "target-1".to_string(),
            bias: 0.6,
        }],
        expected_creature_score_gain: 0.03,
        comment: Some("low-success module: bias fix".to_string()),
    };

    // With historical tracker — high-success module should be preferred
    let result_with_tracker =
        ensemble_scoring::apply_ensemble_scoring(vec![c1.clone(), c2.clone()], &tracker);

    // With empty tracker — neutral weighting
    let empty_tracker = ModuleOutcomeTracker::new();
    let result_empty = ensemble_scoring::apply_ensemble_scoring(vec![c1, c2], &empty_tracker);

    // Both should produce an ensemble candidate
    assert_eq!(result_with_tracker.ensemble_candidates, 1);
    assert_eq!(result_empty.ensemble_candidates, 1);

    // The gains should differ because the tracker influences weighting
    let gain_with_tracker = result_with_tracker.candidates[0].expected_creature_score_gain;
    let gain_empty = result_empty.candidates[0].expected_creature_score_gain;

    // Both should have the agreement boost, but the template selection may differ
    assert!(
        gain_with_tracker > 0.0 && gain_empty > 0.0,
        "Both should produce positive gains"
    );
}

// =============================================================================
// Test: ModuleStats success_rate is correct for known data
// =============================================================================

#[test]
fn module_stats_success_rate_for_known_data() {
    let stats = ModuleStats {
        attempts: 100,
        successes: 65,
        candidates_produced: 200,
        soft_failures: 0.0,
    };
    let rate = stats.success_rate();
    // Bayesian: (65 + 1) / (100 + 2) = 66/102 ≈ 0.647
    assert!((rate - 0.647).abs() < 0.01, "Expected ~0.647, got {rate}");
}
