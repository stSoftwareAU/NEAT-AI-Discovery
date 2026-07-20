//! Tests for Issue #485: Adaptive discovery module weighting based on candidate success rates.
//!
//! The module weights tracker records per-module accept/reject outcomes and computes
//! rolling success rates with Bayesian smoothing. This enables:
//!
//! 1. Tracking which discovery modules produce accepted candidates
//! 2. Computing per-module success rates with Bayesian smoothing
//! 3. Providing boost factors for deadline-constrained prioritisation
//! 4. Reporting per-module effectiveness in analysis metadata
//!
//! ## Key Behaviours Verified
//!
//! - Recording per-module outcomes (success/failure)
//! - Computing Bayesian success rates per module
//! - Providing boost factors based on success rates
//! - Serialisation/deserialisation for persistence
//! - No module is entirely starved (minimum exploration budget)

use neat_ai_discovery::analysis::module_weights::{ModuleOutcomeTracker, ModuleStats};

// =============================================================================
// Basic Recording and Lookup
// =============================================================================

#[test]
fn empty_tracker_has_no_stats() {
    let tracker = ModuleOutcomeTracker::new();
    assert!(tracker.is_empty());
    assert_eq!(tracker.module_count(), 0);
    let stats = tracker.stats("saturation detection");
    assert_eq!(stats.attempts, 0);
    assert_eq!(stats.successes, 0);
}

#[test]
fn record_success_and_retrieve_stats() {
    let mut tracker = ModuleOutcomeTracker::new();
    tracker.record("saturation detection", true);
    tracker.record("saturation detection", true);
    tracker.record("saturation detection", false);

    let stats = tracker.stats("saturation detection");
    assert_eq!(stats.attempts, 3);
    assert_eq!(stats.successes, 2);
}

#[test]
fn multiple_modules_tracked_independently() {
    let mut tracker = ModuleOutcomeTracker::new();
    tracker.record("saturation detection", true);
    tracker.record("dead neuron detection", false);
    tracker.record("dead neuron detection", false);

    assert_eq!(tracker.module_count(), 2);

    let sat_stats = tracker.stats("saturation detection");
    assert_eq!(sat_stats.attempts, 1);
    assert_eq!(sat_stats.successes, 1);

    let dead_stats = tracker.stats("dead neuron detection");
    assert_eq!(dead_stats.attempts, 2);
    assert_eq!(dead_stats.successes, 0);
}

// =============================================================================
// Bayesian Success Rate
// =============================================================================

#[test]
fn bayesian_success_rate_with_no_data_returns_prior() {
    let stats = ModuleStats::default();
    // Beta(1,1) prior → 0.5
    let rate = stats.success_rate();
    assert!(
        (rate - 0.5).abs() < f64::EPSILON,
        "Empty stats should return prior 0.5, got {rate}"
    );
}

#[test]
fn bayesian_success_rate_converges_to_raw_rate() {
    let stats = ModuleStats {
        attempts: 1000,
        successes: 360,
        candidates_produced: 0,
        soft_failures: 0.0,
    };
    let rate = stats.success_rate();
    // Should be close to 0.36 with 1000 samples
    assert!(
        (rate - 0.36).abs() < 0.01,
        "With 1000 samples at 36% raw rate, Bayesian rate should be ~0.36, got {rate}"
    );
}

#[test]
fn bayesian_rate_never_exactly_zero_or_one() {
    let all_fail = ModuleStats {
        attempts: 100,
        successes: 0,
        candidates_produced: 0,
        soft_failures: 0.0,
    };
    assert!(all_fail.success_rate() > 0.0);

    let all_succeed = ModuleStats {
        attempts: 100,
        successes: 100,
        candidates_produced: 0,
        soft_failures: 0.0,
    };
    assert!(all_succeed.success_rate() < 1.0);
}

// =============================================================================
// Boost Factor
// =============================================================================

#[test]
fn boost_factor_neutral_when_insufficient_data() {
    let mut tracker = ModuleOutcomeTracker::new();
    // Record fewer than MIN_BOOST_SAMPLES outcomes
    for i in 0..5 {
        tracker.record(&format!("module-{i}"), true);
    }
    // Each module has only 1 outcome, well below MIN_BOOST_SAMPLES
    let boost = tracker.module_boost("module-0");
    assert!(
        (boost - 1.0).abs() < f64::EPSILON,
        "Insufficient data should give neutral boost 1.0, got {boost}"
    );
}

#[test]
fn boost_factor_reflects_success_rate() {
    let mut tracker = ModuleOutcomeTracker::new();

    // High-success module: 80% (8 of 10)
    for i in 0..10 {
        tracker.record("high-success", i < 8);
    }

    // Low-success module: 20% (2 of 10)
    for i in 0..10 {
        tracker.record("low-success", i < 2);
    }

    let high_boost = tracker.module_boost("high-success");
    let low_boost = tracker.module_boost("low-success");

    assert!(
        high_boost > low_boost,
        "High-success boost ({high_boost}) should be > low-success boost ({low_boost})"
    );

    // High boost should be > 1.0 (positive signal)
    assert!(
        high_boost > 1.0,
        "80% success should give boost > 1.0, got {high_boost}"
    );

    // Low boost should be < 1.0 (penalty signal)
    assert!(
        low_boost < 1.0,
        "20% success should give boost < 1.0, got {low_boost}"
    );
}

#[test]
fn boost_factor_clamped_within_bounds() {
    let mut tracker = ModuleOutcomeTracker::new();

    // Perfect success: should be clamped at max
    for _ in 0..20 {
        tracker.record("perfect", true);
    }

    // Total failure: should be clamped at min
    for _ in 0..20 {
        tracker.record("failure", false);
    }

    let perfect_boost = tracker.module_boost("perfect");
    let failure_boost = tracker.module_boost("failure");

    assert!(
        perfect_boost <= 2.0,
        "Perfect boost should be clamped to <= 2.0, got {perfect_boost}"
    );
    assert!(
        failure_boost >= 0.5,
        "Failure boost should be clamped to >= 0.5, got {failure_boost}"
    );
}

#[test]
fn unknown_module_gets_neutral_boost() {
    let tracker = ModuleOutcomeTracker::new();
    let boost = tracker.module_boost("nonexistent");
    assert!(
        (boost - 1.0).abs() < f64::EPSILON,
        "Unknown module should get neutral boost 1.0, got {boost}"
    );
}

// =============================================================================
// Batch Recording
// =============================================================================

#[test]
fn record_candidates_count() {
    let mut tracker = ModuleOutcomeTracker::new();
    tracker.record_candidates("bottleneck detection", 5);

    let stats = tracker.stats("bottleneck detection");
    assert_eq!(stats.candidates_produced, 5);

    tracker.record_candidates("bottleneck detection", 3);
    let stats = tracker.stats("bottleneck detection");
    assert_eq!(stats.candidates_produced, 8);
}

// =============================================================================
// All Module Stats Summary
// =============================================================================

#[test]
fn all_stats_returns_all_tracked_modules() {
    let mut tracker = ModuleOutcomeTracker::new();
    tracker.record("module-a", true);
    tracker.record("module-b", false);
    tracker.record("module-c", true);

    let all = tracker.all_stats();
    assert_eq!(all.len(), 3);
    assert!(all.contains_key("module-a"));
    assert!(all.contains_key("module-b"));
    assert!(all.contains_key("module-c"));
}

// =============================================================================
// Serialisation Round-Trip
// =============================================================================

#[test]
fn serialisation_round_trip() {
    let mut tracker = ModuleOutcomeTracker::new();
    tracker.record("saturation detection", true);
    tracker.record("saturation detection", false);
    tracker.record("dead neuron detection", true);
    tracker.record_candidates("saturation detection", 3);

    let json = serde_json::to_string(&tracker).expect("Serialisation should succeed");
    let deserialized: ModuleOutcomeTracker =
        serde_json::from_str(&json).expect("Deserialisation should succeed");

    assert_eq!(tracker.module_count(), deserialized.module_count());

    let sat = deserialized.stats("saturation detection");
    assert_eq!(sat.attempts, 2);
    assert_eq!(sat.successes, 1);
    assert_eq!(sat.candidates_produced, 3);

    let dead = deserialized.stats("dead neuron detection");
    assert_eq!(dead.attempts, 1);
    assert_eq!(dead.successes, 1);
}

// =============================================================================
// Integration: Discovery Module Stats in Metadata
// =============================================================================

#[test]
fn discovery_module_stats_populated_after_parallel_dispatch() {
    // Verify that after run_discovery_modules_parallel, the synapse result
    // contains per-module stats in its metadata.
    use neat_ai_discovery::analysis::discovery_dispatch::{
        DiscoveryDetectionResult, DiscoveryModuleSpec,
    };
    let mut syn = empty_synapse_result();

    let modules = vec![
        DiscoveryModuleSpec {
            module_name: "test module A".to_string(),
            phase_name: "test_a",
            max_candidates: 0,
            detect_fn: Box::new(|| {
                Some(DiscoveryDetectionResult {
                    detected_count: 2,
                    candidates: vec![make_candidate(0.5), make_candidate(0.3)],
                })
            }),
        },
        DiscoveryModuleSpec {
            module_name: "test module B".to_string(),
            phase_name: "test_b",
            max_candidates: 0,
            detect_fn: Box::new(|| None), // No detections
        },
        DiscoveryModuleSpec {
            module_name: "test module C".to_string(),
            phase_name: "test_c",
            max_candidates: 0,
            detect_fn: Box::new(|| {
                Some(DiscoveryDetectionResult {
                    detected_count: 1,
                    candidates: vec![make_candidate(0.1)],
                })
            }),
        },
    ];

    let mut tracker = ModuleOutcomeTracker::new();
    neat_ai_discovery::analysis::discovery_dispatch::run_discovery_modules_parallel(
        &mut syn,
        modules,
        None,
        false,
        &mut tracker,
    );

    // Metadata should now contain per-module stats
    let module_stats = &syn.metadata.discovery_module_stats;
    assert!(!module_stats.is_empty(), "Module stats should be populated");

    // Module A produced 2 candidates
    let a_stats = module_stats
        .iter()
        .find(|s| s.module_name == "test module A");
    assert!(a_stats.is_some(), "Should have stats for test module A");
    let a = a_stats.unwrap();
    assert_eq!(a.candidates_produced, 2);

    // Module B produced 0 candidates (returned None)
    let b_stats = module_stats
        .iter()
        .find(|s| s.module_name == "test module B");
    assert!(b_stats.is_some(), "Should have stats for test module B");
    assert_eq!(b_stats.unwrap().candidates_produced, 0);

    // Module C produced 1 candidate
    let c_stats = module_stats
        .iter()
        .find(|s| s.module_name == "test module C");
    assert!(c_stats.is_some(), "Should have stats for test module C");
    assert_eq!(c_stats.unwrap().candidates_produced, 1);
}

// =============================================================================
// Test Helpers
// =============================================================================

fn empty_synapse_result() -> neat_ai_discovery::analysis::shared::AnalyzeSynapsesResult {
    neat_ai_discovery::analysis::shared::AnalyzeSynapsesResult {
        helpful_synapses: Vec::new(),
        harmful_synapses: Vec::new(),
        synapse_weight_updates: Vec::new(),
        coordinated_structural_candidates: Vec::new(),
        candidate_clusters: Vec::new(),
        gpu_used: false,
        no_candidate_reasons: Vec::new(),
        metadata: neat_ai_discovery::analysis::shared::SynapseAnalysisMetadata::default(),
    }
}

fn make_candidate(gain: f32) -> neat_ai_discovery::CoordinatedStructuralCandidateJson {
    neat_ai_discovery::CoordinatedStructuralCandidateJson {
        remove_neuron_compensation: None,
        operations: vec![],
        expected_creature_score_gain: gain,
        comment: Some("test candidate".to_string()),
    }
}
