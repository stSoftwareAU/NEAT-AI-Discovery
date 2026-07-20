//! Tests for Issue #1060: Enhance `ModuleOutcomeTracker` to learn from pre-filtering failures.
//!
//! Verifies:
//! 1. Soft failure recording in `ModuleOutcomeTracker`.
//! 2. Success rate incorporates soft failures.
//! 3. Module gating based on success rate threshold.
//! 4. Per-module gate status exposed in `DiscoveryModuleStatsJson`.
//! 5. Soft failures are decayed alongside real outcomes.
//! 6. Pre-filtering failures are recorded during `merge_discovery_module_results`.

use neat_ai_discovery::analysis::constants::{
    MIN_BOOST_SAMPLES, MODULE_GATE_THRESHOLD, SOFT_FAILURE_WEIGHT,
};
use neat_ai_discovery::analysis::discovery_dispatch::{
    DiscoveryDetectionResult, DiscoveryModuleSpec, detect_discovery_modules_parallel,
    run_discovery_modules_parallel,
};
use neat_ai_discovery::analysis::module_weights::{ModuleOutcomeTracker, ModuleStats};
use neat_ai_discovery::analysis::shared::{self, SynapseAnalysisMetadata};
use neat_ai_discovery::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson};

// =============================================================================
// Constants validation (compile-time)
// =============================================================================

const _: () = assert!(MODULE_GATE_THRESHOLD > 0.0);
const _: () = assert!(MODULE_GATE_THRESHOLD < 1.0);
const _: () = assert!(SOFT_FAILURE_WEIGHT > 0.0);
const _: () = assert!(SOFT_FAILURE_WEIGHT <= 1.0);

// =============================================================================
// Soft failure recording
// =============================================================================

#[test]
fn record_soft_failures_increases_soft_failure_count() {
    let mut tracker = ModuleOutcomeTracker::new();
    tracker.record_soft_failures("test-module", 5, 0.5);

    let stats = tracker.stats("test-module");
    assert!(
        (stats.soft_failures - 2.5).abs() < f64::EPSILON,
        "5 filtered candidates x 0.5 weight = 2.5 soft failures, got {}",
        stats.soft_failures
    );
}

#[test]
fn record_soft_failures_accumulates() {
    let mut tracker = ModuleOutcomeTracker::new();
    tracker.record_soft_failures("test-module", 3, 0.5);
    tracker.record_soft_failures("test-module", 7, 0.5);

    let stats = tracker.stats("test-module");
    assert!(
        (stats.soft_failures - 5.0).abs() < f64::EPSILON,
        "Expected 5.0 soft failures, got {}",
        stats.soft_failures
    );
}

#[test]
fn record_soft_failures_zero_count_is_noop() {
    let mut tracker = ModuleOutcomeTracker::new();
    tracker.record_soft_failures("test-module", 0, 0.5);

    assert!(
        tracker.is_empty(),
        "Zero count should not create a module entry"
    );
}

#[test]
fn record_soft_failures_zero_weight_is_noop() {
    let mut tracker = ModuleOutcomeTracker::new();
    tracker.record_soft_failures("test-module", 5, 0.0);

    assert!(
        tracker.is_empty(),
        "Zero weight should not create a module entry"
    );
}

#[test]
fn record_soft_failures_clamps_weight_to_one() {
    let mut tracker = ModuleOutcomeTracker::new();
    tracker.record_soft_failures("test-module", 10, 2.0);

    let stats = tracker.stats("test-module");
    // Weight is clamped to 1.0, so 10 x 1.0 = 10.0
    assert!(
        (stats.soft_failures - 10.0).abs() < f64::EPSILON,
        "Weight should be clamped to 1.0, expected 10.0 soft failures, got {}",
        stats.soft_failures
    );
}

// =============================================================================
// Success rate with soft failures
// =============================================================================

#[test]
fn success_rate_with_soft_failures_is_lower() {
    let mut tracker = ModuleOutcomeTracker::new();

    // Record 5 successes out of 10 attempts
    for _ in 0..5 {
        tracker.record("test-module", true);
        tracker.record("test-module", false);
    }

    let rate_without = tracker.stats("test-module").success_rate();

    // Add soft failures
    tracker.record_soft_failures("test-module", 20, 0.5);
    let rate_with = tracker.stats("test-module").success_rate();

    assert!(
        rate_with < rate_without,
        "Success rate with soft failures ({rate_with}) should be lower than without ({rate_without})"
    );
}

#[test]
fn success_rate_only_soft_failures_no_real_attempts() {
    let stats = ModuleStats {
        attempts: 0,
        successes: 0,
        candidates_produced: 0,
        soft_failures: 10.0,
    };
    let rate = stats.success_rate();
    // alpha = 0 + 1 = 1, beta = 0 + 10 + 1 = 11, rate = 1/12
    assert!(
        (rate - 1.0 / 12.0).abs() < 0.001,
        "Expected ~0.083, got {rate}"
    );
}

#[test]
fn success_rate_no_soft_failures_unchanged() {
    let stats = ModuleStats {
        attempts: 100,
        successes: 50,
        candidates_produced: 0,
        soft_failures: 0.0,
    };
    // alpha = 51, beta = 50 + 0 + 1 = 51, rate = 51/102 = 0.5
    let rate = stats.success_rate();
    assert!((rate - 0.5).abs() < 0.01, "Expected ~0.5, got {rate}");
}

// =============================================================================
// Module gating
// =============================================================================

#[test]
fn is_gated_returns_false_for_unknown_module() {
    let tracker = ModuleOutcomeTracker::new();
    assert!(
        !tracker.is_gated("unknown-module", MODULE_GATE_THRESHOLD),
        "Unknown module should not be gated"
    );
}

#[test]
fn is_gated_returns_false_with_insufficient_data() {
    let mut tracker = ModuleOutcomeTracker::new();
    for _ in 0..(MIN_BOOST_SAMPLES - 1) {
        tracker.record("test-module", false);
    }
    assert!(
        !tracker.is_gated("test-module", MODULE_GATE_THRESHOLD),
        "Module with insufficient data should not be gated"
    );
}

#[test]
fn is_gated_returns_true_for_very_low_success_rate() {
    let mut tracker = ModuleOutcomeTracker::new();
    // Record 500 failures, 0 successes: Bayesian rate = 1/502 ~ 0.002 < 0.005
    for _ in 0..500 {
        tracker.record("failing-module", false);
    }
    assert!(
        tracker.is_gated("failing-module", MODULE_GATE_THRESHOLD),
        "Module with near-zero success rate ({}) should be gated (threshold = {})",
        tracker.stats("failing-module").success_rate(),
        MODULE_GATE_THRESHOLD
    );
}

#[test]
fn is_gated_returns_false_for_healthy_success_rate() {
    let mut tracker = ModuleOutcomeTracker::new();
    // Record 50% success rate
    for _ in 0..50 {
        tracker.record("healthy-module", true);
        tracker.record("healthy-module", false);
    }
    assert!(
        !tracker.is_gated("healthy-module", MODULE_GATE_THRESHOLD),
        "Module with healthy success rate should not be gated"
    );
}

#[test]
fn is_gated_soft_failures_can_trigger_gating() {
    let mut tracker = ModuleOutcomeTracker::new();
    // Record 10 attempts with 1 success: Bayesian rate = 2/12 ~ 0.167 (above 0.005)
    for _ in 0..9 {
        tracker.record("test-module", false);
    }
    tracker.record("test-module", true);

    assert!(
        !tracker.is_gated("test-module", MODULE_GATE_THRESHOLD),
        "Should not be gated before soft failures"
    );

    // Add heavy soft failures to push rate below threshold
    // alpha = 2, beta = 9 + soft + 1. Need rate < 0.005
    // 2 / (2 + 9 + soft + 1) < 0.005 -> 2 < 0.005 * (12 + soft) -> soft > 388
    tracker.record_soft_failures("test-module", 800, 0.5);

    assert!(
        tracker.is_gated("test-module", MODULE_GATE_THRESHOLD),
        "Should be gated after heavy soft failures (rate = {})",
        tracker.stats("test-module").success_rate()
    );
}

#[test]
fn is_gated_default_uses_module_gate_threshold() {
    let mut tracker = ModuleOutcomeTracker::new();
    for _ in 0..100 {
        tracker.record("failing-module", false);
    }
    assert_eq!(
        tracker.is_gated_default("failing-module"),
        tracker.is_gated("failing-module", MODULE_GATE_THRESHOLD),
        "is_gated_default should use MODULE_GATE_THRESHOLD"
    );
}

// =============================================================================
// Decay includes soft failures
// =============================================================================

#[test]
fn apply_decay_reduces_soft_failures() {
    let mut tracker = ModuleOutcomeTracker::new();
    tracker.record_soft_failures("test-module", 100, 0.5); // 50.0 soft failures

    tracker.apply_decay(0.5);
    let stats = tracker.stats("test-module");
    assert!(
        (stats.soft_failures - 25.0).abs() < 0.01,
        "Soft failures should be halved by 0.5 decay, got {}",
        stats.soft_failures
    );
}

#[test]
fn apply_decay_zero_clears_soft_failures() {
    let mut tracker = ModuleOutcomeTracker::new();
    tracker.record_soft_failures("test-module", 100, 0.5);

    tracker.apply_decay(0.0);
    let stats = tracker.stats("test-module");
    assert!(
        stats.soft_failures.abs() < f64::EPSILON,
        "Soft failures should be zero after full decay, got {}",
        stats.soft_failures
    );
}

// =============================================================================
// Module gating in dispatch
// =============================================================================

fn empty_synapse_result() -> shared::AnalyzeSynapsesResult {
    shared::AnalyzeSynapsesResult {
        helpful_synapses: Vec::new(),
        harmful_synapses: Vec::new(),
        synapse_weight_updates: Vec::new(),
        coordinated_structural_candidates: Vec::new(),
        candidate_clusters: Vec::new(),
        gpu_used: false,
        no_candidate_reasons: Vec::new(),
        metadata: SynapseAnalysisMetadata {
            candidates_found: 0,
            candidates_returned: 0,
            ..Default::default()
        },
    }
}

fn make_candidate(gain: f32) -> CoordinatedStructuralCandidateJson {
    CoordinatedStructuralCandidateJson {
        remove_neuron_compensation: None,
        operations: vec![CoordinatedStructuralOpJson::RemoveSynapse {
            from_neuron_uuid: "a".to_string(),
            to_neuron_uuid: "b".to_string(),
        }],
        expected_creature_score_gain: gain,
        comment: Some(format!("test gain={gain}")),
    }
}

#[test]
fn gated_module_is_skipped_during_detection() {
    let mut tracker = ModuleOutcomeTracker::new();
    // Make "failing-module" gated: 500 failures, 0 successes
    // Bayesian rate = 1/502 ~ 0.002 < MODULE_GATE_THRESHOLD (0.005)
    for _ in 0..500 {
        tracker.record("failing-module", false);
    }
    // Make "healthy-module" not gated: 50% success
    for _ in 0..20 {
        tracker.record("healthy-module", true);
        tracker.record("healthy-module", false);
    }

    let modules = vec![
        DiscoveryModuleSpec {
            module_name: "failing-module".to_string(),
            phase_name: "test_fail",
            max_candidates: 0,
            detect_fn: Box::new(|| {
                Some(DiscoveryDetectionResult {
                    detected_count: 1,
                    candidates: vec![make_candidate(1.0)],
                })
            }),
        },
        DiscoveryModuleSpec {
            module_name: "healthy-module".to_string(),
            phase_name: "test_healthy",
            max_candidates: 0,
            detect_fn: Box::new(|| {
                Some(DiscoveryDetectionResult {
                    detected_count: 1,
                    candidates: vec![make_candidate(2.0)],
                })
            }),
        },
    ];

    let results = detect_discovery_modules_parallel(modules, None, Some(&tracker));

    assert_eq!(results.entries.len(), 2);

    // Failing module should be skipped (gated).
    assert!(
        results.entries[0].result.is_none(),
        "Gated module should produce None result"
    );

    // Healthy module should run normally.
    assert!(
        results.entries[1].result.is_some(),
        "Non-gated module should produce results"
    );
}

#[test]
fn gated_module_without_tracker_runs_normally() {
    // Without a tracker, no modules should be gated.
    let modules = vec![DiscoveryModuleSpec {
        module_name: "any-module".to_string(),
        phase_name: "test_any",
        max_candidates: 0,
        detect_fn: Box::new(|| {
            Some(DiscoveryDetectionResult {
                detected_count: 1,
                candidates: vec![make_candidate(1.0)],
            })
        }),
    }];

    let results = detect_discovery_modules_parallel(modules, None, None);

    assert_eq!(results.entries.len(), 1);
    assert!(
        results.entries[0].result.is_some(),
        "Without tracker, no modules should be gated"
    );
}

// =============================================================================
// Soft failure recording during merge
// =============================================================================

#[test]
fn merge_records_soft_failures_for_negative_gain_candidates() {
    let mut syn = empty_synapse_result();
    let mut tracker = ModuleOutcomeTracker::new();

    let modules = vec![DiscoveryModuleSpec {
        module_name: "test-module".to_string(),
        phase_name: "test_neg",
        max_candidates: 0,
        detect_fn: Box::new(|| {
            Some(DiscoveryDetectionResult {
                detected_count: 3,
                candidates: vec![
                    make_candidate(1.0),  // passes
                    make_candidate(-0.5), // filtered (negative gain)
                    make_candidate(0.0),  // filtered (zero gain)
                ],
            })
        }),
    }];

    run_discovery_modules_parallel(&mut syn, modules, None, false, &mut tracker);

    let stats = tracker.stats("test-module");
    // 2 candidates filtered out x SOFT_FAILURE_WEIGHT
    assert!(
        stats.soft_failures > 0.0,
        "Soft failures should be recorded for filtered candidates, got {}",
        stats.soft_failures
    );
    assert!(
        (stats.soft_failures - 2.0 * SOFT_FAILURE_WEIGHT).abs() < f64::EPSILON,
        "Expected {} soft failures, got {}",
        2.0 * SOFT_FAILURE_WEIGHT,
        stats.soft_failures
    );
}

#[test]
fn merge_records_soft_failures_for_truncated_candidates() {
    let mut syn = empty_synapse_result();
    let mut tracker = ModuleOutcomeTracker::new();

    // Module produces 5 candidates but has budget of 2.
    let modules = vec![DiscoveryModuleSpec {
        module_name: "test-module".to_string(),
        phase_name: "test_trunc",
        max_candidates: 2,
        detect_fn: Box::new(|| {
            Some(DiscoveryDetectionResult {
                detected_count: 5,
                candidates: vec![
                    make_candidate(5.0),
                    make_candidate(4.0),
                    make_candidate(3.0),
                    make_candidate(2.0),
                    make_candidate(1.0),
                ],
            })
        }),
    }];

    run_discovery_modules_parallel(&mut syn, modules, None, false, &mut tracker);

    let stats = tracker.stats("test-module");
    // 3 candidates truncated x SOFT_FAILURE_WEIGHT
    assert!(
        stats.soft_failures > 0.0,
        "Soft failures should be recorded for truncated candidates"
    );
    assert!(
        (stats.soft_failures - 3.0 * SOFT_FAILURE_WEIGHT).abs() < f64::EPSILON,
        "Expected {} soft failures (3 truncated), got {}",
        3.0 * SOFT_FAILURE_WEIGHT,
        stats.soft_failures
    );
}

// =============================================================================
// Gate status in metadata
// =============================================================================

#[test]
fn discovery_module_stats_include_gate_status() {
    let mut tracker = ModuleOutcomeTracker::new();
    // Populate tracker with a failing module and a healthy module.
    // 500 failures: Bayesian rate = 1/502 ~ 0.002 < MODULE_GATE_THRESHOLD (0.005)
    for _ in 0..500 {
        tracker.record("failing-module", false);
    }
    for _ in 0..20 {
        tracker.record("healthy-module", true);
        tracker.record("healthy-module", false);
    }

    let mut syn = empty_synapse_result();
    let modules = vec![
        DiscoveryModuleSpec {
            module_name: "failing-module".to_string(),
            phase_name: "test_fail_meta",
            max_candidates: 0,
            detect_fn: Box::new(|| {
                Some(DiscoveryDetectionResult {
                    detected_count: 1,
                    candidates: vec![make_candidate(1.0)],
                })
            }),
        },
        DiscoveryModuleSpec {
            module_name: "healthy-module".to_string(),
            phase_name: "test_healthy_meta",
            max_candidates: 0,
            detect_fn: Box::new(|| {
                Some(DiscoveryDetectionResult {
                    detected_count: 1,
                    candidates: vec![make_candidate(2.0)],
                })
            }),
        },
    ];

    run_discovery_modules_parallel(&mut syn, modules, None, false, &mut tracker);

    let stats = &syn.metadata.discovery_module_stats;
    assert_eq!(stats.len(), 2, "Should have stats for both modules");

    let failing = stats
        .iter()
        .find(|s| s.module_name == "failing-module")
        .expect("Should have stats for failing-module");
    assert!(
        failing.gated,
        "Failing module should be marked as gated (rate={}, threshold={}, attempts={}, successes={}, soft_failures={})",
        failing.success_rate,
        MODULE_GATE_THRESHOLD,
        failing.attempts,
        failing.successes,
        failing.soft_failures,
    );

    let healthy = stats
        .iter()
        .find(|s| s.module_name == "healthy-module")
        .expect("Should have stats for healthy-module");
    assert!(
        !healthy.gated,
        "Healthy module should not be marked as gated"
    );
}

#[test]
fn discovery_module_stats_include_soft_failures() {
    let mut tracker = ModuleOutcomeTracker::new();
    tracker.record_soft_failures("test-module", 10, 0.5);

    let mut syn = empty_synapse_result();
    let modules = vec![DiscoveryModuleSpec {
        module_name: "test-module".to_string(),
        phase_name: "test_soft_meta",
        max_candidates: 0,
        detect_fn: Box::new(|| {
            Some(DiscoveryDetectionResult {
                detected_count: 1,
                candidates: vec![make_candidate(1.0)],
            })
        }),
    }];

    run_discovery_modules_parallel(&mut syn, modules, None, false, &mut tracker);

    let stats = &syn.metadata.discovery_module_stats;
    let module_stats = stats
        .iter()
        .find(|s| s.module_name == "test-module")
        .expect("Should have stats for test-module");
    assert!(
        module_stats.soft_failures > 0.0,
        "Soft failures should be exposed in metadata"
    );
}
