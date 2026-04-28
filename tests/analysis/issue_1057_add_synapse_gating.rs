//! Tests for Issue #1057: Add-synapse candidate gating to reduce failure rate.
//!
//! Validates that add-synapse candidates are gated out when:
//! 1. The `ModuleOutcomeTracker` shows a historical success rate below the
//!    configurable threshold (default <1%) for the `"add-synapse"` module.
//! 2. The creature's synapse density (synapses/neurons) exceeds a threshold.
//!
//! ## Key Behaviours Verified
//!
//! - Gating is inactive with insufficient historical data
//! - Gating activates when success rate drops below threshold
//! - Density gate skips dense networks
//! - Custom thresholds are respected
//! - Gating clears `helpful_synapses` but not harmful or coordinated candidates

use neat_ai_discovery::CandidateSynapseJson;
use neat_ai_discovery::analysis::module_weights::ModuleOutcomeTracker;
use neat_ai_discovery::analysis::synapse::add_synapse_gating::{
    ADD_SYNAPSE_MODULE_NAME, DEFAULT_ADD_SYNAPSE_SUCCESS_THRESHOLD,
    DEFAULT_SYNAPSE_DENSITY_THRESHOLD, gate_add_synapse_candidates, should_skip_add_synapse,
    should_skip_add_synapse_by_density, should_skip_add_synapse_by_outcome,
};
use neat_ai_discovery::ffi_types::{CreatureJson, NeuronJson, SynapseJson};

// =============================================================================
// Test Helpers
// =============================================================================

fn make_creature(neuron_count: usize, synapse_count: usize, input_count: usize) -> CreatureJson {
    let neurons: Vec<NeuronJson> = (0..neuron_count)
        .map(|i| NeuronJson {
            uuid: format!("neuron-{i}"),
            neuron_type: "hidden".to_string(),
            squash: "TANH".to_string(),
            bias: 0.0,
        })
        .collect();
    let synapses: Vec<SynapseJson> = (0..synapse_count)
        .map(|i| SynapseJson {
            from_uuid: format!("input-{}", i % input_count.max(1)),
            to_uuid: format!("neuron-{}", i % neuron_count.max(1)),
            weight: 0.5,
            synapse_type: None,
        })
        .collect();
    CreatureJson {
        neurons,
        synapses,
        input: input_count,
        output: 1,
    }
}

fn make_test_candidate(from: &str, to: &str) -> CandidateSynapseJson {
    CandidateSynapseJson {
        from_neuron_uuid: from.to_string(),
        to_neuron_uuid: to.to_string(),
        from_neuron_index: None,
        to_neuron_index: None,
        weight: 0.5,
        target_neuron_impact: 1.0,
        expected_creature_error_reduction: 0.01,
        expected_creature_score_gain: 0.01,
        improved_count: 5,
        total_count: 10,
        improvement_magnitude_ratio: None,
        target_neuron_stats: None,
        outlier_reduction_info: None,
        prediction_confidence: 0.5,
        expected_score_gain_confidence_interval: [0.005, 0.015],
        comment: None,
    }
}

fn tracker_with_failures(count: u32) -> ModuleOutcomeTracker {
    let mut tracker = ModuleOutcomeTracker::new();
    for _ in 0..count {
        tracker.record(ADD_SYNAPSE_MODULE_NAME, false);
    }
    tracker
}

fn tracker_with_mixed(successes: u32, failures: u32) -> ModuleOutcomeTracker {
    let mut tracker = ModuleOutcomeTracker::new();
    for _ in 0..successes {
        tracker.record(ADD_SYNAPSE_MODULE_NAME, true);
    }
    for _ in 0..failures {
        tracker.record(ADD_SYNAPSE_MODULE_NAME, false);
    }
    tracker
}

// =============================================================================
// Outcome Tracker Gate
// =============================================================================

#[test]
fn outcome_gate_inactive_with_no_data() {
    let tracker = ModuleOutcomeTracker::new();
    assert!(!should_skip_add_synapse_by_outcome(
        &tracker,
        DEFAULT_ADD_SYNAPSE_SUCCESS_THRESHOLD
    ));
}

#[test]
fn outcome_gate_inactive_with_sparse_data() {
    // 9 attempts is below MIN_BOOST_SAMPLES (10)
    let tracker = tracker_with_failures(9);
    assert!(!should_skip_add_synapse_by_outcome(
        &tracker,
        DEFAULT_ADD_SYNAPSE_SUCCESS_THRESHOLD
    ));
}

#[test]
fn outcome_gate_activates_with_zero_successes() {
    // 100 failures, 0 successes → Bayesian rate = 1/102 ≈ 0.0098 < 0.01
    let tracker = tracker_with_failures(100);
    assert!(should_skip_add_synapse_by_outcome(
        &tracker,
        DEFAULT_ADD_SYNAPSE_SUCCESS_THRESHOLD
    ));
}

#[test]
fn outcome_gate_inactive_with_moderate_success() {
    // 5 successes, 95 failures → Bayesian rate = 6/102 ≈ 0.0588 > 0.01
    let tracker = tracker_with_mixed(5, 95);
    assert!(!should_skip_add_synapse_by_outcome(
        &tracker,
        DEFAULT_ADD_SYNAPSE_SUCCESS_THRESHOLD
    ));
}

#[test]
fn outcome_gate_custom_threshold() {
    // 5 successes, 95 failures → Bayesian rate ≈ 0.0588
    let tracker = tracker_with_mixed(5, 95);
    // With 10% threshold, should skip (5.88% < 10%)
    assert!(should_skip_add_synapse_by_outcome(&tracker, 0.10));
    // With 1% threshold, should not skip (5.88% > 1%)
    assert!(!should_skip_add_synapse_by_outcome(&tracker, 0.01));
}

#[test]
fn outcome_gate_ignores_other_modules() {
    let mut tracker = ModuleOutcomeTracker::new();
    for _ in 0..100 {
        tracker.record("saturation", false);
    }
    // add-synapse module has no data
    assert!(!should_skip_add_synapse_by_outcome(
        &tracker,
        DEFAULT_ADD_SYNAPSE_SUCCESS_THRESHOLD
    ));
}

// =============================================================================
// Density Gate
// =============================================================================

#[test]
fn density_gate_inactive_for_sparse_network() {
    // 10 neurons + 5 inputs = 15, 20 synapses → ratio 1.33
    let creature = make_creature(10, 20, 5);
    assert!(!should_skip_add_synapse_by_density(
        &creature,
        DEFAULT_SYNAPSE_DENSITY_THRESHOLD
    ));
}

#[test]
fn density_gate_activates_for_dense_network() {
    // 100 neurons + 10 inputs = 110, 2000 synapses → ratio 18.18
    let creature = make_creature(100, 2000, 10);
    assert!(should_skip_add_synapse_by_density(
        &creature,
        DEFAULT_SYNAPSE_DENSITY_THRESHOLD
    ));
}

#[test]
fn density_gate_boundary_at_threshold() {
    // Exactly at threshold (14.0) → not above, should not skip
    let creature = make_creature(100, 1400, 0);
    assert!(!should_skip_add_synapse_by_density(
        &creature,
        DEFAULT_SYNAPSE_DENSITY_THRESHOLD
    ));

    // Just above threshold → should skip
    let creature = make_creature(100, 1401, 0);
    assert!(should_skip_add_synapse_by_density(
        &creature,
        DEFAULT_SYNAPSE_DENSITY_THRESHOLD
    ));
}

#[test]
fn density_gate_safe_for_empty_creature() {
    let creature = make_creature(0, 0, 0);
    assert!(!should_skip_add_synapse_by_density(
        &creature,
        DEFAULT_SYNAPSE_DENSITY_THRESHOLD
    ));
}

// =============================================================================
// Combined Gate
// =============================================================================

#[test]
fn combined_gate_requires_at_least_one_trigger() {
    let tracker = ModuleOutcomeTracker::new(); // no data
    let creature = make_creature(10, 20, 5); // sparse
    assert!(!should_skip_add_synapse(
        &tracker,
        &creature,
        DEFAULT_ADD_SYNAPSE_SUCCESS_THRESHOLD,
        DEFAULT_SYNAPSE_DENSITY_THRESHOLD,
    ));
}

#[test]
fn combined_gate_triggers_on_outcome_alone() {
    let tracker = tracker_with_failures(100);
    let creature = make_creature(10, 20, 5); // sparse
    assert!(should_skip_add_synapse(
        &tracker,
        &creature,
        DEFAULT_ADD_SYNAPSE_SUCCESS_THRESHOLD,
        DEFAULT_SYNAPSE_DENSITY_THRESHOLD,
    ));
}

#[test]
fn combined_gate_triggers_on_density_alone() {
    let tracker = ModuleOutcomeTracker::new(); // no data
    let creature = make_creature(100, 2000, 10); // dense
    assert!(should_skip_add_synapse(
        &tracker,
        &creature,
        DEFAULT_ADD_SYNAPSE_SUCCESS_THRESHOLD,
        DEFAULT_SYNAPSE_DENSITY_THRESHOLD,
    ));
}

// =============================================================================
// Candidate Gating (end-to-end)
// =============================================================================

#[test]
fn gate_clears_helpful_synapses_when_triggered() {
    let tracker = tracker_with_failures(100);
    let creature = make_creature(10, 20, 5);

    let mut helpful = vec![
        make_test_candidate("input-0", "neuron-1"),
        make_test_candidate("input-1", "neuron-2"),
        make_test_candidate("input-2", "neuron-3"),
    ];

    let removed = gate_add_synapse_candidates(&mut helpful, &tracker, &creature);
    assert_eq!(removed, 3);
    assert!(helpful.is_empty());
}

#[test]
fn gate_preserves_helpful_synapses_when_not_triggered() {
    let tracker = ModuleOutcomeTracker::new();
    let creature = make_creature(10, 20, 5);

    let mut helpful = vec![
        make_test_candidate("input-0", "neuron-1"),
        make_test_candidate("input-1", "neuron-2"),
    ];

    let removed = gate_add_synapse_candidates(&mut helpful, &tracker, &creature);
    assert_eq!(removed, 0);
    assert_eq!(helpful.len(), 2);
}

#[test]
fn gate_returns_zero_for_empty_input() {
    let tracker = tracker_with_failures(100);
    let creature = make_creature(10, 20, 5);

    let mut helpful: Vec<CandidateSynapseJson> = Vec::new();
    let removed = gate_add_synapse_candidates(&mut helpful, &tracker, &creature);
    assert_eq!(removed, 0);
}

// =============================================================================
// Constants Sanity Checks
// =============================================================================

#[test]
fn module_name_is_stable() {
    // The module name must remain "add-synapse" for cross-run tracker persistence.
    assert_eq!(ADD_SYNAPSE_MODULE_NAME, "add-synapse");
}
