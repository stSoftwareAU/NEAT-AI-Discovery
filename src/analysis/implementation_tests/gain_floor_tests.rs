//! Absolute expected-gain floor tests (Issue #1191).
//!
//! Acceptance criteria from the issue:
//! - Constructs a candidate set with mixed gain magnitudes and asserts
//!   only those at or above the floor survive.
//! - Asserts the `candidates_below_gain_floor` counter increments by the
//!   correct amount.

use crate::CandidateNeuronJson;
use crate::CandidateSynapseJson;
use crate::analysis::constants::{
    MIN_EXPECTED_CREATURE_SCORE_GAIN, min_expected_creature_score_gain,
};
use crate::analysis::neuron::post_processing::apply_min_expected_gain_floor_for_neurons;
use crate::analysis::synapse::post_processing::apply_min_expected_gain_floor_for_synapses;
use crate::observability::global_gain_floor_metrics;
use serial_test::serial;

fn neuron_candidate(gain: f32) -> CandidateNeuronJson {
    CandidateNeuronJson {
        source_neuron_uuid: format!("source-{gain}"),
        target_neuron_uuid: format!("target-{gain}"),
        source_neuron_index: None,
        target_neuron_index: None,
        incoming_weight: 1.0,
        outgoing_weight: 1.0,
        squash: format!("SQUASH-{gain}"),
        bias: 0.0,
        comment: None,
        target_neuron_impact: 1.0,
        expected_creature_error_reduction: gain,
        expected_creature_score_gain: gain,
        improved_count: 10,
        total_count: 10,
        improvement_magnitude_ratio: None,
        target_neuron_stats: None,
        prediction_confidence: 0.5,
        expected_score_gain_confidence_interval: [gain, gain],
        target_saturation_factor: None,
        variant_key: None,
    }
}

fn synapse_candidate(gain: f32) -> CandidateSynapseJson {
    CandidateSynapseJson {
        from_neuron_uuid: format!("source-{gain}"),
        to_neuron_uuid: format!("target-{gain}"),
        from_neuron_index: None,
        to_neuron_index: None,
        weight: 1.0,
        target_neuron_impact: 1.0,
        expected_creature_error_reduction: gain,
        expected_creature_score_gain: gain,
        improved_count: 10,
        total_count: 10,
        improvement_magnitude_ratio: None,
        target_neuron_stats: None,
        outlier_reduction_info: None,
        prediction_confidence: 0.5,
        expected_score_gain_confidence_interval: [gain, gain],
        comment: None,
        variant_key: None,
    }
}

#[test]
fn neuron_floor_drops_only_below_threshold_candidates() {
    // Mixed gain magnitudes: some well below 1e-5, some at the floor, some
    // comfortably above.
    let mut candidates = vec![
        neuron_candidate(6.6e-7),                           // noise — drop
        neuron_candidate(6.8e-7),                           // noise — drop
        neuron_candidate(3.1e-6),                           // below floor — drop
        neuron_candidate(MIN_EXPECTED_CREATURE_SCORE_GAIN), // exactly at floor — keep
        neuron_candidate(5e-4),                             // well above — keep
        neuron_candidate(1e-2),                             // well above — keep
    ];

    let dropped = apply_min_expected_gain_floor_for_neurons(&mut candidates);

    assert_eq!(dropped, 3, "three noise-level candidates must be dropped");
    assert_eq!(candidates.len(), 3, "three candidates must survive");
    for c in &candidates {
        assert!(
            c.expected_creature_score_gain >= MIN_EXPECTED_CREATURE_SCORE_GAIN,
            "surviving candidate gain {} must be >= floor {}",
            c.expected_creature_score_gain,
            MIN_EXPECTED_CREATURE_SCORE_GAIN
        );
    }
}

#[test]
fn synapse_floor_drops_only_below_threshold_candidates() {
    let mut candidates = vec![
        synapse_candidate(1e-8),
        synapse_candidate(9.9e-6), // just below floor — drop
        synapse_candidate(MIN_EXPECTED_CREATURE_SCORE_GAIN), // at floor — keep
        synapse_candidate(2e-3),
    ];

    let dropped = apply_min_expected_gain_floor_for_synapses(&mut candidates);

    assert_eq!(dropped, 2);
    assert_eq!(candidates.len(), 2);
    for c in &candidates {
        assert!(c.expected_creature_score_gain >= MIN_EXPECTED_CREATURE_SCORE_GAIN);
    }
}

#[test]
fn neuron_floor_keeps_pool_intact_when_all_above_floor() {
    let mut candidates = vec![
        neuron_candidate(0.5),
        neuron_candidate(0.1),
        neuron_candidate(2e-5),
    ];
    let original_len = candidates.len();
    let dropped = apply_min_expected_gain_floor_for_neurons(&mut candidates);
    assert_eq!(
        dropped, 0,
        "no candidates should be dropped above the floor"
    );
    assert_eq!(candidates.len(), original_len);
}

#[test]
fn neuron_floor_empty_input_is_safe() {
    let mut candidates: Vec<CandidateNeuronJson> = Vec::new();
    let dropped = apply_min_expected_gain_floor_for_neurons(&mut candidates);
    assert_eq!(dropped, 0);
    assert!(candidates.is_empty());
}

#[test]
fn counter_increments_on_floor_drops() {
    let metrics = global_gain_floor_metrics();
    let start = metrics.candidates_below_gain_floor_total();

    let mut candidates = vec![
        neuron_candidate(1e-8),
        neuron_candidate(2e-7),
        neuron_candidate(0.5),
    ];
    let dropped = apply_min_expected_gain_floor_for_neurons(&mut candidates);

    assert_eq!(dropped, 2);
    let after = metrics.candidates_below_gain_floor_total();
    assert!(
        after >= start + 2,
        "counter must increment by at least the dropped count (start={start}, after={after})"
    );
}

#[test]
#[serial]
fn env_override_controls_floor() {
    // SAFETY: env access is serialised via `#[serial]`.
    unsafe {
        std::env::set_var("NEAT_AI_DISCOVERY_MIN_EXPECTED_GAIN", "1e-3");
    }
    let configured = min_expected_creature_score_gain();

    // SAFETY: env access is serialised via `#[serial]`.
    unsafe {
        std::env::remove_var("NEAT_AI_DISCOVERY_MIN_EXPECTED_GAIN");
    }

    assert!(
        (configured - 1e-3).abs() < 1e-9,
        "override should be honoured (got {configured})"
    );
}

#[test]
#[serial]
fn env_override_clamps_above_ceiling() {
    // SAFETY: env access is serialised via `#[serial]`.
    unsafe {
        std::env::set_var("NEAT_AI_DISCOVERY_MIN_EXPECTED_GAIN", "1.0");
    }
    let configured = min_expected_creature_score_gain();

    // SAFETY: env access is serialised via `#[serial]`.
    unsafe {
        std::env::remove_var("NEAT_AI_DISCOVERY_MIN_EXPECTED_GAIN");
    }

    // Clamped to ceiling 1e-2 — must not exceed it.
    assert!(
        configured <= 1e-2 + 1e-9,
        "override above ceiling must be clamped (got {configured})"
    );
}
