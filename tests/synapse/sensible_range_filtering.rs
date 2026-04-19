//! Regression tests for "sensible range" filtering.
//!
//! We cache failures and do not retry them, so it is better to avoid proposing
//! candidates with absurd weights/biases up front. These tests ensure we never
//! return out-of-range add-neuron candidates.

use neat_ai_discovery::CandidateNeuronJson;
use neat_ai_discovery::analysis::utils::{
    filter_candidates_to_sensible_ranges, pair_extreme_candidates_with_conservative_variants,
};

#[test]
fn extreme_candidate_is_not_returned_after_sensible_range_filtering() {
    // An intentionally extreme candidate: large incoming and bias.
    let extreme = CandidateNeuronJson {
        source_neuron_uuid: "source-1".to_string(),
        target_neuron_uuid: "target-1".to_string(),
        source_neuron_index: None,
        target_neuron_index: None,
        incoming_weight: 200.0,
        outgoing_weight: 0.1,
        squash: "TANH".to_string(),
        bias: 50.0,
        comment: None,
        target_neuron_impact: 1.0,
        expected_creature_error_reduction: 0.2,
        expected_creature_score_gain: 0.2,
        improved_count: 10,
        total_count: 20,
        target_neuron_stats: None,
        prediction_confidence: 0.0,
        expected_score_gain_confidence_interval: [0.0, 0.0],
        target_saturation_factor: None,
    };

    // Existing production experiment: pair with safety variants.
    let paired = pair_extreme_candidates_with_conservative_variants(vec![extreme], Some(3));
    assert_eq!(paired.len(), 3, "expected original + two safety variants");

    // New policy: do not return out-of-range candidates.
    let filtered = filter_candidates_to_sensible_ranges(paired);

    // Should drop the original extreme candidate, but keep at least one safe variant.
    assert!(
        !filtered.is_empty(),
        "expected at least one sensible candidate to remain"
    );

    // No returned candidate should have absurd bias/weights.
    for c in &filtered {
        assert!(
            c.incoming_weight.abs() <= 20.0,
            "incoming_weight should be kept in a sensible range, got {}",
            c.incoming_weight
        );
        assert!(
            c.bias.abs() <= 10.0,
            "bias should be kept in a sensible range, got {}",
            c.bias
        );
        assert!(
            c.outgoing_weight.abs() <= 0.1,
            "outgoing_weight is clamped in analysis and should remain ≤0.1, got {}",
            c.outgoing_weight
        );
    }
}

#[test]
fn absurd_identity_bias_is_filtered_out() {
    let absurd = CandidateNeuronJson {
        source_neuron_uuid: "source-1".to_string(),
        target_neuron_uuid: "target-1".to_string(),
        source_neuron_index: None,
        target_neuron_index: None,
        incoming_weight: 1.0,
        outgoing_weight: 0.1,
        squash: "IDENTITY".to_string(),
        bias: 95_247_370.0,
        comment: None,
        target_neuron_impact: 1.0,
        expected_creature_error_reduction: 0.2,
        expected_creature_score_gain: 0.2,
        improved_count: 10,
        total_count: 20,
        target_neuron_stats: None,
        prediction_confidence: 0.0,
        expected_score_gain_confidence_interval: [0.0, 0.0],
        target_saturation_factor: None,
    };

    let filtered = filter_candidates_to_sensible_ranges(vec![absurd]);
    assert!(
        filtered.is_empty(),
        "expected absurd IDENTITY bias candidate to be filtered out"
    );
}
