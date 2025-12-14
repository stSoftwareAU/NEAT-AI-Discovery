//! Regression tests for extreme-candidate pairing logic.
//!
//! Production experiment (Dec 2025): when an add-neuron candidate is deemed "extreme",
//! we want to return both the original candidate and a conservative variant.
//!
//! This file ensures we do not accidentally drop the conservative variant due to an
//! overly-narrow duplicate check.

use neat_ai_discovery::{
    analysis::utils::pair_extreme_candidates_with_conservative_variants, CandidateNeuronJson,
};

#[test]
fn extreme_candidate_comment_does_not_claim_pairing_when_limit_prevents_variant() {
    let candidate = CandidateNeuronJson {
        source_neuron_uuid: "source-1".to_string(),
        target_neuron_uuid: "target-1".to_string(),
        incoming_weight: 3.0, // extreme (above clamp max)
        outgoing_weight: 0.1,
        squash: "TANH".to_string(),
        bias: 0.0,
        comment: None,
        target_neuron_impact: 1.0,
        expected_creature_error_reduction: 0.2,
        expected_creature_score_gain: 0.2,
        improved_count: 10,
        total_count: 20,
        target_neuron_stats: None,
    };

    // Limit leaves no room for the conservative variant.
    let paired = pair_extreme_candidates_with_conservative_variants(vec![candidate], Some(1));

    assert_eq!(paired.len(), 1, "expected only the original candidate");
    assert_eq!(
        paired[0].comment.as_deref(),
        Some("Extreme candidate (conservative variant not included)"),
        "comment should not claim pairing when the variant cannot be returned"
    );
}

#[test]
fn extreme_candidate_pairing_considers_outgoing_weight_differences() {
    // This is the edge case highlighted in PR review:
    // - incoming is barely above the clamp threshold, but within 1e-6 after clamping
    // - bias is already within the clamp threshold
    // - outgoing weight is meaningfully reduced by the conservative variant
    //
    // If we don't consider outgoing weight in the duplicate check, we incorrectly drop the
    // conservative variant and return only a single candidate.
    let candidate = CandidateNeuronJson {
        source_neuron_uuid: "source-1".to_string(),
        target_neuron_uuid: "target-1".to_string(),
        incoming_weight: 2.0 + 5e-7,
        outgoing_weight: 0.1,
        squash: "TANH".to_string(),
        bias: 0.0,
        comment: None,
        target_neuron_impact: 1.0,
        expected_creature_error_reduction: 0.2,
        expected_creature_score_gain: 0.2,
        improved_count: 10,
        total_count: 20,
        target_neuron_stats: None,
    };

    let paired = pair_extreme_candidates_with_conservative_variants(vec![candidate], Some(2));

    assert_eq!(paired.len(), 2, "expected original + conservative variant");

    // Original is returned first.
    assert!(paired[0].comment.is_some());

    // Conservative variant should have meaningfully smaller outgoing weight.
    // 0.1 * 0.2 = 0.02 (within outgoing clamp).
    assert!(
        (paired[1].outgoing_weight - 0.02).abs() < 1e-6,
        "expected outgoing weight to be scaled to ~0.02"
    );
    assert!(paired[1].comment.is_some());
}
