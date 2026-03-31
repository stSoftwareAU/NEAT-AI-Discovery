//! Regression tests for extreme-candidate pairing logic.
//!
//! Production experiment (Dec 2025): when an add-neuron candidate is deemed "extreme",
//! we want to return both the original candidate and a conservative variant.
//!
//! This file ensures we do not accidentally drop the conservative variant due to an
//! overly-narrow duplicate check.

use neat_ai_discovery::{
    CandidateNeuronJson, analysis::utils::pair_extreme_candidates_with_conservative_variants,
};

#[test]
fn extreme_candidate_comment_does_not_claim_pairing_when_limit_prevents_variant() {
    let candidate = CandidateNeuronJson {
        source_neuron_uuid: "source-1".to_string(),
        target_neuron_uuid: "target-1".to_string(),
        source_neuron_index: None,
        target_neuron_index: None,
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
        prediction_confidence: 0.0,
        expected_score_gain_confidence_interval: [0.0, 0.0],
    };

    // Limit leaves no room for any safety variants.
    let paired = pair_extreme_candidates_with_conservative_variants(vec![candidate], Some(1));

    assert_eq!(paired.len(), 1, "expected only the original candidate");
    assert_eq!(
        paired[0].comment.as_deref(),
        Some("Extreme candidate (no safety variants included)"),
        "comment should not claim pairing when variants cannot be returned"
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
        source_neuron_index: None,
        target_neuron_index: None,
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
        prediction_confidence: 0.0,
        expected_score_gain_confidence_interval: [0.0, 0.0],
    };

    let paired = pair_extreme_candidates_with_conservative_variants(vec![candidate], Some(2));

    assert_eq!(paired.len(), 2, "expected original + conservative variant");

    // Original is returned first.
    assert_eq!(
        paired[0].comment.as_deref(),
        Some("Extreme candidate (paired with Conservative variant only)"),
        "comment should reflect what was actually returned"
    );

    // Issue #888: Conservative outgoing_abs_max tightened from 0.05 to 0.005.
    // 0.1 * 0.2 = 0.02, clamped to 0.005.
    assert!(
        (paired[1].outgoing_weight - 0.005).abs() < 1e-6,
        "expected outgoing weight to be clamped to ~0.005, got {}",
        paired[1].outgoing_weight
    );
    assert!(paired[1].comment.is_some());
}

#[test]
fn extreme_candidate_includes_gentle_nudge_variant_when_limit_allows() {
    let candidate = CandidateNeuronJson {
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
    };

    let paired = pair_extreme_candidates_with_conservative_variants(vec![candidate], Some(3));

    assert_eq!(paired.len(), 3, "expected original + two safety variants");
    assert_eq!(
        paired[0].comment.as_deref(),
        Some("Extreme candidate (paired with Conservative + Gentle Nudge variants)"),
        "comment should reflect both variants were returned"
    );

    // Issue #888: Conservative outgoing tightened from 0.05 to 0.005, bias stays 1.0.
    assert!(
        (paired[1].outgoing_weight - 0.005).abs() < 1e-6,
        "expected conservative outgoing weight ~0.005, got {}",
        paired[1].outgoing_weight
    );
    assert!(
        (paired[1].bias - 1.0).abs() < 1e-6,
        "expected conservative bias to be clamped to 1.0"
    );

    // Issue #888: Gentle Nudge outgoing 0.01, bias tightened from 10.0 to 2.0.
    assert!(
        (paired[2].outgoing_weight - 0.01).abs() < 1e-6,
        "expected gentle outgoing weight ~0.01"
    );
    assert!(
        (paired[2].bias - 2.0).abs() < 1e-6,
        "expected gentle bias to be clamped to 2.0, got {}",
        paired[2].bias
    );
    assert!(
        paired[2]
            .comment
            .as_deref()
            .unwrap_or_default()
            .contains("Gentle Nudge"),
        "expected gentle variant to be tagged clearly"
    );
}

#[test]
fn gentle_nudge_variants_are_not_deduped_across_different_neuron_pairs() {
    // Regression (Dec 2025):
    // Gentle Nudge variants were incorrectly treated as duplicates across *different*
    // source/target neuron pairs when the clamping produced identical weights.
    //
    // Two candidates connecting different neuron pairs are fundamentally different
    // operations, even if their clamped weights happen to match.
    let candidate_a = CandidateNeuronJson {
        source_neuron_uuid: "source-a".to_string(),
        target_neuron_uuid: "target-a".to_string(),
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
    };

    let candidate_b = CandidateNeuronJson {
        source_neuron_uuid: "source-b".to_string(),
        target_neuron_uuid: "target-b".to_string(),
        source_neuron_index: None,
        target_neuron_index: None,
        incoming_weight: 200.0,
        outgoing_weight: 0.1,
        squash: "TANH".to_string(),
        bias: 50.0,
        comment: None,
        target_neuron_impact: 1.0,
        expected_creature_error_reduction: 0.19,
        expected_creature_score_gain: 0.19,
        improved_count: 9,
        total_count: 20,
        target_neuron_stats: None,
        prediction_confidence: 0.0,
        expected_score_gain_confidence_interval: [0.0, 0.0],
    };

    // Limit allows both originals plus all safety variants per candidate.
    // Issue #888: With tightened constraints, Conservative and Micro-Nudge have the
    // same outgoing_abs_max (0.005), so Micro-Nudge is skipped (not meaningfully different).
    // Issue #962: Feather-Touch and Whisper variants added for weights above threshold.
    // Each extreme candidate gets: original + conservative + gentle nudge + feather-touch
    // + whisper = 5 items. Two candidates = 10, limited to 8.
    let paired = pair_extreme_candidates_with_conservative_variants(
        vec![candidate_a, candidate_b],
        Some(12),
    );

    assert_eq!(
        paired.len(),
        10,
        "expected two originals + two conservative + two Gentle Nudge + two Feather-Touch + two Whisper variants"
    );

    // Expect a Gentle Nudge variant for each distinct neuron pair.
    let gentle_pairs: Vec<(&str, &str)> = paired
        .iter()
        .filter(|c| {
            c.comment
                .as_deref()
                .unwrap_or_default()
                .starts_with("Gentle Nudge variant")
        })
        .map(|c| (c.source_neuron_uuid.as_str(), c.target_neuron_uuid.as_str()))
        .collect();

    assert_eq!(
        gentle_pairs.len(),
        2,
        "expected two Gentle Nudge variants (one per unique neuron pair)"
    );
    assert!(
        gentle_pairs.contains(&("source-a", "target-a")),
        "expected Gentle Nudge for candidate_a connection"
    );
    assert!(
        gentle_pairs.contains(&("source-b", "target-b")),
        "expected Gentle Nudge for candidate_b connection"
    );
}
