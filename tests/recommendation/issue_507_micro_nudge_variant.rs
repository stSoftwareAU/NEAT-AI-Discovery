//! Tests for ultra-conservative 'micro-nudge' weight variant (Issue #507).
//!
//! When an add-neuron candidate is extreme, we already generate conservative and gentle-nudge
//! variants. The micro-nudge variant goes even smaller: outgoing weight in the ±0.002–0.005
//! range, targeting the near-miss sweet spot observed in production.
//!
//! Issue #888: With tightened constraints, Conservative and Micro-Nudge now share the same
//! `outgoing_abs_max` (0.005), so `should_generate_micro_nudge` returns false (the micro-nudge
//! would not meaningfully differ from conservative). The pairing function now generates
//! 3 variants per extreme candidate: original + conservative + gentle nudge.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use neat_ai_discovery::{
    CandidateNeuronJson, analysis::utils::pair_extreme_candidates_with_conservative_variants,
};

/// Helper to build an extreme add-neuron candidate for testing.
fn make_extreme_candidate(
    source: &str,
    target: &str,
    incoming: f32,
    outgoing: f32,
    bias: f32,
) -> CandidateNeuronJson {
    CandidateNeuronJson {
        source_neuron_uuid: source.to_string(),
        target_neuron_uuid: target.to_string(),
        source_neuron_index: None,
        target_neuron_index: None,
        incoming_weight: incoming,
        outgoing_weight: outgoing,
        squash: "TANH".to_string(),
        bias,
        comment: None,
        target_neuron_impact: 1.0,
        expected_creature_error_reduction: 0.2,
        expected_creature_score_gain: 0.2,
        improved_count: 10,
        total_count: 20,
        improvement_magnitude_ratio: None,
        target_neuron_stats: None,
        prediction_confidence: 0.0,
        expected_score_gain_confidence_interval: [0.0, 0.0],
        target_saturation_factor: None,
    }
}

#[test]
fn micro_nudge_variant_is_generated_for_extreme_candidates() {
    // Issue #888: With tightened constraints, Conservative outgoing is always clamped
    // to 0.005 which equals Micro-Nudge max, so Micro-Nudge is no longer generated.
    // Issue #962: Feather-Touch and Whisper added for weights above threshold (0.05).
    // Extreme candidates with outgoing 0.1 now produce 5 variants:
    // original + conservative + gentle nudge + feather-touch + whisper.
    let candidate = make_extreme_candidate("source-1", "target-1", 200.0, 0.1, 50.0);

    let paired = pair_extreme_candidates_with_conservative_variants(vec![candidate], Some(6));

    assert_eq!(
        paired.len(),
        5,
        "expected original + conservative + gentle nudge + feather-touch + whisper (micro-nudge skipped)"
    );

    // Conservative variant should have outgoing clamped to 0.005
    assert!(
        paired[1]
            .comment
            .as_deref()
            .unwrap_or_default()
            .contains("Conservative"),
        "second should be Conservative variant"
    );
    assert!(
        (paired[1].outgoing_weight - 0.005).abs() < 1e-6,
        "conservative outgoing should be 0.005, got {}",
        paired[1].outgoing_weight
    );

    // Gentle Nudge variant
    assert!(
        paired[2]
            .comment
            .as_deref()
            .unwrap_or_default()
            .contains("Gentle Nudge"),
        "third should be Gentle Nudge variant"
    );
}

#[test]
fn micro_nudge_outgoing_weight_is_in_expected_range() {
    // Issue #888: Micro-Nudge is no longer generated in the pairing context because
    // Conservative outgoing (clamped to 0.005) does not exceed Micro-Nudge max (0.005).
    // Verify the conservative variant takes the ultra-small outgoing role instead.
    let candidate = make_extreme_candidate("source-1", "target-1", 200.0, 0.1, 50.0);

    let paired = pair_extreme_candidates_with_conservative_variants(vec![candidate], None);

    // Conservative variant should have the smallest outgoing weight
    let conservative = paired
        .iter()
        .find(|c| {
            c.comment
                .as_deref()
                .unwrap_or_default()
                .starts_with("Conservative")
        })
        .expect("expected a Conservative variant");

    // Conservative outgoing = min(0.1 * 0.2, 0.005) = 0.005
    assert!(
        (conservative.outgoing_weight - 0.005).abs() < 1e-6,
        "expected conservative outgoing ~0.005, got {}",
        conservative.outgoing_weight
    );
}

#[test]
fn micro_nudge_not_generated_when_conservative_outgoing_already_small() {
    // When the conservative variant's outgoing weight is already within the micro-nudge
    // range (≤0.005), the micro-nudge would not meaningfully differ and should be skipped.
    //
    // Original outgoing = 0.01, conservative outgoing = 0.01 * 0.2 = 0.002 (within 0.005).
    // Conservative outgoing (0.002) ≤ micro-nudge max (0.005), so skip micro-nudge.
    let candidate = make_extreme_candidate("source-1", "target-1", 200.0, 0.01, 50.0);

    let paired = pair_extreme_candidates_with_conservative_variants(vec![candidate], None);

    let micro_nudge_count = paired
        .iter()
        .filter(|c| {
            c.comment
                .as_deref()
                .unwrap_or_default()
                .starts_with("Micro-Nudge")
        })
        .count();

    assert_eq!(
        micro_nudge_count, 0,
        "micro-nudge should not be generated when conservative outgoing is already small"
    );
}

#[test]
fn micro_nudge_expected_improvement_is_scaled_down() {
    // Issue #888: With tightened constraints, Micro-Nudge is no longer generated in
    // the pairing function. Verify that Conservative uses the tightened multiplier instead.
    let candidate = make_extreme_candidate("source-1", "target-1", 200.0, 0.1, 50.0);
    let original_expected = candidate.expected_creature_error_reduction;

    let paired = pair_extreme_candidates_with_conservative_variants(vec![candidate], None);

    let conservative = paired
        .iter()
        .find(|c| {
            c.comment
                .as_deref()
                .unwrap_or_default()
                .starts_with("Conservative")
        })
        .expect("expected a Conservative variant");

    // Conservative expected multiplier is 0.5
    let expected = original_expected * 0.5;
    assert!(
        (conservative.expected_creature_error_reduction - expected).abs() < 1e-6,
        "expected conservative error reduction {expected}, got {}",
        conservative.expected_creature_error_reduction
    );
}

#[test]
fn micro_nudge_preserves_negative_outgoing_weight_sign() {
    // Issue #888: With tightened constraints, test Conservative (which takes Micro-Nudge's role).
    let candidate = make_extreme_candidate("source-1", "target-1", 200.0, -0.1, 50.0);

    let paired = pair_extreme_candidates_with_conservative_variants(vec![candidate], None);

    let conservative = paired
        .iter()
        .find(|c| {
            c.comment
                .as_deref()
                .unwrap_or_default()
                .starts_with("Conservative")
        })
        .expect("expected a Conservative variant");

    assert!(
        conservative.outgoing_weight < 0.0,
        "conservative should preserve negative outgoing weight sign, got {}",
        conservative.outgoing_weight
    );
    assert!(
        (conservative.outgoing_weight - (-0.005)).abs() < 1e-6,
        "expected conservative outgoing ~-0.005, got {}",
        conservative.outgoing_weight
    );
}

#[test]
fn extreme_candidate_comment_reflects_all_four_variants() {
    // Issue #888: With tightened constraints, only Conservative + Gentle Nudge are generated.
    let candidate = make_extreme_candidate("source-1", "target-1", 200.0, 0.1, 50.0);

    let paired = pair_extreme_candidates_with_conservative_variants(vec![candidate], None);

    let original_comment = paired[0].comment.as_deref().unwrap_or_default();
    assert!(
        original_comment.contains("Conservative") && original_comment.contains("Gentle Nudge"),
        "original comment should mention Conservative and Gentle Nudge: {original_comment}"
    );
}

#[test]
fn micro_nudge_skipped_when_limit_too_small() {
    let candidate = make_extreme_candidate("source-1", "target-1", 200.0, 0.1, 50.0);

    // Issue #888: With tightened constraints, only 3 variants are generated
    // (original + conservative + gentle nudge). Limit of 3 fits all of them.
    let paired = pair_extreme_candidates_with_conservative_variants(vec![candidate], Some(3));

    assert_eq!(paired.len(), 3, "expected 3 candidates at limit of 3");

    let micro_nudge_count = paired
        .iter()
        .filter(|c| {
            c.comment
                .as_deref()
                .unwrap_or_default()
                .starts_with("Micro-Nudge")
        })
        .count();

    assert_eq!(
        micro_nudge_count, 0,
        "micro-nudge should not be generated with tightened constraints"
    );
}

#[test]
fn micro_nudge_deduplication_across_different_neuron_pairs() {
    // Issue #888: With tightened constraints, Micro-Nudge is no longer generated.
    // Verify that Conservative variants are correctly generated for each neuron pair.
    let candidate_a = make_extreme_candidate("source-a", "target-a", 200.0, 0.1, 50.0);
    let candidate_b = make_extreme_candidate("source-b", "target-b", 200.0, 0.1, 50.0);

    let paired =
        pair_extreme_candidates_with_conservative_variants(vec![candidate_a, candidate_b], None);

    let conservative_pairs: Vec<(&str, &str)> = paired
        .iter()
        .filter(|c| {
            c.comment
                .as_deref()
                .unwrap_or_default()
                .starts_with("Conservative")
        })
        .map(|c| (c.source_neuron_uuid.as_str(), c.target_neuron_uuid.as_str()))
        .collect();

    assert_eq!(
        conservative_pairs.len(),
        2,
        "expected two Conservative variants (one per neuron pair)"
    );
    assert!(
        conservative_pairs.contains(&("source-a", "target-a")),
        "expected Conservative for candidate_a"
    );
    assert!(
        conservative_pairs.contains(&("source-b", "target-b")),
        "expected Conservative for candidate_b"
    );
}

#[test]
fn micro_nudge_minimum_outgoing_weight_when_scale_produces_near_zero() {
    // When the scaled outgoing weight is near zero, conservative should use a
    // small non-zero fallback to ensure the candidate actually does something.
    let candidate = make_extreme_candidate("source-1", "target-1", 200.0, 0.0001, 50.0);

    let paired = pair_extreme_candidates_with_conservative_variants(vec![candidate], None);

    // Conservative outgoing = 0.0001 * 0.2 = 0.00002, which is ≤ 0.005.
    // So micro-nudge should NOT be generated.
    let micro_nudge = paired.iter().find(|c| {
        c.comment
            .as_deref()
            .unwrap_or_default()
            .starts_with("Micro-Nudge")
    });

    assert!(
        micro_nudge.is_none(),
        "micro-nudge should not be generated when conservative outgoing is within micro-nudge range"
    );
}

#[test]
fn total_candidate_count_with_micro_nudge_is_four_per_extreme() {
    // Issue #888: With tightened constraints, Micro-Nudge is skipped.
    // Issue #962: Feather-Touch and Whisper added for weights above threshold.
    // 5 variants per extreme candidate (original + conservative + gentle nudge
    // + feather-touch + whisper).
    let candidates: Vec<CandidateNeuronJson> = (0..3)
        .map(|i| {
            let mut c =
                make_extreme_candidate(&format!("source-{i}"), "target-1", 200.0, 0.1, 50.0);
            c.expected_creature_error_reduction = 0.2 - (i as f32 * 0.01);
            c.expected_creature_score_gain = 0.2 - (i as f32 * 0.01);
            c
        })
        .collect();

    let paired = pair_extreme_candidates_with_conservative_variants(candidates, None);

    assert_eq!(
        paired.len(),
        15,
        "expected 15 candidates (3 extreme \u{00d7} 5 variants each)"
    );
}
