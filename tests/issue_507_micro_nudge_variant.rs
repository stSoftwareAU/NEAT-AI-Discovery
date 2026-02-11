//! Tests for ultra-conservative 'micro-nudge' weight variant (Issue #507).
//!
//! When an add-neuron candidate is extreme, we already generate conservative and gentle-nudge
//! variants. The micro-nudge variant goes even smaller: outgoing weight in the ±0.002–0.005
//! range, targeting the near-miss sweet spot observed in production.
//!
//! The micro-nudge variant is only generated when the conservative variant's outgoing weight
//! exceeds the micro-nudge max (i.e. when it would meaningfully differ from conservative).

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
        target_neuron_stats: None,
        prediction_confidence: 0.0,
        expected_score_gain_confidence_interval: [0.0, 0.0],
    }
}

#[test]
fn micro_nudge_variant_is_generated_for_extreme_candidates() {
    // Extreme candidate with large incoming weight and outgoing weight.
    // Conservative outgoing = 0.1 * 0.2 = 0.02 (above micro-nudge max of 0.005).
    // So the micro-nudge variant should meaningfully differ and be included.
    let candidate = make_extreme_candidate("source-1", "target-1", 200.0, 0.1, 50.0);

    // Limit allows all 4 variants: original + conservative + gentle nudge + micro-nudge
    let paired = pair_extreme_candidates_with_conservative_variants(vec![candidate], Some(4));

    assert_eq!(
        paired.len(),
        4,
        "expected original + conservative + gentle nudge + micro-nudge"
    );

    // Find the micro-nudge variant
    let micro_nudge = paired
        .iter()
        .find(|c| {
            c.comment
                .as_deref()
                .unwrap_or_default()
                .starts_with("Micro-Nudge")
        })
        .expect("expected a Micro-Nudge variant");

    // Micro-nudge outgoing weight should be very small (±0.005 max)
    assert!(
        micro_nudge.outgoing_weight.abs() <= 0.005 + 1e-6,
        "micro-nudge outgoing weight {} should be at most 0.005",
        micro_nudge.outgoing_weight
    );

    // Micro-nudge should preserve the sign of the outgoing weight
    // Original has positive outgoing (0.1), so micro-nudge should also be positive
    assert!(
        micro_nudge.outgoing_weight > 0.0,
        "micro-nudge should preserve outgoing weight sign"
    );

    // Micro-nudge incoming should be clamped (like conservative)
    assert!(
        micro_nudge.incoming_weight.abs() <= 2.0 + 1e-6,
        "micro-nudge incoming weight {} should be clamped to 2.0",
        micro_nudge.incoming_weight
    );

    // Micro-nudge bias should be tightly clamped
    assert!(
        micro_nudge.bias.abs() <= 1.0 + 1e-6,
        "micro-nudge bias {} should be clamped to 1.0",
        micro_nudge.bias
    );
}

#[test]
fn micro_nudge_outgoing_weight_is_in_expected_range() {
    // Conservative outgoing = 0.1 * 0.2 = 0.02, well above micro-nudge max of 0.005.
    // Micro-nudge outgoing = 0.1 * 0.05 = 0.005, clamped to 0.005.
    let candidate = make_extreme_candidate("source-1", "target-1", 200.0, 0.1, 50.0);

    let paired = pair_extreme_candidates_with_conservative_variants(vec![candidate], None);

    let micro_nudge = paired
        .iter()
        .find(|c| {
            c.comment
                .as_deref()
                .unwrap_or_default()
                .starts_with("Micro-Nudge")
        })
        .expect("expected a Micro-Nudge variant");

    // With outgoing_weight=0.1, micro-nudge scale=0.05: 0.1 * 0.05 = 0.005
    assert!(
        (micro_nudge.outgoing_weight - 0.005).abs() < 1e-6,
        "expected micro-nudge outgoing ~0.005, got {}",
        micro_nudge.outgoing_weight
    );
}

#[test]
fn micro_nudge_not_generated_when_conservative_outgoing_already_small() {
    // When the conservative variant's outgoing weight is already within the micro-nudge
    // range (≤0.005), the micro-nudge would not meaningfully differ and should be skipped.
    //
    // Original outgoing = 0.01, conservative outgoing = 0.01 * 0.2 = 0.002 (within 0.005).
    // Micro-nudge outgoing = 0.01 * 0.05 = 0.0005 (would differ, but very close).
    // However, the issue specifies: only generate when conservative outgoing > micro-nudge max.
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
    let candidate = make_extreme_candidate("source-1", "target-1", 200.0, 0.1, 50.0);
    let original_expected = candidate.expected_creature_error_reduction;

    let paired = pair_extreme_candidates_with_conservative_variants(vec![candidate], None);

    let micro_nudge = paired
        .iter()
        .find(|c| {
            c.comment
                .as_deref()
                .unwrap_or_default()
                .starts_with("Micro-Nudge")
        })
        .expect("expected a Micro-Nudge variant");

    // Micro-nudge expected multiplier is 0.25 (from issue spec)
    let expected = original_expected * 0.25;
    assert!(
        (micro_nudge.expected_creature_error_reduction - expected).abs() < 1e-6,
        "expected micro-nudge error reduction {expected}, got {}",
        micro_nudge.expected_creature_error_reduction
    );
    assert!(
        (micro_nudge.expected_creature_score_gain - expected).abs() < 1e-6,
        "expected micro-nudge score gain {expected}, got {}",
        micro_nudge.expected_creature_score_gain
    );
}

#[test]
fn micro_nudge_preserves_negative_outgoing_weight_sign() {
    // Candidate with negative outgoing weight
    let candidate = make_extreme_candidate("source-1", "target-1", 200.0, -0.1, 50.0);

    let paired = pair_extreme_candidates_with_conservative_variants(vec![candidate], None);

    let micro_nudge = paired
        .iter()
        .find(|c| {
            c.comment
                .as_deref()
                .unwrap_or_default()
                .starts_with("Micro-Nudge")
        })
        .expect("expected a Micro-Nudge variant");

    assert!(
        micro_nudge.outgoing_weight < 0.0,
        "micro-nudge should preserve negative outgoing weight sign, got {}",
        micro_nudge.outgoing_weight
    );
    assert!(
        (micro_nudge.outgoing_weight - (-0.005)).abs() < 1e-6,
        "expected micro-nudge outgoing ~-0.005, got {}",
        micro_nudge.outgoing_weight
    );
}

#[test]
fn extreme_candidate_comment_reflects_all_four_variants() {
    let candidate = make_extreme_candidate("source-1", "target-1", 200.0, 0.1, 50.0);

    let paired = pair_extreme_candidates_with_conservative_variants(vec![candidate], None);

    // Original candidate should mention all three variant types
    let original_comment = paired[0].comment.as_deref().unwrap_or_default();
    assert!(
        original_comment.contains("Conservative")
            && original_comment.contains("Gentle Nudge")
            && original_comment.contains("Micro-Nudge"),
        "original comment should mention all three variants: {original_comment}"
    );
}

#[test]
fn micro_nudge_skipped_when_limit_too_small() {
    let candidate = make_extreme_candidate("source-1", "target-1", 200.0, 0.1, 50.0);

    // Limit of 3 leaves room for original + conservative + gentle nudge only
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
        "micro-nudge should be skipped when limit prevents it"
    );
}

#[test]
fn micro_nudge_deduplication_across_different_neuron_pairs() {
    // Two extreme candidates connecting different neuron pairs should each get
    // their own micro-nudge variant (not deduped across pairs).
    let candidate_a = make_extreme_candidate("source-a", "target-a", 200.0, 0.1, 50.0);
    let candidate_b = make_extreme_candidate("source-b", "target-b", 200.0, 0.1, 50.0);

    let paired =
        pair_extreme_candidates_with_conservative_variants(vec![candidate_a, candidate_b], None);

    let micro_nudge_pairs: Vec<(&str, &str)> = paired
        .iter()
        .filter(|c| {
            c.comment
                .as_deref()
                .unwrap_or_default()
                .starts_with("Micro-Nudge")
        })
        .map(|c| (c.source_neuron_uuid.as_str(), c.target_neuron_uuid.as_str()))
        .collect();

    assert_eq!(
        micro_nudge_pairs.len(),
        2,
        "expected two micro-nudge variants (one per neuron pair)"
    );
    assert!(
        micro_nudge_pairs.contains(&("source-a", "target-a")),
        "expected micro-nudge for candidate_a"
    );
    assert!(
        micro_nudge_pairs.contains(&("source-b", "target-b")),
        "expected micro-nudge for candidate_b"
    );
}

#[test]
fn micro_nudge_minimum_outgoing_weight_when_scale_produces_near_zero() {
    // When the scaled outgoing weight is near zero, micro-nudge should use a
    // small non-zero fallback to ensure the candidate actually does something.
    let candidate = make_extreme_candidate("source-1", "target-1", 200.0, 0.0001, 50.0);

    let paired = pair_extreme_candidates_with_conservative_variants(vec![candidate], None);

    let micro_nudge = paired.iter().find(|c| {
        c.comment
            .as_deref()
            .unwrap_or_default()
            .starts_with("Micro-Nudge")
    });

    // Even with near-zero outgoing weight, the micro-nudge (if generated) should have
    // a non-zero weight. However, conservative outgoing = 0.0001 * 0.2 = 0.00002 which
    // is ≤ 0.005, so micro-nudge should NOT be generated per the mitigation rule.
    assert!(
        micro_nudge.is_none(),
        "micro-nudge should not be generated when conservative outgoing is within micro-nudge range"
    );
}

#[test]
fn total_candidate_count_with_micro_nudge_is_four_per_extreme() {
    // 3 extreme candidates with large outgoing weights should produce 12 total:
    // 3 originals + 3 conservative + 3 gentle nudge + 3 micro-nudge
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
        12,
        "expected 12 candidates (3 extreme × 4 variants each)"
    );
}
