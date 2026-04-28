//! Tests for synapse candidate weight variants (Issue #513).
//!
//! When a helpful synapse candidate is discovered, we should generate multiple weight
//! variants (conservative, gentle nudge, micro-nudge) alongside the original. This gives
//! the TypeScript controller more options to test — the discovery process is expensive,
//! but testing each variant is cheap (~1 minute). Having multiple variants of the same
//! candidate maximises the pay-off from the discovery investment.

use neat_ai_discovery::CandidateSynapseJson;
use neat_ai_discovery::analysis::utils::pair_synapse_candidates_with_weight_variants;

/// Helper to build a helpful synapse candidate for testing.
fn make_synapse_candidate(
    from: &str,
    to: &str,
    weight: f32,
    expected_gain: f32,
) -> CandidateSynapseJson {
    CandidateSynapseJson {
        from_neuron_uuid: from.to_string(),
        to_neuron_uuid: to.to_string(),
        from_neuron_index: None,
        to_neuron_index: None,
        weight,
        target_neuron_impact: 1.0,
        expected_creature_error_reduction: expected_gain,
        expected_creature_score_gain: expected_gain,
        improved_count: 10,
        total_count: 20,
        improvement_magnitude_ratio: None,
        target_neuron_stats: None,
        outlier_reduction_info: None,
        prediction_confidence: 0.8,
        expected_score_gain_confidence_interval: [0.0, 0.0],
        comment: None,
        variant_key: None,
    }
}

// =========================================================================
// Core variant generation tests
// =========================================================================

#[test]
fn synapse_candidate_generates_four_variants() {
    // A synapse candidate with weight 0.08 should produce 6 variants:
    // original + conservative + gentle nudge + micro-nudge + feather-touch + whisper.
    // Issue #962: Feather-Touch and Whisper added for weights >= 0.05 threshold.
    let candidate = make_synapse_candidate("input-5", "output-0", 0.08, 0.1);
    let paired = pair_synapse_candidates_with_weight_variants(vec![candidate], None);

    assert_eq!(
        paired.len(),
        6,
        "expected original + conservative + gentle nudge + micro-nudge + feather-touch + whisper, got {}",
        paired.len()
    );
}

#[test]
fn synapse_conservative_variant_has_scaled_weight() {
    let candidate = make_synapse_candidate("input-5", "output-0", 0.08, 0.1);
    let paired = pair_synapse_candidates_with_weight_variants(vec![candidate], None);

    let conservative = paired
        .iter()
        .find(|c| {
            c.comment
                .as_deref()
                .unwrap_or_default()
                .starts_with("Conservative")
        })
        .expect("expected a Conservative variant");

    // Conservative should scale the weight down by 0.5
    let expected_weight = 0.08 * 0.5;
    assert!(
        (conservative.weight - expected_weight).abs() < 1e-6,
        "expected conservative weight {expected_weight}, got {}",
        conservative.weight
    );
}

#[test]
fn synapse_gentle_nudge_variant_has_scaled_weight() {
    let candidate = make_synapse_candidate("input-5", "output-0", 0.08, 0.1);
    let paired = pair_synapse_candidates_with_weight_variants(vec![candidate], None);

    let gentle = paired
        .iter()
        .find(|c| {
            c.comment
                .as_deref()
                .unwrap_or_default()
                .starts_with("Gentle Nudge")
        })
        .expect("expected a Gentle Nudge variant");

    // Gentle nudge should scale the weight down by 0.25
    let expected_weight = 0.08 * 0.25;
    assert!(
        (gentle.weight - expected_weight).abs() < 1e-6,
        "expected gentle nudge weight {expected_weight}, got {}",
        gentle.weight
    );
}

#[test]
fn synapse_micro_nudge_variant_has_scaled_weight() {
    let candidate = make_synapse_candidate("input-5", "output-0", 0.08, 0.1);
    let paired = pair_synapse_candidates_with_weight_variants(vec![candidate], None);

    let micro = paired
        .iter()
        .find(|c| {
            c.comment
                .as_deref()
                .unwrap_or_default()
                .starts_with("Micro-Nudge")
        })
        .expect("expected a Micro-Nudge variant");

    // Micro-nudge should scale the weight down by 0.1
    let expected_weight = 0.08 * 0.1;
    assert!(
        (micro.weight - expected_weight).abs() < 1e-6,
        "expected micro-nudge weight {expected_weight}, got {}",
        micro.weight
    );
}

#[test]
fn synapse_variant_preserves_negative_weight_sign() {
    let candidate = make_synapse_candidate("input-5", "output-0", -0.06, 0.1);
    let paired = pair_synapse_candidates_with_weight_variants(vec![candidate], None);

    for variant in &paired {
        assert!(
            variant.weight < 0.0 || variant.weight.abs() < 1e-6,
            "all variants should preserve negative weight sign, got {}",
            variant.weight
        );
    }

    let conservative = paired
        .iter()
        .find(|c| {
            c.comment
                .as_deref()
                .unwrap_or_default()
                .starts_with("Conservative")
        })
        .expect("expected Conservative");

    assert!(
        (conservative.weight - (-0.06 * 0.5)).abs() < 1e-6,
        "conservative should preserve negative weight, got {}",
        conservative.weight
    );
}

#[test]
fn synapse_variant_expected_gain_is_scaled() {
    let candidate = make_synapse_candidate("input-5", "output-0", 0.08, 0.1);
    let paired = pair_synapse_candidates_with_weight_variants(vec![candidate], None);

    // Original keeps full gain
    assert!(
        (paired[0].expected_creature_score_gain - 0.1).abs() < 1e-6,
        "original should keep full gain"
    );

    let conservative = paired
        .iter()
        .find(|c| {
            c.comment
                .as_deref()
                .unwrap_or_default()
                .starts_with("Conservative")
        })
        .unwrap();
    assert!(
        (conservative.expected_creature_score_gain - 0.1 * 0.5).abs() < 1e-6,
        "conservative gain should be 0.5x"
    );

    let gentle = paired
        .iter()
        .find(|c| {
            c.comment
                .as_deref()
                .unwrap_or_default()
                .starts_with("Gentle Nudge")
        })
        .unwrap();
    assert!(
        (gentle.expected_creature_score_gain - 0.1 * 0.75).abs() < 1e-6,
        "gentle nudge gain should be 0.75x"
    );

    let micro = paired
        .iter()
        .find(|c| {
            c.comment
                .as_deref()
                .unwrap_or_default()
                .starts_with("Micro-Nudge")
        })
        .unwrap();
    assert!(
        (micro.expected_creature_score_gain - 0.1 * 0.25).abs() < 1e-6,
        "micro-nudge gain should be 0.25x"
    );
}

#[test]
fn synapse_variant_respects_max_candidates_limit() {
    let candidate = make_synapse_candidate("input-5", "output-0", 0.08, 0.1);

    // Limit of 2 should produce original + conservative only
    let paired = pair_synapse_candidates_with_weight_variants(vec![candidate], Some(2));
    assert_eq!(paired.len(), 2, "should respect max_candidates limit");
}

#[test]
fn synapse_variant_skips_near_zero_weight_candidates() {
    // A candidate with near-zero weight should not generate variants that are
    // essentially identical to the original (all near zero).
    let candidate = make_synapse_candidate("input-5", "output-0", 0.001, 0.01);
    let paired = pair_synapse_candidates_with_weight_variants(vec![candidate], None);

    // Should still get at least the original
    assert!(
        !paired.is_empty(),
        "should at least return the original candidate"
    );

    // Variants with near-zero scaled weight may be deduplicated
    for variant in &paired {
        assert!(
            variant.weight.abs() > 1e-7,
            "variants should have non-negligible weight"
        );
    }
}

#[test]
fn synapse_variant_original_comment_lists_included_variants() {
    let candidate = make_synapse_candidate("input-5", "output-0", 0.08, 0.1);
    let paired = pair_synapse_candidates_with_weight_variants(vec![candidate], None);

    let original_comment = paired[0].comment.as_deref().unwrap_or_default();
    // Issue #962: Now includes Feather-Touch and Whisper for weights above threshold.
    assert!(
        original_comment.contains("Conservative")
            && original_comment.contains("Gentle Nudge")
            && original_comment.contains("Micro-Nudge")
            && original_comment.contains("Feather-Touch")
            && original_comment.contains("Whisper"),
        "original comment should mention all five variants: {original_comment}"
    );
}

#[test]
fn synapse_variant_multiple_candidates_each_get_variants() {
    let candidates = vec![
        make_synapse_candidate("input-1", "output-0", 0.08, 0.2),
        make_synapse_candidate("input-2", "output-0", 0.06, 0.15),
    ];
    let paired = pair_synapse_candidates_with_weight_variants(candidates, None);

    // Issue #962: 2 candidates × 6 variants = 12 (weights 0.08 and 0.06 both above threshold).
    assert_eq!(
        paired.len(),
        12,
        "expected 12 candidates (2 \u{00d7} 6 variants each), got {}",
        paired.len()
    );
}

#[test]
fn synapse_variant_from_to_neuron_uuids_match_original() {
    let candidate = make_synapse_candidate("input-5", "output-0", 0.08, 0.1);
    let paired = pair_synapse_candidates_with_weight_variants(vec![candidate], None);

    for variant in &paired {
        assert_eq!(
            variant.from_neuron_uuid, "input-5",
            "from_neuron_uuid should match original"
        );
        assert_eq!(
            variant.to_neuron_uuid, "output-0",
            "to_neuron_uuid should match original"
        );
    }
}
