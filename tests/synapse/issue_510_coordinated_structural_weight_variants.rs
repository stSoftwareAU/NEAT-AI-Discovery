//! Tests for coordinated-structural conservative weight variants (Issue #510).
//!
//! Coordinated-structural candidates (epistatic/synergistic pairs) previously used a
//! fixed weight of 0.1 for addSynapse operations. Production evidence showed this
//! magnitude consistently fails, while conservative variants (~0.02) succeed.
//!
//! This test file verifies that the weight variant generation for coordinated-structural
//! candidates produces conservative, gentle-nudge, and micro-nudge variants with
//! progressively smaller weights — the same strategy used for synapse candidates (#513).

use neat_ai_discovery::CoordinatedStructuralCandidateJson;
use neat_ai_discovery::CoordinatedStructuralOpJson;
use neat_ai_discovery::analysis::utils::pair_coordinated_structural_with_weight_variants;

/// Helper to build a coordinated-structural candidate with two `AddSynapse` operations.
fn make_coordinated_candidate(
    source_a: &str,
    source_b: &str,
    target: &str,
    weight_a: f32,
    weight_b: f32,
    expected_gain: f32,
) -> CoordinatedStructuralCandidateJson {
    CoordinatedStructuralCandidateJson {
        operations: vec![
            CoordinatedStructuralOpJson::AddSynapse {
                from_neuron_uuid: source_a.to_string(),
                to_neuron_uuid: target.to_string(),
                weight: weight_a,
            },
            CoordinatedStructuralOpJson::AddSynapse {
                from_neuron_uuid: source_b.to_string(),
                to_neuron_uuid: target.to_string(),
                weight: weight_b,
            },
        ],
        expected_creature_score_gain: expected_gain,
        comment: Some("Epistatic pair: test".to_string()),
    }
}

// =========================================================================
// Core variant generation tests
// =========================================================================

#[test]
fn coordinated_generates_four_variants() {
    // A coordinated candidate with weight 0.1 should produce 4 variants:
    // original + conservative + gentle-nudge + micro-nudge.
    let candidate = make_coordinated_candidate("input-0", "input-1", "output-0", 0.1, 0.08, 0.05);
    let paired = pair_coordinated_structural_with_weight_variants(vec![candidate], None);

    assert_eq!(
        paired.len(),
        4,
        "expected original + 3 variants, got {}",
        paired.len()
    );
}

#[test]
fn coordinated_conservative_variant_scales_add_synapse_weights() {
    let candidate = make_coordinated_candidate("input-0", "input-1", "output-0", 0.1, 0.08, 0.05);
    let paired = pair_coordinated_structural_with_weight_variants(vec![candidate], None);

    let conservative = paired
        .iter()
        .find(|c| {
            c.comment
                .as_deref()
                .unwrap_or_default()
                .starts_with("Conservative")
        })
        .expect("expected a Conservative variant");

    // Conservative scales AddSynapse weights by CONSERVATIVE_OUTGOING_SCALE (0.2)
    let weights: Vec<f32> = conservative
        .operations
        .iter()
        .filter_map(|op| match op {
            CoordinatedStructuralOpJson::AddSynapse { weight, .. } => Some(*weight),
            _ => None,
        })
        .collect();

    assert_eq!(weights.len(), 2, "should have 2 AddSynapse operations");
    assert!(
        (weights[0] - 0.1 * 0.2).abs() < 1e-6,
        "conservative weight_a should be 0.1 * 0.2 = 0.02, got {}",
        weights[0]
    );
    assert!(
        (weights[1] - 0.08 * 0.2).abs() < 1e-6,
        "conservative weight_b should be 0.08 * 0.2 = 0.016, got {}",
        weights[1]
    );
}

#[test]
fn coordinated_gentle_nudge_variant_scales_weights() {
    let candidate = make_coordinated_candidate("input-0", "input-1", "output-0", 0.1, 0.08, 0.05);
    let paired = pair_coordinated_structural_with_weight_variants(vec![candidate], None);

    let gentle = paired
        .iter()
        .find(|c| {
            c.comment
                .as_deref()
                .unwrap_or_default()
                .starts_with("Gentle Nudge")
        })
        .expect("expected a Gentle Nudge variant");

    let weights: Vec<f32> = gentle
        .operations
        .iter()
        .filter_map(|op| match op {
            CoordinatedStructuralOpJson::AddSynapse { weight, .. } => Some(*weight),
            _ => None,
        })
        .collect();

    assert_eq!(weights.len(), 2);
    assert!(
        (weights[0] - 0.1 * 0.1).abs() < 1e-6,
        "gentle nudge weight_a should be 0.1 * 0.1 = 0.01, got {}",
        weights[0]
    );
    assert!(
        (weights[1] - 0.08 * 0.1).abs() < 1e-6,
        "gentle nudge weight_b should be 0.08 * 0.1 = 0.008, got {}",
        weights[1]
    );
}

#[test]
fn coordinated_micro_nudge_variant_scales_weights() {
    let candidate = make_coordinated_candidate("input-0", "input-1", "output-0", 0.1, 0.08, 0.05);
    let paired = pair_coordinated_structural_with_weight_variants(vec![candidate], None);

    let micro = paired
        .iter()
        .find(|c| {
            c.comment
                .as_deref()
                .unwrap_or_default()
                .starts_with("Micro-Nudge")
        })
        .expect("expected a Micro-Nudge variant");

    let weights: Vec<f32> = micro
        .operations
        .iter()
        .filter_map(|op| match op {
            CoordinatedStructuralOpJson::AddSynapse { weight, .. } => Some(*weight),
            _ => None,
        })
        .collect();

    assert_eq!(weights.len(), 2);
    assert!(
        (weights[0] - 0.1 * 0.05).abs() < 1e-6,
        "micro-nudge weight_a should be 0.1 * 0.05 = 0.005, got {}",
        weights[0]
    );
    assert!(
        (weights[1] - 0.08 * 0.05).abs() < 1e-6,
        "micro-nudge weight_b should be 0.08 * 0.05 = 0.004, got {}",
        weights[1]
    );
}

#[test]
fn coordinated_variant_scales_expected_gain() {
    let candidate = make_coordinated_candidate("input-0", "input-1", "output-0", 0.1, 0.08, 0.05);
    let paired = pair_coordinated_structural_with_weight_variants(vec![candidate], None);

    // Original keeps full gain
    assert!(
        (paired[0].expected_creature_score_gain - 0.05).abs() < 1e-6,
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
        (conservative.expected_creature_score_gain - 0.05 * 0.5).abs() < 1e-6,
        "conservative gain should be 0.5x, got {}",
        conservative.expected_creature_score_gain
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
        (gentle.expected_creature_score_gain - 0.05 * 0.75).abs() < 1e-6,
        "gentle nudge gain should be 0.75x, got {}",
        gentle.expected_creature_score_gain
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
        (micro.expected_creature_score_gain - 0.05 * 0.25).abs() < 1e-6,
        "micro-nudge gain should be 0.25x, got {}",
        micro.expected_creature_score_gain
    );
}

#[test]
fn coordinated_variant_preserves_non_add_synapse_operations() {
    // A coordinated candidate with mixed ops (RemoveSynapse + AddSynapse)
    // should only scale AddSynapse weights, leaving other ops untouched.
    let candidate = CoordinatedStructuralCandidateJson {
        operations: vec![
            CoordinatedStructuralOpJson::RemoveSynapse {
                from_neuron_uuid: "input-0".to_string(),
                to_neuron_uuid: "output-0".to_string(),
            },
            CoordinatedStructuralOpJson::AddSynapse {
                from_neuron_uuid: "input-1".to_string(),
                to_neuron_uuid: "output-0".to_string(),
                weight: 0.1,
            },
        ],
        expected_creature_score_gain: 0.05,
        comment: Some("Noisy vs trusted".to_string()),
    };

    let paired = pair_coordinated_structural_with_weight_variants(vec![candidate], None);

    // Should still generate variants
    assert!(
        paired.len() >= 2,
        "should generate at least original + conservative, got {}",
        paired.len()
    );

    // Each variant should preserve the RemoveSynapse operation
    for variant in &paired {
        let has_remove = variant.operations.iter().any(|op| {
            matches!(
                op,
                CoordinatedStructuralOpJson::RemoveSynapse {
                    from_neuron_uuid,
                    ..
                } if from_neuron_uuid == "input-0"
            )
        });
        assert!(
            has_remove,
            "all variants must preserve RemoveSynapse operations"
        );
    }
}

#[test]
fn coordinated_variant_preserves_negative_weight_sign() {
    let candidate = make_coordinated_candidate("input-0", "input-1", "output-0", -0.06, 0.08, 0.05);
    let paired = pair_coordinated_structural_with_weight_variants(vec![candidate], None);

    for variant in &paired {
        let weights: Vec<f32> = variant
            .operations
            .iter()
            .filter_map(|op| match op {
                CoordinatedStructuralOpJson::AddSynapse {
                    from_neuron_uuid,
                    weight,
                    ..
                } if from_neuron_uuid == "input-0" => Some(*weight),
                _ => None,
            })
            .collect();

        for w in &weights {
            assert!(
                *w < 0.0 || w.abs() < 1e-6,
                "weight sign for input-0 should remain negative, got {w}"
            );
        }
    }
}

#[test]
fn coordinated_variant_respects_max_candidates_limit() {
    let candidate = make_coordinated_candidate("input-0", "input-1", "output-0", 0.1, 0.08, 0.05);
    let paired = pair_coordinated_structural_with_weight_variants(vec![candidate], Some(2));

    assert_eq!(
        paired.len(),
        2,
        "should respect max_candidates limit of 2, got {}",
        paired.len()
    );
}

#[test]
fn coordinated_variant_skips_candidates_without_add_synapse() {
    // A candidate with only RemoveSynapse ops has no weights to scale.
    let candidate = CoordinatedStructuralCandidateJson {
        operations: vec![
            CoordinatedStructuralOpJson::RemoveSynapse {
                from_neuron_uuid: "input-0".to_string(),
                to_neuron_uuid: "output-0".to_string(),
            },
            CoordinatedStructuralOpJson::RemoveNeuron {
                neuron_uuid: "hidden-0".to_string(),
            },
        ],
        expected_creature_score_gain: 0.05,
        comment: None,
    };

    let paired = pair_coordinated_structural_with_weight_variants(vec![candidate], None);

    // No AddSynapse ops → no weight variants to generate, just the original
    assert_eq!(
        paired.len(),
        1,
        "candidate without AddSynapse ops should not generate variants, got {}",
        paired.len()
    );
}

#[test]
fn coordinated_variant_original_comment_lists_included_variants() {
    let candidate = make_coordinated_candidate("input-0", "input-1", "output-0", 0.1, 0.08, 0.05);
    let paired = pair_coordinated_structural_with_weight_variants(vec![candidate], None);

    // The original should preserve existing context and append variant info
    let original_comment = paired[0].comment.as_deref().unwrap_or_default();
    assert!(
        original_comment.contains("Epistatic pair: test"),
        "original comment should preserve existing context: {original_comment}"
    );
    assert!(
        original_comment.contains("Conservative")
            && original_comment.contains("Gentle Nudge")
            && original_comment.contains("Micro-Nudge"),
        "original comment should mention all variants: {original_comment}"
    );
}

#[test]
fn coordinated_variant_multiple_candidates_each_get_variants() {
    let candidates = vec![
        make_coordinated_candidate("input-0", "input-1", "output-0", 0.1, 0.08, 0.05),
        make_coordinated_candidate("input-2", "input-3", "output-0", 0.06, 0.04, 0.03),
    ];
    let paired = pair_coordinated_structural_with_weight_variants(candidates, None);

    // 2 candidates × 4 variants = 8
    assert_eq!(
        paired.len(),
        8,
        "expected 8 candidates (2 × 4 variants each), got {}",
        paired.len()
    );
}
