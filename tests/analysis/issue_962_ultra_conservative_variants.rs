//! Tests for ultra-conservative weight variants (Issue #962).
//!
//! Two new variant tiers below Micro-Nudge for networks approaching equilibrium:
//! - **Feather-Touch**: `outgoing_scale` ~0.01, `expected_multiplier` ~0.25
//! - **Whisper**: `outgoing_scale` ~0.005, `expected_multiplier` ~0.1
//!
//! These are only generated when the base weight is above a configurable threshold
//! to avoid generating near-zero variants of already-small weights.

use neat_ai_discovery::analysis::utils::variant_generation::{
    FEATHER_TOUCH_CONFIG, MICRO_NUDGE_CONFIG, SYNAPSE_FEATHER_TOUCH_CONFIG, SYNAPSE_WHISPER_CONFIG,
    ULTRA_CONSERVATIVE_BASE_WEIGHT_THRESHOLD, WHISPER_CONFIG, make_neuron_variant,
    make_synapse_variant,
};
use neat_ai_discovery::analysis::utils::{
    pair_coordinated_structural_with_weight_variants,
    pair_extreme_candidates_with_conservative_variants,
    pair_synapse_candidates_with_weight_variants,
};
use neat_ai_discovery::{
    CandidateNeuronJson, CandidateSynapseJson, CoordinatedStructuralCandidateJson,
    CoordinatedStructuralOpJson,
};

/// Helper to build a test add-neuron candidate.
fn make_test_neuron_candidate(incoming: f32, outgoing: f32, bias: f32) -> CandidateNeuronJson {
    CandidateNeuronJson {
        source_neuron_uuid: "source-1".to_string(),
        target_neuron_uuid: "target-1".to_string(),
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

/// Helper to build a test synapse candidate.
fn make_test_synapse_candidate(weight: f32, expected_gain: f32) -> CandidateSynapseJson {
    CandidateSynapseJson {
        from_neuron_uuid: "input-5".to_string(),
        to_neuron_uuid: "output-0".to_string(),
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
    }
}

// =========================================================================
// Feather-Touch neuron variant config tests
// =========================================================================

#[test]
fn feather_touch_neuron_config_has_smaller_outgoing_scale_than_micro_nudge() {
    assert!(
        FEATHER_TOUCH_CONFIG.outgoing_scale < MICRO_NUDGE_CONFIG.outgoing_scale,
        "Feather-Touch outgoing_scale ({}) should be smaller than Micro-Nudge ({})",
        FEATHER_TOUCH_CONFIG.outgoing_scale,
        MICRO_NUDGE_CONFIG.outgoing_scale
    );
}

#[test]
fn feather_touch_neuron_config_outgoing_scale_is_001() {
    assert!(
        (FEATHER_TOUCH_CONFIG.outgoing_scale - 0.01).abs() < 1e-6,
        "Feather-Touch outgoing_scale should be 0.01, got {}",
        FEATHER_TOUCH_CONFIG.outgoing_scale
    );
}

#[test]
fn feather_touch_neuron_config_expected_multiplier_is_025() {
    assert!(
        (FEATHER_TOUCH_CONFIG.expected_multiplier - 0.25).abs() < 1e-6,
        "Feather-Touch expected_multiplier should be 0.25, got {}",
        FEATHER_TOUCH_CONFIG.expected_multiplier
    );
}

#[test]
fn feather_touch_neuron_variant_produces_correct_outgoing() {
    let candidate = make_test_neuron_candidate(200.0, 0.1, 50.0);
    let variant = make_neuron_variant(&candidate, &FEATHER_TOUCH_CONFIG);

    // 0.1 * 0.01 = 0.001, within outgoing_abs_max
    assert!(
        variant.outgoing_weight.abs() <= FEATHER_TOUCH_CONFIG.outgoing_abs_max + 1e-6,
        "Feather-Touch outgoing should be at most {}, got {}",
        FEATHER_TOUCH_CONFIG.outgoing_abs_max,
        variant.outgoing_weight
    );
}

#[test]
fn feather_touch_neuron_variant_sets_comment() {
    let candidate = make_test_neuron_candidate(200.0, 0.1, 50.0);
    let variant = make_neuron_variant(&candidate, &FEATHER_TOUCH_CONFIG);

    assert!(
        variant
            .comment
            .as_deref()
            .unwrap_or("")
            .contains("Feather-Touch"),
        "Feather-Touch variant should have appropriate comment, got {:?}",
        variant.comment
    );
}

#[test]
fn feather_touch_neuron_variant_scales_expected_improvement() {
    let candidate = make_test_neuron_candidate(200.0, 0.1, 50.0);
    let variant = make_neuron_variant(&candidate, &FEATHER_TOUCH_CONFIG);

    let expected = 0.2 * FEATHER_TOUCH_CONFIG.expected_multiplier;
    assert!(
        (variant.expected_creature_error_reduction - expected).abs() < 1e-6,
        "Feather-Touch expected improvement should be {expected}, got {}",
        variant.expected_creature_error_reduction
    );
}

// =========================================================================
// Whisper neuron variant config tests
// =========================================================================

#[test]
fn whisper_neuron_config_has_smaller_outgoing_scale_than_feather_touch() {
    assert!(
        WHISPER_CONFIG.outgoing_scale < FEATHER_TOUCH_CONFIG.outgoing_scale,
        "Whisper outgoing_scale ({}) should be smaller than Feather-Touch ({})",
        WHISPER_CONFIG.outgoing_scale,
        FEATHER_TOUCH_CONFIG.outgoing_scale
    );
}

#[test]
fn whisper_neuron_config_outgoing_scale_is_0005() {
    assert!(
        (WHISPER_CONFIG.outgoing_scale - 0.005).abs() < 1e-6,
        "Whisper outgoing_scale should be 0.005, got {}",
        WHISPER_CONFIG.outgoing_scale
    );
}

#[test]
fn whisper_neuron_config_expected_multiplier_is_01() {
    assert!(
        (WHISPER_CONFIG.expected_multiplier - 0.1).abs() < 1e-6,
        "Whisper expected_multiplier should be 0.1, got {}",
        WHISPER_CONFIG.expected_multiplier
    );
}

#[test]
fn whisper_neuron_variant_produces_correct_outgoing() {
    let candidate = make_test_neuron_candidate(200.0, 0.1, 50.0);
    let variant = make_neuron_variant(&candidate, &WHISPER_CONFIG);

    assert!(
        variant.outgoing_weight.abs() <= WHISPER_CONFIG.outgoing_abs_max + 1e-6,
        "Whisper outgoing should be at most {}, got {}",
        WHISPER_CONFIG.outgoing_abs_max,
        variant.outgoing_weight
    );
}

#[test]
fn whisper_neuron_variant_sets_comment() {
    let candidate = make_test_neuron_candidate(200.0, 0.1, 50.0);
    let variant = make_neuron_variant(&candidate, &WHISPER_CONFIG);

    assert!(
        variant.comment.as_deref().unwrap_or("").contains("Whisper"),
        "Whisper variant should have appropriate comment, got {:?}",
        variant.comment
    );
}

// =========================================================================
// Feather-Touch synapse variant config tests
// =========================================================================

#[test]
fn synapse_feather_touch_config_weight_scale_is_005() {
    assert!(
        (SYNAPSE_FEATHER_TOUCH_CONFIG.weight_scale - 0.05).abs() < 1e-6,
        "Synapse Feather-Touch weight_scale should be 0.05, got {}",
        SYNAPSE_FEATHER_TOUCH_CONFIG.weight_scale
    );
}

#[test]
fn synapse_feather_touch_config_expected_multiplier_is_025() {
    assert!(
        (SYNAPSE_FEATHER_TOUCH_CONFIG.expected_multiplier - 0.25).abs() < 1e-6,
        "Synapse Feather-Touch expected_multiplier should be 0.25, got {}",
        SYNAPSE_FEATHER_TOUCH_CONFIG.expected_multiplier
    );
}

#[test]
fn synapse_feather_touch_scales_weight_correctly() {
    let candidate = make_test_synapse_candidate(0.5, 0.1);
    let variant = make_synapse_variant(&candidate, &SYNAPSE_FEATHER_TOUCH_CONFIG);

    let expected_weight = 0.5 * 0.05;
    assert!(
        (variant.weight - expected_weight).abs() < 1e-6,
        "Feather-Touch synapse weight should be {expected_weight}, got {}",
        variant.weight
    );
}

#[test]
fn synapse_feather_touch_sets_comment() {
    let candidate = make_test_synapse_candidate(0.5, 0.1);
    let variant = make_synapse_variant(&candidate, &SYNAPSE_FEATHER_TOUCH_CONFIG);

    assert!(
        variant
            .comment
            .as_deref()
            .unwrap_or("")
            .contains("Feather-Touch"),
        "Feather-Touch synapse variant should have appropriate comment"
    );
}

// =========================================================================
// Whisper synapse variant config tests
// =========================================================================

#[test]
fn synapse_whisper_config_weight_scale_is_002() {
    assert!(
        (SYNAPSE_WHISPER_CONFIG.weight_scale - 0.02).abs() < 1e-6,
        "Synapse Whisper weight_scale should be 0.02, got {}",
        SYNAPSE_WHISPER_CONFIG.weight_scale
    );
}

#[test]
fn synapse_whisper_config_expected_multiplier_is_01() {
    assert!(
        (SYNAPSE_WHISPER_CONFIG.expected_multiplier - 0.1).abs() < 1e-6,
        "Synapse Whisper expected_multiplier should be 0.1, got {}",
        SYNAPSE_WHISPER_CONFIG.expected_multiplier
    );
}

#[test]
fn synapse_whisper_scales_weight_correctly() {
    let candidate = make_test_synapse_candidate(0.5, 0.1);
    let variant = make_synapse_variant(&candidate, &SYNAPSE_WHISPER_CONFIG);

    let expected_weight = 0.5 * 0.02;
    assert!(
        (variant.weight - expected_weight).abs() < 1e-6,
        "Whisper synapse weight should be {expected_weight}, got {}",
        variant.weight
    );
}

#[test]
fn synapse_whisper_sets_comment() {
    let candidate = make_test_synapse_candidate(0.5, 0.1);
    let variant = make_synapse_variant(&candidate, &SYNAPSE_WHISPER_CONFIG);

    assert!(
        variant.comment.as_deref().unwrap_or("").contains("Whisper"),
        "Whisper synapse variant should have appropriate comment"
    );
}

// =========================================================================
// Ultra-conservative threshold gating tests
// =========================================================================

#[test]
fn ultra_conservative_threshold_is_positive() {
    const { assert!(ULTRA_CONSERVATIVE_BASE_WEIGHT_THRESHOLD > 0.0) };
}

#[test]
fn synapse_pairing_generates_ultra_conservative_for_large_weight() {
    // Weight well above threshold — should get Feather-Touch and Whisper variants
    let candidate = make_test_synapse_candidate(0.5, 0.1);
    let result = pair_synapse_candidates_with_weight_variants(vec![candidate], None);

    // Original + Conservative + Gentle Nudge + Micro-Nudge + Feather-Touch + Whisper = up to 6
    assert!(
        result.len() >= 4,
        "large-weight synapse should generate ultra-conservative variants, got {} entries",
        result.len()
    );

    let has_feather_touch = result
        .iter()
        .any(|c| c.comment.as_deref().unwrap_or("").contains("Feather-Touch"));
    let has_whisper = result
        .iter()
        .any(|c| c.comment.as_deref().unwrap_or("").contains("Whisper"));

    assert!(
        has_feather_touch,
        "should generate Feather-Touch variant for large weight"
    );
    assert!(
        has_whisper,
        "should generate Whisper variant for large weight"
    );
}

#[test]
fn synapse_pairing_skips_ultra_conservative_for_small_weight() {
    // Weight below threshold — should NOT get Feather-Touch or Whisper
    let small_weight = ULTRA_CONSERVATIVE_BASE_WEIGHT_THRESHOLD * 0.5;
    let candidate = make_test_synapse_candidate(small_weight, 0.1);
    let result = pair_synapse_candidates_with_weight_variants(vec![candidate], None);

    let has_feather_touch = result
        .iter()
        .any(|c| c.comment.as_deref().unwrap_or("").contains("Feather-Touch"));
    let has_whisper = result
        .iter()
        .any(|c| c.comment.as_deref().unwrap_or("").contains("Whisper"));

    assert!(
        !has_feather_touch,
        "should NOT generate Feather-Touch for small weight ({small_weight})",
    );
    assert!(
        !has_whisper,
        "should NOT generate Whisper for small weight ({small_weight})",
    );
}

// =========================================================================
// Neuron pairing with ultra-conservative variants
// =========================================================================

#[test]
fn neuron_pairing_generates_ultra_conservative_for_extreme_candidate() {
    // Extreme candidate with large outgoing weight (above threshold)
    let candidate = make_test_neuron_candidate(200.0, 0.1, 50.0);
    let result = pair_extreme_candidates_with_conservative_variants(vec![candidate], None);

    let has_feather_touch = result
        .iter()
        .any(|c| c.comment.as_deref().unwrap_or("").contains("Feather-Touch"));
    let has_whisper = result
        .iter()
        .any(|c| c.comment.as_deref().unwrap_or("").contains("Whisper"));

    assert!(
        has_feather_touch,
        "should generate Feather-Touch variant for extreme neuron candidate"
    );
    assert!(
        has_whisper,
        "should generate Whisper variant for extreme neuron candidate"
    );
}

#[test]
fn neuron_pairing_skips_ultra_conservative_for_small_outgoing() {
    // Extreme incoming but very small outgoing (below threshold)
    let small_outgoing = ULTRA_CONSERVATIVE_BASE_WEIGHT_THRESHOLD * 0.5;
    let candidate = make_test_neuron_candidate(200.0, small_outgoing, 50.0);
    let result = pair_extreme_candidates_with_conservative_variants(vec![candidate], None);

    let has_feather_touch = result
        .iter()
        .any(|c| c.comment.as_deref().unwrap_or("").contains("Feather-Touch"));
    let has_whisper = result
        .iter()
        .any(|c| c.comment.as_deref().unwrap_or("").contains("Whisper"));

    assert!(
        !has_feather_touch,
        "should NOT generate Feather-Touch for small outgoing weight"
    );
    assert!(
        !has_whisper,
        "should NOT generate Whisper for small outgoing weight"
    );
}

// =========================================================================
// Coordinated-structural with ultra-conservative variants
// =========================================================================

#[test]
fn coordinated_pairing_generates_ultra_conservative_for_large_weight() {
    let candidate = CoordinatedStructuralCandidateJson {
        operations: vec![CoordinatedStructuralOpJson::AddSynapse {
            from_neuron_uuid: "n1".to_string(),
            to_neuron_uuid: "n2".to_string(),
            weight: 0.5,
        }],
        expected_creature_score_gain: 0.1,
        comment: None,
    };

    let result = pair_coordinated_structural_with_weight_variants(vec![candidate], None);

    let has_feather_touch = result
        .iter()
        .any(|c| c.comment.as_deref().unwrap_or("").contains("Feather-Touch"));
    let has_whisper = result
        .iter()
        .any(|c| c.comment.as_deref().unwrap_or("").contains("Whisper"));

    assert!(
        has_feather_touch,
        "should generate Feather-Touch coordinated variant for large weight"
    );
    assert!(
        has_whisper,
        "should generate Whisper coordinated variant for large weight"
    );
}

#[test]
fn coordinated_pairing_skips_ultra_conservative_for_small_weight() {
    let small_weight = ULTRA_CONSERVATIVE_BASE_WEIGHT_THRESHOLD * 0.5;
    let candidate = CoordinatedStructuralCandidateJson {
        operations: vec![CoordinatedStructuralOpJson::AddSynapse {
            from_neuron_uuid: "n1".to_string(),
            to_neuron_uuid: "n2".to_string(),
            weight: small_weight,
        }],
        expected_creature_score_gain: 0.1,
        comment: None,
    };

    let result = pair_coordinated_structural_with_weight_variants(vec![candidate], None);

    let has_feather_touch = result
        .iter()
        .any(|c| c.comment.as_deref().unwrap_or("").contains("Feather-Touch"));
    let has_whisper = result
        .iter()
        .any(|c| c.comment.as_deref().unwrap_or("").contains("Whisper"));

    assert!(
        !has_feather_touch,
        "should NOT generate Feather-Touch coordinated variant for small weight"
    );
    assert!(
        !has_whisper,
        "should NOT generate Whisper coordinated variant for small weight"
    );
}
