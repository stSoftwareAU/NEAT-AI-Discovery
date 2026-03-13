//! Tests for DRY parameterised variant generation (Issue #806).
//!
//! The three near-identical variant generation functions (conservative, gentle nudge,
//! micro-nudge) have been consolidated into a single parameterised function driven by
//! `NeuronVariantConfig` and `SynapseVariantConfig` structs.
//!
//! These tests verify that the consolidated functions produce identical results to the
//! original implementations and that the config-driven approach is correct.

use neat_ai_discovery::analysis::utils::variant_generation::{
    CONSERVATIVE_CONFIG, GENTLE_NUDGE_CONFIG, MICRO_NUDGE_CONFIG, NeuronVariantConfig,
    SYNAPSE_CONSERVATIVE_CONFIG, SYNAPSE_GENTLE_NUDGE_CONFIG, SYNAPSE_MICRO_NUDGE_CONFIG,
    SynapseVariantConfig, make_neuron_variant, make_synapse_variant,
};
use neat_ai_discovery::{CandidateNeuronJson, CandidateSynapseJson};

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
        target_neuron_stats: None,
        prediction_confidence: 0.0,
        expected_score_gain_confidence_interval: [0.0, 0.0],
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
        target_neuron_stats: None,
        outlier_reduction_info: None,
        prediction_confidence: 0.8,
        expected_score_gain_confidence_interval: [0.0, 0.0],
        comment: None,
    }
}

// =========================================================================
// NeuronVariantConfig tests
// =========================================================================

#[test]
fn conservative_config_clamps_incoming_weight() {
    let candidate = make_test_neuron_candidate(200.0, 0.1, 50.0);
    let variant = make_neuron_variant(&candidate, &CONSERVATIVE_CONFIG);

    assert!(
        (variant.incoming_weight - 2.0).abs() < 1e-6,
        "conservative should clamp incoming to 2.0, got {}",
        variant.incoming_weight
    );
}

#[test]
fn conservative_config_clamps_bias() {
    let candidate = make_test_neuron_candidate(200.0, 0.1, 50.0);
    let variant = make_neuron_variant(&candidate, &CONSERVATIVE_CONFIG);

    assert!(
        (variant.bias - 1.0).abs() < 1e-6,
        "conservative should clamp bias to 1.0, got {}",
        variant.bias
    );
}

#[test]
fn conservative_config_scales_outgoing() {
    let candidate = make_test_neuron_candidate(200.0, 0.1, 50.0);
    let variant = make_neuron_variant(&candidate, &CONSERVATIVE_CONFIG);

    // 0.1 * 0.2 = 0.02, clamped to 0.05 max → 0.02
    let expected = (0.1_f32 * 0.2).clamp(-0.05, 0.05);
    assert!(
        (variant.outgoing_weight - expected).abs() < 1e-6,
        "conservative outgoing should be {expected}, got {}",
        variant.outgoing_weight
    );
}

#[test]
fn conservative_config_scales_expected_improvement() {
    let candidate = make_test_neuron_candidate(200.0, 0.1, 50.0);
    let variant = make_neuron_variant(&candidate, &CONSERVATIVE_CONFIG);

    let expected = 0.2 * 0.5;
    assert!(
        (variant.expected_creature_error_reduction - expected).abs() < 1e-6,
        "conservative expected improvement should be {expected}, got {}",
        variant.expected_creature_error_reduction
    );
    assert!(
        (variant.expected_creature_score_gain - expected).abs() < 1e-6,
        "conservative score gain should match error reduction"
    );
}

#[test]
fn gentle_nudge_config_preserves_moderate_incoming() {
    let candidate = make_test_neuron_candidate(10.0, 0.1, 5.0);
    let variant = make_neuron_variant(&candidate, &GENTLE_NUDGE_CONFIG);

    // Gentle nudge has incoming_abs_max = 20.0, so 10.0 is within range
    assert!(
        (variant.incoming_weight - 10.0).abs() < 1e-6,
        "gentle nudge should preserve incoming within range, got {}",
        variant.incoming_weight
    );
}

#[test]
fn gentle_nudge_config_has_tight_outgoing() {
    let candidate = make_test_neuron_candidate(200.0, 0.1, 50.0);
    let variant = make_neuron_variant(&candidate, &GENTLE_NUDGE_CONFIG);

    // 0.1 * 0.1 = 0.01, clamped to 0.02 max → 0.01
    assert!(
        variant.outgoing_weight.abs() <= 0.02 + 1e-6,
        "gentle nudge outgoing should be at most 0.02, got {}",
        variant.outgoing_weight
    );
}

#[test]
fn micro_nudge_config_produces_ultra_small_outgoing() {
    let candidate = make_test_neuron_candidate(200.0, 0.1, 50.0);
    let variant = make_neuron_variant(&candidate, &MICRO_NUDGE_CONFIG);

    // 0.1 * 0.05 = 0.005, clamped to 0.005 max → 0.005
    assert!(
        variant.outgoing_weight.abs() <= 0.005 + 1e-6,
        "micro-nudge outgoing should be at most 0.005, got {}",
        variant.outgoing_weight
    );
}

#[test]
fn neuron_variant_preserves_negative_signs() {
    let candidate = make_test_neuron_candidate(-200.0, -0.1, -50.0);
    let variant = make_neuron_variant(&candidate, &CONSERVATIVE_CONFIG);

    assert!(
        variant.incoming_weight < 0.0,
        "should preserve negative incoming sign, got {}",
        variant.incoming_weight
    );
    assert!(
        variant.outgoing_weight < 0.0,
        "should preserve negative outgoing sign, got {}",
        variant.outgoing_weight
    );
}

#[test]
fn neuron_variant_uses_fallback_when_outgoing_near_zero() {
    let candidate = make_test_neuron_candidate(200.0, 0.00001, 50.0);
    let variant = make_neuron_variant(&candidate, &CONSERVATIVE_CONFIG);

    // When outgoing * scale ≈ 0, should use fallback (0.01 for conservative)
    assert!(
        variant.outgoing_weight.abs() > 1e-6,
        "should use non-zero fallback, got {}",
        variant.outgoing_weight
    );
}

#[test]
fn neuron_variant_sets_comment() {
    let candidate = make_test_neuron_candidate(200.0, 0.1, 50.0);

    let conservative = make_neuron_variant(&candidate, &CONSERVATIVE_CONFIG);
    assert!(
        conservative
            .comment
            .as_deref()
            .unwrap_or("")
            .contains("Conservative"),
        "conservative variant should have appropriate comment"
    );

    let gentle = make_neuron_variant(&candidate, &GENTLE_NUDGE_CONFIG);
    assert!(
        gentle
            .comment
            .as_deref()
            .unwrap_or("")
            .contains("Gentle Nudge"),
        "gentle nudge variant should have appropriate comment"
    );

    let micro = make_neuron_variant(&candidate, &MICRO_NUDGE_CONFIG);
    assert!(
        micro
            .comment
            .as_deref()
            .unwrap_or("")
            .contains("Micro-Nudge"),
        "micro-nudge variant should have appropriate comment"
    );
}

// =========================================================================
// SynapseVariantConfig tests
// =========================================================================

#[test]
fn synapse_conservative_config_scales_weight() {
    let candidate = make_test_synapse_candidate(0.08, 0.1);
    let variant = make_synapse_variant(&candidate, &SYNAPSE_CONSERVATIVE_CONFIG);

    let expected_weight = 0.08 * 0.5;
    assert!(
        (variant.weight - expected_weight).abs() < 1e-6,
        "conservative synapse weight should be {expected_weight}, got {}",
        variant.weight
    );
}

#[test]
fn synapse_gentle_nudge_config_scales_weight() {
    let candidate = make_test_synapse_candidate(0.08, 0.1);
    let variant = make_synapse_variant(&candidate, &SYNAPSE_GENTLE_NUDGE_CONFIG);

    let expected_weight = 0.08 * 0.25;
    assert!(
        (variant.weight - expected_weight).abs() < 1e-6,
        "gentle nudge synapse weight should be {expected_weight}, got {}",
        variant.weight
    );
}

#[test]
fn synapse_micro_nudge_config_scales_weight() {
    let candidate = make_test_synapse_candidate(0.08, 0.1);
    let variant = make_synapse_variant(&candidate, &SYNAPSE_MICRO_NUDGE_CONFIG);

    let expected_weight = 0.08 * 0.1;
    assert!(
        (variant.weight - expected_weight).abs() < 1e-6,
        "micro-nudge synapse weight should be {expected_weight}, got {}",
        variant.weight
    );
}

#[test]
fn synapse_variant_scales_expected_gain() {
    let candidate = make_test_synapse_candidate(0.08, 0.1);
    let variant = make_synapse_variant(&candidate, &SYNAPSE_CONSERVATIVE_CONFIG);

    let expected_gain = 0.1 * 0.5;
    assert!(
        (variant.expected_creature_score_gain - expected_gain).abs() < 1e-6,
        "conservative synapse gain should be {expected_gain}, got {}",
        variant.expected_creature_score_gain
    );
    assert!(
        (variant.expected_creature_error_reduction - expected_gain).abs() < 1e-6,
        "conservative synapse error reduction should match"
    );
}

#[test]
fn synapse_variant_preserves_negative_weight() {
    let candidate = make_test_synapse_candidate(-0.06, 0.1);
    let variant = make_synapse_variant(&candidate, &SYNAPSE_CONSERVATIVE_CONFIG);

    assert!(
        variant.weight < 0.0,
        "should preserve negative weight sign, got {}",
        variant.weight
    );
    let expected = -0.06 * 0.5;
    assert!(
        (variant.weight - expected).abs() < 1e-6,
        "conservative weight should be {expected}, got {}",
        variant.weight
    );
}

#[test]
fn synapse_variant_sets_comment() {
    let candidate = make_test_synapse_candidate(0.08, 0.1);

    let conservative = make_synapse_variant(&candidate, &SYNAPSE_CONSERVATIVE_CONFIG);
    assert!(
        conservative
            .comment
            .as_deref()
            .unwrap_or("")
            .contains("Conservative"),
        "should have Conservative in comment"
    );

    let gentle = make_synapse_variant(&candidate, &SYNAPSE_GENTLE_NUDGE_CONFIG);
    assert!(
        gentle
            .comment
            .as_deref()
            .unwrap_or("")
            .contains("Gentle Nudge"),
        "should have Gentle Nudge in comment"
    );

    let micro = make_synapse_variant(&candidate, &SYNAPSE_MICRO_NUDGE_CONFIG);
    assert!(
        micro
            .comment
            .as_deref()
            .unwrap_or("")
            .contains("Micro-Nudge"),
        "should have Micro-Nudge in comment"
    );
}

// =========================================================================
// Custom config tests — verify the parameterised approach works generically
// =========================================================================

#[test]
fn custom_neuron_config_produces_expected_variant() {
    let config = NeuronVariantConfig {
        incoming_abs_max: 5.0,
        bias_abs_max: 3.0,
        outgoing_abs_max: 0.03,
        outgoing_scale: 0.15,
        expected_multiplier: 0.6,
        min_outgoing_fallback: 0.008,
        comment: "Custom variant",
    };

    let candidate = make_test_neuron_candidate(100.0, 0.1, 20.0);
    let variant = make_neuron_variant(&candidate, &config);

    assert!(
        (variant.incoming_weight - 5.0).abs() < 1e-6,
        "incoming should be clamped to custom max"
    );
    assert!(
        (variant.bias - 3.0).abs() < 1e-6,
        "bias should be clamped to custom max"
    );
    // 0.1 * 0.15 = 0.015, within 0.03 max
    assert!(
        (variant.outgoing_weight - 0.015).abs() < 1e-6,
        "outgoing should be scaled, got {}",
        variant.outgoing_weight
    );
    let expected_gain = 0.2 * 0.6;
    assert!(
        (variant.expected_creature_error_reduction - expected_gain).abs() < 1e-6,
        "expected improvement should use custom multiplier"
    );
}

#[test]
fn custom_synapse_config_produces_expected_variant() {
    let config = SynapseVariantConfig {
        weight_scale: 0.33,
        expected_multiplier: 0.4,
        comment: "Custom synapse variant",
    };

    let candidate = make_test_synapse_candidate(0.09, 0.2);
    let variant = make_synapse_variant(&candidate, &config);

    let expected_weight = 0.09 * 0.33;
    assert!(
        (variant.weight - expected_weight).abs() < 1e-4,
        "weight should be scaled by custom factor"
    );
    let expected_gain = 0.2 * 0.4;
    assert!(
        (variant.expected_creature_score_gain - expected_gain).abs() < 1e-4,
        "gain should use custom multiplier"
    );
    assert_eq!(
        variant.comment.as_deref().unwrap_or(""),
        "Custom synapse variant"
    );
}
