//! Tests for Issue #888: Tightened weight constraints to match successful candidate patterns.
//!
//! GRQ-sampler discovery cache shows that successful add-neuron candidates have
//! dramatically different weight magnitudes than failures. These tests verify the
//! tightened constraints and sensible-range filtering.
//!
//! ## Key Behaviours Verified
//!
//! - `MAX_OUTGOING_WEIGHT` is reduced to 0.01
//! - New `MAX_INCOMING_WEIGHT` and `MAX_BIAS_MAGNITUDE` constants are defined
//! - Sensible-range filtering rejects extreme candidates
//! - Micro-Nudge variant is boosted relative to other variants
//! - Weight clamping respects the tightened ceiling
//! - Variant configs match cache-evidence constraints

use neat_ai_discovery::CandidateNeuronJson;
use neat_ai_discovery::analysis::constants::{
    MAX_BIAS_MAGNITUDE, MAX_INCOMING_WEIGHT, MICRO_NUDGE_VARIANT_BOOST,
};
use neat_ai_discovery::analysis::scoring::weights::MAX_OUTGOING_WEIGHT;
use neat_ai_discovery::analysis::utils::variant_generation::{
    CONSERVATIVE_CONFIG, GENTLE_NUDGE_CONFIG, MICRO_NUDGE_CONFIG,
    filter_candidates_to_sensible_ranges, make_neuron_variant,
};

// =============================================================================
// Compile-Time Constant Range Validation
// =============================================================================

// MAX_OUTGOING_WEIGHT must be tightened to 0.01 (Issue #888).
const _: () = assert!(MAX_OUTGOING_WEIGHT <= 0.01);
// Must still be positive.
const _: () = assert!(MAX_OUTGOING_WEIGHT > 0.0);

// MAX_INCOMING_WEIGHT must be reasonable.
const _: () = assert!(MAX_INCOMING_WEIGHT >= 2.0);
const _: () = assert!(MAX_INCOMING_WEIGHT <= 10.0);

// MAX_BIAS_MAGNITUDE must be reasonable.
const _: () = assert!(MAX_BIAS_MAGNITUDE >= 1.0);
const _: () = assert!(MAX_BIAS_MAGNITUDE <= 5.0);

// MICRO_NUDGE_VARIANT_BOOST must be a boost.
const _: () = assert!(MICRO_NUDGE_VARIANT_BOOST >= 1.0);

// =============================================================================
// Helper
// =============================================================================

fn make_test_candidate(incoming: f32, outgoing: f32, bias: f32) -> CandidateNeuronJson {
    CandidateNeuronJson {
        source_neuron_uuid: "src-uuid".to_string(),
        source_neuron_index: Some(0),
        target_neuron_uuid: "tgt-uuid".to_string(),
        target_neuron_index: Some(1),
        squash: "TANH".to_string(),
        incoming_weight: incoming,
        outgoing_weight: outgoing,
        bias,
        target_neuron_impact: 1.0,
        expected_creature_error_reduction: 0.01,
        expected_creature_score_gain: 0.01,
        improved_count: 10,
        total_count: 20,
        target_neuron_stats: None,
        prediction_confidence: 0.5,
        expected_score_gain_confidence_interval: [0.005, 0.015],
        comment: None,
    }
}

// =============================================================================
// Sensible-Range Filtering Tests
// =============================================================================

#[test]
fn test_sensible_range_accepts_micro_nudge_pattern() {
    // The dominant success pattern: incoming ~2, outgoing ~0.003, bias ~0.5
    let candidates = vec![make_test_candidate(2.0, 0.003, 0.5)];
    let filtered = filter_candidates_to_sensible_ranges(candidates);
    assert_eq!(
        filtered.len(),
        1,
        "Micro-Nudge pattern should pass sensible-range filter"
    );
}

#[test]
fn test_sensible_range_rejects_extreme_incoming_weight() {
    // Incoming weight of 10 is in the failure range
    let candidates = vec![make_test_candidate(10.0, 0.003, 0.5)];
    let filtered = filter_candidates_to_sensible_ranges(candidates);
    assert!(
        filtered.is_empty(),
        "Extreme incoming weight (10.0) should be filtered out"
    );
}

#[test]
fn test_sensible_range_rejects_extreme_bias() {
    // Bias of 5.0 is in the failure range
    let candidates = vec![make_test_candidate(2.0, 0.003, 5.0)];
    let filtered = filter_candidates_to_sensible_ranges(candidates);
    assert!(
        filtered.is_empty(),
        "Extreme bias (5.0) should be filtered out"
    );
}

#[test]
fn test_sensible_range_rejects_extreme_outgoing_weight() {
    // Outgoing weight of 0.05 is in the failure range
    let candidates = vec![make_test_candidate(2.0, 0.05, 0.5)];
    let filtered = filter_candidates_to_sensible_ranges(candidates);
    assert!(
        filtered.is_empty(),
        "Extreme outgoing weight (0.05) should be filtered out"
    );
}

#[test]
fn test_sensible_range_accepts_boundary_values() {
    // Values at the boundary should still pass
    let candidates = vec![make_test_candidate(5.0, 0.01, 2.0)];
    let filtered = filter_candidates_to_sensible_ranges(candidates);
    assert_eq!(
        filtered.len(),
        1,
        "Boundary values should pass sensible-range filter"
    );
}

#[test]
fn test_sensible_range_rejects_negative_extreme_bias() {
    // Negative extreme bias should also be filtered
    let candidates = vec![make_test_candidate(2.0, 0.003, -5.0)];
    let filtered = filter_candidates_to_sensible_ranges(candidates);
    assert!(
        filtered.is_empty(),
        "Negative extreme bias (-5.0) should be filtered out"
    );
}

#[test]
fn test_sensible_range_mixed_candidates() {
    let candidates = vec![
        make_test_candidate(2.0, 0.003, 0.5), // Good: Micro-Nudge pattern
        make_test_candidate(20.0, 0.05, 10.0), // Bad: extreme everything
        make_test_candidate(1.5, 0.001, 0.0), // Good: conservative
        make_test_candidate(2.0, 0.003, -10.0), // Bad: extreme bias
    ];
    let filtered = filter_candidates_to_sensible_ranges(candidates);
    assert_eq!(
        filtered.len(),
        2,
        "Only the two good candidates should survive filtering"
    );
}

// =============================================================================
// Variant Config Tests
// =============================================================================

#[test]
fn test_conservative_config_within_constraints() {
    assert!(
        CONSERVATIVE_CONFIG.incoming_abs_max <= MAX_INCOMING_WEIGHT,
        "Conservative incoming_abs_max should be within MAX_INCOMING_WEIGHT"
    );
    assert!(
        CONSERVATIVE_CONFIG.bias_abs_max <= MAX_BIAS_MAGNITUDE,
        "Conservative bias_abs_max should be within MAX_BIAS_MAGNITUDE"
    );
    assert!(
        CONSERVATIVE_CONFIG.outgoing_abs_max <= MAX_OUTGOING_WEIGHT,
        "Conservative outgoing_abs_max should be within MAX_OUTGOING_WEIGHT"
    );
}

#[test]
fn test_gentle_nudge_config_within_constraints() {
    assert!(
        GENTLE_NUDGE_CONFIG.incoming_abs_max <= MAX_INCOMING_WEIGHT,
        "Gentle Nudge incoming_abs_max should be within MAX_INCOMING_WEIGHT"
    );
    assert!(
        GENTLE_NUDGE_CONFIG.bias_abs_max <= MAX_BIAS_MAGNITUDE,
        "Gentle Nudge bias_abs_max should be within MAX_BIAS_MAGNITUDE"
    );
    assert!(
        GENTLE_NUDGE_CONFIG.outgoing_abs_max <= MAX_OUTGOING_WEIGHT,
        "Gentle Nudge outgoing_abs_max should be within MAX_OUTGOING_WEIGHT"
    );
}

#[test]
fn test_micro_nudge_config_within_constraints() {
    assert!(
        MICRO_NUDGE_CONFIG.incoming_abs_max <= MAX_INCOMING_WEIGHT,
        "Micro-Nudge incoming_abs_max should be within MAX_INCOMING_WEIGHT"
    );
    assert!(
        MICRO_NUDGE_CONFIG.bias_abs_max <= MAX_BIAS_MAGNITUDE,
        "Micro-Nudge bias_abs_max should be within MAX_BIAS_MAGNITUDE"
    );
    assert!(
        MICRO_NUDGE_CONFIG.outgoing_abs_max <= MAX_OUTGOING_WEIGHT,
        "Micro-Nudge outgoing_abs_max should be within MAX_OUTGOING_WEIGHT"
    );
}

#[test]
fn test_micro_nudge_boosted_over_conservative() {
    // Micro-Nudge expected_multiplier should be >= Conservative's, reflecting
    // the dominance of Micro-Nudge in successful candidates (Issue #888).
    assert!(
        MICRO_NUDGE_CONFIG.expected_multiplier >= CONSERVATIVE_CONFIG.expected_multiplier,
        "Micro-Nudge expected_multiplier ({}) should be >= Conservative's ({})",
        MICRO_NUDGE_CONFIG.expected_multiplier,
        CONSERVATIVE_CONFIG.expected_multiplier,
    );
}

// =============================================================================
// make_neuron_variant Tests
// =============================================================================

#[test]
fn test_micro_nudge_variant_clamps_outgoing_weight() {
    // An extreme candidate with large outgoing weight
    let candidate = make_test_candidate(2.0, 0.5, 0.0);
    let variant = make_neuron_variant(&candidate, &MICRO_NUDGE_CONFIG);

    assert!(
        variant.outgoing_weight.abs() <= MICRO_NUDGE_CONFIG.outgoing_abs_max,
        "Micro-Nudge variant outgoing weight ({}) should be clamped to {}",
        variant.outgoing_weight.abs(),
        MICRO_NUDGE_CONFIG.outgoing_abs_max,
    );
}

#[test]
fn test_conservative_variant_clamps_incoming_weight() {
    let candidate = make_test_candidate(10.0, 0.003, 0.0);
    let variant = make_neuron_variant(&candidate, &CONSERVATIVE_CONFIG);

    assert!(
        variant.incoming_weight.abs() <= CONSERVATIVE_CONFIG.incoming_abs_max,
        "Conservative variant incoming weight ({}) should be clamped to {}",
        variant.incoming_weight.abs(),
        CONSERVATIVE_CONFIG.incoming_abs_max,
    );
}

#[test]
fn test_gentle_nudge_variant_clamps_bias() {
    let candidate = make_test_candidate(2.0, 0.003, 8.0);
    let variant = make_neuron_variant(&candidate, &GENTLE_NUDGE_CONFIG);

    assert!(
        variant.bias.abs() <= GENTLE_NUDGE_CONFIG.bias_abs_max,
        "Gentle Nudge variant bias ({}) should be clamped to {}",
        variant.bias.abs(),
        GENTLE_NUDGE_CONFIG.bias_abs_max,
    );
}

#[test]
fn test_variant_preserves_sign() {
    let candidate = make_test_candidate(-2.0, -0.003, -0.5);
    let variant = make_neuron_variant(&candidate, &CONSERVATIVE_CONFIG);

    assert!(
        variant.incoming_weight < 0.0,
        "Negative incoming weight should preserve sign"
    );
    assert!(
        variant.outgoing_weight < 0.0,
        "Negative outgoing weight should preserve sign"
    );
}

// =============================================================================
// Weight Clamping Tests
// =============================================================================

#[test]
fn test_optimal_weight_clamped_to_tightened_ceiling() {
    use neat_ai_discovery::analysis::scoring::weights::calculate_optimal_outgoing_weight;

    // Large raw weight (sum_error_activation=10, sum_activation_sq=1 => raw=10)
    // should be clamped to MAX_OUTGOING_WEIGHT (0.01)
    let result = calculate_optimal_outgoing_weight(10.0, 1.0, 1.0);
    assert!(result.is_some());
    let weight = result.unwrap();
    assert!(
        (weight - MAX_OUTGOING_WEIGHT).abs() < 1e-6,
        "Weight {weight} should be clamped to MAX_OUTGOING_WEIGHT {MAX_OUTGOING_WEIGHT}",
    );
}

#[test]
fn test_incoming_weight_2_passes_ratio_check_with_tightened_ceiling() {
    use neat_ai_discovery::analysis::scoring::weights::calculate_optimal_outgoing_weight;

    // With MAX_OUTGOING_WEIGHT=0.01, incoming=2 gives ratio=200 which passes
    // the MIN_WEIGHT_RATIO (50) check. This is correct because incoming ~2
    // is the dominant success pattern.
    let result = calculate_optimal_outgoing_weight(1.0, 1.0, 2.0);
    assert!(
        result.is_some(),
        "incoming_weight=2 should pass ratio check with tightened ceiling (ratio=200)"
    );
}
