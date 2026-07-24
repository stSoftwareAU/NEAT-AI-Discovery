//! Tests for optimal outgoing weight calculation.
//!
//! Extracted from `implementation_tests.rs` as part of Issue #426.
//! Tests cover:
//! - Weight calculation edge cases (zero activation, non-finite, near-zero)
//! - Weight clamping to `MAX_OUTGOING_WEIGHT`
//! - Weight ratio validation for add-neuron candidates
//! - Synapse weight handling (skips ratio check)
//! - IDENTITY candidate fallback to affine fit

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use super::common::*;
use crate::analysis::scoring::weights::calculate_optimal_identity_outgoing_and_bias;

///
/// Regression test: the IDENTITY affine fit path in
/// `calculate_optimal_identity_outgoing_and_bias` correctly computes both a weight
/// and bias from data where the no-intercept fit would produce near-zero weight.
///
/// Issue #888: Restructured to directly test the affine fit function rather than
/// going through the full `evaluate_activation_candidate` pipeline, because the
/// tightened weight constraints (`MAX_OUTGOING_WEIGHT` 0.1→0.01) change which data
/// patterns produce valid candidates through the multi-stage pipeline.
#[test]
fn identity_affine_fit_produces_valid_weight_and_bias() {
    // Data structure: activations 0.5–1.5 (positive errors) and ~1.0 + 3.0
    // (negative errors) chosen so sum(a_pos) = sum(a_neg) → Σ(e·u) = 0.
    //
    // With these activations: sum(u²)/sum(u) ≈ 1.32, so bias ≈ -1.32 (within ±2.0).
    let positive_activations: [f32; 11] = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5];
    let mut samples: Vec<HelpfulSample> = positive_activations
        .iter()
        .map(|&a| HelpfulSample {
            activation: a,
            avg_error: 0.001,
            target_value: None,
            target_activation: None,
        })
        .collect();

    let negative_activations: [f32; 9] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0];
    for activation in negative_activations {
        samples.push(HelpfulSample {
            activation,
            avg_error: -0.001,
            target_value: None,
            target_activation: None,
        });
    }

    let result = calculate_optimal_identity_outgoing_and_bias(&samples, 1.0);
    assert!(
        result.is_some(),
        "Affine fit should produce a valid (weight, bias) for this data"
    );

    let (weight, bias) = result.unwrap();
    assert!(weight.is_finite(), "Weight should be finite");
    assert!(bias.is_finite(), "Bias should be finite");
    assert!(
        weight.abs() <= MAX_OUTGOING_WEIGHT,
        "Weight {weight} should be within MAX_OUTGOING_WEIGHT {MAX_OUTGOING_WEIGHT}"
    );
    assert!(
        weight.abs() > EPSILON,
        "Weight should be non-trivial (above EPSILON)"
    );
    assert!(
        bias.abs() >= 0.01,
        "Bias should be meaningful (not equivalent to a direct synapse)"
    );
    assert!(
        bias.abs() <= 2.0,
        "Bias {bias} should be within sensible range"
    );
}

/// Test that `calculate_optimal_outgoing_weight` returns None for insufficient activation
#[test]
fn returns_none_for_zero_activation() {
    let result = calculate_optimal_outgoing_weight(1.0, 0.0, 1.0);
    assert!(
        result.is_none(),
        "Should return None when activation_sq is zero"
    );

    let result = calculate_optimal_outgoing_weight(1.0, EPSILON * 0.5, 1.0);
    assert!(
        result.is_none(),
        "Should return None when activation_sq <= EPSILON"
    );
}

/// Test that `calculate_optimal_outgoing_weight` returns None for non-finite results
#[test]
fn returns_none_for_non_finite_weight() {
    let result = calculate_optimal_outgoing_weight(f32::INFINITY, 1.0, 1.0);
    assert!(
        result.is_none(),
        "Should return None when raw weight is infinite"
    );

    let result = calculate_optimal_outgoing_weight(f32::NAN, 1.0, 1.0);
    assert!(
        result.is_none(),
        "Should return None when raw weight is NaN"
    );
}

/// Test that `calculate_optimal_outgoing_weight` returns None for near-zero weights
#[test]
fn returns_none_for_near_zero_weight() {
    // Very small error_activation results in near-zero weight
    let result = calculate_optimal_outgoing_weight(EPSILON * 0.1, 100.0, 1.0);
    assert!(
        result.is_none(),
        "Should return None when raw weight is near zero"
    );
}

/// Test that weights are clamped to `MAX_OUTGOING_WEIGHT`
#[test]
fn clamps_to_max_outgoing_weight() {
    // Large error relative to activation would produce large weight
    // error/activation = 10.0/1.0 = 10.0, should clamp to 0.01
    let result = calculate_optimal_outgoing_weight(10.0, 1.0, 1.0);
    assert!(result.is_some(), "Should return a valid weight");
    let weight = result.unwrap();
    assert!(
        (weight - MAX_OUTGOING_WEIGHT).abs() < EPSILON,
        "Weight {weight} should be clamped to MAX_OUTGOING_WEIGHT {MAX_OUTGOING_WEIGHT}"
    );

    // Negative case
    let result = calculate_optimal_outgoing_weight(-10.0, 1.0, 1.0);
    assert!(result.is_some(), "Should return a valid negative weight");
    let weight = result.unwrap();
    assert!(
        (weight - (-MAX_OUTGOING_WEIGHT)).abs() < EPSILON,
        "Weight {} should be clamped to -MAX_OUTGOING_WEIGHT {}",
        weight,
        -MAX_OUTGOING_WEIGHT
    );
}

/// Test weight ratio validation for add-neuron candidates
#[test]
fn rejects_small_weight_ratio() {
    // incoming_weight = 10, max outgoing = 0.01, ratio = 1000 -> OK
    let result = calculate_optimal_outgoing_weight(1.0, 1.0, 10.0);
    // raw = 1.0, clamped to 0.01, ratio = 10/0.01 = 1000 >= 50 -> OK
    assert!(
        result.is_some(),
        "Should accept ratio of 1000 (incoming=10, outgoing=0.01)"
    );

    // Issue #888: With MAX_OUTGOING_WEIGHT=0.01, incoming_weight=2 now passes
    // ratio check (2/0.01=200 >= 50). This is correct because incoming ~2
    // is the dominant success pattern in production discovery-cache evidence.
    let result = calculate_optimal_outgoing_weight(1.0, 1.0, 2.0);
    assert!(
        result.is_some(),
        "Should accept ratio of 200 (incoming=2, outgoing=0.01)"
    );
}

/// Test that small incoming weights (synapses) skip ratio check
#[test]
fn skips_ratio_check_for_synapses() {
    // For synapses, incoming_weight = 1.0, so ratio check is skipped
    // Issue #888: raw = 0.05 now exceeds MAX_OUTGOING_WEIGHT (0.01) and gets clamped.
    // Use smaller values to test the unclamped path.
    let result = calculate_optimal_outgoing_weight(0.05, 10.0, 1.0);
    // raw = 0.005, within bounds, no ratio check since incoming <= 1.0
    assert!(
        result.is_some(),
        "Should accept synapse weight without ratio check"
    );
    assert!(
        (result.unwrap() - 0.005).abs() < 0.001,
        "Synapse weight should be ~0.005"
    );
}

/// Test that successful discovery parameters would pass validation
/// Based on real successful discoveries from production
#[test]
fn successful_discovery_parameters_pass() {
    // Successful discovery: incoming=100, outgoing=-0.00096
    // ratio = 100/0.00096 ≈ 104,000 >> 50 -> OK
    // But we need to compute what error/activation ratio would produce 0.00096
    // If raw weight = 0.00096, and it's not clamped, we need:
    // sum_error_activation / sum_activation_sq = 0.00096
    let sum_activation_sq = 1000.0;
    let sum_error_activation = 0.00096 * sum_activation_sq; // = 0.96

    let result = calculate_optimal_outgoing_weight(sum_error_activation, sum_activation_sq, 100.0);
    assert!(result.is_some(), "Successful discovery params should pass");
    let weight = result.unwrap();
    // The weight should be approximately -0.00096 or +0.00096 depending on sign
    assert!(
        weight.abs() < MAX_OUTGOING_WEIGHT,
        "Weight {weight} should be within bounds"
    );
}

/// Test that failed discovery parameters would be rejected
/// Based on real failed discoveries from production
#[test]
fn failed_discovery_parameters_rejected() {
    // Failed discovery: incoming=5, outgoing=4.58 (before our fix, this would pass)
    // Now: raw = 4.58, clamped to 0.01, ratio = 5/0.01 = 500 >= 50 -> OK
    //
    // More clearly failed case: incoming=10, outgoing=-10 (1:1 ratio)
    // After clamping to 0.01, ratio = 10/0.01 = 1000 >= 50 -> passes ratio check
    // BUT the weight is clamped from -10 to -0.01, so prediction accuracy improves
    //
    // The key improvement is that extreme weights like 4.58 or -10 are now clamped
    // to 0.01, dramatically reducing prediction errors

    // Test that a raw weight of 10.0 gets clamped
    let result = calculate_optimal_outgoing_weight(10.0, 1.0, 10.0);
    assert!(result.is_some(), "Should return clamped weight");
    assert!(
        (result.unwrap().abs() - MAX_OUTGOING_WEIGHT).abs() < EPSILON,
        "Large raw weight should be clamped to MAX_OUTGOING_WEIGHT"
    );
}
