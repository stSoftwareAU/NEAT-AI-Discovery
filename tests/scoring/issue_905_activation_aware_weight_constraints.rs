//! Tests for Issue #905: Weight constraints must not systematically reject
//! non-IDENTITY and hidden-source candidates.
//!
//! The tightened constraints from Issue #888 (`MAX_OUTGOING_WEIGHT = 0.01`,
//! `MIN_WEIGHT_RATIO = 50.0`) were calibrated on IDENTITY-dominated success
//! data. Non-linear activations (TANH, GELU, etc.) operate in different weight
//! regimes and need relaxed constraints to avoid systematic rejection.
//!
//! ## Key Behaviours Verified
//!
//! - `calculate_activation_aware_outgoing_weight` uses relaxed constraints for
//!   non-linear activations
//! - Non-linear candidates with valid weights are not rejected
//! - IDENTITY candidates still use tight constraints (no regression)
//! - Hidden-source candidates with smaller incoming weights pass validation

use neat_ai_discovery::analysis::scoring::weights::{
    MAX_OUTGOING_WEIGHT, calculate_activation_aware_outgoing_weight,
    calculate_optimal_outgoing_weight, max_outgoing_weight_for_activation,
    min_weight_ratio_for_activation,
};

// =============================================================================
// Activation-Aware Constraint Functions
// =============================================================================

#[test]
fn test_identity_gets_tight_max_outgoing_weight() {
    let max_out = max_outgoing_weight_for_activation("IDENTITY");
    assert!(
        (max_out - MAX_OUTGOING_WEIGHT).abs() < 1e-6,
        "IDENTITY should use the tight MAX_OUTGOING_WEIGHT ({MAX_OUTGOING_WEIGHT}), got {max_out}"
    );
}

#[test]
fn test_tanh_gets_relaxed_max_outgoing_weight() {
    let max_out = max_outgoing_weight_for_activation("TANH");
    assert!(
        max_out > MAX_OUTGOING_WEIGHT,
        "TANH should have a larger max outgoing weight than IDENTITY ({MAX_OUTGOING_WEIGHT}), got {max_out}"
    );
}

#[test]
fn test_gelu_gets_relaxed_max_outgoing_weight() {
    let max_out = max_outgoing_weight_for_activation("GELU");
    assert!(
        max_out > MAX_OUTGOING_WEIGHT,
        "GELU should have a larger max outgoing weight than IDENTITY ({MAX_OUTGOING_WEIGHT}), got {max_out}"
    );
}

#[test]
fn test_relu_gets_relaxed_max_outgoing_weight() {
    let max_out = max_outgoing_weight_for_activation("ReLU");
    assert!(
        max_out > MAX_OUTGOING_WEIGHT,
        "ReLU should have a larger max outgoing weight than IDENTITY ({MAX_OUTGOING_WEIGHT}), got {max_out}"
    );
}

#[test]
fn test_identity_gets_tight_min_weight_ratio() {
    let min_ratio = min_weight_ratio_for_activation("IDENTITY");
    assert!(
        (min_ratio - 50.0).abs() < 1e-6,
        "IDENTITY should use the tight MIN_WEIGHT_RATIO (50.0), got {min_ratio}"
    );
}

#[test]
fn test_non_linear_gets_relaxed_min_weight_ratio() {
    for activation in &["TANH", "GELU", "ReLU", "HARD_TANH", "CLIPPED"] {
        let min_ratio = min_weight_ratio_for_activation(activation);
        assert!(
            min_ratio < 50.0,
            "{activation} should have a lower min weight ratio than IDENTITY (50.0), got {min_ratio}"
        );
        assert!(
            min_ratio > 0.0,
            "{activation} min weight ratio should be positive, got {min_ratio}"
        );
    }
}

// =============================================================================
// Activation-Aware Weight Calculation
// =============================================================================

#[test]
fn test_activation_aware_identity_matches_default() {
    // IDENTITY should produce the same result as the default function
    let default_result = calculate_optimal_outgoing_weight(1.0, 1.0, 2.0);
    let aware_result = calculate_activation_aware_outgoing_weight(1.0, 1.0, 2.0, "IDENTITY");
    assert_eq!(
        default_result, aware_result,
        "IDENTITY activation-aware result should match default"
    );
}

#[test]
fn test_activation_aware_tanh_allows_larger_weight() {
    // With a raw weight of 0.03, IDENTITY would clamp to 0.01,
    // but TANH should allow the larger weight
    let sum_error_activation = 0.3;
    let sum_activation_sq = 10.0; // raw_weight = 0.3/10 = 0.03
    let incoming_weight = 1.0;

    let identity_result =
        calculate_optimal_outgoing_weight(sum_error_activation, sum_activation_sq, incoming_weight);
    let tanh_result = calculate_activation_aware_outgoing_weight(
        sum_error_activation,
        sum_activation_sq,
        incoming_weight,
        "TANH",
    );

    assert!(identity_result.is_some());
    assert!(tanh_result.is_some());

    let identity_weight = identity_result.unwrap();
    let tanh_weight = tanh_result.unwrap();

    assert!(
        (identity_weight - MAX_OUTGOING_WEIGHT).abs() < 1e-6,
        "IDENTITY weight should be clamped to {MAX_OUTGOING_WEIGHT}, got {identity_weight}"
    );
    assert!(
        tanh_weight > identity_weight,
        "TANH weight ({tanh_weight}) should be larger than clamped IDENTITY weight ({identity_weight})"
    );
    assert!(
        (tanh_weight - 0.03).abs() < 0.001,
        "TANH weight should be ~0.03 (unclamped), got {tanh_weight}"
    );
}

#[test]
fn test_activation_aware_non_linear_passes_ratio_with_smaller_incoming() {
    // Hidden-source candidate: incoming_weight=1.5, raw outgoing ~0.03
    // IDENTITY: clamped to 0.01, ratio=150, passes (but weight was distorted)
    // TANH: not clamped, ratio=50 with relaxed threshold, passes
    let sum_error_activation = 0.3;
    let sum_activation_sq = 10.0; // raw_weight = 0.03
    let incoming_weight = 1.5;

    let tanh_result = calculate_activation_aware_outgoing_weight(
        sum_error_activation,
        sum_activation_sq,
        incoming_weight,
        "TANH",
    );
    assert!(
        tanh_result.is_some(),
        "TANH candidate with incoming=1.5 should pass activation-aware validation"
    );
}

#[test]
fn test_activation_aware_still_rejects_invalid_weights() {
    // Even with relaxed constraints, invalid weights should still be rejected
    let result = calculate_activation_aware_outgoing_weight(f32::NAN, 1.0, 1.0, "TANH");
    assert!(result.is_none(), "NaN input should still be rejected");

    let result = calculate_activation_aware_outgoing_weight(1.0, 0.0, 1.0, "TANH");
    assert!(
        result.is_none(),
        "Zero activation_sq should still be rejected"
    );

    let result = calculate_activation_aware_outgoing_weight(f32::INFINITY, 1.0, 1.0, "GELU");
    assert!(result.is_none(), "Infinite input should still be rejected");
}

#[test]
fn test_activation_aware_no_regression_for_identity() {
    // Ensure IDENTITY candidates work exactly as before
    // incoming=2, raw_weight=10 -> clamped to 0.01, ratio=200 -> passes
    let result = calculate_activation_aware_outgoing_weight(10.0, 1.0, 2.0, "IDENTITY");
    assert!(result.is_some());
    let weight = result.unwrap();
    assert!(
        (weight - MAX_OUTGOING_WEIGHT).abs() < 1e-6,
        "IDENTITY weight should be clamped to MAX_OUTGOING_WEIGHT"
    );

    // incoming=1.0, ratio check skipped -> passes
    let result = calculate_activation_aware_outgoing_weight(10.0, 1.0, 1.0, "IDENTITY");
    assert!(
        result.is_some(),
        "IDENTITY with incoming=1.0 should skip ratio check"
    );
}

// =============================================================================
// Non-Linear Candidate Acceptance
// =============================================================================

#[test]
fn test_tanh_candidate_not_systematically_rejected() {
    // TANH neuron with incoming_weight=0.5 (hidden source), raw outgoing ~0.005
    // This should pass: small weight, no ratio check (incoming <= 1.0)
    let sum_error_activation = 0.05;
    let sum_activation_sq = 10.0; // raw_weight = 0.005
    let incoming_weight = 0.5;

    let result = calculate_activation_aware_outgoing_weight(
        sum_error_activation,
        sum_activation_sq,
        incoming_weight,
        "TANH",
    );
    assert!(
        result.is_some(),
        "TANH candidate with small incoming weight should not be rejected"
    );
}

#[test]
fn test_gelu_candidate_with_moderate_weight_accepted() {
    // GELU neuron with incoming_weight=2.0, raw outgoing ~0.02
    // IDENTITY would clamp to 0.01; GELU should keep 0.02
    let sum_error_activation = 0.2;
    let sum_activation_sq = 10.0; // raw_weight = 0.02
    let incoming_weight = 2.0;

    let identity_result =
        calculate_optimal_outgoing_weight(sum_error_activation, sum_activation_sq, incoming_weight);
    let gelu_result = calculate_activation_aware_outgoing_weight(
        sum_error_activation,
        sum_activation_sq,
        incoming_weight,
        "GELU",
    );

    // IDENTITY clamps to 0.01
    assert!(identity_result.is_some());
    assert!(
        (identity_result.unwrap() - MAX_OUTGOING_WEIGHT).abs() < 1e-6,
        "IDENTITY should clamp to MAX_OUTGOING_WEIGHT"
    );

    // GELU should keep the unclamped weight
    assert!(gelu_result.is_some());
    assert!(
        gelu_result.unwrap() > MAX_OUTGOING_WEIGHT,
        "GELU should allow weight > MAX_OUTGOING_WEIGHT"
    );
}
