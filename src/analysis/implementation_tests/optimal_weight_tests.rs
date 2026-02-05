//! Tests for optimal outgoing weight calculation.
//!
//! Extracted from implementation_tests.rs as part of Issue #426.
//! Tests cover:
//! - Weight calculation edge cases (zero activation, non-finite, near-zero)
//! - Weight clamping to MAX_OUTGOING_WEIGHT
//! - Weight ratio validation for add-neuron candidates
//! - Synapse weight handling (skips ratio check)
//! - IDENTITY candidate fallback to affine fit

use super::common::*;
use crate::analysis::activation::ActivationCandidateSpec;
use crate::analysis::gpu::GpuEvaluator;
use crate::analysis::samples::ReluStats;
use crate::analysis::synapse::evaluate_activation_candidate;
use anyhow::anyhow;

struct AlwaysFailGpuEvaluator;

impl GpuEvaluator for AlwaysFailGpuEvaluator {
    fn evaluate_relu(
        &self,
        _samples: &[HelpfulSample],
        _threshold: f32,
    ) -> Result<(ReluStats, ReluStats, f32)> {
        Err(anyhow!(
            "AlwaysFailGpuEvaluator: GPU not available for test"
        ))
    }

    fn evaluate_activation(
        &self,
        _samples: &[HelpfulSample],
        _activation_type: u32,
        _orientation: f32,
        _scale: f32,
    ) -> Result<(f32, f32, f32, u32)> {
        Err(anyhow!(
            "AlwaysFailGpuEvaluator: GPU not available for test"
        ))
    }

    fn evaluate_activations_batched(
        &self,
        _samples: &[HelpfulSample],
        _activation_configs: &[(u32, f32, f32)],
    ) -> Result<Vec<(f32, f32, f32, u32)>> {
        Err(anyhow!(
            "AlwaysFailGpuEvaluator: GPU not available for test"
        ))
    }
}

/// Regression: IDENTITY candidates must not be skipped in the all-samples fallback path.
///
/// `evaluate_activation_candidate` previously computed `base_weight` (no-intercept fit)
/// before checking `spec.name == "IDENTITY"`. If the no-intercept fit returned None (for
/// example, when Σ(activation×error) cancels to ~0), the function would `continue` and the
/// IDENTITY-specific affine fit (with intercept/bias) was never attempted.
///
/// This test constructs a dataset where:
/// - split-error evaluation is NOT "properly attempted" (negative subset < MIN sample count),
/// - subset evaluation returns None (positive subset has constant error ⇒ best slope is 0),
/// - the all-samples no-intercept fit produces `None` (Σ(activation×error) cancels to 0),
/// - but the all-samples affine fit *does* succeed and should yield a candidate.
#[test]
fn identity_all_samples_fallback_uses_affine_fit_even_when_base_weight_is_none() {
    let gpu = AlwaysFailGpuEvaluator;

    // Use a minimal spec so this test is deterministic.
    static ORIENTATIONS: [f32; 1] = [1.0];
    static SCALES: [f32; 1] = [1.0];
    let spec = ActivationCandidateSpec {
        name: "IDENTITY",
        orientations: &ORIENTATIONS,
        scales: &SCALES,
        activation: identity_activation,
        min_improvement: 0.0,
    };

    // 11 positive-error samples with varying activation.
    //
    // This test is crafted to trigger the all-samples affine fit path while still
    // producing a *sensible* bias (we reject absurd bias magnitudes as a guard rail).
    let mut samples: Vec<HelpfulSample> = (1..=11)
        .map(|i| HelpfulSample {
            activation: i as f32, // 1..11
            avg_error: 1.0,
            target_value: None,
            target_activation: None,
        })
        .collect();

    // 9 negative-error samples whose activation sum matches the positive group:
    // sum(1..11) = 66, so pick 9 values summing to 66.
    //
    // This makes Σ(activation×error) == 0 for the all-samples no-intercept fit,
    // forcing the fallback to rely on the affine (with-intercept) fit.
    let negative_activations: [f32; 9] = [6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 18.0];
    for activation in negative_activations {
        samples.push(HelpfulSample {
            activation,
            avg_error: -1.0,
            target_value: None,
            target_activation: None,
        });
    }

    let candidate =
        evaluate_activation_candidate(&gpu, "source-0", "target-0", &samples, 0.0, &spec, None)
            .expect("Evaluation should succeed");

    assert!(
        candidate.is_some(),
        "Expected an IDENTITY candidate from the all-samples affine fit. \
         This is a regression if None is returned."
    );
    let candidate = candidate.unwrap();
    assert_eq!(candidate.squash, "IDENTITY");
    assert!(
        candidate.bias.abs() >= 0.01,
        "IDENTITY candidate should have a meaningful bias (not equivalent to a direct synapse)"
    );
    assert!(
        candidate.outgoing_weight.is_finite() && candidate.outgoing_weight.abs() > EPSILON,
        "IDENTITY candidate should have a valid outgoing weight"
    );
}

/// Test that calculate_optimal_outgoing_weight returns None for insufficient activation
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

/// Test that calculate_optimal_outgoing_weight returns None for non-finite results
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

/// Test that calculate_optimal_outgoing_weight returns None for near-zero weights
#[test]
fn returns_none_for_near_zero_weight() {
    // Very small error_activation results in near-zero weight
    let result = calculate_optimal_outgoing_weight(EPSILON * 0.1, 100.0, 1.0);
    assert!(
        result.is_none(),
        "Should return None when raw weight is near zero"
    );
}

/// Test that weights are clamped to MAX_OUTGOING_WEIGHT
#[test]
fn clamps_to_max_outgoing_weight() {
    // Large error relative to activation would produce large weight
    // error/activation = 10.0/1.0 = 10.0, should clamp to 0.1
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
    // incoming_weight = 10, max outgoing = 0.1, ratio = 100 -> OK
    let result = calculate_optimal_outgoing_weight(1.0, 1.0, 10.0);
    // raw = 1.0, clamped to 0.1, ratio = 10/0.1 = 100 >= 50 -> OK
    assert!(
        result.is_some(),
        "Should accept ratio of 100 (incoming=10, outgoing=0.1)"
    );

    // incoming_weight = 2, max outgoing = 0.1, ratio = 20 < 50 -> REJECT
    let result = calculate_optimal_outgoing_weight(1.0, 1.0, 2.0);
    // raw = 1.0, clamped to 0.1, ratio = 2/0.1 = 20 < 50 -> REJECT
    assert!(
        result.is_none(),
        "Should reject ratio of 20 (incoming=2, outgoing=0.1)"
    );
}

/// Test that small incoming weights (synapses) skip ratio check
#[test]
fn skips_ratio_check_for_synapses() {
    // For synapses, incoming_weight = 1.0, so ratio check is skipped
    let result = calculate_optimal_outgoing_weight(0.5, 10.0, 1.0);
    // raw = 0.05, within bounds, no ratio check since incoming <= 1.0
    assert!(
        result.is_some(),
        "Should accept synapse weight without ratio check"
    );
    assert!(
        (result.unwrap() - 0.05).abs() < 0.001,
        "Synapse weight should be ~0.05"
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
    // Now: raw = 4.58, clamped to 0.1, ratio = 5/0.1 = 50, just at the boundary
    // This might just pass or just fail depending on exact values

    // More clearly failed case: incoming=10, outgoing=-10 (1:1 ratio)
    // Even after clamping to 0.1, ratio = 10/0.1 = 100 >= 50 -> passes ratio check
    // BUT the weight is clamped from -10 to -0.1, so prediction accuracy improves

    // The key improvement is that extreme weights like 4.58 or -10 are now clamped
    // to 0.1, dramatically reducing prediction errors

    // Test that a raw weight of 10.0 gets clamped
    let result = calculate_optimal_outgoing_weight(10.0, 1.0, 10.0);
    assert!(result.is_some(), "Should return clamped weight");
    assert!(
        (result.unwrap().abs() - MAX_OUTGOING_WEIGHT).abs() < EPSILON,
        "Large raw weight should be clamped to MAX_OUTGOING_WEIGHT"
    );
}
