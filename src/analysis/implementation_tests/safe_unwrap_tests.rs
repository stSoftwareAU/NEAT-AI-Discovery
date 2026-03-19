//! Issue #521: Tests verifying safe unwrap in synapse scoring functions.
//!
//! These tests exercise the target simulation code paths in
//! `compute_relu_improvement_and_count` and `compute_activation_improvement_and_count`
//! that access `sample.target_value` and `sample.target_activation`.
//! The functions must use safe unwraps (not `unwrap_unchecked`) to avoid
//! undefined behaviour if invariants are ever violated.

use super::common::*;
use crate::analysis::synapse::scoring::{
    compute_activation_improvement_and_count, compute_relu_improvement_and_count,
};

/// Hard tanh activation for testing (matches production).
fn test_hard_tanh(x: f32) -> f32 {
    x.clamp(-1.0, 1.0)
}

/// Logistic activation for testing activation-function candidates.
fn test_logistic(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// =============================================================================
// compute_relu_improvement_and_count — target simulation path
// =============================================================================

/// Verifies that `compute_relu_improvement_and_count` correctly accesses
/// `target_value` and `target_activation` via safe unwrap when
/// `target_activation_fn` is `Some`.
#[test]
fn relu_improvement_with_target_simulation_uses_safe_access() {
    let samples: Vec<HelpfulSample> = (0..10)
        .map(|_| HelpfulSample {
            activation: 1.0,
            avg_error: 0.2,
            target_value: Some(0.3),
            target_activation: Some(test_hard_tanh(0.3)),
        })
        .collect();

    let baseline_error_sq: f32 = samples.iter().map(|s| s.avg_error.powi(2)).sum();

    let (improvement, improved_count, total_count) = compute_relu_improvement_and_count(
        &samples,
        1.0, // incoming_weight
        0.2, // outgoing_weight
        0.0, // bias
        baseline_error_sq,
        Some(test_hard_tanh),
    );

    assert_eq!(total_count, 10);
    assert!(improved_count > 0, "should improve some samples");
    assert!(improvement > 0.0, "improvement should be positive");
    assert!(improvement.is_finite(), "improvement must be finite");
}

/// Verifies that the `ReLU` function correctly computes improvement in the
/// activation domain when target simulation is enabled.
#[test]
fn relu_improvement_target_simulation_activation_domain() {
    // Place target near the saturation boundary to exercise the
    // activation-domain computation path.
    let target_value = 0.9;
    let samples: Vec<HelpfulSample> = (0..20)
        .map(|_| HelpfulSample {
            activation: 0.5,
            avg_error: 0.3, // Pushes into saturation: 0.9 + 0.3 = 1.2 → clamped to 1.0
            target_value: Some(target_value),
            target_activation: Some(test_hard_tanh(target_value)),
        })
        .collect();

    let baseline_error_sq: f32 = samples.iter().map(|s| s.avg_error.powi(2)).sum();

    // Without target simulation (linear approximation)
    let (improvement_linear, _, _) =
        compute_relu_improvement_and_count(&samples, 1.0, 0.3, 0.0, baseline_error_sq, None);

    // With target simulation (saturation-aware)
    let (improvement_sim, _, _) = compute_relu_improvement_and_count(
        &samples,
        1.0,
        0.3,
        0.0,
        baseline_error_sq,
        Some(test_hard_tanh),
    );

    // Both should be positive but the saturation-aware model should differ
    // from the linear one near boundaries.
    assert!(
        improvement_linear > 0.0,
        "linear improvement should be positive"
    );
    assert!(
        improvement_sim > 0.0,
        "simulated improvement should be positive"
    );
    assert!(
        (improvement_linear - improvement_sim).abs() > EPSILON,
        "saturation-aware model should differ from linear near boundary"
    );
}

// =============================================================================
// compute_activation_improvement_and_count — target simulation path
// =============================================================================

/// Verifies that `compute_activation_improvement_and_count` correctly accesses
/// `target_value` and `target_activation` via safe unwrap when
/// `target_activation_fn` is `Some`.
#[test]
fn activation_improvement_with_target_simulation_uses_safe_access() {
    let samples: Vec<HelpfulSample> = (0..10)
        .map(|_| HelpfulSample {
            activation: 1.0,
            avg_error: 0.2,
            target_value: Some(0.3),
            target_activation: Some(test_hard_tanh(0.3)),
        })
        .collect();

    let baseline_error_sq: f32 = samples.iter().map(|s| s.avg_error.powi(2)).sum();

    let (improvement, improved_count, total_count) = compute_activation_improvement_and_count(
        &samples,
        1.0,           // incoming_weight
        0.2,           // outgoing_weight
        0.0,           // bias
        test_logistic, // activation_fn (the candidate neuron's squash)
        baseline_error_sq,
        Some(test_hard_tanh), // target_activation_fn
    );

    assert_eq!(total_count, 10);
    assert!(improved_count > 0, "should improve some samples");
    assert!(improvement > 0.0, "improvement should be positive");
    assert!(improvement.is_finite(), "improvement must be finite");
}

/// Verifies that the activation function correctly computes improvement in the
/// activation domain when target simulation is enabled near saturation.
#[test]
fn activation_improvement_target_simulation_near_saturation() {
    let target_value = 0.9;
    let samples: Vec<HelpfulSample> = (0..20)
        .map(|_| HelpfulSample {
            activation: 0.5,
            avg_error: 0.3,
            target_value: Some(target_value),
            target_activation: Some(test_hard_tanh(target_value)),
        })
        .collect();

    let baseline_error_sq: f32 = samples.iter().map(|s| s.avg_error.powi(2)).sum();

    // Without target simulation
    let (improvement_linear, _, _) = compute_activation_improvement_and_count(
        &samples,
        1.0,
        0.3,
        0.0,
        test_logistic,
        baseline_error_sq,
        None,
    );

    // With target simulation
    let (improvement_sim, _, _) = compute_activation_improvement_and_count(
        &samples,
        1.0,
        0.3,
        0.0,
        test_logistic,
        baseline_error_sq,
        Some(test_hard_tanh),
    );

    assert!(
        improvement_linear.is_finite(),
        "linear improvement must be finite"
    );
    assert!(
        improvement_sim.is_finite(),
        "simulated improvement must be finite"
    );
}
