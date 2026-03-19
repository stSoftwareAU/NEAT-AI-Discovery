//! Synthetic tests to verify prediction accuracy against manual simulation.
//!
//! Extracted from `implementation_tests.rs` as part of Issue #426.
//! These tests create controlled scenarios where we know exactly what the
//! predicted and actual improvements should be.
//!
//! Tests cover:
//! - Prediction vs manual simulation in linear region
//! - Prediction vs manual simulation near saturation
//! - Negative error handling
//! - Mixed error handling
//! - Simulated TypeScript evaluation matching
//! - Contribution direction verification

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use super::common::*;
use crate::analysis::synapse::compute_relu_improvement_and_count;

/// Hard tanh activation for testing (same as production)
fn test_hard_tanh(x: f32) -> f32 {
    x.clamp(-1.0, 1.0)
}

/// Create synthetic samples with known properties.
/// Returns (samples, `baseline_error_sq_sum`).
fn create_synthetic_samples(
    count: usize,
    avg_error: f32,
    source_activation: f32,
    target_value: f32,
) -> (Vec<HelpfulSample>, f32) {
    let samples: Vec<HelpfulSample> = (0..count)
        .map(|_| HelpfulSample {
            activation: source_activation,
            avg_error,
            target_value: Some(target_value),
            target_activation: Some(test_hard_tanh(target_value)),
        })
        .collect();

    let baseline_error_sq: f32 = samples.iter().map(|s| s.avg_error.powi(2)).sum();
    (samples, baseline_error_sq)
}

/// CORE TEST: Verify that our prediction formula gives the correct result.
///
/// This test creates a simple scenario:
/// - 100 samples all with the same properties
/// - Known source activation, error, target value
/// - Compute optimal weight
/// - Predict improvement
/// - Manually simulate what the actual improvement would be
/// - Compare predicted vs manually simulated
#[test]
fn prediction_matches_manual_simulation_linear_region() {
    // Scenario: Target in LINEAR region of HARD_TANH (value between -1 and 1)
    let source_activation = 1.0;
    let avg_error = 0.2; // VALUE domain: need to ADD 0.2 to target value
    let target_value = 0.3; // Current pre-activation (in linear region)
    let incoming_weight = 1.0;
    let bias = 0.0;

    let (samples, baseline_error_sq) =
        create_synthetic_samples(100, avg_error, source_activation, target_value);

    // Compute optimal weight using the production formula
    let error_activation_sum: f32 = samples.iter().map(|s| s.avg_error * s.activation).sum();
    let activation_sq_sum: f32 = samples.iter().map(|s| s.activation.powi(2)).sum();
    let raw_outgoing_weight = error_activation_sum / activation_sq_sum;
    let outgoing_weight = raw_outgoing_weight.clamp(-MAX_OUTGOING_WEIGHT, MAX_OUTGOING_WEIGHT);

    eprintln!(
        "LINEAR REGION TEST: error_act_sum={error_activation_sum:.4}, act_sq_sum={activation_sq_sum:.4}, raw_w={raw_outgoing_weight:.6}, clamped_w={outgoing_weight:.6}"
    );

    // Predict improvement using production function
    let (predicted_improvement, improved_count, total_count) = compute_relu_improvement_and_count(
        &samples,
        incoming_weight,
        outgoing_weight,
        bias,
        baseline_error_sq,
        Some(test_hard_tanh),
    );

    // MANUALLY simulate what the actual improvement would be
    // This is what TypeScript evaluation does
    let mut manual_baseline_error_sq = 0.0f32;
    let mut manual_new_error_sq = 0.0f32;

    for sample in &samples {
        let target_val = sample.target_value.unwrap();
        let target_act = sample.target_activation.unwrap();
        let desired_value = target_val + sample.avg_error;
        let expected_activation = test_hard_tanh(desired_value);

        // Baseline error in ACTIVATION domain
        let baseline_err = expected_activation - target_act;
        manual_baseline_error_sq += baseline_err.powi(2);

        // Simulate the new neuron's contribution
        let pre_act = incoming_weight * sample.activation + bias;
        let relu_output = pre_act.max(0.0);
        let contribution = outgoing_weight * relu_output;

        // New target value after contribution
        let new_target_value = target_val + contribution;
        let new_target_activation = test_hard_tanh(new_target_value);

        // New error in ACTIVATION domain
        let new_err = expected_activation - new_target_activation;
        manual_new_error_sq += new_err.powi(2);
    }

    let manual_improvement = if manual_baseline_error_sq > EPSILON {
        (manual_baseline_error_sq - manual_new_error_sq) / manual_baseline_error_sq
    } else {
        0.0
    };

    eprintln!(
        "LINEAR REGION RESULT: predicted={:.6} ({:.4}%), manual={:.6} ({:.4}%), diff={:.6}",
        predicted_improvement,
        predicted_improvement * 100.0,
        manual_improvement,
        manual_improvement * 100.0,
        (predicted_improvement - manual_improvement).abs()
    );
    eprintln!(
        "  improved_count={improved_count}/{total_count}, manual_baseline_sq={manual_baseline_error_sq:.6}, manual_new_sq={manual_new_error_sq:.6}"
    );

    // Prediction and manual simulation should match closely
    let diff = (predicted_improvement - manual_improvement).abs();
    assert!(
        diff < 0.01,
        "Predicted ({predicted_improvement:.6}) and manual ({manual_improvement:.6}) improvement should match within 1%"
    );

    // Both should be positive (error should decrease)
    assert!(
        predicted_improvement > 0.0,
        "Predicted improvement should be positive"
    );
    assert!(
        manual_improvement > 0.0,
        "Manual improvement should be positive"
    );
}

/// Test with target near SATURATION (value close to 1.0)
#[test]
fn prediction_matches_manual_simulation_saturation_region() {
    // Scenario: Target near SATURATION of HARD_TANH
    let source_activation = 1.0;
    let avg_error = 0.1; // VALUE domain: need to ADD 0.1 to target value
    let target_value = 0.95; // Current pre-activation (near saturation!)
    let incoming_weight = 1.0;
    let bias = 0.0;

    let (samples, baseline_error_sq) =
        create_synthetic_samples(100, avg_error, source_activation, target_value);

    // Compute optimal weight
    let error_activation_sum: f32 = samples.iter().map(|s| s.avg_error * s.activation).sum();
    let activation_sq_sum: f32 = samples.iter().map(|s| s.activation.powi(2)).sum();
    let raw_outgoing_weight = error_activation_sum / activation_sq_sum;
    let outgoing_weight = raw_outgoing_weight.clamp(-MAX_OUTGOING_WEIGHT, MAX_OUTGOING_WEIGHT);

    eprintln!(
        "SATURATION TEST: error_act_sum={error_activation_sum:.4}, act_sq_sum={activation_sq_sum:.4}, raw_w={raw_outgoing_weight:.6}, clamped_w={outgoing_weight:.6}"
    );

    // Predict improvement
    let (predicted_improvement, improved_count, total_count) = compute_relu_improvement_and_count(
        &samples,
        incoming_weight,
        outgoing_weight,
        bias,
        baseline_error_sq,
        Some(test_hard_tanh),
    );

    // Manual simulation
    let mut manual_baseline_error_sq = 0.0f32;
    let mut manual_new_error_sq = 0.0f32;

    for sample in &samples {
        let target_val = sample.target_value.unwrap();
        let target_act = sample.target_activation.unwrap();
        let desired_value = target_val + sample.avg_error;
        let expected_activation = test_hard_tanh(desired_value);

        let baseline_err = expected_activation - target_act;
        manual_baseline_error_sq += baseline_err.powi(2);

        let pre_act = incoming_weight * sample.activation + bias;
        let relu_output = pre_act.max(0.0);
        let contribution = outgoing_weight * relu_output;

        let new_target_value = target_val + contribution;
        let new_target_activation = test_hard_tanh(new_target_value);

        let new_err = expected_activation - new_target_activation;
        manual_new_error_sq += new_err.powi(2);
    }

    let manual_improvement = if manual_baseline_error_sq > EPSILON {
        (manual_baseline_error_sq - manual_new_error_sq) / manual_baseline_error_sq
    } else {
        0.0
    };

    eprintln!(
        "SATURATION RESULT: predicted={:.6} ({:.4}%), manual={:.6} ({:.4}%), diff={:.6}",
        predicted_improvement,
        predicted_improvement * 100.0,
        manual_improvement,
        manual_improvement * 100.0,
        (predicted_improvement - manual_improvement).abs()
    );
    eprintln!(
        "  improved_count={improved_count}/{total_count}, manual_baseline_sq={manual_baseline_error_sq:.6}, manual_new_sq={manual_new_error_sq:.6}"
    );

    // Prediction and manual simulation should match
    let diff = (predicted_improvement - manual_improvement).abs();
    assert!(
        diff < 0.01,
        "Predicted ({predicted_improvement:.6}) and manual ({manual_improvement:.6}) improvement should match within 1%"
    );
}

/// Test with NEGATIVE error (target output should be LOWER)
#[test]
fn prediction_matches_manual_simulation_negative_error() {
    // Scenario: Target output is too HIGH, need to REDUCE it
    let source_activation = 1.0;
    let avg_error = -0.2; // VALUE domain: need to SUBTRACT 0.2 from target value
    let target_value = 0.5; // Current pre-activation
    let incoming_weight = 1.0;
    let bias = 0.0;

    let (samples, baseline_error_sq) =
        create_synthetic_samples(100, avg_error, source_activation, target_value);

    // Compute optimal weight (should be NEGATIVE to reduce error)
    let error_activation_sum: f32 = samples.iter().map(|s| s.avg_error * s.activation).sum();
    let activation_sq_sum: f32 = samples.iter().map(|s| s.activation.powi(2)).sum();
    let raw_outgoing_weight = error_activation_sum / activation_sq_sum;
    let outgoing_weight = raw_outgoing_weight.clamp(-MAX_OUTGOING_WEIGHT, MAX_OUTGOING_WEIGHT);

    eprintln!(
        "NEGATIVE ERROR TEST: error_act_sum={error_activation_sum:.4}, act_sq_sum={activation_sq_sum:.4}, raw_w={raw_outgoing_weight:.6}, clamped_w={outgoing_weight:.6}"
    );

    // Verify optimal weight is negative (to reduce target value)
    assert!(
        outgoing_weight < 0.0,
        "Outgoing weight should be negative to reduce target value"
    );

    // Predict improvement
    let (predicted_improvement, improved_count, total_count) = compute_relu_improvement_and_count(
        &samples,
        incoming_weight,
        outgoing_weight,
        bias,
        baseline_error_sq,
        Some(test_hard_tanh),
    );

    // Manual simulation
    let mut manual_baseline_error_sq = 0.0f32;
    let mut manual_new_error_sq = 0.0f32;

    for sample in &samples {
        let target_val = sample.target_value.unwrap();
        let target_act = sample.target_activation.unwrap();
        let desired_value = target_val + sample.avg_error;
        let expected_activation = test_hard_tanh(desired_value);

        let baseline_err = expected_activation - target_act;
        manual_baseline_error_sq += baseline_err.powi(2);

        let pre_act = incoming_weight * sample.activation + bias;
        let relu_output = pre_act.max(0.0);
        let contribution = outgoing_weight * relu_output;

        let new_target_value = target_val + contribution;
        let new_target_activation = test_hard_tanh(new_target_value);

        let new_err = expected_activation - new_target_activation;
        manual_new_error_sq += new_err.powi(2);
    }

    let manual_improvement = if manual_baseline_error_sq > EPSILON {
        (manual_baseline_error_sq - manual_new_error_sq) / manual_baseline_error_sq
    } else {
        0.0
    };

    eprintln!(
        "NEGATIVE ERROR RESULT: predicted={:.6} ({:.4}%), manual={:.6} ({:.4}%), diff={:.6}",
        predicted_improvement,
        predicted_improvement * 100.0,
        manual_improvement,
        manual_improvement * 100.0,
        (predicted_improvement - manual_improvement).abs()
    );
    eprintln!(
        "  improved_count={improved_count}/{total_count}, manual_baseline_sq={manual_baseline_error_sq:.6}, manual_new_sq={manual_new_error_sq:.6}"
    );

    // Prediction and manual simulation should match
    let diff = (predicted_improvement - manual_improvement).abs();
    assert!(
        diff < 0.01,
        "Predicted ({predicted_improvement:.6}) and manual ({manual_improvement:.6}) improvement should match within 1%"
    );

    // Both should be positive (error should decrease)
    assert!(
        predicted_improvement > 0.0,
        "Predicted improvement should be positive"
    );
    assert!(
        manual_improvement > 0.0,
        "Manual improvement should be positive"
    );
}

/// Test with MIXED errors (some positive, some negative)
/// This simulates real-world scenarios where samples have varied errors.
#[test]
fn prediction_matches_manual_simulation_mixed_errors() {
    // Create samples with varied errors
    let samples: Vec<HelpfulSample> = vec![
        // Samples that need INCREASE (positive error)
        HelpfulSample {
            activation: 1.0,
            avg_error: 0.2,
            target_value: Some(0.3),
            target_activation: Some(0.3),
        },
        HelpfulSample {
            activation: 0.8,
            avg_error: 0.15,
            target_value: Some(0.4),
            target_activation: Some(0.4),
        },
        HelpfulSample {
            activation: 1.2,
            avg_error: 0.1,
            target_value: Some(0.2),
            target_activation: Some(0.2),
        },
        // Samples that need DECREASE (negative error)
        HelpfulSample {
            activation: 0.9,
            avg_error: -0.15,
            target_value: Some(0.6),
            target_activation: Some(0.6),
        },
        HelpfulSample {
            activation: 1.1,
            avg_error: -0.1,
            target_value: Some(0.5),
            target_activation: Some(0.5),
        },
    ];

    let baseline_error_sq: f32 = samples.iter().map(|s| s.avg_error.powi(2)).sum();

    // Compute optimal weight (weighted average)
    let error_activation_sum: f32 = samples.iter().map(|s| s.avg_error * s.activation).sum();
    let activation_sq_sum: f32 = samples.iter().map(|s| s.activation.powi(2)).sum();
    let raw_outgoing_weight = error_activation_sum / activation_sq_sum;
    let outgoing_weight = raw_outgoing_weight.clamp(-MAX_OUTGOING_WEIGHT, MAX_OUTGOING_WEIGHT);

    let incoming_weight = 1.0;
    let bias = 0.0;

    eprintln!(
        "MIXED ERRORS TEST: error_act_sum={error_activation_sum:.4}, act_sq_sum={activation_sq_sum:.4}, raw_w={raw_outgoing_weight:.6}, clamped_w={outgoing_weight:.6}"
    );

    // Predict improvement
    let (predicted_improvement, improved_count, total_count) = compute_relu_improvement_and_count(
        &samples,
        incoming_weight,
        outgoing_weight,
        bias,
        baseline_error_sq,
        Some(test_hard_tanh),
    );

    // Manual simulation
    let mut manual_baseline_error_sq = 0.0f32;
    let mut manual_new_error_sq = 0.0f32;
    let mut manual_improved = 0u32;
    let mut manual_worsened = 0u32;

    for sample in &samples {
        let target_val = sample.target_value.unwrap();
        let target_act = sample.target_activation.unwrap();
        let desired_value = target_val + sample.avg_error;
        let expected_activation = test_hard_tanh(desired_value);

        let baseline_err = expected_activation - target_act;
        manual_baseline_error_sq += baseline_err.powi(2);

        let pre_act = incoming_weight * sample.activation + bias;
        let relu_output = pre_act.max(0.0);
        let contribution = outgoing_weight * relu_output;

        let new_target_value = target_val + contribution;
        let new_target_activation = test_hard_tanh(new_target_value);

        let new_err = expected_activation - new_target_activation;
        manual_new_error_sq += new_err.powi(2);

        if new_err.abs() < baseline_err.abs() - EPSILON {
            manual_improved += 1;
        } else if new_err.abs() > baseline_err.abs() + EPSILON {
            manual_worsened += 1;
        }
    }

    let manual_improvement = if manual_baseline_error_sq > EPSILON {
        (manual_baseline_error_sq - manual_new_error_sq) / manual_baseline_error_sq
    } else {
        0.0
    };

    eprintln!(
        "MIXED ERRORS RESULT: predicted={:.6} ({:.4}%), manual={:.6} ({:.4}%), diff={:.6}",
        predicted_improvement,
        predicted_improvement * 100.0,
        manual_improvement,
        manual_improvement * 100.0,
        (predicted_improvement - manual_improvement).abs()
    );
    eprintln!(
        "  func: improved={improved_count}/{total_count}, manual: improved={manual_improved}, worsened={manual_worsened}"
    );

    // Prediction and manual simulation should match
    let diff = (predicted_improvement - manual_improvement).abs();
    assert!(
        diff < 0.01,
        "Predicted ({predicted_improvement:.6}) and manual ({manual_improvement:.6}) improvement should match within 1%"
    );
}

/// KEY TEST: Simulate what TypeScript evaluation actually does.
/// This is the most realistic test - it matches the production evaluation flow.
#[test]
fn prediction_matches_simulated_typescript_evaluation() {
    // Create samples that match production data characteristics
    let samples: Vec<HelpfulSample> = (0..1000)
        .map(|i| {
            let variation = (i as f32 / 100.0).sin() * 0.1;
            let error_variation = (i as f32 / 50.0).cos() * 0.05;
            HelpfulSample {
                activation: 0.5 + variation,
                avg_error: 0.1 + error_variation,
                target_value: Some(0.4 + variation * 0.5),
                target_activation: Some(test_hard_tanh(0.4 + variation * 0.5)),
            }
        })
        .collect();

    let baseline_error_sq: f32 = samples.iter().map(|s| s.avg_error.powi(2)).sum();

    // Compute optimal weight
    let error_activation_sum: f32 = samples.iter().map(|s| s.avg_error * s.activation).sum();
    let activation_sq_sum: f32 = samples.iter().map(|s| s.activation.powi(2)).sum();
    let raw_outgoing_weight = error_activation_sum / activation_sq_sum;
    let outgoing_weight = raw_outgoing_weight.clamp(-MAX_OUTGOING_WEIGHT, MAX_OUTGOING_WEIGHT);

    let incoming_weight = 1.0;
    let bias = 0.0;

    // Predict improvement (what Rust returns)
    let (predicted_improvement, _improved_count, _total_count) = compute_relu_improvement_and_count(
        &samples,
        incoming_weight,
        outgoing_weight,
        bias,
        baseline_error_sq,
        Some(test_hard_tanh),
    );

    // Simulate TypeScript evaluation
    // TypeScript computes: actualErrorReduction = originalError - candidateError
    // Where error is typically MSE or similar across all training samples

    // Original creature MSE (before adding neuron)
    let original_mse: f32 = samples
        .iter()
        .map(|s| {
            let target_act = s.target_activation.unwrap();
            let desired_value = s.target_value.unwrap() + s.avg_error;
            let expected = test_hard_tanh(desired_value);
            (expected - target_act).powi(2)
        })
        .sum::<f32>()
        / samples.len() as f32;

    // Candidate creature MSE (after adding neuron)
    let candidate_mse: f32 = samples
        .iter()
        .map(|s| {
            let target_val = s.target_value.unwrap();
            let desired_value = target_val + s.avg_error;
            let expected = test_hard_tanh(desired_value);

            // New neuron contribution
            let pre_act = incoming_weight * s.activation + bias;
            let relu_output = pre_act.max(0.0);
            let contribution = outgoing_weight * relu_output;

            let new_target_value = target_val + contribution;
            let new_activation = test_hard_tanh(new_target_value);
            (expected - new_activation).powi(2)
        })
        .sum::<f32>()
        / samples.len() as f32;

    // TypeScript reports: actualErrorReduction = originalError - candidateError
    // If we interpret this as raw error change:
    let original_error = original_mse.sqrt(); // RMSE
    let candidate_error = candidate_mse.sqrt();
    let actual_error_reduction = original_error - candidate_error;

    // For comparison with our percentage, convert to ratio
    let actual_improvement_ratio = actual_error_reduction / original_error;

    // Also compute MSE-based ratio (should match our prediction more closely)
    let mse_improvement_ratio = (original_mse - candidate_mse) / original_mse;

    eprintln!(
        "TYPESCRIPT SIMULATION: predicted={:.6} ({:.4}%)",
        predicted_improvement,
        predicted_improvement * 100.0
    );
    eprintln!(
        "  original_mse={:.8}, candidate_mse={:.8}, mse_improvement={:.6} ({:.4}%)",
        original_mse,
        candidate_mse,
        mse_improvement_ratio,
        mse_improvement_ratio * 100.0
    );
    eprintln!(
        "  original_rmse={:.6}, candidate_rmse={:.6}, rmse_reduction={:.6} ({:.4}%)",
        original_error,
        candidate_error,
        actual_improvement_ratio,
        actual_improvement_ratio * 100.0
    );

    // Our prediction should match MSE-based improvement
    let diff = (predicted_improvement - mse_improvement_ratio).abs();
    assert!(
        diff < 0.01,
        "Predicted ({predicted_improvement:.6}) and MSE improvement ({mse_improvement_ratio:.6}) should match within 1%"
    );

    // Both should be positive (error should decrease)
    assert!(
        predicted_improvement > 0.0,
        "Predicted improvement should be positive"
    );
    assert!(
        mse_improvement_ratio > 0.0,
        "MSE improvement should be positive"
    );
}

/// Test that verifies the sign is correct when contribution SHOULD help.
/// If this test fails, it indicates a sign error in the formula.
#[test]
fn contribution_in_correct_direction_reduces_error() {
    // Simple scenario: positive error, positive activation, positive weight = positive contribution
    // Positive contribution ADDS to target value, reducing positive error
    let sample = HelpfulSample {
        activation: 1.0, // positive source activation
        avg_error: 0.2,  // positive VALUE error: need to ADD 0.2
        target_value: Some(0.3),
        target_activation: Some(0.3),
    };

    // Optimal weight formula: w = Σ(error×activation) / Σ(activation²) = 0.2/1 = 0.2
    // Contribution = w × ReLU(source) = 0.2 × 1 = 0.2
    // New target value = 0.3 + 0.2 = 0.5
    // Desired value = 0.3 + 0.2 = 0.5 (should match!)

    let outgoing_weight = 0.1; // Clamped from 0.2
    let incoming_weight = 1.0;
    let bias = 0.0;

    let pre_activation = incoming_weight * sample.activation + bias;
    let relu_output = pre_activation.max(0.0);
    let contribution = outgoing_weight * relu_output;

    // Verify contribution is in the right direction
    assert!(
        contribution > 0.0,
        "Contribution should be positive for positive error"
    );
    assert!(
        contribution.signum() == sample.avg_error.signum(),
        "Contribution sign ({}) should match error sign ({})",
        contribution.signum(),
        sample.avg_error.signum()
    );

    // Verify new error is smaller
    let target_value = sample.target_value.unwrap();
    let target_activation = sample.target_activation.unwrap();
    let desired_value = target_value + sample.avg_error;
    let expected = test_hard_tanh(desired_value);

    let baseline_error = expected - target_activation;
    let new_target_value = target_value + contribution;
    let new_activation = test_hard_tanh(new_target_value);
    let new_error = expected - new_activation;

    eprintln!(
        "DIRECTION TEST: baseline_err={:.4}, new_err={:.4}, reduction={:.4}",
        baseline_error.abs(),
        new_error.abs(),
        baseline_error.abs() - new_error.abs()
    );

    assert!(
        new_error.abs() < baseline_error.abs(),
        "New error ({:.4}) should be smaller than baseline ({:.4})",
        new_error.abs(),
        baseline_error.abs()
    );
}
