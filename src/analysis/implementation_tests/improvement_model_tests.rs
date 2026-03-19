//! Tests for improvement model calculations (linear vs `HARD_TANH`).
//!
//! Tests cover:
//! - Linear model accuracy when errors are aligned
//! - `HARD_TANH` saturation behaviour differences from linear model
//! - Fallback to linear when target data is missing
//! - Sample counting with different models

use super::common::*;

/// Extended sample for `HARD_TANH` testing - includes target neuron's pre-activation value
struct HardTanhSample {
    source_activation: f32,
    target_value: f32,      // Pre-activation input sum
    target_activation: f32, // Post-activation output (clamped to [-1, 1])
    target_error: f32,      // expected - actual
}

/// Apply `HARD_TANH` activation function
fn hard_tanh(x: f32) -> f32 {
    x.clamp(-1.0, 1.0)
}

/// TDD Test: When errors are aligned, linear model should be accurate.
#[test]
fn test_linear_model_accurate_when_errors_aligned() {
    // All positive errors, source always positive
    // This is the ideal case for ReLU - linear model should work perfectly
    let samples = vec![
        HelpfulSample {
            activation: 1.0,
            avg_error: 0.5,
            target_value: None,
            target_activation: None,
        },
        HelpfulSample {
            activation: 0.5,
            avg_error: 0.25,
            target_value: None,
            target_activation: None,
        },
        HelpfulSample {
            activation: 0.8,
            avg_error: 0.4,
            target_value: None,
            target_activation: None,
        },
    ];

    // Compute optimal weight: w = Σ(error×activation) / Σ(activation²)
    let error_act_sum: f32 = samples.iter().map(|s| s.avg_error * s.activation).sum();
    let act_sq_sum: f32 = samples.iter().map(|s| s.activation.powi(2)).sum();
    let outgoing_weight = error_act_sum / act_sq_sum;
    let incoming_weight = 1.0f32;

    let baseline_error_sq: f32 = samples.iter().map(|s| s.avg_error.powi(2)).sum();

    let predicted_improvement = compute_net_improvement_with_squash(
        &samples,
        incoming_weight,
        outgoing_weight,
        0.0, // bias=0 for this test
        baseline_error_sq,
        None, // Linear model
    );

    // Manually compute (bias=0)
    let mut new_error_sq = 0.0f32;
    for sample in &samples {
        let relu_out = (incoming_weight * sample.activation).max(0.0);
        let new_err = sample.avg_error - outgoing_weight * relu_out;
        new_error_sq += new_err.powi(2);
    }
    let actual_improvement = (baseline_error_sq - new_error_sq) / baseline_error_sq;

    eprintln!(
        "Aligned errors: weight={outgoing_weight:.4}, predicted={:.4}%, actual={:.4}%",
        predicted_improvement * 100.0,
        actual_improvement * 100.0
    );

    assert!(
        (predicted_improvement - actual_improvement).abs() < 0.0001,
        "Predicted {predicted_improvement:.4} must match actual {actual_improvement:.4}",
    );

    assert!(
        actual_improvement > 0.1,
        "With aligned errors, should see significant improvement, got {:.4}%",
        actual_improvement * 100.0
    );
}

/// TEST: Demonstrates that linear model is WRONG for `HARD_TANH` targets near saturation.
/// The linear model predicts disaster (-125%) but `HARD_TANH` actually gives perfect result!
#[test]
fn test_hard_tanh_linear_model_is_wrong_near_saturation() {
    let samples = vec![HardTanhSample {
        source_activation: 0.5,
        target_value: 0.9,      // Near saturation
        target_activation: 0.9, // hard_tanh(0.9) = 0.9
        target_error: 0.1,      // expected (1.0) - actual (0.9)
    }];

    let incoming_weight = 1.0;
    let outgoing_weight = 0.5;

    // Compute baseline error
    let baseline_error_sq: f32 = samples.iter().map(|s| s.target_error.powi(2)).sum();

    // LINEAR MODEL prediction (current behaviour)
    let mut linear_new_error_sq = 0.0f32;
    for sample in &samples {
        let relu_output = (incoming_weight * sample.source_activation).max(0.0);
        let contribution = outgoing_weight * relu_output;
        let linear_new_error = sample.target_error - contribution;
        linear_new_error_sq += linear_new_error.powi(2);
    }
    let linear_improvement = (baseline_error_sq - linear_new_error_sq) / baseline_error_sq;

    // ACTUAL HARD_TANH behaviour
    let mut hard_tanh_new_error_sq = 0.0f32;
    for sample in &samples {
        let relu_output = (incoming_weight * sample.source_activation).max(0.0);
        let contribution = outgoing_weight * relu_output;
        let new_input = sample.target_value + contribution;
        let new_output = hard_tanh(new_input);
        let expected = sample.target_activation + sample.target_error;
        let new_error = expected - new_output;
        hard_tanh_new_error_sq += new_error.powi(2);
    }
    let hard_tanh_improvement = (baseline_error_sq - hard_tanh_new_error_sq) / baseline_error_sq;

    eprintln!("Baseline error²: {baseline_error_sq:.4}");
    eprintln!(
        "Linear model: new_error²={linear_new_error_sq:.4}, improvement={:.1}%",
        linear_improvement * 100.0
    );
    eprintln!(
        "HARD_TANH actual: new_error²={hard_tanh_new_error_sq:.4}, improvement={:.1}%",
        hard_tanh_improvement * 100.0
    );

    // The linear model predicts NEGATIVE improvement (making things worse)
    assert!(
        linear_improvement < 0.0,
        "Linear model should predict negative improvement near saturation, got {:.1}%",
        linear_improvement * 100.0
    );

    // But the actual HARD_TANH behaviour shows PERFECT improvement!
    assert!(
        hard_tanh_improvement > 0.99,
        "HARD_TANH should show ~100% improvement (error goes to 0), got {:.1}%",
        hard_tanh_improvement * 100.0
    );
}

/// TEST: Linear model predicts improvement but `HARD_TANH` shows NO improvement (already saturated)
#[test]
fn test_hard_tanh_linear_model_wrong_when_already_saturated() {
    let samples = vec![HardTanhSample {
        source_activation: 0.5,
        target_value: 1.5,      // Already beyond saturation
        target_activation: 1.0, // Clamped at max
        target_error: -0.2,     // expected (0.8) - actual (1.0)
    }];

    let incoming_weight = 1.0;
    let outgoing_weight = -0.3; // Trying to push output down

    let baseline_error_sq: f32 = samples.iter().map(|s| s.target_error.powi(2)).sum();

    // LINEAR MODEL
    let mut linear_new_error_sq = 0.0f32;
    for sample in &samples {
        let relu_output = (incoming_weight * sample.source_activation).max(0.0);
        let contribution = outgoing_weight * relu_output;
        let linear_new_error = sample.target_error - contribution;
        linear_new_error_sq += linear_new_error.powi(2);
    }
    let linear_improvement = (baseline_error_sq - linear_new_error_sq) / baseline_error_sq;

    // ACTUAL HARD_TANH
    let mut hard_tanh_new_error_sq = 0.0f32;
    for sample in &samples {
        let relu_output = (incoming_weight * sample.source_activation).max(0.0);
        let contribution = outgoing_weight * relu_output;
        let new_input = sample.target_value + contribution;
        let new_output = hard_tanh(new_input);
        let expected = sample.target_activation + sample.target_error;
        let new_error = expected - new_output;
        hard_tanh_new_error_sq += new_error.powi(2);
    }
    let hard_tanh_improvement = (baseline_error_sq - hard_tanh_new_error_sq) / baseline_error_sq;

    // Linear model predicts big improvement
    assert!(
        linear_improvement > 0.9,
        "Linear model should predict ~93% improvement, got {:.1}%",
        linear_improvement * 100.0
    );

    // But HARD_TANH shows NO improvement (still saturated)
    assert!(
        hard_tanh_improvement.abs() < 0.01,
        "HARD_TANH should show ~0% improvement (still saturated), got {:.1}%",
        hard_tanh_improvement * 100.0
    );
}

/// Test that `compute_net_improvement_with_squash` uses `HARD_TANH` model when specified.
#[test]
fn test_compute_net_improvement_uses_hard_tanh_model() {
    let samples = vec![HelpfulSample {
        activation: 0.5,
        avg_error: 0.1,
        target_value: Some(0.9), // Near saturation
        target_activation: Some(0.9),
    }];

    let incoming_weight = 1.0;
    let outgoing_weight = 0.5;
    let baseline_error_sq: f32 = samples.iter().map(|s| s.avg_error.powi(2)).sum();

    // Test with LINEAR model
    let linear_improvement = compute_net_improvement_with_squash(
        &samples,
        incoming_weight,
        outgoing_weight,
        0.0,
        baseline_error_sq,
        None,
    );

    // Test with HARD_TANH model
    let hard_tanh_improvement = compute_net_improvement_with_squash(
        &samples,
        incoming_weight,
        outgoing_weight,
        0.0,
        baseline_error_sq,
        Some("HARD_TANH"),
    );

    assert!(
        linear_improvement < 0.0,
        "Linear model should predict negative improvement, got {:.1}%",
        linear_improvement * 100.0
    );

    assert!(
        hard_tanh_improvement > 0.99,
        "HARD_TANH should show ~100% improvement, got {:.1}%",
        hard_tanh_improvement * 100.0
    );
}

/// Test that `HARD_TANH` model falls back to linear when target data is missing.
#[test]
fn test_compute_net_improvement_falls_back_to_linear_without_target_data() {
    let samples = vec![HelpfulSample {
        activation: 0.5,
        avg_error: 0.1,
        target_value: None,
        target_activation: None,
    }];

    let incoming_weight = 1.0;
    let outgoing_weight = 0.5;
    let baseline_error_sq: f32 = samples.iter().map(|s| s.avg_error.powi(2)).sum();

    let improvement = compute_net_improvement_with_squash(
        &samples,
        incoming_weight,
        outgoing_weight,
        0.0,
        baseline_error_sq,
        Some("HARD_TANH"),
    );

    let expected_linear = -1.25;
    assert!(
        (improvement - expected_linear).abs() < 0.01,
        "Should fall back to linear model without target data, got {:.1}% (expected {:.1}%)",
        improvement * 100.0,
        expected_linear * 100.0
    );
}

/// TEST: `count_improved_samples` must use `HARD_TANH` model for accurate sample counts.
#[test]
fn test_count_improved_samples_uses_hard_tanh_model() {
    let samples = vec![HelpfulSample {
        activation: 0.5,
        avg_error: 0.1,
        target_value: Some(0.9),
        target_activation: Some(0.9),
    }];

    let incoming_weight = 1.0;
    let outgoing_weight = 0.5;

    let (improved_count, total_count) = count_improved_samples(
        &samples,
        incoming_weight,
        outgoing_weight,
        0.0,
        Some("HARD_TANH"),
    );

    assert_eq!(total_count, 1, "Should have 1 total sample");
    assert_eq!(
        improved_count, 1,
        "HARD_TANH model should show sample is improved (saturates at 1.0), got {improved_count} improved",
    );
}

/// TEST: `count_improved_samples` falls back to linear model when target data is missing.
#[test]
fn test_count_improved_samples_falls_back_to_linear_without_target_data() {
    let samples = vec![HelpfulSample {
        activation: 0.5,
        avg_error: 0.1,
        target_value: None,
        target_activation: None,
    }];

    let incoming_weight = 1.0;
    let outgoing_weight = 0.5;

    let (improved_count, total_count) = count_improved_samples(
        &samples,
        incoming_weight,
        outgoing_weight,
        0.0,
        Some("HARD_TANH"),
    );

    assert_eq!(total_count, 1, "Should have 1 total sample");
    // Linear model: new_error = 0.1 - 0.25 = -0.15, |new_error| > |old_error|
    // So the sample is NOT improved
    assert_eq!(
        improved_count, 0,
        "Linear model (fallback) should show sample is NOT improved (overshoot)"
    );
}

/// TEST: `count_improved_samples` uses linear model for non-HARD_TANH activations.
#[test]
fn test_count_improved_samples_uses_linear_for_other_activations() {
    let samples = vec![HelpfulSample {
        activation: 0.5,
        avg_error: 0.3,
        target_value: Some(0.5),
        target_activation: Some(0.5),
    }];

    let incoming_weight = 1.0;
    let outgoing_weight = 0.5; // contribution = 0.25
    // Linear: new_error = 0.3 - 0.25 = 0.05, |new_error| < |old_error| = 0.3, IMPROVED

    // With TANH (not HARD_TANH), should use linear model (bias=0)
    let (improved_count, _) = count_improved_samples(
        &samples,
        incoming_weight,
        outgoing_weight,
        0.0,
        Some("TANH"),
    );

    assert_eq!(
        improved_count, 1,
        "Linear model should show sample is improved for TANH"
    );

    // With None squash, should also use linear model (bias=0)
    let (improved_count_none, _) =
        count_improved_samples(&samples, incoming_weight, outgoing_weight, 0.0, None);

    assert_eq!(
        improved_count_none, 1,
        "Linear model should show sample is improved for None squash"
    );
}
