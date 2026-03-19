//! Tests for bias calculation functionality.
//!
//! Tests cover:
//! - Bias calculation for various activation functions (TANH, `ReLU`, IDENTITY, etc.)
//! - Bias boundary ranges
//! - Error reduction improvements with optimal bias
//! - Edge cases (empty samples, insufficient samples, non-finite values)
//! - Threshold activation identification

use super::common::*;

/// Test bias calculation for TANH activation function
#[test]
fn test_bias_calculation_tanh() {
    let samples = create_test_samples();
    let bias = calculate_optimal_bias(&samples, 1.0, -0.5, tanh_activation, "TANH", None, None);

    // Bias should be in expanded TANH range
    assert!(
        (-1.0..=1.0).contains(&bias),
        "Bias for TANH should be in range [-1.0, 1.0], got {bias}"
    );
}

/// Test bias calculation for `ReLU` activation function
#[test]
fn test_bias_calculation_relu() {
    let samples = create_test_samples();
    let relu_fn = |x: f32| x.max(0.0);
    let bias = calculate_optimal_bias(&samples, 1.0, 0.5, relu_fn, "ReLU", None, None);

    // ReLU can now use negative bias for threshold shifting (expanded range)
    assert!(bias >= -1.0, "Bias for ReLU should be >= -1.0, got {bias}");
    assert!(bias <= 1.0, "Bias for ReLU should be <= 1.0, got {bias}");
}

/// Test bias improves error reduction compared to zero bias
#[test]
fn test_bias_improves_error_reduction() {
    let samples = create_test_samples();
    let incoming = 1.5;
    let outgoing = -0.18;

    // Calculate error with zero bias
    let mut zero_bias_error_sq = 0.0;
    for sample in &samples {
        let pre_activation = incoming * sample.activation;
        let new_neuron_activation = identity_activation(pre_activation);
        let correction = outgoing * new_neuron_activation;
        let new_error = sample.avg_error - correction;
        zero_bias_error_sq += new_error * new_error;
    }

    // Calculate optimal bias
    let optimal_bias = calculate_optimal_bias(
        &samples,
        incoming,
        outgoing,
        identity_activation,
        "IDENTITY",
        None,
        None,
    );

    // Calculate error with optimal bias
    let mut optimal_bias_error_sq = 0.0;
    for sample in &samples {
        let pre_activation = incoming * sample.activation + optimal_bias;
        let new_neuron_activation = identity_activation(pre_activation);
        let correction = outgoing * new_neuron_activation;
        let new_error = sample.avg_error - correction;
        optimal_bias_error_sq += new_error * new_error;
    }

    // Optimal bias should give equal or better error reduction than zero bias
    assert!(
        optimal_bias_error_sq <= zero_bias_error_sq + EPSILON,
        "Optimal bias should improve or equal zero bias error reduction: zero_bias_error={zero_bias_error_sq}, optimal_bias_error={optimal_bias_error_sq}"
    );
}

/// Test bias range boundaries for different activation functions
#[test]
fn test_bias_within_reasonable_range() {
    let samples = create_test_samples();

    type ActivationTestCase = (&'static str, fn(f32) -> f32, f32, f32);
    let test_cases: Vec<ActivationTestCase> = vec![
        ("TANH", tanh_activation as fn(f32) -> f32, -10.0, 10.0),
        (
            "LOGISTIC",
            logistic_activation as fn(f32) -> f32,
            -10.0,
            10.0,
        ),
        (
            "IDENTITY",
            identity_activation as fn(f32) -> f32,
            -50.0,
            50.0,
        ),
    ];

    for (name, activation_fn, min_expected, max_expected) in test_cases {
        let bias = calculate_optimal_bias(&samples, 1.0, 1.0, activation_fn, name, None, None);
        assert!(
            bias >= min_expected && bias <= max_expected,
            "Bias for {name} should be in range [{min_expected}, {max_expected}], got {bias}"
        );
    }
}

/// Test bias calculation handles empty samples
#[test]
fn test_bias_calculation_empty_samples() {
    let samples: Vec<HelpfulSample> = vec![];
    let bias = calculate_optimal_bias(&samples, 1.0, 0.5, tanh_activation, "TANH", None, None);

    // Should return 0.0 for empty samples
    assert_eq!(bias, 0.0, "Empty samples should return bias of 0.0");
}

/// Test bias calculation handles insufficient samples
#[test]
fn test_bias_calculation_insufficient_samples() {
    // Only 5 samples (less than MIN_NEURON_SAMPLE_COUNT of 10)
    let samples = vec![
        HelpfulSample {
            activation: 0.5,
            avg_error: 0.2,
            target_value: None,
            target_activation: None,
        },
        HelpfulSample {
            activation: -0.3,
            avg_error: -0.15,
            target_value: None,
            target_activation: None,
        },
        HelpfulSample {
            activation: 0.8,
            avg_error: 0.25,
            target_value: None,
            target_activation: None,
        },
        HelpfulSample {
            activation: -0.6,
            avg_error: -0.1,
            target_value: None,
            target_activation: None,
        },
        HelpfulSample {
            activation: 0.4,
            avg_error: 0.18,
            target_value: None,
            target_activation: None,
        },
    ];

    let bias = calculate_optimal_bias(&samples, 1.0, 0.5, tanh_activation, "TANH", None, None);

    // Should still return a valid bias in range
    assert!(
        (-10.0..=10.0).contains(&bias),
        "Bias should be in reasonable range, got {bias}"
    );
}

/// Test bias calculation with non-finite values
#[test]
fn test_bias_calculation_with_non_finite_values() {
    let mut samples = create_test_samples();
    // Add some non-finite values
    samples[1].activation = f32::NAN;
    samples[2].avg_error = f32::INFINITY;

    let bias = calculate_optimal_bias(&samples, 1.0, 0.5, tanh_activation, "TANH", None, None);

    // Should handle non-finite values gracefully and return a finite bias
    assert!(
        bias.is_finite(),
        "Bias should be finite even with non-finite input values"
    );
    assert!(
        (-1.0..=1.0).contains(&bias),
        "Bias should be in reasonable range, got {bias}"
    );
}

/// Test `is_threshold_activation` identifies threshold functions (STEP/BIPOLAR).
/// These use a specialised threshold-crossing model instead of the linear model.
/// All other activations use the standard linear error model - none are skipped.
#[test]
fn test_is_threshold_activation() {
    // Threshold activations - use threshold-crossing model
    assert!(is_threshold_activation("STEP"), "STEP uses threshold model");
    assert!(is_threshold_activation("step"), "case insensitive");
    assert!(
        is_threshold_activation("BIPOLAR"),
        "BIPOLAR uses threshold model"
    );

    // All other activations use standard linear model (not skipped)
    assert!(
        !is_threshold_activation("IF"),
        "IF uses standard model (correlation still works)"
    );
    assert!(
        !is_threshold_activation("MAXIMUM"),
        "MAXIMUM uses standard model"
    );
    assert!(
        !is_threshold_activation("MINIMUM"),
        "MINIMUM uses standard model"
    );
    assert!(
        !is_threshold_activation("HARD_TANH"),
        "HARD_TANH uses standard model"
    );
    assert!(
        !is_threshold_activation("CLIPPED"),
        "CLIPPED uses standard model"
    );
    assert!(
        !is_threshold_activation("ReLU6"),
        "ReLU6 uses standard model"
    );
    assert!(!is_threshold_activation("TANH"), "TANH uses standard model");
    assert!(
        !is_threshold_activation("LOGISTIC"),
        "LOGISTIC uses standard model"
    );
    assert!(!is_threshold_activation("ReLU"), "ReLU uses standard model");
    assert!(
        !is_threshold_activation("LeakyReLU"),
        "LeakyReLU uses standard model"
    );
    assert!(!is_threshold_activation("ELU"), "ELU uses standard model");
    assert!(!is_threshold_activation("SELU"), "SELU uses standard model");
    assert!(!is_threshold_activation("GELU"), "GELU uses standard model");
    assert!(
        !is_threshold_activation("IDENTITY"),
        "IDENTITY uses standard model"
    );
    assert!(
        !is_threshold_activation("Softplus"),
        "Softplus uses standard model"
    );
    assert!(
        !is_threshold_activation("BENT_IDENTITY"),
        "BENT_IDENTITY uses standard model"
    );
    assert!(
        !is_threshold_activation("ArcTan"),
        "ArcTan uses standard model"
    );
    assert!(
        !is_threshold_activation("Swish"),
        "Swish uses standard model"
    );
    assert!(!is_threshold_activation("Mish"), "Mish uses standard model");
    assert!(
        !is_threshold_activation("UNKNOWN"),
        "Unknown uses standard model"
    );
}

// Helper function to create standard test samples
fn create_test_samples() -> Vec<HelpfulSample> {
    vec![
        HelpfulSample {
            activation: 0.5,
            avg_error: 0.2,
            target_value: None,
            target_activation: None,
        },
        HelpfulSample {
            activation: -0.3,
            avg_error: -0.15,
            target_value: None,
            target_activation: None,
        },
        HelpfulSample {
            activation: 0.8,
            avg_error: 0.25,
            target_value: None,
            target_activation: None,
        },
        HelpfulSample {
            activation: -0.6,
            avg_error: -0.1,
            target_value: None,
            target_activation: None,
        },
        HelpfulSample {
            activation: 0.4,
            avg_error: 0.18,
            target_value: None,
            target_activation: None,
        },
        HelpfulSample {
            activation: -0.2,
            avg_error: -0.08,
            target_value: None,
            target_activation: None,
        },
        HelpfulSample {
            activation: 0.7,
            avg_error: 0.22,
            target_value: None,
            target_activation: None,
        },
        HelpfulSample {
            activation: -0.5,
            avg_error: -0.12,
            target_value: None,
            target_activation: None,
        },
        HelpfulSample {
            activation: 0.6,
            avg_error: 0.19,
            target_value: None,
            target_activation: None,
        },
        HelpfulSample {
            activation: -0.4,
            avg_error: -0.09,
            target_value: None,
            target_activation: None,
        },
    ]
}
