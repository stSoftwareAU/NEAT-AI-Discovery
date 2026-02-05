//! Tests for sample matching and filtering functionality.
//!
//! Tests cover:
//! - Sample matching filters non-finite values
//! - Sample matching retains legitimate zero samples
//! - Sample matching preserves target values and activations
//! - Sample matching enables target activation simulation

use super::common::*;

#[test]
fn sample_matching_filters_non_finite_values() {
    // No GPU needed - tests the CPU build_samples function used in production
    let huge = f32::MAX;
    let target_records = vec![
        DiscoverRecord::new(0, "target".to_string(), None, 0.0, vec![0.5, -0.25]),
        DiscoverRecord::new(1, "target".to_string(), None, 0.0, vec![huge, huge]),
    ];
    let from_records = vec![
        DiscoverRecord::new(0, "from".to_string(), None, f32::INFINITY, Vec::new()),
        DiscoverRecord::new(1, "from".to_string(), None, 1.0, Vec::new()),
    ];

    let samples = build_samples(&target_records, &from_records);
    assert!(
        samples.is_empty(),
        "Sample matching should exclude non-finite samples"
    );
}

#[test]
fn sample_matching_retains_legitimate_zero_samples() {
    // No GPU needed - tests the CPU build_samples function used in production
    let target_records = vec![DiscoverRecord::new(
        42,
        "target".to_string(),
        None,
        0.0,
        vec![0.0, 0.0],
    )];
    let from_records = vec![DiscoverRecord::new(
        42,
        "from".to_string(),
        None,
        0.0,
        Vec::new(),
    )];

    let samples = build_samples(&target_records, &from_records);
    assert_eq!(
        samples.len(),
        1,
        "Sample matching should include legitimate zero-valued samples"
    );

    let sample = samples[0];
    assert_eq!(
        sample.activation, 0.0,
        "Zero activation should be preserved"
    );
    assert_eq!(
        sample.avg_error, 0.0,
        "Zero average error should be preserved"
    );
}

/// Test that sample matching preserves target_value and target_activation.
/// This is critical for accurate improvement predictions with non-linear
/// activation functions (TANH, LOGISTIC, HARD_TANH, etc.).
#[test]
fn sample_matching_preserves_target_value_and_activation() {
    // No GPU needed - tests the production build_samples function
    let target_value = 0.8; // Pre-activation input sum
    let target_activation = 0.6; // Post-activation output (e.g., after TANH)
    let target_records = vec![DiscoverRecord::new(
        0,
        "target".to_string(),
        Some(target_value),
        target_activation,
        vec![0.1, -0.2],
    )];
    let from_records = vec![DiscoverRecord::new(
        0,
        "from".to_string(),
        None, // Source value not used
        0.5,  // Source activation
        Vec::new(),
    )];

    let samples = build_samples(&target_records, &from_records);
    assert_eq!(samples.len(), 1, "Should find one matching sample");
    assert_eq!(
        samples[0].target_value,
        Some(target_value),
        "Sample matching must preserve target_value for activation function simulation"
    );
    assert_eq!(
        samples[0].target_activation,
        Some(target_activation),
        "Sample matching must preserve target_activation for error calculation"
    );
    assert_eq!(
        samples[0].activation, 0.5,
        "Source activation should be preserved"
    );
}

/// Test that target_value enables proper activation function simulation.
/// When target_value is available, get_target_simulation_fn should return
/// the activation function, enabling saturation-aware improvement predictions.
#[test]
fn sample_matching_enables_target_activation_simulation() {
    // No GPU needed - tests the production build_samples function
    // Create samples near HARD_TANH saturation to test simulation accuracy
    let target_records = vec![
        DiscoverRecord::new(
            0,
            "target".to_string(),
            Some(0.95), // Near saturation
            0.95,       // HARD_TANH clips to 1.0 when input >= 1.0
            vec![0.1],  // Small positive error (output should be higher)
        ),
        DiscoverRecord::new(
            1,
            "target".to_string(),
            Some(-0.8),
            -0.8,
            vec![-0.15], // Small negative error (output should be lower)
        ),
    ];
    let from_records = vec![
        DiscoverRecord::new(0, "from".to_string(), None, 0.5, Vec::new()),
        DiscoverRecord::new(1, "from".to_string(), None, -0.3, Vec::new()),
    ];

    let samples = build_samples(&target_records, &from_records);

    assert_eq!(samples.len(), 2, "Should match both records");

    // Verify all samples have target data (required for simulation)
    for (i, sample) in samples.iter().enumerate() {
        assert!(
            sample.target_value.is_some(),
            "Sample {i} must have target_value for activation simulation"
        );
        assert!(
            sample.target_activation.is_some(),
            "Sample {i} must have target_activation for error calculation"
        );
    }

    // With target data available, get_target_simulation_fn should return Some
    // for activations that need simulation (HARD_TANH, TANH, ReLU, etc.)
    let simulation_fn = get_target_simulation_fn(&samples, Some("HARD_TANH"));
    assert!(
        simulation_fn.is_some(),
        "Should enable HARD_TANH simulation when samples have target data"
    );

    let simulation_fn = get_target_simulation_fn(&samples, Some("TANH"));
    assert!(
        simulation_fn.is_some(),
        "Should enable TANH simulation when samples have target data"
    );
}
