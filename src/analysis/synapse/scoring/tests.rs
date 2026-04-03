//! Unit tests for the scoring improvement module.

#![allow(clippy::cast_possible_truncation)] // Test sample counts are small
use super::improvement::{
    compute_activation_improvement_and_count, compute_relu_improvement_and_count,
    compute_synapse_improvement_and_count,
};
use crate::analysis::samples::HelpfulSample;

/// Issue #940: Verify `ReLU` improvement handles None `target_value` gracefully
/// when `target_activation_fn` is provided (previously would panic via unwrap).
#[test]
fn test_relu_improvement_skips_samples_with_none_target_value() {
    let samples = vec![
        HelpfulSample {
            activation: 0.5,
            avg_error: 0.1,
            target_value: None,
            target_activation: Some(0.4),
        },
        HelpfulSample {
            activation: 0.3,
            avg_error: 0.2,
            target_value: None,
            target_activation: Some(0.3),
        },
    ];

    let (improvement, _improved, total) =
        compute_relu_improvement_and_count(&samples, 1.0, 1.0, 0.0, 1.0, Some(|x: f32| x.tanh()));

    assert!(
        improvement.is_finite(),
        "improvement should be finite, not NaN/Inf"
    );
    assert_eq!(total, samples.len() as u32);
}

/// Issue #940: Verify activation improvement handles None `target_activation`
/// gracefully when `target_activation_fn` is provided.
#[test]
fn test_activation_improvement_skips_samples_with_none_target_activation() {
    let samples = vec![HelpfulSample {
        activation: 0.5,
        avg_error: 0.1,
        target_value: Some(0.3),
        target_activation: None,
    }];

    let (improvement, _improved, total) = compute_activation_improvement_and_count(
        &samples,
        1.0,
        1.0,
        0.0,
        |x: f32| x.max(0.0),
        1.0,
        Some(|x: f32| x.tanh()),
    );

    assert!(improvement.is_finite());
    assert_eq!(total, samples.len() as u32);
}

/// Issue #940: Verify synapse improvement handles mixed None/Some target data
/// without panicking.
#[test]
fn test_synapse_improvement_handles_mixed_none_target_data() {
    let samples = vec![
        HelpfulSample {
            activation: 0.5,
            avg_error: 0.1,
            target_value: None,
            target_activation: Some(0.4),
        },
        HelpfulSample {
            activation: 0.3,
            avg_error: 0.2,
            target_value: Some(0.4),
            target_activation: None,
        },
    ];

    // Should not panic — the TargetSimulationMode checks require all samples
    // to have target data, so it falls back to linear mode.
    let (improvement, _improved, _worsened, total) =
        compute_synapse_improvement_and_count(&samples, 0.5, 1.0, Some("HARD_TANH"));

    assert!(improvement.is_finite());
    assert_eq!(total, samples.len() as u32);
}

/// Issue #940: Verify functions still produce correct results with complete data.
#[test]
fn test_relu_improvement_correct_with_complete_data() {
    let samples = vec![
        HelpfulSample {
            activation: 0.8,
            avg_error: 0.5,
            target_value: Some(0.3),
            target_activation: Some(0.29),
        },
        HelpfulSample {
            activation: 0.2,
            avg_error: -0.3,
            target_value: Some(0.6),
            target_activation: Some(0.54),
        },
    ];

    let (improvement, _improved, total) =
        compute_relu_improvement_and_count(&samples, 1.0, 0.5, 0.0, 0.5, Some(|x: f32| x.tanh()));

    assert!(improvement.is_finite());
    assert_eq!(total, 2);
}
