//! Tests for shared statistical utility functions (Issue #750).
//!
//! Verifies that the shared `pearson_correlation`, `compute_mean`, and
//! `compute_variance` functions handle edge cases correctly and always
//! produce clamped, NaN-free results.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use neat_ai_discovery::analysis::detection::stats::{
    compute_mean, compute_variance, pearson_correlation,
};

// ── pearson_correlation ──────────────────────────────────────────────

#[test]
fn pearson_empty_inputs_return_zero() {
    assert_eq!(pearson_correlation(&[], &[]), 0.0);
}

#[test]
fn pearson_single_element_returns_zero() {
    assert_eq!(pearson_correlation(&[1.0], &[2.0]), 0.0);
}

#[test]
fn pearson_constant_sequences_return_zero_not_nan() {
    let result = pearson_correlation(&[5.0, 5.0, 5.0], &[3.0, 3.0, 3.0]);
    assert_eq!(result, 0.0);
    assert!(!result.is_nan());
}

#[test]
fn pearson_perfect_positive_correlation() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let r = pearson_correlation(&x, &y);
    assert!((r - 1.0).abs() < 1e-6, "Expected ~1.0, got {r}");
}

#[test]
fn pearson_perfect_negative_correlation() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![10.0, 8.0, 6.0, 4.0, 2.0];
    let r = pearson_correlation(&x, &y);
    assert!((r - (-1.0)).abs() < 1e-6, "Expected ~-1.0, got {r}");
}

#[test]
fn pearson_identical_sequences_return_exactly_one() {
    let x = vec![3.21, 2.71, 1.41, 0.577];
    let r = pearson_correlation(&x, &x);
    assert_eq!(
        r, 1.0,
        "Identical sequences must produce exactly 1.0, got {r}"
    );
}

#[test]
fn pearson_result_always_clamped() {
    // Even with values that might cause slight floating-point overshoot,
    // the result must be in [-1.0, 1.0].
    let x: Vec<f32> = (0..1000).map(|i| i as f32 * 0.001).collect();
    let y: Vec<f32> = x.iter().map(|v| v * 2.0 + 0.5).collect();
    let r = pearson_correlation(&x, &y);
    assert!(
        (-1.0..=1.0).contains(&r),
        "Result {r} is outside [-1.0, 1.0]"
    );
}

#[test]
fn pearson_mismatched_lengths_uses_shorter() {
    // When lengths differ, should use the shorter length
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.0, 6.0];
    let r = pearson_correlation(&x, &y);
    // Should still produce a valid result using first 3 elements
    assert!(
        (-1.0..=1.0).contains(&r),
        "Result {r} is outside [-1.0, 1.0]"
    );
}

// ── compute_mean ─────────────────────────────────────────────────────

#[test]
fn mean_empty_returns_zero() {
    assert_eq!(compute_mean(&[]), 0.0);
}

#[test]
fn mean_single_element() {
    assert_eq!(compute_mean(&[42.0]), 42.0);
}

#[test]
fn mean_multiple_elements() {
    let result = compute_mean(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    assert!((result - 3.0).abs() < 1e-6);
}

// ── compute_variance ─────────────────────────────────────────────────

#[test]
fn variance_empty_returns_zero() {
    assert_eq!(compute_variance(&[]), 0.0);
}

#[test]
fn variance_single_element_returns_zero() {
    assert_eq!(compute_variance(&[42.0]), 0.0);
}

#[test]
fn variance_constant_values_returns_zero() {
    assert_eq!(compute_variance(&[7.0, 7.0, 7.0, 7.0]), 0.0);
}

#[test]
fn variance_known_values() {
    // Values: [1, 2, 3, 4, 5], mean = 3
    // Variance = ((1-3)^2 + (2-3)^2 + (3-3)^2 + (4-3)^2 + (5-3)^2) / 5
    //          = (4 + 1 + 0 + 1 + 4) / 5 = 2.0
    let result = compute_variance(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    assert!((result - 2.0).abs() < 1e-6, "Expected 2.0, got {result}");
}
