//! Tests for Issue #483: NaN-safe floating-point sorting across analysis modules.
//!
//! Verifies that analysis functions handle NaN values gracefully — no panics,
//! deterministic ordering, and correct results for the finite values.
//!
//! These tests exercise real library functions with NaN-containing data to ensure
//! the `total_cmp` migration is complete and correct across all modules.

mod common;

use neat_ai_discovery::analysis::constants::{cmp_f32_asc, cmp_f32_desc, cmp_f64_desc};
use neat_ai_discovery::analysis::error_distribution::ErrorDistribution;
use neat_ai_discovery::analysis::sample_weighted::{
    SampleWeightedConfig, compute_sample_weights, detect_high_error_neurons, stratify_samples,
};
use neat_ai_discovery::types::DiscoverRecord;

// =============================================================================
// Helper: create records with NaN errors
// =============================================================================

/// Build a `DiscoverRecord` with a specified error vector.
fn record_with_errors(uuid: &str, obs: u32, activation: f32, errors: Vec<f32>) -> DiscoverRecord {
    DiscoverRecord {
        obs_index: obs,
        neuron_uuid: uuid.to_string(),
        value: Some(activation),
        activation,
        errors,
    }
}

// =============================================================================
// f64 NaN-safe helper tests
// =============================================================================

/// Verify `cmp_f64_desc` handles NaN without panic — same guarantee as f32 helpers.
#[test]
fn test_cmp_f64_desc_nan_deterministic() {
    let mut values = [1.0_f64, f64::NAN, 3.0, f64::NAN, 2.0];
    values.sort_by(cmp_f64_desc);

    // NaN sorts as greater than all finite values in total_cmp, so in descending
    // order NaN comes first, then real values descending.
    assert!(values[0].is_nan());
    assert!(values[1].is_nan());
    assert_eq!(values[2], 3.0);
    assert_eq!(values[3], 2.0);
    assert_eq!(values[4], 1.0);
}

/// Verify `cmp_f64_desc` handles infinity correctly.
#[test]
fn test_cmp_f64_desc_infinity() {
    let mut values = [f64::INFINITY, 1.0, f64::NEG_INFINITY, 0.0];
    values.sort_by(cmp_f64_desc);
    assert_eq!(values[0], f64::INFINITY);
    assert_eq!(values[1], 1.0);
    assert_eq!(values[2], 0.0);
    assert_eq!(values[3], f64::NEG_INFINITY);
}

// =============================================================================
// ErrorDistribution with NaN-containing data
// =============================================================================

/// `ErrorDistribution::from_errors` must not panic when given a mix of
/// finite and NaN values. The internal percentile sort uses `total_cmp`.
#[test]
fn test_error_distribution_from_errors_with_nan() {
    let errors = vec![0.1, 0.5, f32::NAN, 0.3, 0.7, f32::NAN, 0.2];
    // NaN propagates through arithmetic, but the function should not panic.
    let result = ErrorDistribution::from_errors(&errors);
    // Should return Some (has data), not panic
    assert!(result.is_some());
}

/// `ErrorDistribution::from_errors` with all-NaN input should still produce
/// a result without panicking.
#[test]
fn test_error_distribution_all_nan_no_panic() {
    let errors = vec![f32::NAN, f32::NAN, f32::NAN];
    // Should not panic — NaN arithmetic produces NaN but that's fine
    let _result = ErrorDistribution::from_errors(&errors);
    // Just verifying no panic
}

/// `ErrorDistribution::from_errors` with purely finite values must produce
/// correct statistics (sanity check that total_cmp doesn't change results).
#[test]
fn test_error_distribution_finite_values_correct() {
    let errors = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let dist = ErrorDistribution::from_errors(&errors).expect("should produce distribution");

    // Mean of [1,2,3,4,5] = 3.0
    assert!((dist.mean - 3.0).abs() < 0.001);
    assert_eq!(dist.sample_count, 5);
    assert!((dist.min - 1.0).abs() < 0.001);
    assert!((dist.max - 5.0).abs() < 0.001);
    // Median (p50) should be 3.0
    assert!((dist.percentiles[2] - 3.0).abs() < 0.001);
}

// =============================================================================
// Sample weighting with NaN errors
// =============================================================================

/// `compute_sample_weights` must not panic when records contain NaN errors.
/// NaN errors should be treated as zero weight (non-finite filtering).
#[test]
fn test_compute_sample_weights_nan_errors_no_panic() {
    let records = vec![
        record_with_errors("n1", 0, 0.5, vec![0.1]),
        record_with_errors("n1", 1, 0.3, vec![f32::NAN]),
        record_with_errors("n1", 2, 0.7, vec![0.5]),
    ];

    let weights = compute_sample_weights(&records);

    // Should return 3 weights without panicking
    assert_eq!(weights.len(), 3);

    // NaN record gets zero weight; others get proportional weights
    // Record 0: error=0.1, Record 1: NaN→0.0, Record 2: error=0.5
    // Total = 0.6, weights = [0.1/0.6, 0.0/0.6, 0.5/0.6]
    assert!(
        weights[1] < 0.001,
        "NaN error record should have near-zero weight"
    );
    assert!(
        weights[0] > 0.0,
        "Finite error record should have positive weight"
    );
    assert!(
        weights[2] > weights[0],
        "Higher error should get higher weight"
    );
}

/// `stratify_samples` must not panic when records contain NaN errors.
#[test]
fn test_stratify_samples_nan_errors_no_panic() {
    let records = vec![
        record_with_errors("n1", 0, 0.5, vec![0.1]),
        record_with_errors("n1", 1, 0.3, vec![f32::NAN]),
        record_with_errors("n1", 2, 0.7, vec![0.5]),
        record_with_errors("n1", 3, 0.9, vec![0.8]),
        record_with_errors("n1", 4, 0.2, vec![0.05]),
    ];

    let result = stratify_samples(&records);

    // Should complete without panic
    assert!(
        !result.easy_samples.is_empty() || !result.hard_samples.is_empty(),
        "Should stratify into at least one non-empty group"
    );
}

/// `detect_high_error_neurons` must not panic with NaN in the error vectors.
#[test]
fn test_detect_high_error_neurons_nan_no_panic() {
    // Create enough records to meet the minimum sample threshold
    let mut records = Vec::new();
    for i in 0..15 {
        let error = if i % 3 == 0 {
            f32::NAN
        } else {
            0.5 + i as f32 * 0.1
        };
        records.push(record_with_errors("n1", i, 0.5, vec![error]));
    }

    let neuron_records = vec![("n1".to_string(), records)];
    let config = SampleWeightedConfig {
        min_weighted_error: 0.1,
        min_samples: 5,
    };

    let candidates = detect_high_error_neurons(&neuron_records, &config);

    // Should not panic — may or may not find candidates depending on thresholds
    // Just verify it completes
    let _ = candidates;
}

// =============================================================================
// Sort stability: NaN values must not corrupt ordering of finite values
// =============================================================================

/// When sorting candidate-like f32 values that include NaN, the relative order
/// of finite values must be preserved correctly.
#[test]
fn test_nan_does_not_corrupt_finite_ordering_asc() {
    let mut values = [3.0_f32, f32::NAN, 1.0, 5.0, f32::NAN, 2.0, 4.0];
    values.sort_by(cmp_f32_asc);

    // Extract finite values — they must be in ascending order
    let finite: Vec<f32> = values.iter().copied().filter(|v| v.is_finite()).collect();
    assert_eq!(finite, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
}

/// When sorting in descending order with NaN, finite values remain correctly ordered.
#[test]
fn test_nan_does_not_corrupt_finite_ordering_desc() {
    let mut values = [3.0_f32, f32::NAN, 1.0, 5.0, f32::NAN, 2.0, 4.0];
    values.sort_by(cmp_f32_desc);

    // Extract finite values — they must be in descending order
    let finite: Vec<f32> = values.iter().copied().filter(|v| v.is_finite()).collect();
    assert_eq!(finite, vec![5.0, 4.0, 3.0, 2.0, 1.0]);
}

/// Verify that subnormal (denormalised) float values sort correctly.
/// These are edge-case values near zero that could cause issues with naive comparisons.
#[test]
fn test_subnormal_values_sort_correctly() {
    let subnormal = f32::MIN_POSITIVE / 2.0; // A subnormal value
    let mut values = [1.0_f32, subnormal, 0.0, -subnormal, -1.0];
    values.sort_by(cmp_f32_asc);

    assert_eq!(values[0], -1.0);
    assert!(values[1] < 0.0); // -subnormal
    assert_eq!(values[2], 0.0); // could be -0.0 or 0.0
    assert!(values[3] > 0.0); // subnormal
    assert_eq!(values[4], 1.0);
}

/// Verify mixed special values (NaN, infinity, subnormal) all sort without panic.
#[test]
fn test_mixed_special_values_no_panic() {
    let mut values = vec![
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0,
        -0.0,
        f32::MIN_POSITIVE,
        f32::MIN_POSITIVE / 2.0, // subnormal
        f32::MAX,
        f32::MIN,
        1.0,
    ];
    values.sort_by(cmp_f32_asc);

    // Verify no panic occurred and finite values are in order
    let finite: Vec<f32> = values.iter().copied().filter(|v| v.is_finite()).collect();
    for w in finite.windows(2) {
        assert!(
            w[0] <= w[1],
            "Finite values must be in ascending order: {} > {}",
            w[0],
            w[1]
        );
    }
}
