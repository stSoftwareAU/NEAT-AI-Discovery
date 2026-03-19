//! Issue #527: Integration tests for confidence metrics
//!
//! Tests the confidence interval calculations in `src/analysis/confidence.rs`:
//! - `compute_confidence_metrics` with known statistical distributions
//! - High-variance vs low-variance candidates produce different interval widths
//! - Edge cases: empty samples, single sample, identical samples, extreme outliers
//!
//! These tests exercise real public functions with test data and verify
//! that confidence metrics behave correctly for candidate reliability assessment.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use neat_ai_discovery::analysis::samples::HelpfulSample;
use neat_ai_discovery::analysis::scoring::confidence::compute_confidence_metrics;

// =============================================================================
// Helper
// =============================================================================

fn make_sample(activation: f32, avg_error: f32) -> HelpfulSample {
    HelpfulSample {
        activation,
        avg_error,
        target_value: None,
        target_activation: None,
    }
}

// =============================================================================
// Empty / single-sample edge cases
// =============================================================================

#[test]
fn empty_samples_return_zero_confidence() {
    let metrics = compute_confidence_metrics(&[], 0.1, None);
    assert_eq!(
        metrics.prediction_confidence, 0.0,
        "Empty samples must produce zero confidence"
    );
    assert_eq!(
        metrics.expected_score_gain_confidence_interval,
        [0.0, 0.0],
        "Empty samples must produce a zero-width interval"
    );
}

#[test]
fn single_sample_produces_low_confidence() {
    let samples = vec![make_sample(0.5, 0.1)];
    let metrics = compute_confidence_metrics(&samples, 0.05, None);

    assert!(
        metrics.prediction_confidence < 0.2,
        "Single sample should yield low confidence, got {}",
        metrics.prediction_confidence
    );
}

// =============================================================================
// Identical samples (zero variance)
// =============================================================================

#[test]
fn identical_samples_produce_low_variance_confidence() {
    // All samples have exactly the same activation — zero source variance.
    // Confidence should be low because there is no correlation signal.
    let samples: Vec<HelpfulSample> = (0..50).map(|_| make_sample(0.5, 0.1)).collect();
    let metrics = compute_confidence_metrics(&samples, 0.05, None);

    assert!(
        metrics.prediction_confidence < 0.5,
        "Constant activations should produce low confidence, got {}",
        metrics.prediction_confidence
    );
}

// =============================================================================
// High-variance candidates get wider confidence intervals
// =============================================================================

#[test]
fn high_error_variance_produces_wider_interval_than_low_error_variance() {
    // Both sets have the same sample count and activation variance,
    // but different error variance.
    let n = 100;
    let expected_gain = 0.1;

    // Low error variance: all errors near 0.1
    let low_var_samples: Vec<HelpfulSample> = (0..n)
        .map(|i| {
            let activation = if i % 2 == 0 { 0.2 } else { 0.8 };
            make_sample(activation, 0.10)
        })
        .collect();

    // High error variance: errors spread between 0.0 and 0.5
    let high_var_samples: Vec<HelpfulSample> = (0..n)
        .map(|i| {
            let activation = if i % 2 == 0 { 0.2 } else { 0.8 };
            let error = (i as f32 / n as f32) * 0.5;
            make_sample(activation, error)
        })
        .collect();

    let low_var_metrics = compute_confidence_metrics(&low_var_samples, expected_gain, None);
    let high_var_metrics = compute_confidence_metrics(&high_var_samples, expected_gain, None);

    let low_var_width = low_var_metrics.expected_score_gain_confidence_interval[1]
        - low_var_metrics.expected_score_gain_confidence_interval[0];
    let high_var_width = high_var_metrics.expected_score_gain_confidence_interval[1]
        - high_var_metrics.expected_score_gain_confidence_interval[0];

    assert!(
        high_var_width >= low_var_width,
        "Higher error variance should produce wider (or equal) CI: low_var={low_var_width}, high_var={high_var_width}"
    );
}

// =============================================================================
// More samples produce narrower intervals
// =============================================================================

#[test]
fn more_samples_produce_narrower_confidence_interval() {
    let expected_gain = 0.1;

    let make_samples = |count: usize| -> Vec<HelpfulSample> {
        (0..count)
            .map(|i| {
                let activation = if i % 2 == 0 { 0.3 } else { 0.7 };
                make_sample(activation, 0.1)
            })
            .collect()
    };

    let small = make_samples(20);
    let large = make_samples(200);

    let small_metrics = compute_confidence_metrics(&small, expected_gain, None);
    let large_metrics = compute_confidence_metrics(&large, expected_gain, None);

    let small_width = small_metrics.expected_score_gain_confidence_interval[1]
        - small_metrics.expected_score_gain_confidence_interval[0];
    let large_width = large_metrics.expected_score_gain_confidence_interval[1]
        - large_metrics.expected_score_gain_confidence_interval[0];

    assert!(
        large_width <= small_width,
        "More samples should give narrower CI: small={small_width}, large={large_width}"
    );
}

// =============================================================================
// Confidence interval contains the point estimate
// =============================================================================

#[test]
fn confidence_interval_contains_expected_score_gain() {
    let expected_gain = 0.15;
    let samples: Vec<HelpfulSample> = (0..100)
        .map(|i| {
            let activation = if i % 2 == 0 { 0.3 } else { 0.7 };
            make_sample(activation, 0.1)
        })
        .collect();

    let metrics = compute_confidence_metrics(&samples, expected_gain, None);
    let [lower, upper] = metrics.expected_score_gain_confidence_interval;

    assert!(
        lower <= expected_gain && expected_gain <= upper,
        "CI [{lower}, {upper}] should contain expected gain {expected_gain}"
    );
}

// =============================================================================
// Model R² affects confidence
// =============================================================================

#[test]
fn high_r_squared_produces_higher_confidence_than_low_r_squared() {
    let samples: Vec<HelpfulSample> = (0..100)
        .map(|i| {
            let activation = if i % 2 == 0 { 0.2 } else { 0.8 };
            make_sample(activation, 0.1)
        })
        .collect();
    let expected_gain = 0.1;

    let low_r2 = compute_confidence_metrics(&samples, expected_gain, Some(0.1));
    let high_r2 = compute_confidence_metrics(&samples, expected_gain, Some(0.9));

    assert!(
        high_r2.prediction_confidence > low_r2.prediction_confidence,
        "Higher R² should yield higher confidence: low_r2={}, high_r2={}",
        low_r2.prediction_confidence,
        high_r2.prediction_confidence
    );
}

// =============================================================================
// Extreme outliers
// =============================================================================

#[test]
fn extreme_outlier_activations_still_produce_valid_metrics() {
    // Mix of normal activations with extreme outliers
    let mut samples: Vec<HelpfulSample> = (0..98)
        .map(|i| {
            let activation = if i % 2 == 0 { 0.3 } else { 0.7 };
            make_sample(activation, 0.1)
        })
        .collect();
    // Add extreme outliers
    samples.push(make_sample(1000.0, 0.1));
    samples.push(make_sample(-1000.0, 0.1));

    let metrics = compute_confidence_metrics(&samples, 0.1, None);

    // Confidence should still be valid (0..=1)
    assert!(
        (0.0..=1.0).contains(&metrics.prediction_confidence),
        "Confidence must be in [0, 1], got {}",
        metrics.prediction_confidence
    );

    // Interval should still be finite
    let [lower, upper] = metrics.expected_score_gain_confidence_interval;
    assert!(lower.is_finite(), "Lower bound must be finite, got {lower}");
    assert!(upper.is_finite(), "Upper bound must be finite, got {upper}");
    assert!(lower <= upper, "Lower {lower} must be <= upper {upper}");
}

// =============================================================================
// Confidence bounds are clamped to [0.0, 1.0]
// =============================================================================

#[test]
fn prediction_confidence_always_clamped_zero_to_one() {
    // Test with various sample sizes to cover the confidence range
    for count in [2, 10, 50, 100, 500] {
        let samples: Vec<HelpfulSample> = (0..count)
            .map(|i| {
                let activation = if i % 2 == 0 { 0.2 } else { 0.8 };
                make_sample(activation, 0.1)
            })
            .collect();

        let metrics = compute_confidence_metrics(&samples, 0.1, Some(0.5));
        assert!(
            (0.0..=1.0).contains(&metrics.prediction_confidence),
            "Confidence must be in [0, 1] for count={count}, got {}",
            metrics.prediction_confidence
        );
    }
}

// =============================================================================
// Good data produces high confidence
// =============================================================================

#[test]
fn large_varied_dataset_with_good_fit_yields_high_confidence() {
    // 200 samples with high activation variance and good R²
    let samples: Vec<HelpfulSample> = (0..200)
        .map(|i| {
            let activation = if i % 2 == 0 { 0.0 } else { 1.0 };
            make_sample(activation, 0.1)
        })
        .collect();

    let metrics = compute_confidence_metrics(&samples, 0.1, Some(0.8));
    assert!(
        metrics.prediction_confidence > 0.7,
        "Large varied dataset with good R² should have high confidence, got {}",
        metrics.prediction_confidence
    );
}

// =============================================================================
// No R² supplied defaults to neutral (not penalised)
// =============================================================================

#[test]
fn none_r_squared_does_not_penalise_confidence() {
    let samples: Vec<HelpfulSample> = (0..100)
        .map(|i| {
            let activation = if i % 2 == 0 { 0.2 } else { 0.8 };
            make_sample(activation, 0.1)
        })
        .collect();
    let expected_gain = 0.1;

    let with_none = compute_confidence_metrics(&samples, expected_gain, None);
    let with_perfect = compute_confidence_metrics(&samples, expected_gain, Some(1.0));

    // None defaults to model_confidence = 1.0 (neutral), same as perfect R²
    assert!(
        (with_none.prediction_confidence - with_perfect.prediction_confidence).abs() < 1e-6,
        "None R² should behave like perfect R²: none={}, perfect={}",
        with_none.prediction_confidence,
        with_perfect.prediction_confidence
    );
}
