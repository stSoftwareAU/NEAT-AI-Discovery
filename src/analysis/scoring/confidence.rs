//! Confidence interval calculations for discovery predictions (Issue #194).
//!
//! This module provides functions to compute prediction confidence and confidence intervals
//! for discovery candidates. The confidence metrics help callers:
//! - Prioritise high-confidence candidates
//! - Understand prediction uncertainty
//! - Filter out unreliable predictions
//!
//! ## Confidence Factors
//!
//! The overall confidence score is computed from:
//! 1. **Sample size** - More samples = higher confidence
//! 2. **Source variance** - Higher source variance = more reliable correlation
//! 3. **Model fit** - How well the linear model fits the data (R²)
//!
//! ## Confidence Interval
//!
//! The confidence interval provides bounds [lower, upper] around the point estimate.
//! The interval width is inversely proportional to:
//! - Sample count (more samples = narrower interval)
//! - Source variance (higher variance = narrower interval)

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use serde::Serialize;

use crate::analysis::samples::HelpfulSample;

/// Minimum sample count for full confidence credit.
/// Below this, sample confidence is linearly scaled.
const MIN_CONFIDENT_SAMPLES: f32 = 100.0;

/// Minimum source standard deviation for full confidence credit.
/// Sources with std dev below this are progressively discounted.
const MIN_CONFIDENT_STD_DEV: f32 = 0.05;

/// t-distribution critical values for 95% confidence interval (two-tailed).
///
/// Each entry is `(degrees_of_freedom, t_critical)`. For df values between
/// entries, linear interpolation is used. For df above the largest entry,
/// the normal approximation (1.960) applies.
const T_CRITICAL_95_TABLE: &[(u32, f32)] = &[
    (1, 12.706),
    (2, 4.303),
    (3, 3.182),
    (4, 2.776),
    (5, 2.571),
    (6, 2.447),
    (7, 2.365),
    (8, 2.306),
    (9, 2.262),
    (10, 2.228),
    (15, 2.131),
    (20, 2.086),
    (25, 2.060),
    (30, 2.042),
    (40, 2.021),
    (50, 2.009),
    (60, 2.000),
    (80, 1.990),
    (100, 1.984),
    (120, 1.980),
];

/// Z-score for 95% confidence interval — used for large df (normal limit).
const Z_SCORE_95: f32 = 1.960;

// =============================================================================
// Confidence Calculation Result
// =============================================================================

/// Confidence metrics for a prediction.
///
/// Contains both the overall confidence score and the confidence interval bounds.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PredictionConfidenceMetrics {
    /// Overall confidence score (0.0 to 1.0).
    ///
    /// Higher values indicate more reliable predictions.
    /// Computed as geometric mean of sample confidence, variance confidence, and model fit.
    pub prediction_confidence: f32,

    /// Confidence interval for the expected score gain: [`lower_bound`, `upper_bound`].
    ///
    /// The point estimate (expectedCreatureScoreGain) should fall within this interval.
    pub expected_score_gain_confidence_interval: [f32; 2],
}

impl Default for PredictionConfidenceMetrics {
    fn default() -> Self {
        Self {
            prediction_confidence: 0.0,
            expected_score_gain_confidence_interval: [0.0, 0.0],
        }
    }
}

// =============================================================================
// Confidence Calculation Functions
// =============================================================================

/// Compute confidence metrics for a synapse/neuron candidate.
///
/// # Arguments
/// * `samples` - The samples used for the prediction
/// * `expected_score_gain` - The point estimate of expected score gain
/// * `model_r_squared` - The R² goodness-of-fit metric (optional)
///
/// # Returns
/// Confidence metrics including overall score and confidence interval.
pub fn compute_confidence_metrics(
    samples: &[HelpfulSample],
    expected_score_gain: f32,
    model_r_squared: Option<f32>,
) -> PredictionConfidenceMetrics {
    if samples.is_empty() {
        return PredictionConfidenceMetrics::default();
    }

    // Compute individual confidence factors
    let sample_confidence = compute_sample_confidence(samples.len());
    let variance_confidence = compute_source_variance_confidence(samples);
    let model_confidence = model_r_squared.map_or(1.0, compute_model_fit_confidence);

    // Overall confidence: geometric mean of all factors
    // All factors matter, so we use geometric mean (any low factor drags down the result)
    let overall_confidence =
        (sample_confidence * variance_confidence * model_confidence).powf(1.0 / 3.0);

    // Compute confidence interval
    let confidence_interval = compute_confidence_interval(samples, expected_score_gain);

    PredictionConfidenceMetrics {
        prediction_confidence: overall_confidence.clamp(0.0, 1.0),
        expected_score_gain_confidence_interval: confidence_interval,
    }
}

/// Compute sample-size-based confidence factor.
///
/// More samples lead to higher confidence.
/// Returns a value between 0.0 and 1.0.
fn compute_sample_confidence(sample_count: usize) -> f32 {
    (sample_count as f32 / MIN_CONFIDENT_SAMPLES).min(1.0)
}

/// Compute source-variance-based confidence factor.
///
/// Higher source activation variance leads to more reliable correlation signals.
/// Returns a value between 0.0 and 1.0.
fn compute_source_variance_confidence(samples: &[HelpfulSample]) -> f32 {
    if samples.len() < 2 {
        return 0.0;
    }

    // Compute standard deviation of source activations
    let mut activation_sum = 0.0f64;
    let mut activation_sq_sum = 0.0f64;
    let mut count = 0u32;

    for sample in samples {
        if sample.activation.is_finite() {
            let a = sample.activation as f64;
            activation_sum += a;
            activation_sq_sum += a * a;
            count += 1;
        }
    }

    if count < 2 {
        return 0.0;
    }

    let n = count as f64;
    let mean = activation_sum / n;
    let variance = (activation_sq_sum / n) - (mean * mean);
    let std_dev = variance.max(0.0).sqrt() as f32;

    // Linear scaling: full confidence at MIN_CONFIDENT_STD_DEV, zero at 0
    (std_dev / MIN_CONFIDENT_STD_DEV).clamp(0.0, 1.0)
}

/// Compute model-fit-based confidence factor from R².
///
/// Higher R² indicates better linear model fit.
/// Returns a value between 0.0 and 1.0.
fn compute_model_fit_confidence(r_squared: f32) -> f32 {
    r_squared.clamp(0.0, 1.0)
}

/// Look up the t-distribution critical value for a 95% two-tailed confidence
/// interval given `df` degrees of freedom.
///
/// Uses linear interpolation between table entries. For df ≥ 120, returns
/// the normal approximation (1.960).
fn t_critical_95(df: usize) -> f32 {
    if df == 0 {
        return Z_SCORE_95;
    }
    let df = df as u32;

    // Beyond table range, use normal approximation
    // T_CRITICAL_95_TABLE is a non-empty compile-time constant (Issue #940).
    let Some(&(last_df, last_t)) = T_CRITICAL_95_TABLE.last() else {
        return Z_SCORE_95;
    };
    if df >= last_df {
        return if df == last_df { last_t } else { Z_SCORE_95 };
    }

    // Find the two bracketing entries and interpolate
    for window in T_CRITICAL_95_TABLE.windows(2) {
        let (df_lo, t_lo) = window[0];
        let (df_hi, t_hi) = window[1];
        if df <= df_lo {
            return t_lo;
        }
        if df < df_hi {
            let frac = (df - df_lo) as f32 / (df_hi - df_lo) as f32;
            return t_lo + frac * (t_hi - t_lo);
        }
    }

    Z_SCORE_95
}

/// Compute 95% confidence interval for expected score gain.
///
/// Uses the t-distribution critical value for `df = n - 1` degrees of freedom,
/// which properly accounts for small-sample uncertainty without the
/// unprincipled `1 / confidence_factor` inflation that was used previously.
fn compute_confidence_interval(samples: &[HelpfulSample], expected_score_gain: f32) -> [f32; 2] {
    if samples.is_empty() {
        return [0.0, 0.0];
    }

    let n = samples.len() as f32;

    // Compute standard error of the prediction
    let error_variance = compute_error_variance(samples);
    let error_std_dev = error_variance.sqrt();

    // Standard error of the mean: σ / √n
    let standard_error = if n > 1.0 {
        error_std_dev / n.sqrt()
    } else {
        error_std_dev
    };

    // Use t-distribution critical value for df = n - 1
    let df = if samples.len() > 1 {
        samples.len() - 1
    } else {
        1
    };
    let t_crit = t_critical_95(df);

    let margin = (t_crit * standard_error).clamp(0.0, expected_score_gain.abs().max(0.5));

    let lower = expected_score_gain - margin;
    let upper = expected_score_gain + margin;

    [lower, upper]
}

/// Compute variance of errors from samples.
fn compute_error_variance(samples: &[HelpfulSample]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }

    let mut error_sum = 0.0f64;
    let mut error_sq_sum = 0.0f64;
    let mut count = 0u32;

    for sample in samples {
        if sample.avg_error.is_finite() {
            let e = sample.avg_error as f64;
            error_sum += e;
            error_sq_sum += e * e;
            count += 1;
        }
    }

    if count == 0 {
        return 0.0;
    }

    let n = count as f64;
    let mean = error_sum / n;
    let variance = (error_sq_sum / n) - (mean * mean);

    variance.max(0.0) as f32
}

// =============================================================================
// Unit Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sample(activation: f32, avg_error: f32) -> HelpfulSample {
        HelpfulSample {
            activation,
            avg_error,
            target_value: None,
            target_activation: None,
        }
    }

    #[test]
    fn test_sample_confidence_empty() {
        let confidence = compute_sample_confidence(0);
        assert_eq!(confidence, 0.0);
    }

    #[test]
    fn test_sample_confidence_low() {
        let confidence = compute_sample_confidence(10);
        assert!(
            (confidence - 0.1).abs() < 0.001,
            "Expected ~0.1, got {confidence}"
        );
    }

    #[test]
    fn test_sample_confidence_medium() {
        let confidence = compute_sample_confidence(50);
        assert!(
            (confidence - 0.5).abs() < 0.001,
            "Expected ~0.5, got {confidence}"
        );
    }

    #[test]
    fn test_sample_confidence_high() {
        let confidence = compute_sample_confidence(100);
        assert!(
            (confidence - 1.0).abs() < 0.001,
            "Expected 1.0, got {confidence}"
        );
    }

    #[test]
    fn test_sample_confidence_above_threshold() {
        let confidence = compute_sample_confidence(500);
        assert_eq!(confidence, 1.0, "Should cap at 1.0");
    }

    #[test]
    fn test_variance_confidence_empty() {
        let confidence = compute_source_variance_confidence(&[]);
        assert_eq!(confidence, 0.0);
    }

    #[test]
    fn test_variance_confidence_single() {
        let samples = vec![make_sample(0.5, 0.1)];
        let confidence = compute_source_variance_confidence(&samples);
        assert_eq!(
            confidence, 0.0,
            "Single sample should have no variance confidence"
        );
    }

    #[test]
    fn test_variance_confidence_constant() {
        // All same activation = zero variance
        let samples = vec![
            make_sample(0.5, 0.1),
            make_sample(0.5, 0.2),
            make_sample(0.5, 0.3),
        ];
        let confidence = compute_source_variance_confidence(&samples);
        assert!(
            confidence < 0.01,
            "Constant source should have ~0 confidence, got {confidence}"
        );
    }

    #[test]
    fn test_variance_confidence_high_variance() {
        // Large variance in activations
        let samples = vec![
            make_sample(0.0, 0.1),
            make_sample(1.0, 0.2),
            make_sample(0.0, 0.3),
            make_sample(1.0, 0.4),
        ];
        let confidence = compute_source_variance_confidence(&samples);
        assert!(
            confidence > 0.9,
            "High variance should have high confidence, got {confidence}"
        );
    }

    #[test]
    fn test_model_fit_confidence() {
        assert_eq!(compute_model_fit_confidence(0.0), 0.0);
        assert_eq!(compute_model_fit_confidence(0.5), 0.5);
        assert_eq!(compute_model_fit_confidence(1.0), 1.0);
        assert_eq!(compute_model_fit_confidence(-0.5), 0.0); // Clamped
        assert_eq!(compute_model_fit_confidence(1.5), 1.0); // Clamped
    }

    #[test]
    fn test_overall_confidence_empty() {
        let metrics = compute_confidence_metrics(&[], 0.1, None);
        assert_eq!(metrics.prediction_confidence, 0.0);
    }

    #[test]
    fn test_overall_confidence_single_sample() {
        let samples = vec![make_sample(0.5, 0.1)];
        let metrics = compute_confidence_metrics(&samples, 0.1, None);
        // Single sample = low sample confidence AND low variance confidence
        assert!(
            metrics.prediction_confidence < 0.2,
            "Single sample should have low confidence"
        );
    }

    #[test]
    fn test_overall_confidence_good_data() {
        // Many samples with high variance
        let mut samples = Vec::new();
        for i in 0..200 {
            let activation = if i % 2 == 0 { 0.0 } else { 1.0 };
            samples.push(make_sample(activation, 0.1));
        }
        let metrics = compute_confidence_metrics(&samples, 0.1, Some(0.8));
        assert!(
            metrics.prediction_confidence > 0.7,
            "Good data should have high confidence, got {}",
            metrics.prediction_confidence
        );
    }

    #[test]
    fn test_confidence_interval_contains_expected() {
        let mut samples = Vec::new();
        for i in 0..100 {
            let activation = if i % 2 == 0 { 0.3 } else { 0.7 };
            samples.push(make_sample(activation, 0.1));
        }
        let expected = 0.15;
        let metrics = compute_confidence_metrics(&samples, expected, None);

        let [lower, upper] = metrics.expected_score_gain_confidence_interval;
        assert!(
            lower <= expected,
            "Lower bound {lower} should be <= expected {expected}"
        );
        assert!(
            upper >= expected,
            "Upper bound {upper} should be >= expected {expected}"
        );
    }

    #[test]
    fn test_confidence_interval_wider_with_fewer_samples() {
        // Same variance, different sample counts
        let make_samples = |count: usize| -> Vec<HelpfulSample> {
            (0..count)
                .map(|i| make_sample(if i % 2 == 0 { 0.3 } else { 0.7 }, 0.1))
                .collect()
        };

        let samples_low = make_samples(20);
        let samples_high = make_samples(200);

        let metrics_low = compute_confidence_metrics(&samples_low, 0.1, None);
        let metrics_high = compute_confidence_metrics(&samples_high, 0.1, None);

        let width_low = metrics_low.expected_score_gain_confidence_interval[1]
            - metrics_low.expected_score_gain_confidence_interval[0];
        let width_high = metrics_high.expected_score_gain_confidence_interval[1]
            - metrics_high.expected_score_gain_confidence_interval[0];

        assert!(
            width_high <= width_low,
            "More samples should give narrower CI: low={width_low}, high={width_high}"
        );
    }

    #[test]
    fn test_t_distribution_interval_ratio_10_vs_100_samples() {
        // With t-distribution, 10 samples (df=9, t≈2.262) should produce an interval
        // roughly 2–5× wider than 100 samples (df=99, t≈1.984), accounting for both
        // the t-critical ratio (~1.14) and the √n ratio (√100/√10 ≈ 3.16).
        let make_samples = |count: usize| -> Vec<HelpfulSample> {
            (0..count)
                .map(|i| {
                    let error = 0.05 + 0.1 * (i as f32 / count as f32);
                    make_sample(if i % 2 == 0 { 0.3 } else { 0.7 }, error)
                })
                .collect()
        };

        let samples_10 = make_samples(10);
        let samples_100 = make_samples(100);

        let metrics_10 = compute_confidence_metrics(&samples_10, 0.5, None);
        let metrics_100 = compute_confidence_metrics(&samples_100, 0.5, None);

        let width_10 = metrics_10.expected_score_gain_confidence_interval[1]
            - metrics_10.expected_score_gain_confidence_interval[0];
        let width_100 = metrics_100.expected_score_gain_confidence_interval[1]
            - metrics_100.expected_score_gain_confidence_interval[0];

        assert!(
            width_10 > 0.0 && width_100 > 0.0,
            "Both intervals should have positive width: w10={width_10}, w100={width_100}"
        );

        let ratio = width_10 / width_100;
        assert!(
            (2.0..=5.0).contains(&ratio),
            "10-sample interval should be ~2-5× wider than 100-sample: ratio={ratio}, w10={width_10}, w100={width_100}"
        );
    }

    #[test]
    fn test_nan_samples_do_not_produce_nan_intervals() {
        let samples = vec![
            make_sample(f32::NAN, f32::NAN),
            make_sample(0.5, 0.1),
            make_sample(f32::NAN, 0.2),
            make_sample(0.7, f32::NAN),
            make_sample(0.3, 0.15),
        ];
        let metrics = compute_confidence_metrics(&samples, 0.1, None);

        let [lower, upper] = metrics.expected_score_gain_confidence_interval;
        assert!(
            lower.is_finite(),
            "Lower bound should be finite, got {lower}"
        );
        assert!(
            upper.is_finite(),
            "Upper bound should be finite, got {upper}"
        );
        assert!(
            metrics.prediction_confidence.is_finite(),
            "Confidence should be finite, got {}",
            metrics.prediction_confidence
        );
    }

    #[test]
    fn test_interval_always_contains_point_estimate() {
        // Test with various sample counts and expected gains
        for &count in &[5_usize, 10, 25, 50, 100, 500] {
            for &expected in &[-0.5_f32, -0.1, 0.0, 0.1, 0.5] {
                let samples: Vec<HelpfulSample> = (0..count)
                    .map(|i| make_sample(if i % 2 == 0 { 0.2 } else { 0.8 }, 0.1))
                    .collect();
                let metrics = compute_confidence_metrics(&samples, expected, None);
                let [lower, upper] = metrics.expected_score_gain_confidence_interval;
                assert!(
                    lower <= expected && upper >= expected,
                    "Interval [{lower}, {upper}] should contain estimate {expected} (n={count})"
                );
            }
        }
    }

    #[test]
    fn test_t_critical_lookup_known_values() {
        // Verify lookup table returns expected t-critical values for 95% CI
        // df=1: t=12.706, df=9: t≈2.262, df=120+: t≈1.96
        let t1 = t_critical_95(1);
        assert!(
            (t1 - 12.706).abs() < 0.01,
            "t(df=1) should be ~12.706, got {t1}"
        );

        let t9 = t_critical_95(9);
        assert!(
            (t9 - 2.262).abs() < 0.01,
            "t(df=9) should be ~2.262, got {t9}"
        );

        let t_large = t_critical_95(200);
        assert!(
            (t_large - 1.96).abs() < 0.01,
            "t(df=200) should be ~1.96, got {t_large}"
        );
    }
}
