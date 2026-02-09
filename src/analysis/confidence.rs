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

use serde::Serialize;

use super::samples::{EPSILON, HelpfulSample};

/// Minimum sample count for full confidence credit.
/// Below this, sample confidence is linearly scaled.
const MIN_CONFIDENT_SAMPLES: f32 = 100.0;

/// Minimum source standard deviation for full confidence credit.
/// Sources with std dev below this are progressively discounted.
const MIN_CONFIDENT_STD_DEV: f32 = 0.05;

/// Z-score for 95% confidence interval (approximately 1.96).
const Z_SCORE_95: f32 = 1.96;

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

    /// Confidence interval for the expected score gain: [lower_bound, upper_bound].
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
    let model_confidence = model_r_squared
        .map(compute_model_fit_confidence)
        .unwrap_or(1.0);

    // Overall confidence: geometric mean of all factors
    // All factors matter, so we use geometric mean (any low factor drags down the result)
    let overall_confidence =
        (sample_confidence * variance_confidence * model_confidence).powf(1.0 / 3.0);

    // Compute confidence interval
    let confidence_interval = compute_confidence_interval(
        samples,
        expected_score_gain,
        sample_confidence,
        variance_confidence,
    );

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

/// Compute 95% confidence interval for expected score gain.
///
/// The interval is computed using the standard error formula, adjusted for
/// sample size and source variance confidence.
fn compute_confidence_interval(
    samples: &[HelpfulSample],
    expected_score_gain: f32,
    sample_confidence: f32,
    variance_confidence: f32,
) -> [f32; 2] {
    if samples.is_empty() {
        return [0.0, 0.0];
    }

    let n = samples.len() as f32;

    // Compute standard error of the prediction
    // Based on error variance in the samples
    let error_variance = compute_error_variance(samples);
    let error_std_dev = error_variance.sqrt();

    // Standard error of the mean: σ / √n
    let standard_error = if n > 1.0 {
        error_std_dev / n.sqrt()
    } else {
        error_std_dev
    };

    // Adjust margin based on confidence factors
    // Lower confidence = wider interval
    let confidence_factor = (sample_confidence * variance_confidence).max(EPSILON);
    let adjusted_margin = if confidence_factor > EPSILON {
        (Z_SCORE_95 * standard_error) / confidence_factor
    } else {
        // Very low confidence: use a wide interval
        expected_score_gain.abs().max(0.1)
    };

    // Clamp margin to reasonable bounds
    let margin = adjusted_margin.clamp(0.0, expected_score_gain.abs().max(0.5));

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
}
