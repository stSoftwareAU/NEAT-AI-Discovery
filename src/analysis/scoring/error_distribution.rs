//! Error distribution analysis for targeted discovery (Issue #192).
//!
//! This module computes distribution statistics — mean, variance, skewness,
//! kurtosis, and percentiles — to help identify non-uniform error patterns
//! that may benefit from targeted discovery approaches.
//!
//! ## Quantised `{0, 1}` error regime (Issue #1247)
//!
//! Under `CATEGORICAL_ERROR` the underlying `avg_error` collapses to a
//! Bernoulli sample over `{0, 1}`. The moment-based statistics still
//! evaluate cleanly:
//!
//! - `variance = p(1 − p)` (finite, bounded by `0.25`)
//! - `skewness = (1 − 2p) / sqrt(p(1 − p))` (finite when `p ∉ {0, 1}`)
//! - `kurtosis = (1 − 6p(1 − p)) / (p(1 − p)) + 3` (finite when
//!   `p ∉ {0, 1}`)
//!
//! The existing `std_dev > 1e-10` guard already handles the all-zero
//! and all-one degenerate cases by emitting `skewness = 0` and
//! `kurtosis = 3` (the Gaussian default) instead of dividing by zero.
//! Percentile interpolation also remains exact because every value is
//! one of `{0, 1}`.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use crate::analysis::samples::HelpfulSample;
use serde::Serialize;

// =============================================================================
// Error Distribution Statistics
// =============================================================================

/// Comprehensive error distribution statistics.
///
/// This structure provides detailed information about the error distribution
/// across samples, enabling targeted discovery for specific error patterns.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ErrorDistribution {
    /// Mean error value.
    pub mean: f32,

    /// Standard deviation of errors.
    pub std_dev: f32,

    /// Variance of errors (`std_dev^2`).
    pub variance: f32,

    /// Skewness - asymmetry indicator.
    ///
    /// - Negative: tail on the left (more high values)
    /// - Zero: symmetric distribution
    /// - Positive: tail on the right (outliers on high end)
    pub skewness: f32,

    /// Kurtosis - tail heaviness indicator.
    ///
    /// - Low (< 3): lighter tails, flatter peak (platykurtic)
    /// - Normal (= 3): normal distribution (mesokurtic)
    /// - High (> 3): heavier tails, sharper peak (leptokurtic)
    pub kurtosis: f32,

    /// Percentiles: [p10, p25, p50 (median), p75, p90].
    pub percentiles: [f32; 5],

    /// Minimum error value observed.
    pub min: f32,

    /// Maximum error value observed.
    pub max: f32,

    /// Number of samples used to compute distribution.
    pub sample_count: usize,

    /// Interquartile range (p75 - p25).
    pub iqr: f32,
}

impl ErrorDistribution {
    /// Compute error distribution statistics from samples.
    ///
    /// Returns `None` if samples is empty.
    pub fn from_samples(samples: &[HelpfulSample]) -> Option<Self> {
        if samples.is_empty() {
            return None;
        }

        // Extract errors and filter out non-finite values
        let errors: Vec<f32> = samples
            .iter()
            .map(|s| s.avg_error)
            .filter(|e| e.is_finite())
            .collect();

        if errors.is_empty() {
            return None;
        }

        Self::from_errors(&errors)
    }

    /// Compute error distribution statistics from raw error values.
    pub fn from_errors(errors: &[f32]) -> Option<Self> {
        if errors.is_empty() {
            return None;
        }

        let n = errors.len() as f64;

        // Compute mean
        let sum: f64 = errors.iter().map(|&e| e as f64).sum();
        let mean = sum / n;

        // Compute variance and std_dev
        let variance_sum: f64 = errors.iter().map(|&e| (e as f64 - mean).powi(2)).sum();
        let variance = variance_sum / n;
        let std_dev = variance.sqrt();

        // Compute skewness (third standardised moment)
        let skewness = if std_dev > 1e-10 {
            let m3: f64 = errors.iter().map(|&e| (e as f64 - mean).powi(3)).sum();
            (m3 / n) / std_dev.powi(3)
        } else {
            0.0
        };

        // Compute kurtosis (fourth standardised moment)
        let kurtosis = if std_dev > 1e-10 {
            let m4: f64 = errors.iter().map(|&e| (e as f64 - mean).powi(4)).sum();
            (m4 / n) / std_dev.powi(4)
        } else {
            3.0 // Default to normal distribution kurtosis
        };

        // Compute percentiles
        let percentiles = compute_percentiles(errors);

        // Compute min and max
        let min = errors.iter().copied().fold(f32::INFINITY, f32::min);
        let max = errors.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Interquartile range
        let iqr = percentiles[3] - percentiles[1];

        Some(Self {
            mean: mean as f32,
            std_dev: std_dev as f32,
            variance: variance as f32,
            skewness: skewness as f32,
            kurtosis: kurtosis as f32,
            percentiles,
            min,
            max,
            sample_count: errors.len(),
            iqr,
        })
    }

}

/// Compute percentiles [p10, p25, p50, p75, p90] from a slice of values.
fn compute_percentiles(values: &[f32]) -> [f32; 5] {
    if values.is_empty() {
        return [0.0; 5];
    }

    let mut sorted: Vec<f32> = values.to_vec();
    sorted.sort_by(f32::total_cmp);

    let n = sorted.len();

    // Percentile calculation using linear interpolation
    let percentile = |p: f64| -> f32 {
        let idx = p * (n - 1) as f64;
        let lower = idx.floor() as usize;
        let upper = idx.ceil() as usize;
        let frac = idx - lower as f64;

        if lower == upper || upper >= n {
            sorted[lower.min(n - 1)]
        } else {
            sorted[lower] * (1.0 - frac as f32) + sorted[upper] * (frac as f32)
        }
    };

    [
        percentile(0.10),
        percentile(0.25),
        percentile(0.50),
        percentile(0.75),
        percentile(0.90),
    ]
}

// =============================================================================
// Unit Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sample(error: f32) -> HelpfulSample {
        HelpfulSample {
            activation: 0.5,
            avg_error: error,
            target_value: None,
            target_activation: None,
        }
    }

    #[test]
    fn test_error_distribution_basic() {
        let samples: Vec<HelpfulSample> = (0..100).map(|i| make_sample(i as f32 / 100.0)).collect();

        let dist = ErrorDistribution::from_samples(&samples).unwrap();

        // Mean should be ~0.495
        assert!((dist.mean - 0.495).abs() < 0.01, "Mean: {}", dist.mean);
        assert!(dist.std_dev > 0.0);
        assert_eq!(dist.sample_count, 100);
    }

    #[test]
    fn test_percentile_calculation() {
        let values: Vec<f32> = (1..=100).map(|i| i as f32).collect();
        let percentiles = compute_percentiles(&values);

        assert!((percentiles[0] - 10.0).abs() < 2.0); // p10
        assert!((percentiles[1] - 25.0).abs() < 2.0); // p25
        assert!((percentiles[2] - 50.0).abs() < 2.0); // p50
        assert!((percentiles[3] - 75.0).abs() < 2.0); // p75
        assert!((percentiles[4] - 90.0).abs() < 2.0); // p90
    }

    #[test]
    fn test_skewness_symmetric() {
        let samples: Vec<HelpfulSample> = (-50..50).map(|i| make_sample(i as f32)).collect();

        let dist = ErrorDistribution::from_samples(&samples).unwrap();

        assert!(
            dist.skewness.abs() < 0.1,
            "Symmetric distribution skewness: {}",
            dist.skewness
        );
    }

    #[test]
    fn test_skewness_right_skewed() {
        let mut errors: Vec<f32> = (0..90).map(|i| i as f32 * 0.1).collect();
        errors.extend(vec![
            50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0,
        ]);

        let samples: Vec<HelpfulSample> = errors.iter().map(|&e| make_sample(e)).collect();

        let dist = ErrorDistribution::from_samples(&samples).unwrap();

        assert!(
            dist.skewness > 0.5,
            "Right-skewed distribution skewness: {}",
            dist.skewness
        );
    }

    #[test]
    fn test_empty_samples() {
        let dist = ErrorDistribution::from_samples(&[]);
        assert!(dist.is_none());
    }

    #[test]
    fn test_single_sample() {
        let samples = vec![make_sample(1.0)];
        let dist = ErrorDistribution::from_samples(&samples);

        if let Some(d) = dist {
            assert_eq!(d.mean, 1.0);
            assert_eq!(d.std_dev, 0.0);
        }
    }

}
