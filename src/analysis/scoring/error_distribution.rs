//! Error distribution analysis for targeted discovery (Issue #192).
//!
//! This module provides analysis of error distribution patterns to enable targeted
//! discovery for specific error patterns:
//! - Outlier samples with high error
//! - Bimodal error distributions
//! - Error clusters
//!
//! The analysis computes distribution statistics including percentiles, skewness,
//! and kurtosis to help identify non-uniform error patterns that may benefit from
//! targeted discovery approaches.

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

    /// Variance of errors (std_dev^2).
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

    /// Count samples that qualify as outliers based on a percentile threshold.
    ///
    /// # Arguments
    /// * `samples` - The samples to analyse
    /// * `percentile` - The percentile threshold (e.g., 90 for p90)
    ///
    /// Returns the number of samples whose absolute error exceeds the threshold.
    pub fn count_outliers(&self, samples: &[HelpfulSample], percentile: u8) -> usize {
        let threshold = self.get_percentile_threshold(percentile);

        samples
            .iter()
            .filter(|s| s.avg_error.abs() > threshold.abs())
            .count()
    }

    /// Get the error threshold for a given percentile.
    pub fn get_percentile_threshold(&self, percentile: u8) -> f32 {
        match percentile {
            p if p <= 10 => self.percentiles[0],
            p if p <= 25 => self.percentiles[1],
            p if p <= 50 => self.percentiles[2],
            p if p <= 75 => self.percentiles[3],
            _ => self.percentiles[4], // 90 or above
        }
    }

    /// Identify samples that are outliers based on the p90 threshold.
    pub fn filter_outliers<'a>(
        &self,
        samples: &'a [HelpfulSample],
        percentile: u8,
    ) -> Vec<&'a HelpfulSample> {
        let threshold = self.get_percentile_threshold(percentile);

        samples
            .iter()
            .filter(|s| s.avg_error.abs() > threshold.abs())
            .collect()
    }

    /// Check if the distribution is likely bimodal based on statistics.
    ///
    /// Indicators of bimodality:
    /// - High variance relative to the range
    /// - Large gap between median and mean
    /// - Unusual kurtosis (often < 3 for bimodal)
    pub fn is_likely_bimodal(&self) -> bool {
        // Use Sarle's bimodality coefficient: b = (g^2 + 1) / (k + 3(n-1)^2/(n-2)(n-3))
        // For large n, this simplifies to: b = (skewness^2 + 1) / (kurtosis)
        // A bimodality coefficient > 5/9 suggests bimodality

        if self.sample_count < 20 {
            return false;
        }

        let bimodality_coefficient = (self.skewness.powi(2) as f64 + 1.0) / (self.kurtosis as f64);
        let threshold = 5.0 / 9.0;

        // Also check for large gap between mean and median (p50)
        let mean_median_gap = (self.mean - self.percentiles[2]).abs();
        let large_gap = mean_median_gap > self.std_dev * 0.5;

        bimodality_coefficient > threshold || large_gap
    }

    /// Check if the distribution has significant outliers.
    ///
    /// Uses the IQR method: outliers are values beyond Q1 - 1.5*IQR or Q3 + 1.5*IQR
    pub fn has_significant_outliers(&self) -> bool {
        let lower_fence = self.percentiles[1] - 1.5 * self.iqr;
        let upper_fence = self.percentiles[3] + 1.5 * self.iqr;

        self.min < lower_fence || self.max > upper_fence
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
// Error Mode Detection
// =============================================================================

/// Represents a detected mode (cluster) in the error distribution.
#[derive(Debug, Clone)]
pub struct ErrorMode {
    /// Centre (mean) of this mode.
    pub centre: f32,

    /// Standard deviation within this mode.
    pub std_dev: f32,

    /// Number of samples in this mode.
    pub sample_count: usize,

    /// Proportion of total samples in this mode (0.0 to 1.0).
    pub proportion: f32,
}

/// Detect error modes (clusters) in the distribution.
///
/// Uses a simple histogram-based approach to identify distinct modes.
/// More sophisticated methods (k-means, Gaussian mixture models) could be added
/// if needed.
pub fn detect_error_modes(samples: &[HelpfulSample]) -> Vec<ErrorMode> {
    if samples.len() < 10 {
        return Vec::new();
    }

    let errors: Vec<f32> = samples
        .iter()
        .map(|s| s.avg_error)
        .filter(|e| e.is_finite())
        .collect();

    if errors.is_empty() {
        return Vec::new();
    }

    // Use histogram-based mode detection
    detect_modes_histogram(&errors)
}

/// Histogram-based mode detection.
///
/// Creates a histogram and identifies local maxima as modes.
fn detect_modes_histogram(errors: &[f32]) -> Vec<ErrorMode> {
    const NUM_BINS: usize = 20;

    let min = errors.iter().copied().fold(f32::INFINITY, f32::min);
    let max = errors.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let range = max - min;
    if range < 1e-6 {
        // All values are essentially the same
        return vec![ErrorMode {
            centre: min,
            std_dev: 0.0,
            sample_count: errors.len(),
            proportion: 1.0,
        }];
    }

    let bin_width = range / NUM_BINS as f32;

    // Count samples per bin
    let mut bins = vec![Vec::new(); NUM_BINS];
    for &error in errors {
        let bin_idx = ((error - min) / bin_width).floor() as usize;
        let bin_idx = bin_idx.min(NUM_BINS - 1); // Handle edge case
        bins[bin_idx].push(error);
    }

    // Find local maxima (bins with more samples than neighbours)
    let mut modes = Vec::new();
    let total = errors.len() as f32;

    for i in 0..NUM_BINS {
        let count = bins[i].len();
        if count < 3 {
            continue; // Skip nearly empty bins
        }

        let left_count = if i > 0 { bins[i - 1].len() } else { 0 };
        let right_count = if i < NUM_BINS - 1 {
            bins[i + 1].len()
        } else {
            0
        };

        // Is this a local maximum?
        // Require at least 10% more samples than neighbours to be considered a mode
        // (prevents detecting minor fluctuations as separate modes)
        let min_neighbour = left_count.min(right_count);
        let significant_peak = count > min_neighbour + (min_neighbour / 10).max(2);
        if count >= left_count && count >= right_count && significant_peak {
            let bin_errors = &bins[i];
            let sum: f64 = bin_errors.iter().map(|&e| e as f64).sum();
            let centre = (sum / bin_errors.len() as f64) as f32;

            // Compute std dev within bin
            let var: f64 = bin_errors
                .iter()
                .map(|&e| (e as f64 - centre as f64).powi(2))
                .sum::<f64>()
                / bin_errors.len() as f64;
            let std_dev = var.sqrt() as f32;

            modes.push(ErrorMode {
                centre,
                std_dev,
                sample_count: count,
                proportion: count as f32 / total,
            });
        }
    }

    // Filter out modes that are too small (< 5% of samples)
    modes.retain(|m| m.proportion >= 0.05);

    // Sort by sample count (largest first)
    modes.sort_by(|a, b| b.sample_count.cmp(&a.sample_count));

    modes
}

// =============================================================================
// Outlier Reduction Information
// =============================================================================

/// Information about how a candidate affects outlier samples.
///
/// Included with candidates when outlier analysis is enabled.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OutlierReductionInfo {
    /// Expected error reduction specifically for outlier samples.
    pub outlier_error_reduction: f32,

    /// Number of outlier samples affected by this candidate.
    pub outlier_samples_affected: usize,

    /// Total number of outliers in the dataset.
    pub total_outliers: usize,

    /// The percentile threshold used to define outliers.
    pub outlier_percentile: u8,
}

// =============================================================================
// Environment Variable Configuration
// =============================================================================

/// Check if outlier analysis is enabled via environment variable.
///
/// Delegates to [`crate::config::outlier_analysis()`].
pub fn outlier_analysis_enabled() -> bool {
    crate::config::outlier_analysis()
}

/// Get the outlier percentile threshold from environment variable.
///
/// Delegates to [`crate::config::outlier_percentile()`].
pub fn outlier_percentile_from_env() -> u8 {
    crate::config::outlier_percentile()
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

    #[test]
    fn test_outlier_counting() {
        // 90 samples with low error, 10 with high error
        let mut errors: Vec<f32> = vec![0.1; 90];
        errors.extend(vec![1.0; 10]);

        let samples: Vec<HelpfulSample> = errors.iter().map(|&e| make_sample(e)).collect();

        let dist = ErrorDistribution::from_samples(&samples).unwrap();
        let outlier_count = dist.count_outliers(&samples, 90);

        // Should be approximately 10
        assert!(
            (9..=11).contains(&outlier_count),
            "Outlier count: {outlier_count}"
        );
    }

    #[test]
    fn test_mode_detection_unimodal() {
        let samples: Vec<HelpfulSample> = (0..100)
            .map(|i| make_sample(0.5 + (i as f32 / 1000.0) - 0.05))
            .collect();

        let modes = detect_error_modes(&samples);

        // Should detect 1-2 modes at most
        assert!(
            modes.len() <= 2,
            "Unimodal should have ≤2 modes, got {}",
            modes.len()
        );
    }

    #[test]
    fn test_mode_detection_bimodal() {
        let mut errors: Vec<f32> = (0..50).map(|i| 0.1 + (i as f32 / 500.0)).collect();
        errors.extend((0..50).map(|i| 0.9 + (i as f32 / 500.0)));

        let samples: Vec<HelpfulSample> = errors.iter().map(|&e| make_sample(e)).collect();

        let modes = detect_error_modes(&samples);

        // Should detect 2+ modes
        assert!(
            modes.len() >= 2,
            "Bimodal should detect ≥2 modes, got {}",
            modes.len()
        );
    }

    #[test]
    #[serial_test::serial]
    fn test_env_var_defaults() {
        // SAFETY: Serialised via #[serial] — no concurrent env access.
        unsafe {
            std::env::remove_var("NEAT_AI_DISCOVERY_OUTLIER_ANALYSIS");
            std::env::remove_var("NEAT_AI_DISCOVERY_OUTLIER_PERCENTILE");
        }

        assert!(!outlier_analysis_enabled());
        assert_eq!(outlier_percentile_from_env(), 90);
    }
}
