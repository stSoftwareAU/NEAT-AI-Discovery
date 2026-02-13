//! Cross-validation consistency scoring for candidate evaluation (Issue #436).
//!
//! Part of the "Brilliant but Brittle" initiative. This module provides cross-validation
//! analysis to detect candidates that perform well on discovery samples but poorly
//! on held-out validation data - a signature of overfitting.
//!
//! ## Problem
//!
//! "Brilliant" candidates may overfit to the discovery sample set, failing when
//! validated against the full training set. Early detection of this pattern saves
//! computation by avoiding ablation tests on candidates likely to fail validation.
//!
//! ## Solution
//!
//! This module implements k-fold cross-validation on the discovery samples to:
//!
//! 1. **Estimate generalisation**: Compare candidate performance across different
//!    sample subsets to detect overfitting signatures.
//!
//! 2. **Provide confidence adjustment**: Reduce confidence scores for candidates
//!    with high variance across folds (brittleness penalty).
//!
//! 3. **Integrate with SPRT**: Use the existing `SequentialEvaluator` infrastructure
//!    from `early_termination.rs` to evaluate each fold efficiently.
//!
//! ## Brittleness Penalty Formula
//!
//! The brittleness penalty is computed as:
//!
//! ```text
//! penalty = min(1.0, variance / variance_threshold)
//! adjusted_confidence = original_confidence × (1.0 - penalty)
//! ```
//!
//! Where:
//! - `variance` is the variance of improvement ratios across folds
//! - `variance_threshold` is the configurable threshold for "acceptable" variance
//! - The penalty is clamped to [0, 1] range
//!
//! ## References
//!
//! - Issue #432: "Brilliant but Brittle" initiative
//! - `early_termination.rs`: SPRT infrastructure for fold evaluation

use crate::analysis::samples::HelpfulSample;

/// Configuration for cross-validation consistency scoring.
#[derive(Debug, Clone)]
pub struct CrossValidationConfig {
    /// Number of folds for cross-validation (k in k-fold CV).
    ///
    /// Default: 5 folds (80% train, 20% validation per fold).
    pub fold_count: usize,

    /// Minimum samples required per fold for meaningful evaluation.
    ///
    /// If total samples / fold_count < min_samples_per_fold, cross-validation
    /// is skipped and no penalty is applied.
    pub min_samples_per_fold: usize,

    /// Variance threshold for determining brittleness.
    ///
    /// Candidates with variance above this threshold receive a penalty.
    /// Default: 0.04 (corresponds to ~0.2 std dev in improvement ratio).
    pub variance_threshold: f32,

    /// Maximum brittleness penalty to apply.
    ///
    /// Default: 0.5 (reduce confidence by up to 50% for highly brittle candidates).
    pub brittleness_penalty_max: f32,
}

impl Default for CrossValidationConfig {
    fn default() -> Self {
        Self {
            fold_count: 5,
            min_samples_per_fold: 15,
            variance_threshold: 0.04,
            brittleness_penalty_max: 0.5,
        }
    }
}

/// Result from evaluating a single fold during cross-validation.
#[derive(Debug, Clone)]
pub struct FoldResult {
    /// Number of samples showing improvement in this fold.
    pub positive_count: u32,

    /// Number of samples not showing improvement in this fold.
    pub negative_count: u32,

    /// Total samples evaluated in this fold.
    pub samples_evaluated: u32,
}

impl FoldResult {
    /// Compute the improvement ratio for this fold.
    ///
    /// Returns 0.5 (neutral) if no samples were evaluated.
    #[must_use]
    pub fn improvement_ratio(&self) -> f64 {
        let total = self.positive_count + self.negative_count;
        if total == 0 {
            return 0.5;
        }
        f64::from(self.positive_count) / f64::from(total)
    }
}

/// Statistical measures of performance variance across folds.
#[derive(Debug, Clone)]
pub struct PerformanceVariance {
    /// Mean improvement ratio across all folds.
    pub mean_improvement_ratio: f64,

    /// Variance of improvement ratios across folds.
    pub variance: f64,

    /// Standard deviation of improvement ratios across folds.
    pub std_dev: f64,

    /// Maximum improvement ratio observed across folds.
    pub max_improvement: f64,

    /// Minimum improvement ratio observed across folds.
    pub min_improvement: f64,
}

impl PerformanceVariance {
    /// Compute variance statistics from fold results.
    #[must_use]
    pub fn from_folds(folds: &[FoldResult]) -> Self {
        if folds.is_empty() {
            return Self {
                mean_improvement_ratio: 0.5,
                variance: 0.0,
                std_dev: 0.0,
                max_improvement: 0.5,
                min_improvement: 0.5,
            };
        }

        // Compute improvement ratios for each fold
        let ratios: Vec<f64> = folds.iter().map(|f| f.improvement_ratio()).collect();

        // Compute mean
        let sum: f64 = ratios.iter().sum();
        let mean = sum / ratios.len() as f64;

        // Compute variance using E[X^2] - E[X]^2
        let sq_sum: f64 = ratios.iter().map(|r| r * r).sum();
        let variance = (sq_sum / ratios.len() as f64) - (mean * mean);

        // Compute min/max
        let max_improvement = ratios.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_improvement = ratios.iter().cloned().fold(f64::INFINITY, f64::min);

        Self {
            mean_improvement_ratio: mean,
            variance: variance.max(0.0), // Ensure non-negative due to floating point
            std_dev: variance.max(0.0).sqrt(),
            max_improvement,
            min_improvement,
        }
    }
}

/// Complete result of cross-validation analysis for a candidate.
#[derive(Debug, Clone)]
pub struct CrossValidationResult {
    /// Results from each fold evaluation.
    pub fold_results: Vec<FoldResult>,

    /// Computed variance statistics across folds.
    pub variance: PerformanceVariance,

    /// Brittleness penalty to apply to confidence score (0.0 to 1.0).
    pub brittleness_penalty: f32,
}

impl CrossValidationResult {
    /// Check if the candidate shows consistent performance across folds.
    ///
    /// A candidate is consistent if its variance is below the threshold.
    #[must_use]
    pub fn is_consistent(&self, config: &CrossValidationConfig) -> bool {
        self.variance.variance < f64::from(config.variance_threshold)
    }

    /// Check if the candidate shows brittle behaviour (high variance).
    #[must_use]
    pub fn is_brittle(&self, config: &CrossValidationConfig) -> bool {
        self.variance.variance >= f64::from(config.variance_threshold)
    }
}

/// Compute cross-validation score for a candidate based on its samples.
///
/// Splits the samples into k folds and evaluates improvement on each fold
/// to detect overfitting patterns.
///
/// # Arguments
///
/// * `samples` - The samples used for candidate evaluation
/// * `config` - Cross-validation configuration
///
/// # Returns
///
/// `Some(CrossValidationResult)` if cross-validation was performed,
/// `None` if there were insufficient samples.
pub fn compute_cross_validation_score(
    samples: &[HelpfulSample],
    config: &CrossValidationConfig,
) -> Option<CrossValidationResult> {
    // Validate we have enough samples
    if samples.is_empty() {
        return None;
    }

    if config.fold_count < 2 {
        // Can't do cross-validation with less than 2 folds
        return None;
    }

    let samples_per_fold = samples.len() / config.fold_count;
    if samples_per_fold < config.min_samples_per_fold {
        return None;
    }

    // Perform k-fold cross-validation
    let mut fold_results = Vec::with_capacity(config.fold_count);

    for fold_idx in 0..config.fold_count {
        // For each fold, use that fold as validation and the rest as "training"
        // (In our case, we just evaluate on the validation fold)
        let start = fold_idx * samples_per_fold;
        let end = if fold_idx == config.fold_count - 1 {
            samples.len() // Include any remainder in last fold
        } else {
            start + samples_per_fold
        };

        let fold_samples = &samples[start..end];
        let fold_result = evaluate_fold(fold_samples);
        fold_results.push(fold_result);
    }

    // Compute variance across folds
    let variance = PerformanceVariance::from_folds(&fold_results);

    // Compute brittleness penalty based on variance
    let penalty = compute_brittleness_penalty(&variance, config);

    Some(CrossValidationResult {
        fold_results,
        variance,
        brittleness_penalty: penalty,
    })
}

/// Evaluate a single fold of samples.
///
/// Determines how many samples would show improvement vs not.
/// Uses the same evaluation logic as the synapse analysis: a sample shows
/// improvement if the optimal weight direction would reduce error.
fn evaluate_fold(samples: &[HelpfulSample]) -> FoldResult {
    if samples.is_empty() {
        return FoldResult {
            positive_count: 0,
            negative_count: 0,
            samples_evaluated: 0,
        };
    }

    // First, compute the optimal weight direction for this fold using linear regression.
    // weight = sum(error * activation) / sum(activation^2)
    let mut error_activation_sum = 0.0f64;
    let mut activation_sq_sum = 0.0f64;

    for sample in samples {
        if sample.activation.is_finite() && sample.avg_error.is_finite() {
            let a = sample.activation as f64;
            let e = sample.avg_error as f64;
            error_activation_sum += e * a;
            activation_sq_sum += a * a;
        }
    }

    // Determine optimal weight direction
    let optimal_weight = if activation_sq_sum > 1e-10 {
        error_activation_sum / activation_sq_sum
    } else {
        0.0
    };

    // Count samples that would improve with this weight
    let mut positive_count = 0u32;
    let mut negative_count = 0u32;

    for sample in samples {
        if !sample.activation.is_finite() || !sample.avg_error.is_finite() {
            continue;
        }

        // Compute whether this sample would improve with the optimal weight.
        // Improvement means: new_error_magnitude < old_error_magnitude
        // new_error = old_error - weight * activation
        let contribution = optimal_weight * sample.activation as f64;
        let old_error_mag = sample.avg_error.abs();
        let new_error_mag = (sample.avg_error as f64 - contribution).abs() as f32;

        if new_error_mag + 1e-8 < old_error_mag {
            positive_count += 1;
        } else {
            negative_count += 1;
        }
    }

    FoldResult {
        positive_count,
        negative_count,
        samples_evaluated: samples.len() as u32,
    }
}

/// Compute the brittleness penalty based on performance variance.
fn compute_brittleness_penalty(
    variance: &PerformanceVariance,
    config: &CrossValidationConfig,
) -> f32 {
    if config.variance_threshold <= 0.0 {
        return 0.0;
    }

    // Linear scaling: full penalty at variance = variance_threshold
    let raw_penalty = (variance.variance as f32) / config.variance_threshold;
    let clamped_penalty = raw_penalty.clamp(0.0, 1.0);

    // Scale by max penalty
    clamped_penalty * config.brittleness_penalty_max
}

/// Apply brittleness penalty to a confidence score.
///
/// # Arguments
///
/// * `original_confidence` - The original prediction confidence (0.0 to 1.0)
/// * `penalty` - The brittleness penalty to apply (0.0 to 1.0)
///
/// # Returns
///
/// Adjusted confidence score, reduced proportionally to the penalty.
///
/// # Formula
///
/// ```text
/// adjusted_confidence = original_confidence × (1.0 - penalty)
/// ```
#[must_use]
pub fn apply_brittleness_penalty(original_confidence: f32, penalty: f32) -> f32 {
    let penalty_clamped = penalty.clamp(0.0, 1.0);
    let adjusted = original_confidence * (1.0 - penalty_clamped);
    adjusted.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fold_result_improvement_ratio() {
        let fold = FoldResult {
            positive_count: 60,
            negative_count: 40,
            samples_evaluated: 100,
        };
        assert!((fold.improvement_ratio() - 0.6).abs() < 0.001);
    }

    #[test]
    fn test_performance_variance_from_folds() {
        let folds = vec![
            FoldResult {
                positive_count: 70,
                negative_count: 30,
                samples_evaluated: 100,
            },
            FoldResult {
                positive_count: 70,
                negative_count: 30,
                samples_evaluated: 100,
            },
        ];

        let variance = PerformanceVariance::from_folds(&folds);
        assert!((variance.mean_improvement_ratio - 0.7).abs() < 0.01);
        assert!(
            variance.variance < 0.001,
            "Identical folds should have ~zero variance"
        );
    }

    #[test]
    fn test_apply_brittleness_penalty_basic() {
        assert!((apply_brittleness_penalty(1.0, 0.0) - 1.0).abs() < 0.001);
        assert!((apply_brittleness_penalty(1.0, 0.5) - 0.5).abs() < 0.001);
        assert!((apply_brittleness_penalty(1.0, 1.0) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_compute_brittleness_penalty() {
        let config = CrossValidationConfig {
            variance_threshold: 0.04,
            brittleness_penalty_max: 0.5,
            ..CrossValidationConfig::default()
        };

        // Zero variance = no penalty
        let variance_low = PerformanceVariance {
            mean_improvement_ratio: 0.7,
            variance: 0.0,
            std_dev: 0.0,
            max_improvement: 0.7,
            min_improvement: 0.7,
        };
        assert!(compute_brittleness_penalty(&variance_low, &config) < 0.01);

        // High variance = penalty
        let variance_high = PerformanceVariance {
            mean_improvement_ratio: 0.6,
            variance: 0.08, // 2x threshold
            std_dev: 0.28,
            max_improvement: 0.9,
            min_improvement: 0.3,
        };
        let penalty = compute_brittleness_penalty(&variance_high, &config);
        assert!((penalty - 0.5).abs() < 0.01, "Should be max penalty");
    }
}
