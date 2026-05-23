//! Sample-weighted discovery module (Issue #423).
//!
//! Prioritises high-error samples during discovery analysis. Current discovery
//! treats all samples equally, but samples with high error are more important
//! for improvement and should receive proportional analysis attention.
//!
//! ## Approach
//!
//! 1. **Compute sample importance**: Weight each sample by absolute error magnitude
//! 2. **Detect high-error neurons**: Identify neurons with disproportionate high-error samples
//! 3. **Stratified analysis**: Separately characterise easy vs hard samples
//! 4. **Candidate generation**: Produce coordinated candidates targeting high-error patterns
//!
//! ## Configuration
//!
//! Detection thresholds can be configured via `SampleWeightedConfig`:
//! - `min_weighted_error`: Minimum weighted error to flag a neuron (default: 0.25)
//! - `min_samples`: Minimum samples required for statistical reliability (default: 10)
//!
//! ## Quantised `{0, 1}` error regime (Issue #1247)
//!
//! Under `CATEGORICAL_ERROR`, the per-record absolute error collapses to
//! `{0, 1}` — every "weight" is either zero (correctly-classified
//! sample) or one (misclassified). The detector then reduces to:
//!
//! - `weighted_mean_error == misclassification_rate`
//! - the stratified hard/easy split == correctly-classified vs
//!   misclassified samples
//! - `hard_to_easy_ratio` becomes unbounded when no sample is
//!   correctly classified (already clamped to ≤ `10` downstream)
//!
//! The detector remains **finite and well-formed** — the existing
//! `total <= EPSILON` and `easy_mean > EPSILON` guards prevent NaN /
//! division-by-zero — and still produces a useful ranking signal
//! (neurons with high misclassification rates outscore neurons with
//! low ones). Magnitude is on a different scale to a continuous-error
//! batch, so downstream `min_weighted_error` thresholds may need to
//! be re-tuned for `CATEGORICAL_ERROR`. Detection is *degraded but
//! well-formed* under this regime (Issue #1247).
//!
//! See `crate::analysis::quantised_error::is_quantised_zero_one` for
//! the runtime predicate.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson};

// =============================================================================
// Constants
// =============================================================================

/// Default minimum weighted error to consider a neuron as high-error.
const DEFAULT_MIN_WEIGHTED_ERROR: f32 = 0.25;

/// Default minimum samples required for detection.
const DEFAULT_MIN_SAMPLES: usize = 10;

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for sample-weighted discovery.
#[derive(Debug, Clone)]
pub struct SampleWeightedConfig {
    /// Minimum weighted error to flag a neuron as high-error.
    pub min_weighted_error: f32,
    /// Minimum samples required for detection.
    pub min_samples: usize,
}

impl Default for SampleWeightedConfig {
    fn default() -> Self {
        Self {
            min_weighted_error: DEFAULT_MIN_WEIGHTED_ERROR,
            min_samples: DEFAULT_MIN_SAMPLES,
        }
    }
}

// =============================================================================
// Detection Types
// =============================================================================

/// A neuron identified as contributing disproportionately to high-error samples.
#[derive(Debug, Clone)]
pub struct HighErrorNeuronCandidate {
    /// UUID of the neuron with high weighted error.
    pub neuron_uuid: String,
    /// Weighted mean error across all samples.
    pub weighted_mean_error: f32,
    /// Proportion of samples in the high-error stratum (0.0 to 1.0).
    pub high_error_proportion: f32,
    /// Number of samples analysed.
    pub sample_count: usize,
    /// Hard-to-easy error ratio from stratified analysis.
    pub hard_to_easy_ratio: f32,
    /// Estimated creature score improvement from addressing this neuron.
    pub estimated_improvement: f32,
}

/// Result of stratifying samples into easy and hard groups.
#[derive(Debug, Clone)]
pub struct StratifiedAnalysis {
    /// Indices of easy (low-error) samples.
    pub easy_samples: Vec<usize>,
    /// Indices of hard (high-error) samples.
    pub hard_samples: Vec<usize>,
    /// Mean error of easy samples.
    pub easy_mean_error: f32,
    /// Mean error of hard samples.
    pub hard_mean_error: f32,
    /// Ratio of hard mean error to easy mean error.
    pub hard_to_easy_ratio: f32,
}

// =============================================================================
// Core Functions
// =============================================================================

/// Compute importance weights for each sample based on absolute error magnitude.
///
/// Returns a vector of normalised weights (summing to 1.0) where higher-error
/// samples receive proportionally higher weights.
///
/// Non-finite error values are filtered out and receive zero weight.
pub fn compute_sample_weights(records: &[DiscoverRecord]) -> Vec<f32> {
    if records.is_empty() {
        return Vec::new();
    }

    // Extract absolute error for each record, filtering non-finite values
    let abs_errors: Vec<f32> = records
        .iter()
        .map(|r| {
            let avg_err = if r.errors.is_empty() {
                0.0
            } else {
                r.errors.iter().sum::<f32>() / r.errors.len() as f32
            };
            if avg_err.is_finite() {
                avg_err.abs()
            } else {
                0.0
            }
        })
        .collect();

    let total: f32 = abs_errors.iter().sum();

    if total <= f32::EPSILON {
        // All errors are zero or negligible — return uniform weights
        let uniform = 1.0 / abs_errors.len() as f32;
        return vec![uniform; abs_errors.len()];
    }

    // Normalise to sum to 1.0
    abs_errors.iter().map(|&e| e / total).collect()
}

/// Stratify samples into easy (low-error) and hard (high-error) groups.
///
/// Uses the median absolute error as the split point. Returns indices into
/// the original records vector for each group.
pub fn stratify_samples(records: &[DiscoverRecord]) -> StratifiedAnalysis {
    if records.is_empty() {
        return StratifiedAnalysis {
            easy_samples: Vec::new(),
            hard_samples: Vec::new(),
            easy_mean_error: 0.0,
            hard_mean_error: 0.0,
            hard_to_easy_ratio: 1.0,
        };
    }

    // Compute absolute error for each record
    let abs_errors: Vec<f32> = records
        .iter()
        .map(|r| {
            if r.errors.is_empty() {
                0.0
            } else {
                let avg = r.errors.iter().sum::<f32>() / r.errors.len() as f32;
                if avg.is_finite() { avg.abs() } else { 0.0 }
            }
        })
        .collect();

    // Issue #943: Compute median using select_nth_unstable (O(n) average)
    // instead of cloning the entire Vec and sorting (O(n log n) + allocation).
    let median_idx = abs_errors.len() / 2;
    let mut median_scratch: Vec<f32> = abs_errors.clone();
    median_scratch.select_nth_unstable_by(median_idx, f32::total_cmp);
    let median = median_scratch[median_idx];

    // Split into easy (≤ median) and hard (> median)
    let mut easy_samples = Vec::new();
    let mut hard_samples = Vec::new();
    let mut easy_sum = 0.0f32;
    let mut hard_sum = 0.0f32;

    for (i, &err) in abs_errors.iter().enumerate() {
        if err <= median {
            easy_samples.push(i);
            easy_sum += err;
        } else {
            hard_samples.push(i);
            hard_sum += err;
        }
    }

    // Handle edge case where all values equal median (all go to easy)
    if hard_samples.is_empty() && !easy_samples.is_empty() {
        // Split in half: move second half to hard
        let mid = easy_samples.len() / 2;
        hard_samples = easy_samples.split_off(mid);
        // Recalculate sums
        easy_sum = easy_samples.iter().map(|&i| abs_errors[i]).sum();
        hard_sum = hard_samples.iter().map(|&i| abs_errors[i]).sum();
    }

    let easy_mean = if easy_samples.is_empty() {
        0.0
    } else {
        easy_sum / easy_samples.len() as f32
    };

    let hard_mean = if hard_samples.is_empty() {
        0.0
    } else {
        hard_sum / hard_samples.len() as f32
    };

    let ratio = if easy_mean > f32::EPSILON {
        hard_mean / easy_mean
    } else if hard_mean > f32::EPSILON {
        hard_mean / f32::EPSILON
    } else {
        1.0
    };

    StratifiedAnalysis {
        easy_samples,
        hard_samples,
        easy_mean_error: easy_mean,
        hard_mean_error: hard_mean,
        hard_to_easy_ratio: ratio,
    }
}

/// Detect neurons with disproportionately high weighted error.
///
/// For each neuron, computes a weighted mean error (weighted by sample importance)
/// and flags neurons that exceed the configured threshold.
///
/// # Arguments
/// * `records` - Vector of (`neuron_uuid`, records) tuples
/// * `config` - Configuration for detection thresholds
///
/// # Returns
/// Vector of candidates, sorted by estimated improvement (best first).
pub fn detect_high_error_neurons(
    records: &[(String, Vec<DiscoverRecord>)],
    config: &SampleWeightedConfig,
) -> Vec<HighErrorNeuronCandidate> {
    let mut candidates = Vec::new();

    for (neuron_uuid, neuron_records) in records {
        if neuron_records.len() < config.min_samples {
            continue;
        }

        // Compute sample weights
        let weights = compute_sample_weights(neuron_records);
        if weights.is_empty() {
            continue;
        }

        // Compute absolute errors
        let abs_errors: Vec<f32> = neuron_records
            .iter()
            .map(|r| {
                if r.errors.is_empty() {
                    0.0
                } else {
                    let avg = r.errors.iter().sum::<f32>() / r.errors.len() as f32;
                    if avg.is_finite() { avg.abs() } else { 0.0 }
                }
            })
            .collect();

        // Compute weighted mean error
        let weighted_mean: f32 = weights
            .iter()
            .zip(abs_errors.iter())
            .map(|(&w, &e)| w * e)
            .sum();

        if weighted_mean < config.min_weighted_error {
            continue;
        }

        // Stratify to get hard-to-easy ratio and high-error proportion
        let stratified = stratify_samples(neuron_records);
        let high_error_proportion =
            stratified.hard_samples.len() as f32 / neuron_records.len() as f32;

        // Estimate improvement: weighted mean error scaled by the hard-to-easy
        // ratio, capped at a reasonable maximum
        let estimated_improvement =
            (weighted_mean * stratified.hard_to_easy_ratio.min(10.0) * 0.01).min(0.1);

        candidates.push(HighErrorNeuronCandidate {
            neuron_uuid: neuron_uuid.clone(),
            weighted_mean_error: weighted_mean,
            high_error_proportion,
            sample_count: neuron_records.len(),
            hard_to_easy_ratio: stratified.hard_to_easy_ratio,
            estimated_improvement,
        });
    }

    // Sort by estimated improvement (best first)
    candidates.sort_by(|a, b| b.estimated_improvement.total_cmp(&a.estimated_improvement));

    candidates
}

// =============================================================================
// Candidate Conversion
// =============================================================================

/// Convert high-error neuron candidates to coordinated structural candidates.
///
/// Generates `setBias` operations to adjust the bias of high-error neurons,
/// which can help shift the neuron's operating point to reduce error on
/// difficult samples.
pub fn high_error_neurons_to_coordinated_candidates(
    candidates: &[HighErrorNeuronCandidate],
) -> Vec<CoordinatedStructuralCandidateJson> {
    let mut result: Vec<CoordinatedStructuralCandidateJson> = candidates
        .iter()
        .map(|c| {
            // Recommend a bias adjustment proportional to the weighted error
            let bias_adjustment = -c.weighted_mean_error * 0.1;

            CoordinatedStructuralCandidateJson {
                operations: vec![CoordinatedStructuralOpJson::SetBias {
                    neuron_uuid: c.neuron_uuid.clone(),
                    bias: bias_adjustment,
                }],
                expected_creature_score_gain: c.estimated_improvement,
                comment: Some(format!(
                    "Sample-weighted discovery: weighted error {:.3}, {:.0}% high-error samples, hard/easy ratio {:.1}x ({} samples)",
                    c.weighted_mean_error,
                    c.high_error_proportion * 100.0,
                    c.hard_to_easy_ratio,
                    c.sample_count,
                )),
            }
        })
        .collect();

    // Sort by improvement (best first)
    result.sort_by(|a, b| {
        b.expected_creature_score_gain
            .total_cmp(&a.expected_creature_score_gain)
    });

    result
}
