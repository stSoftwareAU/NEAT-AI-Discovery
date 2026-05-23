//! Bimodal neuron detection module (Issue #640).
//!
//! Identifies hidden neurons whose pre-activation (`value`) distribution is bimodal
//! or multi-modal across observations. A bimodal distribution indicates the neuron is
//! effectively serving two distinct input regimes and should be split into two
//! specialised neurons, each handling one mode.
//!
//! ## Key differences from related modules
//!
//! - `oscillating_neuron.rs` (Issue #358): Detects post-activation sign changes
//!   (temporal oscillation). Bimodal detection examines the **shape** of the
//!   pre-activation distribution regardless of sign or temporal ordering.
//! - `activation_mismatch.rs` (Issue #543): Checks RELU negative fraction.
//!   Bimodal detection looks for multi-modal clustering in the full value range.
//!
//! ## Detection Algorithm
//!
//! Uses a gap-based approach to detect bimodality:
//! 1. Sort the pre-activation values.
//! 2. Compute gaps between consecutive sorted values.
//! 3. Find the largest gap that satisfies minimum cluster size constraints.
//! 4. If the largest gap is significantly larger than the median gap (indicating
//!    a clear valley between two modes), flag as bimodal.
//!
//! This approach is robust against uniform and unimodal distributions, which have
//! roughly equal gaps, while bimodal distributions have one dominant gap.
//!
//! ## Recommended Actions
//!
//! When bimodality is detected, we recommend adding a new neuron to split the
//! bimodal neuron. Each candidate includes bias offsets targeting each mode.
//!
//! ## `CATEGORICAL_ERROR` / quantised error regime (Issue #1247)
//!
//! This detector inspects the recorded **pre-activation value**
//! (`DiscoverRecord.value`), not the error field, so it is *not*
//! sensitive to the `CATEGORICAL_ERROR` `{0, 1}` regime described in
//! `docs/COST_FUNCTION_NOTES.md`. The cluster coherence and gap-ratio
//! guards (`MIN_GAP_RATIO`, `MAX_CLUSTER_VARIANCE_RATIO`) keep the
//! detector well-formed even when the pre-activation distribution
//! happens to collapse to two points: each cluster has zero internal
//! variance, which still satisfies the coherence test, and the
//! candidate's `bimodality_score` and `estimated_improvement` remain
//! finite.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use super::helpers::build_record_map;
use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson};

use crate::analysis::constants::MIN_DISCOVERY_SAMPLE_COUNT;

/// Minimum ratio of the largest gap to the median gap to consider a distribution
/// bimodal. A value of 5.0 means the biggest gap must be at least 5× the median.
const MIN_GAP_RATIO: f32 = 5.0;

/// Minimum fraction of samples in the smaller cluster.
/// Prevents flagging distributions where one "cluster" has very few points.
const MIN_CLUSTER_FRACTION: f32 = 0.15;

/// Maximum allowed ratio of within-cluster variance to overall variance (Issue #751).
///
/// After detecting a gap, both clusters must have variance below this fraction
/// of the overall variance. This ensures each cluster is internally coherent
/// rather than a broad spread of values that happens to sit on one side of a gap.
///
/// A value of 0.5 means each cluster's variance must be less than half the
/// overall variance. True bimodal distributions typically have ratios well
/// below 0.1, while false positives from skewed unimodal distributions have
/// ratios near 1.0.
const MAX_CLUSTER_VARIANCE_RATIO: f32 = 0.5;

/// Result of detecting a bimodal neuron.
#[derive(Debug, Clone)]
pub struct BimodalNeuronCandidate {
    /// UUID of the bimodal neuron.
    pub neuron_uuid: String,
    /// Current activation function of the neuron.
    pub current_squash: String,
    /// Current bias of the neuron.
    pub current_bias: f32,
    /// Bimodality score (ratio of largest gap to median gap).
    pub bimodality_score: f32,
    /// Mean of the lower mode.
    pub lower_mode_mean: f32,
    /// Mean of the upper mode.
    pub upper_mode_mean: f32,
    /// Number of samples in the lower mode.
    pub lower_mode_count: usize,
    /// Number of samples in the upper mode.
    pub upper_mode_count: usize,
    /// Number of samples analysed (with valid `value`).
    pub sample_count: usize,
    /// Estimated creature score improvement from splitting.
    pub estimated_improvement: f32,
}

/// Detect bimodal neurons from their recorded pre-activation values.
///
/// # Arguments
/// * `neurons` - List of `(neuron_uuid, squash, bias)` tuples for hidden neurons to check.
/// * `neuron_records` - List of `(neuron_uuid, records)` tuples with the recorded data.
///
/// # Returns
/// A list of `BimodalNeuronCandidate` for neurons with bimodal pre-activation
/// distributions, sorted by estimated improvement (best first).
pub fn detect_bimodal_neurons(
    neurons: &[(String, String, f32)],
    neuron_records: &[(String, Vec<DiscoverRecord>)],
) -> Vec<BimodalNeuronCandidate> {
    let mut candidates = Vec::with_capacity(neurons.len());

    let records_map = build_record_map(neuron_records);

    for (uuid, squash, bias) in neurons {
        let Some(records) = records_map.get(uuid.as_str()) else {
            continue;
        };

        // Extract pre-activation values, skipping None
        let mut values: Vec<f32> = records.iter().filter_map(|r| r.value).collect();

        if values.len() < MIN_DISCOVERY_SAMPLE_COUNT {
            continue;
        }

        values.sort_by(f32::total_cmp);

        if let Some(result) = compute_bimodality(&values) {
            let separation = (result.upper_mean - result.lower_mean).abs();
            let estimated_improvement = separation * result.gap_ratio * 0.001;

            candidates.push(BimodalNeuronCandidate {
                neuron_uuid: uuid.clone(),
                current_squash: squash.clone(),
                current_bias: *bias,
                bimodality_score: result.gap_ratio,
                lower_mode_mean: result.lower_mean,
                upper_mode_mean: result.upper_mean,
                lower_mode_count: result.lower_count,
                upper_mode_count: result.upper_count,
                sample_count: values.len(),
                estimated_improvement,
            });
        }
    }

    candidates.sort_by(|a, b| b.estimated_improvement.total_cmp(&a.estimated_improvement));
    candidates
}

/// Result of bimodality computation.
struct BimodalityResult {
    gap_ratio: f32,
    lower_mean: f32,
    upper_mean: f32,
    lower_count: usize,
    upper_count: usize,
}

/// Compute bimodality score using a gap-based approach.
///
/// For sorted `values`, compute the gap between each consecutive pair. The largest
/// gap that respects cluster size constraints is compared against the median gap.
/// A large ratio indicates a clear valley between two modes.
fn compute_bimodality(values: &[f32]) -> Option<BimodalityResult> {
    let n = values.len();
    if n < 2 {
        return None;
    }

    let min_cluster_size = ((n as f32) * MIN_CLUSTER_FRACTION).ceil() as usize;
    if min_cluster_size < 1 || n < 2 * min_cluster_size {
        return None;
    }

    // Compute gaps between consecutive sorted values
    let gaps: Vec<f32> = values.windows(2).map(|w| w[1] - w[0]).collect();

    // Find median gap
    let mut sorted_gaps = gaps.clone();
    sorted_gaps.sort_by(f32::total_cmp);
    let median_gap = sorted_gaps[sorted_gaps.len() / 2];

    if median_gap < 1e-10 {
        // If median gap is ~0, check if there's any gap at all
        // (e.g., mostly identical values with a few outliers)
        let max_gap = sorted_gaps.last().copied().unwrap_or(0.0);
        if max_gap < 1e-6 {
            return None; // All values essentially identical
        }
        // Use a small positive denominator to avoid division by zero
        // but still allow detection when there's a genuine gap
    }

    // Find the largest gap that satisfies cluster size constraints
    let mut best_gap: f32 = 0.0;
    let mut best_split_index = 0;

    for (i, &gap) in gaps.iter().enumerate() {
        let split = i + 1; // split after index i
        if split >= min_cluster_size && (n - split) >= min_cluster_size && gap > best_gap {
            best_gap = gap;
            best_split_index = split;
        }
    }

    if best_gap <= 0.0 {
        return None;
    }

    // Compute gap ratio (largest gap / median gap)
    let gap_ratio = if median_gap > 1e-10 {
        best_gap / median_gap
    } else {
        // Median gap is ~0 but we have a real gap → very strong bimodality
        1e6_f32.min(best_gap * 1e6)
    };

    if gap_ratio < MIN_GAP_RATIO {
        return None;
    }

    let lower = &values[..best_split_index];
    let upper = &values[best_split_index..];

    // Cluster coherence validation (Issue #751): verify that both clusters
    // have variance meaningfully lower than the overall variance. This rejects
    // false positives from skewed unimodal distributions where a large gap
    // exists but one or both "clusters" are not internally coherent.
    let overall_var = variance(values);
    if overall_var > 1e-10 {
        let lower_var = variance(lower);
        let upper_var = variance(upper);
        if lower_var / overall_var > MAX_CLUSTER_VARIANCE_RATIO
            || upper_var / overall_var > MAX_CLUSTER_VARIANCE_RATIO
        {
            return None;
        }
    }

    Some(BimodalityResult {
        gap_ratio,
        lower_mean: mean(lower),
        upper_mean: mean(upper),
        lower_count: lower.len(),
        upper_count: upper.len(),
    })
}

fn mean(values: &[f32]) -> f32 {
    values.iter().sum::<f32>() / values.len() as f32
}

/// Compute population variance of a slice of f32 values.
fn variance(values: &[f32]) -> f32 {
    let m = mean(values);
    values.iter().map(|&v| (v - m) * (v - m)).sum::<f32>() / values.len() as f32
}

/// Convert bimodal neuron candidates into coordinated structural candidates.
///
/// Each bimodal neuron produces an `AddNeuron` coordinated candidate to split
/// the neuron into two, with bias offsets targeting each mode.
pub fn bimodal_neurons_to_coordinated_candidates(
    candidates: &[BimodalNeuronCandidate],
) -> Vec<CoordinatedStructuralCandidateJson> {
    let mut results = Vec::with_capacity(candidates.len());

    for c in candidates {
        // Create a new neuron biased toward the mode that the original neuron
        // does not naturally favour. The original neuron keeps its current bias
        // (close to one mode); the new neuron is biased toward the other mode.
        let new_neuron_bias = c.lower_mode_mean;

        let operations = vec![CoordinatedStructuralOpJson::AddNeuron {
            neuron_uuid: format!("{}-split-lower", c.neuron_uuid),
            neuron_type: "hidden".to_string(),
            squash: c.current_squash.clone(),
            bias: new_neuron_bias,
            insert_before_neuron_uuid: Some(c.neuron_uuid.clone()),
        }];

        results.push(CoordinatedStructuralCandidateJson {
            operations,
            expected_creature_score_gain: c.estimated_improvement,
            comment: Some(format!(
                "Bimodal neuron {}: pre-activation distribution has two modes at {:.3} ({} samples) and {:.3} ({} samples), bimodality score {:.2} — split to specialise each mode",
                c.neuron_uuid, c.lower_mode_mean, c.lower_mode_count, c.upper_mode_mean, c.upper_mode_count, c.bimodality_score
            )),
        });
    }

    results.sort_by(|a, b| {
        b.expected_creature_score_gain
            .total_cmp(&a.expected_creature_score_gain)
    });

    results
}
