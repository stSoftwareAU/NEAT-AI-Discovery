//! Bounded range neuron detection module (Issue #395).
//!
//! Identifies neurons whose activations are concentrated in a narrow sub-range,
//! with a significant fraction of samples at a "sentinel" value (e.g., -1 or 0)
//! that likely represents null/invalid data rather than a meaningful signal.
//!
//! For example, an observation like "Debt-to-Equity" may have meaningful values
//! in 0.2..0.8, but uses -1 as a null indicator. Multiplying -1 by a weight
//! produces a large negative contribution that distorts the output. This module
//! detects that pattern and recommends bias adjustments to shift the neuron's
//! operating point so the sentinel region has minimal effect.
//!
//! ## Detection Criteria
//!
//! A neuron has a "bounded range" pattern if:
//! 1. **Bimodal distribution**: activations cluster into a "meaningful" range and
//!    a separate "sentinel" cluster at the range boundary.
//! 2. **Sentinel fraction**: at least 10% of samples fall in the sentinel cluster.
//! 3. **Meaningful range is narrow**: the active sub-range covers less than 70%
//!    of the neuron's total activation span.
//! 4. **Not constant**: the neuron must have meaningful variance (not dead/stuck).
//!
//! ## Recommended Actions
//!
//! When a bounded range is detected, we recommend:
//! 1. **Adjust bias**: shift the neuron's operating point so sentinel values
//!    map to a neutral (near-zero) output.

use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson};

/// Minimum samples required for reliable bounded range detection.
const MIN_SAMPLES_FOR_BOUNDED_RANGE: usize = 20;

/// Minimum fraction of samples that must be sentinels to trigger detection.
const MIN_SENTINEL_FRACTION: f32 = 0.10;

/// Maximum fraction of the total span that the active range may cover.
/// If the active range covers more than this, the neuron is using its full range.
const MAX_ACTIVE_RANGE_RATIO: f32 = 0.70;

/// Minimum standard deviation to consider the neuron non-constant.
const MIN_STD_DEV: f32 = 0.05;

/// Distance threshold (in standard deviations) for identifying sentinel outliers.
const SENTINEL_DISTANCE_THRESHOLD: f32 = 1.0;

/// Result of detecting a bounded range neuron.
#[derive(Debug, Clone)]
pub struct BoundedRangeCandidate {
    /// UUID of the neuron with a bounded range pattern.
    pub neuron_uuid: String,
    /// Current activation function of the neuron.
    pub current_squash: String,
    /// Lower bound of the active (meaningful) range.
    pub active_range_low: f32,
    /// Upper bound of the active (meaningful) range.
    pub active_range_high: f32,
    /// Fraction of samples that are sentinels (outside the active range).
    pub sentinel_fraction: f32,
    /// Estimated improvement from addressing the bounded range.
    pub estimated_improvement: f32,
}

/// Detect neurons with bounded range patterns from their recorded activations.
///
/// # Arguments
/// * `neurons` - List of `(neuron_uuid, squash, bias)` tuples for neurons to check.
/// * `neuron_records` - List of `(neuron_uuid, records)` tuples with recorded activations.
///
/// # Returns
/// A list of `BoundedRangeCandidate` for neurons that exhibit bounded range patterns.
pub fn detect_bounded_range_neurons(
    neurons: &[(String, String, f32)],
    neuron_records: &[(String, Vec<DiscoverRecord>)],
) -> Vec<BoundedRangeCandidate> {
    let mut candidates = Vec::new();

    let records_map: std::collections::HashMap<&str, &Vec<DiscoverRecord>> = neuron_records
        .iter()
        .map(|(uuid, records)| (uuid.as_str(), records))
        .collect();

    for (uuid, squash, _bias) in neurons {
        let Some(records) = records_map.get(uuid.as_str()) else {
            continue;
        };

        if records.len() < MIN_SAMPLES_FOR_BOUNDED_RANGE {
            continue;
        }

        if let Some(candidate) = analyse_neuron_range(uuid, squash, records) {
            candidates.push(candidate);
        }
    }

    // Sort by estimated improvement (best first)
    candidates.sort_by(|a, b| {
        b.estimated_improvement
            .partial_cmp(&a.estimated_improvement)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    candidates
}

/// Analyse a single neuron's activation distribution for bounded range patterns.
fn analyse_neuron_range(
    uuid: &str,
    squash: &str,
    records: &[DiscoverRecord],
) -> Option<BoundedRangeCandidate> {
    let activations: Vec<f32> = records.iter().map(|r| r.activation).collect();
    let n = activations.len() as f32;

    // Compute basic statistics
    let mean: f32 = activations.iter().sum::<f32>() / n;
    let variance: f32 = activations
        .iter()
        .map(|a| (a - mean) * (a - mean))
        .sum::<f32>()
        / n;
    let std_dev = variance.sqrt();

    // Skip constant neurons (no variance means dead/stuck, handled elsewhere)
    if std_dev < MIN_STD_DEV {
        return None;
    }

    // Find the overall span
    let min_val = activations.iter().copied().fold(f32::INFINITY, f32::min);
    let max_val = activations
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let total_span = max_val - min_val;

    if total_span < MIN_STD_DEV {
        return None;
    }

    // Identify sentinel cluster: values that are far from the mean, clustered
    // at the extremes. We look for a gap in the distribution.
    let sentinel_threshold = SENTINEL_DISTANCE_THRESHOLD * std_dev;

    // Sort activations to find natural gaps
    let mut sorted = activations.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Find the largest gap in the sorted activations
    let (gap_idx, gap_size) = find_largest_gap(&sorted);

    // The gap must be significant relative to the total span
    if gap_size < sentinel_threshold || gap_size < total_span * 0.15 {
        return None;
    }

    // Determine which side is the "sentinel" cluster (smaller group)
    let lower_count = gap_idx + 1;
    let upper_count = sorted.len() - lower_count;

    let (sentinel_count, active_range_low, active_range_high) = if lower_count <= upper_count {
        // Lower cluster is smaller → it's the sentinel cluster
        let active_low = sorted[gap_idx + 1];
        let active_high = sorted[sorted.len() - 1];
        (lower_count, active_low, active_high)
    } else {
        // Upper cluster is smaller → it's the sentinel cluster
        let active_low = sorted[0];
        let active_high = sorted[gap_idx];
        (upper_count, active_low, active_high)
    };

    let sentinel_fraction = sentinel_count as f32 / n;

    // Check thresholds
    if sentinel_fraction < MIN_SENTINEL_FRACTION {
        return None;
    }

    let active_range = active_range_high - active_range_low;
    let active_range_ratio = active_range / total_span;

    if active_range_ratio > MAX_ACTIVE_RANGE_RATIO {
        return None;
    }

    // Estimated improvement: proportional to how much of the signal is wasted
    // on sentinel values. More sentinels and narrower active range = more to gain.
    let estimated_improvement = sentinel_fraction * (1.0 - active_range_ratio) * 0.01;

    Some(BoundedRangeCandidate {
        neuron_uuid: uuid.to_string(),
        current_squash: squash.to_string(),
        active_range_low,
        active_range_high,
        sentinel_fraction,
        estimated_improvement,
    })
}

/// Find the largest gap in a sorted slice of floats.
/// Returns `(index, gap_size)` where the gap is between `sorted[index]` and `sorted[index+1]`.
fn find_largest_gap(sorted: &[f32]) -> (usize, f32) {
    let mut best_idx = 0;
    let mut best_gap = 0.0_f32;

    for i in 0..sorted.len().saturating_sub(1) {
        let gap = sorted[i + 1] - sorted[i];
        if gap > best_gap {
            best_gap = gap;
            best_idx = i;
        }
    }

    (best_idx, best_gap)
}

/// Convert bounded range candidates into coordinated structural candidates.
///
/// Each bounded range neuron produces a `SetBias` candidate that shifts the
/// neuron's operating point so sentinel values map to a neutral output region.
pub fn bounded_range_to_coordinated_candidates(
    candidates: &[BoundedRangeCandidate],
) -> Vec<CoordinatedStructuralCandidateJson> {
    let mut results = Vec::with_capacity(candidates.len());

    for c in candidates {
        // Compute a bias adjustment that centres the active range around 0.
        // The idea: shift so the midpoint of the active range maps to 0,
        // which pushes the sentinel values further into a "don't care" zone.
        let active_midpoint = (c.active_range_low + c.active_range_high) / 2.0;
        let bias_delta = -active_midpoint;

        results.push(CoordinatedStructuralCandidateJson {
            operations: vec![CoordinatedStructuralOpJson::SetBias {
                neuron_uuid: c.neuron_uuid.clone(),
                bias: bias_delta,
            }],
            expected_creature_score_gain: c.estimated_improvement,
            comment: Some(format!(
                "Bounded range neuron {}: {} (active range {:.3}..{:.3}, {:.0}% sentinel) \
                 → adjust bias by {:.3} to centre active range",
                c.neuron_uuid,
                c.current_squash,
                c.active_range_low,
                c.active_range_high,
                c.sentinel_fraction * 100.0,
                bias_delta
            )),
        });
    }

    // Sort by expected improvement (best first)
    results.sort_by(|a, b| {
        b.expected_creature_score_gain
            .partial_cmp(&a.expected_creature_score_gain)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    results
}
