//! Range-aware weight normalisation and scaling (Issue #402).
//!
//! This module handles sentinel-aware weight computation, filtering out samples
//! at detected sentinel values before computing optimal weights.

use crate::analysis::observation_range::ObservationRangeResult;
use crate::analysis::samples::HelpfulSample;

use super::calculation::calculate_optimal_outgoing_weight;

/// Compute `sum_error_activation` and `sum_activation_sq` after excluding samples
/// whose source activation is at a sentinel value.
///
/// This is the pre-filter step described in Issue #402. It accepts observation range
/// metadata from `detect_observation_ranges()` (Issue #398) and removes sentinel
/// samples before accumulating the sums that feed into
/// `calculate_optimal_outgoing_weight()`.
///
/// # Arguments
/// * `samples` - The full set of helpful samples (source activation + target error).
/// * `range` - Observation range metadata identifying sentinel values and effective range.
/// * `sentinel_tolerance` - Tolerance for matching activations to sentinel values.
///
/// # Returns
/// A tuple of `(sum_error_activation, sum_activation_sq, effective_count)` computed
/// only from samples that are **not** at a sentinel value.
pub fn compute_range_aware_sums(
    samples: &[HelpfulSample],
    range: &ObservationRangeResult,
    sentinel_tolerance: f32,
) -> (f32, f32, usize) {
    let mut sum_error_activation: f32 = 0.0;
    let mut sum_activation_sq: f32 = 0.0;
    let mut count: usize = 0;

    for sample in samples {
        if !sample.activation.is_finite() || !sample.avg_error.is_finite() {
            continue;
        }

        // Check whether this sample is at a sentinel value
        let is_sentinel = range
            .sentinel_values
            .iter()
            .any(|&sv| (sample.activation - sv).abs() <= sentinel_tolerance);

        if is_sentinel {
            continue;
        }

        sum_error_activation += sample.avg_error * sample.activation;
        sum_activation_sq += sample.activation * sample.activation;
        count += 1;
    }

    (sum_error_activation, sum_activation_sq, count)
}

/// Compute an optimal outgoing weight after filtering out sentinel samples.
///
/// This is a wrapper around `calculate_optimal_outgoing_weight()` that first
/// excludes samples where the source observation is at a detected sentinel value.
/// The core weight computation function is not modified (DRY principle).
///
/// # Arguments
/// * `samples` - The full set of helpful samples.
/// * `range` - Observation range metadata from `detect_observation_ranges()` (#398).
/// * `incoming_weight` - For neurons: the incoming weight; for synapses: use 1.0.
/// * `sentinel_tolerance` - Tolerance for matching activations to sentinel values.
///
/// # Returns
/// * `Some(weight)` - Optimal weight computed from non-sentinel samples only.
/// * `None` - If insufficient non-sentinel samples or weight cannot be computed.
pub fn calculate_range_aware_weight(
    samples: &[HelpfulSample],
    range: &ObservationRangeResult,
    incoming_weight: f32,
    sentinel_tolerance: f32,
) -> Option<f32> {
    let (sum_ea, sum_aa, _count) = compute_range_aware_sums(samples, range, sentinel_tolerance);

    calculate_optimal_outgoing_weight(sum_ea, sum_aa, incoming_weight)
}
