//! Individual neuron ranking score computation.
//!
//! Contains the `RankedNeuron` type and functions for computing error, activation,
//! frequency, and variance metrics from discovery records.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use super::super::gradient::GradientFlowStats;
use crate::types::DiscoverRecord;

use std::collections::HashMap;

/// Statistics for selection-based neurons (MINIMUM, MAXIMUM, IF).
/// Maps (`from_uuid`, `to_uuid`) -> win probability (0.0 to 1.0).
/// For MIN/MAX: probability this synapse provides the min/max value.
/// For IF: probability this synapse's branch is taken (condition always 1.0).
pub type SelectionStats = HashMap<(String, String), f32>;

#[derive(Debug)]
pub struct RankedNeuron {
    pub neuron_uuid: String,
    pub total_error: f32,
    /// Unclamped mean absolute error from recorded samples.
    ///
    /// We clamp `total_error` for focus ranking so hidden neurons with extreme raw errors do not
    /// dominate selection purely due to scale differences. However, when proposing exploratory
    /// ablation candidates we still want access to the true magnitude.
    pub raw_error: f32,
    /// Structural impact based on weight paths to output
    pub impact: f32,
    /// Mean absolute activation value from recorded samples
    pub mean_activation: f32,
    /// Activation-weighted impact = `structural_impact` × `mean_activation`
    /// This reflects the actual contribution the neuron makes during inference
    pub activation_weighted_impact: f32,
    /// Issue #206: Gradient flow statistics for this neuron.
    ///
    /// These metrics help identify neurons with high learning potential:
    /// - `avg_gradient_magnitude`: How much error signal can flow through
    /// - `saturation_ratio`: % of samples in saturated activation region
    /// - `dead_ratio`: % of samples with zero gradient (`ReLU` dead zones)
    pub gradient_flow: GradientFlowStats,
    /// Issue #204: Activation frequency (proportion of samples where neuron fires).
    ///
    /// Calculated as: `count_nonzero_activations` / `total_samples`
    /// where "fires" means |activation| > small threshold (avoiding floating point issues).
    ///
    /// This helps identify neurons with extreme firing patterns:
    /// - `activation_frequency` < 0.1: Rarely fires, limited influence on most samples
    /// - `activation_frequency` > 0.9: Always fires, behaves like a constant (no discriminative power)
    /// - 0.1 <= `activation_frequency` <= 0.9: "Sweet spot" with good discriminative power
    pub activation_frequency: f32,
}

pub(super) fn average_absolute_error_from_records(records: &[DiscoverRecord]) -> f32 {
    let mut sum = 0.0f32;
    let mut count: u32 = 0;

    for record in records {
        for err in &record.errors {
            if err.is_finite() {
                sum += err.abs();
                count += 1;
            }
        }
    }

    if count == 0 { 0.0 } else { sum / count as f32 }
}

/// Margin-weighted average absolute error (Issue #1318).
///
/// When `obs_weights` is provided, each record's contribution is multiplied by
/// the per-observation weight before averaging. Observations absent from the
/// map default to weight `1.0` (i.e. they contribute as in the unweighted mean).
///
/// When `obs_weights` is `None`, this is identical to
/// [`average_absolute_error_from_records`] — used by the regression guard so
/// callers without an `OneHot` / `Margin` descriptor get the legacy ranking.
pub(super) fn weighted_average_absolute_error_from_records(
    records: &[DiscoverRecord],
    obs_weights: Option<&HashMap<u32, f32>>,
) -> f32 {
    let Some(weights) = obs_weights else {
        return average_absolute_error_from_records(records);
    };

    let mut weighted_sum = 0.0f32;
    let mut weight_total = 0.0f32;

    for record in records {
        let w = weights.get(&record.obs_index).copied().unwrap_or(1.0);
        if !w.is_finite() || w <= 0.0 {
            continue;
        }
        for err in &record.errors {
            if err.is_finite() {
                weighted_sum += w * err.abs();
                weight_total += w;
            }
        }
    }

    if weight_total <= 0.0 {
        0.0
    } else {
        weighted_sum / weight_total
    }
}

/// Compute mean absolute activation from discovery records.
/// Sum of |activation| divided by number of finite records.
///
/// Non-finite values (NaN, Infinity) are filtered out to prevent
/// corruption of `activation_weighted_impact` calculations and sorting.
pub(super) fn mean_absolute_activation_from_records(records: &[DiscoverRecord]) -> f32 {
    if records.is_empty() {
        return 0.0;
    }

    let mut sum = 0.0f32;
    let mut count: u32 = 0;

    for record in records {
        if record.activation.is_finite() {
            sum += record.activation.abs();
            count += 1;
        }
    }

    if count == 0 { 0.0 } else { sum / count as f32 }
}

/// Issue #204: Compute activation frequency from discovery records.
///
/// Activation frequency = `count_nonzero_activations` / `total_samples`
/// where "fires" means |activation| > threshold to avoid floating-point issues.
///
/// # Arguments
/// * `records` - Discovery records for a single neuron
///
/// # Returns
/// A value in [0.0, 1.0] representing the proportion of samples where the neuron fires.
/// Returns 0.0 if there are no valid (finite) records.
///
/// # Rationale
/// - Neurons that rarely fire (frequency < 0.1) have limited influence on most samples
/// - Neurons that always fire (frequency > 0.9) behave like constants with no discriminative power
/// - Neurons with moderate frequency (0.1 to 0.9) are in the "sweet spot" for discovery
const ACTIVATION_FIRING_THRESHOLD: f32 = 1e-6;

pub(super) fn activation_frequency_from_records(records: &[DiscoverRecord]) -> f32 {
    if records.is_empty() {
        return 0.0;
    }

    let mut firing_count: u32 = 0;
    let mut total_count: u32 = 0;

    for record in records {
        if record.activation.is_finite() {
            total_count += 1;
            // A neuron "fires" when its absolute activation exceeds the threshold
            if record.activation.abs() > ACTIVATION_FIRING_THRESHOLD {
                firing_count += 1;
            }
        }
    }

    if total_count == 0 {
        0.0
    } else {
        firing_count as f32 / total_count as f32
    }
}

/// Issue #204: Compute frequency factor for focus neuron ranking.
///
/// Neurons with extreme activation frequencies (rarely or always firing) are
/// less useful for discovery analysis:
/// - Rarely-firing neurons (< 10%) have limited influence on most samples
/// - Always-firing neurons (> 90%) behave like constants with no discriminative power
///
/// # Arguments
/// * `activation_frequency` - The proportion of samples where the neuron fires [0.0, 1.0]
///
/// # Returns
/// * 0.8 if `activation_frequency` < 0.1 (rarely fires) - 20% penalty
/// * 0.8 if `activation_frequency` > 0.9 (always fires) - 20% penalty
/// * 1.0 otherwise (moderate frequency) - no penalty
const FREQUENCY_LOW_THRESHOLD: f32 = 0.1;
const FREQUENCY_HIGH_THRESHOLD: f32 = 0.9;
const FREQUENCY_PENALTY_FACTOR: f32 = 0.8;

pub(super) fn compute_frequency_factor(activation_frequency: f32) -> f32 {
    if (FREQUENCY_LOW_THRESHOLD..=FREQUENCY_HIGH_THRESHOLD).contains(&activation_frequency) {
        1.0 // No penalty for moderate frequency (10-90%)
    } else {
        FREQUENCY_PENALTY_FACTOR // 0.8x penalty for extreme frequencies
    }
}

/// Compute activation variance and mean from discovery records.
///
/// Issue #306: Used to detect constant-value neurons for removal with bias adjustments.
/// A neuron with near-zero variance has constant activation and can be removed,
/// with its effect folded into bias adjustments for downstream neurons.
///
/// # Returns
/// A tuple of (`mean_activation`, variance) where:
/// - `mean_activation` is the arithmetic mean (NOT absolute value)
/// - `variance` is the statistical variance of activations
///
/// Returns (0.0, 0.0) if there are insufficient records.
pub(super) fn activation_mean_and_variance_from_records(records: &[DiscoverRecord]) -> (f32, f32) {
    if records.len() < 2 {
        return (0.0, 0.0);
    }

    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut count = 0u32;

    for record in records {
        if record.activation.is_finite() {
            let a = record.activation as f64;
            sum += a;
            sum_sq += a * a;
            count += 1;
        }
    }

    if count < 2 {
        return (0.0, 0.0);
    }

    let n = count as f64;
    let mean = sum / n;
    let variance = (sum_sq / n) - (mean * mean);

    (mean as f32, variance.max(0.0) as f32)
}
