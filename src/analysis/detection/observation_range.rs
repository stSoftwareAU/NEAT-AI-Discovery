//! Observation effective range discovery module (Issue #398).
//!
//! Observations are finite numbers, often normalised to -1…1. Since observations lack
//! a concept of null, sentinel values like -1 or 0 are used instead. For example, a
//! lower "Debt-to-Equity" value might influence the output, but -1 (meaning "no data")
//! should not.
//!
//! This module detects the "effective range" of each observation (input neuron) by
//! analysing recorded samples. It identifies sentinel clusters (values with low error
//! variance, indicating no meaningful correlation) and computes the effective range
//! where the observation actually influences the output.
//!
//! ## Detection Approach
//!
//! 1. For each observation, collect all activation values and their corresponding errors.
//! 2. Check candidate sentinel values (-1, 0, +1) for density clusters.
//! 3. Compare error variance in the sentinel cluster vs the non-sentinel (useful) range.
//! 4. If the sentinel cluster has significantly lower error variance, it indicates
//!    the sentinel values do not meaningfully influence the output.
//! 5. Compute effective_min, effective_max from the non-sentinel values.
//! 6. Compute utilisation_ratio as the fraction of the full observed range that is effective.
//!
//! ## Differences from `bounded_range.rs`
//!
//! `bounded_range.rs` detects boundary clustering and recommends gating neurons.
//! This module focuses on **characterising** the effective range and sentinel values
//! using error correlation analysis, providing metadata for downstream use.

use std::collections::HashSet;

use crate::CreatureJson;
use crate::types::DiscoverRecord;

// Constants moved to constants.rs (Issue #424)
use crate::analysis::constants::{
    CANDIDATE_SENTINELS, MIN_DISCOVERY_SAMPLE_COUNT as MIN_SAMPLES_FOR_RANGE,
    MIN_SENTINEL_FRACTION, MIN_SENTINEL_GAP as MIN_GAP, SENTINEL_TOLERANCE,
};

/// Result of observation range analysis for a single input neuron.
#[derive(Debug, Clone)]
pub struct ObservationRangeResult {
    /// UUID of the observation (input neuron).
    pub neuron_uuid: String,
    /// Minimum of the effective (non-sentinel) range.
    pub effective_min: f32,
    /// Maximum of the effective (non-sentinel) range.
    pub effective_max: f32,
    /// Detected sentinel/null values (e.g., -1.0, 0.0).
    pub sentinel_values: Vec<f32>,
    /// Fraction of the full observed range that is effective (0.0 to 1.0).
    pub utilisation_ratio: f32,
    /// Number of samples analysed.
    pub sample_count: usize,
}

/// Detect observation effective ranges from recorded samples.
///
/// Analyses each input neuron's recorded activations to identify sentinel clusters
/// and compute the effective range where the observation actually correlates with
/// meaningful error changes.
///
/// # Arguments
/// * `creature` - The creature's network topology (used to identify input neurons).
/// * `neuron_records` - List of `(neuron_uuid, records)` tuples with recorded activations.
///
/// # Returns
/// A list of `ObservationRangeResult` for observations where sentinel values were detected,
/// sorted by neuron UUID for stable output.
pub fn detect_observation_ranges(
    creature: &CreatureJson,
    neuron_records: &[(String, Vec<DiscoverRecord>)],
) -> Vec<ObservationRangeResult> {
    // Only analyse input neurons (observations)
    let input_uuids: HashSet<&str> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "input")
        .map(|n| n.uuid.as_str())
        .collect();

    let mut results = Vec::with_capacity(neuron_records.len());

    for (uuid, records) in neuron_records {
        if !input_uuids.contains(uuid.as_str()) {
            continue;
        }

        if records.len() < MIN_SAMPLES_FOR_RANGE {
            continue;
        }

        if let Some(result) = analyse_observation_range(uuid, records) {
            results.push(result);
        }
    }

    // Sort by neuron UUID for stable output
    results.sort_by(|a, b| a.neuron_uuid.cmp(&b.neuron_uuid));

    results
}

/// Analyse a single observation's recorded samples to detect sentinel clusters
/// and compute the effective range.
fn analyse_observation_range(
    uuid: &str,
    records: &[DiscoverRecord],
) -> Option<ObservationRangeResult> {
    let activations: Vec<f32> = records.iter().map(|r| r.activation).collect();
    let errors: Vec<f32> = records
        .iter()
        .map(|r| {
            if r.errors.is_empty() {
                0.0
            } else {
                r.errors[0]
            }
        })
        .collect();

    let n = activations.len() as f32;

    // Compute overall activation range
    let overall_min = activations.iter().copied().fold(f32::INFINITY, f32::min);
    let overall_max = activations
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let overall_range = overall_max - overall_min;

    if overall_range < 1e-6 {
        // All values essentially identical — no range to analyse
        return None;
    }

    // Detect sentinel clusters
    let mut detected_sentinels: Vec<f32> = Vec::new();
    let mut sentinel_indices: HashSet<usize> = HashSet::new();

    for &sentinel in &CANDIDATE_SENTINELS {
        // Count and collect indices of samples near this sentinel
        let cluster_indices: Vec<usize> = activations
            .iter()
            .enumerate()
            .filter(|&(_, &a)| (a - sentinel).abs() <= SENTINEL_TOLERANCE)
            .map(|(i, _)| i)
            .collect();

        let cluster_fraction = cluster_indices.len() as f32 / n;

        if cluster_fraction < MIN_SENTINEL_FRACTION {
            continue;
        }

        // Compute error variance for the sentinel cluster
        let sentinel_error_var = compute_error_variance(&errors, &cluster_indices);

        // Compute error variance for non-sentinel samples
        let non_sentinel_indices: Vec<usize> = (0..activations.len())
            .filter(|i| !cluster_indices.contains(i))
            .collect();

        if non_sentinel_indices.is_empty() {
            continue;
        }

        let non_sentinel_error_var = compute_error_variance(&errors, &non_sentinel_indices);

        // Check that sentinel cluster has lower error variance (less correlation)
        // or that there is a sufficient gap between sentinel and useful range
        let useful_values: Vec<f32> = non_sentinel_indices
            .iter()
            .map(|&i| activations[i])
            .collect();

        let useful_min = useful_values.iter().copied().fold(f32::INFINITY, f32::min);
        let useful_max = useful_values
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);

        // Check gap between sentinel and useful range
        let gap = if sentinel <= useful_min {
            useful_min - (sentinel + SENTINEL_TOLERANCE)
        } else if sentinel >= useful_max {
            (sentinel - SENTINEL_TOLERANCE) - useful_max
        } else {
            // Sentinel is inside the useful range — not a clear boundary
            0.0
        };

        if gap < MIN_GAP {
            continue;
        }

        // Accept if sentinel has lower error variance OR clear gap exists
        let is_sentinel = sentinel_error_var < non_sentinel_error_var || gap >= MIN_GAP;

        if is_sentinel {
            detected_sentinels.push(sentinel);
            sentinel_indices.extend(cluster_indices);
        }
    }

    if detected_sentinels.is_empty() {
        return None;
    }

    // Compute effective range from non-sentinel values
    let effective_activations: Vec<f32> = activations
        .iter()
        .enumerate()
        .filter(|(i, _)| !sentinel_indices.contains(i))
        .map(|(_, &a)| a)
        .collect();

    if effective_activations.is_empty() {
        return None;
    }

    let effective_min = effective_activations
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    let effective_max = effective_activations
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);

    let effective_range = effective_max - effective_min;
    let utilisation_ratio = if overall_range > 1e-6 {
        (effective_range / overall_range).clamp(0.0, 1.0)
    } else {
        1.0
    };

    Some(ObservationRangeResult {
        neuron_uuid: uuid.to_string(),
        effective_min,
        effective_max,
        sentinel_values: detected_sentinels,
        utilisation_ratio,
        sample_count: records.len(),
    })
}

/// Compute the variance of error values at the given indices.
fn compute_error_variance(errors: &[f32], indices: &[usize]) -> f32 {
    if indices.is_empty() {
        return 0.0;
    }

    let n = indices.len() as f32;
    let sum: f32 = indices.iter().map(|&i| errors[i]).sum();
    let mean = sum / n;

    let variance: f32 = indices
        .iter()
        .map(|&i| (errors[i] - mean).powi(2))
        .sum::<f32>()
        / n;

    variance
}
