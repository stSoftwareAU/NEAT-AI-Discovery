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
//! 4. If the sentinel cluster has lower error variance *and* is clear of the useful
//!    range by at least `MIN_SENTINEL_GAP`, the sentinel values do not meaningfully
//!    influence the output. That decision is shared with `sentinel_gating` (Issue #400)
//!    and defined once in
//!    [`sentinel_cluster::assess_sentinel_cluster`](super::sentinel_cluster::assess_sentinel_cluster)
//!    (Issue #2042).
//! 5. Compute `effective_min`, `effective_max` from the non-sentinel values.
//! 6. Compute `utilisation_ratio` as the fraction of the full observed range that is effective.
//!
//! ## Differences from `bounded_range.rs`
//!
//! `bounded_range.rs` detects boundary clustering and recommends gating neurons.
//! This module focuses on **characterising** the effective range and sentinel values
//! using error correlation analysis, providing metadata for downstream use.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use std::collections::HashSet;

use crate::CreatureJson;
use crate::types::DiscoverRecord;

// Constants moved to constants.rs (Issue #424)
use crate::analysis::constants::{
    CANDIDATE_SENTINELS, MIN_DISCOVERY_SAMPLE_COUNT as MIN_SAMPLES_FOR_RANGE,
};
// Shared sentinel-cluster rule (Issue #2042)
use crate::analysis::detection::sentinel_cluster::assess_sentinel_cluster;

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
    neuron_records: &[(String, impl AsRef<[DiscoverRecord]>)],
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
        let records = records.as_ref();
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
        // The accept/reject rule lives in one place (Issue #2042): a cluster is a
        // sentinel only when it is dense, clearly separated from the useful range,
        // and carries lower error variance than that range.
        if let Some(cluster) = assess_sentinel_cluster(&activations, &errors, sentinel) {
            detected_sentinels.push(cluster.sentinel_value);
            sentinel_indices.extend(cluster.sentinel_indices);
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
