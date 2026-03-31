//! Detection of individually successful candidates from recorded data.
//!
//! Scans recorded neuron activations and errors to identify source → target
//! synapse additions that individually reduce the target's prediction error.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for neural network computation (Issue #873)
#![allow(clippy::cast_possible_truncation)] // f64→f32 regression results are intentionally truncated

use std::collections::{HashMap, HashSet};

use crate::CreatureJson;
use crate::analysis::constants::MIN_DISCOVERY_SAMPLE_COUNT;
use crate::types::DiscoverRecord;

use super::IndividualCandidate;

/// Minimum fraction of target error variance a candidate must explain to be
/// considered individually successful.
const MIN_INDIVIDUAL_IMPROVEMENT: f32 = 0.01;

/// Maximum number of individually successful candidates to return.
const MAX_INDIVIDUAL_CANDIDATES: usize = 50;

/// Detect individually successful source → target synapse candidates.
///
/// For each target neuron (output/hidden) with error records, evaluates
/// potential source neurons and identifies those that individually reduce
/// the target's prediction error above `MIN_INDIVIDUAL_IMPROVEMENT`.
///
/// # Arguments
/// * `creature` — Network topology.
/// * `neuron_records` — Recorded activations and errors per neuron.
///
/// # Returns
/// Individually successful candidates sorted by improvement (best first).
pub fn detect_individually_successful(
    creature: &CreatureJson,
    neuron_records: &[(String, Vec<DiscoverRecord>)],
) -> Vec<IndividualCandidate> {
    if neuron_records.is_empty() {
        return Vec::new();
    }

    // Build record lookup by neuron UUID.
    let record_map: HashMap<&str, &Vec<DiscoverRecord>> = neuron_records
        .iter()
        .map(|(uuid, recs)| (uuid.as_str(), recs))
        .collect();

    // Build existing synapse set to avoid duplicating existing connections.
    let existing_synapses: HashSet<(&str, &str)> = creature
        .synapses
        .iter()
        .map(|s| (s.from_uuid.as_str(), s.to_uuid.as_str()))
        .collect();

    // Identify target neurons (output/hidden) with sufficient error records.
    let targets: Vec<&str> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "output" || n.neuron_type == "hidden")
        .filter(|n| {
            record_map.get(n.uuid.as_str()).is_some_and(|recs| {
                recs.len() >= MIN_DISCOVERY_SAMPLE_COUNT
                    && recs.iter().any(|r| !r.errors.is_empty())
            })
        })
        .map(|n| n.uuid.as_str())
        .collect();

    // Identify source neurons (input/hidden).
    let sources: Vec<&str> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "input" || n.neuron_type == "hidden")
        .filter(|n| {
            record_map
                .get(n.uuid.as_str())
                .is_some_and(|recs| recs.len() >= MIN_DISCOVERY_SAMPLE_COUNT)
        })
        .map(|n| n.uuid.as_str())
        .collect();

    let mut candidates = Vec::new();

    for &target_uuid in &targets {
        let target_records = match record_map.get(target_uuid) {
            Some(r) => *r,
            None => continue,
        };

        // Build target error map: obs_index → first error value.
        let target_errors: HashMap<u32, f32> = target_records
            .iter()
            .filter(|r| !r.errors.is_empty())
            .map(|r| (r.obs_index, r.errors[0]))
            .collect();

        if target_errors.len() < MIN_DISCOVERY_SAMPLE_COUNT {
            continue;
        }

        for &source_uuid in &sources {
            // Skip if synapse already exists.
            if existing_synapses.contains(&(source_uuid, target_uuid)) {
                continue;
            }
            // Skip self-connections.
            if source_uuid == target_uuid {
                continue;
            }

            let source_records = match record_map.get(source_uuid) {
                Some(r) => *r,
                None => continue,
            };

            if let Some(candidate) =
                evaluate_individual(source_uuid, source_records, target_uuid, &target_errors)
            {
                candidates.push(candidate);
            }
        }
    }

    // Sort by improvement (best first) and truncate.
    candidates.sort_by(|a, b| b.improvement.total_cmp(&a.improvement));
    candidates.truncate(MAX_INDIVIDUAL_CANDIDATES);

    candidates
}

/// Evaluate a single source → target pair for individual improvement.
fn evaluate_individual(
    source_uuid: &str,
    source_records: &[DiscoverRecord],
    target_uuid: &str,
    target_errors: &HashMap<u32, f32>,
) -> Option<IndividualCandidate> {
    // Build source activation map.
    let source_activations: HashMap<u32, f32> = source_records
        .iter()
        .map(|r| (r.obs_index, r.activation))
        .collect();

    // Find shared observation indices.
    let shared_indices: Vec<u32> = target_errors
        .keys()
        .copied()
        .filter(|idx| source_activations.contains_key(idx))
        .collect();

    if shared_indices.len() < MIN_DISCOVERY_SAMPLE_COUNT {
        return None;
    }

    let activations: Vec<f32> = shared_indices
        .iter()
        .map(|idx| source_activations[idx])
        .collect();
    let errors: Vec<f32> = shared_indices
        .iter()
        .map(|idx| target_errors[idx])
        .collect();

    // Least-squares weight: w = Σ(act × err) / Σ(act²).
    let sum_act_sq: f64 = activations
        .iter()
        .map(|a| f64::from(*a) * f64::from(*a))
        .sum();
    if sum_act_sq < 1e-10 {
        return None;
    }

    let sum_act_err: f64 = activations
        .iter()
        .zip(errors.iter())
        .map(|(a, e)| f64::from(*a) * f64::from(*e))
        .sum();

    let weight = (sum_act_err / sum_act_sq) as f32;

    // Compute error reduction: improvement = 1 - (residual_sse / original_sse).
    let original_sse: f64 = errors.iter().map(|e| f64::from(*e) * f64::from(*e)).sum();
    if original_sse < 1e-10 {
        return None;
    }

    let residual_sse: f64 = activations
        .iter()
        .zip(errors.iter())
        .map(|(a, e)| {
            let residual = f64::from(*e) - f64::from(weight) * f64::from(*a);
            residual * residual
        })
        .sum();

    let improvement = (1.0 - (residual_sse / original_sse)) as f32;

    if !improvement.is_finite() || improvement < MIN_INDIVIDUAL_IMPROVEMENT {
        return None;
    }

    Some(IndividualCandidate {
        source_uuid: source_uuid.to_string(),
        target_uuid: target_uuid.to_string(),
        weight,
        improvement,
        sample_count: shared_indices.len(),
    })
}
