//! Multi-hop candidate analysis module (Issue #230).
//!
//! Current discovery only considers single-hop improvements (adding one synapse or neuron).
//! For deep networks, multi-hop improvements (adding a path of 2-3 connections) may be more
//! effective. This module analyses intermediate neurons to find deeper structural improvements.
//!
//! See `docs/DISCOVERY_TYPES.md` § "Multi-Hop Candidate Analysis" for full documentation.
//!
//! ## Detection Method
//!
//! 1. **Identify target neurons with errors**: Focus on output and hidden neurons that have
//!    recorded errors.
//! 2. **Find correlated intermediates**: For each target, find neurons whose activation
//!    correlates with the target's error but are not directly connected.
//! 3. **Build multi-hop paths**: Chain source → intermediate(s) → target paths where each
//!    hop has a correlation-based improvement estimate.
//! 4. **Prune aggressively**: Limit depth to 3 hops max, filter by correlation threshold,
//!    and skip already-connected pairs.
//!
//! ## Recommended Actions
//!
//! When multi-hop opportunities are detected, we recommend:
//! 1. **Add bypass synapse**: Connect the best source directly to the target.
//! 2. **Add relay neuron**: Insert a new hidden neuron along the path to relay information.
//!
//! These are emitted as `CoordinatedStructuralCandidateJson` with `AddNeuron` and/or
//! `AddSynapse` operations.

use std::collections::{HashMap, HashSet};

use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson};

/// Maximum path length (number of nodes) for multi-hop analysis.
/// A path of 4 nodes = 3 hops. Deeper paths have diminishing returns and exponential cost.
const MAX_PATH_LENGTH: usize = 4;

/// Minimum samples required for reliable correlation estimation.
const MIN_SAMPLES_FOR_CORRELATION: usize = 20;

/// Minimum absolute Pearson correlation between a neuron's activation and a target's
/// error to consider the neuron as a useful intermediate.
const CORRELATION_THRESHOLD: f32 = 0.3;

/// Maximum number of intermediate candidates to consider per target neuron.
/// Controls combinatorial explosion.
const MAX_INTERMEDIATES_PER_TARGET: usize = 10;

/// Maximum total candidates returned to limit computational cost.
const MAX_TOTAL_CANDIDATES: usize = 50;

/// A multi-hop candidate describing a path of neurons that could improve the target.
#[derive(Debug, Clone)]
pub struct MultiHopCandidate {
    /// Ordered path of neuron UUIDs: [source, intermediate1, ..., target].
    pub path: Vec<String>,
    /// Estimated creature score improvement from adding this path.
    pub estimated_improvement: f32,
    /// Mean correlation strength along the path.
    pub correlation_strength: f32,
}

/// Detect multi-hop candidates by finding neurons whose activations correlate with
/// target errors but are not directly connected.
///
/// # Arguments
/// * `creature` - The creature's network topology (neurons and synapses).
/// * `neuron_records` - List of `(neuron_uuid, records)` tuples with recorded
///   activations and errors.
///
/// # Returns
/// A list of `MultiHopCandidate` sorted by estimated improvement (best first).
pub fn detect_multi_hop_candidates(
    creature: &CreatureJson,
    neuron_records: &[(String, Vec<DiscoverRecord>)],
) -> Vec<MultiHopCandidate> {
    if neuron_records.is_empty() {
        return Vec::new();
    }

    // Build topology: existing synapse connections
    let existing_synapses: HashSet<(&str, &str)> = creature
        .synapses
        .iter()
        .map(|s| (s.from_uuid.as_str(), s.to_uuid.as_str()))
        .collect();

    // Classify neurons by type
    let output_uuids: HashSet<&str> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "output")
        .map(|n| n.uuid.as_str())
        .collect();

    let input_uuids: HashSet<&str> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "input")
        .map(|n| n.uuid.as_str())
        .collect();

    let hidden_uuids: HashSet<&str> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "hidden")
        .map(|n| n.uuid.as_str())
        .collect();

    // Build per-neuron activation-by-obs and error-by-obs maps
    let mut activation_by_obs: HashMap<&str, HashMap<u32, f32>> = HashMap::new();
    let mut error_by_obs: HashMap<&str, HashMap<u32, f32>> = HashMap::new();

    for (uuid, records) in neuron_records {
        let uuid_str = uuid.as_str();
        let mut act_map = HashMap::new();
        let mut err_map = HashMap::new();
        for r in records {
            act_map.insert(r.obs_index, r.activation);
            if let Some(&err) = r.errors.first() {
                err_map.insert(r.obs_index, err);
            }
        }
        activation_by_obs.insert(uuid_str, act_map);
        if !err_map.is_empty() {
            error_by_obs.insert(uuid_str, err_map);
        }
    }

    // Find target neurons: output neurons (and potentially hidden neurons) with error data
    let target_uuids: Vec<&str> = output_uuids
        .iter()
        .chain(hidden_uuids.iter())
        .filter(|&&uuid| {
            error_by_obs
                .get(uuid)
                .is_some_and(|e| e.len() >= MIN_SAMPLES_FOR_CORRELATION)
        })
        .copied()
        .collect();

    if target_uuids.is_empty() {
        return Vec::new();
    }

    // Neurons that can serve as intermediates or sources (input + hidden)
    let source_candidate_uuids: Vec<&str> = input_uuids
        .iter()
        .chain(hidden_uuids.iter())
        .filter(|&&uuid| {
            activation_by_obs
                .get(uuid)
                .is_some_and(|a| a.len() >= MIN_SAMPLES_FOR_CORRELATION)
        })
        .copied()
        .collect();

    let mut all_candidates: Vec<MultiHopCandidate> = Vec::new();

    for &target_uuid in &target_uuids {
        let target_errors = match error_by_obs.get(target_uuid) {
            Some(e) => e,
            None => continue,
        };

        // Find intermediates: neurons whose activation correlates with this target's error
        // but are NOT directly connected to the target.
        let mut intermediates: Vec<(&str, f32)> = Vec::new();

        for &source_uuid in &source_candidate_uuids {
            if source_uuid == target_uuid {
                continue;
            }

            // Skip if already directly connected to target
            if existing_synapses.contains(&(source_uuid, target_uuid)) {
                continue;
            }

            let source_activations = match activation_by_obs.get(source_uuid) {
                Some(a) => a,
                None => continue,
            };

            let corr = compute_activation_error_correlation(source_activations, target_errors);

            if corr.abs() >= CORRELATION_THRESHOLD {
                intermediates.push((source_uuid, corr));
            }
        }

        // Sort by absolute correlation (strongest first) and limit
        intermediates.sort_by(|a, b| {
            b.1.abs()
                .partial_cmp(&a.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        intermediates.truncate(MAX_INTERMEDIATES_PER_TARGET);

        // Build two-hop candidates: source → intermediate → target
        for &(intermediate_uuid, corr) in &intermediates {
            // For a two-hop path, find a source that connects to or correlates
            // with the intermediate
            let estimated_improvement = corr.abs() * compute_mean_abs_error(target_errors) * 0.01;

            if estimated_improvement <= 0.0 {
                continue;
            }

            // Simple two-hop: just the intermediate directly to target
            all_candidates.push(MultiHopCandidate {
                path: vec![intermediate_uuid.to_string(), target_uuid.to_string()],
                estimated_improvement,
                correlation_strength: corr.abs(),
            });

            // Try to extend to three-hop if path length allows
            if all_candidates.len() < MAX_TOTAL_CANDIDATES && MAX_PATH_LENGTH >= 3 {
                find_three_hop_extensions(
                    intermediate_uuid,
                    target_uuid,
                    corr,
                    &source_candidate_uuids,
                    &activation_by_obs,
                    &existing_synapses,
                    &output_uuids,
                    target_errors,
                    &mut all_candidates,
                );
            }
        }
    }

    // Sort by estimated improvement (best first)
    all_candidates.sort_by(|a, b| {
        b.estimated_improvement
            .partial_cmp(&a.estimated_improvement)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Limit total candidates
    all_candidates.truncate(MAX_TOTAL_CANDIDATES);

    all_candidates
}

/// Extend a two-hop candidate (intermediate → target) to a three-hop candidate
/// (source → intermediate → target) by finding sources whose activation correlates
/// with the intermediate's activation.
#[allow(clippy::too_many_arguments)]
fn find_three_hop_extensions(
    intermediate_uuid: &str,
    target_uuid: &str,
    intermediate_target_corr: f32,
    source_candidates: &[&str],
    activation_by_obs: &HashMap<&str, HashMap<u32, f32>>,
    existing_synapses: &HashSet<(&str, &str)>,
    output_uuids: &HashSet<&str>,
    target_errors: &HashMap<u32, f32>,
    candidates: &mut Vec<MultiHopCandidate>,
) {
    let intermediate_activations = match activation_by_obs.get(intermediate_uuid) {
        Some(a) => a,
        None => return,
    };

    for &source_uuid in source_candidates {
        if source_uuid == intermediate_uuid || source_uuid == target_uuid {
            continue;
        }

        // Skip output neurons as intermediates
        if output_uuids.contains(source_uuid) {
            continue;
        }

        // Skip if source already connected to intermediate
        if existing_synapses.contains(&(source_uuid, intermediate_uuid)) {
            continue;
        }

        // Skip if source already connected to target
        if existing_synapses.contains(&(source_uuid, target_uuid)) {
            continue;
        }

        let source_activations = match activation_by_obs.get(source_uuid) {
            Some(a) => a,
            None => continue,
        };

        // Check correlation between source activation and intermediate activation
        let source_intermediate_corr =
            compute_activation_activation_correlation(source_activations, intermediate_activations);

        if source_intermediate_corr.abs() < CORRELATION_THRESHOLD {
            continue;
        }

        // Combined correlation: geometric mean of the two correlations
        let combined_corr =
            (source_intermediate_corr.abs() * intermediate_target_corr.abs()).sqrt();

        let estimated_improvement = combined_corr * compute_mean_abs_error(target_errors) * 0.005;

        if estimated_improvement <= 0.0 {
            continue;
        }

        candidates.push(MultiHopCandidate {
            path: vec![
                source_uuid.to_string(),
                intermediate_uuid.to_string(),
                target_uuid.to_string(),
            ],
            estimated_improvement,
            correlation_strength: combined_corr,
        });

        if candidates.len() >= MAX_TOTAL_CANDIDATES {
            return;
        }
    }
}

/// Compute Pearson correlation between a neuron's activation and a target's error,
/// both indexed by obs_index.
fn compute_activation_error_correlation(
    activations: &HashMap<u32, f32>,
    errors: &HashMap<u32, f32>,
) -> f32 {
    let shared: Vec<u32> = activations
        .keys()
        .filter(|k| errors.contains_key(k))
        .copied()
        .collect();

    let n = shared.len();
    if n < MIN_SAMPLES_FOR_CORRELATION {
        return 0.0;
    }

    let n_f = n as f32;

    let mean_a: f32 = shared.iter().map(|k| activations[k]).sum::<f32>() / n_f;
    let mean_e: f32 = shared.iter().map(|k| errors[k]).sum::<f32>() / n_f;

    let mut cov = 0.0_f32;
    let mut var_a = 0.0_f32;
    let mut var_e = 0.0_f32;

    for &k in &shared {
        let da = activations[&k] - mean_a;
        let de = errors[&k] - mean_e;
        cov += da * de;
        var_a += da * da;
        var_e += de * de;
    }

    let denom = (var_a * var_e).sqrt();
    if denom < 1e-10 {
        return 0.0;
    }

    cov / denom
}

/// Compute Pearson correlation between two activation vectors indexed by obs_index.
fn compute_activation_activation_correlation(
    activations_a: &HashMap<u32, f32>,
    activations_b: &HashMap<u32, f32>,
) -> f32 {
    let shared: Vec<u32> = activations_a
        .keys()
        .filter(|k| activations_b.contains_key(k))
        .copied()
        .collect();

    let n = shared.len();
    if n < MIN_SAMPLES_FOR_CORRELATION {
        return 0.0;
    }

    let n_f = n as f32;

    let mean_a: f32 = shared.iter().map(|k| activations_a[k]).sum::<f32>() / n_f;
    let mean_b: f32 = shared.iter().map(|k| activations_b[k]).sum::<f32>() / n_f;

    let mut cov = 0.0_f32;
    let mut var_a = 0.0_f32;
    let mut var_b = 0.0_f32;

    for &k in &shared {
        let da = activations_a[&k] - mean_a;
        let db = activations_b[&k] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    let denom = (var_a * var_b).sqrt();
    if denom < 1e-10 {
        return 0.0;
    }

    cov / denom
}

/// Compute mean absolute error from an error-by-obs map.
fn compute_mean_abs_error(errors: &HashMap<u32, f32>) -> f32 {
    if errors.is_empty() {
        return 0.0;
    }

    let sum: f32 = errors.values().map(|e| e.abs()).sum();
    sum / errors.len() as f32
}

/// Generate a deterministic UUID for a multi-hop relay neuron.
fn multi_hop_relay_uuid(path: &[String], index: usize) -> String {
    let mut key = format!("multi-hop-relay|{index}");
    for uuid in path {
        key.push('|');
        key.push_str(uuid);
    }
    // FNV-1a 64-bit
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in key.as_bytes() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("mh-{hash:016x}")
}

/// Convert multi-hop candidates into coordinated structural candidates.
///
/// Each candidate produces a coordinated structural operation:
/// - **Two-hop** (source → target): Add a bypass synapse from source to target.
/// - **Three-hop** (source → intermediate → target): Add a relay neuron with
///   synapses from source to relay and relay to target.
///
/// # Arguments
/// * `candidates` - Detected multi-hop candidates.
/// * `creature` - The creature topology (needed for neuron lookup and placement).
pub fn multi_hop_to_coordinated_candidates(
    candidates: &[MultiHopCandidate],
    creature: &CreatureJson,
) -> Vec<CoordinatedStructuralCandidateJson> {
    // Find the first output neuron UUID for insert_before placement
    let first_output_uuid = creature
        .neurons
        .iter()
        .find(|n| n.neuron_type == "output")
        .map(|n| n.uuid.clone());

    let mut results = Vec::new();

    for (idx, candidate) in candidates.iter().enumerate() {
        let path_len = candidate.path.len();

        if path_len < 2 {
            continue;
        }

        if path_len == 2 {
            // Two-hop: add bypass synapse source → target
            let source = &candidate.path[0];
            let target = &candidate.path[1];

            // Compute weight based on correlation direction
            let weight = if candidate.correlation_strength > 0.0 {
                0.1
            } else {
                -0.1
            };

            results.push(CoordinatedStructuralCandidateJson {
                operations: vec![CoordinatedStructuralOpJson::AddSynapse {
                    from_neuron_uuid: source.clone(),
                    to_neuron_uuid: target.clone(),
                    weight,
                }],
                expected_creature_score_gain: candidate.estimated_improvement,
                comment: Some(format!(
                    "Multi-hop bypass: {} → {} (correlation {:.2})",
                    source, target, candidate.correlation_strength
                )),
            });
        } else {
            // Three-hop or more: add relay neuron with synapses
            let source = &candidate.path[0];
            let target = &candidate.path[path_len - 1];

            let relay_uuid = multi_hop_relay_uuid(&candidate.path, idx);

            // Determine insert position: before the target or before the first output
            let insert_before = if creature
                .neurons
                .iter()
                .any(|n| n.uuid == *target && n.neuron_type == "output")
            {
                Some(target.clone())
            } else {
                first_output_uuid.clone()
            };

            let operations = vec![
                // Add relay neuron
                CoordinatedStructuralOpJson::AddNeuron {
                    neuron_uuid: relay_uuid.clone(),
                    neuron_type: "hidden".to_string(),
                    squash: "TANH".to_string(),
                    bias: 0.0,
                    insert_before_neuron_uuid: insert_before,
                },
                // Add synapse from source to relay
                CoordinatedStructuralOpJson::AddSynapse {
                    from_neuron_uuid: source.clone(),
                    to_neuron_uuid: relay_uuid.clone(),
                    weight: 0.5,
                },
                // Add synapse from relay to target
                CoordinatedStructuralOpJson::AddSynapse {
                    from_neuron_uuid: relay_uuid,
                    to_neuron_uuid: target.clone(),
                    weight: 0.1,
                },
            ];

            let path_str = candidate.path.join(" → ");
            results.push(CoordinatedStructuralCandidateJson {
                operations,
                expected_creature_score_gain: candidate.estimated_improvement,
                comment: Some(format!(
                    "Multi-hop relay: {} (correlation {:.2})",
                    path_str, candidate.correlation_strength
                )),
            });
        }
    }

    // Sort by expected improvement (best first)
    results.sort_by(|a, b| {
        b.expected_creature_score_gain
            .partial_cmp(&a.expected_creature_score_gain)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    results
}

#[cfg(test)]
#[path = "multi_hop_tests.rs"]
mod tests;
