//! Dead neuron detection module (Issue #341).
//!
//! Identifies neurons that have become effectively dead (always outputting zero or
//! near-zero activation) and recommends their removal. Dead neurons waste computation
//! without contributing to the network's output.
//!
//! See `docs/DISCOVERY_TYPES.md` § "Dead Neuron Detection" for full documentation.
//!
//! ## Detection Criteria
//!
//! A neuron is "dead" if:
//! 1. **Near-zero activation**: Mean absolute activation < threshold (e.g., 1e-6) across
//!    all samples.
//! 2. **Zero variance**: Standard deviation of activation ≈ 0 (always outputs the same value).
//! 3. **Only hidden neurons**: Output and input neurons are excluded.
//!
//! ## Recommended Actions
//!
//! When a dead neuron is detected, we recommend:
//! 1. **Remove the neuron**: Emit a `RemoveNeuron` operation via `CoordinatedStructuralCandidateJson`.
//!
//! Dead neurons consume GPU resources during both training and inference without contributing
//! useful information, so removal is the primary recommendation.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use std::collections::HashSet;

use super::helpers::{
    ConfidenceFactor, build_record_map, compute_activation_stats, compute_mean_abs_activation,
    sort_candidates_by_score_gain, weighted_confidence,
};

use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson};

use super::topology_cache::CreatureTopologyCache;

// MIN_SAMPLES_FOR_DEAD_DETECTION moved to constants.rs (Issue #424)
use crate::analysis::constants::MIN_DISCOVERY_SAMPLE_COUNT as MIN_SAMPLES_FOR_DEAD_DETECTION;

/// Threshold for mean absolute activation to consider a neuron dead.
/// Activations below this are effectively zero.
const DEAD_ACTIVATION_THRESHOLD: f32 = 1e-6;

/// Maximum activation standard deviation for a dead neuron.
/// If output varies significantly, the neuron is not truly dead.
const MAX_DEAD_STD_DEV: f32 = 1e-6;

/// Minimum fraction of samples where a neuron must be active to avoid being
/// flagged as dead. A neuron active on even a small fraction of samples with
/// meaningful activation is not dead.
const MIN_ACTIVE_FRACTION: f32 = 0.01;

/// Activation magnitude threshold for considering a single sample "active".
const ACTIVE_SAMPLE_THRESHOLD: f32 = 0.01;

/// Result of detecting a dead neuron.
#[derive(Debug, Clone)]
pub struct DeadNeuronCandidate {
    /// UUID of the dead neuron.
    pub neuron_uuid: String,
    /// Mean absolute activation across all samples.
    pub mean_abs_activation: f32,
    /// Standard deviation of activation across samples.
    pub activation_std_dev: f32,
    /// Number of samples analysed.
    pub sample_count: usize,
    /// UUIDs of output neurons reachable from this neuron.
    pub connected_outputs: Vec<String>,
    /// Confidence that removing this neuron is safe (0.0 to 1.0).
    pub removal_confidence: f32,
    /// Estimated creature score improvement from removing this neuron.
    pub estimated_improvement: f32,
}

/// Detect dead neurons from the creature topology and recorded activations.
///
/// # Arguments
/// * `creature` - The creature's network topology (neurons and synapses).
/// * `neuron_records` - List of `(neuron_uuid, records)` tuples with recorded activations.
/// * `topo` - Optional pre-computed topology cache (Issue #754). When `None`,
///   topology maps are built locally (backward-compatible path).
///
/// # Returns
/// A list of `DeadNeuronCandidate` for neurons that are dead,
/// sorted by removal confidence (highest first).
pub fn detect_dead_neurons(
    creature: &CreatureJson,
    neuron_records: &[(String, Vec<DiscoverRecord>)],
    topo: Option<&CreatureTopologyCache>,
) -> Vec<DeadNeuronCandidate> {
    // Use pre-computed cache or build locally.
    let local_cache;
    let topo = match topo {
        Some(t) => t,
        None => {
            local_cache = CreatureTopologyCache::new(creature);
            &local_cache
        }
    };

    // Build records lookup
    let records_map = build_record_map(neuron_records);

    let mut candidates = Vec::with_capacity(topo.hidden_uuids.len());

    for uuid in &topo.hidden_uuids {
        let Some(records) = records_map.get(uuid.as_str()) else {
            continue;
        };

        if records.len() < MIN_SAMPLES_FOR_DEAD_DETECTION {
            continue;
        }

        let n = records.len() as f32;

        // Compute mean absolute activation (Issue #941: shared helper)
        let mean_abs_activation = compute_mean_abs_activation(records);

        // Compute activation standard deviation (Issue #941: shared helper)
        let activation_std_dev = compute_activation_stats(records).std_dev;

        // Check if neuron is dead: near-zero mean absolute activation AND low variance
        if mean_abs_activation >= DEAD_ACTIVATION_THRESHOLD {
            continue;
        }

        if activation_std_dev >= MAX_DEAD_STD_DEV {
            continue;
        }

        // Check active fraction: if the neuron fires meaningfully on even a small
        // fraction of samples, it is not dead (prevents false positives).
        let active_count = records
            .iter()
            .filter(|r| r.activation.abs() >= ACTIVE_SAMPLE_THRESHOLD)
            .count();
        let active_fraction = active_count as f32 / n;

        if active_fraction >= MIN_ACTIVE_FRACTION {
            continue;
        }

        // Find output neurons reachable from this neuron (BFS)
        let connected_outputs = find_connected_outputs_cached(uuid, topo);

        // Compute removal confidence based on how dead the neuron is
        let confidence = compute_removal_confidence(mean_abs_activation, activation_std_dev, n);

        // Estimated improvement: removing a dead neuron saves computation.
        // The improvement is small but positive (reduced overhead).
        let estimated_improvement = confidence * 0.001;

        candidates.push(DeadNeuronCandidate {
            neuron_uuid: uuid.clone(),
            mean_abs_activation,
            activation_std_dev,
            sample_count: records.len(),
            connected_outputs,
            removal_confidence: confidence,
            estimated_improvement,
        });
    }

    // Sort by removal confidence (highest first)
    candidates.sort_by(|a, b| b.removal_confidence.total_cmp(&a.removal_confidence));

    candidates
}

/// Find output neurons reachable from a given neuron via BFS using the topology cache.
///
/// Uses `HashSet<&str>` keyed on borrowed slices from the topology cache to
/// avoid allocating a new `String` at every BFS step (Issue #755).
fn find_connected_outputs_cached(start_uuid: &str, topo: &CreatureTopologyCache) -> Vec<String> {
    let neuron_count = topo.hidden_uuids.len() + topo.output_uuids.len();
    let mut visited: HashSet<&str> = HashSet::with_capacity(neuron_count);
    let mut queue: Vec<&str> = Vec::with_capacity(neuron_count);
    queue.push(start_uuid);
    let mut connected: Vec<String> = Vec::new();

    while let Some(current) = queue.pop() {
        if !visited.insert(current) {
            continue;
        }

        for next in topo.fan_out_for(current) {
            if topo.output_uuids.contains(next.as_str()) {
                connected.push(next.clone());
            }
            if !visited.contains(next.as_str()) {
                queue.push(next.as_str());
            }
        }
    }

    connected.sort();
    connected.dedup();
    connected
}

/// Compute removal confidence based on activation statistics.
///
/// Higher confidence when:
/// - Mean absolute activation is closer to zero
/// - Standard deviation is closer to zero
/// - More samples were analysed
///
/// Issue #941: refactored to use `weighted_confidence` shared helper.
fn compute_removal_confidence(
    mean_abs_activation: f32,
    activation_std_dev: f32,
    sample_count: f32,
) -> f32 {
    let activation_factor = 1.0 - (mean_abs_activation / DEAD_ACTIVATION_THRESHOLD).min(1.0);
    let variance_factor = 1.0 - (activation_std_dev / MAX_DEAD_STD_DEV).min(1.0);
    let sample_factor = (sample_count / 1000.0).min(1.0);

    weighted_confidence(
        &[
            ConfidenceFactor {
                value: activation_factor,
                weight: 0.4,
            },
            ConfidenceFactor {
                value: variance_factor,
                weight: 0.4,
            },
            ConfidenceFactor {
                value: sample_factor,
                weight: 0.2,
            },
        ],
        0.5,
        1.0,
    )
}

/// Convert dead neuron candidates into coordinated structural candidates.
///
/// Each dead neuron produces a `RemoveNeuron` coordinated candidate.
/// The NEAT-AI controller will validate the removal through ablation testing
/// before actually applying it.
pub fn dead_neurons_to_coordinated_candidates(
    candidates: &[DeadNeuronCandidate],
) -> Vec<CoordinatedStructuralCandidateJson> {
    let mut results = Vec::with_capacity(candidates.len());

    for c in candidates {
        results.push(CoordinatedStructuralCandidateJson {
            operations: vec![CoordinatedStructuralOpJson::RemoveNeuron {
                neuron_uuid: c.neuron_uuid.clone(),
            }],
            expected_creature_score_gain: c.estimated_improvement,
            comment: Some(format!(
                "Dead neuron {}: mean abs activation {:.2e}, std dev {:.2e}, {} samples → remove to reduce wasted computation",
                c.neuron_uuid, c.mean_abs_activation, c.activation_std_dev, c.sample_count
            )),
        });
    }

    // Sort by expected improvement (best first) (Issue #941: shared helper)
    sort_candidates_by_score_gain(&mut results);

    results
}
