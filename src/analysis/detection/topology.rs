//! Topology-aware discovery module (Issue #422).
//!
//! Analyses network structure to identify structural improvements based on
//! path lengths, connectivity balance, and fan-in/fan-out optimisation.
//!
//! See `docs/DISCOVERY_TYPES.md` § "Topology-Aware Structure Analysis" for full documentation.
//!
//! ## Detection Criteria
//!
//! 1. **Long path**: A hidden neuron whose shortest path to any output exceeds
//!    `MAX_EFFICIENT_PATH_LENGTH`. Suggests adding a skip connection to shorten
//!    the path and reduce gradient attenuation.
//! 2. **Connectivity imbalance**: Hidden neurons in the same network have
//!    significantly different fan-in counts, indicating uneven information flow.
//!    Suggests rebalancing by adding synapses to starved neurons.
//!
//! ## Recommended Actions
//!
//! - **Long path → addSynapse** (skip connection): Connect a distant neuron
//!   closer to an output, shortening the effective path.
//! - **Connectivity imbalance → addSynapse**: Add inputs to starved neurons
//!   to balance information flow.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::SystemTime;

use crate::analysis::recommendation::epistatic::{BoundedScan, ScanTruncation};
use crate::analysis::utils::deadline::deadline_passed;
use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson};

use super::topology_cache::CreatureTopologyCache;

// MIN_SAMPLES_FOR_TOPOLOGY moved to constants.rs (Issue #424)
use crate::analysis::constants::MIN_DISCOVERY_SAMPLE_COUNT as MIN_SAMPLES_FOR_TOPOLOGY;

/// Maximum efficient path length (hops) from a hidden neuron to an output.
/// Neurons with longer shortest paths are candidates for skip connections.
const MAX_EFFICIENT_PATH_LENGTH: usize = 3;

/// Minimum fan-in ratio between the most-connected and least-connected hidden
/// neurons for connectivity imbalance detection. A ratio of 3.0 means the
/// most-connected neuron has at least 3x the fan-in of the least-connected.
const MIN_IMBALANCE_RATIO: f32 = 3.0;

/// Result of detecting a topology issue.
#[derive(Debug, Clone)]
pub struct TopologyCandidate {
    /// UUID of the neuron at the centre of the topology issue.
    pub neuron_uuid: String,
    /// Type of topology issue: "`long_path`" or "`connectivity_imbalance`".
    pub issue_type: String,
    /// Shortest path length to an output (for `long_path` issues).
    pub path_length: usize,
    /// Fan-in count for this neuron.
    pub fan_in: usize,
    /// Fan-out count for this neuron.
    pub fan_out: usize,
    /// Mean absolute error for this neuron across samples.
    pub mean_abs_error: f32,
    /// Estimated creature score improvement from resolving the issue.
    pub estimated_improvement: f32,
    /// Suggested target neuron for a skip connection (if applicable).
    pub skip_target_uuid: Option<String>,
    /// Suggested source neuron for an added synapse (if applicable).
    pub add_source_uuid: Option<String>,
}

/// Compute shortest path lengths from every neuron to any output neuron using
/// reverse BFS from all outputs simultaneously.
fn compute_shortest_paths_to_output(
    creature: &CreatureJson,
    deadline: &Option<SystemTime>,
) -> Option<HashMap<String, usize>> {
    // Build reverse adjacency: to_uuid → [from_uuid, ...]
    let mut reverse_adj: HashMap<&str, Vec<&str>> = HashMap::new();
    for s in &creature.synapses {
        if deadline_passed(deadline) {
            return None;
        }
        reverse_adj
            .entry(s.to_uuid.as_str())
            .or_default()
            .push(s.from_uuid.as_str());
    }

    let mut output_uuids = HashSet::new();
    for neuron in &creature.neurons {
        if deadline_passed(deadline) {
            return None;
        }
        if neuron.neuron_type == "output" {
            output_uuids.insert(neuron.uuid.as_str());
        }
    }

    let mut distances: HashMap<String, usize> = HashMap::new();
    let mut queue: VecDeque<(&str, usize)> = VecDeque::new();

    // Seed: outputs are at distance 0
    for &uuid in &output_uuids {
        if deadline_passed(deadline) {
            return None;
        }
        distances.insert(uuid.to_string(), 0);
        queue.push_back((uuid, 0));
    }

    // BFS backwards through the graph
    while let Some((current, dist)) = queue.pop_front() {
        if deadline_passed(deadline) {
            return None;
        }
        if let Some(predecessors) = reverse_adj.get(current) {
            for &pred in predecessors {
                if deadline_passed(deadline) {
                    return None;
                }
                if !distances.contains_key(pred) {
                    distances.insert(pred.to_string(), dist + 1);
                    queue.push_back((pred, dist + 1));
                }
            }
        }
    }

    Some(distances)
}

/// Detect topology issues from the creature's network structure and recorded activations.
///
/// # Arguments
/// * `creature` - The creature's network topology (neurons and synapses).
/// * `neuron_records` - List of `(neuron_uuid, records)` tuples with recorded activations.
///
/// # Returns
/// A list of `TopologyCandidate` sorted by estimated improvement (best first).
/// This compatibility wrapper has no analysis deadline but observes global cancellation.
pub fn detect_topology_issues(
    creature: &CreatureJson,
    neuron_records: &[(String, impl AsRef<[DiscoverRecord]>)],
    topo: Option<&CreatureTopologyCache>,
) -> Vec<TopologyCandidate> {
    detect_topology_issues_with_deadline(creature, neuron_records, topo, &None).candidates
}

/// Detect topology issues while observing the analysis deadline and global cancellation.
///
/// If interrupted, valid candidates found so far are returned with
/// `truncation = Some(DeadlinePassed)`.
pub fn detect_topology_issues_with_deadline(
    creature: &CreatureJson,
    neuron_records: &[(String, impl AsRef<[DiscoverRecord]>)],
    topo: Option<&CreatureTopologyCache>,
    deadline: &Option<SystemTime>,
) -> BoundedScan<TopologyCandidate> {
    if deadline_passed(deadline) {
        return BoundedScan {
            candidates: Vec::new(),
            truncation: Some(ScanTruncation::DeadlinePassed),
        };
    }

    // Use pre-computed cache or build locally.
    let local_cache;
    let topo = match topo {
        Some(t) => t,
        None => {
            local_cache = CreatureTopologyCache::new(creature);
            &local_cache
        }
    };

    if deadline_passed(deadline) {
        return BoundedScan {
            candidates: Vec::new(),
            truncation: Some(ScanTruncation::DeadlinePassed),
        };
    }

    if topo.hidden_uuids.is_empty() {
        return BoundedScan {
            candidates: Vec::new(),
            truncation: None,
        };
    }

    // Build records lookup
    let mut records_map = HashMap::with_capacity(neuron_records.len());
    for (uuid, records) in neuron_records {
        if deadline_passed(deadline) {
            return BoundedScan {
                candidates: Vec::new(),
                truncation: Some(ScanTruncation::DeadlinePassed),
            };
        }
        records_map.insert(uuid.as_str(), records.as_ref());
    }

    // Only consider hidden neurons with sufficient samples
    let mut qualified_hidden = Vec::with_capacity(topo.hidden_uuids.len());
    for uuid in &topo.hidden_uuids {
        if deadline_passed(deadline) {
            return BoundedScan {
                candidates: Vec::new(),
                truncation: Some(ScanTruncation::DeadlinePassed),
            };
        }
        if records_map
            .get(uuid.as_str())
            .is_some_and(|r| r.len() >= MIN_SAMPLES_FOR_TOPOLOGY)
        {
            qualified_hidden.push(uuid.as_str());
        }
    }

    if qualified_hidden.is_empty() {
        return BoundedScan {
            candidates: Vec::new(),
            truncation: None,
        };
    }

    // Compute shortest path to any output for each neuron
    let Some(path_distances) = compute_shortest_paths_to_output(creature, deadline) else {
        return BoundedScan {
            candidates: Vec::new(),
            truncation: Some(ScanTruncation::DeadlinePassed),
        };
    };

    // Compute mean absolute error for each qualified hidden neuron
    let mut mean_errors = HashMap::with_capacity(qualified_hidden.len());
    for &uuid in &qualified_hidden {
        if deadline_passed(deadline) {
            return BoundedScan {
                candidates: Vec::new(),
                truncation: Some(ScanTruncation::DeadlinePassed),
            };
        }
        let Some(records) = records_map.get(uuid) else {
            continue;
        };
        let mean_error = match mean_absolute_error(records, deadline) {
            Ok(mean_error) => mean_error,
            Err(reason) => {
                return BoundedScan {
                    candidates: Vec::new(),
                    truncation: Some(reason),
                };
            }
        };
        mean_errors.insert(uuid, mean_error);
    }

    // Copy HashSet iteration order once into compact storage, then stop at the
    // first source without an existing incoming edge for each starved neuron.
    let mut input_uuids = Vec::with_capacity(topo.input_uuids.len());
    for input_uuid in &topo.input_uuids {
        if deadline_passed(deadline) {
            return BoundedScan {
                candidates: Vec::new(),
                truncation: Some(ScanTruncation::DeadlinePassed),
            };
        }
        input_uuids.push(input_uuid.as_str());
    }

    // Every output has BFS distance zero. A >3-hop hidden neuron cannot already
    // connect to an output, so one target can serve all long-path candidates.
    let skip_target = topo.output_uuids.iter().next();

    let mut candidates = Vec::with_capacity(qualified_hidden.len());
    let mut truncation = None;

    // --- Detection 1: Long path to output ---
    for &uuid in &qualified_hidden {
        if deadline_passed(deadline) {
            truncation = Some(ScanTruncation::DeadlinePassed);
            break;
        }
        let path_len = match path_distances.get(uuid) {
            Some(&d) => d,
            None => continue, // unreachable from any output
        };

        if path_len <= MAX_EFFICIENT_PATH_LENGTH {
            continue;
        }

        let fan_in = topo.fan_in_for(uuid).len();
        let fan_out = topo.fan_out_for(uuid).len();
        let mean_err = mean_errors.get(uuid).copied().unwrap_or(0.0);

        // Keep the existing-edge exclusion even if a caller supplies a stale cache.
        let candidate_target = skip_target
            .filter(|out| !topo.synapse_exists(uuid, out))
            .cloned();

        // Estimated improvement: longer paths with higher error benefit more.
        // Scale: excess hops × error × small factor
        let excess_hops = (path_len - MAX_EFFICIENT_PATH_LENGTH) as f32;
        let estimated_improvement = excess_hops * mean_err * 0.005;

        // Issue #2351: non-finite recorded errors give a NaN/inf gain.
        if !estimated_improvement.is_finite() || estimated_improvement <= 0.0 {
            continue;
        }

        candidates.push(TopologyCandidate {
            neuron_uuid: uuid.to_string(),
            issue_type: "long_path".to_string(),
            path_length: path_len,
            fan_in,
            fan_out,
            mean_abs_error: mean_err,
            estimated_improvement,
            skip_target_uuid: candidate_target,
            add_source_uuid: None,
        });
    }

    if truncation.is_some() {
        candidates.sort_by(|a, b| b.estimated_improvement.total_cmp(&a.estimated_improvement));
        // Sorting is not interruptible, but report cancellation observed after it.
        if deadline_passed(deadline) {
            truncation = Some(ScanTruncation::DeadlinePassed);
        }
        return BoundedScan {
            candidates,
            truncation,
        };
    }

    // --- Detection 2: Connectivity imbalance ---
    // Compute fan-in stats across all qualified hidden neurons
    let mut fan_ins = Vec::with_capacity(qualified_hidden.len());
    let mut max_fan_in = 0;
    let mut min_fan_in = usize::MAX;
    for &uuid in &qualified_hidden {
        if deadline_passed(deadline) {
            truncation = Some(ScanTruncation::DeadlinePassed);
            break;
        }
        let fan_in = topo.fan_in_for(uuid).len();
        max_fan_in = max_fan_in.max(fan_in);
        min_fan_in = min_fan_in.min(fan_in);
        fan_ins.push((uuid, fan_in));
    }

    if truncation.is_none() && fan_ins.len() >= 2 {
        // Only flag imbalance if there's a significant ratio difference
        if min_fan_in > 0 && max_fan_in as f32 / min_fan_in as f32 >= MIN_IMBALANCE_RATIO {
            // Find the starved neurons (those with min fan-in)
            let starved_threshold = (max_fan_in as f32 / MIN_IMBALANCE_RATIO).ceil() as usize;

            for &(uuid, fi) in &fan_ins {
                if deadline_passed(deadline) {
                    truncation = Some(ScanTruncation::DeadlinePassed);
                    break;
                }
                if fi > starved_threshold {
                    continue;
                }

                let fan_out = topo.fan_out_for(uuid).len();
                let mean_err = mean_errors.get(uuid).copied().unwrap_or(0.0);
                let path_len = path_distances.get(uuid).copied().unwrap_or(0);

                // Find a good source to add a connection from.
                // Prefer input neurons not already connected to this neuron.
                let add_source = match first_unconnected_input(uuid, &input_uuids, topo, deadline) {
                    Ok(source) => source,
                    Err(reason) => {
                        truncation = Some(reason);
                        break;
                    }
                };

                // Estimated improvement: imbalance ratio × error × factor
                let imbalance_ratio = max_fan_in as f32 / fi.max(1) as f32;
                let estimated_improvement = (imbalance_ratio - 1.0) * mean_err * 0.003;

                // Issue #2351: non-finite recorded errors give a NaN/inf gain.
                if !estimated_improvement.is_finite() || estimated_improvement <= 0.0 {
                    continue;
                }

                candidates.push(TopologyCandidate {
                    neuron_uuid: uuid.to_string(),
                    issue_type: "connectivity_imbalance".to_string(),
                    path_length: path_len,
                    fan_in: fi,
                    fan_out,
                    mean_abs_error: mean_err,
                    estimated_improvement,
                    skip_target_uuid: None,
                    add_source_uuid: add_source,
                });
            }
        }
    }

    // Sort by estimated improvement (best first), including valid partial results.
    candidates.sort_by(|a, b| b.estimated_improvement.total_cmp(&a.estimated_improvement));
    // Sorting is not interruptible, but report cancellation observed after it.
    if deadline_passed(deadline) {
        truncation = Some(ScanTruncation::DeadlinePassed);
    }

    BoundedScan {
        candidates,
        truncation,
    }
}

fn mean_absolute_error(
    records: &[DiscoverRecord],
    deadline: &Option<SystemTime>,
) -> Result<f32, ScanTruncation> {
    let mut total_error = 0.0_f32;
    let mut count = 0_usize;
    let mut work_since_check = 0_usize;
    for record in records {
        work_since_check += 1;
        if work_since_check >= 1024 {
            if deadline_passed(deadline) {
                return Err(ScanTruncation::DeadlinePassed);
            }
            work_since_check = 0;
        }
        for error in &record.errors {
            work_since_check += 1;
            if work_since_check >= 1024 {
                if deadline_passed(deadline) {
                    return Err(ScanTruncation::DeadlinePassed);
                }
                work_since_check = 0;
            }
            total_error += error.abs();
            count += 1;
        }
    }
    Ok(if count > 0 {
        total_error / count as f32
    } else {
        0.0
    })
}

fn first_unconnected_input(
    neuron_uuid: &str,
    input_uuids: &[&str],
    topo: &CreatureTopologyCache,
    deadline: &Option<SystemTime>,
) -> Result<Option<String>, ScanTruncation> {
    for &input_uuid in input_uuids {
        if deadline_passed(deadline) {
            return Err(ScanTruncation::DeadlinePassed);
        }
        if !topo.synapse_exists(input_uuid, neuron_uuid) {
            return Ok(Some(input_uuid.to_string()));
        }
    }
    Ok(None)
}

#[cfg(test)]
#[path = "../topology_deadline_tests.rs"]
mod topology_deadline_tests;

/// Convert topology candidates into coordinated structural candidates.
///
/// - **Long path** → `addSynapse` (skip connection from neuron to a closer output)
/// - **Connectivity imbalance** → `addSynapse` (new input to starved neuron)
pub fn topology_issues_to_coordinated_candidates(
    candidates: &[TopologyCandidate],
    creature: &CreatureJson,
    topo: Option<&CreatureTopologyCache>,
) -> Vec<CoordinatedStructuralCandidateJson> {
    let local_cache;
    let topo = match topo {
        Some(t) => t,
        None => {
            local_cache = CreatureTopologyCache::new(creature);
            &local_cache
        }
    };

    let mut results = Vec::new();

    for c in candidates {
        match c.issue_type.as_str() {
            "long_path" => {
                if let Some(target) = &c.skip_target_uuid {
                    // Don't suggest if synapse already exists
                    if topo.synapse_exists(&c.neuron_uuid, target) {
                        continue;
                    }

                    // Skip connection: neuron → output with small initial weight
                    let weight = 0.1 * c.mean_abs_error.min(1.0);
                    results.push(CoordinatedStructuralCandidateJson {
                        remove_neuron_compensation: None,
        constant_neuron_bias_fold: None,
                        operations: vec![CoordinatedStructuralOpJson::AddSynapse {
                            from_neuron_uuid: c.neuron_uuid.clone(),
                            to_neuron_uuid: target.clone(),
                            weight,
                        }],
                        expected_creature_score_gain: c.estimated_improvement,
                        comment: Some(format!(
                            "Topology: long path ({} hops) from {} to output → add skip connection to {}, mean error {:.4}",
                            c.path_length, c.neuron_uuid, target, c.mean_abs_error
                        )),
                    });
                }
            }
            "connectivity_imbalance" => {
                if let Some(source) = &c.add_source_uuid {
                    // Don't suggest if synapse already exists
                    if topo.synapse_exists(source, &c.neuron_uuid) {
                        continue;
                    }

                    // Add connection from input to starved neuron with small weight
                    let weight = 0.1;
                    results.push(CoordinatedStructuralCandidateJson {
                        remove_neuron_compensation: None,
        constant_neuron_bias_fold: None,
                        operations: vec![CoordinatedStructuralOpJson::AddSynapse {
                            from_neuron_uuid: source.clone(),
                            to_neuron_uuid: c.neuron_uuid.clone(),
                            weight,
                        }],
                        expected_creature_score_gain: c.estimated_improvement,
                        comment: Some(format!(
                            "Topology: connectivity imbalance — {} has fan-in={} (starved) → add connection from {}, mean error {:.4}",
                            c.neuron_uuid, c.fan_in, source, c.mean_abs_error
                        )),
                    });
                }
            }
            _ => {}
        }
    }

    // Sort by expected improvement (best first)
    results.sort_by(|a, b| {
        b.expected_creature_score_gain
            .total_cmp(&a.expected_creature_score_gain)
    });

    results
}
