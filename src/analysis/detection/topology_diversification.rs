//! Topology diversification for structural jumps (Issue #549).
//!
//! Detects when the network topology is too simple for the observed error patterns.
//! Weight and bias adjustments operate in a fixed-dimension space; adding neurons
//! changes the dimensionality of the solution space, allowing the network to reach
//! solutions unreachable by parameter tuning alone.
//!
//! ## Detection Criteria
//!
//! 1. **Insufficient non-linear depth**: All paths from inputs to an output neuron
//!    pass through zero hidden neurons (direct connections only). When error is high,
//!    this indicates the problem requires non-linear intermediate processing.
//! 2. **Healthy intermediates but high output error**: When hidden neurons exist on
//!    some paths but the output error is still high and the hidden neurons themselves
//!    show low individual error variance, the issue is structural — more non-linear
//!    capacity is needed.
//!
//! ## Recommended Actions
//!
//! - **addNeuron** at strategic insertion points between inputs and outputs where
//!   non-linear transformations would provide the most benefit.
//! - Focus on paths where error is high but individual neuron metrics look healthy,
//!   indicating the issue is structural, not parametric.

use std::collections::{HashMap, HashSet};

use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson};

use crate::analysis::constants::MIN_DISCOVERY_SAMPLE_COUNT;

use super::helpers::build_record_map;

/// Minimum mean absolute output error to consider the topology insufficient.
/// Below this threshold the network is performing adequately.
const MIN_OUTPUT_ERROR_FOR_DIVERSIFICATION: f32 = 0.05;

/// Maximum number of hidden neurons on any input→output path before we consider
/// the topology "sufficient". If all paths have fewer hidden neurons than this,
/// the network may lack non-linear capacity.
const MIN_NONLINEAR_DEPTH: usize = 1;

/// Result of detecting a topology diversification opportunity.
#[derive(Debug, Clone)]
pub struct TopologyDiversificationCandidate {
    /// UUID of the output neuron with high error and insufficient topology.
    pub output_neuron_uuid: String,
    /// UUID of the best input neuron to use as source for the new hidden neuron.
    pub source_input_uuid: String,
    /// Maximum depth of hidden neurons on paths to this output.
    pub max_hidden_depth: usize,
    /// Number of direct (zero-hidden) input→output paths.
    pub direct_path_count: usize,
    /// Mean absolute error at the output neuron.
    pub mean_output_error: f32,
    /// Estimated creature score improvement from adding a neuron.
    pub estimated_improvement: f32,
}

/// Compute the maximum number of hidden neurons on any path from any input to a
/// specific output neuron, using DFS through the synapse graph.
fn max_hidden_depth_to_output(
    creature: &CreatureJson,
    output_uuid: &str,
    hidden_set: &HashSet<&str>,
) -> usize {
    // Build reverse adjacency: to_uuid → [from_uuid, ...]
    let mut reverse_adj: HashMap<&str, Vec<&str>> = HashMap::new();
    for s in &creature.synapses {
        reverse_adj
            .entry(s.to_uuid.as_str())
            .or_default()
            .push(s.from_uuid.as_str());
    }

    let input_set: HashSet<&str> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "input")
        .map(|n| n.uuid.as_str())
        .collect();

    // DFS backwards from output, counting hidden neurons on each path
    let mut max_depth: usize = 0;
    let mut stack: Vec<(&str, usize, HashSet<&str>)> = Vec::new();

    let mut initial_visited = HashSet::new();
    initial_visited.insert(output_uuid);
    stack.push((output_uuid, 0, initial_visited));

    while let Some((current, hidden_count, visited)) = stack.pop() {
        if input_set.contains(current) {
            // Reached an input — record the hidden depth of this path
            max_depth = max_depth.max(hidden_count);
            continue;
        }

        if let Some(predecessors) = reverse_adj.get(current) {
            for &pred in predecessors {
                if visited.contains(pred) {
                    continue; // Avoid cycles
                }
                let mut new_visited = visited.clone();
                new_visited.insert(pred);
                let added_hidden = if hidden_set.contains(pred) { 1 } else { 0 };
                stack.push((pred, hidden_count + added_hidden, new_visited));
            }
        }
    }

    max_depth
}

/// Count direct (zero-hidden) paths from inputs to a specific output neuron.
fn count_direct_input_paths(creature: &CreatureJson, output_uuid: &str) -> usize {
    let input_set: HashSet<&str> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "input")
        .map(|n| n.uuid.as_str())
        .collect();

    creature
        .synapses
        .iter()
        .filter(|s| s.to_uuid == output_uuid && input_set.contains(s.from_uuid.as_str()))
        .count()
}

/// Find the best input neuron to use as source for an inserted hidden neuron.
/// Prefers the input with the highest activation variance (most informative signal).
fn best_source_input(
    creature: &CreatureJson,
    output_uuid: &str,
    records_map: &HashMap<&str, &Vec<DiscoverRecord>>,
) -> Option<String> {
    let direct_inputs: Vec<&str> = creature
        .synapses
        .iter()
        .filter(|s| s.to_uuid == output_uuid)
        .filter(|s| {
            creature
                .neurons
                .iter()
                .any(|n| n.uuid == s.from_uuid && n.neuron_type == "input")
        })
        .map(|s| s.from_uuid.as_str())
        .collect();

    if direct_inputs.is_empty() {
        return None;
    }

    // Pick the input with the highest activation variance
    direct_inputs
        .into_iter()
        .max_by(|&a, &b| {
            let var_a = activation_variance(records_map.get(a).copied());
            let var_b = activation_variance(records_map.get(b).copied());
            var_a.total_cmp(&var_b)
        })
        .map(std::string::ToString::to_string)
}

/// Compute activation variance for a set of records.
fn activation_variance(records: Option<&Vec<DiscoverRecord>>) -> f32 {
    let records = match records {
        Some(r) if !r.is_empty() => r,
        _ => return 0.0,
    };

    let n = records.len() as f32;
    let mean: f32 = records.iter().map(|r| r.activation).sum::<f32>() / n;
    let variance: f32 = records
        .iter()
        .map(|r| (r.activation - mean).powi(2))
        .sum::<f32>()
        / n;
    variance
}

/// Check whether any hidden neuron on the paths to this output has high error
/// variance (indicating a parametric issue, not structural).
fn has_unhealthy_intermediates(
    creature: &CreatureJson,
    output_uuid: &str,
    hidden_set: &HashSet<&str>,
    records_map: &HashMap<&str, &Vec<DiscoverRecord>>,
) -> bool {
    // Find hidden neurons that feed (directly or indirectly) into this output
    let mut reverse_adj: HashMap<&str, Vec<&str>> = HashMap::new();
    for s in &creature.synapses {
        reverse_adj
            .entry(s.to_uuid.as_str())
            .or_default()
            .push(s.from_uuid.as_str());
    }

    // BFS backwards from output to find hidden neurons on paths
    let mut visited = HashSet::new();
    let mut queue = vec![output_uuid];
    visited.insert(output_uuid);

    let mut path_hidden_neurons: Vec<&str> = Vec::new();

    while let Some(current) = queue.pop() {
        if let Some(predecessors) = reverse_adj.get(current) {
            for &pred in predecessors {
                if visited.contains(pred) {
                    continue;
                }
                visited.insert(pred);
                if hidden_set.contains(pred) {
                    path_hidden_neurons.push(pred);
                }
                queue.push(pred);
            }
        }
    }

    // Check if any on-path hidden neuron has high error variance
    const HIGH_ERROR_CV_THRESHOLD: f32 = 0.8;

    for &h_uuid in &path_hidden_neurons {
        if let Some(records) = records_map.get(h_uuid) {
            if records.len() < MIN_DISCOVERY_SAMPLE_COUNT {
                continue;
            }
            let errors: Vec<f32> = records
                .iter()
                .flat_map(|r| r.errors.iter())
                .map(|e| e.abs())
                .collect();
            if errors.is_empty() {
                continue;
            }
            let n = errors.len() as f32;
            let mean = errors.iter().sum::<f32>() / n;
            if mean < 0.001 {
                continue;
            }
            let std_dev = (errors.iter().map(|e| (e - mean).powi(2)).sum::<f32>() / n).sqrt();
            let cv = std_dev / mean;
            if cv > HIGH_ERROR_CV_THRESHOLD {
                return true;
            }
        }
    }

    false
}

/// Detect topology diversification candidates from the creature's network
/// structure and recorded activations.
///
/// # Arguments
/// * `creature` - The creature's network topology (neurons and synapses).
/// * `neuron_records` - List of `(neuron_uuid, records)` tuples with recorded activations.
///
/// # Returns
/// A list of `TopologyDiversificationCandidate` sorted by estimated improvement (best first).
pub fn detect_topology_diversification_candidates(
    creature: &CreatureJson,
    neuron_records: &[(String, Vec<DiscoverRecord>)],
) -> Vec<TopologyDiversificationCandidate> {
    let output_neurons: Vec<&str> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "output")
        .map(|n| n.uuid.as_str())
        .collect();

    if output_neurons.is_empty() {
        return Vec::new();
    }

    let hidden_set: HashSet<&str> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "hidden")
        .map(|n| n.uuid.as_str())
        .collect();

    let records_map = build_record_map(neuron_records);

    let mut candidates = Vec::with_capacity(output_neurons.len());

    for &output_uuid in &output_neurons {
        // Check sample count
        let output_records = match records_map.get(output_uuid) {
            Some(r) if r.len() >= MIN_DISCOVERY_SAMPLE_COUNT => *r,
            _ => continue,
        };

        // Compute mean absolute error at this output
        let errors: Vec<f32> = output_records
            .iter()
            .flat_map(|r| r.errors.iter())
            .map(|e| e.abs())
            .collect();
        if errors.is_empty() {
            continue;
        }
        let mean_output_error = errors.iter().sum::<f32>() / errors.len() as f32;

        // Skip if error is already low — topology is adequate
        if mean_output_error < MIN_OUTPUT_ERROR_FOR_DIVERSIFICATION {
            continue;
        }

        // Compute maximum hidden depth on any input→output path
        let max_depth = max_hidden_depth_to_output(creature, output_uuid, &hidden_set);

        // Only flag if topology lacks non-linear depth
        if max_depth >= MIN_NONLINEAR_DEPTH {
            continue;
        }

        // Count direct input→output paths
        let direct_paths = count_direct_input_paths(creature, output_uuid);
        if direct_paths == 0 {
            continue; // No input paths at all — different problem
        }

        // Check for unhealthy intermediates (parametric issue, not structural)
        if has_unhealthy_intermediates(creature, output_uuid, &hidden_set, &records_map) {
            continue;
        }

        // Find the best input source for the new neuron
        let source_uuid = match best_source_input(creature, output_uuid, &records_map) {
            Some(s) => s,
            None => continue,
        };

        // Estimated improvement: proportional to error and structural deficit
        let depth_deficit = (MIN_NONLINEAR_DEPTH - max_depth) as f32;
        let estimated_improvement = mean_output_error * depth_deficit * 0.15;

        if estimated_improvement <= 0.0 {
            continue;
        }

        candidates.push(TopologyDiversificationCandidate {
            output_neuron_uuid: output_uuid.to_string(),
            source_input_uuid: source_uuid,
            max_hidden_depth: max_depth,
            direct_path_count: direct_paths,
            mean_output_error,
            estimated_improvement,
        });
    }

    // Sort by estimated improvement (best first)
    candidates.sort_by(|a, b| b.estimated_improvement.total_cmp(&a.estimated_improvement));

    candidates
}

/// Generate a deterministic UUID for a topology diversification neuron.
fn topology_diversification_neuron_uuid(source_uuid: &str, output_uuid: &str) -> String {
    let key = format!("topology-diversification|{source_uuid}|{output_uuid}");
    // FNV-1a 64-bit
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in key.as_bytes() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("td-{hash:016x}")
}

/// Convert topology diversification candidates into coordinated structural candidates.
///
/// Each candidate produces a coordinated structural operation consisting of:
/// 1. `AddNeuron` — a new hidden neuron with TANH activation between input and output
/// 2. `AddSynapse` — connection from the source input to the new neuron
/// 3. `AddSynapse` — connection from the new neuron to the output
pub fn topology_diversification_to_coordinated_candidates(
    candidates: &[TopologyDiversificationCandidate],
    creature: &CreatureJson,
) -> Vec<CoordinatedStructuralCandidateJson> {
    let existing_synapses: HashSet<(String, String)> = creature
        .synapses
        .iter()
        .map(|s| (s.from_uuid.clone(), s.to_uuid.clone()))
        .collect();

    let mut results = Vec::new();

    for c in candidates {
        let neuron_uuid =
            topology_diversification_neuron_uuid(&c.source_input_uuid, &c.output_neuron_uuid);

        // Skip if the new neuron's connections would duplicate existing synapses
        if existing_synapses.contains(&(neuron_uuid.clone(), c.output_neuron_uuid.clone())) {
            continue;
        }

        let operations = vec![
            CoordinatedStructuralOpJson::AddNeuron {
                neuron_uuid: neuron_uuid.clone(),
                neuron_type: "hidden".to_string(),
                squash: "TANH".to_string(),
                bias: 0.0,
                insert_before_neuron_uuid: Some(c.output_neuron_uuid.clone()),
            },
            CoordinatedStructuralOpJson::AddSynapse {
                from_neuron_uuid: c.source_input_uuid.clone(),
                to_neuron_uuid: neuron_uuid.clone(),
                weight: 0.5,
            },
            CoordinatedStructuralOpJson::AddSynapse {
                from_neuron_uuid: neuron_uuid,
                to_neuron_uuid: c.output_neuron_uuid.clone(),
                weight: 0.1,
            },
        ];

        results.push(CoordinatedStructuralCandidateJson {
            operations,
            expected_creature_score_gain: c.estimated_improvement,
            comment: Some(format!(
                "Issue #549: Topology diversification — add hidden neuron between {} and {} \
                 (max hidden depth={}, direct paths={}, mean output error={:.4})",
                c.source_input_uuid,
                c.output_neuron_uuid,
                c.max_hidden_depth,
                c.direct_path_count,
                c.mean_output_error,
            )),
        });
    }

    // Sort by expected improvement (best first)
    results.sort_by(|a, b| {
        b.expected_creature_score_gain
            .total_cmp(&a.expected_creature_score_gain)
    });

    results
}
