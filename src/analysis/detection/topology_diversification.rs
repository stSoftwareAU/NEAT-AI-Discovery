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
//!
//! ## Bounds (Issue #2489)
//!
//! The reverse-adjacency and direct-input index is built once per scan, in a single
//! O(N + S) pass over the synapses. Each output then does work proportional to its own
//! edges: its direct inputs are read from the index, and its unhealthy-intermediate walk
//! visits only that output's ancestors. The previous code rescanned every synapse for each
//! output, so the cost grew with outputs times synapses.
//!
//! The analysis deadline (via `deadline_passed`) is checked once per output, at the top of
//! the per-output loop. A stop keeps the candidates gathered so far, still sorted by
//! estimated improvement, and reports the reason via `TopologyDiversificationScan::stop`.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use std::collections::{HashMap, HashSet};
use std::time::SystemTime;

use crate::analysis::utils::deadline_passed;
use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson};

use crate::analysis::constants::MIN_DISCOVERY_SAMPLE_COUNT;

use super::error_dispersion::error_dispersion;
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

/// Why a topology-diversification scan stopped early (Issue #2489).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopologyDiversificationStop {
    /// The analysis deadline elapsed or the host requested cancellation.
    DeadlinePassed,
}

/// Candidates from a deadline-bounded topology-diversification scan, plus why it stopped
/// early (Issue #2489).
#[derive(Debug, Clone)]
pub struct TopologyDiversificationScan {
    /// Detected candidates, sorted by estimated improvement (best first).
    pub candidates: Vec<TopologyDiversificationCandidate>,
    /// `Some` when the scan stopped early.
    pub stop: Option<TopologyDiversificationStop>,
}

/// Compute input reachability and maximum hidden depth in validated evaluation
/// order. Dangling synapse endpoints remain tolerated but cannot extend a path.
fn input_reachable_hidden_depths(
    creature: &CreatureJson,
    hidden_set: &HashSet<&str>,
) -> Vec<usize> {
    let index_by_uuid: HashMap<&str, usize> = creature
        .neurons
        .iter()
        .enumerate()
        .map(|(index, neuron)| (neuron.uuid.as_str(), index))
        .collect();
    let mut reverse_adj = vec![Vec::new(); creature.neurons.len()];

    for synapse in &creature.synapses {
        let (Some(&from_index), Some(&to_index)) = (
            index_by_uuid.get(synapse.from_uuid.as_str()),
            index_by_uuid.get(synapse.to_uuid.as_str()),
        ) else {
            continue;
        };
        reverse_adj[to_index].push(from_index);
    }

    let mut input_reachable = vec![false; creature.neurons.len()];
    let mut depths = vec![0; creature.neurons.len()];

    for (index, neuron) in creature.neurons.iter().enumerate() {
        if neuron.neuron_type == "input" {
            input_reachable[index] = true;
            continue;
        }

        let mut best_predecessor_depth = 0;
        for &predecessor in &reverse_adj[index] {
            if input_reachable[predecessor] {
                best_predecessor_depth = best_predecessor_depth.max(depths[predecessor]);
                input_reachable[index] = true;
            }
        }

        if input_reachable[index] && hidden_set.contains(neuron.uuid.as_str()) {
            best_predecessor_depth += 1;
        }
        depths[index] = best_predecessor_depth;
    }

    depths
}

/// Edge index for one scan (Issue #2489), built once in a single O(N + S) pass over the
/// synapses. Each output's direct-input and predecessor lookups are then O(1) plus that
/// output's own edges.
struct SynapseIndex<'a> {
    /// `to_uuid` -> `from_uuid` of each synapse whose source is an input neuron, in synapse
    /// order. Duplicate synapses are kept, so each one counts as a direct path.
    direct_inputs: HashMap<&'a str, Vec<&'a str>>,
    /// `to_uuid` -> `from_uuid` of every synapse, dangling endpoints included, so the
    /// unhealthy-intermediate walk sees the same predecessors as before the index.
    reverse_adj: HashMap<&'a str, Vec<&'a str>>,
}

impl<'a> SynapseIndex<'a> {
    /// Build the index. `on_step` is called once per synapse.
    fn build(creature: &'a CreatureJson, on_step: &mut impl FnMut()) -> Self {
        let input_set: HashSet<&str> = creature
            .neurons
            .iter()
            .filter(|n| n.neuron_type == "input")
            .map(|n| n.uuid.as_str())
            .collect();

        let mut direct_inputs: HashMap<&str, Vec<&str>> = HashMap::new();
        let mut reverse_adj: HashMap<&str, Vec<&str>> = HashMap::new();
        for s in &creature.synapses {
            on_step();
            let from = s.from_uuid.as_str();
            let to = s.to_uuid.as_str();
            reverse_adj.entry(to).or_default().push(from);
            if input_set.contains(from) {
                direct_inputs.entry(to).or_default().push(from);
            }
        }

        Self {
            direct_inputs,
            reverse_adj,
        }
    }
}

/// Find the best input neuron to use as source for an inserted hidden neuron.
/// Prefers the input with the highest activation variance (most informative signal).
///
/// `direct_inputs` is one output's direct-input slice from [`SynapseIndex`]. The result is
/// `None` only when that slice is empty.
fn best_source_input<'a>(
    direct_inputs: &[&'a str],
    records_map: &HashMap<&str, &[DiscoverRecord]>,
) -> Option<&'a str> {
    // Pick the input with the highest activation variance
    direct_inputs.iter().copied().max_by(|&a, &b| {
        let var_a = activation_variance(records_map.get(a).copied());
        let var_b = activation_variance(records_map.get(b).copied());
        var_a.total_cmp(&var_b)
    })
}

/// Compute activation variance for a set of records.
fn activation_variance(records: Option<&[DiscoverRecord]>) -> f32 {
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
///
/// `reverse_adj` is the scan's predecessor index (Issue #2489). `on_step` is called once per
/// predecessor edge examined.
fn has_unhealthy_intermediates(
    reverse_adj: &HashMap<&str, Vec<&str>>,
    output_uuid: &str,
    hidden_set: &HashSet<&str>,
    records_map: &HashMap<&str, &[DiscoverRecord]>,
    on_step: &mut impl FnMut(),
) -> bool {
    // Find hidden neurons that feed (directly or indirectly) into this output.
    // Walk backwards from the output (stack order) to find hidden neurons on paths.
    let mut visited = HashSet::new();
    let mut queue = vec![output_uuid];
    visited.insert(output_uuid);

    let mut path_hidden_neurons: Vec<&str> = Vec::new();

    while let Some(current) = queue.pop() {
        if let Some(predecessors) = reverse_adj.get(current) {
            for &pred in predecessors {
                on_step();
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

    /// Mean absolute error below which a hidden neuron counts as converged, so
    /// its error variance carries no diversification signal.
    const MIN_MEAN_ERROR_FOR_CV: f32 = 0.001;

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
            // Same mean → std_dev → CV chain as the plateau detectors, but read
            // in the opposite direction: high variance, not tight clustering
            // (Issue #2044).
            let Some(dispersion) = error_dispersion(&errors, MIN_MEAN_ERROR_FOR_CV) else {
                continue;
            };
            if dispersion.cv > HIGH_ERROR_CV_THRESHOLD {
                return true;
            }
        }
    }

    false
}

/// Detect topology diversification candidates from the creature's network
/// structure and recorded activations.
///
/// Unbounded by deadline: this is a thin wrapper over
/// [`detect_topology_diversification_candidates_with_deadline`] with `deadline` set to `None`.
///
/// # Arguments
/// * `creature` - The creature's network topology (neurons and synapses).
/// * `neuron_records` - List of `(neuron_uuid, records)` tuples with recorded activations.
///
/// # Returns
/// A list of `TopologyDiversificationCandidate` sorted by estimated improvement (best first).
pub fn detect_topology_diversification_candidates(
    creature: &CreatureJson,
    neuron_records: &[(String, impl AsRef<[DiscoverRecord]>)],
) -> Vec<TopologyDiversificationCandidate> {
    detect_topology_diversification_candidates_with_deadline(creature, neuron_records, &None)
        .candidates
}

/// Detect topology diversification candidates, bounded by `deadline` (Issue #2489).
///
/// # Arguments
/// * `creature` - The creature's network topology (neurons and synapses).
/// * `neuron_records` - List of `(neuron_uuid, records)` tuples with recorded activations.
/// * `deadline` - When set, the scan stops early (and reports why via
///   `TopologyDiversificationScan::stop`) once the deadline elapses or global cancellation is
///   requested.
///
/// # Returns
/// A [`TopologyDiversificationScan`] with the detected candidates (sorted by estimated
/// improvement, best first) and, when the scan stopped early, the reason why.
pub fn detect_topology_diversification_candidates_with_deadline(
    creature: &CreatureJson,
    neuron_records: &[(String, impl AsRef<[DiscoverRecord]>)],
    deadline: &Option<SystemTime>,
) -> TopologyDiversificationScan {
    detect_topology_diversification_observed(creature, neuron_records, deadline, || {})
}

/// Detect topology diversification candidates, bounded by `deadline`, with a step observer
/// (Issue #2489).
///
/// `on_step` is the deterministic work counter used by step-count tests. It is called once per
/// synapse when the edge index is built, and once per predecessor edge examined by each
/// output's unhealthy-intermediate walk.
///
/// The edge index is built once per call, so the scan costs O(N + S) plus work proportional to
/// each output's own edges. The deadline is checked once per output, at the top of the loop.
fn detect_topology_diversification_observed(
    creature: &CreatureJson,
    neuron_records: &[(String, impl AsRef<[DiscoverRecord]>)],
    deadline: &Option<SystemTime>,
    mut on_step: impl FnMut(),
) -> TopologyDiversificationScan {
    let output_neurons: Vec<usize> = creature
        .neurons
        .iter()
        .enumerate()
        .filter(|(_, neuron)| neuron.neuron_type == "output")
        .map(|(index, _)| index)
        .collect();

    if output_neurons.is_empty() {
        return TopologyDiversificationScan {
            candidates: Vec::new(),
            stop: None,
        };
    }

    // SIMPLE-ON-PURPOSE: the O(N + S) index and depth setup runs unchecked; the loop checks the deadline. Upgrade if setup alone overruns it.
    let hidden_set: HashSet<&str> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "hidden")
        .map(|n| n.uuid.as_str())
        .collect();
    let hidden_depths = input_reachable_hidden_depths(creature, &hidden_set);

    let records_map = build_record_map(neuron_records);
    let index = SynapseIndex::build(creature, &mut on_step);

    let mut stop: Option<TopologyDiversificationStop> = None;
    let mut candidates = Vec::with_capacity(output_neurons.len());

    for &output_index in &output_neurons {
        // Issue #2489: checked once per output. A stop keeps the candidates gathered so far.
        if deadline_passed(deadline) {
            stop = Some(TopologyDiversificationStop::DeadlinePassed);
            break;
        }

        let output_uuid = creature.neurons[output_index].uuid.as_str();
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
        let max_depth = hidden_depths[output_index];

        // Only flag if topology lacks non-linear depth
        if max_depth >= MIN_NONLINEAR_DEPTH {
            continue;
        }

        // Direct input→output paths, read from the index (Issue #2489). Duplicates count.
        let direct_inputs: &[&str] = index
            .direct_inputs
            .get(output_uuid)
            .map_or(&[], Vec::as_slice);
        let direct_paths = direct_inputs.len();

        // Find the best input source for the new neuron. `None` means there are no input
        // paths at all, which is a different problem.
        let Some(source_uuid) = best_source_input(direct_inputs, &records_map) else {
            continue;
        };

        // Check for unhealthy intermediates (parametric issue, not structural)
        if has_unhealthy_intermediates(
            &index.reverse_adj,
            output_uuid,
            &hidden_set,
            &records_map,
            &mut on_step,
        ) {
            continue;
        }

        // Estimated improvement: proportional to error and structural deficit
        let depth_deficit = (MIN_NONLINEAR_DEPTH - max_depth) as f32;
        let estimated_improvement = mean_output_error * depth_deficit * 0.15;

        if estimated_improvement <= 0.0 {
            continue;
        }

        candidates.push(TopologyDiversificationCandidate {
            output_neuron_uuid: output_uuid.to_string(),
            source_input_uuid: source_uuid.to_string(),
            max_hidden_depth: max_depth,
            direct_path_count: direct_paths,
            mean_output_error,
            estimated_improvement,
        });
    }

    // Sort by estimated improvement (best first)
    candidates.sort_by(|a, b| b.estimated_improvement.total_cmp(&a.estimated_improvement));

    TopologyDiversificationScan { candidates, stop }
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
    // Issue #983: Use borrowed references to avoid cloning every synapse UUID pair.
    let existing_synapses: HashSet<(&str, &str)> = creature
        .synapses
        .iter()
        .map(|s| (s.from_uuid.as_str(), s.to_uuid.as_str()))
        .collect();

    let mut results = Vec::new();

    for c in candidates {
        let neuron_uuid =
            topology_diversification_neuron_uuid(&c.source_input_uuid, &c.output_neuron_uuid);

        // Skip if the new neuron's connections would duplicate existing synapses
        if existing_synapses.contains(&(neuron_uuid.as_str(), c.output_neuron_uuid.as_str())) {
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
            remove_neuron_compensation: None,
            constant_neuron_bias_fold: None,
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

#[cfg(test)]
#[path = "../topology_diversification_deadline_tests.rs"]
mod topology_diversification_deadline_tests;
