//! Skip-connection discovery module (Issue #570).
//!
//! Analyses neuron topological depth and gradient attenuation to identify
//! beneficial residual (skip) connections across layers. Deep neurons with
//! attenuated error gradients are connected directly to shallow neurons or
//! inputs, bridging large depth gaps.
//!
//! ## Detection Criteria
//!
//! 1. **Layer depth analysis**: Compute topological depth of each neuron from
//!    inputs using a Kahn topological pass (longest path from any input).
//! 2. **Gradient attenuation**: Identify neurons at depth > 2 whose mean
//!    absolute error is significantly lower than the mean error at shallow
//!    depths (ratio below `ATTENUATION_THRESHOLD`).
//! 3. **Candidate generation**: For attenuated neurons, suggest addSynapse
//!    candidates connecting a shallow source (input or shallow hidden) directly
//!    to the deep neuron, prioritising the largest depth gaps and using
//!    conservative initial weights near zero.
//!
//! ## Recommended Actions
//!
//! - **Attenuated gradient → addSynapse** (skip connection): Connect a shallow
//!   neuron directly to a deep neuron to restore gradient signal flow.
//!
//! ## Bounds (Issue #2368)
//!
//! The depth pass is a Kahn topological sort, O(V + E): each neuron is popped
//! once and each synapse is relaxed once. A cycle has no topological depth, so
//! the scan returns no candidates and reports `SkipConnectionStop::CycleDetected`
//! rather than looping. The target loop is indexed: each deep target scans the
//! shallow sources once, skipping only its existing in-edges, so the pair search
//! is O(E + D) rather than O(D · S).
//!
//! The analysis deadline (via `deadline_passed`) is checked before the depth
//! pass, once per neuron popped in it, and once per deep target. A stop during
//! the depth pass returns no candidates because the depths are incomplete. A stop
//! in the target loop keeps the candidates gathered so far. Either way the reason
//! is reported via `SkipConnectionScan::stop`.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::SystemTime;

use crate::analysis::utils::deadline_passed;
use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson};

use crate::analysis::constants::MIN_DISCOVERY_SAMPLE_COUNT;

use super::helpers::build_record_map;

/// Minimum topological depth from inputs for a neuron to be considered "deep"
/// and eligible for skip-connection candidates.
const MIN_DEEP_DEPTH: usize = 3;

/// Attenuation threshold: a deep neuron's mean absolute error must be below
/// this fraction of the shallow mean error to be considered gradient-attenuated.
const ATTENUATION_THRESHOLD: f32 = 0.5;

/// Result of detecting a skip-connection candidate.
#[derive(Debug, Clone)]
pub struct SkipConnectionCandidate {
    /// UUID of the deep neuron that would receive the skip connection.
    pub target_uuid: String,
    /// Topological depth of the target neuron from inputs.
    pub target_depth: usize,
    /// UUID of the shallow source neuron for the skip connection.
    pub source_uuid: String,
    /// Topological depth of the source neuron from inputs.
    pub source_depth: usize,
    /// Depth gap bridged by this skip connection.
    pub depth_gap: usize,
    /// Mean absolute error of the target neuron.
    pub target_mean_error: f32,
    /// Estimated creature score improvement from adding this connection.
    pub estimated_improvement: f32,
}

/// Why a skip-connection scan stopped early (Issue #2368).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkipConnectionStop {
    /// The synapse graph has a cycle, so no topological depth exists. No candidates are returned.
    CycleDetected,
    /// The analysis deadline elapsed or the host requested cancellation.
    DeadlinePassed,
}

/// Candidates from a bounded skip-connection scan, plus why it stopped early (Issue #2368).
#[derive(Debug, Clone)]
pub struct SkipConnectionScan {
    /// Detected candidates, sorted by estimated improvement (best first).
    pub candidates: Vec<SkipConnectionCandidate>,
    /// `Some` when the scan stopped early or the synapse graph has a cycle.
    pub stop: Option<SkipConnectionStop>,
}

/// Outcome of the topological depth pass (Issue #2368).
#[derive(Debug)]
enum DepthPass {
    /// Longest-path depth of every neuron reachable from an input.
    Depths(HashMap<String, usize>),
    /// The synapse graph has a cycle, so no topological depth exists.
    CycleDetected,
    /// The deadline elapsed or cancellation was requested during the pass.
    DeadlinePassed,
}

/// Scan result with no candidates and the given stop reason.
fn empty_scan(stop: Option<SkipConnectionStop>) -> SkipConnectionScan {
    SkipConnectionScan {
        candidates: Vec::new(),
        stop,
    }
}

/// Index `uuid` in `index`, assigning the next free index on first sight (Issue #2368).
fn intern_uuid<'a>(
    uuid: &'a str,
    index: &mut HashMap<&'a str, usize>,
    names: &mut Vec<&'a str>,
) -> usize {
    *index.entry(uuid).or_insert_with(|| {
        names.push(uuid);
        names.len() - 1
    })
}

/// Compute topological depth of each neuron from inputs using a Kahn topological pass.
///
/// Input neurons have depth 0. A neuron's depth is the longest path from any input,
/// so each layer increments by 1 from the deepest predecessor. Neurons not reachable
/// from an input have no depth and are absent from the returned map.
///
/// Nodes are every neuron UUID plus every synapse endpoint. The pass runs in O(V + E)
/// (Issue #2368). If some node is never popped, the graph has a cycle and no depth is
/// defined. `on_step` is called once per node popped and once per edge relaxed. The
/// deadline is checked before the pass and once per node popped.
fn compute_depths_from_inputs<F: FnMut()>(
    creature: &CreatureJson,
    deadline: &Option<SystemTime>,
    on_step: &mut F,
) -> DepthPass {
    // Densely index every node: neuron UUIDs first, then any synapse endpoint.
    let mut index: HashMap<&str, usize> = HashMap::new();
    let mut names: Vec<&str> = Vec::new();
    let mut input_indices: Vec<usize> = Vec::new();
    for n in &creature.neurons {
        let i = intern_uuid(n.uuid.as_str(), &mut index, &mut names);
        if n.neuron_type == "input" {
            input_indices.push(i);
        }
    }
    let mut edges: Vec<(usize, usize)> = Vec::with_capacity(creature.synapses.len());
    for s in &creature.synapses {
        let from = intern_uuid(s.from_uuid.as_str(), &mut index, &mut names);
        let to = intern_uuid(s.to_uuid.as_str(), &mut index, &mut names);
        edges.push((from, to));
    }
    let node_count = names.len();

    let mut successors: Vec<Vec<usize>> = vec![Vec::new(); node_count];
    let mut in_degree: Vec<usize> = vec![0; node_count];
    for &(from, to) in &edges {
        successors[from].push(to);
        in_degree[to] += 1;
    }

    // Input neurons seed the pass at depth 0.
    let mut depth: Vec<Option<usize>> = vec![None; node_count];
    for &i in &input_indices {
        depth[i] = Some(0);
    }

    if deadline_passed(deadline) {
        return DepthPass::DeadlinePassed;
    }

    let mut queue: VecDeque<usize> = (0..node_count).filter(|&i| in_degree[i] == 0).collect();
    let mut popped = 0_usize;
    while let Some(u) = queue.pop_front() {
        if deadline_passed(deadline) {
            return DepthPass::DeadlinePassed;
        }
        on_step();
        popped += 1;
        for &v in &successors[u] {
            on_step();
            // Longest path: a successor takes the deepest predecessor plus one.
            // Unreachable predecessors (depth None) leave the successor unchanged.
            if let Some(d) = depth[u] {
                depth[v] = Some(depth[v].map_or(d + 1, |cur| cur.max(d + 1)));
            }
            in_degree[v] -= 1;
            if in_degree[v] == 0 {
                queue.push_back(v);
            }
        }
    }

    // A node that never reached in-degree 0 sits on or behind a cycle.
    if popped < node_count {
        return DepthPass::CycleDetected;
    }

    DepthPass::Depths(
        names
            .iter()
            .zip(depth)
            .filter_map(|(name, d)| d.map(|d| ((*name).to_string(), d)))
            .collect(),
    )
}

/// Detect skip-connection candidates based on gradient attenuation in deep neurons.
///
/// Unbounded by deadline: this is a thin wrapper over
/// [`detect_skip_connection_candidates_with_deadline`] with `deadline` set to `None`.
///
/// # Arguments
/// * `creature` - The creature's network topology.
/// * `neuron_records` - List of `(neuron_uuid, records)` tuples with recorded data.
///
/// # Returns
/// A list of `SkipConnectionCandidate` sorted by estimated improvement (best first).
pub fn detect_skip_connection_candidates(
    creature: &CreatureJson,
    neuron_records: &[(String, impl AsRef<[DiscoverRecord]>)],
) -> Vec<SkipConnectionCandidate> {
    detect_skip_connection_candidates_with_deadline(creature, neuron_records, &None).candidates
}

/// Detect skip-connection candidates, bounded by `deadline` (Issue #2368).
///
/// # Arguments
/// * `creature` - The creature's network topology.
/// * `neuron_records` - List of `(neuron_uuid, records)` tuples with recorded data.
/// * `deadline` - When set, the scan stops early (and reports why via
///   `SkipConnectionScan::stop`) once the deadline elapses or global cancellation is requested.
///
/// # Returns
/// A [`SkipConnectionScan`] with the detected candidates (sorted by estimated improvement,
/// best first) and, when the scan stopped early or the graph has a cycle, the reason why.
pub fn detect_skip_connection_candidates_with_deadline(
    creature: &CreatureJson,
    neuron_records: &[(String, impl AsRef<[DiscoverRecord]>)],
    deadline: &Option<SystemTime>,
) -> SkipConnectionScan {
    detect_skip_connection_candidates_observed(creature, neuron_records, deadline, || {})
}

/// Detect skip-connection candidates, bounded by `deadline`, with a step observer (Issue #2368).
///
/// `on_step` is the deterministic work counter used by step-count tests. It is called once per
/// neuron popped in the depth pass, once per synapse relaxed in the depth pass, and once per
/// candidate source examined in the target loop.
///
/// # Arguments
/// * `creature` - The creature's network topology.
/// * `neuron_records` - List of `(neuron_uuid, records)` tuples with recorded data.
/// * `deadline` - When set, the scan stops early once the deadline elapses or cancellation is
///   requested.
/// * `on_step` - Called once per unit of work (see above).
///
/// # Returns
/// A [`SkipConnectionScan`] as for [`detect_skip_connection_candidates_with_deadline`].
pub fn detect_skip_connection_candidates_observed(
    creature: &CreatureJson,
    neuron_records: &[(String, impl AsRef<[DiscoverRecord]>)],
    deadline: &Option<SystemTime>,
    mut on_step: impl FnMut(),
) -> SkipConnectionScan {
    let hidden_uuids: HashSet<&str> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "hidden")
        .map(|n| n.uuid.as_str())
        .collect();

    if hidden_uuids.is_empty() {
        return empty_scan(None);
    }

    // Build records lookup
    let records_map = build_record_map(neuron_records);

    // Only consider hidden neurons with sufficient samples
    let qualified_hidden: Vec<&str> = hidden_uuids
        .iter()
        .filter(|&&uuid| {
            records_map
                .get(uuid)
                .is_some_and(|r| r.len() >= MIN_DISCOVERY_SAMPLE_COUNT)
        })
        .copied()
        .collect();

    if qualified_hidden.is_empty() {
        return empty_scan(None);
    }

    // Compute topological depth from inputs (Issue #2368: Kahn pass, O(V + E)).
    let depths = match compute_depths_from_inputs(creature, deadline, &mut on_step) {
        DepthPass::Depths(depths) => depths,
        DepthPass::CycleDetected => return empty_scan(Some(SkipConnectionStop::CycleDetected)),
        DepthPass::DeadlinePassed => return empty_scan(Some(SkipConnectionStop::DeadlinePassed)),
    };

    // Compute mean absolute error for each qualified hidden neuron
    let mean_errors: HashMap<&str, f32> = qualified_hidden
        .iter()
        .filter_map(|&uuid| {
            let records = records_map.get(uuid)?;
            let total_err: f32 = records
                .iter()
                .flat_map(|r| r.errors.iter())
                .map(|e| e.abs())
                .sum();
            let count = records.iter().map(|r| r.errors.len()).sum::<usize>() as f32;
            if count > 0.0 {
                Some((uuid, total_err / count))
            } else {
                Some((uuid, 0.0))
            }
        })
        .collect();

    // Compute shallow mean error (neurons at depth <= 2)
    let shallow_errors: Vec<f32> = qualified_hidden
        .iter()
        .filter(|&&uuid| depths.get(uuid).copied().unwrap_or(0) <= 2)
        .filter_map(|&uuid| mean_errors.get(uuid).copied())
        .filter(|&e| e > 0.0)
        .collect();

    if shallow_errors.is_empty() {
        return empty_scan(None);
    }

    let shallow_mean_error: f32 = shallow_errors.iter().sum::<f32>() / shallow_errors.len() as f32;

    if shallow_mean_error <= 0.0 {
        return empty_scan(None);
    }

    // Find deep neurons with attenuated gradients
    let deep_attenuated: Vec<(&str, usize, f32)> = qualified_hidden
        .iter()
        .filter_map(|&uuid| {
            let depth = depths.get(uuid).copied()?;
            if depth < MIN_DEEP_DEPTH {
                return None;
            }
            let mean_err = mean_errors.get(uuid).copied()?;
            let attenuation_ratio = mean_err / shallow_mean_error;
            if attenuation_ratio < ATTENUATION_THRESHOLD {
                Some((uuid, depth, mean_err))
            } else {
                None
            }
        })
        .collect();

    if deep_attenuated.is_empty() {
        return empty_scan(None);
    }

    // Build set of existing synapses for deduplication
    let existing_synapses: HashSet<(&str, &str)> = creature
        .synapses
        .iter()
        .map(|s| (s.from_uuid.as_str(), s.to_uuid.as_str()))
        .collect();

    // Split shallow sources (inputs and hidden at depth <= 1) by depth, in creature order.
    // Issue #2368: a depth-0 source always gives a larger gap than a depth-1 source for a
    // deep target (depth >= MIN_DEEP_DEPTH), so each target takes the first unconnected
    // depth-0 source, else the first unconnected depth-1 source.
    let mut depth0: Vec<&str> = Vec::new();
    let mut depth1: Vec<&str> = Vec::new();
    for n in &creature.neurons {
        if n.neuron_type != "input" && n.neuron_type != "hidden" {
            continue;
        }
        match depths.get(n.uuid.as_str()).copied().unwrap_or(0) {
            0 => depth0.push(n.uuid.as_str()),
            1 => depth1.push(n.uuid.as_str()),
            _ => {}
        }
    }

    let mut stop: Option<SkipConnectionStop> = None;
    let mut candidates = Vec::with_capacity(deep_attenuated.len());

    for &(target_uuid, target_depth, target_mean_err) in &deep_attenuated {
        // Issue #2368: checked once per deep target. A stop keeps the candidates gathered so far.
        if deadline_passed(deadline) {
            stop = Some(SkipConnectionStop::DeadlinePassed);
            break;
        }

        // Best source: largest depth gap, not already connected. Work per target is bounded
        // by its existing in-edges plus one.
        let mut best: Option<(&str, usize)> = None;
        for &source_uuid in &depth0 {
            on_step();
            if !existing_synapses.contains(&(source_uuid, target_uuid)) {
                best = Some((source_uuid, 0));
                break;
            }
        }
        if best.is_none() {
            for &source_uuid in &depth1 {
                on_step();
                if !existing_synapses.contains(&(source_uuid, target_uuid)) {
                    best = Some((source_uuid, 1));
                    break;
                }
            }
        }

        if let Some((source_uuid, source_depth)) = best {
            let best_gap = target_depth.saturating_sub(source_depth);
            // Estimated improvement: depth gap × attenuation severity × factor
            let attenuation_severity = 1.0 - (target_mean_err / shallow_mean_error);
            let estimated_improvement = best_gap as f32 * attenuation_severity * 0.005;

            if estimated_improvement > 0.0 {
                candidates.push(SkipConnectionCandidate {
                    target_uuid: target_uuid.to_string(),
                    target_depth,
                    source_uuid: source_uuid.to_string(),
                    source_depth,
                    depth_gap: best_gap,
                    target_mean_error: target_mean_err,
                    estimated_improvement,
                });
            }
        }
    }

    // Sort by estimated improvement (best first), using depth_gap as tiebreaker
    candidates.sort_by(|a, b| {
        b.estimated_improvement
            .total_cmp(&a.estimated_improvement)
            .then_with(|| b.depth_gap.cmp(&a.depth_gap))
    });

    SkipConnectionScan { candidates, stop }
}

/// Convert skip-connection candidates into coordinated structural candidates.
///
/// Each candidate becomes an `addSynapse` operation with a conservative
/// initial weight near zero to avoid disrupting existing network behaviour.
pub fn skip_connections_to_coordinated_candidates(
    candidates: &[SkipConnectionCandidate],
    creature: &CreatureJson,
) -> Vec<CoordinatedStructuralCandidateJson> {
    let existing_synapses: HashSet<(&str, &str)> = creature
        .synapses
        .iter()
        .map(|s| (s.from_uuid.as_str(), s.to_uuid.as_str()))
        .collect();

    let mut results = Vec::with_capacity(candidates.len());

    for c in candidates {
        // Skip if synapse already exists
        if existing_synapses.contains(&(c.source_uuid.as_str(), c.target_uuid.as_str())) {
            continue;
        }

        // Conservative weight: small and proportional to attenuation
        let weight = 0.01 * c.target_mean_error.min(1.0);

        results.push(CoordinatedStructuralCandidateJson {
            remove_neuron_compensation: None,
        constant_neuron_bias_fold: None,
            operations: vec![CoordinatedStructuralOpJson::AddSynapse {
                from_neuron_uuid: c.source_uuid.clone(),
                to_neuron_uuid: c.target_uuid.clone(),
                weight,
            }],
            expected_creature_score_gain: c.estimated_improvement,
            comment: Some(format!(
                "Skip connection: {} (depth {}) → {} (depth {}), gap={}, mean error {:.4}, attenuation detected",
                c.source_uuid, c.source_depth, c.target_uuid, c.target_depth,
                c.depth_gap, c.target_mean_error
            )),
        });
    }

    results
}

#[cfg(test)]
#[path = "skip_connection_deadline_tests.rs"]
mod skip_connection_deadline_tests;
