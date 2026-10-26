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
//!    inputs using a single-pass longest-path traversal (see Scan bounds).
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
//! ## Scan bounds (Issue #2368)
//!
//! - **Depth pass**: depths are computed in a single O(V+E) Kahn-style
//!   longest-path pass. Nodes are processed only once their in-degree reaches
//!   zero, so there is no re-queueing. A cyclic synapse graph has no
//!   well-defined depths: the detector fails closed, returning no candidates
//!   and setting [`SkipConnectionScan::depth_cycle_detected`].
//! - **Source lookup**: shallow sources (depth 0 and depth 1) are indexed once
//!   in creature order. Each deep target scans them only until the first
//!   unconnected source, so its cost is bounded by its in-degree.
//! - **Deadline**: [`detect_skip_connection_candidates_with_deadline`] checks the
//!   shared analysis deadline / global cancellation once per deep target and
//!   returns the candidates found so far, flagged via
//!   [`SkipConnectionScan::truncation`].

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use std::collections::{HashMap, HashSet, VecDeque};

use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson};

use crate::analysis::constants::MIN_DISCOVERY_SAMPLE_COUNT;
use crate::analysis::recommendation::epistatic::ScanTruncation;
use crate::analysis::utils::deadline_passed;

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

/// Result of a deadline-bounded skip-connection scan (Issue #2368).
#[derive(Debug, Clone)]
pub struct SkipConnectionScan {
    /// Candidates found before the scan finished or stopped, sorted by
    /// estimated improvement (best first). Empty when the synapse graph
    /// contains a cycle.
    pub candidates: Vec<SkipConnectionCandidate>,
    /// `Some` when the scan stopped before visiting every deep target.
    pub truncation: Option<ScanTruncation>,
    /// `true` when the synapse graph contains a cycle, so depths are undefined
    /// and no candidates are proposed (fail closed).
    pub depth_cycle_detected: bool,
}

/// Result of the single-pass depth computation (Issue #2368).
#[derive(Debug)]
struct DepthScan {
    /// Longest-path depth from inputs for each reached neuron UUID. Neurons
    /// unreachable from any input have no entry.
    depths: HashMap<String, usize>,
    /// `true` when the synapse graph contains a cycle. `depths` is incomplete
    /// in that case and must not be used.
    cycle_detected: bool,
}

/// Intern a UUID as a dense node index, recording its name on first sight.
fn intern_uuid<'a>(
    index: &mut HashMap<&'a str, usize>,
    names: &mut Vec<&'a str>,
    uuid: &'a str,
) -> usize {
    *index.entry(uuid).or_insert_with(|| {
        names.push(uuid);
        names.len() - 1
    })
}

/// Compute topological depth of each neuron from inputs in a single
/// Kahn-style pass (Issue #2368).
///
/// Input neurons start at depth 0. A neuron's depth is the longest path from
/// any input: the maximum of `predecessor depth + 1` over predecessors that
/// have a depth. Neurons unreachable from any input get no entry. Runs in
/// O(V + E). A cycle leaves some nodes unprocessed, which is reported through
/// `cycle_detected`.
fn compute_depths_from_inputs(creature: &CreatureJson) -> DepthScan {
    let mut index: HashMap<&str, usize> = HashMap::new();
    let mut names: Vec<&str> = Vec::new();

    let mut input_nodes: Vec<usize> = Vec::new();
    for n in &creature.neurons {
        let id = intern_uuid(&mut index, &mut names, n.uuid.as_str());
        if n.neuron_type == "input" {
            input_nodes.push(id);
        }
    }

    // Synapse endpoints not declared in `creature.neurons` are still interned
    // as nodes, so they receive depths exactly as before.
    let edges: Vec<(usize, usize)> = creature
        .synapses
        .iter()
        .map(|s| {
            let from = intern_uuid(&mut index, &mut names, s.from_uuid.as_str());
            let to = intern_uuid(&mut index, &mut names, s.to_uuid.as_str());
            (from, to)
        })
        .collect();

    let node_count = names.len();
    let mut out_edges: Vec<Vec<usize>> = vec![Vec::new(); node_count];
    let mut in_degree: Vec<usize> = vec![0; node_count];
    for &(from, to) in &edges {
        out_edges[from].push(to);
        in_degree[to] += 1;
    }

    // `depth[i]` is `Some` once node `i` is known to be reachable from an input.
    let mut depth: Vec<Option<usize>> = vec![None; node_count];
    for &id in &input_nodes {
        depth[id] = Some(0);
    }

    // A node is processed only once every predecessor has been processed, so
    // its depth is final when it is popped.
    let mut queue: VecDeque<usize> = (0..node_count).filter(|&i| in_degree[i] == 0).collect();
    let mut processed: usize = 0;
    while let Some(node) = queue.pop_front() {
        processed += 1;
        for &succ in &out_edges[node] {
            if let Some(d) = depth[node] {
                let candidate = d + 1;
                depth[succ] = Some(depth[succ].map_or(candidate, |cur| cur.max(candidate)));
            }
            in_degree[succ] -= 1;
            if in_degree[succ] == 0 {
                queue.push_back(succ);
            }
        }
    }

    let depths: HashMap<String, usize> = names
        .iter()
        .zip(depth.iter().copied())
        .filter_map(|(name, d)| d.map(|d| ((*name).to_string(), d)))
        .collect();

    DepthScan {
        depths,
        cycle_detected: processed < node_count,
    }
}

/// Detect skip-connection candidates based on gradient attenuation in deep neurons.
///
/// Applies no deadline (Issue #2368); equivalent to calling
/// [`detect_skip_connection_candidates_with_deadline`] with a `None` deadline.
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

/// Deadline- and cancellation-aware variant of [`detect_skip_connection_candidates`]
/// (Issue #2368).
///
/// Checks `deadline` once per deep attenuated target, so the scan stops promptly
/// on an elapsed deadline or global cancellation and keeps the candidates found
/// so far. A creature whose synapse graph contains a cycle has no well-defined
/// depths, so it yields no candidates and sets `depth_cycle_detected`.
pub fn detect_skip_connection_candidates_with_deadline(
    creature: &CreatureJson,
    neuron_records: &[(String, impl AsRef<[DiscoverRecord]>)],
    deadline: &Option<std::time::SystemTime>,
) -> SkipConnectionScan {
    let no_candidates = || SkipConnectionScan {
        candidates: Vec::new(),
        truncation: None,
        depth_cycle_detected: false,
    };

    let hidden_uuids: HashSet<&str> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "hidden")
        .map(|n| n.uuid.as_str())
        .collect();

    if hidden_uuids.is_empty() {
        return no_candidates();
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
        return no_candidates();
    }

    // Compute topological depth from inputs
    let depth_scan = compute_depths_from_inputs(creature);
    if depth_scan.cycle_detected {
        return SkipConnectionScan {
            candidates: Vec::new(),
            truncation: None,
            depth_cycle_detected: true,
        };
    }
    let depths = depth_scan.depths;

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
        return no_candidates();
    }

    let shallow_mean_error: f32 = shallow_errors.iter().sum::<f32>() / shallow_errors.len() as f32;

    if shallow_mean_error <= 0.0 {
        return no_candidates();
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
        return no_candidates();
    }

    // Build set of existing synapses for deduplication
    let existing_synapses: HashSet<(&str, &str)> = creature
        .synapses
        .iter()
        .map(|s| (s.from_uuid.as_str(), s.to_uuid.as_str()))
        .collect();

    // Index shallow source neurons (inputs and hidden at depth <= 1) by depth,
    // in creature order. Unreached neurons count as depth 0, as before.
    let mut shallow_depth0: Vec<&str> = Vec::new();
    let mut shallow_depth1: Vec<&str> = Vec::new();
    for n in creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "input" || n.neuron_type == "hidden")
    {
        match depths.get(n.uuid.as_str()).copied().unwrap_or(0) {
            0 => shallow_depth0.push(n.uuid.as_str()),
            1 => shallow_depth1.push(n.uuid.as_str()),
            _ => {}
        }
    }

    let mut candidates = Vec::with_capacity(deep_attenuated.len());
    let mut truncation: Option<ScanTruncation> = None;

    for &(target_uuid, target_depth, target_mean_err) in &deep_attenuated {
        if deadline_passed(deadline) {
            truncation = Some(ScanTruncation::DeadlinePassed);
            break;
        }

        // Best shallow source: the largest gap wins. A depth-0 source gives gap
        // `target_depth`, which beats any depth-1 source (gap `target_depth - 1`).
        // Within each depth, the first unconnected source in creature order wins.
        let best_source: Option<(&str, usize)> = shallow_depth0
            .iter()
            .copied()
            .find(|&s| !existing_synapses.contains(&(s, target_uuid)))
            .map(|s| (s, 0_usize))
            .or_else(|| {
                shallow_depth1
                    .iter()
                    .copied()
                    .find(|&s| !existing_synapses.contains(&(s, target_uuid)))
                    .map(|s| (s, 1_usize))
            });

        if let Some((source_uuid, source_depth)) = best_source {
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

    SkipConnectionScan {
        candidates,
        truncation,
        depth_cycle_detected: false,
    }
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
#[path = "skip_connection_tests.rs"]
mod tests;
