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
//!    inputs using forward BFS.
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

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use std::collections::{HashMap, HashSet, VecDeque};

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

/// Compute topological depth of each neuron from inputs using forward BFS.
///
/// Input neurons have depth 0. Each subsequent layer increments by 1.
/// Returns a map from neuron UUID to its depth.
fn compute_depths_from_inputs(creature: &CreatureJson) -> HashMap<String, usize> {
    // Build forward adjacency: from_uuid → [to_uuid, ...]
    let mut forward_adj: HashMap<&str, Vec<&str>> = HashMap::new();
    for s in &creature.synapses {
        forward_adj
            .entry(s.from_uuid.as_str())
            .or_default()
            .push(s.to_uuid.as_str());
    }

    let mut depths: HashMap<String, usize> = HashMap::new();
    let mut queue: VecDeque<(&str, usize)> = VecDeque::new();

    // Seed: input neurons at depth 0
    for n in &creature.neurons {
        if n.neuron_type == "input" {
            depths.insert(n.uuid.clone(), 0);
            queue.push_back((n.uuid.as_str(), 0));
        }
    }

    // BFS forward through the graph, taking maximum depth for each neuron
    // (so depth reflects the longest path from any input, giving true layer depth)
    while let Some((current, depth)) = queue.pop_front() {
        if let Some(successors) = forward_adj.get(current) {
            for &succ in successors {
                let new_depth = depth + 1;
                let entry = depths.entry(succ.to_string()).or_insert(0);
                if new_depth > *entry {
                    *entry = new_depth;
                    queue.push_back((succ, new_depth));
                }
            }
        }
    }

    depths
}

/// Detect skip-connection candidates based on gradient attenuation in deep neurons.
///
/// # Arguments
/// * `creature` - The creature's network topology.
/// * `neuron_records` - List of `(neuron_uuid, records)` tuples with recorded data.
///
/// # Returns
/// A list of `SkipConnectionCandidate` sorted by estimated improvement (best first).
pub fn detect_skip_connection_candidates(
    creature: &CreatureJson,
    neuron_records: &[(String, Vec<DiscoverRecord>)],
) -> Vec<SkipConnectionCandidate> {
    let hidden_uuids: HashSet<&str> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "hidden")
        .map(|n| n.uuid.as_str())
        .collect();

    if hidden_uuids.is_empty() {
        return Vec::new();
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
        return Vec::new();
    }

    // Compute topological depth from inputs
    let depths = compute_depths_from_inputs(creature);

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
        return Vec::new();
    }

    let shallow_mean_error: f32 = shallow_errors.iter().sum::<f32>() / shallow_errors.len() as f32;

    if shallow_mean_error <= 0.0 {
        return Vec::new();
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
        return Vec::new();
    }

    // Build set of existing synapses for deduplication
    let existing_synapses: HashSet<(&str, &str)> = creature
        .synapses
        .iter()
        .map(|s| (s.from_uuid.as_str(), s.to_uuid.as_str()))
        .collect();

    // Collect shallow source neurons (inputs and hidden at depth <= 1)
    let shallow_sources: Vec<(&str, usize)> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "input" || n.neuron_type == "hidden")
        .filter_map(|n| {
            let depth = depths.get(n.uuid.as_str()).copied().unwrap_or(0);
            if depth <= 1 {
                Some((n.uuid.as_str(), depth))
            } else {
                None
            }
        })
        .collect();

    let mut candidates = Vec::with_capacity(deep_attenuated.len());

    for &(target_uuid, target_depth, target_mean_err) in &deep_attenuated {
        // Find the best shallow source: prefer largest depth gap, not already connected
        let mut best_source: Option<(&str, usize)> = None;
        let mut best_gap: usize = 0;

        for &(source_uuid, source_depth) in &shallow_sources {
            if existing_synapses.contains(&(source_uuid, target_uuid)) {
                continue;
            }
            let gap = target_depth.saturating_sub(source_depth);
            if gap > best_gap {
                best_gap = gap;
                best_source = Some((source_uuid, source_depth));
            }
        }

        if let Some((source_uuid, source_depth)) = best_source {
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

    candidates
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
