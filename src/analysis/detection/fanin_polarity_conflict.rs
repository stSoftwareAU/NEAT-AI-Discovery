//! Fan-in weight polarity conflict detector (Issue #641).
//!
//! Identifies hidden neurons whose incoming synapses have sharply conflicting
//! polarities — strongly positive weights fighting strongly negative weights.
//! When this happens, most of the input signal cancels out, wasting
//! representational capacity.
//!
//! ## How it differs from related modules
//!
//! - `opposing_synapse.rs` detects two synapses from the *same source* to the
//!   *same target* with opposing signs — a narrow, per-synapse case.
//! - `weight_coherence.rs` checks incoming/outgoing weight *ratio* consistency
//!   and symmetric cancellation of *correlated* sources.
//! - This module detects the general case: the entire fan-in is in fundamental
//!   polarity tension regardless of source identity or correlation.
//!
//! ## Detection
//!
//! For each hidden neuron:
//! 1. Partition incoming synapse weights by sign.
//! 2. Compute `conflict_score = min(|positive_sum|, |negative_sum|) / max(…)`.
//!    A score of 1.0 means perfect cancellation; 0.0 means no conflict.
//! 3. Flag neurons where the conflict score exceeds a threshold and both groups
//!    have meaningful total magnitude.
//!
//! ## Recommended action
//!
//! Produce a `coordinatedStructural` candidate with `addNeuron` to split the
//! positive and negative pathways into separate neurons, plus `addSynapse` ops
//! to wire the new neuron into the network.

use std::collections::{HashMap, HashSet};

use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson};

use crate::analysis::constants::MIN_DISCOVERY_SAMPLE_COUNT;

/// Minimum conflict score to flag a neuron (ratio of minority to majority group).
/// 0.4 means the smaller-magnitude group is at least 40% of the larger group.
const MIN_CONFLICT_SCORE: f32 = 0.4;

/// Minimum total absolute weight across all incoming synapses to consider
/// meaningful. Very small weights are not worth flagging.
const MIN_TOTAL_WEIGHT_MAGNITUDE: f32 = 1.0;

/// Minimum number of incoming synapses required for conflict analysis.
/// A neuron with fewer than 2 incoming synapses cannot have polarity conflict.
const MIN_INCOMING_SYNAPSES: usize = 2;

// =============================================================================
// Candidate type
// =============================================================================

/// A hidden neuron whose incoming synapses exhibit polarity conflict.
#[derive(Debug, Clone)]
pub struct FaninPolarityConflictCandidate {
    /// UUID of the conflicted hidden neuron.
    pub neuron_uuid: String,
    /// Sum of positive incoming weights.
    pub positive_weight_sum: f32,
    /// Sum of absolute negative incoming weights.
    pub negative_weight_sum: f32,
    /// Number of positive incoming synapses.
    pub positive_count: usize,
    /// Number of negative incoming synapses.
    pub negative_count: usize,
    /// Conflict score: min(pos_sum, neg_sum) / max(pos_sum, neg_sum).
    pub conflict_score: f32,
    /// Number of samples available for this neuron.
    pub sample_count: usize,
    /// Estimated improvement from reorganising the fan-in.
    pub estimated_improvement: f32,
}

// =============================================================================
// Detection
// =============================================================================

/// Detect hidden neurons with sharply conflicting incoming synapse polarities.
///
/// # Arguments
/// * `creature` - The creature's network topology.
/// * `neuron_records` - List of `(neuron_uuid, records)` tuples with recorded data.
///
/// # Returns
/// A list of `FaninPolarityConflictCandidate` sorted by conflict score (worst first).
pub fn detect_fanin_polarity_conflicts(
    creature: &CreatureJson,
    neuron_records: &[(String, Vec<DiscoverRecord>)],
) -> Vec<FaninPolarityConflictCandidate> {
    // Identify hidden neuron UUIDs
    let hidden_uuids: HashSet<&str> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "hidden")
        .map(|n| n.uuid.as_str())
        .collect();

    if hidden_uuids.is_empty() {
        return Vec::new();
    }

    // Build records lookup for sample count validation
    let records_map: HashMap<&str, usize> = neuron_records
        .iter()
        .map(|(uuid, recs)| (uuid.as_str(), recs.len()))
        .collect();

    // Group incoming synapses by target neuron
    let mut incoming_by_target: HashMap<&str, Vec<f32>> = HashMap::new();
    for synapse in &creature.synapses {
        if hidden_uuids.contains(synapse.to_uuid.as_str()) {
            incoming_by_target
                .entry(synapse.to_uuid.as_str())
                .or_default()
                .push(synapse.weight);
        }
    }

    let mut candidates = Vec::new();

    for (neuron_uuid, weights) in &incoming_by_target {
        // Need at least MIN_INCOMING_SYNAPSES incoming connections
        if weights.len() < MIN_INCOMING_SYNAPSES {
            continue;
        }

        // Need sufficient recorded samples
        let sample_count = records_map.get(neuron_uuid).copied().unwrap_or(0);
        if sample_count < MIN_DISCOVERY_SAMPLE_COUNT {
            continue;
        }

        // Partition by sign
        let positive_sum: f32 = weights.iter().filter(|w| **w > 0.0).sum();
        let negative_sum: f32 = weights.iter().filter(|w| **w < 0.0).map(|w| w.abs()).sum();

        // Both groups must have meaningful magnitude
        let total_magnitude = positive_sum + negative_sum;
        if total_magnitude < MIN_TOTAL_WEIGHT_MAGNITUDE {
            continue;
        }

        // Skip if either group is empty (no conflict)
        if positive_sum <= 0.0 || negative_sum <= 0.0 {
            continue;
        }

        let positive_count = weights.iter().filter(|w| **w > 0.0).count();
        let negative_count = weights.iter().filter(|w| **w < 0.0).count();

        // Conflict score: ratio of minority to majority group
        let conflict_score = positive_sum.min(negative_sum) / positive_sum.max(negative_sum);

        if conflict_score < MIN_CONFLICT_SCORE {
            continue;
        }

        // Estimated improvement scales with conflict severity and total magnitude
        let estimated_improvement = 0.01 * conflict_score * (total_magnitude / 10.0).min(1.0);

        candidates.push(FaninPolarityConflictCandidate {
            neuron_uuid: neuron_uuid.to_string(),
            positive_weight_sum: positive_sum,
            negative_weight_sum: negative_sum,
            positive_count,
            negative_count,
            conflict_score,
            sample_count,
            estimated_improvement,
        });
    }

    // Sort by conflict score descending (worst conflicts first)
    candidates.sort_by(|a, b| b.conflict_score.total_cmp(&a.conflict_score));

    candidates
}

// =============================================================================
// Candidate conversion
// =============================================================================

/// Convert fan-in polarity conflict candidates to coordinated structural candidates.
///
/// Each candidate proposes splitting the conflicting fan-in by adding a new hidden
/// neuron to handle the negative-polarity pathway, rewiring the negative incoming
/// synapses to the new neuron, and connecting the new neuron to the original target.
pub fn fanin_polarity_conflicts_to_coordinated_candidates(
    candidates: &[FaninPolarityConflictCandidate],
    creature: &CreatureJson,
) -> Vec<CoordinatedStructuralCandidateJson> {
    // Build incoming synapses lookup: target_uuid -> Vec<(from_uuid, weight)>
    let mut incoming_synapses: HashMap<&str, Vec<(&str, f32)>> = HashMap::new();
    for synapse in &creature.synapses {
        incoming_synapses
            .entry(synapse.to_uuid.as_str())
            .or_default()
            .push((synapse.from_uuid.as_str(), synapse.weight));
    }

    let mut results = Vec::with_capacity(candidates.len());

    for c in candidates {
        // Determine which group is the minority (to be split off)
        let split_negative = c.negative_weight_sum <= c.positive_weight_sum;

        // Get the synapses that will be rewired to the new neuron
        let synapses_for_neuron = incoming_synapses
            .get(c.neuron_uuid.as_str())
            .cloned()
            .unwrap_or_default();

        let synapses_to_move: Vec<(&str, f32)> = if split_negative {
            synapses_for_neuron
                .iter()
                .filter(|(_, w)| *w < 0.0)
                .copied()
                .collect()
        } else {
            synapses_for_neuron
                .iter()
                .filter(|(_, w)| *w > 0.0)
                .copied()
                .collect()
        };

        if synapses_to_move.is_empty() {
            continue;
        }

        // Determine the activation function from the original neuron
        let squash = creature
            .neurons
            .iter()
            .find(|n| n.uuid == c.neuron_uuid)
            .map_or_else(|| "TANH".to_string(), |n| n.squash.clone());

        // Generate a deterministic UUID for the new neuron
        let new_neuron_uuid = format!("fanin-split-{}", c.neuron_uuid);

        let mut operations = Vec::new();

        // 1. Add the new neuron (placed before the conflicted neuron)
        operations.push(CoordinatedStructuralOpJson::AddNeuron {
            neuron_uuid: new_neuron_uuid.clone(),
            neuron_type: "hidden".to_string(),
            squash,
            bias: 0.0,
            insert_before_neuron_uuid: Some(c.neuron_uuid.clone()),
        });

        // 2. Add synapses from the moved sources to the new neuron
        for (from_uuid, weight) in &synapses_to_move {
            operations.push(CoordinatedStructuralOpJson::AddSynapse {
                from_neuron_uuid: from_uuid.to_string(),
                to_neuron_uuid: new_neuron_uuid.clone(),
                weight: *weight,
            });
        }

        // 3. Connect the new neuron to the original target
        operations.push(CoordinatedStructuralOpJson::AddSynapse {
            from_neuron_uuid: new_neuron_uuid.clone(),
            to_neuron_uuid: c.neuron_uuid.clone(),
            weight: 1.0,
        });

        // 4. Remove the conflicting synapses from the original neuron
        for (from_uuid, _) in &synapses_to_move {
            operations.push(CoordinatedStructuralOpJson::RemoveSynapse {
                from_neuron_uuid: from_uuid.to_string(),
                to_neuron_uuid: c.neuron_uuid.clone(),
            });
        }

        results.push(CoordinatedStructuralCandidateJson {
            operations,
            expected_creature_score_gain: c.estimated_improvement,
            comment: Some(format!(
                "Fan-in polarity conflict at {}: {} positive synapses (sum={:.2}) vs {} negative synapses (sum={:.2}), conflict score {:.3} — split minority pathway into separate neuron",
                c.neuron_uuid,
                c.positive_count,
                c.positive_weight_sum,
                c.negative_count,
                c.negative_weight_sum,
                c.conflict_score
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
