//! Candidate aggregation and post-processing (Issue #562).
//!
//! This module handles the post-processing steps after synapse and neuron analysis:
//! - Converting add-neuron candidates to coordinated structural replacements
//!   when a direct synapse already exists (Issue #173)
//! - Merging coordinated structural replacements into synapse results
//! - Truncating candidate sets to respect `max_synapse_candidates`

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use std::collections::HashMap;
use std::mem;
use std::sync::Arc;

use crate::{
    AnalyzeAllInput, CandidateNeuronJson, CoordinatedStructuralCandidateJson,
    CoordinatedStructuralOpJson,
};

use super::constants::{COORDINATED_OPERATION_DISCOUNT, MIN_COORDINATED_MULTI_OP_GAIN};
use super::{cache, shared, synapse};

/// Compute the operation-count discount for a coordinated candidate (Issue #732).
///
/// Multi-operation candidates suffer compounding prediction uncertainty.
/// Each additional operation beyond the first applies a multiplicative discount
/// of `COORDINATED_OPERATION_DISCOUNT`, so a 4-operation candidate receives
/// `COORDINATED_OPERATION_DISCOUNT^3` ≈ 0.512 discount.
///
/// Single-operation candidates receive no discount (returns original gain).
pub fn apply_operation_count_discount(candidate: &CoordinatedStructuralCandidateJson) -> f32 {
    let op_count = candidate.operations.len();
    if op_count <= 1 {
        return candidate.expected_creature_score_gain;
    }
    let exponent = (op_count - 1) as f32;
    candidate.expected_creature_score_gain * COORDINATED_OPERATION_DISCOUNT.powf(exponent)
}

/// Validate whether a coordinated candidate's gain exceeds the minimum threshold (Issue #732).
///
/// Single-operation candidates only require positive gain. Multi-operation
/// candidates (>= 2 operations) must exceed `MIN_COORDINATED_MULTI_OP_GAIN`
/// to filter out near-zero predictions that almost never succeed in practice.
pub fn validate_coordinated_candidate_gain(candidate: &CoordinatedStructuralCandidateJson) -> bool {
    if candidate.operations.len() <= 1 {
        return candidate.expected_creature_score_gain > 0.0;
    }
    let discounted = apply_operation_count_discount(candidate);
    discounted > MIN_COORDINATED_MULTI_OP_GAIN
}

/// Merge coordinated structural replacement candidates into the synapse result.
///
/// Filters out candidates with non-positive expected gain (Issue #557),
/// applies operation-count discount for multi-operation candidates (Issue #732),
/// sorts by expected gain (unless diversified), and truncates to the
/// combined synapse candidate limit.
pub(crate) fn merge_coordinated_structural_replacements(
    synapse: &mut shared::AnalyzeSynapsesResult,
    mut replacements: Vec<CoordinatedStructuralCandidateJson>,
    max_synapse_candidates: Option<usize>,
    diversify: bool,
) {
    if replacements.is_empty() {
        return;
    }

    // Issue #557: Filter out candidates with non-positive expected_creature_score_gain.
    // Only candidates predicted to improve the creature's score should be returned.
    replacements.retain(|c| c.expected_creature_score_gain > 0.0);
    if replacements.is_empty() {
        return;
    }

    // Issue #732: Apply operation-count discount and minimum gain validation.
    // Multi-operation candidates have compounding prediction uncertainty.
    for c in &mut replacements {
        if c.operations.len() > 1 {
            c.expected_creature_score_gain = apply_operation_count_discount(c);
        }
    }
    // After discounting, filter out candidates below the minimum gain threshold.
    // Note: we check the gain directly here since discounting has already been applied.
    replacements.retain(|c| {
        if c.operations.len() <= 1 {
            c.expected_creature_score_gain > 0.0
        } else {
            c.expected_creature_score_gain > MIN_COORDINATED_MULTI_OP_GAIN
        }
    });
    if replacements.is_empty() {
        return;
    }

    synapse
        .coordinated_structural_candidates
        .append(&mut replacements);

    // Keep deterministic ordering for non-deadline runs.
    //
    // Deadline-diversified runs rely on upstream per-bucket ordering and round-robin selection,
    // so we avoid resorting here when `diversify` is enabled.
    if !diversify {
        synapse.coordinated_structural_candidates.sort_by(|a, b| {
            b.expected_creature_score_gain
                .total_cmp(&a.expected_creature_score_gain)
        });
    }

    if let Some(limit) = max_synapse_candidates {
        let (helpful, harmful, coordinated) = synapse::truncate_combined_synapse_candidate_sets(
            mem::take(&mut synapse.helpful_synapses),
            mem::take(&mut synapse.harmful_synapses),
            mem::take(&mut synapse.coordinated_structural_candidates),
            limit,
            diversify,
        );
        synapse.helpful_synapses = helpful;
        synapse.harmful_synapses = harmful;
        synapse.coordinated_structural_candidates = coordinated;
    }

    // Ensure metadata reflects what we actually return to callers.
    synapse.metadata.candidates_returned = synapse.helpful_synapses.len()
        + synapse.harmful_synapses.len()
        + synapse.coordinated_structural_candidates.len();
}

/// Update neuron result after filtering out candidates converted to coordinated replacements.
pub(crate) fn apply_kept_neuron_candidates(
    neuron: &mut shared::AnalyzeNeuronsResult,
    kept_neurons: Vec<CandidateNeuronJson>,
) {
    // Regression fix (7-Jan-2026):
    // Post-processing may convert some add-neuron candidates into coordinated structural
    // replacements. When we filter those out of `helpful_neurons`, we must also update the
    // metadata so JSON output remains consistent with the returned arrays.
    neuron.helpful_neurons = kept_neurons;
    neuron.metadata.candidates_returned = neuron.helpful_neurons.len();
}

/// Convert eligible add-neuron candidates into coordinated structural replacements.
///
/// Issue #173 (7-Jan-2026): When a direct synapse already exists between (source -> target),
/// applying an add-neuron candidate without removing the synapse is often not the intended edit.
/// We instead emit a coordinated group that removes the synapse and inserts the hidden neuron
/// path atomically.
///
/// Normal add-neurons discovery works as-is for cases where no direct synapse exists.
pub(crate) fn convert_neurons_to_coordinated_replacements(
    input: &AnalyzeAllInput,
    syn: &mut shared::AnalyzeSynapsesResult,
    neuron: &mut shared::AnalyzeNeuronsResult,
    shared_cache: &Arc<cache::RecordCache>,
) {
    // Build a quick lookup from (from_uuid,to_uuid) to existing weight.
    let mut direct_synapse_weight: HashMap<(String, String), f32> = HashMap::new();
    for s in &input.creature.synapses {
        direct_synapse_weight.insert((s.from_uuid.clone(), s.to_uuid.clone()), s.weight);
    }

    let mut kept_neurons: Vec<CandidateNeuronJson> =
        Vec::with_capacity(neuron.helpful_neurons.len());
    let mut replacements: Vec<CoordinatedStructuralCandidateJson> = Vec::new();

    for candidate in &neuron.helpful_neurons {
        let key = (
            candidate.source_neuron_uuid.clone(),
            candidate.target_neuron_uuid.clone(),
        );
        let Some(&old_weight) = direct_synapse_weight.get(&key) else {
            kept_neurons.push(candidate.clone());
            continue;
        };

        let new_neuron_uuid = synapse::deterministic_coordinated_neuron_uuid(
            &candidate.source_neuron_uuid,
            &candidate.target_neuron_uuid,
            &candidate.squash,
            candidate.incoming_weight,
            candidate.outgoing_weight,
            candidate.bias,
        );

        let replace_params = synapse::ReplaceSynapseParams {
            cache: shared_cache.as_ref(),
            source_uuid: &candidate.source_neuron_uuid,
            target_uuid: &candidate.target_neuron_uuid,
            old_weight,
            incoming_weight: candidate.incoming_weight,
            outgoing_weight: candidate.outgoing_weight,
            bias: candidate.bias,
            squash: &candidate.squash,
        };
        let mut expected_gain =
            synapse::expected_gain_replace_synapse_with_hidden_neuron(&replace_params)
                .unwrap_or(candidate.expected_creature_score_gain);

        // Apply a conservative impact discount when the target is hidden.
        // This mirrors the synapse/neurons discounting semantics without requiring deep graph analysis.
        let is_target_output = input
            .creature
            .neurons
            .iter()
            .find(|n| n.uuid == candidate.target_neuron_uuid)
            .is_some_and(|n| n.neuron_type == "output");
        if !is_target_output {
            expected_gain *= 0.1;
        }

        replacements.push(CoordinatedStructuralCandidateJson {
            operations: vec![
                CoordinatedStructuralOpJson::RemoveSynapse {
                    from_neuron_uuid: candidate.source_neuron_uuid.clone(),
                    to_neuron_uuid: candidate.target_neuron_uuid.clone(),
                },
                CoordinatedStructuralOpJson::AddNeuron {
                    neuron_uuid: new_neuron_uuid.clone(),
                    neuron_type: "hidden".to_string(),
                    squash: candidate.squash.clone(),
                    bias: candidate.bias,
                    insert_before_neuron_uuid: Some(candidate.target_neuron_uuid.clone()),
                },
                CoordinatedStructuralOpJson::AddSynapse {
                    from_neuron_uuid: candidate.source_neuron_uuid.clone(),
                    to_neuron_uuid: new_neuron_uuid.clone(),
                    weight: candidate.incoming_weight,
                },
                CoordinatedStructuralOpJson::AddSynapse {
                    from_neuron_uuid: new_neuron_uuid,
                    to_neuron_uuid: candidate.target_neuron_uuid.clone(),
                    weight: candidate.outgoing_weight,
                },
            ],
            expected_creature_score_gain: expected_gain,
            comment: Some(format!(
                "Coordinated replacement: remove synapse and insert {} hidden neuron",
                candidate.squash
            )),
        });
    }

    // Keep non-replacement add-neurons unchanged.
    apply_kept_neuron_candidates(neuron, kept_neurons);

    merge_coordinated_structural_replacements(
        syn,
        replacements,
        input.max_synapse_candidates,
        input.analysis_deadline_ms.is_some(),
    );
}
