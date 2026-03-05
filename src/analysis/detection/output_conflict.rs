//! Per-output error disaggregation detector for hidden neurons (Issue #639).
//!
//! Identifies hidden neurons with conflicting per-output error contributions.
//! A hidden neuron can have a net positive effect when errors are summed, but
//! actively harm a specific output. For example, a hidden neuron might reduce
//! error on output-0 by 0.5 but increase error on output-1 by 0.3 — its net
//! effect looks positive, but it is actively harming output-1.
//!
//! ## Detection Method
//!
//! 1. For each hidden neuron, compute the mean error per output index across
//!    all observations using `errors[0..n]`.
//! 2. Check whether the sign of the mean error differs across outputs
//!    (some negative = helping, some positive = harming).
//! 3. Filter out weak conflicts where all per-output mean errors are near zero.
//! 4. Rank by conflict severity: the product of the largest positive and
//!    largest negative mean errors (higher magnitude = more severe).
//!
//! ## Recommended Actions
//!
//! When a conflicting hidden neuron is detected, we recommend:
//! 1. **Add a gating synapse**: Insert a gating synapse so the neuron's
//!    contribution to the harmed output is attenuated.
//! 2. **Split the neuron**: Clone the neuron into output-specific copies,
//!    each connected only to the outputs it helps.
//!
//! These are emitted as `CoordinatedStructuralCandidateJson` candidates.

use std::collections::HashSet;

use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson};

use crate::analysis::constants::MIN_DISCOVERY_SAMPLE_COUNT as MIN_SAMPLES;

/// Minimum absolute mean error per output to be considered significant.
/// Below this, the error contribution is treated as noise.
const MIN_SIGNIFICANT_ERROR: f32 = 0.01;

/// Result of detecting a hidden neuron with conflicting per-output errors.
#[derive(Debug, Clone)]
pub struct OutputConflictNeuron {
    /// UUID of the conflicting hidden neuron.
    pub neuron_uuid: String,
    /// Activation function of this neuron.
    pub squash: String,
    /// Bias of this neuron.
    pub bias: f32,
    /// Mean error contribution per output index.
    /// Negative means helping (reducing error), positive means harming.
    pub per_output_mean_error: Vec<f32>,
    /// Severity of the conflict: product of max positive and |max negative| mean errors.
    pub conflict_severity: f32,
    /// Number of samples analysed.
    pub sample_count: usize,
    /// Estimated creature score improvement from resolving the conflict.
    pub estimated_improvement: f32,
}

/// Detect hidden neurons with conflicting per-output error contributions.
///
/// Analyses the full `errors` vector (not just `errors[0]`) for each hidden
/// neuron to find cases where the neuron helps some outputs but harms others.
///
/// # Arguments
/// * `creature` - The creature's network topology.
/// * `neuron_records` - List of `(neuron_uuid, records)` tuples.
///
/// # Returns
/// A list of `OutputConflictNeuron` sorted by conflict severity (worst first).
pub fn detect_output_conflict_neurons(
    creature: &CreatureJson,
    neuron_records: &[(String, Vec<DiscoverRecord>)],
) -> Vec<OutputConflictNeuron> {
    // Need at least 2 outputs for cross-output conflict
    if creature.output < 2 {
        return Vec::new();
    }

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

    // Build a lookup from UUID to neuron info
    let neuron_info: std::collections::HashMap<&str, (&str, f32)> = creature
        .neurons
        .iter()
        .map(|n| (n.uuid.as_str(), (n.squash.as_str(), n.bias)))
        .collect();

    let mut results: Vec<OutputConflictNeuron> = Vec::with_capacity(neuron_records.len());

    for (uuid, records) in neuron_records {
        // Only analyse hidden neurons
        if !hidden_uuids.contains(uuid.as_str()) {
            continue;
        }

        // Need sufficient samples
        if records.len() < MIN_SAMPLES {
            continue;
        }

        // Determine the number of output error channels from the records.
        // Use the first record with non-empty errors to determine channel count.
        let num_outputs = match records.iter().find(|r| !r.errors.is_empty()) {
            Some(r) => r.errors.len(),
            None => continue,
        };

        // Need at least 2 error channels for conflict detection
        if num_outputs < 2 {
            continue;
        }

        // Compute mean error per output index
        let mut sum_errors = vec![0.0_f32; num_outputs];
        let mut count = 0_usize;

        for r in records {
            if r.errors.len() >= num_outputs {
                for (idx, &err) in r.errors.iter().enumerate().take(num_outputs) {
                    sum_errors[idx] += err;
                }
                count += 1;
            }
        }

        if count < MIN_SAMPLES {
            continue;
        }

        let mean_errors: Vec<f32> = sum_errors.iter().map(|&s| s / count as f32).collect();

        // Check for sign conflict: at least one negative and one positive mean error
        let has_negative = mean_errors.iter().any(|&e| e < -MIN_SIGNIFICANT_ERROR);
        let has_positive = mean_errors.iter().any(|&e| e > MIN_SIGNIFICANT_ERROR);

        if !has_negative || !has_positive {
            continue;
        }

        // Compute conflict severity: max positive × |min negative|
        let max_positive = mean_errors
            .iter()
            .copied()
            .filter(|&e| e > 0.0)
            .fold(0.0_f32, f32::max);
        let max_negative_abs = mean_errors
            .iter()
            .copied()
            .filter(|&e| e < 0.0)
            .map(f32::abs)
            .fold(0.0_f32, f32::max);

        let conflict_severity = max_positive * max_negative_abs;

        // Look up neuron info
        let (squash, bias) = neuron_info
            .get(uuid.as_str())
            .copied()
            .unwrap_or(("IDENTITY", 0.0));

        // Estimated improvement: addressing the conflict could reduce the harmed output's
        // error contribution. Use the mean of the positive (harmful) errors as estimate.
        let harmful_mean: f32 = mean_errors.iter().filter(|&&e| e > 0.0).sum::<f32>()
            / mean_errors.iter().filter(|&&e| e > 0.0).count().max(1) as f32;
        let estimated_improvement = harmful_mean * 0.1; // conservative 10% capture

        results.push(OutputConflictNeuron {
            neuron_uuid: uuid.clone(),
            squash: squash.to_string(),
            bias,
            per_output_mean_error: mean_errors,
            conflict_severity,
            sample_count: count,
            estimated_improvement,
        });
    }

    // Sort by conflict severity (worst first)
    results.sort_by(|a, b| b.conflict_severity.total_cmp(&a.conflict_severity));

    results
}

/// Generate a deterministic UUID for a split neuron.
fn split_neuron_uuid(original_uuid: &str, output_index: usize) -> String {
    let key = format!("output-conflict-split|{original_uuid}|{output_index}");
    // FNV-1a 64-bit
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in key.as_bytes() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("ocs-{hash:016x}")
}

/// Convert output conflict detections into coordinated structural candidates.
///
/// For each conflicting neuron, produces a candidate that adds a gating synapse
/// to attenuate the neuron's contribution to the harmed output(s).
///
/// # Arguments
/// * `conflicts` - Detected output conflict neurons.
/// * `creature` - The creature topology.
pub fn output_conflicts_to_coordinated_candidates(
    conflicts: &[OutputConflictNeuron],
    creature: &CreatureJson,
) -> Vec<CoordinatedStructuralCandidateJson> {
    // Find output neuron UUIDs in order
    let output_uuids: Vec<&str> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "output")
        .map(|n| n.uuid.as_str())
        .collect();

    // Find the first output neuron UUID for insert_before placement
    let first_output_uuid = output_uuids.first().map(std::string::ToString::to_string);

    let mut results = Vec::new();

    for conflict in conflicts {
        // Strategy: add a new gating neuron that routes the original neuron's
        // output only to the helped outputs, bypassing the harmed ones.
        // We create a split: a new hidden neuron connected only to the harmed outputs
        // with a corrective weight.

        let mut operations = Vec::new();

        // Identify which outputs are harmed (positive mean error)
        let harmed_outputs: Vec<(usize, f32)> = conflict
            .per_output_mean_error
            .iter()
            .enumerate()
            .filter(|(_, e)| **e > MIN_SIGNIFICANT_ERROR)
            .map(|(i, &e)| (i, e))
            .collect();

        // Find which existing synapses connect this neuron to the harmed outputs
        let existing_synapses: Vec<&crate::SynapseJson> = creature
            .synapses
            .iter()
            .filter(|s| s.from_uuid == conflict.neuron_uuid)
            .collect();

        // For each harmed output, adjust the synapse weight to reduce the harm
        for &(output_idx, harm_magnitude) in &harmed_outputs {
            if output_idx >= output_uuids.len() {
                continue;
            }
            let output_uuid = output_uuids[output_idx];

            // Check if there's a direct synapse from the conflicting neuron to this output
            if let Some(syn) = existing_synapses.iter().find(|s| s.to_uuid == output_uuid) {
                // Reduce the weight toward zero to attenuate harm
                let reduced_weight = syn.weight * 0.3; // reduce to 30%
                operations.push(CoordinatedStructuralOpJson::SetWeight {
                    from_neuron_uuid: conflict.neuron_uuid.clone(),
                    to_neuron_uuid: output_uuid.to_string(),
                    weight: reduced_weight,
                });
            } else {
                // No direct synapse — add a compensating neuron between the
                // conflicting hidden neuron and the harmed output
                let new_uuid = split_neuron_uuid(&conflict.neuron_uuid, output_idx);

                operations.push(CoordinatedStructuralOpJson::AddNeuron {
                    neuron_uuid: new_uuid.clone(),
                    neuron_type: "hidden".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 0.0,
                    insert_before_neuron_uuid: first_output_uuid.clone(),
                });

                // Connect: conflicting neuron → new gating neuron
                operations.push(CoordinatedStructuralOpJson::AddSynapse {
                    from_neuron_uuid: conflict.neuron_uuid.clone(),
                    to_neuron_uuid: new_uuid.clone(),
                    weight: -harm_magnitude * 0.5,
                });

                // Connect: new gating neuron → harmed output
                operations.push(CoordinatedStructuralOpJson::AddSynapse {
                    from_neuron_uuid: new_uuid,
                    to_neuron_uuid: output_uuid.to_string(),
                    weight: 1.0,
                });
            }
        }

        if operations.is_empty() {
            continue;
        }

        let helped_outputs: Vec<usize> = conflict
            .per_output_mean_error
            .iter()
            .enumerate()
            .filter(|(_, e)| **e < -MIN_SIGNIFICANT_ERROR)
            .map(|(i, _)| i)
            .collect();

        let harmed_indices: Vec<usize> = harmed_outputs.iter().map(|&(i, _)| i).collect();

        results.push(CoordinatedStructuralCandidateJson {
            operations,
            expected_creature_score_gain: conflict.estimated_improvement,
            comment: Some(format!(
                "Output conflict on hidden neuron {}: helps output(s) {:?}, harms output(s) {:?} \
                 (severity {:.3}, {} samples)",
                conflict.neuron_uuid,
                helped_outputs,
                harmed_indices,
                conflict.conflict_severity,
                conflict.sample_count,
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
