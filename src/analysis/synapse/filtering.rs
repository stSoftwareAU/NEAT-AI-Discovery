//! Candidate filtering, deduplication, and truncation
//!
//! This module contains functions for truncating combined candidate sets,
//! generating deterministic UUIDs for coordinated candidates, and computing
//! expected gains for replace-synapse-with-neuron operations.

use crate::CandidateSynapseJson;
use crate::analysis::activation::{
    absolute_activation, arctan_activation, bent_identity_activation, bipolar_activation,
    clipped_activation, elu_activation, gelu_activation, hard_tanh_activation, identity_activation,
    logistic_activation, mish_activation, relu6_activation, softplus_activation,
    softsign_activation, tanh_activation,
};
use crate::analysis::cache::RecordCache;
use crate::analysis::diagnostics::TargetMap;
use crate::analysis::samples::EPSILON;

use super::scoring::{
    compute_activation_improvement_and_count, compute_relu_improvement_and_count,
};

// =============================================================================
// Coordinated Structural Helpers
// =============================================================================

/// Deterministic UUID generator for coordinated structural `addNeuron` operations.
pub(crate) fn deterministic_coordinated_neuron_uuid(
    source_uuid: &str,
    target_uuid: &str,
    squash: &str,
    incoming_weight: f32,
    outgoing_weight: f32,
    bias: f32,
) -> String {
    let key = format!(
        "replace-synapse-with-neuron|{source_uuid}|{target_uuid}|{squash}|{incoming_weight:.6}|{outgoing_weight:.6}|{bias:.6}"
    );
    // FNV-1a 64-bit
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in key.as_bytes() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("coordinated-hidden-{hash:016x}")
}

/// Parameters for computing expected gain when replacing a synapse with a
/// hidden neuron.
pub(crate) struct ReplaceSynapseParams<'a> {
    pub cache: &'a RecordCache,
    pub source_uuid: &'a str,
    pub target_uuid: &'a str,
    pub old_weight: f32,
    pub incoming_weight: f32,
    pub outgoing_weight: f32,
    pub bias: f32,
    pub squash: &'a str,
}

/// Compute expected gain for a coordinated "replace synapse with neuron" group.
///
/// This models the coordinated operation sequence:
/// 1) remove direct synapse (source -> target)
/// 2) add hidden neuron with (source -> newNeuron) and (newNeuron -> target)
pub(crate) fn expected_gain_replace_synapse_with_hidden_neuron(
    params: &ReplaceSynapseParams<'_>,
) -> Option<f32> {
    let from_records_arc = params.cache.get(params.source_uuid).ok()?;
    let target_records_arc = params.cache.get(params.target_uuid).ok()?;
    if from_records_arc.is_empty() || target_records_arc.is_empty() {
        return None;
    }

    let target_map = TargetMap::from_records(target_records_arc.as_ref());
    if target_map.map.is_empty() {
        return None;
    }

    let mut samples = target_map.build_samples_from(from_records_arc.as_ref());
    if samples.is_empty() {
        return None;
    }

    // Adjust baseline errors for the removal of the existing direct synapse.
    for s in &mut samples {
        s.avg_error += params.old_weight * s.activation;
    }

    let total_baseline_error_sq: f32 = samples.iter().map(|s| s.avg_error * s.avg_error).sum();
    if total_baseline_error_sq <= EPSILON {
        return None;
    }

    if params.squash.eq_ignore_ascii_case("ReLU") {
        let (improvement, _improved, _total) = compute_relu_improvement_and_count(
            samples.as_slice(),
            params.incoming_weight,
            params.outgoing_weight,
            params.bias,
            total_baseline_error_sq,
            None, // linear domain
        );
        return Some(improvement);
    }

    // Map squash names to activation functions
    let activation_fn: fn(f32) -> f32 = match params.squash {
        "GELU" => gelu_activation,
        "ELU" => elu_activation,
        "Softplus" => softplus_activation,
        "LOGISTIC" => logistic_activation,
        "TANH" => tanh_activation,
        "IDENTITY" => identity_activation,
        "BIPOLAR" => bipolar_activation,
        "CLIPPED" => clipped_activation,
        "ABSOLUTE" => absolute_activation,
        "Mish" => mish_activation,
        "HARD_TANH" => hard_tanh_activation,
        "Softsign" => softsign_activation,
        "BentIdentity" => bent_identity_activation,
        "Arctan" => arctan_activation,
        "ReLU6" => relu6_activation,
        _ => return None,
    };

    let (improvement, _improved, _total) = compute_activation_improvement_and_count(
        samples.as_slice(),
        params.incoming_weight,
        params.outgoing_weight,
        params.bias,
        activation_fn,
        total_baseline_error_sq,
        None, // linear domain
    );
    Some(improvement)
}

// =============================================================================
// Truncate Combined Synapse Candidate Sets
// =============================================================================

/// Truncate synapse candidate buckets to a global cap, preserving ordering semantics.
pub(crate) fn truncate_combined_synapse_candidate_sets(
    helpful: Vec<CandidateSynapseJson>,
    harmful: Vec<CandidateSynapseJson>,
    coordinated: Vec<crate::CoordinatedStructuralCandidateJson>,
    limit: usize,
    diversify: bool,
) -> (
    Vec<CandidateSynapseJson>,
    Vec<CandidateSynapseJson>,
    Vec<crate::CoordinatedStructuralCandidateJson>,
) {
    if limit == 0 {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    // If we're already under the global cap, keep the caller's ordering exactly.
    let total = helpful.len() + harmful.len() + coordinated.len();
    if total <= limit {
        return (helpful, harmful, coordinated);
    }

    // In diversified mode we intentionally preserve the per-bucket ordering
    if diversify {
        use std::collections::VecDeque;

        let mut helpful_q: VecDeque<CandidateSynapseJson> = VecDeque::from(helpful);
        let mut harmful_q: VecDeque<CandidateSynapseJson> = VecDeque::from(harmful);
        let mut coordinated_q: VecDeque<crate::CoordinatedStructuralCandidateJson> =
            VecDeque::from(coordinated);

        let mut helpful_out = Vec::new();
        let mut harmful_out = Vec::new();
        let mut coordinated_out = Vec::new();

        let mut returned = 0usize;
        while returned < limit {
            let mut progressed = false;

            if let Some(c) = helpful_q.pop_front() {
                helpful_out.push(c);
                returned += 1;
                progressed = true;
                if returned >= limit {
                    break;
                }
            }
            if let Some(c) = harmful_q.pop_front() {
                harmful_out.push(c);
                returned += 1;
                progressed = true;
                if returned >= limit {
                    break;
                }
            }
            if let Some(c) = coordinated_q.pop_front() {
                coordinated_out.push(c);
                returned += 1;
                progressed = true;
                if returned >= limit {
                    break;
                }
            }

            if !progressed {
                break;
            }
        }

        return (helpful_out, harmful_out, coordinated_out);
    }

    enum Any {
        Helpful(CandidateSynapseJson),
        Harmful(CandidateSynapseJson),
        Coordinated(crate::CoordinatedStructuralCandidateJson),
    }

    fn score(item: &Any) -> f32 {
        match item {
            Any::Helpful(c) => c.expected_creature_score_gain,
            Any::Harmful(c) => c.expected_creature_score_gain,
            Any::Coordinated(c) => c.expected_creature_score_gain,
        }
    }

    let mut combined: Vec<Any> = Vec::with_capacity(total);
    combined.extend(helpful.into_iter().map(Any::Helpful));
    combined.extend(harmful.into_iter().map(Any::Harmful));
    combined.extend(coordinated.into_iter().map(Any::Coordinated));

    combined.sort_by(|a, b| score(b).total_cmp(&score(a)));
    combined.truncate(limit);

    let mut helpful_out = Vec::new();
    let mut harmful_out = Vec::new();
    let mut coordinated_out = Vec::new();
    for item in combined {
        match item {
            Any::Helpful(c) => helpful_out.push(c),
            Any::Harmful(c) => harmful_out.push(c),
            Any::Coordinated(c) => coordinated_out.push(c),
        }
    }

    (helpful_out, harmful_out, coordinated_out)
}
