//! Candidate diversity enforcement (Issue #610).
//!
//! Penalises structurally similar candidates so the evaluation budget is spread
//! across different regions of mutation space. When multiple candidates propose
//! similar structural changes (e.g., removing adjacent synapses to the same
//! target), diversity-aware reranking demotes lower-ranked similar candidates
//! in favour of structurally distinct alternatives.
//!
//! ## Algorithm
//!
//! 1. Sort candidates by `expected_creature_score_gain` (best first).
//! 2. For each candidate from position 1 onward, compute the maximum structural
//!    similarity to any already-accepted candidate.
//! 3. Penalise the candidate's effective score: `effective = score × (1 - penalty × similarity)`.
//! 4. Re-sort by effective score to produce the final ranking.
//!
//! ## Structural Similarity
//!
//! Two candidates are similar when they affect the same neurons/synapses with
//! the same operation types. The similarity metric considers:
//! - **Operation type overlap** — same op types score higher.
//! - **Affected neuron overlap** — targeting/sourcing the same neurons scores higher.
//! - Both factors are combined into a single [0.0, 1.0] score.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use std::collections::HashSet;

use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson};

/// Configuration for diversity-aware reranking.
#[derive(Debug, Clone)]
pub struct DiversityConfig {
    /// Strength of the diversity penalty (0.0 = no penalty, 1.0 = maximum penalty).
    ///
    /// At `penalty_strength = 1.0`, a candidate identical to a higher-ranked one
    /// has its effective score reduced to zero. At `0.5`, the effective score is
    /// halved for identical candidates.
    pub penalty_strength: f32,
}

impl Default for DiversityConfig {
    fn default() -> Self {
        Self {
            penalty_strength: 0.5,
        }
    }
}

/// Compute the structural similarity between two coordinated candidates.
///
/// Returns a value in [0.0, 1.0] where:
/// - 1.0 means identical structure (same operations, same neurons)
/// - 0.0 means completely unrelated structure
///
/// The metric combines:
/// - **Operation type overlap**: Jaccard similarity of op-type multisets.
/// - **Affected neuron overlap**: Jaccard similarity of affected neuron UUID sets.
pub fn compute_structural_similarity(
    a: &CoordinatedStructuralCandidateJson,
    b: &CoordinatedStructuralCandidateJson,
) -> f32 {
    let types_a = op_type_set(&a.operations);
    let types_b = op_type_set(&b.operations);
    let type_similarity = jaccard_similarity(&types_a, &types_b);

    let neurons_a = affected_neurons(&a.operations);
    let neurons_b = affected_neurons(&b.operations);
    let neuron_similarity = jaccard_similarity_str(&neurons_a, &neurons_b);

    // Weighted combination: neuron overlap is more important than op-type overlap
    // because two RemoveSynapse candidates targeting different parts of the network
    // are genuinely diverse despite sharing the same op type.
    const NEURON_WEIGHT: f32 = 0.7;
    const TYPE_WEIGHT: f32 = 0.3;

    NEURON_WEIGHT * neuron_similarity + TYPE_WEIGHT * type_similarity
}

/// Rerank candidates with a diversity penalty.
///
/// Candidates already sorted by `expected_creature_score_gain` are penalised
/// when structurally similar to higher-ranked candidates. The penalty reduces
/// each candidate's effective score proportional to its maximum similarity
/// to any already-accepted candidate.
///
/// All candidates are preserved (none are removed), but their order may change.
pub fn rerank_with_diversity(
    mut candidates: Vec<CoordinatedStructuralCandidateJson>,
    config: &DiversityConfig,
) -> Vec<CoordinatedStructuralCandidateJson> {
    if candidates.len() <= 1 || config.penalty_strength <= 0.0 {
        return candidates;
    }

    // Sort by original score (best first)
    candidates.sort_by(|a, b| {
        b.expected_creature_score_gain
            .total_cmp(&a.expected_creature_score_gain)
    });

    // Compute effective scores with diversity penalty.
    // For each candidate, find max similarity to any higher-ranked candidate.
    let n = candidates.len();
    let mut effective_scores: Vec<f32> = Vec::with_capacity(n);

    // First candidate always keeps its original score
    effective_scores.push(candidates[0].expected_creature_score_gain);

    for i in 1..n {
        let mut max_similarity: f32 = 0.0;
        for j in 0..i {
            let sim = compute_structural_similarity(&candidates[i], &candidates[j]);
            max_similarity = max_similarity.max(sim);
        }

        let original_score = candidates[i].expected_creature_score_gain;
        let penalty = config.penalty_strength * max_similarity;
        let effective = original_score * (1.0 - penalty);
        effective_scores.push(effective);
    }

    // Create index-score pairs and sort by effective score (best first)
    let mut indexed: Vec<(usize, f32)> = effective_scores.into_iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.total_cmp(&a.1));

    // Reorder candidates by effective score
    let mut reranked: Vec<CoordinatedStructuralCandidateJson> = Vec::with_capacity(n);
    // Use a placeholder swap approach to avoid double-borrow issues
    let mut slots: Vec<Option<CoordinatedStructuralCandidateJson>> =
        candidates.into_iter().map(Some).collect();

    for (idx, _score) in indexed {
        if let Some(candidate) = slots[idx].take() {
            reranked.push(candidate);
        }
    }

    reranked
}

/// Extract operation type tags from a list of operations.
fn op_type_set(ops: &[CoordinatedStructuralOpJson]) -> HashSet<&'static str> {
    ops.iter().map(op_type_tag).collect()
}

/// Get a tag string for an operation type.
fn op_type_tag(op: &CoordinatedStructuralOpJson) -> &'static str {
    match op {
        CoordinatedStructuralOpJson::RemoveSynapse { .. } => "RemoveSynapse",
        CoordinatedStructuralOpJson::AddSynapse { .. } => "AddSynapse",
        CoordinatedStructuralOpJson::AddNeuron { .. } => "AddNeuron",
        CoordinatedStructuralOpJson::RemoveNeuron { .. } => "RemoveNeuron",
        CoordinatedStructuralOpJson::ChangeSquash { .. } => "ChangeSquash",
        CoordinatedStructuralOpJson::SetBias { .. } => "SetBias",
        CoordinatedStructuralOpJson::SetWeight { .. } => "SetWeight",
    }
}

/// Collect the set of neuron UUIDs affected by a list of operations.
fn affected_neurons(ops: &[CoordinatedStructuralOpJson]) -> HashSet<String> {
    let mut neurons = HashSet::new();
    for op in ops {
        match op {
            CoordinatedStructuralOpJson::RemoveSynapse {
                from_neuron_uuid,
                to_neuron_uuid,
            }
            | CoordinatedStructuralOpJson::AddSynapse {
                from_neuron_uuid,
                to_neuron_uuid,
                ..
            }
            | CoordinatedStructuralOpJson::SetWeight {
                from_neuron_uuid,
                to_neuron_uuid,
                ..
            } => {
                neurons.insert(from_neuron_uuid.clone());
                neurons.insert(to_neuron_uuid.clone());
            }
            CoordinatedStructuralOpJson::AddNeuron { neuron_uuid, .. }
            | CoordinatedStructuralOpJson::RemoveNeuron { neuron_uuid }
            | CoordinatedStructuralOpJson::ChangeSquash { neuron_uuid, .. }
            | CoordinatedStructuralOpJson::SetBias { neuron_uuid, .. } => {
                neurons.insert(neuron_uuid.clone());
            }
        }
    }
    neurons
}

/// Jaccard similarity for `HashSet<&str>`.
fn jaccard_similarity(a: &HashSet<&str>, b: &HashSet<&str>) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let intersection = a.intersection(b).count() as f32;
    let union = a.union(b).count() as f32;
    if union == 0.0 {
        return 0.0;
    }
    intersection / union
}

/// Jaccard similarity for `HashSet<String>`.
fn jaccard_similarity_str(a: &HashSet<String>, b: &HashSet<String>) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let intersection = a.intersection(b).count() as f32;
    let union = a.union(b).count() as f32;
    if union == 0.0 {
        return 0.0;
    }
    intersection / union
}
