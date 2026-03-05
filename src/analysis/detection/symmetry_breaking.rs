//! Symmetry-breaking detection module (Issue #569).
//!
//! Identifies pairs of hidden neurons whose weight configurations and activation
//! functions have converged to near-identical states, effectively reducing the
//! network's representational capacity. Two neurons computing the same function
//! waste a network slot.
//!
//! ## Detection Criteria
//!
//! A pair of hidden neurons is "symmetric" if:
//! 1. **Cosine similarity** of incoming weight vectors exceeds a threshold (0.95)
//! 2. **Bias values** are within a tolerance
//! 3. **Activation functions** are identical
//!
//! ## Recommended Actions
//!
//! For each detected symmetric pair, we recommend perturbation of one neuron:
//! - `SetBias` — shift one neuron's bias to break symmetry
//! - `SetWeight` — scale one neuron's incoming synapse weights
//! - `ChangeSquash` — change one neuron's activation function (if suitable)

use std::collections::HashMap;

use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson};

use crate::analysis::constants::MIN_DISCOVERY_SAMPLE_COUNT;

/// Minimum cosine similarity to consider two weight vectors as symmetric.
const COSINE_SIMILARITY_THRESHOLD: f32 = 0.95;

/// Maximum absolute bias difference to consider two neurons as symmetric.
const BIAS_TOLERANCE: f32 = 0.5;

/// Weight scaling factor applied to one neuron to break symmetry.
const PERTURBATION_WEIGHT_SCALE: f32 = 0.7;

/// Bias perturbation applied to one neuron to break symmetry.
const PERTURBATION_BIAS_DELTA: f32 = 0.3;

/// Base estimated improvement for breaking a symmetric pair.
const BASE_IMPROVEMENT: f32 = 0.005;

/// Candidate representing a detected symmetric neuron pair.
#[derive(Debug, Clone)]
pub struct SymmetricPairCandidate {
    /// UUID of the first neuron in the pair.
    pub neuron_a_uuid: String,
    /// UUID of the second neuron in the pair (the one to perturb).
    pub neuron_b_uuid: String,
    /// Current activation function (shared by both).
    pub squash: String,
    /// Bias of neuron B (the one to perturb).
    pub neuron_b_bias: f32,
    /// Cosine similarity of incoming weight vectors.
    pub cosine_similarity: f32,
    /// Absolute bias difference between the two neurons.
    pub bias_difference: f32,
    /// Incoming synapse weights for neuron B: (from_uuid, weight).
    pub neuron_b_incoming_weights: Vec<(String, f32)>,
    /// Estimated improvement from breaking this symmetry.
    pub estimated_improvement: f32,
}

/// Detect symmetric neuron pairs in the creature's hidden layer.
///
/// # Arguments
/// * `creature` - The creature's network topology (neurons and synapses).
/// * `neuron_records` - List of `(neuron_uuid, records)` tuples with recorded activations.
///
/// # Returns
/// A list of `SymmetricPairCandidate` for detected symmetric pairs,
/// sorted by cosine similarity (most symmetric first).
pub fn detect_symmetric_neurons(
    creature: &CreatureJson,
    neuron_records: &[(String, Vec<DiscoverRecord>)],
) -> Vec<SymmetricPairCandidate> {
    // Collect hidden neurons
    let hidden_neurons: Vec<&crate::NeuronJson> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "hidden")
        .collect();

    if hidden_neurons.len() < 2 {
        return Vec::new();
    }

    // Build records lookup for minimum sample count filtering
    let records_map: HashMap<&str, &Vec<DiscoverRecord>> = neuron_records
        .iter()
        .map(|(uuid, records)| (uuid.as_str(), records))
        .collect();

    // Filter to neurons with sufficient samples
    let eligible_neurons: Vec<&&crate::NeuronJson> = hidden_neurons
        .iter()
        .filter(|n| {
            records_map
                .get(n.uuid.as_str())
                .is_some_and(|r| r.len() >= MIN_DISCOVERY_SAMPLE_COUNT)
        })
        .collect();

    if eligible_neurons.len() < 2 {
        return Vec::new();
    }

    // Build incoming weight vectors for each hidden neuron
    let mut incoming_weights: HashMap<&str, Vec<(&str, f32)>> = HashMap::new();
    for synapse in &creature.synapses {
        if hidden_neurons.iter().any(|n| n.uuid == synapse.to_uuid) {
            incoming_weights
                .entry(synapse.to_uuid.as_str())
                .or_default()
                .push((synapse.from_uuid.as_str(), synapse.weight));
        }
    }

    // Collect all unique source UUIDs across all hidden neurons for consistent ordering
    let mut all_sources: Vec<&str> = incoming_weights
        .values()
        .flat_map(|weights| weights.iter().map(|(src, _)| *src))
        .collect();
    all_sources.sort();
    all_sources.dedup();

    let mut candidates = Vec::with_capacity(eligible_neurons.len());

    // Compare all pairs of eligible hidden neurons
    for i in 0..eligible_neurons.len() {
        for j in (i + 1)..eligible_neurons.len() {
            let neuron_a = eligible_neurons[i];
            let neuron_b = eligible_neurons[j];

            // Criterion 3: Activation functions must be identical
            if neuron_a.squash != neuron_b.squash {
                continue;
            }

            // Criterion 2: Bias values within tolerance
            let bias_diff = (neuron_a.bias - neuron_b.bias).abs();
            if bias_diff > BIAS_TOLERANCE {
                continue;
            }

            // Criterion 1: Cosine similarity of incoming weight vectors
            let weights_a = build_weight_vector(&neuron_a.uuid, &incoming_weights, &all_sources);
            let weights_b = build_weight_vector(&neuron_b.uuid, &incoming_weights, &all_sources);

            let similarity = cosine_similarity(&weights_a, &weights_b);
            if similarity < COSINE_SIMILARITY_THRESHOLD {
                continue;
            }

            // Compute estimated improvement — higher similarity = more wasted capacity
            let severity =
                (similarity - COSINE_SIMILARITY_THRESHOLD) / (1.0 - COSINE_SIMILARITY_THRESHOLD);
            let estimated_improvement = BASE_IMPROVEMENT * (0.5 + 0.5 * severity);

            // Collect incoming weights for neuron B (the one we will perturb)
            let b_incoming = incoming_weights
                .get(neuron_b.uuid.as_str())
                .map(|ws| ws.iter().map(|(src, w)| (src.to_string(), *w)).collect())
                .unwrap_or_default();

            candidates.push(SymmetricPairCandidate {
                neuron_a_uuid: neuron_a.uuid.clone(),
                neuron_b_uuid: neuron_b.uuid.clone(),
                squash: neuron_a.squash.clone(),
                neuron_b_bias: neuron_b.bias,
                cosine_similarity: similarity,
                bias_difference: bias_diff,
                neuron_b_incoming_weights: b_incoming,
                estimated_improvement,
            });
        }
    }

    // Sort by cosine similarity (most symmetric first — highest improvement potential)
    candidates.sort_by(|a, b| b.cosine_similarity.total_cmp(&a.cosine_similarity));

    candidates
}

/// Build a weight vector for a neuron, ordered by `all_sources`.
///
/// Missing sources get weight 0.0 so both vectors have the same dimensionality.
fn build_weight_vector(
    neuron_uuid: &str,
    incoming_weights: &HashMap<&str, Vec<(&str, f32)>>,
    all_sources: &[&str],
) -> Vec<f32> {
    let weights = incoming_weights.get(neuron_uuid);
    all_sources
        .iter()
        .map(|src| {
            weights
                .and_then(|ws| ws.iter().find(|(s, _)| s == src).map(|(_, w)| *w))
                .unwrap_or(0.0)
        })
        .collect()
}

/// Compute cosine similarity between two vectors.
///
/// Returns 0.0 if either vector has zero magnitude (avoids division by zero).
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if mag_a < f32::EPSILON || mag_b < f32::EPSILON {
        return 0.0;
    }

    (dot / (mag_a * mag_b)).clamp(-1.0, 1.0)
}

/// Convert symmetric pair candidates into coordinated structural candidates.
///
/// For each symmetric pair, we perturb neuron B with:
/// 1. A bias shift (`SetBias`) to move its operating point
/// 2. Weight scaling (`SetWeight`) on incoming synapses to differentiate computation
///
/// The NEAT-AI controller validates each candidate through ablation testing.
pub fn symmetric_neurons_to_coordinated_candidates(
    candidates: &[SymmetricPairCandidate],
) -> Vec<CoordinatedStructuralCandidateJson> {
    let mut results = Vec::with_capacity(candidates.len());

    for c in candidates {
        let mut operations = Vec::new();

        // Perturb neuron B's bias
        let new_bias = c.neuron_b_bias + PERTURBATION_BIAS_DELTA;
        operations.push(CoordinatedStructuralOpJson::SetBias {
            neuron_uuid: c.neuron_b_uuid.clone(),
            bias: new_bias,
        });

        // Scale neuron B's incoming weights
        for (from_uuid, weight) in &c.neuron_b_incoming_weights {
            operations.push(CoordinatedStructuralOpJson::SetWeight {
                from_neuron_uuid: from_uuid.clone(),
                to_neuron_uuid: c.neuron_b_uuid.clone(),
                weight: weight * PERTURBATION_WEIGHT_SCALE,
            });
        }

        results.push(CoordinatedStructuralCandidateJson {
            operations,
            expected_creature_score_gain: c.estimated_improvement,
            comment: Some(format!(
                "[#569] Symmetry-breaking: neurons {} and {} have cosine similarity {:.3}, bias diff {:.3} — perturbing {} to unlock latent capacity",
                c.neuron_a_uuid, c.neuron_b_uuid, c.cosine_similarity, c.bias_difference, c.neuron_b_uuid
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
