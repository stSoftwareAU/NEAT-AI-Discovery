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
//!
//! ## Bounded scan (Issue #2349)
//!
//! The pair scan is inherently O(E²) in the number of eligible hidden
//! neurons. Two changes keep it from being an uncancellable, super-linear
//! cost centre:
//! - Each eligible neuron's incoming-weight vector is built exactly **once**
//!   (`O(E · (A + F))` total), rather than being rebuilt from scratch for
//!   both sides of every pair via a linear `find` (`O(E² · A · F)`).
//! - The pair scan itself checks the discovery deadline / global
//!   cancellation flag before every outer iteration, and stops once
//!   [`MAX_SYMMETRIC_PAIR_CANDIDATES`] candidates have been emitted. Either
//!   stop reports its reason via [`BoundedScan::truncation`] so a truncated
//!   scan is never presented as a complete one.

use std::collections::{HashMap, HashSet};
use std::time::SystemTime;

use crate::analysis::recommendation::epistatic::{BoundedScan, ScanTruncation};
use crate::analysis::utils::deadline_passed;
use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson};

use crate::analysis::constants::MIN_DISCOVERY_SAMPLE_COUNT;

use super::helpers::build_record_map;

/// Ceiling on emitted symmetric-pair candidates (Issue #2349).
///
/// Without a ceiling, a creature with many identical neurons yields
/// `E(E-1)/2` candidates, each cloning neuron B's fan-in weights — a cost
/// that grows quadratically with no bound.
pub const MAX_SYMMETRIC_PAIR_CANDIDATES: usize = 256;

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
    /// Incoming synapse weights for neuron B: (`from_uuid`, weight).
    pub neuron_b_incoming_weights: Vec<(String, f32)>,
    /// Estimated improvement from breaking this symmetry.
    pub estimated_improvement: f32,
}

/// Detect symmetric neuron pairs in the creature's hidden layer.
///
/// Unbounded by deadline — use [`detect_symmetric_neurons_with_deadline`] on
/// the discovery hot path. Still capped by [`MAX_SYMMETRIC_PAIR_CANDIDATES`].
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
    neuron_records: &[(String, impl AsRef<[DiscoverRecord]>)],
) -> Vec<SymmetricPairCandidate> {
    detect_symmetric_neurons_with_deadline(creature, neuron_records, &None).candidates
}

/// [`detect_symmetric_neurons`] that stops at `deadline` or on global
/// cancellation, and reports why it stopped via [`BoundedScan::truncation`]
/// (Issue #2349).
pub fn detect_symmetric_neurons_with_deadline(
    creature: &CreatureJson,
    neuron_records: &[(String, impl AsRef<[DiscoverRecord]>)],
    deadline: &Option<SystemTime>,
) -> BoundedScan<SymmetricPairCandidate> {
    detect_symmetric_neurons_observed(creature, neuron_records, deadline, || {})
}

/// [`detect_symmetric_neurons_with_deadline`] with `on_weight_vector_build`
/// called once per weight vector built, so a test can assert on how the work
/// grows without timing it (Issue #2349).
pub fn detect_symmetric_neurons_observed(
    creature: &CreatureJson,
    neuron_records: &[(String, impl AsRef<[DiscoverRecord]>)],
    deadline: &Option<SystemTime>,
    mut on_weight_vector_build: impl FnMut(),
) -> BoundedScan<SymmetricPairCandidate> {
    // Collect hidden neurons
    let hidden_neurons: Vec<&crate::NeuronJson> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "hidden")
        .collect();

    if hidden_neurons.len() < 2 {
        return BoundedScan {
            candidates: Vec::new(),
            truncation: None,
        };
    }

    // Build records lookup for minimum sample count filtering
    let records_map = build_record_map(neuron_records);

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
        return BoundedScan {
            candidates: Vec::new(),
            truncation: None,
        };
    }

    if deadline_passed(deadline) {
        return BoundedScan {
            candidates: Vec::new(),
            truncation: Some(ScanTruncation::DeadlinePassed),
        };
    }

    // Build incoming weight vectors for each hidden neuron. Issue #2349: use
    // a HashSet for the hidden-neuron membership test instead of `.any(..)`
    // over the whole hidden-neuron list for every synapse.
    let hidden_uuids: HashSet<&str> = hidden_neurons.iter().map(|n| n.uuid.as_str()).collect();

    let mut incoming_weights: HashMap<&str, Vec<(&str, f32)>> = HashMap::new();
    for synapse in &creature.synapses {
        if hidden_uuids.contains(synapse.to_uuid.as_str()) {
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

    let source_index: HashMap<&str, usize> = all_sources
        .iter()
        .enumerate()
        .map(|(idx, src)| (*src, idx))
        .collect();

    // Issue #2349: build each eligible neuron's weight vector exactly once,
    // instead of rebuilding both sides of every pair via a linear `find`.
    let vectors: Vec<Vec<f32>> = eligible_neurons
        .iter()
        .map(|neuron| {
            on_weight_vector_build();
            let mut vector = vec![0.0f32; all_sources.len()];
            if let Some(weights) = incoming_weights.get(neuron.uuid.as_str()) {
                // Preserve the legacy `find`-based semantics: the FIRST
                // synapse (in `creature.synapses` order) for a duplicated
                // source wins. Scattering in reverse and overwriting
                // unconditionally means the write for the first original
                // occurrence happens last, so it is the one left standing.
                for (src, weight) in weights.iter().rev() {
                    if let Some(&idx) = source_index.get(src) {
                        vector[idx] = *weight;
                    }
                }
            }
            vector
        })
        .collect();

    let mut candidates = Vec::with_capacity(eligible_neurons.len());
    let mut truncation = None;

    // Compare all pairs of eligible hidden neurons
    'outer: for i in 0..eligible_neurons.len() {
        if deadline_passed(deadline) {
            truncation = Some(ScanTruncation::DeadlinePassed);
            break 'outer;
        }
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
            let similarity = cosine_similarity(&vectors[i], &vectors[j]);
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

            if candidates.len() >= MAX_SYMMETRIC_PAIR_CANDIDATES {
                truncation = Some(ScanTruncation::CandidateCeiling);
                break 'outer;
            }
        }
    }

    // Sort by cosine similarity (most symmetric first — highest improvement potential)
    candidates.sort_by(|a, b| b.cosine_similarity.total_cmp(&a.cosine_similarity));

    BoundedScan {
        candidates,
        truncation,
    }
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
            remove_neuron_compensation: None,
        constant_neuron_bias_fold: None,
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
