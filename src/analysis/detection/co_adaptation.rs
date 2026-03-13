//! Activation co-adaptation detection module (Issue #571).
//!
//! Identifies pairs of hidden neurons whose activations are highly correlated
//! (or anti-correlated), meaning they effectively encode the same information.
//! This wastes network capacity and makes the network fragile.
//!
//! Unlike symmetry detection (Issue #569) which compares weight configurations,
//! co-adaptation detection analyses the *recorded activations* — two neurons may
//! have completely different weights and activation functions but still produce
//! correlated outputs due to receiving similar input patterns.
//!
//! ## Detection Criteria
//!
//! A pair of hidden neurons is "co-adapted" if:
//! 1. |Pearson correlation| of their activations across samples exceeds a
//!    threshold (default 0.9)
//! 2. Both neurons have sufficient recorded samples
//!
//! ## Recommended Actions
//!
//! For each co-adapted pair:
//! - `RemoveNeuron` — remove the lower-impact neuron (reuse `remove-low-impact`)
//! - `SetWeight` — perturb one neuron's incoming weights to break the co-adaptation

use std::collections::HashMap;

use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson};

use crate::analysis::constants::MIN_DISCOVERY_SAMPLE_COUNT;

use super::helpers::build_record_map;

/// Minimum |correlation| to consider a pair co-adapted.
const CO_ADAPTATION_THRESHOLD: f32 = 0.9;

/// Weight perturbation factor applied to break co-adaptation.
const PERTURBATION_SCALE: f32 = 0.6;

/// Base estimated improvement for removing a redundant neuron.
const BASE_IMPROVEMENT: f32 = 0.003;

/// Candidate representing a detected co-adapted neuron pair.
#[derive(Debug, Clone)]
pub struct CoAdaptedPairCandidate {
    /// UUID of the first neuron in the pair.
    pub neuron_a_uuid: String,
    /// UUID of the second neuron in the pair.
    pub neuron_b_uuid: String,
    /// Pearson correlation coefficient between their activations.
    pub correlation: f32,
    /// Number of samples used to compute correlation.
    pub sample_count: usize,
    /// Mean absolute activation of neuron A.
    pub mean_abs_activation_a: f32,
    /// Mean absolute activation of neuron B.
    pub mean_abs_activation_b: f32,
    /// Estimated improvement from resolving the co-adaptation.
    pub estimated_improvement: f32,
}

/// Detect co-adapted neuron pairs from recorded activations.
///
/// Computes pairwise Pearson correlation between hidden neurons' activations
/// and flags pairs where |correlation| exceeds the threshold.
///
/// # Arguments
/// * `creature` - The creature's network topology (neurons and synapses).
/// * `neuron_records` - List of `(neuron_uuid, records)` tuples with recorded activations.
///
/// # Returns
/// A list of `CoAdaptedPairCandidate` sorted by |correlation| (highest first).
pub fn detect_co_adapted_neurons(
    creature: &CreatureJson,
    neuron_records: &[(String, Vec<DiscoverRecord>)],
) -> Vec<CoAdaptedPairCandidate> {
    // Identify hidden neuron UUIDs
    let hidden_uuids: Vec<&str> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "hidden")
        .map(|n| n.uuid.as_str())
        .collect();

    if hidden_uuids.len() < 2 {
        return Vec::new();
    }

    // Build records lookup
    let records_map = build_record_map(neuron_records);

    // Filter to hidden neurons with sufficient samples
    let eligible: Vec<&str> = hidden_uuids
        .iter()
        .filter(|&&uuid| {
            records_map
                .get(uuid)
                .is_some_and(|r| r.len() >= MIN_DISCOVERY_SAMPLE_COUNT)
        })
        .copied()
        .collect();

    if eligible.len() < 2 {
        return Vec::new();
    }

    // Pre-compute per-neuron activation vectors indexed by obs_index
    // for consistent pairing across samples.
    let activation_maps: HashMap<&str, HashMap<u32, f32>> = eligible
        .iter()
        .map(|&uuid| {
            let recs = records_map[uuid];
            let map: HashMap<u32, f32> = recs.iter().map(|r| (r.obs_index, r.activation)).collect();
            (uuid, map)
        })
        .collect();

    let mut candidates = Vec::with_capacity(eligible.len());

    // Compare all pairs of eligible hidden neurons
    for i in 0..eligible.len() {
        for j in (i + 1)..eligible.len() {
            let uuid_a = eligible[i];
            let uuid_b = eligible[j];

            let map_a = &activation_maps[uuid_a];
            let map_b = &activation_maps[uuid_b];

            // Find shared observation indices
            let shared: Vec<(f32, f32)> = map_a
                .iter()
                .filter_map(|(&obs, &act_a)| map_b.get(&obs).map(|&act_b| (act_a, act_b)))
                .collect();

            if shared.len() < MIN_DISCOVERY_SAMPLE_COUNT {
                continue;
            }

            let (vals_a, vals_b): (Vec<f32>, Vec<f32>) = shared.iter().copied().unzip();
            let correlation = super::stats::pearson_correlation(&vals_a, &vals_b);

            if correlation.abs() < CO_ADAPTATION_THRESHOLD {
                continue;
            }

            // Compute mean absolute activations for impact estimation
            let mean_abs_a = shared.iter().map(|(a, _)| a.abs()).sum::<f32>() / shared.len() as f32;
            let mean_abs_b = shared.iter().map(|(_, b)| b.abs()).sum::<f32>() / shared.len() as f32;

            // Higher |correlation| → more redundancy → higher improvement
            let severity =
                (correlation.abs() - CO_ADAPTATION_THRESHOLD) / (1.0 - CO_ADAPTATION_THRESHOLD);
            let estimated_improvement = BASE_IMPROVEMENT * (0.5 + 0.5 * severity);

            candidates.push(CoAdaptedPairCandidate {
                neuron_a_uuid: uuid_a.to_string(),
                neuron_b_uuid: uuid_b.to_string(),
                correlation,
                sample_count: shared.len(),
                mean_abs_activation_a: mean_abs_a,
                mean_abs_activation_b: mean_abs_b,
                estimated_improvement,
            });
        }
    }

    // Sort by |correlation| descending (most co-adapted first)
    candidates.sort_by(|a, b| b.correlation.abs().total_cmp(&a.correlation.abs()));

    candidates
}

/// Convert co-adapted pair candidates into coordinated structural candidates.
///
/// For each pair, generates two candidate strategies:
/// 1. **Remove** the lower-impact neuron (`RemoveNeuron`)
/// 2. **Perturb** one neuron's incoming weights (`SetWeight`) to break co-adaptation
///
/// The NEAT-AI controller validates each candidate through ablation testing.
pub fn co_adapted_pairs_to_coordinated_candidates(
    candidates: &[CoAdaptedPairCandidate],
    creature: &CreatureJson,
) -> Vec<CoordinatedStructuralCandidateJson> {
    let mut results = Vec::with_capacity(candidates.len() * 2);

    // Build incoming synapse lookup for weight perturbation
    let mut incoming_synapses: HashMap<&str, Vec<(&str, f32)>> = HashMap::new();
    for syn in &creature.synapses {
        incoming_synapses
            .entry(syn.to_uuid.as_str())
            .or_default()
            .push((syn.from_uuid.as_str(), syn.weight));
    }

    for c in candidates {
        // Strategy 1: Remove the lower-impact neuron
        let remove_uuid = if c.mean_abs_activation_a <= c.mean_abs_activation_b {
            &c.neuron_a_uuid
        } else {
            &c.neuron_b_uuid
        };

        results.push(CoordinatedStructuralCandidateJson {
            operations: vec![CoordinatedStructuralOpJson::RemoveNeuron {
                neuron_uuid: remove_uuid.clone(),
            }],
            expected_creature_score_gain: c.estimated_improvement,
            comment: Some(format!(
                "[#571] Co-adaptation: neurons {} and {} have correlation {:.3} ({} samples) \
                 — removing lower-impact neuron {}",
                c.neuron_a_uuid, c.neuron_b_uuid, c.correlation, c.sample_count, remove_uuid
            )),
        });

        // Strategy 2: Perturb one neuron's incoming weights to break co-adaptation
        let perturb_uuid = remove_uuid; // Perturb the same (lower-impact) neuron
        if let Some(synapses) = incoming_synapses.get(perturb_uuid.as_str()) {
            let ops: Vec<CoordinatedStructuralOpJson> = synapses
                .iter()
                .map(
                    |(from_uuid, weight)| CoordinatedStructuralOpJson::SetWeight {
                        from_neuron_uuid: from_uuid.to_string(),
                        to_neuron_uuid: perturb_uuid.clone(),
                        weight: weight * PERTURBATION_SCALE,
                    },
                )
                .collect();

            if !ops.is_empty() {
                results.push(CoordinatedStructuralCandidateJson {
                    operations: ops,
                    expected_creature_score_gain: c.estimated_improvement * 0.8,
                    comment: Some(format!(
                        "[#571] Co-adaptation: neurons {} and {} have correlation {:.3} \
                         — perturbing {} incoming weights to break co-adaptation",
                        c.neuron_a_uuid, c.neuron_b_uuid, c.correlation, perturb_uuid
                    )),
                });
            }
        }
    }

    // Sort by expected improvement (best first)
    results.sort_by(|a, b| {
        b.expected_creature_score_gain
            .total_cmp(&a.expected_creature_score_gain)
    });

    results
}
