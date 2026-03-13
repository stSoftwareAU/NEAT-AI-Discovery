//! Dormant synapse detection module (Issue #359).
//!
//! Identifies synapses with near-zero weights that contribute negligible signal
//! to their target neuron. Dormant synapses waste computation during both forward
//! pass and discovery analysis without providing meaningful information flow.
//!
//! See `docs/DISCOVERY_TYPES.md` § "Dormant Synapse Detection" for full documentation.
//!
//! ## Detection Criteria
//!
//! A synapse is "dormant" if:
//! 1. **Near-zero weight**: The absolute weight is below a threshold (e.g., 1e-4).
//! 2. **Low contribution**: The product of source activation and synapse weight
//!    is negligible relative to other inputs to the target neuron.
//! 3. **Not the sole connection**: The target neuron has other incoming synapses
//!    (removing the only input would be destructive).
//!
//! ## Recommended Actions
//!
//! When a dormant synapse is detected, we recommend:
//! 1. **Remove the synapse**: Emit a `RemoveSynapse` operation via
//!    `CoordinatedStructuralCandidateJson`.
//!
//! Removing dormant synapses reduces network complexity and may allow the controller
//! to allocate evaluation budget to more promising candidates.

use std::collections::HashMap;

use super::helpers::build_record_map;
use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson};

// MIN_SAMPLES_FOR_DORMANT moved to constants.rs (Issue #424)
use crate::analysis::constants::MIN_DISCOVERY_SAMPLE_COUNT as MIN_SAMPLES_FOR_DORMANT;

/// Maximum absolute weight to consider a synapse dormant.
const DORMANT_WEIGHT_THRESHOLD: f32 = 1e-4;

/// Maximum mean absolute contribution (|weight × source_activation|) for dormancy.
const DORMANT_CONTRIBUTION_THRESHOLD: f32 = 1e-4;

/// Result of detecting a dormant synapse.
#[derive(Debug, Clone)]
pub struct DormantSynapseCandidate {
    /// UUID of the source neuron.
    pub from_neuron_uuid: String,
    /// UUID of the target neuron.
    pub to_neuron_uuid: String,
    /// Current weight of the synapse.
    pub weight: f32,
    /// Mean absolute contribution (|weight × activation|) across samples.
    pub mean_abs_contribution: f32,
    /// Number of other incoming synapses to the target neuron.
    pub other_fan_in: usize,
    /// Number of samples analysed.
    pub sample_count: usize,
    /// Estimated creature score improvement from removing this synapse.
    pub estimated_improvement: f32,
}

/// Detect dormant synapses from the creature topology and recorded activations.
///
/// # Arguments
/// * `creature` - The creature's network topology (neurons and synapses).
/// * `neuron_records` - List of `(neuron_uuid, records)` tuples with recorded activations.
///
/// # Returns
/// A list of `DormantSynapseCandidate` for synapses that are dormant,
/// sorted by estimated improvement (best first).
pub fn detect_dormant_synapses(
    creature: &CreatureJson,
    neuron_records: &[(String, Vec<DiscoverRecord>)],
) -> Vec<DormantSynapseCandidate> {
    // Build fan-in count map (how many synapses target each neuron)
    let mut fan_in_count: HashMap<&str, usize> = HashMap::new();
    for synapse in &creature.synapses {
        *fan_in_count.entry(synapse.to_uuid.as_str()).or_insert(0) += 1;
    }

    // Build records lookup
    let records_map = build_record_map(neuron_records);

    let mut candidates = Vec::with_capacity(creature.synapses.len());

    for synapse in &creature.synapses {
        // Skip if weight is not near zero
        if synapse.weight.abs() > DORMANT_WEIGHT_THRESHOLD {
            continue;
        }

        // Skip if this is the only input to the target neuron
        let total_fan_in = fan_in_count
            .get(synapse.to_uuid.as_str())
            .copied()
            .unwrap_or(0);
        if total_fan_in <= 1 {
            continue;
        }

        // Get source neuron activation records
        let Some(source_records) = records_map.get(synapse.from_uuid.as_str()) else {
            continue;
        };

        if source_records.len() < MIN_SAMPLES_FOR_DORMANT {
            continue;
        }

        // Compute mean absolute contribution
        let n = source_records.len() as f32;
        let sum_abs_contribution: f32 = source_records
            .iter()
            .map(|r| (synapse.weight * r.activation).abs())
            .sum();
        let mean_abs_contribution = sum_abs_contribution / n;

        if mean_abs_contribution > DORMANT_CONTRIBUTION_THRESHOLD {
            continue;
        }

        // Estimated improvement: removing a dormant synapse reduces complexity.
        // The improvement is small but positive (reduced overhead).
        let estimated_improvement =
            0.001 * (1.0 - mean_abs_contribution / DORMANT_CONTRIBUTION_THRESHOLD);

        candidates.push(DormantSynapseCandidate {
            from_neuron_uuid: synapse.from_uuid.clone(),
            to_neuron_uuid: synapse.to_uuid.clone(),
            weight: synapse.weight,
            mean_abs_contribution,
            other_fan_in: total_fan_in - 1,
            sample_count: source_records.len(),
            estimated_improvement,
        });
    }

    // Sort by estimated improvement (best first)
    candidates.sort_by(|a, b| b.estimated_improvement.total_cmp(&a.estimated_improvement));

    candidates
}

/// Convert dormant synapse candidates into coordinated structural candidates.
///
/// Each dormant synapse produces a `RemoveSynapse` coordinated candidate.
/// The NEAT-AI controller will validate the removal through ablation testing
/// before actually applying it.
pub fn dormant_synapses_to_coordinated_candidates(
    candidates: &[DormantSynapseCandidate],
) -> Vec<CoordinatedStructuralCandidateJson> {
    let mut results = Vec::with_capacity(candidates.len());

    for c in candidates {
        results.push(CoordinatedStructuralCandidateJson {
            operations: vec![CoordinatedStructuralOpJson::RemoveSynapse {
                from_neuron_uuid: c.from_neuron_uuid.clone(),
                to_neuron_uuid: c.to_neuron_uuid.clone(),
            }],
            expected_creature_score_gain: c.estimated_improvement,
            comment: Some(format!(
                "Dormant synapse {} → {}: weight {:.2e}, mean abs contribution {:.2e}, {} samples → remove to reduce complexity",
                c.from_neuron_uuid, c.to_neuron_uuid, c.weight, c.mean_abs_contribution, c.sample_count
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
