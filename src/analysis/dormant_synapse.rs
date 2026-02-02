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

use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson};

/// Minimum samples required for reliable dormant synapse detection.
const MIN_SAMPLES_FOR_DORMANT: usize = 20;

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
    let records_map: HashMap<&str, &Vec<DiscoverRecord>> = neuron_records
        .iter()
        .map(|(uuid, records)| (uuid.as_str(), records))
        .collect();

    let mut candidates = Vec::new();

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
    candidates.sort_by(|a, b| {
        b.estimated_improvement
            .partial_cmp(&a.estimated_improvement)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

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
            .partial_cmp(&a.expected_creature_score_gain)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DiscoverRecord;
    use crate::{CreatureJson, NeuronJson, SynapseJson};

    fn neuron(uuid: &str, ntype: &str) -> NeuronJson {
        NeuronJson {
            uuid: uuid.to_string(),
            neuron_type: ntype.to_string(),
            squash: "TANH".to_string(),
            bias: 0.0,
        }
    }

    fn syn(from: &str, to: &str, weight: f32) -> SynapseJson {
        SynapseJson {
            from_uuid: from.to_string(),
            to_uuid: to.to_string(),
            weight,
            synapse_type: None,
        }
    }

    fn records_for(uuid: &str, count: usize, activation: f32) -> (String, Vec<DiscoverRecord>) {
        let recs = (0..count)
            .map(|i| DiscoverRecord {
                obs_index: i as u32,
                neuron_uuid: uuid.to_string(),
                value: None,
                activation,
                errors: vec![0.01],
            })
            .collect();
        (uuid.to_string(), recs)
    }

    /// Creature: i0 → h1 → o0, with two synapses targeting h1.
    fn dormant_creature(dormant_weight: f32, active_weight: f32) -> CreatureJson {
        CreatureJson {
            neurons: vec![
                neuron("i0", "input"),
                neuron("i1", "input"),
                neuron("h1", "hidden"),
                neuron("o0", "output"),
            ],
            synapses: vec![
                syn("i0", "h1", dormant_weight),
                syn("i1", "h1", active_weight),
                syn("h1", "o0", 1.0),
            ],
            input: 2,
            output: 1,
        }
    }

    // -----------------------------------------------------------------------
    // Detection criteria
    // -----------------------------------------------------------------------

    #[test]
    fn near_zero_weight_synapse_detected() {
        let creature = dormant_creature(1e-5, 1.0);
        let records = vec![records_for("i0", 30, 0.5), records_for("i1", 30, 0.5)];
        let candidates = detect_dormant_synapses(&creature, &records);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].from_neuron_uuid, "i0");
        assert_eq!(candidates[0].to_neuron_uuid, "h1");
    }

    #[test]
    fn estimated_improvement_positive() {
        let creature = dormant_creature(1e-5, 1.0);
        let records = vec![records_for("i0", 30, 0.5), records_for("i1", 30, 0.5)];
        let candidates = detect_dormant_synapses(&creature, &records);
        assert!(candidates[0].estimated_improvement > 0.0);
    }

    // -----------------------------------------------------------------------
    // Exclusion criteria
    // -----------------------------------------------------------------------

    #[test]
    fn active_synapse_not_flagged() {
        let creature = dormant_creature(1.0, 1.0);
        let records = vec![records_for("i0", 30, 0.5), records_for("i1", 30, 0.5)];
        let candidates = detect_dormant_synapses(&creature, &records);
        assert!(candidates.is_empty(), "active weight should not be flagged");
    }

    #[test]
    fn sole_connection_not_flagged() {
        // h1 has only one incoming synapse — removing it would be destructive
        let creature = CreatureJson {
            neurons: vec![
                neuron("i0", "input"),
                neuron("h1", "hidden"),
                neuron("o0", "output"),
            ],
            synapses: vec![syn("i0", "h1", 1e-5), syn("h1", "o0", 1.0)],
            input: 1,
            output: 1,
        };
        let records = vec![records_for("i0", 30, 0.5)];
        let candidates = detect_dormant_synapses(&creature, &records);
        assert!(candidates.is_empty(), "sole connection should be protected");
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn insufficient_samples_not_flagged() {
        let creature = dormant_creature(1e-5, 1.0);
        let records = vec![records_for("i0", 5, 0.5)];
        let candidates = detect_dormant_synapses(&creature, &records);
        assert!(candidates.is_empty());
    }

    #[test]
    fn empty_synapses_no_candidates() {
        let creature = CreatureJson {
            neurons: vec![neuron("i0", "input")],
            synapses: vec![],
            input: 1,
            output: 0,
        };
        let candidates = detect_dormant_synapses(&creature, &[]);
        assert!(candidates.is_empty());
    }

    // -----------------------------------------------------------------------
    // Conversion
    // -----------------------------------------------------------------------

    #[test]
    fn coordinated_candidate_uses_remove_synapse() {
        let candidates = vec![DormantSynapseCandidate {
            from_neuron_uuid: "i0".to_string(),
            to_neuron_uuid: "h1".to_string(),
            weight: 1e-5,
            mean_abs_contribution: 1e-6,
            other_fan_in: 1,
            sample_count: 30,
            estimated_improvement: 0.001,
        }];
        let coordinated = dormant_synapses_to_coordinated_candidates(&candidates);
        assert_eq!(coordinated.len(), 1);
        assert!(matches!(
            &coordinated[0].operations[0],
            CoordinatedStructuralOpJson::RemoveSynapse { from_neuron_uuid, to_neuron_uuid }
            if from_neuron_uuid == "i0" && to_neuron_uuid == "h1"
        ));
    }
}
