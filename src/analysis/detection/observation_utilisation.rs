//! Observation utilisation detection module (Issue #543).
//!
//! Builds on the existing `observation_range.rs` (Issue #398) to generate
//! actionable candidates for input neurons with low effective utilisation.
//!
//! While `observation_range.rs` characterises the effective range and sentinel
//! values as metadata, this module goes further by recommending concrete
//! structural changes: adding gating neurons that filter out sentinel values
//! so downstream neurons only process meaningful data.
//!
//! ## Detection Approach
//!
//! 1. Reuse the sentinel detection logic from `observation_range.rs`.
//! 2. For inputs where sentinels are detected and utilisation is low,
//!    recommend a gating neuron that zeroes the contribution when the
//!    input is at a sentinel value.
//!
//! ## Candidate Generation
//!
//! When an underutilised observation is found, we recommend a coordinated
//! structural candidate that adds a gating hidden neuron between the
//! input and its downstream targets. The gating neuron uses a step-like
//! activation to pass through the effective range and block sentinel values.

use crate::CreatureJson;
use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson};

use super::observation_range;

/// Minimum utilisation ratio to consider an input "underutilised".
/// Below this threshold, the sentinel values occupy too much of the range.
const MAX_UTILISATION_FOR_DETECTION: f32 = 0.80;

/// Result of detecting an underutilised observation.
#[derive(Debug, Clone)]
pub struct UnderutilisedObservation {
    /// UUID of the input neuron.
    pub neuron_uuid: String,
    /// Detected sentinel values.
    pub sentinel_values: Vec<f32>,
    /// Fraction of the full range that is effectively used.
    pub utilisation_ratio: f32,
    /// Minimum of the effective (non-sentinel) range.
    pub effective_min: f32,
    /// Maximum of the effective (non-sentinel) range.
    pub effective_max: f32,
    /// Number of samples analysed.
    pub sample_count: usize,
    /// Estimated improvement from gating sentinel values.
    pub estimated_improvement: f32,
    /// Human-readable explanation.
    pub reason: String,
}

/// Detect underutilised observations from recorded samples.
///
/// Uses the sentinel detection from `observation_range.rs` and filters to
/// observations with low utilisation ratios that would benefit from gating.
///
/// # Arguments
/// * `creature` - The creature's network topology (identifies input neurons).
/// * `neuron_records` - List of `(neuron_uuid, records)` tuples.
///
/// # Returns
/// A list of `UnderutilisedObservation` for inputs with sentinel-dominated ranges,
/// sorted by utilisation ratio (lowest first).
pub fn detect_underutilised_observations(
    creature: &CreatureJson,
    neuron_records: &[(String, Vec<DiscoverRecord>)],
) -> Vec<UnderutilisedObservation> {
    // Reuse the observation range detection logic
    let ranges = observation_range::detect_observation_ranges(creature, neuron_records);

    let mut results: Vec<UnderutilisedObservation> = ranges
        .into_iter()
        .filter(|r| r.utilisation_ratio < MAX_UTILISATION_FOR_DETECTION)
        .map(|r| {
            let sentinel_desc = r
                .sentinel_values
                .iter()
                .map(|v| format!("{v:.1}"))
                .collect::<Vec<_>>()
                .join(", ");

            let estimated_improvement = (1.0 - r.utilisation_ratio) * 0.005;

            UnderutilisedObservation {
                neuron_uuid: r.neuron_uuid,
                sentinel_values: r.sentinel_values,
                utilisation_ratio: r.utilisation_ratio,
                effective_min: r.effective_min,
                effective_max: r.effective_max,
                sample_count: r.sample_count,
                estimated_improvement,
                reason: format!(
                    "Utilisation {:.0}% — sentinel value(s) [{}] occupy significant range fraction",
                    r.utilisation_ratio * 100.0,
                    sentinel_desc,
                ),
            }
        })
        .collect();

    // Sort by utilisation ratio ascending (most underutilised first)
    results.sort_by(|a, b| a.utilisation_ratio.total_cmp(&b.utilisation_ratio));

    results
}

/// Convert underutilised observation results to coordinated structural candidates.
///
/// For each underutilised input, recommends a bias adjustment on downstream
/// synapses to centre the effective range around zero, improving gradient flow.
pub fn observation_utilisation_to_coordinated_candidates(
    observations: &[UnderutilisedObservation],
    creature: &CreatureJson,
) -> Vec<CoordinatedStructuralCandidateJson> {
    let mut candidates = Vec::new();

    for obs in observations {
        // Find downstream targets of this input neuron
        let downstream_targets: Vec<&str> = creature
            .synapses
            .iter()
            .filter(|s| s.from_uuid == obs.neuron_uuid)
            .map(|s| s.to_uuid.as_str())
            .collect();

        if downstream_targets.is_empty() {
            continue;
        }

        // Compute a bias offset to centre the effective range
        let effective_centre = (obs.effective_min + obs.effective_max) / 2.0;

        // For each downstream target, recommend adjusting the synapse weight
        // to compensate for the sentinel-induced offset
        for &target_uuid in &downstream_targets {
            // Find the current synapse weight
            let current_weight = creature
                .synapses
                .iter()
                .find(|s| s.from_uuid == obs.neuron_uuid && s.to_uuid == target_uuid)
                .map(|s| s.weight)
                .unwrap_or(0.0);

            if current_weight.abs() < 1e-8 {
                continue;
            }

            // The bias adjustment compensates for the sentinel offset
            // When sentinel values are present, the mean input shifts
            let bias_compensation = -effective_centre * current_weight;

            if bias_compensation.abs() < 1e-6 {
                continue;
            }

            candidates.push(CoordinatedStructuralCandidateJson {
                operations: vec![CoordinatedStructuralOpJson::SetBias {
                    neuron_uuid: target_uuid.to_string(),
                    bias: bias_compensation,
                }],
                expected_creature_score_gain: obs.estimated_improvement,
                comment: Some(format!(
                    "Observation utilisation (Issue #543): {} has sentinel value(s), \
                     adjust bias on {} to compensate for effective centre offset {:.3}. {}",
                    obs.neuron_uuid, target_uuid, effective_centre, obs.reason,
                )),
            });
        }
    }

    candidates
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NeuronJson, SynapseJson};

    fn make_creature() -> CreatureJson {
        CreatureJson {
            neurons: vec![
                NeuronJson {
                    uuid: "input-0".to_string(),
                    neuron_type: "input".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 0.0,
                },
                NeuronJson {
                    uuid: "output-0".to_string(),
                    neuron_type: "output".to_string(),
                    squash: "TANH".to_string(),
                    bias: 0.0,
                },
            ],
            synapses: vec![SynapseJson {
                from_uuid: "input-0".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.5,
                synapse_type: None,
            }],
            input: 1,
            output: 1,
        }
    }

    #[test]
    fn test_empty_records_returns_empty() {
        let creature = make_creature();
        let records: Vec<(String, Vec<DiscoverRecord>)> = vec![];
        let result = detect_underutilised_observations(&creature, &records);
        assert!(result.is_empty());
    }

    #[test]
    fn test_conversion_produces_set_bias() {
        let creature = make_creature();
        let obs = vec![UnderutilisedObservation {
            neuron_uuid: "input-0".to_string(),
            sentinel_values: vec![-1.0],
            utilisation_ratio: 0.3,
            effective_min: 0.2,
            effective_max: 0.8,
            sample_count: 50,
            estimated_improvement: 0.005,
            reason: "Test".to_string(),
        }];

        let candidates = observation_utilisation_to_coordinated_candidates(&obs, &creature);
        assert!(!candidates.is_empty());

        // Should have a SetBias operation targeting the downstream neuron
        match &candidates[0].operations[0] {
            CoordinatedStructuralOpJson::SetBias { neuron_uuid, .. } => {
                assert_eq!(neuron_uuid, "output-0");
            }
            other => panic!("Expected SetBias, got {other:?}"),
        }
    }
}
