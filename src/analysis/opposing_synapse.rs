//! Opposing synapse detection module (Issue #360).
//!
//! Identifies synapses whose contribution consistently works against error reduction.
//! When a synapse's activation–error correlation is strongly positive (meaning the
//! synapse increases the output when the error is already positive, or decreases it
//! when the error is already negative), the synapse is actively hindering performance.
//!
//! See `docs/DISCOVERY_TYPES.md` § "Opposing Synapse Detection" for full documentation.
//!
//! ## Detection Criteria
//!
//! A synapse is "opposing" if:
//! 1. **Positive contribution–error correlation**: The Pearson correlation between
//!    `weight × source_activation` and the target neuron's error is strongly positive
//!    (≥ threshold).
//! 2. **Meaningful contribution**: The synapse has a non-negligible mean absolute
//!    contribution (distinguishing from dormant synapses).
//! 3. **Sufficient samples**: Enough recorded samples for statistical reliability.
//!
//! ## Recommended Actions
//!
//! When an opposing synapse is detected, we recommend:
//! 1. **Remove the synapse**: If the opposition is strong, removing the synapse eliminates
//!    the harmful contribution.
//! 2. **Flip the weight sign**: If the synapse carries useful magnitude but wrong direction,
//!    negating the weight may help.
//!
//! These are emitted as `CoordinatedStructuralCandidateJson` with `RemoveSynapse` or
//! `SetWeight` operations.

use std::collections::HashMap;

use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson};

/// Minimum samples required for reliable opposing synapse detection.
const MIN_SAMPLES_FOR_OPPOSING: usize = 20;

/// Minimum Pearson correlation between synapse contribution and error
/// to consider the synapse opposing.
const MIN_OPPOSING_CORRELATION: f32 = 0.3;

/// Minimum mean absolute contribution for the synapse to be considered
/// non-dormant (dormant synapses are handled separately).
const MIN_CONTRIBUTION_FOR_OPPOSING: f32 = 0.01;

/// Result of detecting an opposing synapse.
#[derive(Debug, Clone)]
pub struct OpposingSynapseCandidate {
    /// UUID of the source neuron.
    pub from_neuron_uuid: String,
    /// UUID of the target neuron.
    pub to_neuron_uuid: String,
    /// Current weight of the synapse.
    pub weight: f32,
    /// Pearson correlation between synapse contribution and target error.
    pub contribution_error_correlation: f32,
    /// Mean absolute contribution of the synapse.
    pub mean_abs_contribution: f32,
    /// Number of samples analysed.
    pub sample_count: usize,
    /// Whether removing the synapse is recommended (vs. flipping weight).
    pub recommend_removal: bool,
    /// Estimated creature score improvement from fixing this synapse.
    pub estimated_improvement: f32,
}

/// Detect opposing synapses from the creature topology and recorded activations.
///
/// # Arguments
/// * `creature` - The creature's network topology (neurons and synapses).
/// * `neuron_records` - List of `(neuron_uuid, records)` tuples with recorded activations.
///
/// # Returns
/// A list of `OpposingSynapseCandidate` for synapses that oppose error reduction,
/// sorted by estimated improvement (best first).
pub fn detect_opposing_synapses(
    creature: &CreatureJson,
    neuron_records: &[(String, Vec<DiscoverRecord>)],
) -> Vec<OpposingSynapseCandidate> {
    // Build records lookup
    let records_map: HashMap<&str, &Vec<DiscoverRecord>> = neuron_records
        .iter()
        .map(|(uuid, records)| (uuid.as_str(), records))
        .collect();

    // Build a mapping from obs_index to records for target neurons
    let mut target_error_map: HashMap<&str, HashMap<u32, &DiscoverRecord>> = HashMap::new();
    for (uuid, records) in neuron_records {
        let obs_map: HashMap<u32, &DiscoverRecord> =
            records.iter().map(|r| (r.obs_index, r)).collect();
        target_error_map.insert(uuid.as_str(), obs_map);
    }

    // Identify output neuron UUIDs (primary targets for opposing detection)
    let output_uuids: std::collections::HashSet<&str> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "output")
        .map(|n| n.uuid.as_str())
        .collect();

    let mut candidates = Vec::new();

    for synapse in &creature.synapses {
        // Only analyse synapses targeting output neurons (where error is directly measured)
        if !output_uuids.contains(synapse.to_uuid.as_str()) {
            continue;
        }

        // Get source neuron activation records
        let Some(source_records) = records_map.get(synapse.from_uuid.as_str()) else {
            continue;
        };

        if source_records.len() < MIN_SAMPLES_FOR_OPPOSING {
            continue;
        }

        // Get target error records
        let Some(target_obs_map) = target_error_map.get(synapse.to_uuid.as_str()) else {
            continue;
        };

        // Compute synapse contribution and correlate with target error
        let mut contributions: Vec<f32> = Vec::new();
        let mut errors: Vec<f32> = Vec::new();

        for source_record in source_records.iter() {
            let contribution = synapse.weight * source_record.activation;

            if let Some(target_record) = target_obs_map.get(&source_record.obs_index) {
                if let Some(&error) = target_record.errors.first() {
                    contributions.push(contribution);
                    errors.push(error);
                }
            }
        }

        if contributions.len() < MIN_SAMPLES_FOR_OPPOSING {
            continue;
        }

        // Compute mean absolute contribution
        let n = contributions.len() as f32;
        let mean_abs_contribution: f32 = contributions.iter().map(|c| c.abs()).sum::<f32>() / n;

        if mean_abs_contribution < MIN_CONTRIBUTION_FOR_OPPOSING {
            continue;
        }

        // Compute Pearson correlation between contributions and errors
        let correlation = pearson_correlation(&contributions, &errors);

        if correlation < MIN_OPPOSING_CORRELATION {
            continue;
        }

        // Higher correlation and higher contribution = more harmful
        let harm_score = correlation * mean_abs_contribution;
        let estimated_improvement = harm_score * 0.05;

        // Recommend removal if correlation is very strong; otherwise flip the weight
        let recommend_removal = correlation > 0.5;

        candidates.push(OpposingSynapseCandidate {
            from_neuron_uuid: synapse.from_uuid.clone(),
            to_neuron_uuid: synapse.to_uuid.clone(),
            weight: synapse.weight,
            contribution_error_correlation: correlation,
            mean_abs_contribution,
            sample_count: contributions.len(),
            recommend_removal,
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

/// Compute Pearson correlation coefficient between two vectors.
fn pearson_correlation(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len() as f32;
    if n < 2.0 {
        return 0.0;
    }

    let mean_x: f32 = x.iter().sum::<f32>() / n;
    let mean_y: f32 = y.iter().sum::<f32>() / n;

    let mut cov = 0.0_f32;
    let mut var_x = 0.0_f32;
    let mut var_y = 0.0_f32;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom < 1e-12 {
        return 0.0;
    }

    cov / denom
}

/// Convert opposing synapse candidates into coordinated structural candidates.
///
/// Each opposing synapse produces either:
/// - A `RemoveSynapse` operation (for strongly opposing synapses), or
/// - A `SetWeight` operation with negated weight (for moderately opposing synapses).
pub fn opposing_synapses_to_coordinated_candidates(
    candidates: &[OpposingSynapseCandidate],
) -> Vec<CoordinatedStructuralCandidateJson> {
    let mut results = Vec::with_capacity(candidates.len());

    for c in candidates {
        if c.recommend_removal {
            results.push(CoordinatedStructuralCandidateJson {
                operations: vec![CoordinatedStructuralOpJson::RemoveSynapse {
                    from_neuron_uuid: c.from_neuron_uuid.clone(),
                    to_neuron_uuid: c.to_neuron_uuid.clone(),
                }],
                expected_creature_score_gain: c.estimated_improvement,
                comment: Some(format!(
                    "Opposing synapse {} → {}: contribution–error correlation {:.3}, weight {:.4} → remove harmful connection",
                    c.from_neuron_uuid, c.to_neuron_uuid, c.contribution_error_correlation, c.weight
                )),
            });
        } else {
            results.push(CoordinatedStructuralCandidateJson {
                operations: vec![CoordinatedStructuralOpJson::SetWeight {
                    from_neuron_uuid: c.from_neuron_uuid.clone(),
                    to_neuron_uuid: c.to_neuron_uuid.clone(),
                    weight: -c.weight,
                }],
                expected_creature_score_gain: c.estimated_improvement * 0.7,
                comment: Some(format!(
                    "Opposing synapse {} → {}: contribution–error correlation {:.3}, weight {:.4} → flip weight sign to reverse harmful direction",
                    c.from_neuron_uuid, c.to_neuron_uuid, c.contribution_error_correlation, c.weight
                )),
            });
        }
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

    /// Build opposing records: source activation and target error are positively correlated
    /// (both increase together), so weight * activation correlates with error.
    fn opposing_records(
        count: usize,
        weight: f32,
    ) -> (Vec<(String, Vec<DiscoverRecord>)>, CreatureJson) {
        let source_recs: Vec<DiscoverRecord> = (0..count)
            .map(|i| DiscoverRecord {
                obs_index: i as u32,
                neuron_uuid: "h1".to_string(),
                value: None,
                activation: 0.1 + (i as f32) * 0.1,
                errors: vec![],
            })
            .collect();

        let target_recs: Vec<DiscoverRecord> = (0..count)
            .map(|i| DiscoverRecord {
                obs_index: i as u32,
                neuron_uuid: "o0".to_string(),
                value: None,
                activation: 0.5,
                // Error correlates positively with source activation × weight
                errors: vec![weight * (0.1 + (i as f32) * 0.1) + 0.01],
            })
            .collect();

        let records = vec![
            ("h1".to_string(), source_recs),
            ("o0".to_string(), target_recs),
        ];

        let creature = CreatureJson {
            neurons: vec![
                neuron("i0", "input"),
                neuron("h1", "hidden"),
                neuron("o0", "output"),
            ],
            synapses: vec![syn("i0", "h1", 1.0), syn("h1", "o0", weight)],
            input: 1,
            output: 1,
        };

        (records, creature)
    }

    // -----------------------------------------------------------------------
    // Detection criteria
    // -----------------------------------------------------------------------

    #[test]
    fn detects_opposing_synapse() {
        let (records, creature) = opposing_records(30, 1.0);
        let candidates = detect_opposing_synapses(&creature, &records);
        assert!(
            !candidates.is_empty(),
            "opposing synapse should be detected"
        );
        assert_eq!(candidates[0].from_neuron_uuid, "h1");
        assert_eq!(candidates[0].to_neuron_uuid, "o0");
    }

    #[test]
    fn strongly_opposing_recommends_removal() {
        let (records, creature) = opposing_records(30, 1.0);
        let candidates = detect_opposing_synapses(&creature, &records);
        if !candidates.is_empty() && candidates[0].contribution_error_correlation > 0.5 {
            assert!(candidates[0].recommend_removal);
        }
    }

    // -----------------------------------------------------------------------
    // Exclusion criteria
    // -----------------------------------------------------------------------

    #[test]
    fn hidden_target_synapses_not_analysed() {
        let creature = CreatureJson {
            neurons: vec![
                neuron("i0", "input"),
                neuron("h1", "hidden"),
                neuron("h2", "hidden"),
                neuron("o0", "output"),
            ],
            synapses: vec![
                syn("i0", "h1", 1.0),
                syn("h1", "h2", 1.0), // targets hidden neuron
                syn("h2", "o0", 1.0),
            ],
            input: 1,
            output: 1,
        };
        let recs: Vec<DiscoverRecord> = (0..30)
            .map(|i| DiscoverRecord {
                obs_index: i,
                neuron_uuid: "h1".to_string(),
                value: None,
                activation: i as f32 * 0.1,
                errors: vec![],
            })
            .collect();
        let records = vec![("h1".to_string(), recs)];
        let candidates = detect_opposing_synapses(&creature, &records);
        // h1 → h2 should not be analysed (h2 is hidden, not output)
        let h1_to_h2 = candidates.iter().any(|c| c.to_neuron_uuid == "h2");
        assert!(!h1_to_h2, "hidden target synapses should not be analysed");
    }

    #[test]
    fn low_contribution_excluded() {
        // weight is very small → contribution below MIN_CONTRIBUTION_FOR_OPPOSING
        let creature = CreatureJson {
            neurons: vec![
                neuron("i0", "input"),
                neuron("h1", "hidden"),
                neuron("o0", "output"),
            ],
            synapses: vec![
                syn("i0", "h1", 1.0),
                syn("h1", "o0", 0.0001), // Very small weight
            ],
            input: 1,
            output: 1,
        };
        let source_recs: Vec<DiscoverRecord> = (0..30)
            .map(|i| DiscoverRecord {
                obs_index: i,
                neuron_uuid: "h1".to_string(),
                value: None,
                activation: 0.01, // Small activation too
                errors: vec![],
            })
            .collect();
        let target_recs: Vec<DiscoverRecord> = (0..30)
            .map(|i| DiscoverRecord {
                obs_index: i,
                neuron_uuid: "o0".to_string(),
                value: None,
                activation: 0.5,
                errors: vec![0.5],
            })
            .collect();
        let records = vec![
            ("h1".to_string(), source_recs),
            ("o0".to_string(), target_recs),
        ];
        let candidates = detect_opposing_synapses(&creature, &records);
        assert!(candidates.is_empty(), "low contribution should be excluded");
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn insufficient_samples_not_flagged() {
        let (mut records, creature) = opposing_records(30, 1.0);
        // Truncate to < MIN_SAMPLES
        records[0].1.truncate(5);
        records[1].1.truncate(5);
        let candidates = detect_opposing_synapses(&creature, &records);
        assert!(candidates.is_empty());
    }

    #[test]
    fn empty_creature_no_candidates() {
        let creature = CreatureJson {
            neurons: vec![],
            synapses: vec![],
            input: 0,
            output: 0,
        };
        let candidates = detect_opposing_synapses(&creature, &[]);
        assert!(candidates.is_empty());
    }

    // -----------------------------------------------------------------------
    // Conversion
    // -----------------------------------------------------------------------

    #[test]
    fn removal_candidate_uses_remove_synapse() {
        let candidates = vec![OpposingSynapseCandidate {
            from_neuron_uuid: "h1".to_string(),
            to_neuron_uuid: "o0".to_string(),
            weight: 1.0,
            contribution_error_correlation: 0.8,
            mean_abs_contribution: 0.5,
            sample_count: 30,
            recommend_removal: true,
            estimated_improvement: 0.02,
        }];
        let coordinated = opposing_synapses_to_coordinated_candidates(&candidates);
        assert_eq!(coordinated.len(), 1);
        assert!(matches!(
            &coordinated[0].operations[0],
            CoordinatedStructuralOpJson::RemoveSynapse { .. }
        ));
    }

    #[test]
    fn moderate_opposition_uses_set_weight() {
        let candidates = vec![OpposingSynapseCandidate {
            from_neuron_uuid: "h1".to_string(),
            to_neuron_uuid: "o0".to_string(),
            weight: 0.5,
            contribution_error_correlation: 0.35,
            mean_abs_contribution: 0.3,
            sample_count: 30,
            recommend_removal: false,
            estimated_improvement: 0.01,
        }];
        let coordinated = opposing_synapses_to_coordinated_candidates(&candidates);
        assert_eq!(coordinated.len(), 1);
        assert!(matches!(
            &coordinated[0].operations[0],
            CoordinatedStructuralOpJson::SetWeight { weight, .. } if *weight == -0.5
        ));
    }
}
