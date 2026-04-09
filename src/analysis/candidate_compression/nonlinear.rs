//! Non-linear compression logic (Issue #922).
//!
//! Uses TANH or GELU hidden neurons to capture interaction effects between inputs.
//! The combined signal through a non-linear neuron is not simply the sum of
//! individual effects — saturation-aware gain estimation accounts for diminished
//! returns in saturated regimes.

#![allow(clippy::cast_possible_truncation)] // Intentional numeric casts for neural network computation (Issue #873)
#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for neural network computation (Issue #873)

use crate::{
    CandidateSynapseJson, CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson,
    CreatureJson,
};

use super::CompressibleGroup;
use super::gain_estimation::estimate_nonlinear_gain;
use super::grouping::detect_compressible_groups;
use crate::analysis::constants::{
    MAX_COMPRESSION_INPUTS, MIN_COMPRESSED_SOURCES, MIN_COORDINATED_MULTI_OP_GAIN,
    coordinated_empirical_discount,
};

/// Supported non-linear squash functions for compression.
const NONLINEAR_SQUASH_FUNCTIONS: &[&str] = &["TANH", "GELU"];

/// Select the squash function for non-linear compression.
///
/// If the target neuron already uses a non-linear squash (TANH or GELU), use
/// that. Otherwise default to TANH (matching fan-in module behaviour).
fn select_nonlinear_squash(creature: &CreatureJson, target_uuid: &str) -> &'static str {
    // Issue #983: Squash strings are already uppercase at load time (Issue #771),
    // so compare directly without allocating via `to_uppercase()`.
    if let Some(target) = creature.neurons.iter().find(|n| n.uuid == target_uuid) {
        for &s in NONLINEAR_SQUASH_FUNCTIONS {
            if target.squash == s {
                return s;
            }
        }
    }
    // Default to TANH (matches fan_in.rs:59).
    "TANH"
}

/// Compress a group of candidates into a non-linear coordinated structural candidate.
///
/// Returns `None` if:
/// - The non-linear gain estimation fails or is below threshold
/// - The discounted gain does not exceed `MIN_COORDINATED_MULTI_OP_GAIN`
fn compress_group_nonlinear(
    group: &CompressibleGroup,
    creature: &CreatureJson,
) -> Option<CoordinatedStructuralCandidateJson> {
    let mut sorted_candidates = group.candidates.clone();
    sorted_candidates.sort_by(|a, b| {
        b.expected_creature_score_gain
            .total_cmp(&a.expected_creature_score_gain)
    });
    sorted_candidates.truncate(MAX_COMPRESSION_INPUTS);

    if sorted_candidates.len() < MIN_COMPRESSED_SOURCES {
        return None;
    }

    let squash_name = select_nonlinear_squash(creature, &group.to_neuron_uuid);

    // Estimate non-linear gain with saturation awareness.
    let combined_gain = estimate_nonlinear_gain(&sorted_candidates, squash_name)?;

    if combined_gain <= 0.0 {
        return None;
    }

    let input_uuids: Vec<String> = sorted_candidates
        .iter()
        .map(|c| c.from_neuron_uuid.clone())
        .collect();

    let neuron_uuid = super::generate_compression_uuid(&input_uuids, &group.to_neuron_uuid);

    // N inputs → N+2 operations (1 AddNeuron + N AddSynapse inputs + 1 AddSynapse output).
    let op_count = sorted_candidates.len() + 2;
    let discounted_gain = combined_gain * coordinated_empirical_discount(op_count);

    if discounted_gain <= MIN_COORDINATED_MULTI_OP_GAIN {
        return None;
    }

    // Insert before target neuron.
    let insert_before = creature
        .neurons
        .iter()
        .find(|n| n.uuid == group.to_neuron_uuid)
        .map(|n| n.uuid.clone());

    let mut operations = Vec::with_capacity(op_count);

    // 1. Add the hidden non-linear neuron with bias=0.
    operations.push(CoordinatedStructuralOpJson::AddNeuron {
        neuron_uuid: neuron_uuid.clone(),
        neuron_type: "hidden".to_string(),
        squash: squash_name.to_string(),
        bias: 0.0,
        insert_before_neuron_uuid: insert_before,
    });

    // 2. Add synapses from each input to the hidden neuron.
    for candidate in &sorted_candidates {
        operations.push(CoordinatedStructuralOpJson::AddSynapse {
            from_neuron_uuid: candidate.from_neuron_uuid.clone(),
            to_neuron_uuid: neuron_uuid.clone(),
            weight: candidate.weight,
        });
    }

    // 3. Add synapse from hidden neuron to target (weight=0.1 for non-linear,
    //    matching fan-in conservative output weight).
    operations.push(CoordinatedStructuralOpJson::AddSynapse {
        from_neuron_uuid: neuron_uuid,
        to_neuron_uuid: group.to_neuron_uuid.clone(),
        weight: 0.1,
    });

    let input_labels: Vec<&str> = sorted_candidates
        .iter()
        .map(|c| c.from_neuron_uuid.as_str())
        .collect();

    Some(CoordinatedStructuralCandidateJson {
        operations,
        expected_creature_score_gain: discounted_gain,
        comment: Some(format!(
            "Compressed {squash_name}: {} inputs [{}] → {}",
            sorted_candidates.len(),
            input_labels.join(", "),
            group.to_neuron_uuid,
        )),
    })
}

/// Compress compatible candidates into non-linear coordinated structural candidates (Issue #922).
///
/// For each compressible group, attempts TANH/GELU compression with saturation-aware
/// gain estimation. Only emits candidates where the combined gain exceeds the best
/// individual gain by `COMPRESSION_MIN_BENEFIT_RATIO` (1.05).
///
/// Returns non-linear compressed candidates alongside (not replacing) IDENTITY
/// compressed candidates. The caller merges both sets into the pipeline.
pub fn compress_nonlinear_candidates(
    helpful_synapses: &[CandidateSynapseJson],
    creature: &CreatureJson,
) -> Vec<CoordinatedStructuralCandidateJson> {
    let groups = detect_compressible_groups(helpful_synapses);

    let mut compressed = Vec::new();
    for group in &groups {
        if let Some(candidate) = compress_group_nonlinear(group, creature) {
            compressed.push(candidate);
        }
    }

    // Sort by gain descending for deterministic output.
    compressed.sort_by(|a, b| {
        b.expected_creature_score_gain
            .total_cmp(&a.expected_creature_score_gain)
    });

    compressed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NeuronJson, SynapseJson};

    fn make_candidate(from: &str, to: &str, weight: f32, gain: f32) -> CandidateSynapseJson {
        CandidateSynapseJson {
            from_neuron_uuid: from.to_string(),
            to_neuron_uuid: to.to_string(),
            from_neuron_index: None,
            to_neuron_index: None,
            weight,
            target_neuron_impact: 1.0,
            expected_creature_error_reduction: gain,
            expected_creature_score_gain: gain,
            improved_count: 80,
            total_count: 100,
            target_neuron_stats: None,
            outlier_reduction_info: None,
            prediction_confidence: 0.8,
            expected_score_gain_confidence_interval: [gain * 0.5, gain * 1.5],
            comment: None,
        }
    }

    fn make_creature(neurons: Vec<NeuronJson>, synapses: Vec<SynapseJson>) -> CreatureJson {
        let input_count = neurons.iter().filter(|n| n.neuron_type == "input").count();
        let output_count = neurons.iter().filter(|n| n.neuron_type == "output").count();
        CreatureJson {
            neurons,
            synapses,
            input: input_count,
            output: output_count,
        }
    }

    fn neuron(uuid: &str, neuron_type: &str) -> NeuronJson {
        NeuronJson {
            uuid: uuid.to_string(),
            neuron_type: neuron_type.to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        }
    }

    fn neuron_with_squash(uuid: &str, neuron_type: &str, squash: &str) -> NeuronJson {
        NeuronJson {
            uuid: uuid.to_string(),
            neuron_type: neuron_type.to_string(),
            squash: squash.to_string(),
            bias: 0.0,
        }
    }

    fn synapse(from: &str, to: &str, weight: f32) -> SynapseJson {
        SynapseJson {
            from_uuid: from.to_string(),
            to_uuid: to.to_string(),
            weight,
            synapse_type: None,
        }
    }

    #[test]
    fn test_nonlinear_tanh_compression_linear_regime() {
        // Small weights keep TANH in its linear regime — compression should succeed.
        let candidates = vec![
            make_candidate("input-a", "output-1", 0.3, 0.05),
            make_candidate("input-b", "output-1", 0.4, 0.06),
        ];
        let creature = make_creature(
            vec![
                neuron("input-a", "input"),
                neuron("input-b", "input"),
                neuron("output-1", "output"),
            ],
            vec![
                synapse("input-a", "output-1", 0.3),
                synapse("input-b", "output-1", 0.4),
            ],
        );

        let compressed = compress_nonlinear_candidates(&candidates, &creature);
        assert!(
            !compressed.is_empty(),
            "TANH compression should succeed in linear regime"
        );

        // Verify the hidden neuron uses TANH (default).
        match &compressed[0].operations[0] {
            CoordinatedStructuralOpJson::AddNeuron { squash, .. } => {
                assert_eq!(squash, "TANH", "Should default to TANH squash");
            }
            _ => panic!("First operation should be AddNeuron"),
        }
    }

    #[test]
    fn test_nonlinear_tanh_saturated_inputs_diminished() {
        // Large weights push TANH into saturation — gain should be diminished.
        let candidates = vec![
            make_candidate("input-a", "output-1", 3.0, 0.05),
            make_candidate("input-b", "output-1", 3.0, 0.06),
        ];
        let creature = make_creature(
            vec![
                neuron("input-a", "input"),
                neuron("input-b", "input"),
                neuron("output-1", "output"),
            ],
            vec![
                synapse("input-a", "output-1", 3.0),
                synapse("input-b", "output-1", 3.0),
            ],
        );

        // With saturated inputs, combined TANH output ≈ 1.0, individual outputs ≈ 1.0 each.
        // The benefit ratio check should filter this out since combining saturated
        // inputs adds no benefit.
        let compressed = compress_nonlinear_candidates(&candidates, &creature);

        // Either filtered out entirely or has reduced gain.
        if !compressed.is_empty() {
            // Gain should be less than sum of individual gains (saturation penalty).
            let individual_sum: f32 = candidates
                .iter()
                .map(|c| c.expected_creature_score_gain)
                .sum();
            assert!(
                compressed[0].expected_creature_score_gain < individual_sum,
                "Saturated TANH should produce lower gain than linear sum"
            );
        }
    }

    #[test]
    fn test_nonlinear_gelu_compression() {
        // GELU compression with moderate positive weights.
        let candidates = vec![
            make_candidate("input-a", "output-1", 0.5, 0.05),
            make_candidate("input-b", "output-1", 0.6, 0.06),
        ];
        let creature = make_creature(
            vec![
                neuron("input-a", "input"),
                neuron("input-b", "input"),
                neuron_with_squash("output-1", "output", "GELU"),
            ],
            vec![
                synapse("input-a", "output-1", 0.5),
                synapse("input-b", "output-1", 0.6),
            ],
        );

        let compressed = compress_nonlinear_candidates(&candidates, &creature);

        // Should produce a GELU compressed candidate (target neuron uses GELU).
        if !compressed.is_empty() {
            match &compressed[0].operations[0] {
                CoordinatedStructuralOpJson::AddNeuron { squash, .. } => {
                    assert_eq!(squash, "GELU", "Should use target's GELU squash");
                }
                _ => panic!("First operation should be AddNeuron"),
            }
        }
    }

    #[test]
    fn test_nonlinear_selects_target_squash() {
        let creature = make_creature(
            vec![
                neuron("input-a", "input"),
                neuron_with_squash("output-1", "output", "GELU"),
            ],
            vec![],
        );
        assert_eq!(
            select_nonlinear_squash(&creature, "output-1"),
            "GELU",
            "Should select target neuron's GELU squash"
        );
    }

    #[test]
    fn test_nonlinear_defaults_to_tanh() {
        let creature = make_creature(
            vec![
                neuron("input-a", "input"),
                neuron("output-1", "output"), // IDENTITY squash
            ],
            vec![],
        );
        assert_eq!(
            select_nonlinear_squash(&creature, "output-1"),
            "TANH",
            "Should default to TANH when target is not non-linear"
        );
    }

    #[test]
    fn test_nonlinear_output_weight_is_conservative() {
        // Non-linear compression should use conservative output weight (0.1).
        let candidates = vec![
            make_candidate("input-a", "output-1", 0.3, 0.05),
            make_candidate("input-b", "output-1", 0.4, 0.06),
        ];
        let creature = make_creature(
            vec![
                neuron("input-a", "input"),
                neuron("input-b", "input"),
                neuron("output-1", "output"),
            ],
            vec![
                synapse("input-a", "output-1", 0.3),
                synapse("input-b", "output-1", 0.4),
            ],
        );

        let compressed = compress_nonlinear_candidates(&candidates, &creature);
        assert!(!compressed.is_empty());

        // Last operation is AddSynapse to target with weight=0.1.
        let last_op = compressed[0].operations.last().unwrap();
        match last_op {
            CoordinatedStructuralOpJson::AddSynapse { weight, .. } => {
                assert!(
                    (*weight - 0.1).abs() < f32::EPSILON,
                    "Non-linear output synapse weight should be 0.1, got {weight}"
                );
            }
            _ => panic!("Last operation should be AddSynapse"),
        }
    }

    #[test]
    fn test_nonlinear_comment_mentions_squash() {
        let candidates = vec![
            make_candidate("input-a", "output-1", 0.3, 0.05),
            make_candidate("input-b", "output-1", 0.4, 0.06),
        ];
        let creature = make_creature(
            vec![
                neuron("input-a", "input"),
                neuron("input-b", "input"),
                neuron("output-1", "output"),
            ],
            vec![
                synapse("input-a", "output-1", 0.3),
                synapse("input-b", "output-1", 0.4),
            ],
        );

        let compressed = compress_nonlinear_candidates(&candidates, &creature);
        assert!(!compressed.is_empty());

        let comment = compressed[0].comment.as_ref().unwrap();
        assert!(
            comment.contains("TANH"),
            "Comment should mention TANH, got: {comment}"
        );
    }

    #[test]
    fn test_nonlinear_empty_input() {
        let creature = make_creature(vec![neuron("output-1", "output")], vec![]);
        let compressed = compress_nonlinear_candidates(&[], &creature);
        assert!(compressed.is_empty(), "Empty input should produce nothing");
    }
}
