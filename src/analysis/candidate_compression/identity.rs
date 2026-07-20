//! IDENTITY neuron compression logic (Issue #921).
//!
//! Uses a hidden IDENTITY neuron that sums all inputs — mathematically equivalent
//! to applying the individual synapses separately.

#![allow(clippy::cast_possible_truncation)] // Intentional numeric casts for neural network computation (Issue #873)
#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for neural network computation (Issue #873)

use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson};

use crate::CandidateSynapseJson;

use super::CompressibleGroup;
use super::grouping::detect_compressible_groups;
use crate::analysis::constants::{
    MAX_COMPRESSION_INPUTS, MIN_COMPRESSED_SOURCES, MIN_COORDINATED_MULTI_OP_GAIN,
    coordinated_empirical_discount,
};

/// Compress a group of compatible IDENTITY candidates into a single coordinated
/// structural candidate.
///
/// Returns `None` if the discounted gain does not exceed `MIN_COORDINATED_MULTI_OP_GAIN`.
fn compress_group(
    group: &CompressibleGroup,
    creature: &CreatureJson,
) -> Option<CoordinatedStructuralCandidateJson> {
    // Cap the number of inputs per compressed candidate.
    let mut sorted_candidates = group.candidates.clone();
    sorted_candidates.sort_by(|a, b| {
        b.expected_creature_score_gain
            .total_cmp(&a.expected_creature_score_gain)
    });
    sorted_candidates.truncate(MAX_COMPRESSION_INPUTS);

    if sorted_candidates.len() < MIN_COMPRESSED_SOURCES {
        return None;
    }

    // Combined gain = sum of individual gains (linear additivity for IDENTITY).
    let combined_gain: f32 = sorted_candidates
        .iter()
        .map(|c| c.expected_creature_score_gain)
        .sum();

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

    // Determine insert position: insert before target neuron.
    let insert_before = creature
        .neurons
        .iter()
        .find(|n| n.uuid == group.to_neuron_uuid)
        .map(|n| n.uuid.clone());

    let mut operations = Vec::with_capacity(op_count);

    // 1. Add the hidden IDENTITY neuron with bias=0.
    operations.push(CoordinatedStructuralOpJson::AddNeuron {
        neuron_uuid: neuron_uuid.clone(),
        neuron_type: "hidden".to_string(),
        squash: "IDENTITY".to_string(),
        bias: 0.0,
        insert_before_neuron_uuid: insert_before,
    });

    // 2. Add synapses from each input to the hidden neuron, preserving original weights.
    for candidate in &sorted_candidates {
        operations.push(CoordinatedStructuralOpJson::AddSynapse {
            from_neuron_uuid: candidate.from_neuron_uuid.clone(),
            to_neuron_uuid: neuron_uuid.clone(),
            weight: candidate.weight,
        });
    }

    // 3. Add synapse from hidden neuron to target (weight=1.0 since IDENTITY passes sum).
    operations.push(CoordinatedStructuralOpJson::AddSynapse {
        from_neuron_uuid: neuron_uuid,
        to_neuron_uuid: group.to_neuron_uuid.clone(),
        weight: 1.0,
    });

    let input_labels: Vec<&str> = sorted_candidates
        .iter()
        .map(|c| c.from_neuron_uuid.as_str())
        .collect();

    Some(CoordinatedStructuralCandidateJson {
        remove_neuron_compensation: None,
        constant_neuron_bias_fold: None,
        operations,
        expected_creature_score_gain: discounted_gain,
        comment: Some(format!(
            "Compressed IDENTITY: {} inputs [{}] → {}",
            sorted_candidates.len(),
            input_labels.join(", "),
            group.to_neuron_uuid,
        )),
    })
}

/// Compress compatible IDENTITY candidates into coordinated structural candidates.
///
/// Detects compressible groups from `helpful_synapses`, compresses each group,
/// and returns the resulting coordinated candidates. The original individual
/// candidates are preserved alongside the compressed ones (the caller is responsible
/// for merging).
pub fn compress_identity_candidates(
    helpful_synapses: &[CandidateSynapseJson],
    creature: &CreatureJson,
) -> Vec<CoordinatedStructuralCandidateJson> {
    let groups = detect_compressible_groups(helpful_synapses);

    let mut compressed = Vec::new();
    for group in &groups {
        if let Some(candidate) = compress_group(group, creature) {
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
    use crate::analysis::constants::coordinated_empirical_discount;
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
            improvement_magnitude_ratio: None,
            target_neuron_stats: None,
            outlier_reduction_info: None,
            prediction_confidence: 0.8,
            expected_score_gain_confidence_interval: [gain * 0.5, gain * 1.5],
            comment: None,
            variant_key: None,
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

    fn synapse(from: &str, to: &str, weight: f32) -> SynapseJson {
        SynapseJson {
            from_uuid: from.to_string(),
            to_uuid: to.to_string(),
            weight,
            synapse_type: None,
        }
    }

    #[test]
    fn test_compress_produces_valid_operations() {
        let candidates = vec![
            make_candidate("input-a", "output-1", 0.3, 0.05),
            make_candidate("input-b", "output-1", 0.5, 0.06),
        ];
        let creature = make_creature(
            vec![
                neuron("input-a", "input"),
                neuron("input-b", "input"),
                neuron("output-1", "output"),
            ],
            vec![
                synapse("input-a", "output-1", 0.3),
                synapse("input-b", "output-1", 0.5),
            ],
        );

        let compressed = compress_identity_candidates(&candidates, &creature);
        assert_eq!(
            compressed.len(),
            1,
            "Should produce one compressed candidate"
        );

        let c = &compressed[0];
        // 2 inputs → 4 operations: 1 AddNeuron + 2 AddSynapse (inputs) + 1 AddSynapse (output).
        assert_eq!(c.operations.len(), 4, "Should have 4 operations");

        // Verify operation types via JSON serialisation.
        let ops_json = serde_json::to_string(&c.operations).unwrap();
        assert!(ops_json.contains("addNeuron"), "Should include addNeuron");
        assert!(ops_json.contains("addSynapse"), "Should include addSynapse");
        assert!(
            ops_json.contains("IDENTITY"),
            "Hidden neuron should use IDENTITY squash"
        );
    }

    #[test]
    fn test_compress_gain_calculation() {
        let candidates = vec![
            make_candidate("input-a", "output-1", 0.3, 0.05),
            make_candidate("input-b", "output-1", 0.5, 0.06),
        ];
        let creature = make_creature(
            vec![
                neuron("input-a", "input"),
                neuron("input-b", "input"),
                neuron("output-1", "output"),
            ],
            vec![
                synapse("input-a", "output-1", 0.3),
                synapse("input-b", "output-1", 0.5),
            ],
        );

        let compressed = compress_identity_candidates(&candidates, &creature);
        assert_eq!(compressed.len(), 1);

        let c = &compressed[0];
        // Combined gain = 0.05 + 0.06 = 0.11
        // 4 operations → empirical discount for 4+ ops.
        let expected_combined = 0.05_f32 + 0.06;
        let expected_discounted = expected_combined * coordinated_empirical_discount(4);
        let tolerance = 1e-6;
        assert!(
            (c.expected_creature_score_gain - expected_discounted).abs() < tolerance,
            "Discounted gain should be ~{expected_discounted}, got {}",
            c.expected_creature_score_gain
        );
    }

    #[test]
    fn test_compress_below_min_gain_threshold() {
        // Issue #1058: With lowered threshold (1e-5) and empirical discount (0.1 for 4+ ops),
        // use truly tiny gains. Combined = 2e-5, discounted = 2e-5 × 0.1 = 2e-6 < 1e-5.
        let candidates = vec![
            make_candidate("input-a", "output-1", 0.3, 1e-5),
            make_candidate("input-b", "output-1", 0.5, 1e-5),
        ];
        let creature = make_creature(
            vec![
                neuron("input-a", "input"),
                neuron("input-b", "input"),
                neuron("output-1", "output"),
            ],
            vec![
                synapse("input-a", "output-1", 0.3),
                synapse("input-b", "output-1", 0.5),
            ],
        );

        let compressed = compress_identity_candidates(&candidates, &creature);
        assert!(
            compressed.is_empty(),
            "Candidates below minimum gain threshold should be filtered out"
        );
    }

    #[test]
    fn test_compress_preserves_original_weights() {
        let candidates = vec![
            make_candidate("input-a", "output-1", 0.3, 0.05),
            make_candidate("input-b", "output-1", 0.7, 0.06),
        ];
        let creature = make_creature(
            vec![
                neuron("input-a", "input"),
                neuron("input-b", "input"),
                neuron("output-1", "output"),
            ],
            vec![
                synapse("input-a", "output-1", 0.3),
                synapse("input-b", "output-1", 0.7),
            ],
        );

        let compressed = compress_identity_candidates(&candidates, &creature);
        assert_eq!(compressed.len(), 1);

        // Check that input synapse weights match original candidate weights.
        let ops_json = serde_json::to_string(&compressed[0].operations).unwrap();
        assert!(
            ops_json.contains("0.3") || ops_json.contains("0.30"),
            "Should preserve weight 0.3"
        );
        assert!(
            ops_json.contains("0.7") || ops_json.contains("0.70"),
            "Should preserve weight 0.7"
        );
        // Output synapse weight should be 1.0.
        assert!(
            ops_json.contains("1.0") || ops_json.contains("1.00"),
            "Output synapse weight should be 1.0"
        );
    }

    #[test]
    fn test_compress_caps_at_max_inputs() {
        // Create more candidates than MAX_COMPRESSION_INPUTS.
        let mut candidates = Vec::new();
        let weights: [f32; 8] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let gains: [f32; 8] = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12];
        for (i, (&w, &g)) in weights.iter().zip(gains.iter()).enumerate() {
            candidates.push(make_candidate(&format!("input-{i}"), "output-1", w, g));
        }
        let mut neurons = vec![neuron("output-1", "output")];
        let mut synapses = Vec::new();
        for (i, &w) in weights.iter().enumerate() {
            neurons.push(neuron(&format!("input-{i}"), "input"));
            synapses.push(synapse(&format!("input-{i}"), "output-1", w));
        }
        let creature = make_creature(neurons, synapses);

        let compressed = compress_identity_candidates(&candidates, &creature);

        if !compressed.is_empty() {
            // The number of AddSynapse ops to the hidden neuron should be capped.
            let input_synapse_count = compressed[0]
                .operations
                .iter()
                .filter(|op| {
                    matches!(op, CoordinatedStructuralOpJson::AddSynapse { to_neuron_uuid, .. }
                        if to_neuron_uuid.starts_with("compress-"))
                })
                .count();
            assert!(
                input_synapse_count <= MAX_COMPRESSION_INPUTS,
                "Input count should be capped at {MAX_COMPRESSION_INPUTS}, got {input_synapse_count}"
            );
        }
    }

    #[test]
    fn test_compress_has_descriptive_comment() {
        let candidates = vec![
            make_candidate("input-a", "output-1", 0.3, 0.05),
            make_candidate("input-b", "output-1", 0.5, 0.06),
        ];
        let creature = make_creature(
            vec![
                neuron("input-a", "input"),
                neuron("input-b", "input"),
                neuron("output-1", "output"),
            ],
            vec![
                synapse("input-a", "output-1", 0.3),
                synapse("input-b", "output-1", 0.5),
            ],
        );

        let compressed = compress_identity_candidates(&candidates, &creature);
        assert_eq!(compressed.len(), 1);
        assert!(
            compressed[0].comment.is_some(),
            "Compressed candidate should have a comment"
        );
        let comment = compressed[0].comment.as_ref().unwrap();
        assert!(
            comment.contains("IDENTITY"),
            "Comment should mention IDENTITY"
        );
    }

    #[test]
    fn test_compress_insert_before_target() {
        let candidates = vec![
            make_candidate("input-a", "output-1", 0.3, 0.05),
            make_candidate("input-b", "output-1", 0.5, 0.06),
        ];
        let creature = make_creature(
            vec![
                neuron("input-a", "input"),
                neuron("input-b", "input"),
                neuron("output-1", "output"),
            ],
            vec![
                synapse("input-a", "output-1", 0.3),
                synapse("input-b", "output-1", 0.5),
            ],
        );

        let compressed = compress_identity_candidates(&candidates, &creature);
        assert_eq!(compressed.len(), 1);

        // First operation should be AddNeuron with insert_before_neuron_uuid = Some("output-1").
        match &compressed[0].operations[0] {
            CoordinatedStructuralOpJson::AddNeuron {
                insert_before_neuron_uuid,
                ..
            } => {
                assert_eq!(
                    insert_before_neuron_uuid.as_deref(),
                    Some("output-1"),
                    "Should insert before target neuron"
                );
            }
            _ => panic!("First operation should be AddNeuron"),
        }
    }
}
