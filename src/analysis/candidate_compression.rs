//! Candidate compression for IDENTITY squash candidates (Issue #921).
//!
//! When multiple independently-discovered `CandidateSynapseJson` candidates target
//! the same output neuron via IDENTITY, this module compresses them into a single
//! `CoordinatedStructuralCandidateJson`:
//!
//! ```text
//! Before (N separate candidates):
//!   input-A → output    (gain = 0.01)
//!   input-B → output    (gain = 0.02)
//!
//! After (1 compressed candidate):
//!   input-A --+
//!             +-→ [hidden IDENTITY, bias=0] → output   (combined gain)
//!   input-B --+
//! ```
//!
//! The compressed candidate uses one hidden IDENTITY neuron that sums all inputs —
//! mathematically equivalent to applying the individual synapses separately, but
//! tested as a single atomic mutation.

#![allow(clippy::cast_possible_truncation)] // Intentional numeric casts for neural network computation (Issue #873)
#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for neural network computation (Issue #873)

use std::collections::HashMap;

use crate::{
    CandidateSynapseJson, CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson,
    CreatureJson,
};

use super::constants::{
    COORDINATED_OPERATION_DISCOUNT, MAX_COMPRESSION_INPUTS, MIN_COMPRESSED_SOURCES,
    MIN_COORDINATED_MULTI_OP_GAIN,
};

/// A group of synapse candidates that share the same target neuron and can be
/// compressed into a single coordinated structural candidate.
#[derive(Debug)]
pub struct CompressibleGroup {
    pub to_neuron_uuid: String,
    pub candidates: Vec<CandidateSynapseJson>,
}

/// Detect groups of helpful synapse candidates that can be compressed.
///
/// Groups candidates by `to_neuron_uuid`, then filters for groups with
/// ≥ `MIN_COMPRESSED_SOURCES` candidates from distinct `from_neuron_uuid` sources.
pub fn detect_compressible_groups(
    helpful_synapses: &[CandidateSynapseJson],
) -> Vec<CompressibleGroup> {
    // Group by target neuron.
    let mut by_target: HashMap<String, Vec<CandidateSynapseJson>> = HashMap::new();
    for candidate in helpful_synapses {
        by_target
            .entry(candidate.to_neuron_uuid.clone())
            .or_default()
            .push(candidate.clone());
    }

    let mut groups = Vec::new();
    for (to_uuid, candidates) in by_target {
        // Deduplicate by from_neuron_uuid — keep the best candidate per source.
        let mut best_by_source: HashMap<String, CandidateSynapseJson> = HashMap::new();
        for c in candidates {
            best_by_source
                .entry(c.from_neuron_uuid.clone())
                .and_modify(|existing| {
                    if c.expected_creature_score_gain > existing.expected_creature_score_gain {
                        *existing = c.clone();
                    }
                })
                .or_insert(c);
        }

        let distinct_candidates: Vec<CandidateSynapseJson> = best_by_source.into_values().collect();

        if distinct_candidates.len() >= MIN_COMPRESSED_SOURCES {
            groups.push(CompressibleGroup {
                to_neuron_uuid: to_uuid,
                candidates: distinct_candidates,
            });
        }
    }

    // Sort groups by target UUID for deterministic output.
    groups.sort_by(|a, b| a.to_neuron_uuid.cmp(&b.to_neuron_uuid));
    groups
}

/// Generate a deterministic UUID for a compressed hidden neuron using FNV-1a hash.
///
/// The UUID is derived from the sorted input UUIDs and target UUID, following
/// the same pattern as `fan_in.rs`.
pub fn generate_compression_uuid(input_uuids: &[String], target_uuid: &str) -> String {
    let mut sorted_inputs: Vec<&str> = input_uuids.iter().map(String::as_str).collect();
    sorted_inputs.sort();

    // FNV-1a 64-bit hash.
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let fnv_prime: u64 = 0x0100_0000_01b3;

    for byte in b"compress:" {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(fnv_prime);
    }
    for input in &sorted_inputs {
        for byte in input.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(fnv_prime);
        }
        // Separator between inputs.
        hash ^= u64::from(b'+');
        hash = hash.wrapping_mul(fnv_prime);
    }
    for byte in b"->" {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(fnv_prime);
    }
    for byte in target_uuid.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(fnv_prime);
    }

    format!("compress-{hash:016x}")
}

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

    let neuron_uuid = generate_compression_uuid(&input_uuids, &group.to_neuron_uuid);

    // N inputs → N+2 operations (1 AddNeuron + N AddSynapse inputs + 1 AddSynapse output).
    let op_count = sorted_candidates.len() + 2;
    let exponent = (op_count - 1) as f32;
    let discounted_gain = combined_gain * COORDINATED_OPERATION_DISCOUNT.powf(exponent);

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

    fn synapse(from: &str, to: &str, weight: f32) -> SynapseJson {
        SynapseJson {
            from_uuid: from.to_string(),
            to_uuid: to.to_string(),
            weight,
            synapse_type: None,
        }
    }

    // -------------------------------------------------------------------------
    // Detection tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_detect_compressible_groups_basic() {
        let candidates = vec![
            make_candidate("input-a", "output-1", 0.3, 0.02),
            make_candidate("input-b", "output-1", 0.5, 0.03),
        ];

        let groups = detect_compressible_groups(&candidates);
        assert_eq!(groups.len(), 1, "Should detect one compressible group");
        assert_eq!(groups[0].to_neuron_uuid, "output-1");
        assert_eq!(groups[0].candidates.len(), 2);
    }

    #[test]
    fn test_detect_no_groups_for_single_candidate() {
        let candidates = vec![make_candidate("input-a", "output-1", 0.3, 0.02)];

        let groups = detect_compressible_groups(&candidates);
        assert!(
            groups.is_empty(),
            "Single candidate per target should not form a group"
        );
    }

    #[test]
    fn test_detect_no_groups_for_same_source() {
        // Two candidates from the same source to the same target — only one distinct source.
        let candidates = vec![
            make_candidate("input-a", "output-1", 0.3, 0.02),
            make_candidate("input-a", "output-1", 0.5, 0.03),
        ];

        let groups = detect_compressible_groups(&candidates);
        assert!(
            groups.is_empty(),
            "Same source should not form a compressible group"
        );
    }

    #[test]
    fn test_detect_multiple_groups() {
        let candidates = vec![
            make_candidate("input-a", "output-1", 0.3, 0.02),
            make_candidate("input-b", "output-1", 0.5, 0.03),
            make_candidate("input-c", "output-2", 0.4, 0.01),
            make_candidate("input-d", "output-2", 0.6, 0.04),
        ];

        let groups = detect_compressible_groups(&candidates);
        assert_eq!(groups.len(), 2, "Should detect two compressible groups");
    }

    #[test]
    fn test_detect_empty_input() {
        let groups = detect_compressible_groups(&[]);
        assert!(groups.is_empty(), "Empty input should produce no groups");
    }

    // -------------------------------------------------------------------------
    // UUID generation tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_deterministic_uuid() {
        let inputs = vec!["input-a".to_string(), "input-b".to_string()];
        let uuid1 = generate_compression_uuid(&inputs, "output-1");
        let uuid2 = generate_compression_uuid(&inputs, "output-1");
        assert_eq!(uuid1, uuid2, "UUIDs should be deterministic");
    }

    #[test]
    fn test_uuid_order_independent() {
        let inputs1 = vec!["input-b".to_string(), "input-a".to_string()];
        let inputs2 = vec!["input-a".to_string(), "input-b".to_string()];
        let uuid1 = generate_compression_uuid(&inputs1, "output-1");
        let uuid2 = generate_compression_uuid(&inputs2, "output-1");
        assert_eq!(
            uuid1, uuid2,
            "UUIDs should be the same regardless of input order"
        );
    }

    #[test]
    fn test_uuid_differs_for_different_targets() {
        let inputs = vec!["input-a".to_string(), "input-b".to_string()];
        let uuid1 = generate_compression_uuid(&inputs, "output-1");
        let uuid2 = generate_compression_uuid(&inputs, "output-2");
        assert_ne!(
            uuid1, uuid2,
            "Different targets should produce different UUIDs"
        );
    }

    #[test]
    fn test_uuid_starts_with_compress_prefix() {
        let inputs = vec!["input-a".to_string(), "input-b".to_string()];
        let uuid = generate_compression_uuid(&inputs, "output-1");
        assert!(
            uuid.starts_with("compress-"),
            "UUID should start with 'compress-' prefix"
        );
    }

    // -------------------------------------------------------------------------
    // Compression output tests
    // -------------------------------------------------------------------------

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
        // 4 operations → discount = 0.65^3 ≈ 0.274625
        // Discounted gain ≈ 0.11 * 0.274625 ≈ 0.03021
        let expected_combined = 0.05_f32 + 0.06;
        let expected_discounted = expected_combined * COORDINATED_OPERATION_DISCOUNT.powf(3.0);
        let tolerance = 1e-6;
        assert!(
            (c.expected_creature_score_gain - expected_discounted).abs() < tolerance,
            "Discounted gain should be ~{expected_discounted}, got {}",
            c.expected_creature_score_gain
        );
    }

    #[test]
    fn test_compress_below_min_gain_threshold() {
        // Very small gains that after discounting will be below MIN_COORDINATED_MULTI_OP_GAIN.
        let candidates = vec![
            make_candidate("input-a", "output-1", 0.3, 0.001),
            make_candidate("input-b", "output-1", 0.5, 0.001),
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
        // Combined = 0.002, discounted = 0.002 * 0.65^3 ≈ 0.000549
        // This is below MIN_COORDINATED_MULTI_OP_GAIN (1e-3), so should be empty.
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
