//! Integration tests for Issue #921: Compress compatible IDENTITY candidates
//! into single coordinated structural candidates.
//!
//! ## TDD Plan
//! 1. Verify compressible groups are detected from helpful synapses
//! 2. Verify compressed candidates have valid coordinated operations
//! 3. Verify deterministic UUIDs across runs
//! 4. Verify operation-count discount is applied correctly
//! 5. Verify original individual candidates are preserved alongside compressed ones
//! 6. Verify edge cases (empty input, single candidate, all same source)
//! 7. Verify compressed candidates survive dedup/ensemble/clustering pipeline

use neat_ai_discovery::analysis::candidate_compression::{
    compress_identity_candidates, detect_compressible_groups, generate_compression_uuid,
};
use neat_ai_discovery::analysis::constants::{
    MAX_COMPRESSION_INPUTS, MIN_COMPRESSED_SOURCES, coordinated_empirical_discount,
};
use neat_ai_discovery::{
    CandidateSynapseJson, CoordinatedStructuralOpJson, CreatureJson, NeuronJson, SynapseJson,
};

/// Helper: create a `CandidateSynapseJson`.
fn candidate(from: &str, to: &str, weight: f32, gain: f32) -> CandidateSynapseJson {
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
    }
}

/// Helper: build a `NeuronJson`.
fn neuron(uuid: &str, neuron_type: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
        squash: "IDENTITY".to_string(),
        bias: 0.0,
    }
}

/// Helper: build a `SynapseJson`.
fn synapse(from: &str, to: &str, weight: f32) -> SynapseJson {
    SynapseJson {
        from_uuid: from.to_string(),
        to_uuid: to.to_string(),
        weight,
        synapse_type: None,
    }
}

/// Helper: build a minimal creature.
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

/// Test 1: Compressible groups are detected when multiple candidates target the same neuron.
#[test]
fn test_detect_compressible_groups() {
    let candidates = vec![
        candidate("input-a", "output-1", 0.3, 0.02),
        candidate("input-b", "output-1", 0.5, 0.03),
        candidate("input-c", "output-1", 0.4, 0.01),
    ];

    let groups = detect_compressible_groups(&candidates);
    assert_eq!(groups.len(), 1, "Should detect one compressible group");
    assert_eq!(groups[0].to_neuron_uuid, "output-1");
    assert!(
        groups[0].candidates.len() >= MIN_COMPRESSED_SOURCES,
        "Group should have at least {MIN_COMPRESSED_SOURCES} candidates"
    );
}

/// Test 2: Compressed candidate has valid coordinated operations structure.
#[test]
fn test_compressed_candidate_structure() {
    let candidates = vec![
        candidate("input-a", "output-1", 0.3, 0.05),
        candidate("input-b", "output-1", 0.5, 0.06),
        candidate("input-c", "output-1", 0.4, 0.04),
    ];
    let creature = make_creature(
        vec![
            neuron("input-a", "input"),
            neuron("input-b", "input"),
            neuron("input-c", "input"),
            neuron("output-1", "output"),
        ],
        vec![
            synapse("input-a", "output-1", 0.3),
            synapse("input-b", "output-1", 0.5),
            synapse("input-c", "output-1", 0.4),
        ],
    );

    let compressed = compress_identity_candidates(&candidates, &creature);
    assert!(
        !compressed.is_empty(),
        "Should produce compressed candidates"
    );

    let c = &compressed[0];
    // 3 inputs → 5 operations: 1 AddNeuron + 3 AddSynapse (inputs) + 1 AddSynapse (output).
    assert_eq!(
        c.operations.len(),
        5,
        "Should have 5 operations (1 AddNeuron + 3 input synapses + 1 output synapse)"
    );

    // First operation must be AddNeuron.
    assert!(
        matches!(
            &c.operations[0],
            CoordinatedStructuralOpJson::AddNeuron { .. }
        ),
        "First operation should be AddNeuron"
    );

    // Last operation must be AddSynapse to target with weight 1.0.
    match &c.operations[c.operations.len() - 1] {
        CoordinatedStructuralOpJson::AddSynapse {
            to_neuron_uuid,
            weight,
            ..
        } => {
            assert_eq!(
                to_neuron_uuid, "output-1",
                "Last synapse should target output-1"
            );
            assert!(
                (*weight - 1.0).abs() < f32::EPSILON,
                "Output synapse weight should be 1.0"
            );
        }
        _ => panic!("Last operation should be AddSynapse to target"),
    }
}

/// Test 3: Deterministic UUIDs — same inputs produce the same UUID.
#[test]
fn test_deterministic_compression_uuid() {
    let inputs = vec![
        "input-a".to_string(),
        "input-b".to_string(),
        "input-c".to_string(),
    ];
    let uuid1 = generate_compression_uuid(&inputs, "output-1");
    let uuid2 = generate_compression_uuid(&inputs, "output-1");
    assert_eq!(uuid1, uuid2, "UUID should be deterministic");
    assert!(
        uuid1.starts_with("compress-"),
        "UUID should have 'compress-' prefix"
    );
}

/// Test 4: Operation-count discount is applied correctly.
#[test]
fn test_operation_count_discount() {
    let candidates = vec![
        candidate("input-a", "output-1", 0.3, 0.05),
        candidate("input-b", "output-1", 0.5, 0.06),
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
    // 4 operations → empirical discount for 4+ ops
    let expected = (0.05_f32 + 0.06) * coordinated_empirical_discount(4);
    assert!(
        (c.expected_creature_score_gain - expected).abs() < 1e-6,
        "Discounted gain should be ~{expected}, got {}",
        c.expected_creature_score_gain
    );
}

/// Test 5: Original individual candidates are preserved (caller responsibility —
/// compression returns new candidates, originals untouched).
#[test]
fn test_original_candidates_preserved() {
    let candidates = vec![
        candidate("input-a", "output-1", 0.3, 0.05),
        candidate("input-b", "output-1", 0.5, 0.06),
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

    // The function takes a reference, so original candidates are not consumed.
    let _compressed = compress_identity_candidates(&candidates, &creature);

    // Verify originals still exist unchanged.
    assert_eq!(
        candidates.len(),
        2,
        "Original candidates should be preserved"
    );
    assert_eq!(candidates[0].from_neuron_uuid, "input-a");
    assert_eq!(candidates[1].from_neuron_uuid, "input-b");
}

/// Test 6: Empty input produces no compressed candidates.
#[test]
fn test_empty_input_no_compression() {
    let creature = make_creature(vec![neuron("output-1", "output")], vec![]);
    let compressed = compress_identity_candidates(&[], &creature);
    assert!(
        compressed.is_empty(),
        "Empty input should produce no compressed candidates"
    );
}

/// Test 7: Single candidate per target — no compression.
#[test]
fn test_single_candidate_no_compression() {
    let candidates = vec![candidate("input-a", "output-1", 0.3, 0.05)];
    let creature = make_creature(
        vec![neuron("input-a", "input"), neuron("output-1", "output")],
        vec![synapse("input-a", "output-1", 0.3)],
    );

    let compressed = compress_identity_candidates(&candidates, &creature);
    assert!(
        compressed.is_empty(),
        "Single candidate should not produce compressed candidate"
    );
}

/// Test 8: All candidates from same source — no compression (only 1 distinct source).
#[test]
fn test_same_source_no_compression() {
    let candidates = vec![
        candidate("input-a", "output-1", 0.3, 0.05),
        candidate("input-a", "output-1", 0.5, 0.06),
    ];
    let creature = make_creature(
        vec![neuron("input-a", "input"), neuron("output-1", "output")],
        vec![synapse("input-a", "output-1", 0.3)],
    );

    let compressed = compress_identity_candidates(&candidates, &creature);
    assert!(
        compressed.is_empty(),
        "Same source candidates should not be compressed"
    );
}

/// Test 9: Candidates below gain threshold after discounting are filtered.
#[test]
fn test_below_min_gain_filtered() {
    // Issue #1058: With lowered threshold (1e-5) and empirical discount (0.1 for 4+ ops),
    // use truly tiny gains that will fall below the threshold.
    // Combined = 2e-5, discounted = 2e-5 × 0.1 = 2e-6 < 1e-5.
    let candidates = vec![
        candidate("input-a", "output-1", 0.3, 1e-5),
        candidate("input-b", "output-1", 0.5, 1e-5),
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
        "Candidates below MIN_COORDINATED_MULTI_OP_GAIN after discounting should be filtered"
    );
}

/// Test 10: `MAX_COMPRESSION_INPUTS` caps the number of inputs per compressed candidate.
#[test]
fn test_max_inputs_capped() {
    let weights: [f32; 8] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let gains: [f32; 8] = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12];
    let mut candidates = Vec::new();
    for (i, (&w, &g)) in weights.iter().zip(gains.iter()).enumerate() {
        candidates.push(candidate(&format!("input-{i}"), "output-1", w, g));
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
        // Count input synapses to the hidden neuron.
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
            "Should cap inputs at {MAX_COMPRESSION_INPUTS}, got {input_synapse_count}"
        );
    }
}

/// Test 11: Compressed candidate includes IDENTITY activation for the hidden neuron.
#[test]
fn test_compressed_uses_identity_activation() {
    let candidates = vec![
        candidate("input-a", "output-1", 0.3, 0.05),
        candidate("input-b", "output-1", 0.5, 0.06),
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

    match &compressed[0].operations[0] {
        CoordinatedStructuralOpJson::AddNeuron { squash, bias, .. } => {
            assert_eq!(
                squash, "IDENTITY",
                "Hidden neuron should use IDENTITY squash"
            );
            assert!(
                (*bias - 0.0).abs() < f32::EPSILON,
                "Hidden neuron bias should be 0.0"
            );
        }
        _ => panic!("First operation should be AddNeuron"),
    }
}

/// Test 12: Multiple groups targeting different neurons produce separate candidates.
#[test]
fn test_multiple_target_groups() {
    let candidates = vec![
        candidate("input-a", "output-1", 0.3, 0.05),
        candidate("input-b", "output-1", 0.5, 0.06),
        candidate("input-c", "output-2", 0.4, 0.07),
        candidate("input-d", "output-2", 0.6, 0.08),
    ];
    let creature = make_creature(
        vec![
            neuron("input-a", "input"),
            neuron("input-b", "input"),
            neuron("input-c", "input"),
            neuron("input-d", "input"),
            neuron("output-1", "output"),
            neuron("output-2", "output"),
        ],
        vec![
            synapse("input-a", "output-1", 0.3),
            synapse("input-b", "output-1", 0.5),
            synapse("input-c", "output-2", 0.4),
            synapse("input-d", "output-2", 0.6),
        ],
    );

    let compressed = compress_identity_candidates(&candidates, &creature);
    assert_eq!(
        compressed.len(),
        2,
        "Should produce two compressed candidates for two target groups"
    );
}
