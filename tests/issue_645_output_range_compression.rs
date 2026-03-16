//! Tests for Issue #645: Output neuron range compression detector.
//!
//! Detects output neurons operating in a compressed sub-range of their
//! activation function's theoretical range. For example, a TANH neuron
//! (range [-1, 1]) whose activations only span [0.3, 0.7] is using
//! only 20% of the available range, reducing dynamic resolution.
//!
//! ## TDD Plan
//! 1. Output neuron using <40% of activation range SHOULD be flagged
//! 2. Output neuron using >40% of range should NOT be flagged
//! 3. Hidden neurons are excluded (handled by restricted_range module)
//! 4. Insufficient samples produce no detections
//! 5. Unbounded activation functions (IDENTITY, RELU) are excluded
//! 6. Produces valid coordinated candidates (setBias+setWeight or changeSquash)
//! 7. Distinguishes from output_squash_mismatch (wrong function type vs compression)
//! 8. Multiple output neurons — only compressed ones flagged
//! 9. Near-saturated neurons excluded (saturation detection handles those)
//! 10. Dead output neurons excluded (near-zero range)

use neat_ai_discovery::analysis::detection::output_range_compression::{
    OutputRangeCompressionConfig, OutputRangeCompressionNeuron, detect_output_range_compression,
    output_range_compression_to_coordinated_candidates,
};
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

/// Helper: create a DiscoverRecord.
fn record(neuron_uuid: &str, obs_index: u32, activation: f32, errors: Vec<f32>) -> DiscoverRecord {
    DiscoverRecord {
        obs_index,
        neuron_uuid: neuron_uuid.to_string(),
        value: Some(activation),
        activation,
        errors,
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

fn neuron(uuid: &str, neuron_type: &str, squash: &str, bias: f32) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
        squash: squash.to_string(),
        bias,
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

/// Test 1: Output neuron using TANH with activations in [0.3, 0.7] — only 20% of [-1, 1].
/// Should be flagged as compressed.
#[test]
fn test_compressed_output_detected() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY", 0.0),
            neuron("hidden-1", "hidden", "TANH", 0.0),
            neuron("output-1", "output", "TANH", 0.5),
        ],
        vec![
            synapse("input-1", "hidden-1", 0.5),
            synapse("hidden-1", "output-1", 0.3),
        ],
    );

    // Activations confined to [0.3, 0.7] out of TANH's [-1, 1] range = 20% utilisation
    let records: Vec<DiscoverRecord> = (0..50)
        .map(|i| {
            let activation = 0.3 + (i as f32 / 49.0) * 0.4;
            record("output-1", i, activation, vec![0.1])
        })
        .collect();

    let neuron_records = vec![("output-1".to_string(), records)];

    let config = OutputRangeCompressionConfig::default();
    let detected = detect_output_range_compression(&creature, &neuron_records, &config);

    assert!(
        !detected.is_empty(),
        "Output neuron using only 20% of TANH range should be flagged"
    );
    assert_eq!(detected[0].neuron_uuid, "output-1");
    assert!(
        detected[0].range_utilisation < 0.40,
        "Range utilisation should be below threshold, got {:.2}",
        detected[0].range_utilisation
    );
}

/// Test 2: Output neuron using most of its range should NOT be flagged.
#[test]
fn test_full_range_not_flagged() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY", 0.0),
            neuron("output-1", "output", "TANH", 0.0),
        ],
        vec![synapse("input-1", "output-1", 1.0)],
    );

    // Activations span [-0.8, 0.8] out of [-1, 1] = 80% utilisation
    let records: Vec<DiscoverRecord> = (0..50)
        .map(|i| {
            let activation = -0.8 + (i as f32 / 49.0) * 1.6;
            record("output-1", i, activation, vec![0.05])
        })
        .collect();

    let neuron_records = vec![("output-1".to_string(), records)];

    let config = OutputRangeCompressionConfig::default();
    let detected = detect_output_range_compression(&creature, &neuron_records, &config);

    assert!(
        detected.is_empty(),
        "Output neuron using 80% of range should not be flagged, got {} detections",
        detected.len()
    );
}

/// Test 3: Hidden neurons should be excluded (restricted_range handles those).
#[test]
fn test_hidden_neurons_excluded() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY", 0.0),
            neuron("hidden-1", "hidden", "TANH", 0.0),
            neuron("output-1", "output", "IDENTITY", 0.0),
        ],
        vec![
            synapse("input-1", "hidden-1", 0.5),
            synapse("hidden-1", "output-1", 0.8),
        ],
    );

    // Compressed hidden neuron — should NOT be flagged by this module
    let records: Vec<DiscoverRecord> = (0..50)
        .map(|i| {
            let activation = 0.1 + (i as f32 / 49.0) * 0.2;
            record("hidden-1", i, activation, vec![0.1])
        })
        .collect();

    let neuron_records = vec![("hidden-1".to_string(), records)];

    let config = OutputRangeCompressionConfig::default();
    let detected = detect_output_range_compression(&creature, &neuron_records, &config);

    assert!(
        detected.is_empty(),
        "Hidden neurons should not be flagged by output range compression detector"
    );
}

/// Test 4: Insufficient samples should not trigger detection.
#[test]
fn test_output_range_compression_insufficient_samples_no_detection() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY", 0.0),
            neuron("output-1", "output", "TANH", 0.5),
        ],
        vec![synapse("input-1", "output-1", 0.3)],
    );

    // Only 5 samples (below MIN_SAMPLES threshold)
    let records: Vec<DiscoverRecord> = (0..5)
        .map(|i| {
            let activation = 0.3 + (i as f32 / 4.0) * 0.4;
            record("output-1", i, activation, vec![0.1])
        })
        .collect();

    let neuron_records = vec![("output-1".to_string(), records)];

    let config = OutputRangeCompressionConfig::default();
    let detected = detect_output_range_compression(&creature, &neuron_records, &config);

    assert!(
        detected.is_empty(),
        "Insufficient samples should produce no detections"
    );
}

/// Test 5: Unbounded activation functions should be excluded.
#[test]
fn test_unbounded_activation_excluded() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY", 0.0),
            neuron("output-1", "output", "IDENTITY", 0.0),
        ],
        vec![synapse("input-1", "output-1", 0.3)],
    );

    // IDENTITY has no bounded range — cannot be "compressed"
    let records: Vec<DiscoverRecord> = (0..50)
        .map(|i| {
            let activation = 0.3 + (i as f32 / 49.0) * 0.1;
            record("output-1", i, activation, vec![0.1])
        })
        .collect();

    let neuron_records = vec![("output-1".to_string(), records)];

    let config = OutputRangeCompressionConfig::default();
    let detected = detect_output_range_compression(&creature, &neuron_records, &config);

    assert!(
        detected.is_empty(),
        "Unbounded activation functions should not be flagged"
    );
}

/// Test 6: Produces valid coordinated candidates (setBias+setWeight or changeSquash).
#[test]
fn test_output_range_compression_produces_valid_coordinated_candidates() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY", 0.0),
            neuron("hidden-1", "hidden", "TANH", 0.0),
            neuron("output-1", "output", "TANH", 0.5),
        ],
        vec![
            synapse("input-1", "hidden-1", 0.5),
            synapse("hidden-1", "output-1", 0.3),
        ],
    );

    let detected = vec![OutputRangeCompressionNeuron {
        neuron_uuid: "output-1".to_string(),
        squash: "TANH".to_string(),
        bias: 0.5,
        activation_min: 0.3,
        activation_max: 0.7,
        theoretical_range: 2.0,
        range_utilisation: 0.20,
        sample_count: 50,
    }];

    let candidates = output_range_compression_to_coordinated_candidates(&detected, &creature);

    assert!(
        !candidates.is_empty(),
        "Should produce at least one coordinated candidate"
    );

    for c in &candidates {
        assert!(
            c.expected_creature_score_gain > 0.0,
            "Expected improvement should be positive"
        );
        assert!(c.comment.is_some(), "Should have a descriptive comment");
    }

    // Check we get both changeSquash and setBias+setWeight type candidates
    let ops_json = serde_json::to_string(&candidates).unwrap();
    let has_change_squash = ops_json.contains("changeSquash");
    let has_set_bias = ops_json.contains("setBias");
    let has_set_weight = ops_json.contains("setWeight");
    assert!(has_change_squash, "Should include changeSquash candidate");
    assert!(
        has_set_bias && has_set_weight,
        "Should include coordinated setBias+setWeight candidate"
    );
}

/// Test 7: LOGISTIC output neuron compressed to [0.6, 0.8] — only 20% of [0, 1].
#[test]
fn test_logistic_output_compressed() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY", 0.0),
            neuron("output-1", "output", "LOGISTIC", 1.0),
        ],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    // LOGISTIC range is [0, 1]; activations in [0.6, 0.8] = 20% utilisation
    let records: Vec<DiscoverRecord> = (0..50)
        .map(|i| {
            let activation = 0.6 + (i as f32 / 49.0) * 0.2;
            record("output-1", i, activation, vec![0.1])
        })
        .collect();

    let neuron_records = vec![("output-1".to_string(), records)];

    let config = OutputRangeCompressionConfig::default();
    let detected = detect_output_range_compression(&creature, &neuron_records, &config);

    assert!(
        !detected.is_empty(),
        "LOGISTIC output compressed to 20% of range should be flagged"
    );
    assert_eq!(detected[0].squash, "LOGISTIC");
}

/// Test 8: Multiple output neurons — only compressed ones flagged.
#[test]
fn test_multiple_outputs_only_compressed_flagged() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY", 0.0),
            neuron("output-good", "output", "TANH", 0.0),
            neuron("output-bad", "output", "TANH", 0.5),
        ],
        vec![
            synapse("input-1", "output-good", 1.0),
            synapse("input-1", "output-bad", 0.2),
        ],
    );

    // output-good: uses 80% of range [-0.8, 0.8]
    let good_records: Vec<DiscoverRecord> = (0..50)
        .map(|i| {
            let activation = -0.8 + (i as f32 / 49.0) * 1.6;
            record("output-good", i, activation, vec![0.05])
        })
        .collect();

    // output-bad: uses only 15% of range [0.2, 0.5]
    let bad_records: Vec<DiscoverRecord> = (0..50)
        .map(|i| {
            let activation = 0.2 + (i as f32 / 49.0) * 0.3;
            record("output-bad", i, activation, vec![0.2])
        })
        .collect();

    let neuron_records = vec![
        ("output-good".to_string(), good_records),
        ("output-bad".to_string(), bad_records),
    ];

    let config = OutputRangeCompressionConfig::default();
    let detected = detect_output_range_compression(&creature, &neuron_records, &config);

    let flagged_uuids: Vec<&str> = detected.iter().map(|d| d.neuron_uuid.as_str()).collect();
    assert!(
        flagged_uuids.contains(&"output-bad"),
        "output-bad should be flagged, got: {flagged_uuids:?}"
    );
    assert!(
        !flagged_uuids.contains(&"output-good"),
        "output-good should NOT be flagged, got: {flagged_uuids:?}"
    );
}

/// Test 9: Near-saturated output neuron should be excluded.
#[test]
fn test_near_saturated_excluded() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY", 0.0),
            neuron("output-1", "output", "TANH", 0.0),
        ],
        vec![synapse("input-1", "output-1", 1.0)],
    );

    // Activations near upper bound [0.92, 0.98] — this is saturation, not compression
    let records: Vec<DiscoverRecord> = (0..50)
        .map(|i| {
            let activation = 0.92 + (i as f32 / 49.0) * 0.06;
            record("output-1", i, activation, vec![0.1])
        })
        .collect();

    let neuron_records = vec![("output-1".to_string(), records)];

    let config = OutputRangeCompressionConfig::default();
    let detected = detect_output_range_compression(&creature, &neuron_records, &config);

    assert!(
        detected.is_empty(),
        "Near-saturated output should not be flagged as compressed"
    );
}

/// Test 10: Dead output neuron (near-zero range) should be excluded.
#[test]
fn test_dead_output_excluded() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY", 0.0),
            neuron("output-1", "output", "TANH", 0.0),
        ],
        vec![synapse("input-1", "output-1", 1.0)],
    );

    // All activations essentially the same value (dead neuron)
    let records: Vec<DiscoverRecord> = (0..50)
        .map(|i| {
            let activation = 0.5 + (i as f32 / 49.0) * 0.005;
            record("output-1", i, activation, vec![0.1])
        })
        .collect();

    let neuron_records = vec![("output-1".to_string(), records)];

    let config = OutputRangeCompressionConfig::default();
    let detected = detect_output_range_compression(&creature, &neuron_records, &config);

    assert!(
        detected.is_empty(),
        "Dead output neuron (near-zero range) should not be flagged"
    );
}

/// Test 11: Empty records produce no detections.
#[test]
fn test_output_range_compression_empty_records_no_detections() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY", 0.0),
            neuron("output-1", "output", "TANH", 0.0),
        ],
        vec![synapse("input-1", "output-1", 1.0)],
    );

    let config = OutputRangeCompressionConfig::default();
    let detected = detect_output_range_compression(&creature, &[], &config);

    assert!(
        detected.is_empty(),
        "Empty records should produce no detections"
    );
}

/// Test 12: Results sorted by range utilisation (worst first).
#[test]
fn test_results_sorted_by_utilisation() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY", 0.0),
            neuron("output-a", "output", "TANH", 0.0),
            neuron("output-b", "output", "TANH", 0.0),
        ],
        vec![
            synapse("input-1", "output-a", 0.5),
            synapse("input-1", "output-b", 0.5),
        ],
    );

    // output-a: 30% utilisation [0.1, 0.7]
    let records_a: Vec<DiscoverRecord> = (0..50)
        .map(|i| {
            let activation = 0.1 + (i as f32 / 49.0) * 0.6;
            record("output-a", i, activation, vec![0.1])
        })
        .collect();

    // output-b: 15% utilisation [0.3, 0.6]
    let records_b: Vec<DiscoverRecord> = (0..50)
        .map(|i| {
            let activation = 0.3 + (i as f32 / 49.0) * 0.3;
            record("output-b", i, activation, vec![0.1])
        })
        .collect();

    let neuron_records = vec![
        ("output-a".to_string(), records_a),
        ("output-b".to_string(), records_b),
    ];

    let config = OutputRangeCompressionConfig::default();
    let detected = detect_output_range_compression(&creature, &neuron_records, &config);

    assert!(detected.len() >= 2, "Should detect both compressed outputs");

    // Worst utilisation (lowest) should be first
    assert!(
        detected[0].range_utilisation <= detected[1].range_utilisation,
        "Results should be sorted by utilisation (lowest first), got {:.2} then {:.2}",
        detected[0].range_utilisation,
        detected[1].range_utilisation
    );
}
