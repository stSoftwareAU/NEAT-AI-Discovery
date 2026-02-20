//! Tests for Issue #356: Detect output neuron bias drift for bias adjustment candidates.
//!
//! Output neurons with consistent error sign bias (predominantly positive or negative
//! errors) indicate a systematic prediction offset that can be corrected by adjusting
//! the neuron's bias parameter.
//!
//! ## TDD Plan
//! 1. Create test network with intentionally biased output neurons
//! 2. Verify detection identifies them with correct statistics
//! 3. Verify balanced output neurons are not flagged
//! 4. Test edge cases: insufficient samples, small errors
//! 5. Test coordinated structural candidate conversion

use neat_ai_discovery::analysis::recommendation::output_bias_drift::{
    detect_output_bias_drift, output_bias_drift_to_coordinated_candidates,
};
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

/// Helper: create a DiscoverRecord.
fn record(neuron_uuid: &str, obs_index: u32, activation: f32, errors: Vec<f32>) -> DiscoverRecord {
    DiscoverRecord {
        obs_index,
        neuron_uuid: neuron_uuid.to_string(),
        value: Some(activation * 0.8),
        activation,
        errors,
    }
}

/// Helper: build a minimal creature.
fn make_creature(neurons: Vec<NeuronJson>, synapses: Vec<SynapseJson>) -> CreatureJson {
    CreatureJson {
        neurons,
        synapses,
        input: 1,
        output: 1,
    }
}

/// Helper: build a NeuronJson.
fn neuron(uuid: &str, neuron_type: &str, bias: f32) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
        squash: "IDENTITY".to_string(),
        bias,
    }
}

/// Helper: build a SynapseJson.
fn synapse(from: &str, to: &str, weight: f32) -> SynapseJson {
    SynapseJson {
        from_uuid: from.to_string(),
        to_uuid: to.to_string(),
        weight,
        synapse_type: None,
    }
}

/// Test 1: Output neuron with predominantly positive errors is detected.
#[test]
fn test_detects_positive_bias_drift() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", 0.0),
            neuron("output-1", "output", 0.0),
        ],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    // 80% positive errors → output predicting too low
    let output_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let error = if i < 80 { 0.3 } else { -0.1 };
            record("output-1", i, 0.5, vec![error])
        })
        .collect();

    let candidates =
        detect_output_bias_drift(&creature, &[("output-1".to_string(), output_records)]);

    assert_eq!(candidates.len(), 1, "Should detect one biased output");
    let c = &candidates[0];
    assert_eq!(c.neuron_uuid, "output-1");
    assert!(c.mean_error > 0.0, "Mean error should be positive");
    assert!(
        c.positive_error_fraction > 0.7,
        "Positive error fraction should be > 0.7"
    );
    assert!(
        c.recommended_bias_delta < 0.0,
        "Should recommend negative bias adjustment for positive errors"
    );
}

/// Test 2: Output neuron with predominantly negative errors is detected.
#[test]
fn test_detects_negative_bias_drift() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", 0.0),
            neuron("output-1", "output", 0.5),
        ],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    // 85% negative errors → output predicting too high
    let output_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let error = if i < 85 { -0.4 } else { 0.1 };
            record("output-1", i, 0.5, vec![error])
        })
        .collect();

    let candidates =
        detect_output_bias_drift(&creature, &[("output-1".to_string(), output_records)]);

    assert_eq!(candidates.len(), 1, "Should detect one biased output");
    let c = &candidates[0];
    assert!(c.mean_error < 0.0, "Mean error should be negative");
    assert!(
        c.recommended_bias_delta > 0.0,
        "Should recommend positive bias adjustment for negative errors"
    );
}

/// Test 3: Output neuron with balanced errors is NOT flagged.
#[test]
fn test_balanced_errors_not_flagged() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", 0.0),
            neuron("output-1", "output", 0.0),
        ],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    // Roughly 50/50 positive and negative errors
    let output_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let error = if i % 2 == 0 { 0.2 } else { -0.2 };
            record("output-1", i, 0.5, vec![error])
        })
        .collect();

    let candidates =
        detect_output_bias_drift(&creature, &[("output-1".to_string(), output_records)]);

    assert!(
        candidates.is_empty(),
        "Balanced errors should not be flagged as bias drift"
    );
}

/// Test 4: Hidden neuron is NOT flagged (only outputs).
#[test]
fn test_hidden_neuron_not_flagged() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", 0.0),
            neuron("hidden-1", "hidden", 0.0),
            neuron("output-1", "output", 0.0),
        ],
        vec![
            synapse("input-1", "hidden-1", 0.5),
            synapse("hidden-1", "output-1", 0.3),
        ],
    );

    // Hidden neuron with biased errors — should NOT be detected
    let hidden_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| record("hidden-1", i, 0.5, vec![0.5]))
        .collect();

    let candidates =
        detect_output_bias_drift(&creature, &[("hidden-1".to_string(), hidden_records)]);

    assert!(
        candidates.is_empty(),
        "Hidden neurons should not be flagged"
    );
}

/// Test 5: Insufficient samples should not trigger detection.
#[test]
fn test_insufficient_samples_not_flagged() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", 0.0),
            neuron("output-1", "output", 0.0),
        ],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    let output_records: Vec<DiscoverRecord> = (0..5)
        .map(|i| record("output-1", i, 0.5, vec![0.5]))
        .collect();

    let candidates =
        detect_output_bias_drift(&creature, &[("output-1".to_string(), output_records)]);

    assert!(
        candidates.is_empty(),
        "Too few samples should not trigger detection"
    );
}

/// Test 6: Very small errors (noise) are NOT flagged.
#[test]
fn test_noise_level_errors_not_flagged() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", 0.0),
            neuron("output-1", "output", 0.0),
        ],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    // All positive but tiny errors (noise level)
    let output_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| record("output-1", i, 0.5, vec![0.001]))
        .collect();

    let candidates =
        detect_output_bias_drift(&creature, &[("output-1".to_string(), output_records)]);

    assert!(
        candidates.is_empty(),
        "Noise-level errors should not be flagged"
    );
}

/// Test 7: Bias drift candidates produce correct coordinated SetBias operations.
#[test]
fn test_candidates_produce_coordinated_set_bias_operations() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", 0.0),
            neuron("output-1", "output", 0.0),
        ],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    let output_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let error = if i < 80 { 0.3 } else { -0.1 };
            record("output-1", i, 0.5, vec![error])
        })
        .collect();

    let candidates =
        detect_output_bias_drift(&creature, &[("output-1".to_string(), output_records)]);
    assert!(!candidates.is_empty(), "Should detect bias drift");

    let coordinated = output_bias_drift_to_coordinated_candidates(&candidates);

    assert_eq!(
        coordinated.len(),
        1,
        "Should produce one coordinated candidate"
    );
    let c = &coordinated[0];
    assert!(
        c.expected_creature_score_gain > 0.0,
        "Expected improvement should be positive"
    );
    assert!(c.comment.is_some(), "Should have a comment");

    // Check that operations include a SetBias
    let ops_json = serde_json::to_string(&c.operations).unwrap();
    assert!(
        ops_json.contains("setBias"),
        "Should include setBias operation, got: {ops_json}"
    );
}

/// Test 8: Multiple output neurons — only biased ones are detected.
#[test]
fn test_multiple_outputs_only_biased_detected() {
    let creature = CreatureJson {
        neurons: vec![
            neuron("input-1", "input", 0.0),
            neuron("output-1", "output", 0.0),
            neuron("output-2", "output", 0.0),
        ],
        synapses: vec![
            synapse("input-1", "output-1", 0.5),
            synapse("input-1", "output-2", 0.3),
        ],
        input: 1,
        output: 2,
    };

    // output-1: biased (80% positive)
    let output_1_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let error = if i < 80 { 0.3 } else { -0.1 };
            record("output-1", i, 0.5, vec![error])
        })
        .collect();

    // output-2: balanced
    let output_2_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let error = if i % 2 == 0 { 0.2 } else { -0.2 };
            record("output-2", i, 0.5, vec![error])
        })
        .collect();

    let candidates = detect_output_bias_drift(
        &creature,
        &[
            ("output-1".to_string(), output_1_records),
            ("output-2".to_string(), output_2_records),
        ],
    );

    assert_eq!(candidates.len(), 1, "Should detect only the biased output");
    assert_eq!(candidates[0].neuron_uuid, "output-1");
}

/// Test 9: Current bias is correctly recorded.
#[test]
fn test_current_bias_recorded() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", 0.0),
            neuron("output-1", "output", 0.3),
        ],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    let output_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let error = if i < 80 { 0.3 } else { -0.1 };
            record("output-1", i, 0.5, vec![error])
        })
        .collect();

    let candidates =
        detect_output_bias_drift(&creature, &[("output-1".to_string(), output_records)]);

    assert_eq!(candidates.len(), 1);
    assert!(
        (candidates[0].current_bias - 0.3).abs() < 1e-6,
        "Current bias should be recorded correctly"
    );
}
