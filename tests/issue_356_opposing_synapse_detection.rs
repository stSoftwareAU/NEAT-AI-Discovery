//! Tests for Issue #356: Detect opposing synapses for removal or weight flip candidates.
//!
//! Opposing synapses have contributions that correlate positively with target error,
//! meaning they actively make predictions worse. This module detects them and recommends
//! removal or weight sign reversal.
//!
//! ## TDD Plan
//! 1. Create test network with intentionally opposing synapses
//! 2. Verify detection identifies them with correct statistics
//! 3. Verify helpful synapses are not flagged
//! 4. Test coordinated structural candidate conversion (removal vs weight flip)

use neat_ai_discovery::analysis::opposing_synapse::{
    detect_opposing_synapses, opposing_synapses_to_coordinated_candidates,
};
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

/// Helper: create a DiscoverRecord for a neuron.
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
        input: 2,
        output: 1,
    }
}

/// Helper: build a NeuronJson.
fn neuron(uuid: &str, neuron_type: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
        squash: "IDENTITY".to_string(),
        bias: 0.0,
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

/// Test 1: Synapse with strong positive contribution–error correlation is detected.
#[test]
fn test_detects_opposing_synapse() {
    let creature = make_creature(
        vec![neuron("input-1", "input"), neuron("output-1", "output")],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    // When source activation is high (positive contribution), error is also high
    // → positive correlation between contribution and error → opposing synapse
    let input_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = (i as f32 - 50.0) / 50.0; // Range [-1, 1]
            record("input-1", i, activation, vec![])
        })
        .collect();

    let output_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = (i as f32 - 50.0) / 50.0;
            let contribution = 0.5 * activation; // weight × activation
            let error = contribution * 0.8 + 0.1; // Strongly correlated with contribution
            record("output-1", i, 0.0, vec![error])
        })
        .collect();

    let candidates = detect_opposing_synapses(
        &creature,
        &[
            ("input-1".to_string(), input_records),
            ("output-1".to_string(), output_records),
        ],
    );

    assert_eq!(candidates.len(), 1, "Should detect one opposing synapse");
    let c = &candidates[0];
    assert_eq!(c.from_neuron_uuid, "input-1");
    assert_eq!(c.to_neuron_uuid, "output-1");
    assert!(
        c.contribution_error_correlation > 0.3,
        "Correlation should be positive: got {}",
        c.contribution_error_correlation
    );
}

/// Test 2: Helpful synapse (negative contribution–error correlation) is NOT flagged.
#[test]
fn test_helpful_synapse_not_flagged() {
    let creature = make_creature(
        vec![neuron("input-1", "input"), neuron("output-1", "output")],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    // When source activation is high, error is LOW → negative correlation → helpful
    let input_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = (i as f32 - 50.0) / 50.0;
            record("input-1", i, activation, vec![])
        })
        .collect();

    let output_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = (i as f32 - 50.0) / 50.0;
            let contribution = 0.5 * activation;
            let error = -contribution * 0.8; // Negatively correlated with contribution → helpful
            record("output-1", i, 0.0, vec![error])
        })
        .collect();

    let candidates = detect_opposing_synapses(
        &creature,
        &[
            ("input-1".to_string(), input_records),
            ("output-1".to_string(), output_records),
        ],
    );

    assert!(
        candidates.is_empty(),
        "Helpful synapse should not be flagged as opposing"
    );
}

/// Test 3: Hidden-to-hidden synapses are not analysed (only output targets).
#[test]
fn test_hidden_target_synapses_not_analysed() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input"),
            neuron("hidden-1", "hidden"),
            neuron("output-1", "output"),
        ],
        vec![
            synapse("input-1", "hidden-1", 0.5), // Hidden target — skipped
            synapse("hidden-1", "output-1", 0.3),
        ],
    );

    let input_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = (i as f32 - 50.0) / 50.0;
            record("input-1", i, activation, vec![])
        })
        .collect();

    let hidden_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = (i as f32 - 50.0) / 50.0;
            let error = activation * 0.5; // Would be opposing if analysed
            record("hidden-1", i, activation, vec![error])
        })
        .collect();

    let output_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| record("output-1", i, 0.0, vec![0.01]))
        .collect();

    let candidates = detect_opposing_synapses(
        &creature,
        &[
            ("input-1".to_string(), input_records),
            ("hidden-1".to_string(), hidden_records),
            ("output-1".to_string(), output_records),
        ],
    );

    // The input-1 → hidden-1 synapse is not analysed because hidden-1 is not output
    // The hidden-1 → output-1 synapse may or may not be detected depending on correlation
    // But we should NOT detect input-1 → hidden-1 as opposing
    let non_output_targets: Vec<_> = candidates
        .iter()
        .filter(|c| c.to_neuron_uuid == "hidden-1")
        .collect();
    assert!(
        non_output_targets.is_empty(),
        "Should not analyse synapses targeting hidden neurons"
    );
}

/// Test 4: Insufficient samples should not trigger detection.
#[test]
fn test_insufficient_samples_not_flagged() {
    let creature = make_creature(
        vec![neuron("input-1", "input"), neuron("output-1", "output")],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    let input_records: Vec<DiscoverRecord> =
        (0..5).map(|i| record("input-1", i, 0.5, vec![])).collect();

    let output_records: Vec<DiscoverRecord> = (0..5)
        .map(|i| record("output-1", i, 0.0, vec![0.5]))
        .collect();

    let candidates = detect_opposing_synapses(
        &creature,
        &[
            ("input-1".to_string(), input_records),
            ("output-1".to_string(), output_records),
        ],
    );

    assert!(
        candidates.is_empty(),
        "Too few samples should not trigger detection"
    );
}

/// Test 5: Strongly opposing synapse recommends removal.
#[test]
fn test_strongly_opposing_recommends_removal() {
    let creature = make_creature(
        vec![neuron("input-1", "input"), neuron("output-1", "output")],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    // Very strong positive correlation (contribution mirrors error)
    let input_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = (i as f32 - 50.0) / 50.0;
            record("input-1", i, activation, vec![])
        })
        .collect();

    let output_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = (i as f32 - 50.0) / 50.0;
            let error = 0.5 * activation + 0.05; // Strong positive correlation
            record("output-1", i, 0.0, vec![error])
        })
        .collect();

    let candidates = detect_opposing_synapses(
        &creature,
        &[
            ("input-1".to_string(), input_records),
            ("output-1".to_string(), output_records),
        ],
    );

    assert!(!candidates.is_empty(), "Should detect opposing synapse");

    let coordinated = opposing_synapses_to_coordinated_candidates(&candidates);
    assert!(
        !coordinated.is_empty(),
        "Should produce coordinated candidates"
    );

    let c = &coordinated[0];
    assert!(c.expected_creature_score_gain > 0.0);

    let ops_json = serde_json::to_string(&c.operations).unwrap();
    // Should contain either removeSynapse or setWeight
    assert!(
        ops_json.contains("removeSynapse") || ops_json.contains("setWeight"),
        "Should include removeSynapse or setWeight operation, got: {ops_json}"
    );
}
