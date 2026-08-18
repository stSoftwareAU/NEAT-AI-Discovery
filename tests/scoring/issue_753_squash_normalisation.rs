//! Tests for Issue #753: Squash string normalisation at load time.
//!
//! Verifies that `NeuronJson` squash strings are normalised to ASCII uppercase
//! during deserialisation, so detection modules never see mixed-case names.

use neat_ai_discovery::{CreatureJson, NeuronJson};

/// Squash names should be normalised to uppercase during deserialisation.
#[test]
fn squash_normalised_to_uppercase_on_deserialise() {
    let json = r#"{
        "neurons": [
            {"uuid": "n1", "type": "hidden", "squash": "Tanh", "bias": 0.0},
            {"uuid": "n2", "type": "hidden", "squash": "logistic", "bias": 0.1},
            {"uuid": "n3", "type": "hidden", "squash": "Hard_Tanh", "bias": 0.2},
            {"uuid": "n4", "type": "hidden", "squash": "RELU", "bias": 0.0},
            {"uuid": "n5", "type": "hidden", "squash": "softPlus", "bias": 0.0}
        ],
        "synapses": [],
        "input": 1,
        "output": 1
    }"#;

    let creature: CreatureJson = serde_json::from_str(json).unwrap();

    assert_eq!(creature.neurons[0].squash, "TANH");
    assert_eq!(creature.neurons[1].squash, "LOGISTIC");
    assert_eq!(creature.neurons[2].squash, "HARD_TANH");
    assert_eq!(creature.neurons[3].squash, "RELU");
    assert_eq!(creature.neurons[4].squash, "SOFTPLUS");
}

/// Default squash should remain "IDENTITY" (already uppercase).
#[test]
fn default_squash_is_uppercase() {
    let json = r#"{
        "neurons": [
            {"uuid": "n1", "type": "constant"}
        ],
        "synapses": [],
        "input": 1,
        "output": 1
    }"#;

    let creature: CreatureJson = serde_json::from_str(json).unwrap();
    assert_eq!(creature.neurons[0].squash, "IDENTITY");
}

/// Whitespace-only squash should normalise to empty string.
#[test]
fn whitespace_squash_normalises_to_empty() {
    let json = r#"{
        "neurons": [
            {"uuid": "n1", "type": "hidden", "squash": "  ", "bias": 0.0}
        ],
        "synapses": [],
        "input": 1,
        "output": 1
    }"#;

    let creature: CreatureJson = serde_json::from_str(json).unwrap();
    assert_eq!(creature.neurons[0].squash, "");
}

/// Programmatically constructed `NeuronJson` retains its squash value unchanged
/// (normalisation only applies during serde deserialisation).
#[test]
fn programmatic_construction_preserves_squash() {
    let neuron = NeuronJson {
        uuid: "test".to_string(),
        neuron_type: "hidden".to_string(),
        squash: "TANH".to_string(),
        bias: 0.0,
    };
    assert_eq!(neuron.squash, "TANH");
}

/// Detection modules should work correctly with pre-normalised squash strings.
/// Exercises saturation detection with uppercase squash names.
#[test]
fn saturation_detection_works_with_prenormalised_squash() {
    use neat_ai_discovery::analysis::detection::saturation::detect_saturated_neurons;
    use neat_ai_discovery::types::DiscoverRecord;

    let neurons = vec![("sat-neuron".to_string(), "TANH".to_string(), 0.0)];

    // Create records that show saturation (all activations near 1.0)
    let records: Vec<DiscoverRecord> = (0..50)
        .map(|i| DiscoverRecord::new(i, "sat-neuron".to_string(), Some(5.0), 0.999, vec![0.01]))
        .collect();

    let neuron_records = vec![("sat-neuron".to_string(), records)];

    let candidates = detect_saturated_neurons(&neurons, &neuron_records);
    assert!(
        !candidates.is_empty(),
        "Should detect saturated TANH neuron with pre-normalised squash"
    );
    assert_eq!(candidates[0].current_squash, "TANH");
}
