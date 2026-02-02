//! Unit tests for dead neuron detection (Issue #376).

use super::*;
use crate::{CreatureJson, NeuronJson, SynapseJson};

/// Helper: build a NeuronJson.
fn neuron(uuid: &str, neuron_type: &str, squash: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
        squash: squash.to_string(),
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

/// Helper: create records with near-zero activations (dead neuron).
fn dead_records(uuid: &str, count: usize) -> Vec<DiscoverRecord> {
    (0..count)
        .map(|i| DiscoverRecord {
            obs_index: i as u32,
            neuron_uuid: uuid.to_string(),
            value: None,
            activation: 0.0,
            errors: vec![0.01],
        })
        .collect()
}

/// Helper: create records with non-zero activations (alive neuron).
fn alive_records(uuid: &str, count: usize, activation: f32) -> Vec<DiscoverRecord> {
    (0..count)
        .map(|i| DiscoverRecord {
            obs_index: i as u32,
            neuron_uuid: uuid.to_string(),
            value: None,
            activation,
            errors: vec![0.01],
        })
        .collect()
}

fn make_creature_with_hidden(hidden_uuid: &str) -> CreatureJson {
    CreatureJson {
        neurons: vec![
            neuron("i0", "input", "IDENTITY"),
            neuron(hidden_uuid, "hidden", "TANH"),
            neuron("o1", "output", "IDENTITY"),
        ],
        synapses: vec![
            synapse("i0", hidden_uuid, 1.0),
            synapse(hidden_uuid, "o1", 1.0),
        ],
        input: 1,
        output: 1,
    }
}

// ── Detection criteria ──────────────────────────────────────────────────────

#[test]
fn neuron_with_zero_activation_across_all_samples_is_detected() {
    let creature = make_creature_with_hidden("h1");
    let records = vec![("h1".to_string(), dead_records("h1", 30))];

    let detected = detect_dead_neurons(&creature, &records);

    assert!(!detected.is_empty(), "expected dead neuron to be detected");
    assert_eq!(detected[0].neuron_uuid, "h1");
    assert!(detected[0].removal_confidence >= 0.5);
    assert!(detected[0].estimated_improvement > 0.0);
}

// ── Exclusion criteria ──────────────────────────────────────────────────────

#[test]
fn neuron_with_meaningful_activation_is_not_detected() {
    let creature = make_creature_with_hidden("h1");
    let records = vec![("h1".to_string(), alive_records("h1", 30, 0.5))];

    let detected = detect_dead_neurons(&creature, &records);

    assert!(
        detected.is_empty(),
        "neuron with activation 0.5 should not be detected as dead"
    );
}

#[test]
fn output_neuron_is_not_detected_as_dead() {
    let creature = CreatureJson {
        neurons: vec![
            neuron("i0", "input", "IDENTITY"),
            neuron("o1", "output", "IDENTITY"),
        ],
        synapses: vec![synapse("i0", "o1", 1.0)],
        input: 1,
        output: 1,
    };

    // Output neuron with zero activation — should still not be flagged
    let records = vec![("o1".to_string(), dead_records("o1", 30))];

    let detected = detect_dead_neurons(&creature, &records);

    assert!(
        detected.is_empty(),
        "output neurons should never be flagged as dead"
    );
}

#[test]
fn neuron_active_on_small_fraction_is_not_detected() {
    let creature = make_creature_with_hidden("h1");
    // 29 records at 0.0 and 1 record at 0.5 — active_fraction = 1/30 > MIN_ACTIVE_FRACTION
    let mut records = dead_records("h1", 29);
    records.push(DiscoverRecord {
        obs_index: 29,
        neuron_uuid: "h1".to_string(),
        value: None,
        activation: 0.5,
        errors: vec![0.01],
    });

    let neuron_records = vec![("h1".to_string(), records)];
    let detected = detect_dead_neurons(&creature, &neuron_records);

    assert!(
        detected.is_empty(),
        "neuron active on >=1% of samples should not be flagged as dead"
    );
}

// ── Edge cases ──────────────────────────────────────────────────────────────

#[test]
fn insufficient_samples_prevents_dead_detection() {
    let creature = make_creature_with_hidden("h1");
    let records = vec![("h1".to_string(), dead_records("h1", 5))];

    let detected = detect_dead_neurons(&creature, &records);

    assert!(
        detected.is_empty(),
        "fewer than 20 samples should prevent detection"
    );
}

#[test]
fn dead_neuron_connected_outputs_are_identified() {
    let creature = CreatureJson {
        neurons: vec![
            neuron("i0", "input", "IDENTITY"),
            neuron("h1", "hidden", "TANH"),
            neuron("h2", "hidden", "TANH"),
            neuron("o1", "output", "IDENTITY"),
        ],
        synapses: vec![
            synapse("i0", "h1", 1.0),
            synapse("h1", "h2", 1.0),
            synapse("h2", "o1", 1.0),
        ],
        input: 1,
        output: 1,
    };

    let records = vec![("h1".to_string(), dead_records("h1", 30))];
    let detected = detect_dead_neurons(&creature, &records);

    assert!(!detected.is_empty());
    assert!(
        detected[0].connected_outputs.contains(&"o1".to_string()),
        "should identify reachable output neurons"
    );
}

// ── Conversion ──────────────────────────────────────────────────────────────

#[test]
fn conversion_produces_remove_neuron_operation() {
    let candidates = vec![DeadNeuronCandidate {
        neuron_uuid: "h1".to_string(),
        mean_abs_activation: 0.0,
        activation_std_dev: 0.0,
        sample_count: 100,
        connected_outputs: vec!["o1".to_string()],
        removal_confidence: 0.95,
        estimated_improvement: 0.00095,
    }];

    let coordinated = dead_neurons_to_coordinated_candidates(&candidates);

    assert_eq!(coordinated.len(), 1);
    let first = &coordinated[0];
    let has_remove = first.operations.iter().any(|op| {
        matches!(op, CoordinatedStructuralOpJson::RemoveNeuron { neuron_uuid } if neuron_uuid == "h1")
    });
    assert!(has_remove, "expected RemoveNeuron operation for h1");
    assert!(first.expected_creature_score_gain > 0.0);
}
