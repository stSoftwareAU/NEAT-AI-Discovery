//! Unit tests for bottleneck neuron detection (Issue #376).

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

/// Helper: create records with non-zero errors for a neuron.
fn records_with_errors(uuid: &str, count: usize, error: f32) -> Vec<DiscoverRecord> {
    (0..count)
        .map(|i| DiscoverRecord {
            obs_index: i as u32,
            neuron_uuid: uuid.to_string(),
            value: None,
            activation: 0.5,
            errors: vec![error],
        })
        .collect()
}

// ── Detection criteria ──────────────────────────────────────────────────────

#[test]
fn hidden_neuron_with_high_fan_in_low_fan_out_is_detected() {
    // Bottleneck: fan-in=4, fan-out=1, ratio=4.0
    let creature = CreatureJson {
        neurons: vec![
            neuron("i0", "input", "IDENTITY"),
            neuron("i1", "input", "IDENTITY"),
            neuron("i2", "input", "IDENTITY"),
            neuron("i3", "input", "IDENTITY"),
            neuron("h1", "hidden", "TANH"),
            neuron("o1", "output", "IDENTITY"),
        ],
        synapses: vec![
            synapse("i0", "h1", 1.0),
            synapse("i1", "h1", 0.5),
            synapse("i2", "h1", 0.3),
            synapse("i3", "h1", 0.2),
            synapse("h1", "o1", 1.0),
        ],
        input: 4,
        output: 1,
    };

    let records = vec![("h1".to_string(), records_with_errors("h1", 30, 0.1))];

    let detected = detect_bottleneck_neurons(&creature, &records);

    assert!(
        !detected.is_empty(),
        "expected bottleneck detection for fan-in=4, fan-out=1"
    );
    assert_eq!(detected[0].neuron_uuid, "h1");
    assert_eq!(detected[0].fan_in, 4);
    assert_eq!(detected[0].fan_out, 1);
    assert!(detected[0].estimated_improvement > 0.0);
}

// ── Exclusion criteria ──────────────────────────────────────────────────────

#[test]
fn neuron_with_balanced_fan_in_fan_out_is_not_detected() {
    // fan-in=2, fan-out=2, ratio=1.0 — below MIN_FAN_IN_FOR_BOTTLENECK (3)
    let creature = CreatureJson {
        neurons: vec![
            neuron("i0", "input", "IDENTITY"),
            neuron("i1", "input", "IDENTITY"),
            neuron("h1", "hidden", "TANH"),
            neuron("o1", "output", "IDENTITY"),
            neuron("o2", "output", "IDENTITY"),
        ],
        synapses: vec![
            synapse("i0", "h1", 1.0),
            synapse("i1", "h1", 0.5),
            synapse("h1", "o1", 1.0),
            synapse("h1", "o2", 0.5),
        ],
        input: 2,
        output: 2,
    };

    let records = vec![("h1".to_string(), records_with_errors("h1", 30, 0.1))];

    let detected = detect_bottleneck_neurons(&creature, &records);

    assert!(
        detected.is_empty(),
        "balanced fan-in/fan-out should not be detected"
    );
}

#[test]
fn output_neuron_is_not_detected_as_bottleneck() {
    // Output neurons have high fan-in naturally — should be excluded
    let creature = CreatureJson {
        neurons: vec![
            neuron("i0", "input", "IDENTITY"),
            neuron("i1", "input", "IDENTITY"),
            neuron("i2", "input", "IDENTITY"),
            neuron("o1", "output", "IDENTITY"),
        ],
        synapses: vec![
            synapse("i0", "o1", 1.0),
            synapse("i1", "o1", 0.5),
            synapse("i2", "o1", 0.3),
        ],
        input: 3,
        output: 1,
    };

    let records = vec![("o1".to_string(), records_with_errors("o1", 30, 0.1))];

    let detected = detect_bottleneck_neurons(&creature, &records);

    assert!(
        detected.is_empty(),
        "output neurons should not be detected as bottlenecks"
    );
}

// ── Edge cases ──────────────────────────────────────────────────────────────

#[test]
fn insufficient_samples_prevents_bottleneck_detection() {
    let creature = CreatureJson {
        neurons: vec![
            neuron("i0", "input", "IDENTITY"),
            neuron("i1", "input", "IDENTITY"),
            neuron("i2", "input", "IDENTITY"),
            neuron("h1", "hidden", "TANH"),
            neuron("o1", "output", "IDENTITY"),
        ],
        synapses: vec![
            synapse("i0", "h1", 1.0),
            synapse("i1", "h1", 0.5),
            synapse("i2", "h1", 0.3),
            synapse("h1", "o1", 1.0),
        ],
        input: 3,
        output: 1,
    };

    // Only 5 samples — below MIN_SAMPLES_FOR_BOTTLENECK (20)
    let records = vec![("h1".to_string(), records_with_errors("h1", 5, 0.1))];

    let detected = detect_bottleneck_neurons(&creature, &records);

    assert!(
        detected.is_empty(),
        "fewer than 20 samples should prevent detection"
    );
}

// ── Conversion ──────────────────────────────────────────────────────────────

#[test]
fn conversion_produces_add_neuron_and_add_synapse_operations() {
    let creature = CreatureJson {
        neurons: vec![
            neuron("i0", "input", "IDENTITY"),
            neuron("i1", "input", "IDENTITY"),
            neuron("i2", "input", "IDENTITY"),
            neuron("i3", "input", "IDENTITY"),
            neuron("h1", "hidden", "TANH"),
            neuron("o1", "output", "IDENTITY"),
        ],
        synapses: vec![
            synapse("i0", "h1", 1.0),
            synapse("i1", "h1", 0.5),
            synapse("i2", "h1", 0.3),
            synapse("i3", "h1", 0.2),
            synapse("h1", "o1", 1.0),
        ],
        input: 4,
        output: 1,
    };

    let candidates = vec![BottleneckNeuronCandidate {
        neuron_uuid: "h1".to_string(),
        fan_in: 4,
        fan_out: 1,
        error_contribution_ratio: 0.5,
        bottleneck_score: 0.6,
        estimated_improvement: 0.006,
        recommended_actions: vec![
            "addParallelNeuron".to_string(),
            "addBypassSynapse".to_string(),
        ],
        upstream_uuids: vec![
            "i0".to_string(),
            "i1".to_string(),
            "i2".to_string(),
            "i3".to_string(),
        ],
        downstream_uuids: vec!["o1".to_string()],
    }];

    let coordinated = bottleneck_neurons_to_coordinated_candidates(&candidates, &creature);

    assert!(
        !coordinated.is_empty(),
        "expected at least one coordinated candidate"
    );

    // The first candidate should include an AddNeuron operation
    let first = &coordinated[0];
    let has_add_neuron = first
        .operations
        .iter()
        .any(|op| matches!(op, CoordinatedStructuralOpJson::AddNeuron { .. }));
    assert!(
        has_add_neuron,
        "parallel neuron candidate should include AddNeuron"
    );
    assert!(first.expected_creature_score_gain > 0.0);
}
