//! Unit tests for correlated error pattern detection (Issue #376).

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

/// Helper: create error records for an output neuron with specified error values.
fn output_error_records(uuid: &str, errors: &[f32]) -> Vec<DiscoverRecord> {
    errors
        .iter()
        .enumerate()
        .map(|(i, &error)| DiscoverRecord {
            obs_index: i as u32,
            neuron_uuid: uuid.to_string(),
            value: None,
            activation: 0.0,
            errors: vec![error],
        })
        .collect()
}

/// Helper: create input activation records.
fn input_activation_records(uuid: &str, activations: &[f32]) -> Vec<DiscoverRecord> {
    activations
        .iter()
        .enumerate()
        .map(|(i, &activation)| DiscoverRecord {
            obs_index: i as u32,
            neuron_uuid: uuid.to_string(),
            value: None,
            activation,
            errors: vec![],
        })
        .collect()
}

// ── Detection criteria ──────────────────────────────────────────────────────

#[test]
fn two_outputs_with_highly_correlated_errors_are_grouped() {
    let creature = CreatureJson {
        neurons: vec![
            neuron("i0", "input", "IDENTITY"),
            neuron("o1", "output", "IDENTITY"),
            neuron("o2", "output", "IDENTITY"),
        ],
        synapses: vec![synapse("i0", "o1", 1.0), synapse("i0", "o2", 1.0)],
        input: 1,
        output: 2,
    };

    // Both outputs have identical error patterns — correlation = 1.0
    let errors: Vec<f32> = (0..30).map(|i| (i as f32) / 30.0 - 0.5).collect();
    let records = vec![
        ("o1".to_string(), output_error_records("o1", &errors)),
        ("o2".to_string(), output_error_records("o2", &errors)),
        ("i0".to_string(), input_activation_records("i0", &errors)),
    ];

    let detected = detect_correlated_error_patterns(&creature, &records);

    assert!(
        !detected.is_empty(),
        "expected correlated error group to be detected"
    );
    let group = &detected[0];
    assert!(group.output_neuron_uuids.len() >= 2);
    assert!(group.mean_correlation >= 0.7);
}

// ── Exclusion criteria ──────────────────────────────────────────────────────

#[test]
fn uncorrelated_outputs_are_not_grouped() {
    let creature = CreatureJson {
        neurons: vec![
            neuron("i0", "input", "IDENTITY"),
            neuron("o1", "output", "IDENTITY"),
            neuron("o2", "output", "IDENTITY"),
        ],
        synapses: vec![synapse("i0", "o1", 1.0), synapse("i0", "o2", 1.0)],
        input: 1,
        output: 2,
    };

    // o1 errors increase, o2 errors are random — low correlation
    let errors_o1: Vec<f32> = (0..30).map(|i| (i as f32) / 30.0).collect();
    let errors_o2: Vec<f32> = (0..30)
        .map(|i| {
            if i % 3 == 0 {
                0.5
            } else if i % 3 == 1 {
                -0.3
            } else {
                0.1
            }
        })
        .collect();
    let records = vec![
        ("o1".to_string(), output_error_records("o1", &errors_o1)),
        ("o2".to_string(), output_error_records("o2", &errors_o2)),
    ];

    let detected = detect_correlated_error_patterns(&creature, &records);

    assert!(
        detected.is_empty(),
        "outputs with uncorrelated errors should not form a group"
    );
}

#[test]
fn single_output_neuron_returns_empty() {
    let creature = CreatureJson {
        neurons: vec![
            neuron("i0", "input", "IDENTITY"),
            neuron("o1", "output", "IDENTITY"),
        ],
        synapses: vec![synapse("i0", "o1", 1.0)],
        input: 1,
        output: 1,
    };

    let errors: Vec<f32> = (0..30).map(|i| (i as f32) / 30.0).collect();
    let records = vec![("o1".to_string(), output_error_records("o1", &errors))];

    let detected = detect_correlated_error_patterns(&creature, &records);

    assert!(
        detected.is_empty(),
        "need at least 2 output neurons for correlation"
    );
}

// ── Edge cases ──────────────────────────────────────────────────────────────

#[test]
fn insufficient_samples_prevents_correlation_detection() {
    let creature = CreatureJson {
        neurons: vec![
            neuron("i0", "input", "IDENTITY"),
            neuron("o1", "output", "IDENTITY"),
            neuron("o2", "output", "IDENTITY"),
        ],
        synapses: vec![synapse("i0", "o1", 1.0), synapse("i0", "o2", 1.0)],
        input: 1,
        output: 2,
    };

    // Only 5 samples — below MIN_SAMPLES_FOR_CORRELATION (20)
    let errors: Vec<f32> = (0..5).map(|i| (i as f32) / 5.0).collect();
    let records = vec![
        ("o1".to_string(), output_error_records("o1", &errors)),
        ("o2".to_string(), output_error_records("o2", &errors)),
    ];

    let detected = detect_correlated_error_patterns(&creature, &records);

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
            neuron("o1", "output", "IDENTITY"),
            neuron("o2", "output", "IDENTITY"),
        ],
        synapses: vec![synapse("i0", "o1", 1.0), synapse("i0", "o2", 1.0)],
        input: 1,
        output: 2,
    };

    let groups = vec![CorrelatedErrorGroup {
        output_neuron_uuids: vec!["o1".to_string(), "o2".to_string()],
        mean_correlation: 0.9,
        shared_error_sample_count: 25,
        total_sample_count: 30,
        predictive_input_uuids: vec!["i0".to_string()],
        estimated_improvement: 0.01,
    }];

    let coordinated = correlated_errors_to_coordinated_candidates(&groups, &creature);

    assert!(!coordinated.is_empty());
    let first = &coordinated[0];

    let has_add_neuron = first
        .operations
        .iter()
        .any(|op| matches!(op, CoordinatedStructuralOpJson::AddNeuron { .. }));
    assert!(
        has_add_neuron,
        "expected AddNeuron for shared hidden neuron"
    );

    let synapse_count = first
        .operations
        .iter()
        .filter(|op| matches!(op, CoordinatedStructuralOpJson::AddSynapse { .. }))
        .count();
    // Should have: i0→shared + shared→o1 + shared→o2 = 3 synapses
    assert!(
        synapse_count >= 3,
        "expected at least 3 AddSynapse operations"
    );
}
