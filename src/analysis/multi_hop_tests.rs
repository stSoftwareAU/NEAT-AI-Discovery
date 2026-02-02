//! Unit tests for multi-hop candidate analysis (Issue #376).

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

/// Helper: create activation records for a neuron.
fn activation_records(uuid: &str, activations: &[f32]) -> Vec<DiscoverRecord> {
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

/// Helper: create error records for a target neuron.
fn error_records(uuid: &str, errors: &[f32]) -> Vec<DiscoverRecord> {
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

// ── Detection criteria ──────────────────────────────────────────────────────

#[test]
fn unconnected_neuron_correlating_with_target_error_is_detected() {
    // h1 is not connected to o1, but its activation correlates with o1's error
    let creature = CreatureJson {
        neurons: vec![
            neuron("i0", "input", "IDENTITY"),
            neuron("h1", "hidden", "TANH"),
            neuron("o1", "output", "IDENTITY"),
        ],
        synapses: vec![
            synapse("i0", "h1", 1.0),
            synapse("i0", "o1", 1.0),
            // No synapse from h1 to o1
        ],
        input: 1,
        output: 1,
    };

    // h1 activation and o1 error are correlated
    let activations: Vec<f32> = (0..30).map(|i| (i as f32) / 30.0).collect();
    let errors: Vec<f32> = (0..30).map(|i| (i as f32) / 30.0 * 0.5).collect();

    let records = vec![
        ("h1".to_string(), activation_records("h1", &activations)),
        ("o1".to_string(), error_records("o1", &errors)),
        ("i0".to_string(), activation_records("i0", &activations)),
    ];

    let detected = detect_multi_hop_candidates(&creature, &records);

    assert!(
        !detected.is_empty(),
        "expected multi-hop candidate for correlated unconnected neuron"
    );
    // The path should connect h1 to o1
    let has_relevant_path = detected
        .iter()
        .any(|c| c.path.contains(&"h1".to_string()) && c.path.last() == Some(&"o1".to_string()));
    assert!(has_relevant_path, "expected path involving h1 → o1");
}

// ── Exclusion criteria ──────────────────────────────────────────────────────

#[test]
fn already_connected_neuron_is_not_detected() {
    let creature = CreatureJson {
        neurons: vec![
            neuron("i0", "input", "IDENTITY"),
            neuron("h1", "hidden", "TANH"),
            neuron("o1", "output", "IDENTITY"),
        ],
        synapses: vec![
            synapse("i0", "h1", 1.0),
            synapse("h1", "o1", 1.0), // Already connected
            synapse("i0", "o1", 0.5),
        ],
        input: 1,
        output: 1,
    };

    let activations: Vec<f32> = (0..30).map(|i| (i as f32) / 30.0).collect();
    let errors: Vec<f32> = (0..30).map(|i| (i as f32) / 30.0 * 0.5).collect();

    let records = vec![
        ("h1".to_string(), activation_records("h1", &activations)),
        ("o1".to_string(), error_records("o1", &errors)),
        ("i0".to_string(), activation_records("i0", &activations)),
    ];

    let detected = detect_multi_hop_candidates(&creature, &records);

    // h1 → o1 path should be excluded because h1 is already connected to o1
    let has_direct_h1_o1 = detected
        .iter()
        .any(|c| c.path.len() == 2 && c.path[0] == "h1" && c.path[1] == "o1");
    assert!(
        !has_direct_h1_o1,
        "already-connected pairs should be excluded from two-hop candidates"
    );
}

#[test]
fn uncorrelated_neuron_is_not_detected() {
    let creature = CreatureJson {
        neurons: vec![
            neuron("i0", "input", "IDENTITY"),
            neuron("h1", "hidden", "TANH"),
            neuron("o1", "output", "IDENTITY"),
        ],
        synapses: vec![synapse("i0", "h1", 1.0), synapse("i0", "o1", 1.0)],
        input: 1,
        output: 1,
    };

    // h1 activation is constant — no correlation with anything
    let activations: Vec<f32> = vec![0.5; 30];
    let errors: Vec<f32> = (0..30).map(|i| (i as f32) / 30.0).collect();

    let records = vec![
        ("h1".to_string(), activation_records("h1", &activations)),
        ("o1".to_string(), error_records("o1", &errors)),
        (
            "i0".to_string(),
            activation_records("i0", &(0..30).map(|i| i as f32 / 30.0).collect::<Vec<_>>()),
        ),
    ];

    let detected = detect_multi_hop_candidates(&creature, &records);

    // h1 has constant activation → zero correlation → should not appear
    let has_h1_path = detected.iter().any(|c| c.path.contains(&"h1".to_string()));
    assert!(
        !has_h1_path,
        "constant-activation neuron should not produce candidates"
    );
}

// ── Edge cases ──────────────────────────────────────────────────────────────

#[test]
fn empty_records_returns_empty() {
    let creature = CreatureJson {
        neurons: vec![
            neuron("i0", "input", "IDENTITY"),
            neuron("o1", "output", "IDENTITY"),
        ],
        synapses: vec![synapse("i0", "o1", 1.0)],
        input: 1,
        output: 1,
    };

    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![];

    let detected = detect_multi_hop_candidates(&creature, &records);

    assert!(detected.is_empty());
}

#[test]
fn insufficient_samples_prevents_multi_hop_detection() {
    let creature = CreatureJson {
        neurons: vec![
            neuron("i0", "input", "IDENTITY"),
            neuron("h1", "hidden", "TANH"),
            neuron("o1", "output", "IDENTITY"),
        ],
        synapses: vec![synapse("i0", "h1", 1.0), synapse("i0", "o1", 1.0)],
        input: 1,
        output: 1,
    };

    // Only 5 samples — below MIN_SAMPLES_FOR_CORRELATION (20)
    let activations: Vec<f32> = (0..5).map(|i| (i as f32) / 5.0).collect();
    let errors: Vec<f32> = (0..5).map(|i| (i as f32) / 5.0).collect();

    let records = vec![
        ("h1".to_string(), activation_records("h1", &activations)),
        ("o1".to_string(), error_records("o1", &errors)),
    ];

    let detected = detect_multi_hop_candidates(&creature, &records);

    assert!(
        detected.is_empty(),
        "fewer than 20 samples should prevent detection"
    );
}

// ── Conversion ──────────────────────────────────────────────────────────────

#[test]
fn two_hop_conversion_produces_add_synapse() {
    let creature = CreatureJson {
        neurons: vec![
            neuron("h1", "hidden", "TANH"),
            neuron("o1", "output", "IDENTITY"),
        ],
        synapses: vec![],
        input: 0,
        output: 1,
    };

    let candidates = vec![MultiHopCandidate {
        path: vec!["h1".to_string(), "o1".to_string()],
        estimated_improvement: 0.005,
        correlation_strength: 0.5,
    }];

    let coordinated = multi_hop_to_coordinated_candidates(&candidates, &creature);

    assert_eq!(coordinated.len(), 1);
    let has_add_synapse = coordinated[0].operations.iter().any(|op| {
        matches!(op, CoordinatedStructuralOpJson::AddSynapse {
            from_neuron_uuid, to_neuron_uuid, ..
        } if from_neuron_uuid == "h1" && to_neuron_uuid == "o1")
    });
    assert!(
        has_add_synapse,
        "two-hop should produce AddSynapse from source to target"
    );
}

#[test]
fn three_hop_conversion_produces_add_neuron_and_synapses() {
    let creature = CreatureJson {
        neurons: vec![
            neuron("i0", "input", "IDENTITY"),
            neuron("h1", "hidden", "TANH"),
            neuron("o1", "output", "IDENTITY"),
        ],
        synapses: vec![],
        input: 1,
        output: 1,
    };

    let candidates = vec![MultiHopCandidate {
        path: vec!["i0".to_string(), "h1".to_string(), "o1".to_string()],
        estimated_improvement: 0.003,
        correlation_strength: 0.4,
    }];

    let coordinated = multi_hop_to_coordinated_candidates(&candidates, &creature);

    assert_eq!(coordinated.len(), 1);
    let first = &coordinated[0];

    let has_add_neuron = first
        .operations
        .iter()
        .any(|op| matches!(op, CoordinatedStructuralOpJson::AddNeuron { .. }));
    assert!(
        has_add_neuron,
        "three-hop should produce AddNeuron for relay"
    );

    let synapse_count = first
        .operations
        .iter()
        .filter(|op| matches!(op, CoordinatedStructuralOpJson::AddSynapse { .. }))
        .count();
    assert_eq!(
        synapse_count, 2,
        "three-hop should produce 2 AddSynapse operations"
    );
}
