//! Unit tests for dormant synapse detection (Issue #376).

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
fn make_synapse(from: &str, to: &str, weight: f32) -> SynapseJson {
    SynapseJson {
        from_uuid: from.to_string(),
        to_uuid: to.to_string(),
        weight,
        synapse_type: None,
    }
}

/// Helper: create source activation records.
fn source_records(uuid: &str, count: usize, activation: f32) -> Vec<DiscoverRecord> {
    (0..count)
        .map(|i| DiscoverRecord {
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
fn synapse_with_near_zero_weight_and_low_contribution_is_detected() {
    let creature = CreatureJson {
        neurons: vec![
            neuron("i0", "input", "IDENTITY"),
            neuron("i1", "input", "IDENTITY"),
            neuron("h1", "hidden", "TANH"),
        ],
        synapses: vec![
            make_synapse("i0", "h1", 0.00005), // dormant — weight < 1e-4
            make_synapse("i1", "h1", 1.0),     // active — ensures fan-in > 1
        ],
        input: 2,
        output: 0,
    };

    let records = vec![("i0".to_string(), source_records("i0", 30, 0.5))];

    let detected = detect_dormant_synapses(&creature, &records);

    assert!(
        !detected.is_empty(),
        "expected dormant synapse to be detected"
    );
    assert_eq!(detected[0].from_neuron_uuid, "i0");
    assert_eq!(detected[0].to_neuron_uuid, "h1");
    assert!(detected[0].estimated_improvement > 0.0);
}

// ── Exclusion criteria ──────────────────────────────────────────────────────

#[test]
fn synapse_with_significant_weight_is_not_detected() {
    let creature = CreatureJson {
        neurons: vec![
            neuron("i0", "input", "IDENTITY"),
            neuron("i1", "input", "IDENTITY"),
            neuron("h1", "hidden", "TANH"),
        ],
        synapses: vec![
            make_synapse("i0", "h1", 0.5), // weight well above 1e-4
            make_synapse("i1", "h1", 1.0),
        ],
        input: 2,
        output: 0,
    };

    let records = vec![("i0".to_string(), source_records("i0", 30, 0.5))];

    let detected = detect_dormant_synapses(&creature, &records);

    assert!(
        detected.is_empty(),
        "synapse with weight 0.5 should not be flagged as dormant"
    );
}

#[test]
fn sole_input_synapse_is_not_detected() {
    // Only one synapse to the target — removal would be destructive
    let creature = CreatureJson {
        neurons: vec![
            neuron("i0", "input", "IDENTITY"),
            neuron("h1", "hidden", "TANH"),
        ],
        synapses: vec![
            make_synapse("i0", "h1", 0.00005), // dormant weight but sole input
        ],
        input: 1,
        output: 0,
    };

    let records = vec![("i0".to_string(), source_records("i0", 30, 0.5))];

    let detected = detect_dormant_synapses(&creature, &records);

    assert!(
        detected.is_empty(),
        "sole input synapse should not be flagged (would be destructive)"
    );
}

// ── Edge cases ──────────────────────────────────────────────────────────────

#[test]
fn insufficient_samples_prevents_dormant_detection() {
    let creature = CreatureJson {
        neurons: vec![
            neuron("i0", "input", "IDENTITY"),
            neuron("i1", "input", "IDENTITY"),
            neuron("h1", "hidden", "TANH"),
        ],
        synapses: vec![
            make_synapse("i0", "h1", 0.00005),
            make_synapse("i1", "h1", 1.0),
        ],
        input: 2,
        output: 0,
    };

    // Only 5 samples — below MIN_SAMPLES_FOR_DORMANT (20)
    let records = vec![("i0".to_string(), source_records("i0", 5, 0.5))];

    let detected = detect_dormant_synapses(&creature, &records);

    assert!(
        detected.is_empty(),
        "fewer than 20 samples should prevent detection"
    );
}

#[test]
fn empty_creature_returns_empty() {
    let creature = CreatureJson {
        neurons: vec![],
        synapses: vec![],
        input: 0,
        output: 0,
    };

    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![];

    let detected = detect_dormant_synapses(&creature, &records);

    assert!(detected.is_empty());
}

// ── Conversion ──────────────────────────────────────────────────────────────

#[test]
fn conversion_produces_remove_synapse_operation() {
    let candidates = vec![DormantSynapseCandidate {
        from_neuron_uuid: "i0".to_string(),
        to_neuron_uuid: "h1".to_string(),
        weight: 0.00005,
        mean_abs_contribution: 0.000025,
        other_fan_in: 1,
        sample_count: 30,
        estimated_improvement: 0.00075,
    }];

    let coordinated = dormant_synapses_to_coordinated_candidates(&candidates);

    assert_eq!(coordinated.len(), 1);
    let first = &coordinated[0];
    let has_remove_synapse = first.operations.iter().any(|op| {
        matches!(op, CoordinatedStructuralOpJson::RemoveSynapse { from_neuron_uuid, to_neuron_uuid }
            if from_neuron_uuid == "i0" && to_neuron_uuid == "h1")
    });
    assert!(has_remove_synapse, "expected RemoveSynapse operation");
    assert!(first.expected_creature_score_gain > 0.0);
}
