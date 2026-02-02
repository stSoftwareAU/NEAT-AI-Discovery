//! Unit tests for opposing synapse detection (Issue #376).

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

/// Helper: create source activation records with obs_index alignment.
fn source_activation_records(uuid: &str, activations: &[f32]) -> Vec<DiscoverRecord> {
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

/// Helper: create target error records with obs_index alignment.
fn target_error_records(uuid: &str, errors: &[f32]) -> Vec<DiscoverRecord> {
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
fn synapse_with_contribution_positively_correlated_to_error_is_detected() {
    // Source activation and target error are positively correlated,
    // meaning the synapse pushes the output in the wrong direction
    let activations: Vec<f32> = (0..30).map(|i| (i as f32) / 30.0).collect();
    let errors: Vec<f32> = (0..30).map(|i| (i as f32) / 30.0).collect(); // same direction

    let creature = CreatureJson {
        neurons: vec![
            neuron("src", "input", "IDENTITY"),
            neuron("out", "output", "IDENTITY"),
        ],
        synapses: vec![make_synapse("src", "out", 1.0)],
        input: 1,
        output: 1,
    };

    let records = vec![
        (
            "src".to_string(),
            source_activation_records("src", &activations),
        ),
        ("out".to_string(), target_error_records("out", &errors)),
    ];

    let detected = detect_opposing_synapses(&creature, &records);

    assert!(
        !detected.is_empty(),
        "expected opposing synapse to be detected"
    );
    assert_eq!(detected[0].from_neuron_uuid, "src");
    assert_eq!(detected[0].to_neuron_uuid, "out");
    assert!(detected[0].contribution_error_correlation >= 0.3);
}

// ── Exclusion criteria ──────────────────────────────────────────────────────

#[test]
fn synapse_with_negative_correlation_is_not_detected() {
    // Source activation and target error are negatively correlated — synapse is helpful
    let activations: Vec<f32> = (0..30).map(|i| (i as f32) / 30.0).collect();
    let errors: Vec<f32> = (0..30).map(|i| -((i as f32) / 30.0)).collect(); // opposite direction

    let creature = CreatureJson {
        neurons: vec![
            neuron("src", "input", "IDENTITY"),
            neuron("out", "output", "IDENTITY"),
        ],
        synapses: vec![make_synapse("src", "out", 1.0)],
        input: 1,
        output: 1,
    };

    let records = vec![
        (
            "src".to_string(),
            source_activation_records("src", &activations),
        ),
        ("out".to_string(), target_error_records("out", &errors)),
    ];

    let detected = detect_opposing_synapses(&creature, &records);

    assert!(
        detected.is_empty(),
        "negatively correlated synapse should not be flagged as opposing"
    );
}

#[test]
fn synapse_to_hidden_neuron_is_not_detected() {
    // Opposing detection only analyses synapses targeting output neurons
    let activations: Vec<f32> = (0..30).map(|i| (i as f32) / 30.0).collect();
    let errors: Vec<f32> = (0..30).map(|i| (i as f32) / 30.0).collect();

    let creature = CreatureJson {
        neurons: vec![
            neuron("src", "input", "IDENTITY"),
            neuron("h1", "hidden", "TANH"),
            neuron("out", "output", "IDENTITY"),
        ],
        synapses: vec![
            make_synapse("src", "h1", 1.0),
            make_synapse("h1", "out", 1.0),
        ],
        input: 1,
        output: 1,
    };

    let records = vec![
        (
            "src".to_string(),
            source_activation_records("src", &activations),
        ),
        ("h1".to_string(), target_error_records("h1", &errors)),
    ];

    let detected = detect_opposing_synapses(&creature, &records);

    assert!(
        detected.is_empty(),
        "synapses to hidden neurons should not be detected"
    );
}

#[test]
fn dormant_synapse_is_not_detected_as_opposing() {
    // Near-zero contribution should be excluded (handled by dormant synapse module)
    let activations: Vec<f32> = (0..30).map(|i| (i as f32) / 30.0).collect();
    let errors: Vec<f32> = (0..30).map(|i| (i as f32) / 30.0).collect();

    let creature = CreatureJson {
        neurons: vec![
            neuron("src", "input", "IDENTITY"),
            neuron("out", "output", "IDENTITY"),
        ],
        synapses: vec![make_synapse("src", "out", 0.0001)], // tiny weight → negligible contribution
        input: 1,
        output: 1,
    };

    let records = vec![
        (
            "src".to_string(),
            source_activation_records("src", &activations),
        ),
        ("out".to_string(), target_error_records("out", &errors)),
    ];

    let detected = detect_opposing_synapses(&creature, &records);

    assert!(
        detected.is_empty(),
        "dormant synapse with negligible contribution should not be flagged"
    );
}

// ── Edge cases ──────────────────────────────────────────────────────────────

#[test]
fn insufficient_aligned_samples_prevents_detection() {
    // Source has obs_indices 0-29, target has obs_indices 1000-1029 — no overlap
    let source_recs: Vec<DiscoverRecord> = (0..30)
        .map(|i| DiscoverRecord {
            obs_index: i,
            neuron_uuid: "src".to_string(),
            value: None,
            activation: (i as f32) / 30.0,
            errors: vec![],
        })
        .collect();

    let target_recs: Vec<DiscoverRecord> = (0..30)
        .map(|i| DiscoverRecord {
            obs_index: i + 1000,
            neuron_uuid: "out".to_string(),
            value: None,
            activation: 0.0,
            errors: vec![(i as f32) / 30.0],
        })
        .collect();

    let creature = CreatureJson {
        neurons: vec![
            neuron("src", "input", "IDENTITY"),
            neuron("out", "output", "IDENTITY"),
        ],
        synapses: vec![make_synapse("src", "out", 1.0)],
        input: 1,
        output: 1,
    };

    let records = vec![
        ("src".to_string(), source_recs),
        ("out".to_string(), target_recs),
    ];

    let detected = detect_opposing_synapses(&creature, &records);

    assert!(
        detected.is_empty(),
        "no aligned obs_indices should prevent detection"
    );
}

// ── Conversion ──────────────────────────────────────────────────────────────

#[test]
fn high_correlation_recommends_removal() {
    let candidates = vec![OpposingSynapseCandidate {
        from_neuron_uuid: "src".to_string(),
        to_neuron_uuid: "out".to_string(),
        weight: 1.0,
        contribution_error_correlation: 0.8,
        mean_abs_contribution: 0.5,
        sample_count: 30,
        recommend_removal: true,
        estimated_improvement: 0.02,
    }];

    let coordinated = opposing_synapses_to_coordinated_candidates(&candidates);

    assert_eq!(coordinated.len(), 1);
    let has_remove = coordinated[0]
        .operations
        .iter()
        .any(|op| matches!(op, CoordinatedStructuralOpJson::RemoveSynapse { .. }));
    assert!(has_remove, "high correlation should produce RemoveSynapse");
}

#[test]
fn moderate_correlation_recommends_weight_flip() {
    let candidates = vec![OpposingSynapseCandidate {
        from_neuron_uuid: "src".to_string(),
        to_neuron_uuid: "out".to_string(),
        weight: 0.5,
        contribution_error_correlation: 0.35,
        mean_abs_contribution: 0.2,
        sample_count: 30,
        recommend_removal: false,
        estimated_improvement: 0.0035,
    }];

    let coordinated = opposing_synapses_to_coordinated_candidates(&candidates);

    assert_eq!(coordinated.len(), 1);
    let has_set_weight = coordinated[0].operations.iter().any(
        |op| matches!(op, CoordinatedStructuralOpJson::SetWeight { weight, .. } if *weight == -0.5),
    );
    assert!(
        has_set_weight,
        "moderate correlation should produce SetWeight with negated weight"
    );
}
