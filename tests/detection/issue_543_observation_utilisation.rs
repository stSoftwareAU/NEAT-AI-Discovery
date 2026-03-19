//! Issue #543: Integration tests for observation utilisation detection.
//!
//! Tests the `observation_utilisation` detection module which identifies input
//! neurons (observations) with low effective utilisation ratios due to sentinel
//! values and recommends bias adjustments to improve their contribution.
//!
//! These tests exercise real detection and conversion functions with test data.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use neat_ai_discovery::analysis::detection::observation_utilisation::{
    detect_underutilised_observations, observation_utilisation_to_coordinated_candidates,
};
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

// =============================================================================
// Helpers
// =============================================================================

fn make_record(uuid: &str, idx: u32, activation: f32, error: f32) -> DiscoverRecord {
    DiscoverRecord {
        obs_index: idx,
        neuron_uuid: uuid.to_string(),
        value: Some(activation),
        activation,
        errors: vec![error],
    }
}

fn make_creature(input_count: usize) -> CreatureJson {
    let mut neurons = Vec::new();
    for i in 0..input_count {
        neurons.push(NeuronJson {
            uuid: format!("input-{i}"),
            neuron_type: "input".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        });
    }
    neurons.push(NeuronJson {
        uuid: "output-0".to_string(),
        neuron_type: "output".to_string(),
        squash: "TANH".to_string(),
        bias: 0.0,
    });

    let synapses: Vec<SynapseJson> = (0..input_count)
        .map(|i| SynapseJson {
            from_uuid: format!("input-{i}"),
            to_uuid: "output-0".to_string(),
            weight: 0.5,
            synapse_type: None,
        })
        .collect();

    CreatureJson {
        neurons,
        synapses,
        input: input_count,
        output: 1,
    }
}

// =============================================================================
// Input with sentinel values at -1 and low utilisation
// =============================================================================

#[test]
fn input_with_sentinel_cluster_detected() {
    let creature = make_creature(1);

    // Input-0 has 40% of values at -1.0 (sentinel) and 60% in the range [0.2, 0.8].
    // There is a clear gap between sentinel and effective range.
    let mut records: Vec<DiscoverRecord> = Vec::new();
    for i in 0..20 {
        records.push(make_record("input-0", i, -1.0, 0.01)); // sentinel, low error
    }
    for i in 20..50 {
        let activation = 0.2 + (i as f32 - 20.0) / 50.0;
        records.push(make_record("input-0", i, activation, 0.1));
    }

    let neuron_records = vec![("input-0".to_string(), records)];
    let detected = detect_underutilised_observations(&creature, &neuron_records);

    assert!(
        !detected.is_empty(),
        "Input with sentinel cluster should be detected as underutilised"
    );

    let c = &detected[0];
    assert_eq!(c.neuron_uuid, "input-0");
    assert!(
        c.utilisation_ratio < 1.0,
        "Utilisation ratio should be less than 1.0"
    );
    assert!(
        !c.sentinel_values.is_empty(),
        "Should detect sentinel values"
    );
}

// =============================================================================
// Input without sentinels → NOT detected
// =============================================================================

#[test]
fn input_without_sentinels_not_detected() {
    let creature = make_creature(1);

    // Input-0 has uniformly distributed values across [0.0, 1.0] — no sentinel
    let records: Vec<DiscoverRecord> = (0..50)
        .map(|i| make_record("input-0", i, i as f32 / 50.0, 0.1))
        .collect();

    let neuron_records = vec![("input-0".to_string(), records)];
    let detected = detect_underutilised_observations(&creature, &neuron_records);

    assert!(
        detected.is_empty(),
        "Input without sentinel values should NOT be detected"
    );
}

// =============================================================================
// Too few samples returns empty
// =============================================================================

#[test]
fn too_few_samples_returns_empty() {
    let creature = make_creature(1);

    let records: Vec<DiscoverRecord> = (0..5)
        .map(|i| make_record("input-0", i, -1.0, 0.01))
        .collect();

    let neuron_records = vec![("input-0".to_string(), records)];
    let detected = detect_underutilised_observations(&creature, &neuron_records);

    assert!(detected.is_empty(), "Too few samples should return empty");
}

// =============================================================================
// Hidden neurons are ignored (only input neurons are observations)
// =============================================================================

#[test]
fn hidden_neurons_ignored() {
    let mut creature = make_creature(1);
    creature.neurons.push(NeuronJson {
        uuid: "hidden-0".to_string(),
        neuron_type: "hidden".to_string(),
        squash: "TANH".to_string(),
        bias: 0.0,
    });

    // Records for hidden neuron with sentinel-like patterns
    let records: Vec<DiscoverRecord> = (0..50)
        .map(|i| {
            if i < 25 {
                make_record("hidden-0", i, -1.0, 0.01)
            } else {
                make_record("hidden-0", i, 0.5, 0.1)
            }
        })
        .collect();

    let neuron_records = vec![("hidden-0".to_string(), records)];
    let detected = detect_underutilised_observations(&creature, &neuron_records);

    assert!(
        detected.is_empty(),
        "Hidden neurons should be ignored by observation utilisation detection"
    );
}

// =============================================================================
// Coordinated candidate conversion
// =============================================================================

#[test]
fn underutilised_observations_convert_to_coordinated_candidates() {
    use neat_ai_discovery::analysis::detection::observation_utilisation::UnderutilisedObservation;

    let observations = vec![UnderutilisedObservation {
        neuron_uuid: "input-0".to_string(),
        sentinel_values: vec![-1.0],
        utilisation_ratio: 0.4,
        effective_min: 0.2,
        effective_max: 0.8,
        sample_count: 50,
        estimated_improvement: 0.005,
        reason: "40% of values are sentinel -1.0".to_string(),
    }];

    let coordinated =
        observation_utilisation_to_coordinated_candidates(&observations, &make_creature(1));

    assert!(
        !coordinated.is_empty(),
        "Should produce coordinated candidates"
    );
    assert!(coordinated[0].expected_creature_score_gain > 0.0);
    assert!(coordinated[0].comment.as_ref().unwrap().contains("543"));
}

// =============================================================================
// Multiple inputs can produce multiple candidates
// =============================================================================

#[test]
fn multiple_underutilised_inputs_detected() {
    let creature = make_creature(2);

    // Both inputs have sentinel clusters
    let records_0: Vec<DiscoverRecord> = (0..50)
        .map(|i| {
            if i < 20 {
                make_record("input-0", i, -1.0, 0.01)
            } else {
                make_record("input-0", i, 0.2 + (i as f32 - 20.0) / 50.0, 0.1)
            }
        })
        .collect();

    let records_1: Vec<DiscoverRecord> = (0..50)
        .map(|i| {
            if i < 25 {
                make_record("input-1", i, -1.0, 0.01)
            } else {
                make_record("input-1", i, 0.3 + (i as f32 - 25.0) / 50.0, 0.1)
            }
        })
        .collect();

    let neuron_records = vec![
        ("input-0".to_string(), records_0),
        ("input-1".to_string(), records_1),
    ];

    let detected = detect_underutilised_observations(&creature, &neuron_records);

    assert_eq!(
        detected.len(),
        2,
        "Both underutilised inputs should be detected"
    );
}
