#![allow(clippy::cast_precision_loss)] // Small fixed sample indices make fixture activations.

use std::collections::HashMap;

use neat_ai_discovery::analysis::detection::{
    input_sensitivity::{self, InputSensitivityConfig},
    observation_utilisation::{self, UnderutilisedObservation},
};
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

fn neuron(uuid: &str, neuron_type: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
        squash: "IDENTITY".to_string(),
        bias: 0.0,
    }
}

fn synapse(from: &str, to: &str, weight: f32) -> SynapseJson {
    SynapseJson {
        from_uuid: from.to_string(),
        to_uuid: to.to_string(),
        weight,
        synapse_type: None,
    }
}

#[test]
fn issue_2379_observation_candidates_keep_target_weights_order_and_duplicate_semantics() {
    let creature = CreatureJson {
        neurons: vec![
            neuron("input", "input"),
            neuron("target-a", "output"),
            neuron("target-b", "output"),
        ],
        synapses: vec![
            synapse("input", "target-a", 0.5),
            synapse("input", "target-b", -0.25),
            synapse("input", "target-a", 9.0),
        ],
        input: 1,
        output: 2,
    };
    let observations = [UnderutilisedObservation {
        neuron_uuid: "input".to_string(),
        sentinel_values: vec![-1.0],
        utilisation_ratio: 0.3,
        effective_min: 0.2,
        effective_max: 0.8,
        sample_count: 50,
        estimated_improvement: 0.005,
        reason: "fixture".to_string(),
    }];

    let candidates = observation_utilisation::observation_utilisation_to_coordinated_candidates(
        &observations,
        &creature,
    );
    let biases: Vec<(&str, f32)> = candidates
        .iter()
        .map(|candidate| match &candidate.operations[0] {
            neat_ai_discovery::CoordinatedStructuralOpJson::SetBias { neuron_uuid, bias } => {
                (neuron_uuid.as_str(), *bias)
            }
            other => panic!("expected SetBias, got {other:?}"),
        })
        .collect();

    assert_eq!(biases.len(), 3);
    assert_eq!(biases[0], ("target-a", -0.25));
    assert_eq!(biases[1], ("target-b", 0.125));
    assert_eq!(biases[2], ("target-a", -0.25));
}

fn records(uuid: &str) -> Vec<DiscoverRecord> {
    (0..3)
        .map(|obs_index| {
            let value = obs_index as f32;
            DiscoverRecord::new(obs_index, uuid.to_string(), None, value, vec![value])
        })
        .collect()
}

#[test]
fn issue_2379_dominant_inputs_preserve_direct_hidden_and_record_filters() {
    let creature = CreatureJson {
        neurons: vec![
            neuron("input", "input"),
            neuron("hidden-connected", "hidden"),
            neuron("hidden-disconnected", "hidden"),
            neuron("hidden-no-records", "hidden"),
            neuron("output", "output"),
        ],
        synapses: vec![
            synapse("input", "output", 2.0),
            synapse("input", "hidden-connected", 2.0),
            synapse("input", "hidden-disconnected", 2.0),
            synapse("input", "hidden-no-records", 2.0),
            synapse("hidden-connected", "output", 1.0),
            synapse("hidden-no-records", "output", 1.0),
        ],
        input: 1,
        output: 1,
    };
    let neuron_records = vec![
        ("input".to_string(), records("input")),
        ("output".to_string(), records("output")),
        ("hidden-connected".to_string(), records("hidden-connected")),
        (
            "hidden-disconnected".to_string(),
            records("hidden-disconnected"),
        ),
    ];
    let config = InputSensitivityConfig {
        dominance_threshold: 0.01,
        gradient_threshold: 10.0,
        min_samples: 2,
    };

    let candidates = input_sensitivity::detect_dominant_inputs(&creature, &neuron_records, &config);
    let targets: HashMap<&str, usize> =
        candidates
            .iter()
            .fold(HashMap::new(), |mut counts, candidate| {
                *counts
                    .entry(candidate.target_neuron_uuid.as_str())
                    .or_default() += 1;
                counts
            });

    assert_eq!(
        targets.get("output"),
        Some(&1),
        "direct output remains eligible"
    );
    assert_eq!(
        targets.get("hidden-connected"),
        Some(&1),
        "recorded hidden target remains eligible when it reaches an output"
    );
    assert!(!targets.contains_key("hidden-disconnected"));
    assert!(!targets.contains_key("hidden-no-records"));
}
