#![allow(clippy::cast_precision_loss)] // Benchmark activations use small integer sample indices.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use neat_ai_discovery::analysis::detection::{
    input_sensitivity::{self, InputSensitivityConfig},
    observation_utilisation::{self, UnderutilisedObservation},
};
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
use std::hint::black_box;

fn neuron(uuid: String, neuron_type: &str) -> NeuronJson {
    NeuronJson {
        uuid,
        neuron_type: neuron_type.to_string(),
        squash: "IDENTITY".to_string(),
        bias: 0.0,
    }
}

fn observation_fixture(size: usize) -> (CreatureJson, Vec<UnderutilisedObservation>) {
    let mut neurons = vec![neuron("input-0".to_string(), "input")];
    let synapses = (0..size)
        .map(|index| {
            let target = format!("output-{index}");
            neurons.push(neuron(target.clone(), "output"));
            SynapseJson {
                from_uuid: "input-0".to_string(),
                to_uuid: target,
                weight: 0.5,
                synapse_type: None,
            }
        })
        .collect();
    (
        CreatureJson {
            neurons,
            synapses,
            input: 1,
            output: size,
        },
        vec![UnderutilisedObservation {
            neuron_uuid: "input-0".to_string(),
            sentinel_values: vec![-1.0],
            utilisation_ratio: 0.3,
            effective_min: 0.2,
            effective_max: 0.8,
            sample_count: 50,
            estimated_improvement: 0.005,
            reason: "benchmark".to_string(),
        }],
    )
}

fn dominant_input_fixture(
    size: usize,
) -> (
    CreatureJson,
    Vec<(String, Vec<neat_ai_discovery::types::DiscoverRecord>)>,
) {
    let mut neurons = vec![
        neuron("shared-hidden".to_string(), "hidden"),
        neuron("output-0".to_string(), "output"),
    ];
    let mut synapses = Vec::with_capacity(size * 2);
    let mut records = Vec::with_capacity(size);
    for index in 0..size {
        let input_uuid = format!("input-{index}");
        neurons.push(neuron(input_uuid.clone(), "input"));
        synapses.push(SynapseJson {
            from_uuid: input_uuid.clone(),
            to_uuid: "shared-hidden".to_string(),
            weight: 0.5,
            synapse_type: None,
        });
        records.push((
            input_uuid.clone(),
            (0..20)
                .map(|obs_index| {
                    neat_ai_discovery::types::DiscoverRecord::new(
                        obs_index,
                        input_uuid.clone(),
                        None,
                        obs_index as f32,
                        vec![0.0],
                    )
                })
                .collect(),
        ));
    }
    for index in 0..size {
        let target = format!("hidden-target-{index}");
        neurons.push(neuron(target.clone(), "hidden"));
        synapses.push(SynapseJson {
            from_uuid: "shared-hidden".to_string(),
            to_uuid: target,
            weight: 0.5,
            synapse_type: None,
        });
    }
    (
        CreatureJson {
            neurons,
            synapses,
            input: size,
            output: 1,
        },
        records,
    )
}

fn synapse_rescans(c: &mut Criterion) {
    let mut group = c.benchmark_group("issue_2379_synapse_rescans");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(2));
    for size in [128, 512] {
        let (creature, observations) = observation_fixture(size);
        group.bench_with_input(
            BenchmarkId::new("observation_utilisation", size),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(
                        observation_utilisation::observation_utilisation_to_coordinated_candidates(
                            &observations,
                            &creature,
                        ),
                    )
                });
            },
        );

        let (creature, records) = dominant_input_fixture(size);
        let config = InputSensitivityConfig::default();
        group.bench_with_input(BenchmarkId::new("dominant_inputs", size), &size, |b, _| {
            b.iter(|| {
                black_box(input_sensitivity::detect_dominant_inputs(
                    &creature, &records, &config,
                ))
            });
        });
    }
    group.finish();
}

criterion_group!(benches, synapse_rescans);
criterion_main!(benches);
