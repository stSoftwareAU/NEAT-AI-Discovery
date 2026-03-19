//! Benchmark for Issue #487: Clone reduction in hot analysis paths.
//!
//! Measures performance of modules that had avoidable `clone()` calls:
//! - Bottleneck detection and candidate generation
//! - Restricted range detection and candidate generation
//! - Bounded range detection and candidate generation
//!
//! These benchmarks exercise the same code paths that were optimised
//! to use borrows instead of clones.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use neat_ai_discovery::analysis::detection::bottleneck::{
    bottleneck_neurons_to_coordinated_candidates, detect_bottleneck_neurons,
};
use neat_ai_discovery::analysis::detection::bounded_range::{
    bounded_range_to_coordinated_candidates, detect_bounded_range_neurons,
};
use neat_ai_discovery::analysis::detection::restricted_range::{
    RestrictedRangeConfig, detect_restricted_range_neurons,
    restricted_range_to_coordinated_candidates,
};
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

/// Create a creature with multiple bottleneck neurons (fan-in=8, fan-out=2).
fn create_bottleneck_creature(
    bottleneck_count: usize,
) -> (CreatureJson, Vec<(String, Vec<DiscoverRecord>)>) {
    let inputs_per_bottleneck = 8;
    let total_inputs = bottleneck_count * inputs_per_bottleneck;

    let mut neurons = Vec::new();
    let mut synapses = Vec::new();

    // Create bottleneck hidden neurons
    for b in 0..bottleneck_count {
        neurons.push(NeuronJson {
            uuid: format!("bottleneck-{b}"),
            neuron_type: "hidden".to_string(),
            squash: "TANH".to_string(),
            bias: 0.0,
        });

        // Connect inputs to this bottleneck
        for i in 0..inputs_per_bottleneck {
            let input_idx = b * inputs_per_bottleneck + i;
            synapses.push(SynapseJson {
                from_uuid: format!("input-{input_idx}"),
                to_uuid: format!("bottleneck-{b}"),
                weight: 0.3 + 0.1 * (i as f32),
                synapse_type: None,
            });
        }
    }

    // Output neurons (2 per bottleneck)
    for o in 0..2 {
        neurons.push(NeuronJson {
            uuid: format!("output-{o}"),
            neuron_type: "output".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        });

        for b in 0..bottleneck_count {
            synapses.push(SynapseJson {
                from_uuid: format!("bottleneck-{b}"),
                to_uuid: format!("output-{o}"),
                weight: 0.5,
                synapse_type: None,
            });
        }
    }

    let creature = CreatureJson {
        neurons,
        synapses,
        input: total_inputs,
        output: 2,
    };

    // Create records for bottleneck neurons
    let mut neuron_records = Vec::new();
    for b in 0..bottleneck_count {
        let uuid = format!("bottleneck-{b}");
        let records: Vec<DiscoverRecord> = (0..100)
            .map(|i| DiscoverRecord {
                obs_index: i,
                neuron_uuid: uuid.clone(),
                value: Some(0.5),
                activation: 0.3 + 0.01 * i as f32,
                errors: vec![0.05 * (i as f32 / 100.0)],
            })
            .collect();
        neuron_records.push((uuid, records));
    }

    (creature, neuron_records)
}

/// Create a creature with restricted-range hidden neurons.
fn create_restricted_range_creature(
    neuron_count: usize,
) -> (CreatureJson, Vec<(String, Vec<DiscoverRecord>)>) {
    let mut neurons = Vec::new();
    let mut synapses = Vec::new();

    for n in 0..neuron_count {
        neurons.push(NeuronJson {
            uuid: format!("hidden-{n}"),
            neuron_type: "hidden".to_string(),
            squash: "TANH".to_string(),
            bias: 0.0,
        });

        synapses.push(SynapseJson {
            from_uuid: format!("input-{n}"),
            to_uuid: format!("hidden-{n}"),
            weight: 0.1,
            synapse_type: None,
        });
    }

    neurons.push(NeuronJson {
        uuid: "output-0".to_string(),
        neuron_type: "output".to_string(),
        squash: "IDENTITY".to_string(),
        bias: 0.0,
    });

    for n in 0..neuron_count {
        synapses.push(SynapseJson {
            from_uuid: format!("hidden-{n}"),
            to_uuid: "output-0".to_string(),
            weight: 0.5,
            synapse_type: None,
        });
    }

    let creature = CreatureJson {
        neurons,
        synapses,
        input: neuron_count,
        output: 1,
    };

    // Records in narrow range (10% of TANH range) — restricted
    let mut neuron_records = Vec::new();
    for n in 0..neuron_count {
        let uuid = format!("hidden-{n}");
        let records: Vec<DiscoverRecord> = (0..100)
            .map(|i| DiscoverRecord {
                obs_index: i,
                neuron_uuid: uuid.clone(),
                value: Some(0.2),
                activation: 0.15 + 0.02 * (i as f32 / 100.0), // [0.15, 0.17] — narrow range
                errors: vec![0.01],
            })
            .collect();
        neuron_records.push((uuid, records));
    }

    (creature, neuron_records)
}

/// Create a creature with bounded-range (sentinel) neurons.
fn create_bounded_range_creature(
    neuron_count: usize,
) -> (CreatureJson, Vec<(String, Vec<DiscoverRecord>)>) {
    let mut neurons = Vec::new();

    for n in 0..neuron_count {
        neurons.push(NeuronJson {
            uuid: format!("hidden-{n}"),
            neuron_type: "hidden".to_string(),
            squash: "TANH".to_string(),
            bias: 0.0,
        });
    }

    neurons.push(NeuronJson {
        uuid: "output-0".to_string(),
        neuron_type: "output".to_string(),
        squash: "IDENTITY".to_string(),
        bias: 0.0,
    });

    let creature = CreatureJson {
        neurons,
        synapses: Vec::new(),
        input: neuron_count,
        output: 1,
    };

    // Records with 30% at sentinel -1.0, rest in useful range [0.2, 0.8]
    let mut neuron_records = Vec::new();
    for n in 0..neuron_count {
        let uuid = format!("hidden-{n}");
        let records: Vec<DiscoverRecord> = (0..100)
            .map(|i| {
                let activation = if i < 30 {
                    -1.0 // sentinel
                } else {
                    0.2 + 0.6 * (i as f32 / 100.0) // useful range
                };
                DiscoverRecord {
                    obs_index: i,
                    neuron_uuid: uuid.clone(),
                    value: Some(0.5),
                    activation,
                    errors: vec![0.01],
                }
            })
            .collect();
        neuron_records.push((uuid, records));
    }

    (creature, neuron_records)
}

fn bench_bottleneck_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("bottleneck_clone_reduction");

    for count in [5, 20, 50] {
        let (creature, records) = create_bottleneck_creature(count);

        group.bench_function(format!("detect_{count}_bottlenecks"), |b| {
            b.iter(|| detect_bottleneck_neurons(black_box(&creature), black_box(&records), None));
        });

        let candidates = detect_bottleneck_neurons(&creature, &records, None);
        if !candidates.is_empty() {
            group.bench_function(format!("convert_{count}_bottlenecks"), |b| {
                b.iter(|| {
                    bottleneck_neurons_to_coordinated_candidates(
                        black_box(&candidates),
                        black_box(&creature),
                        None,
                    )
                });
            });
        }
    }

    group.finish();
}

fn bench_restricted_range(c: &mut Criterion) {
    let mut group = c.benchmark_group("restricted_range_clone_reduction");
    let config = RestrictedRangeConfig::default();

    for count in [10, 50, 100] {
        let (creature, records) = create_restricted_range_creature(count);

        group.bench_function(format!("detect_{count}_neurons"), |b| {
            b.iter(|| {
                detect_restricted_range_neurons(
                    black_box(&creature),
                    black_box(&records),
                    black_box(&config),
                )
            });
        });

        let detected = detect_restricted_range_neurons(&creature, &records, &config);
        if !detected.is_empty() {
            group.bench_function(format!("convert_{count}_neurons"), |b| {
                b.iter(|| {
                    restricted_range_to_coordinated_candidates(
                        black_box(&detected),
                        black_box(&creature),
                    )
                });
            });
        }
    }

    group.finish();
}

fn bench_bounded_range(c: &mut Criterion) {
    let mut group = c.benchmark_group("bounded_range_clone_reduction");

    for count in [10, 50, 100] {
        let (creature, records) = create_bounded_range_creature(count);

        group.bench_function(format!("detect_{count}_neurons"), |b| {
            b.iter(|| detect_bounded_range_neurons(black_box(&creature), black_box(&records)));
        });

        let detected = detect_bounded_range_neurons(&creature, &records);
        if !detected.is_empty() {
            group.bench_function(format!("convert_{count}_neurons"), |b| {
                b.iter(|| bounded_range_to_coordinated_candidates(black_box(&detected)));
            });
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_bottleneck_detection,
    bench_restricted_range,
    bench_bounded_range,
);
criterion_main!(benches);
