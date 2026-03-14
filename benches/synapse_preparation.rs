//! Benchmark for Issue #808: String cloning in synapse preparation and candidate generation.
//!
//! Measures the cost of building creature lookup maps and per-target UUID patterns,
//! which are hot paths in the synapse analysis pipeline. These benchmarks simulate
//! the same patterns used in `preparation.rs` and `candidate_generation.rs`.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
use std::collections::{HashMap, HashSet};

/// Create a creature with a given number of hidden neurons and synapses.
fn create_test_creature(hidden_count: usize) -> CreatureJson {
    let input_count = hidden_count;
    let mut neurons = Vec::with_capacity(hidden_count + 1);
    let mut synapses = Vec::with_capacity(hidden_count * 2);

    for h in 0..hidden_count {
        neurons.push(NeuronJson {
            uuid: format!("hidden-{h}"),
            neuron_type: "hidden".to_string(),
            squash: "TANH".to_string(),
            bias: 0.1 * h as f32,
        });
        synapses.push(SynapseJson {
            from_uuid: format!("input-{h}"),
            to_uuid: format!("hidden-{h}"),
            weight: 0.5,
            synapse_type: None,
        });
    }

    neurons.push(NeuronJson {
        uuid: "output-0".to_string(),
        neuron_type: "output".to_string(),
        squash: "IDENTITY".to_string(),
        bias: 0.0,
    });
    for h in 0..hidden_count {
        synapses.push(SynapseJson {
            from_uuid: format!("hidden-{h}"),
            to_uuid: "output-0".to_string(),
            weight: 0.3,
            synapse_type: None,
        });
    }

    CreatureJson {
        neurons,
        synapses,
        input: input_count,
        output: 1,
    }
}

/// Simulate the String-cloning preparation pattern (baseline).
/// This mirrors what `build_creature_lookups` does with owned String keys.
#[allow(clippy::type_complexity)]
fn build_lookup_maps_owned(creature: &CreatureJson) {
    let neuron_squash_map: HashMap<String, String> = creature
        .neurons
        .iter()
        .map(|n| (n.uuid.clone(), n.squash.clone()))
        .collect();
    black_box(&neuron_squash_map);

    let used_inputs: HashSet<String> = creature
        .synapses
        .iter()
        .filter(|s| s.from_uuid.starts_with("input-"))
        .map(|s| s.from_uuid.clone())
        .collect();
    black_box(&used_inputs);

    let neuron_bias_map: HashMap<String, f32> = creature
        .neurons
        .iter()
        .map(|n| (n.uuid.clone(), n.bias))
        .collect();
    black_box(&neuron_bias_map);
}

/// Simulate the borrowed-reference preparation pattern (optimised).
/// This mirrors the optimised `build_creature_lookups` with &str keys.
fn build_lookup_maps_borrowed(creature: &CreatureJson) {
    let neuron_squash_map: HashMap<&str, &str> = creature
        .neurons
        .iter()
        .map(|n| (n.uuid.as_str(), n.squash.as_str()))
        .collect();
    black_box(&neuron_squash_map);

    let used_inputs: HashSet<&str> = creature
        .synapses
        .iter()
        .filter(|s| s.from_uuid.starts_with("input-"))
        .map(|s| s.from_uuid.as_str())
        .collect();
    black_box(&used_inputs);

    let neuron_bias_map: HashMap<&str, f32> = creature
        .neurons
        .iter()
        .map(|n| (n.uuid.as_str(), n.bias))
        .collect();
    black_box(&neuron_bias_map);
}

/// Simulate per-target UUID cloning in sample building (baseline).
fn simulate_per_target_uuid_cloning(creature: &CreatureJson, target_count: usize) {
    for _target in 0..target_count {
        for neuron in &creature.neurons {
            // Simulates the double-clone pattern:
            // 1. build_samples_for_locality_group clones uuid
            let source_uuid: String = neuron.uuid.clone();
            // 2. Consumer clones again for HelpfulWork
            let _work_uuid: String = source_uuid.clone();
            black_box(&_work_uuid);
        }
    }
}

/// Simulate per-target UUID borrowing in sample building (optimised).
fn simulate_per_target_uuid_borrowing(creature: &CreatureJson, target_count: usize) {
    for _target in 0..target_count {
        for neuron in &creature.neurons {
            // Optimised: borrow from source, only allocate once for HelpfulWork
            let source_uuid: &str = neuron.uuid.as_str();
            let _work_uuid: String = source_uuid.to_string();
            black_box(&_work_uuid);
        }
    }
}

fn bench_lookup_map_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("synapse_preparation_lookup_maps");

    for neuron_count in [50, 200, 500] {
        let creature = create_test_creature(neuron_count);

        group.bench_function(format!("owned_{neuron_count}_neurons"), |b| {
            b.iter(|| build_lookup_maps_owned(black_box(&creature)));
        });

        group.bench_function(format!("borrowed_{neuron_count}_neurons"), |b| {
            b.iter(|| build_lookup_maps_borrowed(black_box(&creature)));
        });
    }

    group.finish();
}

fn bench_per_target_uuid_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("synapse_candidate_uuid_cloning");

    for (neuron_count, target_count) in [(50, 20), (200, 50), (500, 100)] {
        let creature = create_test_creature(neuron_count);

        group.bench_function(
            format!("double_clone_{neuron_count}n_{target_count}t"),
            |b| {
                b.iter(|| {
                    simulate_per_target_uuid_cloning(black_box(&creature), target_count);
                });
            },
        );

        group.bench_function(
            format!("borrow_then_clone_{neuron_count}n_{target_count}t"),
            |b| {
                b.iter(|| {
                    simulate_per_target_uuid_borrowing(black_box(&creature), target_count);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_lookup_map_construction,
    bench_per_target_uuid_patterns,
);
criterion_main!(benches);
