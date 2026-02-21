//! Benchmark for Issue #210: Neuron UUID string interning.
//!
//! This benchmark compares the performance and memory characteristics of:
//! - Old approach: String-based HashMap/HashSet keys (cloning UUIDs)
//! - New approach: u32-indexed keys using NeuronIndex interning
//!
//! Expected improvements:
//! - ~89% memory reduction for synapse pair collections
//! - Faster HashMap/HashSet operations due to cheaper key comparison

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use neat_ai_discovery::intern::NeuronIndex;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
use std::collections::{HashMap, HashSet};

/// Create a test creature with specified number of neurons and synapses.
fn create_test_creature(
    num_neurons: usize,
    num_synapses: usize,
    num_inputs: usize,
) -> CreatureJson {
    let mut neurons = Vec::with_capacity(num_neurons);

    // Hidden neurons
    for i in 0..(num_neurons - num_neurons / 4) {
        neurons.push(NeuronJson {
            uuid: format!("hidden-{i}"),
            neuron_type: "hidden".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        });
    }

    // Output neurons
    for i in 0..(num_neurons / 4) {
        neurons.push(NeuronJson {
            uuid: format!("output-{i}"),
            neuron_type: "output".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        });
    }

    // Create synapses with varied connectivity
    let mut synapses = Vec::with_capacity(num_synapses);
    for i in 0..num_synapses {
        let from_uuid = if i % 3 == 0 {
            format!("input-{}", i % num_inputs)
        } else {
            format!("hidden-{}", i % (num_neurons - num_neurons / 4))
        };

        let to_idx = (i * 7 + 3) % num_neurons;
        let to_uuid = if to_idx < num_neurons - num_neurons / 4 {
            format!("hidden-{to_idx}")
        } else {
            format!("output-{}", to_idx - (num_neurons - num_neurons / 4))
        };

        synapses.push(SynapseJson {
            from_uuid,
            to_uuid,
            weight: 0.5,
            synapse_type: None,
        });
    }

    CreatureJson {
        neurons,
        synapses,
        input: num_inputs,
        output: num_neurons / 4,
    }
}

/// OLD approach: Build existing_synapses HashSet using String keys.
fn build_existing_synapses_string(creature: &CreatureJson) -> HashSet<(String, String)> {
    creature
        .synapses
        .iter()
        .map(|s| (s.from_uuid.clone(), s.to_uuid.clone()))
        .collect()
}

/// NEW approach: Build existing_synapses HashSet using interned u32 keys.
fn build_existing_synapses_interned(creature: &CreatureJson) -> (HashSet<(u32, u32)>, NeuronIndex) {
    let mut index = NeuronIndex::with_capacity(creature.neurons.len() + creature.input);

    // Pre-intern all UUIDs
    for i in 0..creature.input {
        index.intern(&format!("input-{i}"));
    }
    for neuron in &creature.neurons {
        index.intern(&neuron.uuid);
    }

    let set: HashSet<(u32, u32)> = creature
        .synapses
        .iter()
        .map(|s| (index.intern(&s.from_uuid), index.intern(&s.to_uuid)))
        .collect();

    (set, index)
}

/// OLD approach: Build synapse weights HashMap using String keys.
fn build_synapse_weights_string(creature: &CreatureJson) -> HashMap<(String, String), f32> {
    creature
        .synapses
        .iter()
        .map(|s| ((s.from_uuid.clone(), s.to_uuid.clone()), s.weight))
        .collect()
}

/// NEW approach: Build synapse weights HashMap using interned u32 keys.
fn build_synapse_weights_interned(
    creature: &CreatureJson,
) -> (HashMap<(u32, u32), f32>, NeuronIndex) {
    let mut index = NeuronIndex::with_capacity(creature.neurons.len() + creature.input);

    // Pre-intern all UUIDs
    for i in 0..creature.input {
        index.intern(&format!("input-{i}"));
    }
    for neuron in &creature.neurons {
        index.intern(&neuron.uuid);
    }

    let map: HashMap<(u32, u32), f32> = creature
        .synapses
        .iter()
        .map(|s| {
            (
                (index.intern(&s.from_uuid), index.intern(&s.to_uuid)),
                s.weight,
            )
        })
        .collect();

    (map, index)
}

/// OLD approach: Lookup synapse existence using String keys.
fn lookup_synapses_string(set: &HashSet<(String, String)>, queries: &[(String, String)]) -> usize {
    queries.iter().filter(|q| set.contains(q)).count()
}

/// NEW approach: Lookup synapse existence using interned keys.
fn lookup_synapses_interned(
    set: &HashSet<(u32, u32)>,
    index: &NeuronIndex,
    queries: &[(String, String)],
) -> usize {
    queries
        .iter()
        .filter(
            |(from, to)| match (index.get_index(from), index.get_index(to)) {
                (Some(from_idx), Some(to_idx)) => set.contains(&(from_idx, to_idx)),
                _ => false,
            },
        )
        .count()
}

fn bench_build_existing_synapses(c: &mut Criterion) {
    let mut group = c.benchmark_group("build_existing_synapses");

    let sizes: Vec<(usize, usize, usize)> = vec![
        (100, 500, 10),      // Small
        (500, 10_000, 50),   // Medium (issue example)
        (1000, 50_000, 100), // Large
    ];

    for (num_neurons, num_synapses, num_inputs) in sizes {
        let creature = create_test_creature(num_neurons, num_synapses, num_inputs);
        let id = format!("{num_neurons}n_{num_synapses}s");

        group.bench_with_input(BenchmarkId::new("string_keys", &id), &creature, |b, c| {
            b.iter(|| build_existing_synapses_string(black_box(c)));
        });

        group.bench_with_input(BenchmarkId::new("interned_keys", &id), &creature, |b, c| {
            b.iter(|| build_existing_synapses_interned(black_box(c)));
        });
    }

    group.finish();
}

fn bench_build_synapse_weights(c: &mut Criterion) {
    let mut group = c.benchmark_group("build_synapse_weights");

    let sizes: Vec<(usize, usize, usize)> =
        vec![(100, 500, 10), (500, 10_000, 50), (1000, 50_000, 100)];

    for (num_neurons, num_synapses, num_inputs) in sizes {
        let creature = create_test_creature(num_neurons, num_synapses, num_inputs);
        let id = format!("{num_neurons}n_{num_synapses}s");

        group.bench_with_input(BenchmarkId::new("string_keys", &id), &creature, |b, c| {
            b.iter(|| build_synapse_weights_string(black_box(c)));
        });

        group.bench_with_input(BenchmarkId::new("interned_keys", &id), &creature, |b, c| {
            b.iter(|| build_synapse_weights_interned(black_box(c)));
        });
    }

    group.finish();
}

fn bench_lookup_synapses(c: &mut Criterion) {
    let mut group = c.benchmark_group("lookup_synapses");

    // Create a medium-sized creature
    let creature = create_test_creature(500, 10_000, 50);

    // Build both data structures
    let string_set = build_existing_synapses_string(&creature);
    let (interned_set, index) = build_existing_synapses_interned(&creature);

    // Create lookup queries (mix of existing and non-existing)
    let queries: Vec<(String, String)> = (0..1000)
        .map(|i| {
            if i % 2 == 0 {
                // Existing synapse
                let s = &creature.synapses[i % creature.synapses.len()];
                (s.from_uuid.clone(), s.to_uuid.clone())
            } else {
                // Non-existing synapse
                (format!("hidden-{i}"), format!("output-{}", i % 10))
            }
        })
        .collect();

    group.bench_function("string_keys_1000_lookups", |b| {
        b.iter(|| lookup_synapses_string(black_box(&string_set), black_box(&queries)));
    });

    group.bench_function("interned_keys_1000_lookups", |b| {
        b.iter(|| {
            lookup_synapses_interned(
                black_box(&interned_set),
                black_box(&index),
                black_box(&queries),
            )
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_build_existing_synapses,
    bench_build_synapse_weights,
    bench_lookup_synapses
);
criterion_main!(benches);
