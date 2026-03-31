//! Benchmark for Issue #208: Pre-compute synapse counts performance.
//!
//! This benchmark compares the performance of:
//! - Old approach: O(n×m) - scanning all synapses for each neuron lookup
//! - New approach: O(n+m) - pre-compute `HashMaps`, then O(1) lookup
//!
//! Expected improvement: ~1000x for large creatures (500 neurons, 10,000 synapses).

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use neat_ai_discovery::focus::SynapseCounts;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
use std::hint::black_box;

/// Create a test creature with specified number of neurons and synapses.
///
/// The synapse connectivity is randomised but deterministic (based on indices).
fn create_test_creature(num_neurons: usize, num_synapses: usize) -> CreatureJson {
    // Create neurons: 1 input, (num_neurons - 2) hidden, 1 output
    let mut neurons = Vec::with_capacity(num_neurons);

    // Hidden neurons (input neurons are not in the neurons list per NEAT-AI model)
    for i in 0..(num_neurons - 1) {
        neurons.push(NeuronJson {
            uuid: format!("hidden-{i}"),
            neuron_type: "hidden".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        });
    }

    // Output neuron
    neurons.push(NeuronJson {
        uuid: "output-0".to_string(),
        neuron_type: "output".to_string(),
        squash: "IDENTITY".to_string(),
        bias: 0.0,
    });

    // Create synapses with pseudo-random connectivity
    let mut synapses = Vec::with_capacity(num_synapses);
    for i in 0..num_synapses {
        // Deterministic "random" from/to selection based on index
        let from_idx = i % num_neurons;
        let to_idx = (i * 7 + 3) % num_neurons;

        let from_uuid = if from_idx == 0 {
            "input-0".to_string()
        } else if from_idx == num_neurons - 1 {
            "output-0".to_string()
        } else {
            format!("hidden-{}", from_idx - 1)
        };

        let to_uuid = if to_idx == num_neurons - 1 {
            "output-0".to_string()
        } else {
            format!("hidden-{to_idx}")
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
        input: 1,
        output: 1,
    }
}

/// Simulate the OLD approach: O(m) scan for each neuron lookup.
/// This is what `count_synapses_for_neuron` used to do.
fn old_count_synapses_for_neuron(neuron_uuid: &str, creature: &CreatureJson) -> (usize, usize) {
    let incoming = creature
        .synapses
        .iter()
        .filter(|s| s.to_uuid == neuron_uuid)
        .count();
    let outgoing = creature
        .synapses
        .iter()
        .filter(|s| s.from_uuid == neuron_uuid)
        .count();
    (incoming, outgoing)
}

/// Benchmark: Count synapses for ALL neurons using the old O(n×m) approach.
fn bench_old_approach(creature: &CreatureJson) -> Vec<(usize, usize)> {
    let all_uuids: Vec<&str> = creature
        .neurons
        .iter()
        .map(|n| n.uuid.as_str())
        .chain(std::iter::once("input-0"))
        .collect();

    all_uuids
        .iter()
        .map(|uuid| old_count_synapses_for_neuron(uuid, creature))
        .collect()
}

/// Benchmark: Count synapses for ALL neurons using the new O(n+m) approach.
fn bench_new_approach(creature: &CreatureJson) -> Vec<(usize, usize)> {
    let synapse_counts = SynapseCounts::new(creature);

    let all_uuids: Vec<&str> = creature
        .neurons
        .iter()
        .map(|n| n.uuid.as_str())
        .chain(std::iter::once("input-0"))
        .collect();

    all_uuids
        .iter()
        .map(|uuid| synapse_counts.get(uuid))
        .collect()
}

fn bench_synapse_counts(c: &mut Criterion) {
    let mut group = c.benchmark_group("synapse_counts");

    // Test different creature sizes as specified in the issue
    let sizes: Vec<(usize, usize)> = vec![
        (100, 500),     // Small: 100 neurons, 500 synapses
        (500, 10_000),  // Medium: 500 neurons, 10,000 synapses (issue example)
        (1000, 50_000), // Large: 1000 neurons, 50,000 synapses
    ];

    for (num_neurons, num_synapses) in sizes {
        let creature = create_test_creature(num_neurons, num_synapses);
        let id = format!("{num_neurons}n_{num_synapses}s");

        // Benchmark old approach
        group.bench_with_input(BenchmarkId::new("old_O(n×m)", &id), &creature, |b, c| {
            b.iter(|| bench_old_approach(black_box(c)));
        });

        // Benchmark new approach
        group.bench_with_input(BenchmarkId::new("new_O(n+m)", &id), &creature, |b, c| {
            b.iter(|| bench_new_approach(black_box(c)));
        });
    }

    group.finish();
}

criterion_group!(benches, bench_synapse_counts);
criterion_main!(benches);
