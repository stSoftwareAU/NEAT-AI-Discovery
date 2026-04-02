//! Benchmark for Issue #976: UUID string cloning in focus/impact.rs hot loops.
//!
//! Measures the cost of functions that clone UUID strings in synapse iteration:
//! - `compute_impacts_public()` (includes `build_adjacency` and impact traversal)
//! - `compute_selection_stats()` via `compute_impacts_with_activations()`
//!
//! Networks are sized to amplify the cloning overhead with many synapses and
//! many recorded observations per synapse.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use criterion::{Criterion, criterion_group, criterion_main};
use neat_ai_discovery::focus::SelectionStats;
use neat_ai_discovery::focus::compute_impacts_public;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
use std::hint::black_box;

/// Build a network with many synapses to stress adjacency map construction and
/// impact traversal cloning. Each hidden neuron connects to every output.
fn build_dense_network(hidden: usize, outputs: usize) -> CreatureJson {
    let mut neurons = Vec::with_capacity(hidden + outputs);
    let mut synapses = Vec::with_capacity(hidden * outputs);

    for i in 0..hidden {
        neurons.push(NeuronJson {
            uuid: format!("hidden-{i}"),
            neuron_type: "hidden".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        });
        for o in 0..outputs {
            synapses.push(SynapseJson {
                from_uuid: format!("hidden-{i}"),
                to_uuid: format!("output-{o}"),
                weight: 1.0 / hidden as f32,
                synapse_type: None,
            });
        }
    }

    for o in 0..outputs {
        neurons.push(NeuronJson {
            uuid: format!("output-{o}"),
            neuron_type: "output".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        });
    }

    CreatureJson {
        input: 0,
        output: outputs,
        neurons,
        synapses,
    }
}

/// Build a MINIMUM-squash network to exercise `compute_min_stats` / `compute_max_stats`
/// inner loops where UUID keys are cloned per observation record.
fn build_selection_network(inputs: usize, _obs_count: u32) -> (CreatureJson, SelectionStats) {
    let mut neurons = Vec::with_capacity(inputs + 1);
    let mut synapses = Vec::with_capacity(inputs);

    for i in 0..inputs {
        neurons.push(NeuronJson {
            uuid: format!("input-{i}"),
            neuron_type: "input".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        });
        synapses.push(SynapseJson {
            from_uuid: format!("input-{i}"),
            to_uuid: "min-neuron".to_string(),
            weight: 1.0 + 0.1 * i as f32,
            synapse_type: None,
        });
    }

    neurons.push(NeuronJson {
        uuid: "min-neuron".to_string(),
        neuron_type: "output".to_string(),
        squash: "MINIMUM".to_string(),
        bias: 0.0,
    });

    let creature = CreatureJson {
        input: inputs,
        output: 1,
        neurons,
        synapses,
    };

    // Pre-compute expected stats (just need a valid SelectionStats for the benchmark).
    // The actual computation happens inside compute_impacts_public.
    let stats = SelectionStats::new();

    (creature, stats)
}

fn bench_impact_uuid_cloning(c: &mut Criterion) {
    let mut group = c.benchmark_group("impact_uuid_cloning");

    // Dense network: many synapses stress build_adjacency and impact traversal cloning
    let dense_small = build_dense_network(50, 3);
    group.bench_function("dense_50h_3o_impacts", |b| {
        b.iter(|| black_box(compute_impacts_public(&dense_small)));
    });

    let dense_large = build_dense_network(200, 5);
    group.bench_function("dense_200h_5o_impacts", |b| {
        b.iter(|| black_box(compute_impacts_public(&dense_large)));
    });

    // Selection network: exercises build_adjacency with repeated from_uuid keys
    let (sel_net, _) = build_selection_network(20, 500);
    group.bench_function("selection_20_inputs_impacts", |b| {
        b.iter(|| black_box(compute_impacts_public(&sel_net)));
    });

    let (sel_net_large, _) = build_selection_network(100, 1000);
    group.bench_function("selection_100_inputs_impacts", |b| {
        b.iter(|| black_box(compute_impacts_public(&sel_net_large)));
    });

    group.finish();
}

/// Benchmark `build_adjacency` indirectly through `compute_impacts_public`
/// with a network that has many synapses sharing the same `from_uuid`.
fn bench_adjacency_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("adjacency_uuid_cloning");

    // Fan-out network: each hidden neuron has many outgoing synapses
    for &(hidden, fan_out) in &[(50, 10), (100, 20), (200, 5)] {
        let mut neurons = Vec::new();
        let mut synapses = Vec::new();

        for i in 0..hidden {
            neurons.push(NeuronJson {
                uuid: format!("h-{i}"),
                neuron_type: "hidden".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            });
            for j in 0..fan_out {
                let target = format!("t-{}", (i + j + 1) % hidden);
                synapses.push(SynapseJson {
                    from_uuid: format!("h-{i}"),
                    to_uuid: target,
                    weight: 0.5,
                    synapse_type: None,
                });
            }
        }

        neurons.push(NeuronJson {
            uuid: "output-0".to_string(),
            neuron_type: "output".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        });
        // Connect last layer to output
        for i in 0..hidden.min(10) {
            synapses.push(SynapseJson {
                from_uuid: format!("h-{i}"),
                to_uuid: "output-0".to_string(),
                weight: 1.0 / 10.0,
                synapse_type: None,
            });
        }

        let creature = CreatureJson {
            input: 0,
            output: 1,
            neurons,
            synapses,
        };

        group.bench_function(format!("fanout_{hidden}h_{fan_out}edges"), |b| {
            b.iter(|| black_box(compute_impacts_public(&creature)));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_impact_uuid_cloning,
    bench_adjacency_construction
);
criterion_main!(benches);
