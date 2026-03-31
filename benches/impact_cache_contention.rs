//! Benchmark for Issue #835: Lock contention on shared cache in impact computation.
//!
//! Measures the cost of `compute_impacts_public()` with networks of varying size
//! and depth. The key optimisation replaces Mutex<HashMap> with `DashMap` to
//! reduce lock contention during parallel recursive impact computation.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use criterion::{Criterion, criterion_group, criterion_main};
use neat_ai_discovery::focus::compute_impacts_public;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
use std::hint::black_box;

/// Build a wide network: many hidden neurons each connecting to the output.
fn build_wide_network(width: usize) -> CreatureJson {
    let mut neurons = Vec::with_capacity(width + 1);
    let mut synapses = Vec::with_capacity(width);

    for i in 0..width {
        neurons.push(NeuronJson {
            uuid: format!("hidden-{i}"),
            neuron_type: "hidden".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        });
        synapses.push(SynapseJson {
            from_uuid: format!("hidden-{i}"),
            to_uuid: "output-0".to_string(),
            weight: 1.0 / width as f32,
            synapse_type: None,
        });
    }

    neurons.push(NeuronJson {
        uuid: "output-0".to_string(),
        neuron_type: "output".to_string(),
        squash: "IDENTITY".to_string(),
        bias: 0.0,
    });

    CreatureJson {
        input: 0,
        output: 1,
        neurons,
        synapses,
    }
}

/// Build a deep chain network: input -> h0 -> h1 -> ... -> hN -> output.
fn build_deep_network(depth: usize) -> CreatureJson {
    let mut neurons = Vec::with_capacity(depth + 1);
    let mut synapses = Vec::with_capacity(depth);

    for i in 0..depth {
        neurons.push(NeuronJson {
            uuid: format!("hidden-{i}"),
            neuron_type: "hidden".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        });
        if i > 0 {
            synapses.push(SynapseJson {
                from_uuid: format!("hidden-{}", i - 1),
                to_uuid: format!("hidden-{i}"),
                weight: 0.9,
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

    synapses.push(SynapseJson {
        from_uuid: format!("hidden-{}", depth - 1),
        to_uuid: "output-0".to_string(),
        weight: 1.0,
        synapse_type: None,
    });

    CreatureJson {
        input: 0,
        output: 1,
        neurons,
        synapses,
    }
}

/// Build a diamond mesh: layers of neurons with cross-connections.
fn build_diamond_mesh(layers: usize, width: usize) -> CreatureJson {
    let mut neurons = Vec::new();
    let mut synapses = Vec::new();

    for layer in 0..layers {
        for i in 0..width {
            neurons.push(NeuronJson {
                uuid: format!("h-{layer}-{i}"),
                neuron_type: "hidden".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            });

            if layer > 0 {
                // Connect from all neurons in previous layer
                for j in 0..width {
                    synapses.push(SynapseJson {
                        from_uuid: format!("h-{}-{j}", layer - 1),
                        to_uuid: format!("h-{layer}-{i}"),
                        weight: 1.0 / width as f32,
                        synapse_type: None,
                    });
                }
            }
        }
    }

    // Output neuron connected from last layer
    neurons.push(NeuronJson {
        uuid: "output-0".to_string(),
        neuron_type: "output".to_string(),
        squash: "IDENTITY".to_string(),
        bias: 0.0,
    });
    for i in 0..width {
        synapses.push(SynapseJson {
            from_uuid: format!("h-{}-{i}", layers - 1),
            to_uuid: "output-0".to_string(),
            weight: 1.0 / width as f32,
            synapse_type: None,
        });
    }

    CreatureJson {
        input: 0,
        output: 1,
        neurons,
        synapses,
    }
}

fn bench_impact_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("impact_cache_contention");

    // Wide network: many parallel threads competing for the cache
    let wide = build_wide_network(200);
    group.bench_function("wide_200_neurons", |b| {
        b.iter(|| black_box(compute_impacts_public(&wide)));
    });

    // Deep network: long recursive chains with cache lookups
    let deep = build_deep_network(100);
    group.bench_function("deep_100_layers", |b| {
        b.iter(|| black_box(compute_impacts_public(&deep)));
    });

    // Diamond mesh: combines width and depth for maximum contention
    let mesh = build_diamond_mesh(10, 20);
    group.bench_function("mesh_10x20", |b| {
        b.iter(|| black_box(compute_impacts_public(&mesh)));
    });

    group.finish();
}

criterion_group!(benches, bench_impact_cache);
criterion_main!(benches);
