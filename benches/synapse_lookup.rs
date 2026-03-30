//! Benchmark for Issue #769: Zero-allocation synapse lookups.
//!
//! Measures the cost of `synapse_exists()` and `synapse_weight()` calls
//! on `CreatureTopologyCache`, which are called in hot loops in
//! `bottleneck.rs` and `topology.rs`.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use neat_ai_discovery::analysis::detection::topology_cache::CreatureTopologyCache;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
use std::hint::black_box;

/// Create a creature with the given number of hidden neurons and ~3× synapses.
fn create_test_creature(hidden_count: usize) -> CreatureJson {
    let input_count = (hidden_count / 5).max(2);
    let output_count = (hidden_count / 10).max(1);

    let mut neurons = Vec::with_capacity(input_count + hidden_count + output_count);
    let mut synapses = Vec::new();

    for i in 0..input_count {
        neurons.push(NeuronJson {
            uuid: format!("inp-{i}"),
            neuron_type: "input".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        });
    }

    for i in 0..hidden_count {
        neurons.push(NeuronJson {
            uuid: format!("hid-{i}"),
            neuron_type: "hidden".to_string(),
            squash: "TANH".to_string(),
            bias: 0.1 * (i % 5) as f32,
        });
    }

    for i in 0..output_count {
        neurons.push(NeuronJson {
            uuid: format!("out-{i}"),
            neuron_type: "output".to_string(),
            squash: "LOGISTIC".to_string(),
            bias: 0.0,
        });
    }

    for i in 0..hidden_count {
        let inp_idx = i % input_count;
        synapses.push(SynapseJson {
            from_uuid: format!("inp-{inp_idx}"),
            to_uuid: format!("hid-{i}"),
            weight: 0.5 - 0.1 * (i % 10) as f32,
            ..Default::default()
        });

        if i + 1 < hidden_count {
            synapses.push(SynapseJson {
                from_uuid: format!("hid-{i}"),
                to_uuid: format!("hid-{}", i + 1),
                weight: 0.3,
                ..Default::default()
            });
        }

        if i % 3 == 0 {
            let out_idx = i % output_count;
            synapses.push(SynapseJson {
                from_uuid: format!("hid-{i}"),
                to_uuid: format!("out-{out_idx}"),
                weight: 0.7,
                ..Default::default()
            });
        }
    }

    CreatureJson {
        neurons,
        synapses,
        input: input_count,
        output: output_count,
    }
}

fn bench_synapse_lookup(c: &mut Criterion) {
    let sizes: &[usize] = &[50, 200, 500];

    let mut group = c.benchmark_group("synapse_lookup");

    for &size in sizes {
        let creature = create_test_creature(size);
        let cache = CreatureTopologyCache::new(&creature);
        let id = format!("{size}_neurons");

        // Benchmark synapse_exists in a loop (simulates hot-path usage)
        group.bench_with_input(
            BenchmarkId::new("synapse_exists", &id),
            &cache,
            |b, cache| {
                b.iter(|| {
                    for i in 0..size {
                        let from = format!("hid-{i}");
                        let to = format!("hid-{}", (i + 1) % size);
                        black_box(cache.synapse_exists(&from, &to));
                    }
                });
            },
        );

        // Benchmark synapse_weight in a loop
        group.bench_with_input(
            BenchmarkId::new("synapse_weight", &id),
            &cache,
            |b, cache| {
                b.iter(|| {
                    for i in 0..size {
                        let from = format!("hid-{i}");
                        let to = format!("hid-{}", (i + 1) % size);
                        black_box(cache.synapse_weight(&from, &to));
                    }
                });
            },
        );

        // Benchmark with pre-allocated strings (shows overhead of method itself)
        let lookup_pairs: Vec<(String, String)> = (0..size)
            .map(|i| (format!("hid-{i}"), format!("hid-{}", (i + 1) % size)))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("synapse_exists_prealloc", &id),
            &(&cache, &lookup_pairs),
            |b, (cache, pairs)| {
                b.iter(|| {
                    for (from, to) in *pairs {
                        black_box(cache.synapse_exists(from, to));
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("synapse_weight_prealloc", &id),
            &(&cache, &lookup_pairs),
            |b, (cache, pairs)| {
                b.iter(|| {
                    for (from, to) in *pairs {
                        black_box(cache.synapse_weight(from, to));
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_synapse_lookup);
criterion_main!(benches);
