//! Benchmark for Issue #943: Clone reduction in candidate pipeline.
//!
//! Measures performance of candidate pipeline operations that previously
//! used unnecessary `clone()` calls:
//! - `stratify_samples` (`Vec<f32>` clone for median computation)
//! - Neuron type map construction (String clones vs borrowed &str)
//! - Synapse weight lookup map construction (String pair clones vs borrowed &str)
//!
//! These benchmarks exercise the same code paths optimised to use borrows
//! instead of clones where the data is read-only.

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use criterion::{Criterion, criterion_group, criterion_main};
use neat_ai_discovery::analysis::recommendation::sample_weighted::stratify_samples;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
use std::collections::HashMap;
use std::hint::black_box;

/// Create test records with varying error magnitudes for stratification.
fn create_stratification_records(count: usize) -> Vec<DiscoverRecord> {
    (0..count)
        .map(|i| {
            let error = if i % 3 == 0 {
                0.5 + 0.1 * (i as f32 / count as f32) // high-error
            } else {
                0.01 + 0.02 * (i as f32 / count as f32) // low-error
            };
            DiscoverRecord {
                obs_index: i as u32,
                neuron_uuid: format!("neuron-{}", i % 10),
                value: Some(0.5),
                activation: 0.3 + 0.01 * i as f32,
                errors: vec![error],
            }
        })
        .collect()
}

/// Create a creature with neurons and synapses for map-building benchmarks.
fn create_pipeline_creature(neuron_count: usize) -> CreatureJson {
    let mut neurons = Vec::with_capacity(neuron_count);
    let mut synapses = Vec::with_capacity(neuron_count * 2);

    for n in 0..neuron_count {
        let neuron_type = if n < neuron_count / 2 {
            "hidden"
        } else {
            "output"
        };
        neurons.push(NeuronJson {
            uuid: format!("neuron-{n}"),
            neuron_type: neuron_type.to_string(),
            squash: "TANH".to_string(),
            bias: 0.0,
        });
    }

    // Create synapses between consecutive neurons
    for n in 0..neuron_count.saturating_sub(1) {
        synapses.push(SynapseJson {
            from_uuid: format!("neuron-{n}"),
            to_uuid: format!("neuron-{}", n + 1),
            weight: 0.5,
            synapse_type: None,
        });
    }
    // Add input synapses
    let input_count = neuron_count.min(10);
    for i in 0..input_count {
        synapses.push(SynapseJson {
            from_uuid: format!("input-{i}"),
            to_uuid: format!("neuron-{i}"),
            weight: 0.3,
            synapse_type: None,
        });
    }

    CreatureJson {
        neurons,
        synapses,
        input: input_count,
        output: neuron_count / 2,
    }
}

/// Benchmark `stratify_samples` which computes median via sorting.
fn bench_stratify_samples(c: &mut Criterion) {
    let mut group = c.benchmark_group("candidate_pipeline_stratify");

    for count in [50, 200, 1000] {
        let records = create_stratification_records(count);

        group.bench_function(format!("stratify_{count}_records"), |b| {
            b.iter(|| stratify_samples(black_box(&records)));
        });
    }

    group.finish();
}

/// Benchmark neuron type map construction patterns.
///
/// The post-processing pipeline builds a `neuron_type_map` from creature neurons.
/// This benchmark compares the clone-based approach with the borrow-based approach.
fn bench_neuron_type_map(c: &mut Criterion) {
    let mut group = c.benchmark_group("candidate_pipeline_neuron_type_map");

    for count in [20, 100, 500] {
        let creature = create_pipeline_creature(count);

        // Benchmark: HashMap<String, String> with clones (original pattern)
        group.bench_function(format!("cloned_{count}_neurons"), |b| {
            b.iter(|| {
                let map: HashMap<String, String> = black_box(&creature)
                    .neurons
                    .iter()
                    .map(|n| (n.uuid.clone(), n.neuron_type.clone()))
                    .collect();
                black_box(map)
            });
        });

        // Benchmark: HashMap<&str, &str> with borrows (optimised pattern)
        group.bench_function(format!("borrowed_{count}_neurons"), |b| {
            b.iter(|| {
                let map: HashMap<&str, &str> = black_box(&creature)
                    .neurons
                    .iter()
                    .map(|n| (n.uuid.as_str(), n.neuron_type.as_str()))
                    .collect();
                black_box(map)
            });
        });
    }

    group.finish();
}

/// Benchmark synapse weight lookup map construction patterns.
///
/// The candidate aggregation pipeline builds a `direct_synapse_weight` map
/// using (`from_uuid`, `to_uuid`) pairs as keys.
fn bench_synapse_weight_map(c: &mut Criterion) {
    let mut group = c.benchmark_group("candidate_pipeline_synapse_weight_map");

    for count in [20, 100, 500] {
        let creature = create_pipeline_creature(count);

        // Benchmark: HashMap<(String, String), f32> with clones (original pattern)
        group.bench_function(format!("cloned_{count}_synapses"), |b| {
            b.iter(|| {
                let map: HashMap<(String, String), f32> = black_box(&creature)
                    .synapses
                    .iter()
                    .map(|s| ((s.from_uuid.clone(), s.to_uuid.clone()), s.weight))
                    .collect();
                black_box(map)
            });
        });

        // Benchmark: HashMap<(&str, &str), f32> with borrows (optimised pattern)
        group.bench_function(format!("borrowed_{count}_synapses"), |b| {
            b.iter(|| {
                let map: HashMap<(&str, &str), f32> = black_box(&creature)
                    .synapses
                    .iter()
                    .map(|s| ((s.from_uuid.as_str(), s.to_uuid.as_str()), s.weight))
                    .collect();
                black_box(map)
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_stratify_samples,
    bench_neuron_type_map,
    bench_synapse_weight_map,
);
criterion_main!(benches);
