//! Benchmark for Issue #754: Pre-computed creature topology cache.
//!
//! Measures the cost of building topology maps (fan-in, fan-out, hidden/output
//! UUID sets, synapse weights) repeatedly per module versus pre-computing them
//! once in a `CreatureTopologyCache` and sharing across modules.
//!
//! Varies creature size from 50 to 500 hidden neurons with proportional
//! synapse counts (3× neurons).

use std::collections::{HashMap, HashSet};

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use neat_ai_discovery::analysis::detection::topology_cache::CreatureTopologyCache;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

/// Number of detection modules that rebuild topology maps per `analyze_all` call.
const MODULE_COUNT: usize = 30;

/// Create a creature with the given number of hidden neurons and ~3× synapses.
fn create_test_creature(hidden_count: usize) -> CreatureJson {
    let input_count = (hidden_count / 5).max(2);
    let output_count = (hidden_count / 10).max(1);

    let mut neurons = Vec::with_capacity(input_count + hidden_count + output_count);
    let mut synapses = Vec::new();

    // Input neurons
    for i in 0..input_count {
        neurons.push(NeuronJson {
            uuid: format!("inp-{i}"),
            neuron_type: "input".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        });
    }

    // Hidden neurons
    for i in 0..hidden_count {
        neurons.push(NeuronJson {
            uuid: format!("hid-{i}"),
            neuron_type: "hidden".to_string(),
            squash: "TANH".to_string(),
            bias: 0.1 * (i % 5) as f32,
        });
    }

    // Output neurons
    for i in 0..output_count {
        neurons.push(NeuronJson {
            uuid: format!("out-{i}"),
            neuron_type: "output".to_string(),
            squash: "LOGISTIC".to_string(),
            bias: 0.0,
        });
    }

    // Synapses: input→hidden, hidden→hidden, hidden→output
    // Target ~3× hidden_count total synapses.
    for i in 0..hidden_count {
        // Each hidden gets at least one input connection
        let inp_idx = i % input_count;
        synapses.push(SynapseJson {
            from_uuid: format!("inp-{inp_idx}"),
            to_uuid: format!("hid-{i}"),
            weight: 0.5 - 0.1 * (i % 10) as f32,
            ..Default::default()
        });

        // Hidden→hidden connections (forward only)
        if i + 1 < hidden_count {
            synapses.push(SynapseJson {
                from_uuid: format!("hid-{i}"),
                to_uuid: format!("hid-{}", i + 1),
                weight: 0.3,
                ..Default::default()
            });
        }

        // Some hidden→output connections
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

/// Simulate what a single detection module does: build topology maps from scratch.
fn build_topology_per_module(creature: &CreatureJson) {
    let _hidden_uuids: HashSet<&str> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "hidden")
        .map(|n| n.uuid.as_str())
        .collect();

    let _output_uuids: HashSet<&str> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "output")
        .map(|n| n.uuid.as_str())
        .collect();

    let mut _fan_in: HashMap<&str, Vec<&str>> = HashMap::new();
    let mut _fan_out: HashMap<&str, Vec<&str>> = HashMap::new();
    for s in &creature.synapses {
        _fan_in
            .entry(s.to_uuid.as_str())
            .or_default()
            .push(s.from_uuid.as_str());
        _fan_out
            .entry(s.from_uuid.as_str())
            .or_default()
            .push(s.to_uuid.as_str());
    }

    let _existing: HashSet<(&str, &str)> = creature
        .synapses
        .iter()
        .map(|s| (s.from_uuid.as_str(), s.to_uuid.as_str()))
        .collect();

    let _weights: HashMap<(&str, &str), f32> = creature
        .synapses
        .iter()
        .map(|s| ((s.from_uuid.as_str(), s.to_uuid.as_str()), s.weight))
        .collect();
}

fn bench_topology_cache(c: &mut Criterion) {
    let sizes: &[usize] = &[50, 100, 200, 500];

    let mut group = c.benchmark_group("topology_cache");

    for &size in sizes {
        let creature = create_test_creature(size);
        let id = format!("{size}_neurons");

        // Baseline: build topology maps MODULE_COUNT times (simulates per-module rebuild)
        group.bench_with_input(
            BenchmarkId::new("per_module_rebuild", &id),
            &creature,
            |b, creature| {
                b.iter(|| {
                    for _ in 0..MODULE_COUNT {
                        build_topology_per_module(black_box(creature));
                    }
                });
            },
        );

        // New: build CreatureTopologyCache once, then do MODULE_COUNT lookups
        group.bench_with_input(
            BenchmarkId::new("shared_cache", &id),
            &creature,
            |b, creature| {
                b.iter(|| {
                    let cache = CreatureTopologyCache::new(black_box(creature));
                    // Simulate MODULE_COUNT modules doing lookups
                    for i in 0..MODULE_COUNT {
                        let uuid = format!("hid-{i}");
                        black_box(cache.fan_in_for(&uuid));
                        black_box(cache.fan_out_for(&uuid));
                        black_box(cache.hidden_uuids.contains(&uuid));
                    }
                });
            },
        );

        // Measure just the cache construction cost
        group.bench_with_input(
            BenchmarkId::new("cache_construction", &id),
            &creature,
            |b, creature| {
                b.iter(|| {
                    black_box(CreatureTopologyCache::new(black_box(creature)));
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_topology_cache);
criterion_main!(benches);
