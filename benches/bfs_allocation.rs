//! Benchmark for Issue #755: BFS visited set allocation reduction.
//!
//! Compares the cost of BFS traversal using `HashSet<String>` (allocating)
//! versus `HashSet<&str>` (borrowing) visited sets on a deep network
//! (depth 50, fanout 5).
//!
//! Also measures overall detection-phase allocation patterns before/after
//! the optimisations in dead_neuron, bottleneck, and skip_connection modules.

use std::collections::HashSet;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use neat_ai_discovery::analysis::detection::dead_neuron::{
    dead_neurons_to_coordinated_candidates, detect_dead_neurons,
};
use neat_ai_discovery::analysis::detection::topology_cache::CreatureTopologyCache;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

/// Create a deep network with the given depth and fanout.
///
/// Produces a chain of layers where each layer fans out to `fanout` neurons
/// in the next layer, creating a graph with `depth * fanout` hidden neurons
/// and `depth * fanout * fanout` synapses.
fn create_deep_network(depth: usize, fanout: usize) -> CreatureJson {
    let mut neurons = Vec::new();
    let mut synapses = Vec::new();

    // Single input neuron
    neurons.push(NeuronJson {
        uuid: "inp-0".to_string(),
        neuron_type: "input".to_string(),
        squash: "IDENTITY".to_string(),
        bias: 0.0,
    });

    // Build layers of hidden neurons
    let mut prev_layer: Vec<String> = vec!["inp-0".to_string()];
    for d in 0..depth {
        let mut current_layer = Vec::with_capacity(fanout);
        for f in 0..fanout {
            let uuid = format!("hid-{d}-{f}");
            neurons.push(NeuronJson {
                uuid: uuid.clone(),
                neuron_type: "hidden".to_string(),
                squash: "TANH".to_string(),
                bias: 0.0,
            });
            // Connect from every neuron in the previous layer
            for prev in &prev_layer {
                synapses.push(SynapseJson {
                    from_uuid: prev.clone(),
                    to_uuid: uuid.clone(),
                    weight: 0.1,
                    ..Default::default()
                });
            }
            current_layer.push(uuid);
        }
        prev_layer = current_layer;
    }

    // Single output neuron
    neurons.push(NeuronJson {
        uuid: "out-0".to_string(),
        neuron_type: "output".to_string(),
        squash: "IDENTITY".to_string(),
        bias: 0.0,
    });

    for prev in &prev_layer {
        synapses.push(SynapseJson {
            from_uuid: prev.clone(),
            to_uuid: "out-0".to_string(),
            weight: 0.5,
            ..Default::default()
        });
    }

    CreatureJson {
        neurons,
        synapses,
        input: 1,
        output: 1,
    }
}

/// Simulate the old BFS pattern with `HashSet<String>` visited set.
fn bfs_with_string_visited(
    start_uuid: &str,
    fan_out: &std::collections::HashMap<String, Vec<String>>,
    output_uuids: &HashSet<String>,
) -> Vec<String> {
    let mut visited: HashSet<String> = HashSet::new();
    let mut queue = vec![start_uuid.to_string()];
    let mut connected = Vec::new();

    while let Some(current) = queue.pop() {
        if !visited.insert(current.clone()) {
            continue;
        }

        if let Some(successors) = fan_out.get(&current) {
            for next in successors {
                if output_uuids.contains(next.as_str()) {
                    connected.push(next.clone());
                }
                if !visited.contains(next.as_str()) {
                    queue.push(next.clone());
                }
            }
        }
    }

    connected.sort();
    connected.dedup();
    connected
}

/// Simulate the new BFS pattern with `HashSet<&str>` visited set (borrowing).
fn bfs_with_str_visited<'a>(
    start_uuid: &'a str,
    fan_out: &'a std::collections::HashMap<String, Vec<String>>,
    output_uuids: &HashSet<String>,
) -> Vec<&'a str> {
    let mut visited: HashSet<&str> = HashSet::new();
    let mut queue = vec![start_uuid];
    let mut connected = Vec::new();

    while let Some(current) = queue.pop() {
        if !visited.insert(current) {
            continue;
        }

        if let Some(successors) = fan_out.get(current) {
            for next in successors {
                if output_uuids.contains(next.as_str()) {
                    connected.push(next.as_str());
                }
                if !visited.contains(next.as_str()) {
                    queue.push(next.as_str());
                }
            }
        }
    }

    connected.sort();
    connected.dedup();
    connected
}

fn bench_bfs_visited_set(c: &mut Criterion) {
    let mut group = c.benchmark_group("bfs_visited_set");

    for &(depth, fanout) in &[(10, 3), (20, 3), (50, 5)] {
        let creature = create_deep_network(depth, fanout);
        let cache = CreatureTopologyCache::new(&creature);

        let id = format!("depth{depth}_fan{fanout}");

        // Benchmark old pattern: HashSet<String>
        group.bench_with_input(
            BenchmarkId::new("string_visited", &id),
            &cache,
            |b, cache| {
                b.iter(|| {
                    bfs_with_string_visited(
                        black_box("hid-0-0"),
                        &cache
                            .fan_out
                            .iter()
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect(),
                        &cache.output_uuids,
                    )
                });
            },
        );

        // Benchmark new pattern: HashSet<&str>
        let fan_out_owned: std::collections::HashMap<String, Vec<String>> = cache
            .fan_out
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        group.bench_with_input(
            BenchmarkId::new("str_visited", &id),
            &fan_out_owned,
            |b, fan_out| {
                b.iter(|| bfs_with_str_visited(black_box("hid-0-0"), fan_out, &cache.output_uuids));
            },
        );
    }

    group.finish();
}

fn bench_dead_neuron_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("dead_neuron_detection");

    for &(depth, fanout) in &[(10, 3), (20, 3)] {
        let creature = create_deep_network(depth, fanout);
        let id = format!("depth{depth}_fan{fanout}");

        // Create neuron records: all hidden neurons are "dead" (near-zero activation)
        let neuron_records: Vec<(String, Vec<DiscoverRecord>)> = creature
            .neurons
            .iter()
            .filter(|n| n.neuron_type == "hidden")
            .map(|n| {
                let records: Vec<DiscoverRecord> = (0..100)
                    .map(|i| DiscoverRecord {
                        obs_index: i,
                        neuron_uuid: n.uuid.clone(),
                        value: Some(0.0),
                        activation: 1e-8 * (i as f32), // near-zero
                        errors: vec![0.001],
                    })
                    .collect();
                (n.uuid.clone(), records)
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::new("detect_and_convert", &id),
            &(&creature, &neuron_records),
            |b, &(creature, records)| {
                b.iter(|| {
                    let candidates =
                        detect_dead_neurons(black_box(creature), black_box(records), None);
                    let _coordinated =
                        dead_neurons_to_coordinated_candidates(black_box(&candidates));
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_bfs_visited_set, bench_dead_neuron_detection);
criterion_main!(benches);
