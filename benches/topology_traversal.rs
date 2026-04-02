//! Benchmark for Issue #979: Backtracking vs `HashSet` clone in topology traversal.
//!
//! Measures the cost of `detect_topology_diversification_candidates` which
//! traverses all paths from outputs back to inputs, counting hidden neuron
//! depth. The optimisation replaces per-recursion `HashSet` clones with a
//! backtracking approach (insert before recurse, remove after return).
//!
//! Uses layered topologies with controlled fan-out to create realistic
//! visited-set pressure without exponential path explosion.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use neat_ai_discovery::analysis::detection::topology_diversification::detect_topology_diversification_candidates;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
use std::hint::black_box;

/// Create a creature with layered topology: inputs → layer1 → layer2 → ... → outputs.
/// Each layer has `width` neurons with fan-out of 2 to the next layer, creating
/// moderate path sharing without exponential blowup.
fn create_layered_creature(depth: usize, width: usize) -> CreatureJson {
    let input_count = 3;
    let output_count = 2;

    let mut neurons = Vec::new();
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

    // Hidden layers
    for layer in 0..depth {
        for w in 0..width {
            let idx = layer * width + w;
            neurons.push(NeuronJson {
                uuid: format!("hid-{idx}"),
                neuron_type: "hidden".to_string(),
                squash: "TANH".to_string(),
                bias: 0.0,
            });
        }
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

    // Wire inputs to first hidden layer
    for w in 0..width {
        let inp_idx = w % input_count;
        synapses.push(SynapseJson {
            from_uuid: format!("inp-{inp_idx}"),
            to_uuid: format!("hid-{w}"),
            weight: 0.5,
            ..Default::default()
        });
    }

    // Wire between hidden layers: each neuron connects to 2 neurons in the next layer
    for layer in 0..depth.saturating_sub(1) {
        for w in 0..width {
            let from_idx = layer * width + w;
            // Connect to same position and next position (mod width) in next layer
            for &target_w in &[w, (w + 1) % width] {
                let to_idx = (layer + 1) * width + target_w;
                synapses.push(SynapseJson {
                    from_uuid: format!("hid-{from_idx}"),
                    to_uuid: format!("hid-{to_idx}"),
                    weight: 0.3,
                    ..Default::default()
                });
            }
        }
    }

    // Wire last hidden layer to outputs
    let last_layer_start = (depth - 1) * width;
    for w in 0..width {
        let from_idx = last_layer_start + w;
        let out_idx = w % output_count;
        synapses.push(SynapseJson {
            from_uuid: format!("hid-{from_idx}"),
            to_uuid: format!("out-{out_idx}"),
            weight: 0.6,
            ..Default::default()
        });
    }

    // Add a direct input→output path to trigger diversification analysis
    synapses.push(SynapseJson {
        from_uuid: "inp-0".to_string(),
        to_uuid: "out-0".to_string(),
        weight: 0.1,
        ..Default::default()
    });

    CreatureJson {
        neurons,
        synapses,
        input: input_count,
        output: output_count,
    }
}

/// Create records for the given creature with enough samples to pass thresholds.
fn create_records(creature: &CreatureJson) -> Vec<(String, Vec<DiscoverRecord>)> {
    let sample_count = 40_u32;

    creature
        .neurons
        .iter()
        .map(|n| {
            let error = if n.neuron_type == "output" {
                0.25
            } else {
                0.02
            };
            let records: Vec<DiscoverRecord> = (0..sample_count)
                .map(|i| DiscoverRecord {
                    obs_index: i,
                    neuron_uuid: n.uuid.clone(),
                    value: Some(0.5 + (i as f32 * 0.01)),
                    activation: 0.5 + (i as f32 * 0.01),
                    errors: vec![error],
                })
                .collect();
            (n.uuid.clone(), records)
        })
        .collect()
}

fn bench_topology_traversal(c: &mut Criterion) {
    // (depth, width) combinations — total hidden neurons = depth × width
    let configs: &[(usize, usize)] = &[
        (3, 4),   // 12 hidden neurons
        (5, 4),   // 20 hidden neurons
        (5, 8),   // 40 hidden neurons
        (8, 8),   // 64 hidden neurons
        (10, 10), // 100 hidden neurons
    ];

    let mut group = c.benchmark_group("topology_traversal");

    for &(depth, width) in configs {
        let total = depth * width;
        let creature = create_layered_creature(depth, width);
        let records = create_records(&creature);
        let id = format!("{total}_hidden_{depth}d_{width}w");

        group.bench_with_input(
            BenchmarkId::new("detect_diversification", &id),
            &(&creature, &records),
            |b, &(creature, records)| {
                b.iter(|| {
                    black_box(detect_topology_diversification_candidates(
                        black_box(creature),
                        black_box(records),
                    ));
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_topology_traversal);
criterion_main!(benches);
