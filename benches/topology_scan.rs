//! Issue #2369 detector benchmarks for long paths and starved neurons.

#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;
use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use neat_ai_discovery::analysis::detection::topology::detect_topology_issues;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

fn neuron(uuid: String, neuron_type: &str) -> NeuronJson {
    NeuronJson {
        uuid,
        neuron_type: neuron_type.to_string(),
        squash: "IDENTITY".to_string(),
        bias: 0.0,
    }
}

fn edge(from: String, to: String) -> SynapseJson {
    SynapseJson {
        from_uuid: from,
        to_uuid: to,
        weight: 0.5,
        synapse_type: None,
    }
}

fn assert_forward_only(creature: &CreatureJson) {
    let indices: HashMap<_, _> = creature
        .neurons
        .iter()
        .enumerate()
        .map(|(index, neuron)| (neuron.uuid.as_str(), index))
        .collect();
    for synapse in &creature.synapses {
        assert!(
            matches!(
                (
                    indices.get(synapse.from_uuid.as_str()),
                    indices.get(synapse.to_uuid.as_str())
                ),
                (Some(from), Some(to)) if from < to
            ),
            "benchmark edge {} -> {} must follow neuron evaluation order",
            synapse.from_uuid,
            synapse.to_uuid
        );
    }
}

fn long_path(
    hidden_count: usize,
    dense: bool,
) -> (CreatureJson, Vec<(String, Vec<DiscoverRecord>)>) {
    let width = if dense { (hidden_count / 5).max(2) } else { 1 };
    let layers = (hidden_count / width).max(5);
    let actual_hidden = layers * width;
    let output_count = (hidden_count / 4).max(1);
    let mut neurons = Vec::with_capacity(actual_hidden + output_count);
    let mut synapses = Vec::new();
    for layer in 0..layers {
        for index in 0..width {
            neurons.push(neuron(format!("h-{layer}-{index}"), "hidden"));
        }
    }
    for index in 0..output_count {
        neurons.push(neuron(format!("o-{index}"), "output"));
    }
    for layer in 0..layers.saturating_sub(1) {
        for from in 0..width {
            for to in 0..if dense { width } else { 1 } {
                synapses.push(edge(
                    format!("h-{layer}-{from}"),
                    format!("h-{}-{to}", layer + 1),
                ));
            }
        }
    }
    for index in 0..width {
        synapses.push(edge(
            format!("h-{}-{index}", layers - 1),
            format!("o-{}", index % output_count),
        ));
    }
    let records = neurons
        .iter()
        .filter(|n| n.neuron_type == "hidden")
        .map(|n| {
            let rows = (0..20)
                .map(|i| DiscoverRecord::new(i, n.uuid.clone(), Some(0.5), 0.5, vec![0.2]))
                .collect();
            (n.uuid.clone(), rows)
        })
        .collect();
    (
        CreatureJson {
            neurons,
            synapses,
            input: 0,
            output: output_count,
        },
        records,
    )
}

fn starved(
    hidden_count: usize,
    input_count: usize,
    dense: bool,
) -> (CreatureJson, Vec<(String, Vec<DiscoverRecord>)>) {
    let mut neurons = Vec::with_capacity(input_count + hidden_count + 1);
    let mut synapses = Vec::new();
    for i in 0..input_count {
        neurons.push(neuron(format!("i-{i}"), "input"));
    }
    let hub_index = hidden_count.saturating_sub(1);
    for i in 0..hidden_count {
        neurons.push(neuron(format!("h-{i}"), "hidden"));
    }
    neurons.push(neuron("o-0".to_string(), "output"));

    let connected_inputs = if dense {
        input_count.saturating_sub(1)
    } else {
        1
    };
    for i in 0..hub_index {
        for source in 0..connected_inputs {
            synapses.push(edge(format!("i-{source}"), format!("h-{i}")));
        }
        // Each starved target rejects every connected input before finding its
        // one missing input, exposing aggregate edge-search work.
        if !dense && input_count > 1 {
            synapses.push(edge(format!("i-{}", input_count - 1), format!("h-{i}")));
        }
        if !dense {
            continue;
        }
    }
    for source in 0..input_count {
        synapses.push(edge(format!("i-{source}"), format!("h-{hub_index}")));
    }
    for source in 0..hub_index {
        synapses.push(edge(format!("h-{source}"), format!("h-{hub_index}")));
    }
    synapses.push(edge(format!("h-{hub_index}"), "o-0".to_string()));

    let records = neurons
        .iter()
        .filter(|n| n.neuron_type == "hidden")
        .map(|n| {
            let rows = (0..20)
                .map(|i| DiscoverRecord::new(i, n.uuid.clone(), Some(0.5), 0.5, vec![0.2]))
                .collect();
            (n.uuid.clone(), rows)
        })
        .collect();
    (
        CreatureJson {
            neurons,
            synapses,
            input: input_count,
            output: 1,
        },
        records,
    )
}

fn bench_topology_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("topology_scan");
    for &(size, dense) in &[
        (64, false),
        (256, false),
        (64, true),
        (256, true),
        (1024, true),
    ] {
        let (creature, records) = long_path(size, dense);
        assert_forward_only(&creature);
        let edge_count = creature.synapses.len();
        group.bench_with_input(
            BenchmarkId::new(
                format!("long_path_{}", if dense { "dense" } else { "sparse" }),
                format!("hidden={size},edges={edge_count}"),
            ),
            &(&creature, &records),
            |b, (creature, records)| {
                b.iter(|| {
                    black_box(detect_topology_issues(
                        black_box(creature),
                        black_box(records),
                        None,
                    ))
                });
            },
        );

        let input_count = if dense { (size / 4).max(3) } else { size };
        let (creature, records) = starved(size, input_count, dense);
        assert_forward_only(&creature);
        let edge_count = creature.synapses.len();
        group.bench_with_input(
            BenchmarkId::new(
                format!("starved_{}", if dense { "dense" } else { "sparse" }),
                format!("hidden={size},inputs={input_count},edges={edge_count}"),
            ),
            &(&creature, &records),
            |b, (creature, records)| {
                b.iter(|| {
                    black_box(detect_topology_issues(
                        black_box(creature),
                        black_box(records),
                        None,
                    ))
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_topology_scan);
criterion_main!(benches);
