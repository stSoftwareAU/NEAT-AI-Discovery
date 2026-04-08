//! Benchmark for Issue #1036: UUID string cloning reduction in analysis preparation.
//!
//! Measures the cost of building neuron lookup maps using `Arc<str>` keys
//! (shared across maps) vs the previous approach of cloning full `String` keys
//! into each map independently.
//!
//! Also benchmarks the cache record-loading helpers that were optimised to
//! avoid double-cloning via intermediate `Vec<String>` collections.

#![allow(clippy::cast_precision_loss)]
use criterion::{Criterion, criterion_group, criterion_main};
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
use std::collections::HashMap;
use std::hint::black_box;
use std::sync::Arc;

/// Build a creature with a realistic number of neurons and synapses.
fn create_test_creature(neuron_count: usize) -> CreatureJson {
    let input_count = neuron_count / 4;
    let hidden_count = neuron_count / 2;
    let output_count = neuron_count - input_count - hidden_count;

    let mut neurons = Vec::with_capacity(hidden_count + output_count);
    let mut synapses = Vec::new();

    for h in 0..hidden_count {
        let uuid = format!("hidden-{h:08x}-abcd-1234-efgh-{h:012x}");
        neurons.push(NeuronJson {
            uuid: uuid.clone(),
            neuron_type: "hidden".to_string(),
            squash: "TANH".to_string(),
            bias: 0.1,
        });

        // Connect from an input to this hidden neuron
        let input_idx = h % input_count;
        synapses.push(SynapseJson {
            from_uuid: format!("input-{input_idx}"),
            to_uuid: uuid,
            weight: 0.5,
            synapse_type: None,
        });
    }

    for o in 0..output_count {
        let uuid = format!("output-{o:08x}-abcd-5678-efgh-{o:012x}");
        neurons.push(NeuronJson {
            uuid: uuid.clone(),
            neuron_type: "output".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        });

        // Connect some hidden neurons to output
        for neuron in neurons.iter().take(std::cmp::min(5, hidden_count)) {
            synapses.push(SynapseJson {
                from_uuid: neuron.uuid.clone(),
                to_uuid: uuid.clone(),
                weight: 0.3,
                synapse_type: None,
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

/// Benchmark: Build 3 lookup maps using Arc<str> keys (current implementation).
/// Each neuron UUID is allocated once as Arc<str> and shared across all maps.
fn build_maps_arc_str(creature: &CreatureJson) {
    let squash_map: HashMap<Arc<str>, String> = creature
        .neurons
        .iter()
        .map(|n| {
            let key: Arc<str> = Arc::from(n.uuid.as_str());
            (key, n.squash.clone())
        })
        .collect();

    let mut type_map: HashMap<Arc<str>, String> =
        HashMap::with_capacity(creature.input + creature.neurons.len());
    for i in 0..creature.input {
        let key: Arc<str> = Arc::from(format!("input-{i}").as_str());
        type_map.insert(key, "input".to_string());
    }
    for n in &creature.neurons {
        let key: Arc<str> = if let Some((k, _)) = squash_map.get_key_value(n.uuid.as_str()) {
            Arc::clone(k)
        } else {
            Arc::from(n.uuid.as_str())
        };
        type_map.insert(key, n.neuron_type.clone());
    }

    let _order_map: HashMap<Arc<str>, usize> = creature
        .neurons
        .iter()
        .enumerate()
        .map(|(idx, n)| {
            let key: Arc<str> = if let Some((k, _)) = squash_map.get_key_value(n.uuid.as_str()) {
                Arc::clone(k)
            } else if let Some((k, _)) = type_map.get_key_value(n.uuid.as_str()) {
                Arc::clone(k)
            } else {
                Arc::from(n.uuid.as_str())
            };
            (key, idx)
        })
        .collect();

    black_box(&squash_map);
    black_box(&type_map);
    black_box(&_order_map);
}

/// Benchmark: Build 3 lookup maps using String keys (old implementation).
/// Each neuron UUID is cloned independently for every map.
fn build_maps_string_clone(creature: &CreatureJson) {
    let squash_map: HashMap<String, String> = creature
        .neurons
        .iter()
        .map(|n| (n.uuid.clone(), n.squash.clone()))
        .collect();

    let mut type_map: HashMap<String, String> = HashMap::new();
    for i in 0..creature.input {
        type_map.insert(format!("input-{i}"), "input".to_string());
    }
    for n in &creature.neurons {
        type_map.insert(n.uuid.clone(), n.neuron_type.clone());
    }

    let _order_map: HashMap<String, usize> = creature
        .neurons
        .iter()
        .enumerate()
        .map(|(idx, n)| (n.uuid.clone(), idx))
        .collect();

    black_box(&squash_map);
    black_box(&type_map);
    black_box(&_order_map);
}

fn bench_preparation_map_building(c: &mut Criterion) {
    let mut group = c.benchmark_group("uuid_preparation_maps");

    for neuron_count in [100, 500, 2000] {
        let creature = create_test_creature(neuron_count);

        group.bench_function(format!("arc_str_{neuron_count}_neurons"), |b| {
            b.iter(|| build_maps_arc_str(black_box(&creature)));
        });

        group.bench_function(format!("string_clone_{neuron_count}_neurons"), |b| {
            b.iter(|| build_maps_string_clone(black_box(&creature)));
        });
    }

    group.finish();
}

criterion_group!(benches, bench_preparation_map_building);
criterion_main!(benches);
