//! Regression coverage for Issue #2383: bounded topology depth traversal.

#![allow(clippy::cast_precision_loss)]

use neat_ai_discovery::analysis::detection::topology_diversification::
    detect_topology_diversification_candidates;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

fn neuron(uuid: impl Into<String>, neuron_type: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.into(),
        neuron_type: neuron_type.to_string(),
        squash: "IDENTITY".to_string(),
        bias: 0.0,
    }
}

fn synapse(from: impl Into<String>, to: impl Into<String>) -> SynapseJson {
    SynapseJson {
        from_uuid: from.into(),
        to_uuid: to.into(),
        weight: 0.5,
        ..Default::default()
    }
}

fn records_for(creature: &CreatureJson) -> Vec<(String, Vec<DiscoverRecord>)> {
    creature
        .neurons
        .iter()
        .map(|neuron| {
            let error = if neuron.neuron_type == "output" {
                0.25
            } else {
                0.01
            };
            let records = (0..20)
                .map(|obs_index| DiscoverRecord {
                    obs_index,
                    neuron_uuid: neuron.uuid.clone(),
                    value: Some(0.5),
                    activation: 0.5,
                    errors: vec![error],
                })
                .collect();
            (neuron.uuid.clone(), records)
        })
        .collect()
}

fn creature(neurons: Vec<NeuronJson>, synapses: Vec<SynapseJson>) -> CreatureJson {
    let input = neurons
        .iter()
        .filter(|neuron| neuron.neuron_type == "input")
        .count();
    let output = neurons
        .iter()
        .filter(|neuron| neuron.neuron_type == "output")
        .count();
    CreatureJson {
        neurons,
        synapses,
        input,
        output,
    }
}

#[test]
fn deep_hidden_chain_does_not_overflow_a_small_stack() {
    std::thread::Builder::new()
        .stack_size(2 * 1024 * 1024)
        .spawn(|| {
            let hidden_count = 50_000;
            let mut neurons = Vec::with_capacity(hidden_count + 2);
            let mut synapses = Vec::with_capacity(hidden_count + 1);
            neurons.push(neuron("input", "input"));
            for index in 0..hidden_count {
                neurons.push(neuron(format!("hidden-{index}"), "hidden"));
                let from = if index == 0 {
                    "input".to_string()
                } else {
                    format!("hidden-{}", index - 1)
                };
                synapses.push(synapse(from, format!("hidden-{index}")));
            }
            neurons.push(neuron("output", "output"));
            synapses.push(synapse(format!("hidden-{}", hidden_count - 1), "output"));

            let creature = creature(neurons, synapses);
            let records = records_for(&creature);
            assert!(detect_topology_diversification_candidates(&creature, &records).is_empty());
        })
        .expect("small-stack thread should spawn")
        .join()
        .expect("50,000 hidden neurons must not overflow the traversal stack");
}

#[test]
fn width_two_shared_path_ladder_keeps_deep_outputs_unflagged() {
    let depth = 16;
    let mut neurons = vec![neuron("input", "input")];
    let mut synapses = Vec::new();
    for layer in 0..depth {
        for width_index in 0..2 {
            let current = format!("hidden-{layer}-{width_index}");
            neurons.push(neuron(&current, "hidden"));
            if layer == 0 {
                synapses.push(synapse("input", &current));
            } else {
                for previous_width_index in 0..2 {
                    synapses.push(synapse(
                        format!("hidden-{}-{previous_width_index}", layer - 1),
                        &current,
                    ));
                }
            }
        }
    }
    neurons.push(neuron("output", "output"));
    for width_index in 0..2 {
        synapses.push(synapse(format!("hidden-{}-{width_index}", depth - 1), "output"));
    }
    synapses.push(synapse("input", "output"));

    let creature = creature(neurons, synapses);
    let candidates = detect_topology_diversification_candidates(&creature, &records_for(&creature));
    assert!(candidates.is_empty(), "deep input paths must suppress diversification");
}

#[test]
fn mixed_outputs_diamond_and_disconnected_chains_keep_candidate_results() {
    let neurons = vec![
        neuron("input", "input"),
        neuron("diamond-left", "hidden"),
        neuron("diamond-right", "hidden"),
        neuron("diamond-join", "hidden"),
        neuron("disconnected-1", "hidden"),
        neuron("disconnected-2", "hidden"),
        neuron("output-deep", "output"),
        neuron("output-shallow", "output"),
        neuron("output-diamond", "output"),
    ];
    let synapses = vec![
        synapse("input", "diamond-left"),
        synapse("input", "diamond-right"),
        synapse("diamond-left", "diamond-join"),
        synapse("diamond-right", "diamond-join"),
        synapse("diamond-join", "output-diamond"),
        synapse("input", "output-diamond"),
        synapse("input", "output-deep"),
        synapse("diamond-join", "output-deep"),
        synapse("input", "output-shallow"),
        synapse("disconnected-1", "disconnected-2"),
    ];
    let creature = creature(neurons, synapses);
    let candidates = detect_topology_diversification_candidates(&creature, &records_for(&creature));

    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].output_neuron_uuid, "output-shallow");
    assert_eq!(candidates[0].max_hidden_depth, 0);
    assert_eq!(candidates[0].direct_path_count, 1);
}

#[test]
fn dangling_endpoint_chains_do_not_extend_graph_traversal() {
    let dangling_count = 50_000;
    let mut synapses = Vec::with_capacity(dangling_count + 1);
    for index in 0..dangling_count {
        synapses.push(synapse(format!("ghost-{index}"), format!("ghost-{}", index + 1)));
    }
    synapses.push(synapse(format!("ghost-{dangling_count}"), "output"));
    synapses.push(synapse("input", "output"));
    let creature = creature(
        vec![neuron("input", "input"), neuron("output", "output")],
        synapses,
    );

    let candidates = detect_topology_diversification_candidates(&creature, &records_for(&creature));
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].output_neuron_uuid, "output");
    assert_eq!(candidates[0].max_hidden_depth, 0);
}
