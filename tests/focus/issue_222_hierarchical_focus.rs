//! Tests for Issue #222: Hierarchical focus neuron selection for large creatures.
//!
//! This module tests the hierarchical selection strategy that organises neurons
//! into layers based on their network depth (distance from inputs) and allocates
//! focus budget proportionally across layers.
//!
//! Benefits:
//! - Better coverage: Guaranteed analysis of neurons at all network depths
//! - Improved discovery: Output-adjacent neurons often have highest impact
//! - More representative selection for large creatures (500+ neurons)

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use neat_ai_discovery::focus::{
    AllocationStrategy, compute_network_layers, hierarchical_focus_selection,
};
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
use std::collections::{HashMap, HashSet};

/// Helper to create a creature with specified neurons and synapses.
/// Input neurons in the `neurons` parameter are used only to count `input`.
/// They are NOT included in `creature.neurons` as per the NEAT-AI data model.
fn create_creature(
    neurons: Vec<(&str, &str)>,       // (uuid, type)
    synapses: Vec<(&str, &str, f32)>, // (from, to, weight)
) -> CreatureJson {
    let input_count = neurons.iter().filter(|(_, t)| *t == "input").count();
    let output_count = neurons.iter().filter(|(_, t)| *t == "output").count();
    CreatureJson {
        neurons: neurons
            .into_iter()
            .filter(|(_, neuron_type)| *neuron_type != "input")
            .map(|(uuid, neuron_type)| NeuronJson {
                uuid: uuid.to_string(),
                neuron_type: neuron_type.to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            })
            .collect(),
        synapses: synapses
            .into_iter()
            .map(|(from, to, weight)| SynapseJson {
                from_uuid: from.to_string(),
                to_uuid: to.to_string(),
                weight,
                synapse_type: None,
            })
            .collect(),
        input: input_count,
        output: output_count,
    }
}

/// Neuron list type alias for `create_deep_network` return value.
type NeuronList = Vec<(&'static str, &'static str)>;
/// Synapse list type alias for `create_deep_network` return value.
type SynapseList = Vec<(&'static str, &'static str, f32)>;

/// Create a deep network with specified number of layers and neurons per layer.
/// Returns neurons and synapses.
fn create_deep_network(layers: usize, neurons_per_layer: usize) -> (NeuronList, SynapseList) {
    let mut neurons = Vec::new();
    let mut synapses = Vec::new();

    // Create input neurons
    for i in 0..neurons_per_layer {
        let uuid: &'static str = Box::leak(format!("input-{i}").into_boxed_str());
        neurons.push((uuid, "input"));
    }

    // Create hidden layers
    for layer in 0..(layers - 1) {
        for i in 0..neurons_per_layer {
            let uuid: &'static str = Box::leak(format!("hidden-L{layer}-{i}").into_boxed_str());
            neurons.push((uuid, "hidden"));

            // Connect from previous layer
            for j in 0..neurons_per_layer {
                let from_uuid: &'static str = if layer == 0 {
                    Box::leak(format!("input-{j}").into_boxed_str())
                } else {
                    Box::leak(format!("hidden-L{}-{j}", layer - 1).into_boxed_str())
                };
                // Sparse connections to avoid O(n^2) explosion
                if (i + j) % 3 == 0 {
                    synapses.push((from_uuid, uuid, 0.5));
                }
            }
        }
    }

    // Create output neurons (last layer)
    for i in 0..neurons_per_layer {
        let uuid: &'static str = Box::leak(format!("output-{i}").into_boxed_str());
        neurons.push((uuid, "output"));

        // Connect from last hidden layer
        let last_hidden_layer = layers - 2;
        for j in 0..neurons_per_layer {
            let from_uuid: &'static str =
                Box::leak(format!("hidden-L{last_hidden_layer}-{j}").into_boxed_str());
            if (i + j) % 2 == 0 {
                synapses.push((from_uuid, uuid, 0.5));
            }
        }
    }

    (neurons, synapses)
}

// =============================================================================
// Tests for compute_network_layers()
// =============================================================================

#[test]
fn test_compute_network_layers_simple_chain() {
    // Simple chain: input -> hidden -> output
    // Expected: layer 0 (depth 1 from input) = hidden, layer 1 (depth 2) = output
    let creature = create_creature(
        vec![
            ("input-0", "input"),
            ("hidden-1", "hidden"),
            ("output-0", "output"),
        ],
        vec![("input-0", "hidden-1", 1.0), ("hidden-1", "output-0", 1.0)],
    );

    let layers = compute_network_layers(&creature);

    // Should have 2 layers (hidden and output, inputs not included in layers)
    assert_eq!(layers.len(), 2, "Expected 2 layers for simple chain");

    // Layer 0 (depth 1) should contain hidden-1
    assert!(
        layers[0].neurons.iter().any(|n| n.uuid == "hidden-1"),
        "Layer 0 should contain hidden-1"
    );
    assert_eq!(layers[0].depth, 1, "First layer should have depth 1");

    // Layer 1 (depth 2) should contain output-0
    assert!(
        layers[1].neurons.iter().any(|n| n.uuid == "output-0"),
        "Layer 1 should contain output-0"
    );
    assert_eq!(layers[1].depth, 2, "Second layer should have depth 2");
}

#[test]
fn test_compute_network_layers_parallel_paths() {
    // Parallel paths with different depths
    // input -> hidden-1 -> output (depth 2)
    // input -> hidden-2 -> hidden-3 -> output (hidden-3 at depth 3)
    let creature = create_creature(
        vec![
            ("input-0", "input"),
            ("hidden-1", "hidden"),
            ("hidden-2", "hidden"),
            ("hidden-3", "hidden"),
            ("output-0", "output"),
        ],
        vec![
            ("input-0", "hidden-1", 1.0),
            ("input-0", "hidden-2", 1.0),
            ("hidden-1", "output-0", 1.0),
            ("hidden-2", "hidden-3", 1.0),
            ("hidden-3", "output-0", 1.0),
        ],
    );

    let layers = compute_network_layers(&creature);

    // Neurons should be grouped by their maximum depth from inputs
    // hidden-1 and hidden-2: depth 1
    // hidden-3: depth 2
    // output-0: depth 3 (max of path through hidden-3)

    // Find the layer containing hidden-1
    let hidden1_layer = layers
        .iter()
        .find(|l| l.neurons.iter().any(|n| n.uuid == "hidden-1"));
    assert!(hidden1_layer.is_some(), "hidden-1 should be in some layer");

    // Find the layer containing hidden-3
    let hidden3_layer = layers
        .iter()
        .find(|l| l.neurons.iter().any(|n| n.uuid == "hidden-3"));
    assert!(hidden3_layer.is_some(), "hidden-3 should be in some layer");

    // hidden-3 should be in a deeper layer than hidden-1
    assert!(
        hidden3_layer.unwrap().depth > hidden1_layer.unwrap().depth,
        "hidden-3 (depth {}) should be deeper than hidden-1 (depth {})",
        hidden3_layer.unwrap().depth,
        hidden1_layer.unwrap().depth
    );
}

#[test]
fn test_compute_network_layers_handles_cycles() {
    // Network with a cycle: hidden-1 <-> hidden-2
    let creature = create_creature(
        vec![
            ("input-0", "input"),
            ("hidden-1", "hidden"),
            ("hidden-2", "hidden"),
            ("output-0", "output"),
        ],
        vec![
            ("input-0", "hidden-1", 1.0),
            ("hidden-1", "hidden-2", 1.0),
            ("hidden-2", "hidden-1", 1.0), // Cycle!
            ("hidden-2", "output-0", 1.0),
        ],
    );

    // Should not panic or infinite loop
    let layers = compute_network_layers(&creature);

    // All neurons should still be assigned to layers
    let all_neuron_uuids: HashSet<_> = layers
        .iter()
        .flat_map(|l| l.neurons.iter().map(|n| n.uuid.as_str()))
        .collect();

    assert!(
        all_neuron_uuids.contains("hidden-1"),
        "hidden-1 should be in layers despite cycle"
    );
    assert!(
        all_neuron_uuids.contains("hidden-2"),
        "hidden-2 should be in layers despite cycle"
    );
    assert!(
        all_neuron_uuids.contains("output-0"),
        "output-0 should be in layers"
    );
}

#[test]
fn test_compute_network_layers_disconnected_component() {
    // Disconnected neuron (orphan)
    let creature = create_creature(
        vec![
            ("input-0", "input"),
            ("hidden-1", "hidden"),
            ("orphan", "hidden"), // Not connected to main network
            ("output-0", "output"),
        ],
        vec![
            ("input-0", "hidden-1", 1.0),
            ("hidden-1", "output-0", 1.0),
            // orphan has no connections
        ],
    );

    let layers = compute_network_layers(&creature);

    // Orphan should still be included (possibly in a special layer for disconnected neurons)
    let all_neuron_uuids: HashSet<_> = layers
        .iter()
        .flat_map(|l| l.neurons.iter().map(|n| n.uuid.as_str()))
        .collect();

    assert!(
        all_neuron_uuids.contains("orphan"),
        "Orphan neuron should still be included in layer computation"
    );
}

#[test]
fn test_compute_network_layers_deep_network() {
    // Create a deep network with 10 layers
    let (neurons, synapses) = create_deep_network(10, 5);
    let creature = create_creature(neurons, synapses);

    let layers = compute_network_layers(&creature);

    // Should have multiple layers (at least 10 for a 10-layer network)
    assert!(
        layers.len() >= 9,
        "Deep network should have at least 9 layers (excluding inputs), got {}",
        layers.len()
    );

    // Layers should be sorted by depth
    for i in 1..layers.len() {
        assert!(
            layers[i].depth >= layers[i - 1].depth,
            "Layers should be sorted by depth"
        );
    }

    // Output neurons should be in the deepest layer
    let deepest_layer = layers.last().unwrap();
    let has_output = deepest_layer
        .neurons
        .iter()
        .any(|n| n.neuron_type == "output");
    assert!(has_output, "Output neurons should be in the deepest layer");
}

// =============================================================================
// Tests for allocation strategies
// =============================================================================

#[test]
fn test_allocation_strategy_equal() {
    let creature = create_creature(
        vec![
            ("input-0", "input"),
            ("h1", "hidden"),
            ("h2", "hidden"),
            ("h3", "hidden"),
            ("h4", "hidden"),
            ("output-0", "output"),
        ],
        vec![
            ("input-0", "h1", 1.0),
            ("input-0", "h2", 1.0),
            ("h1", "h3", 1.0),
            ("h2", "h4", 1.0),
            ("h3", "output-0", 1.0),
            ("h4", "output-0", 1.0),
        ],
    );

    let layers = compute_network_layers(&creature);
    let max_focus = 10;

    // Equal allocation divides focus evenly among layers
    let result = hierarchical_focus_selection(
        &creature,
        &layers,
        max_focus,
        AllocationStrategy::Equal,
        &HashMap::new(), // empty scores for this test
    );

    // Should return up to max_focus neurons
    assert!(result.len() <= max_focus, "Should not exceed max_focus");
}

#[test]
fn test_allocation_strategy_proportional() {
    // Create a network where layers have different sizes
    let creature = create_creature(
        vec![
            ("input-0", "input"),
            ("h1", "hidden"),
            ("h2", "hidden"),
            ("h3", "hidden"),
            ("h4", "hidden"),
            ("h5", "hidden"),
            ("output-0", "output"),
        ],
        vec![
            ("input-0", "h1", 1.0),
            ("input-0", "h2", 1.0),
            ("input-0", "h3", 1.0),
            ("h1", "h4", 1.0),
            ("h2", "h4", 1.0),
            ("h3", "h5", 1.0),
            ("h4", "output-0", 1.0),
            ("h5", "output-0", 1.0),
        ],
    );

    let layers = compute_network_layers(&creature);
    let max_focus = 6;

    // Proportional allocation gives more slots to larger layers
    let result = hierarchical_focus_selection(
        &creature,
        &layers,
        max_focus,
        AllocationStrategy::Proportional,
        &HashMap::new(),
    );

    assert!(result.len() <= max_focus, "Should not exceed max_focus");
}

#[test]
fn test_allocation_strategy_output_first() {
    // OutputFirst prioritises layers closer to output
    let creature = create_creature(
        vec![
            ("input-0", "input"),
            ("h1", "hidden"),
            ("h2", "hidden"),
            ("output-0", "output"),
        ],
        vec![
            ("input-0", "h1", 1.0),
            ("h1", "h2", 1.0),
            ("h2", "output-0", 1.0),
        ],
    );

    let layers = compute_network_layers(&creature);
    let max_focus = 2;

    // With OutputFirst and only 2 slots, output and closest hidden should be selected
    let result = hierarchical_focus_selection(
        &creature,
        &layers,
        max_focus,
        AllocationStrategy::OutputFirst,
        &HashMap::new(),
    );

    // Output neuron should definitely be included
    assert!(
        result.iter().any(|uuid| uuid == "output-0"),
        "OutputFirst should include output neuron"
    );
}

// =============================================================================
// Tests for hierarchical_focus_selection()
// =============================================================================

#[test]
fn test_hierarchical_focus_selection_guarantees_layer_coverage() {
    // Create a deep network
    let (neurons, synapses) = create_deep_network(5, 10);
    let creature = create_creature(neurons.clone(), synapses);

    let layers = compute_network_layers(&creature);
    let max_focus = 20; // Enough to sample from each layer

    // Create scores for neurons
    let mut scores: HashMap<String, f32> = HashMap::new();
    for (uuid, neuron_type) in &neurons {
        if *neuron_type != "input" {
            scores.insert(uuid.to_string(), 1.0); // Equal scores
        }
    }

    let result = hierarchical_focus_selection(
        &creature,
        &layers,
        max_focus,
        AllocationStrategy::Proportional,
        &scores,
    );

    // Count how many layers have at least one neuron selected
    let mut layers_covered = 0;
    for layer in &layers {
        let has_selected = layer.neurons.iter().any(|n| result.contains(&n.uuid));
        if has_selected {
            layers_covered += 1;
        }
    }

    // With enough focus budget, all layers should have representation
    assert!(
        layers_covered >= layers.len().min(max_focus),
        "Expected at least {} layers covered, got {}",
        layers.len().min(max_focus),
        layers_covered
    );
}

#[test]
fn test_hierarchical_selection_respects_max_focus() {
    let (neurons, synapses) = create_deep_network(5, 20);
    let creature = create_creature(neurons.clone(), synapses);

    let layers = compute_network_layers(&creature);
    let max_focus = 10;

    let mut scores: HashMap<String, f32> = HashMap::new();
    for (uuid, neuron_type) in &neurons {
        if *neuron_type != "input" {
            scores.insert(uuid.to_string(), 1.0);
        }
    }

    let result = hierarchical_focus_selection(
        &creature,
        &layers,
        max_focus,
        AllocationStrategy::Equal,
        &scores,
    );

    assert_eq!(
        result.len(),
        max_focus,
        "Should return exactly max_focus neurons when enough are available"
    );
}

#[test]
fn test_hierarchical_selection_uses_scores_within_layer() {
    // Within each layer, highest-scoring neurons should be selected
    let creature = create_creature(
        vec![
            ("input-0", "input"),
            ("high-score", "hidden"),
            ("low-score", "hidden"),
            ("output-0", "output"),
        ],
        vec![
            ("input-0", "high-score", 1.0),
            ("input-0", "low-score", 1.0),
            ("high-score", "output-0", 1.0),
            ("low-score", "output-0", 1.0),
        ],
    );

    let layers = compute_network_layers(&creature);

    let mut scores: HashMap<String, f32> = HashMap::new();
    scores.insert("high-score".to_string(), 10.0);
    scores.insert("low-score".to_string(), 1.0);
    scores.insert("output-0".to_string(), 5.0);

    // With only 2 slots and 2 layers (hidden and output), we can select 1 per layer
    let max_focus = 2;
    let result = hierarchical_focus_selection(
        &creature,
        &layers,
        max_focus,
        AllocationStrategy::Equal,
        &scores,
    );

    // high-score should be selected from the hidden layer (higher score than low-score)
    assert!(
        result.contains(&"high-score".to_string()),
        "Higher-scoring neuron should be selected within layer"
    );
}

#[test]
fn test_hierarchical_selection_handles_empty_layers() {
    // Edge case: some layers might become empty after filtering
    let creature = create_creature(
        vec![
            ("input-0", "input"),
            ("hidden-1", "hidden"),
            ("output-0", "output"),
        ],
        vec![("input-0", "hidden-1", 1.0), ("hidden-1", "output-0", 1.0)],
    );

    let layers = compute_network_layers(&creature);
    let max_focus = 10; // More than available neurons

    let mut scores: HashMap<String, f32> = HashMap::new();
    scores.insert("hidden-1".to_string(), 1.0);
    scores.insert("output-0".to_string(), 1.0);

    let result = hierarchical_focus_selection(
        &creature,
        &layers,
        max_focus,
        AllocationStrategy::Equal,
        &scores,
    );

    // Should return all available neurons (2)
    assert_eq!(
        result.len(),
        2,
        "Should return all available neurons when max_focus exceeds count"
    );
}

// =============================================================================
// Tests for integration with rank_focus_neurons
// =============================================================================

#[test]
fn test_hierarchical_selection_improves_layer_distribution() {
    // This test verifies that hierarchical selection provides better layer distribution
    // compared to flat selection (which might over-represent certain layers)

    let (neurons, synapses) = create_deep_network(5, 10);
    let creature = create_creature(neurons.clone(), synapses);

    let layers = compute_network_layers(&creature);

    // Create scores that would bias flat selection towards one layer
    let mut scores: HashMap<String, f32> = HashMap::new();
    for (uuid, neuron_type) in &neurons {
        if *neuron_type != "input" {
            // Hidden neurons in layer 0 have very high scores
            if uuid.contains("L0") {
                scores.insert(uuid.to_string(), 100.0);
            } else {
                scores.insert(uuid.to_string(), 1.0);
            }
        }
    }

    let max_focus = 10;
    let result = hierarchical_focus_selection(
        &creature,
        &layers,
        max_focus,
        AllocationStrategy::Proportional,
        &scores,
    );

    // Count neurons from layer 0 (the high-score layer)
    let layer0_count = result.iter().filter(|uuid| uuid.contains("L0")).count();

    // With hierarchical selection, layer 0 shouldn't dominate
    // (flat selection would select mostly from layer 0)
    assert!(
        layer0_count < max_focus,
        "Hierarchical selection should not over-represent layer 0. Got {layer0_count} out of {max_focus}",
    );
}

#[test]
fn test_large_creature_hierarchical_selection() {
    // Test with a realistic large creature (500+ neurons)
    let layers = 10;
    let neurons_per_layer = 50; // ~500 neurons total
    let (neurons, synapses) = create_deep_network(layers, neurons_per_layer);
    let creature = create_creature(neurons.clone(), synapses);

    let network_layers = compute_network_layers(&creature);
    let max_focus = 64; // Typical focus budget

    let mut scores: HashMap<String, f32> = HashMap::new();
    for (uuid, neuron_type) in &neurons {
        if *neuron_type != "input" {
            // Vary scores to make selection interesting
            let score = if uuid.contains("output") {
                10.0
            } else {
                1.0 + (uuid.len() % 10) as f32
            };
            scores.insert(uuid.to_string(), score);
        }
    }

    let result = hierarchical_focus_selection(
        &creature,
        &network_layers,
        max_focus,
        AllocationStrategy::Proportional,
        &scores,
    );

    assert_eq!(
        result.len(),
        max_focus,
        "Should return exactly max_focus neurons for large creature"
    );

    // Verify that output neurons are represented
    let has_output = result.iter().any(|uuid| uuid.contains("output"));
    assert!(
        has_output,
        "Selection should include output neurons for large creature"
    );

    // Verify that we have neurons from multiple layers (not just one)
    let unique_layers: HashSet<_> = result
        .iter()
        .filter_map(|uuid| {
            if uuid.contains("-L") {
                // Extract layer number from hidden-L{n}-{m}
                uuid.split("-L").nth(1).and_then(|s| s.split('-').next())
            } else if uuid.contains("output") {
                Some("output")
            } else {
                None
            }
        })
        .collect();

    assert!(
        unique_layers.len() >= 3,
        "Selection should span multiple layers, got {} layers: {:?}",
        unique_layers.len(),
        unique_layers
    );
}
