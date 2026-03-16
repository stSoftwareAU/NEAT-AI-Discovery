//! Issue #523: Targeted tests for the focus `layers` sub-module.
//!
//! Tests BFS layer computation with known network topologies:
//! - Linear chains
//! - Fan-out networks
//! - Skip connections
//! - Multiple inputs
//! - Disconnected neurons

use neat_ai_discovery::focus::{NeuronLayer, compute_network_layers};
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
use std::collections::HashSet;

/// Helper to build a creature from neurons and synapses.
/// Input neurons are identified by type "input" and excluded from `creature.neurons`.
fn make_creature(
    neurons: Vec<(&str, &str, &str)>, // (uuid, type, squash)
    synapses: Vec<(&str, &str, f32)>, // (from, to, weight)
) -> CreatureJson {
    let input_count = neurons.iter().filter(|(_, t, _)| *t == "input").count();
    let output_count = neurons.iter().filter(|(_, t, _)| *t == "output").count();
    CreatureJson {
        neurons: neurons
            .into_iter()
            .filter(|(_, t, _)| *t != "input")
            .map(|(uuid, ntype, squash)| NeuronJson {
                uuid: uuid.to_string(),
                neuron_type: ntype.to_string(),
                squash: squash.to_string(),
                bias: 0.0,
            })
            .collect(),
        synapses: synapses
            .into_iter()
            .map(|(from, to, w)| SynapseJson {
                from_uuid: from.to_string(),
                to_uuid: to.to_string(),
                weight: w,
                synapse_type: None,
            })
            .collect(),
        input: input_count,
        output: output_count,
    }
}

/// Collect all neuron UUIDs across all layers.
fn all_uuids(layers: &[NeuronLayer]) -> HashSet<String> {
    layers
        .iter()
        .flat_map(|l| l.neurons.iter().map(|n| n.uuid.clone()))
        .collect()
}

/// Find the depth assigned to a specific neuron UUID.
fn depth_of(layers: &[NeuronLayer], uuid: &str) -> Option<usize> {
    layers
        .iter()
        .find(|l| l.neurons.iter().any(|n| n.uuid == uuid))
        .map(|l| l.depth)
}

// =============================================================================
// Linear chain topology
// =============================================================================

#[test]
fn linear_chain_assigns_increasing_depths() {
    // input-0 -> h1 -> h2 -> h3 -> out
    let creature = make_creature(
        vec![
            ("input-0", "input", "IDENTITY"),
            ("h1", "hidden", "TANH"),
            ("h2", "hidden", "TANH"),
            ("h3", "hidden", "TANH"),
            ("out", "output", "LOGISTIC"),
        ],
        vec![
            ("input-0", "h1", 1.0),
            ("h1", "h2", 1.0),
            ("h2", "h3", 1.0),
            ("h3", "out", 1.0),
        ],
    );

    let layers = compute_network_layers(&creature);

    // Each neuron should be at strictly increasing depth
    assert_eq!(depth_of(&layers, "h1"), Some(1));
    assert_eq!(depth_of(&layers, "h2"), Some(2));
    assert_eq!(depth_of(&layers, "h3"), Some(3));
    assert_eq!(depth_of(&layers, "out"), Some(4));

    // Layers should be sorted by depth
    for i in 1..layers.len() {
        assert!(layers[i].depth >= layers[i - 1].depth);
    }
}

// =============================================================================
// Fan-out topology
// =============================================================================

#[test]
fn fan_out_groups_siblings_at_same_depth() {
    // input-0 fans out to h1, h2, h3 (all at depth 1), then all connect to out (depth 2)
    let creature = make_creature(
        vec![
            ("input-0", "input", "IDENTITY"),
            ("h1", "hidden", "TANH"),
            ("h2", "hidden", "TANH"),
            ("h3", "hidden", "TANH"),
            ("out", "output", "LOGISTIC"),
        ],
        vec![
            ("input-0", "h1", 1.0),
            ("input-0", "h2", 1.0),
            ("input-0", "h3", 1.0),
            ("h1", "out", 0.5),
            ("h2", "out", 0.3),
            ("h3", "out", 0.8),
        ],
    );

    let layers = compute_network_layers(&creature);

    // All three hidden neurons should be at the same depth
    let h1_depth = depth_of(&layers, "h1").expect("h1 should have a depth");
    let h2_depth = depth_of(&layers, "h2").expect("h2 should have a depth");
    let h3_depth = depth_of(&layers, "h3").expect("h3 should have a depth");

    assert_eq!(h1_depth, h2_depth, "h1 and h2 should be at same depth");
    assert_eq!(h2_depth, h3_depth, "h2 and h3 should be at same depth");
    assert_eq!(h1_depth, 1, "Fan-out hidden neurons should be at depth 1");

    // Output should be deeper than hidden neurons
    let out_depth = depth_of(&layers, "out").expect("out should have a depth");
    assert!(
        out_depth > h1_depth,
        "Output ({out_depth}) should be deeper than hidden ({h1_depth})"
    );
}

// =============================================================================
// Skip connections
// =============================================================================

#[test]
fn skip_connection_uses_maximum_depth() {
    // input-0 -> h1 -> h2 -> out
    //   AND input-0 -> out (skip connection)
    //
    // BFS uses MAXIMUM depth, so `out` should be at depth 3 (via h1 -> h2 -> out),
    // not depth 1 (via the skip connection).
    let creature = make_creature(
        vec![
            ("input-0", "input", "IDENTITY"),
            ("h1", "hidden", "TANH"),
            ("h2", "hidden", "TANH"),
            ("out", "output", "LOGISTIC"),
        ],
        vec![
            ("input-0", "h1", 1.0),
            ("h1", "h2", 1.0),
            ("h2", "out", 1.0),
            ("input-0", "out", 0.1), // Skip connection
        ],
    );

    let layers = compute_network_layers(&creature);

    // Output should be at the maximum depth (3), not depth 1
    let out_depth = depth_of(&layers, "out").expect("out should have a depth");
    assert_eq!(
        out_depth, 3,
        "Output should be at maximum depth 3 (not short-path depth 1), got {out_depth}"
    );
}

#[test]
fn skip_connection_does_not_affect_intermediate_depths() {
    // input-0 -> h1 -> h2 -> out
    //        \-> h2 (skip connection from input directly to h2)
    //
    // h2 has two paths: depth 1 (direct) and depth 2 (via h1).
    // With maximum depth, h2 should be at depth 2.
    let creature = make_creature(
        vec![
            ("input-0", "input", "IDENTITY"),
            ("h1", "hidden", "TANH"),
            ("h2", "hidden", "TANH"),
            ("out", "output", "LOGISTIC"),
        ],
        vec![
            ("input-0", "h1", 1.0),
            ("input-0", "h2", 0.5), // Skip: direct to h2
            ("h1", "h2", 1.0),      // Normal: h1 -> h2
            ("h2", "out", 1.0),
        ],
    );

    let layers = compute_network_layers(&creature);

    let h1_depth = depth_of(&layers, "h1").expect("h1 present");
    let h2_depth = depth_of(&layers, "h2").expect("h2 present");

    assert_eq!(h1_depth, 1);
    assert_eq!(h2_depth, 2, "h2 should be at max depth 2 (via h1), not 1");
}

// =============================================================================
// Multiple inputs
// =============================================================================

#[test]
fn multiple_inputs_all_contribute_to_depth() {
    // input-0 -> h1 -> out
    // input-1 -> h2 -> h1 -> out
    //
    // h1 receives from input-0 (depth 1) and from h2 (depth 2).
    // Maximum depth: h1 should be at depth 2.
    let creature = make_creature(
        vec![
            ("input-0", "input", "IDENTITY"),
            ("input-1", "input", "IDENTITY"),
            ("h1", "hidden", "TANH"),
            ("h2", "hidden", "TANH"),
            ("out", "output", "LOGISTIC"),
        ],
        vec![
            ("input-0", "h1", 1.0),
            ("input-1", "h2", 1.0),
            ("h2", "h1", 1.0),
            ("h1", "out", 1.0),
        ],
    );

    let layers = compute_network_layers(&creature);

    let h2_depth = depth_of(&layers, "h2").expect("h2 present");
    let h1_depth = depth_of(&layers, "h1").expect("h1 present");

    assert_eq!(h2_depth, 1, "h2 direct from input should be at depth 1");
    assert_eq!(
        h1_depth, 2,
        "h1 should be at max depth 2 (via input-1 -> h2 -> h1)"
    );
}

// =============================================================================
// Disconnected / orphan neurons
// =============================================================================

#[test]
fn disconnected_neuron_assigned_to_unreachable_layer() {
    // input-0 -> h1 -> out, orphan is disconnected
    let creature = make_creature(
        vec![
            ("input-0", "input", "IDENTITY"),
            ("h1", "hidden", "TANH"),
            ("orphan", "hidden", "TANH"),
            ("out", "output", "LOGISTIC"),
        ],
        vec![("input-0", "h1", 1.0), ("h1", "out", 1.0)],
    );

    let layers = compute_network_layers(&creature);

    let uuids = all_uuids(&layers);
    assert!(
        uuids.contains("orphan"),
        "Orphan neuron should still appear in layers"
    );

    // Orphan should be in a layer deeper than all connected neurons (usize::MAX)
    let orphan_depth = depth_of(&layers, "orphan").expect("orphan present");
    let out_depth = depth_of(&layers, "out").expect("out present");
    assert!(
        orphan_depth > out_depth,
        "Orphan ({orphan_depth}) should be deeper than output ({out_depth})"
    );
}

// =============================================================================
// Input and constant neurons excluded from layers
// =============================================================================

#[test]
fn input_and_constant_neurons_excluded_from_layers() {
    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![
            NeuronJson {
                uuid: "const-1".to_string(),
                neuron_type: "constant".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 1.0,
            },
            NeuronJson {
                uuid: "h1".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "TANH".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "out".to_string(),
                neuron_type: "output".to_string(),
                squash: "LOGISTIC".to_string(),
                bias: 0.0,
            },
        ],
        synapses: vec![
            SynapseJson {
                from_uuid: "input-0".to_string(),
                to_uuid: "h1".to_string(),
                weight: 1.0,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "const-1".to_string(),
                to_uuid: "h1".to_string(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "h1".to_string(),
                to_uuid: "out".to_string(),
                weight: 1.0,
                synapse_type: None,
            },
        ],
    };

    let layers = compute_network_layers(&creature);
    let uuids = all_uuids(&layers);

    // Constant neurons should be filtered out of layers (not selectable)
    assert!(
        !uuids.contains("const-1"),
        "Constant neuron should not appear in layers"
    );

    // Hidden and output should still be present
    assert!(uuids.contains("h1"));
    assert!(uuids.contains("out"));
}

// =============================================================================
// Empty / minimal creatures
// =============================================================================

#[test]
fn empty_creature_returns_no_layers() {
    let creature = CreatureJson {
        input: 0,
        output: 0,
        neurons: vec![],
        synapses: vec![],
    };

    let layers = compute_network_layers(&creature);
    assert!(layers.is_empty(), "Empty creature should produce no layers");
}

#[test]
fn output_only_creature_produces_single_layer() {
    // A creature with just an output neuron and no synapses
    let creature = CreatureJson {
        input: 0,
        output: 1,
        neurons: vec![NeuronJson {
            uuid: "out".to_string(),
            neuron_type: "output".to_string(),
            squash: "LOGISTIC".to_string(),
            bias: 0.0,
        }],
        synapses: vec![],
    };

    let layers = compute_network_layers(&creature);

    // Output neuron is disconnected (unreachable from inputs since there are none)
    // but should still appear in layers
    assert!(!layers.is_empty(), "Should have at least one layer");
    let uuids = all_uuids(&layers);
    assert!(uuids.contains("out"), "Output should be in layers");
}

// =============================================================================
// Diamond topology
// =============================================================================

#[test]
fn diamond_topology_assigns_correct_depths() {
    // input -> h1 -> h3 -> out
    //     \-> h2 -> h3 -> out
    //
    // h3 should be at depth 2 (max of paths through h1 and h2)
    let creature = make_creature(
        vec![
            ("input-0", "input", "IDENTITY"),
            ("h1", "hidden", "TANH"),
            ("h2", "hidden", "TANH"),
            ("h3", "hidden", "TANH"),
            ("out", "output", "LOGISTIC"),
        ],
        vec![
            ("input-0", "h1", 1.0),
            ("input-0", "h2", 1.0),
            ("h1", "h3", 1.0),
            ("h2", "h3", 1.0),
            ("h3", "out", 1.0),
        ],
    );

    let layers = compute_network_layers(&creature);

    assert_eq!(depth_of(&layers, "h1"), Some(1));
    assert_eq!(depth_of(&layers, "h2"), Some(1));
    assert_eq!(depth_of(&layers, "h3"), Some(2));
    assert_eq!(depth_of(&layers, "out"), Some(3));
}
