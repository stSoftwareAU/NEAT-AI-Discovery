//! Tests for Issue #208: Pre-compute synapse counts to eliminate O(n×m) complexity.
//!
//! The `count_synapses_for_neuron` function previously scanned ALL synapses twice
//! (incoming + outgoing) for each neuron being ranked. With n neurons and m synapses,
//! this was O(n × m) complexity.
//!
//! The fix pre-computes synapse counts into HashMaps, reducing complexity to O(n + m).
//!
//! These tests verify:
//! - Correct synapse counts match the original implementation
//! - Edge cases: neurons with 0 synapses, self-loops
//! - Empty creatures are handled correctly

use neat_ai_discovery::focus::SynapseCounts;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

/// Helper to create a simple creature with specified neurons and synapses
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

#[test]
fn test_synapse_counts_basic() {
    // Simple network: input-0 -> hidden-1 -> output-0
    let creature = create_creature(
        vec![
            ("input-0", "input"),
            ("hidden-1", "hidden"),
            ("output-0", "output"),
        ],
        vec![
            ("input-0", "hidden-1", 1.0),  // hidden-1 has 1 incoming
            ("hidden-1", "output-0", 1.0), // hidden-1 has 1 outgoing, output-0 has 1 incoming
        ],
    );

    let counts = SynapseCounts::new(&creature);

    // Check hidden-1: 1 incoming (from input-0), 1 outgoing (to output-0)
    let (incoming, outgoing) = counts.get("hidden-1");
    assert_eq!(incoming, 1, "hidden-1 should have 1 incoming synapse");
    assert_eq!(outgoing, 1, "hidden-1 should have 1 outgoing synapse");

    // Check output-0: 1 incoming (from hidden-1), 0 outgoing
    let (incoming, outgoing) = counts.get("output-0");
    assert_eq!(incoming, 1, "output-0 should have 1 incoming synapse");
    assert_eq!(outgoing, 0, "output-0 should have 0 outgoing synapses");

    // Check input-0: 0 incoming, 1 outgoing (to hidden-1)
    let (incoming, outgoing) = counts.get("input-0");
    assert_eq!(incoming, 0, "input-0 should have 0 incoming synapses");
    assert_eq!(outgoing, 1, "input-0 should have 1 outgoing synapse");
}

#[test]
fn test_synapse_counts_multiple_connections() {
    // Network with multiple connections:
    // input-0 -> hidden-1 -> output-0
    // input-1 -> hidden-1
    // hidden-1 -> output-1
    let creature = create_creature(
        vec![
            ("input-0", "input"),
            ("input-1", "input"),
            ("hidden-1", "hidden"),
            ("output-0", "output"),
            ("output-1", "output"),
        ],
        vec![
            ("input-0", "hidden-1", 1.0),
            ("input-1", "hidden-1", 1.0),
            ("hidden-1", "output-0", 1.0),
            ("hidden-1", "output-1", 1.0),
        ],
    );

    let counts = SynapseCounts::new(&creature);

    // hidden-1: 2 incoming (from input-0, input-1), 2 outgoing (to output-0, output-1)
    let (incoming, outgoing) = counts.get("hidden-1");
    assert_eq!(incoming, 2, "hidden-1 should have 2 incoming synapses");
    assert_eq!(outgoing, 2, "hidden-1 should have 2 outgoing synapses");
}

#[test]
fn test_synapse_counts_neuron_with_no_synapses() {
    // A neuron with no connections (orphan)
    let creature = create_creature(
        vec![
            ("input-0", "input"),
            ("orphan", "hidden"), // No connections
            ("connected", "hidden"),
            ("output-0", "output"),
        ],
        vec![
            ("input-0", "connected", 1.0),
            ("connected", "output-0", 1.0),
            // Note: orphan has no synapses
        ],
    );

    let counts = SynapseCounts::new(&creature);

    // orphan: 0 incoming, 0 outgoing
    let (incoming, outgoing) = counts.get("orphan");
    assert_eq!(incoming, 0, "orphan should have 0 incoming synapses");
    assert_eq!(outgoing, 0, "orphan should have 0 outgoing synapses");
}

#[test]
fn test_synapse_counts_self_loop() {
    // A neuron with a self-loop (synapse from itself to itself)
    let creature = create_creature(
        vec![
            ("input-0", "input"),
            ("self-loop", "hidden"),
            ("output-0", "output"),
        ],
        vec![
            ("input-0", "self-loop", 1.0),
            ("self-loop", "self-loop", 0.5), // Self-loop
            ("self-loop", "output-0", 1.0),
        ],
    );

    let counts = SynapseCounts::new(&creature);

    // self-loop: 2 incoming (from input-0 + itself), 2 outgoing (to output-0 + itself)
    let (incoming, outgoing) = counts.get("self-loop");
    assert_eq!(
        incoming, 2,
        "self-loop should have 2 incoming synapses (input + self)"
    );
    assert_eq!(
        outgoing, 2,
        "self-loop should have 2 outgoing synapses (output + self)"
    );
}

#[test]
fn test_synapse_counts_empty_creature() {
    // Empty creature with no neurons or synapses
    let creature = CreatureJson {
        neurons: vec![],
        synapses: vec![],
        input: 0,
        output: 0,
    };

    let counts = SynapseCounts::new(&creature);

    // Any neuron lookup should return (0, 0)
    let (incoming, outgoing) = counts.get("nonexistent");
    assert_eq!(incoming, 0, "nonexistent neuron should have 0 incoming");
    assert_eq!(outgoing, 0, "nonexistent neuron should have 0 outgoing");
}

#[test]
fn test_synapse_counts_nonexistent_neuron() {
    // Lookup a neuron that doesn't exist in the creature
    let creature = create_creature(
        vec![
            ("input-0", "input"),
            ("hidden-1", "hidden"),
            ("output-0", "output"),
        ],
        vec![("input-0", "hidden-1", 1.0), ("hidden-1", "output-0", 1.0)],
    );

    let counts = SynapseCounts::new(&creature);

    // Lookup nonexistent neuron should return (0, 0)
    let (incoming, outgoing) = counts.get("nonexistent-uuid");
    assert_eq!(incoming, 0, "nonexistent neuron should have 0 incoming");
    assert_eq!(outgoing, 0, "nonexistent neuron should have 0 outgoing");
}

#[test]
fn test_synapse_counts_complex_network() {
    // A more complex network to test cumulative counting
    //
    // input-0 ─┬─> hidden-1 ─┬─> output-0
    //          │             │
    // input-1 ─┴─> hidden-2 ─┴─> output-1
    //                 │
    //                 └─> hidden-3 ─> output-0
    let creature = create_creature(
        vec![
            ("input-0", "input"),
            ("input-1", "input"),
            ("hidden-1", "hidden"),
            ("hidden-2", "hidden"),
            ("hidden-3", "hidden"),
            ("output-0", "output"),
            ("output-1", "output"),
        ],
        vec![
            ("input-0", "hidden-1", 1.0),  // hidden-1: 1 incoming
            ("input-1", "hidden-1", 1.0),  // hidden-1: 2 incoming
            ("input-0", "hidden-2", 1.0),  // hidden-2: 1 incoming
            ("input-1", "hidden-2", 1.0),  // hidden-2: 2 incoming
            ("hidden-1", "output-0", 1.0), // hidden-1: 1 outgoing, output-0: 1 incoming
            ("hidden-1", "output-1", 1.0), // hidden-1: 2 outgoing, output-1: 1 incoming
            ("hidden-2", "output-0", 1.0), // hidden-2: 1 outgoing, output-0: 2 incoming
            ("hidden-2", "output-1", 1.0), // hidden-2: 2 outgoing, output-1: 2 incoming
            ("hidden-2", "hidden-3", 1.0), // hidden-2: 3 outgoing, hidden-3: 1 incoming
            ("hidden-3", "output-0", 1.0), // hidden-3: 1 outgoing, output-0: 3 incoming
        ],
    );

    let counts = SynapseCounts::new(&creature);

    // Verify hidden-1: 2 incoming (from input-0, input-1), 2 outgoing (to output-0, output-1)
    let (incoming, outgoing) = counts.get("hidden-1");
    assert_eq!(incoming, 2, "hidden-1 should have 2 incoming synapses");
    assert_eq!(outgoing, 2, "hidden-1 should have 2 outgoing synapses");

    // Verify hidden-2: 2 incoming, 3 outgoing (output-0, output-1, hidden-3)
    let (incoming, outgoing) = counts.get("hidden-2");
    assert_eq!(incoming, 2, "hidden-2 should have 2 incoming synapses");
    assert_eq!(outgoing, 3, "hidden-2 should have 3 outgoing synapses");

    // Verify hidden-3: 1 incoming (from hidden-2), 1 outgoing (to output-0)
    let (incoming, outgoing) = counts.get("hidden-3");
    assert_eq!(incoming, 1, "hidden-3 should have 1 incoming synapse");
    assert_eq!(outgoing, 1, "hidden-3 should have 1 outgoing synapse");

    // Verify output-0: 3 incoming (from hidden-1, hidden-2, hidden-3), 0 outgoing
    let (incoming, outgoing) = counts.get("output-0");
    assert_eq!(incoming, 3, "output-0 should have 3 incoming synapses");
    assert_eq!(outgoing, 0, "output-0 should have 0 outgoing synapses");

    // Verify output-1: 2 incoming (from hidden-1, hidden-2), 0 outgoing
    let (incoming, outgoing) = counts.get("output-1");
    assert_eq!(incoming, 2, "output-1 should have 2 incoming synapses");
    assert_eq!(outgoing, 0, "output-1 should have 0 outgoing synapses");

    // Verify input-0: 0 incoming, 2 outgoing (to hidden-1, hidden-2)
    let (incoming, outgoing) = counts.get("input-0");
    assert_eq!(incoming, 0, "input-0 should have 0 incoming synapses");
    assert_eq!(outgoing, 2, "input-0 should have 2 outgoing synapses");
}

#[test]
fn test_synapse_counts_matches_original_implementation() {
    // Verify that SynapseCounts produces the same results as the original
    // O(n) implementation for a variety of neurons.
    //
    // This test ensures the optimised implementation is a drop-in replacement.
    let creature = create_creature(
        vec![
            ("input-0", "input"),
            ("input-1", "input"),
            ("hidden-1", "hidden"),
            ("hidden-2", "hidden"),
            ("hidden-3", "hidden"),
            ("output-0", "output"),
        ],
        vec![
            ("input-0", "hidden-1", 1.0),
            ("input-0", "hidden-2", 0.5),
            ("input-1", "hidden-1", 0.8),
            ("hidden-1", "hidden-2", 0.3),
            ("hidden-1", "hidden-3", 0.4),
            ("hidden-2", "output-0", 0.6),
            ("hidden-3", "output-0", 0.7),
        ],
    );

    let counts = SynapseCounts::new(&creature);

    // Helper function mimicking the original implementation
    fn count_original(neuron_uuid: &str, creature: &CreatureJson) -> (usize, usize) {
        let incoming = creature
            .synapses
            .iter()
            .filter(|s| s.to_uuid == neuron_uuid)
            .count();
        let outgoing = creature
            .synapses
            .iter()
            .filter(|s| s.from_uuid == neuron_uuid)
            .count();
        (incoming, outgoing)
    }

    // Test all neurons
    let neurons_to_test = vec![
        "input-0",
        "input-1",
        "hidden-1",
        "hidden-2",
        "hidden-3",
        "output-0",
        "nonexistent",
    ];

    for neuron_uuid in neurons_to_test {
        let (expected_in, expected_out) = count_original(neuron_uuid, &creature);
        let (actual_in, actual_out) = counts.get(neuron_uuid);
        assert_eq!(
            (actual_in, actual_out),
            (expected_in, expected_out),
            "SynapseCounts.get({neuron_uuid}) should match original implementation"
        );
    }
}
