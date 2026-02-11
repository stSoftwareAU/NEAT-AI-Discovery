//! Issue #491: Verify that splitting focus.rs into submodules preserves all public API.
//!
//! These tests confirm that the focus module's public types, functions, and constants
//! remain accessible via `neat_ai_discovery::focus::*` after the split.
//! Each test exercises real functionality — not source code inspection.

use neat_ai_discovery::focus::{
    AllocationStrategy, GradientFlowStats, HIERARCHICAL_SELECTION_THRESHOLD, NeuronInfo,
    NeuronLayer, SynapseCounts,
};
use neat_ai_discovery::focus::{
    calculate_removal_savings, compute_impacts_public, compute_network_layers,
    hierarchical_focus_selection,
};
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
use std::collections::HashMap;

/// Helper to build a minimal creature with inputs, hidden neurons, and an output neuron.
fn make_test_creature() -> CreatureJson {
    CreatureJson {
        input: 2,
        output: 1,
        neurons: vec![
            NeuronJson {
                uuid: "hidden-1".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "TANH".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "output-1".to_string(),
                neuron_type: "output".to_string(),
                squash: "LOGISTIC".to_string(),
                bias: 0.0,
            },
        ],
        synapses: vec![
            SynapseJson {
                from_uuid: "input-0".to_string(),
                to_uuid: "hidden-1".to_string(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-1".to_string(),
                to_uuid: "hidden-1".to_string(),
                weight: 0.3,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "hidden-1".to_string(),
                to_uuid: "output-1".to_string(),
                weight: 0.8,
                synapse_type: None,
            },
        ],
    }
}

#[test]
fn compute_network_layers_returns_correct_depths() {
    let creature = make_test_creature();
    let layers = compute_network_layers(&creature);

    // Should have at least one layer (hidden + output neurons)
    assert!(
        !layers.is_empty(),
        "Expected at least one layer from compute_network_layers"
    );

    // Layers should be sorted by depth
    for i in 1..layers.len() {
        assert!(
            layers[i].depth >= layers[i - 1].depth,
            "Layers should be sorted by ascending depth"
        );
    }

    // All neurons should be assigned to some layer
    let total_neurons_in_layers: usize = layers.iter().map(|l| l.neurons.len()).sum();
    assert!(
        total_neurons_in_layers > 0,
        "Expected neurons in at least one layer"
    );
}

#[test]
fn hierarchical_focus_selection_respects_max_focus() {
    let creature = make_test_creature();
    let layers = compute_network_layers(&creature);

    let mut scores = HashMap::new();
    scores.insert("hidden-1".to_string(), 1.0_f32);
    scores.insert("output-1".to_string(), 0.5_f32);

    let selected = hierarchical_focus_selection(
        &creature,
        &layers,
        1, // max_focus = 1
        AllocationStrategy::Equal,
        &scores,
    );

    assert!(
        selected.len() <= 1,
        "hierarchical_focus_selection should respect max_focus"
    );
}

#[test]
fn allocation_strategy_variants_are_accessible() {
    // Verify all three variants can be constructed and compared
    let strategies = [
        AllocationStrategy::Equal,
        AllocationStrategy::Proportional,
        AllocationStrategy::OutputFirst,
    ];

    assert_eq!(strategies[0], AllocationStrategy::Equal);
    assert_eq!(strategies[1], AllocationStrategy::Proportional);
    assert_eq!(strategies[2], AllocationStrategy::OutputFirst);
    assert_ne!(strategies[0], strategies[1]);
}

#[test]
fn hierarchical_selection_threshold_is_100() {
    assert_eq!(
        HIERARCHICAL_SELECTION_THRESHOLD, 100,
        "HIERARCHICAL_SELECTION_THRESHOLD should be 100"
    );
}

#[test]
fn compute_impacts_public_returns_all_neurons() {
    let creature = make_test_creature();
    let impacts = compute_impacts_public(&creature);

    // Output neurons should have impact = 1.0
    assert!(
        (impacts.get("output-1").copied().unwrap_or(0.0) - 1.0).abs() < f32::EPSILON,
        "Output neuron should have impact 1.0"
    );

    // Hidden neurons should have positive impact
    let hidden_impact = impacts.get("hidden-1").copied().unwrap_or(0.0);
    assert!(
        hidden_impact > 0.0,
        "Hidden neuron should have positive impact, got {hidden_impact}"
    );
    assert!(
        hidden_impact <= 1.0,
        "Hidden neuron impact should be <= 1.0, got {hidden_impact}"
    );
}

#[test]
fn calculate_removal_savings_formula_correct() {
    // Based on NEAT-AI Score.ts: savings = growth_cost × (1 + (N + M) / 10)
    let savings = calculate_removal_savings(3, 2, 1e-7);

    // Expected: 1e-7 × (1 + 5/10) = 1e-7 × 1.5 = 1.5e-7
    let expected = 1e-7_f32 * 1.5;
    assert!(
        (savings - expected).abs() < 1e-15,
        "Removal savings formula incorrect: got {savings}, expected {expected}"
    );
}

#[test]
fn synapse_counts_returns_correct_counts() {
    let creature = make_test_creature();
    let counts = SynapseCounts::new(&creature);

    // hidden-1 has 2 incoming (from input-0, input-1) and 1 outgoing (to output-1)
    let (incoming, outgoing) = counts.get("hidden-1");
    assert_eq!(incoming, 2, "hidden-1 should have 2 incoming synapses");
    assert_eq!(outgoing, 1, "hidden-1 should have 1 outgoing synapse");

    // output-1 has 1 incoming (from hidden-1) and 0 outgoing
    let (incoming, outgoing) = counts.get("output-1");
    assert_eq!(incoming, 1, "output-1 should have 1 incoming synapse");
    assert_eq!(outgoing, 0, "output-1 should have 0 outgoing synapses");

    // Non-existent neuron
    let (incoming, outgoing) = counts.get("does-not-exist");
    assert_eq!(incoming, 0);
    assert_eq!(outgoing, 0);
}

#[test]
fn gradient_flow_stats_default_values() {
    let stats = GradientFlowStats::default();
    assert!(
        (stats.avg_gradient_magnitude - 1.0).abs() < f32::EPSILON,
        "Default gradient magnitude should be 1.0"
    );
    assert!(
        stats.saturation_ratio.abs() < f32::EPSILON,
        "Default saturation ratio should be 0.0"
    );
    assert!(
        stats.dead_ratio.abs() < f32::EPSILON,
        "Default dead ratio should be 0.0"
    );
}

#[test]
fn neuron_layer_and_info_types_accessible() {
    let layer = NeuronLayer {
        depth: 1,
        neurons: vec![NeuronInfo {
            uuid: "test-uuid".to_string(),
            neuron_type: "hidden".to_string(),
        }],
    };

    assert_eq!(layer.depth, 1);
    assert_eq!(layer.neurons.len(), 1);
    assert_eq!(layer.neurons[0].uuid, "test-uuid");
    assert_eq!(layer.neurons[0].neuron_type, "hidden");
}
