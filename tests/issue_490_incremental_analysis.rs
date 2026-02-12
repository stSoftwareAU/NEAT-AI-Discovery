//! Tests for Issue #490: Incremental analysis — skip unchanged neurons between discovery runs
//!
//! ## TDD Plan
//!
//! 1. Test fingerprint computation from creature topology for a single neuron
//! 2. Test fingerprints change when synapse weights change
//! 3. Test fingerprints change when activation function changes
//! 4. Test fingerprints change when bias changes
//! 5. Test fingerprints change when incoming synapses are added/removed
//! 6. Test fingerprints are stable when nothing changes
//! 7. Test filtering focus neurons by unchanged fingerprints
//! 8. Test new neurons (no previous fingerprint) are always analysed
//! 9. Test removed neurons are correctly handled (stale fingerprints ignored)
//! 10. Test metadata reports cache hit/miss counts

mod common;

use neat_ai_discovery::analysis::neuron_fingerprint::{
    NeuronFingerprint, compute_neuron_fingerprints, filter_changed_neurons,
};
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
use std::collections::HashMap;

fn make_test_creature() -> CreatureJson {
    CreatureJson {
        neurons: vec![
            NeuronJson {
                uuid: "output-0".to_string(),
                neuron_type: "output".to_string(),
                squash: "LOGISTIC".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "hidden-1".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "TANH".to_string(),
                bias: 0.1,
            },
            NeuronJson {
                uuid: "hidden-2".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "RELU".to_string(),
                bias: -0.5,
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
                weight: -0.3,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "hidden-1".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 1.0,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-0".to_string(),
                to_uuid: "hidden-2".to_string(),
                weight: 0.7,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "hidden-2".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.4,
                synapse_type: None,
            },
        ],
        input: 2,
        output: 1,
    }
}

// -------------------------------------------------------------------------
// 1. Test fingerprint computation
// -------------------------------------------------------------------------
#[test]
fn fingerprint_is_computed_for_each_neuron() {
    let creature = make_test_creature();
    let fingerprints = compute_neuron_fingerprints(&creature);

    // Should have fingerprints for all neurons (output + hidden)
    assert!(
        fingerprints.contains_key("output-0"),
        "Should contain output-0"
    );
    assert!(
        fingerprints.contains_key("hidden-1"),
        "Should contain hidden-1"
    );
    assert!(
        fingerprints.contains_key("hidden-2"),
        "Should contain hidden-2"
    );
}

// -------------------------------------------------------------------------
// 2. Test fingerprints change when synapse weights change
// -------------------------------------------------------------------------
#[test]
fn fingerprint_changes_when_synapse_weight_changes() {
    let creature1 = make_test_creature();
    let fp1 = compute_neuron_fingerprints(&creature1);

    let mut creature2 = make_test_creature();
    // Change weight of input-0 -> hidden-1 synapse
    creature2.synapses[0].weight = 0.9;
    let fp2 = compute_neuron_fingerprints(&creature2);

    assert_ne!(
        fp1["hidden-1"], fp2["hidden-1"],
        "Fingerprint should change when incoming synapse weight changes"
    );

    // hidden-2 should be unchanged (different neuron, no topology change)
    assert_eq!(
        fp1["hidden-2"], fp2["hidden-2"],
        "Fingerprint for unrelated neuron should be unchanged"
    );
}

// -------------------------------------------------------------------------
// 3. Test fingerprints change when activation function changes
// -------------------------------------------------------------------------
#[test]
fn fingerprint_changes_when_squash_changes() {
    let creature1 = make_test_creature();
    let fp1 = compute_neuron_fingerprints(&creature1);

    let mut creature2 = make_test_creature();
    creature2.neurons[1].squash = "RELU".to_string(); // hidden-1: TANH -> RELU
    let fp2 = compute_neuron_fingerprints(&creature2);

    assert_ne!(
        fp1["hidden-1"], fp2["hidden-1"],
        "Fingerprint should change when activation function changes"
    );
}

// -------------------------------------------------------------------------
// 4. Test fingerprints change when bias changes
// -------------------------------------------------------------------------
#[test]
fn fingerprint_changes_when_bias_changes() {
    let creature1 = make_test_creature();
    let fp1 = compute_neuron_fingerprints(&creature1);

    let mut creature2 = make_test_creature();
    creature2.neurons[1].bias = 0.5; // hidden-1 bias: 0.1 -> 0.5
    let fp2 = compute_neuron_fingerprints(&creature2);

    assert_ne!(
        fp1["hidden-1"], fp2["hidden-1"],
        "Fingerprint should change when bias changes"
    );
}

// -------------------------------------------------------------------------
// 5. Test fingerprints change when incoming synapses are added/removed
// -------------------------------------------------------------------------
#[test]
fn fingerprint_changes_when_synapse_added() {
    let creature1 = make_test_creature();
    let fp1 = compute_neuron_fingerprints(&creature1);

    let mut creature2 = make_test_creature();
    // Add a new synapse to hidden-1
    creature2.synapses.push(SynapseJson {
        from_uuid: "input-1".to_string(),
        to_uuid: "hidden-2".to_string(),
        weight: 0.2,
        synapse_type: None,
    });
    let fp2 = compute_neuron_fingerprints(&creature2);

    assert_ne!(
        fp1["hidden-2"], fp2["hidden-2"],
        "Fingerprint should change when a new incoming synapse is added"
    );

    // hidden-1 should be unchanged
    assert_eq!(
        fp1["hidden-1"], fp2["hidden-1"],
        "Fingerprint for unrelated neuron should be unchanged"
    );
}

#[test]
fn fingerprint_changes_when_synapse_removed() {
    let creature1 = make_test_creature();
    let fp1 = compute_neuron_fingerprints(&creature1);

    let mut creature2 = make_test_creature();
    // Remove synapse input-1 -> hidden-1 (index 1)
    creature2.synapses.remove(1);
    let fp2 = compute_neuron_fingerprints(&creature2);

    assert_ne!(
        fp1["hidden-1"], fp2["hidden-1"],
        "Fingerprint should change when an incoming synapse is removed"
    );
}

// -------------------------------------------------------------------------
// 6. Test fingerprints are stable when nothing changes
// -------------------------------------------------------------------------
#[test]
fn fingerprint_is_stable_when_topology_unchanged() {
    let creature = make_test_creature();
    let fp1 = compute_neuron_fingerprints(&creature);
    let fp2 = compute_neuron_fingerprints(&creature);

    assert_eq!(
        fp1, fp2,
        "Fingerprints should be identical for the same topology"
    );
}

// -------------------------------------------------------------------------
// 7. Test filtering focus neurons by unchanged fingerprints
// -------------------------------------------------------------------------
#[test]
fn filter_skips_unchanged_neurons() {
    let creature = make_test_creature();
    let previous_fingerprints = compute_neuron_fingerprints(&creature);

    let focus_neurons = vec![
        "output-0".to_string(),
        "hidden-1".to_string(),
        "hidden-2".to_string(),
    ];

    // With unchanged topology, all focus neurons should be skipped
    let result = filter_changed_neurons(&focus_neurons, &creature, &previous_fingerprints);

    assert!(
        result.changed.is_empty(),
        "No neurons should be changed: got {:?}",
        result.changed
    );
    assert_eq!(
        result.cache_hits, 3,
        "All 3 focus neurons should be cache hits"
    );
    assert_eq!(result.cache_misses, 0, "No cache misses expected");
}

#[test]
fn filter_includes_changed_neurons() {
    let creature1 = make_test_creature();
    let previous_fingerprints = compute_neuron_fingerprints(&creature1);

    // Now change hidden-1's bias
    let mut creature2 = make_test_creature();
    creature2.neurons[1].bias = 0.5;

    let focus_neurons = vec![
        "output-0".to_string(),
        "hidden-1".to_string(),
        "hidden-2".to_string(),
    ];

    let result = filter_changed_neurons(&focus_neurons, &creature2, &previous_fingerprints);

    assert_eq!(result.changed.len(), 1, "Only hidden-1 should be changed");
    assert_eq!(result.changed[0], "hidden-1");
    assert_eq!(result.cache_hits, 2, "output-0 and hidden-2 are unchanged");
    assert_eq!(result.cache_misses, 1, "hidden-1 is changed");
}

// -------------------------------------------------------------------------
// 8. Test new neurons are always analysed
// -------------------------------------------------------------------------
#[test]
fn filter_includes_new_neurons() {
    let creature1 = make_test_creature();
    let previous_fingerprints = compute_neuron_fingerprints(&creature1);

    // Add a new neuron
    let mut creature2 = make_test_creature();
    creature2.neurons.push(NeuronJson {
        uuid: "hidden-3".to_string(),
        neuron_type: "hidden".to_string(),
        squash: "TANH".to_string(),
        bias: 0.0,
    });
    creature2.synapses.push(SynapseJson {
        from_uuid: "input-0".to_string(),
        to_uuid: "hidden-3".to_string(),
        weight: 0.3,
        synapse_type: None,
    });

    let focus_neurons = vec![
        "output-0".to_string(),
        "hidden-1".to_string(),
        "hidden-3".to_string(),
    ];

    let result = filter_changed_neurons(&focus_neurons, &creature2, &previous_fingerprints);

    assert!(
        result.changed.contains(&"hidden-3".to_string()),
        "New neuron should always be included for analysis"
    );
    assert_eq!(result.cache_misses, 1, "New neuron counts as a cache miss");
}

// -------------------------------------------------------------------------
// 9. Test removed neurons are ignored (stale fingerprints)
// -------------------------------------------------------------------------
#[test]
fn filter_ignores_stale_fingerprints_for_removed_neurons() {
    let creature1 = make_test_creature();
    let previous_fingerprints = compute_neuron_fingerprints(&creature1);

    // Remove hidden-2 from the creature
    let mut creature2 = make_test_creature();
    creature2.neurons.retain(|n| n.uuid != "hidden-2");
    creature2
        .synapses
        .retain(|s| s.from_uuid != "hidden-2" && s.to_uuid != "hidden-2");

    // Focus neurons don't include removed neuron
    let focus_neurons = vec!["output-0".to_string(), "hidden-1".to_string()];

    let result = filter_changed_neurons(&focus_neurons, &creature2, &previous_fingerprints);

    // output-0 should be changed because its incoming synapses changed
    // (hidden-2 -> output-0 was removed)
    assert!(
        result.changed.contains(&"output-0".to_string()),
        "output-0 should be flagged as changed because its incoming synapse from hidden-2 was removed"
    );
}

// -------------------------------------------------------------------------
// 10. Test metadata reports cache hit/miss counts
// -------------------------------------------------------------------------
#[test]
fn filter_result_reports_correct_counts() {
    let creature = make_test_creature();
    let previous_fingerprints = compute_neuron_fingerprints(&creature);

    let mut creature2 = make_test_creature();
    creature2.neurons[1].bias = 0.5; // Change hidden-1

    let focus_neurons = vec![
        "output-0".to_string(),
        "hidden-1".to_string(),
        "hidden-2".to_string(),
    ];

    let result = filter_changed_neurons(&focus_neurons, &creature2, &previous_fingerprints);

    assert_eq!(result.total_focus_neurons, 3);
    assert_eq!(
        result.cache_hits + result.cache_misses,
        result.total_focus_neurons
    );
    assert_eq!(result.skipped_uuids.len(), result.cache_hits);
}

// -------------------------------------------------------------------------
// 11. Test empty previous fingerprints means all neurons analysed
// -------------------------------------------------------------------------
#[test]
fn filter_with_empty_previous_fingerprints_analyses_all() {
    let creature = make_test_creature();
    let empty_fingerprints: HashMap<String, NeuronFingerprint> = HashMap::new();

    let focus_neurons = vec![
        "output-0".to_string(),
        "hidden-1".to_string(),
        "hidden-2".to_string(),
    ];

    let result = filter_changed_neurons(&focus_neurons, &creature, &empty_fingerprints);

    assert_eq!(
        result.changed.len(),
        3,
        "All neurons should be analysed when no previous fingerprints exist"
    );
    assert_eq!(result.cache_misses, 3);
    assert_eq!(result.cache_hits, 0);
}

// -------------------------------------------------------------------------
// 12. Test fingerprints serialise/deserialise via JSON correctly
// -------------------------------------------------------------------------
#[test]
fn fingerprints_round_trip_through_json() {
    let creature = make_test_creature();
    let fingerprints = compute_neuron_fingerprints(&creature);

    // Serialise to JSON and back — this is how callers store and replay fingerprints
    let json = serde_json::to_string(&fingerprints).expect("serialisation should succeed");
    let restored: HashMap<String, NeuronFingerprint> =
        serde_json::from_str(&json).expect("deserialisation should succeed");

    assert_eq!(
        fingerprints, restored,
        "Fingerprints should survive JSON round-trip"
    );
}

// -------------------------------------------------------------------------
// 13. Test fingerprint includes outgoing synapse changes
// -------------------------------------------------------------------------
#[test]
fn fingerprint_changes_when_outgoing_synapse_weight_changes() {
    let creature1 = make_test_creature();
    let fp1 = compute_neuron_fingerprints(&creature1);

    let mut creature2 = make_test_creature();
    // Change weight of hidden-1 -> output-0 synapse (index 2)
    creature2.synapses[2].weight = 2.0;
    let fp2 = compute_neuron_fingerprints(&creature2);

    // hidden-1's fingerprint should change because its outgoing synapse weight changed.
    // The outgoing weight affects how this neuron's activation contributes to downstream error.
    assert_ne!(
        fp1["hidden-1"], fp2["hidden-1"],
        "Fingerprint should change when outgoing synapse weight changes"
    );
}
