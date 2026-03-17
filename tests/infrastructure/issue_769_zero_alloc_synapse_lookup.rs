//! Issue #769: Zero-allocation synapse lookups in `CreatureTopologyCache`.
//!
//! Verifies that `synapse_exists()` and `synapse_weight()` return correct
//! results using the nested `HashMap` internal storage that avoids per-call
//! `String` allocations.

use neat_ai_discovery::analysis::detection::topology_cache::CreatureTopologyCache;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

fn make_creature() -> CreatureJson {
    CreatureJson {
        neurons: vec![
            NeuronJson {
                uuid: "i1".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "i2".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "h1".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "TANH".to_string(),
                bias: 0.1,
            },
            NeuronJson {
                uuid: "h2".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "RELU".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "o1".to_string(),
                neuron_type: "output".to_string(),
                squash: "LOGISTIC".to_string(),
                bias: 0.0,
            },
        ],
        synapses: vec![
            SynapseJson {
                from_uuid: "i1".to_string(),
                to_uuid: "h1".to_string(),
                weight: 0.5,
                ..Default::default()
            },
            SynapseJson {
                from_uuid: "i2".to_string(),
                to_uuid: "h1".to_string(),
                weight: -0.3,
                ..Default::default()
            },
            SynapseJson {
                from_uuid: "i1".to_string(),
                to_uuid: "h2".to_string(),
                weight: 0.7,
                ..Default::default()
            },
            SynapseJson {
                from_uuid: "h1".to_string(),
                to_uuid: "o1".to_string(),
                weight: 0.8,
                ..Default::default()
            },
            SynapseJson {
                from_uuid: "h2".to_string(),
                to_uuid: "o1".to_string(),
                weight: -0.4,
                ..Default::default()
            },
        ],
        input: 2,
        output: 1,
    }
}

#[test]
fn synapse_exists_returns_true_for_existing_synapses() {
    let cache = CreatureTopologyCache::new(&make_creature());

    assert!(cache.synapse_exists("i1", "h1"));
    assert!(cache.synapse_exists("i2", "h1"));
    assert!(cache.synapse_exists("i1", "h2"));
    assert!(cache.synapse_exists("h1", "o1"));
    assert!(cache.synapse_exists("h2", "o1"));
}

#[test]
fn synapse_exists_returns_false_for_missing_synapses() {
    let cache = CreatureTopologyCache::new(&make_creature());

    // Reverse direction
    assert!(!cache.synapse_exists("h1", "i1"));
    // No direct connection
    assert!(!cache.synapse_exists("i1", "o1"));
    assert!(!cache.synapse_exists("h1", "h2"));
    // Non-existent neuron
    assert!(!cache.synapse_exists("nope", "h1"));
    assert!(!cache.synapse_exists("h1", "nope"));
}

#[test]
fn synapse_weight_returns_correct_values() {
    let cache = CreatureTopologyCache::new(&make_creature());

    assert_eq!(cache.synapse_weight("i1", "h1"), Some(0.5));
    assert_eq!(cache.synapse_weight("i2", "h1"), Some(-0.3));
    assert_eq!(cache.synapse_weight("i1", "h2"), Some(0.7));
    assert_eq!(cache.synapse_weight("h1", "o1"), Some(0.8));
    assert_eq!(cache.synapse_weight("h2", "o1"), Some(-0.4));
}

#[test]
fn synapse_weight_returns_none_for_missing() {
    let cache = CreatureTopologyCache::new(&make_creature());

    assert_eq!(cache.synapse_weight("h1", "i1"), None);
    assert_eq!(cache.synapse_weight("nope", "h1"), None);
    assert_eq!(cache.synapse_weight("h1", "nope"), None);
}

#[test]
fn synapse_count_matches_creature_synapses() {
    let cache = CreatureTopologyCache::new(&make_creature());
    assert_eq!(cache.synapse_count(), 5);
}

#[test]
fn multiple_outgoing_from_same_source() {
    let cache = CreatureTopologyCache::new(&make_creature());

    // i1 has two outgoing synapses: i1→h1 and i1→h2
    assert!(cache.synapse_exists("i1", "h1"));
    assert!(cache.synapse_exists("i1", "h2"));
    assert_eq!(cache.synapse_weight("i1", "h1"), Some(0.5));
    assert_eq!(cache.synapse_weight("i1", "h2"), Some(0.7));
}

#[test]
fn multiple_incoming_to_same_target() {
    let cache = CreatureTopologyCache::new(&make_creature());

    // o1 has two incoming synapses: h1→o1 and h2→o1
    assert!(cache.synapse_exists("h1", "o1"));
    assert!(cache.synapse_exists("h2", "o1"));
    assert_eq!(cache.synapse_weight("h1", "o1"), Some(0.8));
    assert_eq!(cache.synapse_weight("h2", "o1"), Some(-0.4));
}

#[test]
fn empty_creature_has_no_synapses() {
    let creature = CreatureJson {
        neurons: vec![NeuronJson {
            uuid: "i1".to_string(),
            neuron_type: "input".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        }],
        synapses: vec![],
        input: 1,
        output: 0,
    };
    let cache = CreatureTopologyCache::new(&creature);

    assert!(!cache.synapse_exists("i1", "anything"));
    assert_eq!(cache.synapse_weight("i1", "anything"), None);
    assert_eq!(cache.synapse_count(), 0);
}
