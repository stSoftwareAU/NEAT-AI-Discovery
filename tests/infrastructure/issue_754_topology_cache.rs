//! Integration tests for Issue #754: Pre-computed creature topology cache.
//!
//! Verifies that detection modules produce identical results when using a
//! pre-computed `CreatureTopologyCache` versus building topology locally.

use crate::common::{hidden, make_creature, output, record, synapse};
use neat_ai_discovery::NeuronJson;
use neat_ai_discovery::analysis::detection::bottleneck::{
    bottleneck_neurons_to_coordinated_candidates, detect_bottleneck_neurons,
};
use neat_ai_discovery::analysis::detection::dead_neuron::detect_dead_neurons;
use neat_ai_discovery::analysis::detection::topology::{
    detect_topology_issues, topology_issues_to_coordinated_candidates,
};
use neat_ai_discovery::analysis::detection::topology_cache::CreatureTopologyCache;

/// Helper: build a creature with a bottleneck topology for testing.
fn bottleneck_creature() -> neat_ai_discovery::CreatureJson {
    make_creature(
        vec![
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
                uuid: "i3".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "i4".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            hidden("h1", "TANH"),
            output("o1", "LOGISTIC"),
        ],
        vec![
            synapse("i1", "h1", 0.5),
            synapse("i2", "h1", 0.3),
            synapse("i3", "h1", 0.7),
            synapse("i4", "h1", -0.2),
            synapse("h1", "o1", 0.9),
        ],
    )
}

/// Helper: build a creature with a long path topology for testing.
fn long_path_creature() -> neat_ai_discovery::CreatureJson {
    make_creature(
        vec![
            NeuronJson {
                uuid: "i1".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            hidden("h1", "TANH"),
            hidden("h2", "TANH"),
            hidden("h3", "TANH"),
            hidden("h4", "TANH"),
            output("o1", "LOGISTIC"),
        ],
        vec![
            synapse("i1", "h1", 0.5),
            synapse("h1", "h2", 0.3),
            synapse("h2", "h3", 0.7),
            synapse("h3", "h4", 0.4),
            synapse("h4", "o1", 0.9),
        ],
    )
}

#[test]
fn test_dead_neuron_cache_vs_no_cache() {
    let creature = make_creature(
        vec![
            NeuronJson {
                uuid: "i1".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            hidden("h1", "TANH"),
            output("o1", "LOGISTIC"),
        ],
        vec![synapse("i1", "h1", 0.5), synapse("h1", "o1", 0.8)],
    );

    // Dead neuron: near-zero activations
    let records: Vec<_> = (0..100).map(|i| record("h1", i, 0.0, Some(0.0))).collect();
    let neuron_records = vec![("h1".to_string(), records)];

    // Without cache
    let without = detect_dead_neurons(&creature, &neuron_records, None);

    // With cache
    let cache = CreatureTopologyCache::new(&creature);
    let with = detect_dead_neurons(&creature, &neuron_records, Some(&cache));

    assert_eq!(without.len(), with.len());
    for (a, b) in without.iter().zip(with.iter()) {
        assert_eq!(a.neuron_uuid, b.neuron_uuid);
        assert_eq!(a.mean_abs_activation, b.mean_abs_activation);
        assert_eq!(a.removal_confidence, b.removal_confidence);
        assert_eq!(a.connected_outputs, b.connected_outputs);
    }
}

#[test]
fn test_bottleneck_cache_vs_no_cache() {
    let creature = bottleneck_creature();

    let records: Vec<_> = (0..100)
        .map(|i| record("h1", i, 0.5 + 0.01 * i as f32, Some(0.3)))
        .collect();
    let neuron_records = vec![("h1".to_string(), records)];

    let without = detect_bottleneck_neurons(&creature, &neuron_records, None);
    let cache = CreatureTopologyCache::new(&creature);
    let with = detect_bottleneck_neurons(&creature, &neuron_records, Some(&cache));

    assert_eq!(without.len(), with.len());
    for (a, b) in without.iter().zip(with.iter()) {
        assert_eq!(a.neuron_uuid, b.neuron_uuid);
        assert_eq!(a.fan_in, b.fan_in);
        assert_eq!(a.fan_out, b.fan_out);
        assert_eq!(a.bottleneck_score, b.bottleneck_score);
        // upstream/downstream may be in different order due to HashSet iteration
        let mut a_up = a.upstream_uuids.clone();
        let mut b_up = b.upstream_uuids.clone();
        a_up.sort();
        b_up.sort();
        assert_eq!(a_up, b_up);
    }
}

#[test]
fn test_bottleneck_conversion_cache_vs_no_cache() {
    let creature = bottleneck_creature();

    let records: Vec<_> = (0..100)
        .map(|i| record("h1", i, 0.5 + 0.01 * i as f32, Some(0.3)))
        .collect();
    let neuron_records = vec![("h1".to_string(), records)];

    let cache = CreatureTopologyCache::new(&creature);
    let detected = detect_bottleneck_neurons(&creature, &neuron_records, Some(&cache));

    let without = bottleneck_neurons_to_coordinated_candidates(&detected, &creature, None);
    let with = bottleneck_neurons_to_coordinated_candidates(&detected, &creature, Some(&cache));

    assert_eq!(without.len(), with.len());
    for (a, b) in without.iter().zip(with.iter()) {
        assert_eq!(a.operations.len(), b.operations.len());
        assert_eq!(
            a.expected_creature_score_gain,
            b.expected_creature_score_gain
        );
    }
}

#[test]
fn test_topology_cache_vs_no_cache() {
    let creature = long_path_creature();

    let mut neuron_records = Vec::new();
    for uuid in &["h1", "h2", "h3", "h4"] {
        let records: Vec<_> = (0..100)
            .map(|i| {
                let mut r = record(uuid, i, 0.5, Some(0.3));
                r.errors = vec![0.1 * (i % 5) as f32];
                r
            })
            .collect();
        neuron_records.push((uuid.to_string(), records));
    }

    let without = detect_topology_issues(&creature, &neuron_records, None);
    let cache = CreatureTopologyCache::new(&creature);
    let with = detect_topology_issues(&creature, &neuron_records, Some(&cache));

    assert_eq!(without.len(), with.len());
    for (a, b) in without.iter().zip(with.iter()) {
        assert_eq!(a.neuron_uuid, b.neuron_uuid);
        assert_eq!(a.issue_type, b.issue_type);
        assert_eq!(a.path_length, b.path_length);
        assert_eq!(a.fan_in, b.fan_in);
        assert_eq!(a.fan_out, b.fan_out);
    }
}

#[test]
fn test_topology_conversion_cache_vs_no_cache() {
    let creature = long_path_creature();

    let mut neuron_records = Vec::new();
    for uuid in &["h1", "h2", "h3", "h4"] {
        let records: Vec<_> = (0..100)
            .map(|i| {
                let mut r = record(uuid, i, 0.5, Some(0.3));
                r.errors = vec![0.1 * (i % 5) as f32];
                r
            })
            .collect();
        neuron_records.push((uuid.to_string(), records));
    }

    let cache = CreatureTopologyCache::new(&creature);
    let detected = detect_topology_issues(&creature, &neuron_records, Some(&cache));

    let without = topology_issues_to_coordinated_candidates(&detected, &creature, None);
    let with = topology_issues_to_coordinated_candidates(&detected, &creature, Some(&cache));

    assert_eq!(without.len(), with.len());
    for (a, b) in without.iter().zip(with.iter()) {
        assert_eq!(a.operations.len(), b.operations.len());
        assert_eq!(
            a.expected_creature_score_gain,
            b.expected_creature_score_gain
        );
    }
}

#[test]
fn test_cache_construction_correctness() {
    let creature = bottleneck_creature();
    let cache = CreatureTopologyCache::new(&creature);

    // Verify neuron classification
    assert_eq!(cache.hidden_uuids.len(), 1);
    assert!(cache.hidden_uuids.contains("h1"));
    assert_eq!(cache.output_uuids.len(), 1);
    assert!(cache.output_uuids.contains("o1"));
    assert_eq!(cache.input_uuids.len(), 4);

    // Verify fan-in/fan-out
    assert_eq!(cache.fan_in_for("h1").len(), 4); // i1,i2,i3,i4 → h1
    assert_eq!(cache.fan_out_for("h1").len(), 1); // h1 → o1
    assert_eq!(cache.fan_in_for("o1").len(), 1); // h1 → o1

    // Verify synapse lookups
    assert!(cache.synapse_exists("i1", "h1"));
    assert!(cache.synapse_exists("h1", "o1"));
    assert!(!cache.synapse_exists("o1", "h1"));
    assert_eq!(cache.synapse_weight("i1", "h1"), Some(0.5));
    assert_eq!(cache.synapse_weight("h1", "o1"), Some(0.9));
}
