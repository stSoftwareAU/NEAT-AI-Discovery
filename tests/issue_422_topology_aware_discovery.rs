//! Tests for Issue #422: Topology-aware discovery — network structure analysis.
//!
//! Analyses network structure to identify structural improvements based on
//! path lengths, connectivity balance, and fan-in/fan-out optimisation.
//!
//! ## TDD Plan
//! 1. Test path length analysis: detect neurons with unnecessarily long paths to outputs
//! 2. Test connectivity imbalance: find regions with too few/many connections
//! 3. Test fan-in/fan-out optimisation: balance information flow capacity
//! 4. Verify non-issues are not flagged (healthy topology)
//! 5. Test conversion to coordinated structural candidates
//! 6. Test edge cases (empty network, no hidden neurons, insufficient samples)

mod common;

use neat_ai_discovery::analysis::detection::topology::{
    detect_topology_issues, topology_issues_to_coordinated_candidates,
};
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

/// Helper: create a DiscoverRecord with custom errors.
fn record(neuron_uuid: &str, obs_index: u32, activation: f32, errors: Vec<f32>) -> DiscoverRecord {
    DiscoverRecord {
        obs_index,
        neuron_uuid: neuron_uuid.to_string(),
        value: None,
        activation,
        errors,
    }
}

/// Build records for a neuron: 30 samples with varying activations and errors.
fn make_records(uuid: &str, count: u32, activation_scale: f32, error: f32) -> Vec<DiscoverRecord> {
    (0..count)
        .map(|i| {
            record(
                uuid,
                i,
                (i as f32 * 0.1).sin() * activation_scale,
                vec![error * (1.0 + (i as f32 * 0.05).sin())],
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Test network builders
// ---------------------------------------------------------------------------

/// Deep chain: input → h0 → h1 → h2 → h3 → output
/// h0 has path length 4 to output (unnecessarily deep for a single stream).
fn deep_chain_creature() -> CreatureJson {
    CreatureJson {
        neurons: vec![
            NeuronJson {
                uuid: "input-0".into(),
                neuron_type: "input".into(),
                squash: "IDENTITY".into(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "h0".into(),
                neuron_type: "hidden".into(),
                squash: "TANH".into(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "h1".into(),
                neuron_type: "hidden".into(),
                squash: "TANH".into(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "h2".into(),
                neuron_type: "hidden".into(),
                squash: "TANH".into(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "h3".into(),
                neuron_type: "hidden".into(),
                squash: "TANH".into(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "output-0".into(),
                neuron_type: "output".into(),
                squash: "TANH".into(),
                bias: 0.0,
            },
        ],
        synapses: vec![
            SynapseJson {
                from_uuid: "input-0".into(),
                to_uuid: "h0".into(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "h0".into(),
                to_uuid: "h1".into(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "h1".into(),
                to_uuid: "h2".into(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "h2".into(),
                to_uuid: "h3".into(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "h3".into(),
                to_uuid: "output-0".into(),
                weight: 0.5,
                synapse_type: None,
            },
        ],
        input: 1,
        output: 1,
    }
}

/// Imbalanced connectivity creature:
/// input-0 → h0 → output-0 (h0 has fan-in=1, fan-out=1)
/// input-1, input-2, input-3 are disconnected from h0 but connect to output
/// through h1 which has fan-in=3, fan-out=1.
/// h0 is starved (low connectivity) while h1 is overloaded.
fn imbalanced_creature() -> CreatureJson {
    CreatureJson {
        neurons: vec![
            NeuronJson {
                uuid: "input-0".into(),
                neuron_type: "input".into(),
                squash: "IDENTITY".into(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "input-1".into(),
                neuron_type: "input".into(),
                squash: "IDENTITY".into(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "input-2".into(),
                neuron_type: "input".into(),
                squash: "IDENTITY".into(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "input-3".into(),
                neuron_type: "input".into(),
                squash: "IDENTITY".into(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "h0".into(),
                neuron_type: "hidden".into(),
                squash: "TANH".into(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "h1".into(),
                neuron_type: "hidden".into(),
                squash: "TANH".into(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "output-0".into(),
                neuron_type: "output".into(),
                squash: "TANH".into(),
                bias: 0.0,
            },
        ],
        synapses: vec![
            // h0: fan-in=1 (starved)
            SynapseJson {
                from_uuid: "input-0".into(),
                to_uuid: "h0".into(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "h0".into(),
                to_uuid: "output-0".into(),
                weight: 0.5,
                synapse_type: None,
            },
            // h1: fan-in=3 (overloaded)
            SynapseJson {
                from_uuid: "input-1".into(),
                to_uuid: "h1".into(),
                weight: 0.4,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-2".into(),
                to_uuid: "h1".into(),
                weight: 0.3,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-3".into(),
                to_uuid: "h1".into(),
                weight: 0.6,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "h1".into(),
                to_uuid: "output-0".into(),
                weight: 0.7,
                synapse_type: None,
            },
        ],
        input: 4,
        output: 1,
    }
}

/// Well-balanced creature: 2 inputs → 2 hidden → 1 output, each hidden gets 1 input
/// No topology issues expected.
fn balanced_creature() -> CreatureJson {
    CreatureJson {
        neurons: vec![
            NeuronJson {
                uuid: "input-0".into(),
                neuron_type: "input".into(),
                squash: "IDENTITY".into(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "input-1".into(),
                neuron_type: "input".into(),
                squash: "IDENTITY".into(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "h0".into(),
                neuron_type: "hidden".into(),
                squash: "TANH".into(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "h1".into(),
                neuron_type: "hidden".into(),
                squash: "TANH".into(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "output-0".into(),
                neuron_type: "output".into(),
                squash: "TANH".into(),
                bias: 0.0,
            },
        ],
        synapses: vec![
            SynapseJson {
                from_uuid: "input-0".into(),
                to_uuid: "h0".into(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-1".into(),
                to_uuid: "h1".into(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "h0".into(),
                to_uuid: "output-0".into(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "h1".into(),
                to_uuid: "output-0".into(),
                weight: 0.5,
                synapse_type: None,
            },
        ],
        input: 2,
        output: 1,
    }
}

/// Deep chain with high error: same as deep_chain but neurons carry significant error,
/// making skip connections worthwhile.
fn deep_chain_with_error_creature() -> CreatureJson {
    deep_chain_creature()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Test 1: Detect long path in a deep chain — suggest skip connection.
#[test]
fn test_detects_long_path_in_deep_chain() {
    let creature = deep_chain_with_error_creature();

    let neuron_records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        ("h0".into(), make_records("h0", 30, 0.5, 0.2)),
        ("h1".into(), make_records("h1", 30, 0.4, 0.15)),
        ("h2".into(), make_records("h2", 30, 0.3, 0.12)),
        ("h3".into(), make_records("h3", 30, 0.2, 0.1)),
    ];

    let candidates = detect_topology_issues(&creature, &neuron_records, None);

    assert!(
        !candidates.is_empty(),
        "Should detect topology issues in a deep 4-hop chain"
    );

    // At least one candidate should suggest a skip connection (path shortening)
    let has_long_path = candidates.iter().any(|c| c.issue_type == "long_path");
    assert!(
        has_long_path,
        "Should identify long-path issues in a deep chain, found: {:?}",
        candidates.iter().map(|c| &c.issue_type).collect::<Vec<_>>()
    );
}

/// Test 2: Detect connectivity imbalance — one hidden neuron starved, another overloaded.
#[test]
fn test_detects_connectivity_imbalance() {
    let creature = imbalanced_creature();

    let neuron_records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        ("h0".into(), make_records("h0", 30, 0.3, 0.1)),
        ("h1".into(), make_records("h1", 30, 0.6, 0.25)),
    ];

    let candidates = detect_topology_issues(&creature, &neuron_records, None);

    let has_imbalance = candidates
        .iter()
        .any(|c| c.issue_type == "connectivity_imbalance");
    assert!(
        has_imbalance,
        "Should detect connectivity imbalance between h0 (fan-in=1) and h1 (fan-in=3)"
    );
}

/// Test 3: Well-balanced topology produces no candidates.
#[test]
fn test_balanced_topology_no_issues() {
    let creature = balanced_creature();

    let neuron_records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        ("h0".into(), make_records("h0", 30, 0.4, 0.05)),
        ("h1".into(), make_records("h1", 30, 0.4, 0.05)),
    ];

    let candidates = detect_topology_issues(&creature, &neuron_records, None);

    assert!(
        candidates.is_empty(),
        "Balanced topology should not flag any issues, found {} candidates",
        candidates.len()
    );
}

/// Test 4: Output neurons are never flagged as topology issues.
#[test]
fn test_output_neurons_not_flagged() {
    let creature = deep_chain_creature();

    // Only provide records for the output neuron
    let neuron_records: Vec<(String, Vec<DiscoverRecord>)> =
        vec![("output-0".into(), make_records("output-0", 30, 0.5, 0.1))];

    let candidates = detect_topology_issues(&creature, &neuron_records, None);

    let flags_output = candidates.iter().any(|c| c.neuron_uuid == "output-0");
    assert!(
        !flags_output,
        "Output neurons should never be flagged as topology issues"
    );
}

/// Test 5: Input neurons are never flagged.
#[test]
fn test_input_neurons_not_flagged() {
    let creature = deep_chain_creature();

    let neuron_records: Vec<(String, Vec<DiscoverRecord>)> =
        vec![("input-0".into(), make_records("input-0", 30, 1.0, 0.0))];

    let candidates = detect_topology_issues(&creature, &neuron_records, None);

    let flags_input = candidates.iter().any(|c| c.neuron_uuid == "input-0");
    assert!(
        !flags_input,
        "Input neurons should never be flagged as topology issues"
    );
}

/// Test 6: Insufficient samples should not trigger detection.
#[test]
fn test_insufficient_samples_not_flagged() {
    let creature = deep_chain_creature();

    let neuron_records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        ("h0".into(), make_records("h0", 5, 0.5, 0.2)),
        ("h1".into(), make_records("h1", 5, 0.4, 0.15)),
    ];

    let candidates = detect_topology_issues(&creature, &neuron_records, None);

    assert!(
        candidates.is_empty(),
        "Too few samples should not trigger detection"
    );
}

/// Test 7: Empty network produces no candidates.
#[test]
fn test_empty_network_no_candidates() {
    let creature = CreatureJson {
        neurons: vec![
            NeuronJson {
                uuid: "input-0".into(),
                neuron_type: "input".into(),
                squash: "IDENTITY".into(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "output-0".into(),
                neuron_type: "output".into(),
                squash: "TANH".into(),
                bias: 0.0,
            },
        ],
        synapses: vec![SynapseJson {
            from_uuid: "input-0".into(),
            to_uuid: "output-0".into(),
            weight: 0.5,
            synapse_type: None,
        }],
        input: 1,
        output: 1,
    };

    let neuron_records: Vec<(String, Vec<DiscoverRecord>)> = vec![];

    let candidates = detect_topology_issues(&creature, &neuron_records, None);

    assert!(
        candidates.is_empty(),
        "Network with no hidden neurons should produce no topology candidates"
    );
}

/// Test 8: Candidates have positive estimated improvement.
#[test]
fn test_candidates_have_positive_improvement() {
    let creature = deep_chain_with_error_creature();

    let neuron_records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        ("h0".into(), make_records("h0", 30, 0.5, 0.2)),
        ("h1".into(), make_records("h1", 30, 0.4, 0.15)),
        ("h2".into(), make_records("h2", 30, 0.3, 0.12)),
        ("h3".into(), make_records("h3", 30, 0.2, 0.1)),
    ];

    let candidates = detect_topology_issues(&creature, &neuron_records, None);

    for c in &candidates {
        assert!(
            c.estimated_improvement > 0.0,
            "All topology candidates should have positive estimated improvement, got {} for {}",
            c.estimated_improvement,
            c.neuron_uuid
        );
    }
}

/// Test 9: Topology candidates convert to valid coordinated structural candidates.
#[test]
fn test_conversion_to_coordinated_candidates() {
    let creature = deep_chain_with_error_creature();

    let neuron_records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        ("h0".into(), make_records("h0", 30, 0.5, 0.2)),
        ("h1".into(), make_records("h1", 30, 0.4, 0.15)),
        ("h2".into(), make_records("h2", 30, 0.3, 0.12)),
        ("h3".into(), make_records("h3", 30, 0.2, 0.1)),
    ];

    let detected = detect_topology_issues(&creature, &neuron_records, None);
    assert!(!detected.is_empty(), "Should detect issues first");

    let coordinated = topology_issues_to_coordinated_candidates(&detected, &creature, None);

    assert!(
        !coordinated.is_empty(),
        "Should produce coordinated structural candidates"
    );

    for c in &coordinated {
        assert!(
            c.expected_creature_score_gain > 0.0,
            "Coordinated candidate should have positive expected gain"
        );
        assert!(
            c.comment.is_some(),
            "Coordinated candidate should have a comment"
        );
        assert!(
            !c.operations.is_empty(),
            "Coordinated candidate should have at least one operation"
        );
    }
}

/// Test 10: Long-path candidate suggests skip connection (addSynapse).
#[test]
fn test_long_path_suggests_skip_connection() {
    let creature = deep_chain_with_error_creature();

    let neuron_records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        ("h0".into(), make_records("h0", 30, 0.5, 0.2)),
        ("h1".into(), make_records("h1", 30, 0.4, 0.15)),
        ("h2".into(), make_records("h2", 30, 0.3, 0.12)),
        ("h3".into(), make_records("h3", 30, 0.2, 0.1)),
    ];

    let detected = detect_topology_issues(&creature, &neuron_records, None);
    let coordinated = topology_issues_to_coordinated_candidates(&detected, &creature, None);

    // At least one candidate should include addSynapse (skip connection)
    let has_add_synapse = coordinated.iter().any(|c| {
        let ops_json = serde_json::to_string(&c.operations).unwrap_or_default();
        ops_json.contains("addSynapse")
    });
    assert!(
        has_add_synapse,
        "Long-path topology issue should suggest addSynapse (skip connection)"
    );
}

/// Test 11: Candidates are sorted by estimated improvement (best first).
#[test]
fn test_candidates_sorted_by_improvement() {
    let creature = deep_chain_with_error_creature();

    let neuron_records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        ("h0".into(), make_records("h0", 30, 0.5, 0.2)),
        ("h1".into(), make_records("h1", 30, 0.4, 0.15)),
        ("h2".into(), make_records("h2", 30, 0.3, 0.12)),
        ("h3".into(), make_records("h3", 30, 0.2, 0.1)),
    ];

    let candidates = detect_topology_issues(&creature, &neuron_records, None);

    for window in candidates.windows(2) {
        assert!(
            window[0].estimated_improvement >= window[1].estimated_improvement,
            "Candidates should be sorted by estimated improvement (best first)"
        );
    }
}

/// Test 12: No records for any hidden neuron produces no candidates.
#[test]
fn test_no_hidden_records_no_candidates() {
    let creature = deep_chain_creature();

    let neuron_records: Vec<(String, Vec<DiscoverRecord>)> = vec![];

    let candidates = detect_topology_issues(&creature, &neuron_records, None);

    assert!(
        candidates.is_empty(),
        "No records should produce no candidates"
    );
}

/// Test 13: Network with skip connection already present should not suggest duplicate.
#[test]
fn test_existing_skip_connection_not_duplicated() {
    // Deep chain with existing skip: input-0 → h0 → h1 → h2 → output-0
    // Plus skip: h0 → output-0 already exists
    let creature = CreatureJson {
        neurons: vec![
            NeuronJson {
                uuid: "input-0".into(),
                neuron_type: "input".into(),
                squash: "IDENTITY".into(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "h0".into(),
                neuron_type: "hidden".into(),
                squash: "TANH".into(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "h1".into(),
                neuron_type: "hidden".into(),
                squash: "TANH".into(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "h2".into(),
                neuron_type: "hidden".into(),
                squash: "TANH".into(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "output-0".into(),
                neuron_type: "output".into(),
                squash: "TANH".into(),
                bias: 0.0,
            },
        ],
        synapses: vec![
            SynapseJson {
                from_uuid: "input-0".into(),
                to_uuid: "h0".into(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "h0".into(),
                to_uuid: "h1".into(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "h1".into(),
                to_uuid: "h2".into(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "h2".into(),
                to_uuid: "output-0".into(),
                weight: 0.5,
                synapse_type: None,
            },
            // Skip connection already exists
            SynapseJson {
                from_uuid: "h0".into(),
                to_uuid: "output-0".into(),
                weight: 0.3,
                synapse_type: None,
            },
        ],
        input: 1,
        output: 1,
    };

    let neuron_records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        ("h0".into(), make_records("h0", 30, 0.5, 0.15)),
        ("h1".into(), make_records("h1", 30, 0.4, 0.1)),
        ("h2".into(), make_records("h2", 30, 0.3, 0.08)),
    ];

    let detected = detect_topology_issues(&creature, &neuron_records, None);
    let coordinated = topology_issues_to_coordinated_candidates(&detected, &creature, None);

    // If skip connections are suggested, none should duplicate h0→output-0
    for c in &coordinated {
        let ops_json = serde_json::to_string(&c.operations).unwrap_or_default();
        let duplicates_skip = ops_json.contains("\"from_neuron_uuid\":\"h0\"")
            && ops_json.contains("\"to_neuron_uuid\":\"output-0\"")
            && ops_json.contains("addSynapse");
        assert!(
            !duplicates_skip,
            "Should not suggest skip connection that already exists (h0→output-0)"
        );
    }
}
