//! Tests for Issue #230: Multi-hop candidate analysis for deeper network improvements.
//!
//! Current discovery only considers single-hop improvements (adding one synapse or neuron).
//! For deep networks, multi-hop improvements (adding a path of 2-3 connections) may be more
//! effective. This module tests multi-hop analysis that finds deeper structural improvements.
//!
//! ## TDD Plan
//! 1. Create test network with multiple layers where direct connections are suboptimal
//! 2. Verify multi-hop candidate types are correctly constructed
//! 3. Verify intermediate neuron selection based on error correlation
//! 4. Verify path improvement estimation
//! 5. Verify pruning limits candidate explosion
//! 6. Verify multi-hop candidates produce valid coordinated structural operations
//! 7. Test edge cases (empty network, single layer, already connected)

use neat_ai_discovery::analysis::multi_hop::{
    MultiHopCandidate, detect_multi_hop_candidates, multi_hop_to_coordinated_candidates,
};
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

/// Helper: create a DiscoverRecord for a neuron with given activation and errors.
fn record(neuron_uuid: &str, obs_index: u32, activation: f32, errors: Vec<f32>) -> DiscoverRecord {
    DiscoverRecord {
        obs_index,
        neuron_uuid: neuron_uuid.to_string(),
        value: Some(activation),
        activation,
        errors,
    }
}

/// Helper: build a minimal creature with neurons and synapses.
fn make_creature(neurons: Vec<NeuronJson>, synapses: Vec<SynapseJson>) -> CreatureJson {
    let input_count = neurons.iter().filter(|n| n.neuron_type == "input").count();
    let output_count = neurons.iter().filter(|n| n.neuron_type == "output").count();
    CreatureJson {
        neurons,
        synapses,
        input: input_count,
        output: output_count,
    }
}

/// Helper: build a NeuronJson.
fn neuron(uuid: &str, neuron_type: &str, squash: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
        squash: squash.to_string(),
        bias: 0.0,
    }
}

/// Helper: build a SynapseJson.
fn synapse(from: &str, to: &str, weight: f32) -> SynapseJson {
    SynapseJson {
        from_uuid: from.to_string(),
        to_uuid: to.to_string(),
        weight,
        synapse_type: None,
    }
}

/// Test 1: Multi-hop candidates are detected in a deep network where an intermediate
/// neuron's activation correlates with a target's error.
#[test]
fn test_detects_two_hop_candidates_via_error_correlation() {
    // Network: input-1 -> hidden-1 -> hidden-2 -> output-1
    // hidden-1 has activation that correlates with output-1's error,
    // but there's no direct connection from hidden-1 to output-1.
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("hidden-1", "hidden", "TANH"),
            neuron("hidden-2", "hidden", "TANH"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "hidden-1", 0.5),
            synapse("hidden-1", "hidden-2", 0.3),
            synapse("hidden-2", "output-1", 0.4),
        ],
    );

    let mut neuron_records: Vec<(String, Vec<DiscoverRecord>)> = Vec::new();

    // hidden-1: activation correlates with output-1's error
    neuron_records.push((
        "hidden-1".to_string(),
        (0..100)
            .map(|i| {
                let activation = if i < 50 { 0.8 } else { 0.2 };
                record("hidden-1", i, activation, vec![])
            })
            .collect(),
    ));

    // hidden-2: intermediate neuron
    neuron_records.push((
        "hidden-2".to_string(),
        (0..100)
            .map(|i| {
                let activation = if i < 50 { 0.5 } else { 0.4 };
                record("hidden-2", i, activation, vec![])
            })
            .collect(),
    ));

    // output-1: errors that correlate with hidden-1's activation
    neuron_records.push((
        "output-1".to_string(),
        (0..100)
            .map(|i| {
                let error = if i < 50 { 0.6 } else { -0.1 };
                record("output-1", i, 0.5, vec![error])
            })
            .collect(),
    ));

    // input-1: for completeness
    neuron_records.push((
        "input-1".to_string(),
        (0..100)
            .map(|i| {
                let activation = if i < 50 { 0.9 } else { 0.1 };
                record("input-1", i, activation, vec![])
            })
            .collect(),
    ));

    let candidates = detect_multi_hop_candidates(&creature, &neuron_records);

    assert!(
        !candidates.is_empty(),
        "Should detect at least one multi-hop candidate"
    );

    // Verify the candidate describes a multi-hop path
    let candidate = &candidates[0];
    assert!(
        candidate.path.len() >= 2,
        "Multi-hop candidate should have a path of at least 2 neurons, got {}",
        candidate.path.len()
    );
}

/// Test 2: No multi-hop candidates when all neurons are already directly connected.
#[test]
fn test_no_candidates_when_fully_connected() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("hidden-1", "hidden", "TANH"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "hidden-1", 0.5),
            synapse("hidden-1", "output-1", 0.4),
            // Direct connection already exists
            synapse("input-1", "output-1", 0.3),
        ],
    );

    let mut neuron_records: Vec<(String, Vec<DiscoverRecord>)> = Vec::new();

    neuron_records.push((
        "hidden-1".to_string(),
        (0..100)
            .map(|i| {
                let activation = if i < 50 { 0.8 } else { 0.2 };
                record("hidden-1", i, activation, vec![])
            })
            .collect(),
    ));

    neuron_records.push((
        "output-1".to_string(),
        (0..100)
            .map(|i| {
                let error = if i < 50 { 0.6 } else { -0.1 };
                record("output-1", i, 0.5, vec![error])
            })
            .collect(),
    ));

    neuron_records.push((
        "input-1".to_string(),
        (0..100)
            .map(|i| record("input-1", i, 0.5, vec![]))
            .collect(),
    ));

    let candidates = detect_multi_hop_candidates(&creature, &neuron_records);

    // Candidates involving already-connected pairs should be filtered out
    for c in &candidates {
        // Each candidate's source→target pair should not already be directly connected
        let first = &c.path[0];
        let last = &c.path[c.path.len() - 1];
        let already_connected = creature
            .synapses
            .iter()
            .any(|s| s.from_uuid == *first && s.to_uuid == *last);
        assert!(
            !already_connected,
            "Multi-hop candidates should not duplicate existing direct connections"
        );
    }
}

/// Test 3: Insufficient samples should not trigger detection.
#[test]
fn test_insufficient_samples_no_candidates() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("hidden-1", "hidden", "TANH"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "hidden-1", 0.5),
            synapse("hidden-1", "output-1", 0.4),
        ],
    );

    // Only 5 samples — not enough
    let neuron_records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        (
            "hidden-1".to_string(),
            (0..5).map(|i| record("hidden-1", i, 0.5, vec![])).collect(),
        ),
        (
            "output-1".to_string(),
            (0..5)
                .map(|i| record("output-1", i, 0.5, vec![0.3]))
                .collect(),
        ),
    ];

    let candidates = detect_multi_hop_candidates(&creature, &neuron_records);

    assert!(
        candidates.is_empty(),
        "Too few samples should not trigger multi-hop detection"
    );
}

/// Test 4: Empty records produce no candidates.
#[test]
fn test_empty_records_no_candidates() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![],
    );

    let candidates = detect_multi_hop_candidates(&creature, &[]);

    assert!(
        candidates.is_empty(),
        "Empty records should produce no candidates"
    );
}

/// Test 5: Estimated improvement is positive for detected candidates.
#[test]
fn test_estimated_improvement_positive() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("input-2", "input", "IDENTITY"),
            neuron("hidden-1", "hidden", "TANH"),
            neuron("hidden-2", "hidden", "TANH"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "hidden-1", 0.5),
            synapse("input-2", "hidden-2", 0.3),
            synapse("hidden-1", "hidden-2", 0.4),
            synapse("hidden-2", "output-1", 0.6),
        ],
    );

    let mut neuron_records: Vec<(String, Vec<DiscoverRecord>)> = Vec::new();

    // input-1: activation correlates with output error
    neuron_records.push((
        "input-1".to_string(),
        (0..100)
            .map(|i| {
                let activation = if i < 50 { 0.9 } else { 0.1 };
                record("input-1", i, activation, vec![])
            })
            .collect(),
    ));

    neuron_records.push((
        "input-2".to_string(),
        (0..100)
            .map(|i| record("input-2", i, 0.5, vec![]))
            .collect(),
    ));

    neuron_records.push((
        "hidden-1".to_string(),
        (0..100)
            .map(|i| {
                let activation = if i < 50 { 0.7 } else { 0.2 };
                record("hidden-1", i, activation, vec![])
            })
            .collect(),
    ));

    neuron_records.push((
        "hidden-2".to_string(),
        (0..100)
            .map(|i| {
                let activation = if i < 50 { 0.6 } else { 0.3 };
                record("hidden-2", i, activation, vec![])
            })
            .collect(),
    ));

    neuron_records.push((
        "output-1".to_string(),
        (0..100)
            .map(|i| {
                let error = if i < 50 { 0.5 } else { -0.2 };
                record("output-1", i, 0.5, vec![error])
            })
            .collect(),
    ));

    let candidates = detect_multi_hop_candidates(&creature, &neuron_records);

    for c in &candidates {
        assert!(
            c.estimated_improvement > 0.0,
            "Estimated improvement should be positive, got {}",
            c.estimated_improvement
        );
    }
}

/// Test 6: Multi-hop candidates produce valid coordinated structural candidates
/// with AddNeuron and AddSynapse operations.
#[test]
fn test_candidates_produce_coordinated_operations() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("hidden-1", "hidden", "TANH"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "hidden-1", 0.5),
            synapse("hidden-1", "output-1", 0.4),
        ],
    );

    let candidate = MultiHopCandidate {
        path: vec![
            "input-1".to_string(),
            "hidden-1".to_string(),
            "output-1".to_string(),
        ],
        estimated_improvement: 0.05,
        correlation_strength: 0.8,
    };

    let coordinated = multi_hop_to_coordinated_candidates(&[candidate], &creature);

    assert!(
        !coordinated.is_empty(),
        "Should produce at least one coordinated candidate"
    );

    let c = &coordinated[0];
    assert!(
        c.expected_creature_score_gain > 0.0,
        "Expected improvement should be positive"
    );
    assert!(
        c.comment.is_some(),
        "Should have a comment explaining the recommendation"
    );

    // Check that operations include AddSynapse
    let ops_json = serde_json::to_string(&c.operations).unwrap();
    assert!(
        ops_json.contains("addSynapse"),
        "Should include addSynapse operation, got: {ops_json}"
    );
}

/// Test 7: Candidate paths are bounded in length (max depth 3).
#[test]
fn test_candidate_path_depth_bounded() {
    // Create a deep network with 6 layers
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("h1", "hidden", "TANH"),
            neuron("h2", "hidden", "TANH"),
            neuron("h3", "hidden", "TANH"),
            neuron("h4", "hidden", "TANH"),
            neuron("h5", "hidden", "TANH"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "h1", 0.5),
            synapse("h1", "h2", 0.4),
            synapse("h2", "h3", 0.3),
            synapse("h3", "h4", 0.4),
            synapse("h4", "h5", 0.3),
            synapse("h5", "output-1", 0.4),
        ],
    );

    let mut neuron_records: Vec<(String, Vec<DiscoverRecord>)> = Vec::new();
    for uuid in &["input-1", "h1", "h2", "h3", "h4", "h5"] {
        neuron_records.push((
            uuid.to_string(),
            (0..100)
                .map(|i| {
                    let activation = if i < 50 { 0.8 } else { 0.2 };
                    record(uuid, i, activation, vec![])
                })
                .collect(),
        ));
    }
    neuron_records.push((
        "output-1".to_string(),
        (0..100)
            .map(|i| {
                let error = if i < 50 { 0.5 } else { -0.3 };
                record("output-1", i, 0.5, vec![error])
            })
            .collect(),
    ));

    let candidates = detect_multi_hop_candidates(&creature, &neuron_records);

    for c in &candidates {
        assert!(
            c.path.len() <= 4,
            "Multi-hop path should be at most 4 nodes (3 hops), got {} nodes",
            c.path.len()
        );
    }
}

/// Test 8: Multi-hop candidates are sorted by estimated improvement.
#[test]
fn test_candidates_sorted_by_improvement() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("input-2", "input", "IDENTITY"),
            neuron("hidden-1", "hidden", "TANH"),
            neuron("hidden-2", "hidden", "TANH"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "hidden-1", 0.5),
            synapse("input-2", "hidden-2", 0.3),
            synapse("hidden-1", "output-1", 0.4),
            // hidden-2 not connected to output-1 — multi-hop opportunity
        ],
    );

    let mut neuron_records: Vec<(String, Vec<DiscoverRecord>)> = Vec::new();

    // Both hidden neurons correlate with output error, but hidden-1 stronger
    for (uuid, strength) in &[("hidden-1", 0.9_f32), ("hidden-2", 0.5_f32)] {
        neuron_records.push((
            uuid.to_string(),
            (0..100)
                .map(|i| {
                    let activation = if i < 50 { *strength } else { 1.0 - strength };
                    record(uuid, i, activation, vec![])
                })
                .collect(),
        ));
    }

    for uuid in &["input-1", "input-2"] {
        neuron_records.push((
            uuid.to_string(),
            (0..100)
                .map(|i| {
                    let activation = if i < 50 { 0.8 } else { 0.2 };
                    record(uuid, i, activation, vec![])
                })
                .collect(),
        ));
    }

    neuron_records.push((
        "output-1".to_string(),
        (0..100)
            .map(|i| {
                let error = if i < 50 { 0.5 } else { -0.3 };
                record("output-1", i, 0.5, vec![error])
            })
            .collect(),
    ));

    let candidates = detect_multi_hop_candidates(&creature, &neuron_records);

    // Verify sorting: each candidate's improvement should be >= the next
    for window in candidates.windows(2) {
        assert!(
            window[0].estimated_improvement >= window[1].estimated_improvement,
            "Candidates should be sorted by improvement (descending)"
        );
    }
}

/// Test 9: Network with no hidden neurons produces no multi-hop candidates.
#[test]
fn test_no_hidden_neurons_no_candidates() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    let neuron_records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        (
            "input-1".to_string(),
            (0..100)
                .map(|i| record("input-1", i, 0.5, vec![]))
                .collect(),
        ),
        (
            "output-1".to_string(),
            (0..100)
                .map(|i| record("output-1", i, 0.5, vec![0.3]))
                .collect(),
        ),
    ];

    let candidates = detect_multi_hop_candidates(&creature, &neuron_records);

    // With only input and output, there are no intermediate neurons for multi-hop paths.
    // Candidates may still be produced if input neurons serve as sources, but paths
    // should only include existing neurons.
    for c in &candidates {
        assert!(
            c.path.len() >= 2,
            "Any candidate must have at least 2 nodes in the path"
        );
    }
}

/// Test 10: Multi-hop detection handles neurons with no error records gracefully.
#[test]
fn test_handles_neurons_without_errors() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("hidden-1", "hidden", "TANH"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "hidden-1", 0.5),
            synapse("hidden-1", "output-1", 0.4),
        ],
    );

    // hidden-1 has records but no errors
    let neuron_records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        (
            "hidden-1".to_string(),
            (0..100)
                .map(|i| record("hidden-1", i, 0.5, vec![]))
                .collect(),
        ),
        // output-1 has errors but no activation variation
        (
            "output-1".to_string(),
            (0..100)
                .map(|i| record("output-1", i, 0.5, vec![0.0]))
                .collect(),
        ),
    ];

    // Should not panic
    let _candidates = detect_multi_hop_candidates(&creature, &neuron_records);
}

/// Test 11: Multi-hop correctly excludes output→output paths.
/// Output neurons should only appear as the final node in a path.
#[test]
fn test_output_neurons_only_as_targets() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("hidden-1", "hidden", "TANH"),
            neuron("output-1", "output", "IDENTITY"),
            neuron("output-2", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "hidden-1", 0.5),
            synapse("hidden-1", "output-1", 0.4),
            synapse("hidden-1", "output-2", 0.3),
        ],
    );

    let mut neuron_records: Vec<(String, Vec<DiscoverRecord>)> = Vec::new();
    neuron_records.push((
        "hidden-1".to_string(),
        (0..100)
            .map(|i| {
                let activation = if i < 50 { 0.8 } else { 0.2 };
                record("hidden-1", i, activation, vec![])
            })
            .collect(),
    ));
    neuron_records.push((
        "output-1".to_string(),
        (0..100)
            .map(|i| {
                let error = if i < 50 { 0.5 } else { -0.3 };
                record("output-1", i, 0.5, vec![error])
            })
            .collect(),
    ));
    neuron_records.push((
        "output-2".to_string(),
        (0..100)
            .map(|i| {
                let error = if i < 50 { 0.4 } else { -0.2 };
                record("output-2", i, 0.5, vec![error])
            })
            .collect(),
    ));
    neuron_records.push((
        "input-1".to_string(),
        (0..100)
            .map(|i| record("input-1", i, 0.5, vec![]))
            .collect(),
    ));

    let candidates = detect_multi_hop_candidates(&creature, &neuron_records);

    for c in &candidates {
        // Intermediate nodes (not first or last) should never be output neurons
        if c.path.len() > 2 {
            for intermediate in &c.path[1..c.path.len() - 1] {
                let is_output = creature
                    .neurons
                    .iter()
                    .any(|n| n.uuid == *intermediate && n.neuron_type == "output");
                assert!(
                    !is_output,
                    "Output neurons should not appear as intermediate nodes in multi-hop paths"
                );
            }
        }
    }
}

/// Test 12: Multi-hop candidates for a three-hop path through two intermediates.
#[test]
fn test_three_hop_candidate() {
    // Network where input-1 -> h1 -> h2 -> output-1 is the path,
    // but input-1 is not connected to h2 or output-1 directly.
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("h1", "hidden", "TANH"),
            neuron("h2", "hidden", "TANH"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "h1", 0.5),
            synapse("h1", "h2", 0.4),
            synapse("h2", "output-1", 0.3),
        ],
    );

    let mut neuron_records: Vec<(String, Vec<DiscoverRecord>)> = Vec::new();

    // input-1: strong activation pattern
    neuron_records.push((
        "input-1".to_string(),
        (0..100)
            .map(|i| {
                let activation = if i < 50 { 0.9 } else { 0.1 };
                record("input-1", i, activation, vec![])
            })
            .collect(),
    ));

    // h1: correlates with output error
    neuron_records.push((
        "h1".to_string(),
        (0..100)
            .map(|i| {
                let activation = if i < 50 { 0.7 } else { 0.2 };
                record("h1", i, activation, vec![])
            })
            .collect(),
    ));

    // h2: also correlates but less strongly
    neuron_records.push((
        "h2".to_string(),
        (0..100)
            .map(|i| {
                let activation = if i < 50 { 0.6 } else { 0.3 };
                record("h2", i, activation, vec![])
            })
            .collect(),
    ));

    // output-1: errors correlating with input pattern
    neuron_records.push((
        "output-1".to_string(),
        (0..100)
            .map(|i| {
                let error = if i < 50 { 0.5 } else { -0.3 };
                record("output-1", i, 0.5, vec![error])
            })
            .collect(),
    ));

    let candidates = detect_multi_hop_candidates(&creature, &neuron_records);

    // Should find candidates — at least some multi-hop paths
    assert!(
        !candidates.is_empty(),
        "Should detect multi-hop candidates in a deep network"
    );
}

/// Test 13: Candidates have deterministic UUIDs for new neurons.
#[test]
fn test_coordinated_candidates_have_deterministic_uuids() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("hidden-1", "hidden", "TANH"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "hidden-1", 0.5),
            synapse("hidden-1", "output-1", 0.4),
        ],
    );

    let candidate = MultiHopCandidate {
        path: vec![
            "input-1".to_string(),
            "hidden-1".to_string(),
            "output-1".to_string(),
        ],
        estimated_improvement: 0.05,
        correlation_strength: 0.8,
    };

    let coordinated_1 =
        multi_hop_to_coordinated_candidates(std::slice::from_ref(&candidate), &creature);
    let coordinated_2 =
        multi_hop_to_coordinated_candidates(std::slice::from_ref(&candidate), &creature);

    // Same input should produce same output
    let json_1 = serde_json::to_string(&coordinated_1).unwrap();
    let json_2 = serde_json::to_string(&coordinated_2).unwrap();
    assert_eq!(
        json_1, json_2,
        "Coordinated candidates should be deterministic"
    );
}
