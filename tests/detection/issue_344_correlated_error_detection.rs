//! Tests for Issue #344: Detect correlated error patterns for shared-cause identification.
//!
//! When multiple output neurons consistently err in the same direction on the same samples,
//! it suggests a missing input feature or hidden representation that would benefit all of them.
//! This module detects correlated error groups and recommends shared structural changes.
//!
//! ## TDD Plan
//! 1. Create test network where multiple outputs depend on a missing feature
//! 2. Generate samples showing correlated errors
//! 3. Verify detection identifies the correlated group
//! 4. Verify recommended structural change addresses the shared cause
//! 5. Test with independent errors (should not form groups)
//! 6. Test single output neuron (should skip — nothing to correlate)

use neat_ai_discovery::analysis::detection::correlated_error::{
    CorrelatedErrorGroup, correlated_errors_to_coordinated_candidates,
    detect_correlated_error_patterns,
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

/// Test 1: Strongly correlated errors across three output neurons are detected.
///
/// All three outputs err positively on the same samples, suggesting a shared
/// missing cause.
#[test]
fn test_detects_strongly_correlated_output_errors() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("input-2", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
            neuron("output-2", "output", "IDENTITY"),
            neuron("output-3", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "output-1", 0.5),
            synapse("input-1", "output-2", 0.3),
            synapse("input-2", "output-3", 0.4),
        ],
    );

    // Generate correlated errors: all three outputs err similarly on the same samples
    let mut output_records: Vec<(String, Vec<DiscoverRecord>)> = Vec::new();
    for output_id in &["output-1", "output-2", "output-3"] {
        let records: Vec<DiscoverRecord> = (0..100)
            .map(|i| {
                // All outputs have correlated positive error on even samples,
                // correlated negative error on odd samples
                let base_error = if i % 2 == 0 { 0.5 } else { -0.3 };
                let noise = (i as f32 * 0.001) % 0.01; // tiny noise
                record(output_id, i, 0.5, vec![base_error + noise])
            })
            .collect();
        output_records.push((output_id.to_string(), records));
    }

    let groups = detect_correlated_error_patterns(&creature, &output_records);

    assert!(
        !groups.is_empty(),
        "Should detect at least one correlated error group"
    );
    let group = &groups[0];
    assert!(
        group.output_neuron_uuids.len() >= 2,
        "Group should contain at least 2 output neurons"
    );
    assert!(
        group.mean_correlation >= 0.7,
        "Mean correlation should be high, got {}",
        group.mean_correlation
    );
}

/// Test 2: Independent (uncorrelated) errors do NOT form groups.
#[test]
fn test_independent_errors_no_groups() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
            neuron("output-2", "output", "IDENTITY"),
            neuron("output-3", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "output-1", 0.5),
            synapse("input-1", "output-2", 0.3),
            synapse("input-1", "output-3", 0.4),
        ],
    );

    // Generate independent errors: each output has a different pattern
    let mut output_records: Vec<(String, Vec<DiscoverRecord>)> = Vec::new();

    // output-1: positive on even, negative on odd
    let records_1: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let error = if i % 2 == 0 { 0.5 } else { -0.5 };
            record("output-1", i, 0.5, vec![error])
        })
        .collect();
    output_records.push(("output-1".to_string(), records_1));

    // output-2: positive on multiples of 3, negative otherwise (different pattern)
    let records_2: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let error = if i % 3 == 0 { 0.5 } else { -0.5 };
            record("output-2", i, 0.5, vec![error])
        })
        .collect();
    output_records.push(("output-2".to_string(), records_2));

    // output-3: positive on multiples of 7, negative otherwise (different pattern)
    let records_3: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let error = if i % 7 == 0 { 0.5 } else { -0.5 };
            record("output-3", i, 0.5, vec![error])
        })
        .collect();
    output_records.push(("output-3".to_string(), records_3));

    let groups = detect_correlated_error_patterns(&creature, &output_records);

    assert!(
        groups.is_empty(),
        "Independent errors should not form correlated groups"
    );
}

/// Test 3: Single output neuron — should skip (nothing to correlate).
#[test]
fn test_single_output_skips() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| record("output-1", i, 0.5, vec![0.3]))
        .collect();

    let groups = detect_correlated_error_patterns(&creature, &[("output-1".to_string(), records)]);

    assert!(
        groups.is_empty(),
        "Single output neuron should not form any groups"
    );
}

/// Test 4: Insufficient samples should not trigger detection.
#[test]
fn test_correlated_error_insufficient_samples_no_detection() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
            neuron("output-2", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "output-1", 0.5),
            synapse("input-1", "output-2", 0.3),
        ],
    );

    // Only 5 samples — not enough
    let mut output_records: Vec<(String, Vec<DiscoverRecord>)> = Vec::new();
    for output_id in &["output-1", "output-2"] {
        let records: Vec<DiscoverRecord> = (0..5)
            .map(|i| record(output_id, i, 0.5, vec![0.3]))
            .collect();
        output_records.push((output_id.to_string(), records));
    }

    let groups = detect_correlated_error_patterns(&creature, &output_records);

    assert!(
        groups.is_empty(),
        "Too few samples should not trigger detection"
    );
}

/// Test 5: Correlated error group produces coordinated structural candidate
/// with AddNeuron and AddSynapse operations.
#[test]
fn test_correlated_error_candidates_produce_coordinated_operations() {
    let group = CorrelatedErrorGroup {
        output_neuron_uuids: vec![
            "output-1".to_string(),
            "output-2".to_string(),
            "output-3".to_string(),
        ],
        mean_correlation: 0.85,
        shared_error_sample_count: 60,
        total_sample_count: 100,
        predictive_input_uuids: vec!["input-1".to_string(), "input-4".to_string()],
        estimated_improvement: 0.02,
    };

    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("input-4", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
            neuron("output-2", "output", "IDENTITY"),
            neuron("output-3", "output", "IDENTITY"),
        ],
        vec![],
    );

    let coordinated = correlated_errors_to_coordinated_candidates(&[group], &creature);

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

    // Check that operations include AddNeuron and AddSynapse
    let ops_json = serde_json::to_string(&c.operations).unwrap();
    assert!(
        ops_json.contains("addNeuron"),
        "Should include addNeuron operation, got: {ops_json}"
    );
    assert!(
        ops_json.contains("addSynapse"),
        "Should include addSynapse operation, got: {ops_json}"
    );
}

/// Test 6: Two separate correlated groups are detected independently.
#[test]
fn test_two_independent_correlated_groups() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
            neuron("output-2", "output", "IDENTITY"),
            neuron("output-3", "output", "IDENTITY"),
            neuron("output-4", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "output-1", 0.5),
            synapse("input-1", "output-2", 0.3),
            synapse("input-1", "output-3", 0.4),
            synapse("input-1", "output-4", 0.2),
        ],
    );

    let mut output_records: Vec<(String, Vec<DiscoverRecord>)> = Vec::new();

    // Group A: output-1 and output-2 correlate (even/odd pattern)
    for output_id in &["output-1", "output-2"] {
        let records: Vec<DiscoverRecord> = (0..100)
            .map(|i| {
                let error = if i % 2 == 0 { 0.5 } else { -0.5 };
                record(output_id, i, 0.5, vec![error])
            })
            .collect();
        output_records.push((output_id.to_string(), records));
    }

    // Group B: output-3 and output-4 correlate (multiples of 5 pattern — different from A)
    for output_id in &["output-3", "output-4"] {
        let records: Vec<DiscoverRecord> = (0..100)
            .map(|i| {
                let error = if i % 5 == 0 { 0.8 } else { -0.2 };
                record(output_id, i, 0.5, vec![error])
            })
            .collect();
        output_records.push((output_id.to_string(), records));
    }

    let groups = detect_correlated_error_patterns(&creature, &output_records);

    assert!(
        groups.len() >= 2,
        "Should detect at least 2 separate groups, got {}",
        groups.len()
    );
}

/// Test 7: Negatively correlated errors should NOT form a group.
/// Negative correlation means errors go in opposite directions — not a shared cause.
#[test]
fn test_negatively_correlated_errors_no_group() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
            neuron("output-2", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "output-1", 0.5),
            synapse("input-1", "output-2", -0.5),
        ],
    );

    let mut output_records: Vec<(String, Vec<DiscoverRecord>)> = Vec::new();

    // output-1: positive on even, negative on odd
    let records_1: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let error = if i % 2 == 0 { 0.5 } else { -0.5 };
            record("output-1", i, 0.5, vec![error])
        })
        .collect();
    output_records.push(("output-1".to_string(), records_1));

    // output-2: OPPOSITE pattern — negative on even, positive on odd
    let records_2: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let error = if i % 2 == 0 { -0.5 } else { 0.5 };
            record("output-2", i, 0.5, vec![error])
        })
        .collect();
    output_records.push(("output-2".to_string(), records_2));

    let groups = detect_correlated_error_patterns(&creature, &output_records);

    assert!(
        groups.is_empty(),
        "Negatively correlated errors should not form groups"
    );
}

/// Test 8: Predictive input identification — the detection should find which
/// input neuron activations predict the shared error.
#[test]
fn test_predictive_inputs_identified() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("input-2", "input", "IDENTITY"),
            neuron("input-3", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
            neuron("output-2", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "output-1", 0.5),
            synapse("input-2", "output-1", 0.3),
            synapse("input-3", "output-2", 0.4),
        ],
    );

    // input-1 is predictive: when input-1 is high, both outputs err high
    let mut output_records: Vec<(String, Vec<DiscoverRecord>)> = Vec::new();
    for output_id in &["output-1", "output-2"] {
        let records: Vec<DiscoverRecord> = (0..100)
            .map(|i| {
                // Error correlates with a hidden pattern
                let error = if i < 50 { 0.5 } else { -0.3 };
                record(output_id, i, 0.5, vec![error])
            })
            .collect();
        output_records.push((output_id.to_string(), records));
    }

    // input records: input-1 activation correlates with the error pattern
    output_records.push((
        "input-1".to_string(),
        (0..100)
            .map(|i| {
                let activation = if i < 50 { 0.9 } else { 0.1 };
                record("input-1", i, activation, vec![])
            })
            .collect(),
    ));

    // input-2: random, not predictive
    output_records.push((
        "input-2".to_string(),
        (0..100)
            .map(|i| {
                let activation = ((i as f32) * 0.7).sin() * 0.5 + 0.5;
                record("input-2", i, activation, vec![])
            })
            .collect(),
    ));

    // input-3: random, not predictive
    output_records.push((
        "input-3".to_string(),
        (0..100)
            .map(|i| {
                let activation = ((i as f32) * 1.3).cos() * 0.5 + 0.5;
                record("input-3", i, activation, vec![])
            })
            .collect(),
    ));

    let groups = detect_correlated_error_patterns(&creature, &output_records);

    assert!(!groups.is_empty(), "Should detect correlated error group");
    let group = &groups[0];
    assert!(
        !group.predictive_input_uuids.is_empty(),
        "Should identify at least one predictive input"
    );
    assert!(
        group
            .predictive_input_uuids
            .contains(&"input-1".to_string()),
        "input-1 should be identified as predictive, got: {:?}",
        group.predictive_input_uuids
    );
}

/// Test 9: Shared error sample count is correctly computed.
#[test]
fn test_shared_error_sample_count() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
            neuron("output-2", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "output-1", 0.5),
            synapse("input-1", "output-2", 0.3),
        ],
    );

    let mut output_records: Vec<(String, Vec<DiscoverRecord>)> = Vec::new();
    for output_id in &["output-1", "output-2"] {
        let records: Vec<DiscoverRecord> = (0..100)
            .map(|i| {
                // Same error pattern: positive on first 60, negative on last 40
                let error = if i < 60 { 0.5 } else { -0.3 };
                record(output_id, i, 0.5, vec![error])
            })
            .collect();
        output_records.push((output_id.to_string(), records));
    }

    let groups = detect_correlated_error_patterns(&creature, &output_records);

    assert!(!groups.is_empty(), "Should detect correlated error group");
    let group = &groups[0];
    assert_eq!(
        group.total_sample_count, 100,
        "Total sample count should match"
    );
    assert!(
        group.shared_error_sample_count > 0,
        "Should have shared error samples"
    );
}

/// Test 10: Estimated improvement is positive for detected groups.
#[test]
fn test_correlated_error_estimated_improvement_positive() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
            neuron("output-2", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "output-1", 0.5),
            synapse("input-1", "output-2", 0.3),
        ],
    );

    let mut output_records: Vec<(String, Vec<DiscoverRecord>)> = Vec::new();
    for output_id in &["output-1", "output-2"] {
        let records: Vec<DiscoverRecord> = (0..100)
            .map(|i| {
                let error = if i % 2 == 0 { 0.5 } else { -0.3 };
                record(output_id, i, 0.5, vec![error])
            })
            .collect();
        output_records.push((output_id.to_string(), records));
    }

    let groups = detect_correlated_error_patterns(&creature, &output_records);

    assert!(!groups.is_empty(), "Should detect correlated error group");
    for group in &groups {
        assert!(
            group.estimated_improvement > 0.0,
            "Estimated improvement should be positive, got {}",
            group.estimated_improvement
        );
    }
}

/// Test 11: Empty records produce no groups.
#[test]
fn test_empty_records_no_groups() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
            neuron("output-2", "output", "IDENTITY"),
        ],
        vec![],
    );

    let groups: Vec<CorrelatedErrorGroup> = detect_correlated_error_patterns(&creature, &[]);

    assert!(groups.is_empty(), "Empty records should produce no groups");
}

/// Test 12: Only output neurons are included in correlated groups (hidden neurons excluded).
#[test]
fn test_only_output_neurons_in_groups() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("hidden-1", "hidden", "TANH"),
            neuron("output-1", "output", "IDENTITY"),
            neuron("output-2", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "hidden-1", 0.5),
            synapse("hidden-1", "output-1", 0.5),
            synapse("hidden-1", "output-2", 0.3),
        ],
    );

    let mut output_records: Vec<(String, Vec<DiscoverRecord>)> = Vec::new();

    // Correlated output errors
    for output_id in &["output-1", "output-2"] {
        let records: Vec<DiscoverRecord> = (0..100)
            .map(|i| {
                let error = if i % 2 == 0 { 0.5 } else { -0.3 };
                record(output_id, i, 0.5, vec![error])
            })
            .collect();
        output_records.push((output_id.to_string(), records));
    }

    // Hidden neuron records (should not be in any group)
    let hidden_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let error = if i % 2 == 0 { 0.5 } else { -0.3 };
            record("hidden-1", i, 0.5, vec![error])
        })
        .collect();
    output_records.push(("hidden-1".to_string(), hidden_records));

    let groups = detect_correlated_error_patterns(&creature, &output_records);

    assert!(!groups.is_empty(), "Should detect correlated group");
    for group in &groups {
        for uuid in &group.output_neuron_uuids {
            assert!(
                !uuid.starts_with("hidden"),
                "Hidden neurons should not appear in correlated error groups"
            );
        }
    }
}

/// Test 13: Multi-error outputs — correlation computed per error index.
/// When output neurons have multiple error values, we correlate based on the
/// first (primary) error value.
#[test]
fn test_multi_error_outputs_use_first_error() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
            neuron("output-2", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "output-1", 0.5),
            synapse("input-1", "output-2", 0.3),
        ],
    );

    let mut output_records: Vec<(String, Vec<DiscoverRecord>)> = Vec::new();
    for output_id in &["output-1", "output-2"] {
        let records: Vec<DiscoverRecord> = (0..100)
            .map(|i| {
                let primary_error = if i % 2 == 0 { 0.5 } else { -0.3 };
                // Second error is uncorrelated noise
                let secondary_error = ((i as f32) * 0.37).sin();
                record(output_id, i, 0.5, vec![primary_error, secondary_error])
            })
            .collect();
        output_records.push((output_id.to_string(), records));
    }

    let groups = detect_correlated_error_patterns(&creature, &output_records);

    assert!(
        !groups.is_empty(),
        "Should detect correlated group from primary error"
    );
}
