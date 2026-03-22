//! Tests for Issue #908: Fan-in candidate generation (multiple inputs converging
//! to one hidden neuron).
//!
//! ## TDD Plan
//! 1. Verify fan-in candidates are detected when correlated input pairs exist
//! 2. Verify no candidates for uncorrelated inputs
//! 3. Verify non-linear activations are used (not IDENTITY)
//! 4. Verify coordinated structural operations are valid
//! 5. Verify deterministic UUID generation
//! 6. Verify edge cases (empty records, insufficient samples, single input)
//! 7. Verify candidates are sorted by estimated improvement
//! 8. Verify redundant (highly correlated) inputs are filtered out

use neat_ai_discovery::analysis::recommendation::fan_in::{
    FanInCandidate, detect_fan_in_candidates, fan_in_to_coordinated_candidates,
};
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

/// Helper: create a `DiscoverRecord`.
fn record(neuron_uuid: &str, obs_index: u32, activation: f32, errors: Vec<f32>) -> DiscoverRecord {
    DiscoverRecord {
        obs_index,
        neuron_uuid: neuron_uuid.to_string(),
        value: Some(activation),
        activation,
        errors,
    }
}

/// Helper: build a minimal creature.
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

/// Helper: build a `NeuronJson`.
fn neuron(uuid: &str, neuron_type: &str, squash: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
        squash: squash.to_string(),
        bias: 0.0,
    }
}

/// Helper: build a `SynapseJson`.
fn synapse(from: &str, to: &str, weight: f32) -> SynapseJson {
    SynapseJson {
        from_uuid: from.to_string(),
        to_uuid: to.to_string(),
        weight,
        synapse_type: None,
    }
}

/// Build a network with two inputs and one output where both inputs' activations
/// correlate with the output error but have complementary patterns (low mutual
/// correlation). The error depends equally on both inputs so the combined
/// regression is significantly better than either individual.
fn make_fan_in_network_and_records() -> (CreatureJson, Vec<(String, Vec<DiscoverRecord>)>) {
    let creature = make_creature(
        vec![
            neuron("input-a", "input", "IDENTITY"),
            neuron("input-b", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-a", "output-1", 0.1),
            synapse("input-b", "output-1", 0.1),
        ],
    );

    let n = 100_u32;
    let mut records: Vec<(String, Vec<DiscoverRecord>)> = Vec::new();

    // input-a: varies based on first/second half (uncorrelated with input-b).
    records.push((
        "input-a".to_string(),
        (0..n)
            .map(|i| {
                let act = if i < n / 2 { 0.9 } else { 0.1 };
                record("input-a", i, act, vec![])
            })
            .collect(),
    ));

    // input-b: varies based on even/odd index (uncorrelated with input-a).
    records.push((
        "input-b".to_string(),
        (0..n)
            .map(|i| {
                let act = if i % 2 == 0 { 0.9 } else { 0.1 };
                record("input-b", i, act, vec![])
            })
            .collect(),
    ));

    // output-1: error depends equally on both inputs so neither alone is sufficient.
    // error ≈ 0.5 * act_a + 0.5 * act_b - 0.5 (centred around zero).
    records.push((
        "output-1".to_string(),
        (0..n)
            .map(|i| {
                let act_a = if i < n / 2 { 0.9 } else { 0.1 };
                let act_b = if i % 2 == 0 { 0.9 } else { 0.1 };
                let error = 0.5 * act_a + 0.5 * act_b - 0.5;
                record("output-1", i, 0.5, vec![error])
            })
            .collect(),
    ));

    (creature, records)
}

/// Test 1: Fan-in candidates are detected when correlated input pairs exist.
#[test]
fn test_detects_fan_in_candidates_for_correlated_inputs() {
    let (creature, records) = make_fan_in_network_and_records();
    let candidates = detect_fan_in_candidates(&creature, &records);

    assert!(
        !candidates.is_empty(),
        "Should detect at least one fan-in candidate"
    );

    let c = &candidates[0];
    assert_eq!(
        c.input_uuids.len(),
        2,
        "Fan-in candidate should have exactly 2 inputs"
    );
    assert!(
        c.input_uuids.contains(&"input-a".to_string()),
        "Should include input-a"
    );
    assert!(
        c.input_uuids.contains(&"input-b".to_string()),
        "Should include input-b"
    );
    assert_eq!(c.target_uuid, "output-1");
}

/// Test 2: No candidates when inputs are uncorrelated with target error.
#[test]
fn test_no_candidates_for_uncorrelated_inputs() {
    let creature = make_creature(
        vec![
            neuron("input-a", "input", "IDENTITY"),
            neuron("input-b", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-a", "output-1", 0.1),
            synapse("input-b", "output-1", 0.1),
        ],
    );

    let n = 100_u32;
    let mut records: Vec<(String, Vec<DiscoverRecord>)> = Vec::new();

    // Both inputs: random-like constant activations — no correlation with error.
    records.push((
        "input-a".to_string(),
        (0..n).map(|i| record("input-a", i, 0.5, vec![])).collect(),
    ));
    records.push((
        "input-b".to_string(),
        (0..n).map(|i| record("input-b", i, 0.5, vec![])).collect(),
    ));

    // Output: error with a pattern unrelated to inputs.
    records.push((
        "output-1".to_string(),
        (0..n)
            .map(|i| {
                let error = if i % 3 == 0 { 0.5 } else { -0.2 };
                record("output-1", i, 0.5, vec![error])
            })
            .collect(),
    ));

    let candidates = detect_fan_in_candidates(&creature, &records);

    assert!(
        candidates.is_empty(),
        "Should not detect fan-in candidates for uncorrelated inputs"
    );
}

/// Test 3: Fan-in neurons use non-linear activations (not IDENTITY).
#[test]
fn test_fan_in_uses_non_linear_activation() {
    let (creature, records) = make_fan_in_network_and_records();
    let candidates = detect_fan_in_candidates(&creature, &records);

    assert!(!candidates.is_empty());

    for c in &candidates {
        assert_ne!(
            c.activation, "IDENTITY",
            "Fan-in neurons should not use IDENTITY activation"
        );
        assert!(
            c.activation == "TANH" || c.activation == "GELU",
            "Fan-in neurons should use TANH or GELU, got {}",
            c.activation
        );
    }
}

/// Test 4: Coordinated structural operations are valid.
#[test]
fn test_fan_in_produces_valid_coordinated_operations() {
    let (creature, records) = make_fan_in_network_and_records();
    let candidates = detect_fan_in_candidates(&creature, &records);

    assert!(!candidates.is_empty());

    let coordinated = fan_in_to_coordinated_candidates(&candidates, &creature);

    assert!(
        !coordinated.is_empty(),
        "Should produce coordinated candidates"
    );

    let c = &coordinated[0];
    assert!(
        c.expected_creature_score_gain > 0.0,
        "Expected improvement should be positive"
    );
    assert!(c.comment.is_some(), "Should have a descriptive comment");

    // Check operations: should have AddNeuron + 2 AddSynapse (inputs) + 1 AddSynapse (output).
    let ops_json = serde_json::to_string(&c.operations).unwrap();
    assert!(
        ops_json.contains("addNeuron"),
        "Should include addNeuron operation: {ops_json}"
    );
    assert!(
        ops_json.contains("addSynapse"),
        "Should include addSynapse operations: {ops_json}"
    );

    // Verify operation count: 1 AddNeuron + 3 AddSynapse = 4 operations.
    assert_eq!(
        c.operations.len(),
        4,
        "Fan-in candidate should have 4 operations (1 AddNeuron + 3 AddSynapse)"
    );
}

/// Test 5: Deterministic UUID generation — same inputs produce same UUID.
#[test]
fn test_fan_in_deterministic_uuid() {
    let (creature, records) = make_fan_in_network_and_records();
    let candidates = detect_fan_in_candidates(&creature, &records);

    assert!(!candidates.is_empty());

    let coord_1 = fan_in_to_coordinated_candidates(&candidates, &creature);
    let coord_2 = fan_in_to_coordinated_candidates(&candidates, &creature);

    let json_1 = serde_json::to_string(&coord_1).unwrap();
    let json_2 = serde_json::to_string(&coord_2).unwrap();
    assert_eq!(
        json_1, json_2,
        "Coordinated candidates should be deterministic"
    );
}

/// Test 6: Empty records produce no candidates.
#[test]
fn test_fan_in_empty_records() {
    let creature = make_creature(
        vec![
            neuron("input-a", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![synapse("input-a", "output-1", 0.5)],
    );

    let candidates = detect_fan_in_candidates(&creature, &[]);
    assert!(
        candidates.is_empty(),
        "Empty records should produce no candidates"
    );
}

/// Test 7: Insufficient samples produce no candidates.
#[test]
fn test_fan_in_insufficient_samples() {
    let creature = make_creature(
        vec![
            neuron("input-a", "input", "IDENTITY"),
            neuron("input-b", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-a", "output-1", 0.1),
            synapse("input-b", "output-1", 0.1),
        ],
    );

    // Only 5 samples — below MIN_DISCOVERY_SAMPLE_COUNT.
    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        (
            "input-a".to_string(),
            (0..5).map(|i| record("input-a", i, 0.8, vec![])).collect(),
        ),
        (
            "input-b".to_string(),
            (0..5).map(|i| record("input-b", i, 0.3, vec![])).collect(),
        ),
        (
            "output-1".to_string(),
            (0..5)
                .map(|i| record("output-1", i, 0.5, vec![0.3]))
                .collect(),
        ),
    ];

    let candidates = detect_fan_in_candidates(&creature, &records);
    assert!(
        candidates.is_empty(),
        "Insufficient samples should produce no candidates"
    );
}

/// Test 8: Single input neuron — cannot form fan-in pair.
#[test]
fn test_fan_in_single_input_no_candidates() {
    let creature = make_creature(
        vec![
            neuron("input-a", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![synapse("input-a", "output-1", 0.5)],
    );

    let n = 100_u32;
    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        (
            "input-a".to_string(),
            (0..n)
                .map(|i| {
                    let act = if i < 50 { 0.9 } else { 0.1 };
                    record("input-a", i, act, vec![])
                })
                .collect(),
        ),
        (
            "output-1".to_string(),
            (0..n)
                .map(|i| {
                    let error = if i < 50 { 0.5 } else { -0.2 };
                    record("output-1", i, 0.5, vec![error])
                })
                .collect(),
        ),
    ];

    let candidates = detect_fan_in_candidates(&creature, &records);
    assert!(
        candidates.is_empty(),
        "Single input cannot form fan-in pair"
    );
}

/// Test 9: Candidates are sorted by estimated improvement (best first).
#[test]
fn test_fan_in_candidates_sorted_by_improvement() {
    // Three inputs with varying correlation strength.
    let creature = make_creature(
        vec![
            neuron("input-a", "input", "IDENTITY"),
            neuron("input-b", "input", "IDENTITY"),
            neuron("input-c", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-a", "output-1", 0.1),
            synapse("input-b", "output-1", 0.1),
            synapse("input-c", "output-1", 0.1),
        ],
    );

    let n = 100_u32;
    let mut records: Vec<(String, Vec<DiscoverRecord>)> = Vec::new();

    // input-a: strong correlation with error.
    records.push((
        "input-a".to_string(),
        (0..n)
            .map(|i| {
                let act = if i < n / 2 { 0.95 } else { 0.05 };
                record("input-a", i, act, vec![])
            })
            .collect(),
    ));

    // input-b: moderate correlation, complementary to input-a.
    records.push((
        "input-b".to_string(),
        (0..n)
            .map(|i| {
                let act = if i % 3 == 0 { 0.8 } else { 0.2 };
                record("input-b", i, act, vec![])
            })
            .collect(),
    ));

    // input-c: weaker correlation, complementary pattern.
    records.push((
        "input-c".to_string(),
        (0..n)
            .map(|i| {
                let act = if i % 5 == 0 { 0.7 } else { 0.3 };
                record("input-c", i, act, vec![])
            })
            .collect(),
    ));

    // Output error correlates with inputs.
    records.push((
        "output-1".to_string(),
        (0..n)
            .map(|i| {
                let from_a = if i < n / 2 { 0.5 } else { -0.1 };
                let from_b = if i % 3 == 0 { 0.15 } else { -0.05 };
                let from_c = if i % 5 == 0 { 0.1 } else { -0.02 };
                record("output-1", i, 0.5, vec![from_a + from_b + from_c])
            })
            .collect(),
    ));

    let candidates = detect_fan_in_candidates(&creature, &records);

    // Verify sorting: each candidate's improvement >= the next.
    for window in candidates.windows(2) {
        assert!(
            window[0].estimated_improvement >= window[1].estimated_improvement,
            "Candidates should be sorted by improvement (descending): {} >= {}",
            window[0].estimated_improvement,
            window[1].estimated_improvement,
        );
    }
}

/// Test 10: Highly correlated (redundant) inputs are filtered out.
#[test]
fn test_fan_in_filters_redundant_inputs() {
    let creature = make_creature(
        vec![
            neuron("input-a", "input", "IDENTITY"),
            neuron("input-b", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-a", "output-1", 0.1),
            synapse("input-b", "output-1", 0.1),
        ],
    );

    let n = 100_u32;
    let mut records: Vec<(String, Vec<DiscoverRecord>)> = Vec::new();

    // Both inputs have nearly identical activation patterns (high mutual correlation).
    records.push((
        "input-a".to_string(),
        (0..n)
            .map(|i| {
                let act = if i < n / 2 { 0.9 } else { 0.1 };
                record("input-a", i, act, vec![])
            })
            .collect(),
    ));
    records.push((
        "input-b".to_string(),
        (0..n)
            .map(|i| {
                let act = if i < n / 2 { 0.88 } else { 0.12 };
                record("input-b", i, act, vec![])
            })
            .collect(),
    ));

    records.push((
        "output-1".to_string(),
        (0..n)
            .map(|i| {
                let error = if i < n / 2 { 0.5 } else { -0.2 };
                record("output-1", i, 0.5, vec![error])
            })
            .collect(),
    ));

    let candidates = detect_fan_in_candidates(&creature, &records);

    // Redundant inputs should be filtered — high mutual correlation means
    // a single-input path would suffice.
    assert!(
        candidates.is_empty(),
        "Highly correlated (redundant) inputs should be filtered out"
    );
}

/// Test 11: Fan-in candidates target hidden neurons as well as outputs.
#[test]
fn test_fan_in_targets_hidden_neurons() {
    let creature = make_creature(
        vec![
            neuron("input-a", "input", "IDENTITY"),
            neuron("input-b", "input", "IDENTITY"),
            neuron("hidden-1", "hidden", "TANH"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-a", "hidden-1", 0.1),
            synapse("hidden-1", "output-1", 0.5),
        ],
    );

    let n = 100_u32;
    let mut records: Vec<(String, Vec<DiscoverRecord>)> = Vec::new();

    records.push((
        "input-a".to_string(),
        (0..n)
            .map(|i| {
                let act = if i < n / 2 { 0.9 } else { 0.1 };
                record("input-a", i, act, vec![])
            })
            .collect(),
    ));

    records.push((
        "input-b".to_string(),
        (0..n)
            .map(|i| {
                let act = if i % 2 == 0 { 0.9 } else { 0.1 };
                record("input-b", i, act, vec![])
            })
            .collect(),
    ));

    // hidden-1 has error records — can be a fan-in target.
    // Error depends equally on both inputs.
    records.push((
        "hidden-1".to_string(),
        (0..n)
            .map(|i| {
                let act_a = if i < n / 2 { 0.9 } else { 0.1 };
                let act_b = if i % 2 == 0 { 0.9 } else { 0.1 };
                let error = 0.5 * act_a + 0.5 * act_b - 0.5;
                record("hidden-1", i, 0.5, vec![error])
            })
            .collect(),
    ));

    records.push((
        "output-1".to_string(),
        (0..n)
            .map(|i| record("output-1", i, 0.5, vec![0.1]))
            .collect(),
    ));

    let candidates = detect_fan_in_candidates(&creature, &records);

    // Should find candidates targeting hidden-1.
    let hidden_targets: Vec<_> = candidates
        .iter()
        .filter(|c| c.target_uuid == "hidden-1")
        .collect();

    assert!(
        !hidden_targets.is_empty(),
        "Should detect fan-in candidates targeting hidden neurons"
    );
}

/// Test 12: Estimated improvement is always positive for detected candidates.
#[test]
fn test_fan_in_improvement_positive() {
    let (creature, records) = make_fan_in_network_and_records();
    let candidates = detect_fan_in_candidates(&creature, &records);

    for c in &candidates {
        assert!(
            c.estimated_improvement > 0.0,
            "Estimated improvement should be positive, got {}",
            c.estimated_improvement
        );
    }
}

/// Test 13: Fan-in candidate conversion with fewer than 2 inputs produces nothing.
#[test]
fn test_fan_in_conversion_requires_two_inputs() {
    let creature = make_creature(
        vec![
            neuron("input-a", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![synapse("input-a", "output-1", 0.5)],
    );

    // Manually create a degenerate candidate with only 1 input.
    let bad_candidate = FanInCandidate {
        input_uuids: vec!["input-a".to_string()],
        target_uuid: "output-1".to_string(),
        input_weights: vec![0.5],
        output_weight: 0.1,
        activation: "TANH".to_string(),
        estimated_improvement: 0.01,
        mean_input_error_correlation: 0.5,
        input_mutual_correlation: 0.0,
        sample_count: 100,
        reason: "test".to_string(),
    };

    let coordinated = fan_in_to_coordinated_candidates(&[bad_candidate], &creature);
    assert!(
        coordinated.is_empty(),
        "Should not produce coordinated candidate with fewer than 2 inputs"
    );
}
