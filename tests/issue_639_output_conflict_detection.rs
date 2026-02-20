//! Tests for Issue #639: Per-output error disaggregation detector for hidden neurons.
//!
//! Detects hidden neurons with conflicting per-output error contributions.
//! A hidden neuron might reduce error on output-0 but increase error on output-1,
//! meaning its net positive effect masks harm to a specific output.
//!
//! ## TDD Plan
//! 1. Hidden neuron helping one output but harming another SHOULD be flagged
//! 2. Hidden neuron consistently helpful across all outputs should NOT be flagged
//! 3. Only hidden neurons are analysed (input/output neurons excluded)
//! 4. Insufficient samples produce no detections
//! 5. Single-output networks produce no detections (no conflict possible)
//! 6. Produces valid coordinated structural candidates
//! 7. Multiple conflicting hidden neurons — all flagged
//! 8. Hidden neuron with mixed but weak conflicts should NOT be flagged
//! 9. Results sorted by conflict severity (worst first)
//! 10. Empty records produce no detections

use neat_ai_discovery::analysis::detection::output_conflict::{
    OutputConflictNeuron, detect_output_conflict_neurons,
    output_conflicts_to_coordinated_candidates,
};
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

/// Helper: create a DiscoverRecord with multi-output errors.
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

fn neuron(uuid: &str, neuron_type: &str, squash: &str, bias: f32) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
        squash: squash.to_string(),
        bias,
    }
}

fn synapse(from: &str, to: &str, weight: f32) -> SynapseJson {
    SynapseJson {
        from_uuid: from.to_string(),
        to_uuid: to.to_string(),
        weight,
        synapse_type: None,
    }
}

/// Build a two-output creature with one hidden neuron.
fn two_output_creature() -> CreatureJson {
    make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY", 0.0),
            neuron("hidden-1", "hidden", "TANH", 0.0),
            neuron("output-0", "output", "TANH", 0.0),
            neuron("output-1", "output", "TANH", 0.0),
        ],
        vec![
            synapse("input-1", "hidden-1", 0.5),
            synapse("hidden-1", "output-0", 0.8),
            synapse("hidden-1", "output-1", -0.3),
        ],
    )
}

/// Test 1: Hidden neuron that helps output-0 but harms output-1 should be flagged.
///
/// When a hidden neuron is active, error on output-0 decreases (negative error contribution)
/// but error on output-1 increases (positive error contribution). The sign conflict should
/// be detected.
#[test]
fn test_conflicting_hidden_neuron_detected() {
    let creature = two_output_creature();

    // hidden-1: when active, reduces error on output-0 but increases error on output-1
    let records: Vec<DiscoverRecord> = (0..50)
        .map(|i| {
            let activation = 0.3 + (i as f32 / 49.0) * 0.4;
            // errors[0] negative (helping), errors[1] positive (harming)
            record("hidden-1", i, activation, vec![-0.5, 0.3])
        })
        .collect();

    let neuron_records = vec![("hidden-1".to_string(), records)];

    let detected = detect_output_conflict_neurons(&creature, &neuron_records);

    assert!(
        !detected.is_empty(),
        "Hidden neuron helping output-0 but harming output-1 should be flagged"
    );
    assert_eq!(detected[0].neuron_uuid, "hidden-1");
}

/// Test 2: Hidden neuron consistently helpful across all outputs should NOT be flagged.
#[test]
fn test_consistently_helpful_not_flagged() {
    let creature = two_output_creature();

    // hidden-1: reduces error on both outputs (both negative)
    let records: Vec<DiscoverRecord> = (0..50)
        .map(|i| {
            let activation = 0.3 + (i as f32 / 49.0) * 0.4;
            record("hidden-1", i, activation, vec![-0.3, -0.2])
        })
        .collect();

    let neuron_records = vec![("hidden-1".to_string(), records)];

    let detected = detect_output_conflict_neurons(&creature, &neuron_records);

    assert!(
        detected.is_empty(),
        "Consistently helpful hidden neuron should NOT be flagged"
    );
}

/// Test 3: Only hidden neurons should be analysed.
#[test]
fn test_only_hidden_neurons_analysed() {
    let creature = two_output_creature();

    // Input neuron with conflicting errors — should be ignored
    let input_records: Vec<DiscoverRecord> = (0..50)
        .map(|i| record("input-1", i, 0.5, vec![-0.5, 0.3]))
        .collect();

    // Output neuron with conflicting errors — should be ignored
    let output_records: Vec<DiscoverRecord> = (0..50)
        .map(|i| record("output-0", i, 0.5, vec![-0.5, 0.3]))
        .collect();

    let neuron_records = vec![
        ("input-1".to_string(), input_records),
        ("output-0".to_string(), output_records),
    ];

    let detected = detect_output_conflict_neurons(&creature, &neuron_records);

    assert!(
        detected.is_empty(),
        "Input and output neurons should not be analysed for output conflicts"
    );
}

/// Test 4: Insufficient samples produce no detections.
#[test]
fn test_insufficient_samples_no_detection() {
    let creature = two_output_creature();

    // Only 5 samples — below minimum threshold
    let records: Vec<DiscoverRecord> = (0..5)
        .map(|i| record("hidden-1", i, 0.5, vec![-0.5, 0.3]))
        .collect();

    let neuron_records = vec![("hidden-1".to_string(), records)];

    let detected = detect_output_conflict_neurons(&creature, &neuron_records);

    assert!(
        detected.is_empty(),
        "Insufficient samples should produce no detections"
    );
}

/// Test 5: Single-output networks produce no detections.
#[test]
fn test_single_output_no_detection() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY", 0.0),
            neuron("hidden-1", "hidden", "TANH", 0.0),
            neuron("output-0", "output", "TANH", 0.0),
        ],
        vec![
            synapse("input-1", "hidden-1", 0.5),
            synapse("hidden-1", "output-0", 0.8),
        ],
    );

    let records: Vec<DiscoverRecord> = (0..50)
        .map(|i| record("hidden-1", i, 0.5, vec![-0.5]))
        .collect();

    let neuron_records = vec![("hidden-1".to_string(), records)];

    let detected = detect_output_conflict_neurons(&creature, &neuron_records);

    assert!(
        detected.is_empty(),
        "Single-output network cannot have output conflicts"
    );
}

/// Test 6: Produces valid coordinated structural candidates.
#[test]
fn test_produces_valid_coordinated_candidates() {
    let creature = two_output_creature();

    let detected = vec![OutputConflictNeuron {
        neuron_uuid: "hidden-1".to_string(),
        squash: "TANH".to_string(),
        bias: 0.0,
        per_output_mean_error: vec![-0.5, 0.3],
        conflict_severity: 0.8,
        sample_count: 50,
        estimated_improvement: 0.05,
    }];

    let candidates = output_conflicts_to_coordinated_candidates(&detected, &creature);

    assert!(
        !candidates.is_empty(),
        "Should produce at least one coordinated candidate"
    );

    for c in &candidates {
        assert!(
            c.expected_creature_score_gain > 0.0,
            "Expected improvement should be positive"
        );
        assert!(c.comment.is_some(), "Should have a descriptive comment");
        assert!(
            !c.operations.is_empty(),
            "Should have at least one structural operation"
        );
    }
}

/// Test 7: Multiple conflicting hidden neurons — all flagged.
#[test]
fn test_multiple_conflicting_neurons_all_flagged() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY", 0.0),
            neuron("hidden-1", "hidden", "TANH", 0.0),
            neuron("hidden-2", "hidden", "TANH", 0.0),
            neuron("output-0", "output", "TANH", 0.0),
            neuron("output-1", "output", "TANH", 0.0),
        ],
        vec![
            synapse("input-1", "hidden-1", 0.5),
            synapse("input-1", "hidden-2", 0.5),
            synapse("hidden-1", "output-0", 0.8),
            synapse("hidden-1", "output-1", -0.3),
            synapse("hidden-2", "output-0", -0.4),
            synapse("hidden-2", "output-1", 0.6),
        ],
    );

    let records_1: Vec<DiscoverRecord> = (0..50)
        .map(|i| record("hidden-1", i, 0.5, vec![-0.5, 0.3]))
        .collect();

    let records_2: Vec<DiscoverRecord> = (0..50)
        .map(|i| record("hidden-2", i, 0.4, vec![0.4, -0.6]))
        .collect();

    let neuron_records = vec![
        ("hidden-1".to_string(), records_1),
        ("hidden-2".to_string(), records_2),
    ];

    let detected = detect_output_conflict_neurons(&creature, &neuron_records);

    assert!(
        detected.len() >= 2,
        "Both conflicting hidden neurons should be flagged, got {}",
        detected.len()
    );

    let uuids: Vec<&str> = detected.iter().map(|d| d.neuron_uuid.as_str()).collect();
    assert!(uuids.contains(&"hidden-1"), "hidden-1 should be flagged");
    assert!(uuids.contains(&"hidden-2"), "hidden-2 should be flagged");
}

/// Test 8: Hidden neuron with mixed but weak conflicts should NOT be flagged.
///
/// If the per-output mean errors have opposing signs but are very small,
/// the conflict is not significant enough to warrant structural changes.
#[test]
fn test_weak_conflict_not_flagged() {
    let creature = two_output_creature();

    // Very small opposing errors — noise, not real conflict
    let records: Vec<DiscoverRecord> = (0..50)
        .map(|i| {
            let activation = 0.3 + (i as f32 / 49.0) * 0.4;
            record("hidden-1", i, activation, vec![-0.001, 0.002])
        })
        .collect();

    let neuron_records = vec![("hidden-1".to_string(), records)];

    let detected = detect_output_conflict_neurons(&creature, &neuron_records);

    assert!(
        detected.is_empty(),
        "Very weak conflicts should not be flagged"
    );
}

/// Test 9: Results sorted by conflict severity (worst first).
#[test]
fn test_results_sorted_by_severity() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY", 0.0),
            neuron("hidden-mild", "hidden", "TANH", 0.0),
            neuron("hidden-severe", "hidden", "TANH", 0.0),
            neuron("output-0", "output", "TANH", 0.0),
            neuron("output-1", "output", "TANH", 0.0),
        ],
        vec![
            synapse("input-1", "hidden-mild", 0.5),
            synapse("input-1", "hidden-severe", 0.5),
            synapse("hidden-mild", "output-0", 0.5),
            synapse("hidden-mild", "output-1", -0.2),
            synapse("hidden-severe", "output-0", 0.8),
            synapse("hidden-severe", "output-1", -0.8),
        ],
    );

    // hidden-mild: mild conflict
    let records_mild: Vec<DiscoverRecord> = (0..50)
        .map(|i| record("hidden-mild", i, 0.5, vec![-0.1, 0.05]))
        .collect();

    // hidden-severe: severe conflict
    let records_severe: Vec<DiscoverRecord> = (0..50)
        .map(|i| record("hidden-severe", i, 0.5, vec![-0.8, 0.7]))
        .collect();

    let neuron_records = vec![
        ("hidden-mild".to_string(), records_mild),
        ("hidden-severe".to_string(), records_severe),
    ];

    let detected = detect_output_conflict_neurons(&creature, &neuron_records);

    assert!(
        detected.len() >= 2,
        "Should detect both conflicting neurons"
    );

    // Most severe conflict should be first
    assert!(
        detected[0].conflict_severity >= detected[1].conflict_severity,
        "Results should be sorted by severity (highest first), got {:.3} then {:.3}",
        detected[0].conflict_severity,
        detected[1].conflict_severity
    );
}

/// Test 10: Empty records produce no detections.
#[test]
fn test_empty_records_no_detections() {
    let creature = two_output_creature();

    let detected = detect_output_conflict_neurons(&creature, &[]);

    assert!(
        detected.is_empty(),
        "Empty records should produce no detections"
    );
}

/// Test 11: Three-output network with conflict on only one output.
#[test]
fn test_three_output_partial_conflict() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY", 0.0),
            neuron("hidden-1", "hidden", "TANH", 0.0),
            neuron("output-0", "output", "TANH", 0.0),
            neuron("output-1", "output", "TANH", 0.0),
            neuron("output-2", "output", "TANH", 0.0),
        ],
        vec![
            synapse("input-1", "hidden-1", 0.5),
            synapse("hidden-1", "output-0", 0.8),
            synapse("hidden-1", "output-1", 0.5),
            synapse("hidden-1", "output-2", -0.6),
        ],
    );

    // Helps output-0 and output-1, harms output-2
    let records: Vec<DiscoverRecord> = (0..50)
        .map(|i| record("hidden-1", i, 0.5, vec![-0.3, -0.2, 0.4]))
        .collect();

    let neuron_records = vec![("hidden-1".to_string(), records)];

    let detected = detect_output_conflict_neurons(&creature, &neuron_records);

    assert!(
        !detected.is_empty(),
        "Hidden neuron with sign conflict across 3 outputs should be flagged"
    );
    assert_eq!(detected[0].per_output_mean_error.len(), 3);
}
