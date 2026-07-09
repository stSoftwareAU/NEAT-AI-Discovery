//! Tests for Issue #643: Activation-error monotonicity detector.
//!
//! A well-functioning hidden neuron should have a monotonic relationship between
//! its activation and the output error — higher activation consistently correlates
//! with either lower or higher error. A non-monotonic relationship suggests the
//! neuron is encoding contradictory information and should be restructured.
//!
//! ## TDD Plan
//! 1. Monotonic activation-error relationship should NOT be flagged
//! 2. Non-monotonic (contradictory) relationship SHOULD be flagged
//! 3. Input/output neurons are excluded
//! 4. Insufficient samples produce no detections
//! 5. Produces valid structural candidates (addNeuron or changeSquash)
//! 6. Distinguishes from noise-to-signal detection (variance-based)
//! 7. Multiple neurons — only non-monotonic ones flagged

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use neat_ai_discovery::analysis::detection::monotonicity::{
    MonotonicityCandidate, detect_non_monotonic_neurons,
    non_monotonic_neurons_to_coordinated_candidates,
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

fn neuron(uuid: &str, neuron_type: &str, squash: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
        squash: squash.to_string(),
        bias: 0.0,
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

/// Test 1: Monotonically increasing activation-error (higher activation → higher error)
/// should NOT be flagged as non-monotonic.
#[test]
fn test_monotonic_increasing_not_flagged() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("hidden-1", "hidden", "LOGISTIC"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "hidden-1", 0.5),
            synapse("hidden-1", "output-1", 0.8),
        ],
    );

    // Monotonically increasing: as activation goes up, error goes up
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = i as f32 / 100.0;
            let error = 0.01 + activation * 0.8;
            record("hidden-1", i, activation, vec![error])
        })
        .collect();

    let neuron_records = vec![("hidden-1".to_string(), records)];

    let candidates = detect_non_monotonic_neurons(&creature, &neuron_records);

    assert!(
        candidates.is_empty(),
        "Monotonically increasing activation-error should not be flagged, got {} candidates",
        candidates.len()
    );
}

/// Test 2: Monotonically decreasing activation-error (higher activation → lower error)
/// should NOT be flagged.
#[test]
fn test_monotonic_decreasing_not_flagged() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("hidden-1", "hidden", "LOGISTIC"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "hidden-1", 0.5),
            synapse("hidden-1", "output-1", 0.8),
        ],
    );

    // Monotonically decreasing: as activation goes up, error goes down
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = i as f32 / 100.0;
            let error = 0.8 - activation * 0.7;
            record("hidden-1", i, activation, vec![error])
        })
        .collect();

    let neuron_records = vec![("hidden-1".to_string(), records)];

    let candidates = detect_non_monotonic_neurons(&creature, &neuron_records);

    assert!(
        candidates.is_empty(),
        "Monotonically decreasing activation-error should not be flagged, got {} candidates",
        candidates.len()
    );
}

/// Test 3: Non-monotonic (U-shaped) activation-error SHOULD be flagged.
/// Low and high activations have high error, mid activations have low error.
#[test]
fn test_non_monotonic_u_shape_flagged() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("hidden-1", "hidden", "LOGISTIC"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "hidden-1", 0.5),
            synapse("hidden-1", "output-1", 0.8),
        ],
    );

    // U-shaped: error is high at both extremes, low in the middle
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = i as f32 / 100.0;
            let mid = 0.5;
            let error = 0.02 + (activation - mid).powi(2) * 3.0;
            record("hidden-1", i, activation, vec![error])
        })
        .collect();

    let neuron_records = vec![("hidden-1".to_string(), records)];

    let candidates = detect_non_monotonic_neurons(&creature, &neuron_records);

    assert!(
        !candidates.is_empty(),
        "Non-monotonic (U-shaped) activation-error should be flagged"
    );

    let c = &candidates[0];
    assert_eq!(c.neuron_uuid, "hidden-1");
    assert!(
        c.monotonicity_score.abs() < 0.5,
        "U-shaped pattern should have low monotonicity score, got {}",
        c.monotonicity_score
    );
}

/// Test 4: Non-monotonic (inverted-U) activation-error SHOULD be flagged.
/// Mid activations have high error, extremes have low error.
#[test]
fn test_non_monotonic_inverted_u_flagged() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("hidden-1", "hidden", "LOGISTIC"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "hidden-1", 0.5),
            synapse("hidden-1", "output-1", 0.8),
        ],
    );

    // Inverted-U: error is high in the middle, low at extremes
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = i as f32 / 100.0;
            let mid = 0.5;
            let error = 0.8 - (activation - mid).powi(2) * 3.0;
            let error = error.max(0.01);
            record("hidden-1", i, activation, vec![error])
        })
        .collect();

    let neuron_records = vec![("hidden-1".to_string(), records)];

    let candidates = detect_non_monotonic_neurons(&creature, &neuron_records);

    assert!(
        !candidates.is_empty(),
        "Non-monotonic (inverted-U) activation-error should be flagged"
    );
}

/// Test 5: Input and output neurons should be excluded from detection.
#[test]
fn test_input_output_neurons_excluded() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    // Non-monotonic records for both input and output neurons
    let make_non_monotonic = |uuid: &str| -> Vec<DiscoverRecord> {
        (0..100)
            .map(|i| {
                let activation = i as f32 / 100.0;
                let error = 0.02 + (activation - 0.5).powi(2) * 3.0;
                record(uuid, i, activation, vec![error])
            })
            .collect()
    };

    let neuron_records = vec![
        ("input-1".to_string(), make_non_monotonic("input-1")),
        ("output-1".to_string(), make_non_monotonic("output-1")),
    ];

    let candidates = detect_non_monotonic_neurons(&creature, &neuron_records);

    assert!(
        candidates.is_empty(),
        "Input and output neurons should not be flagged"
    );
}

/// Test 6: Insufficient samples should not trigger detection.
#[test]
fn test_activation_monotonicity_insufficient_samples_no_detection() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("hidden-1", "hidden", "LOGISTIC"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "hidden-1", 0.5),
            synapse("hidden-1", "output-1", 0.8),
        ],
    );

    // Only 5 samples
    let records: Vec<DiscoverRecord> = (0..5)
        .map(|i| {
            let activation = i as f32 / 5.0;
            let error = 0.02 + (activation - 0.5).powi(2) * 3.0;
            record("hidden-1", i, activation, vec![error])
        })
        .collect();

    let neuron_records = vec![("hidden-1".to_string(), records)];

    let candidates = detect_non_monotonic_neurons(&creature, &neuron_records);

    assert!(
        candidates.is_empty(),
        "Insufficient samples should not produce detections"
    );
}

/// Test 7: Produces valid structural candidates (addNeuron to split, or changeSquash).
#[test]
fn test_produces_valid_structural_candidates() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("hidden-1", "hidden", "LOGISTIC"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "hidden-1", 0.5),
            synapse("hidden-1", "output-1", 0.8),
        ],
    );

    let candidate = MonotonicityCandidate {
        neuron_uuid: "hidden-1".to_string(),
        monotonicity_score: 0.1,
        sample_count: 100,
        estimated_improvement: 0.005,
    };

    let coordinated = non_monotonic_neurons_to_coordinated_candidates(&[candidate], &creature);

    assert!(
        !coordinated.is_empty(),
        "Should produce at least one coordinated candidate"
    );

    let c = &coordinated[0];
    assert!(
        c.expected_creature_score_gain > 0.0,
        "Expected improvement should be positive"
    );
    assert!(c.comment.is_some(), "Should have a descriptive comment");

    // Check operations include addNeuron or changeSquash
    let ops_json = serde_json::to_string(&c.operations).unwrap();
    let has_structural_op = ops_json.contains("addNeuron") || ops_json.contains("changeSquash");
    assert!(
        has_structural_op,
        "Should produce addNeuron or changeSquash operation, got: {ops_json}"
    );
}

/// Test 8: Multiple neurons — only non-monotonic ones are flagged.
#[test]
fn test_multiple_neurons_only_non_monotonic_flagged() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("hidden-good", "hidden", "LOGISTIC"),
            neuron("hidden-bad", "hidden", "LOGISTIC"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "hidden-good", 0.5),
            synapse("input-1", "hidden-bad", 0.5),
            synapse("hidden-good", "output-1", 0.8),
            synapse("hidden-bad", "output-1", 0.3),
        ],
    );

    // hidden-good: monotonically decreasing (activation up → error down)
    let good_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = i as f32 / 100.0;
            let error = 0.8 - activation * 0.7;
            record("hidden-good", i, activation, vec![error])
        })
        .collect();

    // hidden-bad: non-monotonic (U-shaped)
    let bad_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = i as f32 / 100.0;
            let error = 0.02 + (activation - 0.5).powi(2) * 3.0;
            record("hidden-bad", i, activation, vec![error])
        })
        .collect();

    let neuron_records = vec![
        ("hidden-good".to_string(), good_records),
        ("hidden-bad".to_string(), bad_records),
    ];

    let candidates = detect_non_monotonic_neurons(&creature, &neuron_records);

    assert!(
        !candidates.is_empty(),
        "Should detect at least one non-monotonic neuron"
    );

    let flagged_uuids: Vec<&str> = candidates.iter().map(|c| c.neuron_uuid.as_str()).collect();
    assert!(
        flagged_uuids.contains(&"hidden-bad"),
        "hidden-bad should be flagged, got: {flagged_uuids:?}"
    );
    assert!(
        !flagged_uuids.contains(&"hidden-good"),
        "hidden-good should NOT be flagged, got: {flagged_uuids:?}"
    );
}

/// Test 9: Estimated improvement is positive for detected non-monotonic neurons.
#[test]
fn test_activation_monotonicity_estimated_improvement_positive() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("hidden-1", "hidden", "LOGISTIC"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "hidden-1", 0.5),
            synapse("hidden-1", "output-1", 0.8),
        ],
    );

    // Strongly non-monotonic: W-shaped error pattern relative to activation
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = i as f32 / 100.0;
            // W-shape — two valleys, no consistent direction
            let error = if activation < 0.25 {
                0.7 - activation * 2.0
            } else if activation < 0.5 {
                0.2 + (activation - 0.25) * 2.0
            } else if activation < 0.75 {
                0.7 - (activation - 0.5) * 2.0
            } else {
                0.2 + (activation - 0.75) * 2.0
            };
            record("hidden-1", i, activation, vec![error.max(0.01)])
        })
        .collect();

    let neuron_records = vec![("hidden-1".to_string(), records)];

    let candidates = detect_non_monotonic_neurons(&creature, &neuron_records);

    assert!(!candidates.is_empty(), "Should detect non-monotonic neuron");

    for c in &candidates {
        assert!(
            c.estimated_improvement > 0.0,
            "Estimated improvement should be positive, got {}",
            c.estimated_improvement
        );
    }
}

/// Test 10: Empty records produce no detections.
#[test]
fn test_activation_monotonicity_empty_records_no_detections() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("hidden-1", "hidden", "LOGISTIC"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![],
    );

    let candidates =
        detect_non_monotonic_neurons(&creature, &Vec::<(String, Vec<DiscoverRecord>)>::new());

    assert!(
        candidates.is_empty(),
        "Empty records should produce no detections"
    );
}
