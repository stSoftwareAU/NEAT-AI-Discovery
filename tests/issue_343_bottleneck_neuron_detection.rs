//! Tests for Issue #343: Detect bottleneck neurons limiting information flow.
//!
//! Bottleneck neurons are hidden neurons where many input signals converge through
//! a single neuron before reaching outputs. They limit the network's ability to
//! represent complex input combinations because one neuron's activation range must
//! encode all upstream information.
//!
//! ## TDD Plan
//! 1. Create test network with an intentional bottleneck (many inputs → 1 neuron → outputs)
//! 2. Verify detection identifies the bottleneck
//! 3. Verify non-bottleneck neurons are not flagged
//! 4. Test structural recommendations (parallel path, bypass)
//! 5. Verify with network that has natural fan-in (should not flag)

use neat_ai_discovery::analysis::bottleneck::{
    bottleneck_neurons_to_coordinated_candidates, detect_bottleneck_neurons,
    BottleneckNeuronCandidate,
};
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

/// Helper: create a DiscoverRecord for a neuron with given activation and error.
fn record(
    neuron_uuid: &str,
    obs_index: u32,
    activation: f32,
    value: Option<f32>,
    errors: Vec<f32>,
) -> DiscoverRecord {
    DiscoverRecord {
        obs_index,
        neuron_uuid: neuron_uuid.to_string(),
        value,
        activation,
        errors,
    }
}

/// Build a creature with an intentional bottleneck:
/// 5 inputs → hidden-bottleneck → 2 outputs
/// All information must flow through a single hidden neuron.
fn bottleneck_creature() -> CreatureJson {
    CreatureJson {
        neurons: vec![
            NeuronJson {
                uuid: "input-0".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "input-1".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "input-2".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "input-3".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "input-4".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "hidden-bottleneck".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "TANH".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "output-0".to_string(),
                neuron_type: "output".to_string(),
                squash: "TANH".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "output-1".to_string(),
                neuron_type: "output".to_string(),
                squash: "TANH".to_string(),
                bias: 0.0,
            },
        ],
        synapses: vec![
            SynapseJson {
                from_uuid: "input-0".to_string(),
                to_uuid: "hidden-bottleneck".to_string(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-1".to_string(),
                to_uuid: "hidden-bottleneck".to_string(),
                weight: 0.3,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-2".to_string(),
                to_uuid: "hidden-bottleneck".to_string(),
                weight: -0.4,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-3".to_string(),
                to_uuid: "hidden-bottleneck".to_string(),
                weight: 0.6,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-4".to_string(),
                to_uuid: "hidden-bottleneck".to_string(),
                weight: -0.2,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "hidden-bottleneck".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.8,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "hidden-bottleneck".to_string(),
                to_uuid: "output-1".to_string(),
                weight: -0.5,
                synapse_type: None,
            },
        ],
        input: 5,
        output: 2,
    }
}

/// Build a creature with a wide topology — no bottleneck.
/// 3 inputs → 3 hidden neurons → 2 outputs (each hidden gets 1 input)
fn no_bottleneck_creature() -> CreatureJson {
    CreatureJson {
        neurons: vec![
            NeuronJson {
                uuid: "input-0".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "input-1".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "input-2".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "hidden-0".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "TANH".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "hidden-1".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "TANH".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "hidden-2".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "TANH".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "output-0".to_string(),
                neuron_type: "output".to_string(),
                squash: "TANH".to_string(),
                bias: 0.0,
            },
        ],
        synapses: vec![
            SynapseJson {
                from_uuid: "input-0".to_string(),
                to_uuid: "hidden-0".to_string(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-1".to_string(),
                to_uuid: "hidden-1".to_string(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-2".to_string(),
                to_uuid: "hidden-2".to_string(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "hidden-0".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "hidden-1".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "hidden-2".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.5,
                synapse_type: None,
            },
        ],
        input: 3,
        output: 1,
    }
}

/// Test 1: A single hidden neuron with fan-in=5 and fan-out=2 is detected as a bottleneck.
#[test]
fn test_detects_bottleneck_with_high_fan_in() {
    let creature = bottleneck_creature();

    // Generate records with diverse inputs but correlated errors (showing the bottleneck
    // concentrates error)
    let neuron_records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "hidden-bottleneck".to_string(),
        (0..100)
            .map(|i| {
                let activation = (i as f32 * 0.1).sin() * 0.5;
                record(
                    "hidden-bottleneck",
                    i,
                    activation,
                    Some(i as f32 * 0.05),
                    vec![0.1 * (1.0 + (i as f32 * 0.03).sin())],
                )
            })
            .collect(),
    )];

    let candidates = detect_bottleneck_neurons(&creature, &neuron_records);

    assert!(
        !candidates.is_empty(),
        "Should detect the bottleneck neuron"
    );
    let c = &candidates[0];
    assert_eq!(c.neuron_uuid, "hidden-bottleneck");
    assert_eq!(c.fan_in, 5);
    assert_eq!(c.fan_out, 2);
    assert!(
        c.estimated_improvement > 0.0,
        "Expected improvement should be positive"
    );
}

/// Test 2: Non-bottleneck neurons in a wide topology are not flagged.
#[test]
fn test_does_not_flag_wide_topology() {
    let creature = no_bottleneck_creature();

    let neuron_records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        (
            "hidden-0".to_string(),
            (0..100)
                .map(|i| {
                    record(
                        "hidden-0",
                        i,
                        (i as f32 * 0.1).sin() * 0.3,
                        Some(i as f32 * 0.05),
                        vec![0.05],
                    )
                })
                .collect(),
        ),
        (
            "hidden-1".to_string(),
            (0..100)
                .map(|i| {
                    record(
                        "hidden-1",
                        i,
                        (i as f32 * 0.2).cos() * 0.4,
                        Some(i as f32 * 0.03),
                        vec![0.04],
                    )
                })
                .collect(),
        ),
        (
            "hidden-2".to_string(),
            (0..100)
                .map(|i| {
                    record(
                        "hidden-2",
                        i,
                        (i as f32 * 0.15).sin() * 0.2,
                        Some(i as f32 * 0.04),
                        vec![0.06],
                    )
                })
                .collect(),
        ),
    ];

    let candidates = detect_bottleneck_neurons(&creature, &neuron_records);

    assert!(
        candidates.is_empty(),
        "Wide topology with fan-in=1 should not flag any bottleneck, found: {}",
        candidates.len()
    );
}

/// Test 3: Output neurons are never flagged as bottlenecks (they are natural convergence points).
#[test]
fn test_output_neurons_not_flagged() {
    // Even though output neurons may have high fan-in, they are expected convergence points
    let creature = CreatureJson {
        neurons: vec![
            NeuronJson {
                uuid: "input-0".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "input-1".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "input-2".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "output-0".to_string(),
                neuron_type: "output".to_string(),
                squash: "TANH".to_string(),
                bias: 0.0,
            },
        ],
        synapses: vec![
            SynapseJson {
                from_uuid: "input-0".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-1".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.3,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-2".to_string(),
                to_uuid: "output-0".to_string(),
                weight: -0.4,
                synapse_type: None,
            },
        ],
        input: 3,
        output: 1,
    };

    let neuron_records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "output-0".to_string(),
        (0..100)
            .map(|i| {
                record(
                    "output-0",
                    i,
                    (i as f32 * 0.1).sin() * 0.5,
                    Some(0.5),
                    vec![0.1],
                )
            })
            .collect(),
    )];

    let candidates = detect_bottleneck_neurons(&creature, &neuron_records);
    assert!(
        candidates.is_empty(),
        "Output neurons should never be flagged as bottlenecks"
    );
}

/// Test 4: Input neurons are never flagged.
#[test]
fn test_input_neurons_not_flagged() {
    let creature = bottleneck_creature();

    let neuron_records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "input-0".to_string(),
        (0..100)
            .map(|i| record("input-0", i, i as f32 * 0.1, Some(i as f32 * 0.1), vec![]))
            .collect(),
    )];

    let candidates = detect_bottleneck_neurons(&creature, &neuron_records);
    assert!(
        candidates.is_empty(),
        "Input neurons should never be flagged"
    );
}

/// Test 5: Insufficient samples should not trigger detection.
#[test]
fn test_insufficient_samples_not_flagged() {
    let creature = bottleneck_creature();

    let neuron_records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "hidden-bottleneck".to_string(),
        (0..5)
            .map(|i| record("hidden-bottleneck", i, 0.5, Some(0.5), vec![0.1]))
            .collect(),
    )];

    let candidates = detect_bottleneck_neurons(&creature, &neuron_records);
    assert!(
        candidates.is_empty(),
        "Too few samples should not trigger detection"
    );
}

/// Test 6: Candidates produce correct coordinated structural operations.
#[test]
fn test_candidates_produce_coordinated_operations() {
    let candidate = BottleneckNeuronCandidate {
        neuron_uuid: "hidden-bottleneck".to_string(),
        fan_in: 5,
        fan_out: 2,
        error_contribution_ratio: 0.45,
        bottleneck_score: 0.8,
        estimated_improvement: 0.02,
        recommended_actions: vec![
            "addParallelNeuron".to_string(),
            "addBypassSynapse".to_string(),
        ],
        upstream_uuids: vec![
            "input-0".to_string(),
            "input-1".to_string(),
            "input-2".to_string(),
            "input-3".to_string(),
            "input-4".to_string(),
        ],
        downstream_uuids: vec!["output-0".to_string(), "output-1".to_string()],
    };

    let creature = bottleneck_creature();
    let coordinated = bottleneck_neurons_to_coordinated_candidates(&[candidate], &creature);

    assert!(
        !coordinated.is_empty(),
        "Should produce coordinated candidates"
    );

    // Check that at least one candidate has positive expected improvement
    let c = &coordinated[0];
    assert!(
        c.expected_creature_score_gain > 0.0,
        "Expected improvement should be positive"
    );
    assert!(
        c.comment.is_some(),
        "Should have a comment explaining the change"
    );

    // The operations should include AddNeuron and AddSynapse for parallel path
    let ops_json = serde_json::to_string(&c.operations).unwrap();
    let has_add_neuron = ops_json.contains("addNeuron");
    let has_add_synapse = ops_json.contains("addSynapse");
    assert!(
        has_add_neuron || has_add_synapse,
        "Should include addNeuron or addSynapse operations, got: {ops_json}"
    );
}

/// Test 7: Bottleneck score is higher for neurons with higher fan-in ratio.
#[test]
fn test_higher_fan_in_ratio_scores_higher() {
    // Creature with two hidden neurons: one with fan-in=8, one with fan-in=2
    let creature = CreatureJson {
        neurons: vec![
            NeuronJson {
                uuid: "input-0".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "input-1".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "input-2".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "input-3".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "input-4".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "input-5".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "input-6".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "input-7".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "hidden-big".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "TANH".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "hidden-small".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "TANH".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "output-0".to_string(),
                neuron_type: "output".to_string(),
                squash: "TANH".to_string(),
                bias: 0.0,
            },
        ],
        synapses: vec![
            // hidden-big: fan-in = 8
            SynapseJson {
                from_uuid: "input-0".to_string(),
                to_uuid: "hidden-big".to_string(),
                weight: 0.3,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-1".to_string(),
                to_uuid: "hidden-big".to_string(),
                weight: 0.3,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-2".to_string(),
                to_uuid: "hidden-big".to_string(),
                weight: 0.3,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-3".to_string(),
                to_uuid: "hidden-big".to_string(),
                weight: 0.3,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-4".to_string(),
                to_uuid: "hidden-big".to_string(),
                weight: 0.3,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-5".to_string(),
                to_uuid: "hidden-big".to_string(),
                weight: 0.3,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-6".to_string(),
                to_uuid: "hidden-big".to_string(),
                weight: 0.3,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-7".to_string(),
                to_uuid: "hidden-big".to_string(),
                weight: 0.3,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "hidden-big".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.5,
                synapse_type: None,
            },
            // hidden-small: fan-in = 2
            SynapseJson {
                from_uuid: "input-0".to_string(),
                to_uuid: "hidden-small".to_string(),
                weight: 0.3,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-1".to_string(),
                to_uuid: "hidden-small".to_string(),
                weight: 0.3,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "hidden-small".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.5,
                synapse_type: None,
            },
        ],
        input: 8,
        output: 1,
    };

    let neuron_records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        (
            "hidden-big".to_string(),
            (0..100)
                .map(|i| {
                    record(
                        "hidden-big",
                        i,
                        (i as f32 * 0.05).sin() * 0.7,
                        Some(i as f32 * 0.1),
                        vec![0.15],
                    )
                })
                .collect(),
        ),
        (
            "hidden-small".to_string(),
            (0..100)
                .map(|i| {
                    record(
                        "hidden-small",
                        i,
                        (i as f32 * 0.1).cos() * 0.3,
                        Some(i as f32 * 0.05),
                        vec![0.05],
                    )
                })
                .collect(),
        ),
    ];

    let candidates = detect_bottleneck_neurons(&creature, &neuron_records);

    // hidden-big should be flagged (fan-in=8, fan-out=1), hidden-small should not (fan-in=2)
    assert!(
        !candidates.is_empty(),
        "Should detect at least one bottleneck"
    );

    // If both are detected, the bigger bottleneck should rank first
    if candidates.len() > 1 {
        assert!(
            candidates[0].fan_in >= candidates[1].fan_in,
            "Higher fan-in bottleneck should rank first"
        );
    }

    // The first candidate should be hidden-big
    assert_eq!(candidates[0].neuron_uuid, "hidden-big");
}

/// Test 8: Bottleneck detection with error concentration — neurons with high
/// error contribution ratio should score higher.
#[test]
fn test_error_contribution_increases_score() {
    let creature = bottleneck_creature();

    // Records with high error values (concentrated error through bottleneck)
    let high_error_records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "hidden-bottleneck".to_string(),
        (0..100)
            .map(|i| {
                record(
                    "hidden-bottleneck",
                    i,
                    (i as f32 * 0.1).sin() * 0.5,
                    Some(i as f32 * 0.05),
                    vec![0.5],
                )
            })
            .collect(),
    )];

    let candidates = detect_bottleneck_neurons(&creature, &high_error_records);

    assert!(
        !candidates.is_empty(),
        "Should detect bottleneck with high error"
    );
    assert!(
        candidates[0].error_contribution_ratio > 0.0,
        "Error contribution ratio should be positive"
    );
}

/// Test 9: A neuron with fan-in=1 and fan-out=1 (pass-through) is not a bottleneck.
#[test]
fn test_pass_through_neuron_not_bottleneck() {
    let creature = CreatureJson {
        neurons: vec![
            NeuronJson {
                uuid: "input-0".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "hidden-pass".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "TANH".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "output-0".to_string(),
                neuron_type: "output".to_string(),
                squash: "TANH".to_string(),
                bias: 0.0,
            },
        ],
        synapses: vec![
            SynapseJson {
                from_uuid: "input-0".to_string(),
                to_uuid: "hidden-pass".to_string(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "hidden-pass".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.8,
                synapse_type: None,
            },
        ],
        input: 1,
        output: 1,
    };

    let neuron_records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "hidden-pass".to_string(),
        (0..100)
            .map(|i| {
                record(
                    "hidden-pass",
                    i,
                    (i as f32 * 0.1).sin() * 0.5,
                    Some(0.5),
                    vec![0.1],
                )
            })
            .collect(),
    )];

    let candidates = detect_bottleneck_neurons(&creature, &neuron_records);
    assert!(
        candidates.is_empty(),
        "Pass-through neuron (fan-in=1) should not be a bottleneck"
    );
}

/// Test 10: Bypass synapse recommendation connects upstream directly to downstream.
#[test]
fn test_bypass_synapse_candidate() {
    let candidate = BottleneckNeuronCandidate {
        neuron_uuid: "hidden-bottleneck".to_string(),
        fan_in: 5,
        fan_out: 2,
        error_contribution_ratio: 0.45,
        bottleneck_score: 0.8,
        estimated_improvement: 0.02,
        recommended_actions: vec!["addBypassSynapse".to_string()],
        upstream_uuids: vec![
            "input-0".to_string(),
            "input-1".to_string(),
            "input-2".to_string(),
            "input-3".to_string(),
            "input-4".to_string(),
        ],
        downstream_uuids: vec!["output-0".to_string(), "output-1".to_string()],
    };

    let creature = bottleneck_creature();
    let coordinated = bottleneck_neurons_to_coordinated_candidates(&[candidate], &creature);

    // Should have bypass candidates that add direct connections
    let bypass_candidates: Vec<_> = coordinated
        .iter()
        .filter(|c| c.comment.as_ref().is_some_and(|s| s.contains("bypass")))
        .collect();

    assert!(
        !bypass_candidates.is_empty(),
        "Should produce bypass synapse candidates"
    );

    // The bypass candidate should add a synapse from an upstream to a downstream neuron
    for bc in &bypass_candidates {
        let ops_json = serde_json::to_string(&bc.operations).unwrap();
        assert!(
            ops_json.contains("addSynapse"),
            "Bypass candidate should contain addSynapse operation"
        );
        assert!(
            bc.expected_creature_score_gain > 0.0,
            "Bypass candidate should have positive expected gain"
        );
    }
}

/// Test 11: Parallel path candidate adds a new neuron sharing inputs/outputs.
#[test]
fn test_parallel_path_candidate() {
    let candidate = BottleneckNeuronCandidate {
        neuron_uuid: "hidden-bottleneck".to_string(),
        fan_in: 5,
        fan_out: 2,
        error_contribution_ratio: 0.45,
        bottleneck_score: 0.8,
        estimated_improvement: 0.02,
        recommended_actions: vec!["addParallelNeuron".to_string()],
        upstream_uuids: vec![
            "input-0".to_string(),
            "input-1".to_string(),
            "input-2".to_string(),
            "input-3".to_string(),
            "input-4".to_string(),
        ],
        downstream_uuids: vec!["output-0".to_string(), "output-1".to_string()],
    };

    let creature = bottleneck_creature();
    let coordinated = bottleneck_neurons_to_coordinated_candidates(&[candidate], &creature);

    // Should have parallel path candidates that add a new neuron
    let parallel_candidates: Vec<_> = coordinated
        .iter()
        .filter(|c| c.comment.as_ref().is_some_and(|s| s.contains("parallel")))
        .collect();

    assert!(
        !parallel_candidates.is_empty(),
        "Should produce parallel neuron candidates"
    );

    for pc in &parallel_candidates {
        let ops_json = serde_json::to_string(&pc.operations).unwrap();
        assert!(
            ops_json.contains("addNeuron"),
            "Parallel candidate should contain addNeuron operation"
        );
        assert!(
            ops_json.contains("addSynapse"),
            "Parallel candidate should contain addSynapse operations for wiring"
        );
    }
}

/// Test 12: Neuron with no recorded samples is skipped gracefully.
#[test]
fn test_no_records_for_neuron() {
    let creature = bottleneck_creature();

    // No records at all
    let neuron_records: Vec<(String, Vec<DiscoverRecord>)> = vec![];

    let candidates = detect_bottleneck_neurons(&creature, &neuron_records);
    assert!(
        candidates.is_empty(),
        "No records should produce no candidates"
    );
}

/// Test 13: Bottleneck score considers error magnitude.
#[test]
fn test_bottleneck_score_considers_errors() {
    let creature = bottleneck_creature();

    // High error records
    let neuron_records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "hidden-bottleneck".to_string(),
        (0..100)
            .map(|i| {
                record(
                    "hidden-bottleneck",
                    i,
                    (i as f32 * 0.1).sin() * 0.5,
                    Some(i as f32 * 0.05),
                    vec![0.8],
                )
            })
            .collect(),
    )];

    let candidates = detect_bottleneck_neurons(&creature, &neuron_records);
    assert!(!candidates.is_empty(), "Should detect bottleneck");

    let c = &candidates[0];
    assert!(
        c.bottleneck_score > 0.0,
        "Bottleneck score should be positive"
    );
}
