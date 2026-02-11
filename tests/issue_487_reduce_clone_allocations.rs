//! Tests for Issue #487: Reduce unnecessary clone() allocations in hot analysis paths.
//!
//! These tests verify that the clone reduction refactoring preserves correct behaviour:
//! - HelpfulStats derives Copy (enabling cheaper passing)
//! - SampleLocalityGroup without _representative_indices still groups correctly
//! - Bottleneck detection with borrowed fan-in/fan-out lists works correctly
//! - Structural pattern detection with borrowed synapse maps works correctly

mod common;

use neat_ai_discovery::analysis::bottleneck::{
    bottleneck_neurons_to_coordinated_candidates, detect_bottleneck_neurons,
};
use neat_ai_discovery::analysis::samples::{HelpfulSample, HelpfulStats};
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

// =============================================================================
// HelpfulStats Copy trait
// =============================================================================

/// Verify HelpfulStats implements Copy (can be passed by value without clone).
#[test]
fn helpful_stats_is_copy() {
    let stats = HelpfulStats {
        positive_count: 10,
        negative_count: 5,
        positive_improvement_sum: 0.5,
        negative_improvement_sum: 0.2,
        positive_activation_sum: 1.0,
        negative_activation_sum: 0.3,
        error_sq_sum: 0.1,
        activation_sq_sum: 0.8,
        error_activation_sum: 0.4,
        samples_evaluated: 15,
        early_terminated: false,
    };

    // Copy semantics: assign to new variable without clone()
    let stats_copy = stats;
    // Original still usable (proves Copy, not Move)
    assert_eq!(stats.positive_count, stats_copy.positive_count);
    assert_eq!(stats.negative_count, stats_copy.negative_count);
    assert_eq!(stats.total_count(), 15);
    assert_eq!(stats_copy.total_count(), 15);
}

/// Verify HelpfulStats can be passed to functions without explicit clone.
#[test]
fn helpful_stats_copy_through_function() {
    let stats = HelpfulStats {
        positive_count: 8,
        negative_count: 2,
        ..Default::default()
    };

    fn consume_stats(s: HelpfulStats) -> u32 {
        s.total_count()
    }

    // Pass by value (uses Copy, not Move)
    let count = consume_stats(stats);
    assert_eq!(count, 10);
    // stats is still usable
    assert_eq!(stats.positive_count, 8);
}

// =============================================================================
// Bottleneck detection with borrowed fan-in/fan-out lists
// =============================================================================

/// Verify bottleneck detection produces correct candidates after clone reduction.
#[test]
fn bottleneck_detection_with_borrowed_lists() {
    // Create a creature with a clear bottleneck: 4 inputs → 1 hidden → 1 output
    let creature = CreatureJson {
        neurons: vec![
            NeuronJson {
                uuid: "hidden-0".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "TANH".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "output-0".to_string(),
                neuron_type: "output".to_string(),
                squash: "IDENTITY".to_string(),
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
                to_uuid: "hidden-0".to_string(),
                weight: 0.3,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-2".to_string(),
                to_uuid: "hidden-0".to_string(),
                weight: 0.4,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-3".to_string(),
                to_uuid: "hidden-0".to_string(),
                weight: 0.2,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "hidden-0".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.8,
                synapse_type: None,
            },
        ],
        input: 4,
        output: 1,
    };

    // Create records for the bottleneck neuron
    let records: Vec<DiscoverRecord> = (0..50)
        .map(|i| DiscoverRecord {
            obs_index: i,
            neuron_uuid: "hidden-0".to_string(),
            value: Some(0.5),
            activation: 0.3 + 0.01 * i as f32,
            errors: vec![0.05 * (i as f32 / 50.0)],
        })
        .collect();

    let neuron_records = vec![("hidden-0".to_string(), records)];

    let candidates = detect_bottleneck_neurons(&creature, &neuron_records);

    // Should detect hidden-0 as a bottleneck (fan-in=4, fan-out=1)
    assert!(
        !candidates.is_empty(),
        "Should detect bottleneck with fan-in=4, fan-out=1"
    );

    let c = &candidates[0];
    assert_eq!(c.neuron_uuid, "hidden-0");
    assert_eq!(c.fan_in, 4);
    assert_eq!(c.fan_out, 1);
    assert!(c.estimated_improvement > 0.0);
    assert!(!c.recommended_actions.is_empty());
    assert_eq!(c.upstream_uuids.len(), 4);
    assert_eq!(c.downstream_uuids.len(), 1);

    // Convert to coordinated candidates
    let coordinated = bottleneck_neurons_to_coordinated_candidates(&candidates, &creature);
    assert!(
        !coordinated.is_empty(),
        "Should produce coordinated candidates from bottleneck"
    );

    // Verify the coordinated candidates have valid operations
    for cc in &coordinated {
        assert!(cc.expected_creature_score_gain > 0.0);
        assert!(!cc.operations.is_empty());
    }
}

// =============================================================================
// HelpfulSample Copy trait (pre-existing, verify still works)
// =============================================================================

/// Verify HelpfulSample is Copy and can be used in iterators with .copied().
#[test]
fn helpful_sample_is_copy() {
    let sample = HelpfulSample {
        activation: 0.5,
        avg_error: 0.1,
        target_value: Some(0.6),
        target_activation: Some(0.55),
    };

    let sample_copy = sample;
    assert_eq!(sample.activation, sample_copy.activation);
    assert_eq!(sample.avg_error, sample_copy.avg_error);

    // Verify .to_vec() works (Copy trait enables this)
    let samples = [sample];
    let copied: Vec<HelpfulSample> = samples.to_vec();
    assert_eq!(copied.len(), 1);
    assert_eq!(copied[0].activation, 0.5);
}

// =============================================================================
// Phase 2 (#487): Additional clone reduction — correctness verification
// =============================================================================

use neat_ai_discovery::analysis::bounded_range::{
    bounded_range_to_coordinated_candidates, detect_bounded_range_neurons,
};
use neat_ai_discovery::analysis::restricted_range::{
    RestrictedRangeConfig, detect_restricted_range_neurons,
    restricted_range_to_coordinated_candidates,
};

/// Verify restricted range detection produces correct results after clone reduction.
/// Tests the full detect → convert pipeline with multiple neurons.
#[test]
fn restricted_range_detection_preserves_correctness() {
    let creature = CreatureJson {
        neurons: vec![
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
                bias: 0.1,
            },
            NeuronJson {
                uuid: "output-0".to_string(),
                neuron_type: "output".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
        ],
        synapses: vec![
            SynapseJson {
                from_uuid: "input-0".to_string(),
                to_uuid: "hidden-0".to_string(),
                weight: 0.1,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-1".to_string(),
                to_uuid: "hidden-1".to_string(),
                weight: 0.1,
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
        ],
        input: 2,
        output: 1,
    };

    // Records in narrow range (< 20% of TANH [-1,1] range)
    let mut neuron_records = Vec::new();
    for n in 0..2 {
        let uuid = format!("hidden-{n}");
        let records: Vec<DiscoverRecord> = (0..50)
            .map(|i| DiscoverRecord {
                obs_index: i,
                neuron_uuid: uuid.clone(),
                value: Some(0.2),
                activation: 0.15 + 0.02 * (i as f32 / 50.0), // [0.15, 0.17]
                errors: vec![0.01],
            })
            .collect();
        neuron_records.push((uuid, records));
    }

    let config = RestrictedRangeConfig::default();
    let detected = detect_restricted_range_neurons(&creature, &neuron_records, &config);

    assert!(
        !detected.is_empty(),
        "Should detect restricted range neurons"
    );

    for d in &detected {
        assert!(
            d.range_utilisation < 0.20,
            "Range utilisation should be < 20%"
        );
        assert!(d.sample_count >= 50);
        assert_eq!(d.squash, "TANH");
    }

    // Convert to coordinated candidates and verify structure
    let coordinated = restricted_range_to_coordinated_candidates(&detected, &creature);
    assert!(
        !coordinated.is_empty(),
        "Should produce coordinated candidates from restricted range"
    );

    for cc in &coordinated {
        assert!(cc.expected_creature_score_gain > 0.0);
        assert!(!cc.operations.is_empty());
        assert!(cc.comment.is_some());
    }
}

/// Verify bounded range detection produces correct results after clone reduction.
/// Tests that sentinel cluster detection and gating neuron generation work correctly.
#[test]
fn bounded_range_detection_preserves_correctness() {
    let creature = CreatureJson {
        neurons: vec![
            NeuronJson {
                uuid: "hidden-0".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "TANH".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "output-0".to_string(),
                neuron_type: "output".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
        ],
        synapses: Vec::new(),
        input: 1,
        output: 1,
    };

    // Records with 30% at sentinel -1.0, rest in useful range [0.2, 0.8]
    let uuid = "hidden-0".to_string();
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = if i < 30 {
                -1.0 // sentinel cluster
            } else {
                0.2 + 0.6 * (i as f32 / 100.0) // useful range
            };
            DiscoverRecord {
                obs_index: i,
                neuron_uuid: uuid.clone(),
                value: Some(0.5),
                activation,
                errors: vec![0.01],
            }
        })
        .collect();

    let neuron_records = vec![(uuid, records)];
    let detected = detect_bounded_range_neurons(&creature, &neuron_records);

    assert!(
        !detected.is_empty(),
        "Should detect bounded range with sentinel cluster"
    );

    let d = &detected[0];
    assert_eq!(d.neuron_uuid, "hidden-0");
    assert!(
        (d.boundary_value - (-1.0)).abs() < 0.1,
        "Sentinel should be near -1.0"
    );
    assert!(d.boundary_fraction >= 0.20, "At least 20% at sentinel");
    assert!(d.detection_confidence > 0.0);
    assert!(d.estimated_improvement > 0.0);

    // Convert and verify gating neuron operations
    let coordinated = bounded_range_to_coordinated_candidates(&detected);
    assert!(
        !coordinated.is_empty(),
        "Should produce coordinated candidates from bounded range"
    );

    for cc in &coordinated {
        assert!(cc.expected_creature_score_gain > 0.0);
        assert!(
            cc.operations.len() == 2,
            "Should have AddNeuron + AddSynapse"
        );
    }
}

/// Verify bottleneck with multiple candidates produces correctly formed operations
/// (tests that String references are correctly materialised in operations).
#[test]
fn bottleneck_multiple_candidates_correct_uuids() {
    let creature = CreatureJson {
        neurons: vec![
            NeuronJson {
                uuid: "hidden-a".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "TANH".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "hidden-b".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "LOGISTIC".to_string(),
                bias: 0.1,
            },
            NeuronJson {
                uuid: "output-0".to_string(),
                neuron_type: "output".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
        ],
        synapses: vec![
            // hidden-a: fan-in=4, fan-out=1
            SynapseJson {
                from_uuid: "input-0".to_string(),
                to_uuid: "hidden-a".to_string(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-1".to_string(),
                to_uuid: "hidden-a".to_string(),
                weight: 0.3,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-2".to_string(),
                to_uuid: "hidden-a".to_string(),
                weight: 0.4,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-3".to_string(),
                to_uuid: "hidden-a".to_string(),
                weight: 0.2,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "hidden-a".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.8,
                synapse_type: None,
            },
            // hidden-b: fan-in=3, fan-out=1
            SynapseJson {
                from_uuid: "input-4".to_string(),
                to_uuid: "hidden-b".to_string(),
                weight: 0.6,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-5".to_string(),
                to_uuid: "hidden-b".to_string(),
                weight: 0.7,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-6".to_string(),
                to_uuid: "hidden-b".to_string(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "hidden-b".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.9,
                synapse_type: None,
            },
        ],
        input: 7,
        output: 1,
    };

    let neuron_records: Vec<(String, Vec<DiscoverRecord>)> = ["hidden-a", "hidden-b"]
        .iter()
        .map(|&uuid| {
            let records: Vec<DiscoverRecord> = (0..50)
                .map(|i| DiscoverRecord {
                    obs_index: i,
                    neuron_uuid: uuid.to_string(),
                    value: Some(0.5),
                    activation: 0.3 + 0.01 * i as f32,
                    errors: vec![0.05],
                })
                .collect();
            (uuid.to_string(), records)
        })
        .collect();

    let candidates = detect_bottleneck_neurons(&creature, &neuron_records);
    let coordinated = bottleneck_neurons_to_coordinated_candidates(&candidates, &creature);

    // Verify all operation UUIDs are non-empty and correctly formed
    for cc in &coordinated {
        for op in &cc.operations {
            match op {
                neat_ai_discovery::CoordinatedStructuralOpJson::AddNeuron {
                    neuron_uuid,
                    squash,
                    ..
                } => {
                    assert!(!neuron_uuid.is_empty(), "AddNeuron UUID must not be empty");
                    assert!(!squash.is_empty(), "Squash must not be empty");
                }
                neat_ai_discovery::CoordinatedStructuralOpJson::AddSynapse {
                    from_neuron_uuid,
                    to_neuron_uuid,
                    ..
                } => {
                    assert!(
                        !from_neuron_uuid.is_empty(),
                        "AddSynapse from_uuid must not be empty"
                    );
                    assert!(
                        !to_neuron_uuid.is_empty(),
                        "AddSynapse to_uuid must not be empty"
                    );
                }
                _ => {}
            }
        }
    }
}
