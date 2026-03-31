//! Tests for Issue #965: Batch-successful candidate grouping module.
//!
//! ## TDD Plan
//! 1. Verify individually successful candidates are detected with clear improvement signals
//! 2. Verify no candidates when improvement is too low
//! 3. Verify structural conflict detection (same source+target = conflict)
//! 4. Verify no conflict for different sources to same target
//! 5. Verify batch grouping produces groups of 2–4 candidates
//! 6. Verify batch respects max size (no batches > 4)
//! 7. Verify conversion produces valid coordinated candidates with `AddSynapse` ops
//! 8. Verify combined improvement is positive
//! 9. Verify empty/insufficient inputs produce no results
//! 10. Verify comments distinguish batch-successful from other candidate types
//! 11. Verify `detect_batch_successful_groups` end-to-end pipeline
//! 12. Verify existing synapses are excluded from detection

use neat_ai_discovery::analysis::recommendation::batch_successful::{
    BatchSuccessfulGroup, IndividualCandidate, batch_successful_to_coordinated_candidates,
    detect_batch_successful_groups, detect_individually_successful, group_into_batches,
    has_structural_conflict,
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

/// Helper: build a `NeuronJson`.
fn neuron(uuid: &str, neuron_type: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
        squash: "IDENTITY".to_string(),
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

/// Build a creature with two inputs and one output where both inputs'
/// activations strongly predict the output error independently.
fn make_two_successful_inputs() -> (CreatureJson, Vec<(String, Vec<DiscoverRecord>)>) {
    let creature = CreatureJson {
        neurons: vec![
            neuron("input-a", "input"),
            neuron("input-b", "input"),
            neuron("output-1", "output"),
        ],
        // No existing synapses from input-a or input-b to output-1.
        synapses: vec![],
        input: 2,
        output: 1,
    };

    let n = 100_u32;
    let mut records: Vec<(String, Vec<DiscoverRecord>)> = Vec::new();

    // input-a: activation pattern that strongly predicts error.
    records.push((
        "input-a".to_string(),
        (0..n)
            .map(|i| {
                let act = if i < n / 2 { 0.9 } else { 0.1 };
                record("input-a", i, act, vec![])
            })
            .collect(),
    ));

    // input-b: different activation pattern, also predicts error.
    records.push((
        "input-b".to_string(),
        (0..n)
            .map(|i| {
                let act = if i % 2 == 0 { 0.8 } else { 0.2 };
                record("input-b", i, act, vec![])
            })
            .collect(),
    ));

    // output-1: error depends on both inputs.
    records.push((
        "output-1".to_string(),
        (0..n)
            .map(|i| {
                let act_a = if i < n / 2 { 0.9 } else { 0.1 };
                let act_b = if i % 2 == 0 { 0.8 } else { 0.2 };
                let error = 0.4 * act_a + 0.3 * act_b;
                record("output-1", i, 0.5, vec![error])
            })
            .collect(),
    ));

    (creature, records)
}

// ─── Test 1: Detection of individually successful candidates ────────────────

#[test]
fn test_detects_individually_successful_candidates() {
    let (creature, records) = make_two_successful_inputs();
    let candidates = detect_individually_successful(&creature, &records);

    assert!(
        !candidates.is_empty(),
        "Should detect individually successful candidates"
    );

    // Both input-a and input-b should be detected.
    let has_a = candidates.iter().any(|c| c.source_uuid == "input-a");
    let has_b = candidates.iter().any(|c| c.source_uuid == "input-b");
    assert!(has_a, "Should detect input-a as individually successful");
    assert!(has_b, "Should detect input-b as individually successful");
}

// ─── Test 2: No candidates when improvement is too low ──────────────────────

#[test]
fn test_no_candidates_for_low_improvement() {
    let creature = CreatureJson {
        neurons: vec![neuron("input-a", "input"), neuron("output-1", "output")],
        synapses: vec![],
        input: 1,
        output: 1,
    };

    let n = 100_u32;
    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        // Constant activation — no correlation with error.
        (
            "input-a".to_string(),
            (0..n).map(|i| record("input-a", i, 0.5, vec![])).collect(),
        ),
        (
            "output-1".to_string(),
            (0..n)
                .map(|i| {
                    let error = if i % 3 == 0 { 0.1 } else { -0.05 };
                    record("output-1", i, 0.5, vec![error])
                })
                .collect(),
        ),
    ];

    let candidates = detect_individually_successful(&creature, &records);
    assert!(
        candidates.is_empty(),
        "Should not detect candidates when improvement is too low"
    );
}

// ─── Test 3: Structural conflict detection ──────────────────────────────────

#[test]
fn test_structural_conflict_same_source_and_target() {
    let a = IndividualCandidate {
        source_uuid: "input-a".to_string(),
        target_uuid: "output-1".to_string(),
        weight: 0.5,
        improvement: 0.05,
        sample_count: 100,
    };
    let b = IndividualCandidate {
        source_uuid: "input-a".to_string(),
        target_uuid: "output-1".to_string(),
        weight: 0.3,
        improvement: 0.03,
        sample_count: 100,
    };

    assert!(
        has_structural_conflict(&a, &b),
        "Same source + target should be a conflict"
    );
}

// ─── Test 4: No conflict for different sources to same target ───────────────

#[test]
fn test_no_conflict_different_sources_same_target() {
    let a = IndividualCandidate {
        source_uuid: "input-a".to_string(),
        target_uuid: "output-1".to_string(),
        weight: 0.5,
        improvement: 0.05,
        sample_count: 100,
    };
    let b = IndividualCandidate {
        source_uuid: "input-b".to_string(),
        target_uuid: "output-1".to_string(),
        weight: 0.3,
        improvement: 0.03,
        sample_count: 100,
    };

    assert!(
        !has_structural_conflict(&a, &b),
        "Different sources to same target should not conflict"
    );
}

// ─── Test 5: Batch grouping produces groups of 2–4 ─────────────────────────

#[test]
fn test_group_into_batches_produces_valid_groups() {
    let candidates = vec![
        IndividualCandidate {
            source_uuid: "input-a".to_string(),
            target_uuid: "output-1".to_string(),
            weight: 0.5,
            improvement: 0.10,
            sample_count: 100,
        },
        IndividualCandidate {
            source_uuid: "input-b".to_string(),
            target_uuid: "output-1".to_string(),
            weight: 0.3,
            improvement: 0.08,
            sample_count: 100,
        },
        IndividualCandidate {
            source_uuid: "input-c".to_string(),
            target_uuid: "output-1".to_string(),
            weight: 0.2,
            improvement: 0.05,
            sample_count: 100,
        },
    ];

    let groups = group_into_batches(&candidates);

    assert!(
        !groups.is_empty(),
        "Should produce at least one batch group"
    );

    for group in &groups {
        assert!(
            group.candidates.len() >= 2,
            "Each batch should have at least 2 candidates, got {}",
            group.candidates.len()
        );
        assert!(
            group.candidates.len() <= 4,
            "Each batch should have at most 4 candidates, got {}",
            group.candidates.len()
        );
    }
}

// ─── Test 6: Batch respects max size ────────────────────────────────────────

#[test]
fn test_batch_max_size_respected() {
    // 6 non-conflicting candidates — should form batches of max 4.
    let weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    let improvements = [0.10, 0.09, 0.08, 0.07, 0.06, 0.05];
    let candidates: Vec<IndividualCandidate> = (0..6)
        .map(|i| IndividualCandidate {
            source_uuid: format!("input-{i}"),
            target_uuid: "output-1".to_string(),
            weight: weights[i],
            improvement: improvements[i],
            sample_count: 100,
        })
        .collect();

    let groups = group_into_batches(&candidates);

    for group in &groups {
        assert!(
            group.candidates.len() <= 4,
            "Batch size should not exceed 4, got {}",
            group.candidates.len()
        );
    }
}

// ─── Test 7: Conversion produces valid coordinated candidates ───────────────

#[test]
fn test_conversion_produces_valid_coordinated_candidates() {
    let groups = vec![BatchSuccessfulGroup {
        candidates: vec![
            IndividualCandidate {
                source_uuid: "input-a".to_string(),
                target_uuid: "output-1".to_string(),
                weight: 0.5,
                improvement: 0.10,
                sample_count: 100,
            },
            IndividualCandidate {
                source_uuid: "input-b".to_string(),
                target_uuid: "output-1".to_string(),
                weight: 0.3,
                improvement: 0.08,
                sample_count: 100,
            },
        ],
        combined_improvement: 0.0018,
        reason: "Batch-successful: test".to_string(),
    }];

    let coordinated = batch_successful_to_coordinated_candidates(&groups);

    assert_eq!(
        coordinated.len(),
        1,
        "Should produce one coordinated candidate"
    );
    assert_eq!(
        coordinated[0].operations.len(),
        2,
        "Should have 2 AddSynapse operations"
    );
    assert!(
        coordinated[0].expected_creature_score_gain > 0.0,
        "Expected gain should be positive"
    );

    // Verify all operations are AddSynapse.
    let ops_json = serde_json::to_string(&coordinated[0].operations).unwrap();
    assert!(
        ops_json.contains("addSynapse"),
        "All operations should be addSynapse: {ops_json}"
    );
}

// ─── Test 8: Combined improvement is positive ──────────────────────────────

#[test]
fn test_combined_improvement_positive() {
    let candidates = vec![
        IndividualCandidate {
            source_uuid: "input-a".to_string(),
            target_uuid: "output-1".to_string(),
            weight: 0.5,
            improvement: 0.10,
            sample_count: 100,
        },
        IndividualCandidate {
            source_uuid: "input-b".to_string(),
            target_uuid: "output-1".to_string(),
            weight: 0.3,
            improvement: 0.08,
            sample_count: 100,
        },
    ];

    let groups = group_into_batches(&candidates);

    for group in &groups {
        assert!(
            group.combined_improvement > 0.0,
            "Combined improvement should be positive, got {}",
            group.combined_improvement
        );
    }
}

// ─── Test 9: Empty/insufficient inputs produce no results ──────────────────

#[test]
fn test_empty_records_no_candidates() {
    let creature = CreatureJson {
        neurons: vec![neuron("input-a", "input"), neuron("output-1", "output")],
        synapses: vec![],
        input: 1,
        output: 1,
    };

    let candidates = detect_individually_successful(&creature, &[]);
    assert!(
        candidates.is_empty(),
        "Empty records should produce no candidates"
    );
}

#[test]
fn test_insufficient_samples_no_candidates() {
    let creature = CreatureJson {
        neurons: vec![neuron("input-a", "input"), neuron("output-1", "output")],
        synapses: vec![],
        input: 1,
        output: 1,
    };

    // Only 5 samples — below MIN_DISCOVERY_SAMPLE_COUNT (20).
    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        (
            "input-a".to_string(),
            (0..5).map(|i| record("input-a", i, 0.8, vec![])).collect(),
        ),
        (
            "output-1".to_string(),
            (0..5)
                .map(|i| record("output-1", i, 0.5, vec![0.3]))
                .collect(),
        ),
    ];

    let candidates = detect_individually_successful(&creature, &records);
    assert!(
        candidates.is_empty(),
        "Insufficient samples should produce no candidates"
    );
}

#[test]
fn test_single_candidate_no_batch() {
    let candidates = vec![IndividualCandidate {
        source_uuid: "input-a".to_string(),
        target_uuid: "output-1".to_string(),
        weight: 0.5,
        improvement: 0.10,
        sample_count: 100,
    }];

    let groups = group_into_batches(&candidates);
    assert!(
        groups.is_empty(),
        "Single candidate should not form a batch"
    );
}

// ─── Test 10: Comments distinguish batch-successful ────────────────────────

#[test]
fn test_comments_identify_batch_successful() {
    let groups = vec![BatchSuccessfulGroup {
        candidates: vec![
            IndividualCandidate {
                source_uuid: "input-a".to_string(),
                target_uuid: "output-1".to_string(),
                weight: 0.5,
                improvement: 0.10,
                sample_count: 100,
            },
            IndividualCandidate {
                source_uuid: "input-b".to_string(),
                target_uuid: "output-1".to_string(),
                weight: 0.3,
                improvement: 0.08,
                sample_count: 100,
            },
        ],
        combined_improvement: 0.0018,
        reason: "Batch-successful: 2 individually proven sources".to_string(),
    }];

    let coordinated = batch_successful_to_coordinated_candidates(&groups);

    assert_eq!(coordinated.len(), 1);
    let comment = coordinated[0].comment.as_ref().unwrap();
    assert!(
        comment.contains("Batch-successful"),
        "Comment should identify batch-successful type: {comment}"
    );
}

// ─── Test 11: End-to-end pipeline ──────────────────────────────────────────

#[test]
fn test_detect_batch_successful_groups_end_to_end() {
    let (creature, records) = make_two_successful_inputs();
    let groups = detect_batch_successful_groups(&creature, &records);

    // With two strong individually successful inputs targeting the same output,
    // should produce at least one batch group.
    assert!(
        !groups.is_empty(),
        "End-to-end pipeline should produce batch groups"
    );

    let coordinated = batch_successful_to_coordinated_candidates(&groups);
    assert!(
        !coordinated.is_empty(),
        "Should produce coordinated candidates from batch groups"
    );

    // Verify the coordinated candidate has multiple operations.
    for c in &coordinated {
        assert!(
            c.operations.len() >= 2,
            "Batch candidate should have at least 2 operations"
        );
    }
}

// ─── Test 12: Existing synapses excluded from detection ────────────────────

#[test]
fn test_existing_synapses_excluded() {
    let creature = CreatureJson {
        neurons: vec![
            neuron("input-a", "input"),
            neuron("input-b", "input"),
            neuron("output-1", "output"),
        ],
        // input-a already connected to output-1.
        synapses: vec![synapse("input-a", "output-1", 0.5)],
        input: 2,
        output: 1,
    };

    let n = 100_u32;
    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        (
            "input-a".to_string(),
            (0..n)
                .map(|i| {
                    let act = if i < n / 2 { 0.9 } else { 0.1 };
                    record("input-a", i, act, vec![])
                })
                .collect(),
        ),
        (
            "input-b".to_string(),
            (0..n)
                .map(|i| {
                    let act = if i % 2 == 0 { 0.8 } else { 0.2 };
                    record("input-b", i, act, vec![])
                })
                .collect(),
        ),
        (
            "output-1".to_string(),
            (0..n)
                .map(|i| {
                    let act_a = if i < n / 2 { 0.9 } else { 0.1 };
                    let error = 0.5 * act_a;
                    record("output-1", i, 0.5, vec![error])
                })
                .collect(),
        ),
    ];

    let candidates = detect_individually_successful(&creature, &records);

    // input-a → output-1 should be excluded (synapse exists).
    let has_a_to_output = candidates
        .iter()
        .any(|c| c.source_uuid == "input-a" && c.target_uuid == "output-1");
    assert!(
        !has_a_to_output,
        "Existing synapse input-a → output-1 should be excluded"
    );
}
