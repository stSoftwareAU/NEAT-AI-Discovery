//! Tests for Issue #489: Cross-module candidate deduplication.
//!
//! When 25+ discovery modules run in parallel, different modules can independently
//! propose candidates targeting the same neuron or synapse with the same operation.
//! Cross-module deduplication removes these redundancies to avoid wasted ablation tests.
//!
//! ## TDD Plan
//! 1. Identical single-op candidates from different modules are deduplicated
//! 2. Best candidate (highest expected improvement) is kept as representative
//! 3. Candidates with different operation types are NOT deduplicated
//! 4. Candidates targeting different neurons are NOT deduplicated
//! 5. Empty input produces empty output
//! 6. Single candidate passes through unchanged
//! 7. Multi-operation candidates with identical ops are deduplicated
//! 8. Conflicting operations on same neuron are flagged in metadata
//! 9. Deduplication metadata reports counts correctly
//! 10. Large candidate sets are handled efficiently

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use neat_ai_discovery::CoordinatedStructuralCandidateJson;
use neat_ai_discovery::CoordinatedStructuralOpJson;
use neat_ai_discovery::analysis::candidate_clustering::deduplicate_cross_module_candidates;

/// Helper: create a single-op `ChangeSquash` candidate.
fn change_squash_candidate(
    neuron_uuid: &str,
    squash: &str,
    gain: f32,
    comment: &str,
) -> CoordinatedStructuralCandidateJson {
    CoordinatedStructuralCandidateJson {
        remove_neuron_compensation: None,
        operations: vec![CoordinatedStructuralOpJson::ChangeSquash {
            neuron_uuid: neuron_uuid.to_string(),
            squash: squash.to_string(),
        }],
        expected_creature_score_gain: gain,
        comment: Some(comment.to_string()),
    }
}

/// Helper: create a single-op `RemoveNeuron` candidate.
fn remove_neuron_candidate(
    neuron_uuid: &str,
    gain: f32,
    comment: &str,
) -> CoordinatedStructuralCandidateJson {
    CoordinatedStructuralCandidateJson {
        remove_neuron_compensation: None,
        operations: vec![CoordinatedStructuralOpJson::RemoveNeuron {
            neuron_uuid: neuron_uuid.to_string(),
        }],
        expected_creature_score_gain: gain,
        comment: Some(comment.to_string()),
    }
}

/// Helper: create a single-op `SetBias` candidate.
fn set_bias_candidate(
    neuron_uuid: &str,
    bias: f32,
    gain: f32,
    comment: &str,
) -> CoordinatedStructuralCandidateJson {
    CoordinatedStructuralCandidateJson {
        remove_neuron_compensation: None,
        operations: vec![CoordinatedStructuralOpJson::SetBias {
            neuron_uuid: neuron_uuid.to_string(),
            bias,
        }],
        expected_creature_score_gain: gain,
        comment: Some(comment.to_string()),
    }
}

/// Helper: create an `AddSynapse` candidate.
fn add_synapse_candidate(
    from: &str,
    to: &str,
    weight: f32,
    gain: f32,
    comment: &str,
) -> CoordinatedStructuralCandidateJson {
    CoordinatedStructuralCandidateJson {
        remove_neuron_compensation: None,
        operations: vec![CoordinatedStructuralOpJson::AddSynapse {
            from_neuron_uuid: from.to_string(),
            to_neuron_uuid: to.to_string(),
            weight,
        }],
        expected_creature_score_gain: gain,
        comment: Some(comment.to_string()),
    }
}

/// Test 1: Identical single-op candidates from different modules are deduplicated.
/// Both saturation and oscillating detection propose changeSquash for the same neuron
/// with the same activation — only the best should survive.
#[test]
fn test_identical_change_squash_deduplicated() {
    let candidates = vec![
        change_squash_candidate("neuron-5", "TANH", 0.05, "saturation"),
        change_squash_candidate("neuron-5", "TANH", 0.048, "oscillating"),
    ];

    let result = deduplicate_cross_module_candidates(candidates);

    assert_eq!(
        result.candidates.len(),
        1,
        "Identical changeSquash for same neuron should be deduplicated to 1"
    );
    assert_eq!(
        result.candidates[0].expected_creature_score_gain, 0.05,
        "Should keep the candidate with highest expected improvement"
    );
    assert_eq!(result.duplicates_removed, 1);
}

/// Test 2: Best candidate (highest expected improvement) is kept as representative.
#[test]
fn test_best_candidate_kept() {
    let candidates = vec![
        change_squash_candidate("neuron-5", "RELU", 0.02, "module-A"),
        change_squash_candidate("neuron-5", "RELU", 0.08, "module-B"),
        change_squash_candidate("neuron-5", "RELU", 0.05, "module-C"),
    ];

    let result = deduplicate_cross_module_candidates(candidates);

    assert_eq!(result.candidates.len(), 1);
    assert_eq!(result.candidates[0].expected_creature_score_gain, 0.08);
    assert_eq!(result.duplicates_removed, 2);
}

/// Test 3: Candidates with different operation types targeting the same neuron
/// are NOT deduplicated — they represent genuinely different interventions.
#[test]
fn test_different_op_types_not_deduplicated() {
    let candidates = vec![
        change_squash_candidate("neuron-5", "TANH", 0.05, "saturation"),
        remove_neuron_candidate("neuron-5", 0.03, "dead-neuron"),
    ];

    let result = deduplicate_cross_module_candidates(candidates);

    assert_eq!(
        result.candidates.len(),
        2,
        "Different operation types should both be kept"
    );
    // Conflicts should be flagged
    assert!(
        result.conflicts_detected > 0,
        "Contradictory operations on same neuron should be flagged as conflicts"
    );
}

/// Test 4: Candidates targeting different neurons are NOT deduplicated.
#[test]
fn test_different_targets_not_deduplicated() {
    let candidates = vec![
        change_squash_candidate("neuron-5", "TANH", 0.05, "saturation"),
        change_squash_candidate("neuron-10", "TANH", 0.04, "saturation"),
    ];

    let result = deduplicate_cross_module_candidates(candidates);

    assert_eq!(
        result.candidates.len(),
        2,
        "Different target neurons should both be kept"
    );
    assert_eq!(result.duplicates_removed, 0);
}

/// Test 5: Empty input produces empty output.
#[test]
fn test_empty_input() {
    let result = deduplicate_cross_module_candidates(vec![]);

    assert!(result.candidates.is_empty());
    assert_eq!(result.duplicates_removed, 0);
    assert_eq!(result.conflicts_detected, 0);
}

/// Test 6: Single candidate passes through unchanged.
#[test]
fn test_cross_module_dedup_single_candidate_passthrough() {
    let candidates = vec![change_squash_candidate(
        "neuron-5",
        "TANH",
        0.05,
        "saturation",
    )];

    let result = deduplicate_cross_module_candidates(candidates);

    assert_eq!(result.candidates.len(), 1);
    assert_eq!(result.duplicates_removed, 0);
}

/// Test 7: Multi-operation candidates with identical operation lists are deduplicated.
#[test]
fn test_multi_op_candidates_deduplicated() {
    let candidate1 = CoordinatedStructuralCandidateJson {
        remove_neuron_compensation: None,
        operations: vec![
            CoordinatedStructuralOpJson::RemoveSynapse {
                from_neuron_uuid: "input-0".to_string(),
                to_neuron_uuid: "output-0".to_string(),
            },
            CoordinatedStructuralOpJson::AddSynapse {
                from_neuron_uuid: "input-0".to_string(),
                to_neuron_uuid: "hidden-1".to_string(),
                weight: 0.5,
            },
        ],
        expected_creature_score_gain: 0.10,
        comment: Some("module-A".to_string()),
    };

    let candidate2 = CoordinatedStructuralCandidateJson {
        remove_neuron_compensation: None,
        operations: vec![
            CoordinatedStructuralOpJson::RemoveSynapse {
                from_neuron_uuid: "input-0".to_string(),
                to_neuron_uuid: "output-0".to_string(),
            },
            CoordinatedStructuralOpJson::AddSynapse {
                from_neuron_uuid: "input-0".to_string(),
                to_neuron_uuid: "hidden-1".to_string(),
                weight: 0.5,
            },
        ],
        expected_creature_score_gain: 0.12,
        comment: Some("module-B".to_string()),
    };

    let result = deduplicate_cross_module_candidates(vec![candidate1, candidate2]);

    assert_eq!(
        result.candidates.len(),
        1,
        "Identical multi-op candidates should be deduplicated"
    );
    assert_eq!(result.candidates[0].expected_creature_score_gain, 0.12);
    assert_eq!(result.duplicates_removed, 1);
}

/// Test 8: Conflicting operations on the same neuron are flagged in metadata.
/// e.g., one module says "remove neuron-5" and another says "change its activation".
#[test]
fn test_conflicting_ops_flagged() {
    let candidates = vec![
        remove_neuron_candidate("neuron-5", 0.10, "dead-neuron"),
        change_squash_candidate("neuron-5", "TANH", 0.08, "saturation"),
        set_bias_candidate("neuron-5", 0.1, 0.06, "bias-drift"),
    ];

    let result = deduplicate_cross_module_candidates(candidates);

    // All three should be kept (different operations)
    assert_eq!(result.candidates.len(), 3);
    // But conflicts involving the remove vs modify should be detected
    assert!(
        result.conflicts_detected > 0,
        "RemoveNeuron vs modify operations on same neuron should flag conflicts"
    );
}

/// Test 9: Deduplication metadata reports counts correctly.
#[test]
fn test_deduplication_metadata_counts() {
    let candidates = vec![
        // Group 1: three identical changeSquash for neuron-5 → 2 removed
        change_squash_candidate("neuron-5", "TANH", 0.05, "module-A"),
        change_squash_candidate("neuron-5", "TANH", 0.048, "module-B"),
        change_squash_candidate("neuron-5", "TANH", 0.04, "module-C"),
        // Group 2: two identical removeNeuron for neuron-10 → 1 removed
        remove_neuron_candidate("neuron-10", 0.03, "module-D"),
        remove_neuron_candidate("neuron-10", 0.025, "module-E"),
        // Unique candidate
        change_squash_candidate("neuron-20", "RELU", 0.07, "module-F"),
    ];

    let result = deduplicate_cross_module_candidates(candidates);

    assert_eq!(
        result.candidates.len(),
        3,
        "Should have 3 unique candidates after deduplication"
    );
    assert_eq!(
        result.duplicates_removed, 3,
        "Should have removed 3 duplicates (2 from group 1 + 1 from group 2)"
    );
    assert_eq!(
        result.original_count, 6,
        "Original count should reflect total input"
    );
}

/// Test 10: Large candidate sets are handled correctly.
#[test]
fn test_large_candidate_set_deduplication() {
    let mut candidates = Vec::new();

    // 20 modules each propose changeSquash for the same 5 neurons
    for module_idx in 0..20 {
        for neuron_idx in 0..5 {
            candidates.push(change_squash_candidate(
                &format!("neuron-{neuron_idx}"),
                "TANH",
                0.05 + (module_idx as f32 * 0.001),
                &format!("module-{module_idx}"),
            ));
        }
    }

    assert_eq!(candidates.len(), 100);

    let result = deduplicate_cross_module_candidates(candidates);

    assert_eq!(
        result.candidates.len(),
        5,
        "Should deduplicate to 5 unique candidates (one per neuron)"
    );
    assert_eq!(result.duplicates_removed, 95);
    assert_eq!(result.original_count, 100);
}

/// Test 11: `ChangeSquash` with different target squash values are NOT deduplicated
/// (they represent different proposed activations).
#[test]
fn test_different_squash_values_not_deduplicated() {
    let candidates = vec![
        change_squash_candidate("neuron-5", "TANH", 0.05, "module-A"),
        change_squash_candidate("neuron-5", "RELU", 0.04, "module-B"),
    ];

    let result = deduplicate_cross_module_candidates(candidates);

    assert_eq!(
        result.candidates.len(),
        2,
        "Different squash targets should both be kept"
    );
    assert_eq!(result.duplicates_removed, 0);
}

/// Test 12: `SetBias` candidates with very different bias values are NOT deduplicated.
#[test]
fn test_different_bias_values_not_deduplicated() {
    let candidates = vec![
        set_bias_candidate("neuron-5", 0.1, 0.05, "module-A"),
        set_bias_candidate("neuron-5", -0.5, 0.04, "module-B"),
    ];

    let result = deduplicate_cross_module_candidates(candidates);

    assert_eq!(
        result.candidates.len(),
        2,
        "SetBias with different bias values should both be kept"
    );
}

/// Test 13: `SetBias` candidates with similar bias values ARE deduplicated.
#[test]
fn test_similar_bias_values_deduplicated() {
    let candidates = vec![
        set_bias_candidate("neuron-5", 0.100, 0.05, "module-A"),
        set_bias_candidate("neuron-5", 0.101, 0.04, "module-B"),
    ];

    let result = deduplicate_cross_module_candidates(candidates);

    assert_eq!(
        result.candidates.len(),
        1,
        "SetBias with very similar bias values should be deduplicated"
    );
    assert_eq!(result.candidates[0].expected_creature_score_gain, 0.05);
}

/// Test 14: `AddSynapse` candidates with same endpoints but very different weights
/// are NOT deduplicated.
#[test]
fn test_add_synapse_different_weights_not_deduplicated() {
    let candidates = vec![
        add_synapse_candidate("input-0", "output-0", 0.5, 0.05, "module-A"),
        add_synapse_candidate("input-0", "output-0", -0.5, 0.04, "module-B"),
    ];

    let result = deduplicate_cross_module_candidates(candidates);

    assert_eq!(
        result.candidates.len(),
        2,
        "AddSynapse with very different weights should both be kept"
    );
}

/// Test 15: `AddSynapse` candidates with same endpoints and similar weights
/// ARE deduplicated.
#[test]
fn test_add_synapse_similar_weights_deduplicated() {
    let candidates = vec![
        add_synapse_candidate("input-0", "output-0", 0.50, 0.05, "module-A"),
        add_synapse_candidate("input-0", "output-0", 0.51, 0.04, "module-B"),
    ];

    let result = deduplicate_cross_module_candidates(candidates);

    assert_eq!(
        result.candidates.len(),
        1,
        "AddSynapse with similar weights should be deduplicated"
    );
}

/// Test 16: Output is sorted by expected improvement (best first).
#[test]
fn test_output_sorted_by_improvement() {
    let candidates = vec![
        change_squash_candidate("neuron-1", "TANH", 0.02, "module-A"),
        change_squash_candidate("neuron-2", "RELU", 0.08, "module-B"),
        change_squash_candidate("neuron-3", "TANH", 0.05, "module-C"),
    ];

    let result = deduplicate_cross_module_candidates(candidates);

    assert_eq!(result.candidates.len(), 3);
    assert!(
        result.candidates[0].expected_creature_score_gain
            >= result.candidates[1].expected_creature_score_gain
    );
    assert!(
        result.candidates[1].expected_creature_score_gain
            >= result.candidates[2].expected_creature_score_gain
    );
}
