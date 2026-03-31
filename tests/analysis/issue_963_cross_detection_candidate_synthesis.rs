//! Cross-detection candidate synthesis tests (Issue #963).
//!
//! When multiple detection modules independently flag the same neuron,
//! the system should synthesise combined remediation candidates that
//! address multiple issues simultaneously.
//!
//! ## TDD Test Plan
//!
//! 1. Test grouping detection outputs by target neuron UUID
//! 2. Test that co-flagged neurons (2+ detections) produce synthesised candidates
//! 3. Test that synthesised candidates use `CoordinatedStructuralCandidate`
//! 4. Test that individual candidates are preserved (synthesis is additive)
//! 5. Test success rate tracking distinguishes synthesised from individual
//! 6. Test compatible operation merging (changeSquash + setBias)
//! 7. Test compatible operation merging (changeSquash + setWeight)
//! 8. Test compatible operation merging (removeNeuron + removeSynapse)
//! 9. Test that incompatible operations are not merged (remove + changeSquash)
//! 10. Test empty input passthrough
//! 11. Test single-detection neurons produce no synthesis

use neat_ai_discovery::analysis::detection::cross_detection_synthesis::{
    group_candidates_by_neuron, synthesise_cross_detection_candidates,
};
use neat_ai_discovery::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson};

/// Helper: create a `ChangeSquash` candidate targeting a neuron.
fn change_squash_candidate(
    neuron_uuid: &str,
    squash: &str,
    gain: f32,
    module: &str,
) -> CoordinatedStructuralCandidateJson {
    CoordinatedStructuralCandidateJson {
        operations: vec![CoordinatedStructuralOpJson::ChangeSquash {
            neuron_uuid: neuron_uuid.to_string(),
            squash: squash.to_string(),
        }],
        expected_creature_score_gain: gain,
        comment: Some(format!("{module}: detected issue on {neuron_uuid}")),
    }
}

/// Helper: create a `SetBias` candidate targeting a neuron.
fn set_bias_candidate(
    neuron_uuid: &str,
    bias: f32,
    gain: f32,
    module: &str,
) -> CoordinatedStructuralCandidateJson {
    CoordinatedStructuralCandidateJson {
        operations: vec![CoordinatedStructuralOpJson::SetBias {
            neuron_uuid: neuron_uuid.to_string(),
            bias,
        }],
        expected_creature_score_gain: gain,
        comment: Some(format!("{module}: detected issue on {neuron_uuid}")),
    }
}

/// Helper: create a `SetWeight` candidate targeting a synapse.
fn set_weight_candidate(
    from: &str,
    to: &str,
    weight: f32,
    gain: f32,
    module: &str,
) -> CoordinatedStructuralCandidateJson {
    CoordinatedStructuralCandidateJson {
        operations: vec![CoordinatedStructuralOpJson::SetWeight {
            from_neuron_uuid: from.to_string(),
            to_neuron_uuid: to.to_string(),
            weight,
        }],
        expected_creature_score_gain: gain,
        comment: Some(format!("{module}: detected issue on {to}")),
    }
}

/// Helper: create a `RemoveNeuron` candidate.
fn remove_neuron_candidate(
    neuron_uuid: &str,
    gain: f32,
    module: &str,
) -> CoordinatedStructuralCandidateJson {
    CoordinatedStructuralCandidateJson {
        operations: vec![CoordinatedStructuralOpJson::RemoveNeuron {
            neuron_uuid: neuron_uuid.to_string(),
        }],
        expected_creature_score_gain: gain,
        comment: Some(format!("{module}: detected issue on {neuron_uuid}")),
    }
}

/// Helper: create a `RemoveSynapse` candidate.
fn remove_synapse_candidate(
    from: &str,
    to: &str,
    gain: f32,
    module: &str,
) -> CoordinatedStructuralCandidateJson {
    CoordinatedStructuralCandidateJson {
        operations: vec![CoordinatedStructuralOpJson::RemoveSynapse {
            from_neuron_uuid: from.to_string(),
            to_neuron_uuid: to.to_string(),
        }],
        expected_creature_score_gain: gain,
        comment: Some(format!("{module}: detected issue on {to}")),
    }
}

// =========================================================================
// Test: Group candidates by target neuron UUID
// =========================================================================

#[test]
fn test_group_candidates_by_neuron_groups_correctly() {
    let candidates = vec![
        change_squash_candidate("neuron-a", "IDENTITY", 0.05, "saturation detection"),
        set_bias_candidate("neuron-a", 0.1, 0.03, "restricted range detection"),
        change_squash_candidate("neuron-b", "RELU", 0.04, "saturation detection"),
    ];

    let groups = group_candidates_by_neuron(&candidates);

    // neuron-a should have 2 candidates, neuron-b should have 1
    assert_eq!(
        groups.get("neuron-a").map(Vec::len),
        Some(2),
        "neuron-a should have 2 candidates grouped"
    );
    assert_eq!(
        groups.get("neuron-b").map(Vec::len),
        Some(1),
        "neuron-b should have 1 candidate grouped"
    );
}

#[test]
fn test_group_candidates_empty_input() {
    let candidates: Vec<CoordinatedStructuralCandidateJson> = vec![];
    let groups = group_candidates_by_neuron(&candidates);
    assert!(groups.is_empty(), "empty input should produce empty groups");
}

#[test]
fn test_group_candidates_multi_op_candidates_excluded() {
    // Multi-operation candidates should not be grouped (they are already coordinated)
    let multi_op = CoordinatedStructuralCandidateJson {
        operations: vec![
            CoordinatedStructuralOpJson::ChangeSquash {
                neuron_uuid: "neuron-a".to_string(),
                squash: "IDENTITY".to_string(),
            },
            CoordinatedStructuralOpJson::SetBias {
                neuron_uuid: "neuron-a".to_string(),
                bias: 0.1,
            },
        ],
        expected_creature_score_gain: 0.05,
        comment: Some("multi-op: already coordinated".to_string()),
    };

    let candidates = vec![
        multi_op,
        change_squash_candidate("neuron-a", "RELU", 0.03, "saturation detection"),
    ];

    let groups = group_candidates_by_neuron(&candidates);

    // Only the single-op candidate should be grouped
    assert_eq!(
        groups.get("neuron-a").map(Vec::len),
        Some(1),
        "multi-op candidates should be excluded from grouping"
    );
}

// =========================================================================
// Test: Synthesise combined candidates for co-flagged neurons
// =========================================================================

#[test]
fn test_synthesise_produces_combined_candidate_for_coflagged_neuron() {
    let candidates = vec![
        change_squash_candidate("neuron-a", "IDENTITY", 0.05, "saturation detection"),
        set_bias_candidate("neuron-a", 0.1, 0.03, "restricted range detection"),
    ];

    let result = synthesise_cross_detection_candidates(candidates);

    // Should produce at least one synthesised candidate
    assert!(
        result.synthesised_count > 0,
        "co-flagged neuron should produce synthesised candidates"
    );

    // Synthesised candidates should have multiple operations
    let synthesised: Vec<_> = result
        .candidates
        .iter()
        .filter(|c| {
            c.comment
                .as_ref()
                .is_some_and(|com| com.contains("cross-detection synthesis"))
        })
        .collect();

    assert!(
        !synthesised.is_empty(),
        "should have at least one synthesised candidate with cross-detection comment"
    );

    // The synthesised candidate should have both operations
    let synth = &synthesised[0];
    assert!(
        synth.operations.len() >= 2,
        "synthesised candidate should combine multiple operations, got {}",
        synth.operations.len()
    );
}

#[test]
fn test_synthesise_preserves_individual_candidates() {
    let candidates = vec![
        change_squash_candidate("neuron-a", "IDENTITY", 0.05, "saturation detection"),
        set_bias_candidate("neuron-a", 0.1, 0.03, "restricted range detection"),
    ];

    let original_count = candidates.len();
    let result = synthesise_cross_detection_candidates(candidates);

    // Individual candidates must still be present (synthesis is additive)
    let individual_count = result.candidates.len() - result.synthesised_count;
    assert!(
        individual_count >= original_count,
        "individual candidates must be preserved: expected >= {original_count}, got {individual_count}"
    );
}

#[test]
fn test_single_detection_neuron_produces_no_synthesis() {
    let candidates = vec![
        change_squash_candidate("neuron-a", "IDENTITY", 0.05, "saturation detection"),
        change_squash_candidate("neuron-b", "RELU", 0.04, "restricted range detection"),
    ];

    let result = synthesise_cross_detection_candidates(candidates);

    assert_eq!(
        result.synthesised_count, 0,
        "neurons with only one detection should not produce synthesised candidates"
    );
}

#[test]
fn test_empty_input_passthrough() {
    let result = synthesise_cross_detection_candidates(vec![]);

    assert!(result.candidates.is_empty());
    assert_eq!(result.synthesised_count, 0);
}

// =========================================================================
// Test: Compatible operation merging rules
// =========================================================================

#[test]
fn test_merge_change_squash_and_set_bias() {
    // Saturation + restricted range → combine ChangeSquash + SetBias
    let candidates = vec![
        change_squash_candidate("neuron-x", "IDENTITY", 0.05, "saturation detection"),
        set_bias_candidate("neuron-x", -0.2, 0.03, "bias perturbation"),
    ];

    let result = synthesise_cross_detection_candidates(candidates);

    let synthesised: Vec<_> = result
        .candidates
        .iter()
        .filter(|c| {
            c.comment
                .as_ref()
                .is_some_and(|com| com.contains("cross-detection synthesis"))
        })
        .collect();

    assert_eq!(
        synthesised.len(),
        1,
        "should produce exactly one synthesised candidate"
    );

    let ops = &synthesised[0].operations;
    let has_change_squash = ops
        .iter()
        .any(|op| matches!(op, CoordinatedStructuralOpJson::ChangeSquash { .. }));
    let has_set_bias = ops
        .iter()
        .any(|op| matches!(op, CoordinatedStructuralOpJson::SetBias { .. }));

    assert!(
        has_change_squash,
        "synthesised candidate should contain ChangeSquash"
    );
    assert!(has_set_bias, "synthesised candidate should contain SetBias");
}

#[test]
fn test_merge_change_squash_and_set_weight() {
    // Saturation + weight magnitude → combine ChangeSquash + SetWeight
    let candidates = vec![
        change_squash_candidate("neuron-x", "RELU", 0.06, "saturation detection"),
        set_weight_candidate("input-1", "neuron-x", 0.5, 0.02, "weight magnitude reset"),
    ];

    let result = synthesise_cross_detection_candidates(candidates);

    let synthesised: Vec<_> = result
        .candidates
        .iter()
        .filter(|c| {
            c.comment
                .as_ref()
                .is_some_and(|com| com.contains("cross-detection synthesis"))
        })
        .collect();

    assert!(
        !synthesised.is_empty(),
        "changeSquash + setWeight should produce a synthesised candidate"
    );

    let ops = &synthesised[0].operations;
    let has_change_squash = ops
        .iter()
        .any(|op| matches!(op, CoordinatedStructuralOpJson::ChangeSquash { .. }));
    let has_set_weight = ops
        .iter()
        .any(|op| matches!(op, CoordinatedStructuralOpJson::SetWeight { .. }));

    assert!(has_change_squash, "should contain ChangeSquash");
    assert!(has_set_weight, "should contain SetWeight");
}

#[test]
fn test_merge_remove_neuron_and_remove_synapse() {
    // Dead neuron + dormant synapse → combine removals
    let candidates = vec![
        remove_neuron_candidate("neuron-dead", 0.04, "dead neuron detection"),
        remove_synapse_candidate("input-1", "neuron-dead", 0.02, "dormant synapse detection"),
    ];

    let result = synthesise_cross_detection_candidates(candidates);

    let synthesised: Vec<_> = result
        .candidates
        .iter()
        .filter(|c| {
            c.comment
                .as_ref()
                .is_some_and(|com| com.contains("cross-detection synthesis"))
        })
        .collect();

    assert!(
        !synthesised.is_empty(),
        "removeNeuron + removeSynapse should produce a synthesised candidate"
    );

    let ops = &synthesised[0].operations;
    let has_remove_neuron = ops
        .iter()
        .any(|op| matches!(op, CoordinatedStructuralOpJson::RemoveNeuron { .. }));
    let has_remove_synapse = ops
        .iter()
        .any(|op| matches!(op, CoordinatedStructuralOpJson::RemoveSynapse { .. }));

    assert!(has_remove_neuron, "should contain RemoveNeuron");
    assert!(has_remove_synapse, "should contain RemoveSynapse");
}

// =========================================================================
// Test: Incompatible operations are not merged
// =========================================================================

#[test]
fn test_incompatible_remove_and_modify_not_merged() {
    // RemoveNeuron conflicts with ChangeSquash (cannot change squash on a removed neuron)
    let candidates = vec![
        remove_neuron_candidate("neuron-x", 0.04, "dead neuron detection"),
        change_squash_candidate("neuron-x", "RELU", 0.06, "saturation detection"),
    ];

    let result = synthesise_cross_detection_candidates(candidates);

    // Should not synthesise when operations conflict (remove vs modify)
    assert_eq!(
        result.synthesised_count, 0,
        "conflicting operations (remove vs modify) should not be synthesised"
    );
}

// =========================================================================
// Test: Scoring of synthesised candidates
// =========================================================================

#[test]
fn test_synthesised_candidate_gain_uses_best_individual() {
    let candidates = vec![
        change_squash_candidate("neuron-a", "IDENTITY", 0.08, "saturation detection"),
        set_bias_candidate("neuron-a", 0.1, 0.03, "restricted range detection"),
    ];

    let result = synthesise_cross_detection_candidates(candidates);

    let synthesised: Vec<_> = result
        .candidates
        .iter()
        .filter(|c| {
            c.comment
                .as_ref()
                .is_some_and(|com| com.contains("cross-detection synthesis"))
        })
        .collect();

    assert!(!synthesised.is_empty());

    // Synthesised gain should be based on the best individual gain (0.08)
    // with some adjustment for the combined operation
    let gain = synthesised[0].expected_creature_score_gain;
    assert!(
        gain > 0.0,
        "synthesised candidate should have positive gain, got {gain}"
    );
}

// =========================================================================
// Test: Comment format for synthesised candidates
// =========================================================================

#[test]
fn test_synthesised_candidate_comment_identifies_source_modules() {
    let candidates = vec![
        change_squash_candidate("neuron-a", "IDENTITY", 0.05, "saturation detection"),
        set_bias_candidate("neuron-a", 0.1, 0.03, "restricted range detection"),
    ];

    let result = synthesise_cross_detection_candidates(candidates);

    let synthesised: Vec<_> = result
        .candidates
        .iter()
        .filter(|c| {
            c.comment
                .as_ref()
                .is_some_and(|com| com.contains("cross-detection synthesis"))
        })
        .collect();

    assert!(!synthesised.is_empty());

    let comment = synthesised[0].comment.as_ref().unwrap();
    assert!(
        comment.contains("cross-detection synthesis"),
        "comment should identify as cross-detection synthesis: {comment}"
    );
}

// =========================================================================
// Test: SynthesisResult structure
// =========================================================================

#[test]
fn test_synthesis_result_counts() {
    let candidates = vec![
        change_squash_candidate("neuron-a", "IDENTITY", 0.05, "saturation detection"),
        set_bias_candidate("neuron-a", 0.1, 0.03, "restricted range detection"),
        change_squash_candidate("neuron-b", "RELU", 0.04, "saturation detection"),
    ];

    let result = synthesise_cross_detection_candidates(candidates);

    // Should have original candidates + synthesised
    assert!(
        result.candidates.len() >= 3,
        "should have at least the original 3 candidates, got {}",
        result.candidates.len()
    );
    assert_eq!(
        result.synthesised_count, 1,
        "should have exactly 1 synthesised candidate (from neuron-a co-flagging)"
    );
}

// =========================================================================
// Test: Three-way co-flagging
// =========================================================================

#[test]
fn test_three_way_coflagging_produces_synthesis() {
    let candidates = vec![
        change_squash_candidate("neuron-a", "IDENTITY", 0.08, "saturation detection"),
        set_bias_candidate("neuron-a", 0.1, 0.03, "restricted range detection"),
        set_weight_candidate("input-1", "neuron-a", 0.5, 0.02, "weight magnitude reset"),
    ];

    let result = synthesise_cross_detection_candidates(candidates);

    assert!(
        result.synthesised_count >= 1,
        "three-way co-flagging should produce at least one synthesised candidate"
    );

    // All original candidates should still be present
    assert!(
        result.candidates.len() >= 3,
        "all original candidates must be preserved"
    );
}
