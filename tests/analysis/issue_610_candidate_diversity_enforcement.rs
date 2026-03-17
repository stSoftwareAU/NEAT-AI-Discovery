//! Tests for Issue #610: Candidate diversity enforcement — penalise structurally similar candidates.
//!
//! When the discovery pipeline returns multiple candidates that are structurally similar
//! (e.g., removing adjacent synapses to the same target, or adding neurons at similar
//! positions), diversity-aware reranking penalises similar candidates so the evaluation
//! budget is spread across different mutation regions.
//!
//! ## TDD Plan
//! 1. Structural similarity metric correctly identifies similar candidates
//! 2. Identical candidates have similarity = 1.0
//! 3. Completely different candidates have similarity = 0.0
//! 4. Diversity reranking demotes similar lower-ranked candidates
//! 5. Diverse candidates remain unaffected by reranking
//! 6. Configurable diversity penalty changes reranking strength
//! 7. Empty or single-candidate input passes through unchanged
//! 8. Integration with existing candidate clustering

use neat_ai_discovery::analysis::candidate_diversity::{
    DiversityConfig, compute_structural_similarity, rerank_with_diversity,
};
use neat_ai_discovery::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson};

/// Helper: create a RemoveSynapse coordinated candidate.
fn remove_synapse_candidate(from: &str, to: &str, gain: f32) -> CoordinatedStructuralCandidateJson {
    CoordinatedStructuralCandidateJson {
        operations: vec![CoordinatedStructuralOpJson::RemoveSynapse {
            from_neuron_uuid: from.to_string(),
            to_neuron_uuid: to.to_string(),
        }],
        expected_creature_score_gain: gain,
        comment: None,
    }
}

/// Helper: create an AddSynapse coordinated candidate.
fn add_synapse_candidate(
    from: &str,
    to: &str,
    weight: f32,
    gain: f32,
) -> CoordinatedStructuralCandidateJson {
    CoordinatedStructuralCandidateJson {
        operations: vec![CoordinatedStructuralOpJson::AddSynapse {
            from_neuron_uuid: from.to_string(),
            to_neuron_uuid: to.to_string(),
            weight,
        }],
        expected_creature_score_gain: gain,
        comment: None,
    }
}

/// Helper: create a ChangeSquash coordinated candidate.
fn change_squash_candidate(
    neuron: &str,
    squash: &str,
    gain: f32,
) -> CoordinatedStructuralCandidateJson {
    CoordinatedStructuralCandidateJson {
        operations: vec![CoordinatedStructuralOpJson::ChangeSquash {
            neuron_uuid: neuron.to_string(),
            squash: squash.to_string(),
        }],
        expected_creature_score_gain: gain,
        comment: None,
    }
}

// =============================================================================
// Structural Similarity Metric Tests
// =============================================================================

/// Test 1: Two candidates removing synapses to the same target have high similarity.
#[test]
fn test_same_target_remove_synapse_high_similarity() {
    let a = remove_synapse_candidate("input-1", "output-0", 0.10);
    let b = remove_synapse_candidate("input-2", "output-0", 0.08);

    let sim = compute_structural_similarity(&a, &b);
    assert!(
        sim >= 0.5,
        "Candidates removing synapses to the same target should have high similarity, got {sim}"
    );
}

/// Test 2: Identical candidates have similarity = 1.0.
#[test]
fn test_identical_candidates_similarity_one() {
    let a = remove_synapse_candidate("input-1", "output-0", 0.10);
    let b = remove_synapse_candidate("input-1", "output-0", 0.10);

    let sim = compute_structural_similarity(&a, &b);
    assert!(
        (sim - 1.0).abs() < f32::EPSILON,
        "Identical candidates should have similarity = 1.0, got {sim}"
    );
}

/// Test 3: Completely different candidates have similarity = 0.0.
#[test]
fn test_different_candidates_low_similarity() {
    let a = remove_synapse_candidate("input-1", "output-0", 0.10);
    let b = change_squash_candidate("hidden-5", "RELU", 0.05);

    let sim = compute_structural_similarity(&a, &b);
    assert!(
        sim < 0.3,
        "Completely different candidates should have low similarity, got {sim}"
    );
}

/// Test 4: Different operation types on the same neuron have moderate similarity.
#[test]
fn test_same_neuron_different_ops_moderate_similarity() {
    let a = change_squash_candidate("hidden-5", "RELU", 0.10);
    let b = CoordinatedStructuralCandidateJson {
        operations: vec![CoordinatedStructuralOpJson::SetBias {
            neuron_uuid: "hidden-5".to_string(),
            bias: 0.5,
        }],
        expected_creature_score_gain: 0.08,
        comment: None,
    };

    let sim = compute_structural_similarity(&a, &b);
    assert!(
        sim > 0.0 && sim < 1.0,
        "Same neuron, different ops should have moderate similarity, got {sim}"
    );
}

// =============================================================================
// Diversity Reranking Tests
// =============================================================================

/// Test 5: Diversity reranking demotes similar lower-ranked candidates.
#[test]
fn test_reranking_demotes_similar_candidates() {
    // Three candidates all targeting the same output-0, two from adjacent inputs
    let candidates = vec![
        remove_synapse_candidate("input-1", "output-0", 0.10), // best
        remove_synapse_candidate("input-2", "output-0", 0.09), // similar to best
        add_synapse_candidate("input-50", "hidden-5", 0.5, 0.08), // different structure
    ];

    let config = DiversityConfig::default();
    let reranked = rerank_with_diversity(candidates, &config);

    assert_eq!(reranked.len(), 3);
    // The best candidate stays at the top
    assert_eq!(reranked[0].expected_creature_score_gain, 0.10);
    // The diverse candidate (add synapse to hidden-5) should be promoted relative to
    // the similar remove-synapse candidate
    let diverse_idx = reranked
        .iter()
        .position(|c| {
            c.operations.iter().any(|op| {
                matches!(op,
                CoordinatedStructuralOpJson::AddSynapse { to_neuron_uuid, .. }
                if to_neuron_uuid == "hidden-5")
            })
        })
        .expect("diverse candidate should be present");
    let similar_idx = reranked
        .iter()
        .position(|c| {
            c.operations.iter().any(|op| {
                matches!(op,
                CoordinatedStructuralOpJson::RemoveSynapse { from_neuron_uuid, .. }
                if from_neuron_uuid == "input-2")
            })
        })
        .expect("similar candidate should be present");

    assert!(
        diverse_idx < similar_idx,
        "Diverse candidate should be ranked above similar candidate after reranking"
    );
}

/// Test 6: Diverse candidates remain unaffected by reranking.
#[test]
fn test_diverse_candidates_unaffected() {
    // Three completely different candidates — ordering should stay by score
    let candidates = vec![
        remove_synapse_candidate("input-1", "output-0", 0.10),
        change_squash_candidate("hidden-5", "RELU", 0.09),
        add_synapse_candidate("input-50", "hidden-10", 0.5, 0.08),
    ];

    let config = DiversityConfig::default();
    let reranked = rerank_with_diversity(candidates, &config);

    // All candidates are diverse, so ordering should be by original score
    assert_eq!(reranked[0].expected_creature_score_gain, 0.10);
    assert_eq!(reranked[1].expected_creature_score_gain, 0.09);
    assert_eq!(reranked[2].expected_creature_score_gain, 0.08);
}

/// Test 7: Empty input passes through unchanged.
#[test]
fn test_empty_candidates_passthrough() {
    let config = DiversityConfig::default();
    let reranked = rerank_with_diversity(Vec::new(), &config);
    assert!(reranked.is_empty());
}

/// Test 8: Single candidate passes through unchanged.
#[test]
fn test_candidate_diversity_single_candidate_passthrough() {
    let candidates = vec![remove_synapse_candidate("input-1", "output-0", 0.10)];

    let config = DiversityConfig::default();
    let reranked = rerank_with_diversity(candidates, &config);

    assert_eq!(reranked.len(), 1);
    assert_eq!(reranked[0].expected_creature_score_gain, 0.10);
}

/// Test 9: Higher diversity penalty increases demotion of similar candidates.
#[test]
fn test_higher_penalty_stronger_demotion() {
    let make_candidates = || {
        vec![
            remove_synapse_candidate("input-1", "output-0", 0.10),
            remove_synapse_candidate("input-2", "output-0", 0.095),
            remove_synapse_candidate("input-3", "output-0", 0.090),
            add_synapse_candidate("input-50", "hidden-5", 0.5, 0.085),
        ]
    };

    let low_penalty = DiversityConfig {
        penalty_strength: 0.1,
    };
    let high_penalty = DiversityConfig {
        penalty_strength: 0.9,
    };

    let reranked_low = rerank_with_diversity(make_candidates(), &low_penalty);
    let reranked_high = rerank_with_diversity(make_candidates(), &high_penalty);

    // With high penalty, the diverse candidate should rank higher than with low penalty
    let diverse_rank_low = reranked_low
        .iter()
        .position(|c| {
            c.operations.iter().any(|op| {
                matches!(op,
                CoordinatedStructuralOpJson::AddSynapse { to_neuron_uuid, .. }
                if to_neuron_uuid == "hidden-5")
            })
        })
        .unwrap();
    let diverse_rank_high = reranked_high
        .iter()
        .position(|c| {
            c.operations.iter().any(|op| {
                matches!(op,
                CoordinatedStructuralOpJson::AddSynapse { to_neuron_uuid, .. }
                if to_neuron_uuid == "hidden-5")
            })
        })
        .unwrap();

    assert!(
        diverse_rank_high <= diverse_rank_low,
        "Higher penalty should promote diverse candidates more (high rank: {diverse_rank_high}, low rank: {diverse_rank_low})"
    );
}

/// Test 10: Zero penalty strength preserves original ordering.
#[test]
fn test_zero_penalty_preserves_ordering() {
    let candidates = vec![
        remove_synapse_candidate("input-1", "output-0", 0.10),
        remove_synapse_candidate("input-2", "output-0", 0.09),
        remove_synapse_candidate("input-3", "output-0", 0.08),
    ];

    let config = DiversityConfig {
        penalty_strength: 0.0,
    };
    let reranked = rerank_with_diversity(candidates, &config);

    assert_eq!(reranked[0].expected_creature_score_gain, 0.10);
    assert_eq!(reranked[1].expected_creature_score_gain, 0.09);
    assert_eq!(reranked[2].expected_creature_score_gain, 0.08);
}

/// Test 11: Candidates with multi-operation groups are compared correctly.
#[test]
fn test_multi_operation_similarity() {
    // Two coordinated candidates that both remove a synapse and add a neuron at similar positions
    let a = CoordinatedStructuralCandidateJson {
        operations: vec![
            CoordinatedStructuralOpJson::RemoveSynapse {
                from_neuron_uuid: "input-1".to_string(),
                to_neuron_uuid: "output-0".to_string(),
            },
            CoordinatedStructuralOpJson::AddNeuron {
                neuron_uuid: "new-1".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "LOGISTIC".to_string(),
                bias: 0.0,
                insert_before_neuron_uuid: Some("output-0".to_string()),
            },
        ],
        expected_creature_score_gain: 0.10,
        comment: None,
    };

    let b = CoordinatedStructuralCandidateJson {
        operations: vec![
            CoordinatedStructuralOpJson::RemoveSynapse {
                from_neuron_uuid: "input-2".to_string(),
                to_neuron_uuid: "output-0".to_string(),
            },
            CoordinatedStructuralOpJson::AddNeuron {
                neuron_uuid: "new-2".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "LOGISTIC".to_string(),
                bias: 0.0,
                insert_before_neuron_uuid: Some("output-0".to_string()),
            },
        ],
        expected_creature_score_gain: 0.08,
        comment: None,
    };

    let sim = compute_structural_similarity(&a, &b);
    // These candidates share op types (RemoveSynapse + AddNeuron) and the target neuron
    // (output-0), but have different generated neuron UUIDs and source neurons, so
    // similarity is moderate rather than high.
    assert!(
        sim >= 0.3,
        "Similar multi-op candidates should have moderate-to-high similarity, got {sim}"
    );
}

/// Test 12: Similarity is symmetric.
#[test]
fn test_similarity_is_symmetric() {
    let a = remove_synapse_candidate("input-1", "output-0", 0.10);
    let b = add_synapse_candidate("input-50", "hidden-5", 0.5, 0.08);

    let sim_ab = compute_structural_similarity(&a, &b);
    let sim_ba = compute_structural_similarity(&b, &a);

    assert!(
        (sim_ab - sim_ba).abs() < f32::EPSILON,
        "Similarity should be symmetric: {sim_ab} vs {sim_ba}"
    );
}

/// Test 13: Reranking preserves all candidates (no filtering).
#[test]
fn test_reranking_preserves_all_candidates() {
    let candidates = vec![
        remove_synapse_candidate("input-1", "output-0", 0.10),
        remove_synapse_candidate("input-2", "output-0", 0.09),
        remove_synapse_candidate("input-3", "output-0", 0.08),
        add_synapse_candidate("input-50", "hidden-5", 0.5, 0.07),
        change_squash_candidate("hidden-5", "RELU", 0.06),
    ];

    let config = DiversityConfig::default();
    let reranked = rerank_with_diversity(candidates, &config);

    assert_eq!(
        reranked.len(),
        5,
        "Reranking should preserve all candidates"
    );
}
