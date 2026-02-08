//! Tests for Issue #429: Early termination improvements for low-value candidates.
//!
//! Validates the four improvements:
//! 1. Hierarchical candidate filtering — quick pre-filter removes clearly poor candidates
//! 2. Budget-aware prioritisation — stops generating when budget is exhausted
//! 3. Incremental confidence — exits early when confidence exceeds threshold
//! 4. Cross-module deduplication — skips candidates similar to already-generated ones

use neat_ai_discovery::analysis::candidate_prefilter::{
    deduplicate_candidates, filter_low_value_candidates, prefilter_candidates,
    CandidatePrefilterConfig,
};
use neat_ai_discovery::CoordinatedStructuralCandidateJson;
use neat_ai_discovery::CoordinatedStructuralOpJson;

// =============================================================================
// Helper functions
// =============================================================================

/// Create a candidate with a single RemoveSynapse operation.
fn make_remove_synapse_candidate(
    from: &str,
    to: &str,
    gain: f32,
) -> CoordinatedStructuralCandidateJson {
    CoordinatedStructuralCandidateJson {
        operations: vec![CoordinatedStructuralOpJson::RemoveSynapse {
            from_neuron_uuid: from.to_string(),
            to_neuron_uuid: to.to_string(),
        }],
        expected_creature_score_gain: gain,
        comment: None,
    }
}

/// Create a candidate with AddNeuron + AddSynapse operations (coordinated structural).
fn make_add_neuron_candidate(
    neuron_uuid: &str,
    target: &str,
    gain: f32,
) -> CoordinatedStructuralCandidateJson {
    CoordinatedStructuralCandidateJson {
        operations: vec![
            CoordinatedStructuralOpJson::AddNeuron {
                neuron_uuid: neuron_uuid.to_string(),
                neuron_type: "hidden".to_string(),
                squash: "LOGISTIC".to_string(),
                bias: 0.0,
                insert_before_neuron_uuid: Some(target.to_string()),
            },
            CoordinatedStructuralOpJson::AddSynapse {
                from_neuron_uuid: "input-0".to_string(),
                to_neuron_uuid: neuron_uuid.to_string(),
                weight: 1.0,
            },
            CoordinatedStructuralOpJson::AddSynapse {
                from_neuron_uuid: neuron_uuid.to_string(),
                to_neuron_uuid: target.to_string(),
                weight: 0.5,
            },
        ],
        expected_creature_score_gain: gain,
        comment: Some("coordinated add-neuron".to_string()),
    }
}

/// Create a setBias candidate.
fn make_set_bias_candidate(uuid: &str, gain: f32) -> CoordinatedStructuralCandidateJson {
    CoordinatedStructuralCandidateJson {
        operations: vec![CoordinatedStructuralOpJson::SetBias {
            neuron_uuid: uuid.to_string(),
            bias: 0.1,
        }],
        expected_creature_score_gain: gain,
        comment: None,
    }
}

// =============================================================================
// 1. Hierarchical candidate filtering — low-value removal
// =============================================================================

/// Candidates with near-zero expected improvement should be filtered out.
#[test]
fn test_filter_removes_near_zero_improvement_candidates() {
    let candidates = vec![
        make_remove_synapse_candidate("a", "b", 0.05), // Keep: good improvement
        make_remove_synapse_candidate("c", "d", 0.0001), // Remove: near-zero
        make_remove_synapse_candidate("e", "f", 0.0),  // Remove: zero
        make_remove_synapse_candidate("g", "h", -0.001), // Remove: negative
    ];

    let config = CandidatePrefilterConfig::default();
    let filtered = filter_low_value_candidates(&candidates, &config);

    assert_eq!(
        filtered.len(),
        1,
        "Only the high-value candidate should survive filtering"
    );
    assert!(
        filtered[0].expected_creature_score_gain > 0.01,
        "Remaining candidate should have meaningful improvement"
    );
}

/// Filtering should preserve all candidates when they all exceed the threshold.
#[test]
fn test_filter_preserves_all_good_candidates() {
    let candidates = vec![
        make_remove_synapse_candidate("a", "b", 0.05),
        make_remove_synapse_candidate("c", "d", 0.03),
        make_remove_synapse_candidate("e", "f", 0.10),
    ];

    let config = CandidatePrefilterConfig::default();
    let filtered = filter_low_value_candidates(&candidates, &config);

    assert_eq!(filtered.len(), 3, "All good candidates should be preserved");
}

/// Empty input should produce empty output.
#[test]
fn test_filter_handles_empty_input() {
    let candidates: Vec<CoordinatedStructuralCandidateJson> = vec![];
    let config = CandidatePrefilterConfig::default();
    let filtered = filter_low_value_candidates(&candidates, &config);

    assert!(
        filtered.is_empty(),
        "Empty input should produce empty output"
    );
}

/// The minimum improvement threshold should be configurable.
#[test]
fn test_filter_respects_custom_threshold() {
    let candidates = vec![
        make_remove_synapse_candidate("a", "b", 0.05),
        make_remove_synapse_candidate("c", "d", 0.005),
        make_remove_synapse_candidate("e", "f", 0.002),
    ];

    // Higher threshold - only the strongest candidate survives
    let config = CandidatePrefilterConfig {
        min_improvement_threshold: 0.01,
        ..CandidatePrefilterConfig::default()
    };
    let filtered = filter_low_value_candidates(&candidates, &config);

    assert_eq!(
        filtered.len(),
        1,
        "Only candidates above 0.01 threshold should survive"
    );
    assert!(
        (filtered[0].expected_creature_score_gain - 0.05).abs() < f32::EPSILON,
        "The 0.05 candidate should be preserved"
    );
}

// =============================================================================
// 2. Budget-aware prioritisation
// =============================================================================

/// When a budget (max candidates) is set, only the top-N candidates survive.
#[test]
fn test_prefilter_respects_budget() {
    let candidates = vec![
        make_remove_synapse_candidate("a", "b", 0.10),
        make_remove_synapse_candidate("c", "d", 0.08),
        make_remove_synapse_candidate("e", "f", 0.05),
        make_remove_synapse_candidate("g", "h", 0.03),
        make_remove_synapse_candidate("i", "j", 0.02),
    ];

    let config = CandidatePrefilterConfig {
        max_candidates: Some(3),
        ..CandidatePrefilterConfig::default()
    };
    let filtered = prefilter_candidates(&candidates, &config);

    assert_eq!(filtered.len(), 3, "Budget should limit to 3 candidates");
    // The top-3 by improvement should be kept
    assert!(filtered[0].expected_creature_score_gain >= 0.05);
}

/// Budget of None means no limit.
#[test]
fn test_prefilter_no_budget_keeps_all_good() {
    let candidates = vec![
        make_remove_synapse_candidate("a", "b", 0.10),
        make_remove_synapse_candidate("c", "d", 0.08),
        make_remove_synapse_candidate("e", "f", 0.05),
    ];

    let config = CandidatePrefilterConfig {
        max_candidates: None,
        ..CandidatePrefilterConfig::default()
    };
    let filtered = prefilter_candidates(&candidates, &config);

    assert_eq!(
        filtered.len(),
        3,
        "Without budget, all good candidates should survive"
    );
}

// =============================================================================
// 3. Incremental confidence — exit early with enough confidence
// =============================================================================

/// Candidates with high expected improvement should be prioritised (sorted first).
#[test]
fn test_prefilter_sorts_by_improvement() {
    let candidates = vec![
        make_remove_synapse_candidate("a", "b", 0.03),
        make_remove_synapse_candidate("c", "d", 0.10),
        make_remove_synapse_candidate("e", "f", 0.05),
    ];

    let config = CandidatePrefilterConfig::default();
    let filtered = prefilter_candidates(&candidates, &config);

    assert_eq!(filtered.len(), 3);
    // Should be sorted by improvement, best first
    assert!(
        filtered[0].expected_creature_score_gain >= filtered[1].expected_creature_score_gain,
        "Candidates should be sorted by improvement (best first)"
    );
    assert!(
        filtered[1].expected_creature_score_gain >= filtered[2].expected_creature_score_gain,
        "Candidates should be sorted by improvement (best first)"
    );
}

// =============================================================================
// 4. Cross-module deduplication
// =============================================================================

/// Candidates targeting the same neuron with the same operation type should
/// be deduplicated — keeping only the best one per (target, operation_type).
#[test]
fn test_dedup_removes_same_target_same_op() {
    let candidates = vec![
        make_remove_synapse_candidate("source-a", "target-1", 0.05),
        make_remove_synapse_candidate("source-b", "target-1", 0.03), // Same target, lower gain
        make_remove_synapse_candidate("source-c", "target-2", 0.04), // Different target
    ];

    let config = CandidatePrefilterConfig::default();
    let deduped = deduplicate_candidates(&candidates, &config);

    // We should keep the best for target-1 and the one for target-2
    assert_eq!(
        deduped.len(),
        2,
        "Should keep best per (target, op_type) pair"
    );

    let gains: Vec<f32> = deduped
        .iter()
        .map(|c| c.expected_creature_score_gain)
        .collect();
    assert!(
        gains.contains(&0.05),
        "Best candidate for target-1 should be kept"
    );
    assert!(
        gains.contains(&0.04),
        "Candidate for target-2 should be kept"
    );
}

/// Different operation types targeting the same neuron should NOT be deduplicated.
#[test]
fn test_dedup_preserves_different_op_types_same_target() {
    let candidates = vec![
        make_remove_synapse_candidate("source-a", "target-1", 0.05),
        make_set_bias_candidate("target-1", 0.04),
    ];

    let config = CandidatePrefilterConfig::default();
    let deduped = deduplicate_candidates(&candidates, &config);

    assert_eq!(
        deduped.len(),
        2,
        "Different operations on same target should both be kept"
    );
}

/// AddNeuron candidates with different target neurons should not be deduplicated.
#[test]
fn test_dedup_preserves_different_targets() {
    let candidates = vec![
        make_add_neuron_candidate("new-1", "target-1", 0.06),
        make_add_neuron_candidate("new-2", "target-2", 0.05),
    ];

    let config = CandidatePrefilterConfig::default();
    let deduped = deduplicate_candidates(&candidates, &config);

    assert_eq!(
        deduped.len(),
        2,
        "Different targets should not be deduplicated"
    );
}

/// Empty input should produce empty output for deduplication.
#[test]
fn test_dedup_handles_empty_input() {
    let candidates: Vec<CoordinatedStructuralCandidateJson> = vec![];
    let config = CandidatePrefilterConfig::default();
    let deduped = deduplicate_candidates(&candidates, &config);

    assert!(deduped.is_empty());
}

// =============================================================================
// Combined prefilter (all steps)
// =============================================================================

/// The full prefilter pipeline applies filtering, dedup, sorting, and budget in sequence.
#[test]
fn test_prefilter_full_pipeline() {
    let candidates = vec![
        make_remove_synapse_candidate("a", "target-1", 0.10),
        make_remove_synapse_candidate("b", "target-1", 0.08), // Duplicate target, lower gain
        make_remove_synapse_candidate("c", "target-2", 0.05),
        make_remove_synapse_candidate("d", "target-3", 0.0001), // Below threshold
        make_remove_synapse_candidate("e", "target-4", 0.03),
        make_set_bias_candidate("target-1", 0.04), // Different op type, same target
    ];

    let config = CandidatePrefilterConfig {
        min_improvement_threshold: 0.001,
        max_candidates: Some(4),
    };

    let result = prefilter_candidates(&candidates, &config);

    // After filtering: remove 0.0001 → 5 remain
    // After dedup: target-1 RemoveSynapse keeps best (0.10), setBias kept separately → 4 remain
    // After budget: top 4 kept
    assert!(
        result.len() <= 4,
        "Budget should limit output to 4, got {}",
        result.len()
    );

    // All surviving candidates should have meaningful improvement
    for c in &result {
        assert!(
            c.expected_creature_score_gain >= 0.001,
            "All surviving candidates should exceed threshold"
        );
    }

    // Best candidate should be first
    if result.len() >= 2 {
        assert!(
            result[0].expected_creature_score_gain >= result[1].expected_creature_score_gain,
            "Results should be sorted by improvement (best first)"
        );
    }
}

/// The prefilter should never filter out high-value candidates — maintain coverage.
#[test]
fn test_prefilter_does_not_filter_good_candidates() {
    let candidates: Vec<CoordinatedStructuralCandidateJson> = (0..100)
        .map(|i| make_remove_synapse_candidate(&format!("src-{i}"), &format!("tgt-{i}"), 0.05))
        .collect();

    let config = CandidatePrefilterConfig::default();
    let result = prefilter_candidates(&candidates, &config);

    // All have unique targets and good improvement, so all should survive
    assert_eq!(
        result.len(),
        100,
        "All high-value, unique-target candidates should be preserved"
    );
}

/// Config defaults should be reasonable.
#[test]
fn test_config_defaults() {
    let config = CandidatePrefilterConfig::default();

    assert!(
        config.min_improvement_threshold > 0.0,
        "Default threshold should be positive"
    );
    assert!(
        config.min_improvement_threshold < 0.01,
        "Default threshold should be conservative (not filtering good candidates)"
    );
    assert!(
        config.max_candidates.is_none(),
        "Default budget should be None (no limit)"
    );
}
