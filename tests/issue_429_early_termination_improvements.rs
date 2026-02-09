//! Tests for early termination improvements for low-value candidates (Issue #429).
//!
//! Tests four new capabilities:
//! 1. Hierarchical candidate filtering — quick pre-filter before detailed SPRT analysis
//! 2. Budget-aware prioritisation — stop evaluating when candidate budget exhausted
//! 3. Incremental confidence — exit early when confidence already exceeds threshold
//! 4. Cross-module deduplication — skip candidates similar to already-generated ones

use neat_ai_discovery::analysis::early_termination::{
    EarlyTerminationConfig, EarlyTerminationDecision,
};
use neat_ai_discovery::analysis::samples::HelpfulStats;

// =============================================================================
// 1. Hierarchical Candidate Filtering
// =============================================================================

/// Test that the quick pre-filter correctly identifies clearly poor candidates
/// based on improvement ratio alone, without needing full SPRT evaluation.
#[test]
fn test_prefilter_rejects_clearly_poor_candidates() {
    use neat_ai_discovery::analysis::early_termination::prefilter_candidates;

    // Candidate with only 5% positive — clearly poor
    let poor = HelpfulStats {
        positive_count: 5,
        negative_count: 95,
        positive_improvement_sum: 0.01,
        negative_improvement_sum: -0.5,
        positive_activation_sum: 0.0,
        negative_activation_sum: 0.0,
        error_sq_sum: 0.0,
        activation_sq_sum: 0.0,
        error_activation_sum: 0.0,
        samples_evaluated: 100,
        early_terminated: false,
    };

    // Candidate with 80% positive — clearly good
    let good = HelpfulStats {
        positive_count: 80,
        negative_count: 20,
        positive_improvement_sum: 0.5,
        negative_improvement_sum: -0.1,
        positive_activation_sum: 0.0,
        negative_activation_sum: 0.0,
        error_sq_sum: 0.0,
        activation_sq_sum: 0.0,
        error_activation_sum: 0.0,
        samples_evaluated: 100,
        early_terminated: false,
    };

    // Candidate with 50% positive — marginal, should pass to SPRT
    let marginal = HelpfulStats {
        positive_count: 50,
        negative_count: 50,
        positive_improvement_sum: 0.2,
        negative_improvement_sum: -0.2,
        positive_activation_sum: 0.0,
        negative_activation_sum: 0.0,
        error_sq_sum: 0.0,
        activation_sq_sum: 0.0,
        error_activation_sum: 0.0,
        samples_evaluated: 100,
        early_terminated: false,
    };

    let stats = vec![poor, good, marginal];
    let result = prefilter_candidates(&stats);

    // Poor candidate should be rejected by prefilter
    assert!(
        result.reject_indices.contains(&0),
        "Clearly poor candidate (5% positive) should be pre-filtered as rejected"
    );
    // Good candidate should be accepted by prefilter
    assert!(
        result.accept_indices.contains(&1),
        "Clearly good candidate (80% positive) should be pre-filtered as accepted"
    );
    // Marginal candidate should continue to full SPRT
    assert!(
        result.continue_indices.contains(&2),
        "Marginal candidate (50% positive) should continue to SPRT"
    );
}

/// Test that prefilter does not reject candidates with insufficient samples.
#[test]
fn test_prefilter_requires_minimum_samples() {
    use neat_ai_discovery::analysis::early_termination::prefilter_candidates;

    // Only 5 samples — not enough for prefilter decision
    let too_few = HelpfulStats {
        positive_count: 0,
        negative_count: 5,
        positive_improvement_sum: 0.0,
        negative_improvement_sum: -0.1,
        positive_activation_sum: 0.0,
        negative_activation_sum: 0.0,
        error_sq_sum: 0.0,
        activation_sq_sum: 0.0,
        error_activation_sum: 0.0,
        samples_evaluated: 5,
        early_terminated: false,
    };

    let result = prefilter_candidates(&[too_few]);

    // Too few samples — must continue, not reject
    assert!(
        result.continue_indices.contains(&0),
        "Candidate with fewer than minimum samples must continue, not be pre-filtered"
    );
    assert!(result.reject_indices.is_empty());
    assert!(result.accept_indices.is_empty());
}

/// Test that prefilter correctly handles an empty input.
#[test]
fn test_prefilter_empty_input() {
    use neat_ai_discovery::analysis::early_termination::prefilter_candidates;

    let result = prefilter_candidates(&[]);
    assert!(result.accept_indices.is_empty());
    assert!(result.reject_indices.is_empty());
    assert!(result.continue_indices.is_empty());
}

// =============================================================================
// 2. Budget-Aware Prioritisation
// =============================================================================

/// Test that budget-aware evaluation stops processing when budget is exhausted.
#[test]
fn test_budget_aware_stops_at_budget() {
    use neat_ai_discovery::analysis::early_termination::budget_aware_evaluate;

    // 10 candidates, budget for only 5
    let stats: Vec<HelpfulStats> = (0..10)
        .map(|i| HelpfulStats {
            positive_count: 50 + i,
            negative_count: 50 - i,
            positive_improvement_sum: 0.1,
            negative_improvement_sum: -0.1,
            positive_activation_sum: 0.0,
            negative_activation_sum: 0.0,
            error_sq_sum: 0.0,
            activation_sq_sum: 0.0,
            error_activation_sum: 0.0,
            samples_evaluated: 100,
            early_terminated: false,
        })
        .collect();

    let config = EarlyTerminationConfig::default();
    let result = budget_aware_evaluate(&stats, &config, 5);

    // Total decisions should cover all candidates
    let total =
        result.accept_indices.len() + result.reject_indices.len() + result.continue_indices.len();
    assert_eq!(total, 10, "All candidates must be accounted for");

    // Candidates beyond budget should be rejected (budget exhausted)
    let evaluated_count = result.accept_indices.len() + result.continue_indices.len();
    assert!(
        evaluated_count <= 5,
        "Budget of 5 should limit evaluated candidates, but got {evaluated_count}"
    );
}

/// Test that budget of zero rejects all candidates.
#[test]
fn test_budget_zero_rejects_all() {
    use neat_ai_discovery::analysis::early_termination::budget_aware_evaluate;

    let stats = vec![HelpfulStats {
        positive_count: 80,
        negative_count: 20,
        positive_improvement_sum: 0.5,
        negative_improvement_sum: -0.1,
        positive_activation_sum: 0.0,
        negative_activation_sum: 0.0,
        error_sq_sum: 0.0,
        activation_sq_sum: 0.0,
        error_activation_sum: 0.0,
        samples_evaluated: 100,
        early_terminated: false,
    }];

    let config = EarlyTerminationConfig::default();
    let result = budget_aware_evaluate(&stats, &config, 0);

    assert_eq!(result.reject_indices.len(), 1);
    assert!(result.accept_indices.is_empty());
    assert!(result.continue_indices.is_empty());
}

/// Test that budget larger than candidates evaluates all.
#[test]
fn test_budget_larger_than_candidates() {
    use neat_ai_discovery::analysis::early_termination::budget_aware_evaluate;

    let stats = vec![HelpfulStats {
        positive_count: 80,
        negative_count: 20,
        positive_improvement_sum: 0.5,
        negative_improvement_sum: -0.1,
        positive_activation_sum: 0.0,
        negative_activation_sum: 0.0,
        error_sq_sum: 0.0,
        activation_sq_sum: 0.0,
        error_activation_sum: 0.0,
        samples_evaluated: 100,
        early_terminated: false,
    }];

    let config = EarlyTerminationConfig::default();
    let result = budget_aware_evaluate(&stats, &config, 100);

    // All candidates should be evaluated (not budget-rejected)
    let total =
        result.accept_indices.len() + result.reject_indices.len() + result.continue_indices.len();
    assert_eq!(total, 1);
}

// =============================================================================
// 3. Incremental Confidence
// =============================================================================

/// Test that incremental confidence exits early when confidence is very high.
#[test]
fn test_incremental_confidence_exits_early_high_confidence() {
    use neat_ai_discovery::analysis::early_termination::evaluate_with_confidence;

    // Strong positive candidate: 90% positive with many samples
    let strong = HelpfulStats {
        positive_count: 900,
        negative_count: 100,
        positive_improvement_sum: 5.0,
        negative_improvement_sum: -0.5,
        positive_activation_sum: 0.0,
        negative_activation_sum: 0.0,
        error_sq_sum: 0.0,
        activation_sq_sum: 0.0,
        error_activation_sum: 0.0,
        samples_evaluated: 1000,
        early_terminated: false,
    };

    let config = EarlyTerminationConfig::default();
    let (decision, confidence) = evaluate_with_confidence(&strong, &config);

    assert!(
        matches!(decision, EarlyTerminationDecision::Accept),
        "90% positive with 1000 samples should accept, got {decision:?}"
    );
    assert!(
        confidence > 0.7,
        "Strong candidate should have high confidence, got {confidence}"
    );
}

/// Test that low-confidence candidates continue evaluation.
#[test]
fn test_incremental_confidence_continues_for_marginal() {
    use neat_ai_discovery::analysis::early_termination::evaluate_with_confidence;

    // Marginal candidate: 52% positive, few samples
    let marginal = HelpfulStats {
        positive_count: 26,
        negative_count: 24,
        positive_improvement_sum: 0.05,
        negative_improvement_sum: -0.04,
        positive_activation_sum: 0.0,
        negative_activation_sum: 0.0,
        error_sq_sum: 0.0,
        activation_sq_sum: 0.0,
        error_activation_sum: 0.0,
        samples_evaluated: 50,
        early_terminated: false,
    };

    let config = EarlyTerminationConfig::default();
    let (decision, confidence) = evaluate_with_confidence(&marginal, &config);

    // Marginal candidates should either continue or have low confidence
    if matches!(decision, EarlyTerminationDecision::Continue) {
        // Expected — not enough signal
    } else {
        // If a decision was reached, confidence should be lower than for a strong candidate
        assert!(
            confidence < 0.95,
            "Marginal candidate confidence should be moderate, got {confidence}"
        );
    }
}

// =============================================================================
// 4. Cross-Module Deduplication
// =============================================================================

/// Test that duplicate candidates are detected and filtered.
#[test]
fn test_cross_module_dedup_filters_duplicates() {
    use neat_ai_discovery::analysis::early_termination::CandidateSignature;
    use neat_ai_discovery::analysis::early_termination::CrossModuleDeduplicator;

    let mut dedup = CrossModuleDeduplicator::new();

    // First candidate — should be accepted (novel)
    let sig1 = CandidateSignature {
        source_uuid: "input-1".to_string(),
        target_uuid: "output-0".to_string(),
        candidate_type: "addSynapse".to_string(),
    };
    assert!(dedup.is_novel(&sig1), "First candidate should be novel");
    dedup.register(&sig1);

    // Same candidate again — should be duplicate
    let sig2 = CandidateSignature {
        source_uuid: "input-1".to_string(),
        target_uuid: "output-0".to_string(),
        candidate_type: "addSynapse".to_string(),
    };
    assert!(
        !dedup.is_novel(&sig2),
        "Identical candidate should be detected as duplicate"
    );

    // Different candidate — should be novel
    let sig3 = CandidateSignature {
        source_uuid: "input-2".to_string(),
        target_uuid: "output-0".to_string(),
        candidate_type: "addSynapse".to_string(),
    };
    assert!(dedup.is_novel(&sig3), "Different source should be novel");
}

/// Test that deduplicator handles different candidate types.
#[test]
fn test_cross_module_dedup_different_types() {
    use neat_ai_discovery::analysis::early_termination::CandidateSignature;
    use neat_ai_discovery::analysis::early_termination::CrossModuleDeduplicator;

    let mut dedup = CrossModuleDeduplicator::new();

    // addSynapse from input-1 to output-0
    let sig_add = CandidateSignature {
        source_uuid: "input-1".to_string(),
        target_uuid: "output-0".to_string(),
        candidate_type: "addSynapse".to_string(),
    };
    dedup.register(&sig_add);

    // removeSynapse from input-1 to output-0 — different type, should be novel
    let sig_remove = CandidateSignature {
        source_uuid: "input-1".to_string(),
        target_uuid: "output-0".to_string(),
        candidate_type: "removeSynapse".to_string(),
    };
    assert!(
        dedup.is_novel(&sig_remove),
        "Different candidate type should be novel even with same source/target"
    );
}

/// Test deduplicator count tracking.
#[test]
fn test_cross_module_dedup_counts() {
    use neat_ai_discovery::analysis::early_termination::CandidateSignature;
    use neat_ai_discovery::analysis::early_termination::CrossModuleDeduplicator;

    let mut dedup = CrossModuleDeduplicator::new();

    assert_eq!(dedup.registered_count(), 0);

    let sig = CandidateSignature {
        source_uuid: "input-1".to_string(),
        target_uuid: "output-0".to_string(),
        candidate_type: "addSynapse".to_string(),
    };
    dedup.register(&sig);
    assert_eq!(dedup.registered_count(), 1);

    // Registering same candidate again should not increase count
    dedup.register(&sig);
    assert_eq!(dedup.registered_count(), 1);

    let sig2 = CandidateSignature {
        source_uuid: "input-2".to_string(),
        target_uuid: "output-0".to_string(),
        candidate_type: "addSynapse".to_string(),
    };
    dedup.register(&sig2);
    assert_eq!(dedup.registered_count(), 2);
}

// =============================================================================
// Integration: Combined Pipeline
// =============================================================================

/// Test the combined pipeline: prefilter → budget → SPRT → confidence.
#[test]
fn test_combined_early_termination_pipeline() {
    use neat_ai_discovery::analysis::early_termination::{
        budget_aware_evaluate, prefilter_candidates,
    };

    // Mix of clearly poor, clearly good, and marginal candidates
    let candidates = vec![
        // Index 0: clearly poor (5% positive)
        HelpfulStats {
            positive_count: 5,
            negative_count: 95,
            positive_improvement_sum: 0.01,
            negative_improvement_sum: -0.5,
            positive_activation_sum: 0.0,
            negative_activation_sum: 0.0,
            error_sq_sum: 0.0,
            activation_sq_sum: 0.0,
            error_activation_sum: 0.0,
            samples_evaluated: 100,
            early_terminated: false,
        },
        // Index 1: clearly good (90% positive)
        HelpfulStats {
            positive_count: 90,
            negative_count: 10,
            positive_improvement_sum: 0.8,
            negative_improvement_sum: -0.05,
            positive_activation_sum: 0.0,
            negative_activation_sum: 0.0,
            error_sq_sum: 0.0,
            activation_sq_sum: 0.0,
            error_activation_sum: 0.0,
            samples_evaluated: 100,
            early_terminated: false,
        },
        // Index 2: marginal (55% positive)
        HelpfulStats {
            positive_count: 55,
            negative_count: 45,
            positive_improvement_sum: 0.2,
            negative_improvement_sum: -0.15,
            positive_activation_sum: 0.0,
            negative_activation_sum: 0.0,
            error_sq_sum: 0.0,
            activation_sq_sum: 0.0,
            error_activation_sum: 0.0,
            samples_evaluated: 100,
            early_terminated: false,
        },
    ];

    // Step 1: Prefilter
    let prefilter_result = prefilter_candidates(&candidates);

    // Clearly poor should be rejected early
    assert!(
        prefilter_result.reject_indices.contains(&0),
        "Prefilter should reject the 5% positive candidate"
    );

    // Step 2: Budget-aware evaluation for remaining candidates
    let config = EarlyTerminationConfig::default();
    let budget_result = budget_aware_evaluate(&candidates, &config, 10);

    // All candidates accounted for
    let total = budget_result.accept_indices.len()
        + budget_result.reject_indices.len()
        + budget_result.continue_indices.len();
    assert_eq!(total, candidates.len());
}
