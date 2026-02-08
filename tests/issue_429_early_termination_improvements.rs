//! Tests for Issue #429: Early termination improvements for low-value candidates.
//!
//! Tests the four proposed improvements:
//! 1. Hierarchical candidate filtering — quick pre-filter before detailed analysis
//! 2. Budget-aware prioritisation — stop generating when budget exhausted
//! 3. Incremental confidence early exit — exit when confidence exceeds threshold
//! 4. Cross-module deduplication — skip candidates similar to already-generated ones

use neat_ai_discovery::analysis::early_termination::{
    BudgetTracker, CandidatePreFilter, CrossModuleDeduplicator, EarlyTerminationConfig,
    EarlyTerminationDecision, PreFilterResult, SequentialEvaluator,
};
use neat_ai_discovery::analysis::samples::HelpfulStats;

// =============================================================================
// Helper functions
// =============================================================================

fn make_stats(positive: u32, negative: u32) -> HelpfulStats {
    HelpfulStats {
        positive_count: positive,
        negative_count: negative,
        positive_improvement_sum: positive as f32 * 0.01,
        negative_improvement_sum: negative as f32 * 0.005,
        positive_activation_sum: positive as f32 * 0.5,
        negative_activation_sum: negative as f32 * 0.5,
        error_sq_sum: (positive + negative) as f32 * 0.01,
        activation_sq_sum: (positive + negative) as f32 * 0.25,
        error_activation_sum: (positive + negative) as f32 * 0.05,
        samples_evaluated: positive + negative,
        early_terminated: false,
    }
}

// =============================================================================
// 1. Hierarchical candidate pre-filtering
// =============================================================================

/// Pre-filter should classify clearly poor candidates quickly without full SPRT evaluation.
#[test]
fn test_pre_filter_rejects_clearly_poor_candidates() {
    let filter = CandidatePreFilter::default();

    // 10% positive rate — clearly poor, should be rejected quickly
    let poor_stats = make_stats(50, 450);
    let result = filter.classify(&poor_stats);
    assert_eq!(
        result,
        PreFilterResult::Reject,
        "10% positive rate should be pre-filter rejected"
    );
}

/// Pre-filter should classify clearly good candidates quickly.
#[test]
fn test_pre_filter_accepts_clearly_good_candidates() {
    let filter = CandidatePreFilter::default();

    // 90% positive rate — clearly good, should be accepted quickly
    let good_stats = make_stats(450, 50);
    let result = filter.classify(&good_stats);
    assert_eq!(
        result,
        PreFilterResult::Accept,
        "90% positive rate should be pre-filter accepted"
    );
}

/// Pre-filter should pass marginal candidates through for full SPRT analysis.
#[test]
fn test_pre_filter_passes_marginal_candidates() {
    let filter = CandidatePreFilter::default();

    // 55% positive rate — marginal, needs full evaluation
    let marginal_stats = make_stats(275, 225);
    let result = filter.classify(&marginal_stats);
    assert_eq!(
        result,
        PreFilterResult::NeedsFullEvaluation,
        "55% positive rate should need full evaluation"
    );
}

/// Pre-filter batch should separate candidates into accept/reject/needs-evaluation.
#[test]
fn test_pre_filter_batch() {
    let filter = CandidatePreFilter::default();

    let stats = vec![
        make_stats(450, 50),  // Good (90%)
        make_stats(50, 450),  // Poor (10%)
        make_stats(275, 225), // Marginal (55%)
        make_stats(400, 100), // Good (80%)
        make_stats(100, 400), // Poor (20%)
    ];

    let result = filter.filter_batch(&stats);

    // Good candidates should be accepted
    assert!(
        !result.accept_indices.is_empty(),
        "Should accept clearly good candidates"
    );
    // Poor candidates should be rejected
    assert!(
        !result.reject_indices.is_empty(),
        "Should reject clearly poor candidates"
    );
    // Total should account for all candidates
    assert_eq!(
        result.accept_indices.len() + result.reject_indices.len() + result.needs_eval_indices.len(),
        stats.len(),
        "All candidates must be classified"
    );
}

/// Pre-filter should require minimum samples before making decisions.
#[test]
fn test_pre_filter_requires_minimum_samples() {
    let filter = CandidatePreFilter::default();

    // Only 5 samples — too few for a reliable pre-filter decision
    let few_samples = make_stats(5, 0);
    let result = filter.classify(&few_samples);
    assert_eq!(
        result,
        PreFilterResult::NeedsFullEvaluation,
        "Too few samples should need full evaluation"
    );
}

/// Pre-filter should not reject candidates that are actually good.
/// This tests the safety margin — we must not filter out quality candidates.
#[test]
fn test_pre_filter_conservative_thresholds() {
    let filter = CandidatePreFilter::default();

    // 65% positive — above random but not clearly beneficial
    // Should NOT be rejected (false negatives are worse than false positives)
    let ok_stats = make_stats(325, 175);
    let result = filter.classify(&ok_stats);
    assert_ne!(
        result,
        PreFilterResult::Reject,
        "65% positive should not be rejected — must maintain coverage"
    );
}

// =============================================================================
// 2. Budget-aware prioritisation
// =============================================================================

/// Budget tracker should stop accepting new candidates when budget is exhausted.
#[test]
fn test_budget_tracker_stops_when_exhausted() {
    let mut tracker = BudgetTracker::new(5);

    // Add 5 candidates (within budget)
    for i in 0..5 {
        assert!(
            tracker.try_add(format!("candidate-{i}"), 0.1),
            "Should accept candidate within budget"
        );
    }

    // 6th candidate should be rejected
    assert!(
        !tracker.try_add("candidate-5".to_string(), 0.1),
        "Should reject candidate when budget exhausted"
    );
}

/// Budget tracker should prioritise higher-value candidates.
#[test]
fn test_budget_tracker_prioritises_high_value() {
    let mut tracker = BudgetTracker::new(3);

    // Fill budget with low-value candidates
    tracker.try_add("low-1".to_string(), 0.01);
    tracker.try_add("low-2".to_string(), 0.02);
    tracker.try_add("low-3".to_string(), 0.03);

    // Higher-value candidate should displace lowest-value
    let displaced = tracker.try_add_with_displacement("high-1".to_string(), 0.50);
    assert!(displaced, "High-value candidate should displace low-value");
    assert_eq!(tracker.count(), 3, "Budget should remain at limit");
    assert!(
        !tracker.contains("low-1"),
        "Lowest-value candidate should be displaced"
    );
    assert!(
        tracker.contains("high-1"),
        "High-value candidate should be present"
    );
}

/// Budget tracker should report remaining capacity.
#[test]
fn test_budget_tracker_remaining_capacity() {
    let mut tracker = BudgetTracker::new(10);
    assert_eq!(tracker.remaining(), 10);

    tracker.try_add("a".to_string(), 0.1);
    assert_eq!(tracker.remaining(), 9);

    for i in 0..9 {
        tracker.try_add(format!("b-{i}"), 0.1);
    }
    assert_eq!(tracker.remaining(), 0);
    assert!(tracker.is_exhausted());
}

// =============================================================================
// 3. Incremental confidence early exit
// =============================================================================

/// When confidence is already high enough, exit early even if SPRT hasn't decided.
#[test]
fn test_incremental_confidence_early_exit() {
    let mut evaluator = SequentialEvaluator::new(0.01, 0.01, 0.0);

    // Add enough samples to be statistically clear (80% positive over 500 samples)
    evaluator.add_batch(400, 100);

    // The confidence should be meaningfully above zero
    // 80% positive → deviation = 0.3 → deviation_factor = 0.6
    // 500 samples → sample_factor = 1.0
    // Combined ≈ 0.6
    let confidence = evaluator.confidence_score();
    assert!(
        confidence > 0.5,
        "80% positive over 500 samples should have meaningful confidence, got {confidence}"
    );
}

/// Low confidence should not trigger early exit.
#[test]
fn test_low_confidence_does_not_trigger_early_exit() {
    let mut evaluator = SequentialEvaluator::new(0.01, 0.01, 0.0);

    // Only 50% positive — uncertain outcome
    evaluator.add_batch(250, 250);

    let confidence = evaluator.confidence_score();
    assert!(
        confidence < 0.3,
        "50/50 split should have low confidence, got {confidence}"
    );
}

/// Confidence should increase with more concordant samples.
#[test]
fn test_confidence_increases_with_concordance() {
    let mut eval_small = SequentialEvaluator::new(0.01, 0.01, 0.0);
    let mut eval_large = SequentialEvaluator::new(0.01, 0.01, 0.0);

    // Same ratio but different sample counts
    eval_small.add_batch(80, 20);
    eval_large.add_batch(800, 200);

    let conf_small = eval_small.confidence_score();
    let conf_large = eval_large.confidence_score();

    assert!(
        conf_large >= conf_small,
        "More samples should give >= confidence: small={conf_small}, large={conf_large}"
    );
}

// =============================================================================
// 4. Cross-module deduplication
// =============================================================================

/// Cross-module deduplicator should detect similar candidates across modules.
#[test]
fn test_cross_module_dedup_detects_similar() {
    let mut dedup = CrossModuleDeduplicator::new();

    // Register a candidate from module A
    let is_new = dedup.register_candidate("source-1", "target-1", "addSynapse", 0.10);
    assert!(is_new, "First candidate should be new");

    // Same source-target pair from module B should be detected as duplicate
    let is_new = dedup.register_candidate("source-1", "target-1", "addSynapse", 0.12);
    assert!(!is_new, "Same source-target-type should be duplicate");
}

/// Different source-target pairs should not be detected as duplicates.
#[test]
fn test_cross_module_dedup_allows_different_pairs() {
    let mut dedup = CrossModuleDeduplicator::new();

    let is_new1 = dedup.register_candidate("source-1", "target-1", "addSynapse", 0.10);
    let is_new2 = dedup.register_candidate("source-2", "target-1", "addSynapse", 0.10);
    let is_new3 = dedup.register_candidate("source-1", "target-2", "addSynapse", 0.10);

    assert!(is_new1, "First candidate should be new");
    assert!(is_new2, "Different source should be new");
    assert!(is_new3, "Different target should be new");
}

/// Same source-target but different operation type should not be duplicate.
#[test]
fn test_cross_module_dedup_different_op_types() {
    let mut dedup = CrossModuleDeduplicator::new();

    let is_new1 = dedup.register_candidate("source-1", "target-1", "addSynapse", 0.10);
    let is_new2 = dedup.register_candidate("source-1", "target-1", "removeSynapse", 0.10);

    assert!(is_new1, "addSynapse should be new");
    assert!(
        is_new2,
        "removeSynapse for same pair should be new (different operation)"
    );
}

/// Deduplicator should track how many duplicates were found.
#[test]
fn test_cross_module_dedup_counts() {
    let mut dedup = CrossModuleDeduplicator::new();

    dedup.register_candidate("s1", "t1", "addSynapse", 0.10);
    dedup.register_candidate("s1", "t1", "addSynapse", 0.12); // dup
    dedup.register_candidate("s2", "t1", "addSynapse", 0.10);
    dedup.register_candidate("s2", "t1", "addSynapse", 0.15); // dup
    dedup.register_candidate("s3", "t1", "addSynapse", 0.10);

    assert_eq!(dedup.unique_count(), 3);
    assert_eq!(dedup.duplicate_count(), 2);
}

// =============================================================================
// Integration: pre-filter + SPRT should agree on clear cases
// =============================================================================

/// Pre-filter Accept should agree with SPRT Accept for clearly good candidates.
#[test]
fn test_pre_filter_and_sprt_agree_on_good_candidates() {
    let filter = CandidatePreFilter::default();
    let config = EarlyTerminationConfig::default();

    // 90% positive rate over 500 samples — clearly good
    let stats = make_stats(450, 50);

    let pre_filter_result = filter.classify(&stats);
    let mut evaluator = config.create_evaluator();
    evaluator.add_batch(stats.positive_count, stats.negative_count);
    let sprt_result = evaluator.should_stop();

    // Both should accept
    assert_eq!(pre_filter_result, PreFilterResult::Accept);
    assert_eq!(sprt_result, EarlyTerminationDecision::Accept);
}

/// Pre-filter Reject should agree with SPRT Reject for clearly poor candidates.
#[test]
fn test_pre_filter_and_sprt_agree_on_poor_candidates() {
    let filter = CandidatePreFilter::default();
    let config = EarlyTerminationConfig::default();

    // 10% positive rate over 500 samples — clearly poor
    let stats = make_stats(50, 450);

    let pre_filter_result = filter.classify(&stats);
    let mut evaluator = config.create_evaluator();
    evaluator.add_batch(stats.positive_count, stats.negative_count);
    let sprt_result = evaluator.should_stop();

    // Both should reject
    assert_eq!(pre_filter_result, PreFilterResult::Reject);
    assert_eq!(sprt_result, EarlyTerminationDecision::Reject);
}
