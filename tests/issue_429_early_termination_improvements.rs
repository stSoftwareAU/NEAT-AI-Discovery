//! Tests for Issue #429: Early termination improvements for low-value candidates.
//!
//! Tests the four improvements to candidate generation:
//! 1. Hierarchical candidate filtering — quick pre-filter before detailed analysis
//! 2. Budget-aware prioritisation — stop generating when budget exhausted
//! 3. Incremental confidence — exit early when confidence exceeds threshold
//! 4. Cross-module deduplication — skip candidates similar to already-generated ones

use neat_ai_discovery::analysis::early_termination::{
    BudgetTracker, CandidatePreFilter, CandidatePreFilterConfig, CrossModuleDeduplicator,
    IncrementalConfidenceChecker,
};

// ============================================================================
// 1. Hierarchical Candidate Pre-Filtering
// ============================================================================

#[test]
fn test_pre_filter_rejects_low_variance_source() {
    let config = CandidatePreFilterConfig::default();
    let pre_filter = CandidatePreFilter::new(config);

    // Source with near-zero variance should be filtered out
    let activations: Vec<f32> = vec![0.5; 100]; // constant activation
    assert!(
        !pre_filter.passes_variance_check(&activations),
        "Constant source should be filtered out by variance check"
    );
}

#[test]
fn test_pre_filter_accepts_high_variance_source() {
    let config = CandidatePreFilterConfig::default();
    let pre_filter = CandidatePreFilter::new(config);

    // Source with good variance should pass
    let activations: Vec<f32> = (0..100).map(|i| (i as f32 / 100.0) * 2.0 - 1.0).collect();
    assert!(
        pre_filter.passes_variance_check(&activations),
        "High-variance source should pass variance check"
    );
}

#[test]
fn test_pre_filter_rejects_insufficient_samples() {
    let config = CandidatePreFilterConfig::default();
    let pre_filter = CandidatePreFilter::new(config);

    // Too few samples should fail
    let activations: Vec<f32> = vec![0.1, 0.9, 0.5];
    assert!(
        !pre_filter.passes_sample_count_check(activations.len()),
        "Insufficient samples should be filtered"
    );
}

#[test]
fn test_pre_filter_accepts_sufficient_samples() {
    let config = CandidatePreFilterConfig::default();
    let pre_filter = CandidatePreFilter::new(config);

    assert!(
        pre_filter.passes_sample_count_check(100),
        "Sufficient samples should pass"
    );
}

#[test]
fn test_pre_filter_rejects_low_error_correlation() {
    let config = CandidatePreFilterConfig::default();
    let pre_filter = CandidatePreFilter::new(config);

    // Activations and errors with zero correlation
    let activations: Vec<f32> = (0..100).map(|i| (i as f32 / 100.0) * 2.0 - 1.0).collect();
    let errors: Vec<f32> = vec![0.5; 100]; // constant error — no correlation
    assert!(
        !pre_filter.passes_error_correlation_check(&activations, &errors),
        "Zero error correlation should be filtered"
    );
}

#[test]
fn test_pre_filter_accepts_high_error_correlation() {
    let config = CandidatePreFilterConfig::default();
    let pre_filter = CandidatePreFilter::new(config);

    // Activations and errors with strong correlation (errors mirror activations)
    let activations: Vec<f32> = (0..100).map(|i| (i as f32 / 100.0) * 2.0 - 1.0).collect();
    let errors: Vec<f32> = activations.iter().map(|a| a * 0.8 + 0.1).collect();
    assert!(
        pre_filter.passes_error_correlation_check(&activations, &errors),
        "Strong error correlation should pass"
    );
}

#[test]
fn test_pre_filter_combined_check() {
    let config = CandidatePreFilterConfig::default();
    let pre_filter = CandidatePreFilter::new(config);

    // Good candidate: enough samples, good variance, correlated errors
    let activations: Vec<f32> = (0..100).map(|i| (i as f32 / 100.0) * 2.0 - 1.0).collect();
    let errors: Vec<f32> = activations.iter().map(|a| a * 0.5).collect();
    assert!(
        pre_filter.should_analyse(&activations, &errors),
        "Good candidate should pass combined check"
    );

    // Bad candidate: constant source
    let bad_activations: Vec<f32> = vec![0.5; 100];
    let bad_errors: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
    assert!(
        !pre_filter.should_analyse(&bad_activations, &bad_errors),
        "Constant source should fail combined check"
    );
}

#[test]
fn test_pre_filter_tracks_statistics() {
    let config = CandidatePreFilterConfig::default();
    let mut pre_filter = CandidatePreFilter::new(config);

    pre_filter.record_check(true);
    pre_filter.record_check(false);
    pre_filter.record_check(true);

    let stats = pre_filter.statistics();
    assert_eq!(stats.total_checked, 3);
    assert_eq!(stats.total_passed, 2);
    assert_eq!(stats.total_filtered, 1);
}

// ============================================================================
// 2. Budget-Aware Prioritisation
// ============================================================================

#[test]
fn test_budget_tracker_basic() {
    let mut tracker = BudgetTracker::new(10);

    assert!(tracker.has_budget(), "Should have budget initially");
    assert_eq!(tracker.remaining(), 10);

    tracker.consume(3);
    assert_eq!(tracker.remaining(), 7);
    assert!(tracker.has_budget());

    tracker.consume(7);
    assert_eq!(tracker.remaining(), 0);
    assert!(!tracker.has_budget(), "Budget should be exhausted");
}

#[test]
fn test_budget_tracker_overflow_protection() {
    let mut tracker = BudgetTracker::new(5);

    tracker.consume(10); // consume more than available
    assert_eq!(tracker.remaining(), 0, "Should not go negative");
    assert!(!tracker.has_budget());
}

#[test]
fn test_budget_tracker_zero_budget() {
    let tracker = BudgetTracker::new(0);
    assert!(!tracker.has_budget(), "Zero budget should have no budget");
}

#[test]
fn test_budget_tracker_should_skip_low_priority() {
    let mut tracker = BudgetTracker::new(100);

    // When budget is mostly unused, don't skip anything
    tracker.consume(10);
    assert!(
        !tracker.should_skip_low_priority(0.5),
        "Should not skip when budget is plentiful"
    );

    // When budget is 80%+ consumed, skip low-priority candidates
    tracker.consume(70);
    assert!(
        tracker.should_skip_low_priority(0.001),
        "Should skip very low priority when budget is mostly consumed"
    );

    // But don't skip high-priority even when budget is tight
    assert!(
        !tracker.should_skip_low_priority(0.5),
        "Should not skip high priority even with tight budget"
    );
}

// ============================================================================
// 3. Incremental Confidence Checking
// ============================================================================

#[test]
fn test_incremental_confidence_basic() {
    let mut checker = IncrementalConfidenceChecker::new(0.8);

    // Not enough candidates yet
    assert!(
        !checker.should_stop_generating(),
        "Should not stop with no candidates"
    );

    // Add some low-confidence candidates
    checker.add_candidate(0.3);
    checker.add_candidate(0.4);
    assert!(
        !checker.should_stop_generating(),
        "Should not stop with low-confidence candidates"
    );
}

#[test]
fn test_incremental_confidence_stops_when_threshold_met() {
    let mut checker = IncrementalConfidenceChecker::new(0.7);

    // Add several high-confidence candidates
    for _ in 0..10 {
        checker.add_candidate(0.9);
    }

    assert!(
        checker.should_stop_generating(),
        "Should stop generating when enough high-confidence candidates exist"
    );
}

#[test]
fn test_incremental_confidence_needs_minimum_candidates() {
    let mut checker = IncrementalConfidenceChecker::new(0.5);

    // Even with high confidence, need minimum candidates before stopping
    checker.add_candidate(0.99);
    assert!(
        !checker.should_stop_generating(),
        "Should not stop with only one candidate even if high confidence"
    );
}

#[test]
fn test_incremental_confidence_returns_best_confidence() {
    let mut checker = IncrementalConfidenceChecker::new(0.8);
    checker.add_candidate(0.3);
    checker.add_candidate(0.9);
    checker.add_candidate(0.5);

    assert!(
        (checker.best_confidence() - 0.9).abs() < f32::EPSILON,
        "Should return the best confidence seen"
    );
}

// ============================================================================
// 4. Cross-Module Deduplication
// ============================================================================

#[test]
fn test_deduplicator_allows_unique_candidates() {
    let mut dedup = CrossModuleDeduplicator::new();

    // First candidate for a target should always be allowed
    let is_dup = dedup.is_duplicate("source-1", "target-1", 0.1);
    assert!(
        !is_dup,
        "First candidate for target should not be duplicate"
    );
}

#[test]
fn test_deduplicator_detects_exact_duplicate() {
    let mut dedup = CrossModuleDeduplicator::new();

    dedup.register("source-1", "target-1", 0.1);
    let is_dup = dedup.is_duplicate("source-1", "target-1", 0.1);
    assert!(is_dup, "Same source-target-gain should be duplicate");
}

#[test]
fn test_deduplicator_detects_similar_candidate() {
    let mut dedup = CrossModuleDeduplicator::new();

    dedup.register("source-1", "target-1", 0.100);
    // Same source/target with very similar gain
    let is_dup = dedup.is_duplicate("source-1", "target-1", 0.102);
    assert!(is_dup, "Similar gain to same target should be duplicate");
}

#[test]
fn test_deduplicator_allows_different_target() {
    let mut dedup = CrossModuleDeduplicator::new();

    dedup.register("source-1", "target-1", 0.1);
    // Different target should not be a duplicate
    let is_dup = dedup.is_duplicate("source-1", "target-2", 0.1);
    assert!(!is_dup, "Different target should not be duplicate");
}

#[test]
fn test_deduplicator_allows_significantly_different_gain() {
    let mut dedup = CrossModuleDeduplicator::new();

    dedup.register("source-1", "target-1", 0.1);
    // Same source/target but very different gain — likely a distinct candidate
    let is_dup = dedup.is_duplicate("source-1", "target-1", 0.5);
    assert!(
        !is_dup,
        "Very different gain on same pair should not be duplicate"
    );
}

#[test]
fn test_deduplicator_tracks_statistics() {
    let mut dedup = CrossModuleDeduplicator::new();

    dedup.register("s1", "t1", 0.1);
    dedup.register("s2", "t2", 0.2);
    let _ = dedup.is_duplicate("s1", "t1", 0.1); // duplicate

    let stats = dedup.statistics();
    assert_eq!(stats.total_registered, 2);
    assert_eq!(stats.total_duplicate_checks, 1);
}

#[test]
fn test_deduplicator_register_and_check() {
    let mut dedup = CrossModuleDeduplicator::new();

    // register_if_unique should register and return true for new candidates
    let unique = dedup.register_if_unique("s1", "t1", 0.1);
    assert!(unique, "New candidate should be unique");

    // duplicate should return false and NOT register
    let unique = dedup.register_if_unique("s1", "t1", 0.101);
    assert!(!unique, "Duplicate should not be unique");

    let stats = dedup.statistics();
    assert_eq!(stats.total_registered, 1, "Only first should be registered");
}
