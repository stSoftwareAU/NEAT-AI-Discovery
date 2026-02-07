//! Tests for Issue #429: Early termination improvements for low-value candidates.
//!
//! Validates the four improvements from the issue:
//! 1. Hierarchical candidate filtering (minimum gain threshold)
//! 2. Budget-aware prioritisation (stop when budget exhausted)
//! 3. Incremental confidence (low-gain filtering)
//! 4. Cross-module deduplication (skip duplicate neuron pairs)

use neat_ai_discovery::analysis::candidate_prefilter::{
    CandidatePreFilter, PreFilterConfig, MIN_CANDIDATE_GAIN,
};
use neat_ai_discovery::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson};

// =============================================================================
// Test Helpers
// =============================================================================

fn make_add_synapse_candidate(
    from: &str,
    to: &str,
    gain: f32,
) -> CoordinatedStructuralCandidateJson {
    CoordinatedStructuralCandidateJson {
        operations: vec![CoordinatedStructuralOpJson::AddSynapse {
            from_neuron_uuid: from.to_string(),
            to_neuron_uuid: to.to_string(),
            weight: 0.5,
        }],
        expected_creature_score_gain: gain,
        comment: None,
    }
}

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

fn make_set_bias_candidate(neuron: &str, gain: f32) -> CoordinatedStructuralCandidateJson {
    CoordinatedStructuralCandidateJson {
        operations: vec![CoordinatedStructuralOpJson::SetBias {
            neuron_uuid: neuron.to_string(),
            bias: 0.1,
        }],
        expected_creature_score_gain: gain,
        comment: None,
    }
}

// =============================================================================
// 1. Hierarchical Candidate Filtering
// =============================================================================

/// Candidates with expected gain below the minimum threshold are rejected.
#[test]
fn test_hierarchical_filtering_rejects_low_gain() {
    let config = PreFilterConfig {
        min_gain: 0.01,
        candidate_budget: 1000,
        dedup_enabled: false,
    };
    let mut filter = CandidatePreFilter::new(config);

    let candidates = vec![
        make_add_synapse_candidate("input-0", "output-0", 0.1), // good
        make_add_synapse_candidate("input-1", "output-0", 0.005), // too low
        make_add_synapse_candidate("input-2", "output-0", 0.05), // good
        make_add_synapse_candidate("input-3", "output-0", 0.0), // zero gain
        make_add_synapse_candidate("input-4", "output-0", -0.01), // negative
    ];

    let result = filter.filter(candidates);

    assert_eq!(
        result.len(),
        2,
        "Only candidates above min_gain threshold should pass"
    );
    assert!(
        result[0].expected_creature_score_gain > 0.01,
        "First kept candidate should exceed threshold"
    );
    assert!(
        result[1].expected_creature_score_gain > 0.01,
        "Second kept candidate should exceed threshold"
    );

    let stats = filter.stats();
    assert_eq!(stats.rejected_low_gain, 3);
}

/// The default MIN_CANDIDATE_GAIN is very conservative (near zero).
#[test]
fn test_default_min_gain_is_conservative() {
    // Use a binding to avoid clippy::assertions_on_constants
    let gain = MIN_CANDIDATE_GAIN;
    assert!(gain > 0.0, "MIN_CANDIDATE_GAIN should be positive");
    assert!(
        gain < 1e-6,
        "MIN_CANDIDATE_GAIN should be very small to avoid filtering genuine improvements"
    );
}

/// Candidates exactly at the threshold are rejected (strict inequality).
#[test]
fn test_filtering_at_exact_threshold() {
    let config = PreFilterConfig {
        min_gain: 0.05,
        candidate_budget: 1000,
        dedup_enabled: false,
    };
    let mut filter = CandidatePreFilter::new(config);

    let candidates = vec![
        make_add_synapse_candidate("a", "b", 0.05), // exactly at threshold — rejected
        make_add_synapse_candidate("c", "d", 0.050_000_01), // just above — accepted
    ];

    let result = filter.filter(candidates);
    assert_eq!(
        result.len(),
        1,
        "Exact threshold should be rejected (strict >)"
    );
}

// =============================================================================
// 2. Budget-Aware Prioritisation
// =============================================================================

/// Budget limits the total number of candidates accepted across all filter calls.
#[test]
fn test_budget_limits_candidates_across_modules() {
    let config = PreFilterConfig {
        min_gain: MIN_CANDIDATE_GAIN,
        candidate_budget: 3,
        dedup_enabled: false,
    };
    let mut filter = CandidatePreFilter::new(config);

    // Module 1: 2 candidates
    let batch1 = vec![
        make_add_synapse_candidate("a", "b", 0.1),
        make_add_synapse_candidate("c", "d", 0.2),
    ];
    let result1 = filter.filter(batch1);
    assert_eq!(result1.len(), 2);
    assert_eq!(filter.remaining_budget(), 1);

    // Module 2: 3 candidates, but only 1 budget left
    let batch2 = vec![
        make_add_synapse_candidate("e", "f", 0.3),
        make_add_synapse_candidate("g", "h", 0.4),
        make_add_synapse_candidate("i", "j", 0.5),
    ];
    let result2 = filter.filter(batch2);
    assert_eq!(result2.len(), 1, "Only 1 more should fit in budget");
    assert!(filter.budget_exhausted());

    let stats = filter.stats();
    assert_eq!(stats.accepted, 3);
    assert_eq!(stats.rejected_budget, 2);
}

/// Once budget is exhausted, `budget_exhausted()` returns true.
#[test]
fn test_budget_exhaustion_flag() {
    let config = PreFilterConfig {
        min_gain: MIN_CANDIDATE_GAIN,
        candidate_budget: 1,
        dedup_enabled: false,
    };
    let mut filter = CandidatePreFilter::new(config);

    assert!(
        !filter.budget_exhausted(),
        "Fresh filter should not be exhausted"
    );

    let batch = vec![make_add_synapse_candidate("a", "b", 0.1)];
    filter.filter(batch);

    assert!(
        filter.budget_exhausted(),
        "Should be exhausted after filling budget"
    );
}

/// Modules can be skipped entirely when budget is exhausted.
#[test]
fn test_module_skip_tracking() {
    let config = PreFilterConfig {
        min_gain: MIN_CANDIDATE_GAIN,
        candidate_budget: 0, // zero budget — all modules should be skipped
        dedup_enabled: false,
    };
    let mut filter = CandidatePreFilter::new(config);

    assert!(filter.budget_exhausted());

    // Simulate 3 modules being skipped
    filter.record_module_skipped();
    filter.record_module_skipped();
    filter.record_module_skipped();

    assert_eq!(filter.stats().modules_skipped, 3);
}

// =============================================================================
// 3. Incremental Confidence (Low-Gain Filtering)
// =============================================================================

/// Filtering correctly removes near-zero and negative gain candidates.
#[test]
fn test_incremental_confidence_near_zero_gains() {
    let config = PreFilterConfig::default();
    let mut filter = CandidatePreFilter::new(config);

    let candidates = vec![
        make_add_synapse_candidate("a", "b", 1e-10), // below MIN_CANDIDATE_GAIN
        make_add_synapse_candidate("c", "d", 0.0),   // zero
        make_add_synapse_candidate("e", "f", -0.001), // negative
        make_add_synapse_candidate("g", "h", 0.001), // above MIN_CANDIDATE_GAIN
    ];

    let result = filter.filter(candidates);

    // Only the last candidate (0.001) should survive the default filter
    assert_eq!(
        result.len(),
        1,
        "Only candidates with gain > MIN_CANDIDATE_GAIN should pass"
    );
    assert!(result[0].expected_creature_score_gain > MIN_CANDIDATE_GAIN);
}

// =============================================================================
// 4. Cross-Module Deduplication
// =============================================================================

/// Duplicate (from, to) pairs across modules are rejected.
#[test]
fn test_cross_module_dedup_rejects_duplicate_pairs() {
    let config = PreFilterConfig {
        min_gain: MIN_CANDIDATE_GAIN,
        candidate_budget: 100,
        dedup_enabled: true,
    };
    let mut filter = CandidatePreFilter::new(config);

    // Module 1 produces a candidate for (input-0, output-0)
    let batch1 = vec![make_add_synapse_candidate("input-0", "output-0", 0.1)];
    let result1 = filter.filter(batch1);
    assert_eq!(result1.len(), 1);

    // Module 2 produces the same pair via RemoveSynapse
    let batch2 = vec![make_remove_synapse_candidate("input-0", "output-0", 0.2)];
    let result2 = filter.filter(batch2);
    assert_eq!(
        result2.len(),
        0,
        "Duplicate pair should be rejected even across different operation types"
    );

    assert_eq!(filter.stats().rejected_dedup, 1);
}

/// Different (from, to) pairs are not considered duplicates.
#[test]
fn test_dedup_allows_different_pairs() {
    let config = PreFilterConfig {
        min_gain: MIN_CANDIDATE_GAIN,
        candidate_budget: 100,
        dedup_enabled: true,
    };
    let mut filter = CandidatePreFilter::new(config);

    let batch1 = vec![make_add_synapse_candidate("input-0", "output-0", 0.1)];
    let batch2 = vec![make_add_synapse_candidate("input-1", "output-0", 0.2)];
    let batch3 = vec![make_add_synapse_candidate("input-0", "output-1", 0.3)];

    let r1 = filter.filter(batch1);
    let r2 = filter.filter(batch2);
    let r3 = filter.filter(batch3);

    assert_eq!(r1.len(), 1);
    assert_eq!(r2.len(), 1);
    assert_eq!(r3.len(), 1);
    assert_eq!(filter.stats().rejected_dedup, 0);
}

/// Candidates without synapse-pair signatures (e.g., SetBias) are never deduplicated.
#[test]
fn test_dedup_does_not_affect_non_pair_candidates() {
    let config = PreFilterConfig {
        min_gain: MIN_CANDIDATE_GAIN,
        candidate_budget: 100,
        dedup_enabled: true,
    };
    let mut filter = CandidatePreFilter::new(config);

    // Two SetBias candidates for the same neuron — both should pass
    // because SetBias has no from/to pair signature
    let batch1 = vec![make_set_bias_candidate("neuron-0", 0.1)];
    let batch2 = vec![make_set_bias_candidate("neuron-0", 0.2)];

    let r1 = filter.filter(batch1);
    let r2 = filter.filter(batch2);

    assert_eq!(r1.len(), 1);
    assert_eq!(r2.len(), 1);
    assert_eq!(filter.stats().rejected_dedup, 0);
}

/// Deduplication can be disabled via configuration.
#[test]
fn test_dedup_disabled_allows_duplicates() {
    let config = PreFilterConfig {
        min_gain: MIN_CANDIDATE_GAIN,
        candidate_budget: 100,
        dedup_enabled: false,
    };
    let mut filter = CandidatePreFilter::new(config);

    let batch1 = vec![make_add_synapse_candidate("input-0", "output-0", 0.1)];
    let batch2 = vec![make_add_synapse_candidate("input-0", "output-0", 0.2)];

    let r1 = filter.filter(batch1);
    let r2 = filter.filter(batch2);

    assert_eq!(r1.len(), 1);
    assert_eq!(r2.len(), 1, "Duplicates should pass when dedup is disabled");
    assert_eq!(filter.stats().rejected_dedup, 0);
}

// =============================================================================
// Combined Filter Behaviour
// =============================================================================

/// All filter stages work together in the correct priority order.
#[test]
fn test_combined_filtering_stages() {
    let config = PreFilterConfig {
        min_gain: 0.01,
        candidate_budget: 3,
        dedup_enabled: true,
    };
    let mut filter = CandidatePreFilter::new(config);

    // Module 1: 2 candidates accepted
    let batch1 = vec![
        make_add_synapse_candidate("a", "b", 0.1), // accepted
        make_add_synapse_candidate("c", "d", 0.2), // accepted
    ];
    let r1 = filter.filter(batch1);
    assert_eq!(r1.len(), 2);

    // Module 2: tests all three rejection reasons
    let batch2 = vec![
        make_add_synapse_candidate("e", "f", 0.005), // rejected: low gain
        make_add_synapse_candidate("a", "b", 0.3),   // rejected: dedup
        make_add_synapse_candidate("g", "h", 0.4),   // accepted (last budget slot)
        make_add_synapse_candidate("i", "j", 0.5),   // rejected: budget
    ];
    let r2 = filter.filter(batch2);
    assert_eq!(r2.len(), 1, "Only one should fit after filters");

    let stats = filter.stats();
    assert_eq!(stats.accepted, 3);
    assert_eq!(stats.rejected_low_gain, 1);
    assert_eq!(stats.rejected_dedup, 1);
    assert_eq!(stats.rejected_budget, 1);
    assert_eq!(stats.total_rejected(), 3);
}

/// PreFilterStats total_rejected sums all rejection categories.
#[test]
fn test_stats_total_rejected_is_sum() {
    let config = PreFilterConfig {
        min_gain: 0.1,
        candidate_budget: 1,
        dedup_enabled: true,
    };
    let mut filter = CandidatePreFilter::new(config);

    let batch = vec![
        make_add_synapse_candidate("a", "b", 0.5),  // accepted
        make_add_synapse_candidate("c", "d", 0.01), // rejected: low gain
        make_add_synapse_candidate("a", "b", 0.3),  // rejected: budget (already at cap)
    ];
    filter.filter(batch);
    filter.record_module_skipped();

    let stats = filter.stats();
    assert_eq!(
        stats.total_rejected(),
        stats.rejected_low_gain + stats.rejected_budget + stats.rejected_dedup,
        "total_rejected should be sum of all rejection categories"
    );
}
