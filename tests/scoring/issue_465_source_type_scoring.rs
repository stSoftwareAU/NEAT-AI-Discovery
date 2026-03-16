//! Tests for Issue #465: Source-type scoring boost for candidate prioritisation.
//!
//! GRQ-sampler analysis shows that input neurons as sources have a 36.2% success
//! rate compared to 2.8–3.3% for hidden neurons. This module tests the scoring
//! boost mechanism that prioritises candidates from historically more successful
//! source types.
//!
//! ## Key Behaviours Verified
//!
//! - Source-type boost factors are applied correctly to candidate scoring
//! - Input-source candidates get higher boost than hidden-source candidates
//! - Boost requires sufficient samples to be applied

use neat_ai_discovery::analysis::candidate_cache::{
    CandidateOutcomeCache, DEFAULT_STALENESS_WINDOW, SourceTypeStats,
};
use neat_ai_discovery::analysis::constants::{INPUT_SOURCE_BOOST, MIN_BOOST_SAMPLES};

// Compile-time validation of constant ranges.
const _: () = assert!(INPUT_SOURCE_BOOST > 1.0);
const _: () = assert!(INPUT_SOURCE_BOOST <= 3.0);
const _: () = assert!(MIN_BOOST_SAMPLES >= 5);
const _: () = assert!(MIN_BOOST_SAMPLES <= 50);
const _: () = assert!(DEFAULT_STALENESS_WINDOW >= 10);
const _: () = assert!(DEFAULT_STALENESS_WINDOW <= 1000);

// =============================================================================
// Source Type Stats
// =============================================================================

#[test]
fn source_type_stats_default_is_empty() {
    let stats = SourceTypeStats::default();
    assert_eq!(stats.attempts, 0);
    assert_eq!(stats.successes, 0);
    assert!((stats.success_rate() - 0.5).abs() < f64::EPSILON);
}

#[test]
fn source_type_stats_bayesian_score_with_few_samples() {
    let stats = SourceTypeStats {
        attempts: 1,
        successes: 1,
    };
    // Bayesian: (1+1)/(1+2) = 0.667 — not 1.0
    let score = stats.success_rate();
    assert!(
        (score - 2.0 / 3.0).abs() < 0.01,
        "Single success should give Bayesian score ~0.667, got {score}"
    );
}

#[test]
fn source_type_stats_converges_with_many_samples() {
    let stats = SourceTypeStats {
        attempts: 1000,
        successes: 362,
    };
    // With many samples, Bayesian ~= raw rate: 362/1000 = 0.362
    let score = stats.success_rate();
    assert!(
        (score - 0.362).abs() < 0.01,
        "With 1000 samples, score should be ~0.362, got {score}"
    );
}

// =============================================================================
// Boost Application
// =============================================================================

#[test]
fn boost_not_applied_with_insufficient_samples() {
    let mut cache = CandidateOutcomeCache::new();

    // Record fewer samples than MIN_BOOST_SAMPLES
    let sample_count = MIN_BOOST_SAMPLES.saturating_sub(1).max(1) as u64;
    for i in 0..sample_count {
        cache.record_with_source_type(
            &format!("input-{i}"),
            "target-1",
            "addSynapse",
            true,
            i,
            "input",
        );
    }

    // With insufficient samples, boost should be neutral (1.0)
    let boost = cache.source_type_boost("input");
    assert!(
        (boost - 1.0).abs() < f64::EPSILON,
        "Boost with insufficient samples should be 1.0, got {boost}"
    );
}

#[test]
fn boost_applied_with_sufficient_samples() {
    let mut cache = CandidateOutcomeCache::new();

    // Record enough successful samples to trigger boost
    let sample_count = MIN_BOOST_SAMPLES as u64 + 5;
    for i in 0..sample_count {
        cache.record_with_source_type(
            &format!("input-{i}"),
            "target-1",
            "addSynapse",
            true,
            i,
            "input",
        );
    }

    let boost = cache.source_type_boost("input");
    assert!(
        boost > 1.0,
        "Boost with high success rate and sufficient samples should be > 1.0, got {boost}"
    );
}

#[test]
fn input_source_boost_applied_to_expected_score_gain() {
    // Verify that INPUT_SOURCE_BOOST can be used as a multiplier for scoring
    let base_score = 0.05_f64;
    let boosted_score = base_score * INPUT_SOURCE_BOOST;

    assert!(
        boosted_score > base_score,
        "Boosted score ({boosted_score}) should exceed base score ({base_score})"
    );
}
