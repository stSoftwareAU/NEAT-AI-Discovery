//! Tests for Issue #465: Candidate outcome cache to improve discovery success rate.
//!
//! The candidate outcome cache tracks which candidates have been suggested and
//! whether they succeeded or failed ablation testing. This enables:
//!
//! 1. Suppressing candidates that have repeatedly failed
//! 2. Re-enabling candidates after a staleness window (the creature may have changed)
//! 3. Boosting candidates from source types with higher historical success rates
//!
//! ## Key Behaviours Verified
//!
//! - Recording candidate outcomes (success/failure)
//! - Suppressing recently-failed candidates
//! - Re-enabling candidates after staleness window expires
//! - Source-type success rate tracking
//! - Serialisation/deserialisation for persistence

#![allow(clippy::cast_sign_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use neat_ai_discovery::analysis::candidate_cache::CandidateOutcomeCache;

// =============================================================================
// Basic Recording and Lookup
// =============================================================================

#[test]
fn empty_cache_returns_no_outcome() {
    let cache = CandidateOutcomeCache::new();
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
    let outcome = cache.get_outcome("source-1", "target-1", "addSynapse");
    assert!(outcome.is_none());
}

#[test]
fn record_success_and_retrieve() {
    let mut cache = CandidateOutcomeCache::new();
    cache.record("source-1", "target-1", "addSynapse", true, 100);

    let outcome = cache.get_outcome("source-1", "target-1", "addSynapse");
    assert!(outcome.is_some());
    let outcome = outcome.unwrap();
    assert!(outcome.succeeded);
    assert_eq!(outcome.epoch, 100);
}

#[test]
fn record_failure_and_retrieve() {
    let mut cache = CandidateOutcomeCache::new();
    cache.record("source-1", "target-1", "addSynapse", false, 200);

    let outcome = cache.get_outcome("source-1", "target-1", "addSynapse");
    assert!(outcome.is_some());
    let outcome = outcome.unwrap();
    assert!(!outcome.succeeded);
    assert_eq!(outcome.epoch, 200);
}

#[test]
fn latest_outcome_overwrites_previous() {
    let mut cache = CandidateOutcomeCache::new();
    cache.record("source-1", "target-1", "addSynapse", false, 100);
    cache.record("source-1", "target-1", "addSynapse", true, 200);

    let outcome = cache
        .get_outcome("source-1", "target-1", "addSynapse")
        .unwrap();
    assert!(outcome.succeeded);
    assert_eq!(outcome.epoch, 200);
}

// =============================================================================
// Candidate Suppression
// =============================================================================

#[test]
fn recently_failed_candidate_is_suppressed() {
    let mut cache = CandidateOutcomeCache::new();
    cache.record("source-1", "target-1", "addSynapse", false, 100);

    // At epoch 105, within default staleness window, candidate should be suppressed
    assert!(cache.is_suppressed("source-1", "target-1", "addSynapse", 105));
}

#[test]
fn successful_candidate_is_not_suppressed() {
    let mut cache = CandidateOutcomeCache::new();
    cache.record("source-1", "target-1", "addSynapse", true, 100);

    // Successful candidates should never be suppressed
    assert!(!cache.is_suppressed("source-1", "target-1", "addSynapse", 101));
}

#[test]
fn failed_candidate_becomes_eligible_after_staleness_window() {
    let mut cache = CandidateOutcomeCache::new();
    let staleness_window = cache.staleness_window();
    cache.record("source-1", "target-1", "addSynapse", false, 100);

    // Just before window expires: still suppressed
    assert!(cache.is_suppressed(
        "source-1",
        "target-1",
        "addSynapse",
        100 + staleness_window - 1
    ));

    // At window boundary: no longer suppressed
    assert!(!cache.is_suppressed("source-1", "target-1", "addSynapse", 100 + staleness_window));
}

#[test]
fn unknown_candidate_is_not_suppressed() {
    let cache = CandidateOutcomeCache::new();
    assert!(!cache.is_suppressed("source-1", "target-1", "addSynapse", 100));
}

// =============================================================================
// Different Operation Types Are Independent
// =============================================================================

#[test]
fn different_operations_tracked_independently() {
    let mut cache = CandidateOutcomeCache::new();
    cache.record("source-1", "target-1", "addSynapse", false, 100);
    cache.record("source-1", "target-1", "addNeuron", true, 100);

    assert!(cache.is_suppressed("source-1", "target-1", "addSynapse", 105));
    assert!(!cache.is_suppressed("source-1", "target-1", "addNeuron", 105));
}

// =============================================================================
// Source-Type Statistics
// =============================================================================

#[test]
fn source_type_stats_track_success_rates() {
    let mut cache = CandidateOutcomeCache::new();

    // Record input-neuron source outcomes
    cache.record_with_source_type("input-1", "target-1", "addSynapse", true, 100, "input");
    cache.record_with_source_type("input-2", "target-1", "addSynapse", true, 101, "input");
    cache.record_with_source_type("input-3", "target-1", "addSynapse", false, 102, "input");

    // Record hidden-neuron source outcomes
    cache.record_with_source_type("hidden-1", "target-1", "addSynapse", false, 100, "hidden");
    cache.record_with_source_type("hidden-2", "target-1", "addSynapse", false, 101, "hidden");
    cache.record_with_source_type("hidden-3", "target-1", "addSynapse", true, 102, "hidden");

    let input_stats = cache.source_type_stats("input");
    assert_eq!(input_stats.attempts, 3);
    assert_eq!(input_stats.successes, 2);

    let hidden_stats = cache.source_type_stats("hidden");
    assert_eq!(hidden_stats.attempts, 3);
    assert_eq!(hidden_stats.successes, 1);

    // Input neurons should have higher success rate
    assert!(input_stats.success_rate() > hidden_stats.success_rate());
}

#[test]
fn source_type_boost_factor_reflects_success_rate() {
    let mut cache = CandidateOutcomeCache::new();

    // Input: 80% success (8 of 10)
    for i in 0..10 {
        cache.record_with_source_type(
            &format!("input-{i}"),
            "target-1",
            "addSynapse",
            i < 8,
            i as u64,
            "input",
        );
    }

    // Hidden: 20% success (2 of 10)
    for i in 0..10 {
        cache.record_with_source_type(
            &format!("hidden-{i}"),
            "target-1",
            "addSynapse",
            i < 2,
            i as u64,
            "hidden",
        );
    }

    let input_boost = cache.source_type_boost("input");
    let hidden_boost = cache.source_type_boost("hidden");

    // Higher success rate should give higher boost
    assert!(
        input_boost > hidden_boost,
        "Input boost ({input_boost}) should be > hidden boost ({hidden_boost})"
    );

    // Unknown types get neutral boost (1.0)
    let unknown_boost = cache.source_type_boost("constant");
    assert!(
        (unknown_boost - 1.0).abs() < f64::EPSILON,
        "Unknown source type should get neutral boost (1.0), got {unknown_boost}"
    );
}

// =============================================================================
// Serialisation Round-Trip
// =============================================================================

#[test]
fn serialisation_round_trip() {
    let mut cache = CandidateOutcomeCache::new();
    cache.record_with_source_type("source-1", "target-1", "addSynapse", true, 100, "input");
    cache.record_with_source_type("source-2", "target-1", "addNeuron", false, 200, "hidden");

    let json = serde_json::to_string(&cache).expect("Serialisation should succeed");
    let deserialized: CandidateOutcomeCache =
        serde_json::from_str(&json).expect("Deserialisation should succeed");

    assert_eq!(cache.len(), deserialized.len());

    let o1 = deserialized
        .get_outcome("source-1", "target-1", "addSynapse")
        .expect("Should find source-1 outcome");
    assert!(o1.succeeded);
    assert_eq!(o1.epoch, 100);

    let o2 = deserialized
        .get_outcome("source-2", "target-1", "addNeuron")
        .expect("Should find source-2 outcome");
    assert!(!o2.succeeded);
    assert_eq!(o2.epoch, 200);
}

// =============================================================================
// Pruning
// =============================================================================

#[test]
fn prune_removes_stale_entries() {
    let mut cache = CandidateOutcomeCache::new();
    cache.record("source-1", "target-1", "addSynapse", false, 100);
    cache.record("source-2", "target-1", "addSynapse", true, 500);

    assert_eq!(cache.len(), 2);

    // Prune entries older than epoch 300
    cache.prune_before_epoch(300);

    assert_eq!(cache.len(), 1);
    assert!(
        cache
            .get_outcome("source-1", "target-1", "addSynapse")
            .is_none()
    );
    assert!(
        cache
            .get_outcome("source-2", "target-1", "addSynapse")
            .is_some()
    );
}

// =============================================================================
// Custom Staleness Window
// =============================================================================

#[test]
fn custom_staleness_window() {
    let mut cache = CandidateOutcomeCache::with_staleness_window(50);
    assert_eq!(cache.staleness_window(), 50);

    cache.record("source-1", "target-1", "addSynapse", false, 100);

    assert!(cache.is_suppressed("source-1", "target-1", "addSynapse", 140));
    assert!(!cache.is_suppressed("source-1", "target-1", "addSynapse", 150));
}
