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
// Issue #1203 changed `is_suppressed` to require a `DiscoveryMode` and a
// `drought_failures` count so the staleness window can adapt to droughts.
// Passing `(DiscoveryMode::Normal, 0)` preserves the original behaviour these
// tests were written against.
use neat_ai_discovery::analysis::discovery_mode::DiscoveryMode;

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
    assert!(cache.is_suppressed(
        "source-1",
        "target-1",
        "addSynapse",
        105,
        DiscoveryMode::Normal,
        0,
    ));
}

#[test]
fn successful_candidate_is_not_suppressed() {
    let mut cache = CandidateOutcomeCache::new();
    cache.record("source-1", "target-1", "addSynapse", true, 100);

    // Successful candidates should never be suppressed
    assert!(!cache.is_suppressed(
        "source-1",
        "target-1",
        "addSynapse",
        101,
        DiscoveryMode::Normal,
        0,
    ));
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
        100 + staleness_window - 1,
        DiscoveryMode::Normal,
        0,
    ));

    // At window boundary: no longer suppressed
    assert!(!cache.is_suppressed(
        "source-1",
        "target-1",
        "addSynapse",
        100 + staleness_window,
        DiscoveryMode::Normal,
        0,
    ));
}

#[test]
fn unknown_candidate_is_not_suppressed() {
    let cache = CandidateOutcomeCache::new();
    assert!(!cache.is_suppressed(
        "source-1",
        "target-1",
        "addSynapse",
        100,
        DiscoveryMode::Normal,
        0,
    ));
}

// =============================================================================
// Different Operation Types Are Independent
// =============================================================================

#[test]
fn different_operations_tracked_independently() {
    let mut cache = CandidateOutcomeCache::new();
    cache.record("source-1", "target-1", "addSynapse", false, 100);
    cache.record("source-1", "target-1", "addNeuron", true, 100);

    assert!(cache.is_suppressed(
        "source-1",
        "target-1",
        "addSynapse",
        105,
        DiscoveryMode::Normal,
        0,
    ));
    assert!(!cache.is_suppressed(
        "source-1",
        "target-1",
        "addNeuron",
        105,
        DiscoveryMode::Normal,
        0,
    ));
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

    assert!(cache.is_suppressed(
        "source-1",
        "target-1",
        "addSynapse",
        140,
        DiscoveryMode::Normal,
        0,
    ));
    assert!(!cache.is_suppressed(
        "source-1",
        "target-1",
        "addSynapse",
        150,
        DiscoveryMode::Normal,
        0,
    ));
}

// =============================================================================
// Issue #1205 — clear_failed_entries + drought-reset tombstone
// =============================================================================

#[test]
fn clear_failed_entries_on_empty_cache_returns_zero() {
    let mut cache = CandidateOutcomeCache::new();
    assert_eq!(cache.clear_failed_entries(5), 0);
    // Tombstone still set so a second call within the same streak is a no-op.
    assert_eq!(cache.drought_reset_tombstone(), Some(5));
}

#[test]
fn clear_failed_entries_removes_only_failures_preserving_successes() {
    let mut cache = CandidateOutcomeCache::new();
    cache.record("s1", "t1", "addSynapse", false, 1);
    cache.record("s2", "t2", "addSynapse", false, 2);
    cache.record("s3", "t3", "addSynapse", true, 3);
    cache.record_with_source_type("s4", "t4", "addSynapse", false, 4, "hidden");

    let removed = cache.clear_failed_entries(10);
    assert_eq!(removed, 3);
    assert_eq!(cache.len(), 1);

    // Successful candidate retained.
    assert!(
        cache
            .get_outcome("s3", "t3", "addSynapse")
            .is_some_and(|o| o.succeeded)
    );

    // Source-type stats preserved (institutional memory).
    let stats = cache.source_type_stats("hidden");
    assert_eq!(stats.attempts, 1);
    assert_eq!(stats.successes, 0);
}

#[test]
fn clear_failed_entries_all_failed() {
    let mut cache = CandidateOutcomeCache::new();
    for i in 0..5 {
        cache.record(&format!("s{i}"), &format!("t{i}"), "addSynapse", false, i);
    }
    let removed = cache.clear_failed_entries(100);
    assert_eq!(removed, 5);
    assert!(cache.is_empty());
}

#[test]
fn clear_failed_entries_all_success() {
    let mut cache = CandidateOutcomeCache::new();
    for i in 0..3 {
        cache.record(&format!("s{i}"), &format!("t{i}"), "addSynapse", true, i);
    }
    let removed = cache.clear_failed_entries(100);
    assert_eq!(removed, 0);
    assert_eq!(cache.len(), 3);
    assert_eq!(cache.drought_reset_tombstone(), Some(100));
}

#[test]
fn record_success_clears_drought_reset_tombstone() {
    let mut cache = CandidateOutcomeCache::new();
    cache.record("s1", "t1", "addSynapse", false, 0);
    cache.clear_failed_entries(3);
    assert_eq!(cache.drought_reset_tombstone(), Some(3));

    // Successful outcome re-arms the lever.
    cache.record("s-good", "t-good", "addSynapse", true, 4);
    assert!(cache.drought_reset_tombstone().is_none());
}

#[test]
fn record_failure_does_not_clear_tombstone() {
    let mut cache = CandidateOutcomeCache::new();
    cache.clear_failed_entries(5);
    cache.record("s1", "t1", "addSynapse", false, 6);
    assert_eq!(cache.drought_reset_tombstone(), Some(5));
}
