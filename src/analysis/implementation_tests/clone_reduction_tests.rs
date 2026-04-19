//! Issue #526: Verify that hash-based candidate deduplication maintains
//! correctness after replacing String-based `HashMap` keys with FNV-1a hash keys.
//!
//! These tests exercise `upsert_candidate` and `compute_candidate_dedup_key`
//! with various candidate configurations, verifying deduplication semantics
//! are preserved after the clone-reduction optimisation.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use super::common::*;

/// Helper to create a candidate with given source/target/squash and weight signs.
fn make_candidate(
    source: &str,
    target: &str,
    squash: &str,
    incoming: f32,
    outgoing: f32,
    gain: f32,
) -> CandidateNeuronJson {
    CandidateNeuronJson {
        source_neuron_uuid: source.to_string(),
        target_neuron_uuid: target.to_string(),
        source_neuron_index: None,
        target_neuron_index: None,
        incoming_weight: incoming,
        outgoing_weight: outgoing,
        squash: squash.to_string(),
        bias: 0.0,
        comment: None,
        target_neuron_impact: 1.0,
        expected_creature_error_reduction: gain,
        expected_creature_score_gain: gain,
        improved_count: 50,
        total_count: 100,
        target_neuron_stats: None,
        prediction_confidence: 0.5,
        expected_score_gain_confidence_interval: [0.01, 0.09],
        target_saturation_factor: None,
    }
}

/// Same source/target/squash and weight signs → should deduplicate (keep higher gain).
#[test]
fn upsert_deduplicates_identical_keys() {
    let mut map: HashMap<u64, CandidateNeuronJson> = HashMap::new();

    let c1 = make_candidate("input-0", "hidden-1", "ReLU", 1.0, 0.5, 0.10);
    let c2 = make_candidate("input-0", "hidden-1", "ReLU", 1.0, 0.3, 0.15); // higher gain

    upsert_candidate(&mut map, c1);
    upsert_candidate(&mut map, c2);

    assert_eq!(map.len(), 1, "Same key should deduplicate to 1 entry");

    let winner = map.values().next().unwrap();
    assert!(
        (winner.expected_creature_score_gain - 0.15).abs() < 1e-6,
        "Higher gain candidate should win: got {}",
        winner.expected_creature_score_gain
    );
}

/// Different source UUIDs → should produce distinct keys.
#[test]
fn different_sources_not_deduplicated() {
    let mut map: HashMap<u64, CandidateNeuronJson> = HashMap::new();

    let c1 = make_candidate("input-0", "hidden-1", "ReLU", 1.0, 0.5, 0.10);
    let c2 = make_candidate("input-1", "hidden-1", "ReLU", 1.0, 0.5, 0.10);

    upsert_candidate(&mut map, c1);
    upsert_candidate(&mut map, c2);

    assert_eq!(
        map.len(),
        2,
        "Different source UUIDs should produce distinct keys"
    );
}

/// Different target UUIDs → should produce distinct keys.
#[test]
fn different_targets_not_deduplicated() {
    let mut map: HashMap<u64, CandidateNeuronJson> = HashMap::new();

    let c1 = make_candidate("input-0", "hidden-1", "ReLU", 1.0, 0.5, 0.10);
    let c2 = make_candidate("input-0", "hidden-2", "ReLU", 1.0, 0.5, 0.10);

    upsert_candidate(&mut map, c1);
    upsert_candidate(&mut map, c2);

    assert_eq!(
        map.len(),
        2,
        "Different target UUIDs should produce distinct keys"
    );
}

/// Different squash functions → should produce distinct keys.
#[test]
fn different_squash_not_deduplicated() {
    let mut map: HashMap<u64, CandidateNeuronJson> = HashMap::new();

    let c1 = make_candidate("input-0", "hidden-1", "ReLU", 1.0, 0.5, 0.10);
    let c2 = make_candidate("input-0", "hidden-1", "TANH", 1.0, 0.5, 0.10);

    upsert_candidate(&mut map, c1);
    upsert_candidate(&mut map, c2);

    assert_eq!(
        map.len(),
        2,
        "Different squash functions should produce distinct keys"
    );
}

/// Different incoming weight signs → should produce distinct keys.
#[test]
fn different_incoming_signs_not_deduplicated() {
    let mut map: HashMap<u64, CandidateNeuronJson> = HashMap::new();

    let c1 = make_candidate("input-0", "hidden-1", "ReLU", 1.0, 0.5, 0.10);
    let c2 = make_candidate("input-0", "hidden-1", "ReLU", -1.0, 0.5, 0.10);

    upsert_candidate(&mut map, c1);
    upsert_candidate(&mut map, c2);

    assert_eq!(
        map.len(),
        2,
        "Different incoming weight signs should produce distinct keys"
    );
}

/// Different outgoing weight signs → should produce distinct keys.
#[test]
fn different_outgoing_signs_not_deduplicated() {
    let mut map: HashMap<u64, CandidateNeuronJson> = HashMap::new();

    let c1 = make_candidate("input-0", "hidden-1", "ReLU", 1.0, 0.5, 0.10);
    let c2 = make_candidate("input-0", "hidden-1", "ReLU", 1.0, -0.5, 0.10);

    upsert_candidate(&mut map, c1);
    upsert_candidate(&mut map, c2);

    assert_eq!(
        map.len(),
        2,
        "Different outgoing weight signs should produce distinct keys"
    );
}

/// Verify no accidental collisions between UUID patterns that could overlap
/// without the separator byte (e.g., "input-1" + "hidden-23" vs "input-12" + "hidden-3").
#[test]
fn no_collision_between_similar_uuid_concatenations() {
    let mut map: HashMap<u64, CandidateNeuronJson> = HashMap::new();

    let c1 = make_candidate("input-1", "hidden-23", "ReLU", 1.0, 0.5, 0.10);
    let c2 = make_candidate("input-12", "hidden-3", "ReLU", 1.0, 0.5, 0.10);

    upsert_candidate(&mut map, c1);
    upsert_candidate(&mut map, c2);

    assert_eq!(
        map.len(),
        2,
        "Different UUIDs that could collide without separator should produce distinct keys"
    );
}

/// Stress test: many unique candidates should all be preserved.
#[test]
fn many_unique_candidates_all_preserved() {
    let mut map: HashMap<u64, CandidateNeuronJson> = HashMap::new();

    // 500 unique combinations: 50 sources × 10 targets
    for src in 0..50 {
        for tgt in 0..10 {
            let c = make_candidate(
                &format!("input-{src}"),
                &format!("hidden-{tgt}"),
                "ReLU",
                1.0,
                0.5,
                0.05,
            );
            upsert_candidate(&mut map, c);
        }
    }

    assert_eq!(
        map.len(),
        500,
        "All 500 unique source/target combinations should be preserved"
    );
}

/// Verify upsert correctly keeps the candidate with higher gain when
/// multiple candidates compete for the same key.
#[test]
fn upsert_keeps_best_gain_across_multiple_updates() {
    let mut map: HashMap<u64, CandidateNeuronJson> = HashMap::new();

    // Insert 10 candidates with same key but different gains
    for i in 0..10 {
        let gain = 0.01 * (i as f32 + 1.0);
        let c = make_candidate("input-0", "hidden-1", "ReLU", 1.0, 0.5, gain);
        upsert_candidate(&mut map, c);
    }

    assert_eq!(map.len(), 1, "All same-key candidates should merge to 1");

    let winner = map.values().next().unwrap();
    assert!(
        (winner.expected_creature_score_gain - 0.10).abs() < 1e-6,
        "Highest gain (0.10) should win: got {}",
        winner.expected_creature_score_gain
    );
}

/// Verify `compute_candidate_dedup_key` produces consistent results.
#[test]
fn dedup_key_is_deterministic() {
    let c1 = make_candidate("input-0", "hidden-1", "ReLU", 1.0, 0.5, 0.10);
    let c2 = make_candidate("input-0", "hidden-1", "ReLU", 1.0, 0.3, 0.15);

    let key1 = compute_candidate_dedup_key(&c1);
    let key2 = compute_candidate_dedup_key(&c2);

    // Same source/target/squash/signs → same key, regardless of gain or weight magnitude
    assert_eq!(
        key1, key2,
        "Same logical key should produce same hash regardless of gain/weight magnitude"
    );
}

/// Verify `compute_candidate_dedup_key` distinguishes all five key components.
#[test]
fn dedup_key_distinguishes_all_components() {
    let base = make_candidate("input-0", "hidden-1", "ReLU", 1.0, 0.5, 0.10);
    let diff_source = make_candidate("input-1", "hidden-1", "ReLU", 1.0, 0.5, 0.10);
    let diff_target = make_candidate("input-0", "hidden-2", "ReLU", 1.0, 0.5, 0.10);
    let diff_squash = make_candidate("input-0", "hidden-1", "TANH", 1.0, 0.5, 0.10);
    let diff_incoming = make_candidate("input-0", "hidden-1", "ReLU", -1.0, 0.5, 0.10);
    let diff_outgoing = make_candidate("input-0", "hidden-1", "ReLU", 1.0, -0.5, 0.10);

    let base_key = compute_candidate_dedup_key(&base);

    assert_ne!(
        base_key,
        compute_candidate_dedup_key(&diff_source),
        "Different source should produce different key"
    );
    assert_ne!(
        base_key,
        compute_candidate_dedup_key(&diff_target),
        "Different target should produce different key"
    );
    assert_ne!(
        base_key,
        compute_candidate_dedup_key(&diff_squash),
        "Different squash should produce different key"
    );
    assert_ne!(
        base_key,
        compute_candidate_dedup_key(&diff_incoming),
        "Different incoming sign should produce different key"
    );
    assert_ne!(
        base_key,
        compute_candidate_dedup_key(&diff_outgoing),
        "Different outgoing sign should produce different key"
    );
}
