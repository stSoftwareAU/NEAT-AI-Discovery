//! Issue #1781: failure-cache entries must expire, and a coarse (target-agnostic)
//! entry must not suppress a whole change type indefinitely.
//!
//! Behaviour under test:
//!
//! 1. `FailureCacheEntry` carries an optional `ageEpochs` field (host-supplied
//!    age in discovery passes) so Rust can expire stale entries.
//! 2. An entry at or beyond `FAILURE_CACHE_MAX_AGE_EPOCHS` suppresses nothing.
//! 3. A wildcard entry (no `targetUuid`) only acts as a wildcard against a
//!    *specific* candidate while it is demonstrably fresh — it still matches an
//!    equally target-agnostic candidate exactly.

use neat_ai_discovery::analysis::discovery_mode::DEFAULT_LOW_SUCCESS_RATE_THRESHOLD;
use neat_ai_discovery::analysis::failure_cache_handshake::{
    CHANGE_TYPE_COORDINATED_STRUCTURAL, CandidateIdentity, FAILURE_CACHE_MAX_AGE_EPOCHS,
    WILDCARD_FAILURE_CACHE_MAX_AGE_EPOCHS, count_suppressed, evaluate,
};
use neat_ai_discovery::analysis::novelty_escalation::DEFAULT_SUPPRESSION_RATIO_THRESHOLD;
use neat_ai_discovery::analysis::scoring::calibration_correction::FailureCacheEntry;

/// Parse a single failure-cache entry from the wire JSON shape NEAT-AI emits.
fn parse_entry(json: &str) -> FailureCacheEntry {
    serde_json::from_str(json).expect("failure-cache entry should deserialise")
}

/// A coordinated-structural entry with an explicit age and optional target.
fn entry(age_epochs: Option<u32>, target_uuid: Option<&str>) -> FailureCacheEntry {
    let age = age_epochs.map_or("null".to_string(), |a| a.to_string());
    let uuid = target_uuid.map_or("null".to_string(), |u| format!("\"{u}\""));
    parse_entry(&format!(
        r#"{{
            "changeType": "coordinated-structural",
            "expectedErrorReduction": 0.1,
            "actualErrorReduction": -0.01,
            "targetUuid": {uuid},
            "ageEpochs": {age}
        }}"#
    ))
}

fn candidate(target_uuid: Option<&str>) -> CandidateIdentity {
    CandidateIdentity::new(
        CHANGE_TYPE_COORDINATED_STRUCTURAL,
        target_uuid.map(str::to_string),
        None,
    )
}

#[test]
fn age_epochs_is_parsed_from_the_wire() {
    let parsed = entry(Some(7), Some("n1"));
    assert_eq!(parsed.age_epochs, Some(7));
}

#[test]
fn legacy_entry_without_age_deserialises_to_none() {
    let parsed = parse_entry(
        r#"{
            "changeType": "coordinated-structural",
            "expectedErrorReduction": 0.1,
            "actualErrorReduction": -0.01
        }"#,
    );
    assert_eq!(parsed.age_epochs, None);
}

#[test]
fn fresh_specific_entry_still_suppresses_its_target() {
    let cache = vec![entry(Some(0), Some("n1"))];
    assert_eq!(count_suppressed(&[candidate(Some("n1"))], &cache), 1);
}

#[test]
fn expired_specific_entry_suppresses_nothing() {
    let cache = vec![entry(Some(FAILURE_CACHE_MAX_AGE_EPOCHS), Some("n1"))];
    assert_eq!(count_suppressed(&[candidate(Some("n1"))], &cache), 0);
}

#[test]
fn specific_entry_one_epoch_before_expiry_still_suppresses() {
    let cache = vec![entry(Some(FAILURE_CACHE_MAX_AGE_EPOCHS - 1), Some("n1"))];
    assert_eq!(count_suppressed(&[candidate(Some("n1"))], &cache), 1);
}

#[test]
fn fresh_wildcard_entry_suppresses_specific_candidates() {
    let cache = vec![entry(Some(0), None)];
    let cands = vec![candidate(Some("a")), candidate(Some("b"))];
    assert_eq!(count_suppressed(&cands, &cache), 2);
}

#[test]
fn aged_wildcard_entry_stops_suppressing_specific_candidates() {
    // Past the wildcard TTL the coarse entry may no longer stand in for every
    // target of its change type — this is the "suppresses a whole change type
    // forever" fault from Issue #1781.
    let cache = vec![entry(Some(WILDCARD_FAILURE_CACHE_MAX_AGE_EPOCHS), None)];
    assert_eq!(count_suppressed(&[candidate(Some("a"))], &cache), 0);
    // ... but an exactly-matching target-agnostic candidate is still suppressed
    // until the entry expires outright.
    assert_eq!(count_suppressed(&[candidate(None)], &cache), 1);
}

#[test]
fn unknown_age_wildcard_entry_does_not_suppress_specific_candidates() {
    // Legacy hosts that supply no age cannot demonstrate freshness, so their
    // coarse entries no longer act as wildcards against specific candidates.
    let cache = vec![entry(None, None)];
    assert_eq!(count_suppressed(&[candidate(Some("a"))], &cache), 0);
    assert_eq!(count_suppressed(&[candidate(None)], &cache), 1);
}

#[test]
fn expired_wildcard_entry_suppresses_nothing_at_all() {
    let cache = vec![entry(Some(FAILURE_CACHE_MAX_AGE_EPOCHS + 5), None)];
    assert_eq!(count_suppressed(&[candidate(None)], &cache), 0);
    assert_eq!(count_suppressed(&[candidate(Some("a"))], &cache), 0);
}

#[test]
fn escalation_is_not_engaged_by_an_expired_cache() {
    // A plateaued creature whose only cache entry has expired is no longer
    // reported as suppressed, so the novelty bypass stays off.
    let cache = vec![entry(Some(FAILURE_CACHE_MAX_AGE_EPOCHS), None)];
    let cands = vec![candidate(Some("a")), candidate(Some("b"))];
    let out = evaluate(
        &cands,
        &cache,
        0.05,
        DEFAULT_LOW_SUCCESS_RATE_THRESHOLD,
        DEFAULT_SUPPRESSION_RATIO_THRESHOLD,
    );
    assert_eq!(out.failure_cache_suppressed_count, 0);
    assert!(!out.novelty_escalation_active);
}

#[test]
fn wildcard_ttl_is_shorter_than_the_outright_expiry() {
    const {
        assert!(WILDCARD_FAILURE_CACHE_MAX_AGE_EPOCHS < FAILURE_CACHE_MAX_AGE_EPOCHS);
    }
}
