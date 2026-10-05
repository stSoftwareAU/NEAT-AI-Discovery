//! Issue #2347: the symmetric-cancellation pair scan
//! (`detect_symmetric_cancellation_with_deadline`) is bounded by a fan-in
//! cap, a candidate ceiling and the analysis deadline, and reports the work
//! it actually did via deterministic work counters rather than wall-clock
//! timing (precedent: Issue #2320).

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for neural network computation (Issue #873)
mod common;

use std::time::{Duration, SystemTime};

use common::{make_creature, neuron, synapse};
use neat_ai_discovery::analysis::detection::weight_coherence::{
    MAX_FANIN_FOR_CANCELLATION_SCAN, MAX_SYMMETRIC_CANCELLATION_CANDIDATES, WeightCoherenceConfig,
    detect_symmetric_cancellation, detect_symmetric_cancellation_with_deadline,
};
use neat_ai_discovery::analysis::recommendation::epistatic::ScanTruncation;
use neat_ai_discovery::types::DiscoverRecord;

const RECORDS_PER_SOURCE: u32 = 25;

/// A single target neuron ("target-1") with `n` input sources, alternating
/// `+5.0` / `-5.0` weights so every cross-parity pair is an opposite-sign,
/// equal-magnitude pair (the only pairs symmetric cancellation can flag).
/// Every source shares the same activation pattern, so every cross-parity
/// pair is perfectly correlated.
fn fanin_fixture(
    n: usize,
) -> (
    neat_ai_discovery::CreatureJson,
    Vec<(String, Vec<DiscoverRecord>)>,
) {
    let mut neurons = vec![neuron("target-1", "output", "IDENTITY")];
    let mut synapses = Vec::with_capacity(n);
    let mut records = Vec::with_capacity(n);

    for i in 0..n {
        let uuid = format!("input-{i}");
        neurons.push(neuron(&uuid, "input", "IDENTITY"));
        let weight = if i % 2 == 0 { 5.0 } else { -5.0 };
        synapses.push(synapse(&uuid, "target-1", weight));

        let source_records: Vec<DiscoverRecord> = (0..RECORDS_PER_SOURCE)
            .map(|idx| {
                let activation = ((idx as f32 * 0.1).sin()).tanh();
                DiscoverRecord {
                    obs_index: idx,
                    neuron_uuid: uuid.clone(),
                    value: Some(activation),
                    activation,
                    errors: vec![0.1],
                }
            })
            .collect();
        records.push((uuid, source_records));
    }

    (make_creature(neurons, synapses), records)
}

fn elapsed() -> Option<SystemTime> {
    Some(SystemTime::UNIX_EPOCH)
}

fn far_future() -> Option<SystemTime> {
    Some(SystemTime::now() + Duration::from_secs(86_400))
}

// =============================================================================
// Test 1: Work grows no faster than the fan-in cap allows.
// =============================================================================

#[test]
fn symmetric_cancellation_work_grows_no_faster_than_the_cap_allows() {
    let config = WeightCoherenceConfig::default();

    let n = 128;
    let (creature_n, records_n) = fanin_fixture(n);
    let scan_n =
        detect_symmetric_cancellation_with_deadline(&creature_n, &records_n, &config, None, &None);

    let four_n = 4 * n;
    let (creature_4n, records_4n) = fanin_fixture(four_n);
    let scan_4n = detect_symmetric_cancellation_with_deadline(
        &creature_4n,
        &records_4n,
        &config,
        None,
        &None,
    );

    assert!(
        scan_n.pairs_correlated > 0,
        "the under-cap target should have been scanned"
    );
    assert!(
        four_n > MAX_FANIN_FOR_CANCELLATION_SCAN,
        "fixture must exceed the fan-in cap to exercise the skip"
    );
    assert_eq!(
        scan_4n.skipped_high_fanin_targets, 1,
        "the over-cap target must be skipped entirely, not partially scanned"
    );
    assert!(
        scan_4n.pairs_correlated <= 4 * scan_n.pairs_correlated,
        "work must not grow faster than the cap allows: {} vs 4x{}",
        scan_4n.pairs_correlated,
        scan_n.pairs_correlated
    );
}

// =============================================================================
// Test 2: Fan-in boundary — cap is scanned, cap + 1 is skipped.
// =============================================================================

#[test]
fn fanin_at_the_cap_is_scanned_fanin_over_the_cap_is_skipped() {
    let config = WeightCoherenceConfig::default();

    let (creature_at_cap, records_at_cap) = fanin_fixture(MAX_FANIN_FOR_CANCELLATION_SCAN);
    let scan_at_cap = detect_symmetric_cancellation_with_deadline(
        &creature_at_cap,
        &records_at_cap,
        &config,
        None,
        &None,
    );
    assert_eq!(
        scan_at_cap.skipped_high_fanin_targets, 0,
        "a target exactly at the cap must be scanned, not skipped"
    );
    assert!(
        scan_at_cap.pairs_correlated > 0,
        "a target exactly at the cap must do pair work"
    );

    let (creature_over_cap, records_over_cap) = fanin_fixture(MAX_FANIN_FOR_CANCELLATION_SCAN + 1);
    let scan_over_cap = detect_symmetric_cancellation_with_deadline(
        &creature_over_cap,
        &records_over_cap,
        &config,
        None,
        &None,
    );
    assert_eq!(
        scan_over_cap.skipped_high_fanin_targets, 1,
        "a target one over the cap must be skipped entirely"
    );
    assert_eq!(
        scan_over_cap.pairs_correlated, 0,
        "a skipped target must do no pair work at all"
    );
}

// =============================================================================
// Test 3: Each source's activation map is built once, not once per pair.
// =============================================================================

#[test]
fn activation_maps_are_cached_across_pairs_not_rebuilt_per_pair() {
    let config = WeightCoherenceConfig::default();
    let n = 20;
    let (creature, records) = fanin_fixture(n);

    let scan =
        detect_symmetric_cancellation_with_deadline(&creature, &records, &config, None, &None);

    assert!(
        scan.pairs_correlated > n,
        "fixture should correlate more pairs than there are sources: {}",
        scan.pairs_correlated
    );
    assert!(
        scan.activation_maps_built <= n,
        "each source's activation map must be built at most once: {} maps for {} sources",
        scan.activation_maps_built,
        n
    );
}

// =============================================================================
// Test 4: Deadline truncation.
// =============================================================================

#[test]
fn elapsed_deadline_stops_the_scan_before_any_pair_work() {
    let config = WeightCoherenceConfig::default();
    let (creature, records) = fanin_fixture(20);

    let scan =
        detect_symmetric_cancellation_with_deadline(&creature, &records, &config, None, &elapsed());

    assert_eq!(scan.truncation, Some(ScanTruncation::DeadlinePassed));
    assert_eq!(scan.pairs_correlated, 0);
    assert!(scan.candidates.is_empty());
}

#[test]
fn no_deadline_does_not_truncate_and_finds_candidates() {
    let config = WeightCoherenceConfig::default();
    let (creature, records) = fanin_fixture(20);

    let scan =
        detect_symmetric_cancellation_with_deadline(&creature, &records, &config, None, &None);

    assert_eq!(scan.truncation, None);
    assert!(!scan.candidates.is_empty());
}

#[test]
fn far_future_deadline_behaves_like_no_deadline() {
    let config = WeightCoherenceConfig::default();
    let (creature, records) = fanin_fixture(20);

    let scan = detect_symmetric_cancellation_with_deadline(
        &creature,
        &records,
        &config,
        None,
        &far_future(),
    );

    assert_eq!(scan.truncation, None);
    assert!(!scan.candidates.is_empty());
}

// =============================================================================
// Test 5: Candidate ceiling truncation.
// =============================================================================

#[test]
fn symmetric_cancellation_stops_at_the_candidate_ceiling_and_reports_it() {
    let config = WeightCoherenceConfig::default();
    // At the fan-in cap (so the target is scanned, not skipped), half the
    // sources are positive and half negative: 128 * 128 = 16,384 opposite-sign,
    // perfectly-correlated pairs, far exceeding the 1,024 candidate ceiling.
    let (creature, records) = fanin_fixture(MAX_FANIN_FOR_CANCELLATION_SCAN);

    let scan =
        detect_symmetric_cancellation_with_deadline(&creature, &records, &config, None, &None);

    assert_eq!(scan.truncation, Some(ScanTruncation::CandidateCeiling));
    assert_eq!(scan.candidates.len(), MAX_SYMMETRIC_CANCELLATION_CANDIDATES);
}

#[test]
fn candidate_ceiling_not_reported_when_the_scan_completes() {
    let config = WeightCoherenceConfig::default();
    let (creature, records) = fanin_fixture(20);

    let scan =
        detect_symmetric_cancellation_with_deadline(&creature, &records, &config, None, &None);

    assert_eq!(scan.truncation, None);
    assert!(scan.candidates.len() < MAX_SYMMETRIC_CANCELLATION_CANDIDATES);
}

// =============================================================================
// Test 6: Behaviour preservation — the legacy wrapper matches the bounded
// scan with no deadline.
// =============================================================================

#[test]
fn legacy_wrapper_matches_the_unbounded_with_deadline_scan() {
    let config = WeightCoherenceConfig::default();
    let (creature, records) = fanin_fixture(20);

    let legacy = detect_symmetric_cancellation(&creature, &records, &config, None);
    let scan =
        detect_symmetric_cancellation_with_deadline(&creature, &records, &config, None, &None);

    assert_eq!(legacy.len(), scan.candidates.len());
    let keys = |v: &[neat_ai_discovery::analysis::detection::weight_coherence::SymmetricCancellationCandidate]| {
        v.iter()
            .map(|c| {
                (
                    c.source1_neuron_uuid.clone(),
                    c.source2_neuron_uuid.clone(),
                    c.target_neuron_uuid.clone(),
                    c.correlation,
                    c.estimated_improvement,
                )
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(keys(&legacy), keys(&scan.candidates));
}
