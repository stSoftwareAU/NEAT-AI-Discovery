//! Issue #2348 (CWE-834, Excessive Iteration): `detect_co_adapted_neurons_with_deadline`
//! bounds the co-adaptation pairwise scan by both a shared analysis deadline
//! and a ceiling on the number of eligible neurons / emitted candidates, and
//! reports the reason a scan stopped early via `ScanTruncation`.
//!
//! Deterministic: an elapsed deadline is `UNIX_EPOCH`, a live one is a day
//! ahead — no sleeps and no timing thresholds.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for synthetic activation waveforms (Issue #873)

mod common;

use std::time::{Duration, SystemTime};

use common::{hidden, make_creature, neuron, output, record, synapse};
use neat_ai_discovery::CreatureJson;
use neat_ai_discovery::analysis::detection::co_adaptation::{
    MAX_CO_ADAPTATION_ELIGIBLE_NEURONS, MAX_CO_ADAPTED_PAIR_CANDIDATES, detect_co_adapted_neurons,
    detect_co_adapted_neurons_with_deadline,
};
use neat_ai_discovery::analysis::recommendation::epistatic::ScanTruncation;
use neat_ai_discovery::types::DiscoverRecord;

fn elapsed() -> Option<SystemTime> {
    Some(SystemTime::UNIX_EPOCH)
}

fn far_future() -> Option<SystemTime> {
    Some(SystemTime::now() + Duration::from_secs(86_400))
}

fn correlated_records(uuid: &str, count: u32) -> Vec<DiscoverRecord> {
    (0..count)
        .map(|i| record(uuid, i, (i as f32 * 0.1).sin(), None))
        .collect()
}

/// Per-neuron-distinct, pairwise-uncorrelated activations: neuron `k` uses a
/// distinct frequency/phase combination so no two neurons correlate highly.
fn uncorrelated_records(uuid: &str, k: usize, count: u32) -> Vec<DiscoverRecord> {
    let freq = 1.3 + (k as f32) * 0.37;
    let phase = (k as f32) * 0.91;
    (0..count)
        .map(|i| {
            let activation = (i as f32 * freq + phase).sin() * 0.5 + (i as f32 * freq * 1.7).cos();
            record(uuid, i, activation, None)
        })
        .collect()
}

fn build_creature_with_hidden(hidden_count: usize) -> CreatureJson {
    let mut neurons = vec![neuron("in-0", "input", "IDENTITY")];
    let mut synapses = Vec::new();
    for h in 0..hidden_count {
        let uuid = format!("h{h}");
        neurons.push(hidden(&uuid, "TANH"));
        synapses.push(synapse("in-0", &uuid, 1.0));
    }
    neurons.push(output("out-0", "TANH"));
    for h in 0..hidden_count {
        let uuid = format!("h{h}");
        synapses.push(synapse(&uuid, "out-0", 0.5));
    }
    make_creature(neurons, synapses)
}

/// An elapsed deadline must stop the scan before any pair is evaluated.
#[test]
fn elapsed_deadline_stops_co_adaptation_scan_before_any_pair() {
    const COUNT: u32 = 20;
    let hidden_count = 4;
    let creature = build_creature_with_hidden(hidden_count);
    let records: Vec<(String, Vec<DiscoverRecord>)> = (0..hidden_count)
        .map(|h| {
            let uuid = format!("h{h}");
            (uuid.clone(), correlated_records(&uuid, COUNT))
        })
        .collect();

    let scan = detect_co_adapted_neurons_with_deadline(&creature, &records, &elapsed());

    assert!(
        scan.candidates.is_empty(),
        "an elapsed deadline must stop the scan before any candidate is emitted, got {}",
        scan.candidates.len()
    );
    assert_eq!(
        scan.pairs_evaluated, 0,
        "an elapsed deadline must stop the scan before the first pair"
    );
    assert!(
        matches!(scan.truncation, Some(ScanTruncation::DeadlinePassed)),
        "expected DeadlinePassed, got {:?}",
        scan.truncation
    );
}

/// A far-future deadline must not truncate the scan: it yields exactly the
/// same candidates as the legacy unbounded `detect_co_adapted_neurons`.
#[test]
fn future_deadline_yields_every_pair_the_unbounded_scan_yields() {
    const COUNT: u32 = 20;
    // Mix: h0/h1 correlated, h2/h3 uncorrelated with everything else.
    let hidden_count = 4;
    let creature = build_creature_with_hidden(hidden_count);
    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        ("h0".to_string(), correlated_records("h0", COUNT)),
        ("h1".to_string(), correlated_records("h1", COUNT)),
        ("h2".to_string(), uncorrelated_records("h2", 2, COUNT)),
        ("h3".to_string(), uncorrelated_records("h3", 5, COUNT)),
    ];

    let scan = detect_co_adapted_neurons_with_deadline(&creature, &records, &far_future());
    let legacy = detect_co_adapted_neurons(&creature, &records);

    assert_eq!(scan.truncation, None);
    assert_eq!(scan.pairs_evaluated, hidden_count * (hidden_count - 1) / 2);
    assert!(
        !scan.candidates.is_empty(),
        "expected at least one co-adapted candidate (h0/h1) for the equality check to be non-vacuous"
    );

    let mut scan_keys: Vec<(String, String)> = scan
        .candidates
        .iter()
        .map(|c| (c.neuron_a_uuid.clone(), c.neuron_b_uuid.clone()))
        .collect();
    let mut legacy_keys: Vec<(String, String)> = legacy
        .iter()
        .map(|c| (c.neuron_a_uuid.clone(), c.neuron_b_uuid.clone()))
        .collect();
    scan_keys.sort();
    legacy_keys.sort();
    assert_eq!(scan_keys, legacy_keys);
}

/// 30 identical hidden neurons produce 435 co-adapted pairs, which exceeds
/// the candidate ceiling — the scan must stop and report `CandidateCeiling`.
#[test]
fn identical_neurons_stop_at_the_candidate_ceiling() {
    const COUNT: u32 = 20;
    let hidden_count = 30_usize;
    assert!(hidden_count * (hidden_count - 1) / 2 > MAX_CO_ADAPTED_PAIR_CANDIDATES);

    let creature = build_creature_with_hidden(hidden_count);
    let records: Vec<(String, Vec<DiscoverRecord>)> = (0..hidden_count)
        .map(|h| {
            let uuid = format!("h{h}");
            (uuid.clone(), correlated_records(&uuid, COUNT))
        })
        .collect();

    let scan = detect_co_adapted_neurons_with_deadline(&creature, &records, &far_future());

    assert_eq!(scan.candidates.len(), MAX_CO_ADAPTED_PAIR_CANDIDATES);
    assert!(matches!(
        scan.truncation,
        Some(ScanTruncation::CandidateCeiling)
    ));
}

/// Eligible hidden neurons beyond `MAX_CO_ADAPTATION_ELIGIBLE_NEURONS` must
/// be skipped (and the skip count recorded), rather than silently included
/// in an unbounded scan. Activations are pairwise-uncorrelated so the
/// candidate ceiling never fires, isolating the eligible-neuron ceiling.
#[test]
fn eligible_neurons_beyond_the_cap_are_skipped_and_recorded() {
    const COUNT: u32 = 20;
    let extra = 40_usize;
    let hidden_count = MAX_CO_ADAPTATION_ELIGIBLE_NEURONS + extra;

    let creature = build_creature_with_hidden(hidden_count);
    let records: Vec<(String, Vec<DiscoverRecord>)> = (0..hidden_count)
        .map(|h| {
            let uuid = format!("h{h}");
            (uuid.clone(), uncorrelated_records(&uuid, h, COUNT))
        })
        .collect();

    let scan = detect_co_adapted_neurons_with_deadline(&creature, &records, &None);

    assert_eq!(scan.eligible_skipped, extra);
    assert_eq!(
        scan.pairs_evaluated,
        MAX_CO_ADAPTATION_ELIGIBLE_NEURONS * (MAX_CO_ADAPTATION_ELIGIBLE_NEURONS - 1) / 2
    );
    assert!(
        scan.candidates.len() < MAX_CO_ADAPTED_PAIR_CANDIDATES,
        "uncorrelated activations must not trip the candidate ceiling, got {} candidates",
        scan.candidates.len()
    );
    assert_eq!(
        scan.truncation, None,
        "the eligible-neuron ceiling alone is not reported as a scan truncation"
    );
}

/// Under the eligible-neuron cap, no neurons are skipped.
#[test]
fn under_cap_scan_records_no_skip() {
    const COUNT: u32 = 20;
    let hidden_count = 10_usize;
    let creature = build_creature_with_hidden(hidden_count);
    let records: Vec<(String, Vec<DiscoverRecord>)> = (0..hidden_count)
        .map(|h| {
            let uuid = format!("h{h}");
            (uuid.clone(), uncorrelated_records(&uuid, h, COUNT))
        })
        .collect();

    let scan = detect_co_adapted_neurons_with_deadline(&creature, &records, &None);

    assert_eq!(scan.eligible_skipped, 0);
}
