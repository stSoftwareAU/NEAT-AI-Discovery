//! In-crate deadline tests for the dead-neuron scan (Issue #2370).
//!
//! These use the `deadline_override` test seam (`#[cfg(test)]`-only, so they
//! cannot live in the `tests/` integration crate) to script `deadline_passed`
//! to return `false` for the first N checks and then `true`, proving each
//! check (the hidden-neuron scan and the reachability walk) is the one that
//! stops a scan that started before the deadline.
//!
//! `#[serial]` keeps these tests from interleaving with any other test that
//! installs a deadline override or flips the global cancellation flag
//! `deadline_passed` also consults.

use super::*;
use crate::analysis::utils::deadline_override::DeadlineOverrideGuard;
use crate::types::DiscoverRecord;
use crate::{CreatureJson, NeuronJson, SynapseJson};
use serial_test::serial;
use std::time::Duration;

const SAMPLES: u32 = 30;

fn records_for(uuid: &str) -> (String, Vec<DiscoverRecord>) {
    let recs = (0..SAMPLES)
        .map(|idx| DiscoverRecord {
            obs_index: idx,
            neuron_uuid: uuid.to_string(),
            value: Some(0.0),
            activation: 0.0,
            errors: vec![0.0],
        })
        .collect();
    (uuid.to_string(), recs)
}

/// `input-0 -> hidden-0 -> hidden-1 -> hidden-2 -> output-0`. Every hidden
/// neuron is dead (zero activation, zero variance) with `SAMPLES` samples.
fn dead_chain_creature() -> (CreatureJson, Vec<(String, Vec<DiscoverRecord>)>) {
    let neuron = |uuid: &str, neuron_type: &str| NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
        squash: "LOGISTIC".to_string(),
        bias: 0.0,
    };
    let synapse = |from: &str, to: &str| SynapseJson {
        from_uuid: from.to_string(),
        to_uuid: to.to_string(),
        weight: 0.5,
        synapse_type: None,
    };

    let neurons = vec![
        neuron("input-0", "input"),
        neuron("hidden-0", "hidden"),
        neuron("hidden-1", "hidden"),
        neuron("hidden-2", "hidden"),
        neuron("output-0", "output"),
    ];
    let synapses = vec![
        synapse("input-0", "hidden-0"),
        synapse("hidden-0", "hidden-1"),
        synapse("hidden-1", "hidden-2"),
        synapse("hidden-2", "output-0"),
    ];
    let records = ["hidden-0", "hidden-1", "hidden-2"]
        .into_iter()
        .map(records_for)
        .collect();

    let creature = CreatureJson {
        neurons,
        synapses,
        input: 1,
        output: 1,
    };
    (creature, records)
}

/// A deadline value is required by the signature, but every call these tests
/// drive is intercepted by the override queue — the real value is only a
/// fallback that must never be consulted while the queue still has entries.
fn unused_live_deadline() -> Option<SystemTime> {
    Some(SystemTime::now() + Duration::from_secs(86_400))
}

/// A deadline reported as passed by the hidden-neuron scan (the first check
/// before `hidden-1`) stops the scan before any reachability work, returning
/// no candidates.
#[test]
#[serial]
fn phase_one_deadline_check_truncates_and_returns_no_candidates() {
    let (creature, records) = dead_chain_creature();

    // [false /*check before hidden-0: proceeds*/,
    //  true  /*check before hidden-1: deadline now reported passed*/]
    let _guard = DeadlineOverrideGuard::with_sequence(vec![false, true]);

    let mut steps = 0;
    let scan =
        detect_dead_neurons_observed(&creature, &records, None, &unused_live_deadline(), || {
            steps += 1;
        });

    assert_eq!(
        scan.truncation,
        Some(ScanTruncation::DeadlinePassed),
        "the hidden-neuron deadline check must report DeadlinePassed"
    );
    assert!(
        scan.candidates.is_empty(),
        "a partial hidden-neuron scan cannot yield outputs, so no candidates"
    );
    assert_eq!(
        steps, 0,
        "the scan must stop before the reachability pass takes any step"
    );
}

/// A deadline reported as passed inside the reachability walk stops the scan
/// after the hidden-neuron scan has completed, returning no candidates.
#[test]
#[serial]
fn reachability_deadline_check_stops_mid_scan() {
    let (creature, records) = dead_chain_creature();

    // Phase 1 makes one check per hidden neuron (3 calls, all false).
    // Phase 2 then makes one check per walk iteration:
    //  [false /*iteration 1: examine the first root's single out-edge*/,
    //   true  /*iteration 2: deadline now reported passed*/]
    // Each hidden neuron has exactly one out-edge, so whichever root the
    // (randomly ordered) hidden-neuron set yields first, iteration 1 examines
    // one edge and iteration 2 aborts. The first root cannot complete within
    // two iterations, so no candidate is finished.
    let _guard = DeadlineOverrideGuard::with_sequence(vec![false, false, false, false, true]);

    let mut steps = 0;
    let scan =
        detect_dead_neurons_observed(&creature, &records, None, &unused_live_deadline(), || {
            steps += 1;
        });

    assert_eq!(
        scan.truncation,
        Some(ScanTruncation::DeadlinePassed),
        "the reachability-walk deadline check must report DeadlinePassed"
    );
    assert_eq!(
        scan.candidates.len(),
        0,
        "the first root is not complete after one examined edge, so no candidate"
    );
    assert_eq!(
        steps, 1,
        "exactly one edge is examined before the second walk check aborts"
    );
}
