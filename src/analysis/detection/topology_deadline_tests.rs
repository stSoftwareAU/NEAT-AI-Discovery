//! Deadline and cancellation tests for the topology scan (Issue #2369).
//!
//! These in-crate tests use the `deadline_override` test seam (`#[cfg(test)]`
//! only) to script `deadline_passed` so each check point is the one that stops
//! the scan. `#[serial]` keeps them from interleaving with other tests that
//! install overrides or flip the global cancellation flag.

use super::*;
use crate::analysis::utils::deadline_override::DeadlineOverrideGuard;
use crate::{NeuronJson, SynapseJson};
use serial_test::serial;
use std::time::Duration;

/// Matches `MIN_SAMPLES_FOR_TOPOLOGY` so every hidden neuron qualifies.
const SAMPLES: u32 = 20;

fn neuron(uuid: &str, neuron_type: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
        squash: "TANH".to_string(),
        bias: 0.0,
    }
}

fn synapse(from: &str, to: &str) -> SynapseJson {
    SynapseJson {
        from_uuid: from.to_string(),
        to_uuid: to.to_string(),
        weight: 0.5,
        synapse_type: None,
    }
}

/// Non-zero errors so every candidate has a positive estimated gain.
fn records_for(uuid: &str) -> (String, Vec<DiscoverRecord>) {
    let recs = (0..SAMPLES)
        .map(|obs_index| DiscoverRecord {
            obs_index,
            neuron_uuid: uuid.to_string(),
            value: Some(0.0),
            activation: 0.2,
            errors: vec![0.2],
        })
        .collect();
    (uuid.to_string(), recs)
}

/// `k` qualified hidden neurons, each fed from `in-0` and each feeding a shared
/// tail `tail-1 -> tail-2 -> tail-3 -> out-0`. Every `hidden-i` is four hops from
/// an output, beyond the three-hop limit, so each is a `long_path` candidate. The
/// tail neurons have no records and are not qualified.
fn long_path_creature(k: usize) -> (CreatureJson, Vec<(String, Vec<DiscoverRecord>)>) {
    let mut neurons = vec![neuron("in-0", "input"), neuron("out-0", "output")];
    let mut synapses = Vec::new();
    let mut records = Vec::new();
    for i in 0..k {
        let uuid = format!("hidden-{i}");
        neurons.push(neuron(&uuid, "hidden"));
        synapses.push(synapse("in-0", &uuid));
        synapses.push(synapse(&uuid, "tail-1"));
        records.push(records_for(&uuid));
    }
    for tail in ["tail-1", "tail-2", "tail-3"] {
        neurons.push(neuron(tail, "hidden"));
    }
    synapses.push(synapse("tail-1", "tail-2"));
    synapses.push(synapse("tail-2", "tail-3"));
    synapses.push(synapse("tail-3", "out-0"));
    (
        CreatureJson {
            neurons,
            synapses,
            input: 1,
            output: 1,
        },
        records,
    )
}

/// Three qualified hidden neurons that each feed `out-0` directly, so none is a
/// long-path candidate. `hidden-a` has fan-in 3 and `hidden-b` and `hidden-c`
/// have fan-in 1, a ratio of 3.0, so `hidden-b` and `hidden-c` are starved.
fn imbalance_creature() -> (CreatureJson, Vec<(String, Vec<DiscoverRecord>)>) {
    let neurons = vec![
        neuron("in-0", "input"),
        neuron("in-1", "input"),
        neuron("in-2", "input"),
        neuron("out-0", "output"),
        neuron("hidden-a", "hidden"),
        neuron("hidden-b", "hidden"),
        neuron("hidden-c", "hidden"),
    ];
    let synapses = vec![
        synapse("in-0", "hidden-a"),
        synapse("in-1", "hidden-a"),
        synapse("in-2", "hidden-a"),
        synapse("in-0", "hidden-b"),
        synapse("in-1", "hidden-c"),
        synapse("hidden-a", "out-0"),
        synapse("hidden-b", "out-0"),
        synapse("hidden-c", "out-0"),
    ];
    let records = ["hidden-a", "hidden-b", "hidden-c"]
        .into_iter()
        .map(records_for)
        .collect();
    (
        CreatureJson {
            neurons,
            synapses,
            input: 3,
            output: 1,
        },
        records,
    )
}

fn uuids(candidates: &[TopologyCandidate]) -> Vec<(String, String)> {
    candidates
        .iter()
        .map(|c| (c.neuron_uuid.clone(), c.issue_type.clone()))
        .collect()
}

/// An elapsed deadline stops the scan before any work, returning no candidates
/// and reporting truncation.
#[test]
#[serial]
fn elapsed_deadline_up_front_returns_no_candidates_and_reports_truncation() {
    {
        let (creature, records) = long_path_creature(2);
        let _guard = DeadlineOverrideGuard::with_sequence(vec![true]);
        let scan = detect_topology_issues_with_deadline(&creature, &records, None, &None);
        assert!(
            scan.candidates.is_empty(),
            "an elapsed deadline must yield no candidates"
        );
        assert_eq!(scan.truncation, Some(ScanTruncation::DeadlinePassed));
    }

    // With no qualified hidden neurons the loops would never consult the
    // deadline, so only the up-front check can report truncation here.
    let (creature, records) = long_path_creature(0);
    let _guard = DeadlineOverrideGuard::with_sequence(vec![true]);
    let scan = detect_topology_issues_with_deadline(&creature, &records, None, &None);
    assert_eq!(
        scan.truncation,
        Some(ScanTruncation::DeadlinePassed),
        "the up-front check must report truncation before any work is done"
    );
}

/// A deadline that fires at the second qualified neuron keeps the candidate
/// already found for the first and skips the imbalance detection.
#[test]
#[serial]
fn long_path_loop_deadline_check_keeps_partial_candidates() {
    let (creature, records) = long_path_creature(2);
    let control = detect_topology_issues(&creature, &records, None);
    assert_eq!(control.len(), 2, "fixture must yield two long-path candidates");

    // [up-front, hidden-0 check, hidden-1 check -> true]
    let _guard = DeadlineOverrideGuard::with_sequence(vec![false, false, true]);
    let scan = detect_topology_issues_with_deadline(&creature, &records, None, &None);
    assert_eq!(scan.truncation, Some(ScanTruncation::DeadlinePassed));
    assert_eq!(
        scan.candidates.len(),
        1,
        "only the candidate found before the deadline may be returned"
    );
    assert_eq!(scan.candidates[0].issue_type, "long_path");
}

/// A deadline that fires at the second starved neuron keeps the first starved
/// candidate and reports truncation.
#[test]
#[serial]
fn imbalance_loop_deadline_check_keeps_partial_candidates() {
    let (creature, records) = imbalance_creature();
    let unbounded = detect_topology_issues(&creature, &records, None);
    assert_eq!(
        unbounded.len(),
        2,
        "fixture must flag both starved neurons, got: {unbounded:?}"
    );
    assert!(
        unbounded
            .iter()
            .all(|c| c.issue_type == "connectivity_imbalance")
    );

    // [up-front, one check per qualified neuron in the long-path loop (3),
    //  starved #1 proceeds, starved #2 -> true]
    let _guard =
        DeadlineOverrideGuard::with_sequence(vec![false, false, false, false, false, true]);
    let scan = detect_topology_issues_with_deadline(&creature, &records, None, &None);
    assert_eq!(scan.truncation, Some(ScanTruncation::DeadlinePassed));
    assert_eq!(
        scan.candidates.len(),
        1,
        "only the starved neuron processed before the deadline may be returned"
    );
    assert!(scan.candidates.len() < unbounded.len());
    assert_eq!(scan.candidates[0].issue_type, "connectivity_imbalance");
}

/// A far-future deadline changes nothing: the scan reports no truncation and
/// matches the unbounded wrapper exactly.
#[test]
#[serial]
fn far_future_deadline_matches_unbounded_scan_without_truncation() {
    let (creature, records) = long_path_creature(3);
    let unbounded = detect_topology_issues(&creature, &records, None);
    assert_eq!(unbounded.len(), 3, "fixture must yield three candidates");

    let far = Some(SystemTime::now() + Duration::from_secs(86_400));
    let scan = detect_topology_issues_with_deadline(&creature, &records, None, &far);
    assert_eq!(scan.truncation, None);
    assert_eq!(uuids(&scan.candidates), uuids(&unbounded));
}
