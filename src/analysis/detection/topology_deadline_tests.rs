//! Issue #2369: in-crate deadline checks for `detect_topology_issues_with_deadline`.
//!
//! Both per-neuron scans check `deadline_passed` at the start of every
//! iteration. These tests use the `deadline_override` test seam
//! (`#[cfg(test)]`-only, so they cannot live in the `tests/` integration crate)
//! to script `deadline_passed`, proving that each check is the one that stops
//! its scan.
//!
//! `#[serial]` keeps these tests from interleaving with any other test that
//! installs a deadline override or flips the global cancellation flag
//! `deadline_passed` also consults.

use super::*;
use crate::analysis::utils::deadline_override::DeadlineOverrideGuard;
use crate::types::DiscoverRecord;
use crate::{CreatureJson, NeuronJson, SynapseJson};
use serial_test::serial;
use std::time::{Duration, SystemTime};

const SAMPLES: u32 = 30;

// The fixtures must qualify their hidden neurons for topology analysis.
const _: () = assert!(SAMPLES as usize >= MIN_SAMPLES_FOR_TOPOLOGY);

type Records = Vec<(String, Vec<DiscoverRecord>)>;

fn neuron(uuid: &str, kind: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: kind.to_string(),
        squash: "TANH".to_string(),
        bias: 0.0,
    }
}

fn syn(from: &str, to: &str) -> SynapseJson {
    SynapseJson {
        from_uuid: from.to_string(),
        to_uuid: to.to_string(),
        weight: 0.5,
        synapse_type: None,
    }
}

/// Qualified records (`SAMPLES` observations, non-zero error) for one neuron.
fn qualified_records(uuid: &str) -> (String, Vec<DiscoverRecord>) {
    let recs = (0..SAMPLES)
        .map(|idx| DiscoverRecord {
            obs_index: idx,
            neuron_uuid: uuid.to_string(),
            value: None,
            activation: 0.5,
            errors: vec![0.5],
        })
        .collect();
    (uuid.to_string(), recs)
}

/// A deadline value is required by the signature, but every call these tests
/// drive is intercepted by the override queue. The real value is only a
/// fallback that must never be consulted while the queue still has entries.
fn unused_live_deadline() -> Option<SystemTime> {
    Some(SystemTime::now() + Duration::from_secs(86_400))
}

fn uuids_of(candidates: &[TopologyCandidate], issue_type: &str) -> Vec<String> {
    let mut uuids: Vec<String> = candidates
        .iter()
        .filter(|c| c.issue_type == issue_type)
        .map(|c| c.neuron_uuid.clone())
        .collect();
    uuids.sort();
    uuids
}

/// `input-0 -> h0 -> h1 -> h2 -> h3 -> h4 -> output-0`. `h0` sits 5 hops from the
/// output and `h1` sits 4, so both are long-path. Only `h0` and `h1` are
/// qualified, so loop 1 visits exactly two neurons and loop 2 is never reached
/// unless loop 1 completes.
fn long_path_only_fixture() -> (CreatureJson, Records) {
    let mut neurons = vec![neuron("input-0", "input")];
    neurons.extend(["h0", "h1", "h2", "h3", "h4"].map(|u| neuron(u, "hidden")));
    neurons.push(neuron("output-0", "output"));
    let synapses = vec![
        syn("input-0", "h0"),
        syn("h0", "h1"),
        syn("h1", "h2"),
        syn("h2", "h3"),
        syn("h3", "h4"),
        syn("h4", "output-0"),
    ];
    let records = vec![qualified_records("h0"), qualified_records("h1")];
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

/// Long-path chain `input-0 -> h0 -> ... -> h4 -> output-0`, plus a `rich`
/// neuron fed by all six inputs and two starved neurons fed only by `in-0` and
/// `in-1`. Every hidden neuron is qualified, so both loops run and yield
/// candidates of both kinds.
fn both_kinds_fixture() -> (CreatureJson, Records) {
    let inputs: Vec<String> = (0..6).map(|i| format!("in-{i}")).collect();
    let hidden = [
        "h0",
        "h1",
        "h2",
        "h3",
        "h4",
        "rich",
        "starved-0",
        "starved-1",
    ];

    let mut neurons: Vec<NeuronJson> = inputs.iter().map(|u| neuron(u, "input")).collect();
    neurons.extend(hidden.iter().map(|u| neuron(u, "hidden")));
    neurons.push(neuron("output-0", "output"));

    let mut synapses = vec![
        syn("in-0", "h0"),
        syn("h0", "h1"),
        syn("h1", "h2"),
        syn("h2", "h3"),
        syn("h3", "h4"),
        syn("h4", "output-0"),
    ];
    for input in &inputs {
        synapses.push(syn(input, "rich"));
    }
    synapses.push(syn("rich", "output-0"));
    for starved in ["starved-0", "starved-1"] {
        synapses.push(syn("in-0", starved));
        synapses.push(syn("in-1", starved));
        synapses.push(syn(starved, "output-0"));
    }

    let records: Records = hidden.iter().map(|u| qualified_records(u)).collect();
    (
        CreatureJson {
            neurons,
            synapses,
            input: inputs.len(),
            output: 1,
        },
        records,
    )
}

/// The loop-1 check (first statement of each long-path iteration) must stop the
/// long-path scan after the first candidate and skip connectivity-imbalance
/// detection entirely.
#[test]
#[serial]
fn loop_one_deadline_check_stops_long_path_scan_and_skips_imbalance() {
    let (creature, records) = long_path_only_fixture();

    let control = detect_topology_issues_with_deadline(&creature, &records, None, &None);
    assert_eq!(
        uuids_of(&control, "long_path").len(),
        2,
        "fixture must yield two long-path candidates with no deadline override, got: {control:?}"
    );

    // [false /*first qualified neuron proceeds*/, true /*second — deadline now passed*/]
    let _guard = DeadlineOverrideGuard::with_sequence(vec![false, true]);
    let scan =
        detect_topology_issues_with_deadline(&creature, &records, None, &unused_live_deadline());

    assert_eq!(
        uuids_of(&scan, "long_path").len(),
        1,
        "the loop-1 check must stop the scan after one long-path candidate, got: {scan:?}"
    );
    assert!(
        uuids_of(&scan, "connectivity_imbalance").is_empty(),
        "expiry in loop 1 must skip connectivity-imbalance detection, got: {scan:?}"
    );
}

/// The loop-2 check (first statement of each imbalance iteration) must stop the
/// imbalance scan while leaving the completed long-path results untouched.
#[test]
#[serial]
fn loop_two_deadline_check_stops_imbalance_scan_after_long_path_scan_completes() {
    let (creature, records) = both_kinds_fixture();

    let control = detect_topology_issues_with_deadline(&creature, &records, None, &None);
    let control_long = uuids_of(&control, "long_path");
    assert!(
        !control_long.is_empty(),
        "fixture must yield long-path candidates, got: {control:?}"
    );
    assert!(
        !uuids_of(&control, "connectivity_imbalance").is_empty(),
        "positive control: fixture must yield connectivity-imbalance candidates, got: {control:?}"
    );

    // One loop-1 check per qualified hidden neuron, then the first loop-2 check
    // reports the deadline as passed.
    let qualified_hidden = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "hidden")
        .filter(|n| {
            records
                .iter()
                .any(|(uuid, recs)| *uuid == n.uuid && recs.len() >= MIN_SAMPLES_FOR_TOPOLOGY)
        })
        .count();
    let mut sequence = vec![false; qualified_hidden];
    sequence.push(true);
    let _guard = DeadlineOverrideGuard::with_sequence(sequence);

    let scan =
        detect_topology_issues_with_deadline(&creature, &records, None, &unused_live_deadline());

    assert_eq!(
        uuids_of(&scan, "long_path"),
        control_long,
        "the completed long-path scan must be kept unchanged"
    );
    assert!(
        uuids_of(&scan, "connectivity_imbalance").is_empty(),
        "the loop-2 check must stop the imbalance scan before any candidate is added, got: {scan:?}"
    );
}
