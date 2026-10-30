//! Deadline regression tests for topology detection (Issue #2369).

use std::time::SystemTime;

use serial_test::serial;

use super::{
    compute_shortest_paths_to_output, detect_topology_issues_with_deadline,
    first_unconnected_input, mean_absolute_error,
};
use crate::analysis::detection::topology_cache::CreatureTopologyCache;
use crate::analysis::recommendation::epistatic::ScanTruncation;
use crate::analysis::utils::deadline_override::DeadlineOverrideGuard;
use crate::types::DiscoverRecord;
use crate::{CreatureJson, NeuronJson, SynapseJson};

fn neuron(uuid: &str, kind: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: kind.to_string(),
        squash: "IDENTITY".to_string(),
        bias: 0.0,
    }
}

fn edge(from: &str, to: &str) -> SynapseJson {
    SynapseJson {
        from_uuid: from.to_string(),
        to_uuid: to.to_string(),
        weight: 0.5,
        synapse_type: None,
    }
}

fn records_for(uuids: &[&str]) -> Vec<(String, Vec<DiscoverRecord>)> {
    uuids
        .iter()
        .map(|uuid| {
            let records = (0..20)
                .map(|i| DiscoverRecord::new(i, (*uuid).to_string(), Some(0.5), 0.5, vec![0.2]))
                .collect();
            ((*uuid).to_string(), records)
        })
        .collect()
}

fn long_path_fixture() -> (CreatureJson, Vec<(String, Vec<DiscoverRecord>)>) {
    let creature = CreatureJson {
        neurons: vec![
            neuron("h0", "hidden"),
            neuron("h1", "hidden"),
            neuron("h2", "hidden"),
            neuron("h3", "hidden"),
            neuron("h4", "hidden"),
            neuron("out", "output"),
        ],
        synapses: vec![
            edge("h0", "h1"),
            edge("h1", "h2"),
            edge("h2", "h3"),
            edge("h3", "h4"),
            edge("h4", "out"),
        ],
        input: 0,
        output: 1,
    };
    (creature, records_for(&["h0", "h1"]))
}

fn imbalanced_fixture() -> (CreatureJson, Vec<(String, Vec<DiscoverRecord>)>) {
    let creature = CreatureJson {
        neurons: vec![
            neuron("seed", "hidden"),
            neuron("a", "hidden"),
            neuron("b", "hidden"),
            neuron("c", "hidden"),
            neuron("rich", "hidden"),
        ],
        synapses: vec![
            edge("seed", "a"),
            edge("seed", "b"),
            edge("seed", "c"),
            edge("seed", "rich"),
            edge("a", "rich"),
            edge("b", "rich"),
            edge("c", "rich"),
        ],
        input: 0,
        output: 0,
    };
    (creature, records_for(&["a", "b", "c", "rich"]))
}

#[test]
#[serial]
fn long_path_loop_deadline_keeps_candidates_found_before_interruption() {
    let (creature, records) = long_path_fixture();
    let topology = CreatureTopologyCache::new(&creature);
    // Preparation, output collection, BFS, and 80 record/error checks precede
    // the second long-path iteration.
    let mut checks = vec![false; 115];
    checks.push(true);
    let _guard = DeadlineOverrideGuard::with_sequence(checks);

    let scan = detect_topology_issues_with_deadline(&creature, &records, Some(&topology), &None);

    assert_eq!(scan.truncation, Some(ScanTruncation::DeadlinePassed));
    assert_eq!(
        scan.candidates.len(),
        1,
        "the completed first long-path candidate remains valid"
    );
    assert_eq!(scan.candidates[0].issue_type, "long_path");
}

#[test]
#[serial]
fn input_scan_checks_cancellation_between_existing_edges() {
    let creature = CreatureJson {
        neurons: vec![
            neuron("i0", "input"),
            neuron("i1", "input"),
            neuron("h", "hidden"),
        ],
        synapses: vec![edge("i0", "h")],
        input: 2,
        output: 0,
    };
    let topology = CreatureTopologyCache::new(&creature);
    let inputs = ["i0", "i1"];
    let _guard = DeadlineOverrideGuard::with_sequence(vec![false, true]);

    let result = first_unconnected_input("h", &inputs, &topology, &None);

    assert_eq!(result, Err(ScanTruncation::DeadlinePassed));
}

#[test]
#[serial]
fn output_collection_checks_cancellation_after_partial_collection() {
    let creature = CreatureJson {
        neurons: vec![
            neuron("h0", "hidden"),
            neuron("h1", "hidden"),
            neuron("h2", "hidden"),
        ],
        synapses: Vec::new(),
        input: 0,
        output: 0,
    };
    let _guard = DeadlineOverrideGuard::with_sequence(vec![false, false, true]);

    assert!(compute_shortest_paths_to_output(&creature, &None).is_none());
}

#[test]
#[serial]
fn reverse_bfs_checks_cancellation_after_graph_preparation() {
    let creature = CreatureJson {
        neurons: vec![
            neuron("h0", "hidden"),
            neuron("h1", "hidden"),
            neuron("out", "output"),
        ],
        synapses: vec![edge("h0", "h1"), edge("h1", "out")],
        input: 0,
        output: 1,
    };
    // Two edges are indexed, three neurons are classified, and the output is
    // seeded before cancellation on the first reverse-BFS queue entry.
    let _guard =
        DeadlineOverrideGuard::with_sequence(vec![false, false, false, false, false, false, true]);

    assert!(compute_shortest_paths_to_output(&creature, &None).is_none());
}

#[test]
#[serial]
fn error_element_traversal_checks_cancellation_inside_a_record() {
    let records = [DiscoverRecord::new(
        0,
        "h".to_string(),
        Some(0.5),
        0.5,
        vec![0.2, 0.4],
    )];
    let _guard = DeadlineOverrideGuard::with_sequence(vec![false, false, true]);

    assert_eq!(
        mean_absolute_error(&records, &None),
        Err(ScanTruncation::DeadlinePassed)
    );
}

#[test]
#[serial]
fn elapsed_deadline_is_reported_before_preparation() {
    let (creature, records) = long_path_fixture();
    let topology = CreatureTopologyCache::new(&creature);
    let scan = detect_topology_issues_with_deadline(
        &creature,
        &records,
        Some(&topology),
        &Some(SystemTime::UNIX_EPOCH),
    );

    assert_eq!(scan.truncation, Some(ScanTruncation::DeadlinePassed));
    assert!(scan.candidates.is_empty());
}

#[test]
#[serial]
fn preparation_deadline_is_reported_during_record_lookup() {
    let (creature, records) = long_path_fixture();
    let topology = CreatureTopologyCache::new(&creature);
    let _guard = DeadlineOverrideGuard::with_sequence(vec![false, false, true]);

    let scan = detect_topology_issues_with_deadline(&creature, &records, Some(&topology), &None);

    assert_eq!(scan.truncation, Some(ScanTruncation::DeadlinePassed));
    assert!(scan.candidates.is_empty());
}

#[test]
#[serial]
fn imbalance_loop_deadline_keeps_candidates_found_before_interruption() {
    let (creature, records) = imbalanced_fixture();
    let topology = CreatureTopologyCache::new(&creature);
    // Preparation, output collection, record/error traversal, input copying,
    // long-path and fan-in checks, and one imbalance candidate complete first.
    let mut checks = vec![false; 196];
    checks.push(true);
    let _guard = DeadlineOverrideGuard::with_sequence(checks);

    let scan = detect_topology_issues_with_deadline(&creature, &records, Some(&topology), &None);

    assert_eq!(scan.truncation, Some(ScanTruncation::DeadlinePassed));
    assert!(!scan.candidates.is_empty());
    assert!(
        scan.candidates
            .iter()
            .all(|candidate| candidate.issue_type == "connectivity_imbalance")
    );
    assert_eq!(
        scan.candidates.len(),
        1,
        "cancellation during the imbalance scan preserves its completed candidate"
    );
}
