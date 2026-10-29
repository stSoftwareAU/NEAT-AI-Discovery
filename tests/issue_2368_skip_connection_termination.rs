//! Termination regression tests for the skip-connection depth pass (Issue #2368).
//!
//! These tests call only `detect_skip_connection_candidates`, which exists unchanged on the
//! base branch, so they compile and run against both the old and the new implementation.
//! They fail on the old one and pass on the new one:
//!
//! * the old depth pass was a re-pushing longest-path BFS, which never terminates on a cycle
//!   and does O(V²) pops on a chain that also has an edge from the input to every link;
//! * the new depth pass is a single Kahn topological pass, O(V + E), that reports a cycle.
//!
//! The scan runs on a worker thread and the test waits on a channel. That is a hang guard,
//! not a performance assertion: the budget is far above the new implementation's cost (a few
//! milliseconds) and far below the old implementation's (tens of seconds for the chain, forever
//! for the cycle). A stuck worker is abandoned and dies with the test process.

use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use neat_ai_discovery::analysis::constants::MIN_DISCOVERY_SAMPLE_COUNT;
use neat_ai_discovery::analysis::detection::skip_connection::{
    SkipConnectionCandidate, detect_skip_connection_candidates,
};
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

/// Generous hang-guard budget. The new scan finishes in milliseconds.
const HANG_GUARD: Duration = Duration::from_secs(20);

/// Number of hidden links in the chain fixture. The old depth pass needs about `n^2 / 2` pops.
const CHAIN_LEN: usize = 30_000;

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

/// `MIN_DISCOVERY_SAMPLE_COUNT` records for one hidden neuron. Having at least one qualified
/// hidden neuron is what lets the scan past its early return and into the depth pass.
fn hidden_records(uuid: &str) -> (String, Vec<DiscoverRecord>) {
    let records = (0..MIN_DISCOVERY_SAMPLE_COUNT)
        .map(|i| DiscoverRecord {
            obs_index: u32::try_from(i).expect("sample index fits in u32"),
            neuron_uuid: uuid.to_string(),
            value: None,
            activation: 0.5,
            errors: vec![0.5],
        })
        .collect();
    (uuid.to_string(), records)
}

/// Runs the public scan on a worker thread. Returns `None` if it has not finished within
/// `HANG_GUARD`.
fn scan_with_hang_guard(
    creature: CreatureJson,
    records: Vec<(String, Vec<DiscoverRecord>)>,
) -> Option<Vec<SkipConnectionCandidate>> {
    let (sender, receiver) = mpsc::channel();
    thread::spawn(move || {
        let candidates = detect_skip_connection_candidates(&creature, &records);
        // The receiver is gone if the guard already expired; nothing to report then.
        let _ = sender.send(candidates);
    });
    receiver.recv_timeout(HANG_GUARD).ok()
}

/// A cycle among hidden neurons reachable from an input (`c1 -> c2 -> c3 -> c1`).
/// The old depth pass raised the depths around the cycle forever. The new one returns no
/// candidates.
#[test]
fn cyclic_graph_terminates_with_no_candidates() {
    let creature = CreatureJson {
        neurons: vec![
            neuron("in0", "input"),
            neuron("c0", "hidden"),
            neuron("c1", "hidden"),
            neuron("c2", "hidden"),
            neuron("c3", "hidden"),
        ],
        synapses: vec![
            synapse("in0", "c0"),
            synapse("c0", "c1"),
            synapse("c1", "c2"),
            synapse("c2", "c3"),
            synapse("c3", "c1"),
        ],
        input: 1,
        output: 0,
    };
    let records = vec![hidden_records("c0"), hidden_records("c1")];

    let candidates = scan_with_hang_guard(creature, records)
        .expect("the skip-connection scan must terminate on a cyclic graph");
    assert!(
        candidates.is_empty(),
        "a cyclic graph must yield no candidates, got: {candidates:?}"
    );
}

/// An acyclic chain `a0 -> a1 -> ... -> a(n-1)` where the input also feeds every link. Every
/// link starts at depth 1 and is then raised one layer per sweep, so the old depth pass popped
/// about `n^2 / 2` entries. The new Kahn pass visits each link once.
#[test]
fn long_chain_with_input_fan_out_terminates() {
    let mut neurons = vec![neuron("in0", "input")];
    let mut synapses = Vec::with_capacity(2 * CHAIN_LEN);
    for i in 0..CHAIN_LEN {
        let uuid = format!("a{i}");
        neurons.push(neuron(&uuid, "hidden"));
        synapses.push(synapse("in0", &uuid));
        if i > 0 {
            synapses.push(synapse(&format!("a{}", i - 1), &uuid));
        }
    }
    let creature = CreatureJson {
        neurons,
        synapses,
        input: 1,
        output: 0,
    };
    let records = vec![hidden_records("a0")];

    let candidates = scan_with_hang_guard(creature, records)
        .expect("the skip-connection depth pass must be linear on an acyclic chain");
    // Only `a0` is qualified, and it is shallow, so nothing is deep and attenuated.
    assert!(
        candidates.is_empty(),
        "no deep attenuated neuron is qualified, got: {candidates:?}"
    );
}
