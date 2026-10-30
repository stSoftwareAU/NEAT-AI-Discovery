//! Issue #2370: `detect_dead_neurons` resolved each dead neuron's connected
//! outputs with its own depth-first search, so the reachability work was
//! O(D * (V + E)) per creature. It is now one shared, memoised
//! strongly-connected-component pass over the fan-out graph.
//!
//! This pins:
//! - the reachability work grows linearly with the dead-neuron count, not
//!   quadratically (counted via the observer, never timed);
//! - a 20,000-deep chain completes without overflowing the thread stack;
//! - a cycle among dead neurons terminates and reports the right outputs;
//! - the outputs match an in-test reference traversal on a branching graph
//!   with a diamond, a shared descendant, a cycle and an output-to-output edge;
//! - an elapsed deadline stops the scan and reports `DeadlinePassed`, and a
//!   far-future deadline does not truncate;
//! - the cache-backed and cache-free entry points agree;
//! - a live chain above a wide fan-out shares its output set instead of copying
//!   it per chain node, so the work does not grow as chain length x outputs;
//! - a merge ceiling stops the scan with `CandidateCeiling`, and only the ceiling
//!   does (the same input with an unbounded ceiling completes).
//!
//! Deterministic throughout: no timing thresholds and no sleeps. An elapsed
//! deadline is `UNIX_EPOCH`, a live one is a day ahead.

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, SystemTime};

use neat_ai_discovery::analysis::detection::dead_neuron::{
    DeadNeuronCandidate, DeadNeuronScan, detect_dead_neurons, detect_dead_neurons_observed,
    detect_dead_neurons_with_deadline,
};
use neat_ai_discovery::analysis::detection::topology_cache::CreatureTopologyCache;
use neat_ai_discovery::analysis::recommendation::epistatic::ScanTruncation;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

/// Per-neuron recorded activations.
type Records = Vec<(String, Vec<DiscoverRecord>)>;

/// Clears `MIN_DISCOVERY_SAMPLE_COUNT` (20).
const SAMPLES: u32 = 30;

/// Production-sized merge ceiling (`MAX_REACHABILITY_MERGE_WORK`).
const PRODUCTION_MERGE_CEILING: usize = 8_000_000;

fn neuron(uuid: &str, neuron_type: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
        squash: "LOGISTIC".to_string(),
        bias: 0.0,
    }
}

fn synapse(from_uuid: &str, to_uuid: &str) -> SynapseJson {
    SynapseJson {
        from_uuid: from_uuid.to_string(),
        to_uuid: to_uuid.to_string(),
        weight: 0.5,
        synapse_type: None,
    }
}

/// `SAMPLES` records for `uuid` at a constant `activation`. A dead neuron uses
/// `0.0` (value `Some(0.0)`, errors `[0.0]`); a live one uses `0.9`.
fn records_of(uuid: &str, activation: f32) -> (String, Vec<DiscoverRecord>) {
    let records = (0..SAMPLES)
        .map(|idx| DiscoverRecord {
            obs_index: idx,
            neuron_uuid: uuid.to_string(),
            value: Some(activation),
            activation,
            errors: vec![0.0],
        })
        .collect();
    (uuid.to_string(), records)
}

/// Build a creature with one input (`input-0`), the given hidden and output
/// neurons, and the given directed edges.
fn build_creature(hidden: &[&str], outputs: &[&str], edges: &[(&str, &str)]) -> CreatureJson {
    let mut neurons = vec![neuron("input-0", "input")];
    neurons.extend(hidden.iter().map(|u| neuron(u, "hidden")));
    neurons.extend(outputs.iter().map(|u| neuron(u, "output")));
    CreatureJson {
        neurons,
        synapses: edges.iter().map(|(from, to)| synapse(from, to)).collect(),
        input: 1,
        output: outputs.len(),
    }
}

/// `input-0 -> h0 -> h1 -> ... -> h{n-1} -> output-0`, every hidden neuron dead.
fn chain(n: usize) -> (CreatureJson, Vec<(String, Vec<DiscoverRecord>)>) {
    let hidden: Vec<String> = (0..n).map(|h| format!("h{h}")).collect();
    let hidden_refs: Vec<&str> = hidden.iter().map(String::as_str).collect();

    let mut edges: Vec<(&str, &str)> = Vec::with_capacity(n + 1);
    let mut prev = "input-0";
    for &h in &hidden_refs {
        edges.push((prev, h));
        prev = h;
    }
    edges.push((prev, "output-0"));

    let creature = build_creature(&hidden_refs, &["output-0"], &edges);
    let records = hidden_refs.iter().map(|h| records_of(h, 0.0)).collect();
    (creature, records)
}

/// Seven dead hidden neurons (`a`..`g`) and one live hidden neuron (`m`) in
/// the middle of a graph with three outputs. It contains a diamond
/// (`a -> b, c -> d`), a shared descendant (`d`), a cycle (`d <-> e`), a live
/// hidden neuron on a path (`c -> m -> g`), and an output with an outgoing
/// edge to another output (`out-1 -> out-2`). Each dead neuron reaches a
/// different output set.
fn branching_graph() -> (CreatureJson, Vec<(String, Vec<DiscoverRecord>)>) {
    let dead = ["a", "b", "c", "d", "e", "f", "g"];
    let hidden: Vec<&str> = dead.iter().copied().chain(["m"]).collect();
    let edges = [
        ("input-0", "a"),
        ("a", "b"),
        ("a", "c"),
        ("b", "d"),
        ("c", "d"),
        ("c", "m"),
        ("m", "g"),
        ("g", "out-1"),
        ("d", "e"),
        ("e", "d"),
        ("d", "out-0"),
        ("e", "f"),
        ("f", "out-2"),
        ("out-1", "out-2"),
    ];
    let creature = build_creature(&hidden, &["out-0", "out-1", "out-2"], &edges);
    let records = dead
        .iter()
        .map(|u| records_of(u, 0.0))
        .chain(std::iter::once(records_of("m", 0.9)))
        .collect();
    (creature, records)
}

/// Reference traversal: BFS with a visited set from `start`, collecting every
/// output node reachable via at least one edge, sorted ascending.
fn reference_outputs(creature: &CreatureJson, start: &str) -> Vec<String> {
    let outputs: HashSet<&str> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "output")
        .map(|n| n.uuid.as_str())
        .collect();

    let mut adjacency: HashMap<&str, Vec<&str>> = HashMap::new();
    for s in &creature.synapses {
        adjacency
            .entry(s.from_uuid.as_str())
            .or_default()
            .push(s.to_uuid.as_str());
    }

    let mut visited: HashSet<&str> = HashSet::new();
    let mut queue: VecDeque<&str> = VecDeque::new();
    let mut found: BTreeSet<String> = BTreeSet::new();

    for &next in adjacency.get(start).into_iter().flatten() {
        if visited.insert(next) {
            queue.push_back(next);
        }
    }
    while let Some(node) = queue.pop_front() {
        if outputs.contains(node) {
            found.insert(node.to_string());
        }
        for &next in adjacency.get(node).into_iter().flatten() {
            if visited.insert(next) {
                queue.push_back(next);
            }
        }
    }
    found.into_iter().collect()
}

/// Order-independent view of a candidate list (candidates are sorted by
/// removal confidence, which ties here, so the order is not meaningful).
fn summary(candidates: &[DeadNeuronCandidate]) -> Vec<(String, Vec<String>, usize, u32)> {
    let mut rows: Vec<_> = candidates
        .iter()
        .map(|c| {
            (
                c.neuron_uuid.clone(),
                c.connected_outputs.clone(),
                c.sample_count,
                c.removal_confidence.to_bits(),
            )
        })
        .collect();
    rows.sort();
    rows
}

/// Run the scan and count every reachability step the observer sees.
fn count_reachability_steps(
    creature: &CreatureJson,
    records: &[(String, Vec<DiscoverRecord>)],
) -> (usize, DeadNeuronScan) {
    let mut steps = 0_usize;
    let scan = detect_dead_neurons_observed(
        creature,
        records,
        None,
        &None,
        PRODUCTION_MERGE_CEILING,
        || {
            steps += 1;
        },
    );
    (steps, scan)
}

fn far_future_deadline() -> Option<SystemTime> {
    Some(SystemTime::now() + Duration::from_secs(24 * 60 * 60))
}

#[test]
fn reachability_grows_linearly_with_dead_count() {
    let (small, small_records) = chain(64);
    let (large, large_records) = chain(256);

    let (count_n, scan_n) = count_reachability_steps(&small, &small_records);
    let (count_4n, scan_4n) = count_reachability_steps(&large, &large_records);

    assert_eq!(scan_n.candidates.len(), 64, "all 64 dead neurons found");
    assert_eq!(scan_4n.candidates.len(), 256, "all 256 dead neurons found");
    assert!(scan_n.truncation.is_none() && scan_4n.truncation.is_none());
    assert!(
        scan_n
            .candidates
            .iter()
            .chain(scan_4n.candidates.iter())
            .all(|c| c.connected_outputs == ["output-0"]),
        "every chain neuron reaches only output-0"
    );

    // Linear work is about 4x for 4x the neurons; the old per-neuron DFS is
    // about 16x. The bound sits between the two.
    assert!(
        count_n > 0 && count_4n > 3 * count_n && count_4n < 6 * count_n,
        "reachability work must grow linearly: N=64 took {count_n} steps, \
         N=256 took {count_4n} steps (linear is ~4x, per-neuron DFS is ~16x)"
    );
}

#[test]
fn deep_chain_does_not_overflow_the_stack() {
    const DEPTH: usize = 20_000;
    let (creature, records) = chain(DEPTH);

    let scan = detect_dead_neurons_with_deadline(&creature, &records, None, &None);

    assert!(scan.truncation.is_none());
    assert_eq!(scan.candidates.len(), DEPTH);
    assert!(
        scan.candidates
            .iter()
            .all(|c| c.connected_outputs == ["output-0"]),
        "every neuron in the deep chain reaches output-0"
    );
}

#[test]
fn cycle_among_dead_neurons_terminates_with_correct_outputs() {
    // h0 -> h1 -> h2 -> h0 (cycle), h2 -> output-0, h1 -> output-1.
    let hidden = ["h0", "h1", "h2"];
    let edges = [
        ("input-0", "h0"),
        ("h0", "h1"),
        ("h1", "h2"),
        ("h2", "h0"),
        ("h2", "output-0"),
        ("h1", "output-1"),
    ];
    let creature = build_creature(&hidden, &["output-0", "output-1"], &edges);
    let records: Vec<(String, Vec<DiscoverRecord>)> =
        hidden.iter().map(|h| records_of(h, 0.0)).collect();

    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let scan = detect_dead_neurons_with_deadline(&creature, &records, None, &None);
        let _ = tx.send(scan);
    });
    let scan = rx
        .recv_timeout(Duration::from_secs(30))
        .expect("cycle scan did not finish within 30 s: possible non-termination");

    assert!(scan.truncation.is_none());
    assert_eq!(scan.candidates.len(), 3);
    for c in &scan.candidates {
        assert_eq!(
            c.connected_outputs,
            ["output-0", "output-1"],
            "{} reaches both outputs through the cycle",
            c.neuron_uuid
        );
    }
}

#[test]
fn matches_a_reference_traversal_on_a_branching_graph() {
    let (creature, records) = branching_graph();

    let scan = detect_dead_neurons_with_deadline(&creature, &records, None, &None);

    assert!(scan.truncation.is_none());
    assert_eq!(
        scan.candidates.len(),
        7,
        "the live neuron m is not a candidate"
    );

    let by_uuid: HashMap<&str, &DeadNeuronCandidate> = scan
        .candidates
        .iter()
        .map(|c| (c.neuron_uuid.as_str(), c))
        .collect();
    assert!(!by_uuid.contains_key("m"), "live neuron m must not appear");

    // Hand-computed expectations, checked against the reference traversal too.
    let expected: [(&str, &[&str]); 7] = [
        ("a", &["out-0", "out-1", "out-2"]),
        ("b", &["out-0", "out-2"]),
        ("c", &["out-0", "out-1", "out-2"]),
        ("d", &["out-0", "out-2"]),
        ("e", &["out-0", "out-2"]),
        ("f", &["out-2"]),
        ("g", &["out-1", "out-2"]),
    ];
    for (uuid, outputs) in expected {
        let candidate = by_uuid
            .get(uuid)
            .unwrap_or_else(|| panic!("{uuid} missing from candidates"));
        assert_eq!(
            candidate.connected_outputs, outputs,
            "hand-computed outputs for {uuid}"
        );
    }

    for c in &scan.candidates {
        assert_eq!(
            c.connected_outputs,
            reference_outputs(&creature, &c.neuron_uuid),
            "reference traversal disagrees for {}",
            c.neuron_uuid
        );
    }
}

#[test]
fn elapsed_deadline_stops_the_scan_and_reports_truncation() {
    let (creature, records) = chain(8);
    let elapsed = Some(SystemTime::UNIX_EPOCH);

    let scan = detect_dead_neurons_with_deadline(&creature, &records, None, &elapsed);

    assert_eq!(scan.truncation, Some(ScanTruncation::DeadlinePassed));
    assert!(scan.candidates.is_empty());
}

#[test]
fn far_future_deadline_does_not_truncate() {
    let (creature, records) = branching_graph();

    let bounded =
        detect_dead_neurons_with_deadline(&creature, &records, None, &far_future_deadline());
    let plain = detect_dead_neurons(&creature, &records, None);

    assert!(bounded.truncation.is_none());
    assert_eq!(bounded.candidates.len(), 7);
    assert_eq!(summary(&bounded.candidates), summary(&plain));
}

#[test]
fn without_a_cache_and_with_a_cache_agree() {
    let (creature, records) = branching_graph();
    let cache = CreatureTopologyCache::new(&creature);

    let uncached = detect_dead_neurons(&creature, &records, None);
    let cached = detect_dead_neurons(&creature, &records, Some(&cache));

    assert_eq!(uncached.len(), 7);
    assert_eq!(summary(&cached), summary(&uncached));
}

/// One dead hidden root `d`, a chain of `n` live hidden nodes (no records), and
/// a last chain node feeding `k` outputs.
fn live_chain_above_fan_out(n: usize, k: usize) -> (CreatureJson, Records, Vec<String>) {
    let chain_nodes: Vec<String> = (0..n).map(|i| format!("c{i}")).collect();
    let outputs: Vec<String> = (0..k).map(|i| format!("out-{i:03}")).collect();

    let mut hidden: Vec<&str> = vec!["d"];
    hidden.extend(chain_nodes.iter().map(String::as_str));
    let output_refs: Vec<&str> = outputs.iter().map(String::as_str).collect();

    let mut edges: Vec<(&str, &str)> = vec![("input-0", "d"), ("d", chain_nodes[0].as_str())];
    for pair in chain_nodes.windows(2) {
        edges.push((pair[0].as_str(), pair[1].as_str()));
    }
    let last = chain_nodes[n - 1].as_str();
    edges.extend(output_refs.iter().map(|&o| (last, o)));

    let creature = build_creature(&hidden, &output_refs, &edges);
    // Only `d` has records, so only `d` is a dead candidate.
    let records = vec![records_of("d", 0.0)];
    (creature, records, outputs)
}

#[test]
fn live_chain_above_wide_fan_out_does_not_copy_sets_per_component() {
    const K: usize = 64;
    let (small, small_records, outputs) = live_chain_above_fan_out(64, K);
    let (large, large_records, _) = live_chain_above_fan_out(256, K);

    let (count_n, scan_n) = count_reachability_steps(&small, &small_records);
    let (count_4n, scan_4n) = count_reachability_steps(&large, &large_records);

    assert!(scan_n.truncation.is_none() && scan_4n.truncation.is_none());
    for scan in [&scan_n, &scan_4n] {
        assert_eq!(scan.candidates.len(), 1, "only d is a dead candidate");
        assert_eq!(scan.candidates[0].neuron_uuid, "d");
        assert_eq!(
            scan.candidates[0].connected_outputs, outputs,
            "d reaches all {K} outputs"
        );
    }

    assert!(
        count_n > 0 && count_4n < 6 * count_n,
        "work must grow linearly in chain length: N=64 took {count_n} steps, \
         N=256 took {count_4n} steps"
    );
    assert!(
        count_4n < 256 * K,
        "work must not be chain length x outputs: N=256 took {count_4n} steps, \
         copying the set per chain node would take over {}",
        256 * K
    );
}

/// `d -> x -> out-0` and `d -> y -> out-1`, with `x` and `y` live, so closing
/// `d` must merge two non-empty successor sets.
fn merging_graph() -> (CreatureJson, Vec<(String, Vec<DiscoverRecord>)>) {
    let edges = [
        ("input-0", "d"),
        ("d", "x"),
        ("d", "y"),
        ("x", "out-0"),
        ("y", "out-1"),
    ];
    let creature = build_creature(&["d", "x", "y"], &["out-0", "out-1"], &edges);
    (creature, vec![records_of("d", 0.0)])
}

#[test]
fn merge_ceiling_stops_the_scan_and_reports_candidate_ceiling() {
    let (creature, records) = merging_graph();

    let ceiling_one = detect_dead_neurons_observed(&creature, &records, None, &None, 1, || {});
    assert_eq!(
        ceiling_one.truncation,
        Some(ScanTruncation::CandidateCeiling),
        "a ceiling of 1 element cannot hold the merge"
    );
    assert!(
        ceiling_one.candidates.is_empty(),
        "d never completed, so it is not a candidate"
    );

    // Positive control: only the ceiling stops the scan.
    let unbounded =
        detect_dead_neurons_observed(&creature, &records, None, &None, usize::MAX, || {});
    assert!(unbounded.truncation.is_none());
    assert_eq!(unbounded.candidates.len(), 1);
    assert_eq!(
        unbounded.candidates[0].connected_outputs,
        ["out-0", "out-1"]
    );

    let production = detect_dead_neurons_observed(
        &creature,
        &records,
        None,
        &None,
        PRODUCTION_MERGE_CEILING,
        || {},
    );
    assert!(production.truncation.is_none());
    assert_eq!(
        summary(&production.candidates),
        summary(&unbounded.candidates)
    );
}

/// One dead neuron `d` feeds 200 live hidden nodes (L0). Every L0 node feeds
/// all 200 live hidden nodes of L1, and every L1 node feeds all 200 outputs
/// (400 live hidden neurons, 200 outputs, 80,200 synapses).
fn dense_two_layer_fan_out() -> (CreatureJson, Records, Vec<String>) {
    const WIDTH: usize = 200;
    let l0: Vec<String> = (0..WIDTH).map(|i| format!("l0-{i:03}")).collect();
    let l1: Vec<String> = (0..WIDTH).map(|i| format!("l1-{i:03}")).collect();
    let outputs: Vec<String> = (0..WIDTH).map(|i| format!("out-{i:03}")).collect();

    let mut hidden: Vec<&str> = vec!["d"];
    hidden.extend(l0.iter().map(String::as_str));
    hidden.extend(l1.iter().map(String::as_str));
    let output_refs: Vec<&str> = outputs.iter().map(String::as_str).collect();

    let mut edges: Vec<(&str, &str)> = vec![("input-0", "d")];
    edges.extend(l0.iter().map(|a| ("d", a.as_str())));
    for a in &l0 {
        edges.extend(l1.iter().map(|b| (a.as_str(), b.as_str())));
    }
    for b in &l1 {
        edges.extend(output_refs.iter().map(|&o| (b.as_str(), o)));
    }

    let creature = build_creature(&hidden, &output_refs, &edges);
    // Only `d` has records, so only `d` is a dead candidate.
    (creature, vec![records_of("d", 0.0)], outputs)
}

/// Each L0 node merges 200 successor sets that all hold the same 200 outputs.
/// The merge ceiling must charge the 200 distinct ids it keeps, not the 40,000
/// it scans; otherwise the 200th L0 close crosses `MAX_REACHABILITY_MERGE_WORK`
/// and the production scan drops `d`, which a per-neuron DFS reported cheaply.
#[test]
fn dense_fan_out_with_duplicate_successor_sets_stays_under_the_merge_ceiling() {
    let (creature, records, outputs) = dense_two_layer_fan_out();

    let scan = detect_dead_neurons_with_deadline(&creature, &records, None, &far_future_deadline());

    assert_eq!(
        scan.truncation, None,
        "duplicate output ids across successors must not be charged to the merge ceiling"
    );
    assert_eq!(scan.candidates.len(), 1, "only d is a dead candidate");
    assert_eq!(scan.candidates[0].neuron_uuid, "d");
    assert_eq!(
        scan.candidates[0].connected_outputs, outputs,
        "d reaches all 200 outputs"
    );
}

/// Two separate, identically shaped dead roots: `a -> out-a` and `b -> out-b`.
/// Whichever root the hidden-neuron set yields first, the walk does the same
/// work, so a stop between the roots is deterministic.
fn two_separate_dead_roots() -> (CreatureJson, Records) {
    let edges = [
        ("input-0", "a"),
        ("input-0", "b"),
        ("a", "out-a"),
        ("b", "out-b"),
    ];
    let creature = build_creature(&["a", "b"], &["out-a", "out-b"], &edges);
    (creature, vec![records_of("a", 0.0), records_of("b", 0.0)])
}

/// A merge-ceiling stop keeps the candidate that completed before it. With a
/// ceiling of 1, the first root keeps its one output id and the second root's
/// merge is refused.
#[test]
fn merge_ceiling_stop_keeps_the_candidate_completed_before_it() {
    let (creature, records) = two_separate_dead_roots();

    let scan = detect_dead_neurons_observed(&creature, &records, None, &None, 1, || {});

    assert_eq!(scan.truncation, Some(ScanTruncation::CandidateCeiling));
    assert_eq!(
        scan.candidates.len(),
        1,
        "the root completed before the ceiling must be kept"
    );
    let kept = &scan.candidates[0];
    assert_eq!(
        kept.connected_outputs,
        [format!("out-{}", kept.neuron_uuid)],
        "the kept candidate carries its own single output"
    );

    // Positive control: with no ceiling both roots complete.
    let unbounded =
        detect_dead_neurons_observed(&creature, &records, None, &None, usize::MAX, || {});
    assert_eq!(unbounded.truncation, None);
    assert_eq!(unbounded.candidates.len(), 2);
}
