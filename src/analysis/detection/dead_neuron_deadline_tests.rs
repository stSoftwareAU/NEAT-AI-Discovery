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
    let scan = detect_dead_neurons_observed(
        &creature,
        &records,
        None,
        &unused_live_deadline(),
        usize::MAX,
        || {
            steps += 1;
        },
    );

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
    let scan = detect_dead_neurons_observed(
        &creature,
        &records,
        None,
        &unused_live_deadline(),
        usize::MAX,
        || {
            steps += 1;
        },
    );

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

/// `a -> out-a` and `b -> out-b`: two separate, identically shaped dead roots,
/// so the walk does the same work whichever root the hidden-neuron set yields
/// first.
fn two_dead_roots_creature() -> (CreatureJson, Vec<(String, Vec<DiscoverRecord>)>) {
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

    let creature = CreatureJson {
        neurons: vec![
            neuron("input-0", "input"),
            neuron("a", "hidden"),
            neuron("b", "hidden"),
            neuron("out-a", "output"),
            neuron("out-b", "output"),
        ],
        synapses: vec![
            synapse("input-0", "a"),
            synapse("input-0", "b"),
            synapse("a", "out-a"),
            synapse("b", "out-b"),
        ],
        input: 1,
        output: 2,
    };
    (creature, vec![records_for("a"), records_for("b")])
}

/// A deadline stop during the reachability pass keeps the candidate whose
/// outputs completed before it, as the `DeadNeuronScan::truncation` docs say.
#[test]
#[serial]
fn reachability_deadline_stop_keeps_the_completed_candidate() {
    let (creature, records) = two_dead_roots_creature();

    // Phase 1: one check per hidden neuron (2, both false). Phase 2, first
    // root: examine its edge, close its output, close the root (3, all false).
    // The second root's first walk check then reports the deadline passed.
    let _guard =
        DeadlineOverrideGuard::with_sequence(vec![false, false, false, false, false, true]);

    let scan = detect_dead_neurons_observed(
        &creature,
        &records,
        None,
        &unused_live_deadline(),
        usize::MAX,
        || {},
    );

    assert_eq!(scan.truncation, Some(ScanTruncation::DeadlinePassed));
    assert_eq!(
        scan.candidates.len(),
        1,
        "the first root completed before the deadline, so it must be kept"
    );
    let kept = &scan.candidates[0];
    assert_eq!(
        kept.connected_outputs,
        [format!("out-{}", kept.neuron_uuid)],
        "the kept candidate carries its own single output"
    );
}

/// Number of outputs `x` and `y` each feed in [`merge_scan_creature`]. Chosen
/// so `d`'s merge scans `2 * MERGE_FAN_OUTPUTS` = 6,000 successor elements,
/// past the first in-merge deadline check at element 4,096.
const MERGE_FAN_OUTPUTS: usize = 3_000;

/// Dead `d` -> live `x`, `y`; `x` and `y` each feed the same
/// [`MERGE_FAN_OUTPUTS`] outputs. Only `d`'s close merges non-empty successor
/// sets (`x`'s and `y`'s, 3,000 ids each), so the merge loop's periodic
/// deadline check is reached only there.
fn merge_scan_creature() -> (CreatureJson, Vec<(String, Vec<DiscoverRecord>)>) {
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
    // A live neuron: alternating 0 / 1 activations, far above every
    // dead-neuron threshold.
    let live_records = |uuid: &str| -> (String, Vec<DiscoverRecord>) {
        let recs = (0..SAMPLES)
            .map(|idx| DiscoverRecord {
                obs_index: idx,
                neuron_uuid: uuid.to_string(),
                value: Some(0.0),
                activation: if idx % 2 == 0 { 0.0 } else { 1.0 },
                errors: vec![0.0],
            })
            .collect();
        (uuid.to_string(), recs)
    };

    let outputs: Vec<String> = (0..MERGE_FAN_OUTPUTS).map(|i| format!("out-{i}")).collect();
    let mut neurons = vec![
        neuron("input-0", "input"),
        neuron("d", "hidden"),
        neuron("x", "hidden"),
        neuron("y", "hidden"),
    ];
    neurons.extend(outputs.iter().map(|o| neuron(o, "output")));

    let mut synapses = vec![
        synapse("input-0", "d"),
        synapse("d", "x"),
        synapse("d", "y"),
    ];
    for o in &outputs {
        synapses.push(synapse("x", o));
        synapses.push(synapse("y", o));
    }

    let creature = CreatureJson {
        neurons,
        synapses,
        input: 1,
        output: MERGE_FAN_OUTPUTS,
    };
    (
        creature,
        vec![records_for("d"), live_records("x"), live_records("y")],
    )
}

/// The deadline check inside the successor merge loop (every
/// `MERGE_DEADLINE_CHECK_INTERVAL` scanned elements) stops a large merge.
///
/// Call order of `deadline_passed`:
/// - Phase 1: one check per hidden neuron (`d`, `x`, `y`): 3.
/// - Phase 2, the walk from `d`: one check per main-loop iteration. That is
///   one per examined edge (2 from `d`, 3,000 from `x`, 3,000 from `y`: 6,002)
///   plus one per node pop (`d`, `x`, `y` and the 3,000 outputs: 3,003), so
///   9,005 in all, every one `false`. `x` merges only the outputs' empty sets,
///   so its close scans no successor element.
/// - `d`'s close, run inside its own pop iteration, then scans `x`'s 3,000
///   ids and `y`'s 3,000. Its first merge-loop check falls at element 4,096,
///   and that is the call scripted `true`.
///
/// Without the merge-loop check the scan would finish and return `d` with no
/// truncation.
#[test]
#[serial]
fn merge_loop_deadline_check_stops_a_large_merge() {
    let (creature, records) = merge_scan_creature();

    let walk_iterations = 6_002 + 3_003;
    // Built with an iterator, not a `vec!` repeat expression, which the chunk
    // 8a ledger would count as a capacity site.
    let sequence: Vec<bool> = std::iter::repeat_n(false, 3 + walk_iterations)
        .chain(std::iter::once(true))
        .collect();
    let _guard = DeadlineOverrideGuard::with_sequence(sequence);

    let scan = detect_dead_neurons_observed(
        &creature,
        &records,
        None,
        &unused_live_deadline(),
        usize::MAX,
        || {},
    );

    assert_eq!(
        scan.truncation,
        Some(ScanTruncation::DeadlinePassed),
        "the merge-loop deadline check must stop d's 6,000-element merge"
    );
    assert!(
        scan.candidates.is_empty(),
        "d's merge did not finish, so it must not be returned as a candidate"
    );
}

/// Dead `a` and `b` in a two-node cycle (`a <-> b`), each also feeding `out-0`
/// and `out-1`. Both share one strongly connected component, so whichever root
/// the hidden-neuron set yields first, its walk closes both, and the second
/// root is a memo hit that runs no walk.
fn dead_cycle_above_two_outputs() -> (CreatureJson, Vec<(String, Vec<DiscoverRecord>)>) {
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

    let creature = CreatureJson {
        neurons: vec![
            neuron("input-0", "input"),
            neuron("a", "hidden"),
            neuron("b", "hidden"),
            neuron("out-0", "output"),
            neuron("out-1", "output"),
        ],
        synapses: vec![
            synapse("input-0", "a"),
            synapse("a", "b"),
            synapse("a", "out-0"),
            synapse("a", "out-1"),
            synapse("b", "a"),
            synapse("b", "out-0"),
            synapse("b", "out-1"),
        ],
        input: 1,
        output: 2,
    };
    (creature, vec![records_for("a"), records_for("b")])
}

/// A dead neuron whose component is already memoised still gets a deadline
/// check before its output list is built (review 5711590564). Without it, a
/// long run of memo hits (a dead chain above a wide fan-out) would build every
/// output list past the deadline and report no truncation.
///
/// Call order of `deadline_passed`:
/// - Phase 1: one check per hidden neuron (`a`, `b`): 2.
/// - Phase 2, the first root's walk (the fixture is symmetric, so either root
///   does the same): 6 examined edges plus 4 node pops (both outputs, the
///   other root, then this root, which closes `{a, b}`): 10, all `false`.
/// - The second root is a memo hit; its check is the one scripted `true`.
#[test]
#[serial]
fn memo_hit_root_deadline_check_stops_the_scan() {
    let (creature, records) = dead_cycle_above_two_outputs();

    let sequence: Vec<bool> = std::iter::repeat_n(false, 2 + 10)
        .chain(std::iter::once(true))
        .collect();
    let _guard = DeadlineOverrideGuard::with_sequence(sequence);

    let mut steps = 0;
    let scan = detect_dead_neurons_observed(
        &creature,
        &records,
        None,
        &unused_live_deadline(),
        usize::MAX,
        || {
            steps += 1;
        },
    );

    assert_eq!(
        scan.truncation,
        Some(ScanTruncation::DeadlinePassed),
        "the memo-hit root's deadline check must stop the scan"
    );
    assert_eq!(
        scan.candidates.len(),
        1,
        "the first root completed before the deadline, so it must be kept"
    );
    assert_eq!(scan.candidates[0].connected_outputs, ["out-0", "out-1"]);
    // Walk: 6 examined edges. Closing `{a, b}`: 6 member edges scanned. The
    // kept candidate's output list: 2 names. The second root adds none.
    assert_eq!(
        steps, 14,
        "the observer must count the walk, the close and the output names handed out"
    );
}
