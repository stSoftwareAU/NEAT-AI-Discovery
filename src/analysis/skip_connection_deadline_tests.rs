//! Deadline, cycle and equivalence tests for the skip-connection scan (Issue #2368).
//!
//! These use the `DeadlineOverrideGuard` test seam (`#[cfg(test)]`-only, so they
//! cannot live in the `tests/` integration crate) to script `deadline_passed`.
//! The depth pass makes one check before the pass and one per neuron popped, so a
//! fixture with N neurons and no cycle makes N + 1 depth-pass checks. Each target
//! then makes one check at the top of the target loop.
//!
//! The `on_step` observer is the deterministic work counter.
//!
//! `#[serial]` keeps these tests from interleaving with any other test that
//! installs a deadline override or flips the global cancellation flag.

use super::*;
use crate::analysis::utils::deadline_override::DeadlineOverrideGuard;
use crate::types::DiscoverRecord;
use crate::{CreatureJson, NeuronJson, SynapseJson};
use serial_test::serial;

/// Samples per hidden neuron, above `MIN_DISCOVERY_SAMPLE_COUNT` (20).
const SAMPLES: u32 = 30;

/// Mean absolute error of a shallow hidden neuron (depth <= 2).
const SHALLOW_ERROR: f32 = 0.5;

/// Mean absolute error of an attenuated deep neuron. Its ratio to the shallow mean
/// is 0.1, below `ATTENUATION_THRESHOLD`.
const DEEP_ERROR: f32 = 0.05;

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

/// `SAMPLES` records for one hidden neuron, each carrying `error`.
fn hidden_records(uuid: &str, error: f32) -> (String, Vec<DiscoverRecord>) {
    let records = (0..SAMPLES)
        .map(|i| DiscoverRecord {
            obs_index: i,
            neuron_uuid: uuid.to_string(),
            value: None,
            activation: 0.5,
            errors: vec![error],
        })
        .collect();
    (uuid.to_string(), records)
}

/// `in0 -> h0 -> h1 -> h2 -> h3`. h0 and h1 are shallow, h2 and h3 are deep and
/// attenuated. The depth pass pops 5 neurons (1 input, 4 hidden), so it makes
/// 6 checks. The two deep targets (h2 at depth 3, h3 at depth 4) each yield one
/// candidate from `in0`.
fn chain_fixture() -> (CreatureJson, Vec<(String, Vec<DiscoverRecord>)>) {
    let creature = CreatureJson {
        neurons: vec![
            neuron("in0", "input"),
            neuron("h0", "hidden"),
            neuron("h1", "hidden"),
            neuron("h2", "hidden"),
            neuron("h3", "hidden"),
        ],
        synapses: vec![
            synapse("in0", "h0"),
            synapse("h0", "h1"),
            synapse("h1", "h2"),
            synapse("h2", "h3"),
        ],
        input: 1,
        output: 0,
    };
    let records = vec![
        hidden_records("h0", SHALLOW_ERROR),
        hidden_records("h1", SHALLOW_ERROR),
        hidden_records("h2", DEEP_ERROR),
        hidden_records("h3", DEEP_ERROR),
    ];
    (creature, records)
}

/// Depth-pass checks for `chain_fixture`: one before the pass plus one per popped neuron.
const CHAIN_DEPTH_PASS_CHECKS: usize = 6;

/// A deadline that fires on the first neuron popped in the depth pass stops the scan
/// before any neuron is processed. The result is empty, reports `DeadlinePassed`, and
/// records no step.
#[test]
#[serial]
fn deadline_on_first_popped_neuron_returns_no_candidates() {
    let (creature, records) = chain_fixture();

    // Positive control: with no override the fixture yields both deep targets.
    let control = detect_skip_connection_candidates_with_deadline(&creature, &records, &None);
    assert_eq!(
        control.candidates.len(),
        2,
        "fixture must yield one candidate per deep target"
    );
    assert_eq!(control.stop, None);

    // [false = pre-pass check, true = first pop check].
    let _guard = DeadlineOverrideGuard::with_sequence(vec![false, true]);
    let mut steps = 0_usize;
    let scan =
        detect_skip_connection_candidates_observed(&creature, &records, &None, || steps += 1);

    assert!(
        scan.candidates.is_empty(),
        "depths are incomplete, so no candidates may be returned"
    );
    assert_eq!(scan.stop, Some(SkipConnectionStop::DeadlinePassed));
    assert_eq!(
        steps, 0,
        "the first pop check must fire before any neuron is processed"
    );
}

/// The per-target check must stop the target loop after the depth pass has completed.
/// The partial result keeps the candidates gathered before the stop.
#[test]
#[serial]
fn deadline_at_per_target_check_keeps_completed_depth_pass_and_candidates_so_far() {
    let (creature, records) = chain_fixture();

    // The first target check fires. The depth pass completed, but no target was processed.
    {
        let mut sequence = vec![false; CHAIN_DEPTH_PASS_CHECKS];
        sequence.push(true);
        let _guard = DeadlineOverrideGuard::with_sequence(sequence);
        let scan = detect_skip_connection_candidates_with_deadline(&creature, &records, &None);
        assert!(
            scan.candidates.is_empty(),
            "no target was processed before the first target check fired"
        );
        assert_eq!(scan.stop, Some(SkipConnectionStop::DeadlinePassed));
    }

    // The first target is processed, then the second target check fires. The one
    // candidate already gathered must be kept.
    let mut sequence = vec![false; CHAIN_DEPTH_PASS_CHECKS];
    sequence.extend([false, true]);
    let _guard = DeadlineOverrideGuard::with_sequence(sequence);
    let scan = detect_skip_connection_candidates_with_deadline(&creature, &records, &None);
    assert_eq!(
        scan.candidates.len(),
        1,
        "the candidate gathered before the second target check must be kept, got: {:?}",
        scan.candidates
    );
    assert_eq!(scan.stop, Some(SkipConnectionStop::DeadlinePassed));
}

/// `in0 -> h0 -> ... -> h(ring_len-1) -> h0`: a cycle of `ring_len` hidden neurons
/// with every hidden neuron qualified.
fn ring_fixture(ring_len: usize) -> (CreatureJson, Vec<(String, Vec<DiscoverRecord>)>) {
    let mut neurons = vec![neuron("in0", "input")];
    let mut synapses = vec![synapse("in0", "h0")];
    let mut records = Vec::new();
    for i in 0..ring_len {
        let uuid = format!("h{i}");
        let next = format!("h{}", (i + 1) % ring_len);
        neurons.push(neuron(&uuid, "hidden"));
        synapses.push(synapse(&uuid, &next));
        records.push(hidden_records(&uuid, SHALLOW_ERROR));
    }
    let creature = CreatureJson {
        neurons,
        synapses,
        input: 1,
        output: 0,
    };
    (creature, records)
}

/// A 2-node cycle must return no candidates and report `CycleDetected`, and the call must
/// terminate.
#[test]
#[serial]
fn two_node_cycle_returns_cycle_detected() {
    let (creature, records) = ring_fixture(2);
    let scan = detect_skip_connection_candidates_with_deadline(&creature, &records, &None);
    assert!(scan.candidates.is_empty());
    assert_eq!(scan.stop, Some(SkipConnectionStop::CycleDetected));
}

/// A 3-node cycle must return no candidates and report `CycleDetected`, and the call must
/// terminate.
#[test]
#[serial]
fn three_node_cycle_returns_cycle_detected() {
    let (creature, records) = ring_fixture(3);
    let scan = detect_skip_connection_candidates_with_deadline(&creature, &records, &None);
    assert!(scan.candidates.is_empty());
    assert_eq!(scan.stop, Some(SkipConnectionStop::CycleDetected));
}

/// The chain `in0 -> h0 -> h1 -> h2 -> h3` with a tail cycle `h3 -> x -> h3`. The
/// pass gives h2 and h3 depths before stalling on `h3` (in-degree 2, one edge
/// unrelaxed). Those depths would produce candidates if the cycle were ignored, so this
/// test shows the cycle check, not the data, empties the result.
#[test]
#[serial]
fn cycle_downstream_of_deep_chain_discards_partial_depths() {
    let (mut creature, mut records) = chain_fixture();
    creature.neurons.push(neuron("x", "hidden"));
    creature.synapses.push(synapse("h3", "x"));
    creature.synapses.push(synapse("x", "h3"));
    records.push(hidden_records("x", DEEP_ERROR));

    let scan = detect_skip_connection_candidates_with_deadline(&creature, &records, &None);
    assert!(
        scan.candidates.is_empty(),
        "a cycle must not expose partial depths, got: {:?}",
        scan.candidates
    );
    assert_eq!(scan.stop, Some(SkipConnectionStop::CycleDetected));
}

/// Reference copy of the pre-#2368 depth pass: a forward BFS that takes the maximum
/// depth for each neuron. It is only used on acyclic fixtures, because it never
/// terminates on a cycle.
fn legacy_depths(creature: &CreatureJson) -> HashMap<String, usize> {
    let mut forward_adj: HashMap<&str, Vec<&str>> = HashMap::new();
    for s in &creature.synapses {
        forward_adj
            .entry(s.from_uuid.as_str())
            .or_default()
            .push(s.to_uuid.as_str());
    }

    let mut depths: HashMap<String, usize> = HashMap::new();
    let mut queue: VecDeque<(&str, usize)> = VecDeque::new();
    for n in &creature.neurons {
        if n.neuron_type == "input" {
            depths.insert(n.uuid.clone(), 0);
            queue.push_back((n.uuid.as_str(), 0));
        }
    }
    while let Some((current, depth)) = queue.pop_front() {
        if let Some(successors) = forward_adj.get(current) {
            for &succ in successors {
                let new_depth = depth + 1;
                let entry = depths.entry(succ.to_string()).or_insert(0);
                if new_depth > *entry {
                    *entry = new_depth;
                    queue.push_back((succ, new_depth));
                }
            }
        }
    }
    depths
}

/// Reference copy of the pre-#2368 detection: the forward BFS above, then the nested
/// "strict > gap, first wins" source loop. Returns `(target, source, depth_gap)` sorted.
fn legacy_candidates(
    creature: &CreatureJson,
    records: &[(String, Vec<DiscoverRecord>)],
) -> Vec<(String, String, usize)> {
    let depths = legacy_depths(creature);
    let hidden: HashSet<&str> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "hidden")
        .map(|n| n.uuid.as_str())
        .collect();

    let mut mean_error: HashMap<&str, f32> = HashMap::new();
    for (uuid, recs) in records {
        if !hidden.contains(uuid.as_str()) || recs.len() < MIN_DISCOVERY_SAMPLE_COUNT {
            continue;
        }
        let total: f32 = recs
            .iter()
            .flat_map(|r| r.errors.iter())
            .map(|e| e.abs())
            .sum();
        let count = recs.iter().map(|r| r.errors.len()).sum::<usize>() as f32;
        mean_error.insert(uuid.as_str(), if count > 0.0 { total / count } else { 0.0 });
    }

    let mut qualified: Vec<&str> = mean_error.keys().copied().collect();
    qualified.sort();

    let shallow_errors: Vec<f32> = qualified
        .iter()
        .copied()
        .filter(|u| depths.get(*u).copied().unwrap_or(0) <= 2)
        .filter_map(|u| mean_error.get(u).copied())
        .filter(|&e| e > 0.0)
        .collect();
    if shallow_errors.is_empty() {
        return Vec::new();
    }
    let shallow_mean = shallow_errors.iter().sum::<f32>() / shallow_errors.len() as f32;
    if shallow_mean <= 0.0 {
        return Vec::new();
    }

    let existing: HashSet<(&str, &str)> = creature
        .synapses
        .iter()
        .map(|s| (s.from_uuid.as_str(), s.to_uuid.as_str()))
        .collect();

    let sources: Vec<(&str, usize)> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "input" || n.neuron_type == "hidden")
        .filter_map(|n| {
            let d = depths.get(n.uuid.as_str()).copied().unwrap_or(0);
            (d <= 1).then_some((n.uuid.as_str(), d))
        })
        .collect();

    let mut out = Vec::new();
    for &u in &qualified {
        let Some(&depth) = depths.get(u) else {
            continue;
        };
        if depth < MIN_DEEP_DEPTH {
            continue;
        }
        let err = mean_error[u];
        if err / shallow_mean >= ATTENUATION_THRESHOLD {
            continue;
        }
        let mut best: Option<(&str, usize)> = None;
        let mut best_gap = 0_usize;
        for &(src, src_depth) in &sources {
            if existing.contains(&(src, u)) {
                continue;
            }
            let gap = depth.saturating_sub(src_depth);
            if gap > best_gap {
                best_gap = gap;
                best = Some((src, src_depth));
            }
        }
        if let Some((src, _)) = best {
            let severity = 1.0 - err / shallow_mean;
            if best_gap as f32 * severity * 0.005 > 0.0 {
                out.push((u.to_string(), src.to_string(), best_gap));
            }
        }
    }
    out.sort();
    out
}

/// Acyclic fixture covering four cases:
/// - a diamond (`in0 -> ha -> hb` and `in0 -> hb`), so `hb`'s longest-path depth is 2;
/// - an unreachable hidden neuron `hu` (no path from any input), which counts as depth 0;
/// - an existing synapse `in0 -> hc`, so `hc` skips `in0` and takes `in1`;
/// - `hd` has every depth-0 source already connected, so it falls back to the depth-1 `ha`.
fn acyclic_fixture() -> (CreatureJson, Vec<(String, Vec<DiscoverRecord>)>) {
    let creature = CreatureJson {
        neurons: vec![
            neuron("in0", "input"),
            neuron("in1", "input"),
            neuron("ha", "hidden"),
            neuron("hb", "hidden"),
            neuron("hc", "hidden"),
            neuron("hd", "hidden"),
            neuron("hu", "hidden"),
            neuron("out0", "output"),
        ],
        synapses: vec![
            synapse("in0", "ha"),
            synapse("ha", "hb"),
            synapse("in0", "hb"),
            synapse("hb", "hc"),
            synapse("hu", "hc"),
            synapse("hc", "hd"),
            synapse("hb", "hd"),
            synapse("hd", "out0"),
            synapse("in0", "hc"),
            synapse("in0", "hd"),
            synapse("in1", "hd"),
            synapse("hu", "hd"),
        ],
        input: 2,
        output: 1,
    };
    let records = vec![
        hidden_records("ha", SHALLOW_ERROR),
        hidden_records("hb", SHALLOW_ERROR),
        hidden_records("hu", SHALLOW_ERROR),
        hidden_records("hc", DEEP_ERROR),
        hidden_records("hd", DEEP_ERROR),
    ];
    (creature, records)
}

/// On an acyclic graph the Kahn pass and indexed target loop must return exactly the
/// `(target, source, depth_gap)` set of the legacy BFS and nested loop, and the fixture
/// must exercise the fallback path so the comparison is not vacuous.
#[test]
#[serial]
fn acyclic_result_matches_legacy_algorithm() {
    let (creature, records) = acyclic_fixture();

    let scan = detect_skip_connection_candidates_with_deadline(&creature, &records, &None);
    assert_eq!(scan.stop, None);

    let mut actual: Vec<(String, String, usize)> = scan
        .candidates
        .iter()
        .map(|c| (c.target_uuid.clone(), c.source_uuid.clone(), c.depth_gap))
        .collect();
    actual.sort();

    let legacy = legacy_candidates(&creature, &records);
    assert_eq!(
        actual, legacy,
        "indexed result must match the legacy algorithm"
    );

    assert_eq!(
        actual,
        vec![
            ("hc".to_string(), "in1".to_string(), 3),
            ("hd".to_string(), "ha".to_string(), 3),
        ],
        "fixture must exercise the next-source and depth-1 fallback paths"
    );
}
