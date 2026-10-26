//! Issue #2368 unit tests for the skip-connection scan bounds: single-pass
//! depth computation, fail-closed cycle handling, deadline-bounded scanning and
//! indexed shallow-source selection.
//!
//! These live in-crate because the `deadline_override` seam is `#[cfg(test)]`
//! only and cannot be used from the `tests/` integration crate.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for synthetic activation waveforms (Issue #873)

use super::*;
use crate::analysis::utils::deadline_override::DeadlineOverrideGuard;
use crate::{NeuronJson, SynapseJson};
use serial_test::serial;
use std::time::{Duration, SystemTime};

const SAMPLES: u32 = 30;

/// Build a creature from `(uuid, neuron_type)` pairs and `(from, to)` pairs.
fn build_creature<S: AsRef<str>>(neurons: &[(S, &str)], synapses: &[(S, S)]) -> CreatureJson {
    CreatureJson {
        neurons: neurons
            .iter()
            .map(|(uuid, neuron_type)| NeuronJson {
                uuid: uuid.as_ref().to_string(),
                neuron_type: neuron_type.to_string(),
                squash: "TANH".to_string(),
                bias: 0.0,
            })
            .collect(),
        synapses: synapses
            .iter()
            .map(|(from, to)| SynapseJson {
                from_uuid: from.as_ref().to_string(),
                to_uuid: to.as_ref().to_string(),
                weight: 0.5,
                synapse_type: None,
            })
            .collect(),
        input: 1,
        output: 1,
    }
}

/// Records for `uuid` whose mean absolute error is roughly `error`.
fn records_for(uuid: &str, error: f32) -> (String, Vec<DiscoverRecord>) {
    let records = (0..SAMPLES)
        .map(|i| DiscoverRecord {
            obs_index: i,
            neuron_uuid: uuid.to_string(),
            value: None,
            activation: (i as f32 * 0.1).sin(),
            errors: vec![error * (1.0 + (i as f32 * 0.05).sin())],
        })
        .collect();
    (uuid.to_string(), records)
}

/// Naive reference: memoised recursive longest path from inputs.
fn naive_depths(creature: &CreatureJson) -> HashMap<String, usize> {
    let inputs: HashSet<&str> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "input")
        .map(|n| n.uuid.as_str())
        .collect();
    let mut preds: HashMap<&str, Vec<&str>> = HashMap::new();
    for s in &creature.synapses {
        preds
            .entry(s.to_uuid.as_str())
            .or_default()
            .push(s.from_uuid.as_str());
    }
    let mut nodes: HashSet<&str> = HashSet::new();
    for n in &creature.neurons {
        nodes.insert(n.uuid.as_str());
    }
    for s in &creature.synapses {
        nodes.insert(s.from_uuid.as_str());
        nodes.insert(s.to_uuid.as_str());
    }

    let mut memo: HashMap<String, Option<usize>> = HashMap::new();
    let mut out = HashMap::new();
    for node in nodes {
        if let Some(d) = longest_path(node, &inputs, &preds, &mut memo) {
            out.insert(node.to_string(), d);
        }
    }
    out
}

fn longest_path(
    node: &str,
    inputs: &HashSet<&str>,
    preds: &HashMap<&str, Vec<&str>>,
    memo: &mut HashMap<String, Option<usize>>,
) -> Option<usize> {
    if let Some(known) = memo.get(node) {
        return *known;
    }
    let mut best: Option<usize> = inputs.contains(node).then_some(0_usize);
    if let Some(parents) = preds.get(node) {
        for &parent in parents {
            if let Some(d) = longest_path(parent, inputs, preds, memo) {
                best = Some(best.map_or(d + 1, |b| b.max(d + 1)));
            }
        }
    }
    memo.insert(node.to_string(), best);
    best
}

/// Chain `in-0 -> s0 -> m1 -> d3 -> d4 -> d5`: `s0` and `m1` are shallow with
/// error 0.8, and `d3`, `d4`, `d5` are deep and attenuated with error 0.05.
/// Each deep target gets one candidate, sourced from `in-0`.
fn chain_creature() -> (CreatureJson, Vec<(String, Vec<DiscoverRecord>)>) {
    let creature = build_creature(
        &[
            ("in-0", "input"),
            ("s0", "hidden"),
            ("m1", "hidden"),
            ("d3", "hidden"),
            ("d4", "hidden"),
            ("d5", "hidden"),
        ],
        &[
            ("in-0", "s0"),
            ("s0", "m1"),
            ("m1", "d3"),
            ("d3", "d4"),
            ("d4", "d5"),
        ],
    );
    let records = vec![
        records_for("s0", 0.8),
        records_for("m1", 0.8),
        records_for("d3", 0.05),
        records_for("d4", 0.05),
        records_for("d5", 0.05),
    ];
    (creature, records)
}

#[test]
fn depths_match_naive_longest_path_on_dense_layered_dag() {
    const LAYERS: usize = 5;
    const WIDTH: usize = 6;

    let mut neurons: Vec<(String, &str)> = Vec::new();
    for layer in 0..LAYERS {
        let kind = if layer == 0 { "input" } else { "hidden" };
        for k in 0..WIDTH {
            neurons.push((format!("L{layer}-{k}"), kind));
        }
    }
    // Unreachable hidden neuron: no incoming synapses from any input.
    neurons.push(("orphan".to_string(), "hidden"));

    let mut synapses: Vec<(String, String)> = Vec::new();
    for layer in 0..LAYERS - 1 {
        for a in 0..WIDTH {
            for b in 0..WIDTH {
                synapses.push((format!("L{layer}-{a}"), format!("L{}-{b}", layer + 1)));
            }
        }
    }
    // Skip edges bridging two or more layers.
    for (from, to) in [("L0-0", "L3-0"), ("L1-2", "L4-5"), ("L0-5", "L2-3")] {
        synapses.push((from.to_string(), to.to_string()));
    }
    // Duplicate synapse, edge out of the unreachable neuron, and an endpoint
    // that is not declared in `neurons`.
    synapses.push(("L0-0".to_string(), "L1-0".to_string()));
    synapses.push(("orphan".to_string(), "L1-1".to_string()));
    synapses.push(("L4-0".to_string(), "ghost".to_string()));

    let creature = build_creature(&neurons, &synapses);

    let scan = compute_depths_from_inputs(&creature);
    assert!(!scan.cycle_detected, "a DAG must not report a cycle");
    assert_eq!(scan.depths, naive_depths(&creature));
    assert_eq!(scan.depths.get("L4-0"), Some(&4));
    assert_eq!(scan.depths.get("ghost"), Some(&5));
    assert!(
        !scan.depths.contains_key("orphan"),
        "neurons unreachable from any input must get no depth"
    );
}

#[test]
fn cyclic_creature_sets_depth_cycle_detected_and_returns_no_candidates() {
    let creature = build_creature(
        &[
            ("in-0", "input"),
            ("h0", "hidden"),
            ("h1", "hidden"),
            ("h2", "hidden"),
            ("h3", "hidden"),
            ("h4", "hidden"),
        ],
        &[
            ("in-0", "h0"),
            ("h0", "h1"),
            ("h1", "h2"),
            ("h2", "h3"),
            ("h3", "h4"),
            ("h4", "h3"),
        ],
    );
    assert!(compute_depths_from_inputs(&creature).cycle_detected);

    let records = vec![
        records_for("h0", 0.8),
        records_for("h1", 0.6),
        records_for("h2", 0.08),
        records_for("h3", 0.05),
        records_for("h4", 0.03),
    ];
    let scan = detect_skip_connection_candidates_with_deadline(&creature, &records, &None);
    assert!(scan.depth_cycle_detected);
    assert!(scan.candidates.is_empty());
    assert!(scan.truncation.is_none());
}

#[test]
#[serial]
fn elapsed_deadline_mid_scan_keeps_partial_candidates_and_reports_truncation() {
    let (creature, records) = chain_creature();
    let full = detect_skip_connection_candidates(&creature, &records);
    assert_eq!(full.len(), 3, "each deep target should yield one candidate");

    // The first per-target check passes; the second reports the deadline elapsed.
    let _guard = DeadlineOverrideGuard::with_sequence(vec![false, true]);
    let scan = detect_skip_connection_candidates_with_deadline(&creature, &records, &None);

    assert!(matches!(
        scan.truncation,
        Some(ScanTruncation::DeadlinePassed)
    ));
    assert_eq!(
        scan.candidates.len(),
        1,
        "only the first deep target is scanned before the deadline"
    );
    let kept = &scan.candidates[0];
    assert!(
        full.iter()
            .any(|f| f.target_uuid == kept.target_uuid && f.depth_gap == kept.depth_gap),
        "the partial candidate must be one the unbounded scan also finds"
    );
}

#[test]
#[serial]
fn far_future_deadline_matches_unbounded_scan_without_truncation() {
    let (creature, records) = chain_creature();
    let far = SystemTime::now() + Duration::from_secs(3600);

    let bounded = detect_skip_connection_candidates_with_deadline(&creature, &records, &Some(far));
    let unbounded = detect_skip_connection_candidates(&creature, &records);

    assert!(bounded.truncation.is_none());
    assert!(!bounded.depth_cycle_detected);
    let key =
        |c: &SkipConnectionCandidate| (c.target_uuid.clone(), c.source_uuid.clone(), c.depth_gap);
    let bounded_keys: Vec<_> = bounded.candidates.iter().map(key).collect();
    let unbounded_keys: Vec<_> = unbounded.iter().map(key).collect();
    assert_eq!(bounded_keys.len(), 3);
    assert_eq!(bounded_keys, unbounded_keys);
}

#[test]
fn shallow_source_selection_skips_connected_sources_and_falls_back_to_depth_one() {
    let creature = build_creature(
        &[
            ("in-0", "input"),
            ("in-1", "input"),
            ("s0", "hidden"),
            ("m1", "hidden"),
            ("d3", "hidden"),
            ("d4", "hidden"),
        ],
        &[
            ("in-0", "s0"),
            ("s0", "m1"),
            ("m1", "d3"),
            ("d3", "d4"),
            // d3 is already connected from the first depth-0 input.
            ("in-0", "d3"),
            // d4 is already connected from every depth-0 input.
            ("in-0", "d4"),
            ("in-1", "d4"),
        ],
    );
    let records = vec![
        records_for("s0", 0.8),
        records_for("m1", 0.8),
        records_for("d3", 0.05),
        records_for("d4", 0.05),
    ];
    let candidates = detect_skip_connection_candidates(&creature, &records);

    let d3 = candidates
        .iter()
        .find(|c| c.target_uuid == "d3")
        .expect("d3 should have a candidate");
    assert_eq!(d3.source_uuid, "in-1");
    assert_eq!((d3.source_depth, d3.depth_gap), (0, 3));

    let d4 = candidates
        .iter()
        .find(|c| c.target_uuid == "d4")
        .expect("d4 should have a candidate");
    assert_eq!(d4.source_uuid, "s0");
    assert_eq!((d4.source_depth, d4.depth_gap), (1, 3));
}
