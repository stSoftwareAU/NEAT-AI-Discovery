//! Issue #2367 (CWE-400): unit tests for the bounded compound degradation scan.
//!
//! These live in-crate rather than in `tests/` so they can reach the private
//! `reachable_outputs` helper and drive `deadline_passed` through the
//! `#[cfg(test)]` deadline override seam. `#[serial]` keeps them from
//! interleaving with other tests that touch the global cancellation flag.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for synthetic fixtures (Issue #873)

use super::*;
use crate::analysis::utils::deadline_override::DeadlineOverrideGuard;
use crate::{NeuronJson, SynapseJson};
use serial_test::serial;
use std::time::Duration;

const SAMPLES: u32 = 40;

/// `hidden_count` hidden neurons, each fed by `in-0` and feeding `out-0`, with
/// records engineered so every hidden neuron yields a bias correction and every
/// hidden -> out-0 synapse yields a weight correction. Every (bias, weight)
/// pair is on a connected path, so the scan emits `hidden_count²` candidates.
fn fixture(hidden_count: usize) -> (CreatureJson, Vec<(String, Vec<DiscoverRecord>)>) {
    let mut neurons = vec![NeuronJson {
        uuid: "in-0".to_string(),
        neuron_type: "input".to_string(),
        squash: "IDENTITY".to_string(),
        bias: 0.0,
    }];
    let mut synapses = Vec::new();
    let mut records = Vec::new();

    for h in 0..hidden_count {
        let uuid = format!("h{h}");
        neurons.push(NeuronJson {
            uuid: uuid.clone(),
            neuron_type: "hidden".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        });
        synapses.push(SynapseJson {
            from_uuid: "in-0".to_string(),
            to_uuid: uuid.clone(),
            weight: 1.0,
            synapse_type: None,
        });
        synapses.push(SynapseJson {
            from_uuid: uuid.clone(),
            to_uuid: "out-0".to_string(),
            weight: 0.5,
            synapse_type: None,
        });
        // Consistent, varying-by-neuron bias offset (>= MIN_BIAS_ERROR).
        let bias_error = 0.1 + 0.01 * h as f32;
        let recs = (0..SAMPLES)
            .map(|o| DiscoverRecord {
                obs_index: o,
                neuron_uuid: uuid.clone(),
                value: Some(0.0),
                activation: 1.0 + 0.5 * (o as f32 * 0.7).sin(),
                errors: vec![bias_error],
            })
            .collect();
        records.push((uuid, recs));
    }

    neurons.push(NeuronJson {
        uuid: "out-0".to_string(),
        neuron_type: "output".to_string(),
        squash: "IDENTITY".to_string(),
        bias: 0.0,
    });

    // Continuous (non-quantised) output error shared by every hidden -> out-0
    // synapse, so each one has a positive least-squares weight delta.
    let out_recs = (0..SAMPLES)
        .map(|o| DiscoverRecord {
            obs_index: o,
            neuron_uuid: "out-0".to_string(),
            value: None,
            activation: 0.0,
            errors: vec![0.2 + 0.1 * (o as f32 * 0.9 + 0.5).sin()],
        })
        .collect();
    records.push(("out-0".to_string(), out_recs));

    let creature = CreatureJson {
        neurons,
        synapses,
        input: 1,
        output: 1,
    };
    (creature, records)
}

/// A deadline value is required by the signature. Under an override queue the
/// real value is only a fallback that must not be consulted while the queue
/// still has entries; a far-future value keeps the live check quiet afterwards.
fn live_deadline() -> Option<SystemTime> {
    Some(SystemTime::now() + Duration::from_secs(86_400))
}

/// Naive reference: enumerate every walk of 1..=`depth` edges from `start`
/// (no visited set, exponential in depth) and collect the outputs it ends on.
fn naive_reachable(
    start: &str,
    down: &HashMap<&str, HashSet<&str>>,
    outputs: &HashSet<&str>,
    depth: usize,
) -> HashSet<String> {
    fn walk(
        node: &str,
        down: &HashMap<&str, HashSet<&str>>,
        outputs: &HashSet<&str>,
        remaining: usize,
        acc: &mut HashSet<String>,
    ) {
        if remaining == 0 {
            return;
        }
        let Some(children) = down.get(node) else {
            return;
        };
        for &child in children {
            if outputs.contains(child) {
                acc.insert(child.to_string());
            }
            walk(child, down, outputs, remaining - 1, acc);
        }
    }

    let mut acc = HashSet::new();
    walk(start, down, outputs, depth, &mut acc);
    acc
}

/// Layered DAG with dense fan-out: `layers` layers of `width` nodes, every node
/// wired to every node in the next layer. Outputs are the last two layers plus
/// `L0-0`, which is reached again through back-edges (a cycle), so the
/// start-as-output case is covered too.
fn layered_dag(width: usize, layers: usize) -> (HashMap<String, HashSet<String>>, HashSet<String>) {
    let node = |l: usize, i: usize| format!("L{l}-{i}");
    let mut edges: HashMap<String, HashSet<String>> = HashMap::new();
    for l in 0..layers - 1 {
        for i in 0..width {
            for j in 0..width {
                edges.entry(node(l, i)).or_default().insert(node(l + 1, j));
            }
        }
    }
    // Back-edges into L0-0: a depth-3 cycle L0-0 -> L1-x -> L2-0 -> L0-0.
    edges.entry(node(2, 0)).or_default().insert(node(0, 0));
    edges
        .entry(node(layers - 1, 0))
        .or_default()
        .insert(node(0, 0));

    let mut outputs: HashSet<String> = HashSet::new();
    for i in 0..width {
        outputs.insert(node(layers - 2, i));
        outputs.insert(node(layers - 1, i));
    }
    outputs.insert(node(0, 0));
    (edges, outputs)
}

/// Issue #2367: the visited-set `reachable_outputs` must return exactly the
/// set of outputs a naive path enumeration reaches within 3 hops, including
/// the cycle case where the start node is itself an output.
#[test]
#[serial]
fn reachable_outputs_matches_naive_path_enumeration_on_dense_layered_dag() {
    let (owned_edges, owned_outputs) = layered_dag(30, 5);
    let down: HashMap<&str, HashSet<&str>> = owned_edges
        .iter()
        .map(|(k, v)| (k.as_str(), v.iter().map(String::as_str).collect()))
        .collect();
    let outputs: HashSet<&str> = owned_outputs.iter().map(String::as_str).collect();

    let mut starts: Vec<String> = (0..30).map(|i| format!("L0-{i}")).collect();
    starts.push("L2-0".to_string());
    starts.push("L4-0".to_string());

    for start in &starts {
        let fast: HashSet<String> = reachable_outputs(start, &down, &outputs, 3)
            .into_iter()
            .map(str::to_string)
            .collect();
        let naive = naive_reachable(start, &down, &outputs, 3);
        assert_eq!(
            fast, naive,
            "visited-set reachability diverged from naive enumeration from {start}"
        );
    }

    let from_origin = reachable_outputs("L0-0", &down, &outputs, 3);
    assert!(
        from_origin.contains("L0-0"),
        "the start node must be reported when a depth-3 cycle returns to it"
    );
    assert!(
        from_origin.contains("L3-0"),
        "layer 3 outputs are three edges from L0-0 and must be reachable"
    );
    assert!(
        !from_origin.contains("L4-0"),
        "layer 4 outputs are four edges from L0-0 and must lie outside the depth limit"
    );
}

/// Issue #2367: an override queue of `[true]` reports the deadline as passed
/// on the first check, so the scan stops before any candidate is emitted.
#[test]
#[serial]
fn elapsed_deadline_truncates_scan_with_deadline_passed() {
    let (creature, records) = fixture(4);
    let unbounded = detect_compound_bias_weight_degradations(&creature, &records);
    assert!(
        !unbounded.is_empty(),
        "fixture must yield candidates so the truncation check is non-vacuous"
    );

    let _guard = DeadlineOverrideGuard::with_sequence(vec![true]);
    let scan = detect_compound_bias_weight_degradations_with_deadline(
        &creature,
        &records,
        &live_deadline(),
    );

    assert_eq!(scan.truncation, Some(ScanTruncation::DeadlinePassed));
    assert!(
        scan.candidates.len() < unbounded.len(),
        "a deadline-truncated scan must return fewer candidates than the unbounded scan, \
         got {} vs {}",
        scan.candidates.len(),
        unbounded.len()
    );
}

/// Issue #2367: the outer-row deadline check must stop the pair loop after the
/// rows already completed, keeping their candidates. The precompute loop makes
/// one check per distinct neuron (`hidden + out-0`), then row 0 proceeds and
/// row 1 reports the deadline passed.
#[test]
#[serial]
fn outer_row_deadline_check_keeps_candidates_from_completed_rows() {
    let hidden = 4;
    let (creature, records) = fixture(hidden);

    let mut sequence = vec![false; hidden + 1 + 1];
    sequence.push(true);
    let _guard = DeadlineOverrideGuard::with_sequence(sequence);
    let scan = detect_compound_bias_weight_degradations_with_deadline(
        &creature,
        &records,
        &live_deadline(),
    );

    assert_eq!(scan.truncation, Some(ScanTruncation::DeadlinePassed));
    assert_eq!(
        scan.candidates.len(),
        hidden,
        "row 0 pairs with every weight correction must be kept before row 1 stops"
    );
}

/// Issue #2367: a 24-hidden fixture yields 576 valid pairs, above the
/// 256-candidate ceiling, so exactly the ceiling is returned, sorted by
/// estimated improvement (descending), with `CandidateCeiling` reported.
#[test]
#[serial]
fn candidate_ceiling_caps_emitted_candidates_and_reports_truncation() {
    let (creature, records) = fixture(24);

    let scan = detect_compound_bias_weight_degradations_with_deadline(&creature, &records, &None);

    assert_eq!(scan.candidates.len(), MAX_COMPOUND_DEGRADATION_CANDIDATES);
    assert_eq!(scan.truncation, Some(ScanTruncation::CandidateCeiling));
    assert!(
        scan.candidates
            .windows(2)
            .all(|w| w[0].estimated_improvement >= w[1].estimated_improvement),
        "capped candidates must remain sorted by estimated improvement, descending"
    );
}

/// Issue #2367: with a far-future deadline on a small fixture, nothing is
/// truncated and the result matches the deadline-free wrapper exactly.
#[test]
#[serial]
fn far_future_deadline_matches_unbounded_scan_without_truncation() {
    let (creature, records) = fixture(6);

    let scan = detect_compound_bias_weight_degradations_with_deadline(
        &creature,
        &records,
        &live_deadline(),
    );
    let unbounded = detect_compound_bias_weight_degradations(&creature, &records);

    assert!(
        scan.truncation.is_none(),
        "no truncation expected, got {:?}",
        scan.truncation
    );
    assert_eq!(scan.candidates.len(), unbounded.len());

    // The weight statistics are summed over `HashMap` iteration, so tied
    // candidates can swap places and differ by an ULP between calls. Compare
    // by canonical key with a float tolerance instead of by position.
    let key = |c: &CompoundDegradationCandidate| {
        (
            c.bias_neuron_uuid.clone(),
            c.weight_from_uuid.clone(),
            c.weight_to_uuid.clone(),
        )
    };
    let mut got: Vec<&CompoundDegradationCandidate> = scan.candidates.iter().collect();
    let mut want: Vec<&CompoundDegradationCandidate> = unbounded.iter().collect();
    got.sort_by_key(|c| key(c));
    want.sort_by_key(|c| key(c));
    for (a, b) in got.iter().zip(&want) {
        assert_eq!(key(a), key(b));
        assert_close(a.recommended_bias, b.recommended_bias);
        assert_close(a.recommended_weight, b.recommended_weight);
        assert_close(a.estimated_improvement, b.estimated_improvement);
    }
}

/// Float comparison with a relative tolerance wide enough for ULP-level drift.
fn assert_close(a: f32, b: f32) {
    assert!(
        (a - b).abs() <= 1e-5 * b.abs().max(1.0),
        "values differ beyond tolerance: {a} vs {b}"
    );
}
