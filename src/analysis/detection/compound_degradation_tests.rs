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

/// Frontier pushes made by `reachable_outputs` from `L0-0` over a 4-layer
/// `layered_dag` of the given width, at the production depth of 3.
fn walk_pushes(width: usize) -> usize {
    let (owned_edges, owned_outputs) = layered_dag(width, 4);
    let down: HashMap<&str, HashSet<&str>> = owned_edges
        .iter()
        .map(|(k, v)| (k.as_str(), v.iter().map(String::as_str).collect()))
        .collect();
    let outputs: HashSet<&str> = owned_outputs.iter().map(String::as_str).collect();
    let mut pushes = 0_usize;
    reachable_outputs_observed("L0-0", &down, &outputs, 3, &mut || pushes += 1);
    pushes
}

/// Issue #2367: the visited set is what removes the F³ factor. With it, each
/// neuron is pushed at most once, so pushes grow linearly with fan-out
/// (3N + 1 here). Without it, every walk is pushed, about N + N² + N³.
#[test]
fn reachable_outputs_pushes_grow_linearly_with_fan_out() {
    let n = 4;
    let pushes_n = walk_pushes(n);
    let pushes_4n = walk_pushes(4 * n);

    assert!(pushes_n > 0, "the walk must push something at fan-out {n}");
    assert!(
        pushes_4n <= 4 * pushes_n,
        "frontier pushes must grow at most linearly with fan-out: {pushes_n} at fan-out {n}, \
         {pushes_4n} at fan-out {} (a walk without a visited set grows as fan-out cubed)",
        4 * n
    );
}

/// Issue #2367: the 256-candidate ceiling keeps the strongest candidates. The
/// reference is every (bias, weight) pair's combined improvement from the same
/// detectors (every pair in `fixture` is connected), so the weakest kept
/// candidate must be at least as strong as the strongest dropped one.
#[test]
#[serial]
fn candidate_ceiling_keeps_the_strongest_candidates() {
    let (creature, records) = fixture(24);
    let scan = detect_compound_bias_weight_degradations_with_deadline(&creature, &records, &None);
    assert_eq!(scan.candidates.len(), MAX_COMPOUND_DEGRADATION_CANDIDATES);

    let records_map = build_record_map(&records);
    let squash: HashMap<&str, &str> = creature
        .neurons
        .iter()
        .map(|n| (n.uuid.as_str(), n.squash.as_str()))
        .collect();
    let bias: HashMap<&str, f32> = creature
        .neurons
        .iter()
        .map(|n| (n.uuid.as_str(), n.bias))
        .collect();
    let bias_corrections = detect_bias_corrections(&creature, &records_map, &squash, &bias);
    let weight_corrections = detect_weight_corrections(&creature, &records_map);
    let mut all: Vec<f32> = bias_corrections
        .iter()
        .flat_map(|b| {
            weight_corrections
                .iter()
                .map(move |w| b.estimated_improvement + w.estimated_improvement * 0.7)
        })
        .collect();
    assert_eq!(
        all.len(),
        576,
        "fixture(24) must give 24 x 24 connected pairs"
    );
    all.sort_by(|a, b| b.total_cmp(a));

    let strongest_dropped = all[MAX_COMPOUND_DEGRADATION_CANDIDATES];
    let weakest_kept = scan
        .candidates
        .last()
        .expect("the capped scan keeps candidates")
        .estimated_improvement;
    assert!(
        weakest_kept >= strongest_dropped - 1e-5,
        "the ceiling must keep the strongest candidates: weakest kept {weakest_kept}, \
         strongest dropped {strongest_dropped}"
    );
}

/// `SAMPLES` records for a hidden neuron with a constant error `error` and the
/// `fixture` activation pattern.
fn hidden_records(uuid: &str, error: f32) -> (String, Vec<DiscoverRecord>) {
    let recs = (0..SAMPLES)
        .map(|o| DiscoverRecord {
            obs_index: o,
            neuron_uuid: uuid.to_string(),
            value: Some(0.0),
            activation: 1.0 + 0.5 * (o as f32 * 0.7).sin(),
            errors: vec![error],
        })
        .collect();
    (uuid.to_string(), recs)
}

/// The continuous `out-0` error shared with `fixture`.
fn out_error(o: u32) -> f32 {
    0.2 + 0.1 * (o as f32 * 0.9 + 0.5).sin()
}

fn out_records() -> (String, Vec<DiscoverRecord>) {
    let recs = (0..SAMPLES)
        .map(|o| DiscoverRecord {
            obs_index: o,
            neuron_uuid: "out-0".to_string(),
            value: None,
            activation: 0.0,
            errors: vec![out_error(o)],
        })
        .collect();
    ("out-0".to_string(), recs)
}

fn test_neuron(uuid: &str, neuron_type: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
        squash: "IDENTITY".to_string(),
        bias: 0.0,
    }
}

fn test_synapse(from: &str, to: &str) -> SynapseJson {
    SynapseJson {
        from_uuid: from.to_string(),
        to_uuid: to.to_string(),
        weight: 0.5,
        synapse_type: None,
    }
}

/// Issue #2367: more than `MAX_COMPOUND_BIAS_CORRECTIONS` bias corrections are
/// capped to the strongest before the pair loop. 300 hidden neurons each yield a
/// bias correction (error rising with the index), but only the weakest (`h0`)
/// and the strongest (`h299`) feed `out-0`, so only they pair with the two
/// weight corrections. The cap drops `h0`, which leaves 2 candidates (under the
/// 256-candidate ceiling), so only the correction cap can report
/// `CandidateCeiling` or exclude `h0`.
#[test]
#[serial]
fn bias_correction_cap_keeps_the_strongest_and_reports_candidate_ceiling() {
    const HIDDEN: usize = 300;
    let mut neurons = Vec::new();
    let mut records = Vec::new();
    for h in 0..HIDDEN {
        let uuid = format!("h{h}");
        neurons.push(test_neuron(&uuid, "hidden"));
        records.push(hidden_records(&uuid, 0.1 + 0.01 * h as f32));
    }
    neurons.push(test_neuron("out-0", "output"));
    records.push(out_records());
    let creature = CreatureJson {
        neurons,
        synapses: vec![test_synapse("h0", "out-0"), test_synapse("h299", "out-0")],
        input: 0,
        output: 1,
    };

    let scan = detect_compound_bias_weight_degradations_with_deadline(&creature, &records, &None);

    assert_eq!(scan.truncation, Some(ScanTruncation::CandidateCeiling));
    assert_eq!(
        scan.candidates.len(),
        2,
        "only h299's bias correction survives the cap and pairs with both weight corrections"
    );
    assert!(
        scan.candidates.iter().all(|c| c.bias_neuron_uuid == "h299"),
        "no candidate may use a bias correction weaker than the strongest 256, got {:?}",
        scan.candidates
            .iter()
            .map(|c| c.bias_neuron_uuid.as_str())
            .collect::<Vec<_>>()
    );
}

/// Issue #2367: more than `MAX_COMPOUND_WEIGHT_CORRECTIONS` weight corrections
/// are capped to the strongest before the pair loop. `w0..w299 -> out-0` each
/// yield a weight correction, and the fit to `out-0`'s error worsens with the
/// index. The single bias neuron `b` feeds only `w0` (strong) and `w299` (weak),
/// so it pairs with just those two. The cap drops `w299`, which leaves 1
/// candidate, so only the correction cap can report `CandidateCeiling` or
/// exclude `w299`.
#[test]
#[serial]
fn weight_correction_cap_keeps_the_strongest_and_reports_candidate_ceiling() {
    const SOURCES: usize = 300;
    let mut neurons = vec![test_neuron("b", "hidden")];
    let mut records = vec![hidden_records("b", 0.3)];
    let mut synapses = vec![test_synapse("b", "w0"), test_synapse("b", "w299")];
    for i in 0..SOURCES {
        let uuid = format!("w{i}");
        neurons.push(test_neuron(&uuid, "hidden"));
        let noise_scale = i as f32 / 100.0;
        let recs = (0..SAMPLES)
            .map(|o| DiscoverRecord {
                obs_index: o,
                neuron_uuid: uuid.clone(),
                value: Some(0.0),
                activation: out_error(o) + noise_scale * 0.2 * (o as f32 * 1.7 + 0.3).cos(),
                // Zero error: no bias correction for the weight sources.
                errors: vec![0.0],
            })
            .collect();
        records.push((uuid.clone(), recs));
        synapses.push(test_synapse(&uuid, "out-0"));
    }
    neurons.push(test_neuron("out-0", "output"));
    records.push(out_records());
    let creature = CreatureJson {
        neurons,
        synapses,
        input: 0,
        output: 1,
    };

    // Preconditions: 300 weight corrections, with w0 inside the strongest 256
    // and w299 outside it.
    let records_map = build_record_map(&records);
    let mut weight_corrections = detect_weight_corrections(&creature, &records_map);
    assert_eq!(weight_corrections.len(), SOURCES);
    weight_corrections.sort_by(|a, b| b.estimated_improvement.total_cmp(&a.estimated_improvement));
    let rank = |from: &str| {
        weight_corrections
            .iter()
            .position(|w| w.from_neuron_uuid == from)
            .expect("every source yields a weight correction")
    };
    assert!(rank("w0") < MAX_COMPOUND_WEIGHT_CORRECTIONS);
    assert!(rank("w299") >= MAX_COMPOUND_WEIGHT_CORRECTIONS);

    let scan = detect_compound_bias_weight_degradations_with_deadline(&creature, &records, &None);

    assert_eq!(scan.truncation, Some(ScanTruncation::CandidateCeiling));
    assert_eq!(
        scan.candidates.len(),
        1,
        "only b x (w0 -> out-0) survives the weight-correction cap"
    );
    assert_eq!(
        scan.candidates[0].weight_from_uuid, "w0",
        "no candidate may use a weight correction weaker than the strongest 256"
    );
}

/// Reachable-output walks made by the full scan over `creature`.
fn scan_walks(creature: &CreatureJson, records: &[(String, Vec<DiscoverRecord>)]) -> usize {
    let mut walks = 0usize;
    detect_compound_bias_weight_degradations_observed(creature, records, &None, &mut || {
        walks += 1;
    });
    walks
}

/// Issue #2367: with no weight corrections no (bias, weight) pair exists, so the
/// scan must not walk the graph for any bias neuron. The control (`fixture`
/// with its `out-0` records) walks once per distinct precomputed neuron: the
/// four hidden bias neurons plus the shared weight target `out-0`.
#[test]
#[serial]
fn scan_with_no_weight_corrections_does_no_reachable_walk() {
    let (creature, mut records) = fixture(4);
    assert_eq!(scan_walks(&creature, &records), 5);

    // Without `out-0` records no synapse has target records, so no weight
    // correction can be detected; the hidden bias corrections remain.
    records.retain(|(uuid, _)| uuid != "out-0");
    let records_map = build_record_map(&records);
    assert!(detect_weight_corrections(&creature, &records_map).is_empty());
    let neuron_squash: HashMap<&str, &str> = creature
        .neurons
        .iter()
        .map(|n| (n.uuid.as_str(), n.squash.as_str()))
        .collect();
    let neuron_bias: HashMap<&str, f32> = creature
        .neurons
        .iter()
        .map(|n| (n.uuid.as_str(), n.bias))
        .collect();
    assert_eq!(
        detect_bias_corrections(&creature, &records_map, &neuron_squash, &neuron_bias).len(),
        4,
        "the bias corrections must remain so the zero-walk check is non-vacuous"
    );

    assert_eq!(scan_walks(&creature, &records), 0);
    let scan = detect_compound_bias_weight_degradations_with_deadline(&creature, &records, &None);
    assert!(scan.candidates.is_empty());
    assert_eq!(scan.truncation, None);
}

/// Issue #2367: the packed reachable-output bitset must agree with hash-set
/// intersection and store no more than `O / 64 + 1` words for `O` outputs.
#[test]
fn output_bitset_intersects_matches_hash_set_intersection() {
    const OUTPUTS: usize = 300;
    let names: Vec<String> = (0..OUTPUTS).map(|i| format!("o{i}")).collect();
    let index: HashMap<&str, usize> = names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i))
        .collect();
    // Deterministic subsets: multiples of a stride from a start, plus empty.
    let mut sets: Vec<HashSet<&str>> = vec![HashSet::new()];
    for (start, stride) in [
        (0, 1),
        (5, 7),
        (63, 64),
        (64, 64),
        (299, 1),
        (130, 50),
        (1, 2),
    ] {
        sets.push(
            (start..OUTPUTS)
                .step_by(stride)
                .map(|i| names[i].as_str())
                .collect(),
        );
    }
    sets.push(HashSet::from([names[10].as_str()]));
    sets.push(HashSet::from([names[200].as_str()]));

    let packed: Vec<OutputBitset> = sets
        .iter()
        .map(|s| OutputBitset::from_outputs(s.iter().copied(), &index))
        .collect();
    for p in &packed {
        assert!(p.words.len() <= OUTPUTS / 64 + 1);
        // The retained allocation, not just the length, must keep to the
        // bound (review 5711830474).
        assert!(
            p.words.capacity() <= OUTPUTS / 64 + 1,
            "bitset capacity {} exceeds {} words",
            p.words.capacity(),
            OUTPUTS / 64 + 1
        );
    }
    // Review 5711830474: bits 64k, 64(k + 1) and 0 (k = 3), packed in that
    // order, made a word-by-word `resize` reserve 4 words, then double to 8 for
    // a 5-word set. Packing must reserve exactly the length whatever the order.
    let high_first = OutputBitset::from_outputs(
        [names[192].as_str(), names[256].as_str(), names[0].as_str()],
        &index,
    );
    assert_eq!(high_first.words.len(), 5);
    assert_eq!(
        high_first.words.capacity(),
        high_first.words.len(),
        "packing high bits first must not leave amortised spare capacity"
    );
    assert_eq!(
        high_first,
        OutputBitset::from_outputs(
            [names[0].as_str(), names[192].as_str(), names[256].as_str()],
            &index,
        ),
        "the packed set must not depend on the input order"
    );
    for (a, pa) in sets.iter().zip(&packed) {
        for (b, pb) in sets.iter().zip(&packed) {
            assert_eq!(
                pa.intersects(pb),
                a.intersection(b).next().is_some(),
                "bitset and hash-set intersection disagree for {a:?} and {b:?}"
            );
        }
    }
}
