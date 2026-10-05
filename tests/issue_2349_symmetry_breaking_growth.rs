//! Issue #2349: `detect_symmetric_neurons` did an uncancellable O(E²) pair
//! scan over hidden neurons, rebuilding BOTH weight vectors per pair (each
//! O(A·F) via a linear `find`), and filtered synapses with an O(S·H)
//! `.any(..)` membership test.
//!
//! This pins:
//! - weight vectors are built exactly once per eligible neuron, not once per
//!   pair-side (linear growth in the neuron count, not quadratic);
//! - the pair scan stops at the discovery deadline / host cancellation and
//!   reports `ScanTruncation::DeadlinePassed`;
//! - the pair scan stops at `MAX_SYMMETRIC_PAIR_CANDIDATES` and reports
//!   `ScanTruncation::CandidateCeiling`;
//! - the deadline-free legacy entry point agrees with the deadline-aware one;
//! - the first synapse (in `creature.synapses` order) for a duplicated
//!   source still wins, matching the pre-#2349 `find`-based semantics.
//!
//! Deterministic throughout: growth and truncation are pinned by counting
//! work done, not by timing it. An elapsed deadline is `UNIX_EPOCH`, a live
//! one is a day ahead — no sleeps and no timing thresholds.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for per-neuron bias offsets (Issue #873)

use std::time::{Duration, SystemTime};

use neat_ai_discovery::analysis::detection::symmetry_breaking::{
    MAX_SYMMETRIC_PAIR_CANDIDATES, MAX_SYMMETRY_ELIGIBLE_NEURONS, detect_symmetric_neurons,
    detect_symmetric_neurons_observed, detect_symmetric_neurons_with_deadline,
};
use neat_ai_discovery::analysis::recommendation::epistatic::ScanTruncation;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

/// Clears `MIN_DISCOVERY_SAMPLE_COUNT` (20).
const SAMPLES: u32 = 30;

fn neuron(uuid: &str, neuron_type: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
        squash: "LOGISTIC".to_string(),
        bias: 0.0,
    }
}

fn synapse(from_uuid: &str, to_uuid: &str, weight: f32) -> SynapseJson {
    SynapseJson {
        from_uuid: from_uuid.to_string(),
        to_uuid: to_uuid.to_string(),
        weight,
        synapse_type: None,
    }
}

fn records_for(uuid: &str) -> (String, Vec<DiscoverRecord>) {
    let recs = (0..SAMPLES)
        .map(|idx| DiscoverRecord {
            obs_index: idx,
            neuron_uuid: uuid.to_string(),
            value: Some(0.0),
            activation: 0.9,
            errors: vec![0.0],
        })
        .collect();
    (uuid.to_string(), recs)
}

/// A creature with 3 input neurons, `n` identical hidden neurons (same
/// squash, bias, and incoming weights from every input) and one output, so
/// every pair of hidden neurons is a symmetric-pair candidate.
fn identical_creature(n: usize) -> (CreatureJson, Vec<(String, Vec<DiscoverRecord>)>) {
    let inputs = ["input-0", "input-1", "input-2"];
    let weights = [0.5_f32, -0.3, 0.8];

    let mut neurons: Vec<NeuronJson> = inputs.iter().map(|u| neuron(u, "input")).collect();
    neurons.push(neuron("output-0", "output"));

    let mut synapses = Vec::new();
    let mut records = Vec::with_capacity(n);

    for h in 0..n {
        let uuid = format!("hidden-{h}");
        neurons.push(neuron(&uuid, "hidden"));
        for (input, weight) in inputs.iter().zip(weights.iter()) {
            synapses.push(synapse(input, &uuid, *weight));
        }
        records.push(records_for(&uuid));
    }

    let creature = CreatureJson {
        neurons,
        synapses,
        input: inputs.len(),
        output: 1,
    };
    (creature, records)
}

fn far_future_deadline() -> Option<SystemTime> {
    Some(SystemTime::now() + Duration::from_secs(24 * 60 * 60))
}

/// Like `identical_creature`, but every hidden neuron's bias is offset by
/// 1.0 from its neighbours — comfortably beyond `BIAS_TOLERANCE` (0.5) — so
/// no pair ever qualifies as symmetric. This isolates the eligible-neuron
/// cap from the candidate ceiling: with zero candidates possible, any
/// truncation observed can only come from the eligible-neuron cap.
fn creature_with_no_symmetric_pairs(
    n: usize,
) -> (CreatureJson, Vec<(String, Vec<DiscoverRecord>)>) {
    let inputs = ["input-0", "input-1", "input-2"];
    let weights = [0.5_f32, -0.3, 0.8];

    let mut neurons: Vec<NeuronJson> = inputs.iter().map(|u| neuron(u, "input")).collect();
    neurons.push(neuron("output-0", "output"));

    let mut synapses = Vec::new();
    let mut records = Vec::with_capacity(n);

    for h in 0..n {
        let uuid = format!("hidden-{h}");
        neurons.push(NeuronJson {
            uuid: uuid.clone(),
            neuron_type: "hidden".to_string(),
            squash: "LOGISTIC".to_string(),
            bias: h as f32,
        });
        for (input, weight) in inputs.iter().zip(weights.iter()) {
            synapses.push(synapse(input, &uuid, *weight));
        }
        records.push(records_for(&uuid));
    }

    let creature = CreatureJson {
        neurons,
        synapses,
        input: inputs.len(),
        output: 1,
    };
    (creature, records)
}

#[test]
fn symmetry_breaking_weight_vectors_are_built_once_per_neuron() {
    // E = 4 eligible hidden neurons → 4·3/2 = 6 pairs, well under the 256
    // candidate ceiling, so the whole scan runs to completion.
    let (c4, r4) = identical_creature(4);
    let mut count_e = 0;
    let scan_e = detect_symmetric_neurons_observed(&c4, &r4, &None, || count_e += 1);
    assert_eq!(
        count_e, 4,
        "exactly one weight-vector build per eligible neuron"
    );
    assert_eq!(scan_e.candidates.len(), 4 * 3 / 2);
    assert_eq!(scan_e.truncation, None);

    // E = 16 eligible hidden neurons → 16·15/2 = 120 pairs, still under the
    // ceiling.
    let (c16, r16) = identical_creature(16);
    let mut count_4e = 0;
    let scan_4e = detect_symmetric_neurons_observed(&c16, &r16, &None, || count_4e += 1);
    assert_eq!(
        count_4e, 16,
        "one weight-vector build per eligible neuron, scaled 4x"
    );
    assert_eq!(scan_4e.candidates.len(), 16 * 15 / 2);
    assert_eq!(scan_4e.truncation, None);

    // Before #2349, the base algorithm rebuilt BOTH weight vectors for every
    // pair: 2 * E(E-1)/2 builds. For E=4 that is 12, for E=16 that is 240 —
    // a 20x growth for a 4x increase in neuron count. The fix builds one
    // vector per eligible neuron, so growth is linear (4x in, 4x out).
    assert!(
        count_4e <= 4 * count_e,
        "weight-vector builds must grow linearly with neuron count, not quadratically: \
         count_e={count_e}, count_4e={count_4e}"
    );
}

#[test]
fn symmetry_breaking_scan_stops_at_an_elapsed_deadline() {
    let (creature, records) = identical_creature(8);

    let elapsed =
        detect_symmetric_neurons_with_deadline(&creature, &records, &Some(SystemTime::UNIX_EPOCH));
    assert_eq!(elapsed.truncation, Some(ScanTruncation::DeadlinePassed));
    assert!(elapsed.candidates.is_empty());

    let live = detect_symmetric_neurons_with_deadline(&creature, &records, &far_future_deadline());
    assert_eq!(live.truncation, None);
    assert_eq!(live.candidates.len(), 8 * 7 / 2);
}

#[test]
fn symmetry_breaking_scan_stops_at_the_candidate_ceiling() {
    // 40 identical hidden neurons → 40·39/2 = 780 candidate pairs, well over
    // the 256 ceiling.
    let (creature, records) = identical_creature(40);

    let scan = detect_symmetric_neurons_with_deadline(&creature, &records, &None);
    assert_eq!(scan.truncation, Some(ScanTruncation::CandidateCeiling));
    assert_eq!(scan.candidates.len(), MAX_SYMMETRIC_PAIR_CANDIDATES);
}

#[test]
fn legacy_detect_symmetric_neurons_matches_deadline_free_scan() {
    let (creature, records) = identical_creature(6);

    let legacy = detect_symmetric_neurons(&creature, &records);
    let bounded = detect_symmetric_neurons_with_deadline(&creature, &records, &None);

    let legacy_pairs: Vec<(String, String)> = legacy
        .iter()
        .map(|c| (c.neuron_a_uuid.clone(), c.neuron_b_uuid.clone()))
        .collect();
    let bounded_pairs: Vec<(String, String)> = bounded
        .candidates
        .iter()
        .map(|c| (c.neuron_a_uuid.clone(), c.neuron_b_uuid.clone()))
        .collect();

    assert_eq!(legacy_pairs, bounded_pairs);
}

#[test]
fn duplicate_source_synapse_keeps_first_weight() {
    // Neuron A has two synapses from the same input: the first with weight
    // 1.0, the second with weight -1.0. Neuron B has a single synapse from
    // that input with weight 1.0. If the first synapse wins (legacy `find`
    // semantics), A's effective weight vector is [1.0] and the pair is
    // detected as symmetric (cosine similarity 1.0). If the scatter
    // regressed to last-wins, A's effective weight would be -1.0 and the
    // pair would not be detected.
    let neurons = vec![
        neuron("input-0", "input"),
        neuron("output-0", "output"),
        neuron("hidden-a", "hidden"),
        neuron("hidden-b", "hidden"),
    ];
    let synapses = vec![
        synapse("input-0", "hidden-a", 1.0),
        synapse("input-0", "hidden-a", -1.0),
        synapse("input-0", "hidden-b", 1.0),
    ];
    let creature = CreatureJson {
        neurons,
        synapses,
        input: 1,
        output: 1,
    };
    let records = vec![records_for("hidden-a"), records_for("hidden-b")];

    let scan = detect_symmetric_neurons_with_deadline(&creature, &records, &None);
    assert_eq!(scan.truncation, None);
    assert_eq!(
        scan.candidates.len(),
        1,
        "the duplicated-source pair must still be detected as symmetric"
    );
    assert!((scan.candidates[0].cosine_similarity - 1.0).abs() < 1e-6);
}

/// Review finding on PR #2400 (Issue #2349 follow-up): eligible hidden
/// neurons beyond `MAX_SYMMETRY_ELIGIBLE_NEURONS` must be skipped (and the
/// skip recorded), rather than driving an unbounded `E x A` weight-vector
/// build. No pair is symmetric here, so the candidate ceiling never fires —
/// isolating the eligible-neuron cap. The weight-vector build count is
/// capped at `MAX_SYMMETRY_ELIGIBLE_NEURONS`, proving the fix: before this
/// cap existed, this test would build `extra` more vectors than it does now.
#[test]
fn eligible_neurons_beyond_the_cap_are_skipped_and_recorded() {
    let extra = 40_usize;
    let hidden_count = MAX_SYMMETRY_ELIGIBLE_NEURONS + extra;

    let (creature, records) = creature_with_no_symmetric_pairs(hidden_count);

    let mut build_count = 0;
    let scan = detect_symmetric_neurons_observed(&creature, &records, &None, || build_count += 1);

    assert_eq!(scan.eligible_skipped, extra);
    assert_eq!(
        build_count, MAX_SYMMETRY_ELIGIBLE_NEURONS,
        "the weight-vector build must stop at the eligible-neuron cap, not build one per hidden neuron"
    );
    assert!(
        scan.candidates.is_empty(),
        "no pair is symmetric by construction, so no candidate ceiling can fire"
    );
    assert_eq!(
        scan.truncation, None,
        "the eligible-neuron cap alone is not reported as a scan truncation"
    );
}

/// Under the eligible-neuron cap, no neurons are skipped.
#[test]
fn under_cap_scan_records_no_skip() {
    let (creature, records) = creature_with_no_symmetric_pairs(10);

    let scan = detect_symmetric_neurons_with_deadline(&creature, &records, &None);

    assert_eq!(scan.eligible_skipped, 0);
}
