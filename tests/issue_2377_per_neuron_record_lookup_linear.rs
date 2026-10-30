//! Regression tests for Issue #2377 (CWE-407): four public detectors looked up
//! each neuron's records with `neuron_records.iter().find(..)` inside the
//! per-neuron loop, so every neuron rescanned the whole records slice
//! (O(N x R)).
//!
//! This is a ratio test, not a wall-clock test: absolute timings vary widely
//! across CI machines, but the ratio t(4N)/t(N) is stable. Linear growth gives
//! about 4x, quadratic about 16x. Each size is timed as the best of five runs.
//!
//! Every neuron in the timed fixtures has one record, which is below
//! `MIN_DISCOVERY_SAMPLE_COUNT`, so each detector `continue`s straight after
//! the lookup and only the lookup is measured. The records are listed in
//! reverse neuron order, so the old scan for neuron `i` walks past the
//! `N - 1 - i` entries in front of its match.
//!
//! Each test also runs a positive control: eight neurons where only the last
//! one has qualifying records. That neuron must be the sole candidate returned,
//! which shows the lookup still pairs each neuron with its own records.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)

use std::hint::black_box;
use std::time::{Duration, Instant};

use neat_ai_discovery::analysis::constants::MIN_DISCOVERY_SAMPLE_COUNT;
use neat_ai_discovery::analysis::cost_function_hint::CostFunctionHint;
use neat_ai_discovery::analysis::detection::bias_perturbation::detect_bias_perturbation_candidates;
use neat_ai_discovery::analysis::detection::error_plateau::detect_error_plateaus;
use neat_ai_discovery::analysis::detection::output_squash_mismatch::detect_output_squash_mismatches_with_cost_hint;
use neat_ai_discovery::analysis::detection::squash_weight_rescale::detect_squash_weight_rescale_candidates;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

const N: usize = 2_000;

// The timed fixtures give each neuron one record. That must stay below the
// detectors' sample threshold, or the detectors would not `continue` early.
const _: () = assert!(1 < MIN_DISCOVERY_SAMPLE_COUNT);

type NeuronTuples = Vec<(String, String, f32)>;
type RecordTable = Vec<(String, Vec<DiscoverRecord>)>;

// =============================================================================
// Helpers
// =============================================================================

fn uuids(prefix: &str, n: usize) -> Vec<String> {
    (0..n).map(|i| format!("{prefix}-{i}")).collect()
}

fn record(uuid: &str, obs_index: u32, value: f32, activation: f32, error: f32) -> DiscoverRecord {
    DiscoverRecord {
        obs_index,
        neuron_uuid: uuid.to_string(),
        value: Some(value),
        activation,
        errors: vec![error],
    }
}

fn neuron(uuid: &str, neuron_type: &str, squash: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
        squash: squash.to_string(),
        bias: 0.0,
    }
}

fn synapse(from: &str, to: &str, weight: f32) -> SynapseJson {
    SynapseJson {
        from_uuid: from.to_string(),
        to_uuid: to.to_string(),
        weight,
        synapse_type: None,
    }
}

fn neuron_tuples(ids: &[String], squash: &str) -> NeuronTuples {
    ids.iter()
        .map(|u| (u.clone(), squash.to_string(), 0.0))
        .collect()
}

/// One sparse record per neuron, listed in reverse neuron order.
fn sparse_records(ids: &[String]) -> RecordTable {
    ids.iter()
        .rev()
        .map(|u| (u.clone(), vec![record(u, 0, 0.0, 0.0, 0.3)]))
        .collect()
}

/// Neuron tuples in neuron order plus reverse-ordered sparse records for `n` neurons.
fn sparse_fixture(prefix: &str, n: usize, squash: &str) -> (NeuronTuples, RecordTable) {
    let ids = uuids(prefix, n);
    (neuron_tuples(&ids, squash), sparse_records(&ids))
}

/// Eight neurons `{prefix}-0..=7`. Only `{prefix}-7` gets `qualifying` records;
/// the others get one sparse record each. Listed in reverse neuron order.
fn positive_records(prefix: &str, qualifying: fn(&str) -> Vec<DiscoverRecord>) -> RecordTable {
    let mut table: RecordTable = uuids(prefix, 8)
        .into_iter()
        .enumerate()
        .map(|(i, u)| {
            let recs = if i == 7 {
                qualifying(&u)
            } else {
                vec![record(&u, 0, 0.0, 0.0, 0.3)]
            };
            (u, recs)
        })
        .collect();
    table.reverse();
    table
}

/// Minimum of five timed runs of `run` over `fixture`.
fn best_of_five<T>(fixture: &T, run: impl Fn(&T) -> usize) -> Duration {
    (0..5)
        .map(|_| {
            let start = Instant::now();
            let out = run(fixture);
            let elapsed = start.elapsed();
            black_box(out);
            elapsed
        })
        .min()
        .expect("five runs")
}

/// Checks that the detector returns nothing for the sparse N-sized fixture, then
/// asserts that the best-of-five time for 4N is under 10x the time for N.
fn assert_linear_growth<T>(detector: &str, build: impl Fn(usize) -> T, run: impl Fn(&T) -> usize) {
    let small_fixture = build(N);
    assert_eq!(
        run(&small_fixture),
        0,
        "{detector}: sparse fixture must yield no candidates"
    );
    let small = best_of_five(&small_fixture, &run);
    let large_fixture = build(4 * N);
    let large = best_of_five(&large_fixture, &run);
    let ratio = large.as_secs_f64() / small.as_secs_f64().max(1e-9);
    eprintln!("{detector}: t(N)={small:?} t(4N)={large:?} ratio={ratio:.2}");
    assert!(
        ratio < 10.0,
        "{detector}: record lookup should grow ~linearly: t({N})={small:?}, t({})={large:?}, ratio={ratio:.2} (linear ~4, quadratic ~16)",
        4 * N
    );
}

// =============================================================================
// Qualifying fixtures for the positive controls
// =============================================================================

/// Pre-activations deep in TANH saturation with a consistent error (Issue #551).
fn bias_qualifying(uuid: &str) -> Vec<DiscoverRecord> {
    (0..60u32)
        .map(|i| {
            let value = 4.5 + (i as f32) / 60.0;
            record(uuid, i, value, value.tanh(), 0.3)
        })
        .collect()
}

/// Tightly clustered non-zero error on a `HARD_TANH` output (Issue #545).
fn plateau_qualifying(uuid: &str) -> Vec<DiscoverRecord> {
    (0..60u32)
        .map(|i| {
            let activation = (i as f32 - 30.0) / 40.0;
            let error = 0.30 + (i as f32 * 0.001).sin() * 0.01;
            record(uuid, i, activation, activation, error)
        })
        .collect()
}

/// `HARD_TANH` output against TANH-shaped targets (Issue #546).
fn output_squash_qualifying(uuid: &str) -> Vec<DiscoverRecord> {
    (0..60u32)
        .map(|i| {
            let x = (i as f32 - 30.0) / 10.0;
            let hard = x.clamp(-1.0, 1.0);
            record(uuid, i, x, hard, (hard - x.tanh()).abs())
        })
        .collect()
}

/// RELU hidden neuron with predominantly negative pre-activations (Issue #548).
fn squash_qualifying(uuid: &str) -> Vec<DiscoverRecord> {
    (0..40u32)
        .map(|i| {
            let value = (i as f32 - 20.0) / 10.0;
            let activation = value.max(0.0);
            let error = 0.15 + (i as f32 * 0.002).sin() * 0.02;
            record(uuid, i, value, activation, error)
        })
        .collect()
}

// =============================================================================
// One growth test per detector
// =============================================================================

#[test]
fn bias_perturbation_record_lookup_grows_linearly() {
    // Positive control: only hidden-7 qualifies. Its bias of 5.0 is far from
    // the TANH active-zone centre, so a regime shift is recommended.
    let mut hidden = neuron_tuples(&uuids("hidden", 8), "TANH");
    hidden[7].2 = 5.0;
    let control = positive_records("hidden", bias_qualifying);
    let out = detect_bias_perturbation_candidates(&hidden, &control);
    let got: Vec<&str> = out.iter().map(|c| c.neuron_uuid.as_str()).collect();
    assert_eq!(got, ["hidden-7"], "positive control");

    assert_linear_growth(
        "bias_perturbation",
        |n| sparse_fixture("hidden", n, "TANH"),
        |(h, r)| detect_bias_perturbation_candidates(h, r).len(),
    );
}

#[test]
fn error_plateau_record_lookup_grows_linearly() {
    let outputs = neuron_tuples(&uuids("output", 8), "HARD_TANH");
    let control = positive_records("output", plateau_qualifying);
    let out = detect_error_plateaus(&outputs, &control);
    let got: Vec<&str> = out.iter().map(|c| c.neuron_uuid.as_str()).collect();
    assert_eq!(got, ["output-7"], "positive control");

    assert_linear_growth(
        "error_plateau",
        |n| sparse_fixture("output", n, "TANH"),
        |(h, r)| detect_error_plateaus(h, r).len(),
    );
}

#[test]
fn output_squash_record_lookup_grows_linearly() {
    let outputs = neuron_tuples(&uuids("output", 8), "HARD_TANH");
    let control = positive_records("output", output_squash_qualifying);
    let out = detect_output_squash_mismatches_with_cost_hint(
        &outputs,
        &control,
        CostFunctionHint::Unknown,
    );
    let got: Vec<&str> = out.iter().map(|c| c.neuron_uuid.as_str()).collect();
    assert_eq!(got, ["output-7"], "positive control");

    assert_linear_growth(
        "output_squash_mismatch",
        |n| sparse_fixture("output", n, "TANH"),
        |(h, r)| {
            detect_output_squash_mismatches_with_cost_hint(h, r, CostFunctionHint::Unknown).len()
        },
    );
}

#[test]
fn squash_weight_rescale_record_lookup_grows_linearly() {
    // Positive control: hidden-7 is RELU with one incoming and one outgoing
    // synapse, as in the Issue #548 fixture.
    let mut hidden = neuron_tuples(&uuids("hidden", 8), "TANH");
    hidden[7].1 = "RELU".to_string();
    let creature = CreatureJson {
        neurons: vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("input-2", "input", "IDENTITY"),
            neuron("hidden-7", "hidden", "RELU"),
            neuron("output-1", "output", "TANH"),
        ],
        synapses: vec![
            synapse("input-1", "hidden-7", 1.5),
            synapse("input-2", "hidden-7", -0.8),
            synapse("hidden-7", "output-1", 1.0),
        ],
        input: 2,
        output: 1,
    };
    let control = positive_records("hidden", squash_qualifying);
    let out = detect_squash_weight_rescale_candidates(&creature, &hidden, &control);
    let got: Vec<&str> = out.iter().map(|c| c.neuron_uuid.as_str()).collect();
    assert_eq!(got, ["hidden-7"], "positive control");

    // Synapse-free creature: the aggregate-feed check is O(0) per neuron, so
    // only the record lookup grows with N.
    assert_linear_growth(
        "squash_weight_rescale",
        |n| {
            let creature = CreatureJson {
                neurons: Vec::new(),
                synapses: Vec::new(),
                input: 1,
                output: 1,
            };
            let (hidden, records) = sparse_fixture("hidden", n, "TANH");
            (creature, hidden, records)
        },
        |(c, h, r)| detect_squash_weight_rescale_candidates(c, h, r).len(),
    );
}
