//! Issue #2367 (CWE-400, Uncontrolled Resource Consumption): the compound
//! bias+weight degradation detector combines every bias correction with every
//! weight correction (`O(B·W)` pairs, each with reachability work) and emits a
//! candidate for every connected pair, with no ceiling on the count.
//!
//! This test is deliberately written against only the pre-existing public API
//! (`detect_compound_bias_weight_degradations`) so that it compiles and runs
//! against the unfixed base code: quadrupling the hidden-neuron count `H`
//! should grow the emitted candidate count by no more than 4x (linear in `H`),
//! once a fix caps the pair scan and the emitted candidates. Against the
//! unfixed code, `H = 24` (576 connected pairs) and `H = 96` (9,216 connected
//! pairs) are both uncapped, so the count grows quadratically (16x) and this
//! test is red. With the ceilings in place both counts converge to 256 and the
//! test goes green.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for synthetic fixtures (Issue #873)

mod common;

use common::{hidden, make_creature, neuron, output, synapse};
use neat_ai_discovery::CreatureJson;
use neat_ai_discovery::analysis::detection::compound_degradation::detect_compound_bias_weight_degradations;
use neat_ai_discovery::types::DiscoverRecord;

/// Observations per neuron. Well above `MIN_DISCOVERY_SAMPLE_COUNT` (20).
const SAMPLES: u32 = 40;

/// Records for a hidden neuron: a constant, non-zero error (a consistent bias
/// offset) and a varying activation (the source side of its weight correction).
fn hidden_records(uuid: &str, bias_error: f32) -> Vec<DiscoverRecord> {
    (0..SAMPLES)
        .map(|o| DiscoverRecord {
            obs_index: o,
            neuron_uuid: uuid.to_string(),
            value: Some(0.0),
            activation: 1.0 + 0.5 * (o as f32 * 0.7).sin(),
            errors: vec![bias_error],
        })
        .collect()
}

/// Records for the output neuron: a continuous, non-quantised error shared by
/// every hidden -> out-0 synapse, so each has a positive least-squares delta.
fn output_records(uuid: &str) -> Vec<DiscoverRecord> {
    (0..SAMPLES)
        .map(|o| DiscoverRecord {
            obs_index: o,
            neuron_uuid: uuid.to_string(),
            value: None,
            activation: 0.0,
            errors: vec![0.2 + 0.1 * (o as f32 * 0.9 + 0.5).sin()],
        })
        .collect()
}

/// `hidden_count` hidden neurons, each fed by `in-0` and feeding `out-0`.
/// Every hidden neuron has a bias correction, every hidden -> out-0 synapse has
/// a weight correction, and every (bias, weight) pair is on a connected path.
fn build_creature(hidden_count: usize) -> (CreatureJson, Vec<(String, Vec<DiscoverRecord>)>) {
    let mut neurons = vec![neuron("in-0", "input", "IDENTITY")];
    let mut synapses = Vec::new();
    let mut records = Vec::new();
    for h in 0..hidden_count {
        let uuid = format!("h{h}");
        neurons.push(hidden(&uuid, "IDENTITY"));
        synapses.push(synapse("in-0", &uuid, 1.0));
        synapses.push(synapse(&uuid, "out-0", 0.5));
        records.push((uuid.clone(), hidden_records(&uuid, 0.1)));
    }
    neurons.push(output("out-0", "IDENTITY"));
    records.push(("out-0".to_string(), output_records("out-0")));
    (make_creature(neurons, synapses), records)
}

/// Guards against CWE-400: quadrupling the hidden-neuron count must not
/// quadruple-squared the emitted candidate count. Growth in candidates must
/// stay no faster than linear in the hidden-neuron count once bounded.
#[test]
fn compound_candidates_grow_no_faster_than_linear_in_neuron_count() {
    let small_hidden = 24_usize;
    let big_hidden = small_hidden * 4;

    let (small_creature, small_records) = build_creature(small_hidden);
    let small = detect_compound_bias_weight_degradations(&small_creature, &small_records);

    let (big_creature, big_records) = build_creature(big_hidden);
    let big = detect_compound_bias_weight_degradations(&big_creature, &big_records);

    assert!(
        !small.is_empty(),
        "fixture must yield compound candidates so the growth check is non-vacuous"
    );
    assert!(
        big.len() <= 4 * small.len(),
        "candidates emitted must grow no faster than linear in the hidden-neuron count, \
         got {} candidates at H={small_hidden} and {} candidates at 4H={big_hidden}",
        small.len(),
        big.len()
    );
}
