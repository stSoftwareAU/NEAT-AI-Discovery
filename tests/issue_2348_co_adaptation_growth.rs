//! Issue #2348 (CWE-834, Excessive Iteration): the co-adaptation detector's
//! pairwise scan is uncancellable and unbounded — it visits every hidden
//! neuron pair (`O(n^2)`) and emits a candidate for every one that crosses
//! the correlation threshold, with no deadline check and no ceiling on the
//! number of candidates produced.
//!
//! This test is deliberately written against only the pre-existing public
//! API (`detect_co_adapted_neurons`) so that it compiles and runs against
//! the unfixed base code: quadrupling the eligible hidden-neuron count `E`
//! should grow the emitted candidate count by no more than 4x (linear in
//! `E`), once a fix caps the candidates emitted. Against the unfixed code,
//! `E = 24` (`C(24,2) = 276` pairs) vs `4E = 96` (`C(96,2) = 4560` pairs) are
//! both fully correlated and uncapped, so the candidate count grows
//! quadratically (~16.5x) and this test is red. A fix that caps candidate
//! emission (e.g. at 256) makes both counts converge to the cap and this
//! test goes green.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for synthetic activation waveforms (Issue #873)

mod common;

use common::{hidden, make_creature, neuron, output, record, synapse};
use neat_ai_discovery::analysis::detection::co_adaptation::detect_co_adapted_neurons;
use neat_ai_discovery::types::DiscoverRecord;

/// Build `count` records for a neuron, all identical across neurons so every
/// pair is perfectly correlated. Activations are non-constant (so Pearson
/// correlation is defined) but follow the same waveform for every neuron.
fn make_correlated_records(uuid: &str, count: u32) -> Vec<DiscoverRecord> {
    (0..count)
        .map(|i| record(uuid, i, (i as f32 * 0.1).sin(), None))
        .collect()
}

/// Build a creature with `hidden_count` hidden neurons, each fed by a single
/// input and feeding a single output, plus a single-input/single-output
/// creature topology.
fn build_creature_with_hidden(hidden_count: usize) -> neat_ai_discovery::CreatureJson {
    let mut neurons = vec![neuron("in-0", "input", "IDENTITY")];
    let mut synapses = Vec::new();
    for h in 0..hidden_count {
        let uuid = format!("h{h}");
        neurons.push(hidden(&uuid, "TANH"));
        synapses.push(synapse("in-0", &uuid, 1.0));
    }
    neurons.push(output("out-0", "TANH"));
    for h in 0..hidden_count {
        let uuid = format!("h{h}");
        synapses.push(synapse(&uuid, "out-0", 0.5));
    }
    make_creature(neurons, synapses)
}

/// Guards against CWE-834 (excessive iteration): quadrupling the eligible
/// hidden-neuron count must not quadruple-squared the emitted candidate
/// count — growth in candidates emitted must stay no faster than linear in
/// the eligible-neuron count once bounded.
#[test]
fn co_adaptation_pairs_evaluated_grow_no_faster_than_the_cap_allows() {
    const COUNT: u32 = 20;

    let small_hidden = 24_usize;
    let big_hidden = small_hidden * 4;

    let small_creature = build_creature_with_hidden(small_hidden);
    let small_records: Vec<(String, Vec<DiscoverRecord>)> = (0..small_hidden)
        .map(|h| {
            let uuid = format!("h{h}");
            (uuid.clone(), make_correlated_records(&uuid, COUNT))
        })
        .collect();
    let small = detect_co_adapted_neurons(&small_creature, &small_records);

    let big_creature = build_creature_with_hidden(big_hidden);
    let big_records: Vec<(String, Vec<DiscoverRecord>)> = (0..big_hidden)
        .map(|h| {
            let uuid = format!("h{h}");
            (uuid.clone(), make_correlated_records(&uuid, COUNT))
        })
        .collect();
    let big = detect_co_adapted_neurons(&big_creature, &big_records);

    assert!(
        big.len() <= 4 * small.len(),
        "candidates emitted must grow no faster than linear in the eligible-neuron count, \
         got {} candidates at E={small_hidden} and {} candidates at 4E={big_hidden}",
        small.len(),
        big.len()
    );
}
