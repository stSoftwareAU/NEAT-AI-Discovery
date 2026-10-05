//! Issue #2350 (CWE-407, Algorithmic Complexity): the fan-in polarity conflict
//! converter looked up each candidate's originating neuron's squash function
//! with `creature.neurons.iter().find(|n| n.uuid == c.neuron_uuid)` — a linear
//! scan performed once per candidate, giving `O(C·N)` work for `C` candidates
//! over `N` neurons.
//!
//! This test is deliberately written against only the pre-existing public API
//! (`fanin_polarity_conflicts_to_coordinated_candidates`) so that it compiles
//! and runs against both the unfixed base code and the fixed code: it times
//! the conversion at `N` hidden neurons/candidates vs `4N`, and asserts that
//! wall-clock time grows no faster than roughly linearly (`ratio < 8.0`,
//! comfortably below the unfixed quadratic blow-up of ~16x and above the
//! fixed linear ~4x). Hidden neurons are placed early in creature evaluation
//! order so the unfixed linear `find` scans roughly half the neuron list on
//! average per candidate.

mod common;

use std::time::Instant;

use common::{make_creature, neuron, output, synapse};
use neat_ai_discovery::analysis::detection::fanin_polarity_conflict::{
    FaninPolarityConflictCandidate, fanin_polarity_conflicts_to_coordinated_candidates,
};

/// Build a creature with `count` conflicted hidden neurons, each fed by one
/// positive-weight and one negative-weight input synapse (so the converter's
/// `synapses_to_move` is never empty and the loop body is always exercised),
/// plus a single output. Hidden neurons are placed immediately after the two
/// inputs (i.e. early in evaluation order) so an unfixed linear `find` scan
/// over `creature.neurons` must walk roughly half the neuron list per lookup.
fn build_creature_with_conflicted_hidden(count: usize) -> neat_ai_discovery::CreatureJson {
    let mut neurons = vec![
        neuron("pos-in", "input", "IDENTITY"),
        neuron("neg-in", "input", "IDENTITY"),
    ];
    let mut synapses = Vec::new();
    for h in 0..count {
        let uuid = format!("h{h}");
        neurons.push(neuron(&uuid, "hidden", "TANH"));
        synapses.push(synapse("pos-in", &uuid, 2.0));
        synapses.push(synapse("neg-in", &uuid, -2.0));
    }
    neurons.push(output("out-0", "IDENTITY"));
    for h in 0..count {
        let uuid = format!("h{h}");
        synapses.push(synapse(&uuid, "out-0", 1.0));
    }
    make_creature(neurons, synapses)
}

/// One candidate per conflicted hidden neuron, mirroring what
/// `detect_fanin_polarity_conflicts` would have produced for the fixture
/// built by [`build_creature_with_conflicted_hidden`].
fn build_candidates(count: usize) -> Vec<FaninPolarityConflictCandidate> {
    (0..count)
        .map(|h| FaninPolarityConflictCandidate {
            neuron_uuid: format!("h{h}"),
            positive_weight_sum: 2.0,
            negative_weight_sum: 2.0,
            positive_count: 1,
            negative_count: 1,
            conflict_score: 1.0,
            sample_count: 30,
            estimated_improvement: 0.01,
        })
        .collect()
}

/// Time `fanin_polarity_conflicts_to_coordinated_candidates` over `repeats`
/// runs and return the minimum duration, to minimise scheduling noise.
fn min_duration_for(count: usize, repeats: u32) -> std::time::Duration {
    let creature = build_creature_with_conflicted_hidden(count);
    let candidates = build_candidates(count);

    let mut best = std::time::Duration::MAX;
    for _ in 0..repeats {
        let start = Instant::now();
        let results = fanin_polarity_conflicts_to_coordinated_candidates(&candidates, &creature);
        let elapsed = start.elapsed();
        assert_eq!(
            results.len(),
            count,
            "every conflicted hidden neuron should convert to a coordinated candidate"
        );
        best = best.min(elapsed);
    }
    best
}

/// Guards against CWE-407 (algorithmic complexity): quadrupling the number of
/// candidates (and hidden neurons) must not quadratically blow up the
/// wall-clock time spent converting them — growth must stay roughly linear
/// once the per-candidate squash lookup is `O(1)`.
#[test]
fn squash_lookup_work_grows_linearly_with_candidates() {
    const N: usize = 2000;
    const BIG_N: usize = N * 4;
    const REPEATS: u32 = 5;

    let small = min_duration_for(N, REPEATS);
    let big = min_duration_for(BIG_N, REPEATS);

    let ratio = big.as_secs_f64() / small.as_secs_f64().max(1e-9);

    assert!(
        ratio < 8.0,
        "conversion time must grow no faster than roughly linearly in the candidate count, \
         got {small:?} at N={N} and {big:?} at 4N={BIG_N} (ratio {ratio:.2})",
    );
}
