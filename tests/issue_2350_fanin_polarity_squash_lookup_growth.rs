//! Issue #2350 (CWE-407, Algorithmic Complexity): the fan-in polarity conflict
//! converter looked up each candidate's originating neuron's squash function
//! with `creature.neurons.iter().find(|n| n.uuid == c.neuron_uuid)` — a linear
//! scan performed once per candidate, giving `O(C·N)` work for `C` candidates
//! over `N` neurons.
//!
//! The regression checks observable output sizes at N and 4N. Deterministic
//! lookup-work counts live beside the private converter helper, while elapsed
//! time is measured by the Criterion benchmark in `benches/`.

mod common;

use common::{make_creature, neuron, output, synapse};
use neat_ai_discovery::CoordinatedStructuralOpJson;
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

fn converted_candidate_count(count: usize) -> usize {
    let creature = build_creature_with_conflicted_hidden(count);
    let candidates = build_candidates(count);
    fanin_polarity_conflicts_to_coordinated_candidates(&candidates, &creature).len()
}

#[test]
fn squash_lookup_conversion_preserves_output_at_n_and_4n() {
    const N: usize = 2_000;
    assert_eq!(converted_candidate_count(N), N);
    assert_eq!(converted_candidate_count(N * 4), N * 4);
}

#[test]
fn conversion_keeps_gain_order_and_inserts_each_neuron_before_its_target() {
    let creature = build_creature_with_conflicted_hidden(2);
    let mut candidates = build_candidates(2);
    candidates[0].estimated_improvement = 0.01;
    candidates[1].estimated_improvement = 0.02;

    let converted = fanin_polarity_conflicts_to_coordinated_candidates(&candidates, &creature);
    let inserted_targets: Vec<_> = converted
        .iter()
        .map(|candidate| {
            candidate
                .operations
                .iter()
                .find_map(|operation| match operation {
                    CoordinatedStructuralOpJson::AddNeuron {
                        neuron_uuid,
                        insert_before_neuron_uuid,
                        ..
                    } => Some((neuron_uuid.as_str(), insert_before_neuron_uuid.as_deref())),
                    _ => None,
                })
                .expect("each candidate adds a split neuron")
        })
        .collect();

    assert_eq!(
        inserted_targets,
        [
            ("fanin-split-h1", Some("h1")),
            ("fanin-split-h0", Some("h0"))
        ]
    );
}
