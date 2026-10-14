//! Regression tests for Issue #2358 (CWE-407): `output_conflicts_to_coordinated_candidates`
//! rescanned `creature.synapses` once per conflict (and scanned the per-neuron
//! synapse list once per harmed output), making conversion O(conflicts x synapses).
//!
//! The growth test is a ratio test rather than a wall-clock budget: absolute
//! timings vary wildly across CI machines, but the ratio t(4N)/t(N) is stable.
//! Linear growth gives about 4x, quadratic about 16x. Each size is timed as the
//! best of several runs to suppress scheduler noise.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use neat_ai_discovery::analysis::detection::output_conflict::{
    OutputConflictNeuron, output_conflicts_to_coordinated_candidates,
};
use neat_ai_discovery::{
    CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson, NeuronJson,
    SynapseJson,
};

fn neuron(uuid: &str, neuron_type: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
        squash: "TANH".to_string(),
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

/// Weight of the hidden -> output-0 synapse of hidden neuron `i`.
fn original_weight(i: usize) -> f32 {
    1.0 + f32::from(u8::try_from(i % 7).expect("small"))
}

/// Even-indexed neurons harm output 1 (no direct synapse, so `AddNeuron`).
/// Odd-indexed neurons harm output 0 (direct synapse, so `SetWeight`).
fn harms_output_zero(i: usize) -> bool {
    i % 2 == 1
}

fn build(k: usize) -> (CreatureJson, Vec<OutputConflictNeuron>) {
    let mut neurons = vec![neuron("in-0", "input"), neuron("in-1", "input")];
    let mut synapses = Vec::new();
    let mut conflicts = Vec::new();
    for i in 0..k {
        let uuid = format!("hidden-{i}");
        neurons.push(neuron(&uuid, "hidden"));
        synapses.push(synapse("in-0", &uuid, 0.5));
        synapses.push(synapse(&uuid, "out-0", original_weight(i)));
        let errors = if harms_output_zero(i) {
            vec![0.5, -0.3]
        } else {
            vec![-0.3, 0.5]
        };
        conflicts.push(OutputConflictNeuron {
            neuron_uuid: uuid,
            squash: "TANH".to_string(),
            bias: 0.0,
            per_output_mean_error: errors,
            conflict_severity: 0.15,
            sample_count: 50,
            estimated_improvement: 0.05,
        });
    }
    neurons.push(neuron("out-0", "output"));
    neurons.push(neuron("out-1", "output"));
    (
        CreatureJson {
            neurons,
            synapses,
            input: 2,
            output: 2,
        },
        conflicts,
    )
}

fn best_of_five(k: usize) -> Duration {
    let (creature, conflicts) = build(k);
    (0..5)
        .map(|_| {
            let start = Instant::now();
            let out = output_conflicts_to_coordinated_candidates(&conflicts, &creature, &None);
            let elapsed = start.elapsed();
            std::hint::black_box(out);
            elapsed
        })
        .min()
        .expect("five runs")
}

fn candidate_uuid(c: &CoordinatedStructuralCandidateJson) -> String {
    let comment = c.comment.as_deref().expect("comment");
    let rest = comment
        .strip_prefix("Output conflict on hidden neuron ")
        .expect("comment prefix");
    rest.split(':').next().expect("uuid").to_string()
}

#[test]
fn conflict_conversion_work_grows_linearly_with_conflicts() {
    const N: usize = 1000;

    // Correctness on the N-sized run so the timing assertion is not vacuous.
    let (creature, conflicts) = build(N);
    let out = output_conflicts_to_coordinated_candidates(&conflicts, &creature, &None);
    assert_eq!(out.len(), N, "one candidate per conflict");
    let by_uuid: HashMap<String, &CoordinatedStructuralCandidateJson> =
        out.iter().map(|c| (candidate_uuid(c), c)).collect();
    assert_eq!(by_uuid.len(), N);
    for i in 0..N {
        let c = by_uuid[&format!("hidden-{i}")];
        if harms_output_zero(i) {
            assert_eq!(c.operations.len(), 1, "hidden-{i}");
            match &c.operations[0] {
                CoordinatedStructuralOpJson::SetWeight { weight, .. } => {
                    assert!(
                        (weight - original_weight(i) * 0.3).abs() < 1e-5,
                        "hidden-{i}"
                    );
                }
                other => panic!("hidden-{i}: expected SetWeight, got {other:?}"),
            }
        } else {
            assert_eq!(c.operations.len(), 3, "hidden-{i}");
            assert!(matches!(
                c.operations[0],
                CoordinatedStructuralOpJson::AddNeuron { .. }
            ));
            assert!(matches!(
                c.operations[1],
                CoordinatedStructuralOpJson::AddSynapse { .. }
            ));
            assert!(matches!(
                c.operations[2],
                CoordinatedStructuralOpJson::AddSynapse { .. }
            ));
        }
    }

    let small = best_of_five(N);
    let large = best_of_five(4 * N);
    let ratio = large.as_secs_f64() / small.as_secs_f64().max(1e-9);
    eprintln!("t(N)={small:?} t(4N)={large:?} ratio={ratio:.2}");
    assert!(
        ratio < 10.0,
        "conversion work should grow ~linearly: t({N})={small:?}, t({})={large:?}, ratio={ratio:.2} (linear ~4, quadratic ~16)",
        4 * N
    );
}

#[test]
fn conflict_conversion_stops_at_passed_deadline() {
    let (creature, conflicts) = build(10);
    let out = output_conflicts_to_coordinated_candidates(
        &conflicts,
        &creature,
        &Some(std::time::UNIX_EPOCH),
    );
    assert!(out.is_empty(), "a passed deadline must stop conversion");
}

#[test]
fn duplicate_synapse_pair_uses_first_weight() {
    let (mut creature, conflicts) = build(1);
    // build(1) has hidden-0 (harms output 1, no direct synapse); use the
    // odd-indexed shape instead: harm output 0 with duplicate synapses.
    creature.synapses.retain(|s| s.to_uuid != "out-0");
    creature.synapses.push(synapse("hidden-0", "out-0", 2.0));
    creature.synapses.push(synapse("hidden-0", "out-0", 7.0));
    let mut conflict = conflicts[0].clone();
    conflict.per_output_mean_error = vec![0.5, -0.3];
    let out = output_conflicts_to_coordinated_candidates(&[conflict], &creature, &None);
    assert_eq!(out.len(), 1);
    match &out[0].operations[..] {
        [CoordinatedStructuralOpJson::SetWeight { weight, .. }] => {
            assert!((weight - 0.6).abs() < 1e-5, "got {weight}");
        }
        other => panic!("expected single SetWeight, got {other:?}"),
    }
}
