//! Integration tests for Issue #1623: bias-fold removal for functionally-constant
//! hidden neurons.
//!
//! A neuron with a constant activation `c` contributes a fixed `w_{n→t}·c` to
//! each target `t` on every observation. Folding `w_{n→t}·c` into `t`'s bias and
//! deleting the neuron is behaviour-preserving on the recorded window. These
//! tests assert:
//!
//! - **Exactness:** post-fold output pre-activations are bit-identical (within
//!   tolerance) to pre-fold across the recorded window, and the neuron and its
//!   edges are gone.
//! - **Multi-target fan-out:** each target's bias receives its own `w × c` fold.
//! - **Gate rejection:** a fold whose evaluation fails is not accepted and the
//!   network is left unchanged.

// Small fixed observation counts cast cleanly to `u32` in these fixtures.
#![allow(clippy::cast_possible_truncation)]

use std::collections::HashMap;

use neat_ai_discovery::analysis::remove_neuron_bias_fold::{
    evaluate_constant_neuron_bias_fold, fold_and_remove_constant_neuron,
};
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

const TOLERANCE: f64 = 1e-6;

fn neuron(uuid: &str, neuron_type: &str, bias: f32) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
        squash: "IDENTITY".to_string(),
        bias,
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

fn record(obs: u32, uuid: &str, activation: f32) -> DiscoverRecord {
    DiscoverRecord::new(obs, uuid.to_string(), None, activation, vec![])
}

/// Pre-activation of `target` at observation `obs`: `bias + Σ w·activation` over
/// the target's incoming synapses, reading source activations from the recorded
/// window. Removed sources contribute via the folded bias instead of a synapse,
/// so this reproduces the network output regardless of whether the fold ran.
fn output_pre_activation(
    creature: &CreatureJson,
    activations: &HashMap<(u32, String), f64>,
    target: &str,
    obs: u32,
) -> f64 {
    let bias = creature
        .neurons
        .iter()
        .find(|n| n.uuid == target)
        .map_or(0.0, |n| f64::from(n.bias));
    let contribution: f64 = creature
        .synapses
        .iter()
        .filter(|s| s.to_uuid == target)
        .map(|s| {
            let act = activations
                .get(&(obs, s.from_uuid.clone()))
                .copied()
                .unwrap_or(0.0);
            f64::from(s.weight) * act
        })
        .sum();
    bias + contribution
}

#[test]
fn constant_neuron_fold_preserves_outputs_across_window_and_removes_neuron() {
    // in0 varies; "const" is functionally constant at 4.0; "out" is an IDENTITY
    // output fed by in0 and const. An incoming edge into const also exists, to
    // confirm dangling incoming edges are cleaned up.
    let mut creature = CreatureJson {
        neurons: vec![
            neuron("in0", "input", 0.0),
            neuron("const", "hidden", 0.0),
            neuron("out", "output", 0.25),
        ],
        synapses: vec![
            synapse("in0", "const", 0.7),
            synapse("in0", "out", 1.5),
            synapse("const", "out", 2.0),
        ],
        input: 1,
        output: 1,
    };

    let obs_inputs = [0.1_f32, -0.4, 0.9, 2.0];
    let mut records = Vec::new();
    let mut activations: HashMap<(u32, String), f64> = HashMap::new();
    for (i, &x) in obs_inputs.iter().enumerate() {
        let obs = i as u32;
        records.push(record(obs, "in0", x));
        records.push(record(obs, "const", 4.0));
        activations.insert((obs, "in0".to_string()), f64::from(x));
        activations.insert((obs, "const".to_string()), 4.0);
    }

    // Output pre-activations before the fold.
    let before: Vec<f64> = (0..obs_inputs.len() as u32)
        .map(|obs| output_pre_activation(&creature, &activations, "out", obs))
        .collect();

    let outcome = fold_and_remove_constant_neuron(&mut creature, &records, "const", TOLERANCE);
    assert!(
        outcome.accepted,
        "constant neuron fold should pass the gate"
    );

    // Neuron and all its edges (incoming and outgoing) are gone.
    assert!(!creature.neurons.iter().any(|n| n.uuid == "const"));
    assert!(
        !creature
            .synapses
            .iter()
            .any(|s| s.from_uuid == "const" || s.to_uuid == "const"),
        "no dangling edge touching the removed neuron should survive"
    );

    // Output pre-activations after the fold are bit-identical within tolerance.
    for obs in 0..obs_inputs.len() as u32 {
        let after = output_pre_activation(&creature, &activations, "out", obs);
        assert!(
            (after - before[obs as usize]).abs() < 1e-5,
            "obs {obs}: output changed {} -> {after}",
            before[obs as usize]
        );
    }
}

#[test]
fn multi_target_fan_out_folds_each_target_independently() {
    // "const" (activation 3.0) fans out to two outputs with distinct weights.
    let mut creature = CreatureJson {
        neurons: vec![
            neuron("const", "hidden", 0.0),
            neuron("out1", "output", 0.1),
            neuron("out2", "output", -0.2),
        ],
        synapses: vec![
            synapse("const", "out1", 2.0),
            synapse("const", "out2", -3.0),
        ],
        input: 1,
        output: 2,
    };
    let records = vec![
        record(0, "const", 3.0),
        record(1, "const", 3.0),
        record(2, "const", 3.0),
    ];

    let outcome = fold_and_remove_constant_neuron(&mut creature, &records, "const", TOLERANCE);
    assert!(outcome.accepted);

    // out1: 0.1 + 2.0 * 3.0 = 6.1
    let out1 = creature.neurons.iter().find(|n| n.uuid == "out1").unwrap();
    assert!((f64::from(out1.bias) - 6.1).abs() < 1e-5);
    // out2: -0.2 + (-3.0) * 3.0 = -9.2
    let out2 = creature.neurons.iter().find(|n| n.uuid == "out2").unwrap();
    assert!((f64::from(out2.bias) - (-9.2)).abs() < 1e-5);

    // Each folded target is reported with its own w × c delta.
    assert_eq!(outcome.folded_targets.len(), 2);
    let delta = |uuid: &str| {
        outcome
            .folded_targets
            .iter()
            .find(|t| t.target_uuid == uuid)
            .map(|t| t.bias_delta)
            .unwrap()
    };
    assert!((delta("out1") - 6.0).abs() < 1e-9);
    assert!((delta("out2") - (-9.0)).abs() < 1e-9);
}

#[test]
fn gate_rejects_non_constant_neuron_and_leaves_network_unchanged() {
    let mut creature = CreatureJson {
        neurons: vec![neuron("vary", "hidden", 0.0), neuron("out", "output", 0.5)],
        synapses: vec![synapse("vary", "out", 2.0)],
        input: 1,
        output: 1,
    };
    let before = creature.clone();
    // Genuinely varying activation — folding a single constant is not exact.
    let records = vec![
        record(0, "vary", 1.0),
        record(1, "vary", 5.0),
        record(2, "vary", -2.0),
    ];

    let outcome = fold_and_remove_constant_neuron(&mut creature, &records, "vary", TOLERANCE);
    assert!(!outcome.accepted, "varying neuron must be rejected");
    assert!(outcome.rejection_reason.is_some());
    assert!(outcome.max_residual > TOLERANCE);

    // Network is byte-for-byte unchanged.
    assert_eq!(creature.neurons.len(), before.neurons.len());
    assert_eq!(creature.synapses.len(), before.synapses.len());
    let out = creature.neurons.iter().find(|n| n.uuid == "out").unwrap();
    assert!((f64::from(out.bias) - 0.5).abs() < f32::EPSILON.into());
    assert!(creature.neurons.iter().any(|n| n.uuid == "vary"));
}

#[test]
fn evaluate_does_not_mutate_creature() {
    let creature = CreatureJson {
        neurons: vec![neuron("const", "hidden", 0.0), neuron("out", "output", 1.0)],
        synapses: vec![synapse("const", "out", 2.0)],
        input: 1,
        output: 1,
    };
    let records = vec![record(0, "const", 5.0), record(1, "const", 5.0)];

    let outcome =
        evaluate_constant_neuron_bias_fold(&creature, &records, "const", TOLERANCE).unwrap();
    assert!(outcome.accepted);
    // Pure evaluation: the creature still contains the neuron and its edge.
    assert!(creature.neurons.iter().any(|n| n.uuid == "const"));
    assert_eq!(creature.synapses.len(), 1);
}
