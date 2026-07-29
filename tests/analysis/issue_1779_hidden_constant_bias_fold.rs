//! Regression tests for Issue #1779: the constant-neuron bias fold must fire for
//! **hidden** functionally-constant neurons.
//!
//! The #1690 wiring gated the #1623 fold on the *declared* neuron class
//! (`neuron_type == "constant"`), but every producer of a sole-op `RemoveNeuron`
//! emits **hidden** neurons — `"constant"` is the NEAT-AI input-side bias class.
//! The gate therefore never fired for an emitted candidate: a
//! functionally-constant hidden neuron (the realistic case) was proposed for
//! removal carrying no fold at all, and fell through to the #1559 redistribution
//! path where a zero-variance neuron short-circuits to a `delta_weight: 0.0`
//! remedy carrying no bias information.
//!
//! The gate is now **measured** constancy — the fold's own evaluate-before-accept
//! result — so a hidden neuron whose recorded activations are constant within
//! tolerance carries the folded per-target bias deltas, whatever its declared
//! type.

use neat_ai_discovery::CoordinatedStructuralCandidateJson;
use neat_ai_discovery::CoordinatedStructuralOpJson;
use neat_ai_discovery::CreatureJson;
use neat_ai_discovery::analysis::discovery_dispatch::{
    apply_constant_neuron_bias_fold, apply_remove_neuron_compensation,
};
use neat_ai_discovery::types::DiscoverRecord;

fn record(obs: u32, uuid: &str, activation: f32) -> DiscoverRecord {
    DiscoverRecord::new(obs, uuid.to_string(), None, activation, vec![])
}

/// A creature whose removable neuron `quiet` is declared **hidden** — exactly what
/// every sole-op `RemoveNeuron` producer emits — feeding a single output with
/// weight `3.0`, alongside a variance-carrying survivor `sib`.
fn creature_with_hidden_neuron() -> CreatureJson {
    serde_json::from_str(
        r#"{
            "input": 1, "output": 1,
            "neurons": [
                {"uuid": "input-0", "type": "input"},
                {"uuid": "quiet", "type": "hidden", "squash": "IDENTITY", "bias": 0.02},
                {"uuid": "sib", "type": "hidden", "squash": "IDENTITY", "bias": 0.0},
                {"uuid": "out-0", "type": "output", "squash": "IDENTITY", "bias": 0.5}
            ],
            "synapses": [
                {"fromUUID": "input-0", "toUUID": "sib", "weight": 1.0},
                {"fromUUID": "quiet", "toUUID": "out-0", "weight": 3.0},
                {"fromUUID": "sib", "toUUID": "out-0", "weight": 1.0}
            ]
        }"#,
    )
    .expect("valid creature JSON")
}

fn remove_neuron_candidate(uuid: &str) -> CoordinatedStructuralCandidateJson {
    CoordinatedStructuralCandidateJson {
        operations: vec![CoordinatedStructuralOpJson::RemoveNeuron {
            neuron_uuid: uuid.to_string(),
        }],
        expected_creature_score_gain: -0.01,
        ..Default::default()
    }
}

/// Constant activations for `quiet`, plus a varying survivor so the fixture is
/// not degenerate.
fn hidden_constant_records() -> Vec<DiscoverRecord> {
    let mut records = Vec::new();
    for (obs, sib) in [0.1_f32, -0.4, 0.9, 0.2].iter().enumerate() {
        let obs = u32::try_from(obs).expect("small index");
        records.push(record(obs, "quiet", 0.02));
        records.push(record(obs, "sib", *sib));
    }
    records
}

/// The core regression: a **hidden** functionally-constant neuron carries the
/// bias fold. Before Issue #1779 the declared-class gate rejected it and no fold
/// was emitted.
#[test]
fn hidden_functionally_constant_neuron_carries_bias_fold() {
    let creature = creature_with_hidden_neuron();
    let records = hidden_constant_records();
    let mut candidates = vec![remove_neuron_candidate("quiet")];

    let attached = apply_constant_neuron_bias_fold(&creature, &records, &mut candidates);
    assert_eq!(
        attached, 1,
        "a hidden neuron measured constant must be folded, not skipped for lacking \
         the declared \"constant\" class"
    );

    let fold = candidates[0]
        .constant_neuron_bias_fold
        .as_ref()
        .expect("bias fold attached to the hidden constant candidate");
    assert!(
        (fold.constant_activation - 0.02).abs() < 1e-6,
        "folded constant must be the measured mean 0.02, got {}",
        fold.constant_activation
    );
    assert_eq!(fold.folded_targets.len(), 1, "one downstream target folded");
    assert_eq!(fold.folded_targets[0].target_neuron_uuid, "out-0");
    // Δbias = w × c = 3.0 × 0.02 = 0.06 — real bias information, not the
    // zero-valued redistribution remedy the old path produced.
    assert!(
        (fold.folded_targets[0].bias_delta - 0.06).abs() < 1e-6,
        "folded bias delta must be w×c = 0.06, got {}",
        fold.folded_targets[0].bias_delta
    );
}

/// Routing stays mutually exclusive under the measured gate: the hidden constant
/// neuron takes the fold and is **not** given the #1559 redistribution remedy
/// (which would short-circuit to a `delta_weight: 0.0` remedy carrying no bias
/// information).
#[test]
fn hidden_constant_neuron_is_not_given_a_redistribution_remedy() {
    let creature = creature_with_hidden_neuron();
    let records = hidden_constant_records();
    let mut candidates = vec![remove_neuron_candidate("quiet")];

    let redistributed = apply_remove_neuron_compensation(&creature, &records, &mut candidates);
    let folded = apply_constant_neuron_bias_fold(&creature, &records, &mut candidates);

    assert_eq!(
        redistributed, 0,
        "a measured-constant neuron carries no variance to redistribute"
    );
    assert_eq!(folded, 1, "it takes the bias fold instead");
    assert!(candidates[0].remove_neuron_compensation.is_none());
    assert!(candidates[0].constant_neuron_bias_fold.is_some());
}

/// A variance-carrying hidden neuron still routes to redistribution and receives
/// no fold — the measured gate must not swallow the #1559 path.
#[test]
fn variance_carrying_hidden_neuron_still_routes_to_redistribution() {
    let creature = creature_with_hidden_neuron();
    let mut records = Vec::new();
    for (obs, a) in [1.0_f32, 2.0, 3.0, 4.0].iter().enumerate() {
        let obs = u32::try_from(obs).expect("small index");
        records.push(record(obs, "quiet", *a));
        records.push(record(obs, "sib", *a));
    }
    let mut candidates = vec![remove_neuron_candidate("quiet")];

    let folded = apply_constant_neuron_bias_fold(&creature, &records, &mut candidates);
    let redistributed = apply_remove_neuron_compensation(&creature, &records, &mut candidates);

    assert_eq!(folded, 0, "a varying neuron is rejected by the fold gate");
    assert_eq!(redistributed, 1, "it takes the redistribution remedy");
    assert!(candidates[0].constant_neuron_bias_fold.is_none());
    assert!(candidates[0].remove_neuron_compensation.is_some());
}

/// The gate is measurement, not declaration: a neuron declared `"constant"` whose
/// recorded activations vary beyond tolerance is still rejected fail-loud, and now
/// routes to redistribution rather than being parked on the fold path.
#[test]
fn declared_constant_class_with_varying_records_is_still_rejected() {
    let creature: CreatureJson = serde_json::from_str(
        r#"{
            "input": 1, "output": 1,
            "neurons": [
                {"uuid": "input-0", "type": "input"},
                {"uuid": "konst", "type": "constant", "squash": "IDENTITY", "bias": 0.0},
                {"uuid": "sib", "type": "hidden", "squash": "IDENTITY", "bias": 0.0},
                {"uuid": "out-0", "type": "output", "squash": "IDENTITY", "bias": 0.0}
            ],
            "synapses": [
                {"fromUUID": "input-0", "toUUID": "sib", "weight": 1.0},
                {"fromUUID": "konst", "toUUID": "out-0", "weight": 2.0},
                {"fromUUID": "sib", "toUUID": "out-0", "weight": 1.0}
            ]
        }"#,
    )
    .expect("valid creature JSON");

    let mut records = Vec::new();
    for (obs, a) in [1.0_f32, 5.0, -2.0, 3.0].iter().enumerate() {
        let obs = u32::try_from(obs).expect("small index");
        records.push(record(obs, "konst", *a));
        records.push(record(obs, "sib", *a));
    }
    let mut candidates = vec![remove_neuron_candidate("konst")];

    let folded = apply_constant_neuron_bias_fold(&creature, &records, &mut candidates);
    assert_eq!(
        folded, 0,
        "the declared class must not buy a fold the measurement rejects"
    );
    assert!(candidates[0].constant_neuron_bias_fold.is_none());

    let redistributed = apply_remove_neuron_compensation(&creature, &records, &mut candidates);
    assert_eq!(
        redistributed, 1,
        "a declared-constant neuron that actually varies takes the redistribution remedy"
    );
}

/// A neuron with no outgoing synapses has nothing to fold: an empty fold is not a
/// remedy, so none is attached (and the candidate is not parked on the fold path).
#[test]
fn neuron_without_outgoing_synapses_gets_no_empty_fold() {
    let creature: CreatureJson = serde_json::from_str(
        r#"{
            "input": 1, "output": 1,
            "neurons": [
                {"uuid": "input-0", "type": "input"},
                {"uuid": "orphan", "type": "hidden", "squash": "IDENTITY", "bias": 0.0},
                {"uuid": "out-0", "type": "output", "squash": "IDENTITY", "bias": 0.0}
            ],
            "synapses": [
                {"fromUUID": "input-0", "toUUID": "orphan", "weight": 1.0}
            ]
        }"#,
    )
    .expect("valid creature JSON");
    let records: Vec<DiscoverRecord> = (0..4).map(|obs| record(obs, "orphan", 0.5)).collect();
    let mut candidates = vec![remove_neuron_candidate("orphan")];

    let folded = apply_constant_neuron_bias_fold(&creature, &records, &mut candidates);
    assert_eq!(folded, 0, "no downstream target ⇒ no fold to attach");
    assert!(candidates[0].constant_neuron_bias_fold.is_none());
}
