//! Integration tests for Issue #1690: wire the #1623 constant-neuron bias fold
//! into the live remove-neuron dispatch path.
//!
//! Milestone #1623 built `evaluate_constant_neuron_bias_fold` /
//! `fold_and_remove_constant_neuron` but nothing in the live pipeline invoked
//! them, so a genuinely-constant remove-neuron candidate carried no bias-fold
//! remedy and the applier fell back to a mean-only fold. This complements the
//! #1559 variance-redistribution wiring (Issue #1689): the two remedies route by
//! neuron *class* and must not overlap.
//!
//! These tests assert the wiring behaviour of
//! `apply_constant_neuron_bias_fold`:
//!
//! - A genuinely-constant remove-neuron candidate is emitted with the folded bias
//!   deltas (`w × c` per target), behind the evaluate-before-accept gate.
//! - The emitted deltas match the accepted [`BiasFoldOutcome`].
//! - Routing is by class: a constant neuron is **not** given a redistribution
//!   remedy, and a variance-carrying neuron is **not** given a bias fold.
//! - A neuron that only *looks* constant (records vary over tolerance) or has no
//!   recorded activations is rejected fail-loud — no fold is emitted and the
//!   creature is never mutated by the wiring.

// Small fixed observation counts cast cleanly to `u32` in these fixtures.
#![allow(clippy::cast_possible_truncation)]

use neat_ai_discovery::CoordinatedStructuralCandidateJson;
use neat_ai_discovery::CoordinatedStructuralOpJson;
use neat_ai_discovery::CreatureJson;
use neat_ai_discovery::analysis::discovery_dispatch::{
    apply_constant_neuron_bias_fold, apply_remove_neuron_compensation,
};
use neat_ai_discovery::analysis::remove_neuron_bias_fold::{
    BIAS_FOLD_GATE_TOLERANCE, evaluate_constant_neuron_bias_fold,
};
use neat_ai_discovery::types::DiscoverRecord;

fn record(obs: u32, uuid: &str, activation: f32) -> DiscoverRecord {
    DiscoverRecord::new(obs, uuid.to_string(), None, activation, vec![])
}

/// A creature with a **constant**-class neuron `konst` feeding a single output
/// `out-0`, plus a variance-carrying survivor `sib` feeding the same output.
fn creature_with_constant_neuron() -> CreatureJson {
    serde_json::from_str(
        r#"{
            "input": 1, "output": 1,
            "neurons": [
                {"uuid": "input-0", "type": "input"},
                {"uuid": "konst", "type": "constant", "squash": "IDENTITY", "bias": 0.0},
                {"uuid": "sib", "type": "hidden", "squash": "IDENTITY", "bias": 0.0},
                {"uuid": "out-0", "type": "output", "squash": "IDENTITY", "bias": 0.5}
            ],
            "synapses": [
                {"fromUUID": "input-0", "toUUID": "sib", "weight": 1.0},
                {"fromUUID": "konst", "toUUID": "out-0", "weight": 2.0},
                {"fromUUID": "sib", "toUUID": "out-0", "weight": 1.0}
            ]
        }"#,
    )
    .expect("valid creature JSON")
}

/// A creature with a **constant**-class neuron `konst` fanning out to two
/// downstream targets, each with its own outgoing weight.
fn creature_with_constant_fanout() -> CreatureJson {
    serde_json::from_str(
        r#"{
            "input": 1, "output": 2,
            "neurons": [
                {"uuid": "input-0", "type": "input"},
                {"uuid": "konst", "type": "constant", "squash": "IDENTITY", "bias": 0.0},
                {"uuid": "out-0", "type": "output", "squash": "IDENTITY", "bias": 0.0},
                {"uuid": "out-1", "type": "output", "squash": "IDENTITY", "bias": 1.0}
            ],
            "synapses": [
                {"fromUUID": "konst", "toUUID": "out-0", "weight": 2.0},
                {"fromUUID": "konst", "toUUID": "out-1", "weight": -3.0}
            ]
        }"#,
    )
    .expect("valid creature JSON")
}

/// A creature with a variance-carrying **hidden** neuron `deep` and survivor
/// `sib` both feeding `out-0`. `deep` is not a constant-class neuron.
fn creature_with_deep_neuron() -> CreatureJson {
    serde_json::from_str(
        r#"{
            "input": 1, "output": 1,
            "neurons": [
                {"uuid": "input-0", "type": "input"},
                {"uuid": "deep", "type": "hidden", "squash": "IDENTITY", "bias": 0.0},
                {"uuid": "sib", "type": "hidden", "squash": "IDENTITY", "bias": 0.0},
                {"uuid": "out-0", "type": "output", "squash": "IDENTITY", "bias": 0.0}
            ],
            "synapses": [
                {"fromUUID": "input-0", "toUUID": "deep", "weight": 1.0},
                {"fromUUID": "input-0", "toUUID": "sib", "weight": 1.0},
                {"fromUUID": "deep", "toUUID": "out-0", "weight": 1.0},
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

/// Constant-within-tolerance activations for a single neuron.
fn constant_records(uuid: &str, value: f32) -> Vec<DiscoverRecord> {
    (0..3).map(|obs| record(obs, uuid, value)).collect()
}

/// The core wiring: a genuinely-constant remove-neuron candidate is emitted with
/// the folded bias deltas (`w × c` per target), behind the gate.
#[test]
fn constant_neuron_candidate_carries_bias_fold() {
    let creature = creature_with_constant_neuron();
    let records = constant_records("konst", 3.0);
    let mut candidates = vec![remove_neuron_candidate("konst")];

    let attached = apply_constant_neuron_bias_fold(&creature, &records, &mut candidates);
    assert_eq!(
        attached, 1,
        "the constant candidate is compensated by a fold"
    );

    let fold = candidates[0]
        .constant_neuron_bias_fold
        .as_ref()
        .expect("bias fold attached to the constant candidate");

    assert!(
        (fold.constant_activation - 3.0).abs() < 1e-5,
        "folded constant must be the mean activation 3.0, got {}",
        fold.constant_activation
    );
    assert_eq!(fold.folded_targets.len(), 1, "one downstream target folded");
    let target = &fold.folded_targets[0];
    assert_eq!(target.target_neuron_uuid, "out-0");
    // Δbias = w × c = 2.0 × 3.0 = 6.0.
    assert!(
        (target.bias_delta - 6.0).abs() < 1e-5,
        "folded bias delta must be w×c = 6.0, got {}",
        target.bias_delta
    );
    // A genuinely-constant neuron leaves ~zero residual and no variance.
    assert!(fold.max_residual <= BIAS_FOLD_GATE_TOLERANCE as f32);
    assert!(fold.activation_variance < 1e-6);
}

/// The emitted deltas must match the accepted [`BiasFoldOutcome`] exactly across
/// a multi-target fan-out.
#[test]
fn emitted_deltas_match_the_accepted_fold() {
    let creature = creature_with_constant_fanout();
    let records = constant_records("konst", 4.0);
    let mut candidates = vec![remove_neuron_candidate("konst")];

    let outcome =
        evaluate_constant_neuron_bias_fold(&creature, &records, "konst", BIAS_FOLD_GATE_TOLERANCE)
            .expect("fold evaluates for a neuron with recorded activations");
    assert!(outcome.accepted, "constant neuron fold is accepted");

    let attached = apply_constant_neuron_bias_fold(&creature, &records, &mut candidates);
    assert_eq!(attached, 1);

    let fold = candidates[0]
        .constant_neuron_bias_fold
        .as_ref()
        .expect("bias fold attached");
    assert_eq!(
        fold.folded_targets.len(),
        outcome.folded_targets.len(),
        "every outgoing target is folded"
    );
    for expected in &outcome.folded_targets {
        let got = fold
            .folded_targets
            .iter()
            .find(|t| t.target_neuron_uuid == expected.target_uuid)
            .unwrap_or_else(|| panic!("target {} present in emitted fold", expected.target_uuid));
        assert!(
            (f64::from(got.bias_delta) - expected.bias_delta).abs() < 1e-5,
            "emitted delta for {} must match the accepted fold ({} vs {})",
            expected.target_uuid,
            got.bias_delta,
            expected.bias_delta
        );
    }
}

/// Routing: a constant neuron takes the bias fold, **not** the #1559
/// redistribution; a variance-carrying neuron takes redistribution, **not** the
/// bias fold. The two remedies are mutually exclusive per candidate.
#[test]
fn bias_fold_and_redistribution_are_mutually_exclusive() {
    // Constant neuron: fold yes, redistribution no.
    let creature = creature_with_constant_neuron();
    let records = constant_records("konst", 3.0);
    let mut candidates = vec![remove_neuron_candidate("konst")];

    let redistributed = apply_remove_neuron_compensation(&creature, &records, &mut candidates);
    let folded = apply_constant_neuron_bias_fold(&creature, &records, &mut candidates);
    assert_eq!(redistributed, 0, "constant neuron gets no redistribution");
    assert_eq!(folded, 1, "constant neuron gets the bias fold");
    assert!(candidates[0].remove_neuron_compensation.is_none());
    assert!(candidates[0].constant_neuron_bias_fold.is_some());

    // Variance-carrying neuron: redistribution yes, fold no.
    let creature = creature_with_deep_neuron();
    let mut deep_records = Vec::new();
    for (obs, a) in [1.0_f32, 2.0, 3.0].iter().enumerate() {
        let obs = obs as u32;
        deep_records.push(record(obs, "deep", *a));
        deep_records.push(record(obs, "sib", *a));
    }
    let mut candidates = vec![remove_neuron_candidate("deep")];

    let folded = apply_constant_neuron_bias_fold(&creature, &deep_records, &mut candidates);
    let redistributed = apply_remove_neuron_compensation(&creature, &deep_records, &mut candidates);
    assert_eq!(folded, 0, "variance-carrying neuron gets no bias fold");
    assert_eq!(
        redistributed, 1,
        "variance-carrying neuron gets redistribution"
    );
    assert!(candidates[0].constant_neuron_bias_fold.is_none());
    assert!(candidates[0].remove_neuron_compensation.is_some());
}

/// A constant-class neuron whose records vary beyond tolerance only *looks*
/// constant: the gate rejects it, so no fold is emitted (fail-loud, never
/// deleted blind).
#[test]
fn looks_constant_over_tolerance_is_rejected_fail_loud() {
    let creature = creature_with_constant_neuron();
    // Records vary well beyond the gate tolerance despite the constant class.
    let records = vec![
        record(0, "konst", 1.0),
        record(1, "konst", 5.0),
        record(2, "konst", -2.0),
    ];
    let mut candidates = vec![remove_neuron_candidate("konst")];

    let folded = apply_constant_neuron_bias_fold(&creature, &records, &mut candidates);
    assert_eq!(
        folded, 0,
        "a looks-constant candidate is rejected by the gate"
    );
    assert!(
        candidates[0].constant_neuron_bias_fold.is_none(),
        "no fold is emitted for a rejected candidate — never deleted blind"
    );
}

/// A constant-class neuron with no recorded activations cannot have its constancy
/// verified: rejected fail-loud, no fold emitted.
#[test]
fn constant_neuron_without_records_is_rejected_fail_loud() {
    let creature = creature_with_constant_neuron();
    let mut candidates = vec![remove_neuron_candidate("konst")];

    let folded = apply_constant_neuron_bias_fold(&creature, &[], &mut candidates);
    assert_eq!(folded, 0, "no records ⇒ constancy cannot be verified");
    assert!(
        candidates[0].constant_neuron_bias_fold.is_none(),
        "no fold is fabricated without per-sample data"
    );
}

/// Multi-operation coordinated candidates are not bare neuron removals, so they
/// receive no bias fold.
#[test]
fn multi_op_candidate_gets_no_bias_fold() {
    let creature = creature_with_constant_neuron();
    let records = constant_records("konst", 3.0);
    let mut candidates = vec![CoordinatedStructuralCandidateJson {
        operations: vec![
            CoordinatedStructuralOpJson::SetBias {
                neuron_uuid: "out-0".to_string(),
                bias: 0.1,
            },
            CoordinatedStructuralOpJson::RemoveNeuron {
                neuron_uuid: "konst".to_string(),
            },
        ],
        expected_creature_score_gain: -0.01,
        ..Default::default()
    }];

    let folded = apply_constant_neuron_bias_fold(&creature, &records, &mut candidates);
    assert_eq!(folded, 0, "multi-op candidates are not bare removals");
    assert!(candidates[0].constant_neuron_bias_fold.is_none());
}

/// Non-`RemoveNeuron` single-op candidates receive no bias fold.
#[test]
fn non_remove_neuron_candidate_gets_no_bias_fold() {
    let creature = creature_with_constant_neuron();
    let records = constant_records("konst", 3.0);
    let mut candidates = vec![CoordinatedStructuralCandidateJson {
        operations: vec![CoordinatedStructuralOpJson::SetBias {
            neuron_uuid: "out-0".to_string(),
            bias: 0.2,
        }],
        expected_creature_score_gain: 0.3,
        ..Default::default()
    }];

    let folded = apply_constant_neuron_bias_fold(&creature, &records, &mut candidates);
    assert_eq!(folded, 0, "non-RemoveNeuron candidates get no fold");
    assert!(candidates[0].constant_neuron_bias_fold.is_none());
}
