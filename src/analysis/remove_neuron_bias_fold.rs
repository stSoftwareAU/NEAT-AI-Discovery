//! Bias-fold removal for functionally-constant hidden neurons (Issue #1623).
//!
//! ## Why bias-fold is exact for a constant neuron
//!
//! A neuron whose activation is a constant `c` on every observation contributes
//! a fixed `w_{n→t} · c` to each downstream target `t`. Adding `w_{n→t} · c` to
//! target `t`'s bias reproduces that contribution exactly, so removing the
//! neuron and applying the bias fold is **behaviour-preserving on the recorded
//! window**: the target's pre-activation — `bias_t + Σ_s w_{s→t}·a_{s,i}` — is
//! unchanged for every observation `i`, because the removed term
//! `w_{n→t}·a_{n,i}` is replaced by the folded constant `w_{n→t}·c` and
//! `a_{n,i} = c`. With the pre-activation preserved, the post-squash activation
//! and everything downstream are preserved too.
//!
//! Unlike the general remove-neuron case (Issue #1559,
//! [`super::remove_neuron_compensation`]), no per-sample variance survives —
//! this is the constant-activation special case where mean-preserving bias
//! compensation is *complete*, not merely mean-cancelling.
//!
//! ## Evaluate-before-accept gate
//!
//! The fold value `c` is the mean activation over the recorded window. The gate
//! is the safety net: for every outgoing connection and every observation it
//! measures the residual `|w_{n→t}·(a_{n,i} − c)|` — the exact per-sample
//! deviation the fold would introduce. A genuinely constant neuron leaves a zero
//! residual and is accepted; a neuron that only *looks* constant leaves a
//! residual above tolerance and is **rejected**, so the neuron is never deleted
//! blind. Rejection leaves the creature untouched.
//!
//! The mean/variance accumulation and the `VARIANCE_EPSILON` threshold are
//! reused from [`super::remove_neuron_compensation`] rather than duplicated.

use crate::CreatureJson;
use crate::analysis::remove_neuron_compensation::{ActivationCovariance, VARIANCE_EPSILON};
use crate::types::DiscoverRecord;

/// Default tolerance for the evaluate-before-accept gate.
///
/// The gate accepts a fold only when the maximum per-sample residual it would
/// introduce across every target is at or below this bound. `1e-6` sits well
/// above `f32`-recording round-off yet far below any activation deviation a
/// genuinely non-constant neuron produces.
pub const BIAS_FOLD_GATE_TOLERANCE: f64 = 1e-6;

/// A single downstream target that receives a folded bias delta (Issue #1623).
#[derive(Debug, Clone, PartialEq)]
pub struct FoldedTarget {
    /// The downstream neuron whose bias absorbs the constant contribution.
    pub target_uuid: String,
    /// The bias delta added to the target: `outgoing_weight × constant_activation`.
    pub bias_delta: f64,
}

/// The outcome of evaluating (and possibly applying) a constant-neuron bias fold
/// (Issue #1623).
#[derive(Debug, Clone, PartialEq)]
pub struct BiasFoldOutcome {
    /// `true` when the evaluate-before-accept gate passed and the fold was
    /// applied; `false` when the gate rejected it (the creature is unchanged).
    pub accepted: bool,
    /// The mean activation used as the folded constant `c`.
    pub constant_activation: f64,
    /// The population variance of the neuron's activation over the window —
    /// (near-)zero for a genuinely constant neuron.
    pub activation_variance: f64,
    /// The maximum per-sample residual `|w·(a_i − c)|` across every outgoing
    /// connection and observation. Compared against the gate tolerance.
    pub max_residual: f64,
    /// Per-target bias deltas that were (or would have been) applied.
    pub folded_targets: Vec<FoldedTarget>,
    /// When the fold could not be evaluated or was rejected, a short reason;
    /// `None` on acceptance.
    pub rejection_reason: Option<String>,
}

/// Collect one neuron's per-observation activations from the recorded window.
fn neuron_activations(records: &[DiscoverRecord], neuron_uuid: &str) -> Vec<f64> {
    records
        .iter()
        .filter(|r| r.neuron_uuid == neuron_uuid)
        .map(|r| f64::from(r.activation))
        .collect()
}

/// Evaluate the constant-neuron bias fold **without** mutating the creature
/// (Issue #1623).
///
/// Derives the folded constant `c` as the mean activation over the recorded
/// window, computes the per-target bias deltas (`w_{n→t} · c`), and runs the
/// evaluate-before-accept gate: the maximum per-sample residual
/// `|w_{n→t}·(a_{n,i} − c)|` across every outgoing connection and observation is
/// compared against `tolerance`.
///
/// Returns `None` when the fold cannot be evaluated at all — the neuron has no
/// recorded activations, so its constancy cannot be verified. A missing neuron
/// is a fail-loud condition (no blind delete), not a silent no-op.
#[must_use]
pub fn evaluate_constant_neuron_bias_fold(
    creature: &CreatureJson,
    records: &[DiscoverRecord],
    neuron_uuid: &str,
    tolerance: f64,
) -> Option<BiasFoldOutcome> {
    let activations = neuron_activations(records, neuron_uuid);
    let stats = ActivationCovariance::from_values(activations.iter().copied())?;
    let constant = stats.candidate_mean;
    let variance = stats.candidate_variance;

    // Fold every outgoing connection into its target's bias, and measure the
    // per-sample residual each connection would leave behind.
    let mut folded_targets = Vec::new();
    let mut max_residual = 0.0_f64;
    let mut max_abs_deviation = 0.0_f64;
    for activation in &activations {
        max_abs_deviation = max_abs_deviation.max((activation - constant).abs());
    }
    for synapse in creature
        .synapses
        .iter()
        .filter(|s| s.from_uuid == neuron_uuid)
    {
        let weight = f64::from(synapse.weight);
        folded_targets.push(FoldedTarget {
            target_uuid: synapse.to_uuid.clone(),
            bias_delta: weight * constant,
        });
        max_residual = max_residual.max(weight.abs() * max_abs_deviation);
    }

    // A genuinely constant neuron leaves a residual at or below tolerance; a
    // neuron that only looks constant is rejected so it is never deleted blind.
    let accepted = variance <= VARIANCE_EPSILON || max_residual <= tolerance;
    let rejection_reason = if accepted {
        None
    } else {
        Some(format!(
            "per-sample residual {max_residual:.3e} exceeds gate tolerance {tolerance:.3e}"
        ))
    };

    Some(BiasFoldOutcome {
        accepted,
        constant_activation: constant,
        activation_variance: variance,
        max_residual,
        folded_targets,
        rejection_reason,
    })
}

/// Apply an accepted fold in place: add each target's bias delta, then delete the
/// neuron and all synapses touching it (Issue #1623).
///
/// Removes both outgoing and incoming connections so no dangling edge survives.
fn apply_fold(creature: &mut CreatureJson, neuron_uuid: &str, folded_targets: &[FoldedTarget]) {
    for folded in folded_targets {
        for neuron in creature
            .neurons
            .iter_mut()
            .filter(|n| n.uuid == folded.target_uuid)
        {
            let updated = f64::from(neuron.bias) + folded.bias_delta;
            // Recorded biases are `f32`; the fold delta is computed in `f64` for
            // precision and narrowed back to the network's `f32` representation.
            #[allow(clippy::cast_possible_truncation)]
            {
                neuron.bias = updated as f32;
            }
        }
    }
    creature.neurons.retain(|n| n.uuid != neuron_uuid);
    creature
        .synapses
        .retain(|s| s.from_uuid != neuron_uuid && s.to_uuid != neuron_uuid);
}

/// Fold a functionally-constant neuron's contribution into its targets' biases
/// and remove it — behind the evaluate-before-accept gate (Issue #1623).
///
/// Evaluates the fold, and **only if the gate passes** applies it: each target's
/// bias gains `outgoing_weight × constant_activation`, then the neuron and every
/// synapse touching it are deleted. If the gate rejects the fold (the neuron's
/// activation varies beyond `tolerance`) or the fold cannot be evaluated (no
/// recorded activations), the creature is left **exactly** as it was and the
/// returned outcome reports why.
pub fn fold_and_remove_constant_neuron(
    creature: &mut CreatureJson,
    records: &[DiscoverRecord],
    neuron_uuid: &str,
    tolerance: f64,
) -> BiasFoldOutcome {
    let Some(outcome) =
        evaluate_constant_neuron_bias_fold(creature, records, neuron_uuid, tolerance)
    else {
        return BiasFoldOutcome {
            accepted: false,
            constant_activation: 0.0,
            activation_variance: 0.0,
            max_residual: 0.0,
            folded_targets: Vec::new(),
            rejection_reason: Some(format!(
                "no recorded activations for neuron {neuron_uuid}; cannot verify constancy"
            )),
        };
    };

    if outcome.accepted {
        apply_fold(creature, neuron_uuid, &outcome.folded_targets);
    }
    outcome
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi_types::{NeuronJson, SynapseJson};

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

    #[test]
    fn constant_neuron_fold_updates_target_bias_and_is_accepted() {
        let mut creature = CreatureJson {
            neurons: vec![neuron("const", "hidden", 0.0), neuron("out", "output", 0.5)],
            synapses: vec![synapse("const", "out", 2.0)],
            input: 1,
            output: 1,
        };
        let records = vec![
            record(0, "const", 3.0),
            record(1, "const", 3.0),
            record(2, "const", 3.0),
        ];

        let outcome = fold_and_remove_constant_neuron(&mut creature, &records, "const", 1e-6);
        assert!(outcome.accepted);
        // 0.5 + 2.0 * 3.0 = 6.5
        let out = creature.neurons.iter().find(|n| n.uuid == "out").unwrap();
        assert!((f64::from(out.bias) - 6.5).abs() < 1e-5);
        assert!(!creature.neurons.iter().any(|n| n.uuid == "const"));
        assert!(creature.synapses.is_empty());
    }

    #[test]
    fn varying_neuron_is_rejected_and_creature_unchanged() {
        let mut creature = CreatureJson {
            neurons: vec![neuron("vary", "hidden", 0.0), neuron("out", "output", 0.5)],
            synapses: vec![synapse("vary", "out", 2.0)],
            input: 1,
            output: 1,
        };
        let before = creature.clone();
        let records = vec![
            record(0, "vary", 1.0),
            record(1, "vary", 5.0),
            record(2, "vary", -2.0),
        ];

        let outcome = fold_and_remove_constant_neuron(&mut creature, &records, "vary", 1e-6);
        assert!(!outcome.accepted);
        assert!(outcome.rejection_reason.is_some());
        assert_eq!(creature.neurons.len(), before.neurons.len());
        assert_eq!(creature.synapses.len(), before.synapses.len());
    }

    #[test]
    fn missing_records_fail_loud_without_deleting() {
        let mut creature = CreatureJson {
            neurons: vec![neuron("const", "hidden", 0.0), neuron("out", "output", 0.0)],
            synapses: vec![synapse("const", "out", 1.0)],
            input: 1,
            output: 1,
        };
        let outcome = fold_and_remove_constant_neuron(&mut creature, &[], "const", 1e-6);
        assert!(!outcome.accepted);
        assert!(outcome.rejection_reason.is_some());
        assert!(creature.neurons.iter().any(|n| n.uuid == "const"));
    }
}
