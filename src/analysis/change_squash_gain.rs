//! Propagation-aware change-squash gain estimation (Issue #1532).
//!
//! Extends the #1518 propagation-aware approach — which fixed the
//! **remove-neuron** estimate — to the **change-squash** estimate path, the
//! second estimate path cited in the production discovery analysis.
//!
//! ## Why the change-squash placeholder was wrong
//!
//! For the recorded failure (`neuron-1481550544`, `SELU → SQUARE`) the pipeline
//! emitted `expectedCreatureScoreGain = 8.6e-10` — a near-zero placeholder that
//! is topology-blind and activation-blind — while the empirically measured
//! effect was `-0.000341` (the change made the trained creature slightly
//! *worse*). That is ~400,000× too small in magnitude and the wrong sign.
//!
//! ## Why structural influence alone is not enough here
//!
//! Removing a neuron zeroes its whole contribution, so its creature-level effect
//! is well approximated by its propagation-aware downstream influence
//! ([`estimate_remove_neuron_gain`](super::estimate_remove_neuron_gain)).
//! Changing a neuron's *activation function* is different: the effect is driven
//! by **how much the neuron's emitted output changes** when the squash is
//! swapped. Swapping `SELU → SQUARE` can amplify the neuron's output by orders
//! of magnitude, so the pure (small-perturbation) structural influence
//! under-predicts the effect badly — for `neuron-1481550544` the structural
//! influence is ~`4.3e-7`, ~800× below the measured `3.4e-4`.
//!
//! ## The propagation-aware estimate
//!
//! We combine two propagation-aware factors:
//!
//! - **Downstream influence** — the neuron's propagation-aware influence on the
//!   output(s), reused from [`compute_impacts_public`] exactly as the
//!   remove-neuron estimator does (DRY). A deep neuron attenuates to a tiny
//!   value.
//! - **Local perturbation scale** — how much the squash swap changes the
//!   neuron's own behaviour, measured by the reduction in the neuron's *local*
//!   error the candidate reports (`current_local_error − proposed_local_error`).
//!   Re-fitting the neuron's local target perturbs its emitted activation by a
//!   comparable amount; that is the quantity that then propagates downstream.
//!
//! The creature-level magnitude is `influence × local_perturbation`, and the
//! **sign is negative**: on a converged network the downstream layers were
//! trained around the neuron's *original* activation, so re-fitting it disrupts
//! that equilibrium and is expected to *reduce* the trained score. This is the
//! same honest non-positive prior the remove-neuron estimator uses, extended to
//! the change-squash perturbation scale — not a fabricated near-zero placeholder.

use crate::CreatureJson;
use crate::focus::compute_impacts_public;

/// Estimate the honest, propagation-aware creature-score gain from changing a
/// neuron's activation function (change-squash).
///
/// The returned value is signed:
/// - Its **magnitude** is the neuron's propagation-aware downstream influence
///   scaled by the local perturbation the squash swap induces (the reduction in
///   the neuron's local error). A deep neuron, or a swap that barely changes the
///   neuron's behaviour, attenuates to a tiny value.
/// - Its **sign** is negative (or zero): on a converged network, re-fitting a
///   neuron's activation disrupts the downstream layers that were trained around
///   its original behaviour, so the honest "gain" from the change is
///   non-positive.
///
/// # Arguments
/// * `creature` - The creature's network topology (neurons and synapses).
/// * `neuron_uuid` - UUID of the neuron whose squash change is being estimated.
/// * `current_local_error` - The neuron's local error under its current squash
///   (the candidate's `currentError`).
/// * `proposed_local_error` - The neuron's local error under the proposed squash
///   (the candidate's `improvedError`).
///
/// # Returns
/// `Some(gain)` where `gain <= 0.0`, or `None` when the neuron is not present in
/// the topology or is an output neuron (output activations are governed by the
/// loss contract, not change-squash candidates).
#[must_use]
pub fn estimate_change_squash_gain(
    creature: &CreatureJson,
    neuron_uuid: &str,
    current_local_error: f64,
    proposed_local_error: f64,
) -> Option<f64> {
    // Output neurons are not change-squash candidates.
    if creature
        .neurons
        .iter()
        .any(|n| n.uuid == neuron_uuid && n.neuron_type == "output")
    {
        return None;
    }

    let impacts = compute_impacts_public(creature);
    let influence = impacts.get(neuron_uuid).copied()?;

    // Downstream propagation (mirrors the remove-neuron estimator). Guard against
    // any negative/NaN influence leaking through.
    let influence = f64::from(influence).max(0.0);

    // Local perturbation scale: the swap re-fits the neuron's local target,
    // reducing its local error by this much and perturbing its emitted output by
    // a comparable amount. A non-improving swap (>= current error) contributes no
    // perturbation.
    let local_perturbation = (current_local_error - proposed_local_error).max(0.0);

    // Honest gain: the perturbation attenuates through the downstream influence,
    // and disrupts a network trained around the original activation, so the
    // creature-level effect is non-positive.
    Some(-(influence * local_perturbation))
}
