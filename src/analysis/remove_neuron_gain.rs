//! Propagation-aware remove-neuron gain estimation (Issue #1518).
//!
//! Replaces the fabricated floor-at-`0.1` placeholder gain (the NEAT-AI Deno
//! `#2483` over-threshold sink `0.1 + (log10(err) − 10)/10 × 0.4`, clamped to
//! `[0.1, 0.5]`) with a **propagation-aware** estimate of the effect on the
//! network output(s) of removing a neuron.
//!
//! ## Why the placeholder was wrong
//!
//! The placeholder ignored topology entirely: it turned a large squash error
//! into a large positive "gain" regardless of where the neuron sat in the
//! network. For a neuron many layers from the output(s), the activation is
//! attenuated / squashed repeatedly on the way to the output, so the true
//! effect of removing it is tiny. For the recorded failure example
//! (`neuron-1802938338`) the placeholder claimed `+0.17882921` while the
//! empirically measured effect was only `-0.000194` — ~920× too large and
//! opposite in sign.
//!
//! ## The propagation-aware estimate
//!
//! [`compute_impacts_public`] already
//! walks every downstream connection, applying each edge weight and the local
//! squash bound / derivative, accumulating the fraction of the output(s)'
//! sensitivity that flows through each neuron. For a deep neuron this
//! attenuates to a tiny value — exactly the propagation the placeholder
//! ignored.
//!
//! We reuse that machinery (DRY) and turn the unsigned influence fraction into
//! a **signed honest gain**:
//!
//! - **Magnitude** = the neuron's propagation-aware influence on the output(s).
//!   Deep neurons attenuate to ~`1e-4`, not the fabricated `0.1+`.
//! - **Sign** = negative. Removing a neuron that still carries genuine
//!   downstream influence removes that contribution, so the trained network's
//!   score is expected to *drop* by roughly its influence. A neuron with no
//!   downstream influence (dead / disconnected) attenuates to ~`0`, so its
//!   honest gain is ~`0` — neither a fabricated win nor a large loss.
//!
//! This keeps over-threshold "harmful" neurons removal-eligible (their honest
//! gain is ≈0 or slightly negative) without letting a fabricated large positive
//! gain crowd out realistic (~`1e-4`) candidates.

use crate::CreatureJson;
use crate::focus::compute_impacts_public;

/// Estimate the honest, propagation-aware creature-score gain from removing a
/// neuron.
///
/// The returned value is signed:
/// - Its **magnitude** is the neuron's propagation-aware influence on the
///   output(s) — the squash-bounded product of downstream edge weights
///   accumulated all the way to the output(s). Neurons many layers from the
///   output attenuate to a tiny value.
/// - Its **sign** is negative (or zero): removing a neuron that still carries
///   downstream influence is expected to *reduce* the trained network's score
///   by roughly its influence, so the honest "gain" from removal is
///   non-positive.
///
/// # Arguments
/// * `creature` - The creature's network topology (neurons and synapses).
/// * `neuron_uuid` - UUID of the neuron whose removal is being estimated.
///
/// # Returns
/// `Some(gain)` where `gain <= 0.0`, or `None` when the neuron is not present
/// in the topology or is an output neuron (outputs are not removal candidates).
#[must_use]
pub fn estimate_remove_neuron_gain(creature: &CreatureJson, neuron_uuid: &str) -> Option<f64> {
    // Output neurons are never remove-neuron candidates.
    if creature
        .neurons
        .iter()
        .any(|n| n.uuid == neuron_uuid && n.neuron_type == "output")
    {
        return None;
    }

    let impacts = compute_impacts_public(creature);
    let influence = impacts.get(neuron_uuid).copied()?;

    // Honest gain: removing a contributing neuron costs its downstream
    // influence, so the score is expected to fall by roughly that amount.
    // Guard against any negative/NaN influence leaking through.
    let magnitude = f64::from(influence).max(0.0);
    Some(-magnitude)
}
