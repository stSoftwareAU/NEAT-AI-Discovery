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

/// Squash-error magnitude above which a neuron is treated as "broken" for
/// hygiene purposes (Issue #1519).
///
/// A neuron whose baseline squash error exceeds this bound is producing
/// astronomically large activation errors that break WASM compilation of the
/// exported network downstream. It must be removed regardless of its estimated
/// gain — the NEAT-AI `#2483` hygiene guarantee. This mirrors the Deno-side
/// `MAX_REASONABLE_SQUASH_ERROR` so the removal-eligibility decision uses the
/// same threshold on both sides of the FFI boundary.
pub const MAX_REASONABLE_SQUASH_ERROR: f64 = 1e10;

/// Decoupled remove-neuron assessment (Issue #1519).
///
/// Separates hygiene **removal-eligibility** from the **ranking value** so an
/// over-threshold broken neuron is removed regardless of its gain, while its
/// reported gain stays honest and never crowds out realistic (~`1e-4`)
/// candidates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RemoveNeuronAssessment {
    /// `true` when the neuron must be removed on hygiene grounds — its squash
    /// error exceeds [`MAX_REASONABLE_SQUASH_ERROR`]. Derived purely from the
    /// hygiene threshold, never from [`gain`](Self::gain).
    pub removal_eligible: bool,
    /// The honest, propagation-aware gain used purely for ranking. Always the
    /// [`estimate_remove_neuron_gain`] value (≈0 or negative for a broken
    /// neuron), never the retired synthetic `[0.1, 0.5]` floor.
    pub gain: f64,
}

/// Assess a remove-neuron candidate, decoupling hygiene removal-eligibility
/// from the honest ranking gain (Issue #1519).
///
/// Removal-eligibility is driven solely by `squash_error` against
/// [`MAX_REASONABLE_SQUASH_ERROR`]; the gain is the honest, propagation-aware
/// [`estimate_remove_neuron_gain`] estimate. The two are independent: a broken
/// (over-threshold) neuron stays removal-eligible even when its honest gain is
/// ≈0 or negative, and that honest gain — not a fabricated floor — is what the
/// downstream ranking sorts on.
///
/// # Arguments
/// * `creature` - The creature's network topology (neurons and synapses).
/// * `neuron_uuid` - UUID of the neuron whose removal is being assessed.
/// * `squash_error` - The neuron's baseline squash error, used only for the
///   hygiene removal-eligibility decision.
///
/// # Returns
/// `Some(assessment)` for a hidden neuron present in the topology, or `None`
/// when the neuron is absent or is an output neuron (outputs are never removal
/// candidates).
#[must_use]
pub fn assess_remove_neuron(
    creature: &CreatureJson,
    neuron_uuid: &str,
    squash_error: f64,
) -> Option<RemoveNeuronAssessment> {
    // The honest gain also gates candidacy: `None` here means the neuron is an
    // output or not present, so it is never a removal candidate.
    let gain = estimate_remove_neuron_gain(creature, neuron_uuid)?;
    Some(RemoveNeuronAssessment {
        removal_eligible: squash_error > MAX_REASONABLE_SQUASH_ERROR,
        gain,
    })
}

/// Estimate the honest, propagation-aware **cost** of removing a neuron — the
/// influence term of the removal gain, not the gain itself.
///
/// The returned value is signed:
/// - Its **magnitude** is the neuron's propagation-aware influence on the
///   output(s) — the squash-bounded product of downstream edge weights
///   accumulated all the way to the output(s). Neurons many layers from the
///   output attenuate to a tiny value.
/// - Its **sign** is negative (or zero): removing a neuron that still carries
///   downstream influence is expected to *reduce* the trained network's score
///   by roughly its influence, so this term is non-positive.
///
/// # Sign and scale contract (Issue #1812)
///
/// The magnitude is a **unitless fraction of output sensitivity in `[0, 1]`**,
/// as returned by [`compute_impacts_public`]. It is *not* a creature-score
/// delta, and it must not be written into `expectedCreatureScoreGain` on its
/// own: that field is consumed as a creature-score **benefit** by the shared
/// gain-descending ranking sort and by the acceptance floor, so a bare
/// non-positive cost can never clear a positive floor (the Issue #1785 / #1810
/// zero yield).
///
/// Two conversions turn this into the emitted gain, both applied by
/// [`removal_net_gain`](super::remove_neuron_net_gain::removal_net_gain): the
/// magnitude is multiplied by `REMOVE_INFLUENCE_CALIBRATION` to reach the
/// creature-score scale, and subtracted from the exact complexity saving. The
/// sign and value returned here are unchanged by that — do **not** flip the sign
/// to make removals acceptable.
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
