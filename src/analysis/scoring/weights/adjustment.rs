//! Dynamic weight adjustment functions.
//!
//! This module contains functions for adjusting existing synapse weights,
//! including delta clamping and coordinated structural candidate computation.

use crate::analysis::samples::EPSILON;

use super::MAX_OUTGOING_WEIGHT;

/// Clamp a proposed synapse weight delta against `MAX_OUTGOING_WEIGHT`.
///
/// Weight update candidates are represented as a *delta* applied to an existing synapse. If the
/// resulting `new_weight` is clamped, the *effective* delta differs from the proposed delta.
///
/// # Arguments
/// * `old_weight` - Current weight of the synapse
/// * `proposed_delta_weight` - Proposed weight change
///
/// # Returns
/// * `Some((new_weight, delta_weight))` - When the effective delta is meaningful
/// * `None` - When the effective delta is too small (below EPSILON)
pub fn clamp_weight_update_delta(
    old_weight: f32,
    proposed_delta_weight: f32,
) -> Option<(f32, f32)> {
    let new_weight =
        (old_weight + proposed_delta_weight).clamp(-MAX_OUTGOING_WEIGHT, MAX_OUTGOING_WEIGHT);
    let delta_weight = new_weight - old_weight;
    if delta_weight.abs() <= EPSILON {
        None
    } else {
        Some((new_weight, delta_weight))
    }
}

/// Compute activation delta for coordinated structural candidates.
///
/// This function calculates the activation delta needed when replacing a noisy
/// synapse with a trusted one in a coordinated structural operation.
///
/// # Arguments
/// * `trusted_activation` - Activation from the trusted source neuron
/// * `noisy_activation` - Activation from the noisy source neuron (being removed)
/// * `noisy_weight` - Weight of the noisy synapse (being removed)
/// * `trusted_weight` - Current weight of the trusted synapse
///
/// # Returns
/// * `Some(delta)` - The activation delta needed for the coordinated candidate
/// * `None` - If noisy_weight is effectively zero (cannot compute scale)
///
/// # Notes (7-Jan-2026)
/// We intentionally do **not** clamp the trusted weight here. The coordinated candidate is
/// derived from existing synapse weights, and NEAT-AI will validate the full ablation on the
/// complete training set. Clamping here changes the candidate semantics and can suppress
/// valid coordinated candidates.
pub fn coordinated_structural_activation_delta(
    trusted_activation: f32,
    noisy_activation: f32,
    noisy_weight: f32,
    trusted_weight: f32,
) -> Option<f32> {
    if noisy_weight.abs() <= EPSILON {
        return None;
    }

    let new_trusted_weight = trusted_weight + noisy_weight;
    let delta_trusted_weight = new_trusted_weight - trusted_weight;

    let scale = delta_trusted_weight / noisy_weight;
    Some(scale * trusted_activation - noisy_activation)
}
