//! Drought-driven deprioritisation of destructive remove-neuron candidates
//! (Issue #1448).
//!
//! On a mature, plateaued creature (the #1418 GRQ-3 case: 1673 neurons at score
//! ~0.4224), the remove-neuron path keeps proposing low-impact hidden neurons
//! whose predicted error reduction never survives evaluation. The failure cache
//! fills with remove-neuron entries (bucket `247b83ab`: 9 of 11 files), the
//! `#1444` failure-cache filter then suppresses re-tries, and on the plateau
//! remove-neuron is the only module producing *any* output — all harmful.
//!
//! The `#1425` failure-cache calibration shrinks remove-neuron predictions, but
//! it learns only *after* the failures have been cached, so it cannot stop the
//! first wave of over-confident proposals on a creature whose search is already
//! exhausted. This module adds the missing front-stop: when the creature is in a
//! **search-exhaustion** drought (the `#1424` / `#1421` classification), it
//! multiplies the expected gain of single-op `RemoveNeuron` coordinated
//! candidates by a deprioritisation factor `< 1.0`. The demoted gains sort below
//! add-synapse / squash / multi-op coordinated candidates and the most
//! over-confident ones fall through the downstream coordinated noise floor
//! (`apply_final_coordinated_gain_floor`) — so the destructive module yields the
//! budget to the constructive ones during a plateau.
//!
//! The deprioritisation is gated on the drought classification on purpose:
//! - an **environmental** drought (memory / GPU gated passes) is not the
//!   creature's fault, so remove-neuron is left untouched; and
//! - outside a drought the module behaves exactly as before — the factor
//!   resolves to the neutral `1.0`.

use crate::CoordinatedStructuralCandidateJson;

use super::creature_drought_alarm::{DroughtClassification, classify_drought};

/// Neutral (no-op) deprioritisation factor.
///
/// Returned whenever the deprioritisation must not engage (no active drought, an
/// environmental drought, or the operator has disabled the lever). Multiplying a
/// gain by `1.0` leaves it unchanged.
pub const NEUTRAL_DEPRIORITISATION_FACTOR: f32 = 1.0;

/// Default multiplier applied to a single-op remove-neuron candidate's expected
/// gain during a search-exhaustion drought (Issue #1448).
///
/// `0.1` demotes the prediction by an order of magnitude — enough to sort
/// remove-neuron proposals below the constructive change types and to push the
/// most over-confident ones (the ~800× over-predictors from bucket `247b83ab`)
/// below the per-op coordinated noise floor, while leaving a small non-zero gain
/// so a genuinely strong removal can still surface. Tune via
/// `NEAT_AI_DISCOVERY_REMOVE_NEURON_DROUGHT_FACTOR`.
pub const DEFAULT_REMOVE_NEURON_DROUGHT_FACTOR: f32 = 0.1;

/// Lower clamp for the configured deprioritisation factor.
///
/// Mirrors the failure-cache calibration floor
/// ([`super::scoring::calibration_correction::MIN_CALIBRATION_CORRECTION`]): a
/// remove-neuron candidate is demoted, never zeroed, so exploration is never
/// fully extinguished.
pub const MIN_REMOVE_NEURON_DROUGHT_FACTOR: f32 = 0.001;

/// Inputs the orchestrator gathers to decide whether remove-neuron candidates
/// should be deprioritised this pass (Issue #1448).
#[derive(Debug, Clone, Copy)]
pub struct DroughtDeprioritisationInputs {
    /// Consecutive trailing empty (search-exhausted) discovery passes — the same
    /// streak the per-pass drought diagnostic counts.
    pub consecutive_failures: u32,
    /// Passes within the drought that the host could not evaluate (memory / GPU
    /// gated). Used only to classify the drought, never to trigger it.
    pub environmentally_disabled_passes: u32,
    /// Consecutive-failure count at which a drought is considered active — the
    /// task-calibrated drought-diagnostic threshold. Must be `>= 1`; a value of
    /// `0` disables the deprioritisation entirely.
    pub drought_threshold: u32,
}

impl DroughtDeprioritisationInputs {
    /// Whether the creature is in an active **search-exhaustion** drought.
    ///
    /// Both conditions must hold: the trailing-failure streak has reached the
    /// drought threshold, and the drought is classified
    /// [`DroughtClassification::SearchExhaustion`] (an environmental drought is
    /// host-driven, so the destructive module is left alone).
    #[must_use]
    pub fn is_search_exhaustion_drought(&self) -> bool {
        if self.drought_threshold == 0 || self.consecutive_failures < self.drought_threshold {
            return false;
        }
        classify_drought(
            self.consecutive_failures,
            self.environmentally_disabled_passes,
        ) == DroughtClassification::SearchExhaustion
    }
}

/// Resolve the effective remove-neuron deprioritisation factor for this pass
/// (Issue #1448).
///
/// Returns [`NEUTRAL_DEPRIORITISATION_FACTOR`] (`1.0`, a no-op) unless the
/// creature is in a search-exhaustion drought, in which case it returns
/// `configured_factor` clamped to
/// `[MIN_REMOVE_NEURON_DROUGHT_FACTOR, NEUTRAL_DEPRIORITISATION_FACTOR]`. A
/// configured factor of `1.0` (or any non-finite value) therefore disables the
/// deprioritisation even during a drought.
#[must_use]
pub fn remove_neuron_deprioritisation_factor(
    inputs: &DroughtDeprioritisationInputs,
    configured_factor: f32,
) -> f32 {
    if !inputs.is_search_exhaustion_drought() {
        return NEUTRAL_DEPRIORITISATION_FACTOR;
    }
    if !configured_factor.is_finite() {
        return NEUTRAL_DEPRIORITISATION_FACTOR;
    }
    configured_factor.clamp(
        MIN_REMOVE_NEURON_DROUGHT_FACTOR,
        NEUTRAL_DEPRIORITISATION_FACTOR,
    )
}

/// Whether a coordinated candidate is a single-op `RemoveNeuron` — the
/// functional `remove-neuron` change keyed under
/// [`super::scoring::calibration_correction::CHANGE_TYPE_REMOVE_NEURON`]
/// (Issue #1425, #1448).
#[must_use]
pub fn is_single_op_remove_neuron(candidate: &CoordinatedStructuralCandidateJson) -> bool {
    candidate.operations.len() == 1
        && matches!(
            candidate.operations.first(),
            Some(crate::CoordinatedStructuralOpJson::RemoveNeuron { .. })
        )
}

/// Multiply the expected gain of every single-op remove-neuron candidate by
/// `factor`, deprioritising them relative to the constructive change types
/// (Issue #1448).
///
/// A `factor >= 1.0` (the neutral case) is a no-op and returns `0`. Only
/// candidates with a finite, strictly positive gain are demoted — non-finite or
/// non-positive gains are left for the dedicated `reject_non_finite_gains` /
/// non-positive filters downstream. Returns the number of candidates whose gain
/// was reduced, so the caller can log how much budget the destructive module
/// surrendered.
pub fn deprioritise_remove_neuron_candidates(
    candidates: &mut [CoordinatedStructuralCandidateJson],
    factor: f32,
) -> u32 {
    // Only a finite factor strictly below the neutral 1.0 deprioritises; a
    // neutral / inflating / non-finite factor is a no-op.
    if !factor.is_finite() || factor >= NEUTRAL_DEPRIORITISATION_FACTOR {
        return 0;
    }
    let mut demoted = 0_u32;
    for candidate in candidates.iter_mut() {
        if is_single_op_remove_neuron(candidate)
            && candidate.expected_creature_score_gain.is_finite()
            && candidate.expected_creature_score_gain > 0.0
        {
            candidate.expected_creature_score_gain *= factor;
            demoted = demoted.saturating_add(1);
        }
    }
    demoted
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs(failures: u32, disabled: u32, threshold: u32) -> DroughtDeprioritisationInputs {
        DroughtDeprioritisationInputs {
            consecutive_failures: failures,
            environmentally_disabled_passes: disabled,
            drought_threshold: threshold,
        }
    }

    #[test]
    fn no_deprioritisation_below_threshold() {
        // Four failures, threshold five — drought is not yet active.
        let factor = remove_neuron_deprioritisation_factor(&inputs(4, 0, 5), 0.1);
        assert_eq!(factor, NEUTRAL_DEPRIORITISATION_FACTOR);
    }

    #[test]
    fn deprioritises_during_search_exhaustion() {
        let factor = remove_neuron_deprioritisation_factor(&inputs(10, 0, 5), 0.1);
        assert!((factor - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn environmental_drought_is_not_deprioritised() {
        // Disabled passes outnumber genuinely-empty passes -> environmental.
        let factor = remove_neuron_deprioritisation_factor(&inputs(6, 20, 5), 0.1);
        assert_eq!(factor, NEUTRAL_DEPRIORITISATION_FACTOR);
    }

    #[test]
    fn threshold_zero_disables() {
        let factor = remove_neuron_deprioritisation_factor(&inputs(50, 0, 0), 0.1);
        assert_eq!(factor, NEUTRAL_DEPRIORITISATION_FACTOR);
    }

    #[test]
    fn configured_factor_one_disables_even_in_drought() {
        let factor = remove_neuron_deprioritisation_factor(&inputs(10, 0, 5), 1.0);
        assert_eq!(factor, NEUTRAL_DEPRIORITISATION_FACTOR);
    }

    #[test]
    fn configured_factor_is_clamped_to_floor() {
        // A below-floor configured factor is clamped up to the floor, never zeroed.
        let factor = remove_neuron_deprioritisation_factor(&inputs(10, 0, 5), 0.0);
        assert!((factor - MIN_REMOVE_NEURON_DROUGHT_FACTOR).abs() < f32::EPSILON);
    }
}
