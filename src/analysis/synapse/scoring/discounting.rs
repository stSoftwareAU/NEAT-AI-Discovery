//! Pessimism discounting and prediction calibration for candidate scoring
//!
//! This module applies pessimism discounts based on sample improvement ratios
//! and per-candidate-type prediction calibration factors (Issues #506, #789, #791, #891).

#![allow(clippy::cast_precision_loss)] // Intentional u32→f32 casts for ratio computation (Issue #873)
use crate::analysis::constants::{
    LOGISTIC_CALIBRATION_FLOOR, LOGISTIC_CALIBRATION_MIDPOINT, LOGISTIC_CALIBRATION_STEEPNESS,
    NEURON_PESSIMISM_CURVE_EXPONENT, NEURON_PESSIMISM_DISCOUNT_FLOOR, PESSIMISM_CURVE_EXPONENT,
    PESSIMISM_DISCOUNT_FLOOR, SATURATION_DISCOUNT_AGGRESSIVE, SYNAPSE_PESSIMISM_CURVE_EXPONENT,
    SYNAPSE_PESSIMISM_DISCOUNT_FLOOR,
};

// =============================================================================
// Pessimism Discount (Issue #506)
// =============================================================================

/// Combine the binary `improved_ratio` with the magnitude-weighted ratio
/// (Issue #1161) so that 76% of samples improving by noise-level amounts no
/// longer scores like 76% of samples improving meaningfully.
///
/// The combined ratio is the geometric mean
/// `sqrt(improved_ratio × magnitude_ratio)` when `magnitude_ratio` is provided.
/// When the magnitude ratio is `None` (legacy callers, or paths that have not
/// computed it), the function falls back to the binary `improved_ratio` to
/// preserve the previous behaviour.
///
/// The geometric mean is chosen because it strongly penalises imbalanced
/// signals: a candidate where 76% of samples improve but only by ~0% of the
/// baseline magnitude collapses to ≈ 0.0, while a candidate where both signals
/// agree at 0.7 stays at 0.7.
///
/// `magnitude_ratio` values that are negative or NaN are clamped to `0.0` so
/// callers cannot accidentally produce non-finite discounts.
#[inline]
fn combined_ratio(improved_ratio: f32, magnitude_ratio: Option<f32>) -> f32 {
    match magnitude_ratio {
        Some(m) => {
            let safe_m = if m.is_finite() { m.max(0.0) } else { 0.0 };
            (improved_ratio.max(0.0) * safe_m).sqrt()
        }
        None => improved_ratio,
    }
}

/// Scale the pessimism floor by the magnitude ratio when one is supplied
/// (Issue #1161).
///
/// The legacy floor (`base_floor`) is the minimum discount that would otherwise
/// guarantee gain × `base_floor` for any non-empty sample set. For noise-level
/// candidates whose magnitude ratio is ≈ 0, this floor masks the underlying
/// signal that the candidate's improvements are not meaningful — discounted
/// gains stay at ≥ `base_floor` × gain even when the actual error reduction is
/// negligible. Multiplying the floor by the magnitude ratio collapses it to
/// near-zero exactly when the candidate's improvements are noise-level, so the
/// final discount can fall below the legacy floor and reach < 1% of the input
/// gain (issue #1160 acceptance criterion).
///
/// `None` magnitude ratio preserves the legacy floor unchanged.
#[inline]
fn effective_floor(base_floor: f32, magnitude_ratio: Option<f32>) -> f32 {
    match magnitude_ratio {
        Some(m) => {
            let safe_m = if m.is_finite() {
                m.clamp(0.0, 1.0)
            } else {
                0.0
            };
            base_floor * safe_m
        }
        None => base_floor,
    }
}

/// Apply a pessimism discount to an expected score gain based on sample improvement ratio.
///
/// Production data (creature b2ff6e45) showed that raw improvement percentages
/// over-estimate creature-level score gains by orders of magnitude: the sole
/// successful candidate predicted +0.0205 but achieved only +0.0000011
/// (18,500× over-estimation). The improvement is computed from a single target
/// neuron's sampled error, but this does not generalise directly to creature-level
/// score gain across the full training set.
///
/// The discount uses the `improved_count / total_count` ratio as a quality signal:
/// candidates that improve more samples are more likely to generalise.
///
/// ## Formula
///
/// Issue #733: Changed from linear to concave (power) curve. The linear formula
/// was too aggressive for add-neurons candidates, discounting moderate-quality
/// candidates excessively.
///
/// ```text
/// improved_ratio = improved_count / total_count
/// adjusted_ratio = improved_ratio ^ PESSIMISM_CURVE_EXPONENT
/// discount = PESSIMISM_DISCOUNT_FLOOR + (1 - PESSIMISM_DISCOUNT_FLOOR) × adjusted_ratio
/// result = gain × discount
/// ```
///
/// ## Returns
///
/// The discounted gain, always preserving the sign of the original gain.
pub fn apply_pessimism_discount(gain: f32, improved_count: u32, total_count: u32) -> f32 {
    if total_count == 0 {
        return gain * PESSIMISM_DISCOUNT_FLOOR;
    }
    let improved_ratio = improved_count as f32 / total_count as f32;
    let adjusted_ratio = improved_ratio.powf(PESSIMISM_CURVE_EXPONENT);
    let discount = PESSIMISM_DISCOUNT_FLOOR + (1.0 - PESSIMISM_DISCOUNT_FLOOR) * adjusted_ratio;
    gain * discount
}

/// Apply a neuron-specific pessimism discount to an expected score gain (Issue #791).
///
/// GRQ-sampler analysis shows add-neurons has a 15% success rate (3,812 / 25,812),
/// indicating the generic pessimism parameters are too generous for neuron candidates.
/// This function uses neuron-calibrated constants that apply more aggressive
/// discounting:
///
/// - Lower floor (0.10 vs 0.15): stronger base discount at low ratios
/// - Higher exponent (0.75 vs 0.6): less forgiving at moderate ratios
///
/// ## Formula
///
/// Issue #1161: The binary `improved_ratio` is combined with a
/// magnitude-weighted ratio via geometric mean before raising to the pessimism
/// exponent. This collapses the score for candidates where many samples
/// improve by noise-level amounts (high binary ratio, low magnitude ratio)
/// while preserving credit for candidates whose improvements are both broad
/// and substantial.
///
/// ```text
/// improved_ratio   = improved_count / total_count
/// combined_ratio   = sqrt(improved_ratio × magnitude_ratio)        # if magnitude is supplied
/// adjusted_ratio   = combined_ratio ^ NEURON_PESSIMISM_CURVE_EXPONENT
/// discount         = NEURON_PESSIMISM_DISCOUNT_FLOOR
///                  + (1 - NEURON_PESSIMISM_DISCOUNT_FLOOR) × adjusted_ratio
/// result           = gain × discount
/// ```
///
/// `improvement_magnitude_ratio` of `None` falls back to the legacy behaviour
/// (binary ratio only) so callers that have not yet been updated keep their
/// previous discount.
pub fn apply_neuron_pessimism_discount(
    gain: f32,
    improved_count: u32,
    total_count: u32,
    improvement_magnitude_ratio: Option<f32>,
) -> f32 {
    if total_count == 0 {
        return gain * NEURON_PESSIMISM_DISCOUNT_FLOOR;
    }
    let improved_ratio = improved_count as f32 / total_count as f32;
    let combined = combined_ratio(improved_ratio, improvement_magnitude_ratio);
    let adjusted_ratio = combined.powf(NEURON_PESSIMISM_CURVE_EXPONENT);
    // Issue #1161: scale the floor by the magnitude ratio so noise-level
    // candidates (low magnitude) can be discounted below the legacy floor.
    let floor = effective_floor(NEURON_PESSIMISM_DISCOUNT_FLOOR, improvement_magnitude_ratio);
    let discount = floor + (1.0 - floor) * adjusted_ratio;
    gain * discount
}

/// Apply a synapse-specific pessimism discount to an expected score gain (Issue #789).
///
/// GRQ-sampler analysis shows add-synapses has a 0% success rate (0 / 31),
/// indicating both the generic and neuron-specific pessimism parameters are too
/// generous for synapse candidates. This function uses synapse-calibrated constants
/// that apply the most aggressive discounting of all candidate types:
///
/// - Lowest floor (0.05 vs 0.10 neuron vs 0.15 generic): strongest base discount
/// - Highest exponent (0.85 vs 0.75 neuron vs 0.6 generic): least forgiving curve
///
/// The aggressive discounting accounts for the multi-weight search (9 weight
/// variants) creating selection bias that overfits to sample data.
///
/// ## Formula
///
/// Issue #1161: The binary `improved_ratio` is combined with the
/// magnitude-weighted ratio via geometric mean before raising to the pessimism
/// exponent (see [`apply_neuron_pessimism_discount`] for the detailed
/// description and rationale).
///
/// ```text
/// improved_ratio   = improved_count / total_count
/// combined_ratio   = sqrt(improved_ratio × magnitude_ratio)        # if magnitude is supplied
/// adjusted_ratio   = combined_ratio ^ SYNAPSE_PESSIMISM_CURVE_EXPONENT
/// discount         = SYNAPSE_PESSIMISM_DISCOUNT_FLOOR
///                  + (1 - SYNAPSE_PESSIMISM_DISCOUNT_FLOOR) × adjusted_ratio
/// result           = gain × discount
/// ```
pub fn apply_synapse_pessimism_discount(
    gain: f32,
    improved_count: u32,
    total_count: u32,
    improvement_magnitude_ratio: Option<f32>,
) -> f32 {
    if total_count == 0 {
        return gain * SYNAPSE_PESSIMISM_DISCOUNT_FLOOR;
    }
    let improved_ratio = improved_count as f32 / total_count as f32;
    let combined = combined_ratio(improved_ratio, improvement_magnitude_ratio);
    let adjusted_ratio = combined.powf(SYNAPSE_PESSIMISM_CURVE_EXPONENT);
    // Issue #1161: scale the floor by the magnitude ratio so noise-level
    // candidates (low magnitude) can be discounted below the legacy floor.
    let floor = effective_floor(
        SYNAPSE_PESSIMISM_DISCOUNT_FLOOR,
        improvement_magnitude_ratio,
    );
    let discount = floor + (1.0 - floor) * adjusted_ratio;
    gain * discount
}

/// Apply a per-candidate-type prediction calibration factor (Issue #891).
///
/// GRQ-sampler discovery cache reveals that `expected_creature_score_gain` overestimates
/// actual outcomes by 100–10,000×, with the magnitude varying by candidate type. This
/// systematic overestimation means cross-type comparisons are unreliable — a synapse
/// prediction of 0.01 is not comparable to a neuron prediction of 0.003.
///
/// This function applies a multiplicative calibration factor to scale predictions
/// closer to observed actual gains. It is applied after pessimism discounting and
/// type-specific boosts.
///
/// ## Arguments
///
/// * `gain` — The expected creature score gain after pessimism discounting
/// * `calibration_factor` — Per-type calibration constant (e.g., `SYNAPSE_PREDICTION_CALIBRATION`)
///
/// ## Returns
///
/// The calibrated gain, preserving the sign of the original.
#[inline]
pub fn apply_prediction_calibration(gain: f32, calibration_factor: f32) -> f32 {
    gain * calibration_factor
}

// =============================================================================
// Logistic Prediction Calibration (Issue #1056)
// =============================================================================

/// Apply non-linear (logistic) prediction calibration based on improved ratio (Issue #1056).
///
/// The linear `gain × calibration_factor` approach was insufficient to bridge the
/// neuron-level → creature-level prediction gap. GRQ-sampler data (30+ creatures)
/// shows the relationship between `improvedCount/totalCount` and actual success
/// probability is non-linear — moderate improved ratios (0.3–0.6) are far more
/// overestimated than high ratios (>0.8).
///
/// This function modulates the base calibration factor using a logistic (sigmoid)
/// curve of the improved ratio:
///
/// ```text
/// sigmoid = 1 / (1 + exp(-steepness × (ratio - midpoint)))
/// modulator = floor + (1 - floor) × sigmoid
/// result = gain × base_calibration × modulator
/// ```
///
/// The effect:
/// - Low ratios (< 0.3): modulator ≈ floor → heavy additional reduction
/// - Moderate ratios (~0.5): modulator ≈ 0.37 → substantial reduction
/// - High ratios (> 0.8): modulator ≈ 0.9+ → near-full base calibration
///
/// ## Arguments
///
/// * `gain` — The expected creature score gain after pessimism discounting
/// * `improved_count` — Number of samples showing improvement
/// * `total_count` — Total number of samples evaluated
/// * `base_calibration` — Per-type calibration constant (e.g., `NEURON_PREDICTION_CALIBRATION`)
///
/// ## Returns
///
/// The calibrated gain, preserving the sign of the original.
pub fn apply_logistic_prediction_calibration(
    gain: f32,
    improved_count: u32,
    total_count: u32,
    base_calibration: f32,
) -> f32 {
    if total_count == 0 {
        return gain * base_calibration * LOGISTIC_CALIBRATION_FLOOR;
    }
    let ratio = improved_count as f32 / total_count as f32;
    let sigmoid = 1.0
        / (1.0 + (-LOGISTIC_CALIBRATION_STEEPNESS * (ratio - LOGISTIC_CALIBRATION_MIDPOINT)).exp());
    let modulator = LOGISTIC_CALIBRATION_FLOOR + (1.0 - LOGISTIC_CALIBRATION_FLOOR) * sigmoid;
    gain * base_calibration * modulator
}

// =============================================================================
// Saturation-Aware Prediction Discount (Issue #1112)
// =============================================================================

/// Compute the saturation discount multiplier for a given saturation factor.
///
/// Linearly interpolates from 1.0 (no discount) at the saturation threshold
/// (0.9) down to `SATURATION_DISCOUNT_AGGRESSIVE` at full saturation (1.0).
///
/// Returns 1.0 for non-saturated targets (factor ≤ threshold or `None`).
#[inline]
fn saturation_discount(factor: f32) -> f32 {
    // Threshold matches TARGET_SATURATION_RANGE_THRESHOLD in preparation.rs
    const THRESHOLD: f32 = 0.90;

    if factor <= THRESHOLD {
        return 1.0;
    }

    // Normalise to [0, 1] within the saturated range
    let normalised = ((factor - THRESHOLD) / (1.0 - THRESHOLD)).clamp(0.0, 1.0);

    // Interpolate from 1.0 down to the aggressive floor
    1.0 - normalised * (1.0 - SATURATION_DISCOUNT_AGGRESSIVE)
}

/// Apply saturation-aware prediction discount for neuron candidates (Issue #1112).
///
/// When the target neuron is operating near its activation saturation bounds,
/// predictions are heavily over-estimated because the target physically cannot
/// move much in response to small perturbations. This function applies an
/// additional multiplicative discount proportional to the saturation level.
///
/// ## Arguments
///
/// * `gain` — The expected creature score gain after pessimism discounting
/// * `target_saturation_factor` — The saturation factor from candidate
///   preparation (`None` for non-saturated targets)
///
/// ## Returns
///
/// The discounted gain. Non-saturated targets (factor = `None`) are returned
/// unchanged.
pub fn apply_saturation_prediction_discount(
    gain: f32,
    target_saturation_factor: Option<f32>,
) -> f32 {
    match target_saturation_factor {
        Some(factor) => gain * saturation_discount(factor),
        None => gain,
    }
}
