//! Synapse scoring and improvement calculation
//!
//! This module contains functions for computing improvement scores, saturation-aware
//! simulation, and source/target type boosting (Issues #413, #467, #468).

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use crate::CandidateNeuronJson;
#[cfg(test)]
use crate::analysis::activation::get_target_simulation_fn;
use crate::analysis::activation::{TargetSimulationMode, get_target_simulation_mode};
use crate::analysis::samples::{EPSILON, HelpfulSample};
use crate::analysis::utils::parse_input_index;
use std::collections::HashMap;

// Boosting and discount constants
use crate::analysis::constants::{
    EXISTING_HIDDEN_TARGET_BOOST, INPUT_SOURCE_BOOST, NEURON_PESSIMISM_CURVE_EXPONENT,
    NEURON_PESSIMISM_DISCOUNT_FLOOR, PESSIMISM_CURVE_EXPONENT, PESSIMISM_DISCOUNT_FLOOR,
    SYNAPSE_PESSIMISM_CURVE_EXPONENT, SYNAPSE_PESSIMISM_DISCOUNT_FLOOR,
};

// =============================================================================
// Source-Type Prioritisation (Issue #467)
// =============================================================================

/// Applies source-type boost to a candidate's expected score gain.
///
/// Input neurons as synapse sources have a 36.2% success rate compared to
/// 2.8–3.3% for hidden neurons (GRQ-sampler data). This function applies
/// [`INPUT_SOURCE_BOOST`] as a multiplier when the source neuron is an input
/// neuron (UUID matches `input-N` pattern).
///
/// Hidden and output neurons receive no boost (multiplier = 1.0).
pub fn apply_source_type_boost(gain: f32, source_uuid: &str) -> f32 {
    if parse_input_index(source_uuid).is_some() {
        gain * INPUT_SOURCE_BOOST as f32
    } else {
        gain
    }
}

// =============================================================================
// Target-Type Prioritisation (Issue #468)
// =============================================================================

/// Applies target-type boost to a candidate's expected score gain.
///
/// Existing hidden neurons as targets have a 31.4% success rate compared to
/// 5.3–5.4% for output or discovery-hidden neurons (GRQ-sampler data). This
/// function applies [`EXISTING_HIDDEN_TARGET_BOOST`] as a multiplier when the
/// target neuron is an existing hidden neuron.
///
/// Output, input, constant, and unknown neurons receive no boost (multiplier = 1.0).
pub fn apply_target_type_boost(
    gain: f32,
    target_uuid: &str,
    neuron_type_map: &HashMap<String, String>,
) -> f32 {
    if neuron_type_map
        .get(target_uuid)
        .is_some_and(|t| t == "hidden")
    {
        gain * EXISTING_HIDDEN_TARGET_BOOST as f32
    } else {
        gain
    }
}

// =============================================================================
// Activation-Function-Aware Neuron Scoring (Issue #887)
// =============================================================================

/// Applies activation-function-aware boost/penalty to a neuron candidate's
/// expected score gain (Issue #887).
///
/// GRQ-sampler cache analysis shows dramatic differences in success rates by
/// activation function (e.g., GELU at 60% vs `HARD_TANH` at 7.2%). This function
/// applies Bayesian-smoothed boost multipliers to prioritise candidates using
/// historically more successful activation functions.
///
/// The boost is applied as a direct multiplier on `expected_creature_score_gain`,
/// similar to [`apply_source_type_boost`] and [`apply_target_type_boost`].
pub fn apply_activation_neuron_boost(gain: f32, squash_name: &str) -> f32 {
    let boost = crate::analysis::constants::activation_neuron_boost(squash_name);
    gain * boost as f32
}

// =============================================================================
// Synapse Improvement Calculation
// =============================================================================

// Note: Target simulation functions (TargetSimulationMode, get_target_activation_fn,
// get_target_simulation_fn, get_target_simulation_mode, can_use_hard_tanh) have been
// moved to the activation module as part of Issue #238.

/// Computes the sign of a weight as an i8 for use in the candidate key.
/// Returns 1 for positive weights, -1 for negative, and 0 for zero (though this
/// shouldn't happen in practice).
pub(crate) fn weight_sign(weight: f32) -> i8 {
    if weight > 0.0 {
        1
    } else if weight < 0.0 {
        -1
    } else {
        0
    }
}

/// Compute a deduplication key for neuron candidates using FNV-1a hashing.
///
/// Issue #526: Replaces the previous `(String, String, String, i8, i8)` tuple key
/// which required 3 String clones per candidate. The hash key is computed from the
/// same components (source UUID, target UUID, squash, weight signs) but avoids
/// heap allocation entirely.
///
/// The key includes signs of BOTH `incoming_weight` AND `outgoing_weight` so that:
/// 1. Different `ReLU` orientations (`incoming_weight` ±1) are kept separately
/// 2. Split-error complementary pairs (same `incoming_weight`, opposite `outgoing_weight`)
///    are also kept separately — one pushes output UP, one pushes DOWN
pub(crate) fn compute_candidate_dedup_key(candidate: &CandidateNeuronJson) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325; // FNV-1a offset basis
    for b in candidate.source_neuron_uuid.as_bytes() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    // Separator byte (0xFF) prevents collisions between e.g. "input-1"+"hidden-2"
    // and "input-12"+"hidden-" which would otherwise produce the same byte sequence
    hash ^= 0xFF;
    hash = hash.wrapping_mul(0x100000001b3);
    for b in candidate.target_neuron_uuid.as_bytes() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash ^= 0xFF;
    hash = hash.wrapping_mul(0x100000001b3);
    for b in candidate.squash.as_bytes() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash ^= 0xFF;
    hash = hash.wrapping_mul(0x100000001b3);
    hash ^= weight_sign(candidate.incoming_weight) as u8 as u64;
    hash = hash.wrapping_mul(0x100000001b3);
    hash ^= weight_sign(candidate.outgoing_weight) as u8 as u64;
    hash = hash.wrapping_mul(0x100000001b3);
    hash
}

/// Insert or update a neuron candidate in the deduplication map.
///
/// Uses a pre-computed FNV-1a hash key (Issue #526) to avoid 3 String clones per call.
/// The key is derived from (`source_uuid`, `target_uuid`, squash, `incoming_weight_sign`,
/// `outgoing_weight_sign`). When a collision occurs, the candidate with the higher
/// `expected_creature_score_gain` wins.
pub(crate) fn upsert_candidate(
    map: &mut HashMap<u64, CandidateNeuronJson>,
    candidate: CandidateNeuronJson,
) {
    use std::collections::hash_map::Entry;

    let key = compute_candidate_dedup_key(&candidate);

    match map.entry(key) {
        Entry::Occupied(mut entry) => {
            // Issue #128: Compare by expected creature score gain
            if candidate.expected_creature_score_gain > entry.get().expected_creature_score_gain {
                entry.insert(candidate);
            }
        }
        Entry::Vacant(entry) => {
            entry.insert(candidate);
        }
    }
}

/// Combined computation of improvement and count for `ReLU` candidates.
/// Single pass over samples for better cache efficiency.
///
/// When `target_activation_fn` is Some, simulates the target neuron's actual activation
/// function for more accurate improvement estimates. Otherwise falls back to linear approximation.
///
/// IMPORTANT: The `bias` parameter is critical for accurate predictions. It shifts the `ReLU`
/// activation threshold, affecting which samples produce non-zero output.
///
/// CRITICAL DOMAIN FIX (v0.1.120): When using `target_activation_fn` simulation,
/// both baseline and new error must be computed in ACTIVATION domain.
///
/// Returns (`improvement_percentage`, `improved_count`, `total_count`)
pub(crate) fn compute_relu_improvement_and_count(
    samples: &[HelpfulSample],
    incoming_weight: f32,
    outgoing_weight: f32,
    bias: f32,
    total_baseline_error_sq: f32,
    target_activation_fn: Option<fn(f32) -> f32>,
) -> (f32, u32, u32) {
    if total_baseline_error_sq <= EPSILON || samples.is_empty() {
        return (0.0, 0, samples.len() as u32);
    }

    let mut baseline_error_sq_sum = 0.0f32; // ACTIVATION domain when simulating
    let mut new_error_sq_sum = 0.0f32;
    let mut improved_count = 0u32;
    let mut worsened_count = 0u32;

    for sample in samples {
        let pre_activation = incoming_weight * sample.activation + bias;
        let relu_output = pre_activation.max(0.0);
        let contribution = outgoing_weight * relu_output;

        let (baseline_error, new_error) = if let Some(target_fn) = target_activation_fn {
            // CRITICAL: Use ACTIVATION domain for BOTH baseline and new error.
            // SAFETY INVARIANT: get_target_simulation_fn() only returns Some when
            // all samples have target_value and target_activation set.
            debug_assert!(
                sample.target_value.is_some(),
                "target_value must be set when target_activation_fn is Some"
            );
            debug_assert!(
                sample.target_activation.is_some(),
                "target_activation must be set when target_activation_fn is Some"
            );
            let target_value = sample.target_value.unwrap();
            let target_activation = sample.target_activation.unwrap();
            let desired_value = target_value + sample.avg_error;
            let expected = target_fn(desired_value);

            // Baseline error in ACTIVATION domain
            let baseline_err = expected - target_activation;

            // New error in ACTIVATION domain
            let new_input = target_value + contribution;
            let new_err = expected - target_fn(new_input);

            (baseline_err, new_err)
        } else {
            // Linear approximation: both errors in VALUE domain
            let baseline_err = sample.avg_error;
            let new_err = sample.avg_error - contribution;

            (baseline_err, new_err)
        };

        if baseline_error.is_finite() {
            baseline_error_sq_sum += baseline_error * baseline_error;
        }
        if new_error.is_finite() {
            new_error_sq_sum += new_error * new_error;
        }

        // Sample is improved if |new_error| < |baseline_error| (consistent domain)
        if new_error.abs() + EPSILON < baseline_error.abs() {
            improved_count += 1;
        } else if new_error.abs() > baseline_error.abs() + EPSILON {
            worsened_count += 1;
        }
    }

    // Use computed ACTIVATION domain baseline when simulating, else use passed VALUE domain
    let effective_baseline = if target_activation_fn.is_some() {
        baseline_error_sq_sum
    } else {
        total_baseline_error_sq
    };

    let improvement = if effective_baseline > EPSILON {
        (effective_baseline - new_error_sq_sum) / effective_baseline
    } else {
        0.0
    };
    let improvement = if improvement.is_finite() {
        improvement
    } else {
        0.0
    };

    let total_count = samples.len() as u32;
    let _ = worsened_count; // Kept for potential future use
    (improvement, improved_count, total_count)
}

/// Combined computation of improvement and count for activation candidates.
/// Single pass over samples for better cache efficiency.
///
/// When `target_activation_fn` is Some, simulates the target neuron's actual activation
/// function for more accurate improvement estimates. Otherwise falls back to linear approximation.
///
/// CRITICAL DOMAIN FIX (v0.1.120): When using `target_activation_fn` simulation,
/// both baseline and new error must be computed in ACTIVATION domain.
///
/// Returns (`improvement_percentage`, `improved_count`, `total_count`)
pub(crate) fn compute_activation_improvement_and_count(
    samples: &[HelpfulSample],
    incoming_weight: f32,
    outgoing_weight: f32,
    bias: f32,
    activation_fn: fn(f32) -> f32,
    total_baseline_error_sq: f32,
    target_activation_fn: Option<fn(f32) -> f32>,
) -> (f32, u32, u32) {
    if total_baseline_error_sq <= EPSILON || samples.is_empty() {
        return (0.0, 0, samples.len() as u32);
    }

    let mut baseline_error_sq_sum = 0.0f32; // ACTIVATION domain when simulating
    let mut new_error_sq_sum = 0.0f32;
    let mut improved_count = 0u32;
    let total_count = samples.len() as u32;

    for sample in samples {
        let pre_activation = incoming_weight * sample.activation + bias;
        let neuron_output = activation_fn(pre_activation);
        let contribution = outgoing_weight * neuron_output;

        let (baseline_error, new_error) = if let Some(target_fn) = target_activation_fn {
            // CRITICAL: Use ACTIVATION domain for BOTH baseline and new error.
            // SAFETY INVARIANT: get_target_simulation_fn() only returns Some when
            // all samples have target_value and target_activation set.
            debug_assert!(
                sample.target_value.is_some(),
                "target_value must be set when target_activation_fn is Some"
            );
            debug_assert!(
                sample.target_activation.is_some(),
                "target_activation must be set when target_activation_fn is Some"
            );
            let target_value = sample.target_value.unwrap();
            let target_activation = sample.target_activation.unwrap();
            let desired_value = target_value + sample.avg_error;
            let expected = target_fn(desired_value);

            // Baseline error in ACTIVATION domain
            let baseline_err = expected - target_activation;

            // New error in ACTIVATION domain
            let new_input = target_value + contribution;
            let new_err = expected - target_fn(new_input);

            (baseline_err, new_err)
        } else {
            // Linear approximation: both errors in VALUE domain
            (sample.avg_error, sample.avg_error - contribution)
        };

        if baseline_error.is_finite() {
            baseline_error_sq_sum += baseline_error * baseline_error;
        }
        if new_error.is_finite() {
            new_error_sq_sum += new_error * new_error;
        }

        // Sample is improved if |new_error| < |baseline_error| (consistent domain)
        if new_error.abs() + EPSILON < baseline_error.abs() {
            improved_count += 1;
        }
    }

    // Use computed ACTIVATION domain baseline when simulating, else use passed VALUE domain
    let effective_baseline = if target_activation_fn.is_some() {
        baseline_error_sq_sum
    } else {
        total_baseline_error_sq
    };

    let improvement = if effective_baseline > EPSILON {
        (effective_baseline - new_error_sq_sum) / effective_baseline
    } else {
        0.0
    };
    let improvement = if improvement.is_finite() {
        improvement
    } else {
        0.0
    };

    (improvement, improved_count, total_count)
}

/// Wrapper for tests - computes improvement only.
/// NOTE: For `ReLU` candidates, bias affects which samples activate. Pass the actual bias
/// that will be used with the new neuron for accurate predictions.
#[cfg(test)]
pub(crate) fn compute_net_improvement_with_squash(
    samples: &[HelpfulSample],
    incoming_weight: f32,
    outgoing_weight: f32,
    bias: f32,
    total_baseline_error_sq: f32,
    target_squash: Option<&str>,
) -> f32 {
    let target_activation_fn = get_target_simulation_fn(samples, target_squash);
    let (improvement, _, _) = compute_relu_improvement_and_count(
        samples,
        incoming_weight,
        outgoing_weight,
        bias,
        total_baseline_error_sq,
        target_activation_fn,
    );
    improvement
}

/// Compute synapse improvement accounting for target neuron's activation function.
///
/// For direct synapse connections (source → target), the contribution is `weight × source_activation`.
/// This function simulates the target's activation function to predict accurate improvement,
/// avoiding overprediction near saturation for `HARD_TANH`, TANH, LOGISTIC, etc.
///
/// Returns `improvement_percentage` only. Used in tests; production uses `compute_synapse_improvement_and_count`.
#[cfg(test)]
pub(crate) fn compute_synapse_improvement_with_target_squash(
    samples: &[HelpfulSample],
    weight: f32,
    total_baseline_error_sq: f32,
    target_squash: Option<&str>,
) -> f32 {
    if total_baseline_error_sq <= EPSILON || samples.is_empty() {
        return 0.0;
    }

    let target_sim = get_target_simulation_mode(samples, target_squash);

    let mut baseline_error_sq_sum = 0.0f32; // ACTIVATION domain when simulating
    let mut new_error_sq_sum = 0.0f32;

    for sample in samples {
        // Direct synapse contribution: weight × source_activation
        let contribution = weight * sample.activation;

        let new_error = match target_sim {
            TargetSimulationMode::None => {
                // Linear approximation - assumes contribution directly reduces error (VALUE domain).
                sample.avg_error - contribution
            }
            TargetSimulationMode::Full(target_fn) => {
                // Saturation-aware model (ACTIVATION domain).
                let target_value = sample.target_value.unwrap();
                let target_activation = sample.target_activation.unwrap();
                let desired_value = target_value + sample.avg_error;
                let expected = target_fn(desired_value);

                let baseline_err = expected - target_activation;
                if baseline_err.is_finite() {
                    baseline_error_sq_sum += baseline_err * baseline_err;
                }

                let new_input = target_value + contribution;
                expected - target_fn(new_input)
            }
            TargetSimulationMode::ApproximateValueFromActivation(target_fn) => {
                // Saturation-aware model (ACTIVATION domain), approximating missing target_value.
                let target_activation = sample.target_activation.unwrap();
                let target_value = sample.target_value.unwrap_or(target_activation);
                let desired_value = target_value + sample.avg_error;
                let expected = target_fn(desired_value);

                let baseline_err = expected - target_activation;
                if baseline_err.is_finite() {
                    baseline_error_sq_sum += baseline_err * baseline_err;
                }

                let new_input = target_value + contribution;
                expected - target_fn(new_input)
            }
        };

        if new_error.is_finite() {
            new_error_sq_sum += new_error * new_error;
        }
    }

    let effective_baseline = match target_sim {
        TargetSimulationMode::None => total_baseline_error_sq,
        _ => baseline_error_sq_sum,
    };
    if effective_baseline <= EPSILON {
        return 0.0;
    }

    let improvement = (effective_baseline - new_error_sq_sum) / effective_baseline;
    if improvement.is_finite() {
        improvement
    } else {
        0.0
    }
}

/// Compute improvement, improved count, and worsened count for synapse candidates.
/// All counts use the same saturation-aware methodology for consistency.
///
/// CRITICAL DOMAIN FIX (v0.1.120): When using `target_activation_fn` simulation,
/// both baseline and new error must be computed in ACTIVATION domain. The passed-in
/// `total_baseline_error_sq` is in VALUE domain, so we compute our own ACTIVATION
/// domain baseline when simulating.
///
/// Returns (`improvement_percentage`, `improved_count`, `worsened_count`, `total_count`)
pub(crate) fn compute_synapse_improvement_and_count(
    samples: &[HelpfulSample],
    weight: f32,
    total_baseline_error_sq: f32,
    target_squash: Option<&str>,
) -> (f32, u32, u32, u32) {
    if total_baseline_error_sq <= EPSILON || samples.is_empty() {
        return (0.0, 0, 0, samples.len() as u32);
    }

    let target_sim = get_target_simulation_mode(samples, target_squash);

    let mut baseline_error_sq_sum = 0.0f32; // ACTIVATION domain when simulating
    let mut new_error_sq_sum = 0.0f32;
    let mut improved_count = 0u32;
    let mut worsened_count = 0u32;
    let total_count = samples.len() as u32;

    for sample in samples {
        let contribution = weight * sample.activation;

        let (baseline_error, new_error) = match target_sim {
            TargetSimulationMode::None => {
                // Linear approximation: both errors in VALUE domain.
                (sample.avg_error, sample.avg_error - contribution)
            }
            TargetSimulationMode::Full(target_fn) => {
                // CRITICAL: Use ACTIVATION domain for BOTH baseline and new error.
                // avg_error is in VALUE domain, but MSE is measured in ACTIVATION domain.
                let target_value = sample.target_value.unwrap();
                let target_activation = sample.target_activation.unwrap();
                let desired_value = target_value + sample.avg_error;
                let expected = target_fn(desired_value);

                let baseline_err = expected - target_activation;
                let new_input = target_value + contribution;
                let new_err = expected - target_fn(new_input);

                (baseline_err, new_err)
            }
            TargetSimulationMode::ApproximateValueFromActivation(target_fn) => {
                // As above, but approximate missing target_value from the observed activation.
                let target_activation = sample.target_activation.unwrap();
                let target_value = sample.target_value.unwrap_or(target_activation);
                let desired_value = target_value + sample.avg_error;
                let expected = target_fn(desired_value);

                let baseline_err = expected - target_activation;
                let new_input = target_value + contribution;
                let new_err = expected - target_fn(new_input);

                (baseline_err, new_err)
            }
        };

        if baseline_error.is_finite() {
            baseline_error_sq_sum += baseline_error * baseline_error;
        }
        if new_error.is_finite() {
            new_error_sq_sum += new_error * new_error;
        }

        // Count improved/worsened samples using consistent domain comparison
        if new_error.abs() + EPSILON < baseline_error.abs() {
            improved_count += 1;
        } else if new_error.abs() > baseline_error.abs() + EPSILON {
            worsened_count += 1;
        }
    }

    // Use computed ACTIVATION domain baseline when simulating, else use passed VALUE domain.
    let effective_baseline = match target_sim {
        TargetSimulationMode::None => total_baseline_error_sq,
        _ => baseline_error_sq_sum,
    };

    let improvement = if effective_baseline > EPSILON {
        (effective_baseline - new_error_sq_sum) / effective_baseline
    } else {
        0.0
    };
    let improvement = if improvement.is_finite() {
        improvement
    } else {
        0.0
    };

    (improvement, improved_count, worsened_count, total_count)
}

// =============================================================================
// Pessimism Discount (Issue #506)
// =============================================================================

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
/// ```text
/// improved_ratio = improved_count / total_count
/// adjusted_ratio = improved_ratio ^ NEURON_PESSIMISM_CURVE_EXPONENT
/// discount = NEURON_PESSIMISM_DISCOUNT_FLOOR + (1 - NEURON_PESSIMISM_DISCOUNT_FLOOR) × adjusted_ratio
/// result = gain × discount
/// ```
pub fn apply_neuron_pessimism_discount(gain: f32, improved_count: u32, total_count: u32) -> f32 {
    if total_count == 0 {
        return gain * NEURON_PESSIMISM_DISCOUNT_FLOOR;
    }
    let improved_ratio = improved_count as f32 / total_count as f32;
    let adjusted_ratio = improved_ratio.powf(NEURON_PESSIMISM_CURVE_EXPONENT);
    let discount =
        NEURON_PESSIMISM_DISCOUNT_FLOOR + (1.0 - NEURON_PESSIMISM_DISCOUNT_FLOOR) * adjusted_ratio;
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
/// ```text
/// improved_ratio = improved_count / total_count
/// adjusted_ratio = improved_ratio ^ SYNAPSE_PESSIMISM_CURVE_EXPONENT
/// discount = SYNAPSE_PESSIMISM_DISCOUNT_FLOOR + (1 - SYNAPSE_PESSIMISM_DISCOUNT_FLOOR) × adjusted_ratio
/// result = gain × discount
/// ```
pub fn apply_synapse_pessimism_discount(gain: f32, improved_count: u32, total_count: u32) -> f32 {
    if total_count == 0 {
        return gain * SYNAPSE_PESSIMISM_DISCOUNT_FLOOR;
    }
    let improved_ratio = improved_count as f32 / total_count as f32;
    let adjusted_ratio = improved_ratio.powf(SYNAPSE_PESSIMISM_CURVE_EXPONENT);
    let discount = SYNAPSE_PESSIMISM_DISCOUNT_FLOOR
        + (1.0 - SYNAPSE_PESSIMISM_DISCOUNT_FLOOR) * adjusted_ratio;
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

/// Wrapper for tests - counts improved samples only.
#[cfg(test)]
pub(crate) fn count_improved_samples(
    samples: &[HelpfulSample],
    incoming_weight: f32,
    outgoing_weight: f32,
    bias: f32,
    target_squash: Option<&str>,
) -> (u32, u32) {
    let total_baseline_error_sq: f32 = samples.iter().map(|s| s.avg_error * s.avg_error).sum();
    let target_activation_fn = get_target_simulation_fn(samples, target_squash);
    let (_, improved, total) = compute_relu_improvement_and_count(
        samples,
        incoming_weight,
        outgoing_weight,
        bias,
        total_baseline_error_sq,
        target_activation_fn,
    );
    (improved, total)
}
