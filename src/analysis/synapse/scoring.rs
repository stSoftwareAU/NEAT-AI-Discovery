//! Synapse scoring and improvement calculation
//!
//! This module contains functions for computing improvement scores, saturation-aware
//! simulation, and source/target type boosting (Issues #413, #467, #468).

use crate::CandidateNeuronJson;
#[cfg(test)]
use crate::analysis::activation::get_target_simulation_fn;
use crate::analysis::activation::{TargetSimulationMode, get_target_simulation_mode};
use crate::analysis::samples::{EPSILON, HelpfulSample};
use crate::analysis::utils::parse_input_index;
use std::collections::HashMap;

// Boosting constants
use crate::analysis::constants::{EXISTING_HIDDEN_TARGET_BOOST, INPUT_SOURCE_BOOST};

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
        .map(|t| t == "hidden")
        .unwrap_or(false)
    {
        gain * EXISTING_HIDDEN_TARGET_BOOST as f32
    } else {
        gain
    }
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

/// Returns true if this add-neuron candidate is "extreme" enough to warrant a conservative pair.
///
/// We intentionally base this on incoming weight and bias (not outgoing), because outgoing
/// weight is already clamped and ReLU candidates commonly use outgoing=0.1 by design.
pub(crate) fn upsert_candidate(
    map: &mut HashMap<(String, String, String, i8, i8), CandidateNeuronJson>,
    candidate: CandidateNeuronJson,
) {
    use std::collections::hash_map::Entry;

    // Key includes signs of BOTH incoming_weight AND outgoing_weight so that:
    // 1. Different ReLU orientations (incoming_weight ±1) are kept separately
    // 2. Split-error complementary pairs (same incoming_weight, opposite outgoing_weight)
    //    are also kept separately - one pushes output UP, one pushes DOWN
    let key = (
        candidate.source_neuron_uuid.clone(),
        candidate.target_neuron_uuid.clone(),
        candidate.squash.clone(),
        weight_sign(candidate.incoming_weight),
        weight_sign(candidate.outgoing_weight),
    );

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

/// Combined computation of improvement and count for ReLU candidates.
/// Single pass over samples for better cache efficiency.
///
/// When `target_activation_fn` is Some, simulates the target neuron's actual activation
/// function for more accurate improvement estimates. Otherwise falls back to linear approximation.
///
/// IMPORTANT: The `bias` parameter is critical for accurate predictions. It shifts the ReLU
/// activation threshold, affecting which samples produce non-zero output.
///
/// CRITICAL DOMAIN FIX (v0.1.120): When using target_activation_fn simulation,
/// both baseline and new error must be computed in ACTIVATION domain.
///
/// Returns (improvement_percentage, improved_count, total_count)
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

    for sample in samples.iter() {
        let pre_activation = incoming_weight * sample.activation + bias;
        let relu_output = pre_activation.max(0.0);
        let contribution = outgoing_weight * relu_output;

        let (baseline_error, new_error) = if let Some(target_fn) = target_activation_fn {
            // CRITICAL: Use ACTIVATION domain for BOTH baseline and new error.
            let target_value = unsafe { sample.target_value.unwrap_unchecked() };
            let target_activation = unsafe { sample.target_activation.unwrap_unchecked() };
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
/// CRITICAL DOMAIN FIX (v0.1.120): When using target_activation_fn simulation,
/// both baseline and new error must be computed in ACTIVATION domain.
///
/// Returns (improvement_percentage, improved_count, total_count)
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
            let target_value = unsafe { sample.target_value.unwrap_unchecked() };
            let target_activation = unsafe { sample.target_activation.unwrap_unchecked() };
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
/// NOTE: For ReLU candidates, bias affects which samples activate. Pass the actual bias
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
/// avoiding overprediction near saturation for HARD_TANH, TANH, LOGISTIC, etc.
///
/// Returns improvement_percentage only. Used in tests; production uses compute_synapse_improvement_and_count.
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
/// CRITICAL DOMAIN FIX (v0.1.120): When using target_activation_fn simulation,
/// both baseline and new error must be computed in ACTIVATION domain. The passed-in
/// total_baseline_error_sq is in VALUE domain, so we compute our own ACTIVATION
/// domain baseline when simulating.
///
/// Returns (improvement_percentage, improved_count, worsened_count, total_count)
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
