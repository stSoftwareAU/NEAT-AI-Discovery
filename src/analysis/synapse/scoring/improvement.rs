//! Core synapse improvement calculation algorithm
//!
//! This module contains functions for computing improvement scores for synapse and
//! neuron candidates, including saturation-aware simulation, candidate deduplication,
//! and sample-level improvement counting (Issues #413, #526).

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use crate::CandidateNeuronJson;
use crate::analysis::activation::{TargetSimulationMode, get_target_simulation_mode};
use crate::analysis::samples::{EPSILON, HelpfulSample};
use std::collections::HashMap;

// =============================================================================
// Candidate Deduplication (Issue #526)
// =============================================================================

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

// =============================================================================
// ReLU Improvement Calculation
// =============================================================================

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
            // Gracefully skip samples missing target data (Issue #940).
            let Some(target_value) = sample.target_value else {
                continue;
            };
            let Some(target_activation) = sample.target_activation else {
                continue;
            };
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

// =============================================================================
// Activation Improvement Calculation
// =============================================================================

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
            // Gracefully skip samples missing target data (Issue #940).
            let Some(target_value) = sample.target_value else {
                continue;
            };
            let Some(target_activation) = sample.target_activation else {
                continue;
            };
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

// =============================================================================
// Synapse Improvement Calculation
// =============================================================================

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
                // Gracefully skip samples missing target data (Issue #940).
                let Some(target_value) = sample.target_value else {
                    continue;
                };
                let Some(target_activation) = sample.target_activation else {
                    continue;
                };
                let desired_value = target_value + sample.avg_error;
                let expected = target_fn(desired_value);

                let baseline_err = expected - target_activation;
                let new_input = target_value + contribution;
                let new_err = expected - target_fn(new_input);

                (baseline_err, new_err)
            }
            TargetSimulationMode::ApproximateValueFromActivation {
                activation_fn: target_fn,
                inverse_fn,
            } => {
                // As above, but approximate missing target_value using the inverse function
                // (Issue #906). Gracefully skip samples missing target_activation (Issue #940).
                let Some(target_activation) = sample.target_activation else {
                    continue;
                };
                let target_value = sample
                    .target_value
                    .unwrap_or_else(|| inverse_fn(target_activation));
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
