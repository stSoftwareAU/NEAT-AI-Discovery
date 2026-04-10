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
// Branchless Helpers (Issue #1075)
// =============================================================================

/// Branchless selection: returns `value` if finite, `0.0` otherwise.
/// Avoids a data-dependent branch that inhibits auto-vectorisation.
#[inline(always)]
fn select_finite(value: f32) -> f32 {
    if value.is_finite() { value } else { 0.0 }
}

/// Compute final improvement from accumulated error sums.
/// Shared by all improvement functions to avoid repetition.
#[inline(always)]
fn finalise_improvement(effective_baseline: f32, new_error_sq_sum: f32) -> f32 {
    if effective_baseline > EPSILON {
        let imp = (effective_baseline - new_error_sq_sum) / effective_baseline;
        select_finite(imp)
    } else {
        0.0
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
/// Issue #1075: Dispatches to specialised branchless variants to improve
/// auto-vectorisation of the hot inner loop.
///
/// Returns (`improvement_percentage`, `improved_count`, `total_count`)
pub fn compute_relu_improvement_and_count(
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

    if let Some(target_fn) = target_activation_fn {
        compute_relu_improvement_with_target(
            samples,
            incoming_weight,
            outgoing_weight,
            bias,
            target_fn,
        )
    } else {
        compute_relu_improvement_no_target(
            samples,
            incoming_weight,
            outgoing_weight,
            bias,
            total_baseline_error_sq,
        )
    }
}

/// Branchless `ReLU` improvement for the no-target (VALUE domain) path.
///
/// Issue #1075: All samples are processed without `continue` or `Option` checks.
/// The `is_finite()` guard uses branchless `select_finite` so the compiler can
/// auto-vectorise the accumulation loop.
#[inline]
fn compute_relu_improvement_no_target(
    samples: &[HelpfulSample],
    incoming_weight: f32,
    outgoing_weight: f32,
    bias: f32,
    total_baseline_error_sq: f32,
) -> (f32, u32, u32) {
    let mut new_error_sq_sum = 0.0f32;
    let mut improved_count = 0u32;

    for sample in samples {
        let pre_activation = incoming_weight * sample.activation + bias;
        let relu_output = pre_activation.max(0.0);
        let contribution = outgoing_weight * relu_output;

        let baseline_error = sample.avg_error;
        let new_error = baseline_error - contribution;

        // Branchless accumulation — select_finite returns 0.0 for NaN/Inf
        let new_err_safe = select_finite(new_error);
        new_error_sq_sum += new_err_safe * new_err_safe;

        if new_error.abs() + EPSILON < baseline_error.abs() {
            improved_count += 1;
        }
    }

    let improvement = finalise_improvement(total_baseline_error_sq, new_error_sq_sum);
    (improvement, improved_count, samples.len() as u32)
}

/// `ReLU` improvement with target activation function (ACTIVATION domain).
///
/// Issue #1075: Separated from the no-target path to eliminate the per-sample
/// `Option` branch. The `continue` on missing target data is inherent to this
/// path and cannot be removed without changing semantics.
#[inline]
fn compute_relu_improvement_with_target(
    samples: &[HelpfulSample],
    incoming_weight: f32,
    outgoing_weight: f32,
    bias: f32,
    target_fn: fn(f32) -> f32,
) -> (f32, u32, u32) {
    let mut baseline_error_sq_sum = 0.0f32;
    let mut new_error_sq_sum = 0.0f32;
    let mut improved_count = 0u32;

    for sample in samples {
        let pre_activation = incoming_weight * sample.activation + bias;
        let relu_output = pre_activation.max(0.0);
        let contribution = outgoing_weight * relu_output;

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

        let baseline_error = expected - target_activation;
        let new_input = target_value + contribution;
        let new_error = expected - target_fn(new_input);

        // Branchless accumulation
        let base_safe = select_finite(baseline_error);
        baseline_error_sq_sum += base_safe * base_safe;
        let new_safe = select_finite(new_error);
        new_error_sq_sum += new_safe * new_safe;

        if new_error.abs() + EPSILON < baseline_error.abs() {
            improved_count += 1;
        }
    }

    let improvement = finalise_improvement(baseline_error_sq_sum, new_error_sq_sum);
    (improvement, improved_count, samples.len() as u32)
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
/// Issue #1075: Dispatches to specialised branchless variants to improve
/// auto-vectorisation of the hot inner loop.
///
/// Returns (`improvement_percentage`, `improved_count`, `total_count`)
pub fn compute_activation_improvement_and_count(
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

    if let Some(target_fn) = target_activation_fn {
        compute_activation_improvement_with_target(
            samples,
            incoming_weight,
            outgoing_weight,
            bias,
            activation_fn,
            target_fn,
        )
    } else {
        compute_activation_improvement_no_target(
            samples,
            incoming_weight,
            outgoing_weight,
            bias,
            activation_fn,
            total_baseline_error_sq,
        )
    }
}

/// Branchless activation improvement for the no-target (VALUE domain) path.
///
/// Issue #1075: All samples are processed without `continue` or `Option` checks.
/// The `is_finite()` guard uses branchless `select_finite` so the compiler can
/// auto-vectorise the accumulation loop.
#[inline]
fn compute_activation_improvement_no_target(
    samples: &[HelpfulSample],
    incoming_weight: f32,
    outgoing_weight: f32,
    bias: f32,
    activation_fn: fn(f32) -> f32,
    total_baseline_error_sq: f32,
) -> (f32, u32, u32) {
    let mut new_error_sq_sum = 0.0f32;
    let mut improved_count = 0u32;

    for sample in samples {
        let pre_activation = incoming_weight * sample.activation + bias;
        let neuron_output = activation_fn(pre_activation);
        let contribution = outgoing_weight * neuron_output;

        let baseline_error = sample.avg_error;
        let new_error = baseline_error - contribution;

        // Branchless accumulation
        let new_err_safe = select_finite(new_error);
        new_error_sq_sum += new_err_safe * new_err_safe;

        if new_error.abs() + EPSILON < baseline_error.abs() {
            improved_count += 1;
        }
    }

    let improvement = finalise_improvement(total_baseline_error_sq, new_error_sq_sum);
    (improvement, improved_count, samples.len() as u32)
}

/// Activation improvement with target function (ACTIVATION domain).
///
/// Issue #1075: Separated from the no-target path to eliminate the per-sample
/// `Option` branch.
#[inline]
fn compute_activation_improvement_with_target(
    samples: &[HelpfulSample],
    incoming_weight: f32,
    outgoing_weight: f32,
    bias: f32,
    activation_fn: fn(f32) -> f32,
    target_fn: fn(f32) -> f32,
) -> (f32, u32, u32) {
    let mut baseline_error_sq_sum = 0.0f32;
    let mut new_error_sq_sum = 0.0f32;
    let mut improved_count = 0u32;

    for sample in samples {
        let pre_activation = incoming_weight * sample.activation + bias;
        let neuron_output = activation_fn(pre_activation);
        let contribution = outgoing_weight * neuron_output;

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

        let baseline_error = expected - target_activation;
        let new_input = target_value + contribution;
        let new_error = expected - target_fn(new_input);

        // Branchless accumulation
        let base_safe = select_finite(baseline_error);
        baseline_error_sq_sum += base_safe * base_safe;
        let new_safe = select_finite(new_error);
        new_error_sq_sum += new_safe * new_safe;

        if new_error.abs() + EPSILON < baseline_error.abs() {
            improved_count += 1;
        }
    }

    let improvement = finalise_improvement(baseline_error_sq_sum, new_error_sq_sum);
    (improvement, improved_count, samples.len() as u32)
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
/// Issue #1075: Dispatches to specialised branchless variants based on the
/// `TargetSimulationMode` to improve auto-vectorisation of the hot inner loop.
///
/// Returns (`improvement_percentage`, `improved_count`, `worsened_count`, `total_count`)
pub fn compute_synapse_improvement_and_count(
    samples: &[HelpfulSample],
    weight: f32,
    total_baseline_error_sq: f32,
    target_squash: Option<&str>,
) -> (f32, u32, u32, u32) {
    if total_baseline_error_sq <= EPSILON || samples.is_empty() {
        return (0.0, 0, 0, samples.len() as u32);
    }

    let target_sim = get_target_simulation_mode(samples, target_squash);

    match target_sim {
        TargetSimulationMode::None => {
            compute_synapse_improvement_no_target(samples, weight, total_baseline_error_sq)
        }
        TargetSimulationMode::Full(target_fn) => {
            compute_synapse_improvement_with_target(samples, weight, target_fn)
        }
        TargetSimulationMode::ApproximateValueFromActivation {
            activation_fn: target_fn,
            inverse_fn,
        } => compute_synapse_improvement_approximate(samples, weight, target_fn, inverse_fn),
    }
}

/// Branchless synapse improvement for the no-target (VALUE domain) path.
///
/// Issue #1075: No `continue`, no `Option` checks, and `is_finite()` guards
/// replaced with branchless `select_finite` for auto-vectorisation.
#[inline]
fn compute_synapse_improvement_no_target(
    samples: &[HelpfulSample],
    weight: f32,
    total_baseline_error_sq: f32,
) -> (f32, u32, u32, u32) {
    let mut new_error_sq_sum = 0.0f32;
    let mut improved_count = 0u32;
    let mut worsened_count = 0u32;

    for sample in samples {
        let contribution = weight * sample.activation;
        let baseline_error = sample.avg_error;
        let new_error = baseline_error - contribution;

        // Branchless accumulation
        let new_err_safe = select_finite(new_error);
        new_error_sq_sum += new_err_safe * new_err_safe;

        let new_abs = new_error.abs();
        let base_abs = baseline_error.abs();
        if new_abs + EPSILON < base_abs {
            improved_count += 1;
        } else if new_abs > base_abs + EPSILON {
            worsened_count += 1;
        }
    }

    let improvement = finalise_improvement(total_baseline_error_sq, new_error_sq_sum);
    (
        improvement,
        improved_count,
        worsened_count,
        samples.len() as u32,
    )
}

/// Synapse improvement with full target simulation (ACTIVATION domain).
///
/// Issue #1075: Separated from the no-target path to eliminate the per-sample
/// `TargetSimulationMode` match.
#[inline]
fn compute_synapse_improvement_with_target(
    samples: &[HelpfulSample],
    weight: f32,
    target_fn: fn(f32) -> f32,
) -> (f32, u32, u32, u32) {
    let mut baseline_error_sq_sum = 0.0f32;
    let mut new_error_sq_sum = 0.0f32;
    let mut improved_count = 0u32;
    let mut worsened_count = 0u32;

    for sample in samples {
        let contribution = weight * sample.activation;

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

        let baseline_error = expected - target_activation;
        let new_input = target_value + contribution;
        let new_error = expected - target_fn(new_input);

        // Branchless accumulation
        let base_safe = select_finite(baseline_error);
        baseline_error_sq_sum += base_safe * base_safe;
        let new_safe = select_finite(new_error);
        new_error_sq_sum += new_safe * new_safe;

        let new_abs = new_error.abs();
        let base_abs = baseline_error.abs();
        if new_abs + EPSILON < base_abs {
            improved_count += 1;
        } else if new_abs > base_abs + EPSILON {
            worsened_count += 1;
        }
    }

    let improvement = finalise_improvement(baseline_error_sq_sum, new_error_sq_sum);
    (
        improvement,
        improved_count,
        worsened_count,
        samples.len() as u32,
    )
}

/// Synapse improvement with approximate inverse simulation (ACTIVATION domain).
///
/// Issue #1075: Separated from other paths to eliminate the per-sample
/// `TargetSimulationMode` match. Uses inverse function to recover `target_value`
/// from `target_activation` when `target_value` is not recorded (Issue #906).
#[inline]
fn compute_synapse_improvement_approximate(
    samples: &[HelpfulSample],
    weight: f32,
    target_fn: fn(f32) -> f32,
    inverse_fn: fn(f32) -> f32,
) -> (f32, u32, u32, u32) {
    let mut baseline_error_sq_sum = 0.0f32;
    let mut new_error_sq_sum = 0.0f32;
    let mut improved_count = 0u32;
    let mut worsened_count = 0u32;

    for sample in samples {
        let contribution = weight * sample.activation;

        // Gracefully skip samples missing target_activation (Issue #940).
        let Some(target_activation) = sample.target_activation else {
            continue;
        };
        let target_value = sample
            .target_value
            .unwrap_or_else(|| inverse_fn(target_activation));
        let desired_value = target_value + sample.avg_error;
        let expected = target_fn(desired_value);

        let baseline_error = expected - target_activation;
        let new_input = target_value + contribution;
        let new_error = expected - target_fn(new_input);

        // Branchless accumulation
        let base_safe = select_finite(baseline_error);
        baseline_error_sq_sum += base_safe * base_safe;
        let new_safe = select_finite(new_error);
        new_error_sq_sum += new_safe * new_safe;

        let new_abs = new_error.abs();
        let base_abs = baseline_error.abs();
        if new_abs + EPSILON < base_abs {
            improved_count += 1;
        } else if new_abs > base_abs + EPSILON {
            worsened_count += 1;
        }
    }

    let improvement = finalise_improvement(baseline_error_sq_sum, new_error_sq_sum);
    (
        improvement,
        improved_count,
        worsened_count,
        samples.len() as u32,
    )
}
