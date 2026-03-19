//! Core weight calculation functions for NEAT-AI Discovery.
//!
//! This module contains the primary weight computation logic used for synapse and
//! neuron candidate evaluation:
//!
//! - `calculate_optimal_outgoing_weight` — least squares optimal weight
//! - `calculate_optimal_identity_outgoing_and_bias` — joint weight/bias for IDENTITY neurons
//! - `calculate_optimal_bias` — grid search (CPU or GPU) for optimal bias

use crate::analysis::activation::{activation_name_to_gpu_id, get_bias_range, get_bias_values};
use crate::analysis::gpu::GpuAnalyzer;
use crate::analysis::samples::{EPSILON, HelpfulSample};

use super::{MAX_OUTGOING_WEIGHT, MIN_WEIGHT_RATIO};

// MIN_NEURON_SAMPLE_COUNT moved to constants.rs (Issue #424)
use crate::analysis::constants::MIN_NEURON_SAMPLE_COUNT;

/// Calculate optimal outgoing weight for add-synapse or add-neuron candidates.
///
/// This is the shared weight calculation function used by both synapse and neuron
/// analysis to ensure consistent behaviour and maintainability (DRY principle).
///
/// The formula used is the standard least squares optimal weight:
/// ```text
/// w = Σ(error × activation) / Σ(activation²)
/// ```
///
/// # Arguments
/// * `sum_error_activation` - Σ(error × activation) from samples
/// * `sum_activation_sq` - Σ(activation²) from samples
/// * `incoming_weight` - For neurons: the incoming weight; for synapses: use 1.0
///
/// # Returns
/// * `Some(weight)` - Optimal outgoing weight, clamped to [-`MAX_OUTGOING_WEIGHT`, `MAX_OUTGOING_WEIGHT`]
/// * `None` - If weight cannot be computed (insufficient activation, invalid result, or
///   weight ratio too small for reliable prediction)
///
/// # Weight Ratio Validation
/// For add-neuron candidates where `incoming_weight` > 1.0, we validate that the
/// incoming/outgoing ratio is at least `MIN_WEIGHT_RATIO`. This is based on analysis
/// showing successful discoveries have much larger incoming than outgoing weights
/// (ratio 71x to 104,000x), while failures often have nearly equal weights.
pub fn calculate_optimal_outgoing_weight(
    sum_error_activation: f32,
    sum_activation_sq: f32,
    incoming_weight: f32,
) -> Option<f32> {
    // Need sufficient activation energy to compute meaningful weight
    if sum_activation_sq <= EPSILON {
        return None;
    }

    // Compute raw optimal weight using least squares formula
    let raw_weight = sum_error_activation / (sum_activation_sq + EPSILON);

    // Reject invalid weights
    if !raw_weight.is_finite() || raw_weight.abs() <= EPSILON {
        return None;
    }

    // Clamp to tight range based on successful discovery analysis
    let clamped = raw_weight.clamp(-MAX_OUTGOING_WEIGHT, MAX_OUTGOING_WEIGHT);

    // For add-neuron candidates with non-trivial incoming weights, validate ratio
    // This catches cases where the computed weight is too large relative to incoming
    if incoming_weight.abs() > 1.0 {
        let ratio = incoming_weight.abs() / (clamped.abs() + EPSILON);
        if ratio < MIN_WEIGHT_RATIO {
            // Weight ratio too small - this configuration is unreliable
            // Skip rather than returning a weight that's likely to fail
            return None;
        }
    }

    Some(clamped)
}

/// Special-case optimisation for IDENTITY candidates: fit an affine correction.
///
/// For IDENTITY, the new neuron's output is linear in the source activation:
/// `output = incoming_weight * activation + bias`.
///
/// The generic search path computes `outgoing_weight` without bias, then searches bias with that
/// fixed outgoing weight. For complement-like shapes (eg `1 - x`) this can miss the optimum
/// because the best `outgoing_weight` depends on the bias/intercept.
///
/// We address that by directly fitting a 2-parameter model:
/// `avg_error ≈ outgoing_weight * (incoming_weight * activation) + intercept`,
/// then converting `intercept` to `bias` via `bias = intercept / outgoing_weight`.
pub fn calculate_optimal_identity_outgoing_and_bias(
    samples: &[HelpfulSample],
    incoming_weight: f32,
) -> Option<(f32, f32)> {
    let mut n: f32 = 0.0;
    let mut sum_a = 0.0f32;
    let mut sum_aa = 0.0f32;
    let mut sum_u = 0.0f32;
    let mut sum_uu = 0.0f32;
    let mut sum_e = 0.0f32;
    let mut sum_eu = 0.0f32;

    for sample in samples {
        if !sample.activation.is_finite() || !sample.avg_error.is_finite() {
            continue;
        }
        sum_a += sample.activation;
        sum_aa += sample.activation * sample.activation;
        let u = incoming_weight * sample.activation;
        n += 1.0;
        sum_u += u;
        sum_uu += u * u;
        sum_e += sample.avg_error;
        sum_eu += sample.avg_error * u;
    }

    if n <= 0.0 {
        return None;
    }

    // If the source activation has low variance, explicitly fitting an intercept (bias)
    // tends to overfit constant-ish sources. We already apply variance discounting later,
    // but keeping IDENTITY bias at 0.0 here avoids inflating the *raw* prediction before
    // the discount is applied (see Issue #130 tests).
    //
    // Use the same threshold as compute_source_variance_discount().
    use crate::analysis::constants::MIN_SOURCE_STD_DEV;
    let mean_a = sum_a / n;
    let var_a = (sum_aa / n) - (mean_a * mean_a);
    let std_dev_a = var_a.max(0.0).sqrt();
    if std_dev_a < MIN_SOURCE_STD_DEV {
        let outgoing_weight = calculate_optimal_outgoing_weight(sum_eu, sum_uu, incoming_weight)?;
        return Some((outgoing_weight, 0.0));
    }

    // Solve normal equations for `e ≈ w*u + c` (w = outgoing_weight, c = intercept).
    // If the determinant is ~0, fall back to the no-intercept weight and compute the best intercept.
    let det = sum_uu * n - sum_u * sum_u;

    let outgoing_weight_raw = if det.abs() > EPSILON {
        (sum_eu * n - sum_e * sum_u) / det
    } else {
        sum_eu / (sum_uu + EPSILON)
    };

    if !outgoing_weight_raw.is_finite() || outgoing_weight_raw.abs() <= EPSILON {
        return None;
    }

    let outgoing_weight = outgoing_weight_raw.clamp(-MAX_OUTGOING_WEIGHT, MAX_OUTGOING_WEIGHT);

    // Apply the same reliability guard as the generic path.
    if incoming_weight.abs() > 1.0 {
        let ratio = incoming_weight.abs() / (outgoing_weight.abs() + EPSILON);
        if ratio < MIN_WEIGHT_RATIO {
            return None;
        }
    }

    // Best intercept for fixed outgoing_weight (least squares).
    let intercept = (sum_e - outgoing_weight * sum_u) / n;
    let bias = intercept / outgoing_weight;

    if !bias.is_finite() {
        return None;
    }

    // Guard rail: absurd IDENTITY biases are almost always brittle in production.
    let bias_abs_max = crate::analysis::utils::sensible_bias_abs_max_for_squash("IDENTITY");
    if bias.abs() > bias_abs_max {
        return None;
    }

    Some((outgoing_weight, bias))
}

/// Calculate optimal bias for a neuron candidate using grid search.
///
/// This function finds the bias value that maximises error reduction when combined
/// with the given weights and activation function. It tests multiple bias values
/// across an activation-function-specific range and selects the one that gives
/// the best improvement.
///
/// Uses GPU-accelerated parallel search when analyzer is provided and GPU is available,
/// otherwise falls back to CPU sequential search.
///
/// # Arguments
/// * `samples` - Training samples (source activations and target errors)
/// * `incoming_weight` - Weight from source to new neuron
/// * `outgoing_weight` - Weight from new neuron to target
/// * `activation_fn` - Activation function to apply
/// * `squash` - Activation function name (for bias range selection and GPU)
/// * `analyzer` - Optional GPU analyzer for accelerated search
/// * `target_squash` - Optional target neuron's squash function (for saturation-aware models)
///
/// # Returns
/// Optimal bias value that maximises error reduction
pub fn calculate_optimal_bias(
    samples: &[HelpfulSample],
    incoming_weight: f32,
    outgoing_weight: f32,
    activation_fn: fn(f32) -> f32,
    squash: &str,
    analyzer: Option<&GpuAnalyzer>,
    target_squash: Option<&str>,
) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }

    let bias_range = get_bias_range(squash);

    // Try GPU-accelerated search first if analyzer available
    if let Some(gpu_analyzer) = analyzer
        && gpu_analyzer.has_gpu()
    {
        let activation_type = activation_name_to_gpu_id(squash);
        if let Ok(optimal_bias) = gpu_analyzer.evaluate_bias_gpu(
            samples,
            incoming_weight,
            outgoing_weight,
            activation_type,
            bias_range,
        ) {
            // Guard rail: only accept biases within sensible ranges.
            let bias_abs_max = crate::analysis::utils::sensible_bias_abs_max_for_squash(squash);
            if optimal_bias.is_finite() && optimal_bias.abs() <= bias_abs_max {
                return optimal_bias;
            }
        }
        // If GPU fails, fall through to CPU
    }

    // Use log-spaced bias values for efficient search
    // Evolution will fine-tune the exact value after discovery
    let bias_values = get_bias_values(squash);

    // Calculate baseline error (no new neuron)
    let mut total_baseline_error_sq = 0.0;
    for sample in samples {
        if sample.avg_error.is_finite() {
            total_baseline_error_sq += sample.avg_error * sample.avg_error;
        }
    }

    if total_baseline_error_sq <= EPSILON {
        return 0.0;
    }

    let mut best_bias = 0.0;
    let mut best_error_reduction = f32::NEG_INFINITY;

    // Check if we can use HARD_TANH model (aka CLIPPED; need target_value for all samples).
    let use_hard_tanh = matches!(
        target_squash,
        Some(s) if s.eq_ignore_ascii_case("HARD_TANH") || s.eq_ignore_ascii_case("CLIPPED")
    ) && samples
        .iter()
        .all(|s| s.target_value.is_some() && s.target_activation.is_some());

    // Search over log-spaced bias values
    for &bias in &bias_values {
        // Calculate error with this bias
        let mut total_new_error_sq = 0.0;
        let mut valid_samples = 0;

        for sample in samples {
            if !sample.avg_error.is_finite() || !sample.activation.is_finite() {
                continue;
            }

            // Calculate new neuron's activation with bias
            let pre_activation = incoming_weight * sample.activation + bias;
            let new_neuron_activation = activation_fn(pre_activation);

            if !new_neuron_activation.is_finite() {
                continue;
            }

            // Calculate new error at target neuron
            let correction = outgoing_weight * new_neuron_activation;
            let new_error = if use_hard_tanh {
                // HARD_TANH model: account for target neuron's clamping
                // CRITICAL: avg_error is in VALUE domain (targetValue - currentValue from TypeScript)
                // So we compute desired_value = target_value + avg_error, then squash to get expected activation
                let target_value = sample.target_value.unwrap();
                let desired_value = target_value + sample.avg_error;
                let expected = hard_tanh(desired_value);
                let new_input = target_value + correction;
                let new_output = hard_tanh(new_input);
                expected - new_output
            } else {
                // Linear model
                sample.avg_error - correction
            };

            if new_error.is_finite() {
                total_new_error_sq += new_error * new_error;
                valid_samples += 1;
            }
        }

        // Only consider if we have valid samples
        if valid_samples < MIN_NEURON_SAMPLE_COUNT {
            continue;
        }

        // Calculate error reduction (positive is good)
        let error_reduction = total_baseline_error_sq - total_new_error_sq;

        if error_reduction > best_error_reduction {
            best_error_reduction = error_reduction;
            best_bias = bias;
        }
    }

    best_bias
}

/// Apply `HARD_TANH` activation function (clamp to [-1, 1])
#[inline(always)]
pub(super) fn hard_tanh(x: f32) -> f32 {
    x.clamp(-1.0, 1.0)
}
