//! Target simulation functions, activation predicates, and variance checking.
//!
//! This module handles:
//! - Target simulation mode selection (full, approximate, none)
//! - Activation function classification predicates
//! - Output variance validation for candidate neurons

#![allow(clippy::cast_possible_truncation)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use crate::analysis::samples::HelpfulSample;

/// Minimum variance of new neuron output across samples.
///
/// Issue #123: When bias is too large relative to the input range, the neuron becomes
/// saturated and outputs nearly-constant values regardless of input. For example:
/// - SOFTSIGN(5 + 0.35×x) ≈ 0.83 for all typical x values
/// - TANH(10 + x) ≈ 1.0 for all x > -9
///
/// A constant-output neuron CANNOT reduce error correlation - it just adds a fixed offset.
/// The prediction model incorrectly assumes the output varies with input, leading to
/// massive prediction failures (e.g., predicting 4.67% improvement when actual is ~0%).
///
/// This threshold rejects candidates where the output standard deviation is below 0.01,
/// meaning the neuron produces nearly identical output across all samples.
const MIN_NEURON_OUTPUT_STD_DEV: f32 = 0.01;

/// Target simulation mode for saturation-aware candidate scoring.
///
/// Most accurate mode requires both `target_value` (pre-activation) and `target_activation`
/// for every sample. In production, `target_value` is not always recorded, but
/// `target_activation` typically is.
///
/// For a small set of squash functions where a reasonable approximation is possible, we can
/// still simulate in activation domain by approximating `target_value` from the observed
/// activation. This is particularly important for `HARD_TANH`, where the linear model can
/// massively overstate improvement near saturation.
#[derive(Copy, Clone)]
pub enum TargetSimulationMode {
    /// No saturation-aware simulation is available; fall back to the linear (value-domain) model.
    None,
    /// Full saturation-aware simulation using recorded `target_value` + `target_activation`.
    Full(fn(f32) -> f32),
    /// Saturation-aware simulation using recorded `target_activation` and an approximation for
    /// `target_value`.
    ApproximateValueFromActivation(fn(f32) -> f32),
}

// ============================================================================
// Activation Function Predicates
// ============================================================================

/// Check if a target neuron uses a threshold-based discrete activation function.
///
/// STEP and BIPOLAR can benefit from a specialised threshold-crossing analysis model
/// that counts how many samples would flip to the correct output if we add a new connection.
///
/// - **STEP**: Output = value > 0 ? 1 : 0
/// - **BIPOLAR**: Output = value > 0 ? 1 : -1
///
/// # Performance (Issue #211)
/// Uses `eq_ignore_ascii_case` for zero-allocation case-insensitive comparison.
/// This function is called in hot loops during candidate evaluation.
#[inline]
pub fn is_threshold_activation(squash: &str) -> bool {
    squash.eq_ignore_ascii_case("STEP") || squash.eq_ignore_ascii_case("BIPOLAR")
}

// ============================================================================
// Target Simulation Functions
// ============================================================================

/// Get the target simulation function for a squash name.
///
/// This delegates to `crate::activations::target_simulation_fn`, which is kept in sync
/// with NEAT-AI's activation registry (and supports aliases + case-insensitive names).
#[inline]
pub fn get_target_activation_fn(squash: &str) -> Option<fn(f32) -> f32> {
    crate::activations::target_simulation_fn(squash)
}

/// Check if samples support target activation simulation (all have target data).
/// Returns the activation function to use, or None if linear approximation should be used.
#[inline]
pub fn get_target_simulation_fn(
    samples: &[HelpfulSample],
    target_squash: Option<&str>,
) -> Option<fn(f32) -> f32> {
    let squash = target_squash?;
    let activation_fn = get_target_activation_fn(squash)?;

    // Verify all samples have the required target data
    if samples
        .iter()
        .all(|s| s.target_value.is_some() && s.target_activation.is_some())
    {
        Some(activation_fn)
    } else {
        None
    }
}

/// Select the best available target simulation mode for the given samples.
#[inline]
pub fn get_target_simulation_mode(
    samples: &[HelpfulSample],
    target_squash: Option<&str>,
) -> TargetSimulationMode {
    let Some(squash) = target_squash else {
        return TargetSimulationMode::None;
    };
    let Some(activation_fn) = get_target_activation_fn(squash) else {
        return TargetSimulationMode::None;
    };

    let all_have_activation = samples.iter().all(|s| s.target_activation.is_some());
    if !all_have_activation {
        return TargetSimulationMode::None;
    }

    let all_have_value = samples.iter().all(|s| s.target_value.is_some());
    if all_have_value {
        return TargetSimulationMode::Full(activation_fn);
    }

    // Approximation path: keep deliberately narrow (Jan 2026).
    //
    // `HARD_TANH` (aka `CLIPPED`) is piecewise linear and, when not saturated,
    // `target_activation == target_value`.
    // When saturated, the exact pre-activation is unknown, but approximating it as ±1 still avoids
    // the linear-model failure mode where we assume the activation can move beyond the clamp.
    if squash.eq_ignore_ascii_case("HARD_TANH") || squash.eq_ignore_ascii_case("CLIPPED") {
        return TargetSimulationMode::ApproximateValueFromActivation(activation_fn);
    }

    TargetSimulationMode::None
}

/// Legacy function for backwards compatibility - returns true only for `HARD_TANH` (or its alias CLIPPED).
/// Deprecated: Use `get_target_simulation_fn` instead for more accurate simulation
#[inline]
pub fn can_use_hard_tanh(samples: &[HelpfulSample], target_squash: Option<&str>) -> bool {
    matches!(
        target_squash,
        Some(s) if s.eq_ignore_ascii_case("HARD_TANH") || s.eq_ignore_ascii_case("CLIPPED")
    ) && samples
        .iter()
        .all(|s| s.target_value.is_some() && s.target_activation.is_some())
}

/// Check if a new neuron would be saturated (producing nearly-constant output).
///
/// Issue #123: Neurons with large bias values relative to typical inputs become saturated
/// and output nearly the same value regardless of input. This causes massive prediction
/// failures because the linear error model assumes output varies with input.
///
/// Returns `true` if the neuron is NOT saturated (output has sufficient variance).
/// Returns `false` if the neuron IS saturated (should be rejected).
///
/// IMPORTANT: We only reject when INPUT has variance but OUTPUT doesn't. If input is
/// already constant (low variance), then constant output is expected and predictions
/// will still be valid.
///
/// # Arguments
/// * `samples` - The samples to evaluate
/// * `incoming_weight` - Weight from source to new neuron
/// * `bias` - Bias of the new neuron
/// * `activation_fn` - Activation function of the new neuron
pub fn has_sufficient_output_variance(
    samples: &[HelpfulSample],
    incoming_weight: f32,
    bias: f32,
    activation_fn: fn(f32) -> f32,
) -> bool {
    if samples.len() < 2 {
        return false;
    }

    // Compute mean and variance of both input and output
    let mut input_sum = 0.0f64;
    let mut input_sum_sq = 0.0f64;
    let mut output_sum = 0.0f64;
    let mut output_sum_sq = 0.0f64;
    let mut count = 0u32;

    for sample in samples {
        let input = sample.activation;
        let pre_activation = incoming_weight * input + bias;
        let output = activation_fn(pre_activation);
        if output.is_finite() && input.is_finite() {
            input_sum += input as f64;
            input_sum_sq += (input as f64) * (input as f64);
            output_sum += output as f64;
            output_sum_sq += (output as f64) * (output as f64);
            count += 1;
        }
    }

    if count < 2 {
        return false;
    }

    let n = count as f64;

    // Calculate input variance
    let input_mean = input_sum / n;
    let input_variance = (input_sum_sq / n) - (input_mean * input_mean);
    let input_std_dev = input_variance.max(0.0).sqrt() as f32;

    // Calculate output variance
    let output_mean = output_sum / n;
    let output_variance = (output_sum_sq / n) - (output_mean * output_mean);
    let output_std_dev = output_variance.max(0.0).sqrt() as f32;

    // If input already has low variance, constant output is expected - allow it
    if input_std_dev < MIN_NEURON_OUTPUT_STD_DEV {
        return true;
    }

    // If input has variance but output doesn't, the neuron is saturated - reject
    output_std_dev >= MIN_NEURON_OUTPUT_STD_DEV
}
