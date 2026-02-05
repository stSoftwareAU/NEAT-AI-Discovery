//! Activation function related code for NEAT-AI Discovery analysis.
//!
//! This module contains:
//! - CPU activation function implementations for candidate evaluation
//! - Activation candidate specifications (ACTIVATION_SPECS)
//! - GPU ID mapping for activation functions
//! - Bias range helpers for different activation types
//! - Predicates for activation function classification
//!
//! Note: This module is separate from `src/activations.rs` which handles
//! TypeScript-Rust interop for squash names and provides `apply_scalar_squash`.
//! This module focuses on candidate discovery and evaluation.

// ============================================================================
// CPU Activation Function Implementations
// ============================================================================

/// GELU activation function.
/// GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
pub fn gelu_activation(x: f32) -> f32 {
    let x_cubed = x * x * x;
    let tanh_arg = 0.797_884_6 * (x + 0.044_715 * x_cubed);
    0.5 * x * (1.0 + tanh_arg.tanh())
}

/// ELU activation function.
/// ELU(x) = x if x >= 0, else exp(x) - 1
pub fn elu_activation(x: f32) -> f32 {
    if x >= 0.0 {
        x
    } else {
        x.exp() - 1.0
    }
}

/// Softplus activation function.
/// Softplus(x) = ln(1 + exp(x)), linearised for x > 20 to avoid overflow.
pub fn softplus_activation(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Logistic (sigmoid) activation function.
/// Logistic(x) = 1 / (1 + exp(-x)), numerically stable.
pub fn logistic_activation(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let exp_x = x.exp();
        exp_x / (1.0 + exp_x)
    }
}

/// Tanh activation function.
pub fn tanh_activation(x: f32) -> f32 {
    x.tanh()
}

/// Identity activation function.
pub fn identity_activation(x: f32) -> f32 {
    x
}

/// Bipolar activation function.
/// BIPOLAR(x) = 1 if x > 0, else -1
pub fn bipolar_activation(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else {
        -1.0
    }
}

/// Clipped activation function.
/// CLIPPED(x) = clamp(x, -1, 1)
pub fn clipped_activation(x: f32) -> f32 {
    x.clamp(-1.0, 1.0)
}

/// Absolute activation function.
pub fn absolute_activation(x: f32) -> f32 {
    x.abs()
}

// ============================================================================
// NEW ACTIVATION FUNCTIONS (v0.1.139)
// Based on analysis of successful discoveries that evolved TO these activations
// ============================================================================

/// Mish activation function.
/// 2 successful discoveries evolved TO Mish (from ELU and Softplus).
/// Self-regularised activation: x * tanh(softplus(x))
pub fn mish_activation(x: f32) -> f32 {
    let sp = if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
    x * sp.tanh()
}

/// Hard tanh activation function.
/// 1 successful discovery evolved CLIPPED → HARD_TANH.
/// Linear in [-1, 1], saturates outside. Same as CLIPPED but named for NEAT-AI.
pub fn hard_tanh_activation(x: f32) -> f32 {
    x.clamp(-1.0, 1.0)
}

/// Softsign activation function.
/// 1 successful discovery neuron with SOFTSIGN.
/// Smooth approximation of sign function: x / (1 + |x|)
pub fn softsign_activation(x: f32) -> f32 {
    x / (1.0 + x.abs())
}

/// Bent identity activation function.
/// 1 successful discovery evolved LeakyReLU → BENT_IDENTITY.
/// Smooth, nearly linear: (sqrt(x² + 1) - 1) / 2 + x
pub fn bent_identity_activation(x: f32) -> f32 {
    ((x * x + 1.0).sqrt() - 1.0) / 2.0 + x
}

/// Arctan activation function.
/// Similar to SOFTSIGN, bounded output.
pub fn arctan_activation(x: f32) -> f32 {
    x.atan()
}

/// ReLU6 activation function.
/// Capped ReLU at 6, useful for quantisation.
pub fn relu6_activation(x: f32) -> f32 {
    x.clamp(0.0, 6.0)
}

// ============================================================================
// Activation Candidate Specification
// ============================================================================

/// Weight orientation options for bidirectional activations.
pub const ORIENTATIONS_BIDIRECTIONAL: [f32; 2] = [1.0, -1.0];

/// Log-spaced scale range for incoming weights - covers multiple orders of magnitude
/// efficiently. Evolution will fine-tune the exact values after discovery.
/// Extended to very large scales (50, 100) for aggressive signal amplification.
/// Note: Very large scales may cause numerical instability with some activations.
pub const SCALES_WIDE: [f32; 9] = [0.1, 0.2, 0.35, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0];

/// Log-spaced scales for smooth activation functions (TANH, LOGISTIC, SELU) that
/// saturate at large inputs. Larger scales included but will saturate the output,
/// which may still be useful for binary-like thresholding behaviour.
pub const SCALES_SMOOTH: [f32; 8] = [0.1, 0.2, 0.35, 0.5, 1.0, 2.0, 4.0, 10.0];

/// Specification for an activation function candidate.
pub struct ActivationCandidateSpec {
    /// Name of the activation function (used in NEAT-AI).
    pub name: &'static str,
    /// Weight orientations to try (typically [1.0, -1.0] for bidirectional).
    pub orientations: &'static [f32],
    /// Weight scales to try.
    pub scales: &'static [f32],
    /// The activation function implementation.
    pub activation: fn(f32) -> f32,
    /// Minimum improvement threshold for this activation (currently unused, kept for future).
    pub min_improvement: f32,
}

/// Array of activation function specifications for candidate generation.
///
/// This array defines which activation functions are considered when searching for
/// new neuron candidates. Each entry specifies the function name, weight orientations
/// to try, scale factors, and the implementation function.
pub const ACTIVATION_SPECS: [ActivationCandidateSpec; 15] = [
    // ========================================================================
    // ORIGINAL ACTIVATIONS (v0.1.x)
    // ========================================================================
    ActivationCandidateSpec {
        name: "GELU",
        orientations: &ORIENTATIONS_BIDIRECTIONAL,
        scales: &SCALES_WIDE,
        activation: gelu_activation,
        min_improvement: 0.0,
    },
    ActivationCandidateSpec {
        name: "ELU",
        orientations: &ORIENTATIONS_BIDIRECTIONAL,
        scales: &SCALES_WIDE,
        activation: elu_activation,
        min_improvement: 0.0,
    },
    ActivationCandidateSpec {
        name: "Softplus",
        orientations: &ORIENTATIONS_BIDIRECTIONAL,
        scales: &SCALES_WIDE,
        activation: softplus_activation,
        min_improvement: 0.0,
    },
    ActivationCandidateSpec {
        name: "LOGISTIC",
        orientations: &ORIENTATIONS_BIDIRECTIONAL,
        scales: &SCALES_SMOOTH,
        activation: logistic_activation,
        min_improvement: 0.0,
    },
    ActivationCandidateSpec {
        name: "TANH",
        orientations: &ORIENTATIONS_BIDIRECTIONAL,
        scales: &SCALES_SMOOTH,
        activation: tanh_activation,
        min_improvement: 0.0,
    },
    ActivationCandidateSpec {
        name: "IDENTITY",
        orientations: &ORIENTATIONS_BIDIRECTIONAL,
        scales: &SCALES_WIDE,
        activation: identity_activation,
        min_improvement: 0.0,
    },
    ActivationCandidateSpec {
        name: "BIPOLAR",
        orientations: &ORIENTATIONS_BIDIRECTIONAL,
        scales: &SCALES_WIDE,
        activation: bipolar_activation,
        min_improvement: 0.0,
    },
    ActivationCandidateSpec {
        name: "CLIPPED",
        orientations: &ORIENTATIONS_BIDIRECTIONAL,
        scales: &SCALES_SMOOTH,
        activation: clipped_activation,
        min_improvement: 0.0,
    },
    ActivationCandidateSpec {
        name: "ABSOLUTE",
        orientations: &ORIENTATIONS_BIDIRECTIONAL,
        scales: &SCALES_WIDE,
        activation: absolute_activation,
        min_improvement: 0.0,
    },
    // ========================================================================
    // NEW ACTIVATIONS (v0.1.139) - Based on successful discovery evolutions
    // ========================================================================
    //
    // NOTE (Issue #134 follow-up): We intentionally do NOT propose LeakyReLU as a
    // new neuron type. In practice it behaves very similarly to ReLU (α=0.01),
    // and production runs show many low-quality LeakyReLU candidates. We still
    // fully support LeakyReLU in existing creatures (targets and sources).
    //
    // NOTE (Issue #148, 26-Dec-2025): We also do not propose Swish or SELU as new
    // neuron squashes. In our value-domain discovery workflow they are close enough
    // to ReLU in practice that scanning them is usually a poor trade in time-bounded
    // runs. This preserves the budget to scan more (source,target) possibilities
    // while still allowing existing creatures to use any squash.
    ActivationCandidateSpec {
        name: "Mish",
        orientations: &ORIENTATIONS_BIDIRECTIONAL,
        scales: &SCALES_SMOOTH,
        activation: mish_activation,
        min_improvement: 0.0, // 2 successful discoveries evolved TO Mish
    },
    ActivationCandidateSpec {
        name: "HARD_TANH",
        orientations: &ORIENTATIONS_BIDIRECTIONAL,
        scales: &SCALES_SMOOTH,
        activation: hard_tanh_activation,
        min_improvement: 0.0, // 1 successful discovery evolved CLIPPED → HARD_TANH
    },
    ActivationCandidateSpec {
        name: "SOFTSIGN",
        orientations: &ORIENTATIONS_BIDIRECTIONAL,
        scales: &SCALES_SMOOTH,
        activation: softsign_activation,
        min_improvement: 0.0, // Successful discovery neuron with SOFTSIGN
    },
    ActivationCandidateSpec {
        name: "BENT_IDENTITY",
        orientations: &ORIENTATIONS_BIDIRECTIONAL,
        scales: &SCALES_WIDE,
        activation: bent_identity_activation,
        min_improvement: 0.0, // 1 successful discovery evolved LeakyReLU → BENT_IDENTITY
    },
    ActivationCandidateSpec {
        name: "ArcTan",
        orientations: &ORIENTATIONS_BIDIRECTIONAL,
        scales: &SCALES_SMOOTH,
        activation: arctan_activation,
        min_improvement: 0.0, // Similar to SOFTSIGN, bounded output
    },
    ActivationCandidateSpec {
        name: "ReLU6",
        orientations: &ORIENTATIONS_BIDIRECTIONAL,
        scales: &SCALES_WIDE,
        activation: relu6_activation,
        min_improvement: 0.0, // Capped ReLU, useful for bounded outputs
    },
];

// ============================================================================
// GPU ID Mapping
// ============================================================================

/// Maps activation function names to GPU shader IDs.
///
/// These IDs must match the activation function implementations in the GPU shaders
/// (activation.wgsl and bias.wgsl).
pub fn activation_name_to_gpu_id(name: &str) -> u32 {
    match name {
        "GELU" => 0,
        "ELU" => 1,
        "SELU" => 2,
        "Softplus" => 3,
        "LOGISTIC" => 4,
        "TANH" => 5,
        "IDENTITY" => 6,
        "BIPOLAR" => 7,
        "CLIPPED" => 8,
        "ABSOLUTE" => 9,
        "INVERSE" => 10,
        // New activations (v0.1.139) - GPU IDs 11-18
        "LeakyReLU" => 11,
        "Mish" => 12,
        "Swish" => 13,
        "HARD_TANH" => 14,
        "SOFTSIGN" => 15,
        "BENT_IDENTITY" => 16,
        "ArcTan" => 17,
        "ReLU6" => 18,
        _ => 6, // Default to IDENTITY
    }
}

// ============================================================================
// Bias Range Helpers
// ============================================================================

/// Get activation-function-specific bias range (min, max, step).
///
/// This is used by the GPU bias search for compatibility.
/// Extended ranges to work with large incoming weights (up to 200).
///
/// Different activation functions benefit from different bias ranges:
/// - ReLU/ELU: Large negative bias for high-threshold neurons
/// - TANH/LOGISTIC: Wide symmetric range to shift operating point
/// - IDENTITY: Widest range as pure offset (scales with large weights)
pub fn get_bias_range(squash: &str) -> (f32, f32, f32) {
    match squash {
        // Sensible default ranges (Dec 2025):
        // We intentionally avoid very large bias grids (e.g. ±25, ±50) because they
        // frequently yield brittle candidates that fail full rescoring.
        "BIPOLAR" => (-10.0, 10.0, 1.0),
        _ => (-10.0, 10.0, 0.5), // Generous default (but still bounded)
    }
}

/// Get log-spaced bias values for a given activation function.
///
/// Uses sinh-like spacing: denser near 0, sparser at extremes.
/// Extended ranges to work with large incoming weights (up to 200).
/// Evolution will fine-tune the exact bias value after discovery.
pub fn get_bias_values(squash: &str) -> Vec<f32> {
    // Base log-spaced positive values (denser near 0, extended to larger values)
    let base_positive: &[f32] = match squash {
        // Keep within sensible ranges (Dec 2025): avoid large offsets.
        "IDENTITY" => &[0.0, 0.5, 1.0, 2.0, 5.0, 10.0],
        _ => &[0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    };

    let base_negative: &[f32] = match squash {
        "IDENTITY" => &[-0.5, -1.0, -2.0, -5.0, -10.0],
        _ => &[-0.1, -0.5, -1.0, -2.0, -5.0, -10.0],
    };

    let mut values: Vec<f32> = base_negative.to_vec();
    values.extend_from_slice(base_positive);
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    values
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

/// Returns `true` when the target squash is a **bounded** activation that can
/// saturate, AND all samples have the target data needed for simulation.
///
/// This is more targeted than `get_target_simulation_fn`, which returns `Some`
/// for nearly all activation functions (including unbounded ones like
/// BENT_IDENTITY and IDENTITY).  Weight-search heuristics (Issue #413) should
/// only be applied to genuinely saturating activations.
#[inline]
pub(crate) fn is_saturating_target(samples: &[HelpfulSample], target_squash: Option<&str>) -> bool {
    let Some(squash) = target_squash else {
        return false;
    };
    let upper = squash.to_ascii_uppercase();
    let bounded = matches!(
        upper.as_str(),
        "TANH"
            | "LOGISTIC"
            | "HARD_TANH"
            | "CLIPPED"
            | "BIPOLAR"
            | "BIPOLAR_SIGMOID"
            | "STEP"
            | "SOFTSIGN"
            | "ISRU"
            | "ARCTAN"
            | "RELU6"
    );
    bounded
        && get_target_activation_fn(squash).is_some()
        && samples
            .iter()
            .all(|s| s.target_value.is_some() && s.target_activation.is_some())
}

/// Legacy function for backwards compatibility - returns true only for HARD_TANH (or its alias CLIPPED).
/// Deprecated: Use get_target_simulation_fn instead for more accurate simulation
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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gelu_activation() {
        // GELU(0) ≈ 0
        assert!((gelu_activation(0.0) - 0.0).abs() < 1e-6);
        // GELU(x) > 0 for x > 0
        assert!(gelu_activation(1.0) > 0.0);
        // GELU(-1) is small negative
        assert!(gelu_activation(-1.0) < 0.0);
    }

    #[test]
    fn test_elu_activation() {
        assert_eq!(elu_activation(0.0), 0.0);
        assert_eq!(elu_activation(1.0), 1.0);
        assert!(elu_activation(-1.0) < 0.0);
        assert!(elu_activation(-1.0) > -1.0); // ELU asymptotes to -1
    }

    #[test]
    fn test_softplus_activation() {
        assert!(softplus_activation(0.0) > 0.0);
        // Softplus(x) ≈ x for large x
        assert!((softplus_activation(30.0) - 30.0).abs() < 0.01);
    }

    #[test]
    fn test_logistic_activation() {
        assert!((logistic_activation(0.0) - 0.5).abs() < 1e-6);
        assert!(logistic_activation(10.0) > 0.99);
        assert!(logistic_activation(-10.0) < 0.01);
    }

    #[test]
    fn test_tanh_activation() {
        assert_eq!(tanh_activation(0.0), 0.0);
        assert!(tanh_activation(3.0) > 0.99);
        assert!(tanh_activation(-3.0) < -0.99);
    }

    #[test]
    fn test_identity_activation() {
        assert_eq!(identity_activation(5.0), 5.0);
        assert_eq!(identity_activation(-3.0), -3.0);
    }

    #[test]
    fn test_bipolar_activation() {
        assert_eq!(bipolar_activation(0.001), 1.0);
        assert_eq!(bipolar_activation(0.0), -1.0);
        assert_eq!(bipolar_activation(-0.001), -1.0);
    }

    #[test]
    fn test_clipped_activation() {
        assert_eq!(clipped_activation(0.5), 0.5);
        assert_eq!(clipped_activation(2.0), 1.0);
        assert_eq!(clipped_activation(-2.0), -1.0);
    }

    #[test]
    fn test_absolute_activation() {
        assert_eq!(absolute_activation(5.0), 5.0);
        assert_eq!(absolute_activation(-5.0), 5.0);
    }

    #[test]
    fn test_mish_activation() {
        assert!((mish_activation(0.0) - 0.0).abs() < 1e-6);
        assert!(mish_activation(1.0) > 0.0);
    }

    #[test]
    fn test_hard_tanh_activation() {
        assert_eq!(hard_tanh_activation(0.5), 0.5);
        assert_eq!(hard_tanh_activation(2.0), 1.0);
        assert_eq!(hard_tanh_activation(-2.0), -1.0);
    }

    #[test]
    fn test_softsign_activation() {
        assert_eq!(softsign_activation(0.0), 0.0);
        assert!(softsign_activation(1.0) > 0.4);
        assert!(softsign_activation(1.0) < 0.6);
    }

    #[test]
    fn test_bent_identity_activation() {
        assert!((bent_identity_activation(0.0) - 0.0).abs() < 1e-6);
        // Bent identity is nearly linear for small x
        assert!((bent_identity_activation(0.1) - 0.1).abs() < 0.1);
    }

    #[test]
    fn test_arctan_activation() {
        assert_eq!(arctan_activation(0.0), 0.0);
        assert!(arctan_activation(1.0) > 0.7);
        assert!(arctan_activation(1.0) < 0.8);
    }

    #[test]
    fn test_relu6_activation() {
        assert_eq!(relu6_activation(-1.0), 0.0);
        assert_eq!(relu6_activation(3.0), 3.0);
        assert_eq!(relu6_activation(10.0), 6.0);
    }

    #[test]
    fn test_activation_name_to_gpu_id() {
        assert_eq!(activation_name_to_gpu_id("GELU"), 0);
        assert_eq!(activation_name_to_gpu_id("ELU"), 1);
        assert_eq!(activation_name_to_gpu_id("IDENTITY"), 6);
        assert_eq!(activation_name_to_gpu_id("LeakyReLU"), 11);
        assert_eq!(activation_name_to_gpu_id("Mish"), 12);
        assert_eq!(activation_name_to_gpu_id("ReLU6"), 18);
        // Unknown defaults to IDENTITY
        assert_eq!(activation_name_to_gpu_id("UNKNOWN"), 6);
    }

    #[test]
    fn test_get_bias_range() {
        let (min, max, step) = get_bias_range("BIPOLAR");
        assert_eq!(min, -10.0);
        assert_eq!(max, 10.0);
        assert_eq!(step, 1.0);

        let (min, max, step) = get_bias_range("TANH");
        assert_eq!(min, -10.0);
        assert_eq!(max, 10.0);
        assert_eq!(step, 0.5);
    }

    #[test]
    fn test_get_bias_values() {
        let values = get_bias_values("TANH");
        assert!(!values.is_empty());
        assert!(values.contains(&0.0));
        // Check values are sorted
        for i in 1..values.len() {
            assert!(values[i] >= values[i - 1]);
        }
    }

    #[test]
    fn test_is_threshold_activation() {
        assert!(is_threshold_activation("STEP"));
        assert!(is_threshold_activation("step"));
        assert!(is_threshold_activation("BIPOLAR"));
        assert!(is_threshold_activation("bipolar"));
        assert!(!is_threshold_activation("TANH"));
        assert!(!is_threshold_activation("RELU"));
        assert!(!is_threshold_activation("IDENTITY"));
    }

    /// Test that `is_threshold_activation` uses zero-allocation case-insensitive comparison.
    /// Issue #211: Uses `eq_ignore_ascii_case` instead of `.to_uppercase()` to avoid
    /// string allocations in hot loops.
    #[test]
    fn test_is_threshold_activation_case_variations() {
        // All case variations should work without allocation
        assert!(is_threshold_activation("STEP"));
        assert!(is_threshold_activation("Step"));
        assert!(is_threshold_activation("step"));
        assert!(is_threshold_activation("sTeP"));
        assert!(is_threshold_activation("BIPOLAR"));
        assert!(is_threshold_activation("Bipolar"));
        assert!(is_threshold_activation("bipolar"));
        assert!(is_threshold_activation("BiPoLaR"));

        // Non-threshold activations with various cases
        assert!(!is_threshold_activation("TANH"));
        assert!(!is_threshold_activation("tanh"));
        assert!(!is_threshold_activation("Tanh"));
        assert!(!is_threshold_activation("RELU"));
        assert!(!is_threshold_activation("relu"));
        assert!(!is_threshold_activation("ReLU"));
        assert!(!is_threshold_activation("IDENTITY"));
        assert!(!is_threshold_activation("identity"));

        // Edge cases
        assert!(!is_threshold_activation(""));
        assert!(!is_threshold_activation("STEP2")); // Not exact match
        assert!(!is_threshold_activation("STEPBIPOLAR")); // Not exact match
    }

    #[test]
    fn test_activation_specs_count() {
        // Verify we have exactly 15 activation specs as documented
        assert_eq!(ACTIVATION_SPECS.len(), 15);
    }

    #[test]
    fn test_activation_specs_have_valid_functions() {
        // Each spec should have a working activation function
        for spec in &ACTIVATION_SPECS {
            // Test that the function works for a typical input
            let result = (spec.activation)(0.5);
            assert!(
                result.is_finite(),
                "{} returned non-finite for 0.5",
                spec.name
            );
        }
    }

    // ========================================================================
    // Target Simulation Function Tests
    // ========================================================================

    #[test]
    fn test_get_target_activation_fn() {
        // Known activations should return Some
        assert!(get_target_activation_fn("TANH").is_some());
        assert!(get_target_activation_fn("HARD_TANH").is_some());
        assert!(get_target_activation_fn("CLIPPED").is_some());
        assert!(get_target_activation_fn("LOGISTIC").is_some());
        assert!(get_target_activation_fn("IDENTITY").is_some());

        // Case insensitive
        assert!(get_target_activation_fn("tanh").is_some());
        assert!(get_target_activation_fn("Tanh").is_some());

        // Aggregate squashes should return None
        assert!(get_target_activation_fn("MINIMUM").is_none());
        assert!(get_target_activation_fn("MAXIMUM").is_none());
        assert!(get_target_activation_fn("IF").is_none());
    }

    #[test]
    fn test_get_target_simulation_fn_with_complete_samples() {
        let samples = vec![
            HelpfulSample {
                activation: 0.5,
                avg_error: 0.1,
                target_value: Some(0.3),
                target_activation: Some(0.29),
            },
            HelpfulSample {
                activation: -0.2,
                avg_error: -0.05,
                target_value: Some(-0.1),
                target_activation: Some(-0.099),
            },
        ];

        // Should return activation function when all samples have target data
        let result = get_target_simulation_fn(&samples, Some("TANH"));
        assert!(result.is_some());
    }

    #[test]
    fn test_get_target_simulation_fn_with_incomplete_samples() {
        let samples = vec![
            HelpfulSample {
                activation: 0.5,
                avg_error: 0.1,
                target_value: Some(0.3),
                target_activation: Some(0.29),
            },
            HelpfulSample {
                activation: -0.2,
                avg_error: -0.05,
                target_value: None, // Missing target_value
                target_activation: Some(-0.099),
            },
        ];

        // Should return None when some samples are missing target data
        let result = get_target_simulation_fn(&samples, Some("TANH"));
        assert!(result.is_none());
    }

    #[test]
    fn test_get_target_simulation_mode_none_squash() {
        let samples = vec![HelpfulSample {
            activation: 0.5,
            avg_error: 0.1,
            target_value: Some(0.3),
            target_activation: Some(0.29),
        }];

        let mode = get_target_simulation_mode(&samples, None);
        assert!(matches!(mode, TargetSimulationMode::None));
    }

    #[test]
    fn test_get_target_simulation_mode_full() {
        let samples = vec![
            HelpfulSample {
                activation: 0.5,
                avg_error: 0.1,
                target_value: Some(0.3),
                target_activation: Some(0.29),
            },
            HelpfulSample {
                activation: -0.2,
                avg_error: -0.05,
                target_value: Some(-0.1),
                target_activation: Some(-0.099),
            },
        ];

        let mode = get_target_simulation_mode(&samples, Some("HARD_TANH"));
        assert!(matches!(mode, TargetSimulationMode::Full(_)));
    }

    #[test]
    fn test_get_target_simulation_mode_approximate() {
        let samples = vec![
            HelpfulSample {
                activation: 0.5,
                avg_error: 0.1,
                target_value: None, // No target_value
                target_activation: Some(0.29),
            },
            HelpfulSample {
                activation: -0.2,
                avg_error: -0.05,
                target_value: None, // No target_value
                target_activation: Some(-0.099),
            },
        ];

        // HARD_TANH should use approximation mode when target_value is missing
        let mode = get_target_simulation_mode(&samples, Some("HARD_TANH"));
        assert!(matches!(
            mode,
            TargetSimulationMode::ApproximateValueFromActivation(_)
        ));

        // CLIPPED (alias for HARD_TANH) should also work
        let mode = get_target_simulation_mode(&samples, Some("CLIPPED"));
        assert!(matches!(
            mode,
            TargetSimulationMode::ApproximateValueFromActivation(_)
        ));

        // Other activations should fall back to None
        let mode = get_target_simulation_mode(&samples, Some("TANH"));
        assert!(matches!(mode, TargetSimulationMode::None));
    }

    #[test]
    fn test_can_use_hard_tanh() {
        let complete_samples = vec![
            HelpfulSample {
                activation: 0.5,
                avg_error: 0.1,
                target_value: Some(0.3),
                target_activation: Some(0.29),
            },
            HelpfulSample {
                activation: -0.2,
                avg_error: -0.05,
                target_value: Some(-0.1),
                target_activation: Some(-0.099),
            },
        ];

        let incomplete_samples = vec![HelpfulSample {
            activation: 0.5,
            avg_error: 0.1,
            target_value: None,
            target_activation: Some(0.29),
        }];

        // Should return true for HARD_TANH with complete samples
        assert!(can_use_hard_tanh(&complete_samples, Some("HARD_TANH")));
        assert!(can_use_hard_tanh(&complete_samples, Some("hard_tanh")));
        assert!(can_use_hard_tanh(&complete_samples, Some("CLIPPED")));

        // Should return false for other activations
        assert!(!can_use_hard_tanh(&complete_samples, Some("TANH")));
        assert!(!can_use_hard_tanh(&complete_samples, None));

        // Should return false for incomplete samples
        assert!(!can_use_hard_tanh(&incomplete_samples, Some("HARD_TANH")));
    }

    #[test]
    fn test_has_sufficient_output_variance_with_variance() {
        // Samples with varying input - output should also vary
        let samples = vec![
            HelpfulSample {
                activation: -1.0,
                avg_error: 0.1,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.0,
                avg_error: 0.1,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 1.0,
                avg_error: 0.1,
                target_value: None,
                target_activation: None,
            },
        ];

        // Identity activation should show variance
        assert!(has_sufficient_output_variance(
            &samples,
            1.0,
            0.0,
            identity_activation
        ));

        // TANH with reasonable weight should show variance
        assert!(has_sufficient_output_variance(
            &samples,
            1.0,
            0.0,
            tanh_activation
        ));
    }

    #[test]
    fn test_has_sufficient_output_variance_saturated() {
        // Samples with varying input
        let samples = vec![
            HelpfulSample {
                activation: -1.0,
                avg_error: 0.1,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.0,
                avg_error: 0.1,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 1.0,
                avg_error: 0.1,
                target_value: None,
                target_activation: None,
            },
        ];

        // TANH with large bias should be saturated (output ≈ 1.0 for all inputs)
        assert!(!has_sufficient_output_variance(
            &samples,
            1.0,
            10.0,
            tanh_activation
        ));
    }

    #[test]
    fn test_has_sufficient_output_variance_constant_input() {
        // Samples with constant input
        let samples = vec![
            HelpfulSample {
                activation: 0.5,
                avg_error: 0.1,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.5,
                avg_error: 0.2,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.5,
                avg_error: 0.3,
                target_value: None,
                target_activation: None,
            },
        ];

        // Constant input should be allowed (output variance check is skipped)
        assert!(has_sufficient_output_variance(
            &samples,
            1.0,
            0.0,
            identity_activation
        ));
    }

    #[test]
    fn test_has_sufficient_output_variance_insufficient_samples() {
        let single_sample = vec![HelpfulSample {
            activation: 0.5,
            avg_error: 0.1,
            target_value: None,
            target_activation: None,
        }];

        let empty_samples: Vec<HelpfulSample> = vec![];

        // Should return false for insufficient samples
        assert!(!has_sufficient_output_variance(
            &single_sample,
            1.0,
            0.0,
            identity_activation
        ));
        assert!(!has_sufficient_output_variance(
            &empty_samples,
            1.0,
            0.0,
            identity_activation
        ));
    }
}
