//! Activation candidate specifications and GPU ID mapping.
//!
//! Defines which activation functions are considered when searching for new
//! neuron candidates, along with GPU shader ID mappings and bias range helpers.

use super::functions::*;

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
    values.sort_by(|a, b| a.total_cmp(b));
    values
}
