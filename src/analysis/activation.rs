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
#[inline]
pub fn is_threshold_activation(squash: &str) -> bool {
    matches!(squash.to_uppercase().as_str(), "STEP" | "BIPOLAR")
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
}
