//! Activation candidate specifications and GPU ID mapping.
//!
//! Defines which activation functions are considered when searching for new
//! neuron candidates, along with GPU shader ID mappings and bias range helpers.

use super::functions::*;
use crate::analysis::task_descriptor::{TargetTopology, TaskDescriptor};

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
    values.sort_by(f32::total_cmp);
    values
}

// ============================================================================
// Role- and Task-aware Scan Candidate Set (Issue #1315)
// ============================================================================

/// Role of the neuron being scanned for an activation candidate.
///
/// The activation/squash scan offers different candidate sets depending on
/// whether we are searching for a replacement squash on an **output** neuron
/// — whose activation shape is constrained by the loss — or on a **hidden**
/// neuron — where any squash is fair game.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeuronRole {
    /// Output (final-layer) neuron. The activation family is constrained by
    /// the cost function via the [`TaskDescriptor`].
    Output,
    /// Hidden (non-output) neuron. The full unbounded scan set applies.
    Hidden,
}

/// Names of the bounded-squash candidates offered when scanning an **output**
/// neuron under a `OneHot` or `Simplex` target topology.
///
/// These activations all produce outputs in `[0, 1]` (or a step-function
/// surrogate), which is the family compatible with `CROSS_ENTROPY` /
/// `CATEGORICAL_ERROR` losses (see [`TaskDescriptor::from_name`]). Names not
/// present in [`ACTIVATION_SPECS`] are silently skipped — at the time of
/// writing `STEP` is recognised by NEAT-AI as a valid squash but does not
/// yet appear in the scan specs, so the effective set is `LOGISTIC` and
/// `BIPOLAR`. If `STEP` is added to [`ACTIVATION_SPECS`] in future it will
/// be picked up automatically.
pub const BOUNDED_OUTPUT_SCAN_NAMES: &[&str] = &["LOGISTIC", "STEP", "BIPOLAR"];

/// Select the activation candidate specs to scan for the given neuron role
/// and task descriptor.
///
/// Behaviour (Issue #1315):
///
/// - **Output neuron** under `OneHot` or `Simplex` topology — returns the
///   bounded squash subset (`LOGISTIC`, `STEP`, `BIPOLAR`; see
///   [`BOUNDED_OUTPUT_SCAN_NAMES`]).
/// - **All other cases** — output neurons under `Independent` / `Margin` /
///   `Unknown` topology, and *every* hidden neuron — return the full
///   [`ACTIVATION_SPECS`]. This includes the `OTHER` / unknown / absent
///   descriptor (which collapses to [`TaskDescriptor::neutral`]) so callers
///   not yet plumbed for task-shape awareness keep their existing scan.
#[must_use]
pub fn scan_specs_for_role(
    role: NeuronRole,
    descriptor: &TaskDescriptor,
) -> Vec<&'static ActivationCandidateSpec> {
    let use_bounded_subset = matches!(role, NeuronRole::Output)
        && matches!(
            descriptor.target_topology,
            TargetTopology::OneHot | TargetTopology::Simplex
        );

    if use_bounded_subset {
        ACTIVATION_SPECS
            .iter()
            .filter(|spec| BOUNDED_OUTPUT_SCAN_NAMES.contains(&spec.name))
            .collect()
    } else {
        ACTIVATION_SPECS.iter().collect()
    }
}

#[cfg(test)]
mod scan_specs_tests {
    use super::*;
    use crate::analysis::task_descriptor::TaskDescriptor;

    fn names(specs: &[&'static ActivationCandidateSpec]) -> Vec<&'static str> {
        specs.iter().map(|s| s.name).collect()
    }

    #[test]
    fn onehot_output_returns_only_bounded_subset() {
        let d = TaskDescriptor::from_name("CATEGORICAL_ERROR", 7);
        let specs = scan_specs_for_role(NeuronRole::Output, &d);
        let names = names(&specs);

        assert!(
            names.contains(&"LOGISTIC"),
            "OneHot output scan must include LOGISTIC, got {names:?}",
        );
        assert!(
            names.contains(&"BIPOLAR"),
            "OneHot output scan must include BIPOLAR, got {names:?}",
        );
        // Unbounded scans must be excluded for OneHot/Simplex output.
        for unbounded in ["GELU", "ELU", "IDENTITY", "Mish", "Softplus"] {
            assert!(
                !names.contains(&unbounded),
                "OneHot output scan must exclude {unbounded}, got {names:?}",
            );
        }
    }

    #[test]
    fn simplex_output_returns_only_bounded_subset() {
        let d = TaskDescriptor::from_name("CROSS_ENTROPY", 10);
        let specs = scan_specs_for_role(NeuronRole::Output, &d);
        let names = names(&specs);

        assert!(names.contains(&"LOGISTIC"));
        assert!(names.contains(&"BIPOLAR"));
        for unbounded in ["GELU", "ELU", "IDENTITY"] {
            assert!(!names.contains(&unbounded));
        }
    }

    #[test]
    fn onehot_hidden_uses_full_scan_set() {
        let d = TaskDescriptor::from_name("CATEGORICAL_ERROR", 7);
        let specs = scan_specs_for_role(NeuronRole::Hidden, &d);
        assert_eq!(
            specs.len(),
            ACTIVATION_SPECS.len(),
            "Hidden neurons always see the full scan set",
        );
    }

    #[test]
    fn simplex_hidden_uses_full_scan_set() {
        let d = TaskDescriptor::from_name("CROSS_ENTROPY", 10);
        let specs = scan_specs_for_role(NeuronRole::Hidden, &d);
        assert_eq!(specs.len(), ACTIVATION_SPECS.len());
    }

    #[test]
    fn unknown_descriptor_returns_full_scan_for_both_roles() {
        // Regression guard: absent / unrecognised cost name collapses to
        // neutral, which must preserve the pre-Issue-#1315 scan set.
        let d = TaskDescriptor::neutral();
        for role in [NeuronRole::Output, NeuronRole::Hidden] {
            let specs = scan_specs_for_role(role, &d);
            assert_eq!(
                specs.len(),
                ACTIVATION_SPECS.len(),
                "Neutral descriptor must yield the full scan set for {role:?}",
            );
        }
    }

    #[test]
    fn other_cost_name_returns_full_scan_for_both_roles() {
        // OTHER is the explicit "I don't know" cost name. Must behave
        // identically to neutral().
        let d = TaskDescriptor::from_name("OTHER", 4);
        for role in [NeuronRole::Output, NeuronRole::Hidden] {
            let specs = scan_specs_for_role(role, &d);
            assert_eq!(specs.len(), ACTIVATION_SPECS.len());
        }
    }

    #[test]
    fn unrecognised_cost_name_returns_full_scan() {
        let d = TaskDescriptor::from_name("EXOTIC_LOSS", 2);
        let specs = scan_specs_for_role(NeuronRole::Output, &d);
        assert_eq!(specs.len(), ACTIVATION_SPECS.len());
    }

    #[test]
    fn independent_output_uses_full_scan() {
        // MSE / MAE / MAPE / MSLE / BCE / HINGE — none of these gate the
        // scan to bounded only (only OneHot and Simplex do).
        for cost in [
            "MSE",
            "MAE",
            "MAPE",
            "MSLE",
            "BINARY_CROSS_ENTROPY",
            "HINGE",
        ] {
            let d = TaskDescriptor::from_name(cost, 3);
            let specs = scan_specs_for_role(NeuronRole::Output, &d);
            assert_eq!(
                specs.len(),
                ACTIVATION_SPECS.len(),
                "Cost {cost} must not gate the output scan to bounded only",
            );
        }
    }

    #[test]
    fn bounded_subset_only_contains_known_spec_names() {
        // Defensive: every name we report in the bounded subset must
        // actually exist in ACTIVATION_SPECS so callers don't get a stale
        // / mis-spelled label. STEP is intentionally allowed to be missing
        // from ACTIVATION_SPECS (it is listed for forward compatibility).
        let d = TaskDescriptor::from_name("CATEGORICAL_ERROR", 7);
        let specs = scan_specs_for_role(NeuronRole::Output, &d);
        for spec in &specs {
            assert!(
                BOUNDED_OUTPUT_SCAN_NAMES.contains(&spec.name),
                "Spec {} returned from bounded scan must be in BOUNDED_OUTPUT_SCAN_NAMES",
                spec.name,
            );
        }
        // Sanity: at least LOGISTIC and BIPOLAR must be present.
        let names = names(&specs);
        assert!(names.contains(&"LOGISTIC"));
        assert!(names.contains(&"BIPOLAR"));
    }
}
