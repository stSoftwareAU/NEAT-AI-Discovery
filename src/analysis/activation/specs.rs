//! Activation candidate specifications and GPU ID mapping.
//!
//! Defines which activation functions are considered when searching for new
//! neuron candidates, along with GPU shader ID mappings and bias range helpers.

use super::functions::*;
use crate::analysis::task_descriptor::{TargetTopology, TaskDescriptor};
use std::collections::HashSet;

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

// ============================================================================
// Squash-aware Hidden-target Scan Pruning (Issue #1545)
// ============================================================================

/// Core hidden-neuron squash scan set used at cold start (Issue #1545).
///
/// This is an **evidence-based** core, not an arbitrary shortlist. It combines:
///
/// - `IDENTITY` (linear), the `ReLU` family (`ELU`, `ReLU6`), and the smooth
///   workhorses `TANH` / `GELU` — the issue's cold-start core (`ReLU` itself is
///   scanned separately via the `ReLU` split path, so its family is represented
///   by the smooth surrogates `ELU`, `ReLU6`, `GELU`), plus
/// - the v0.1.139 families the [`ACTIVATION_SPECS`] comments document as having
///   produced **accepted** add-neuron discoveries — `Mish` (2 discoveries),
///   `HARD_TANH`, `SOFTSIGN`, and `BENT_IDENTITY` — i.e. squashes with non-zero
///   historical success (criterion #1 of the issue).
///
/// The families left out of the core (`Softplus`, `LOGISTIC`, `BIPOLAR`,
/// `CLIPPED`, `ABSOLUTE`, `ArcTan`) are either output-oriented bounded squashes
/// already covered by [`BOUNDED_OUTPUT_SCAN_NAMES`] or near-duplicates of a core
/// member (e.g. `ArcTan` ≈ `SOFTSIGN`), so pruning them from cold-start hidden
/// scans drops ~40 % of the activation GPU configs (256 → 154) without removing
/// a documented winner.
///
/// A creature with **no** recorded squash history scans exactly this set for
/// hidden targets. Families outside the core set are only scanned once the
/// creature has demonstrably adopted them (history-aware widening) or while the
/// search is escalated (novelty / drought — see [`HiddenScanContext::escalated`]).
pub const CORE_HIDDEN_SCAN_NAMES: &[&str] = &[
    "IDENTITY",
    "GELU",
    "ELU",
    "ReLU6",
    "TANH",
    "Mish",
    "HARD_TANH",
    "SOFTSIGN",
    "BENT_IDENTITY",
];

/// Per-creature history and escalation signal that drives hidden-target squash
/// pruning (Issue #1545).
///
/// The scan set for a hidden add-neuron target is derived from three inputs:
/// the fixed [`CORE_HIDDEN_SCAN_NAMES`] core, the squash families the creature
/// has already adopted (`successful_squashes` — non-zero historical success),
/// and whether the search is currently escalated.
pub struct HiddenScanContext<'a> {
    /// Squash family names with non-zero historical success for this creature.
    ///
    /// In production this is the set of squash functions the creature's neurons
    /// already use: a family the creature has adopted has demonstrably worked
    /// for it, so it is worth re-scanning. Empty at cold start.
    pub successful_squashes: &'a HashSet<String>,
    /// When `true`, novelty escalation (#1423) or a drought restores the full
    /// [`ACTIVATION_SPECS`] set so pruning can **never** permanently starve the
    /// search. This is the critical guard against a filter that only helps toy
    /// networks — a plateaued creature widens back to the full set.
    pub escalated: bool,
}

impl HiddenScanContext<'_> {
    /// Select the activation candidate specs to scan for a **hidden**
    /// add-neuron target (Issue #1545).
    ///
    /// - **Escalated** — returns the full [`ACTIVATION_SPECS`] set (widen under
    ///   novelty escalation / drought).
    /// - **Otherwise** — returns [`CORE_HIDDEN_SCAN_NAMES`] widened with every
    ///   family in [`Self::successful_squashes`]. Families that are neither core
    ///   nor historically successful are pruned.
    ///
    /// The result is never empty and never over-prunes below the core set.
    #[must_use]
    pub fn scan_specs(&self) -> Vec<&'static ActivationCandidateSpec> {
        if self.escalated {
            return ACTIVATION_SPECS.iter().collect();
        }
        ACTIVATION_SPECS
            .iter()
            .filter(|spec| {
                CORE_HIDDEN_SCAN_NAMES.contains(&spec.name)
                    // Case-insensitive: creature squashes are uppercased at
                    // deserialisation (Issue #753) while some spec names are
                    // mixed-case (e.g. `Mish`, `ArcTan`, `ReLU6`).
                    || self
                        .successful_squashes
                        .iter()
                        .any(|s| s.eq_ignore_ascii_case(spec.name))
            })
            .collect()
    }
}

/// A concrete GPU scan configuration: which spec (by index into the plan's spec
/// slice) plus the activation type / orientation / scale to evaluate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScanConfig {
    /// Index into the [`SquashScanPlan::specs`] slice this config belongs to.
    pub spec_index: usize,
    /// GPU activation-type id (see [`activation_name_to_gpu_id`]).
    pub activation_type: u32,
    /// Weight orientation (`+1.0` / `-1.0`).
    pub orientation: f32,
    /// Incoming-weight scale.
    pub scale: f32,
}

/// A resolved hidden-target activation scan plan (Issue #1545).
///
/// Combines the pruned squash-family subset with a per-(source, target) config
/// cap. Produced once per neuron-analysis phase and reused for every
/// (source, target) pair, replacing the previous unconditional full-cross-product
/// scan.
pub struct SquashScanPlan {
    /// Squash families to scan (a subset of [`ACTIVATION_SPECS`]).
    pub specs: Vec<&'static ActivationCandidateSpec>,
    /// Maximum number of (orientation × scale) configs per (source, target)
    /// pair after family filtering. `0` disables the cap.
    pub max_configs_per_pair: usize,
}

impl SquashScanPlan {
    /// Build a hidden-target plan from a scan context and a per-pair config cap.
    #[must_use]
    pub fn for_hidden(ctx: &HiddenScanContext, max_configs_per_pair: usize) -> Self {
        Self {
            specs: ctx.scan_specs(),
            max_configs_per_pair,
        }
    }

    /// The full, unpruned, uncapped plan — the pre-Issue-#1545 behaviour.
    ///
    /// Used as the regression-safe default when no per-creature history is
    /// available (e.g. the batched-evaluation unit tests and the sequential
    /// fallback path).
    #[must_use]
    pub fn full() -> Self {
        Self {
            specs: ACTIVATION_SPECS.iter().collect(),
            max_configs_per_pair: 0,
        }
    }

    /// Expand the plan into concrete GPU configs, honouring the per-pair cap.
    ///
    /// With no cap active the configs are returned in natural spec-major order
    /// (spec, then orientation, then scale) — identical to the pre-#1545
    /// ordering. When the cap trims the list, the retained configs are the ones
    /// whose scale is closest to `1.0` (the historically most productive band),
    /// so extreme scales — which the spec comments flag as numerically unstable
    /// — are dropped first.
    #[must_use]
    pub fn configs(&self) -> Vec<ScanConfig> {
        build_scan_configs(&self.specs, self.max_configs_per_pair)
    }
}

/// Expand a spec slice into concrete GPU configs, applying a top-N cap.
///
/// See [`SquashScanPlan::configs`] for the ordering and cap semantics.
#[must_use]
pub fn build_scan_configs(
    specs: &[&'static ActivationCandidateSpec],
    max_configs: usize,
) -> Vec<ScanConfig> {
    let mut configs: Vec<ScanConfig> = Vec::new();
    for (spec_index, spec) in specs.iter().enumerate() {
        let activation_type = activation_name_to_gpu_id(spec.name);
        for &orientation in spec.orientations {
            for &scale in spec.scales {
                configs.push(ScanConfig {
                    spec_index,
                    activation_type,
                    orientation,
                    scale,
                });
            }
        }
    }

    if max_configs > 0 && configs.len() > max_configs {
        // Keep the `max_configs` configs whose scale is closest to 1.0. Ties
        // (and the uncapped remainder) preserve natural spec-major order via the
        // stable sort, so the output is deterministic.
        configs.sort_by(|a, b| {
            scale_centrality(a.scale)
                .partial_cmp(&scale_centrality(b.scale))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        configs.truncate(max_configs);
    }

    configs
}

/// Distance of a scale from the productive centre (`1.0`) in log space.
///
/// Smaller is more central. Used to decide which configs survive the top-N cap.
fn scale_centrality(scale: f32) -> f32 {
    if scale <= 0.0 {
        return f32::INFINITY;
    }
    scale.ln().abs()
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

    // Issue #1545 rewrote the three hidden-target cases below. Before #1545 a
    // hidden target unconditionally scanned the full `ACTIVATION_SPECS` set;
    // hidden scans are now squash-aware via `HiddenScanContext`. The output-role
    // cases (`onehot_output_returns_only_bounded_subset` etc.) are untouched.

    #[test]
    fn onehot_hidden_cold_start_uses_core_set() {
        // Cold start (no recorded squash history): a hidden target scans only
        // the reduced core set, not the full ACTIVATION_SPECS list.
        let history = HashSet::new();
        let ctx = HiddenScanContext {
            successful_squashes: &history,
            escalated: false,
        };
        let specs = ctx.scan_specs();
        let got = names(&specs);
        assert_eq!(
            got.len(),
            CORE_HIDDEN_SCAN_NAMES.len(),
            "cold-start hidden scan must equal the core set, got {got:?}",
        );
        for core in CORE_HIDDEN_SCAN_NAMES {
            assert!(
                got.contains(core),
                "core member {core} missing from {got:?}"
            );
        }
        assert!(specs.len() < ACTIVATION_SPECS.len());
    }

    #[test]
    fn simplex_hidden_escalated_restores_full_scan_set() {
        // Escalation (novelty / drought) widens a hidden target back to the
        // full set — the guard against permanent over-pruning.
        let history = HashSet::new();
        let ctx = HiddenScanContext {
            successful_squashes: &history,
            escalated: true,
        };
        let specs = ctx.scan_specs();
        assert_eq!(specs.len(), ACTIVATION_SPECS.len());
    }

    #[test]
    fn hidden_history_widens_but_output_neutral_stays_full() {
        // Hidden: a non-core family with recorded success is scanned in addition
        // to the core set; a non-core family with no history is pruned.
        let mut history = HashSet::new();
        history.insert("BIPOLAR".to_string());
        let ctx = HiddenScanContext {
            successful_squashes: &history,
            escalated: false,
        };
        let got = names(&ctx.scan_specs());
        assert!(
            got.contains(&"BIPOLAR"),
            "adopted family must be scanned: {got:?}"
        );
        assert!(
            !got.contains(&"CLIPPED"),
            "unadopted family must be pruned: {got:?}"
        );

        // Output role via the neutral descriptor is unaffected by #1545 and
        // still returns the full scan set (regression guard).
        let d = TaskDescriptor::neutral();
        let out = scan_specs_for_role(NeuronRole::Output, &d);
        assert_eq!(out.len(), ACTIVATION_SPECS.len());
    }

    #[test]
    fn build_scan_configs_respects_cap() {
        let history = HashSet::new();
        let ctx = HiddenScanContext {
            successful_squashes: &history,
            escalated: true, // full set → largest config count
        };
        let specs = ctx.scan_specs();
        let uncapped = build_scan_configs(&specs, 0);
        assert!(uncapped.len() > 20, "sanity: full set has many configs");

        let cap = 12;
        let capped = build_scan_configs(&specs, cap);
        assert_eq!(capped.len(), cap, "cap must bound the config count");
        // Every retained config must reference a valid spec index.
        for cfg in &capped {
            assert!(cfg.spec_index < specs.len());
        }
    }

    #[test]
    fn plan_full_is_uncapped_full_set() {
        let plan = SquashScanPlan::full();
        assert_eq!(plan.specs.len(), ACTIVATION_SPECS.len());
        assert_eq!(plan.max_configs_per_pair, 0);
        // Uncapped config expansion preserves natural spec-major order.
        let configs = plan.configs();
        assert_eq!(configs.first().map(|c| c.spec_index), Some(0));
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
