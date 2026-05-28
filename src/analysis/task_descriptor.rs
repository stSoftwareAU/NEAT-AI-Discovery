//! Task descriptor for discovery consumers (Issue #1312).
//!
//! A [`TaskDescriptor`] is a small, FFI-free summary of the **shape of the
//! supervised-learning task** a recorded creature was trained against. It is
//! derived purely from the cost-function name (and the network's output
//! count) and carries no per-record data.
//!
//! Detectors and recommenders consult the descriptor to gate behaviour that
//! depends on the structural assumptions of the loss:
//!
//! - **Target topology** — are the outputs independent scalars, a one-hot
//!   class label, a soft-max simplex, or a signed margin?
//! - **Target range** — what numeric range can the recorded target take?
//! - **Output squash family** — what activation shape is compatible with the
//!   loss on the final layer?
//!
//! The mapping is exhaustive for the seven built-in NEAT-AI costs listed in
//! the issue. Any other name — including `OTHER` and any unrecognised /
//! absent string — falls back to [`TaskDescriptor::neutral`], which encodes
//! "we know nothing; don't gate on this".
//!
//! The descriptor itself is intentionally pure (no FFI surface, no I/O) but
//! it exposes a small projection — [`TaskDescriptor::cost_function_hint`] —
//! that maps the task shape onto the [`crate::analysis::cost_function_hint::CostFunctionHint`]
//! used by the implied-target reconstruction guard (issue #1317). Wiring the
//! descriptor through the FFI ingest path and the remaining per-consumer
//! sites is tracked separately (see issue #1314 and the per-consumer issues
//! that reference #1312).

use crate::analysis::cost_function_hint::CostFunctionHint;

/// Topology of the recorded targets vector for a single training sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TargetTopology {
    /// Each output is an independent scalar target (e.g. regression, BCE).
    Independent,
    /// Targets form a one-hot class label (exactly one output is `1`, the
    /// rest are `0`) and the model produces per-class scores.
    OneHot,
    /// Targets sit on the probability simplex; the output layer is expected
    /// to be a soft-max producing values summing to one.
    Simplex,
    /// Margin-based target — used by hinge-style losses where the recorded
    /// target is a signed margin label rather than a probability.
    Margin,
    /// Topology is unknown or not constrained by the cost. Detectors should
    /// fall back to their generic, pre-Issue-#1312 behaviour.
    #[default]
    Unknown,
}

/// Numeric range the recorded targets can take.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TargetRange {
    /// Targets are unbounded reals.
    #[default]
    Unbounded,
    /// Targets are in `[0, 1]`.
    Unit,
    /// Targets are non-negative reals (`>= 0`).
    Positive,
    /// Targets are in `[-1, 1]`.
    SignedUnit,
}

/// Family of activation functions that is compatible with the loss on the
/// final layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputSquashFamily {
    /// Bounded, unipolar squash (range `[0, 1]`): LOGISTIC, `BIPOLAR_SIGMOID`
    /// rescaled to unipolar, etc.
    BoundedUnipolar,
    /// Bounded, bipolar squash (range `[-1, 1]`): TANH, `BIPOLAR_SIGMOID`.
    BoundedBipolar,
    /// Non-negative output (range `[0, +inf)`): RELU, SOFTPLUS, ABSOLUTE.
    Positive,
    /// Unbounded real output: IDENTITY, SELU on large inputs.
    Unbounded,
    /// No constraint imposed by the loss. Detectors should not gate on the
    /// output activation family.
    #[default]
    Any,
}

/// Cost-function-derived description of the supervised-learning task.
///
/// Constructed via [`TaskDescriptor::from_name`] or
/// [`TaskDescriptor::neutral`]. The struct is `Copy` and trivially cheap to
/// pass by value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TaskDescriptor {
    /// Topology of the targets vector.
    pub target_topology: TargetTopology,
    /// Numeric range of recorded targets.
    pub target_range: TargetRange,
    /// Output squash family compatible with the loss.
    pub output_squash_family: OutputSquashFamily,
    /// Number of output neurons in the network. Captured so that consumers
    /// can sanity-check topology assumptions (e.g. `OneHot` requires more
    /// than one output to be meaningful).
    pub num_outputs: usize,
}

impl TaskDescriptor {
    /// Neutral descriptor — used when the cost name is absent, unrecognised,
    /// or explicitly `OTHER`. Encodes "we know nothing; don't gate on this".
    ///
    /// `num_outputs` is set to `0` because the neutral descriptor carries
    /// no information about the network's shape. Call
    /// [`TaskDescriptor::from_name`] when the output count matters.
    #[must_use]
    pub const fn neutral() -> Self {
        Self {
            target_topology: TargetTopology::Unknown,
            target_range: TargetRange::Unbounded,
            output_squash_family: OutputSquashFamily::Any,
            num_outputs: 0,
        }
    }

    /// Map a NEAT-AI cost name (case-insensitive) and the network's output
    /// count onto a [`TaskDescriptor`].
    ///
    /// Recognised names: `MSE`, `MAE`, `MAPE`, `MSLE`,
    /// `BINARY_CROSS_ENTROPY`, `CROSS_ENTROPY`, `HINGE`, `CATEGORICAL_ERROR`.
    /// Any other input — including `OTHER`, empty strings, and arbitrary
    /// unknown names — returns [`TaskDescriptor::neutral`].
    #[must_use]
    pub fn from_name(name: &str, num_outputs: usize) -> Self {
        match name.to_ascii_uppercase().as_str() {
            "MSE" | "MAE" => Self {
                target_topology: TargetTopology::Independent,
                target_range: TargetRange::Unbounded,
                output_squash_family: OutputSquashFamily::Unbounded,
                num_outputs,
            },
            "MAPE" | "MSLE" => Self {
                target_topology: TargetTopology::Independent,
                target_range: TargetRange::Positive,
                output_squash_family: OutputSquashFamily::Positive,
                num_outputs,
            },
            "BINARY_CROSS_ENTROPY" | "BINARYCROSSENTROPY" | "BCE" => Self {
                target_topology: TargetTopology::Independent,
                target_range: TargetRange::Unit,
                output_squash_family: OutputSquashFamily::BoundedUnipolar,
                num_outputs,
            },
            "CROSS_ENTROPY" | "CROSSENTROPY" | "CE" => Self {
                target_topology: TargetTopology::Simplex,
                target_range: TargetRange::Unit,
                output_squash_family: OutputSquashFamily::BoundedUnipolar,
                num_outputs,
            },
            "HINGE" => Self {
                target_topology: TargetTopology::Margin,
                target_range: TargetRange::SignedUnit,
                output_squash_family: OutputSquashFamily::BoundedBipolar,
                num_outputs,
            },
            "CATEGORICAL_ERROR" | "CATEGORICALERROR" => Self {
                target_topology: TargetTopology::OneHot,
                target_range: TargetRange::Unit,
                output_squash_family: OutputSquashFamily::BoundedUnipolar,
                num_outputs,
            },
            // OTHER and every unrecognised / absent name collapse to neutral.
            _ => Self::neutral(),
        }
    }

    /// Map this descriptor onto a [`CostFunctionHint`] for the
    /// target-reconstruction guard (issue #1317).
    ///
    /// The guard is consulted by detectors that reconstruct an "implied
    /// target" as `activation ± error` (see
    /// `src/analysis/cost_function_hint.rs`, issue #1250). That identity
    /// only holds for **linear-residual** costs — those whose recorded
    /// error obeys `error = target − output` (or its negation).
    ///
    /// Mapping rules:
    ///
    /// - `MSE` / `MAE` — `Independent` topology with `Unbounded` range ⇒
    ///   [`CostFunctionHint::LinearResidual`].
    /// - `BINARY_CROSS_ENTROPY` — `Independent` topology with `Unit` range
    ///   and `BoundedUnipolar` outputs ⇒ [`CostFunctionHint::LinearResidual`]
    ///   (NEAT-AI records BCE error as `target − output`).
    /// - `CROSS_ENTROPY` — `Simplex` topology ⇒
    ///   [`CostFunctionHint::LinearResidual`].
    /// - `MAPE` / `MSLE` — `Independent` topology with `Positive` range ⇒
    ///   [`CostFunctionHint::NonLinearResidual`].
    /// - `HINGE` — `Margin` topology ⇒ [`CostFunctionHint::NonLinearResidual`].
    /// - `CATEGORICAL_ERROR` — `OneHot` topology ⇒
    ///   [`CostFunctionHint::NonLinearResidual`].
    /// - Neutral / unknown / `OTHER` descriptors ⇒
    ///   [`CostFunctionHint::NonLinearResidual`] (conservative skip — see
    ///   issue #1317 acceptance criterion: "OTHER / Unknown / absent ⇒
    ///   conservative skip").
    ///
    /// The conservative-skip mapping for neutral descriptors is intentional:
    /// the existing [`CostFunctionHint::Unknown`] variant preserves the
    /// pre-Issue-#1250 "assume linear" behaviour for callers that have not
    /// been migrated, but at the dispatch level we want the *absence* of a
    /// cost identity to gate the reconstruction-dependent detectors off so
    /// they cannot emit spurious candidates against an unknown loss shape.
    #[must_use]
    pub fn cost_function_hint(&self) -> CostFunctionHint {
        use OutputSquashFamily as F;
        use TargetRange as R;
        use TargetTopology as T;

        match (
            self.target_topology,
            self.target_range,
            self.output_squash_family,
        ) {
            // MSE / MAE
            (T::Independent, R::Unbounded, F::Unbounded) => CostFunctionHint::LinearResidual,
            // BINARY_CROSS_ENTROPY
            (T::Independent, R::Unit, F::BoundedUnipolar) => CostFunctionHint::LinearResidual,
            // CROSS_ENTROPY
            (T::Simplex, R::Unit, _) => CostFunctionHint::LinearResidual,
            // Everything else — MAPE / MSLE (Independent + Positive),
            // HINGE (Margin), CATEGORICAL_ERROR (OneHot), neutral (Unknown).
            _ => CostFunctionHint::NonLinearResidual,
        }
    }
}

impl Default for TaskDescriptor {
    fn default() -> Self {
        Self::neutral()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neutral_is_unknown_unbounded_any() {
        let d = TaskDescriptor::neutral();
        assert_eq!(d.target_topology, TargetTopology::Unknown);
        assert_eq!(d.target_range, TargetRange::Unbounded);
        assert_eq!(d.output_squash_family, OutputSquashFamily::Any);
        assert_eq!(d.num_outputs, 0);
    }

    #[test]
    fn default_descriptor_equals_neutral() {
        assert_eq!(TaskDescriptor::default(), TaskDescriptor::neutral());
    }

    #[test]
    fn mse_maps_to_independent_unbounded_unbounded() {
        let d = TaskDescriptor::from_name("MSE", 3);
        assert_eq!(d.target_topology, TargetTopology::Independent);
        assert_eq!(d.target_range, TargetRange::Unbounded);
        assert_eq!(d.output_squash_family, OutputSquashFamily::Unbounded);
        assert_eq!(d.num_outputs, 3);
    }

    #[test]
    fn mae_maps_to_independent_unbounded_unbounded() {
        let d = TaskDescriptor::from_name("MAE", 5);
        assert_eq!(d.target_topology, TargetTopology::Independent);
        assert_eq!(d.target_range, TargetRange::Unbounded);
        assert_eq!(d.output_squash_family, OutputSquashFamily::Unbounded);
        assert_eq!(d.num_outputs, 5);
    }

    #[test]
    fn mape_maps_to_independent_positive_positive() {
        let d = TaskDescriptor::from_name("MAPE", 2);
        assert_eq!(d.target_topology, TargetTopology::Independent);
        assert_eq!(d.target_range, TargetRange::Positive);
        assert_eq!(d.output_squash_family, OutputSquashFamily::Positive);
        assert_eq!(d.num_outputs, 2);
    }

    #[test]
    fn msle_maps_to_independent_positive_positive() {
        let d = TaskDescriptor::from_name("MSLE", 4);
        assert_eq!(d.target_topology, TargetTopology::Independent);
        assert_eq!(d.target_range, TargetRange::Positive);
        assert_eq!(d.output_squash_family, OutputSquashFamily::Positive);
        assert_eq!(d.num_outputs, 4);
    }

    #[test]
    fn binary_cross_entropy_maps_to_independent_unit_bounded_unipolar() {
        let d = TaskDescriptor::from_name("BINARY_CROSS_ENTROPY", 1);
        assert_eq!(d.target_topology, TargetTopology::Independent);
        assert_eq!(d.target_range, TargetRange::Unit);
        assert_eq!(d.output_squash_family, OutputSquashFamily::BoundedUnipolar);
        assert_eq!(d.num_outputs, 1);
    }

    #[test]
    fn cross_entropy_maps_to_simplex_unit_bounded_unipolar() {
        let d = TaskDescriptor::from_name("CROSS_ENTROPY", 10);
        assert_eq!(d.target_topology, TargetTopology::Simplex);
        assert_eq!(d.target_range, TargetRange::Unit);
        assert_eq!(d.output_squash_family, OutputSquashFamily::BoundedUnipolar);
        assert_eq!(d.num_outputs, 10);
    }

    #[test]
    fn hinge_maps_to_margin_signed_unit_bounded_bipolar() {
        let d = TaskDescriptor::from_name("HINGE", 1);
        assert_eq!(d.target_topology, TargetTopology::Margin);
        assert_eq!(d.target_range, TargetRange::SignedUnit);
        assert_eq!(d.output_squash_family, OutputSquashFamily::BoundedBipolar);
        assert_eq!(d.num_outputs, 1);
    }

    #[test]
    fn categorical_error_maps_to_onehot_unit_bounded_unipolar() {
        let d = TaskDescriptor::from_name("CATEGORICAL_ERROR", 7);
        assert_eq!(d.target_topology, TargetTopology::OneHot);
        assert_eq!(d.target_range, TargetRange::Unit);
        assert_eq!(d.output_squash_family, OutputSquashFamily::BoundedUnipolar);
        assert_eq!(d.num_outputs, 7);
    }

    #[test]
    fn other_collapses_to_neutral() {
        assert_eq!(
            TaskDescriptor::from_name("OTHER", 4),
            TaskDescriptor::neutral(),
        );
    }

    #[test]
    fn unrecognised_name_collapses_to_neutral() {
        assert_eq!(
            TaskDescriptor::from_name("EXOTIC_LOSS", 8),
            TaskDescriptor::neutral(),
        );
    }

    #[test]
    fn empty_name_collapses_to_neutral() {
        assert_eq!(TaskDescriptor::from_name("", 2), TaskDescriptor::neutral(),);
    }

    #[test]
    fn lookup_is_case_insensitive() {
        for name in ["mse", "Mse", "mSe", "MSE"] {
            let d = TaskDescriptor::from_name(name, 1);
            assert_eq!(d.target_topology, TargetTopology::Independent);
            assert_eq!(d.target_range, TargetRange::Unbounded);
            assert_eq!(d.output_squash_family, OutputSquashFamily::Unbounded);
        }
        for name in ["hinge", "Hinge", "HINGE"] {
            let d = TaskDescriptor::from_name(name, 1);
            assert_eq!(d.target_topology, TargetTopology::Margin);
        }
    }

    #[test]
    fn cost_function_hint_for_linear_residual_costs() {
        // Issue #1317: MSE, MAE, BCE, CE map to LinearResidual.
        for name in ["MSE", "MAE", "BINARY_CROSS_ENTROPY", "CROSS_ENTROPY"] {
            let descriptor = TaskDescriptor::from_name(name, 4);
            assert_eq!(
                descriptor.cost_function_hint(),
                CostFunctionHint::LinearResidual,
                "{name} descriptor should map to LinearResidual",
            );
        }
    }

    #[test]
    fn cost_function_hint_for_non_linear_residual_costs() {
        // Issue #1317: MAPE, MSLE, HINGE, CATEGORICAL_ERROR map to
        // NonLinearResidual so the reconstruction-dependent detectors skip.
        for name in ["MAPE", "MSLE", "HINGE", "CATEGORICAL_ERROR"] {
            let descriptor = TaskDescriptor::from_name(name, 4);
            assert_eq!(
                descriptor.cost_function_hint(),
                CostFunctionHint::NonLinearResidual,
                "{name} descriptor should map to NonLinearResidual",
            );
        }
    }

    #[test]
    fn cost_function_hint_for_neutral_is_conservative_skip() {
        // Issue #1317 acceptance: OTHER / Unknown / absent ⇒ conservative
        // skip. The neutral descriptor maps to NonLinearResidual to gate
        // the implied-target reconstructions off when we know nothing
        // about the loss shape.
        assert_eq!(
            TaskDescriptor::neutral().cost_function_hint(),
            CostFunctionHint::NonLinearResidual,
            "neutral descriptor must conservatively skip",
        );
        assert_eq!(
            TaskDescriptor::from_name("OTHER", 3).cost_function_hint(),
            CostFunctionHint::NonLinearResidual,
            "OTHER descriptor must conservatively skip",
        );
        assert_eq!(
            TaskDescriptor::from_name("EXOTIC_LOSS", 3).cost_function_hint(),
            CostFunctionHint::NonLinearResidual,
            "Unrecognised descriptor must conservatively skip",
        );
    }

    #[test]
    fn num_outputs_is_preserved_verbatim() {
        for n in [0_usize, 1, 7, 1024] {
            assert_eq!(TaskDescriptor::from_name("MSE", n).num_outputs, n);
            assert_eq!(TaskDescriptor::from_name("CROSS_ENTROPY", n).num_outputs, n);
            // Unrecognised names go through neutral(), which always has 0
            // outputs by design — the caller should not be relying on the
            // value when the cost is unknown.
            assert_eq!(TaskDescriptor::from_name("OTHER", n).num_outputs, 0);
        }
    }
}
