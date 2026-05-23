//! Cost-function hint for discovery consumers (Issue #1250).
//!
//! A number of detectors reconstruct an "implied target" as
//! `activation ± error`. That identity only holds when the recorded error
//! is a **linear residual** — true for `MSE`, `MAE`, and the soft-max form
//! of `CROSS_ENTROPY`. For `MAPE`, `MSLE`, `HINGE`, and `CATEGORICAL_ERROR`
//! the recorded error is not the additive residual, so the reconstructed
//! target is wrong and any downstream decision based on it is wrong.
//!
//! This module provides a small enum, [`CostFunctionHint`], that callers
//! can pass into the affected detectors so they can either keep their
//! linear-residual logic (when known-safe) or skip the affected code path
//! (when known-unsafe). The default — `Unknown` — preserves the original
//! pre-Issue-#1250 behaviour and treats the cost as linear, which keeps
//! every existing caller backwards-compatible until they are migrated to
//! pass a real hint.
//!
//! See `docs/COST_FUNCTION_NOTES.md` §4 for the full per-cost catalogue.

/// Per-cost behavioural hint for detectors that reconstruct an implied target
/// from `activation` and `errors[i]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CostFunctionHint {
    /// Caller has not supplied a cost function. Detectors keep their
    /// historical "assume linear residual" behaviour. Use this only for
    /// backwards compatibility — pass an explicit hint when the cost is
    /// known.
    #[default]
    Unknown,
    /// Recorded error is a linear residual (`error = target − output`, or
    /// its negation). Covers `MSE`, `MAE`, and NEAT-AI's soft-max
    /// `CROSS_ENTROPY`. `activation ± error` is a valid target estimator.
    LinearResidual,
    /// Recorded error is **not** a linear residual: the additive
    /// reconstruction `activation ± error` does not yield the recorded
    /// target. Covers `MAPE`, `MSLE`, `HINGE`, and `CATEGORICAL_ERROR`.
    /// Detectors must skip any code path that depends on that identity.
    NonLinearResidual,
}

impl CostFunctionHint {
    /// Map a NEAT-AI cost name (case-insensitive) onto the appropriate
    /// hint. Unknown names return [`CostFunctionHint::Unknown`] so the
    /// caller can decide whether to be conservative.
    #[must_use]
    pub fn from_name(name: &str) -> Self {
        match name.to_ascii_uppercase().as_str() {
            "MSE" | "MAE" | "CROSS_ENTROPY" | "CROSSENTROPY" | "CE" => Self::LinearResidual,
            "MAPE" | "MSLE" | "HINGE" | "CATEGORICAL_ERROR" | "CATEGORICALERROR" => {
                Self::NonLinearResidual
            }
            _ => Self::Unknown,
        }
    }

    /// `true` when the recorded error can be added to (or subtracted from)
    /// the activation to recover the recorded target. `Unknown` is treated
    /// as `true` to preserve historical detector behaviour for callers
    /// that have not been migrated to pass an explicit hint.
    #[must_use]
    pub fn allows_linear_target_reconstruction(self) -> bool {
        matches!(self, Self::LinearResidual | Self::Unknown)
    }

    /// `true` when the caller has positively declared the cost as
    /// non-linear. Detectors should disable any linear-residual code path
    /// in this case.
    #[must_use]
    pub fn is_non_linear_residual(self) -> bool {
        matches!(self, Self::NonLinearResidual)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_residual_costs_map_to_linear() {
        for name in ["MSE", "mae", "Cross_Entropy", "CE", "crossentropy"] {
            assert_eq!(
                CostFunctionHint::from_name(name),
                CostFunctionHint::LinearResidual,
                "{name} should map to LinearResidual",
            );
        }
    }

    #[test]
    fn non_linear_residual_costs_map_to_non_linear() {
        for name in [
            "MAPE",
            "msle",
            "HINGE",
            "Categorical_Error",
            "categoricalerror",
        ] {
            assert_eq!(
                CostFunctionHint::from_name(name),
                CostFunctionHint::NonLinearResidual,
                "{name} should map to NonLinearResidual",
            );
        }
    }

    #[test]
    fn unknown_cost_maps_to_unknown() {
        assert_eq!(
            CostFunctionHint::from_name("EXOTIC_LOSS"),
            CostFunctionHint::Unknown,
        );
    }

    #[test]
    fn unknown_preserves_legacy_linear_behaviour() {
        let hint = CostFunctionHint::Unknown;
        assert!(hint.allows_linear_target_reconstruction());
        assert!(!hint.is_non_linear_residual());
    }

    #[test]
    fn non_linear_gates_off() {
        let hint = CostFunctionHint::NonLinearResidual;
        assert!(!hint.allows_linear_target_reconstruction());
        assert!(hint.is_non_linear_residual());
    }

    #[test]
    fn linear_residual_allows_reconstruction() {
        let hint = CostFunctionHint::LinearResidual;
        assert!(hint.allows_linear_target_reconstruction());
        assert!(!hint.is_non_linear_residual());
    }

    #[test]
    fn default_is_unknown() {
        assert_eq!(CostFunctionHint::default(), CostFunctionHint::Unknown);
    }
}
