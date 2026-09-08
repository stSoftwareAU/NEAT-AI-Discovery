//! Shared error-dispersion / plateau assessment (Issue #2044).
//!
//! Three detectors ask the same question of a neuron's recorded absolute
//! errors: *how tightly are they clustered around their mean?* The answer is
//! the coefficient of variation (`std_dev / mean_error`), and two of the three
//! then apply the same "the error is stuck" rule to it:
//!
//! - [`error_plateau`](super::error_plateau) (Issue #545 / #547) — an output
//!   neuron plateaued in a local minimum.
//! - [`weight_magnitude_reset`](super::weight_magnitude_reset) (Issue #1006) —
//!   a synapse feeding a target whose error is stuck.
//! - [`topology_diversification`](super::topology_diversification) (Issue #1013)
//!   — uses the same dispersion, but looks for the *opposite* signal: a
//!   coefficient of variation **above** a ceiling (high error variance).
//!
//! The rule is defined once here:
//!
//! 1. **Business floor** — the mean absolute error must reach the caller's
//!    `min_mean_error`, below which the neuron counts as converged.
//! 2. **Numerical floor** — the coefficient of variation is only computed when
//!    the mean exceeds [`MIN_CV_DENOMINATOR`]; otherwise it is
//!    `f32::INFINITY`, so a near-zero mean can never masquerade as a tight
//!    cluster.
//! 3. **Tightness ceiling** ([`assess_error_plateau`] only) — the coefficient
//!    of variation must sit at or below the caller's ceiling.
//!
//! Keeping the two floors distinct matters: they coincide today only because
//! every caller's business floor happens to sit well above the numerical one
//! (Issue #2044).

use super::stats::{compute_mean, compute_variance};

/// Smallest mean error for which `std_dev / mean` is numerically meaningful.
///
/// Below this the ratio is dominated by floating-point noise, so the
/// coefficient of variation is reported as `f32::INFINITY` instead.
pub const MIN_CV_DENOMINATOR: f32 = 1e-6;

/// Dispersion of a neuron's absolute errors about their mean.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ErrorDispersion {
    /// Mean absolute error across the observations.
    pub mean_error: f32,
    /// Population standard deviation of the absolute errors.
    pub std_dev: f32,
    /// Coefficient of variation (`std_dev / mean_error`), or `f32::INFINITY`
    /// when the mean is at or below [`MIN_CV_DENOMINATOR`].
    pub cv: f32,
}

impl ErrorDispersion {
    /// How tightly the errors cluster, relative to a caller's ceiling.
    ///
    /// `1.0` is a perfectly flat error (`cv == 0`), falling to `0.0` at and
    /// beyond `cv_ceiling`. Returns `0.0` for a non-positive ceiling.
    #[must_use]
    pub fn plateau_tightness(&self, cv_ceiling: f32) -> f32 {
        if cv_ceiling <= 0.0 {
            return 0.0;
        }
        (1.0 - self.cv / cv_ceiling).max(0.0)
    }
}

/// Compute the dispersion of `errors`, gated on the caller's business floor.
///
/// Returns `None` when `errors` is empty or its mean falls below
/// `min_mean_error` (the neuron is already converged).
///
/// # Arguments
/// * `errors` - Absolute errors, one per observation.
/// * `min_mean_error` - Business floor the mean absolute error must reach.
#[must_use]
pub fn error_dispersion(errors: &[f32], min_mean_error: f32) -> Option<ErrorDispersion> {
    if errors.is_empty() {
        return None;
    }

    let mean_error = compute_mean(errors);
    if mean_error < min_mean_error {
        return None;
    }

    let std_dev = compute_variance(errors).sqrt();
    let cv = if mean_error > MIN_CV_DENOMINATOR {
        std_dev / mean_error
    } else {
        f32::INFINITY
    };

    Some(ErrorDispersion {
        mean_error,
        std_dev,
        cv,
    })
}

/// Assess whether `errors` show a plateau: high mean error, tightly clustered.
///
/// Returns `Some(ErrorDispersion)` only when the mean reaches `min_mean_error`
/// **and** the coefficient of variation is at or below `cv_ceiling`, so its
/// existence *is* the plateau decision — callers never re-apply the rule.
///
/// # Arguments
/// * `errors` - Absolute errors, one per observation.
/// * `min_mean_error` - Business floor the mean absolute error must reach.
/// * `cv_ceiling` - Highest coefficient of variation still counted as a plateau.
#[must_use]
pub fn assess_error_plateau(
    errors: &[f32],
    min_mean_error: f32,
    cv_ceiling: f32,
) -> Option<ErrorDispersion> {
    error_dispersion(errors, min_mean_error).filter(|d| d.cv <= cv_ceiling)
}
