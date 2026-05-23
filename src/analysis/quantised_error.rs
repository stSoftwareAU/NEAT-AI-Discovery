//! Quantised `{0, 1}` error regime detection (Issue #1247).
//!
//! NEAT-AI's `CATEGORICAL_ERROR` cost (PR #2739) records output-neuron
//! errors as quantised misclassification flags — every value is exactly
//! `0` (correct prediction) or `1` (incorrect prediction). Detectors that
//! rely on a continuous error signal (Spearman rank correlation, sample
//! variance, SSE-style improvement ratios) degrade in well-defined but
//! sometimes misleading ways under this regime.
//!
//! This helper lets distribution-sensitive detectors recognise the
//! quantised regime cheaply at runtime so they can either:
//!
//! - **Skip** the detection for the affected target, **or**
//! - **Switch** to a quantile- or presence-based alternative that
//!   survives the regime.
//!
//! See `docs/COST_FUNCTION_NOTES.md` §4.7 for the full catalogue of
//! affected consumers.
//!
//! ## Definition
//!
//! A slice of errors is considered to be in the "quantised `{0, 1}`"
//! regime when:
//!
//! 1. Every finite entry is approximately `0.0` or `1.0` (within
//!    [`QUANTISATION_TOLERANCE`]).
//! 2. There is meaningful mass at both modes (at least
//!    [`MIN_MODE_FRACTION`] of finite entries at each of `0` and `1`).
//!
//! Pure all-zero or all-one batches are *not* flagged as quantised —
//! they are zero-variance regimes that detectors already handle via
//! their existing `is_finite` / `variance > eps` guards. The quantised
//! regime is the interesting case: errors look like a Bernoulli sample
//! and the downstream detector's continuous-error assumption fails.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)] // Counts converted to f32 for ratio comparisons.

/// Absolute tolerance for "is this f32 value approximately `0.0` or
/// `1.0`?" The recorded error comes from a `cost(target, output)` call
/// whose output is forced to exact `0.0` / `1.0` by an argmax check, so
/// the only deviation we expect is round-off through `f32`. A tolerance
/// of `1e-6` rejects anything that has been arithmetically combined
/// (e.g. averaged with a continuous residual) without flagging values
/// that have only been copied.
pub const QUANTISATION_TOLERANCE: f32 = 1e-6;

/// Minimum fraction of finite errors that must sit at each mode for the
/// slice to count as a "non-trivial" quantised batch. A batch where
/// 99 % of errors are `0` and only one is `1` is statistically
/// indistinguishable from "the network already learnt this output", so
/// no special handling is warranted. The default of 5 % matches the
/// minimum cluster fraction used by other discovery detectors.
pub const MIN_MODE_FRACTION: f32 = 0.05;

/// Returns `true` when `errors` is in the quantised `{0, 1}` regime.
///
/// See the module-level documentation for the precise definition. The
/// function silently ignores non-finite entries; a slice that is
/// entirely non-finite returns `false`.
#[must_use]
pub fn is_quantised_zero_one(errors: &[f32]) -> bool {
    is_quantised_zero_one_with_fraction(errors, MIN_MODE_FRACTION)
}

/// Variant of [`is_quantised_zero_one`] that accepts a custom minority
/// mode fraction. Intended for callers that want a stricter or looser
/// threshold; production detectors should prefer the default.
#[must_use]
pub fn is_quantised_zero_one_with_fraction(errors: &[f32], min_mode_fraction: f32) -> bool {
    let mut finite_count = 0usize;
    let mut zero_count = 0usize;
    let mut one_count = 0usize;

    for &e in errors {
        if !e.is_finite() {
            continue;
        }
        finite_count += 1;
        if e.abs() <= QUANTISATION_TOLERANCE {
            zero_count += 1;
        } else if (e - 1.0).abs() <= QUANTISATION_TOLERANCE {
            one_count += 1;
        } else {
            // Any value that is neither approximately 0 nor approximately 1
            // disqualifies the entire batch from the quantised regime.
            return false;
        }
    }

    // Need a meaningful number of samples and mass at both modes.
    if finite_count == 0 {
        return false;
    }
    let min_required = ((finite_count as f32) * min_mode_fraction).ceil() as usize;
    let min_required = min_required.max(1);

    zero_count >= min_required && one_count >= min_required
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_slice_is_not_quantised() {
        assert!(!is_quantised_zero_one(&[]));
    }

    #[test]
    fn all_zero_is_not_quantised() {
        // A constant zero batch is a zero-variance regime, not a
        // bimodal quantised one. Distinct detector code already
        // handles "no signal" — we only flag the truly bimodal case.
        let errors = [0.0_f32; 32];
        assert!(!is_quantised_zero_one(&errors));
    }

    #[test]
    fn all_one_is_not_quantised() {
        let errors = [1.0_f32; 32];
        assert!(!is_quantised_zero_one(&errors));
    }

    #[test]
    fn balanced_zero_one_is_quantised() {
        let errors: Vec<f32> = (0..32)
            .map(|i| if i % 2 == 0 { 0.0 } else { 1.0 })
            .collect();
        assert!(is_quantised_zero_one(&errors));
    }

    #[test]
    fn slightly_unbalanced_zero_one_is_quantised() {
        // 28 zeros, 4 ones — still ≥5 % at the minority mode (4/32 = 12.5 %).
        let mut errors = vec![0.0_f32; 28];
        errors.extend([1.0_f32; 4]);
        assert!(is_quantised_zero_one(&errors));
    }

    #[test]
    fn single_outlier_is_not_quantised() {
        // 99 zeros, 1 one — minority mode under 5 %, not flagged.
        let mut errors = vec![0.0_f32; 99];
        errors.push(1.0);
        assert!(!is_quantised_zero_one(&errors));
    }

    #[test]
    fn continuous_errors_are_not_quantised() {
        let errors: Vec<f32> = (0..32).map(|i| (i as f32) / 31.0).collect();
        assert!(!is_quantised_zero_one(&errors));
    }

    #[test]
    fn signed_residuals_are_not_quantised() {
        // -1, 0, 1 — sign-carrying residuals must NOT be confused with
        // the unsigned quantised regime.
        let errors: Vec<f32> = (0..32)
            .map(|i| match i % 3 {
                0 => -1.0,
                1 => 0.0,
                _ => 1.0,
            })
            .collect();
        assert!(!is_quantised_zero_one(&errors));
    }

    #[test]
    fn non_finite_entries_are_ignored() {
        let mut errors: Vec<f32> = (0..30)
            .map(|i| if i % 2 == 0 { 0.0 } else { 1.0 })
            .collect();
        errors.push(f32::NAN);
        errors.push(f32::INFINITY);
        errors.push(f32::NEG_INFINITY);
        assert!(is_quantised_zero_one(&errors));
    }

    #[test]
    fn all_non_finite_is_not_quantised() {
        let errors = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
        assert!(!is_quantised_zero_one(&errors));
    }

    #[test]
    fn tolerance_accepts_round_off_noise() {
        // Small f32 round-off around the modes must still register.
        let mut errors: Vec<f32> = vec![1e-8_f32, -1e-8_f32, 1.0 - 1e-8, 1.0 + 1e-8];
        errors.extend([0.0_f32; 8]);
        errors.extend([1.0_f32; 8]);
        assert!(is_quantised_zero_one(&errors));
    }

    #[test]
    fn out_of_tolerance_disqualifies_batch() {
        // A single 0.5 between batches of {0, 1} is enough to reject
        // the regime — the batch has continuous structure.
        let mut errors: Vec<f32> = (0..30)
            .map(|i| if i % 2 == 0 { 0.0 } else { 1.0 })
            .collect();
        errors.push(0.5);
        assert!(!is_quantised_zero_one(&errors));
    }

    #[test]
    fn custom_fraction_is_respected() {
        // 95 zeros, 5 ones — minority mode at 5 %.
        let mut errors = vec![0.0_f32; 95];
        errors.extend([1.0_f32; 5]);
        // Default (5 %) flags it.
        assert!(is_quantised_zero_one(&errors));
        // 10 % minimum rejects it.
        assert!(!is_quantised_zero_one_with_fraction(&errors, 0.10));
    }
}
