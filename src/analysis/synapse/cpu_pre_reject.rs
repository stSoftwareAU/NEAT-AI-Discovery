//! CPU pre-reject screen for helpful synapse candidates (Issue #1544).
//!
//! At GRQ scale thousands of sources are GPU-evaluated where many end with
//! `gpu_improved_count == 0`. The CPU already holds the built samples, so it can
//! reject the obvious duds more cheaply than a GPU round-trip.
//!
//! This module provides a **provably quality-neutral** screen: it recomputes
//! the least-squares sufficient statistics that the helpful GPU shader would
//! produce and applies [`calculate_optimal_outgoing_weight`] — the *exact* gate
//! the downstream result-collection loop uses (`None => continue` in
//! `target_analysis::evaluation::collect_and_process_helpful_results`). A
//! candidate screened out here has no finite, above-epsilon optimal outgoing
//! weight, so it would have been rejected after the GPU submit with an identical
//! outcome. The screen therefore changes GPU cost, never which candidates
//! survive.
//!
//! Dropped candidates are recorded under
//! [`REJECTION_CPU_PRE_REJECT_NO_SIGNAL`] so drought diagnostics can
//! distinguish "cheaply screened out on CPU" from a genuine candidate drought.
//!
//! [`REJECTION_CPU_PRE_REJECT_NO_SIGNAL`]:
//! crate::analysis::diagnostics::rejection_reasons::REJECTION_CPU_PRE_REJECT_NO_SIGNAL

#![allow(clippy::cast_possible_truncation)] // Intentional f64→f32 narrowing for the GPU-matched weight gate (Issue #873)

use crate::analysis::samples::HelpfulSample;
use crate::analysis::scoring::weights::calculate_optimal_outgoing_weight;

/// Return `true` when a helpful add-synapse candidate provably cannot yield a
/// usable outgoing weight, so it can be dropped before the GPU round-trip.
///
/// The two least-squares sums are accumulated in `f64` to avoid catastrophic
/// cancellation, then narrowed to `f32` for the weight gate so the decision
/// matches the downstream `calculate_optimal_outgoing_weight` call (which
/// returns `None` for a degenerate `Σ activation² ≤ EPSILON` source or a
/// zero-correlation `|Σ activation·avg_error| / Σ activation² ≤ EPSILON`
/// candidate).
///
/// Empty sample sets are treated as no-signal — the downstream loop also
/// rejects a zero-length candidate (`full_total_count == 0 => continue`).
#[must_use]
pub fn helpful_candidate_has_no_signal(samples: &[HelpfulSample]) -> bool {
    if samples.is_empty() {
        return true;
    }

    let mut sum_activation_sq = 0.0f64;
    let mut sum_error_activation = 0.0f64;
    for sample in samples {
        if sample.activation.is_finite() && sample.avg_error.is_finite() {
            let a = f64::from(sample.activation);
            sum_activation_sq += a * a;
            sum_error_activation += a * f64::from(sample.avg_error);
        }
    }

    calculate_optimal_outgoing_weight(sum_error_activation as f32, sum_activation_sq as f32, 1.0)
        .is_none()
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)] // Test fixtures build sample values from small loop indices.
mod tests {
    use super::*;

    fn sample(activation: f32, avg_error: f32) -> HelpfulSample {
        HelpfulSample {
            activation,
            avg_error,
            target_value: None,
            target_activation: None,
        }
    }

    #[test]
    fn empty_batch_is_no_signal() {
        assert!(helpful_candidate_has_no_signal(&[]));
    }

    #[test]
    fn constant_zero_activation_source_is_no_signal() {
        // Σ activation² ≈ 0 ⇒ no meaningful weight can be fitted (a dead source).
        let samples: Vec<HelpfulSample> = (0..64).map(|i| sample(0.0, 0.1 * i as f32)).collect();
        assert!(helpful_candidate_has_no_signal(&samples));
    }

    #[test]
    fn zero_correlation_source_is_no_signal() {
        // Activation varies but is uncorrelated with error (Σ activation·error = 0),
        // so the optimal weight is ~0 and the candidate is a provable dud.
        let samples = vec![
            sample(1.0, 0.5),
            sample(-1.0, 0.5),
            sample(1.0, -0.5),
            sample(-1.0, -0.5),
        ];
        assert!(helpful_candidate_has_no_signal(&samples));
    }

    #[test]
    fn strong_signal_source_survives() {
        // Activation strongly correlated with error ⇒ a usable weight exists,
        // so the candidate must NOT be screened out (guards against silently
        // dropping good candidates — the dangerous failure mode of Issue #1544).
        let samples: Vec<HelpfulSample> = (0..64)
            .map(|i| {
                let a = (i as f32 % 5.0) - 2.0;
                sample(a, a * 0.7)
            })
            .collect();
        assert!(!helpful_candidate_has_no_signal(&samples));
    }

    #[test]
    fn non_finite_samples_are_ignored() {
        // NaN / inf samples are skipped; the remaining finite pair has signal.
        let samples = vec![
            sample(f32::NAN, 1.0),
            sample(f32::INFINITY, 1.0),
            sample(2.0, 1.4),
            sample(-2.0, -1.4),
        ];
        assert!(!helpful_candidate_has_no_signal(&samples));
    }
}
