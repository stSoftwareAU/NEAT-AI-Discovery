//! Shared sentinel-cluster assessment (Issue #2042).
//!
//! Two detectors ask the same question of an observation's recorded samples:
//! *is this cluster of values a sentinel (a "no data" marker) rather than a
//! meaningful signal?*
//!
//! - [`observation_range`](super::observation_range) (Issue #398) — characterises
//!   the effective range left once sentinel clusters are removed.
//! - [`sentinel_gating`](super::sentinel_gating) (Issue #400) — proposes a gate
//!   neuron to suppress the sentinel region.
//!
//! The rule is one rule, defined once here in [`assess_sentinel_cluster`]:
//!
//! 1. **Density** — at least `MIN_SENTINEL_FRACTION` of samples sit within
//!    `SENTINEL_TOLERANCE` of the candidate value.
//! 2. **Separation** — the cluster is at least `MIN_SENTINEL_GAP` away from the
//!    useful (non-sentinel) range, and outside it.
//! 3. **Error decorrelation** — the cluster's error variance is *lower* than the
//!    useful range's, so the value does not meaningfully influence the output.
//!
//! All three must hold. The copies previously diverged on rule 3: the
//! `observation_range` copy combined it with rule 2 using `||` **after** already
//! rejecting insufficient gaps, so the disjunct was always true and the variance
//! test was dead code (Issue #2042).

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)

use crate::analysis::constants::{
    MIN_SENTINEL_FRACTION, MIN_SENTINEL_GAP as MIN_GAP, SENTINEL_TOLERANCE,
};

/// An accepted sentinel cluster and the evidence that accepted it.
///
/// Only produced by [`assess_sentinel_cluster`], so its existence *is* the
/// accept decision — callers never re-apply the rule.
#[derive(Debug, Clone)]
pub struct SentinelCluster {
    /// The candidate sentinel value the cluster formed around.
    pub sentinel_value: f32,
    /// Sample indices inside the cluster.
    pub sentinel_indices: Vec<usize>,
    /// Sample indices outside the cluster (the useful range).
    pub non_sentinel_indices: Vec<usize>,
    /// Fraction of samples inside the cluster (0.0 to 1.0).
    pub sentinel_fraction: f32,
    /// Minimum activation of the useful (non-sentinel) range.
    pub useful_min: f32,
    /// Maximum activation of the useful (non-sentinel) range.
    pub useful_max: f32,
    /// Separation between the cluster edge and the useful range.
    pub gap: f32,
    /// Error variance inside the cluster.
    pub sentinel_error_var: f32,
    /// Error variance across the useful range.
    pub non_sentinel_error_var: f32,
}

/// Assess one candidate sentinel value against an observation's samples.
///
/// Returns `Some(SentinelCluster)` only when the density, separation and error
/// decorrelation rules all hold; `None` means the candidate is not a sentinel.
///
/// # Arguments
/// * `activations` - Recorded activation per sample.
/// * `errors` - Recorded error per sample, index-aligned with `activations`.
/// * `sentinel` - The candidate sentinel value (e.g. -1.0, 0.0, 1.0).
///
/// # Panics
/// Panics if `errors` is shorter than `activations` — the two are per-sample
/// views of the same records and a mismatch is a caller bug, not a data case.
#[must_use]
pub fn assess_sentinel_cluster(
    activations: &[f32],
    errors: &[f32],
    sentinel: f32,
) -> Option<SentinelCluster> {
    assert!(
        errors.len() >= activations.len(),
        "errors must be index-aligned with activations: {} errors for {} activations",
        errors.len(),
        activations.len()
    );

    if activations.is_empty() {
        return None;
    }

    let n = activations.len() as f32;

    // Rule 1: density — enough samples cluster at the candidate value.
    let sentinel_indices: Vec<usize> = activations
        .iter()
        .enumerate()
        .filter(|&(_, &a)| (a - sentinel).abs() <= SENTINEL_TOLERANCE)
        .map(|(i, _)| i)
        .collect();

    let sentinel_fraction = sentinel_indices.len() as f32 / n;
    if sentinel_fraction < MIN_SENTINEL_FRACTION {
        return None;
    }

    let non_sentinel_indices: Vec<usize> = (0..activations.len())
        .filter(|i| !sentinel_indices.contains(i))
        .collect();

    if non_sentinel_indices.is_empty() {
        return None;
    }

    let useful_min = non_sentinel_indices
        .iter()
        .map(|&i| activations[i])
        .fold(f32::INFINITY, f32::min);
    let useful_max = non_sentinel_indices
        .iter()
        .map(|&i| activations[i])
        .fold(f32::NEG_INFINITY, f32::max);

    // Rule 2: separation — the cluster sits clear of the useful range.
    let gap = if sentinel <= useful_min {
        useful_min - (sentinel + SENTINEL_TOLERANCE)
    } else if sentinel >= useful_max {
        (sentinel - SENTINEL_TOLERANCE) - useful_max
    } else {
        // Sentinel is inside the useful range — not a clear boundary.
        0.0
    };

    if gap < MIN_GAP {
        return None;
    }

    // Rule 3: error decorrelation — the cluster must carry *less* error
    // variation than the useful range, or it is still informative.
    let sentinel_error_var = compute_error_variance(errors, &sentinel_indices);
    let non_sentinel_error_var = compute_error_variance(errors, &non_sentinel_indices);

    if sentinel_error_var >= non_sentinel_error_var {
        return None;
    }

    Some(SentinelCluster {
        sentinel_value: sentinel,
        sentinel_indices,
        non_sentinel_indices,
        sentinel_fraction,
        useful_min,
        useful_max,
        gap,
        sentinel_error_var,
        non_sentinel_error_var,
    })
}

/// Compute the population variance of the error values at the given indices.
///
/// Returns `0.0` when `indices` is empty.
#[must_use]
pub fn compute_error_variance(errors: &[f32], indices: &[usize]) -> f32 {
    if indices.is_empty() {
        return 0.0;
    }

    let n = indices.len() as f32;
    let mean = indices.iter().map(|&i| errors[i]).sum::<f32>() / n;

    indices
        .iter()
        .map(|&i| (errors[i] - mean).powi(2))
        .sum::<f32>()
        / n
}
