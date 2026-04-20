//! Per-change-type prediction calibration correction (Issue #1131).
//!
//! The compiled calibration constants (`NEURON_PREDICTION_CALIBRATION`,
//! `SYNAPSE_PREDICTION_CALIBRATION`, `COORDINATED_PREDICTION_CALIBRATION`) are
//! fixed at build time and reflect the historical average over-estimation gap
//! between `expectedCreatureScoreGain` and the actual post-apply creature
//! score delta. Individual creatures can drift from that average: a creature
//! whose recent failure cache shows a 1300× over-estimate for `add-neurons`
//! should discount neuron predictions more than the global constant does.
//!
//! This module computes a scalar correction factor per change-type (e.g.,
//! `add-neurons`, `add-synapses`, `coordinated-structural`) from recent
//! entries of the failure cache. The correction is applied
//! **multiplicatively** to the existing calibration constant — it never
//! overrides the constant, and it is floored at
//! [`MIN_CALIBRATION_CORRECTION`] so a run of failures cannot collapse
//! predictions to zero.
//!
//! # Formula
//!
//! Given failure cache entries with `expected_error_reduction` and
//! `actual_error_reduction`, the per-entry ratio is
//! `actual / expected`. An exponentially-weighted moving average (EWMA)
//! of these ratios is computed so recent outcomes dominate the estimate.
//! The EWMA is then clamped to
//! `[MIN_CALIBRATION_CORRECTION, NEUTRAL_CORRECTION]` (= `[0.001, 1.0]`).
//!
//! Entries where `expected_error_reduction == 0` are skipped.
//!
//! # Integration
//!
//! The correction is loaded at the FFI entry point (`analyze_parallel_internal`)
//! and plumbed through `analyze_all` into the post-processing passes in
//! `synapse::post_processing` and `neuron::post_processing`. Each call site
//! multiplies the base calibration constant by
//! `correction.get_correction(change_type)` before invoking the logistic /
//! flat calibration function.

use serde::Deserialize;
use std::collections::HashMap;

// =============================================================================
// Constants
// =============================================================================

/// Minimum correction factor (lower clamp).
///
/// A run of failures with `actual / expected` at or near zero clamps to this
/// floor rather than collapsing predictions to zero. 0.001 means a creature
/// whose whole recent history shows ~1000× over-estimates still receives a
/// small non-zero prediction — enough for exploration, while staying far
/// below the neutral factor of 1.0.
pub const MIN_CALIBRATION_CORRECTION: f32 = 0.001;

/// Neutral correction factor (upper clamp, returned for empty / unknown caches).
///
/// Predictions are only ever discounted further by this correction — never
/// inflated. The upper clamp of 1.0 ensures the correction can never make the
/// calibration constant larger than its compiled value.
pub const NEUTRAL_CORRECTION: f32 = 1.0;

/// EWMA smoothing coefficient applied to each successive entry (0 < α ≤ 1).
///
/// `α = 0.3` gives recent outcomes roughly 3× the weight of outcomes 3 entries
/// ago. A full-cache override only happens with ~20 entries all moving in the
/// same direction.
pub const CALIBRATION_CORRECTION_EWMA_ALPHA: f32 = 0.3;

// =============================================================================
// Change-type identifiers
// =============================================================================

/// Change-type key for `add-neurons` candidates.
pub const CHANGE_TYPE_ADD_NEURONS: &str = "add-neurons";

/// Change-type key for `add-synapses` candidates.
pub const CHANGE_TYPE_ADD_SYNAPSES: &str = "add-synapses";

/// Change-type key for `coordinated-structural` candidates.
pub const CHANGE_TYPE_COORDINATED_STRUCTURAL: &str = "coordinated-structural";

// =============================================================================
// Failure cache entry
// =============================================================================

/// A single failure-cache record: what we predicted vs what actually happened.
///
/// `change_type` identifies which prediction bucket this outcome belongs to.
/// Stable keys (see the `CHANGE_TYPE_*` constants) let the same cache feed
/// back into the correction for the same class of candidate next run.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FailureCacheEntry {
    /// Candidate classification (e.g., `add-neurons`, `add-synapses`,
    /// `coordinated-structural`). Used as the correction key.
    pub change_type: String,

    /// What we predicted — the `expectedErrorReduction` at the time of proposal.
    pub expected_error_reduction: f32,

    /// What actually happened — the post-apply `actualErrorReduction`. May be
    /// negative for candidates that harmed the network.
    pub actual_error_reduction: f32,
}

// =============================================================================
// CalibrationCorrection
// =============================================================================

/// Per-change-type scalar correction factors derived from recent outcomes.
///
/// A value of 1.0 means "apply the compiled calibration constant as-is".
/// A value of 0.001 (the floor) means "this change-type has been over-estimating
/// so much recently that predictions should be heavily discounted beyond
/// the compiled constant".
#[derive(Debug, Clone, Default)]
pub struct CalibrationCorrection {
    corrections: HashMap<String, f32>,
}

impl CalibrationCorrection {
    /// Returns a neutral correction where every change-type maps to 1.0.
    #[must_use]
    pub fn neutral() -> Self {
        Self::default()
    }

    /// Builds a correction map from the supplied failure cache.
    ///
    /// # Algorithm
    ///
    /// 1. Group entries by `change_type`.
    /// 2. Skip entries with `expected_error_reduction == 0` (undefined ratio).
    /// 3. For each group, compute an EWMA of `actual / expected` in the order
    ///    supplied. Callers pass entries oldest-first; the final EWMA value
    ///    therefore reflects the most recent outcomes most strongly.
    /// 4. Clamp the EWMA to `[MIN_CALIBRATION_CORRECTION, NEUTRAL_CORRECTION]`.
    ///
    /// Empty caches, or caches with no usable entries for a change-type,
    /// produce a [`Self::neutral`] correction for that change-type on lookup.
    #[must_use]
    pub fn from_failure_cache(cache: &[FailureCacheEntry]) -> Self {
        // Group entries by change_type, preserving supplied order within each group.
        let mut grouped: HashMap<&str, Vec<f32>> = HashMap::new();
        for entry in cache {
            // Skip entries with undefined ratios.
            if entry.expected_error_reduction == 0.0 {
                continue;
            }
            let ratio = entry.actual_error_reduction / entry.expected_error_reduction;
            // Guard against NaN or infinities leaking into the EWMA.
            if !ratio.is_finite() {
                continue;
            }
            grouped
                .entry(entry.change_type.as_str())
                .or_default()
                .push(ratio);
        }

        let mut corrections: HashMap<String, f32> = HashMap::new();
        for (change_type, ratios) in grouped {
            let ewma = ewma(&ratios, CALIBRATION_CORRECTION_EWMA_ALPHA);
            let clamped = ewma.clamp(MIN_CALIBRATION_CORRECTION, NEUTRAL_CORRECTION);
            corrections.insert(change_type.to_string(), clamped);
        }

        Self { corrections }
    }

    /// Look up the correction factor for a given change-type.
    ///
    /// Returns [`NEUTRAL_CORRECTION`] (1.0) when the change-type is absent —
    /// this preserves the compiled calibration constant unchanged for
    /// change-types with no recent failure data.
    #[must_use]
    pub fn get_correction(&self, change_type: &str) -> f32 {
        self.corrections
            .get(change_type)
            .copied()
            .unwrap_or(NEUTRAL_CORRECTION)
    }

    /// Returns a reference to the underlying correction map, for metadata
    /// export at the FFI boundary.
    #[must_use]
    pub fn as_map(&self) -> &HashMap<String, f32> {
        &self.corrections
    }

    /// True when no change-type has a correction recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.corrections.is_empty()
    }
}

// =============================================================================
// EWMA helper
// =============================================================================

/// Compute the exponentially-weighted moving average of a slice.
///
/// Starts from the first observation (no seed bias) and applies
/// `ewma = (1 - α) * prev + α * x` for each subsequent observation. Returns
/// [`NEUTRAL_CORRECTION`] for empty input.
fn ewma(values: &[f32], alpha: f32) -> f32 {
    if values.is_empty() {
        return NEUTRAL_CORRECTION;
    }
    let mut acc = values[0];
    for &x in &values[1..] {
        acc = (1.0 - alpha) * acc + alpha * x;
    }
    acc
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(change_type: &str, predicted: f32, actual: f32) -> FailureCacheEntry {
        FailureCacheEntry {
            change_type: change_type.to_string(),
            expected_error_reduction: predicted,
            actual_error_reduction: actual,
        }
    }

    #[test]
    fn empty_cache_returns_neutral_correction() {
        let correction = CalibrationCorrection::from_failure_cache(&[]);
        assert!(correction.is_empty());
        assert!(
            (correction.get_correction(CHANGE_TYPE_ADD_NEURONS) - NEUTRAL_CORRECTION).abs() < 1e-9
        );
        assert!(
            (correction.get_correction(CHANGE_TYPE_ADD_SYNAPSES) - NEUTRAL_CORRECTION).abs() < 1e-9
        );
    }

    #[test]
    fn all_negative_actuals_clamp_to_minimum() {
        // Every entry is a net-harm outcome; raw ratio is negative.
        let cache: Vec<FailureCacheEntry> = (0..10)
            .map(|_| entry(CHANGE_TYPE_ADD_NEURONS, 0.001, -0.0005))
            .collect();
        let correction = CalibrationCorrection::from_failure_cache(&cache);
        let value = correction.get_correction(CHANGE_TYPE_ADD_NEURONS);
        assert!(
            (value - MIN_CALIBRATION_CORRECTION).abs() < 1e-9,
            "expected MIN_CALIBRATION_CORRECTION (={MIN_CALIBRATION_CORRECTION}), got {value}"
        );
    }

    #[test]
    fn uniform_cache_returns_that_mean_ratio() {
        // 20 entries with identical ratio of 1/1000. EWMA of a constant sequence
        // is that constant.
        let cache: Vec<FailureCacheEntry> = (0..20)
            .map(|_| entry(CHANGE_TYPE_ADD_NEURONS, 0.001, 0.000_001))
            .collect();
        let correction = CalibrationCorrection::from_failure_cache(&cache);
        let value = correction.get_correction(CHANGE_TYPE_ADD_NEURONS);
        // Expected ratio = 0.000001 / 0.001 = 0.001, which equals the floor.
        assert!(
            (value - 0.001).abs() < 1e-6,
            "expected ≈ 0.001, got {value}"
        );
    }

    #[test]
    fn mixed_cache_returns_ewma_of_ratios() {
        // Alternating entries: ratios of 0.5 and 0.1. Final EWMA value with
        // α = 0.3 is deterministic and can be recomputed here.
        let cache = vec![
            entry(CHANGE_TYPE_ADD_SYNAPSES, 1.0, 0.5),
            entry(CHANGE_TYPE_ADD_SYNAPSES, 1.0, 0.1),
            entry(CHANGE_TYPE_ADD_SYNAPSES, 1.0, 0.5),
            entry(CHANGE_TYPE_ADD_SYNAPSES, 1.0, 0.1),
        ];
        let correction = CalibrationCorrection::from_failure_cache(&cache);
        let value = correction.get_correction(CHANGE_TYPE_ADD_SYNAPSES);

        // Manual EWMA with α = 0.3:
        //   start = 0.5
        //   step1 = 0.7*0.5 + 0.3*0.1 = 0.38
        //   step2 = 0.7*0.38 + 0.3*0.5 = 0.416
        //   step3 = 0.7*0.416 + 0.3*0.1 = 0.3212
        let expected = 0.321_2_f32;
        assert!(
            (value - expected).abs() < 1e-4,
            "expected ≈ {expected}, got {value}"
        );
    }

    #[test]
    fn zero_predicted_entries_are_skipped() {
        // A zero-predicted entry has an undefined ratio and must not affect the
        // correction. Mix one in amongst legitimate entries.
        let cache = vec![
            entry(CHANGE_TYPE_ADD_NEURONS, 0.0, 0.0),
            entry(CHANGE_TYPE_ADD_NEURONS, 0.01, 0.005),
            entry(CHANGE_TYPE_ADD_NEURONS, 0.01, 0.005),
        ];
        let correction = CalibrationCorrection::from_failure_cache(&cache);
        let value = correction.get_correction(CHANGE_TYPE_ADD_NEURONS);
        // Both usable entries have ratio 0.5, so EWMA = 0.5.
        assert!((value - 0.5).abs() < 1e-6, "expected 0.5, got {value}");
    }

    #[test]
    fn non_finite_ratios_are_ignored() {
        // Infinite or NaN ratios (e.g., extremely tiny predicted near-zero) must
        // not contaminate the EWMA.
        let cache = vec![
            entry(CHANGE_TYPE_ADD_NEURONS, f32::MIN_POSITIVE, f32::MAX),
            entry(CHANGE_TYPE_ADD_NEURONS, 0.01, 0.002),
            entry(CHANGE_TYPE_ADD_NEURONS, 0.01, 0.002),
        ];
        let correction = CalibrationCorrection::from_failure_cache(&cache);
        let value = correction.get_correction(CHANGE_TYPE_ADD_NEURONS);
        assert!((value - 0.2).abs() < 1e-6, "expected 0.2, got {value}");
    }

    #[test]
    fn different_change_types_are_independent() {
        let cache = vec![
            entry(CHANGE_TYPE_ADD_NEURONS, 1.0, 0.2),
            entry(CHANGE_TYPE_ADD_NEURONS, 1.0, 0.2),
            entry(CHANGE_TYPE_ADD_SYNAPSES, 1.0, 0.8),
            entry(CHANGE_TYPE_ADD_SYNAPSES, 1.0, 0.8),
        ];
        let correction = CalibrationCorrection::from_failure_cache(&cache);
        assert!((correction.get_correction(CHANGE_TYPE_ADD_NEURONS) - 0.2).abs() < 1e-6);
        assert!((correction.get_correction(CHANGE_TYPE_ADD_SYNAPSES) - 0.8).abs() < 1e-6);
        // Unknown change-type falls back to neutral.
        assert!(
            (correction.get_correction(CHANGE_TYPE_COORDINATED_STRUCTURAL) - NEUTRAL_CORRECTION)
                .abs()
                < 1e-9
        );
    }

    #[test]
    fn correction_above_one_clamps_to_neutral() {
        // A lucky run where actual exceeded predicted must never inflate the
        // base calibration constant — the correction only ever discounts.
        let cache = vec![
            entry(CHANGE_TYPE_ADD_NEURONS, 0.001, 0.01),
            entry(CHANGE_TYPE_ADD_NEURONS, 0.001, 0.01),
        ];
        let correction = CalibrationCorrection::from_failure_cache(&cache);
        let value = correction.get_correction(CHANGE_TYPE_ADD_NEURONS);
        assert!(
            (value - NEUTRAL_CORRECTION).abs() < 1e-9,
            "expected NEUTRAL_CORRECTION (=1.0), got {value}"
        );
    }

    #[test]
    fn deterministic_for_same_cache() {
        let cache = vec![
            entry(CHANGE_TYPE_ADD_NEURONS, 0.01, 0.001),
            entry(CHANGE_TYPE_ADD_NEURONS, 0.01, 0.002),
            entry(CHANGE_TYPE_ADD_NEURONS, 0.01, 0.001),
        ];
        let a = CalibrationCorrection::from_failure_cache(&cache);
        let b = CalibrationCorrection::from_failure_cache(&cache);
        assert!(
            (a.get_correction(CHANGE_TYPE_ADD_NEURONS) - b.get_correction(CHANGE_TYPE_ADD_NEURONS))
                .abs()
                < 1e-12
        );
    }
}
