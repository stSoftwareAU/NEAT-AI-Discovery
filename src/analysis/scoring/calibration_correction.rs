//! Per-change-type prediction calibration correction (Issue #1131, #1162).
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
//! ## Per-(change-type, target-squash) tracking (Issue #1162)
//!
//! In practice the prediction-vs-actual gap depends strongly on the target
//! neuron's activation function. Failures against an unbounded squash like
//! `SELU` can over-estimate by 1000× while bounded squashes such as
//! `HARD_TANH` over-estimate by a much smaller factor. Grouping failure
//! entries by `change_type` alone loses that signal.
//!
//! When the failure-cache entries supply `targetNeuronInfo.squash`, this
//! module also computes a *specific* correction keyed by
//! `(change_type, target_squash)`. The lookup
//! [`CalibrationCorrection::correction_for`] returns the specific value
//! when at least [`MIN_SPECIFIC_TARGET_SQUASH_SAMPLES`] usable entries are
//! available, otherwise it falls back to the per-`change_type` correction,
//! and finally to [`NEUTRAL_CORRECTION`] (= 1.0) when neither is known.
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

/// Minimum number of usable failure-cache entries required for a
/// `(change_type, target_squash)` group before its specific correction is
/// preferred over the per-`change_type` fallback (Issue #1162).
///
/// Three entries is enough to establish that a (`change_type`, squash)
/// bucket is consistently mis-calibrated without leaking a single noisy
/// outlier into the per-candidate prediction. Smaller groups fall through
/// to the per-`change_type` correction.
pub const MIN_SPECIFIC_TARGET_SQUASH_SAMPLES: usize = 3;

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
///
/// `target_squash` is the activation function of the target neuron the
/// candidate was applied to (Issue #1162). It is parsed from
/// `targetNeuronInfo.squash` in the failure-cache JSON when present and is
/// `None` for legacy entries that omit it. When supplied it lets the
/// calibration learn that, say, `add-neurons` against `SELU` targets needs a
/// much smaller correction multiplier than the generic `add-neurons` group.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", from = "FailureCacheEntryRaw")]
pub struct FailureCacheEntry {
    /// Candidate classification (e.g., `add-neurons`, `add-synapses`,
    /// `coordinated-structural`). Used as the correction key.
    pub change_type: String,

    /// What we predicted — the `expectedErrorReduction` at the time of proposal.
    pub expected_error_reduction: f32,

    /// What actually happened — the post-apply `actualErrorReduction`. May be
    /// negative for candidates that harmed the network.
    pub actual_error_reduction: f32,

    /// Target neuron's activation function (squash), when reported by the
    /// upstream emitter. Populated from `targetNeuronInfo.squash` in the
    /// failure-cache JSON, or from a top-level `targetSquash` field.
    /// `None` for entries that lack target metadata, preserving backward
    /// compatibility (Issue #1162).
    #[serde(default)]
    pub target_squash: Option<String>,
}

/// Wire-format helper for [`FailureCacheEntry`] (Issue #1162).
///
/// Allows the failure JSON to either nest the squash under
/// `targetNeuronInfo` (the upstream NEAT-AI shape) or supply a top-level
/// `targetSquash` field, while keeping the in-memory struct flat.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct FailureCacheEntryRaw {
    change_type: String,
    expected_error_reduction: f32,
    actual_error_reduction: f32,
    #[serde(default)]
    target_neuron_info: Option<TargetNeuronInfoRaw>,
    #[serde(default)]
    target_squash: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct TargetNeuronInfoRaw {
    #[serde(default)]
    squash: Option<String>,
}

impl From<FailureCacheEntryRaw> for FailureCacheEntry {
    fn from(raw: FailureCacheEntryRaw) -> Self {
        // Prefer an explicit top-level `targetSquash`, fall back to the
        // nested `targetNeuronInfo.squash` shape that NEAT-AI emits.
        let target_squash = raw
            .target_squash
            .or_else(|| raw.target_neuron_info.and_then(|info| info.squash));
        Self {
            change_type: raw.change_type,
            expected_error_reduction: raw.expected_error_reduction,
            actual_error_reduction: raw.actual_error_reduction,
            target_squash,
        }
    }
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
///
/// Two layers of correction are tracked (Issue #1162):
///
/// - `corrections` — keyed by `change_type` alone. Computed from every
///   usable entry of that change-type. This is the **fallback** layer used
///   when no specific correction is available for the requested target
///   squash.
/// - `specific_corrections` — keyed by `(change_type, target_squash)`.
///   Only populated for groups with at least
///   [`MIN_SPECIFIC_TARGET_SQUASH_SAMPLES`] usable entries, so that a single
///   noisy outcome cannot dominate per-candidate calibration. This is the
///   **specific** layer queried by [`Self::correction_for`].
#[derive(Debug, Clone, Default)]
pub struct CalibrationCorrection {
    corrections: HashMap<String, f32>,
    specific_corrections: HashMap<(String, String), f32>,
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
    /// 1. Group entries by `change_type` (fallback layer) and by
    ///    `(change_type, target_squash)` (specific layer, Issue #1162).
    /// 2. Skip entries with `expected_error_reduction == 0` (undefined ratio)
    ///    and entries whose ratio is non-finite.
    /// 3. For each group, compute an EWMA of `actual / expected` in the
    ///    order supplied. Callers pass entries oldest-first; the final EWMA
    ///    value therefore reflects the most recent outcomes most strongly.
    /// 4. Clamp the EWMA to
    ///    `[MIN_CALIBRATION_CORRECTION, NEUTRAL_CORRECTION]`.
    /// 5. Only specific groups with at least
    ///    [`MIN_SPECIFIC_TARGET_SQUASH_SAMPLES`] usable entries are kept; the
    ///    rest fall through to the per-`change_type` fallback at lookup.
    ///
    /// Empty caches, or caches with no usable entries for a change-type,
    /// produce a [`Self::neutral`] correction for that change-type on lookup.
    #[must_use]
    pub fn from_failure_cache(cache: &[FailureCacheEntry]) -> Self {
        // Group entries by change_type (fallback) and by (change_type,
        // target_squash) (specific), preserving supplied order within each
        // group so the EWMA reflects oldest -> newest.
        let mut grouped: HashMap<&str, Vec<f32>> = HashMap::new();
        let mut grouped_specific: HashMap<(&str, &str), Vec<f32>> = HashMap::new();
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
            if let Some(squash) = entry.target_squash.as_deref() {
                grouped_specific
                    .entry((entry.change_type.as_str(), squash))
                    .or_default()
                    .push(ratio);
            }
        }

        let mut corrections: HashMap<String, f32> = HashMap::new();
        for (change_type, ratios) in grouped {
            let ewma = ewma(&ratios, CALIBRATION_CORRECTION_EWMA_ALPHA);
            let clamped = ewma.clamp(MIN_CALIBRATION_CORRECTION, NEUTRAL_CORRECTION);
            corrections.insert(change_type.to_string(), clamped);
        }

        let mut specific_corrections: HashMap<(String, String), f32> = HashMap::new();
        for ((change_type, squash), ratios) in grouped_specific {
            // Issue #1162: only retain specific corrections backed by enough
            // samples; smaller groups fall through to the per-change_type
            // fallback at lookup time.
            if ratios.len() < MIN_SPECIFIC_TARGET_SQUASH_SAMPLES {
                continue;
            }
            let ewma = ewma(&ratios, CALIBRATION_CORRECTION_EWMA_ALPHA);
            let clamped = ewma.clamp(MIN_CALIBRATION_CORRECTION, NEUTRAL_CORRECTION);
            specific_corrections.insert((change_type.to_string(), squash.to_string()), clamped);
        }

        Self {
            corrections,
            specific_corrections,
        }
    }

    /// Look up the correction factor for a given change-type.
    ///
    /// Returns [`NEUTRAL_CORRECTION`] (1.0) when the change-type is absent —
    /// this preserves the compiled calibration constant unchanged for
    /// change-types with no recent failure data.
    ///
    /// This is the per-`change_type` fallback. Prefer
    /// [`Self::correction_for`] when the candidate's target squash is known.
    #[must_use]
    pub fn get_correction(&self, change_type: &str) -> f32 {
        self.corrections
            .get(change_type)
            .copied()
            .unwrap_or(NEUTRAL_CORRECTION)
    }

    /// Look up the most specific available correction factor (Issue #1162).
    ///
    /// Resolution order:
    ///
    /// 1. If `target_squash` is supplied **and** at least
    ///    [`MIN_SPECIFIC_TARGET_SQUASH_SAMPLES`] usable failure-cache entries
    ///    exist for `(change_type, target_squash)`, return that specific
    ///    correction.
    /// 2. Otherwise, return the per-`change_type` correction (existing
    ///    behaviour).
    /// 3. Otherwise, return [`NEUTRAL_CORRECTION`] (= 1.0) so the compiled
    ///    calibration constant is used unchanged.
    ///
    /// Callers should pass `None` for `target_squash` when the candidate's
    /// target squash is unknown.
    #[must_use]
    pub fn correction_for(&self, change_type: &str, target_squash: Option<&str>) -> f32 {
        if let Some(squash) = target_squash
            && let Some(value) = self
                .specific_corrections
                .get(&(change_type.to_string(), squash.to_string()))
                .copied()
        {
            return value;
        }
        self.get_correction(change_type)
    }

    /// Returns a reference to the underlying correction map, for metadata
    /// export at the FFI boundary.
    #[must_use]
    pub fn as_map(&self) -> &HashMap<String, f32> {
        &self.corrections
    }

    /// Returns a reference to the per-(`change_type`, `target_squash`) map
    /// (Issue #1162). Exposed for diagnostic tests; the FFI metadata only
    /// includes the per-`change_type` map for backward compatibility.
    #[must_use]
    pub fn specific_as_map(&self) -> &HashMap<(String, String), f32> {
        &self.specific_corrections
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
            target_squash: None,
        }
    }

    fn entry_with_squash(
        change_type: &str,
        predicted: f32,
        actual: f32,
        squash: &str,
    ) -> FailureCacheEntry {
        FailureCacheEntry {
            change_type: change_type.to_string(),
            expected_error_reduction: predicted,
            actual_error_reduction: actual,
            target_squash: Some(squash.to_string()),
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

    // =========================================================================
    // Issue #1162 — per-(change_type, target_squash) calibration tracking
    // =========================================================================

    #[test]
    fn only_change_type_data_is_used_as_fallback() {
        // No entry carries a target_squash, so the specific layer is empty
        // and `correction_for` falls back to the per-change_type value.
        let cache: Vec<FailureCacheEntry> = (0..6)
            .map(|_| entry(CHANGE_TYPE_ADD_NEURONS, 1.0, 0.4))
            .collect();
        let correction = CalibrationCorrection::from_failure_cache(&cache);
        assert!(correction.specific_as_map().is_empty());

        // Specific lookup with any squash falls back to the change_type EWMA.
        let value = correction.correction_for(CHANGE_TYPE_ADD_NEURONS, Some("SELU"));
        assert!(
            (value - 0.4).abs() < 1e-6,
            "expected fallback ≈ 0.4, got {value}"
        );

        // Without a target_squash the result also matches the change_type EWMA.
        let none_value = correction.correction_for(CHANGE_TYPE_ADD_NEURONS, None);
        assert!((none_value - 0.4).abs() < 1e-6);
    }

    #[test]
    fn specific_data_is_preferred_when_threshold_is_met() {
        // Three or more entries against SELU establish a specific correction
        // distinct from the change_type fallback.
        let mut cache = vec![
            // Generic add-neurons history dominated by mid-range outcomes.
            entry(CHANGE_TYPE_ADD_NEURONS, 1.0, 0.5),
            entry(CHANGE_TYPE_ADD_NEURONS, 1.0, 0.5),
            entry(CHANGE_TYPE_ADD_NEURONS, 1.0, 0.5),
        ];
        // SELU-specific entries that consistently massively over-estimate.
        for _ in 0..MIN_SPECIFIC_TARGET_SQUASH_SAMPLES {
            cache.push(entry_with_squash(
                CHANGE_TYPE_ADD_NEURONS,
                1.0,
                0.001,
                "SELU",
            ));
        }
        let correction = CalibrationCorrection::from_failure_cache(&cache);

        // Generic fallback reflects the union of both groups (SELU entries
        // also feed the per-change_type EWMA), but `correction_for(SELU)`
        // must return the SELU-specific value (≈ 0.001) rather than the
        // higher fallback.
        let selu = correction.correction_for(CHANGE_TYPE_ADD_NEURONS, Some("SELU"));
        assert!(
            (selu - 0.001).abs() < 1e-4,
            "expected SELU-specific ≈ 0.001, got {selu}"
        );

        // A different squash with no specific data falls back.
        let fallback = correction.correction_for(CHANGE_TYPE_ADD_NEURONS, Some("HARD_TANH"));
        let direct_fallback = correction.get_correction(CHANGE_TYPE_ADD_NEURONS);
        assert!(
            (fallback - direct_fallback).abs() < 1e-9,
            "expected HARD_TANH to fall back to {direct_fallback}, got {fallback}"
        );
        // The fallback should be larger than the SELU-specific value.
        assert!(
            fallback > selu,
            "fallback ({fallback}) should exceed SELU-specific ({selu})"
        );
    }

    #[test]
    fn insufficient_specific_samples_fall_back_to_change_type() {
        // Fewer than MIN_SPECIFIC_TARGET_SQUASH_SAMPLES SELU entries — the
        // specific bucket must NOT be retained, even though the data exists.
        const _: () = assert!(MIN_SPECIFIC_TARGET_SQUASH_SAMPLES >= 2);
        let mut cache = vec![
            entry(CHANGE_TYPE_ADD_NEURONS, 1.0, 0.5),
            entry(CHANGE_TYPE_ADD_NEURONS, 1.0, 0.5),
            entry(CHANGE_TYPE_ADD_NEURONS, 1.0, 0.5),
        ];
        for _ in 0..(MIN_SPECIFIC_TARGET_SQUASH_SAMPLES - 1) {
            cache.push(entry_with_squash(
                CHANGE_TYPE_ADD_NEURONS,
                1.0,
                0.001,
                "SELU",
            ));
        }
        let correction = CalibrationCorrection::from_failure_cache(&cache);
        assert!(
            correction.specific_as_map().is_empty(),
            "specific corrections must not be retained below the sample threshold"
        );

        // Lookup with the SELU squash returns the per-change_type fallback.
        let lookup = correction.correction_for(CHANGE_TYPE_ADD_NEURONS, Some("SELU"));
        let fallback = correction.get_correction(CHANGE_TYPE_ADD_NEURONS);
        assert!(
            (lookup - fallback).abs() < 1e-9,
            "expected fallback {fallback}, got {lookup}"
        );
    }

    #[test]
    fn nan_and_zero_divisor_specific_entries_are_skipped() {
        // The two zero / non-finite entries against SELU must NOT count toward
        // the sample threshold, so the SELU-specific bucket stays empty and
        // lookups fall back to the per-change_type value.
        let cache = vec![
            // Zero predicted -> undefined ratio.
            entry_with_squash(CHANGE_TYPE_ADD_NEURONS, 0.0, 0.0, "SELU"),
            // Non-finite ratio.
            entry_with_squash(CHANGE_TYPE_ADD_NEURONS, f32::MIN_POSITIVE, f32::MAX, "SELU"),
            // One legitimate SELU entry — still below the threshold.
            entry_with_squash(CHANGE_TYPE_ADD_NEURONS, 1.0, 0.001, "SELU"),
            // Generic change_type entries to seed the fallback.
            entry(CHANGE_TYPE_ADD_NEURONS, 1.0, 0.4),
            entry(CHANGE_TYPE_ADD_NEURONS, 1.0, 0.4),
        ];
        let correction = CalibrationCorrection::from_failure_cache(&cache);
        assert!(
            correction.specific_as_map().is_empty(),
            "non-finite / zero entries must not count toward the specific threshold"
        );

        let lookup = correction.correction_for(CHANGE_TYPE_ADD_NEURONS, Some("SELU"));
        let fallback = correction.get_correction(CHANGE_TYPE_ADD_NEURONS);
        assert!(
            (lookup - fallback).abs() < 1e-9,
            "non-finite / zero specific entries must not affect lookup"
        );
    }

    #[test]
    fn target_squash_parsed_from_target_neuron_info() {
        // Issue #1162: failure-cache JSON nests squash under
        // `targetNeuronInfo.squash`. Round-trip a representative payload.
        let json = r#"{
            "changeType": "add-neurons",
            "expectedErrorReduction": 0.001,
            "actualErrorReduction": 0.000001,
            "targetNeuronInfo": { "squash": "SELU" }
        }"#;
        let parsed: FailureCacheEntry = serde_json::from_str(json).expect("parse");
        assert_eq!(parsed.change_type, CHANGE_TYPE_ADD_NEURONS);
        assert_eq!(parsed.target_squash.as_deref(), Some("SELU"));
    }

    #[test]
    fn target_squash_optional_for_legacy_entries() {
        // Legacy entries that omit `targetNeuronInfo` must still parse.
        let json = r#"{
            "changeType": "add-neurons",
            "expectedErrorReduction": 0.001,
            "actualErrorReduction": 0.000001
        }"#;
        let parsed: FailureCacheEntry = serde_json::from_str(json).expect("parse");
        assert!(parsed.target_squash.is_none());
    }

    #[test]
    fn correction_for_returns_neutral_when_nothing_known() {
        let correction = CalibrationCorrection::neutral();
        let value = correction.correction_for(CHANGE_TYPE_ADD_NEURONS, Some("SELU"));
        assert!(
            (value - NEUTRAL_CORRECTION).abs() < 1e-9,
            "expected NEUTRAL_CORRECTION, got {value}"
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
