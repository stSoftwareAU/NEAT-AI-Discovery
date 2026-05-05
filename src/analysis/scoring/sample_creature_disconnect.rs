//! Sample-vs-creature disconnect detector (Issue #1195).
//!
//! When a candidate's per-sample improvement counter is near 1.0 but the
//! aggregated creature-level `actual_error_reduction` is non-positive, the
//! prediction model has produced small per-sample gains that did not
//! aggregate into a creature-level improvement. This is the diagnostic
//! signature of the failure cluster discussed in #1189 and #1195.
//!
//! Catching this pattern explicitly — rather than waiting for the
//! [`super::calibration_correction::CalibrationCorrection`] EWMA to learn
//! it slowly — accelerates the calibration's response to genuinely broken
//! prediction families. When the detector fires the offending
//! `(change_type, target_squash, variant_key)` triple is recorded as a
//! "disconnect penalty" inside the calibration correction; subsequent
//! lookups for that triple are demoted by
//! [`SAMPLE_DISCONNECT_PENALTY`] (`0.5`) per disconnect, clamped to the
//! existing
//! [`super::calibration_correction::MIN_CALIBRATION_CORRECTION`] floor.
//!
//! ## Thresholds
//!
//! - [`SAMPLE_DISCONNECT_RATIO_THRESHOLD`] (default `0.95`): the minimum
//!   `improved_count / total_count` ratio required to consider a candidate.
//! - The actual aggregate must satisfy
//!   `actual_error_reduction <= 0.0` for the detector to fire.
//!
//! Both bounds are tight by design: the detector targets the specific
//! failure mode where per-sample gains are nearly universal but
//! creature-level gain is absent. Looser thresholds would conflate this
//! pattern with the slower EWMA-learnable miscalibration that the existing
//! correction layers already handle.

use super::calibration_correction::FailureCacheEntry;

/// Minimum `improved_count / total_count` ratio (inclusive) required for
/// the detector to fire (Issue #1195).
///
/// 0.95 was chosen because the failure cluster cited in #1195 shows three
/// candidates each at ≈ 99.7 % per-sample improvement; a 0.95 threshold
/// captures that pattern while excluding the noisier mid-range candidates
/// that the existing EWMA already governs.
pub const SAMPLE_DISCONNECT_RATIO_THRESHOLD: f32 = 0.95;

/// Multiplicative penalty applied to the
/// `(change_type, target_squash, variant_key)` calibration entry each
/// time the detector fires (Issue #1195).
///
/// 0.5 halves the surviving correction per detected disconnect. Repeated
/// disconnects compound multiplicatively but are bounded by
/// [`super::calibration_correction::MIN_CALIBRATION_CORRECTION`]
/// (= `0.001`) at the lower clamp, so a noisy stream of failures cannot
/// collapse the calibration to zero.
pub const SAMPLE_DISCONNECT_PENALTY: f32 = 0.5;

/// True when the detector should fire for the supplied per-candidate
/// statistics (Issue #1195).
///
/// Returns `false` whenever any of the inputs are unusable:
/// - `total_count == 0` (no samples were evaluated).
/// - `actual_error_reduction` is non-finite (cannot be compared).
/// - `improved_count > total_count` (corrupt counters).
///
/// In every other case the function returns
/// `improved_ratio >= SAMPLE_DISCONNECT_RATIO_THRESHOLD &&
/// actual_error_reduction <= 0.0`.
#[must_use]
#[allow(clippy::cast_precision_loss)] // u32 -> f32 is exact for the sample counts we expect.
pub fn detect_disconnect(
    improved_count: u32,
    total_count: u32,
    actual_error_reduction: f32,
) -> bool {
    if total_count == 0 || improved_count > total_count {
        return false;
    }
    if !actual_error_reduction.is_finite() {
        return false;
    }
    let ratio = improved_count as f32 / total_count as f32;
    ratio >= SAMPLE_DISCONNECT_RATIO_THRESHOLD && actual_error_reduction <= 0.0
}

/// Convenience wrapper that runs [`detect_disconnect`] over a
/// [`FailureCacheEntry`]. Returns `false` when the entry omits per-sample
/// statistics (legacy entries) — the detector cannot fire without
/// `improved_count` and `total_count`.
#[must_use]
pub fn detect_disconnect_entry(entry: &FailureCacheEntry) -> bool {
    match (entry.improved_count, entry.total_count) {
        (Some(improved), Some(total)) => {
            detect_disconnect(improved, total, entry.actual_error_reduction)
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry_with_counts(
        change_type: &str,
        squash: &str,
        variant: &str,
        actual_error_reduction: f32,
        improved: Option<u32>,
        total: Option<u32>,
    ) -> FailureCacheEntry {
        FailureCacheEntry {
            change_type: change_type.to_string(),
            expected_error_reduction: 0.001,
            actual_error_reduction,
            target_squash: Some(squash.to_string()),
            variant_key: Some(variant.to_string()),
            target_uuid: None,
            improved_count: improved,
            total_count: total,
        }
    }

    #[test]
    fn fires_for_high_ratio_and_negative_reduction() {
        // 1032 / 1036 ≈ 99.6 %, actual = -0.01 — the canonical pattern.
        assert!(detect_disconnect(1032, 1036, -0.01));
    }

    #[test]
    fn fires_for_high_ratio_and_zero_reduction() {
        // Boundary: actual_error_reduction = 0 still counts as non-positive.
        assert!(detect_disconnect(1032, 1036, 0.0));
    }

    #[test]
    fn does_not_fire_for_high_ratio_and_positive_reduction() {
        // High improved_count but a real positive aggregate — no disconnect.
        assert!(!detect_disconnect(1000, 1036, 0.001));
    }

    #[test]
    fn does_not_fire_for_low_ratio_and_negative_reduction() {
        // Negative reduction but only 50 % of samples improved — the
        // standard EWMA path handles this; the disconnect detector must
        // remain silent.
        assert!(!detect_disconnect(518, 1036, -0.01));
    }

    #[test]
    fn does_not_fire_at_exactly_the_threshold_boundary_minus_epsilon() {
        // Just below the threshold (0.94999…): no disconnect.
        // 949 / 1000 = 0.949
        assert!(!detect_disconnect(949, 1000, -0.01));
        // 950 / 1000 = 0.95 — at the threshold, fires.
        assert!(detect_disconnect(950, 1000, -0.01));
    }

    #[test]
    fn does_not_fire_for_zero_total_count() {
        assert!(!detect_disconnect(0, 0, -0.01));
    }

    #[test]
    fn does_not_fire_for_corrupt_counters() {
        // improved_count > total_count is treated as corrupt input.
        assert!(!detect_disconnect(1100, 1000, -0.01));
    }

    #[test]
    fn does_not_fire_for_non_finite_reduction() {
        assert!(!detect_disconnect(1032, 1036, f32::NAN));
        assert!(!detect_disconnect(1032, 1036, f32::INFINITY));
    }

    #[test]
    fn entry_helper_returns_false_for_legacy_entries() {
        // Legacy entries lack improved_count / total_count and must not
        // trigger the detector.
        let mut e = entry_with_counts("add-neurons", "SINE", "v2", -0.01, None, None);
        assert!(!detect_disconnect_entry(&e));
        // Setting only one half is also insufficient.
        e.improved_count = Some(1032);
        e.total_count = None;
        assert!(!detect_disconnect_entry(&e));
        e.improved_count = None;
        e.total_count = Some(1036);
        assert!(!detect_disconnect_entry(&e));
    }

    #[test]
    fn entry_helper_fires_on_full_pattern() {
        let e = entry_with_counts(
            "add-neurons",
            "SINE",
            "v2_add-neurons_demo",
            -0.01,
            Some(1032),
            Some(1036),
        );
        assert!(detect_disconnect_entry(&e));
    }
}
