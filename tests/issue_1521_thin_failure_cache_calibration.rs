//! Issue #1521 — a thin (2-entry) failure cache must NOT starve the
//! #1131/#1162 calibration EWMA into suppressing the remove-neuron candidate
//! stream.
//!
//! ## Investigation summary (negative result on the "starvation" hypothesis)
//!
//! The parent milestone (#1516) observed that the failure cache for creature
//! `247b83ab` holds only 2 entries and hypothesised that this thin cache
//! starves the per-change-type calibration EWMA, suppressing remove-neuron
//! candidate suggestion. This investigation shows the thin cache does **not**
//! starve the EWMA, because two existing design safeguards in
//! `src/analysis/scoring/calibration_correction.rs` make a thin cache safe:
//!
//! 1. **Correction floor / clamp.** [`CalibrationCorrection::from_failure_cache`]
//!    clamps every per-change-type EWMA to
//!    `[MIN_CALIBRATION_CORRECTION, NEUTRAL_CORRECTION]` = `[0.001, 1.0]`. Even a
//!    2-entry cache dominated by ~800× over-predictions produces a correction of
//!    `0.001` — heavily discounted but strictly non-zero. Because
//!    `apply_prediction_calibration` is a plain multiply, the corrected gain
//!    stays strictly positive, so the candidate stream is never collapsed to
//!    zero by the calibration layer.
//!
//! 2. **Graceful fallback below the specific-sample threshold.** The per-
//!    `(change_type, target_squash)` specific layer requires
//!    [`MIN_SPECIFIC_TARGET_SQUASH_SAMPLES`] (= 3) entries. With only 2 the
//!    specific layer stays empty and [`CalibrationCorrection::correction_for`]
//!    falls back to the per-`change_type` EWMA, which any single usable entry
//!    already populates. A thin cache therefore degrades gracefully rather than
//!    leaving the correction undefined.
//!
//! Low candidate throughput was driven by the **placeholder gain** (parent
//! #1516), addressed by the propagation-aware estimate in #1517/#1518 — not by
//! the thin failure cache. These tests are the documented warm-up fallback the
//! issue calls for: they lock in that a 2-entry remove-neuron cache still yields
//! a non-zero, floored correction and a strictly-positive corrected candidate
//! stream, so a future regression (removing the floor, or gating corrections
//! behind a minimum entry count) that re-suppresses candidates fails here.

#![allow(clippy::cast_precision_loss)]

use neat_ai_discovery::analysis::constants::COORDINATED_PREDICTION_CALIBRATION;
use neat_ai_discovery::analysis::scoring::calibration_correction::{
    CHANGE_TYPE_REMOVE_NEURON, CalibrationCorrection, FailureCacheEntry,
    MIN_CALIBRATION_CORRECTION, MIN_SPECIFIC_TARGET_SQUASH_SAMPLES, NEUTRAL_CORRECTION,
};

/// A single failure-cache entry mirroring the `247b83ab` evidence shape:
/// predicted ≈ 0.166, actual ≈ 0 (slightly negative) — a ~800× over-prediction.
fn remove_neuron_failure() -> FailureCacheEntry {
    FailureCacheEntry {
        change_type: CHANGE_TYPE_REMOVE_NEURON.to_string(),
        expected_error_reduction: 0.165_710_48,
        actual_error_reduction: -0.000_207_19,
        target_squash: None,
        variant_key: None,
        target_uuid: None,
        improved_count: None,
        total_count: None,
        age_epochs: None,
    }
}

/// Build the thin, 2-entry failure cache described in the parent issue.
fn thin_two_entry_cache() -> Vec<FailureCacheEntry> {
    vec![remove_neuron_failure(), remove_neuron_failure()]
}

/// Core guarantee: a 2-entry remove-neuron failure cache still populates the
/// per-change-type correction with a strictly-positive, floored value — the
/// EWMA is not starved into silence by the thin cache.
#[test]
fn thin_two_entry_cache_yields_non_zero_floored_correction() {
    let cache = thin_two_entry_cache();
    assert_eq!(
        cache.len(),
        2,
        "guard: this exercises the thin 2-entry cache"
    );

    let correction = CalibrationCorrection::from_failure_cache(&cache);

    // The per-change-type layer IS populated from just 2 entries.
    assert!(
        !correction.is_empty(),
        "a 2-entry cache must still populate the per-change-type correction"
    );

    let learned = correction.correction_for(CHANGE_TYPE_REMOVE_NEURON, None);
    assert!(
        learned.is_finite() && learned > 0.0,
        "correction must be finite and strictly positive, got {learned}"
    );
    // ~800× over-prediction with only 2 entries clamps to the floor, not zero.
    assert!(
        (learned - MIN_CALIBRATION_CORRECTION).abs() < 1e-6,
        "expected the correction floored at {MIN_CALIBRATION_CORRECTION}, got {learned}"
    );
}

/// The corrected candidate gain stays strictly positive: the calibration layer
/// discounts the fabricated gain but never collapses the stream to zero.
#[test]
fn thin_cache_corrected_gain_is_non_zero() {
    let correction = CalibrationCorrection::from_failure_cache(&thin_two_entry_cache());
    let learned = correction.correction_for(CHANGE_TYPE_REMOVE_NEURON, None);

    let raw_gain = 0.165_710_48_f32;
    let corrected = raw_gain * COORDINATED_PREDICTION_CALIBRATION * learned;

    assert!(
        corrected > 0.0,
        "corrected gain must remain strictly positive (non-zero candidate stream), got {corrected}"
    );
    // The correction still does its job: the fabricated gain is heavily shrunk.
    assert!(
        corrected < raw_gain * 0.05,
        "corrected gain ({corrected}) must be far below the raw gain ({raw_gain})"
    );
}

/// Below the per-squash sample threshold the specific layer stays empty and the
/// lookup falls back to the (non-zero) per-change-type EWMA — a thin cache does
/// not leave the correction undefined or neutral.
#[test]
fn thin_cache_falls_back_to_change_type_not_neutral() {
    // Two entries carrying a target squash — still one short of the specific
    // threshold, so the specific bucket must not be retained.
    const _: () = assert!(MIN_SPECIFIC_TARGET_SQUASH_SAMPLES > 2);
    let cache: Vec<FailureCacheEntry> = (0..2)
        .map(|_| {
            let mut e = remove_neuron_failure();
            e.target_squash = Some("SELU".to_string());
            e
        })
        .collect();

    let correction = CalibrationCorrection::from_failure_cache(&cache);
    assert!(
        correction.specific_as_map().is_empty(),
        "2 entries is below MIN_SPECIFIC_TARGET_SQUASH_SAMPLES; specific layer must stay empty"
    );

    let with_squash = correction.correction_for(CHANGE_TYPE_REMOVE_NEURON, Some("SELU"));
    let fallback = correction.get_correction(CHANGE_TYPE_REMOVE_NEURON);
    assert!(
        (with_squash - fallback).abs() < 1e-9,
        "specific lookup must fall back to the per-change-type EWMA ({fallback}), got {with_squash}"
    );
    // The fallback is the floored, non-zero correction — not the neutral 1.0
    // that would leave the fabricated gain undiscounted.
    assert!(
        with_squash < NEUTRAL_CORRECTION && with_squash > 0.0,
        "fallback must be a discounting, non-zero correction, got {with_squash}"
    );
}

/// Two entries are enough for the EWMA to reflect the observed ratio: a cache of
/// two mild (ratio ≈ 0.5) outcomes yields ≈ 0.5, proving the EWMA is not starved
/// into a degenerate value by the small sample count.
#[test]
fn thin_cache_ewma_reflects_observed_ratio() {
    let cache = vec![
        FailureCacheEntry {
            change_type: CHANGE_TYPE_REMOVE_NEURON.to_string(),
            expected_error_reduction: 1.0,
            actual_error_reduction: 0.5,
            target_squash: None,
            variant_key: None,
            target_uuid: None,
            improved_count: None,
            total_count: None,
            age_epochs: None,
        },
        FailureCacheEntry {
            change_type: CHANGE_TYPE_REMOVE_NEURON.to_string(),
            expected_error_reduction: 1.0,
            actual_error_reduction: 0.5,
            target_squash: None,
            variant_key: None,
            target_uuid: None,
            improved_count: None,
            total_count: None,
            age_epochs: None,
        },
    ];
    let correction = CalibrationCorrection::from_failure_cache(&cache);
    let learned = correction.correction_for(CHANGE_TYPE_REMOVE_NEURON, None);
    assert!(
        (learned - 0.5).abs() < 1e-6,
        "EWMA of two 0.5 ratios must be 0.5 (not starved to floor/neutral), got {learned}"
    );
}

/// The 2-entry correction matches a many-entry uniform cache: the EWMA of a
/// constant sequence is that constant, so two entries produce the same
/// calibration a long history would. The thin cache is not "under-warmed".
#[test]
fn thin_cache_matches_long_uniform_cache() {
    let thin = CalibrationCorrection::from_failure_cache(&thin_two_entry_cache());
    let long: Vec<FailureCacheEntry> = (0..20).map(|_| remove_neuron_failure()).collect();
    let long = CalibrationCorrection::from_failure_cache(&long);

    let thin_value = thin.correction_for(CHANGE_TYPE_REMOVE_NEURON, None);
    let long_value = long.correction_for(CHANGE_TYPE_REMOVE_NEURON, None);
    assert!(
        (thin_value - long_value).abs() < 1e-6,
        "2-entry correction ({thin_value}) must match the 20-entry correction ({long_value})"
    );
}
