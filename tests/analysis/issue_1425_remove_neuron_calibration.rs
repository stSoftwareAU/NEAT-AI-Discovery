//! Integration tests for Issue #1425: failure-cache calibration correction for
//! harmful-neuron (`remove-neuron`) candidates.
//!
//! Harmful-neuron candidates historically over-predicted their score gain by
//! ~800× (failure bucket `247b83ab`). The failure-cache calibration correction
//! that learns from exactly these misses was applied to the add-neuron and
//! add-synapse paths but not to the remove-neuron path. These tests cover the
//! public surface of [`CalibrationCorrection::correction_for`] for the
//! `remove-neuron` change type.

#![allow(clippy::cast_precision_loss)]

use neat_ai_discovery::analysis::scoring::calibration_correction::{
    CHANGE_TYPE_COORDINATED_STRUCTURAL, CHANGE_TYPE_REMOVE_NEURON, CalibrationCorrection,
    FailureCacheEntry,
};

/// A failure-cache entry mirroring the `247b83ab` evidence shape: predicted
/// ≈ 0.166, actual ≈ 0 (slightly negative) — a ~800× over-prediction.
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

/// Acceptance criterion 2/3: replaying the 9 cached `remove-neuron` failures
/// produces a `remove-neuron` correction near the floor, so the predicted gain
/// is shrunk toward the observed ≈ 0.
#[test]
fn nine_remove_neuron_failures_drive_correction_to_floor() {
    let cache: Vec<FailureCacheEntry> = (0..9).map(|_| remove_neuron_failure()).collect();
    let correction = CalibrationCorrection::from_failure_cache(&cache);

    let learned = correction.correction_for(CHANGE_TYPE_REMOVE_NEURON, None);
    assert!(
        learned <= 0.01,
        "remove-neuron correction should collapse toward the floor, got {learned}"
    );

    // Applying the learnt correction to the over-predicted gain shrinks it by
    // roughly the over-prediction factor.
    let raw_gain = 0.165_710_48_f32;
    let corrected = raw_gain * learned;
    assert!(
        corrected < raw_gain * 0.05,
        "corrected gain ({corrected}) must be far below the raw gain ({raw_gain})"
    );
}

/// The `remove-neuron` change type is tracked independently from
/// `coordinated-structural`: remove-neuron failures must not discount
/// coordinated-structural predictions and vice versa.
#[test]
fn remove_neuron_and_coordinated_structural_are_independent() {
    let cache: Vec<FailureCacheEntry> = (0..9).map(|_| remove_neuron_failure()).collect();
    let correction = CalibrationCorrection::from_failure_cache(&cache);

    let remove = correction.correction_for(CHANGE_TYPE_REMOVE_NEURON, None);
    let coordinated = correction.correction_for(CHANGE_TYPE_COORDINATED_STRUCTURAL, None);

    // remove-neuron is heavily discounted; coordinated-structural saw no
    // failures so it remains neutral (1.0).
    assert!(remove < coordinated);
    assert!((coordinated - 1.0).abs() < 1e-9);
}

/// Acceptance criterion 3: a `remove-neuron` failure-cache JSON entry
/// round-trips through serde and feeds the correction for that change type.
#[test]
fn remove_neuron_json_round_trip() {
    let cache_json = r#"[
        {
            "changeType": "remove-neuron",
            "expectedErrorReduction": 0.16571048,
            "actualErrorReduction": -0.00020719
        }
    ]"#;
    let parsed: Vec<FailureCacheEntry> = serde_json::from_str(cache_json).expect("parse");
    assert_eq!(parsed.len(), 1);
    assert_eq!(parsed[0].change_type, CHANGE_TYPE_REMOVE_NEURON);

    // A single entry is below the per-squash specific threshold but still feeds
    // the per-change_type EWMA fallback used by `correction_for`.
    let correction = CalibrationCorrection::from_failure_cache(&parsed);
    let learned = correction.correction_for(CHANGE_TYPE_REMOVE_NEURON, None);
    assert!(
        learned <= 0.01,
        "single severe over-prediction should still discount, got {learned}"
    );
}
