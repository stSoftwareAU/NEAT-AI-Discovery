//! Integration tests for Issue #1162: per-(`change_type`, `target_squash`)
//! calibration tracking from the failure cache.
//!
//! These tests cover the public surface of
//! [`CalibrationCorrection::correction_for`] (the lookup wired into the
//! synapse and neuron post-processing passes) and the JSON parsing of the
//! optional `targetNeuronInfo.squash` field.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use neat_ai_discovery::analysis::scoring::calibration_correction::{
    CHANGE_TYPE_ADD_NEURONS, CHANGE_TYPE_ADD_SYNAPSES, CalibrationCorrection, FailureCacheEntry,
    MIN_SPECIFIC_TARGET_SQUASH_SAMPLES,
};

fn entry_with_squash(
    change_type: &str,
    predicted: f32,
    actual: f32,
    squash: Option<&str>,
) -> FailureCacheEntry {
    FailureCacheEntry {
        change_type: change_type.to_string(),
        expected_error_reduction: predicted,
        actual_error_reduction: actual,
        target_squash: squash.map(str::to_string),
    }
}

/// Acceptance: when only `change_type` data is supplied (no target squash),
/// `correction_for` falls back to the per-`change_type` value for any squash.
#[test]
fn change_type_only_data_falls_back_for_any_squash() {
    let cache: Vec<FailureCacheEntry> = (0..6)
        .map(|_| entry_with_squash(CHANGE_TYPE_ADD_NEURONS, 1.0, 0.25, None))
        .collect();
    let correction = CalibrationCorrection::from_failure_cache(&cache);

    assert!(correction.specific_as_map().is_empty());

    // Per-change_type EWMA dominates because no specific data is recorded.
    let baseline = correction.get_correction(CHANGE_TYPE_ADD_NEURONS);
    let with_squash = correction.correction_for(CHANGE_TYPE_ADD_NEURONS, Some("SELU"));
    let without_squash = correction.correction_for(CHANGE_TYPE_ADD_NEURONS, None);
    assert!((with_squash - baseline).abs() < 1e-9);
    assert!((without_squash - baseline).abs() < 1e-9);
}

/// Acceptance: with enough specific data, `correction_for(SELU)` returns a
/// stronger discount than the per-`change_type` fallback for an unrelated
/// squash. This is the headline behaviour from Issue #1162's GRQ-sampler
/// motivating example.
#[test]
fn specific_correction_is_distinct_from_change_type_fallback() {
    let mut cache = vec![
        // Change_type history: bounded squashes that recently came in close
        // to expectation (ratio ≈ 0.8).
        entry_with_squash(CHANGE_TYPE_ADD_NEURONS, 1.0, 0.8, Some("HARD_TANH")),
        entry_with_squash(CHANGE_TYPE_ADD_NEURONS, 1.0, 0.8, Some("HARD_TANH")),
        entry_with_squash(CHANGE_TYPE_ADD_NEURONS, 1.0, 0.8, Some("HARD_TANH")),
    ];
    // SELU-specific entries that consistently massively over-estimated.
    for _ in 0..MIN_SPECIFIC_TARGET_SQUASH_SAMPLES {
        cache.push(entry_with_squash(
            CHANGE_TYPE_ADD_NEURONS,
            1.0,
            0.001,
            Some("SELU"),
        ));
    }

    let correction = CalibrationCorrection::from_failure_cache(&cache);

    let selu = correction.correction_for(CHANGE_TYPE_ADD_NEURONS, Some("SELU"));
    let hard_tanh = correction.correction_for(CHANGE_TYPE_ADD_NEURONS, Some("HARD_TANH"));

    // SELU-specific must be much smaller than the HARD_TANH-specific value.
    assert!(
        selu < hard_tanh,
        "expected SELU-specific ({selu}) < HARD_TANH-specific ({hard_tanh})"
    );
    // SELU should be near the floor (0.001).
    assert!(
        selu <= 0.01,
        "expected SELU correction near the floor, got {selu}"
    );
}

/// Acceptance: a (`change_type`, squash) bucket below the sample threshold
/// is ignored — `correction_for` returns the per-`change_type` fallback.
#[test]
fn below_threshold_specific_entries_are_ignored() {
    let mut cache = vec![
        entry_with_squash(CHANGE_TYPE_ADD_SYNAPSES, 1.0, 0.5, None),
        entry_with_squash(CHANGE_TYPE_ADD_SYNAPSES, 1.0, 0.5, None),
        entry_with_squash(CHANGE_TYPE_ADD_SYNAPSES, 1.0, 0.5, None),
    ];
    // Only (MIN - 1) entries against TANH — must NOT be retained.
    for _ in 0..(MIN_SPECIFIC_TARGET_SQUASH_SAMPLES - 1) {
        cache.push(entry_with_squash(
            CHANGE_TYPE_ADD_SYNAPSES,
            1.0,
            0.001,
            Some("TANH"),
        ));
    }
    let correction = CalibrationCorrection::from_failure_cache(&cache);
    assert!(correction.specific_as_map().is_empty());

    let tanh_lookup = correction.correction_for(CHANGE_TYPE_ADD_SYNAPSES, Some("TANH"));
    let baseline = correction.get_correction(CHANGE_TYPE_ADD_SYNAPSES);
    assert!(
        (tanh_lookup - baseline).abs() < 1e-9,
        "below-threshold lookup must equal the change_type fallback"
    );
}

/// Acceptance: zero-divisor and NaN ratios on specific entries do not count
/// toward the sample threshold and do not pollute the EWMA.
#[test]
fn nan_and_zero_divisor_specific_entries_skipped() {
    let mut cache = vec![
        entry_with_squash(CHANGE_TYPE_ADD_SYNAPSES, 0.0, 0.0, Some("RELU")),
        entry_with_squash(
            CHANGE_TYPE_ADD_SYNAPSES,
            f32::MIN_POSITIVE,
            f32::MAX,
            Some("RELU"),
        ),
    ];
    // Add legitimate fallback data.
    for _ in 0..3 {
        cache.push(entry_with_squash(CHANGE_TYPE_ADD_SYNAPSES, 1.0, 0.4, None));
    }
    let correction = CalibrationCorrection::from_failure_cache(&cache);
    assert!(
        correction.specific_as_map().is_empty(),
        "non-finite / zero specific entries must not establish a specific bucket"
    );

    let lookup = correction.correction_for(CHANGE_TYPE_ADD_SYNAPSES, Some("RELU"));
    let baseline = correction.get_correction(CHANGE_TYPE_ADD_SYNAPSES);
    assert!((lookup - baseline).abs() < 1e-9);
}

/// Acceptance: a failure-cache JSON entry with `targetNeuronInfo.squash`
/// round-trips through serde into `target_squash`, and entries without it
/// remain backward compatible.
#[test]
fn json_target_neuron_info_round_trip() {
    let cache_json = r#"[
        {
            "changeType": "add-neurons",
            "expectedErrorReduction": 0.001,
            "actualErrorReduction": 0.000001,
            "targetNeuronInfo": { "squash": "SELU" }
        },
        {
            "changeType": "add-neurons",
            "expectedErrorReduction": 0.001,
            "actualErrorReduction": 0.000001
        }
    ]"#;
    let parsed: Vec<FailureCacheEntry> = serde_json::from_str(cache_json).expect("parse");
    assert_eq!(parsed.len(), 2);
    assert_eq!(parsed[0].target_squash.as_deref(), Some("SELU"));
    assert!(parsed[1].target_squash.is_none());
}

/// Acceptance: a fully-empty cache produces a correction that returns
/// neutral (1.0) for every lookup, including those carrying a target squash.
#[test]
fn empty_cache_lookup_returns_neutral() {
    let correction = CalibrationCorrection::from_failure_cache(&[]);
    let value = correction.correction_for(CHANGE_TYPE_ADD_NEURONS, Some("SELU"));
    assert!((value - 1.0).abs() < 1e-9);
    let value_no_squash = correction.correction_for(CHANGE_TYPE_ADD_NEURONS, None);
    assert!((value_no_squash - 1.0).abs() < 1e-9);
}
