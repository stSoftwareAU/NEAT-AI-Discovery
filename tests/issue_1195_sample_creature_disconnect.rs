//! Integration tests for Issue #1195 — sample-vs-creature disconnect
//! detector and calibration penalty.
//!
//! These tests exercise the public API surface only: the detector function
//! in `analysis::scoring::sample_creature_disconnect`, the
//! `SampleCreatureDisconnect` event in `observability`, and the
//! `disconnect_penalty_for` / `correction_for_triple` lookups added to
//! `CalibrationCorrection`.

use neat_ai_discovery::analysis::scoring::calibration_correction::{
    CHANGE_TYPE_ADD_NEURONS, CalibrationCorrection, FailureCacheEntry, MIN_CALIBRATION_CORRECTION,
    NEUTRAL_CORRECTION,
};
use neat_ai_discovery::analysis::scoring::sample_creature_disconnect::{
    SAMPLE_DISCONNECT_PENALTY, SAMPLE_DISCONNECT_RATIO_THRESHOLD, detect_disconnect,
    detect_disconnect_entry,
};
use neat_ai_discovery::observability::{
    SampleCreatureDisconnect, maybe_emit_sample_creature_disconnect,
};

/// Helper that builds a fully-specified failure-cache entry.
#[allow(clippy::too_many_arguments)] // Single-purpose test fixture; flat positional args keep callers compact.
fn fc_entry(
    change_type: &str,
    squash: &str,
    variant: &str,
    target_uuid: &str,
    expected: f32,
    actual: f32,
    improved: u32,
    total: u32,
) -> FailureCacheEntry {
    FailureCacheEntry {
        change_type: change_type.to_string(),
        expected_error_reduction: expected,
        actual_error_reduction: actual,
        target_squash: Some(squash.to_string()),
        variant_key: Some(variant.to_string()),
        target_uuid: Some(target_uuid.to_string()),
        improved_count: Some(improved),
        total_count: Some(total),
        age_epochs: None,
    }
}

// =============================================================================
// Detector behaviour
// =============================================================================

#[test]
fn detector_fires_for_high_ratio_and_negative_reduction() {
    // 1032 / 1036 ≈ 99.6 %, actual = -0.01 — the canonical pattern.
    assert!(detect_disconnect(1032, 1036, -0.01));
    const _: () = assert!(SAMPLE_DISCONNECT_RATIO_THRESHOLD <= 0.95);
}

#[test]
fn detector_silent_for_high_ratio_and_positive_reduction() {
    assert!(!detect_disconnect(1032, 1036, 0.001));
}

#[test]
fn detector_silent_for_low_ratio_and_negative_reduction() {
    assert!(!detect_disconnect(518, 1036, -0.5));
}

// =============================================================================
// Event emission
// =============================================================================

#[test]
fn event_emitted_when_detector_fires() {
    let entry = fc_entry(
        "add-neurons",
        "SINE",
        "v2_add-neurons_demo_e0_e-3_e0",
        "neuron-A",
        0.001,
        -0.01,
        1032,
        1036,
    );
    let event = maybe_emit_sample_creature_disconnect(42, &entry).expect("should fire");
    assert_eq!(event.batch_id, 42);
    assert_eq!(event.change_type, "add-neurons");
    assert_eq!(event.target_squash.as_deref(), Some("SINE"));
    assert_eq!(
        event.variant_key.as_deref(),
        Some("v2_add-neurons_demo_e0_e-3_e0")
    );
    assert_eq!(event.improved_count, 1032);
    assert_eq!(event.total_count, 1036);
}

#[test]
fn event_not_emitted_when_detector_silent() {
    let entry = fc_entry(
        "add-neurons",
        "SINE",
        "v2_add-neurons_demo_e0_e-3_e0",
        "neuron-A",
        0.001,
        // Tiny but positive — disconnect must not fire.
        0.000_001,
        1032,
        1036,
    );
    assert!(maybe_emit_sample_creature_disconnect(42, &entry).is_none());
}

// =============================================================================
// Calibration penalty integration
// =============================================================================

/// Acceptance: replays the three failures captured in Issue #1195
/// (failure-cache `c5ecaafe`) and asserts the detector fires for each
/// and the calibration entry for the matching triple is demoted.
#[test]
fn replays_failure_cluster_and_demotes_calibration_entry() {
    // Three add-neurons failures against a SINE target with the same
    // variant key — the cluster pattern from #1189/#1195. Each entry
    // shows ≈ 99.7 % per-sample improvement but a non-positive
    // creature-level outcome.
    let variant = "v2_add-neurons_demo_e0_e-3_e0";
    let cache = vec![
        fc_entry(
            CHANGE_TYPE_ADD_NEURONS,
            "SINE",
            variant,
            "neuron-A",
            0.001,
            -0.000_5,
            1032,
            1036,
        ),
        fc_entry(
            CHANGE_TYPE_ADD_NEURONS,
            "SINE",
            variant,
            "neuron-A",
            0.001,
            -0.000_4,
            1033,
            1036,
        ),
        fc_entry(
            CHANGE_TYPE_ADD_NEURONS,
            "SINE",
            variant,
            "neuron-A",
            0.001,
            -0.000_3,
            1033,
            1036,
        ),
    ];

    // The detector must fire for every entry in the cluster.
    for entry in &cache {
        assert!(
            detect_disconnect_entry(entry),
            "detector should fire for every clustered failure"
        );
        assert!(
            SampleCreatureDisconnect::from_entry(0, entry).is_some(),
            "event must be constructible for every clustered failure"
        );
    }

    let correction = CalibrationCorrection::from_failure_cache(&cache);

    // Penalty after three disconnects: 0.5 ^ 3 = 0.125, well above the
    // floor and well below the neutral ceiling.
    let penalty = correction.disconnect_penalty_for(CHANGE_TYPE_ADD_NEURONS, "SINE", variant);
    let expected_penalty = SAMPLE_DISCONNECT_PENALTY.powi(3);
    assert!(
        (penalty - expected_penalty).abs() < 1e-6,
        "expected cumulative penalty {expected_penalty}, got {penalty}"
    );
    assert!(penalty > MIN_CALIBRATION_CORRECTION);
    assert!(penalty < NEUTRAL_CORRECTION);

    // The penalty layer must be present in the diagnostic map — proof
    // that the (change_type, target_squash, variant_key) entry was
    // demoted relative to the neutral default.
    let key = (
        CHANGE_TYPE_ADD_NEURONS.to_string(),
        "SINE".to_string(),
        variant.to_string(),
    );
    assert!(
        correction.disconnect_penalties_as_map().contains_key(&key),
        "penalty entry for the matching triple must be recorded"
    );

    // The triple-aware lookup must be at or below the (change_type,
    // target_squash) lookup — never higher. The strict-inequality case
    // is exercised by `triple_lookup_strictly_below_pair_when_above_floor`.
    let triple = correction.correction_for_triple(CHANGE_TYPE_ADD_NEURONS, "SINE", variant);
    let pair = correction.correction_for(CHANGE_TYPE_ADD_NEURONS, Some("SINE"));
    assert!(
        triple <= pair + 1e-9,
        "triple lookup ({triple}) must not exceed pair ({pair})"
    );
}

/// Sanity check that the penalty is strictly applied to the triple lookup
/// when the underlying pair correction is above the floor. We construct
/// a positive-ratio failure cluster that still trips the disconnect
/// detector via the per-sample counters (this is the artificial "edge"
/// case the detector defends against).
#[test]
fn triple_lookup_strictly_below_pair_when_above_floor() {
    let variant = "v2_add-neurons_demo";

    // Three legitimate add-neurons entries against SINE with a moderate
    // (positive) ratio, seeding an above-floor EWMA correction.
    let mut cache: Vec<FailureCacheEntry> = (0..3)
        .map(|_| {
            fc_entry(
                CHANGE_TYPE_ADD_NEURONS,
                "SINE",
                variant,
                "neuron-A",
                1.0,
                0.5,
                500,
                1000,
            )
        })
        .collect();
    // One disconnect entry (high ratio + zero reduction triggers the
    // detector but contributes a 0/1.0 ratio of 0 to the EWMA).
    cache.push(fc_entry(
        CHANGE_TYPE_ADD_NEURONS,
        "SINE",
        variant,
        "neuron-A",
        1.0,
        0.0,
        1032,
        1036,
    ));

    let correction = CalibrationCorrection::from_failure_cache(&cache);

    let pair = correction.correction_for(CHANGE_TYPE_ADD_NEURONS, Some("SINE"));
    let triple = correction.correction_for_triple(CHANGE_TYPE_ADD_NEURONS, "SINE", variant);
    let penalty = correction.disconnect_penalty_for(CHANGE_TYPE_ADD_NEURONS, "SINE", variant);

    assert!(
        pair > MIN_CALIBRATION_CORRECTION + 0.01,
        "pair lookup must sit above the floor for this scenario, got {pair}"
    );
    assert!(
        (penalty - SAMPLE_DISCONNECT_PENALTY).abs() < 1e-6,
        "exactly one disconnect → penalty 0.5, got {penalty}"
    );
    assert!(
        triple < pair,
        "triple lookup ({triple}) must be strictly below pair ({pair})"
    );
}

#[test]
fn cumulative_penalty_clamped_to_floor() {
    // Many disconnects in a row must not collapse the penalty below the
    // existing calibration floor.
    let variant = "v2_add-neurons_demo_e0_e-3_e0";
    // 0.5 ^ N reaches 0.001 at N ≈ 10; pile on 20 to confirm the clamp.
    let cache: Vec<FailureCacheEntry> = (0..20)
        .map(|_| {
            fc_entry(
                CHANGE_TYPE_ADD_NEURONS,
                "SINE",
                variant,
                "neuron-A",
                0.001,
                -0.000_1,
                1032,
                1036,
            )
        })
        .collect();
    let correction = CalibrationCorrection::from_failure_cache(&cache);
    let penalty = correction.disconnect_penalty_for(CHANGE_TYPE_ADD_NEURONS, "SINE", variant);
    assert!(
        (penalty - MIN_CALIBRATION_CORRECTION).abs() < 1e-6,
        "expected floor {MIN_CALIBRATION_CORRECTION}, got {penalty}"
    );
}

#[test]
fn unrelated_triple_unaffected_by_penalty() {
    // A single SINE / variant disconnect must not penalise other targets.
    let entry = fc_entry(
        CHANGE_TYPE_ADD_NEURONS,
        "SINE",
        "v2_add-neurons_demo_e0_e-3_e0",
        "neuron-A",
        0.001,
        -0.000_1,
        1032,
        1036,
    );
    let correction = CalibrationCorrection::from_failure_cache(&[entry]);

    // Different squash, same variant — neutral.
    assert!(
        (correction.disconnect_penalty_for(
            CHANGE_TYPE_ADD_NEURONS,
            "RELU",
            "v2_add-neurons_demo_e0_e-3_e0"
        ) - NEUTRAL_CORRECTION)
            .abs()
            < 1e-9
    );
    // Same squash, different variant — neutral.
    assert!(
        (correction.disconnect_penalty_for(CHANGE_TYPE_ADD_NEURONS, "SINE", "different-variant")
            - NEUTRAL_CORRECTION)
            .abs()
            < 1e-9
    );
}

#[test]
fn legacy_entries_without_counts_do_not_record_penalty() {
    // Legacy failure cache (no improved_count / total_count) must keep
    // the existing calibration behaviour and produce no penalty entries.
    let entry = FailureCacheEntry {
        change_type: CHANGE_TYPE_ADD_NEURONS.to_string(),
        expected_error_reduction: 0.001,
        actual_error_reduction: -0.5,
        target_squash: Some("SINE".to_string()),
        variant_key: Some("v2".to_string()),
        target_uuid: None,
        improved_count: None,
        total_count: None,
        age_epochs: None,
    };
    let correction = CalibrationCorrection::from_failure_cache(&[entry]);
    assert!(correction.disconnect_penalties_as_map().is_empty());
}
