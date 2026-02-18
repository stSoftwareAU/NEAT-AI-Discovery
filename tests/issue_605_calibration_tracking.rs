//! Integration tests for Issue #605: Candidate confidence calibration —
//! track predicted vs actual improvement accuracy.
//!
//! Tests verify that the calibration tracker correctly records predicted
//! vs actual improvements and computes calibration metrics.

use neat_ai_discovery::discovery_history::{CalibrationTracker, DiscoveryHistory};
use neat_ai_discovery::get_calibration_summary_internal;

#[test]
fn calibration_tracker_records_predictions() {
    let mut tracker = CalibrationTracker::new();
    tracker.record_prediction("saturation", "addSynapse", 0.05, 0.03);
    tracker.record_prediction("saturation", "addSynapse", 0.10, 0.08);

    let summary = tracker.calibration_summary();
    assert_eq!(summary.len(), 1);

    let entry = &summary[0];
    assert_eq!(entry.module_name, "saturation");
    assert_eq!(entry.candidate_type, "addSynapse");
    assert_eq!(entry.sample_count, 2);
}

#[test]
fn calibration_tracker_computes_mean_absolute_error() {
    let mut tracker = CalibrationTracker::new();

    // Predicted 0.10, actual 0.08 => error 0.02
    tracker.record_prediction("saturation", "addSynapse", 0.10, 0.08);
    // Predicted 0.05, actual 0.03 => error 0.02
    tracker.record_prediction("saturation", "addSynapse", 0.05, 0.03);

    let summary = tracker.calibration_summary();
    let entry = &summary[0];

    // MAE = (0.02 + 0.02) / 2 = 0.02
    assert!(
        (entry.mean_absolute_error - 0.02).abs() < 1e-6,
        "Expected MAE ~0.02, got {}",
        entry.mean_absolute_error
    );
}

#[test]
fn calibration_tracker_computes_bias_direction() {
    let mut tracker = CalibrationTracker::new();

    // Over-predictions: predicted > actual
    tracker.record_prediction("bottleneck", "addNeuron", 0.10, 0.05);
    tracker.record_prediction("bottleneck", "addNeuron", 0.08, 0.03);

    let summary = tracker.calibration_summary();
    let entry = &summary[0];

    // Bias = mean(predicted - actual) = mean(0.05, 0.05) = 0.05
    // Positive bias means over-prediction
    assert!(
        entry.bias > 0.0,
        "Expected positive bias (over-prediction), got {}",
        entry.bias
    );
    assert!(
        (entry.bias - 0.05).abs() < 1e-6,
        "Expected bias ~0.05, got {}",
        entry.bias
    );
}

#[test]
fn calibration_tracker_negative_bias_for_under_prediction() {
    let mut tracker = CalibrationTracker::new();

    // Under-predictions: predicted < actual
    tracker.record_prediction("dead_neuron", "removeNeuron", 0.02, 0.06);
    tracker.record_prediction("dead_neuron", "removeNeuron", 0.03, 0.07);

    let summary = tracker.calibration_summary();
    let entry = &summary[0];

    // Bias = mean(predicted - actual) = mean(-0.04, -0.04) = -0.04
    assert!(
        entry.bias < 0.0,
        "Expected negative bias (under-prediction), got {}",
        entry.bias
    );
}

#[test]
fn calibration_tracker_per_module_calibration_factor() {
    let mut tracker = CalibrationTracker::new();

    // Module that consistently over-predicts by 2x
    tracker.record_prediction("saturation", "addSynapse", 0.10, 0.05);
    tracker.record_prediction("saturation", "addSynapse", 0.20, 0.10);
    tracker.record_prediction("saturation", "addSynapse", 0.06, 0.03);

    let factor = tracker.calibration_factor("saturation", "addSynapse");

    // actual/predicted = 0.5 on average, so calibration factor should be ~0.5
    assert!(
        (factor - 0.5).abs() < 0.1,
        "Expected calibration factor ~0.5, got {factor}"
    );
}

#[test]
fn calibration_factor_defaults_to_one_for_unknown_module() {
    let tracker = CalibrationTracker::new();

    let factor = tracker.calibration_factor("unknown_module", "addSynapse");
    assert!(
        (factor - 1.0).abs() < 1e-6,
        "Expected default calibration factor of 1.0, got {factor}"
    );
}

#[test]
fn calibration_tracker_separates_modules_and_candidate_types() {
    let mut tracker = CalibrationTracker::new();

    tracker.record_prediction("saturation", "addSynapse", 0.10, 0.08);
    tracker.record_prediction("saturation", "setBias", 0.05, 0.04);
    tracker.record_prediction("bottleneck", "addSynapse", 0.15, 0.12);

    let summary = tracker.calibration_summary();
    assert_eq!(
        summary.len(),
        3,
        "Should have 3 separate entries for different module/type combinations"
    );
}

#[test]
fn calibration_tracker_serialisation_roundtrip() {
    let mut tracker = CalibrationTracker::new();
    tracker.record_prediction("saturation", "addSynapse", 0.10, 0.08);
    tracker.record_prediction("bottleneck", "addNeuron", 0.05, 0.03);

    let json = serde_json::to_string(&tracker).expect("should serialise");
    let restored: CalibrationTracker = serde_json::from_str(&json).expect("should deserialise");

    let original_summary = tracker.calibration_summary();
    let restored_summary = restored.calibration_summary();
    assert_eq!(original_summary.len(), restored_summary.len());
}

#[test]
fn discovery_history_includes_calibration_tracker() {
    let mut history = DiscoveryHistory::new();

    // Record neuron discovery as before
    history.record("hidden-1", true, Some(100));

    // Record calibration data
    history.record_calibration("saturation", "addSynapse", 0.10, 0.08);
    history.record_calibration("bottleneck", "addNeuron", 0.05, 0.03);

    let summary = history.calibration_summary();
    assert_eq!(summary.len(), 2);

    let factor = history.calibration_factor("saturation", "addSynapse");
    assert!(factor > 0.0 && factor <= 2.0);
}

#[test]
fn calibration_tracker_handles_zero_predicted() {
    let mut tracker = CalibrationTracker::new();

    // Edge case: predicted improvement is zero
    tracker.record_prediction("saturation", "addSynapse", 0.0, 0.05);

    let factor = tracker.calibration_factor("saturation", "addSynapse");
    // When predicted is zero but actual is non-zero, calibration factor should be clamped
    assert!(
        (0.1..=10.0).contains(&factor),
        "Calibration factor should be clamped to reasonable range, got {factor}"
    );
}

#[test]
fn ffi_get_calibration_summary_returns_valid_json() {
    let mut history = DiscoveryHistory::new();
    history.record_calibration("saturation", "addSynapse", 0.10, 0.08);
    history.record_calibration("saturation", "addSynapse", 0.20, 0.15);
    history.record_calibration("bottleneck", "addNeuron", 0.05, 0.03);

    let history_json = serde_json::to_string(&history).expect("should serialise history");
    let input_json = serde_json::json!({
        "discoveryHistory": history_json
    })
    .to_string();

    let output_json = get_calibration_summary_internal(&input_json).expect("should return JSON");
    let output: serde_json::Value =
        serde_json::from_str(&output_json).expect("should be valid JSON");

    assert_eq!(output["success"], true);
    let entries = output["calibrationSummary"]
        .as_array()
        .expect("should have calibrationSummary array");
    assert_eq!(entries.len(), 2);

    // Verify fields are present
    let first = &entries[0];
    assert!(first["moduleName"].is_string());
    assert!(first["candidateType"].is_string());
    assert!(first["sampleCount"].is_number());
    assert!(first["meanAbsoluteError"].is_number());
    assert!(first["bias"].is_number());
    assert!(first["calibrationFactor"].is_number());
}

#[test]
fn ffi_get_calibration_summary_empty_history() {
    let history = DiscoveryHistory::new();
    let history_json = serde_json::to_string(&history).expect("should serialise");
    let input_json = serde_json::json!({
        "discoveryHistory": history_json
    })
    .to_string();

    let output_json = get_calibration_summary_internal(&input_json).expect("should return JSON");
    let output: serde_json::Value =
        serde_json::from_str(&output_json).expect("should be valid JSON");

    assert_eq!(output["success"], true);
    let entries = output["calibrationSummary"]
        .as_array()
        .expect("should have calibrationSummary array");
    assert!(entries.is_empty());
}

#[test]
fn calibration_summary_sorted_by_sample_count() {
    let mut tracker = CalibrationTracker::new();

    // Module with 1 sample
    tracker.record_prediction("dead_neuron", "removeNeuron", 0.10, 0.08);

    // Module with 3 samples
    for _ in 0..3 {
        tracker.record_prediction("saturation", "addSynapse", 0.10, 0.08);
    }

    let summary = tracker.calibration_summary();
    assert!(
        summary[0].sample_count >= summary[1].sample_count,
        "Summary should be sorted by sample count descending"
    );
}
