//! Issue #891: Calibrate prediction scaling to reduce overestimation
//!
//! Production discovery-cache analysis reveals that predicted improvements overestimate
//! actual outcomes by 100–10,000×. This systematic overestimation varies by
//! candidate type, making cross-type comparisons unreliable.
//!
//! Per-candidate-type calibration factors correct the magnitude gap:
//! - Synapse predictions: ~1,000× overestimation → calibration 0.001
//! - Neuron predictions: ~100× overestimation → calibration 0.01
//! - Coordinated predictions: ~10,000× overestimation → calibration 0.0001

use neat_ai_discovery::analysis::constants::{
    COORDINATED_PREDICTION_CALIBRATION, NEURON_PREDICTION_CALIBRATION,
    SYNAPSE_PREDICTION_CALIBRATION,
};
use neat_ai_discovery::analysis::synapse::apply_prediction_calibration;

// =============================================================================
// Compile-time calibration constant validation
// =============================================================================

// Synapse calibration: must be in (0.0001, 0.01] to correct ~1,000× overestimation.
const _: () = assert!(SYNAPSE_PREDICTION_CALIBRATION > 0.0);
const _: () = assert!(SYNAPSE_PREDICTION_CALIBRATION <= 0.01);
const _: () = assert!(SYNAPSE_PREDICTION_CALIBRATION >= 0.0001);

// Neuron calibration: must be in (0.001, 0.1] to correct ~100× overestimation.
const _: () = assert!(NEURON_PREDICTION_CALIBRATION > 0.0);
const _: () = assert!(NEURON_PREDICTION_CALIBRATION <= 0.1);
const _: () = assert!(NEURON_PREDICTION_CALIBRATION >= 0.001);

// Coordinated calibration: must be in (0.00001, 0.001] to correct ~10,000× overestimation.
const _: () = assert!(COORDINATED_PREDICTION_CALIBRATION > 0.0);
const _: () = assert!(COORDINATED_PREDICTION_CALIBRATION <= 0.001);
const _: () = assert!(COORDINATED_PREDICTION_CALIBRATION >= 0.00001);

// Relative ordering: neuron > synapse > coordinated (less overestimation → larger factor).
const _: () = assert!(NEURON_PREDICTION_CALIBRATION > SYNAPSE_PREDICTION_CALIBRATION);
const _: () = assert!(SYNAPSE_PREDICTION_CALIBRATION > COORDINATED_PREDICTION_CALIBRATION);

// =============================================================================
// Calibration application tests
// =============================================================================

/// Verify calibration correctly scales a positive gain.
#[test]
fn test_apply_prediction_calibration_positive_gain() {
    let raw_gain = 0.05_f32;
    let calibrated = apply_prediction_calibration(raw_gain, SYNAPSE_PREDICTION_CALIBRATION);

    assert!(calibrated > 0.0, "Calibrated gain should remain positive");
    assert!(
        calibrated < raw_gain,
        "Calibrated gain ({calibrated}) should be less than raw gain ({raw_gain})",
    );
    let expected = raw_gain * SYNAPSE_PREDICTION_CALIBRATION;
    assert!(
        (calibrated - expected).abs() < 1e-10,
        "Calibrated gain should equal raw × calibration factor: expected {expected}, got {calibrated}",
    );
}

/// Verify calibration preserves zero gain.
#[test]
fn test_apply_prediction_calibration_zero_gain() {
    let calibrated = apply_prediction_calibration(0.0, SYNAPSE_PREDICTION_CALIBRATION);
    assert!(
        calibrated.abs() < 1e-15,
        "Zero gain should remain zero after calibration, got {calibrated}",
    );
}

/// Verify calibration preserves sign for negative gains.
#[test]
fn test_apply_prediction_calibration_negative_gain() {
    let raw_gain = -0.03_f32;
    let calibrated = apply_prediction_calibration(raw_gain, SYNAPSE_PREDICTION_CALIBRATION);

    assert!(
        calibrated < 0.0,
        "Negative gain should remain negative after calibration"
    );
    let expected = raw_gain * SYNAPSE_PREDICTION_CALIBRATION;
    assert!(
        (calibrated - expected).abs() < 1e-10,
        "Calibrated negative gain should equal raw × calibration factor"
    );
}

/// Verify cross-type calibration improves ranking fairness.
///
/// A neuron prediction of 0.003 and a synapse prediction of 0.01 should
/// swap relative ranking after calibration (neuron overestimates less).
#[test]
fn test_calibration_improves_cross_type_ranking() {
    let neuron_raw = 0.003_f32;
    let synapse_raw = 0.01_f32;

    // Before calibration: synapse ranks higher (0.01 > 0.003)
    assert!(synapse_raw > neuron_raw);

    let neuron_calibrated = apply_prediction_calibration(neuron_raw, NEURON_PREDICTION_CALIBRATION);
    let synapse_calibrated =
        apply_prediction_calibration(synapse_raw, SYNAPSE_PREDICTION_CALIBRATION);

    // After calibration: neuron should rank higher because its overestimation
    // is smaller (100× vs 1,000×), so the calibrated value is relatively larger.
    assert!(
        neuron_calibrated > synapse_calibrated,
        "After calibration, neuron ({neuron_calibrated}) should rank higher than synapse ({synapse_calibrated}) because neuron overestimates less",
    );
}

/// Verify calibration handles very small gains without losing precision to zero.
#[test]
fn test_calibration_preserves_small_gains() {
    let tiny_gain = 1e-6_f32;
    let calibrated = apply_prediction_calibration(tiny_gain, NEURON_PREDICTION_CALIBRATION);

    assert!(
        calibrated > 0.0,
        "Very small positive gain should remain positive after calibration, got {calibrated}",
    );
}
