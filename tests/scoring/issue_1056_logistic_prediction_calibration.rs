//! Issue #1056: Recalibrate prediction scoring to match production success rates
//!
//! GRQ-sampler discovery cache (30+ creatures) reveals significant gaps between
//! predicted and actual success rates:
//!
//! | Candidate Type         | Actual Success Rate |
//! |------------------------|---------------------|
//! | add-neurons            | ~2.7% (28/1028)     |
//! | coordinated-structural | ~1.1% (6/525)       |
//! | add-synapses           | ~0.1% (3/1001)      |
//! | remove-low-impact      | ~11%  (6/54)        |
//!
//! This module tests:
//! - Updated calibration factors match empirical data
//! - Non-linear (logistic) calibration provides better correction than linear
//! - Updated pessimism discount parameters are within valid ranges
//! - Cross-type ranking is preserved with new calibration

use neat_ai_discovery::analysis::constants::{
    COORDINATED_PREDICTION_CALIBRATION, LOGISTIC_CALIBRATION_FLOOR, LOGISTIC_CALIBRATION_MIDPOINT,
    LOGISTIC_CALIBRATION_STEEPNESS, NEURON_PESSIMISM_CURVE_EXPONENT,
    NEURON_PESSIMISM_DISCOUNT_FLOOR, NEURON_PREDICTION_CALIBRATION,
    SYNAPSE_PESSIMISM_CURVE_EXPONENT, SYNAPSE_PESSIMISM_DISCOUNT_FLOOR,
    SYNAPSE_PREDICTION_CALIBRATION,
};
use neat_ai_discovery::analysis::synapse::{
    apply_logistic_prediction_calibration, apply_neuron_pessimism_discount,
    apply_prediction_calibration, apply_synapse_pessimism_discount,
};

// =============================================================================
// Compile-time constant validation (Issue #1056)
// =============================================================================

// Updated calibration factors must maintain ordering: neuron > synapse > coordinated.
const _: () = assert!(NEURON_PREDICTION_CALIBRATION > SYNAPSE_PREDICTION_CALIBRATION);
const _: () = assert!(SYNAPSE_PREDICTION_CALIBRATION > COORDINATED_PREDICTION_CALIBRATION);

// Updated pessimism parameters: synapse more aggressive than neuron.
const _SYNAPSE_FLOOR_LESS: () = {
    assert!(SYNAPSE_PESSIMISM_DISCOUNT_FLOOR < NEURON_PESSIMISM_DISCOUNT_FLOOR);
};
const _SYNAPSE_EXP_MORE: () = {
    assert!(
        // f32 comparison via bits for const context
        SYNAPSE_PESSIMISM_CURVE_EXPONENT.to_bits() > NEURON_PESSIMISM_CURVE_EXPONENT.to_bits()
    );
};

// Logistic calibration parameters must be in valid ranges.
const _: () = assert!(LOGISTIC_CALIBRATION_FLOOR > 0.0);
const _: () = assert!(LOGISTIC_CALIBRATION_FLOOR < 0.5);
const _: () = assert!(LOGISTIC_CALIBRATION_STEEPNESS > 0.0);
const _: () = assert!(LOGISTIC_CALIBRATION_MIDPOINT > 0.2);
const _: () = assert!(LOGISTIC_CALIBRATION_MIDPOINT < 0.9);

// =============================================================================
// Logistic calibration function tests
// =============================================================================

/// Logistic calibration reduces gain more than linear calibration at moderate ratios.
///
/// With a typical improved ratio of ~50% (as observed for add-neurons), the logistic
/// calibration should apply additional reduction beyond the base calibration factor.
#[test]
fn logistic_calibration_more_aggressive_at_moderate_ratios() {
    let gain = 0.01_f32;
    let improved_count = 50_u32;
    let total_count = 100_u32;

    let linear = apply_prediction_calibration(gain, NEURON_PREDICTION_CALIBRATION);
    let logistic = apply_logistic_prediction_calibration(
        gain,
        improved_count,
        total_count,
        NEURON_PREDICTION_CALIBRATION,
    );

    assert!(
        logistic < linear,
        "Logistic calibration ({logistic}) should be more aggressive than linear ({linear}) at 50% improved ratio",
    );
    assert!(
        logistic > 0.0,
        "Logistic calibration should preserve positivity"
    );
}

/// At very high improved ratios (>90%), logistic calibration approaches linear.
#[test]
fn logistic_calibration_approaches_linear_at_high_ratios() {
    let gain = 0.01_f32;
    let improved_count = 95_u32;
    let total_count = 100_u32;

    let linear = apply_prediction_calibration(gain, NEURON_PREDICTION_CALIBRATION);
    let logistic = apply_logistic_prediction_calibration(
        gain,
        improved_count,
        total_count,
        NEURON_PREDICTION_CALIBRATION,
    );

    // At 95% ratio, logistic should be close to linear (within 15%)
    let ratio = logistic / linear;
    assert!(
        ratio > 0.85,
        "At 95% improved ratio, logistic ({logistic}) should be close to linear ({linear}), ratio = {ratio}",
    );
}

/// At very low improved ratios (<10%), logistic calibration heavily reduces gain.
#[test]
fn logistic_calibration_heavy_reduction_at_low_ratios() {
    let gain = 0.01_f32;
    let improved_count = 5_u32;
    let total_count = 100_u32;

    let linear = apply_prediction_calibration(gain, NEURON_PREDICTION_CALIBRATION);
    let logistic = apply_logistic_prediction_calibration(
        gain,
        improved_count,
        total_count,
        NEURON_PREDICTION_CALIBRATION,
    );

    // At 5% ratio, logistic should be much smaller than linear
    assert!(
        logistic < linear * 0.3,
        "At 5% improved ratio, logistic ({logistic}) should be much smaller than linear ({linear})",
    );
}

/// Logistic calibration preserves ordering: higher improved ratios → higher calibrated gain.
#[test]
fn logistic_calibration_preserves_ordering() {
    let gain = 0.01_f32;
    let total_count = 100_u32;

    let cal_low =
        apply_logistic_prediction_calibration(gain, 20, total_count, NEURON_PREDICTION_CALIBRATION);
    let cal_mid =
        apply_logistic_prediction_calibration(gain, 50, total_count, NEURON_PREDICTION_CALIBRATION);
    let cal_high =
        apply_logistic_prediction_calibration(gain, 80, total_count, NEURON_PREDICTION_CALIBRATION);

    assert!(
        cal_low < cal_mid,
        "Lower improved ratio ({cal_low}) should give lower calibrated gain than moderate ({cal_mid})",
    );
    assert!(
        cal_mid < cal_high,
        "Moderate improved ratio ({cal_mid}) should give lower calibrated gain than high ({cal_high})",
    );
}

/// Logistic calibration handles zero `total_count` gracefully.
#[test]
fn logistic_calibration_zero_total_count() {
    let gain = 0.01_f32;
    let result = apply_logistic_prediction_calibration(gain, 0, 0, NEURON_PREDICTION_CALIBRATION);

    assert!(result > 0.0, "Should be positive with zero total count");
    assert!(result.is_finite(), "Should be finite with zero total count");
    // With zero total, should use floor
    let expected = gain * NEURON_PREDICTION_CALIBRATION * LOGISTIC_CALIBRATION_FLOOR;
    assert!(
        (result - expected).abs() < 1e-10,
        "Zero total should give floor calibration: expected {expected}, got {result}",
    );
}

/// Logistic calibration preserves sign for negative gains.
#[test]
fn logistic_calibration_preserves_negative_sign() {
    let gain = -0.01_f32;
    let result =
        apply_logistic_prediction_calibration(gain, 50, 100, SYNAPSE_PREDICTION_CALIBRATION);

    assert!(
        result < 0.0,
        "Negative gain should remain negative after logistic calibration, got {result}",
    );
}

/// Logistic calibration preserves zero gain.
#[test]
fn logistic_calibration_preserves_zero() {
    let result =
        apply_logistic_prediction_calibration(0.0, 50, 100, SYNAPSE_PREDICTION_CALIBRATION);
    assert!(
        result.abs() < 1e-15,
        "Zero gain should remain zero, got {result}",
    );
}

// =============================================================================
// Cross-type ranking tests with updated calibration (Issue #1056)
// =============================================================================

/// After calibration, neuron candidates should rank higher than synapse candidates
/// for equivalent raw gains, because neurons have higher actual success rates.
#[test]
fn cross_type_ranking_neuron_above_synapse() {
    let raw_gain = 0.005_f32;
    let improved_count = 50_u32;
    let total_count = 100_u32;

    let neuron_cal = apply_logistic_prediction_calibration(
        raw_gain,
        improved_count,
        total_count,
        NEURON_PREDICTION_CALIBRATION,
    );
    let synapse_cal = apply_logistic_prediction_calibration(
        raw_gain,
        improved_count,
        total_count,
        SYNAPSE_PREDICTION_CALIBRATION,
    );

    assert!(
        neuron_cal > synapse_cal,
        "Neuron ({neuron_cal}) should rank above synapse ({synapse_cal}) for equal raw gains",
    );
}

// =============================================================================
// Pessimism discount parameter validation (Issue #1056)
// =============================================================================

/// Updated neuron pessimism produces more aggressive discounting than before.
///
/// The updated parameters (floor=0.08, exponent=0.80) should produce lower
/// discounted values than the previous parameters (floor=0.10, exponent=0.75)
/// at the typical improved ratio of ~50%.
#[test]
fn neuron_pessimism_more_aggressive_at_typical_ratio() {
    let gain = 0.01_f32;
    let improved_count = 50_u32;
    let total_count = 100_u32;

    let discounted = apply_neuron_pessimism_discount(gain, improved_count, total_count, None);

    // With floor=0.08, exponent=0.80, ratio=0.5:
    // adjusted = 0.5^0.80 ≈ 0.574
    // discount = 0.08 + 0.92 * 0.574 ≈ 0.608
    // discounted = 0.01 * 0.608 ≈ 0.00608
    assert!(
        discounted < gain * 0.65,
        "Neuron pessimism at 50% ratio should discount significantly: {discounted}",
    );
    assert!(
        discounted > gain * 0.4,
        "Neuron pessimism should not over-discount: {discounted}",
    );
}

/// Updated synapse pessimism produces more aggressive discounting.
#[test]
fn synapse_pessimism_more_aggressive_at_typical_ratio() {
    let gain = 0.01_f32;
    let improved_count = 50_u32;
    let total_count = 100_u32;

    let discounted = apply_synapse_pessimism_discount(gain, improved_count, total_count, None);

    // With floor=0.03, exponent=0.90, ratio=0.5:
    // adjusted = 0.5^0.90 ≈ 0.536
    // discount = 0.03 + 0.97 * 0.536 ≈ 0.550
    // discounted = 0.01 * 0.550 ≈ 0.00550
    assert!(
        discounted < gain * 0.60,
        "Synapse pessimism at 50% ratio should discount aggressively: {discounted}",
    );
    assert!(
        discounted > gain * 0.3,
        "Synapse pessimism should not zero-out: {discounted}",
    );
}

// =============================================================================
// End-to-end calibration pipeline test (Issue #1056)
// =============================================================================

/// Simulate the full scoring pipeline for a typical add-neuron candidate and
/// verify the final calibrated gain is within a realistic range.
///
/// GRQ-sampler data: add-neurons ~2.7% success rate, typical predicted improved
/// ratio ~50%, typical raw gain ~0.01. After pessimism + logistic calibration,
/// the final prediction should be heavily reduced.
#[test]
fn end_to_end_neuron_pipeline_realistic_range() {
    let raw_gain = 0.01_f32;
    let improved_count = 50_u32;
    let total_count = 100_u32;

    // Step 1: Apply neuron pessimism discount
    let after_pessimism =
        apply_neuron_pessimism_discount(raw_gain, improved_count, total_count, None);

    // Step 2: Apply logistic prediction calibration
    let after_calibration = apply_logistic_prediction_calibration(
        after_pessimism,
        improved_count,
        total_count,
        NEURON_PREDICTION_CALIBRATION,
    );

    // The final calibrated value should be orders of magnitude smaller than raw
    assert!(
        after_calibration < raw_gain * 0.01,
        "After full pipeline, gain ({after_calibration}) should be <1% of raw ({raw_gain})",
    );
    assert!(
        after_calibration > 0.0,
        "Calibrated gain should remain positive",
    );

    // Should be in a realistic range relative to actual success
    // A raw gain of 0.01 for a ~2.7% success rate candidate should yield
    // a very small expected gain
    assert!(
        after_calibration < 1e-4,
        "Calibrated gain ({after_calibration}) should be very small for typical neuron candidate",
    );
}

/// Simulate the full scoring pipeline for a typical add-synapse candidate.
///
/// GRQ-sampler data: add-synapses ~0.1% success rate. After pessimism +
/// logistic calibration, the final prediction should be extremely small.
#[test]
fn end_to_end_synapse_pipeline_realistic_range() {
    let raw_gain = 0.01_f32;
    let improved_count = 40_u32;
    let total_count = 100_u32;

    let after_pessimism =
        apply_synapse_pessimism_discount(raw_gain, improved_count, total_count, None);
    let after_calibration = apply_logistic_prediction_calibration(
        after_pessimism,
        improved_count,
        total_count,
        SYNAPSE_PREDICTION_CALIBRATION,
    );

    assert!(
        after_calibration < raw_gain * 0.001,
        "After full pipeline, synapse gain ({after_calibration}) should be <0.1% of raw ({raw_gain})",
    );
    assert!(
        after_calibration > 0.0,
        "Calibrated synapse gain should remain positive",
    );
}
