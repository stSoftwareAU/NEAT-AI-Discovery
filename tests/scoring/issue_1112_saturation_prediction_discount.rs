//! Issue #1112: Saturation-aware prediction discount for neuron candidates.
//!
//! Production data shows predictions of ~0.01 for neuron candidates targeting
//! `HARD_TANH` at full activation range, while actual results are ~-0.00005.
//! Saturation resistance means the target cannot move much in response to
//! perturbations, so predictions must be discounted proportionally.

use neat_ai_discovery::analysis::constants::SATURATION_DISCOUNT_AGGRESSIVE;
use neat_ai_discovery::analysis::synapse::apply_saturation_prediction_discount;

// =============================================================================
// Compile-time constant validation
// =============================================================================

const _: () = assert!(SATURATION_DISCOUNT_AGGRESSIVE > 0.0);
const _: () = assert!(SATURATION_DISCOUNT_AGGRESSIVE < 0.5);

// =============================================================================
// Unit tests
// =============================================================================

/// Candidate targeting `HARD_TANH` at [-1, 1] (fully saturated, factor = 1.0)
/// should have prediction reduced by ≥60%.
#[test]
fn saturated_hard_tanh_target_gets_heavy_discount() {
    let raw_gain = 0.01_f32;
    // `HARD_TANH` at [-1, 1] → saturation_factor ≈ 1.0
    let discounted = apply_saturation_prediction_discount(raw_gain, Some(1.0));

    let reduction = 1.0 - (discounted / raw_gain);
    assert!(
        reduction >= 0.60,
        "Saturated target prediction should be reduced by ≥60%, but reduction was {:.1}%",
        reduction * 100.0,
    );
    assert!(
        discounted > 0.0,
        "Discounted gain should remain positive, got {discounted}",
    );
}

/// Candidate targeting IDENTITY (unbounded, no saturation factor) should
/// receive no additional discount.
#[test]
fn non_saturated_identity_target_unchanged() {
    let raw_gain = 0.01_f32;
    // IDENTITY → unbounded → target_saturation_factor = None
    let discounted = apply_saturation_prediction_discount(raw_gain, None);

    assert!(
        (discounted - raw_gain).abs() < 1e-10,
        "Non-saturated target should be unchanged: expected {raw_gain}, got {discounted}",
    );
}

/// Targets at the saturation threshold boundary (0.9) should receive no discount.
#[test]
fn threshold_boundary_gets_no_discount() {
    let raw_gain = 0.05_f32;
    let discounted = apply_saturation_prediction_discount(raw_gain, Some(0.90));

    assert!(
        (discounted - raw_gain).abs() < 1e-6,
        "Target at threshold (0.9) should get no discount: expected {raw_gain}, got {discounted}",
    );
}

/// Targets below the saturation threshold should receive no discount even
/// when a saturation factor is provided.
#[test]
fn below_threshold_gets_no_discount() {
    let raw_gain = 0.05_f32;
    let discounted = apply_saturation_prediction_discount(raw_gain, Some(0.5));

    assert!(
        (discounted - raw_gain).abs() < 1e-6,
        "Target below threshold should get no discount: expected {raw_gain}, got {discounted}",
    );
}

/// Discount should be proportional to saturation — a target at 0.95 should
/// receive less discount than a fully saturated target (1.0).
#[test]
fn discount_is_proportional_to_saturation() {
    let raw_gain = 0.01_f32;
    let at_95 = apply_saturation_prediction_discount(raw_gain, Some(0.95));
    let at_100 = apply_saturation_prediction_discount(raw_gain, Some(1.0));

    assert!(
        at_95 > at_100,
        "Partially saturated ({at_95}) should retain more gain than fully saturated ({at_100})",
    );
    assert!(
        at_95 < raw_gain,
        "Partially saturated target should still receive some discount",
    );
}

/// Zero gain should remain zero after saturation discount.
#[test]
fn zero_gain_remains_zero() {
    let discounted = apply_saturation_prediction_discount(0.0, Some(1.0));
    assert!(
        discounted.abs() < 1e-15,
        "Zero gain should remain zero, got {discounted}",
    );
}

/// Negative gain should preserve sign after saturation discount.
#[test]
fn negative_gain_preserves_sign() {
    let raw_gain = -0.03_f32;
    let discounted = apply_saturation_prediction_discount(raw_gain, Some(1.0));

    assert!(
        discounted < 0.0,
        "Negative gain should remain negative after discount",
    );
    assert!(
        discounted.abs() < raw_gain.abs(),
        "Absolute value should decrease: |{discounted}| should be < |{raw_gain}|",
    );
}
