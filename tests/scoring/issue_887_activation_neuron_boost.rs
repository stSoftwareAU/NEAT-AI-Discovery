#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
//! Tests for Issue #887: Activation-function-aware scoring for add-neuron candidates.
//!
//! Production discovery-cache analysis shows dramatic differences in success rates by
//! activation function. This module verifies that per-activation boost/penalty
//! multipliers are correctly defined and applied during neuron candidate scoring.
//!
//! ## Key Behaviours Verified
//!
//! - Boost constants are within valid range \[0.5, 2.0\]
//! - High-success activations get meaningful boosts (> 1.0)
//! - Low-success activations get penalty multipliers (< 1.0)
//! - Boost is correctly applied to `expected_creature_score_gain`
//! - Unknown activations receive neutral boost (1.0)
//! - `ReLU` receives neutral boost (1.0) as it is evaluated separately

use neat_ai_discovery::analysis::constants::{
    ACTIVATION_BOOST_ABSOLUTE, ACTIVATION_BOOST_ARCTAN, ACTIVATION_BOOST_BENT_IDENTITY,
    ACTIVATION_BOOST_BIPOLAR, ACTIVATION_BOOST_CLIPPED, ACTIVATION_BOOST_ELU,
    ACTIVATION_BOOST_GELU, ACTIVATION_BOOST_HARD_TANH, ACTIVATION_BOOST_IDENTITY,
    ACTIVATION_BOOST_MAX, ACTIVATION_BOOST_MIN, ACTIVATION_BOOST_MISH, ACTIVATION_BOOST_RELU6,
    ACTIVATION_BOOST_SINE, ACTIVATION_BOOST_SOFTPLUS, ACTIVATION_BOOST_SOFTSIGN,
    ACTIVATION_BOOST_TANH, activation_neuron_boost,
};
use neat_ai_discovery::analysis::synapse::apply_activation_neuron_boost;

// =============================================================================
// Compile-Time Constant Range Validation
// =============================================================================

// All boost constants must be within the valid range [ACTIVATION_BOOST_MIN, ACTIVATION_BOOST_MAX].
const _: () = assert!(ACTIVATION_BOOST_GELU >= 0.5);
const _: () = assert!(ACTIVATION_BOOST_GELU <= 2.0);
const _: () = assert!(ACTIVATION_BOOST_ABSOLUTE >= 0.5);
const _: () = assert!(ACTIVATION_BOOST_ABSOLUTE <= 2.0);
const _: () = assert!(ACTIVATION_BOOST_MISH >= 0.5);
const _: () = assert!(ACTIVATION_BOOST_MISH <= 2.0);
const _: () = assert!(ACTIVATION_BOOST_RELU6 >= 0.5);
const _: () = assert!(ACTIVATION_BOOST_RELU6 <= 2.0);
const _: () = assert!(ACTIVATION_BOOST_BENT_IDENTITY >= 0.5);
const _: () = assert!(ACTIVATION_BOOST_BENT_IDENTITY <= 2.0);
const _: () = assert!(ACTIVATION_BOOST_ELU >= 0.5);
const _: () = assert!(ACTIVATION_BOOST_ELU <= 2.0);
const _: () = assert!(ACTIVATION_BOOST_SOFTPLUS >= 0.5);
const _: () = assert!(ACTIVATION_BOOST_SOFTPLUS <= 2.0);
const _: () = assert!(ACTIVATION_BOOST_SINE >= 0.5);
const _: () = assert!(ACTIVATION_BOOST_SINE <= 2.0);
const _: () = assert!(ACTIVATION_BOOST_ARCTAN >= 0.5);
const _: () = assert!(ACTIVATION_BOOST_ARCTAN <= 2.0);
const _: () = assert!(ACTIVATION_BOOST_SOFTSIGN >= 0.5);
const _: () = assert!(ACTIVATION_BOOST_SOFTSIGN <= 2.0);
const _: () = assert!(ACTIVATION_BOOST_CLIPPED >= 0.5);
const _: () = assert!(ACTIVATION_BOOST_CLIPPED <= 2.0);
const _: () = assert!(ACTIVATION_BOOST_IDENTITY >= 0.5);
const _: () = assert!(ACTIVATION_BOOST_IDENTITY <= 2.0);
const _: () = assert!(ACTIVATION_BOOST_TANH >= 0.5);
const _: () = assert!(ACTIVATION_BOOST_TANH <= 2.0);
const _: () = assert!(ACTIVATION_BOOST_BIPOLAR >= 0.5);
const _: () = assert!(ACTIVATION_BOOST_BIPOLAR <= 2.0);
const _: () = assert!(ACTIVATION_BOOST_HARD_TANH >= 0.5);
const _: () = assert!(ACTIVATION_BOOST_HARD_TANH <= 2.0);

// Range bounds are valid.
const _: () = assert!(ACTIVATION_BOOST_MIN > 0.0);
const _: () = assert!(ACTIVATION_BOOST_MAX > 1.0);

// =============================================================================
// Boost Ordering (High-Success > Baseline > Low-Success)
// =============================================================================

#[test]
fn high_success_activations_get_boost_above_baseline() {
    // Top performers should have boost > 1.0
    let gelu = ACTIVATION_BOOST_GELU;
    let absolute = ACTIVATION_BOOST_ABSOLUTE;
    let mish = ACTIVATION_BOOST_MISH;
    let relu6 = ACTIVATION_BOOST_RELU6;
    assert!(
        gelu > 1.0,
        "GELU (60% success) should have boost > 1.0, got {gelu}"
    );
    assert!(
        absolute > 1.0,
        "ABSOLUTE (73.6% success) should have boost > 1.0, got {absolute}"
    );
    assert!(
        mish > 1.0,
        "Mish (48% success) should have boost > 1.0, got {mish}"
    );
    assert!(
        relu6 > 1.0,
        "ReLU6 (50% success) should have boost > 1.0, got {relu6}"
    );
}

#[test]
fn low_success_activations_get_penalty_below_baseline() {
    // Bottom performers should have boost < 1.0
    let hard_tanh = ACTIVATION_BOOST_HARD_TANH;
    let bipolar = ACTIVATION_BOOST_BIPOLAR;
    let identity = ACTIVATION_BOOST_IDENTITY;
    assert!(
        hard_tanh < 1.0,
        "HARD_TANH (7.2% success) should have boost < 1.0, got {hard_tanh}"
    );
    assert!(
        bipolar < 1.0,
        "BIPOLAR (12.1% success) should have boost < 1.0, got {bipolar}"
    );
    // Issue #909: IDENTITY penalised due to inflated success rate from candidate-pool dominance
    assert!(
        identity < 1.0,
        "IDENTITY (inflated 14.9%) should have penalty < 1.0, got {identity}"
    );
}

#[test]
fn tanh_is_baseline_neutral() {
    // TANH matches the baseline success rate (13.9%) and should be neutral (1.0)
    let tanh = ACTIVATION_BOOST_TANH;
    assert!(
        (tanh - 1.0).abs() < f64::EPSILON,
        "TANH (13.9% = baseline) should have neutral boost 1.0, got {tanh}"
    );
}

#[test]
fn boost_ordering_reflects_success_rates() {
    // Activations with higher success rates should have higher boosts.
    // Use runtime variables to avoid clippy::assertions_on_constants.
    let boosts: Vec<f64> = vec![
        ACTIVATION_BOOST_GELU,
        ACTIVATION_BOOST_MISH,
        ACTIVATION_BOOST_ELU,
        ACTIVATION_BOOST_SOFTPLUS,
        ACTIVATION_BOOST_SINE,
        ACTIVATION_BOOST_ARCTAN,
        ACTIVATION_BOOST_TANH,
        ACTIVATION_BOOST_BIPOLAR,
        ACTIVATION_BOOST_IDENTITY,
        ACTIVATION_BOOST_HARD_TANH,
    ];
    let names = [
        "GELU",
        "Mish",
        "ELU",
        "Softplus",
        "SINE",
        "ArcTan",
        "TANH",
        "BIPOLAR",
        "IDENTITY",
        "HARD_TANH",
    ];
    for i in 0..boosts.len() - 1 {
        assert!(
            boosts[i] >= boosts[i + 1],
            "{} ({}) should be >= {} ({})",
            names[i],
            boosts[i],
            names[i + 1],
            boosts[i + 1]
        );
    }
}

// =============================================================================
// Lookup Function
// =============================================================================

#[test]
fn activation_neuron_boost_returns_correct_values() {
    assert!((activation_neuron_boost("GELU") - ACTIVATION_BOOST_GELU).abs() < f64::EPSILON);
    assert!((activation_neuron_boost("ABSOLUTE") - ACTIVATION_BOOST_ABSOLUTE).abs() < f64::EPSILON);
    assert!(
        (activation_neuron_boost("HARD_TANH") - ACTIVATION_BOOST_HARD_TANH).abs() < f64::EPSILON
    );
    assert!((activation_neuron_boost("TANH") - ACTIVATION_BOOST_TANH).abs() < f64::EPSILON);
}

#[test]
fn unknown_activation_gets_neutral_boost() {
    let boost = activation_neuron_boost("UnknownSquash");
    assert!(
        (boost - 1.0).abs() < f64::EPSILON,
        "Unknown activation should get neutral boost 1.0, got {boost}"
    );
}

#[test]
fn relu_gets_neutral_boost() {
    let boost = activation_neuron_boost("ReLU");
    assert!(
        (boost - 1.0).abs() < f64::EPSILON,
        "ReLU should get neutral boost 1.0, got {boost}"
    );
}

// =============================================================================
// Boost Application to Score Gain
// =============================================================================

#[test]
fn boost_applied_to_expected_score_gain() {
    let base_gain: f32 = 0.05;

    // GELU (boost 2.0) should double the gain
    let gelu_gain = apply_activation_neuron_boost(base_gain, "GELU");
    let expected_gelu = base_gain * ACTIVATION_BOOST_GELU as f32;
    assert!(
        (gelu_gain - expected_gelu).abs() < 1e-6,
        "GELU boosted gain should be {expected_gelu}, got {gelu_gain}"
    );
    assert!(
        gelu_gain > base_gain,
        "GELU boost should increase gain: {gelu_gain} > {base_gain}"
    );

    // HARD_TANH (boost 0.80) should reduce the gain
    let hard_tanh_gain = apply_activation_neuron_boost(base_gain, "HARD_TANH");
    let expected_ht = base_gain * ACTIVATION_BOOST_HARD_TANH as f32;
    assert!(
        (hard_tanh_gain - expected_ht).abs() < 1e-6,
        "HARD_TANH penalised gain should be {expected_ht}, got {hard_tanh_gain}"
    );
    assert!(
        hard_tanh_gain < base_gain,
        "HARD_TANH penalty should decrease gain: {hard_tanh_gain} < {base_gain}"
    );
}

#[test]
fn boost_preserves_zero_gain() {
    let zero_gain = apply_activation_neuron_boost(0.0, "GELU");
    assert!(
        zero_gain.abs() < f32::EPSILON,
        "Boost of zero gain should remain zero, got {zero_gain}"
    );
}

#[test]
fn boost_preserves_negative_gain_sign() {
    let neg_gain = apply_activation_neuron_boost(-0.01, "GELU");
    assert!(
        neg_gain < 0.0,
        "Boost of negative gain should remain negative, got {neg_gain}"
    );
}

#[test]
fn gelu_candidate_ranked_above_hard_tanh_with_equal_raw_gain() {
    let base_gain: f32 = 0.03;
    let gelu_boosted = apply_activation_neuron_boost(base_gain, "GELU");
    let hard_tanh_boosted = apply_activation_neuron_boost(base_gain, "HARD_TANH");

    assert!(
        gelu_boosted > hard_tanh_boosted,
        "GELU ({gelu_boosted}) should rank above HARD_TANH ({hard_tanh_boosted}) with equal raw gain"
    );

    let ratio = gelu_boosted / hard_tanh_boosted;
    let expected_ratio = ACTIVATION_BOOST_GELU / ACTIVATION_BOOST_HARD_TANH;
    assert!(
        (f64::from(ratio) - expected_ratio).abs() < 0.01,
        "Ratio should be ~{expected_ratio:.2}, got {ratio:.2}"
    );
}

#[test]
fn identity_deprioritised_relative_to_gelu() {
    let base_gain: f32 = 0.05;
    let identity_boosted = apply_activation_neuron_boost(base_gain, "IDENTITY");
    let gelu_boosted = apply_activation_neuron_boost(base_gain, "GELU");

    assert!(
        gelu_boosted > identity_boosted,
        "GELU ({gelu_boosted}) should be prioritised over IDENTITY ({identity_boosted})"
    );
}
