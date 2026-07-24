#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
//! Tests for Issue #909: IDENTITY activation penalty and non-linear activation favouring.
//!
//! IDENTITY's apparent 14.9% success rate is inflated by its dominance in the
//! candidate pool (274 candidates — more than any other activation).
//! Production discovery-cache evidence shows IDENTITY neurons are frequently substituted
//! with non-linear activations like SINE for improvement.
//!
//! ## Key Behaviours Verified
//!
//! - IDENTITY receives a penalty (< 1.0), not a boost
//! - Non-linear activations (GELU, Mish, SINE, ELU) outscore IDENTITY candidates
//!   of equal raw quality
//! - SINE activation is supported with an appropriate boost
//! - SINUSOID maps to the same boost as SINE

use neat_ai_discovery::analysis::constants::{
    ACTIVATION_BOOST_ARCTAN, ACTIVATION_BOOST_BIPOLAR, ACTIVATION_BOOST_ELU, ACTIVATION_BOOST_GELU,
    ACTIVATION_BOOST_HARD_TANH, ACTIVATION_BOOST_IDENTITY, ACTIVATION_BOOST_MISH,
    ACTIVATION_BOOST_SINE, ACTIVATION_BOOST_SOFTPLUS, ACTIVATION_BOOST_SOFTSIGN,
    ACTIVATION_BOOST_TANH, activation_neuron_boost,
};
use neat_ai_discovery::analysis::synapse::apply_activation_neuron_boost;

// =============================================================================
// Issue #909: IDENTITY Must Have a Penalty
// =============================================================================

#[test]
fn identity_has_penalty_below_baseline() {
    let identity = ACTIVATION_BOOST_IDENTITY;
    assert!(
        identity < 1.0,
        "IDENTITY should have a penalty (< 1.0) due to inflated success rate, got {identity}"
    );
    // Specifically, IDENTITY should be penalised more than BIPOLAR
    // because its raw rate is artificially inflated by candidate-pool dominance
    let bipolar = ACTIVATION_BOOST_BIPOLAR;
    assert!(
        identity < bipolar,
        "IDENTITY ({identity}) should be penalised more than BIPOLAR ({bipolar}) \
         due to candidate-pool inflation"
    );
}

#[test]
fn identity_penalty_is_not_too_severe() {
    let identity = ACTIVATION_BOOST_IDENTITY;
    assert!(
        identity >= 0.5,
        "IDENTITY penalty should not go below the minimum valid range (0.5), got {identity}"
    );
    let hard_tanh = ACTIVATION_BOOST_HARD_TANH;
    assert!(
        identity >= hard_tanh,
        "IDENTITY ({identity}) should not be penalised more than HARD_TANH ({hard_tanh})"
    );
}

// =============================================================================
// Issue #909: Non-Linear Activations Score Above IDENTITY
// =============================================================================

#[test]
fn non_linear_activations_outscore_identity_with_equal_raw_gain() {
    let base_gain: f32 = 0.05;
    let identity_boosted = apply_activation_neuron_boost(base_gain, "IDENTITY");

    // All genuinely better non-linear activations should outscore IDENTITY
    let non_linear_activations = [
        ("GELU", ACTIVATION_BOOST_GELU),
        ("Mish", ACTIVATION_BOOST_MISH),
        ("ELU", ACTIVATION_BOOST_ELU),
        ("Softplus", ACTIVATION_BOOST_SOFTPLUS),
        ("SINE", ACTIVATION_BOOST_SINE),
        ("ArcTan", ACTIVATION_BOOST_ARCTAN),
        ("SOFTSIGN", ACTIVATION_BOOST_SOFTSIGN),
        ("TANH", ACTIVATION_BOOST_TANH),
    ];

    for (name, _boost) in &non_linear_activations {
        let boosted = apply_activation_neuron_boost(base_gain, name);
        assert!(
            boosted > identity_boosted,
            "{name} ({boosted}) should outscore IDENTITY ({identity_boosted}) \
             with equal raw gain {base_gain}"
        );
    }
}

#[test]
fn identity_penalty_reduces_score_gain() {
    let base_gain: f32 = 0.05;
    let identity_boosted = apply_activation_neuron_boost(base_gain, "IDENTITY");

    assert!(
        identity_boosted < base_gain,
        "IDENTITY penalty should reduce score gain: {identity_boosted} < {base_gain}"
    );
}

// =============================================================================
// Issue #909: SINE Activation Support
// =============================================================================

#[test]
fn sine_activation_has_boost_above_baseline() {
    let sine = ACTIVATION_BOOST_SINE;
    assert!(
        sine > 1.0,
        "SINE should have a boost > 1.0 based on substitution evidence, got {sine}"
    );
}

#[test]
fn sine_lookup_returns_correct_value() {
    let sine_boost = activation_neuron_boost("SINE");
    let expected = ACTIVATION_BOOST_SINE;
    assert!(
        (sine_boost - expected).abs() < f64::EPSILON,
        "SINE lookup should return {expected}, got {sine_boost}"
    );
}

#[test]
fn sinusoid_maps_to_sine_boost() {
    let sinusoid_boost = activation_neuron_boost("SINUSOID");
    let sine_boost = activation_neuron_boost("SINE");
    assert!(
        (sinusoid_boost - sine_boost).abs() < f64::EPSILON,
        "SINUSOID ({sinusoid_boost}) should map to the same boost as SINE ({sine_boost})"
    );
}

#[test]
fn sine_outscores_identity_with_equal_raw_gain() {
    let base_gain: f32 = 0.03;
    let sine_boosted = apply_activation_neuron_boost(base_gain, "SINE");
    let identity_boosted = apply_activation_neuron_boost(base_gain, "IDENTITY");

    assert!(
        sine_boosted > identity_boosted,
        "SINE ({sine_boosted}) should outscore IDENTITY ({identity_boosted}) \
         — SINE is a demonstrated improvement over IDENTITY"
    );

    // The ratio should reflect the boost difference
    let ratio = sine_boosted / identity_boosted;
    let expected_ratio = ACTIVATION_BOOST_SINE / ACTIVATION_BOOST_IDENTITY;
    assert!(
        (f64::from(ratio) - expected_ratio).abs() < 0.01,
        "Ratio should be ~{expected_ratio:.2}, got {ratio:.2}"
    );
}

// =============================================================================
// Issue #909: Ordering Integrity After Recalibration
// =============================================================================

#[test]
fn identity_ranked_below_tanh_after_recalibration() {
    // IDENTITY should now rank below TANH (which is the baseline at 1.0)
    let identity = ACTIVATION_BOOST_IDENTITY;
    let tanh = ACTIVATION_BOOST_TANH;
    assert!(
        identity < tanh,
        "IDENTITY ({identity}) should rank below TANH ({tanh}) after Issue #909 recalibration"
    );
}

#[test]
fn identity_ranked_between_hard_tanh_and_bipolar() {
    // After recalibration, IDENTITY (0.85) sits between HARD_TANH (0.80) and BIPOLAR (0.95)
    let identity = ACTIVATION_BOOST_IDENTITY;
    let hard_tanh = ACTIVATION_BOOST_HARD_TANH;
    let bipolar = ACTIVATION_BOOST_BIPOLAR;
    assert!(
        identity > hard_tanh && identity < bipolar,
        "IDENTITY ({identity}) should be between HARD_TANH ({hard_tanh}) and BIPOLAR ({bipolar})"
    );
}
