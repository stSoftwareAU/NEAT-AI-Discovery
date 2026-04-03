//! Tests for Issue #982: Verify backward-compatible access to scoring functions
//! after splitting scoring.rs into boost, improvement, and discounting sub-modules.
//!
//! These tests verify that all public scoring functions remain accessible through
//! the same `analysis::synapse::` paths after the module restructuring.

use neat_ai_discovery::analysis::synapse::{
    apply_activation_neuron_boost, apply_neuron_pessimism_discount, apply_pessimism_discount,
    apply_prediction_calibration, apply_source_type_boost, apply_synapse_pessimism_discount,
    apply_target_type_boost,
};
use std::collections::HashMap;

// =============================================================================
// Boost Function Access (scoring/boost_functions.rs)
// =============================================================================

/// Verify `apply_source_type_boost` is accessible and applies input boost.
#[test]
fn test_source_type_boost_accessible_after_split() {
    let boosted = apply_source_type_boost(1.0, "input-0");
    assert!(
        boosted > 1.0,
        "Input source should receive a boost, got {boosted}"
    );
}

/// Verify `apply_source_type_boost` applies hidden boost.
#[test]
fn test_hidden_source_boost_accessible_after_split() {
    let boosted = apply_source_type_boost(1.0, "hidden-abc");
    assert!(
        boosted > 1.0,
        "Hidden source should receive a boost, got {boosted}"
    );
}

/// Verify `apply_source_type_boost` gives no boost to output neurons.
#[test]
fn test_output_source_no_boost_after_split() {
    let result = apply_source_type_boost(1.0, "output-0");
    assert!(
        (result - 1.0).abs() < f32::EPSILON,
        "Output source should receive no boost, got {result}"
    );
}

/// Verify `apply_target_type_boost` is accessible and applies hidden target boost.
#[test]
fn test_target_type_boost_accessible_after_split() {
    let mut type_map = HashMap::new();
    type_map.insert("hidden-1", "hidden");
    let boosted = apply_target_type_boost(1.0, "hidden-1", &type_map);
    assert!(
        boosted > 1.0,
        "Hidden target should receive a boost, got {boosted}"
    );
}

/// Verify `apply_activation_neuron_boost` is accessible.
#[test]
fn test_activation_neuron_boost_accessible_after_split() {
    let result = apply_activation_neuron_boost(1.0, "RELU");
    assert!(
        result.is_finite(),
        "Activation neuron boost should return finite value"
    );
}

// =============================================================================
// Discounting Function Access (scoring/discounting.rs)
// =============================================================================

/// Verify `apply_pessimism_discount` is accessible and reduces gain.
#[test]
fn test_pessimism_discount_accessible_after_split() {
    let discounted = apply_pessimism_discount(1.0, 5, 10);
    assert!(
        discounted < 1.0,
        "Pessimism discount should reduce gain, got {discounted}"
    );
    assert!(
        discounted > 0.0,
        "Pessimism discount should not eliminate gain, got {discounted}"
    );
}

/// Verify `apply_neuron_pessimism_discount` is accessible.
#[test]
fn test_neuron_pessimism_discount_accessible_after_split() {
    let discounted = apply_neuron_pessimism_discount(1.0, 5, 10);
    assert!(
        discounted < 1.0,
        "Neuron pessimism discount should reduce gain, got {discounted}"
    );
}

/// Verify `apply_synapse_pessimism_discount` is accessible.
#[test]
fn test_synapse_pessimism_discount_accessible_after_split() {
    let discounted = apply_synapse_pessimism_discount(1.0, 5, 10);
    assert!(
        discounted < 1.0,
        "Synapse pessimism discount should reduce gain, got {discounted}"
    );
}

/// Verify `apply_prediction_calibration` is accessible.
#[test]
fn test_prediction_calibration_accessible_after_split() {
    let calibrated = apply_prediction_calibration(1.0, 0.5);
    assert!(
        (calibrated - 0.5).abs() < f32::EPSILON,
        "Prediction calibration should multiply by factor, got {calibrated}"
    );
}

// =============================================================================
// Cross-module consistency checks
// =============================================================================

/// Verify that synapse-specific discount is more aggressive than neuron-specific,
/// which is more aggressive than generic discount (Issue #789, #791).
#[test]
fn test_discount_ordering_preserved_after_split() {
    let generic = apply_pessimism_discount(1.0, 5, 10);
    let neuron = apply_neuron_pessimism_discount(1.0, 5, 10);
    let synapse = apply_synapse_pessimism_discount(1.0, 5, 10);

    assert!(
        synapse <= neuron,
        "Synapse discount ({synapse}) should be <= neuron discount ({neuron})"
    );
    assert!(
        neuron <= generic,
        "Neuron discount ({neuron}) should be <= generic discount ({generic})"
    );
}
