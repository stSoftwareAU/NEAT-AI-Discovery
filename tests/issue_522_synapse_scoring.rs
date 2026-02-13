//! Issue #522: Targeted tests for synapse scoring sub-module
//!
//! Tests the scoring functions in `src/analysis/synapse/scoring.rs`:
//! - `apply_source_type_boost` — input-neuron source prioritisation
//! - `apply_target_type_boost` — existing hidden target prioritisation
//! - `apply_pessimism_discount` — sample-ratio-based pessimism discount
//!
//! These tests exercise real functions with known inputs and verify
//! correct improvement calculation and boosting behaviour.

mod common;

use neat_ai_discovery::analysis::constants::{
    EXISTING_HIDDEN_TARGET_BOOST, INPUT_SOURCE_BOOST, PESSIMISM_DISCOUNT_FLOOR,
};
use neat_ai_discovery::analysis::synapse::{
    apply_pessimism_discount, apply_source_type_boost, apply_target_type_boost,
};
use std::collections::HashMap;

// =============================================================================
// apply_pessimism_discount — sample-ratio-based discount (Issue #506)
// =============================================================================

#[test]
fn pessimism_discount_all_samples_improved_gives_full_gain() {
    // When all samples improve, discount = FLOOR + (1 - FLOOR) × 1.0 = 1.0
    let gain = 0.10;
    let result = apply_pessimism_discount(gain, 100, 100);
    let expected = gain * 1.0; // Full discount = 1.0
    assert!(
        (result - expected).abs() < 1e-6,
        "All samples improved should give full gain: expected {expected}, got {result}"
    );
}

#[test]
fn pessimism_discount_no_samples_improved_gives_floor() {
    // When no samples improve, discount = FLOOR + (1 - FLOOR) × 0.0 = FLOOR
    let gain = 0.10;
    let result = apply_pessimism_discount(gain, 0, 100);
    let expected = gain * PESSIMISM_DISCOUNT_FLOOR;
    assert!(
        (result - expected).abs() < 1e-6,
        "No samples improved should give floor discount: expected {expected}, got {result}"
    );
}

#[test]
fn pessimism_discount_half_samples_improved() {
    // When half improve, discount = FLOOR + (1 - FLOOR) × 0.5
    let gain = 0.10;
    let result = apply_pessimism_discount(gain, 50, 100);
    let expected_discount = PESSIMISM_DISCOUNT_FLOOR + (1.0 - PESSIMISM_DISCOUNT_FLOOR) * 0.5;
    let expected = gain * expected_discount;
    assert!(
        (result - expected).abs() < 1e-6,
        "Half samples improved: expected {expected}, got {result}"
    );
}

#[test]
fn pessimism_discount_zero_total_gives_floor() {
    // Edge case: total_count == 0, should return gain × FLOOR
    let gain = 0.10;
    let result = apply_pessimism_discount(gain, 0, 0);
    let expected = gain * PESSIMISM_DISCOUNT_FLOOR;
    assert!(
        (result - expected).abs() < 1e-6,
        "Zero total should give floor discount: expected {expected}, got {result}"
    );
}

#[test]
fn pessimism_discount_preserves_sign_for_negative_gain() {
    // Negative gains should keep their sign after discounting
    let gain = -0.05;
    let result = apply_pessimism_discount(gain, 80, 100);
    assert!(
        result < 0.0,
        "Negative gain should remain negative after discount: got {result}"
    );
    // The absolute value should be reduced (discounted)
    assert!(
        result.abs() <= gain.abs(),
        "Discounted negative gain should have smaller magnitude: |{result}| should be <= |{gain}|"
    );
}

#[test]
fn pessimism_discount_zero_gain_stays_zero() {
    let result = apply_pessimism_discount(0.0, 50, 100);
    assert!(
        result.abs() < 1e-10,
        "Zero gain should remain zero: got {result}"
    );
}

#[test]
fn pessimism_discount_monotonically_increases_with_improved_ratio() {
    // As the improved ratio increases, the discounted gain should increase
    let gain = 0.10;
    let mut prev = apply_pessimism_discount(gain, 0, 100);
    for improved in (10..=100).step_by(10) {
        let current = apply_pessimism_discount(gain, improved, 100);
        assert!(
            current >= prev - 1e-7,
            "Discount should be monotonically non-decreasing: \
            at {improved}/100, got {current} < previous {prev}"
        );
        prev = current;
    }
}

// =============================================================================
// apply_source_type_boost — input neurons as synapse sources (Issue #467)
// =============================================================================

#[test]
fn source_boost_applied_to_various_input_indices() {
    // All "input-N" patterns should receive the boost
    for idx in [0, 1, 5, 42, 999] {
        let uuid = format!("input-{idx}");
        let gain = 0.10;
        let boosted = apply_source_type_boost(gain, &uuid);
        let expected = gain * INPUT_SOURCE_BOOST as f32;
        assert!(
            (boosted - expected).abs() < 1e-6,
            "Input source '{uuid}' should be boosted: expected {expected}, got {boosted}"
        );
    }
}

#[test]
fn source_boost_not_applied_to_hidden_uuids() {
    let test_uuids = [
        "hidden-abc-123",
        "some-random-uuid",
        "output-0",
        "discovery-hidden-xyz",
    ];
    for uuid in test_uuids {
        let gain = 0.10;
        let result = apply_source_type_boost(gain, uuid);
        assert!(
            (result - gain).abs() < 1e-6,
            "Non-input source '{uuid}' should not be boosted: expected {gain}, got {result}"
        );
    }
}

#[test]
fn source_boost_negative_gain_boosted_correctly() {
    // Even negative gains should be multiplied by the boost factor
    let gain = -0.05;
    let boosted = apply_source_type_boost(gain, "input-3");
    let expected = gain * INPUT_SOURCE_BOOST as f32;
    assert!(
        (boosted - expected).abs() < 1e-6,
        "Negative gain should be boosted: expected {expected}, got {boosted}"
    );
    // Negative gain × boost > 1.0 makes it more negative (larger magnitude)
    assert!(
        boosted < gain,
        "Negative gain should become more negative after boost"
    );
}

// =============================================================================
// apply_target_type_boost — existing hidden neurons as targets (Issue #468)
// =============================================================================

#[test]
fn target_boost_applied_to_existing_hidden_neuron() {
    let gain = 0.10;
    let mut type_map = HashMap::new();
    type_map.insert("hidden-abc".to_string(), "hidden".to_string());

    let boosted = apply_target_type_boost(gain, "hidden-abc", &type_map);
    let expected = gain * EXISTING_HIDDEN_TARGET_BOOST as f32;
    assert!(
        (boosted - expected).abs() < 1e-6,
        "Existing hidden target should be boosted: expected {expected}, got {boosted}"
    );
}

#[test]
fn target_boost_not_applied_to_output_neuron() {
    let gain = 0.10;
    let mut type_map = HashMap::new();
    type_map.insert("output-0".to_string(), "output".to_string());

    let result = apply_target_type_boost(gain, "output-0", &type_map);
    assert!(
        (result - gain).abs() < 1e-6,
        "Output target should not be boosted: expected {gain}, got {result}"
    );
}

#[test]
fn target_boost_not_applied_to_missing_neuron() {
    let gain = 0.10;
    let type_map = HashMap::new(); // Empty map — neuron not found

    let result = apply_target_type_boost(gain, "unknown-uuid", &type_map);
    assert!(
        (result - gain).abs() < 1e-6,
        "Unknown target should not be boosted: expected {gain}, got {result}"
    );
}

#[test]
fn target_boost_multiple_neuron_types() {
    let gain = 0.10;
    let mut type_map = HashMap::new();
    type_map.insert("hidden-a".to_string(), "hidden".to_string());
    type_map.insert("hidden-b".to_string(), "hidden".to_string());
    type_map.insert("output-0".to_string(), "output".to_string());
    type_map.insert("input-0".to_string(), "input".to_string());

    // Hidden neurons should be boosted
    let boosted_a = apply_target_type_boost(gain, "hidden-a", &type_map);
    let boosted_b = apply_target_type_boost(gain, "hidden-b", &type_map);
    assert!(boosted_a > gain, "hidden-a should be boosted");
    assert!(boosted_b > gain, "hidden-b should be boosted");
    assert!(
        (boosted_a - boosted_b).abs() < 1e-6,
        "Both hidden neurons should get same boost"
    );

    // Non-hidden neurons should not be boosted
    let output_result = apply_target_type_boost(gain, "output-0", &type_map);
    let input_result = apply_target_type_boost(gain, "input-0", &type_map);
    assert!(
        (output_result - gain).abs() < 1e-6,
        "Output should not be boosted"
    );
    assert!(
        (input_result - gain).abs() < 1e-6,
        "Input should not be boosted"
    );
}

// =============================================================================
// Combined boost and discount interactions
// =============================================================================

#[test]
fn combined_source_and_target_boost_stacks_multiplicatively() {
    // When both source and target boosts apply, the result should be
    // gain × source_boost × target_boost (applied sequentially)
    let gain = 0.10;
    let mut type_map = HashMap::new();
    type_map.insert("hidden-target".to_string(), "hidden".to_string());

    // Apply source boost first (as production does)
    let after_source = apply_source_type_boost(gain, "input-0");
    let after_target = apply_target_type_boost(after_source, "hidden-target", &type_map);

    let expected = gain * INPUT_SOURCE_BOOST as f32 * EXISTING_HIDDEN_TARGET_BOOST as f32;
    assert!(
        (after_target - expected).abs() < 1e-5,
        "Combined boosts should stack: expected {expected}, got {after_target}"
    );
    assert!(after_target > gain, "Combined boosts should increase gain");
}

#[test]
fn pessimism_discount_then_boosts_gives_correct_order() {
    // Production applies pessimism first, then source boost, then target boost
    let gain = 0.10;
    let mut type_map = HashMap::new();
    type_map.insert("hidden-target".to_string(), "hidden".to_string());

    // Apply in production order
    let after_pessimism = apply_pessimism_discount(gain, 80, 100);
    let after_source = apply_source_type_boost(after_pessimism, "input-0");
    let after_target = apply_target_type_boost(after_source, "hidden-target", &type_map);

    // Verify the result is reasonable
    assert!(
        after_target > 0.0,
        "Final score should be positive: got {after_target}"
    );
    assert!(
        after_target.is_finite(),
        "Final score should be finite: got {after_target}"
    );
    // With pessimism discount, the gain should be less than just applying boosts alone
    let boosts_only = gain * INPUT_SOURCE_BOOST as f32 * EXISTING_HIDDEN_TARGET_BOOST as f32;
    assert!(
        after_target < boosts_only,
        "Pessimism discount should reduce gain below boosts-only: {after_target} should be < {boosts_only}"
    );
}
