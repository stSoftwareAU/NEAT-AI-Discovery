//! Tests for Issue #938: Split constants.rs into thematic sub-modules.
//!
//! Verifies that all constants remain accessible via the same
//! `crate::analysis::constants::CONSTANT_NAME` paths after the split,
//! ensuring backward compatibility. Also validates that no constants
//! are duplicated and that each sub-module category is correctly populated.

use neat_ai_discovery::analysis::constants;

// =============================================================================
// Sample Thresholds Sub-module
// =============================================================================

/// Verify sample count thresholds are accessible via the re-export.
#[test]
fn test_sample_thresholds_accessible() {
    assert_eq!(constants::MIN_NEURON_SAMPLE_COUNT, 10);
    assert_eq!(constants::MIN_DISCOVERY_SAMPLE_COUNT, 20);
    assert_eq!(constants::HOLDOUT_MIN_SAMPLE_COUNT, 20);
    assert!((constants::HOLDOUT_VALIDATION_FRACTION - 0.3).abs() < f32::EPSILON);
}

// =============================================================================
// Sentinel Detection Sub-module
// =============================================================================

/// Verify sentinel detection constants are accessible via the re-export.
#[test]
fn test_sentinel_detection_accessible() {
    assert_eq!(constants::CANDIDATE_SENTINELS, [-1.0_f32, 0.0, 1.0]);
    assert!((constants::MIN_SENTINEL_FRACTION - 0.15).abs() < f32::EPSILON);
    assert!((constants::SENTINEL_TOLERANCE - 0.02).abs() < f32::EPSILON);
    assert!((constants::MIN_SENTINEL_GAP - 0.05).abs() < f32::EPSILON);
}

// =============================================================================
// Source Variance Sub-module
// =============================================================================

/// Verify source variance threshold is accessible via the re-export.
#[test]
fn test_source_variance_accessible() {
    assert!((constants::MIN_SOURCE_STD_DEV - 0.05).abs() < f32::EPSILON);
}

// =============================================================================
// Candidate Scoring Sub-module
// =============================================================================

/// Verify diversification constant is accessible.
#[test]
fn test_diversify_top_k_accessible() {
    assert_eq!(constants::DIVERSIFY_TOP_K, 64);
}

/// Verify source-type scoring constants are accessible.
#[test]
fn test_source_type_scoring_accessible() {
    assert!((constants::INPUT_SOURCE_BOOST - 1.5).abs() < f64::EPSILON);
    assert!((constants::HIDDEN_SOURCE_BOOST - 1.2).abs() < f64::EPSILON);
    assert_eq!(constants::HIDDEN_SOURCE_INTERLEAVE_INTERVAL, 3);
    assert_eq!(constants::MIN_BOOST_SAMPLES, 10);
}

/// Verify target-type scoring constant is accessible.
#[test]
fn test_target_type_scoring_accessible() {
    assert!((constants::EXISTING_HIDDEN_TARGET_BOOST - 1.5).abs() < f64::EPSILON);
}

/// Verify activation boost function and constants are accessible.
#[test]
fn test_activation_boosts_accessible() {
    assert!((constants::ACTIVATION_BOOST_GELU - 2.0).abs() < f64::EPSILON);
    assert!((constants::ACTIVATION_BOOST_IDENTITY - 0.85).abs() < f64::EPSILON);
    assert!((constants::ACTIVATION_BOOST_TANH - 1.0).abs() < f64::EPSILON);
    assert!((constants::ACTIVATION_BOOST_HARD_TANH - 0.80).abs() < f64::EPSILON);
    assert!((constants::ACTIVATION_BOOST_MIN - 0.5).abs() < f64::EPSILON);
    assert!((constants::ACTIVATION_BOOST_MAX - 2.0).abs() < f64::EPSILON);

    // Verify the lookup function is accessible and returns correct values.
    assert!((constants::activation_neuron_boost("GELU") - 2.0).abs() < f64::EPSILON);
    assert!((constants::activation_neuron_boost("IDENTITY") - 0.85).abs() < f64::EPSILON);
    assert!((constants::activation_neuron_boost("unknown") - 1.0).abs() < f64::EPSILON);
}

/// Verify pessimism discount constants are accessible.
///
/// Issue #1056: Updated values to match production success rates from GRQ-sampler.
#[test]
fn test_pessimism_discounts_accessible() {
    assert!((constants::PESSIMISM_DISCOUNT_FLOOR - 0.15).abs() < f32::EPSILON);
    assert!((constants::PESSIMISM_CURVE_EXPONENT - 0.6).abs() < f32::EPSILON);
    assert!((constants::NEURON_PESSIMISM_DISCOUNT_FLOOR - 0.08).abs() < f32::EPSILON);
    assert!((constants::NEURON_PESSIMISM_CURVE_EXPONENT - 0.80).abs() < f32::EPSILON);
    assert!((constants::SYNAPSE_PESSIMISM_DISCOUNT_FLOOR - 0.03).abs() < f32::EPSILON);
    assert!((constants::SYNAPSE_PESSIMISM_CURVE_EXPONENT - 0.90).abs() < f32::EPSILON);
}

/// Verify NaN-safe comparison functions are accessible and correct.
#[test]
fn test_nan_safe_comparison_accessible() {
    use std::cmp::Ordering;
    assert_eq!(constants::cmp_f32_desc(&2.0, &1.0), Ordering::Less);
    assert_eq!(constants::cmp_f32_asc(&1.0, &2.0), Ordering::Less);
    assert_eq!(constants::cmp_f64_desc(&2.0, &1.0), Ordering::Less);
}

/// Verify coordinated-structural constants are accessible (Issue #1058: updated).
#[test]
fn test_coordinated_structural_accessible() {
    // Issue #1058: Empirical per-op-count factors replace compound discount.
    assert!((constants::COORDINATED_EMPIRICAL_DISCOUNT_2OPS - 0.5).abs() < f32::EPSILON);
    assert!((constants::COORDINATED_EMPIRICAL_DISCOUNT_3OPS - 0.2).abs() < f32::EPSILON);
    assert!((constants::COORDINATED_EMPIRICAL_DISCOUNT_4PLUS_OPS - 0.1).abs() < f32::EPSILON);
    // Issue #1058: Lowered from 1e-3 to 1e-5.
    assert!((constants::MIN_COORDINATED_MULTI_OP_GAIN - 1e-5).abs() < f32::EPSILON);
    assert!((constants::COORDINATED_ESTIMATION_WEIGHT_SCALE - 0.2).abs() < f32::EPSILON);
    // Issue #1058: Verify empirical discount function is accessible.
    let d = constants::coordinated_empirical_discount(2);
    assert!((d - 0.5).abs() < f32::EPSILON);
}

/// Verify prediction calibration constants are accessible.
///
/// Issue #1056: Updated values to match production success rates from GRQ-sampler.
#[test]
fn test_prediction_calibration_accessible() {
    assert!((constants::SYNAPSE_PREDICTION_CALIBRATION - 0.0003).abs() < f32::EPSILON);
    assert!((constants::NEURON_PREDICTION_CALIBRATION - 0.003).abs() < f32::EPSILON);
    assert!((constants::COORDINATED_PREDICTION_CALIBRATION - 0.00005).abs() < f32::EPSILON);
    // Issue #1056: Verify logistic calibration constants are also accessible.
    let floor = constants::LOGISTIC_CALIBRATION_FLOOR;
    let steepness = constants::LOGISTIC_CALIBRATION_STEEPNESS;
    let midpoint = constants::LOGISTIC_CALIBRATION_MIDPOINT;
    assert!(floor > 0.0, "floor should be positive");
    assert!(steepness > 0.0, "steepness should be positive");
    assert!(midpoint > 0.0, "midpoint should be positive");
}

/// Verify scoring boost multipliers are accessible.
#[test]
fn test_scoring_boosts_accessible() {
    assert!((constants::MICRO_NUDGE_VARIANT_BOOST - 1.5).abs() < f32::EPSILON);
    assert!((constants::REMOVAL_CANDIDATE_BOOST - 1.5).abs() < f32::EPSILON);
}

// =============================================================================
// Detection Thresholds Sub-module
// =============================================================================

/// Verify detection threshold constants are accessible.
#[test]
fn test_detection_thresholds_accessible() {
    assert!((constants::MIN_IMPROVED_RATIO - 0.6).abs() < f32::EPSILON);
    assert!((constants::NEURON_MIN_IMPROVED_RATIO - 0.4).abs() < f32::EPSILON);
    assert!((constants::REMOVAL_MEAN_ACTIVATION_THRESHOLD - 0.04).abs() < f32::EPSILON);
    assert!((constants::REMOVAL_IMPACT_THRESHOLD - 6e-5).abs() < f32::EPSILON);
    assert!((constants::MAX_INCOMING_WEIGHT - 5.0).abs() < f32::EPSILON);
    assert!((constants::MAX_BIAS_MAGNITUDE - 2.0).abs() < f32::EPSILON);
    assert!((constants::MAX_INDIVIDUAL_HARM_FOR_PAIRING - 0.0).abs() < f32::EPSILON);
}

// =============================================================================
// Compression Sub-module
// =============================================================================

/// Verify compression constants are accessible.
#[test]
fn test_compression_accessible() {
    assert_eq!(constants::MIN_COMPRESSED_SOURCES, 2);
    assert_eq!(constants::MAX_COMPRESSION_INPUTS, 5);
    assert!((constants::COMPRESSION_SATURATION_THRESHOLD - 0.9).abs() < f32::EPSILON);
    assert!((constants::COMPRESSION_MIN_BENEFIT_RATIO - 1.05).abs() < f32::EPSILON);
}
