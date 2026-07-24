//! Weight calculation functions for NEAT-AI Discovery.
//!
//! This module contains the core weight computation logic used for synapse and
//! neuron candidate evaluation. These functions implement the "what weight should
//! this connection have?" question that is critical for prediction accuracy.
//!
//! ## Sub-modules
//!
//! - [`calculation`] — core weight calculation (least squares, identity fitting, bias search)
//! - [`normalisation`] — range-aware weight computation (sentinel filtering)
//! - [`adjustment`] — dynamic weight adjustments (delta clamping, coordinated structural)
//!
//! ## Design Notes
//!
//! - IDENTITY outgoing weights are clamped to [-0.01, 0.01] (tightened in Issue #888)
//! - Non-linear activations use relaxed ceiling [-0.05, 0.05] (Issue #905)
//! - Weight ratio validation is activation-aware: IDENTITY requires ratio >= 50,
//!   non-linear activations require ratio >= 10 (Issue #905)
//! - Bias-aware calculation recomputes weights after bias optimisation

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
pub mod adjustment;
pub mod calculation;
pub mod normalisation;

// =============================================================================
// Constants
// =============================================================================

/// Maximum allowed outgoing weight for IDENTITY add-neuron and add-synapse candidates.
///
/// Issue #888: Tightened from 0.1 to 0.01 based on the production discovery cache:
/// - Successful candidates: outgoing weights 0.001–0.005 (exponent e-3)
/// - Failed candidates: outgoing weights 0.01–0.1 (exponent e-2 to e-1)
/// - The previous ceiling of 0.1 was far too permissive; virtually all
///   successes have outgoing weights ≤ 0.005
///
/// Using 0.01 provides margin above the 0.005 success peak while filtering
/// the 0.01–0.1 range that almost always fails.
///
/// Issue #905: This value is now specific to IDENTITY candidates. Non-linear
/// activations use `MAX_OUTGOING_WEIGHT_NON_LINEAR` via the activation-aware
/// helpers.
pub const MAX_OUTGOING_WEIGHT: f32 = 0.01;

/// Maximum allowed outgoing weight for non-linear activation candidates.
///
/// Issue #905: Non-linear activations (TANH, GELU, `ReLU`, etc.) compress their
/// output range, requiring a larger outgoing weight to achieve the same
/// correction magnitude. The IDENTITY-calibrated ceiling of 0.01 systematically
/// rejects valid non-linear candidates whose optimal weight is in the 0.01–0.03
/// range.
pub const MAX_OUTGOING_WEIGHT_NON_LINEAR: f32 = 0.03;

/// Minimum incoming/outgoing weight ratio for IDENTITY predictions.
///
/// Based on successful discovery analysis:
/// - Successful discoveries have ratio 71x to 104,000x
/// - Failures often have nearly equal weights (ratio < 10x)
///
/// We require ratio >= 50 when incoming weight > 1.0.
pub(crate) const MIN_WEIGHT_RATIO: f32 = 50.0;

/// Minimum incoming/outgoing weight ratio for non-linear activation predictions.
///
/// Issue #905: Non-linear activations operate in different weight regimes than
/// IDENTITY. Hidden-source candidates with smaller incoming weights and
/// non-linear activations are valid at lower ratios.
pub(crate) const MIN_WEIGHT_RATIO_NON_LINEAR: f32 = 10.0;

// DEFAULT_SENTINEL_TOLERANCE uses SENTINEL_TOLERANCE from constants.rs (Issue #424)
pub use crate::analysis::constants::SENTINEL_TOLERANCE as DEFAULT_SENTINEL_TOLERANCE;

// =============================================================================
// Re-exports — maintain public API unchanged
// =============================================================================

pub use adjustment::{clamp_weight_update_delta, coordinated_structural_activation_delta};
pub use calculation::{
    calculate_activation_aware_outgoing_weight, calculate_optimal_bias,
    calculate_optimal_identity_outgoing_and_bias, calculate_optimal_outgoing_weight,
    max_outgoing_weight_for_activation, min_weight_ratio_for_activation,
};
pub use normalisation::{calculate_range_aware_weight, compute_range_aware_sums};

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::samples::{EPSILON, HelpfulSample};

    // -------------------------------------------------------------------------
    // calculate_optimal_outgoing_weight tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_optimal_weight_returns_none_for_insufficient_activation() {
        let result = calculate_optimal_outgoing_weight(1.0, 0.0, 1.0);
        assert!(
            result.is_none(),
            "Should return None when sum_activation_sq is zero"
        );
    }

    #[test]
    fn test_optimal_weight_returns_none_for_very_small_activation() {
        let result = calculate_optimal_outgoing_weight(1.0, EPSILON * 0.5, 1.0);
        assert!(
            result.is_none(),
            "Should return None when sum_activation_sq is below EPSILON"
        );
    }

    #[test]
    fn test_optimal_weight_returns_none_for_non_finite_results() {
        let result = calculate_optimal_outgoing_weight(f32::INFINITY, 1.0, 1.0);
        assert!(result.is_none(), "Should return None for infinite input");

        let result = calculate_optimal_outgoing_weight(f32::NAN, 1.0, 1.0);
        assert!(result.is_none(), "Should return None for NaN input");
    }

    #[test]
    fn test_optimal_weight_returns_none_for_near_zero_weights() {
        // Very small sum_error_activation relative to sum_activation_sq
        let result = calculate_optimal_outgoing_weight(EPSILON * 0.1, 100.0, 1.0);
        assert!(
            result.is_none(),
            "Should return None when computed weight is near zero"
        );
    }

    #[test]
    fn test_optimal_weight_is_clamped_to_max_outgoing_weight() {
        // Large positive weight: sum_error_activation=10, sum_activation_sq=1 => raw_weight=10
        let result = calculate_optimal_outgoing_weight(10.0, 1.0, 1.0);
        assert!(result.is_some());
        let weight = result.unwrap();
        assert!(
            (weight - MAX_OUTGOING_WEIGHT).abs() < EPSILON,
            "Weight {weight} should be clamped to MAX_OUTGOING_WEIGHT {MAX_OUTGOING_WEIGHT}"
        );

        // Large negative weight
        let result = calculate_optimal_outgoing_weight(-10.0, 1.0, 1.0);
        assert!(result.is_some());
        let weight = result.unwrap();
        assert!(
            (weight - (-MAX_OUTGOING_WEIGHT)).abs() < EPSILON,
            "Weight {} should be clamped to -MAX_OUTGOING_WEIGHT {}",
            weight,
            -MAX_OUTGOING_WEIGHT
        );
    }

    #[test]
    fn test_optimal_weight_accounts_for_incoming_weight() {
        // With incoming_weight=10, raw_weight=1.0, clamped=0.01
        // ratio = 10/0.01 = 1000 >= 50 (MIN_WEIGHT_RATIO), should pass
        let result = calculate_optimal_outgoing_weight(1.0, 1.0, 10.0);
        assert!(
            result.is_some(),
            "incoming_weight=10 should have ratio >= 50"
        );

        // Issue #888: With MAX_OUTGOING_WEIGHT=0.01, incoming_weight=2 now passes
        // the ratio check (2/0.01=200 >= 50). This is correct because incoming ~2
        // is the dominant success pattern in production discovery-cache evidence.
        let result = calculate_optimal_outgoing_weight(1.0, 1.0, 2.0);
        assert!(
            result.is_some(),
            "incoming_weight=2 with clamped weight=0.01 has ratio=200 >= MIN_WEIGHT_RATIO"
        );

        // With incoming_weight=1.0 (not > 1), ratio check is skipped
        let result = calculate_optimal_outgoing_weight(1.0, 1.0, 1.0);
        assert!(result.is_some(), "incoming_weight=1.0 skips ratio check");
    }

    #[test]
    fn test_optimal_weight_normal_calculation() {
        // Issue #888: With MAX_OUTGOING_WEIGHT=0.01, a weight that would be
        // 0.05 is now clamped to 0.01. Use smaller inputs to test unclamped path.
        // sum_error_activation=0.05, sum_activation_sq=10 => raw_weight=0.005
        let result = calculate_optimal_outgoing_weight(0.05, 10.0, 1.0);
        assert!(result.is_some());
        let weight = result.unwrap();
        assert!(
            (weight - 0.005).abs() < 0.001,
            "Expected weight ~0.005, got {weight}"
        );
    }

    #[test]
    fn test_optimal_weight_ratio_validation() {
        // With large incoming_weight (100), the outgoing weight must be small enough
        // for ratio >= MIN_WEIGHT_RATIO (50)
        // raw_weight = 10/100 = 0.1 => clamped to 0.01
        // ratio = 100/0.01 = 10000 >= 50, should pass
        let sum_error_activation = 10.0;
        let sum_activation_sq = 100.0;
        let incoming_weight = 100.0;
        let result = calculate_optimal_outgoing_weight(
            sum_error_activation,
            sum_activation_sq,
            incoming_weight,
        );

        assert!(result.is_some());
        let weight = result.unwrap();
        assert!(
            weight.abs() <= MAX_OUTGOING_WEIGHT,
            "Weight should be within MAX_OUTGOING_WEIGHT"
        );
    }

    #[test]
    fn test_optimal_weight_rejects_poor_ratio() {
        // Large raw weight that would be clamped to MAX_OUTGOING_WEIGHT
        // With incoming_weight=10, ratio = 10/0.01 = 1000 >= 50, should pass
        let result = calculate_optimal_outgoing_weight(10.0, 1.0, 10.0);
        assert!(
            (result.unwrap().abs() - MAX_OUTGOING_WEIGHT).abs() < EPSILON,
            "Large raw weight should be clamped to MAX_OUTGOING_WEIGHT"
        );
    }

    // -------------------------------------------------------------------------
    // calculate_optimal_identity_outgoing_and_bias tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_identity_returns_none_for_empty_samples() {
        let samples: Vec<HelpfulSample> = vec![];
        let result = calculate_optimal_identity_outgoing_and_bias(&samples, 1.0);
        assert!(result.is_none());
    }

    #[test]
    fn test_identity_returns_none_for_invalid_samples() {
        let samples = vec![
            HelpfulSample {
                activation: f32::NAN,
                avg_error: 0.5,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 1.0,
                avg_error: f32::NAN,
                target_value: None,
                target_activation: None,
            },
        ];
        let result = calculate_optimal_identity_outgoing_and_bias(&samples, 1.0);
        assert!(result.is_none());
    }

    #[test]
    fn test_identity_with_low_variance_returns_zero_bias() {
        // All samples have same activation (low variance)
        let samples = vec![
            HelpfulSample {
                activation: 0.5,
                avg_error: 0.1,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.5,
                avg_error: 0.2,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.5,
                avg_error: 0.15,
                target_value: None,
                target_activation: None,
            },
        ];
        let result = calculate_optimal_identity_outgoing_and_bias(&samples, 1.0);
        if let Some((_, bias)) = result {
            assert!(
                bias.abs() < 0.001,
                "Low variance source should have bias=0, got {bias}"
            );
        }
    }

    #[test]
    fn test_identity_computes_valid_weight_and_bias() {
        // Issue #888: Test data adjusted to produce a weight within the tightened
        // MAX_OUTGOING_WEIGHT (0.01). Error magnitude ~0.005 × activation so
        // that the optimal weight is ~0.005 and the bias remains small.
        let samples = vec![
            HelpfulSample {
                activation: 0.0,
                avg_error: 0.005,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.5,
                avg_error: 0.0025,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 1.0,
                avg_error: 0.0,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.5,
                avg_error: 0.0075,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -1.0,
                avg_error: 0.01,
                target_value: None,
                target_activation: None,
            },
        ];
        let result = calculate_optimal_identity_outgoing_and_bias(&samples, 1.0);
        assert!(result.is_some(), "Should compute valid weight and bias");
        let (weight, bias) = result.unwrap();
        assert!(weight.is_finite(), "Weight should be finite");
        assert!(bias.is_finite(), "Bias should be finite");
        assert!(
            weight.abs() <= MAX_OUTGOING_WEIGHT,
            "Weight should be within MAX_OUTGOING_WEIGHT"
        );
    }

    // -------------------------------------------------------------------------
    // calculate_optimal_bias tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_optimal_bias_returns_zero_for_empty_samples() {
        let samples: Vec<HelpfulSample> = vec![];
        let bias = calculate_optimal_bias(&samples, 1.0, 0.05, |x| x, "IDENTITY", None, None);
        assert!(
            bias.abs() < EPSILON,
            "Empty samples should return bias=0, got {bias}"
        );
    }

    #[test]
    fn test_optimal_bias_returns_zero_for_zero_baseline_error() {
        // All samples have zero error
        let samples = vec![
            HelpfulSample {
                activation: 0.5,
                avg_error: 0.0,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 1.0,
                avg_error: 0.0,
                target_value: None,
                target_activation: None,
            },
        ];
        let bias = calculate_optimal_bias(&samples, 1.0, 0.05, |x| x, "IDENTITY", None, None);
        assert!(
            bias.abs() < EPSILON,
            "Zero baseline error should return bias=0, got {bias}"
        );
    }

    #[test]
    fn test_optimal_bias_with_tanh() {
        // Create samples with error that could be corrected by bias shift
        let samples: Vec<HelpfulSample> = (0..20)
            .map(|i| {
                let activation = (i as f32 - 10.0) / 10.0; // -1 to 1
                HelpfulSample {
                    activation,
                    avg_error: 0.1, // Constant positive error
                    target_value: None,
                    target_activation: None,
                }
            })
            .collect();

        let tanh_fn = |x: f32| x.tanh();
        let bias = calculate_optimal_bias(&samples, 1.0, 0.05, tanh_fn, "TANH", None, None);

        // Should find some bias value (exact value depends on grid search)
        assert!(bias.is_finite(), "Bias should be finite");
    }

    #[test]
    fn test_optimal_bias_with_relu() {
        // Create samples with error pattern suited for ReLU
        let samples: Vec<HelpfulSample> = (0..20)
            .map(|i| {
                let activation = (i as f32 - 10.0) / 10.0; // -1 to 1
                HelpfulSample {
                    activation,
                    avg_error: if activation > 0.0 { 0.1 } else { -0.1 },
                    target_value: None,
                    target_activation: None,
                }
            })
            .collect();

        let relu_fn = |x: f32| x.max(0.0);
        let bias = calculate_optimal_bias(&samples, 1.0, 0.05, relu_fn, "ReLU", None, None);

        assert!(bias.is_finite(), "Bias should be finite");
    }

    // -------------------------------------------------------------------------
    // clamp_weight_update_delta tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_clamp_delta_returns_none_for_tiny_delta() {
        let result = clamp_weight_update_delta(0.005, EPSILON * 0.5);
        assert!(result.is_none(), "Tiny delta should return None");
    }

    #[test]
    fn test_clamp_delta_within_bounds() {
        let result = clamp_weight_update_delta(0.0, 0.005);
        assert!(result.is_some());
        let (new_weight, delta) = result.unwrap();
        assert!((new_weight - 0.005).abs() < EPSILON);
        assert!((delta - 0.005).abs() < EPSILON);
    }

    #[test]
    fn test_clamp_delta_exceeds_upper_bound() {
        // Start at 0.005, try to add 0.01 => clamped to MAX_OUTGOING_WEIGHT (0.01)
        let result = clamp_weight_update_delta(0.005, 0.01);
        assert!(result.is_some());
        let (new_weight, delta) = result.unwrap();
        assert!((new_weight - MAX_OUTGOING_WEIGHT).abs() < EPSILON);
        assert!((delta - 0.005).abs() < EPSILON); // Effective delta is 0.005
    }

    #[test]
    fn test_clamp_delta_exceeds_lower_bound() {
        // Start at -0.005, try to subtract 0.01 => clamped to -MAX_OUTGOING_WEIGHT (-0.01)
        let result = clamp_weight_update_delta(-0.005, -0.01);
        assert!(result.is_some());
        let (new_weight, delta) = result.unwrap();
        assert!((new_weight - (-MAX_OUTGOING_WEIGHT)).abs() < EPSILON);
        assert!((delta - (-0.005)).abs() < EPSILON); // Effective delta is -0.005
    }

    #[test]
    fn test_clamp_delta_already_at_max() {
        // Already at MAX_OUTGOING_WEIGHT, positive delta should be ineffective
        let result = clamp_weight_update_delta(MAX_OUTGOING_WEIGHT, 0.05);
        assert!(result.is_none(), "Delta at max should have no effect");
    }

    // -------------------------------------------------------------------------
    // coordinated_structural_activation_delta tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_coordinated_delta_returns_none_for_zero_noisy_weight() {
        let result = coordinated_structural_activation_delta(1.0, 0.5, 0.0, 0.5);
        assert!(result.is_none(), "Zero noisy weight should return None");

        let result = coordinated_structural_activation_delta(1.0, 0.5, EPSILON * 0.5, 0.5);
        assert!(
            result.is_none(),
            "Near-zero noisy weight should return None"
        );
    }

    #[test]
    fn test_coordinated_delta_computes_correctly() {
        // trusted_activation=1.0, noisy_activation=0.5, noisy_weight=0.2, trusted_weight=0.3
        // new_trusted_weight = 0.3 + 0.2 = 0.5
        // delta_trusted_weight = 0.5 - 0.3 = 0.2
        // scale = 0.2 / 0.2 = 1.0
        // result = 1.0 * 1.0 - 0.5 = 0.5
        let result = coordinated_structural_activation_delta(1.0, 0.5, 0.2, 0.3);
        assert!(result.is_some());
        let delta = result.unwrap();
        assert!(
            (delta - 0.5).abs() < 0.001,
            "Expected delta ~0.5, got {delta}"
        );
    }

    #[test]
    fn test_coordinated_delta_with_negative_weights() {
        // Test with negative weights
        let result = coordinated_structural_activation_delta(1.0, 0.5, -0.2, -0.3);
        assert!(result.is_some());
        // new_trusted_weight = -0.3 + (-0.2) = -0.5
        // delta_trusted_weight = -0.5 - (-0.3) = -0.2
        // scale = -0.2 / -0.2 = 1.0
        // result = 1.0 * 1.0 - 0.5 = 0.5
        let delta = result.unwrap();
        assert!(
            (delta - 0.5).abs() < 0.001,
            "Expected delta ~0.5, got {delta}"
        );
    }

    // -------------------------------------------------------------------------
    // hard_tanh tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_hard_tanh_clamps_correctly() {
        use calculation::hard_tanh;
        assert!((hard_tanh(0.5) - 0.5).abs() < EPSILON);
        assert!((hard_tanh(-0.5) - (-0.5)).abs() < EPSILON);
        assert!((hard_tanh(2.0) - 1.0).abs() < EPSILON);
        assert!((hard_tanh(-2.0) - (-1.0)).abs() < EPSILON);
        assert!((hard_tanh(1.0) - 1.0).abs() < EPSILON);
        assert!((hard_tanh(-1.0) - (-1.0)).abs() < EPSILON);
    }
}
