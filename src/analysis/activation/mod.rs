//! Activation function related code for NEAT-AI Discovery analysis.
//!
//! This module contains:
//! - CPU activation function implementations for candidate evaluation
//! - Activation candidate specifications (`ACTIVATION_SPECS`)
//! - GPU ID mapping for activation functions
//! - Bias range helpers for different activation types
//! - Predicates for activation function classification
//! - Target simulation functions for saturation-aware scoring
//!
//! Note: This module is separate from `src/activations.rs` which handles
//! TypeScript-Rust interop for squash names and provides `apply_scalar_squash`.
//! This module focuses on candidate discovery and evaluation.
//!
//! ## Sub-modules (Issue #607)
//! - `functions` — CPU activation function implementations
//! - `specs` — Activation candidate specifications, GPU ID mapping, bias helpers
//! - `simulation` — Target simulation, predicates, variance checking

pub mod compatibility;
pub mod functions;
pub mod simulation;
pub mod specs;

// Re-export all public items to maintain backward compatibility
pub use compatibility::*;
pub use functions::*;
pub use simulation::*;
pub use specs::*;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::samples::HelpfulSample;

    #[test]
    fn test_gelu_activation() {
        // GELU(0) ≈ 0
        assert!((gelu_activation(0.0) - 0.0).abs() < 1e-6);
        // GELU(x) > 0 for x > 0
        assert!(gelu_activation(1.0) > 0.0);
        // GELU(-1) is small negative
        assert!(gelu_activation(-1.0) < 0.0);
    }

    #[test]
    fn test_elu_activation() {
        assert_eq!(elu_activation(0.0), 0.0);
        assert_eq!(elu_activation(1.0), 1.0);
        assert!(elu_activation(-1.0) < 0.0);
        assert!(elu_activation(-1.0) > -1.0); // ELU asymptotes to -1
    }

    #[test]
    fn test_softplus_activation() {
        assert!(softplus_activation(0.0) > 0.0);
        // Softplus(x) ≈ x for large x
        assert!((softplus_activation(30.0) - 30.0).abs() < 0.01);
    }

    #[test]
    fn test_logistic_activation() {
        assert!((logistic_activation(0.0) - 0.5).abs() < 1e-6);
        assert!(logistic_activation(10.0) > 0.99);
        assert!(logistic_activation(-10.0) < 0.01);
    }

    #[test]
    fn test_tanh_activation() {
        assert_eq!(tanh_activation(0.0), 0.0);
        assert!(tanh_activation(3.0) > 0.99);
        assert!(tanh_activation(-3.0) < -0.99);
    }

    #[test]
    fn test_identity_activation() {
        assert_eq!(identity_activation(5.0), 5.0);
        assert_eq!(identity_activation(-3.0), -3.0);
    }

    #[test]
    fn test_bipolar_activation() {
        assert_eq!(bipolar_activation(0.001), 1.0);
        assert_eq!(bipolar_activation(0.0), -1.0);
        assert_eq!(bipolar_activation(-0.001), -1.0);
    }

    #[test]
    fn test_clipped_activation() {
        assert_eq!(clipped_activation(0.5), 0.5);
        assert_eq!(clipped_activation(2.0), 1.0);
        assert_eq!(clipped_activation(-2.0), -1.0);
    }

    #[test]
    fn test_absolute_activation() {
        assert_eq!(absolute_activation(5.0), 5.0);
        assert_eq!(absolute_activation(-5.0), 5.0);
    }

    #[test]
    fn test_mish_activation() {
        assert!((mish_activation(0.0) - 0.0).abs() < 1e-6);
        assert!(mish_activation(1.0) > 0.0);
    }

    #[test]
    fn test_hard_tanh_activation() {
        assert_eq!(hard_tanh_activation(0.5), 0.5);
        assert_eq!(hard_tanh_activation(2.0), 1.0);
        assert_eq!(hard_tanh_activation(-2.0), -1.0);
    }

    #[test]
    fn test_softsign_activation() {
        assert_eq!(softsign_activation(0.0), 0.0);
        assert!(softsign_activation(1.0) > 0.4);
        assert!(softsign_activation(1.0) < 0.6);
    }

    #[test]
    fn test_bent_identity_activation() {
        assert!((bent_identity_activation(0.0) - 0.0).abs() < 1e-6);
        // Bent identity is nearly linear for small x
        assert!((bent_identity_activation(0.1) - 0.1).abs() < 0.1);
    }

    #[test]
    fn test_arctan_activation() {
        assert_eq!(arctan_activation(0.0), 0.0);
        assert!(arctan_activation(1.0) > 0.7);
        assert!(arctan_activation(1.0) < 0.8);
    }

    #[test]
    fn test_relu6_activation() {
        assert_eq!(relu6_activation(-1.0), 0.0);
        assert_eq!(relu6_activation(3.0), 3.0);
        assert_eq!(relu6_activation(10.0), 6.0);
    }

    #[test]
    fn test_activation_name_to_gpu_id() {
        assert_eq!(activation_name_to_gpu_id("GELU"), 0);
        assert_eq!(activation_name_to_gpu_id("ELU"), 1);
        assert_eq!(activation_name_to_gpu_id("IDENTITY"), 6);
        assert_eq!(activation_name_to_gpu_id("LeakyReLU"), 11);
        assert_eq!(activation_name_to_gpu_id("Mish"), 12);
        assert_eq!(activation_name_to_gpu_id("ReLU6"), 18);
        // Unknown defaults to IDENTITY
        assert_eq!(activation_name_to_gpu_id("UNKNOWN"), 6);
    }

    #[test]
    fn test_get_bias_range() {
        let (min, max, step) = get_bias_range("BIPOLAR");
        assert_eq!(min, -10.0);
        assert_eq!(max, 10.0);
        assert_eq!(step, 1.0);

        let (min, max, step) = get_bias_range("TANH");
        assert_eq!(min, -10.0);
        assert_eq!(max, 10.0);
        assert_eq!(step, 0.5);
    }

    #[test]
    fn test_get_bias_values() {
        let values = get_bias_values("TANH");
        assert!(!values.is_empty());
        assert!(values.contains(&0.0));
        // Check values are sorted
        for i in 1..values.len() {
            assert!(values[i] >= values[i - 1]);
        }
    }

    #[test]
    fn test_is_threshold_activation() {
        assert!(is_threshold_activation("STEP"));
        assert!(is_threshold_activation("step"));
        assert!(is_threshold_activation("BIPOLAR"));
        assert!(is_threshold_activation("bipolar"));
        assert!(!is_threshold_activation("TANH"));
        assert!(!is_threshold_activation("RELU"));
        assert!(!is_threshold_activation("IDENTITY"));
    }

    /// Test that `is_threshold_activation` uses zero-allocation case-insensitive comparison.
    /// Issue #211: Uses `eq_ignore_ascii_case` instead of `.to_uppercase()` to avoid
    /// string allocations in hot loops.
    #[test]
    fn test_is_threshold_activation_case_variations() {
        // All case variations should work without allocation
        assert!(is_threshold_activation("STEP"));
        assert!(is_threshold_activation("Step"));
        assert!(is_threshold_activation("step"));
        assert!(is_threshold_activation("sTeP"));
        assert!(is_threshold_activation("BIPOLAR"));
        assert!(is_threshold_activation("Bipolar"));
        assert!(is_threshold_activation("bipolar"));
        assert!(is_threshold_activation("BiPoLaR"));

        // Non-threshold activations with various cases
        assert!(!is_threshold_activation("TANH"));
        assert!(!is_threshold_activation("tanh"));
        assert!(!is_threshold_activation("Tanh"));
        assert!(!is_threshold_activation("RELU"));
        assert!(!is_threshold_activation("relu"));
        assert!(!is_threshold_activation("ReLU"));
        assert!(!is_threshold_activation("IDENTITY"));
        assert!(!is_threshold_activation("identity"));

        // Edge cases
        assert!(!is_threshold_activation(""));
        assert!(!is_threshold_activation("STEP2")); // Not exact match
        assert!(!is_threshold_activation("STEPBIPOLAR")); // Not exact match
    }

    #[test]
    fn test_activation_specs_count() {
        // Verify we have exactly 15 activation specs as documented
        assert_eq!(ACTIVATION_SPECS.len(), 15);
    }

    #[test]
    fn test_activation_specs_have_valid_functions() {
        // Each spec should have a working activation function
        for spec in &ACTIVATION_SPECS {
            // Test that the function works for a typical input
            let result = (spec.activation)(0.5);
            assert!(
                result.is_finite(),
                "{} returned non-finite for 0.5",
                spec.name
            );
        }
    }

    // ========================================================================
    // Target Simulation Function Tests
    // ========================================================================

    #[test]
    fn test_get_target_activation_fn() {
        // Known activations should return Some
        assert!(get_target_activation_fn("TANH").is_some());
        assert!(get_target_activation_fn("HARD_TANH").is_some());
        assert!(get_target_activation_fn("CLIPPED").is_some());
        assert!(get_target_activation_fn("LOGISTIC").is_some());
        assert!(get_target_activation_fn("IDENTITY").is_some());

        // Case insensitive
        assert!(get_target_activation_fn("tanh").is_some());
        assert!(get_target_activation_fn("Tanh").is_some());

        // Aggregate squashes should return None
        assert!(get_target_activation_fn("MINIMUM").is_none());
        assert!(get_target_activation_fn("MAXIMUM").is_none());
        assert!(get_target_activation_fn("IF").is_none());
    }

    #[test]
    fn test_get_target_simulation_fn_with_complete_samples() {
        let samples = vec![
            HelpfulSample {
                activation: 0.5,
                avg_error: 0.1,
                target_value: Some(0.3),
                target_activation: Some(0.29),
            },
            HelpfulSample {
                activation: -0.2,
                avg_error: -0.05,
                target_value: Some(-0.1),
                target_activation: Some(-0.099),
            },
        ];

        // Should return activation function when all samples have target data
        let result = get_target_simulation_fn(&samples, Some("TANH"));
        assert!(result.is_some());
    }

    #[test]
    fn test_get_target_simulation_fn_with_incomplete_samples() {
        let samples = vec![
            HelpfulSample {
                activation: 0.5,
                avg_error: 0.1,
                target_value: Some(0.3),
                target_activation: Some(0.29),
            },
            HelpfulSample {
                activation: -0.2,
                avg_error: -0.05,
                target_value: None, // Missing target_value
                target_activation: Some(-0.099),
            },
        ];

        // Should return None when some samples are missing target data
        let result = get_target_simulation_fn(&samples, Some("TANH"));
        assert!(result.is_none());
    }

    #[test]
    fn test_get_target_simulation_mode_none_squash() {
        let samples = vec![HelpfulSample {
            activation: 0.5,
            avg_error: 0.1,
            target_value: Some(0.3),
            target_activation: Some(0.29),
        }];

        let mode = get_target_simulation_mode(&samples, None);
        assert!(matches!(mode, TargetSimulationMode::None));
    }

    #[test]
    fn test_get_target_simulation_mode_full() {
        let samples = vec![
            HelpfulSample {
                activation: 0.5,
                avg_error: 0.1,
                target_value: Some(0.3),
                target_activation: Some(0.29),
            },
            HelpfulSample {
                activation: -0.2,
                avg_error: -0.05,
                target_value: Some(-0.1),
                target_activation: Some(-0.099),
            },
        ];

        let mode = get_target_simulation_mode(&samples, Some("HARD_TANH"));
        assert!(matches!(mode, TargetSimulationMode::Full(_)));
    }

    #[test]
    fn test_get_target_simulation_mode_approximate() {
        let samples = vec![
            HelpfulSample {
                activation: 0.5,
                avg_error: 0.1,
                target_value: None, // No target_value
                target_activation: Some(0.29),
            },
            HelpfulSample {
                activation: -0.2,
                avg_error: -0.05,
                target_value: None, // No target_value
                target_activation: Some(-0.099),
            },
        ];

        // HARD_TANH should use approximation mode when target_value is missing
        let mode = get_target_simulation_mode(&samples, Some("HARD_TANH"));
        assert!(matches!(
            mode,
            TargetSimulationMode::ApproximateValueFromActivation { .. }
        ));

        // CLIPPED (alias for HARD_TANH) should also work
        let mode = get_target_simulation_mode(&samples, Some("CLIPPED"));
        assert!(matches!(
            mode,
            TargetSimulationMode::ApproximateValueFromActivation { .. }
        ));

        // Issue #906: TANH should now also use approximation mode (was None before)
        let mode = get_target_simulation_mode(&samples, Some("TANH"));
        assert!(matches!(
            mode,
            TargetSimulationMode::ApproximateValueFromActivation { .. }
        ));
    }

    #[test]
    fn test_can_use_hard_tanh() {
        let complete_samples = vec![
            HelpfulSample {
                activation: 0.5,
                avg_error: 0.1,
                target_value: Some(0.3),
                target_activation: Some(0.29),
            },
            HelpfulSample {
                activation: -0.2,
                avg_error: -0.05,
                target_value: Some(-0.1),
                target_activation: Some(-0.099),
            },
        ];

        let incomplete_samples = vec![HelpfulSample {
            activation: 0.5,
            avg_error: 0.1,
            target_value: None,
            target_activation: Some(0.29),
        }];

        // Should return true for HARD_TANH with complete samples
        assert!(can_use_hard_tanh(&complete_samples, Some("HARD_TANH")));
        assert!(can_use_hard_tanh(&complete_samples, Some("hard_tanh")));
        assert!(can_use_hard_tanh(&complete_samples, Some("CLIPPED")));

        // Should return false for other activations
        assert!(!can_use_hard_tanh(&complete_samples, Some("TANH")));
        assert!(!can_use_hard_tanh(&complete_samples, None));

        // Should return false for incomplete samples
        assert!(!can_use_hard_tanh(&incomplete_samples, Some("HARD_TANH")));
    }

    #[test]
    fn test_has_sufficient_output_variance_with_variance() {
        // Samples with varying input - output should also vary
        let samples = vec![
            HelpfulSample {
                activation: -1.0,
                avg_error: 0.1,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.0,
                avg_error: 0.1,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 1.0,
                avg_error: 0.1,
                target_value: None,
                target_activation: None,
            },
        ];

        // Identity activation should show variance
        assert!(has_sufficient_output_variance(
            &samples,
            1.0,
            0.0,
            identity_activation
        ));

        // TANH with reasonable weight should show variance
        assert!(has_sufficient_output_variance(
            &samples,
            1.0,
            0.0,
            tanh_activation
        ));
    }

    #[test]
    fn test_has_sufficient_output_variance_saturated() {
        // Samples with varying input
        let samples = vec![
            HelpfulSample {
                activation: -1.0,
                avg_error: 0.1,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.0,
                avg_error: 0.1,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 1.0,
                avg_error: 0.1,
                target_value: None,
                target_activation: None,
            },
        ];

        // TANH with large bias should be saturated (output ≈ 1.0 for all inputs)
        assert!(!has_sufficient_output_variance(
            &samples,
            1.0,
            10.0,
            tanh_activation
        ));
    }

    #[test]
    fn test_has_sufficient_output_variance_constant_input() {
        // Samples with constant input
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
                avg_error: 0.3,
                target_value: None,
                target_activation: None,
            },
        ];

        // Constant input should be allowed (output variance check is skipped)
        assert!(has_sufficient_output_variance(
            &samples,
            1.0,
            0.0,
            identity_activation
        ));
    }

    #[test]
    fn test_has_sufficient_output_variance_insufficient_samples() {
        let single_sample = vec![HelpfulSample {
            activation: 0.5,
            avg_error: 0.1,
            target_value: None,
            target_activation: None,
        }];

        let empty_samples: Vec<HelpfulSample> = vec![];

        // Should return false for insufficient samples
        assert!(!has_sufficient_output_variance(
            &single_sample,
            1.0,
            0.0,
            identity_activation
        ));
        assert!(!has_sufficient_output_variance(
            &empty_samples,
            1.0,
            0.0,
            identity_activation
        ));
    }
}
