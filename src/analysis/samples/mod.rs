//! Sample data structures for NEAT-AI Discovery GPU pipeline.
//!
//! This module contains the core sample types and GPU-compatible data formats used
//! throughout the analysis pipeline. These structures are fundamental to the GPU
//! computation workflow:
//!
//! 1. `HelpfulSample` - Core evaluation unit passed between CPU and GPU
//! 2. GPU structs (`*Contribution`, `*Uniforms`) - Tightly coupled with shader code
//! 3. Statistics types - Aggregate GPU computation results

mod gpu_types;
mod statistics;
mod thresholds;

// Re-export all public items for backward compatibility
pub use gpu_types::{
    ActivationOutput, ActivationUniforms, BiasResult, BiasUniforms, GpuHelpfulSample,
    HarmfulContribution, HarmfulUniforms, HelpfulContribution, HelpfulUniforms, ReductionUniforms,
    ReluContribution, ReluUniforms,
};
pub use statistics::{HarmfulStats, HelpfulStats, NeuronStats, ReluOrientation, ReluStats};
pub use thresholds::{
    compute_dynamic_constant_source_threshold, compute_source_std_dev,
    compute_source_variance_discount, constant_source_effect_threshold_from_env,
    get_constant_source_threshold,
};

/// Small epsilon value to prevent division by zero.
pub const EPSILON: f32 = 1e-8;

/// Default threshold for treating an add-synapse candidate as an effective bias change.
///
/// Issue #178 (7-Jan-2026): When the source activation range is ~0, adding a synapse
/// only contributes a near-constant offset to the target. This is better represented as
/// a `setBias` coordinated-structural operation than paying complexity cost for a new edge.
///
/// The heuristic is based on the *range* of the contribution:
///   effect_range approx |weight| * (max_activation - min_activation)
///
/// If `effect_range <= threshold`, we fold the synapse into `setBias`.
pub const DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD: f32 = 1e-7;

// =============================================================================
// Core Sample Types
// =============================================================================

/// Core evaluation sample unit.
///
/// Represents a single sample for evaluating potential synapse/neuron candidates.
/// For accurate HARD_TANH modelling, we need the target's pre-activation value
/// to properly simulate clamping behaviour. When `target_value` is `Some`, we can
/// compute the actual effect of adding a contribution rather than using the linear
/// approximation.
#[derive(Debug, Clone, Copy, Default)]
pub struct HelpfulSample {
    /// Source neuron's activation (what we're considering adding a connection FROM)
    pub activation: f32,
    /// Target neuron's average error (expected - actual output)
    pub avg_error: f32,
    /// Target neuron's pre-activation value (input sum before squash function).
    /// Used for accurate HARD_TANH/clamping calculations. None for GPU-matched samples.
    pub target_value: Option<f32>,
    /// Target neuron's post-activation output (after squash function).
    /// Note: avg_error is in VALUE domain, so expected = squash(target_value + avg_error)
    pub target_activation: Option<f32>,
}

// =============================================================================
// Unit Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_helpful_sample_default() {
        let sample = HelpfulSample::default();
        assert_eq!(sample.activation, 0.0);
        assert_eq!(sample.avg_error, 0.0);
        assert!(sample.target_value.is_none());
        assert!(sample.target_activation.is_none());
    }

    #[test]
    fn test_gpu_helpful_sample_from_helpful_sample() {
        let sample = HelpfulSample {
            activation: 1.5,
            avg_error: -0.3,
            target_value: Some(2.0),
            target_activation: Some(0.9),
        };
        let gpu_sample: GpuHelpfulSample = sample.into();
        assert_eq!(gpu_sample.activation, 1.5);
        assert_eq!(gpu_sample.avg_error, -0.3);
    }

    #[test]
    fn test_bias_result_zeroed() {
        let result = BiasResult::zeroed();
        assert_eq!(result.bias_value, 0.0);
        assert_eq!(result.error_reduction, 0.0);
        assert_eq!(result.valid_sample_count, 0);
    }

    #[test]
    fn test_relu_stats_new() {
        let stats = ReluStats::new(ReluOrientation::Positive);
        assert!(stats.samples.is_empty());
        assert_eq!(stats.activation_sq_sum, 0.0);
        assert_eq!(stats.error_activation_sum, 0.0);
    }

    #[test]
    fn test_harmful_stats_default() {
        let stats = HarmfulStats::default();
        assert_eq!(stats.harmful_count, 0);
        assert_eq!(stats.helpful_count, 0);
        assert_eq!(stats.harmful_error_sum, 0.0);
    }

    #[test]
    fn test_helpful_stats_default() {
        let stats = HelpfulStats::default();
        assert_eq!(stats.positive_count, 0);
        assert_eq!(stats.negative_count, 0);
        assert_eq!(stats.error_sq_sum, 0.0);
    }

    #[test]
    fn test_compute_source_variance_discount_empty() {
        let discount = compute_source_variance_discount(&[]);
        assert_eq!(discount, 0.0);
    }

    #[test]
    fn test_compute_source_variance_discount_single() {
        let samples = vec![HelpfulSample {
            activation: 1.0,
            avg_error: 0.0,
            target_value: None,
            target_activation: None,
        }];
        let discount = compute_source_variance_discount(&samples);
        assert_eq!(discount, 0.0);
    }

    #[test]
    fn test_compute_source_variance_discount_constant() {
        // All samples have the same activation - should be fully discounted
        let samples = vec![
            HelpfulSample {
                activation: 1.0,
                avg_error: 0.1,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 1.0,
                avg_error: -0.1,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 1.0,
                avg_error: 0.2,
                target_value: None,
                target_activation: None,
            },
        ];
        let discount = compute_source_variance_discount(&samples);
        assert!(
            discount < 0.01,
            "Constant source should be heavily discounted"
        );
    }

    #[test]
    fn test_compute_source_variance_discount_high_variance() {
        // Samples with high variance should get full credit
        let samples = vec![
            HelpfulSample {
                activation: -1.0,
                avg_error: 0.1,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.0,
                avg_error: -0.1,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 1.0,
                avg_error: 0.2,
                target_value: None,
                target_activation: None,
            },
        ];
        let discount = compute_source_variance_discount(&samples);
        assert!(
            discount > 0.9,
            "High variance source should get full credit"
        );
    }

    #[test]
    fn test_neuron_stats_from_samples_empty() {
        let result = NeuronStats::from_samples(&[]);
        assert!(result.is_none());
    }

    #[test]
    fn test_neuron_stats_from_samples_valid() {
        let samples = vec![
            HelpfulSample {
                activation: 1.0,
                avg_error: 0.1,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 2.0,
                avg_error: -0.1,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 3.0,
                avg_error: 0.2,
                target_value: None,
                target_activation: None,
            },
        ];
        let stats = NeuronStats::from_samples(&samples).expect("Should compute stats");
        assert!((stats.mean_activation - 2.0).abs() < 0.001);
        assert!(stats.activation_variance > 0.0);
        assert_eq!(stats.activation_min, 1.0);
        assert_eq!(stats.activation_max, 3.0);
    }

    #[test]
    fn test_neuron_stats_to_json() {
        let stats = NeuronStats {
            mean_error: 0.1,
            error_variance: 0.01,
            mean_activation: 0.5,
            activation_variance: 0.25,
            error_spike_count: 2,
            activation_spike_count: 1,
            activation_min: -1.0,
            activation_max: 1.0,
        };
        let json = stats.to_json();
        assert_eq!(json.mean_error, 0.1);
        assert_eq!(json.error_variance, 0.01);
        assert_eq!(json.mean_activation, 0.5);
        assert_eq!(json.activation_variance, 0.25);
        assert_eq!(json.error_spike_count, 2);
        assert_eq!(json.activation_spike_count, 1);
        assert_eq!(json.activation_min, -1.0);
        assert_eq!(json.activation_max, 1.0);
    }

    #[test]
    fn test_gpu_structs_are_pod() {
        // These tests verify that GPU structs can be used with bytemuck
        // by checking they implement Pod and Zeroable
        let _ = bytemuck::bytes_of(&GpuHelpfulSample {
            activation: 0.0,
            avg_error: 0.0,
        });
        let _ = bytemuck::bytes_of(&HelpfulContribution {
            positive_flag: 0,
            negative_flag: 0,
            positive_improvement: 0.0,
            negative_improvement: 0.0,
            positive_activation: 0.0,
            negative_activation: 0.0,
            error_squared: 0.0,
            activation_squared: 0.0,
            error_activation: 0.0,
            pad0: 0.0,
            pad1: 0.0,
            pad2: 0.0,
        });
        let _ = bytemuck::bytes_of(&HelpfulUniforms {
            length: 0,
            pad0: 0,
            epsilon: 0.0,
            pad1: 0.0,
        });
        let _ = bytemuck::bytes_of(&HarmfulContribution {
            harmful_flag: 0,
            helpful_flag: 0,
            error_magnitude: 0.0,
            pad0: 0.0,
        });
        let _ = bytemuck::bytes_of(&HarmfulUniforms {
            length: 0,
            pad0: 0,
            epsilon: 0.0,
            weight: 0.0,
        });
        let _ = bytemuck::bytes_of(&ReluContribution {
            positive_activation_sq: 0.0,
            positive_error_activation: 0.0,
            positive_count: 0,
            negative_activation_sq: 0.0,
            negative_error_activation: 0.0,
            negative_count: 0,
            error_sq: 0.0,
            pad0: 0.0,
            pad1: 0,
            pad2: 0,
        });
        let _ = bytemuck::bytes_of(&ReluUniforms {
            length: 0,
            threshold: 0.0,
            epsilon: 0.0,
            pad0: 0.0,
        });
        let _ = bytemuck::bytes_of(&BiasResult::zeroed());
        let _ = bytemuck::bytes_of(&BiasUniforms {
            sample_count: 0,
            bias_count: 0,
            incoming_weight: 0.0,
            outgoing_weight: 0.0,
            activation_type: 0,
            epsilon: 0.0,
            min_sample_count: 0,
            pad0: 0,
        });
        let _ = bytemuck::bytes_of(&ActivationOutput {
            output: 0.0,
            output_sq: 0.0,
            error_output: 0.0,
            valid: 0,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        });
        let _ = bytemuck::bytes_of(&ActivationUniforms {
            sample_count: 0,
            orientation: 0.0,
            scale: 0.0,
            activation_type: 0,
            epsilon: 0.0,
            pad0: 0.0,
            pad1: 0.0,
        });
    }

    // =============================================================================
    // Issue #199: Dynamic Constant Source Threshold Tests
    // =============================================================================

    #[test]
    fn test_compute_dynamic_constant_source_threshold_low_variance() {
        // With low variance (< 0.05), threshold should stay at default
        let threshold = compute_dynamic_constant_source_threshold(0.01);
        assert_eq!(
            threshold, DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD,
            "Low variance should use default threshold"
        );

        let threshold = compute_dynamic_constant_source_threshold(0.04);
        assert_eq!(
            threshold, DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD,
            "Variance just below reference should use default threshold"
        );
    }

    #[test]
    fn test_compute_dynamic_constant_source_threshold_at_reference() {
        // At exactly the reference value (0.05), threshold should be default
        let threshold = compute_dynamic_constant_source_threshold(0.05);
        assert_eq!(
            threshold, DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD,
            "At reference variance, threshold should be default"
        );
    }

    #[test]
    fn test_compute_dynamic_constant_source_threshold_high_variance() {
        // With high variance, threshold should scale up
        // Formula: threshold = 1e-7 × max(1.0, avg_std_dev / 0.05)

        // std_dev = 0.10, scaling = 0.10 / 0.05 = 2.0
        let threshold = compute_dynamic_constant_source_threshold(0.10);
        let expected = DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD * 2.0;
        assert!(
            (threshold - expected).abs() < 1e-14,
            "Threshold should scale by 2x for std_dev=0.10, got {threshold}, expected {expected}"
        );

        // std_dev = 0.50, scaling = 0.50 / 0.05 = 10.0
        let threshold = compute_dynamic_constant_source_threshold(0.50);
        let expected = DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD * 10.0;
        assert!(
            (threshold - expected).abs() < 1e-13,
            "Threshold should scale by 10x for std_dev=0.50, got {threshold}, expected {expected}"
        );

        // std_dev = 1.0, scaling = 1.0 / 0.05 = 20.0
        let threshold = compute_dynamic_constant_source_threshold(1.0);
        let expected = DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD * 20.0;
        assert!(
            (threshold - expected).abs() < 1e-12,
            "Threshold should scale by 20x for std_dev=1.0, got {threshold}, expected {expected}"
        );
    }

    #[test]
    fn test_compute_dynamic_constant_source_threshold_edge_cases() {
        // Zero variance should return default
        let threshold = compute_dynamic_constant_source_threshold(0.0);
        assert_eq!(threshold, DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD);

        // Negative variance (invalid) should return default
        let threshold = compute_dynamic_constant_source_threshold(-0.1);
        assert_eq!(threshold, DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD);

        // NaN should return default
        let threshold = compute_dynamic_constant_source_threshold(f32::NAN);
        assert_eq!(threshold, DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD);

        // Infinity should return default
        let threshold = compute_dynamic_constant_source_threshold(f32::INFINITY);
        assert_eq!(threshold, DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD);
    }

    #[test]
    fn test_compute_source_std_dev_from_records() {
        use crate::types::DiscoverRecord;

        // Empty records
        let std_dev = compute_source_std_dev(&[]);
        assert_eq!(std_dev, 0.0);

        // Single record
        let records = vec![DiscoverRecord::new(
            0,
            "test".to_string(),
            Some(1.0),
            1.0,
            Vec::new(),
        )];
        let std_dev = compute_source_std_dev(&records);
        assert_eq!(std_dev, 0.0, "Single record should have zero std dev");

        // Constant records (all same activation)
        let records: Vec<DiscoverRecord> = (0..10)
            .map(|i| DiscoverRecord::new(i, "test".to_string(), Some(0.5), 0.5, Vec::new()))
            .collect();
        let std_dev = compute_source_std_dev(&records);
        assert!(
            std_dev < 0.001,
            "Constant records should have near-zero std dev"
        );

        // Variable records (alternating 0 and 1)
        let records: Vec<DiscoverRecord> = (0..10)
            .map(|i| {
                let activation = if i % 2 == 0 { 0.0 } else { 1.0 };
                DiscoverRecord::new(
                    i,
                    "test".to_string(),
                    Some(activation),
                    activation,
                    Vec::new(),
                )
            })
            .collect();
        let std_dev = compute_source_std_dev(&records);
        // std dev of [0, 1, 0, 1, ...] = 0.5
        assert!(
            (std_dev - 0.5).abs() < 0.01,
            "Variable records should have std dev ~0.5, got {std_dev}"
        );
    }

    #[test]
    #[serial_test::serial]
    fn test_get_constant_source_threshold_no_env_var() {
        // Remove env var to test dynamic threshold
        // SAFETY: Serialised via #[serial] — no concurrent env access.
        unsafe {
            std::env::remove_var("NEAT_AI_DISCOVERY_CONSTANT_SOURCE_EFFECT_THRESHOLD");
        }

        // With None, should return default
        let threshold = get_constant_source_threshold(None);
        assert_eq!(threshold, Some(DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD));

        // With low variance, should return default
        let threshold = get_constant_source_threshold(Some(0.01));
        assert_eq!(threshold, Some(DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD));

        // With high variance, should return scaled threshold
        let threshold = get_constant_source_threshold(Some(0.10));
        let expected = DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD * 2.0;
        assert!(
            (threshold.unwrap() - expected).abs() < 1e-14,
            "High variance should scale threshold"
        );
    }
}
