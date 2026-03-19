//! Property-based tests for `HelpfulStats`, `HarmfulStats`, and `NeuronStats` (Issue #680).
//!
//! Uses `proptest` to verify mathematical invariants of the statistics types
//! used in synapse and neuron evaluation, including merge additivity,
//! ratio bounds, and minimum-sample guards.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use neat_ai_discovery::analysis::samples::{
    HarmfulStats, HelpfulSample, HelpfulStats, NeuronStats,
};
use proptest::prelude::*;

/// Strategy for generating moderate finite f32 values.
fn moderate_f32() -> impl Strategy<Value = f32> {
    (-1e4f32..1e4f32).prop_filter("must be finite", |v| v.is_finite())
}

/// Strategy for generating `HelpfulSample` with finite values.
fn helpful_sample_strategy() -> impl Strategy<Value = HelpfulSample> {
    (moderate_f32(), moderate_f32()).prop_map(|(a, e)| HelpfulSample {
        activation: a,
        avg_error: e,
        target_value: None,
        target_activation: None,
    })
}

// =============================================================================
// 1. HelpfulStats Properties
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// total_count() must always equal positive_count + negative_count.
    #[test]
    fn helpful_total_count_equals_sum(
        positive in 0u32..1000,
        negative in 0u32..1000,
    ) {
        let stats = HelpfulStats {
            positive_count: positive,
            negative_count: negative,
            ..HelpfulStats::default()
        };
        prop_assert_eq!(
            stats.total_count(),
            positive + negative,
            "total_count must equal positive + negative"
        );
    }

    /// improvement_ratio() must always be in [0.0, 1.0].
    #[test]
    fn helpful_ratio_bounded(
        positive in 0u32..1000,
        negative in 0u32..1000,
    ) {
        let stats = HelpfulStats {
            positive_count: positive,
            negative_count: negative,
            ..HelpfulStats::default()
        };
        let ratio = stats.improvement_ratio();
        prop_assert!(
            (0.0..=1.0).contains(&ratio),
            "improvement_ratio {ratio} out of [0, 1]"
        );
    }

    /// improvement_ratio() must return 0.5 when total_count == 0.
    #[test]
    fn helpful_empty_ratio_neutral(_dummy in 0u8..1) {
        let stats = HelpfulStats::default();
        prop_assert!(
            (stats.improvement_ratio() - 0.5).abs() < 1e-10,
            "Empty stats should return 0.5 ratio"
        );
    }

    /// is_strongly_beneficial and is_strongly_harmful must be mutually exclusive.
    #[test]
    fn helpful_beneficial_harmful_exclusive(
        positive in 0u32..500,
        negative in 0u32..500,
    ) {
        let stats = HelpfulStats {
            positive_count: positive,
            negative_count: negative,
            ..HelpfulStats::default()
        };
        prop_assert!(
            !(stats.is_strongly_beneficial() && stats.is_strongly_harmful()),
            "Cannot be both beneficial and harmful"
        );
    }

    /// Both is_strongly_beneficial and is_strongly_harmful must return false
    /// when total_count < 30.
    #[test]
    fn helpful_strongly_requires_min_samples(
        positive in 0u32..15,
        negative in 0u32..15,
    ) {
        let stats = HelpfulStats {
            positive_count: positive,
            negative_count: negative,
            ..HelpfulStats::default()
        };
        prop_assert!(!stats.is_strongly_beneficial());
        prop_assert!(!stats.is_strongly_harmful());
    }

    /// merge() must be additive: merged total_count equals sum of both.
    #[test]
    fn helpful_merge_additive(
        p1 in 0u32..500,
        n1 in 0u32..500,
        p2 in 0u32..500,
        n2 in 0u32..500,
    ) {
        let mut stats1 = HelpfulStats {
            positive_count: p1,
            negative_count: n1,
            ..HelpfulStats::default()
        };
        let stats2 = HelpfulStats {
            positive_count: p2,
            negative_count: n2,
            ..HelpfulStats::default()
        };

        stats1.merge(&stats2);

        prop_assert_eq!(stats1.positive_count, p1 + p2);
        prop_assert_eq!(stats1.negative_count, n1 + n2);
        prop_assert_eq!(stats1.total_count(), p1 + n1 + p2 + n2);
    }

    /// merge() must accumulate improvement sums correctly.
    #[test]
    fn helpful_merge_sums(
        pos_sum1 in moderate_f32(),
        neg_sum1 in moderate_f32(),
        pos_sum2 in moderate_f32(),
        neg_sum2 in moderate_f32(),
    ) {
        let mut stats1 = HelpfulStats {
            positive_improvement_sum: pos_sum1,
            negative_improvement_sum: neg_sum1,
            ..HelpfulStats::default()
        };
        let stats2 = HelpfulStats {
            positive_improvement_sum: pos_sum2,
            negative_improvement_sum: neg_sum2,
            ..HelpfulStats::default()
        };

        stats1.merge(&stats2);

        let expected_pos = pos_sum1 + pos_sum2;
        let expected_neg = neg_sum1 + neg_sum2;
        prop_assert!(
            (stats1.positive_improvement_sum - expected_pos).abs() < 1e-3,
            "Positive sum mismatch after merge"
        );
        prop_assert!(
            (stats1.negative_improvement_sum - expected_neg).abs() < 1e-3,
            "Negative sum mismatch after merge"
        );
    }
}

// =============================================================================
// 2. HarmfulStats Properties
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// total_count() must equal harmful_count + helpful_count.
    #[test]
    fn harmful_total_count_equals_sum(
        harmful in 0u32..1000,
        helpful in 0u32..1000,
    ) {
        let stats = HarmfulStats {
            harmful_count: harmful,
            helpful_count: helpful,
            ..HarmfulStats::default()
        };
        prop_assert_eq!(stats.total_count(), harmful + helpful);
    }

    /// harmful_ratio() must always be in [0.0, 1.0].
    #[test]
    fn harmful_ratio_bounded(
        harmful in 0u32..1000,
        helpful in 0u32..1000,
    ) {
        let stats = HarmfulStats {
            harmful_count: harmful,
            helpful_count: helpful,
            ..HarmfulStats::default()
        };
        let ratio = stats.harmful_ratio();
        prop_assert!(
            (0.0..=1.0).contains(&ratio),
            "harmful_ratio {ratio} out of [0, 1]"
        );
    }

    /// harmful_ratio() must return 0.5 when total_count == 0.
    #[test]
    fn harmful_empty_ratio_neutral(_dummy in 0u8..1) {
        let stats = HarmfulStats::default();
        prop_assert!(
            (stats.harmful_ratio() - 0.5).abs() < 1e-10,
            "Empty stats should return 0.5 ratio"
        );
    }

    /// is_clearly_harmful must return false when total_count < 30.
    #[test]
    fn harmful_clearly_requires_min_samples(
        harmful in 0u32..15,
        helpful in 0u32..15,
    ) {
        let stats = HarmfulStats {
            harmful_count: harmful,
            helpful_count: helpful,
            ..HarmfulStats::default()
        };
        prop_assert!(!stats.is_clearly_harmful());
    }

    /// merge() must be additive.
    #[test]
    fn harmful_merge_additive(
        h1 in 0u32..500,
        hp1 in 0u32..500,
        h2 in 0u32..500,
        hp2 in 0u32..500,
    ) {
        let mut stats1 = HarmfulStats {
            harmful_count: h1,
            helpful_count: hp1,
            ..HarmfulStats::default()
        };
        let stats2 = HarmfulStats {
            harmful_count: h2,
            helpful_count: hp2,
            ..HarmfulStats::default()
        };

        stats1.merge(&stats2);

        prop_assert_eq!(stats1.harmful_count, h1 + h2);
        prop_assert_eq!(stats1.helpful_count, hp1 + hp2);
        prop_assert_eq!(stats1.total_count(), h1 + hp1 + h2 + hp2);
    }
}

// =============================================================================
// 3. NeuronStats Properties
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// NeuronStats::from_samples must return None for empty input.
    #[test]
    fn neuron_stats_empty_returns_none(_dummy in 0u8..1) {
        let result = NeuronStats::from_samples(&[]);
        prop_assert!(result.is_none());
    }

    /// NeuronStats variance must be non-negative.
    #[test]
    fn neuron_stats_variance_non_negative(
        samples in prop::collection::vec(helpful_sample_strategy(), 2..100),
    ) {
        if let Some(stats) = NeuronStats::from_samples(&samples) {
            prop_assert!(
                stats.error_variance >= 0.0,
                "Error variance must be non-negative, got {}",
                stats.error_variance
            );
            prop_assert!(
                stats.activation_variance >= 0.0,
                "Activation variance must be non-negative, got {}",
                stats.activation_variance
            );
        }
    }

    /// NeuronStats activation_min must be <= activation_max.
    #[test]
    fn neuron_stats_min_leq_max(
        samples in prop::collection::vec(helpful_sample_strategy(), 2..100),
    ) {
        if let Some(stats) = NeuronStats::from_samples(&samples) {
            prop_assert!(
                stats.activation_min <= stats.activation_max,
                "activation_min {} > activation_max {}",
                stats.activation_min,
                stats.activation_max
            );
        }
    }

    /// NeuronStats mean values must be finite for finite inputs.
    #[test]
    fn neuron_stats_means_finite(
        samples in prop::collection::vec(helpful_sample_strategy(), 1..100),
    ) {
        if let Some(stats) = NeuronStats::from_samples(&samples) {
            prop_assert!(
                stats.mean_error.is_finite(),
                "mean_error must be finite, got {}",
                stats.mean_error
            );
            prop_assert!(
                stats.mean_activation.is_finite(),
                "mean_activation must be finite, got {}",
                stats.mean_activation
            );
        }
    }

    /// NeuronStats must handle samples with NaN/Inf by filtering them out.
    #[test]
    fn neuron_stats_filters_non_finite(
        finite_count in 2usize..20,
    ) {
        let mut samples: Vec<HelpfulSample> = (0..finite_count)
            .map(|i| HelpfulSample {
                activation: i as f32 * 0.1,
                avg_error: 0.05,
                target_value: None,
                target_activation: None,
            })
            .collect();

        // Inject non-finite values
        samples.push(HelpfulSample {
            activation: f32::NAN,
            avg_error: 0.1,
            target_value: None,
            target_activation: None,
        });
        samples.push(HelpfulSample {
            activation: 0.5,
            avg_error: f32::INFINITY,
            target_value: None,
            target_activation: None,
        });

        if let Some(stats) = NeuronStats::from_samples(&samples) {
            prop_assert!(stats.mean_error.is_finite());
            prop_assert!(stats.mean_activation.is_finite());
            prop_assert!(stats.error_variance >= 0.0);
            prop_assert!(stats.activation_variance >= 0.0);
        }
    }

    /// NeuronStats constant samples should produce near-zero variance.
    /// Uses small values to avoid floating-point precision issues with
    /// E[X^2] - E[X]^2 computation (catastrophic cancellation for large means).
    #[test]
    fn neuron_stats_constant_zero_variance(
        activation in -10.0f32..10.0,
        error in -10.0f32..10.0,
        count in 5usize..50,
    ) {
        let samples: Vec<HelpfulSample> = (0..count)
            .map(|_| HelpfulSample {
                activation,
                avg_error: error,
                target_value: None,
                target_activation: None,
            })
            .collect();

        if let Some(stats) = NeuronStats::from_samples(&samples) {
            prop_assert!(
                stats.error_variance < 1e-3,
                "Constant error should have ~0 variance, got {}",
                stats.error_variance
            );
            prop_assert!(
                stats.activation_variance < 1e-3,
                "Constant activation should have ~0 variance, got {}",
                stats.activation_variance
            );
        }
    }
}
