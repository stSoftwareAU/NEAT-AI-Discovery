//! Property-based tests for SPRT early termination (Issue #680).
//!
//! Uses `proptest` to verify mathematical invariants of `SequentialEvaluator`,
//! `EarlyTerminationConfig`, and `check_batch_early_termination`. These tests
//! complement existing unit tests by exploring edge cases with randomised inputs.

#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use neat_ai_discovery::analysis::early_termination::{
    EarlyTerminationConfig, EarlyTerminationDecision, SequentialEvaluator,
    check_batch_early_termination,
};
use neat_ai_discovery::analysis::samples::HelpfulStats;
use proptest::prelude::*;

// =============================================================================
// 1. SequentialEvaluator Core Invariants
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// sample_count() must always equal positive_count() + negative_count().
    #[test]
    fn sample_count_equals_sum(
        positive in 0u32..1000,
        negative in 0u32..1000,
    ) {
        let mut eval = SequentialEvaluator::default();
        eval.add_batch(positive, negative);
        prop_assert_eq!(
            eval.sample_count(),
            u64::from(positive) + u64::from(negative),
            "sample_count must equal positive + negative"
        );
    }

    /// improvement_ratio() must always be in [0.0, 1.0].
    #[test]
    fn improvement_ratio_bounded(
        positive in 0u32..1000,
        negative in 0u32..1000,
    ) {
        let mut eval = SequentialEvaluator::default();
        eval.add_batch(positive, negative);
        let ratio = eval.improvement_ratio();
        prop_assert!(
            (0.0..=1.0).contains(&ratio),
            "improvement_ratio {ratio} out of [0, 1]"
        );
    }

    /// Empty evaluator must return 0.5 for improvement_ratio.
    #[test]
    fn empty_evaluator_neutral_ratio(_dummy in 0u8..1) {
        let eval = SequentialEvaluator::default();
        prop_assert!(
            (eval.improvement_ratio() - 0.5).abs() < 1e-10,
            "Empty evaluator should return 0.5 ratio"
        );
    }

    /// is_strongly_beneficial and is_strongly_harmful must be mutually exclusive.
    #[test]
    fn beneficial_harmful_mutually_exclusive(
        positive in 0u32..500,
        negative in 0u32..500,
    ) {
        let mut eval = SequentialEvaluator::default();
        eval.add_batch(positive, negative);
        prop_assert!(
            !(eval.is_strongly_beneficial() && eval.is_strongly_harmful()),
            "Cannot be both beneficial and harmful: positive={positive}, negative={negative}"
        );
    }

    /// Both is_strongly_beneficial and is_strongly_harmful must return false
    /// when total samples < 30 (the minimum sample threshold).
    #[test]
    fn strongly_requires_minimum_samples(
        positive in 0u32..15,
        negative in 0u32..15,
    ) {
        let mut eval = SequentialEvaluator::default();
        eval.add_batch(positive, negative);
        // Total is at most 28, which is < 30
        prop_assert!(
            !eval.is_strongly_beneficial(),
            "Should not be strongly beneficial with {} samples",
            eval.sample_count()
        );
        prop_assert!(
            !eval.is_strongly_harmful(),
            "Should not be strongly harmful with {} samples",
            eval.sample_count()
        );
    }

    /// should_stop() must return Continue when sample_count < min_samples (30).
    #[test]
    fn should_stop_continues_below_minimum(
        positive in 0u32..15,
        negative in 0u32..15,
    ) {
        let mut eval = SequentialEvaluator::default();
        eval.add_batch(positive, negative);
        prop_assert_eq!(
            eval.should_stop(),
            EarlyTerminationDecision::Continue,
            "Must continue with {} samples (< 30)",
            eval.sample_count()
        );
    }

    /// reset() must zero all counts and restore neutral ratio.
    #[test]
    fn reset_clears_state(
        positive in 0u32..1000,
        negative in 0u32..1000,
    ) {
        let mut eval = SequentialEvaluator::default();
        eval.add_batch(positive, negative);
        eval.reset();
        prop_assert_eq!(eval.sample_count(), 0);
        prop_assert_eq!(eval.positive_count(), 0);
        prop_assert_eq!(eval.negative_count(), 0);
        prop_assert!(
            (eval.improvement_ratio() - 0.5).abs() < 1e-10,
            "Reset evaluator should return 0.5 ratio"
        );
    }

    /// add_sample(true) increments positive_count, add_sample(false) increments negative_count.
    #[test]
    fn add_sample_increments_correctly(
        positives in prop::collection::vec(Just(true), 0..50),
        negatives in prop::collection::vec(Just(false), 0..50),
    ) {
        let mut eval = SequentialEvaluator::default();
        for &is_positive in &positives {
            eval.add_sample(is_positive);
        }
        for &is_positive in &negatives {
            eval.add_sample(is_positive);
        }
        prop_assert_eq!(eval.positive_count(), positives.len() as u64);
        prop_assert_eq!(eval.negative_count(), negatives.len() as u64);
    }
}

// =============================================================================
// 2. Log-Likelihood Ratio Properties
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// LLR must be zero when no samples have been added.
    #[test]
    fn llr_zero_when_empty(
        alpha in 0.001f64..0.1,
        beta in 0.001f64..0.1,
    ) {
        let eval = SequentialEvaluator::new(alpha, beta, 0.0);
        prop_assert!(
            eval.log_likelihood_ratio().abs() < 1e-10,
            "LLR should be 0 with no samples"
        );
    }

    /// Higher positive ratio should yield higher LLR (at fixed sample count).
    #[test]
    fn llr_increases_with_positive_ratio(
        total in 30u32..200,
        low_positive_frac in 0.1f64..0.4,
        high_positive_frac in 0.6f64..0.9,
    ) {
        let low_pos = (total as f64 * low_positive_frac) as u32;
        let high_pos = (total as f64 * high_positive_frac) as u32;

        let mut eval_low = SequentialEvaluator::default();
        eval_low.add_batch(low_pos, total - low_pos);

        let mut eval_high = SequentialEvaluator::default();
        eval_high.add_batch(high_pos, total - high_pos);

        let llr_low = eval_low.log_likelihood_ratio();
        let llr_high = eval_high.log_likelihood_ratio();

        prop_assert!(
            llr_high > llr_low,
            "Higher positive ratio should yield higher LLR: low={llr_low} (pos={low_pos}), high={llr_high} (pos={high_pos})"
        );
    }

    /// LLR must be finite for any valid sample counts.
    #[test]
    fn llr_always_finite(
        positive in 0u32..10000,
        negative in 0u32..10000,
    ) {
        let mut eval = SequentialEvaluator::default();
        eval.add_batch(positive, negative);
        let llr = eval.log_likelihood_ratio();
        prop_assert!(
            llr.is_finite(),
            "LLR must be finite: positive={positive}, negative={negative}, llr={llr}"
        );
    }
}

// =============================================================================
// 3. SPRT Bounds Properties
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// With equal alpha and beta, bounds should be symmetric around zero.
    #[test]
    fn symmetric_bounds_for_equal_error_rates(
        rate in 0.001f64..0.1,
    ) {
        let eval = SequentialEvaluator::new(rate, rate, 0.0);
        let (lower, upper) = eval.get_bounds();
        prop_assert!(
            (lower + upper).abs() < 0.01,
            "Bounds should be symmetric for equal rates: lower={lower}, upper={upper}"
        );
    }

    /// Upper bound must always be greater than lower bound.
    #[test]
    fn upper_bound_exceeds_lower(
        alpha in 0.001f64..0.5,
        beta in 0.001f64..0.5,
    ) {
        let eval = SequentialEvaluator::new(alpha, beta, 0.0);
        let (lower, upper) = eval.get_bounds();
        prop_assert!(
            upper > lower,
            "Upper bound {upper} must exceed lower bound {lower}"
        );
    }
}

// =============================================================================
// 4. Batch Early Termination Properties
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// When early termination is disabled, all candidates must continue.
    #[test]
    fn disabled_config_all_continue(
        count in 1usize..20,
        positive in 0u32..100,
        negative in 0u32..100,
    ) {
        let stats: Vec<HelpfulStats> = (0..count)
            .map(|_| HelpfulStats {
                positive_count: positive,
                negative_count: negative,
                ..HelpfulStats::default()
            })
            .collect();

        let config = EarlyTerminationConfig::disabled();
        let result = check_batch_early_termination(&stats, &config);

        prop_assert!(result.all_continue(), "Disabled config should yield all-continue");
        prop_assert_eq!(result.continue_indices.len(), count);
        prop_assert!(result.accept_indices.is_empty());
        prop_assert!(result.reject_indices.is_empty());
    }

    /// Every candidate index must appear in exactly one of the three result lists.
    #[test]
    fn all_indices_partitioned(
        stats in prop::collection::vec(
            (0u32..200, 0u32..200).prop_map(|(p, n)| HelpfulStats {
                positive_count: p,
                negative_count: n,
                ..HelpfulStats::default()
            }),
            1..20,
        ),
    ) {
        let config = EarlyTerminationConfig::default();
        let result = check_batch_early_termination(&stats, &config);

        let total = result.accept_indices.len()
            + result.reject_indices.len()
            + result.continue_indices.len();
        prop_assert_eq!(
            total,
            stats.len(),
            "All indices must be partitioned: accept={}, reject={}, continue={}, total={}",
            result.accept_indices.len(),
            result.reject_indices.len(),
            result.continue_indices.len(),
            stats.len()
        );
    }
}
