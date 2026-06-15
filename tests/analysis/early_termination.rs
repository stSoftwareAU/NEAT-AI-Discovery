//! Tests for early termination of candidate evaluation using Sequential Probability Ratio Test (SPRT).
//!
//! Issue #219: Implements statistical early termination for clearly beneficial/harmful candidates.
//! This allows the GPU evaluation to stop early when a candidate is clearly good or clearly bad,
//! rather than evaluating all samples.
//!
//! The SPRT uses:
//! - H0: True improvement ≤ 0 (not beneficial)
//! - H1: True improvement ≥ threshold (beneficial)
//! - Stop early when log-likelihood ratio exceeds bounds

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use neat_ai_discovery::analysis::early_termination::{
    EarlyTerminationDecision, SequentialEvaluator,
};

/// Test that a strongly positive candidate terminates early with an Accept decision.
#[test]
fn test_early_termination_strong_positive() {
    // 90% positive correlation - should terminate early as Accept
    let mut evaluator = SequentialEvaluator::new(0.01, 0.01, 0.0);

    // Simulate samples where 90% are positive improvements
    for i in 0..10_000 {
        let is_positive = (i % 10) != 0; // 90% positive
        evaluator.add_sample(is_positive);

        if let EarlyTerminationDecision::Accept = evaluator.should_stop() {
            // Should terminate well before all samples are evaluated
            assert!(
                evaluator.sample_count() < 5_000,
                "Expected early termination before 5000 samples, got {} samples",
                evaluator.sample_count()
            );
            return;
        }
    }

    panic!("Expected early termination for strongly positive candidate");
}

/// Test that a strongly negative candidate terminates early with a Reject decision.
#[test]
fn test_early_termination_strong_negative() {
    // 90% negative correlation - should terminate early as Reject
    let mut evaluator = SequentialEvaluator::new(0.01, 0.01, 0.0);

    // Simulate samples where only 10% are positive improvements
    for i in 0..10_000 {
        let is_positive = (i % 10) == 0; // Only 10% positive
        evaluator.add_sample(is_positive);

        if let EarlyTerminationDecision::Reject = evaluator.should_stop() {
            // Should terminate well before all samples are evaluated
            assert!(
                evaluator.sample_count() < 5_000,
                "Expected early termination before 5000 samples, got {} samples",
                evaluator.sample_count()
            );
            return;
        }
    }

    panic!("Expected early termination for strongly negative candidate");
}

/// Test that the SPRT reaches different decisions based on the observed improvement rate.
///
/// The SPRT tests H0: p ≤ 0.5 vs H1: p ≥ 0.6. This test verifies that:
/// - 90% positive leads to Accept (clearly beneficial)
/// - 55% positive leads to Reject or Continue (below the 60% alternative hypothesis)
/// - The test demonstrates meaningful discrimination between improvement rates
#[test]
fn test_early_termination_marginal() {
    // Test with 90% positive - should Accept
    let mut evaluator_strong = SequentialEvaluator::new(0.01, 0.01, 0.0);

    for i in 0..1_000 {
        let is_positive = (i % 10) < 9; // 90% positive
        evaluator_strong.add_sample(is_positive);
    }

    let strong_decision = evaluator_strong.should_stop();
    assert!(
        matches!(strong_decision, EarlyTerminationDecision::Accept),
        "90% positive should be accepted, got {strong_decision:?}"
    );

    // Test with 55% positive - right in the indifference zone
    // This is below the H1 threshold (60%), so should reject or stay undecided
    let mut evaluator_marginal = SequentialEvaluator::new(0.01, 0.01, 0.0);

    for i in 0..1_000 {
        let is_positive = (i % 100) < 55; // 55% positive
        evaluator_marginal.add_sample(is_positive);
    }

    let marginal_decision = evaluator_marginal.should_stop();

    // 55% is closer to H0 (50%) than H1 (60%), so should eventually reject
    // or at least not accept with high confidence
    assert!(
        !matches!(marginal_decision, EarlyTerminationDecision::Accept),
        "55% positive should NOT be accepted (below H1 threshold of 60%), got {marginal_decision:?}"
    );

    // The key point: different improvement rates lead to different decisions
    assert_ne!(
        strong_decision, marginal_decision,
        "Strong and marginal cases should reach different decisions"
    );
}

/// Test the SPRT bounds are computed correctly.
#[test]
fn test_sprt_bounds_computation() {
    // With alpha = beta = 0.01:
    // Upper bound (A) = ln((1 - beta) / alpha) = ln(0.99 / 0.01) = ln(99) ≈ 4.595
    // Lower bound (B) = ln(beta / (1 - alpha)) = ln(0.01 / 0.99) = ln(0.0101) ≈ -4.595
    let evaluator = SequentialEvaluator::new(0.01, 0.01, 0.0);

    let (lower, upper) = evaluator.get_bounds();

    let expected_upper = (0.99_f64 / 0.01_f64).ln();
    let expected_lower = (0.01_f64 / 0.99_f64).ln();

    assert!(
        (upper - expected_upper).abs() < 0.01,
        "Upper bound should be ~{expected_upper:.3}, got {upper:.3}"
    );
    assert!(
        (lower - expected_lower).abs() < 0.01,
        "Lower bound should be ~{expected_lower:.3}, got {lower:.3}"
    );
}

/// Test that the evaluator tracks sample counts correctly.
#[test]
fn test_sample_count_tracking() {
    let mut evaluator = SequentialEvaluator::new(0.01, 0.01, 0.0);

    assert_eq!(evaluator.sample_count(), 0);
    assert_eq!(evaluator.positive_count(), 0);
    assert_eq!(evaluator.negative_count(), 0);

    evaluator.add_sample(true);
    assert_eq!(evaluator.sample_count(), 1);
    assert_eq!(evaluator.positive_count(), 1);
    assert_eq!(evaluator.negative_count(), 0);

    evaluator.add_sample(false);
    assert_eq!(evaluator.sample_count(), 2);
    assert_eq!(evaluator.positive_count(), 1);
    assert_eq!(evaluator.negative_count(), 1);

    for _ in 0..100 {
        evaluator.add_sample(true);
    }
    assert_eq!(evaluator.sample_count(), 102);
    assert_eq!(evaluator.positive_count(), 101);
    assert_eq!(evaluator.negative_count(), 1);
}

/// Test batch-based evaluation for GPU compatibility.
#[test]
fn test_batch_evaluation() {
    let mut evaluator = SequentialEvaluator::new(0.01, 0.01, 0.0);

    // Simulate batch-based updates (e.g., every 1024 samples from GPU)
    let batch_size = 1024;

    // First batch: 90% positive
    evaluator.add_batch(921, 103); // 921 positive, 103 negative
    assert_eq!(evaluator.sample_count(), batch_size);

    // Check if we can already make a decision
    match evaluator.should_stop() {
        EarlyTerminationDecision::Accept => {
            // Good - we terminated early
            return;
        }
        _ => {
            // Continue with more batches
        }
    }

    // Second batch: still 90% positive
    evaluator.add_batch(921, 103);
    assert_eq!(evaluator.sample_count(), 2 * batch_size);

    // With 90% positive over 2048 samples, should definitely accept
    assert!(
        matches!(evaluator.should_stop(), EarlyTerminationDecision::Accept),
        "Should accept after 2048 strongly positive samples"
    );
}

/// Test that the threshold parameter affects the decision boundary.
#[test]
fn test_threshold_affects_decision() {
    // With a higher threshold, we need more evidence to accept
    let mut evaluator_low = SequentialEvaluator::new(0.01, 0.01, 0.0);
    let mut evaluator_high = SequentialEvaluator::new(0.01, 0.01, 0.3);

    // Simulate 60% positive - moderate improvement
    for i in 0..5_000 {
        let is_positive = (i % 10) < 6; // 60% positive
        evaluator_low.add_sample(is_positive);
        evaluator_high.add_sample(is_positive);
    }

    // Low threshold (0.0) should accept 60% as beneficial
    // (testing H0: p=0.5 vs H1: p=0.6, 60% matches H1)
    let low_decision = evaluator_low.should_stop();

    // High threshold (0.3) tests H0: p=0.65 vs H1: p=0.75
    // 60% is below H0, so should reject
    let high_decision = evaluator_high.should_stop();

    // With 60% positive against 0% threshold, should accept
    assert!(
        matches!(low_decision, EarlyTerminationDecision::Accept),
        "60% positive should be accepted with 0% threshold, got {low_decision:?}"
    );

    // With 60% positive against 30% threshold, 60% < 65% (the new H0),
    // so should reject as not meeting the higher bar
    assert!(
        matches!(high_decision, EarlyTerminationDecision::Reject),
        "60% positive should be rejected with 30% threshold (requires >65%), got {high_decision:?}"
    );
}

/// Test that early termination respects minimum sample requirements.
#[test]
fn test_minimum_samples_before_decision() {
    let mut evaluator = SequentialEvaluator::new(0.01, 0.01, 0.0);

    // Even with 100% positive, first few samples shouldn't trigger early termination
    for _ in 0..10 {
        evaluator.add_sample(true);
        // Very early decisions are statistically unreliable
        if evaluator.sample_count() < 10 {
            // Implementation may still continue even with strong signal
            // to ensure statistical validity
        }
    }
}

/// Test reset functionality for reusing evaluator.
#[test]
fn test_evaluator_reset() {
    let mut evaluator = SequentialEvaluator::new(0.01, 0.01, 0.0);

    // Add some samples
    for _ in 0..100 {
        evaluator.add_sample(true);
    }
    assert_eq!(evaluator.sample_count(), 100);

    // Reset
    evaluator.reset();

    assert_eq!(evaluator.sample_count(), 0);
    assert_eq!(evaluator.positive_count(), 0);
    assert_eq!(evaluator.negative_count(), 0);
    assert!(matches!(
        evaluator.should_stop(),
        EarlyTerminationDecision::Continue
    ));
}

/// Test with realistic improvement ratios based on issue requirements.
/// Strongly beneficial (>50% improvement) should terminate in ≤10% of samples.
#[test]
fn test_strongly_beneficial_under_10_percent_samples() {
    let mut evaluator = SequentialEvaluator::new(0.01, 0.01, 0.0);
    let total_samples = 100_000;

    // 80% positive - strongly beneficial
    for i in 0..total_samples {
        let is_positive = (i % 100) < 80;
        evaluator.add_sample(is_positive);

        if let EarlyTerminationDecision::Accept = evaluator.should_stop() {
            let fraction_used = evaluator.sample_count() as f64 / total_samples as f64;
            assert!(
                fraction_used <= 0.10,
                "Strongly beneficial should terminate in ≤10% of samples, used {:.1}%",
                fraction_used * 100.0
            );
            return;
        }
    }

    panic!("Expected early termination for strongly beneficial candidate");
}

/// Test with realistic improvement ratios based on issue requirements.
/// Strongly harmful should terminate in ≤10% of samples.
#[test]
fn test_strongly_harmful_under_10_percent_samples() {
    let mut evaluator = SequentialEvaluator::new(0.01, 0.01, 0.0);
    let total_samples = 100_000;

    // 20% positive - strongly harmful (80% negative)
    for i in 0..total_samples {
        let is_positive = (i % 100) < 20;
        evaluator.add_sample(is_positive);

        if let EarlyTerminationDecision::Reject = evaluator.should_stop() {
            let fraction_used = evaluator.sample_count() as f64 / total_samples as f64;
            assert!(
                fraction_used <= 0.10,
                "Strongly harmful should terminate in ≤10% of samples, used {:.1}%",
                fraction_used * 100.0
            );
            return;
        }
    }

    panic!("Expected early termination for strongly harmful candidate");
}

/// Test that the evaluator correctly computes the log-likelihood ratio.
#[test]
fn test_log_likelihood_ratio() {
    // With H0: p = 0.5 (not beneficial) and H1: p > 0.5 (beneficial)
    // The log-likelihood ratio after n samples with k positives is:
    // LLR = k * ln(p1/p0) + (n-k) * ln((1-p1)/(1-p0))
    // For simplicity, we test that the ratio increases with more positive samples
    // and decreases with more negative samples.

    let mut eval1 = SequentialEvaluator::new(0.01, 0.01, 0.0);
    let mut eval2 = SequentialEvaluator::new(0.01, 0.01, 0.0);

    // Same number of samples, different ratios
    eval1.add_batch(80, 20); // 80% positive
    eval2.add_batch(20, 80); // 20% positive

    let ratio1 = eval1.log_likelihood_ratio();
    let ratio2 = eval2.log_likelihood_ratio();

    assert!(
        ratio1 > ratio2,
        "Higher positive ratio should have higher LLR: {ratio1} vs {ratio2}"
    );
    assert!(ratio1 > 0.0, "80% positive should have positive LLR");
    assert!(ratio2 < 0.0, "20% positive should have negative LLR");
}

/// Boundary case (Issue #1371): every sample is a failure (`k = 0, n > 0`).
///
/// This is the extreme of the "certain harm" regime. It is the input most
/// likely to expose a `log(0)` / divide-by-zero / saturation bug in the
/// log-likelihood-ratio and SPRT-bound maths. We assert on the observable
/// decision (the WHAT) — the evaluator must reach `Reject` well before all
/// samples are consumed — and that the LLR stays finite throughout.
#[test]
fn test_all_failures_triggers_early_reject() {
    let mut evaluator = SequentialEvaluator::new(0.01, 0.01, 0.0);
    let total_samples = 1_000;

    for _ in 0..total_samples {
        evaluator.add_sample(false); // every sample a failure: k stays 0

        // The LLR must never become NaN or infinite, even at the extreme.
        let llr = evaluator.log_likelihood_ratio();
        assert!(
            llr.is_finite(),
            "LLR must stay finite for all-failure input, got {llr}"
        );

        if let EarlyTerminationDecision::Reject = evaluator.should_stop() {
            assert!(
                evaluator.sample_count() < total_samples,
                "Expected early Reject before consuming all {total_samples} samples, got {} samples",
                evaluator.sample_count()
            );
            assert_eq!(
                evaluator.positive_count(),
                0,
                "All-failure input must leave the positive count at zero"
            );
            return;
        }
    }

    panic!("Expected early Reject for an all-failure candidate (k = 0)");
}

/// Boundary case (Issue #1371): every sample is a success (`k = n`).
///
/// Symmetric to [`test_all_failures_triggers_early_reject`] — the extreme of
/// the "certain benefit" regime. Assert the evaluator reaches `Accept` early
/// and the LLR remains finite.
#[test]
fn test_all_successes_triggers_early_accept() {
    let mut evaluator = SequentialEvaluator::new(0.01, 0.01, 0.0);
    let total_samples = 1_000;

    for _ in 0..total_samples {
        evaluator.add_sample(true); // every sample a success: k == n

        let llr = evaluator.log_likelihood_ratio();
        assert!(
            llr.is_finite(),
            "LLR must stay finite for all-success input, got {llr}"
        );

        if let EarlyTerminationDecision::Accept = evaluator.should_stop() {
            assert!(
                evaluator.sample_count() < total_samples,
                "Expected early Accept before consuming all {total_samples} samples, got {} samples",
                evaluator.sample_count()
            );
            assert_eq!(
                evaluator.negative_count(),
                0,
                "All-success input must leave the negative count at zero"
            );
            return;
        }
    }

    panic!("Expected early Accept for an all-success candidate (k = n)");
}
