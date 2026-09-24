//! Tests for Issue #1247: harden `ErrorDistribution` and mode detection
//! against the quantised `{0, 1}` regime produced by `CATEGORICAL_ERROR`.

#![allow(clippy::cast_precision_loss)]

use neat_ai_discovery::analysis::samples::HelpfulSample;
use neat_ai_discovery::analysis::scoring::error_distribution::ErrorDistribution;

fn make_sample(error: f32) -> HelpfulSample {
    HelpfulSample {
        activation: 0.5,
        avg_error: error,
        target_value: None,
        target_activation: None,
    }
}

/// Distribution statistics on a balanced quantised `{0, 1}` batch must
/// be entirely finite — no NaN from `0/0`, no `Inf` from a zero
/// denominator.
#[test]
fn distribution_stats_finite_for_quantised_zero_one_batch() {
    let samples: Vec<HelpfulSample> = (0..100)
        .map(|i| make_sample(if i % 2 == 0 { 0.0 } else { 1.0 }))
        .collect();

    let dist = ErrorDistribution::from_samples(&samples)
        .expect("distribution should be computed for non-empty samples");

    assert!(dist.mean.is_finite(), "mean must be finite: {}", dist.mean);
    assert!(dist.std_dev.is_finite(), "std_dev must be finite");
    assert!(dist.variance.is_finite(), "variance must be finite");
    assert!(dist.skewness.is_finite(), "skewness must be finite");
    assert!(dist.kurtosis.is_finite(), "kurtosis must be finite");
    for (i, p) in dist.percentiles.iter().enumerate() {
        assert!(p.is_finite(), "percentile[{i}] must be finite: {p}");
    }
    assert!(dist.iqr.is_finite(), "iqr must be finite");

    // Bernoulli(0.5) variance should be near 0.25 — sanity check that
    // we are not silently returning zero.
    assert!(
        (dist.variance - 0.25).abs() < 0.05,
        "Bernoulli(0.5) variance ≈ 0.25, got {}",
        dist.variance,
    );
}

/// Zero-variance batches (all-zero or all-one error) must not divide
/// by zero. Skewness defaults to `0.0` and kurtosis to `3.0`.
#[test]
fn distribution_stats_safe_for_zero_variance_batch() {
    for constant in [0.0_f32, 1.0_f32] {
        let samples = vec![make_sample(constant); 32];
        let dist = ErrorDistribution::from_samples(&samples)
            .expect("non-empty sample should produce a distribution");

        assert!(dist.mean.is_finite());
        assert_eq!(dist.std_dev, 0.0);
        assert_eq!(dist.variance, 0.0);
        // Guard kicks in: skewness becomes 0, kurtosis defaults to 3.
        assert_eq!(dist.skewness, 0.0);
        assert!(dist.kurtosis.is_finite());
    }
}
