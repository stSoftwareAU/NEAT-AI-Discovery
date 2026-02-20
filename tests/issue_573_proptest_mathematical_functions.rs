//! Property-based tests for mathematical functions (Issue #573).
//!
//! Uses `proptest` to verify mathematical properties of scoring, confidence,
//! error distribution, cross-validation, weight calculation, and activation
//! functions. These tests complement existing example-based tests by exploring
//! edge cases (NaN propagation, overflow, extreme values, boundary conditions)
//! that hand-written examples may miss.

mod common;

use neat_ai_discovery::activations::apply_scalar_squash;
use neat_ai_discovery::analysis::constants::{cmp_f32_asc, cmp_f32_desc};
use neat_ai_discovery::analysis::neuron_fingerprint::compute_neuron_fingerprints;
use neat_ai_discovery::analysis::samples::HelpfulSample;
use neat_ai_discovery::analysis::scoring::confidence::compute_confidence_metrics;
use neat_ai_discovery::analysis::scoring::cross_validation::{
    CrossValidationConfig, FoldResult, PerformanceVariance, apply_brittleness_penalty,
    compute_cross_validation_score,
};
use neat_ai_discovery::analysis::scoring::error_distribution::ErrorDistribution;
use neat_ai_discovery::analysis::scoring::weights::{
    MAX_OUTGOING_WEIGHT, calculate_optimal_outgoing_weight, clamp_weight_update_delta,
    coordinated_structural_activation_delta,
};
use proptest::prelude::*;

// =============================================================================
// Test Helpers
// =============================================================================

fn make_sample(activation: f32, avg_error: f32) -> HelpfulSample {
    HelpfulSample {
        activation,
        avg_error,
        target_value: None,
        target_activation: None,
    }
}

/// Strategy for generating finite f32 values (no NaN/Inf).
fn finite_f32() -> impl Strategy<Value = f32> {
    prop::num::f32::NORMAL
        .prop_filter("must be finite", |v| v.is_finite())
        .prop_map(|v| {
            if v == 0.0 && v.is_sign_negative() {
                0.0
            } else {
                v
            }
        })
}

/// Strategy for generating moderate finite f32 values (avoids extremes).
fn moderate_f32() -> impl Strategy<Value = f32> {
    (-1e6f32..1e6f32).prop_filter("must be finite", |v| v.is_finite())
}

/// Strategy for generating small finite f32 values suitable for activations.
fn activation_f32() -> impl Strategy<Value = f32> {
    -10.0f32..10.0f32
}

/// Strategy for generating HelpfulSample with finite values.
fn helpful_sample_strategy() -> impl Strategy<Value = HelpfulSample> {
    (activation_f32(), activation_f32()).prop_map(|(a, e)| make_sample(a, e))
}

/// Strategy for generating a Vec of HelpfulSamples of given size range.
fn helpful_samples(min_size: usize, max_size: usize) -> impl Strategy<Value = Vec<HelpfulSample>> {
    prop::collection::vec(helpful_sample_strategy(), min_size..=max_size)
}

// =============================================================================
// 1. Scoring / Confidence Property Tests
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Confidence score must always be in [0.0, 1.0].
    #[test]
    fn confidence_score_bounded(
        samples in helpful_samples(1, 200),
        gain in moderate_f32(),
        r_squared in prop::option::of(-1.0f32..2.0f32),
    ) {
        let metrics = compute_confidence_metrics(&samples, gain, r_squared);
        prop_assert!(
            (0.0..=1.0).contains(&metrics.prediction_confidence),
            "Confidence {} out of bounds [0, 1]",
            metrics.prediction_confidence
        );
    }

    /// Confidence interval lower bound must be <= upper bound.
    #[test]
    fn confidence_interval_ordered(
        samples in helpful_samples(1, 200),
        gain in moderate_f32(),
    ) {
        let metrics = compute_confidence_metrics(&samples, gain, None);
        let [lower, upper] = metrics.expected_score_gain_confidence_interval;
        prop_assert!(
            lower <= upper || (!lower.is_finite() || !upper.is_finite()),
            "CI bounds not ordered: lower={lower}, upper={upper}"
        );
    }

    /// Empty samples must produce zero confidence.
    #[test]
    fn confidence_empty_samples_zero(
        gain in moderate_f32(),
        r_squared in prop::option::of(-1.0f32..2.0f32),
    ) {
        let metrics = compute_confidence_metrics(&[], gain, r_squared);
        prop_assert_eq!(metrics.prediction_confidence, 0.0);
    }

    /// More samples should not decrease confidence (monotonicity with sample count).
    /// We test with identical samples to isolate the sample-count effect.
    #[test]
    fn confidence_monotonic_with_sample_count(
        activation in activation_f32(),
        error in activation_f32(),
        small_count in 5usize..50,
        large_count in 51usize..200,
    ) {
        let small_samples: Vec<HelpfulSample> = (0..small_count)
            .map(|i| make_sample(
                activation + (i as f32 * 0.01), // slight variation to avoid zero variance
                error,
            ))
            .collect();
        let large_samples: Vec<HelpfulSample> = (0..large_count)
            .map(|i| make_sample(
                activation + (i as f32 * 0.01),
                error,
            ))
            .collect();

        let small_metrics = compute_confidence_metrics(&small_samples, 0.1, None);
        let large_metrics = compute_confidence_metrics(&large_samples, 0.1, None);

        // With more samples, confidence should be >= (or very close due to floating point)
        prop_assert!(
            large_metrics.prediction_confidence >= small_metrics.prediction_confidence - 0.05,
            "Confidence should not decrease much with more samples: small={} ({small_count} samples), large={} ({large_count} samples)",
            small_metrics.prediction_confidence,
            large_metrics.prediction_confidence
        );
    }
}

// =============================================================================
// 2. Error Distribution Property Tests
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Variance must be non-negative.
    #[test]
    fn error_distribution_variance_non_negative(
        errors in prop::collection::vec(moderate_f32(), 2..200),
    ) {
        if let Some(dist) = ErrorDistribution::from_errors(&errors) {
            prop_assert!(
                dist.variance >= 0.0,
                "Variance must be non-negative, got {}",
                dist.variance
            );
        }
    }

    /// Standard deviation must equal sqrt(variance).
    #[test]
    fn error_distribution_std_dev_consistent(
        errors in prop::collection::vec(moderate_f32(), 2..200),
    ) {
        if let Some(dist) = ErrorDistribution::from_errors(&errors) {
            let expected_std_dev = dist.variance.sqrt();
            // Use relative tolerance for large values
            let tolerance = (dist.std_dev.abs() * 1e-4).max(1e-4);
            prop_assert!(
                (dist.std_dev - expected_std_dev).abs() < tolerance,
                "std_dev {} != sqrt(variance) {} (tolerance {})",
                dist.std_dev, expected_std_dev, tolerance
            );
        }
    }

    /// Mean must be between min and max.
    #[test]
    fn error_distribution_mean_bounded(
        errors in prop::collection::vec(moderate_f32(), 1..200),
    ) {
        if let Some(dist) = ErrorDistribution::from_errors(&errors) {
            prop_assert!(
                dist.mean >= dist.min && dist.mean <= dist.max,
                "Mean {} not in [{}, {}]",
                dist.mean, dist.min, dist.max
            );
        }
    }

    /// Percentiles must be ordered: p10 <= p25 <= p50 <= p75 <= p90.
    #[test]
    fn error_distribution_percentiles_ordered(
        errors in prop::collection::vec(moderate_f32(), 2..200),
    ) {
        if let Some(dist) = ErrorDistribution::from_errors(&errors) {
            let [p10, p25, p50, p75, p90] = dist.percentiles;
            prop_assert!(p10 <= p25, "p10={p10} > p25={p25}");
            prop_assert!(p25 <= p50, "p25={p25} > p50={p50}");
            prop_assert!(p50 <= p75, "p50={p50} > p75={p75}");
            prop_assert!(p75 <= p90, "p75={p75} > p90={p90}");
        }
    }

    /// All percentiles must be within [min, max].
    #[test]
    fn error_distribution_percentiles_bounded(
        errors in prop::collection::vec(moderate_f32(), 2..200),
    ) {
        if let Some(dist) = ErrorDistribution::from_errors(&errors) {
            for (i, &p) in dist.percentiles.iter().enumerate() {
                prop_assert!(
                    p >= dist.min && p <= dist.max,
                    "Percentile {} ({}) not in [{}, {}]",
                    i, p, dist.min, dist.max
                );
            }
        }
    }

    /// IQR must be non-negative (p75 >= p25).
    #[test]
    fn error_distribution_iqr_non_negative(
        errors in prop::collection::vec(moderate_f32(), 2..200),
    ) {
        if let Some(dist) = ErrorDistribution::from_errors(&errors) {
            prop_assert!(
                dist.iqr >= 0.0,
                "IQR must be non-negative, got {}",
                dist.iqr
            );
        }
    }

    /// Empty errors must return None.
    #[test]
    fn error_distribution_empty_returns_none(_dummy in 0u8..1) {
        let result = ErrorDistribution::from_errors(&[]);
        prop_assert!(result.is_none());
    }

    /// Constant data must have zero variance and zero std_dev.
    #[test]
    fn error_distribution_constant_zero_variance(
        value in moderate_f32(),
        count in 2usize..50,
    ) {
        let errors: Vec<f32> = vec![value; count];
        if let Some(dist) = ErrorDistribution::from_errors(&errors) {
            prop_assert!(
                dist.variance < 1e-6,
                "Constant data should have ~0 variance, got {}",
                dist.variance
            );
            prop_assert!(
                dist.std_dev < 1e-3,
                "Constant data should have ~0 std_dev, got {}",
                dist.std_dev
            );
        }
    }

    /// Kurtosis of a uniform distribution should be positive.
    #[test]
    fn error_distribution_kurtosis_positive(
        errors in prop::collection::vec(moderate_f32(), 5..200),
    ) {
        if let Some(dist) = ErrorDistribution::from_errors(&errors) {
            // Kurtosis is always positive (it is E[(X-mu)^4] / sigma^4)
            prop_assert!(
                dist.kurtosis >= 0.0 || dist.std_dev < 1e-10,
                "Kurtosis should be non-negative for varying data, got {} (std_dev={})",
                dist.kurtosis, dist.std_dev
            );
        }
    }
}

// =============================================================================
// 3. Cross-Validation and Brittleness Property Tests
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Brittleness penalty must be in [0.0, 1.0].
    #[test]
    fn brittleness_penalty_bounded(
        confidence in 0.0f32..1.0,
        penalty in -0.5f32..1.5,
    ) {
        let adjusted = apply_brittleness_penalty(confidence, penalty);
        prop_assert!(
            (0.0..=1.0).contains(&adjusted),
            "Adjusted confidence {} out of [0, 1]",
            adjusted
        );
    }

    /// Zero penalty should preserve confidence.
    #[test]
    fn brittleness_zero_penalty_preserves(
        confidence in 0.0f32..1.0,
    ) {
        let adjusted = apply_brittleness_penalty(confidence, 0.0);
        prop_assert!(
            (adjusted - confidence).abs() < 1e-6,
            "Zero penalty should preserve confidence: {} vs {}",
            adjusted, confidence
        );
    }

    /// Full penalty should zero out confidence.
    #[test]
    fn brittleness_full_penalty_zeros(
        confidence in 0.0f32..1.0,
    ) {
        let adjusted = apply_brittleness_penalty(confidence, 1.0);
        prop_assert!(
            adjusted.abs() < 1e-6,
            "Full penalty should zero confidence: {}",
            adjusted
        );
    }

    /// Higher penalty should produce lower adjusted confidence (monotonic decrease).
    #[test]
    fn brittleness_penalty_monotonic(
        confidence in 0.0f32..1.0,
        penalty_low in 0.0f32..0.5,
        penalty_high in 0.5f32..1.0,
    ) {
        let adjusted_low = apply_brittleness_penalty(confidence, penalty_low);
        let adjusted_high = apply_brittleness_penalty(confidence, penalty_high);
        prop_assert!(
            adjusted_high <= adjusted_low + 1e-6,
            "Higher penalty should yield lower confidence: low={adjusted_low} (penalty={penalty_low}), high={adjusted_high} (penalty={penalty_high})"
        );
    }

    /// FoldResult improvement_ratio must be in [0.0, 1.0].
    #[test]
    fn fold_result_ratio_bounded(
        positive in 0u32..100,
        negative in 0u32..100,
    ) {
        let fold = FoldResult {
            positive_count: positive,
            negative_count: negative,
            samples_evaluated: positive + negative,
        };
        let ratio = fold.improvement_ratio();
        prop_assert!(
            (0.0..=1.0).contains(&ratio),
            "Improvement ratio {} out of [0, 1]",
            ratio
        );
    }

    /// PerformanceVariance from identical folds should have zero variance.
    #[test]
    fn performance_variance_identical_folds(
        positive in 1u32..100,
        negative in 0u32..100,
        fold_count in 2usize..10,
    ) {
        let folds: Vec<FoldResult> = (0..fold_count)
            .map(|_| FoldResult {
                positive_count: positive,
                negative_count: negative,
                samples_evaluated: positive + negative,
            })
            .collect();
        let variance = PerformanceVariance::from_folds(&folds);
        prop_assert!(
            variance.variance < 1e-10,
            "Identical folds should have ~0 variance, got {}",
            variance.variance
        );
    }

    /// PerformanceVariance must have non-negative variance.
    #[test]
    fn performance_variance_non_negative(
        folds in prop::collection::vec(
            (0u32..100, 0u32..100).prop_map(|(p, n)| FoldResult {
                positive_count: p,
                negative_count: n,
                samples_evaluated: p + n,
            }),
            1..10,
        ),
    ) {
        let variance = PerformanceVariance::from_folds(&folds);
        prop_assert!(
            variance.variance >= 0.0,
            "Variance must be non-negative, got {}",
            variance.variance
        );
    }

    /// Cross-validation with sufficient samples should produce a result.
    #[test]
    fn cross_validation_sufficient_samples(
        samples in helpful_samples(100, 200),
    ) {
        let config = CrossValidationConfig::default();
        let result = compute_cross_validation_score(&samples, &config);
        prop_assert!(
            result.is_some(),
            "Should produce result with {} samples (config: {} folds, {} min per fold)",
            samples.len(), config.fold_count, config.min_samples_per_fold
        );
        if let Some(cv) = result {
            prop_assert!(
                (0.0..=1.0).contains(&cv.brittleness_penalty),
                "Brittleness penalty {} out of [0, 1]",
                cv.brittleness_penalty
            );
        }
    }
}

// =============================================================================
// 4. Weight Calculation Property Tests
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Optimal outgoing weight must be within [-MAX_OUTGOING_WEIGHT, MAX_OUTGOING_WEIGHT].
    #[test]
    fn optimal_weight_clamped(
        sum_ea in moderate_f32(),
        sum_aa in 0.001f32..1e6,
        incoming_weight in 0.01f32..1.0, // Keep <= 1.0 to avoid ratio rejection
    ) {
        if let Some(weight) = calculate_optimal_outgoing_weight(sum_ea, sum_aa, incoming_weight) {
            prop_assert!(
                weight.abs() <= MAX_OUTGOING_WEIGHT + 1e-6,
                "Weight {} exceeds MAX_OUTGOING_WEIGHT {}",
                weight, MAX_OUTGOING_WEIGHT
            );
            prop_assert!(
                weight.is_finite(),
                "Weight must be finite"
            );
        }
    }

    /// Weight with zero activation energy must return None.
    #[test]
    fn optimal_weight_zero_activation_none(
        sum_ea in moderate_f32(),
        incoming_weight in finite_f32(),
    ) {
        let result = calculate_optimal_outgoing_weight(sum_ea, 0.0, incoming_weight);
        prop_assert!(result.is_none(), "Zero activation energy should return None");
    }

    /// Clamp weight delta: result must be within [-MAX, MAX].
    #[test]
    fn clamp_weight_delta_bounded(
        old_weight in -0.1f32..0.1,
        delta in -1.0f32..1.0,
    ) {
        if let Some((new_weight, _)) = clamp_weight_update_delta(old_weight, delta) {
            prop_assert!(
                new_weight.abs() <= MAX_OUTGOING_WEIGHT + 1e-6,
                "Clamped weight {} exceeds MAX",
                new_weight
            );
        }
    }

    /// Clamp weight delta: effective delta must equal new_weight - old_weight.
    #[test]
    fn clamp_weight_delta_consistent(
        old_weight in -0.1f32..0.1,
        delta in -1.0f32..1.0,
    ) {
        if let Some((new_weight, effective_delta)) = clamp_weight_update_delta(old_weight, delta) {
            let expected_delta = new_weight - old_weight;
            prop_assert!(
                (effective_delta - expected_delta).abs() < 1e-6,
                "Delta {} != new({}) - old({})",
                effective_delta, new_weight, old_weight
            );
        }
    }

    /// Coordinated structural delta must be finite when inputs are finite and noisy_weight != 0.
    #[test]
    fn coordinated_delta_finite(
        trusted_activation in activation_f32(),
        noisy_activation in activation_f32(),
        noisy_weight in activation_f32().prop_filter("non-zero", |w| w.abs() > 1e-6),
        trusted_weight in activation_f32(),
    ) {
        if let Some(delta) = coordinated_structural_activation_delta(
            trusted_activation, noisy_activation, noisy_weight, trusted_weight,
        ) {
            prop_assert!(
                delta.is_finite(),
                "Delta must be finite: trusted_act={trusted_activation}, noisy_act={noisy_activation}, noisy_w={noisy_weight}, trusted_w={trusted_weight}"
            );
        }
    }

    /// Coordinated structural delta with zero noisy weight must return None.
    #[test]
    fn coordinated_delta_zero_noisy_weight_none(
        trusted_act in activation_f32(),
        noisy_act in activation_f32(),
        trusted_w in activation_f32(),
    ) {
        let result = coordinated_structural_activation_delta(
            trusted_act, noisy_act, 0.0, trusted_w,
        );
        prop_assert!(result.is_none(), "Zero noisy weight should return None");
    }
}

// =============================================================================
// 5. Activation Function Property Tests
// =============================================================================

/// Scalar activation function names to test.
const BOUNDED_ACTIVATIONS: &[(&str, f32, f32)] = &[
    ("LOGISTIC", 0.0, 1.0),
    ("HARD_TANH", -1.0, 1.0),
    ("TANH", -1.0, 1.0),
    ("RELU6", 0.0, 6.0),
    ("STEP", 0.0, 1.0),
    ("BIPOLAR", -1.0, 1.0),
    ("SOFTSIGN", -1.0, 1.0),
    ("ISRU", -1.0, 1.0),
];

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Bounded activations must stay within their declared bounds.
    #[test]
    fn bounded_activation_within_range(
        x in -100.0f32..100.0,
        idx in 0..BOUNDED_ACTIVATIONS.len(),
    ) {
        let (name, lower, upper) = BOUNDED_ACTIVATIONS[idx];
        if let Some(y) = apply_scalar_squash(name, x) {
            prop_assert!(
                y >= lower - 1e-5 && y <= upper + 1e-5,
                "{name}({x}) = {y} not in [{lower}, {upper}]"
            );
        }
    }

    /// RELU(x) >= 0 for all finite x.
    #[test]
    fn relu_non_negative(x in finite_f32()) {
        if let Some(y) = apply_scalar_squash("RELU", x) {
            prop_assert!(y >= 0.0, "RELU({x}) = {y} should be >= 0");
        }
    }

    /// IDENTITY(x) == x for all finite x.
    #[test]
    fn identity_is_identity(x in finite_f32()) {
        if let Some(y) = apply_scalar_squash("IDENTITY", x) {
            prop_assert!(
                (y - x).abs() < 1e-6,
                "IDENTITY({x}) = {y} should equal {x}"
            );
        }
    }

    /// ABSOLUTE(x) >= 0 for all finite x.
    #[test]
    fn absolute_non_negative(x in finite_f32()) {
        if let Some(y) = apply_scalar_squash("ABSOLUTE", x) {
            prop_assert!(y >= 0.0, "ABSOLUTE({x}) = {y} should be >= 0");
        }
    }

    /// SQUARE(x) >= 0 for all finite x.
    #[test]
    fn square_non_negative(x in moderate_f32()) {
        if let Some(y) = apply_scalar_squash("SQUARE", x) {
            prop_assert!(y >= 0.0, "SQUARE({x}) = {y} should be >= 0");
        }
    }

    /// LOGISTIC is monotonically increasing.
    #[test]
    fn logistic_monotonic(
        x1 in -50.0f32..50.0,
        x2 in -50.0f32..50.0,
    ) {
        let y1 = apply_scalar_squash("LOGISTIC", x1).unwrap();
        let y2 = apply_scalar_squash("LOGISTIC", x2).unwrap();
        if x1 < x2 {
            prop_assert!(
                y1 <= y2 + 1e-6,
                "LOGISTIC not monotonic: f({x1})={y1} > f({x2})={y2}"
            );
        }
    }

    /// TANH is monotonically increasing.
    #[test]
    fn tanh_monotonic(
        x1 in -50.0f32..50.0,
        x2 in -50.0f32..50.0,
    ) {
        let y1 = apply_scalar_squash("TANH", x1).unwrap();
        let y2 = apply_scalar_squash("TANH", x2).unwrap();
        if x1 < x2 {
            prop_assert!(
                y1 <= y2 + 1e-6,
                "TANH not monotonic: f({x1})={y1} > f({x2})={y2}"
            );
        }
    }

    /// RELU is monotonically increasing.
    #[test]
    fn relu_monotonic(
        x1 in moderate_f32(),
        x2 in moderate_f32(),
    ) {
        let y1 = apply_scalar_squash("RELU", x1).unwrap();
        let y2 = apply_scalar_squash("RELU", x2).unwrap();
        if x1 < x2 {
            prop_assert!(
                y1 <= y2 + 1e-6,
                "RELU not monotonic: f({x1})={y1} > f({x2})={y2}"
            );
        }
    }

    /// Determinism: same input always produces same output.
    #[test]
    fn activation_deterministic(
        x in activation_f32(),
        idx in 0..BOUNDED_ACTIVATIONS.len(),
    ) {
        let (name, _, _) = BOUNDED_ACTIVATIONS[idx];
        let y1 = apply_scalar_squash(name, x);
        let y2 = apply_scalar_squash(name, x);
        prop_assert_eq!(
            y1.map(|v| v.to_bits()), y2.map(|v| v.to_bits()),
            "{}({}) not deterministic: {:?} vs {:?}", name, x, y1, y2
        );
    }

    /// GAUSSIAN is symmetric: GAUSSIAN(x) == GAUSSIAN(-x).
    #[test]
    fn gaussian_symmetric(x in activation_f32()) {
        let y_pos = apply_scalar_squash("GAUSSIAN", x).unwrap();
        let y_neg = apply_scalar_squash("GAUSSIAN", -x).unwrap();
        prop_assert!(
            (y_pos - y_neg).abs() < 1e-5,
            "GAUSSIAN not symmetric: f({x})={y_pos}, f({})={y_neg}",
            -x
        );
    }

    /// COMPLEMENT(x) == 1 - x.
    #[test]
    fn complement_is_one_minus_x(x in finite_f32()) {
        if let Some(y) = apply_scalar_squash("COMPLEMENT", x) {
            prop_assert!(
                (y - (1.0 - x)).abs() < 1e-5,
                "COMPLEMENT({x}) = {y} should equal {}",
                1.0 - x
            );
        }
    }
}

// =============================================================================
// 6. NaN/Inf Injection Tests
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Scalar activations must never return NaN for finite inputs.
    #[test]
    fn activation_no_nan_for_finite_inputs(x in activation_f32()) {
        let activations = [
            "RELU", "TANH", "LOGISTIC", "IDENTITY", "HARD_TANH", "ABSOLUTE",
            "SOFTSIGN", "STEP", "BIPOLAR", "RELU6", "LEAKYRELU", "ELU",
            "SELU", "GELU", "MISH", "SWISH", "ISRU", "SQUARE", "GAUSSIAN",
            "SOFTPLUS", "ARCTAN", "SINE", "COSINE", "BENT_IDENTITY",
            "BIPOLAR_SIGMOID", "COMPLEMENT",
        ];
        for name in &activations {
            if let Some(y) = apply_scalar_squash(name, x) {
                prop_assert!(
                    !y.is_nan(),
                    "{name}({x}) produced NaN"
                );
            }
        }
    }

    /// Confidence metrics must handle NaN/Inf in sample activations without panicking.
    #[test]
    fn confidence_nan_inf_samples_no_panic(
        finite_count in 2usize..20,
        gain in moderate_f32(),
    ) {
        let mut samples: Vec<HelpfulSample> = (0..finite_count)
            .map(|i| make_sample(i as f32 * 0.1, 0.05))
            .collect();
        // Inject NaN and Inf
        samples.push(make_sample(f32::NAN, 0.1));
        samples.push(make_sample(f32::INFINITY, 0.1));
        samples.push(make_sample(0.5, f32::NAN));
        samples.push(make_sample(0.5, f32::NEG_INFINITY));

        let metrics = compute_confidence_metrics(&samples, gain, None);
        // Should not panic, and confidence should be a valid number
        prop_assert!(
            metrics.prediction_confidence.is_finite() || metrics.prediction_confidence == 0.0,
            "Confidence {} should be finite or zero",
            metrics.prediction_confidence
        );
    }

    /// ErrorDistribution must handle NaN values in sample errors gracefully.
    #[test]
    fn error_distribution_nan_filtered(
        finite_count in 2usize..20,
    ) {
        let mut samples: Vec<HelpfulSample> = (0..finite_count)
            .map(|i| make_sample(0.5, i as f32 * 0.1))
            .collect();
        // Inject non-finite errors
        samples.push(make_sample(0.5, f32::NAN));
        samples.push(make_sample(0.5, f32::INFINITY));
        samples.push(make_sample(0.5, f32::NEG_INFINITY));

        let result = ErrorDistribution::from_samples(&samples);
        if let Some(dist) = result {
            prop_assert!(dist.mean.is_finite(), "Mean {} should be finite", dist.mean);
            prop_assert!(dist.variance >= 0.0, "Variance {} should be non-negative", dist.variance);
            prop_assert!(
                dist.sample_count == finite_count,
                "Should only count finite samples: {} vs expected {}",
                dist.sample_count, finite_count
            );
        }
    }

    /// Weight calculation must handle NaN/Inf inputs safely.
    #[test]
    fn optimal_weight_nan_inf_safe(
        kind in 0u8..4,
    ) {
        let (sum_ea, sum_aa) = match kind {
            0 => (f32::NAN, 1.0),
            1 => (1.0, f32::NAN),
            2 => (f32::INFINITY, 1.0),
            _ => (1.0, f32::INFINITY),
        };
        let result = calculate_optimal_outgoing_weight(sum_ea, sum_aa, 1.0);
        // Must not panic; if Some, must be finite
        if let Some(w) = result {
            prop_assert!(w.is_finite(), "Weight must be finite, got {w}");
        }
    }

    /// Cross-validation must handle NaN/Inf in samples without panicking.
    #[test]
    fn cross_validation_nan_inf_safe(
        finite_count in 80usize..150,
    ) {
        let mut samples: Vec<HelpfulSample> = (0..finite_count)
            .map(|i| make_sample(
                (i as f32 / 10.0).sin(),
                (i as f32 / 10.0).cos() * 0.1,
            ))
            .collect();
        // Inject non-finite samples
        samples.push(make_sample(f32::NAN, 0.1));
        samples.push(make_sample(0.5, f32::NAN));
        samples.push(make_sample(f32::INFINITY, 0.1));

        let config = CrossValidationConfig::default();
        let result = compute_cross_validation_score(&samples, &config);
        // Should not panic
        if let Some(cv) = result {
            prop_assert!(
                cv.brittleness_penalty.is_finite(),
                "Penalty should be finite"
            );
        }
    }

    /// EXPONENTIAL must not return Inf — it should clamp to JS_MAX_SAFE_INTEGER.
    #[test]
    fn exponential_no_overflow(x in -1000.0f32..1000.0) {
        if let Some(y) = apply_scalar_squash("EXPONENTIAL", x) {
            prop_assert!(
                !y.is_infinite(),
                "EXPONENTIAL({x}) overflowed to Inf"
            );
            prop_assert!(
                !y.is_nan(),
                "EXPONENTIAL({x}) produced NaN"
            );
        }
    }

    /// SOFTPLUS must not return Inf.
    #[test]
    fn softplus_no_overflow(x in -1000.0f32..1000.0) {
        if let Some(y) = apply_scalar_squash("SOFTPLUS", x) {
            prop_assert!(
                !y.is_infinite(),
                "SOFTPLUS({x}) overflowed to Inf"
            );
            prop_assert!(
                !y.is_nan(),
                "SOFTPLUS({x}) produced NaN"
            );
        }
    }

    /// STDINVERSE must not return Inf (epsilon protection near zero).
    #[test]
    fn stdinverse_no_inf_near_zero(x in -0.001f32..0.001) {
        if let Some(y) = apply_scalar_squash("STDINVERSE", x) {
            prop_assert!(
                y.is_finite(),
                "STDINVERSE({x}) should be finite (epsilon protected), got {y}"
            );
        }
    }

    /// SQRT must handle negative inputs gracefully (returns 0.0).
    #[test]
    fn sqrt_negative_safe(x in -1000.0f32..-0.001) {
        if let Some(y) = apply_scalar_squash("SQRT", x) {
            prop_assert_eq!(y, 0.0, "SQRT({}) should return 0.0 for negative input", x);
        }
    }
}

// =============================================================================
// 7. Fingerprinting Property Tests
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Fingerprinting must be deterministic: same creature produces same fingerprints.
    #[test]
    fn fingerprint_deterministic(
        weight in -1.0f32..1.0,
        bias in -1.0f32..1.0,
    ) {
        use common::{hidden_with_bias, make_creature, output, synapse};

        let creature = make_creature(
            vec![
                hidden_with_bias("h1", "TANH", bias),
                output("out", "IDENTITY"),
            ],
            vec![synapse("h1", "out", weight)],
        );

        let fp1 = compute_neuron_fingerprints(&creature);
        let fp2 = compute_neuron_fingerprints(&creature);

        prop_assert_eq!(
            fp1.get("h1"), fp2.get("h1"),
            "Fingerprints should be deterministic"
        );
        prop_assert_eq!(
            fp1.get("out"), fp2.get("out"),
            "Fingerprints should be deterministic"
        );
    }

    /// Changing the weight should change the fingerprint (sensitivity).
    #[test]
    fn fingerprint_sensitive_to_weight(
        weight1 in -1.0f32..0.0,
        weight2 in 0.001f32..1.0,
        bias in -1.0f32..1.0,
    ) {
        use common::{hidden_with_bias, make_creature, output, synapse};

        let creature1 = make_creature(
            vec![
                hidden_with_bias("h1", "TANH", bias),
                output("out", "IDENTITY"),
            ],
            vec![synapse("h1", "out", weight1)],
        );
        let creature2 = make_creature(
            vec![
                hidden_with_bias("h1", "TANH", bias),
                output("out", "IDENTITY"),
            ],
            vec![synapse("h1", "out", weight2)],
        );

        let fp1 = compute_neuron_fingerprints(&creature1);
        let fp2 = compute_neuron_fingerprints(&creature2);

        // Different weights should produce different fingerprints
        // (Not strictly guaranteed due to hash collisions, but extremely unlikely)
        prop_assert_ne!(
            fp1.get("h1"), fp2.get("h1"),
            "Different weights should produce different fingerprints (w1={}, w2={})", weight1, weight2
        );
    }

    /// Changing the bias should change the fingerprint (sensitivity).
    #[test]
    fn fingerprint_sensitive_to_bias(
        weight in -1.0f32..1.0,
        bias1 in -1.0f32..0.0,
        bias2 in 0.001f32..1.0,
    ) {
        use common::{hidden_with_bias, make_creature, output, synapse};

        let creature1 = make_creature(
            vec![
                hidden_with_bias("h1", "TANH", bias1),
                output("out", "IDENTITY"),
            ],
            vec![synapse("h1", "out", weight)],
        );
        let creature2 = make_creature(
            vec![
                hidden_with_bias("h1", "TANH", bias2),
                output("out", "IDENTITY"),
            ],
            vec![synapse("h1", "out", weight)],
        );

        let fp1 = compute_neuron_fingerprints(&creature1);
        let fp2 = compute_neuron_fingerprints(&creature2);

        prop_assert_ne!(
            fp1.get("h1"), fp2.get("h1"),
            "Different biases should produce different fingerprints (b1={}, b2={})", bias1, bias2
        );
    }
}

// =============================================================================
// 8. NaN-safe Comparison Property Tests
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// cmp_f32_desc must provide a total order (no panics on NaN).
    #[test]
    fn cmp_f32_desc_total_order(
        a in prop::num::f32::ANY,
        b in prop::num::f32::ANY,
    ) {
        // Must not panic
        let _ = cmp_f32_desc(&a, &b);
    }

    /// cmp_f32_asc must provide a total order (no panics on NaN).
    #[test]
    fn cmp_f32_asc_total_order(
        a in prop::num::f32::ANY,
        b in prop::num::f32::ANY,
    ) {
        let _ = cmp_f32_asc(&a, &b);
    }

    /// Sorting with cmp_f32_desc and cmp_f32_asc must not panic even with NaN/Inf values.
    #[test]
    fn sort_with_nan_no_panic(
        values in prop::collection::vec(prop::num::f32::ANY, 2..50),
    ) {
        // Descending sort must not panic
        let mut sorted_desc = values.clone();
        sorted_desc.sort_by(cmp_f32_desc);

        // Ascending sort must not panic
        let mut sorted_asc = values.clone();
        sorted_asc.sort_by(cmp_f32_asc);

        // Verify the sort is deterministic: sorting twice gives the same result
        let mut sorted_again = values.clone();
        sorted_again.sort_by(cmp_f32_asc);
        for (i, (a, b)) in sorted_asc.iter().zip(sorted_again.iter()).enumerate() {
            prop_assert_eq!(
                a.to_bits(), b.to_bits(),
                "Ascending sort not deterministic at index {}", i
            );
        }
    }
}
