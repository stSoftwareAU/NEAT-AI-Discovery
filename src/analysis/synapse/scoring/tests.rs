//! Unit tests for the scoring improvement module.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)] // Test sample counts are small
use super::improvement::{
    compute_activation_improvement_and_count, compute_relu_improvement_and_count,
    compute_synapse_improvement_and_count,
};
use crate::analysis::samples::HelpfulSample;

/// Issue #940: Verify `ReLU` improvement handles None `target_value` gracefully
/// when `target_activation_fn` is provided (previously would panic via unwrap).
#[test]
fn test_relu_improvement_skips_samples_with_none_target_value() {
    let samples = vec![
        HelpfulSample {
            activation: 0.5,
            avg_error: 0.1,
            target_value: None,
            target_activation: Some(0.4),
        },
        HelpfulSample {
            activation: 0.3,
            avg_error: 0.2,
            target_value: None,
            target_activation: Some(0.3),
        },
    ];

    let (improvement, _improved, total, _) =
        compute_relu_improvement_and_count(&samples, 1.0, 1.0, 0.0, 1.0, Some(|x: f32| x.tanh()));

    assert!(
        improvement.is_finite(),
        "improvement should be finite, not NaN/Inf"
    );
    assert_eq!(total, samples.len() as u32);
}

/// Issue #940: Verify activation improvement handles None `target_activation`
/// gracefully when `target_activation_fn` is provided.
#[test]
fn test_activation_improvement_skips_samples_with_none_target_activation() {
    let samples = vec![HelpfulSample {
        activation: 0.5,
        avg_error: 0.1,
        target_value: Some(0.3),
        target_activation: None,
    }];

    let (improvement, _improved, total, _) = compute_activation_improvement_and_count(
        &samples,
        1.0,
        1.0,
        0.0,
        |x: f32| x.max(0.0),
        1.0,
        Some(|x: f32| x.tanh()),
    );

    assert!(improvement.is_finite());
    assert_eq!(total, samples.len() as u32);
}

/// Issue #940: Verify synapse improvement handles mixed None/Some target data
/// without panicking.
#[test]
fn test_synapse_improvement_handles_mixed_none_target_data() {
    let samples = vec![
        HelpfulSample {
            activation: 0.5,
            avg_error: 0.1,
            target_value: None,
            target_activation: Some(0.4),
        },
        HelpfulSample {
            activation: 0.3,
            avg_error: 0.2,
            target_value: Some(0.4),
            target_activation: None,
        },
    ];

    // Should not panic — the TargetSimulationMode checks require all samples
    // to have target data, so it falls back to linear mode.
    let (improvement, _improved, _worsened, total, _) =
        compute_synapse_improvement_and_count(&samples, 0.5, 1.0, Some("HARD_TANH"));

    assert!(improvement.is_finite());
    assert_eq!(total, samples.len() as u32);
}

/// Issue #940: Verify functions still produce correct results with complete data.
#[test]
fn test_relu_improvement_correct_with_complete_data() {
    let samples = vec![
        HelpfulSample {
            activation: 0.8,
            avg_error: 0.5,
            target_value: Some(0.3),
            target_activation: Some(0.29),
        },
        HelpfulSample {
            activation: 0.2,
            avg_error: -0.3,
            target_value: Some(0.6),
            target_activation: Some(0.54),
        },
    ];

    let (improvement, _improved, total, _) =
        compute_relu_improvement_and_count(&samples, 1.0, 0.5, 0.0, 0.5, Some(|x: f32| x.tanh()));

    assert!(improvement.is_finite());
    assert_eq!(total, 2);
}

// =============================================================================
// Issue #1075: Branchless variant correctness tests
// =============================================================================

/// Helper to build samples with some non-finite errors (NaN/Inf).
fn build_samples_with_non_finite(count: usize) -> Vec<HelpfulSample> {
    (0..count)
        .map(|i| {
            let phase = i as f32 * 0.1;
            let avg_error = if i % 7 == 0 {
                f32::NAN
            } else if i % 11 == 0 {
                f32::INFINITY
            } else {
                (phase * 0.7).cos() * 0.3
            };
            HelpfulSample {
                activation: phase.sin() * 2.0,
                avg_error,
                target_value: Some(phase.cos()),
                target_activation: Some(phase.cos().tanh()),
            }
        })
        .collect()
}

/// Issue #1075: `ReLU` no-target path produces finite, reasonable results
/// with samples containing non-finite error values.
#[test]
fn test_relu_no_target_branchless_handles_non_finite() {
    let samples = build_samples_with_non_finite(100);
    let baseline_sq: f32 = samples
        .iter()
        .filter(|s| s.avg_error.is_finite())
        .map(|s| s.avg_error * s.avg_error)
        .sum();

    let (improvement, improved, total, _) =
        compute_relu_improvement_and_count(&samples, 0.5, 0.3, 0.1, baseline_sq, None);

    assert!(improvement.is_finite(), "improvement must be finite");
    assert_eq!(total, 100);
    assert!(improved <= total, "improved cannot exceed total");
}

/// Issue #1075: `ReLU` with-target path produces same results as no-target
/// when target function is identity (f(x) = x).
#[test]
fn test_relu_with_target_identity_matches_no_target() {
    // Use samples where all target_value equals avg_error baseline
    let samples: Vec<HelpfulSample> = (0..50)
        .map(|i| {
            let phase = i as f32 * 0.1;
            let activation = phase.sin() * 2.0;
            let error = (phase * 0.7).cos() * 0.3;
            // target_value = 0, target_activation = 0, so:
            //   desired_value = 0 + error = error
            //   expected = identity(error) = error
            //   baseline_err = error - 0 = error  (matches no-target)
            //   new_err = error - identity(contribution) = error - contribution (matches)
            HelpfulSample {
                activation,
                avg_error: error,
                target_value: Some(0.0),
                target_activation: Some(0.0),
            }
        })
        .collect();

    let baseline_sq: f32 = samples.iter().map(|s| s.avg_error * s.avg_error).sum();

    let (imp_no_target, improved_no, _, _) =
        compute_relu_improvement_and_count(&samples, 0.5, 0.3, 0.0, baseline_sq, None);

    let (imp_with_target, improved_with, _, _) = compute_relu_improvement_and_count(
        &samples,
        0.5,
        0.3,
        0.0,
        baseline_sq,
        Some(|x: f32| x), // identity function
    );

    assert!(
        (imp_no_target - imp_with_target).abs() < 1e-5,
        "identity target should match no-target: {imp_no_target} vs {imp_with_target}"
    );
    assert_eq!(improved_no, improved_with);
}

/// Issue #1075: Synapse no-target branchless path produces finite results
/// with non-finite error samples.
#[test]
fn test_synapse_no_target_branchless_handles_non_finite() {
    let samples = build_samples_with_non_finite(100);
    let baseline_sq: f32 = samples
        .iter()
        .filter(|s| s.avg_error.is_finite())
        .map(|s| s.avg_error * s.avg_error)
        .sum();

    let (improvement, improved, worsened, total, _) =
        compute_synapse_improvement_and_count(&samples, 0.35, baseline_sq, None);

    assert!(improvement.is_finite(), "improvement must be finite");
    assert_eq!(total, 100);
    assert!(
        improved + worsened <= total,
        "improved + worsened cannot exceed total"
    );
}

/// Issue #1075: Activation no-target branchless path produces correct results.
#[test]
fn test_activation_no_target_branchless_handles_non_finite() {
    let samples = build_samples_with_non_finite(100);
    let baseline_sq: f32 = samples
        .iter()
        .filter(|s| s.avg_error.is_finite())
        .map(|s| s.avg_error * s.avg_error)
        .sum();

    let (improvement, improved, total, _) = compute_activation_improvement_and_count(
        &samples,
        0.4,
        0.6,
        -0.2,
        |x: f32| x.tanh(),
        baseline_sq,
        None,
    );

    assert!(improvement.is_finite(), "improvement must be finite");
    assert_eq!(total, 100);
    assert!(improved <= total);
}

/// Issue #1075: All three functions return correct zero-improvement for empty samples.
#[test]
fn test_branchless_variants_empty_samples() {
    let empty: Vec<HelpfulSample> = vec![];

    let (imp, _, total, _) = compute_relu_improvement_and_count(&empty, 1.0, 1.0, 0.0, 1.0, None);
    assert_eq!(imp, 0.0);
    assert_eq!(total, 0);

    let (imp, _, _, total, _) = compute_synapse_improvement_and_count(&empty, 0.5, 1.0, None);
    assert_eq!(imp, 0.0);
    assert_eq!(total, 0);

    let (imp, _, total, _) = compute_activation_improvement_and_count(
        &empty,
        1.0,
        1.0,
        0.0,
        |x: f32| x.tanh(),
        1.0,
        None,
    );
    assert_eq!(imp, 0.0);
    assert_eq!(total, 0);
}

/// Issue #1075: Positive weight with positive errors should show improvement.
#[test]
fn test_synapse_improvement_positive_weight_reduces_positive_error() {
    let samples = vec![
        HelpfulSample {
            activation: 1.0,
            avg_error: 0.5,
            target_value: None,
            target_activation: None,
        },
        HelpfulSample {
            activation: 0.8,
            avg_error: 0.4,
            target_value: None,
            target_activation: None,
        },
    ];
    let baseline_sq: f32 = samples.iter().map(|s| s.avg_error * s.avg_error).sum();

    let (improvement, improved, _, total, _) =
        compute_synapse_improvement_and_count(&samples, 0.3, baseline_sq, None);

    assert!(improvement > 0.0, "should show positive improvement");
    assert!(improved > 0, "should have improved samples");
    assert_eq!(total, 2);
}

// =============================================================================
// Issue #1161: magnitude-weighted improvement metric tests
// =============================================================================

use super::discounting::{apply_neuron_pessimism_discount, apply_synapse_pessimism_discount};

/// Build samples with a fixed-magnitude baseline error and a fixed-magnitude
/// new error per sample. Used by the noise/substantial/mixed scenarios below.
///
/// Each sample has `avg_error = baseline`. We then drive the new error via a
/// straight-through linear synapse (`weight = 1`, no target activation) so
/// that `new_error = baseline - activation`. Choosing `activation = baseline -
/// new_error` for each sample produces the desired post-candidate error.
fn samples_with_errors(pairs: &[(f32, f32)]) -> Vec<HelpfulSample> {
    pairs
        .iter()
        .map(|&(baseline, new_err)| HelpfulSample {
            activation: baseline - new_err,
            avg_error: baseline,
            target_value: None,
            target_activation: None,
        })
        .collect()
}

/// All-noise improvements: 76% of samples improve, but only by ≈ 0% of the
/// baseline magnitude. Magnitude ratio must be near zero, and the resulting
/// neuron pessimism discount should kill almost all of the input gain
/// (Issue #1160 failure pattern).
#[test]
fn test_magnitude_ratio_noise_level_improvements_collapse_neuron_gain() {
    // 100 samples baseline = 1.0. 76 improve by 1e-6, 24 stay flat.
    let mut pairs = Vec::with_capacity(100);
    for _ in 0..76 {
        pairs.push((1.0_f32, 1.0_f32 - 1e-6));
    }
    for _ in 0..24 {
        pairs.push((1.0_f32, 1.0_f32));
    }
    let samples = samples_with_errors(&pairs);
    let baseline_sq: f32 = samples.iter().map(|s| s.avg_error * s.avg_error).sum();

    let (_imp, improved, _worsened, total, magnitude_ratio) = compute_synapse_improvement_and_count(
        &samples,
        1.0, // weight that produces new_error = avg_error - activation
        baseline_sq,
        None,
    );

    assert_eq!(total, 100);
    assert_eq!(improved, 76, "76% binary improvement");
    assert!(
        magnitude_ratio < 1e-3,
        "magnitude ratio should be near zero, got {magnitude_ratio}"
    );

    // The combined discount must collapse the input gain to <1% — see issue
    // body acceptance criterion.
    let raw_gain = 0.003_f32;
    let discounted =
        apply_neuron_pessimism_discount(raw_gain, improved, total, Some(magnitude_ratio));
    assert!(
        discounted.abs() < 0.01 * raw_gain.abs(),
        "discounted gain ({discounted}) should be <1% of input ({raw_gain})"
    );
    let synapse_discounted =
        apply_synapse_pessimism_discount(raw_gain, improved, total, Some(magnitude_ratio));
    assert!(
        synapse_discounted.abs() < 0.01 * raw_gain.abs(),
        "synapse-discounted gain ({synapse_discounted}) should be <1% of input ({raw_gain})"
    );
}

/// All-substantial improvements: every sample improves by half its baseline
/// magnitude. Both binary and magnitude ratios must be high and the discount
/// should preserve most of the gain.
#[test]
fn test_magnitude_ratio_substantial_improvements_preserve_gain() {
    let pairs: Vec<(f32, f32)> = (0..100).map(|_| (1.0_f32, 0.5_f32)).collect();
    let samples = samples_with_errors(&pairs);
    let baseline_sq: f32 = samples.iter().map(|s| s.avg_error * s.avg_error).sum();

    let (_imp, improved, _worsened, total, magnitude_ratio) =
        compute_synapse_improvement_and_count(&samples, 1.0, baseline_sq, None);

    assert_eq!(total, 100);
    assert_eq!(improved, 100, "all samples improve");
    assert!(
        (magnitude_ratio - 0.5).abs() < 1e-3,
        "magnitude ratio should be ≈ 0.5, got {magnitude_ratio}"
    );

    // Compare with the legacy (binary-only) discount: the magnitude-aware
    // discount should be no smaller than 50% of the binary-only discount,
    // i.e. it must not unfairly penalise candidates whose magnitude and
    // binary signals agree at a moderate-to-high level.
    let raw_gain = 1.0_f32;
    let combined =
        apply_neuron_pessimism_discount(raw_gain, improved, total, Some(magnitude_ratio));
    let binary_only = apply_neuron_pessimism_discount(raw_gain, improved, total, None);
    assert!(
        combined > 0.5 * binary_only,
        "combined ({combined}) should retain >50% of binary-only ({binary_only})"
    );
}

/// Mixed improvements: half the samples improve substantially, half by
/// noise. The combined discount should fall between the noise-only and the
/// substantial-only scenarios.
#[test]
fn test_magnitude_ratio_mixed_improvements_intermediate_discount() {
    let mut pairs = Vec::with_capacity(100);
    for _ in 0..50 {
        pairs.push((1.0_f32, 0.5_f32)); // substantial
    }
    for _ in 0..50 {
        pairs.push((1.0_f32, 1.0_f32 - 1e-6)); // noise
    }
    let samples = samples_with_errors(&pairs);
    let baseline_sq: f32 = samples.iter().map(|s| s.avg_error * s.avg_error).sum();

    let (_imp, improved, _worsened, total, magnitude_ratio) =
        compute_synapse_improvement_and_count(&samples, 1.0, baseline_sq, None);

    assert_eq!(total, 100);
    assert_eq!(improved, 100);
    // Half-substantial, half-noise → magnitude ≈ 0.25.
    assert!(
        (magnitude_ratio - 0.25).abs() < 0.01,
        "magnitude ratio should be ≈ 0.25, got {magnitude_ratio}"
    );

    let raw_gain = 1.0_f32;
    let mixed = apply_neuron_pessimism_discount(raw_gain, improved, total, Some(magnitude_ratio));
    let substantial = apply_neuron_pessimism_discount(raw_gain, improved, total, Some(0.5_f32));
    let noise = apply_neuron_pessimism_discount(raw_gain, improved, total, Some(1e-4_f32));
    assert!(
        mixed > noise && mixed < substantial,
        "mixed ({mixed}) should fall between noise ({noise}) and substantial ({substantial})"
    );
}

/// `None` magnitude ratio preserves the legacy (binary-only) behaviour: the
/// legacy floor is not scaled, and the geometric-mean term reduces to
/// `improved_ratio` so the resulting discount must equal the discount produced
/// by the original 3-arg formula.
#[test]
fn test_magnitude_ratio_none_falls_back_to_binary_only() {
    use crate::analysis::constants::{
        NEURON_PESSIMISM_CURVE_EXPONENT, NEURON_PESSIMISM_DISCOUNT_FLOOR,
    };

    let raw_gain = 1.0_f32;
    let with_none = apply_neuron_pessimism_discount(raw_gain, 70, 100, None);
    // Recompute the legacy formula directly so the assertion is independent of
    // any future refactor of the production function.
    let legacy_ratio = (0.7_f32).powf(NEURON_PESSIMISM_CURVE_EXPONENT);
    let legacy =
        NEURON_PESSIMISM_DISCOUNT_FLOOR + (1.0 - NEURON_PESSIMISM_DISCOUNT_FLOOR) * legacy_ratio;
    assert!(
        (with_none - legacy).abs() < 1e-5,
        "None must preserve legacy formula: {with_none} vs {legacy}"
    );
}

/// Magnitude ratio is clamped to `[0, 1]`. Negative or NaN ratios must not
/// produce non-finite discounts.
#[test]
fn test_magnitude_ratio_invalid_inputs_remain_finite() {
    let raw_gain = 1.0_f32;
    let neg = apply_neuron_pessimism_discount(raw_gain, 50, 100, Some(-0.5_f32));
    let nan = apply_neuron_pessimism_discount(raw_gain, 50, 100, Some(f32::NAN));
    assert!(
        neg.is_finite(),
        "negative magnitude must yield a finite discount"
    );
    assert!(
        nan.is_finite(),
        "NaN magnitude must yield a finite discount"
    );
}
