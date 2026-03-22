//! Issue #906: Saturation-aware simulation fallback for non-linear activations.
//!
//! When `target_value` is missing but `target_activation` is available, the simulation
//! should use approximate inverse functions to recover `target_value` from `target_activation`
//! for common monotonic non-linear activations (TANH, LOGISTIC, SOFTSIGN, etc.).
//!
//! Previously, only `HARD_TANH`/CLIPPED received saturation-aware scoring in this fallback
//! path; all other non-linear activations fell back to the linear model which ignores
//! saturation effects entirely.

#![allow(clippy::doc_markdown)] // Test file with many activation name references in doc comments

use neat_ai_discovery::analysis::activation::{TargetSimulationMode, get_target_simulation_mode};
use neat_ai_discovery::analysis::samples::HelpfulSample;

// =============================================================================
// 1. Simulation mode selection: non-linear activations should use approximation
// =============================================================================

/// Helper: create samples with target_activation but no target_value.
fn samples_with_activation_only(target_activations: &[f32]) -> Vec<HelpfulSample> {
    target_activations
        .iter()
        .map(|&ta| HelpfulSample {
            activation: 0.5,
            avg_error: 0.1,
            target_value: None,
            target_activation: Some(ta),
        })
        .collect()
}

/// TANH should use ApproximateValueFromActivation when target_value is missing.
#[test]
fn tanh_uses_approximate_mode_when_target_value_missing() {
    let samples = samples_with_activation_only(&[0.46, -0.46, 0.76]);
    let mode = get_target_simulation_mode(&samples, Some("TANH"));
    assert!(
        matches!(
            mode,
            TargetSimulationMode::ApproximateValueFromActivation { .. }
        ),
        "TANH should use approximate inverse, not fall back to None"
    );
}

/// LOGISTIC (sigmoid) should use ApproximateValueFromActivation when target_value is missing.
#[test]
fn logistic_uses_approximate_mode_when_target_value_missing() {
    let samples = samples_with_activation_only(&[0.62, 0.73, 0.38]);
    let mode = get_target_simulation_mode(&samples, Some("LOGISTIC"));
    assert!(
        matches!(
            mode,
            TargetSimulationMode::ApproximateValueFromActivation { .. }
        ),
        "LOGISTIC should use approximate inverse, not fall back to None"
    );
}

/// SOFTSIGN should use ApproximateValueFromActivation when target_value is missing.
#[test]
fn softsign_uses_approximate_mode_when_target_value_missing() {
    let samples = samples_with_activation_only(&[0.33, -0.33, 0.5]);
    let mode = get_target_simulation_mode(&samples, Some("SOFTSIGN"));
    assert!(
        matches!(
            mode,
            TargetSimulationMode::ApproximateValueFromActivation { .. }
        ),
        "SOFTSIGN should use approximate inverse, not fall back to None"
    );
}

/// GELU should use ApproximateValueFromActivation when target_value is missing.
#[test]
fn gelu_uses_approximate_mode_when_target_value_missing() {
    let samples = samples_with_activation_only(&[0.84, -0.16, 0.5]);
    let mode = get_target_simulation_mode(&samples, Some("GELU"));
    assert!(
        matches!(
            mode,
            TargetSimulationMode::ApproximateValueFromActivation { .. }
        ),
        "GELU should use approximate inverse, not fall back to None"
    );
}

/// BIPOLAR_SIGMOID should use ApproximateValueFromActivation when target_value is missing.
#[test]
fn bipolar_sigmoid_uses_approximate_mode_when_target_value_missing() {
    let samples = samples_with_activation_only(&[0.46, -0.46, 0.76]);
    let mode = get_target_simulation_mode(&samples, Some("BIPOLAR_SIGMOID"));
    assert!(
        matches!(
            mode,
            TargetSimulationMode::ApproximateValueFromActivation { .. }
        ),
        "BIPOLAR_SIGMOID should use approximate inverse, not fall back to None"
    );
}

/// ELU should use ApproximateValueFromActivation when target_value is missing.
#[test]
fn elu_uses_approximate_mode_when_target_value_missing() {
    let samples = samples_with_activation_only(&[0.5, -0.39, 1.0]);
    let mode = get_target_simulation_mode(&samples, Some("ELU"));
    assert!(
        matches!(
            mode,
            TargetSimulationMode::ApproximateValueFromActivation { .. }
        ),
        "ELU should use approximate inverse, not fall back to None"
    );
}

/// HARD_TANH should still use ApproximateValueFromActivation (existing behaviour).
#[test]
fn hard_tanh_still_uses_approximate_mode() {
    let samples = samples_with_activation_only(&[0.5, -0.5, 0.9]);
    let mode = get_target_simulation_mode(&samples, Some("HARD_TANH"));
    assert!(
        matches!(
            mode,
            TargetSimulationMode::ApproximateValueFromActivation { .. }
        ),
        "HARD_TANH should still use approximate mode"
    );
}

/// Non-invertible activations (SINE, GAUSSIAN, SQUARE) should still fall back to None.
#[test]
fn non_invertible_activations_fall_back_to_none() {
    let samples = samples_with_activation_only(&[0.5, -0.5, 0.9]);

    let mode = get_target_simulation_mode(&samples, Some("SINE"));
    assert!(
        matches!(mode, TargetSimulationMode::None),
        "SINE is not monotonic, should fall back to None"
    );

    let mode = get_target_simulation_mode(&samples, Some("GAUSSIAN"));
    assert!(
        matches!(mode, TargetSimulationMode::None),
        "GAUSSIAN is not monotonic, should fall back to None"
    );

    let mode = get_target_simulation_mode(&samples, Some("SQUARE"));
    assert!(
        matches!(mode, TargetSimulationMode::None),
        "SQUARE is not monotonic, should fall back to None"
    );
}

/// When all samples have both target_value and target_activation, Full mode should be used.
#[test]
fn full_mode_still_used_when_target_value_present() {
    let samples = vec![
        HelpfulSample {
            activation: 0.5,
            avg_error: 0.1,
            target_value: Some(0.5),
            target_activation: Some(0.5_f32.tanh()),
        },
        HelpfulSample {
            activation: -0.3,
            avg_error: -0.05,
            target_value: Some(-0.3),
            target_activation: Some((-0.3_f32).tanh()),
        },
    ];
    let mode = get_target_simulation_mode(&samples, Some("TANH"));
    assert!(
        matches!(mode, TargetSimulationMode::Full(_)),
        "Should use Full mode when target_value is available"
    );
}

/// Verify that None squash still falls back to None mode.
#[test]
fn no_regression_none_squash_uses_linear() {
    let samples = vec![
        HelpfulSample {
            activation: 0.5,
            avg_error: 0.1,
            target_value: None,
            target_activation: None,
        },
        HelpfulSample {
            activation: -0.2,
            avg_error: -0.05,
            target_value: None,
            target_activation: None,
        },
    ];

    let mode = get_target_simulation_mode(&samples, None);
    assert!(matches!(mode, TargetSimulationMode::None));
}

// =============================================================================
// 2. Saturation-aware fallback through epistatic detection (end-to-end)
// =============================================================================

use neat_ai_discovery::analysis::recommendation::epistatic::{
    build_source_contribution, detect_epistatic_pairs,
};
use neat_ai_discovery::analysis::samples::HelpfulStats;

/// When a TANH target is near saturation and target_value is missing,
/// the approximate inverse should still recognise saturation and produce
/// more conservative estimates than the linear model.
#[test]
fn tanh_saturation_approximate_does_not_overestimate_vs_linear() {
    let n = 64;
    // TANH(2.0) ≈ 0.964 — near saturation. We only have target_activation.
    let target_activation = 2.0_f32.tanh();

    let samples_a: Vec<HelpfulSample> = (0..n)
        .map(|i| HelpfulSample {
            activation: if i < n / 2 { 1.0 } else { 0.0 },
            avg_error: 0.3,
            target_value: None,
            target_activation: Some(target_activation),
        })
        .collect();

    let samples_b: Vec<HelpfulSample> = (0..n)
        .map(|i| HelpfulSample {
            activation: if i >= n / 2 { 1.0 } else { 0.0 },
            avg_error: 0.3,
            target_value: None,
            target_activation: Some(target_activation),
        })
        .collect();

    // With TANH squash (should now use inverse approximation instead of linear)
    let contributions_tanh = vec![
        build_source_contribution(
            "src-a",
            samples_a.clone(),
            HelpfulStats::default(),
            1.5,
            0.0,
        ),
        build_source_contribution(
            "src-b",
            samples_b.clone(),
            HelpfulStats::default(),
            1.5,
            0.0,
        ),
    ];
    let pairs_tanh = detect_epistatic_pairs("output-0", &contributions_tanh, 1.0, Some("TANH"));

    // Without squash info (linear fallback)
    let contributions_linear = vec![
        build_source_contribution("src-a", samples_a, HelpfulStats::default(), 1.5, 0.0),
        build_source_contribution("src-b", samples_b, HelpfulStats::default(), 1.5, 0.0),
    ];
    let pairs_linear = detect_epistatic_pairs("output-0", &contributions_linear, 1.0, None);

    let max_gain_tanh = pairs_tanh
        .iter()
        .map(|p| p.combined_improvement)
        .fold(0.0f32, f32::max);
    let max_gain_linear = pairs_linear
        .iter()
        .map(|p| p.combined_improvement)
        .fold(0.0f32, f32::max);

    // TANH-aware should not exceed linear when pushing into saturation
    assert!(
        max_gain_tanh <= max_gain_linear + 0.001,
        "TANH-aware gain ({max_gain_tanh:.6}) should not exceed linear gain ({max_gain_linear:.6}) \
         when sources push into saturation and target_value is missing"
    );
}

/// LOGISTIC near saturation: approximate inverse should produce finite results
/// even when target_value is missing and activation is near the bounds.
#[test]
fn logistic_saturation_approximate_produces_finite_results() {
    let n = 64;
    let logistic = |x: f32| -> f32 {
        if x >= 0.0 {
            1.0 / (1.0 + (-x).exp())
        } else {
            let exp_x = x.exp();
            exp_x / (1.0 + exp_x)
        }
    };
    // LOGISTIC(4.0) ≈ 0.982 — near saturation
    let target_activation = logistic(4.0);

    let samples_a: Vec<HelpfulSample> = (0..n)
        .map(|i| HelpfulSample {
            activation: if i < n / 2 { 2.0 } else { 0.0 },
            avg_error: 0.1,
            target_value: None,
            target_activation: Some(target_activation),
        })
        .collect();
    let samples_b: Vec<HelpfulSample> = (0..n)
        .map(|i| HelpfulSample {
            activation: if i >= n / 2 { 2.0 } else { 0.0 },
            avg_error: 0.1,
            target_value: None,
            target_activation: Some(target_activation),
        })
        .collect();

    let contributions = vec![
        build_source_contribution("src-a", samples_a, HelpfulStats::default(), 3.0, 0.0),
        build_source_contribution("src-b", samples_b, HelpfulStats::default(), 3.0, 0.0),
    ];

    let pairs = detect_epistatic_pairs("output-0", &contributions, 1.0, Some("LOGISTIC"));

    for pair in &pairs {
        assert!(
            pair.combined_improvement.is_finite(),
            "LOGISTIC approximate improvement should be finite even near saturation: {pair:?}"
        );
    }
}

/// GELU with missing target_value should produce saturation-aware estimates,
/// not fall back to the linear model.
#[test]
fn gelu_approximate_produces_finite_results() {
    let n = 64;
    let gelu = |x: f32| -> f32 {
        let x3 = x * x * x;
        let tanh_arg = 0.797_884_6_f32 * (x + 0.044_715_f32 * x3);
        0.5 * x * (1.0 + tanh_arg.tanh())
    };
    let target_activation = gelu(2.0);

    let samples_a: Vec<HelpfulSample> = (0..n)
        .map(|i| HelpfulSample {
            activation: if i < n / 2 { 1.0 } else { 0.0 },
            avg_error: 0.2,
            target_value: None,
            target_activation: Some(target_activation),
        })
        .collect();
    let samples_b: Vec<HelpfulSample> = (0..n)
        .map(|i| HelpfulSample {
            activation: if i >= n / 2 { 1.0 } else { 0.0 },
            avg_error: 0.2,
            target_value: None,
            target_activation: Some(target_activation),
        })
        .collect();

    let contributions = vec![
        build_source_contribution("src-a", samples_a, HelpfulStats::default(), 1.0, 0.0),
        build_source_contribution("src-b", samples_b, HelpfulStats::default(), 1.0, 0.0),
    ];

    let pairs = detect_epistatic_pairs("output-0", &contributions, 1.0, Some("GELU"));

    for pair in &pairs {
        assert!(
            pair.combined_improvement.is_finite(),
            "GELU approximate improvement should be finite: {pair:?}"
        );
    }
}

// =============================================================================
// 3. Approximate vs Full mode: predictions should be comparable
// =============================================================================

/// For TANH, predictions with inverse approximation (no target_value) should be
/// in the same ballpark as Full mode (with target_value).
#[test]
fn tanh_approximate_improvement_comparable_to_full() {
    let n = 64;
    let target_value = 0.5_f32;
    let target_activation = target_value.tanh();

    // Full samples (have both target_value and target_activation)
    let full_a: Vec<HelpfulSample> = (0..n)
        .map(|i| HelpfulSample {
            activation: if i < n / 2 { 1.0 } else { 0.0 },
            avg_error: 0.3,
            target_value: Some(target_value),
            target_activation: Some(target_activation),
        })
        .collect();
    let full_b: Vec<HelpfulSample> = (0..n)
        .map(|i| HelpfulSample {
            activation: if i >= n / 2 { 1.0 } else { 0.0 },
            avg_error: 0.3,
            target_value: Some(target_value),
            target_activation: Some(target_activation),
        })
        .collect();

    // Approximate samples (target_activation only)
    let approx_a: Vec<HelpfulSample> = full_a
        .iter()
        .map(|s| HelpfulSample {
            target_value: None,
            ..*s
        })
        .collect();
    let approx_b: Vec<HelpfulSample> = full_b
        .iter()
        .map(|s| HelpfulSample {
            target_value: None,
            ..*s
        })
        .collect();

    let full_contributions = vec![
        build_source_contribution("src-a", full_a, HelpfulStats::default(), 0.5, 0.0),
        build_source_contribution("src-b", full_b, HelpfulStats::default(), 0.5, 0.0),
    ];
    let approx_contributions = vec![
        build_source_contribution("src-a", approx_a, HelpfulStats::default(), 0.5, 0.0),
        build_source_contribution("src-b", approx_b, HelpfulStats::default(), 0.5, 0.0),
    ];

    let pairs_full = detect_epistatic_pairs("output-0", &full_contributions, 1.0, Some("TANH"));
    let pairs_approx = detect_epistatic_pairs("output-0", &approx_contributions, 1.0, Some("TANH"));

    let gain_full = pairs_full
        .iter()
        .map(|p| p.combined_improvement)
        .fold(0.0f32, f32::max);
    let gain_approx = pairs_approx
        .iter()
        .map(|p| p.combined_improvement)
        .fold(0.0f32, f32::max);

    // Both should produce similar results — the inverse approximation for TANH
    // should recover the target_value accurately in the non-saturated region.
    let diff = (gain_full - gain_approx).abs();
    assert!(
        diff < 0.15,
        "TANH approximate ({gain_approx:.6}) should be close to full ({gain_full:.6}), \
         diff={diff:.6}"
    );
}
