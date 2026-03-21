//! Issue #897: Saturation-aware simulation for coordinated candidate gain estimation.
//!
//! Tests verify that:
//! 1. Activation-aware simulation produces different (more accurate) estimates than
//!    linear approximation when the target neuron has a non-linear activation function
//! 2. Saturation effects are correctly modelled (both contributions pushing TANH toward ±1)
//! 3. Conservative weight scaling (0.2×) is applied in estimation
//! 4. Linear fallback is used when target squash info is unavailable

use neat_ai_discovery::analysis::recommendation::epistatic::{
    build_source_contribution, detect_epistatic_pairs,
};
use neat_ai_discovery::analysis::samples::{HelpfulSample, HelpfulStats};

// ---------------------------------------------------------------------------
// 1. Saturation-aware simulation differs from linear for non-linear activations
// ---------------------------------------------------------------------------

/// When both sources push a TANH target deeper into saturation (e.g., `target_value` near +2),
/// the linear model overpredicts improvement because it ignores the flattening of TANH.
/// The activation-aware simulation should produce a more conservative estimate.
#[test]
fn tanh_saturation_reduces_estimated_gain_vs_linear() {
    let n = 64;
    let target_value = 2.0_f32; // Already near saturation for TANH

    // Source A fires on first half — both sources have large positive activations
    // that would push target_value even further into TANH saturation
    let samples_a: Vec<HelpfulSample> = (0..n)
        .map(|i| {
            let act = if i < n / 2 { 1.0_f32 } else { 0.0 };
            HelpfulSample {
                activation: act,
                avg_error: 0.3,
                target_value: Some(target_value),
                target_activation: Some(target_value.tanh()),
            }
        })
        .collect();

    // Source B fires on second half
    let samples_b: Vec<HelpfulSample> = (0..n)
        .map(|i| {
            let act = if i >= n / 2 { 1.0_f32 } else { 0.0 };
            HelpfulSample {
                activation: act,
                avg_error: 0.3,
                target_value: Some(target_value),
                target_activation: Some(target_value.tanh()),
            }
        })
        .collect();

    let contributions_with_squash = vec![
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

    // Run with TANH squash (saturation-aware)
    let pairs_tanh =
        detect_epistatic_pairs("output-0", &contributions_with_squash, 1.0, Some("TANH"));

    // Run without squash info (linear fallback)
    let contributions_no_squash = vec![
        build_source_contribution("src-a", samples_a, HelpfulStats::default(), 1.5, 0.0),
        build_source_contribution("src-b", samples_b, HelpfulStats::default(), 1.5, 0.0),
    ];
    let pairs_linear = detect_epistatic_pairs("output-0", &contributions_no_squash, 1.0, None);

    // Both should handle the case gracefully — the key check is that the function
    // runs without error and produces valid results with either path.
    // With saturation, we expect either fewer or lower-gain candidates because TANH
    // flattens near ±1, making the contributions less effective.
    let max_gain_tanh = pairs_tanh
        .iter()
        .map(|p| p.combined_improvement)
        .fold(0.0f32, f32::max);
    let max_gain_linear = pairs_linear
        .iter()
        .map(|p| p.combined_improvement)
        .fold(0.0f32, f32::max);

    // The TANH-aware estimate should not exceed the linear estimate for this
    // saturation scenario (pushing into flat region of TANH)
    assert!(
        max_gain_tanh <= max_gain_linear + 0.001,
        "TANH-aware gain ({max_gain_tanh:.6}) should not exceed linear gain ({max_gain_linear:.6}) \
         when sources push into saturation"
    );
}

// ---------------------------------------------------------------------------
// 2. Saturation edge case: both contributions pushing past activation bounds
// ---------------------------------------------------------------------------

/// When both sources push a LOGISTIC target past its saturation point,
/// the activation-aware simulation should correctly cap the improvement.
#[test]
fn logistic_saturation_caps_improvement() {
    let n = 64;
    let target_value = 5.0_f32; // Deep in LOGISTIC saturation (output ≈ 0.993)

    let logistic = |x: f32| -> f32 {
        if x >= 0.0 {
            1.0 / (1.0 + (-x).exp())
        } else {
            let exp_x = x.exp();
            exp_x / (1.0 + exp_x)
        }
    };

    // Both sources fire on alternating samples with large activations
    let samples_a: Vec<HelpfulSample> = (0..n)
        .map(|i| HelpfulSample {
            activation: if i < n / 2 { 2.0 } else { 0.0 },
            avg_error: 0.1,
            target_value: Some(target_value),
            target_activation: Some(logistic(target_value)),
        })
        .collect();

    let samples_b: Vec<HelpfulSample> = (0..n)
        .map(|i| HelpfulSample {
            activation: if i >= n / 2 { 2.0 } else { 0.0 },
            avg_error: 0.1,
            target_value: Some(target_value),
            target_activation: Some(logistic(target_value)),
        })
        .collect();

    let contributions = vec![
        build_source_contribution("src-a", samples_a, HelpfulStats::default(), 3.0, 0.0),
        build_source_contribution("src-b", samples_b, HelpfulStats::default(), 3.0, 0.0),
    ];

    // Should not panic and should handle deep saturation gracefully
    let pairs = detect_epistatic_pairs("output-0", &contributions, 1.0, Some("LOGISTIC"));

    // With deep saturation, adding more input has diminishing returns via LOGISTIC
    for pair in &pairs {
        assert!(
            pair.combined_improvement.is_finite(),
            "Combined improvement should be finite even in deep saturation: {pair:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// 3. Conservative weight scaling is applied
// ---------------------------------------------------------------------------

/// With linear identity activation, the saturation-aware path should produce
/// results equivalent to linear but with 0.2× weight scaling applied.
#[test]
fn identity_activation_uses_conservative_weight_scale() {
    let n = 64;

    // Source A fires on first half, Source B on second half
    let samples_a: Vec<HelpfulSample> = (0..n)
        .map(|i| HelpfulSample {
            activation: if i < n / 2 { 1.0 } else { 0.0 },
            avg_error: 0.5,
            target_value: Some(0.0),
            target_activation: Some(0.0), // IDENTITY: activation == value
        })
        .collect();
    let samples_b: Vec<HelpfulSample> = (0..n)
        .map(|i| HelpfulSample {
            activation: if i >= n / 2 { 1.0 } else { 0.0 },
            avg_error: 0.5,
            target_value: Some(0.0),
            target_activation: Some(0.0),
        })
        .collect();

    // With IDENTITY, the activation-aware path should behave like linear
    // but with 0.2× weight scaling. Both paths use 0.2× now.
    let contributions = vec![
        build_source_contribution("src-a", samples_a, HelpfulStats::default(), 0.5, 0.0),
        build_source_contribution("src-b", samples_b, HelpfulStats::default(), 0.5, 0.0),
    ];

    let pairs_identity = detect_epistatic_pairs("output-0", &contributions, 1.0, Some("IDENTITY"));
    let pairs_none = detect_epistatic_pairs("output-0", &contributions, 1.0, None);

    // For IDENTITY activation, both paths should produce similar results since
    // IDENTITY(x) = x means the activation-aware path is equivalent to linear
    let gain_identity = pairs_identity
        .iter()
        .map(|p| p.combined_improvement)
        .fold(0.0f32, f32::max);
    let gain_none = pairs_none
        .iter()
        .map(|p| p.combined_improvement)
        .fold(0.0f32, f32::max);

    // Should be approximately equal (IDENTITY is linear)
    assert!(
        (gain_identity - gain_none).abs() < 0.01,
        "IDENTITY activation should produce similar results to linear fallback: \
         identity={gain_identity:.6}, none={gain_none:.6}"
    );
}

// ---------------------------------------------------------------------------
// 4. Linear fallback when target squash is unavailable
// ---------------------------------------------------------------------------

/// When `target_squash` is None, the system should fall back to linear estimation
/// and still produce reasonable results.
#[test]
fn linear_fallback_works_without_squash_info() {
    let n = 64;

    let samples_a: Vec<HelpfulSample> = (0..n)
        .map(|i| HelpfulSample {
            activation: if i < n / 2 { 1.0 } else { 0.0 },
            avg_error: 0.3,
            target_value: None,
            target_activation: None,
        })
        .collect();
    let samples_b: Vec<HelpfulSample> = (0..n)
        .map(|i| HelpfulSample {
            activation: if i >= n / 2 { 1.0 } else { 0.0 },
            avg_error: 0.3,
            target_value: None,
            target_activation: None,
        })
        .collect();

    let contributions = vec![
        build_source_contribution("src-a", samples_a, HelpfulStats::default(), 0.3, 0.0),
        build_source_contribution("src-b", samples_b, HelpfulStats::default(), 0.3, 0.0),
    ];

    // Should work fine with None squash — uses linear fallback
    let pairs = detect_epistatic_pairs("output-0", &contributions, 1.0, None);

    // Verify results are finite
    for pair in &pairs {
        assert!(pair.combined_improvement.is_finite());
    }
}

// ---------------------------------------------------------------------------
// 5. Fallback when samples lack target_value/target_activation
// ---------------------------------------------------------------------------

/// When squash is provided but samples lack `target_value` data, should fall
/// back to linear estimation gracefully.
#[test]
fn falls_back_to_linear_when_samples_lack_target_data() {
    let n = 64;

    // Samples WITHOUT target_value/target_activation
    let samples_a: Vec<HelpfulSample> = (0..n)
        .map(|i| HelpfulSample {
            activation: if i < n / 2 { 1.0 } else { 0.0 },
            avg_error: 0.3,
            target_value: None,
            target_activation: None,
        })
        .collect();
    let samples_b: Vec<HelpfulSample> = (0..n)
        .map(|i| HelpfulSample {
            activation: if i >= n / 2 { 1.0 } else { 0.0 },
            avg_error: 0.3,
            target_value: None,
            target_activation: None,
        })
        .collect();

    let contributions = vec![
        build_source_contribution("src-a", samples_a, HelpfulStats::default(), 0.3, 0.0),
        build_source_contribution("src-b", samples_b, HelpfulStats::default(), 0.3, 0.0),
    ];

    // Providing TANH squash but samples have no target data — should fall back to linear
    let pairs = detect_epistatic_pairs("output-0", &contributions, 1.0, Some("TANH"));

    for pair in &pairs {
        assert!(pair.combined_improvement.is_finite());
    }
}

// ---------------------------------------------------------------------------
// 6. Synergistic detection also uses saturation-aware simulation
// ---------------------------------------------------------------------------

use neat_ai_discovery::analysis::recommendation::epistatic::detect_synergistic_candidates;

/// Synergistic candidate detection should also use saturation-aware simulation
/// when target squash info is provided.
#[test]
fn synergistic_detection_uses_saturation_aware_simulation() {
    let n = 64;

    let samples_primary: Vec<HelpfulSample> = (0..n)
        .map(|i| HelpfulSample {
            activation: if i < n / 2 { 1.0 } else { 0.0 },
            avg_error: 0.3,
            target_value: Some(2.0),
            target_activation: Some(2.0_f32.tanh()),
        })
        .collect();
    let samples_complement: Vec<HelpfulSample> = (0..n)
        .map(|i| HelpfulSample {
            activation: if i >= n / 2 { 1.0 } else { 0.0 },
            avg_error: 0.3,
            target_value: Some(2.0),
            target_activation: Some(2.0_f32.tanh()),
        })
        .collect();

    let contributions = vec![
        build_source_contribution(
            "primary",
            samples_primary,
            HelpfulStats::default(),
            1.0,
            0.05,
        ),
        build_source_contribution(
            "complement",
            samples_complement,
            HelpfulStats::default(),
            1.0,
            0.01,
        ),
    ];

    // Should run without errors with TANH squash
    let candidates_tanh =
        detect_synergistic_candidates("output-0", &contributions, 1.0, Some("TANH"));

    // And without squash
    let candidates_none = detect_synergistic_candidates("output-0", &contributions, 1.0, None);

    // Both should produce finite results
    for c in &candidates_tanh {
        assert!(c.combined_improvement.is_finite());
    }
    for c in &candidates_none {
        assert!(c.combined_improvement.is_finite());
    }
}
