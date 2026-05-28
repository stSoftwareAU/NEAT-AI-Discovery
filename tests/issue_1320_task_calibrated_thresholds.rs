//! Tests for cost-aware early-termination / drought thresholds (Issue #1320).
//!
//! Verifies that the SPRT early-termination config and the drought-diagnostic
//! threshold can be calibrated from a [`TaskDescriptor`]. Classification tasks
//! (one-hot, simplex, margin) get more lenient (later-firing) thresholds so
//! per-sample sparse improvements are not misread as "drought" / "no
//! improvement", while regression and unknown / OTHER descriptors retain the
//! current defaults (regression guard).

use neat_ai_discovery::analysis::drought_diagnostic::drought_threshold_for_task;
use neat_ai_discovery::analysis::early_termination::EarlyTerminationConfig;
use neat_ai_discovery::analysis::task_descriptor::TaskDescriptor;

// =============================================================================
// EarlyTerminationConfig::for_task
// =============================================================================

#[test]
fn early_termination_for_neutral_descriptor_matches_default() {
    // Regression guard: OTHER / Unknown / absent should preserve the current
    // pre-Issue-#1320 defaults exactly.
    let default = EarlyTerminationConfig::default();
    let calibrated = EarlyTerminationConfig::for_task(&TaskDescriptor::neutral());

    assert_eq!(calibrated.enabled, default.enabled);
    assert_eq!(calibrated.alpha, default.alpha);
    assert_eq!(calibrated.beta, default.beta);
    assert_eq!(calibrated.threshold, default.threshold);
    assert_eq!(calibrated.min_samples, default.min_samples);
    assert_eq!(calibrated.check_interval, default.check_interval);
}

#[test]
fn early_termination_for_other_cost_matches_default() {
    let default = EarlyTerminationConfig::default();
    let calibrated = EarlyTerminationConfig::for_task(&TaskDescriptor::from_name("OTHER", 4));
    assert_eq!(calibrated.min_samples, default.min_samples);
    assert_eq!(calibrated.threshold, default.threshold);
}

#[test]
fn early_termination_for_unrecognised_cost_matches_default() {
    let default = EarlyTerminationConfig::default();
    let calibrated = EarlyTerminationConfig::for_task(&TaskDescriptor::from_name("EXOTIC_LOSS", 3));
    assert_eq!(calibrated.min_samples, default.min_samples);
    assert_eq!(calibrated.threshold, default.threshold);
}

#[test]
fn early_termination_for_mse_matches_default_regression_guard() {
    // MSE is the canonical unbounded regression cost — current thresholds are
    // already calibrated to it, so for_task should not move them.
    let default = EarlyTerminationConfig::default();
    let calibrated = EarlyTerminationConfig::for_task(&TaskDescriptor::from_name("MSE", 1));
    assert_eq!(calibrated.min_samples, default.min_samples);
    assert_eq!(calibrated.threshold, default.threshold);
}

#[test]
fn early_termination_for_categorical_error_requires_more_evidence() {
    // CATEGORICAL_ERROR is binary-ish per sample — a 0.1 error already implies
    // ~90% accuracy. Per-sample improvement signal is sparse, so we must
    // require more samples and tighter SPRT before deciding.
    let default = EarlyTerminationConfig::default();
    let calibrated =
        EarlyTerminationConfig::for_task(&TaskDescriptor::from_name("CATEGORICAL_ERROR", 10));
    assert!(
        calibrated.min_samples > default.min_samples,
        "classification should require more samples (got {}, default {})",
        calibrated.min_samples,
        default.min_samples,
    );
    assert!(
        calibrated.threshold > default.threshold,
        "classification should require a stricter SPRT threshold (got {}, default {})",
        calibrated.threshold,
        default.threshold,
    );
}

#[test]
fn early_termination_for_cross_entropy_matches_classification_profile() {
    // CROSS_ENTROPY → Simplex topology → classification profile.
    let cross = EarlyTerminationConfig::for_task(&TaskDescriptor::from_name("CROSS_ENTROPY", 5));
    let cat = EarlyTerminationConfig::for_task(&TaskDescriptor::from_name("CATEGORICAL_ERROR", 5));
    assert_eq!(cross.min_samples, cat.min_samples);
    assert_eq!(cross.threshold, cat.threshold);
}

#[test]
fn early_termination_for_hinge_matches_classification_profile() {
    let hinge = EarlyTerminationConfig::for_task(&TaskDescriptor::from_name("HINGE", 1));
    let cat = EarlyTerminationConfig::for_task(&TaskDescriptor::from_name("CATEGORICAL_ERROR", 5));
    assert_eq!(hinge.min_samples, cat.min_samples);
    assert_eq!(hinge.threshold, cat.threshold);
}

#[test]
fn early_termination_evaluator_inherits_calibrated_min_samples() {
    // The evaluator built from a calibrated config should pick up the larger
    // min_samples count; verify the wiring from the field to the evaluator.
    let calibrated =
        EarlyTerminationConfig::for_task(&TaskDescriptor::from_name("CATEGORICAL_ERROR", 5));
    let evaluator = calibrated.create_evaluator();
    // The evaluator exposes its decision; with 30 strongly-positive samples it
    // would normally trip is_strongly_beneficial at 30, but classification
    // needs more — the calibrated config bumps min_samples past 30 so the
    // heuristic short-circuit must wait.
    let mut e = evaluator;
    e.add_batch(28, 2);
    assert!(
        !e.is_strongly_beneficial(),
        "calibrated min_samples must gate is_strongly_beneficial above 30",
    );
}

// =============================================================================
// drought_threshold_for_task
// =============================================================================

#[test]
fn drought_threshold_for_neutral_is_unchanged() {
    // Regression guard: OTHER / Unknown / absent ⇒ base threshold.
    assert_eq!(drought_threshold_for_task(5, &TaskDescriptor::neutral()), 5);
    assert_eq!(drought_threshold_for_task(7, &TaskDescriptor::default()), 7);
}

#[test]
fn drought_threshold_for_other_is_unchanged() {
    assert_eq!(
        drought_threshold_for_task(5, &TaskDescriptor::from_name("OTHER", 4)),
        5,
    );
}

#[test]
fn drought_threshold_for_unrecognised_is_unchanged() {
    assert_eq!(
        drought_threshold_for_task(5, &TaskDescriptor::from_name("EXOTIC_LOSS", 4)),
        5,
    );
}

#[test]
fn drought_threshold_for_mse_is_unchanged() {
    assert_eq!(
        drought_threshold_for_task(5, &TaskDescriptor::from_name("MSE", 1)),
        5,
    );
    assert_eq!(
        drought_threshold_for_task(5, &TaskDescriptor::from_name("MAE", 1)),
        5,
    );
}

#[test]
fn drought_threshold_for_classification_is_larger() {
    // Classification: sparse per-sample signal → longer trailing streak is
    // expected, so the drought threshold should fire later.
    for name in ["CATEGORICAL_ERROR", "CROSS_ENTROPY", "HINGE"] {
        let descriptor = TaskDescriptor::from_name(name, 10);
        let calibrated = drought_threshold_for_task(5, &descriptor);
        assert!(
            calibrated > 5,
            "{name}: drought threshold must scale up for classification (got {calibrated})",
        );
    }
}

#[test]
fn drought_threshold_for_classification_scales_with_base() {
    // The calibration is multiplicative-ish — a higher base should still come
    // out higher after calibration.
    let descriptor = TaskDescriptor::from_name("CATEGORICAL_ERROR", 10);
    let low = drought_threshold_for_task(5, &descriptor);
    let high = drought_threshold_for_task(20, &descriptor);
    assert!(
        high > low,
        "calibrated threshold should respect base ordering"
    );
}

#[test]
fn drought_threshold_saturates_safely() {
    // u32::MAX as base must not overflow.
    let descriptor = TaskDescriptor::from_name("CATEGORICAL_ERROR", 10);
    let out = drought_threshold_for_task(u32::MAX, &descriptor);
    assert_eq!(out, u32::MAX);
}
