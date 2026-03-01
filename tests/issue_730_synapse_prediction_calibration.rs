//! Issue #730: Fix add-synapses module — 0% success rate in production cache
//!
//! Integration tests for improved synapse prediction accuracy:
//! 1. Creature-level error fraction scaling produces realistic predictions
//! 2. MIN_IMPROVED_RATIO constant is sensible

use neat_ai_discovery::analysis::constants::MIN_IMPROVED_RATIO;
use neat_ai_discovery::analysis::samples::EPSILON;
use neat_ai_discovery::analysis::synapse::post_processing::scale_by_error_fraction;

// =============================================================================
// Creature-level error fraction scaling tests
// =============================================================================

/// Issue #730: Error fraction scaling should reduce raw predictions to be
/// within an order of magnitude of actual creature-level results.
#[test]
fn test_issue_730_error_fraction_scaling_reduces_prediction() {
    // Scenario: target neuron has 5% of total creature error
    let raw_prediction = 0.05; // 5% of target neuron error reduced
    let target_error_sq = 0.01; // Target neuron's mean squared error
    let total_error_sq = 0.20; // Total creature error (20 neurons, this one is 5%)

    let scaled = scale_by_error_fraction(raw_prediction, target_error_sq, total_error_sq);

    // Expected: 0.05 × (0.01 / 0.20) = 0.05 × 0.05 = 0.0025
    assert!(
        (scaled - 0.0025).abs() < 0.001,
        "Error fraction scaling should produce ~0.0025, got {scaled:.6}"
    );

    // The scaled prediction should be much smaller than the raw prediction
    assert!(
        scaled < raw_prediction * 0.5,
        "Scaled prediction ({scaled:.6}) should be significantly smaller than raw ({raw_prediction:.6})"
    );
}

/// Issue #730: When a target has most of the creature error, scaling should
/// preserve most of the prediction.
#[test]
fn test_issue_730_error_fraction_preserves_dominant_target() {
    let raw_prediction = 0.05;
    let target_error_sq = 0.16;
    let total_error_sq = 0.20;

    let scaled = scale_by_error_fraction(raw_prediction, target_error_sq, total_error_sq);

    // Expected: 0.05 × (0.16 / 0.20) = 0.05 × 0.80 = 0.04
    assert!(
        (scaled - 0.04).abs() < 0.001,
        "Dominant target should preserve most of prediction, got {scaled:.6}"
    );
}

/// Issue #730: Error fraction with zero total error should return zero.
#[test]
fn test_issue_730_error_fraction_zero_total_returns_zero() {
    let scaled = scale_by_error_fraction(0.05, 0.0, 0.0);
    assert!(
        scaled.abs() < EPSILON,
        "Zero total error should give zero scaled prediction, got {scaled:.6}"
    );
}

/// Issue #730: Error fraction should clamp to 1.0 when target_error > total_error
/// (numerical edge case).
#[test]
fn test_issue_730_error_fraction_clamps_to_one() {
    // Due to floating point, target might slightly exceed total
    let scaled = scale_by_error_fraction(0.05, 0.21, 0.20);

    // Should be clamped to at most raw_prediction (fraction <= 1.0)
    assert!(
        scaled <= 0.05 + EPSILON,
        "Error fraction should clamp, got {scaled:.6}"
    );
}

// =============================================================================
// MIN_IMPROVED_RATIO constant validation
// =============================================================================

/// Issue #730: MIN_IMPROVED_RATIO should be a sensible threshold.
/// Uses const assertions to validate at compile time.
#[test]
fn test_issue_730_min_improved_ratio_is_sensible() {
    const {
        assert!(MIN_IMPROVED_RATIO > 0.0);
        assert!(MIN_IMPROVED_RATIO <= 0.75);
        assert!(MIN_IMPROVED_RATIO >= 0.3);
    }

    // Runtime check that the constant is usable for filtering
    let test_ratio = 0.6_f32;
    let passes = test_ratio >= MIN_IMPROVED_RATIO;
    assert!(passes, "A 60% improved ratio should pass the threshold");

    let test_ratio_low = 0.2_f32;
    let fails = test_ratio_low < MIN_IMPROVED_RATIO;
    assert!(fails, "A 20% improved ratio should fail the threshold");
}
