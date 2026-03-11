//! Issue #791: Improve add-neurons prediction accuracy (15% success rate)
//!
//! Tests for:
//! 1. Neuron-specific pessimism calibration (more aggressive than synapse)
//! 2. Cross-validation brittleness filtering for neuron candidates

mod common;

use neat_ai_discovery::analysis::constants::{
    NEURON_PESSIMISM_CURVE_EXPONENT, NEURON_PESSIMISM_DISCOUNT_FLOOR, PESSIMISM_CURVE_EXPONENT,
    PESSIMISM_DISCOUNT_FLOOR,
};
use neat_ai_discovery::analysis::synapse::{
    apply_neuron_pessimism_discount, apply_pessimism_discount,
};

use neat_ai_discovery::analysis::scoring::cross_validation::{
    CrossValidationConfig, compute_cross_validation_score,
};

// =============================================================================
// Neuron-specific pessimism constants (Issue #791)
// =============================================================================

/// Issue #791: Neuron-specific pessimism floor should be lower than the synapse
/// floor, applying more aggressive base discounting given the 15% success rate.
#[test]
fn neuron_pessimism_floor_more_aggressive_than_synapse() {
    const {
        assert!(NEURON_PESSIMISM_DISCOUNT_FLOOR < PESSIMISM_DISCOUNT_FLOOR);
    }
}

/// Issue #791: Neuron-specific pessimism exponent should be higher than the synapse
/// exponent, making the curve less forgiving at moderate ratios.
#[test]
fn neuron_pessimism_exponent_less_forgiving_than_synapse() {
    const {
        assert!(NEURON_PESSIMISM_CURVE_EXPONENT > PESSIMISM_CURVE_EXPONENT);
    }
}

/// Issue #791: Both neuron pessimism constants should be within valid ranges.
#[test]
fn neuron_pessimism_constants_in_valid_ranges() {
    const {
        assert!(NEURON_PESSIMISM_DISCOUNT_FLOOR > 0.0);
        assert!(NEURON_PESSIMISM_DISCOUNT_FLOOR < 0.5);
        assert!(NEURON_PESSIMISM_CURVE_EXPONENT > 0.0);
        assert!(NEURON_PESSIMISM_CURVE_EXPONENT <= 1.0);
    }
}

// =============================================================================
// Neuron-specific pessimism discount function (Issue #791)
// =============================================================================

/// Issue #791: Neuron pessimism discount should be more aggressive than synapse
/// pessimism at all improved ratios between 0% and 100% (exclusive).
#[test]
fn neuron_pessimism_more_aggressive_at_all_moderate_ratios() {
    let gain = 0.10;

    // Test at various ratios from 5% to 95%
    for improved in (5..=95).step_by(5) {
        let synapse_result = apply_pessimism_discount(gain, improved, 100);
        let neuron_result = apply_neuron_pessimism_discount(gain, improved, 100);

        assert!(
            neuron_result <= synapse_result,
            "Issue #791: Neuron pessimism at {improved}% should be <= synapse pessimism. \
             Neuron={neuron_result:.6}, Synapse={synapse_result:.6}"
        );
    }
}

/// Issue #791: Neuron pessimism discount should still give full gain at 100% ratio.
#[test]
fn neuron_pessimism_full_ratio_gives_full_gain() {
    let gain = 0.10;
    let result = apply_neuron_pessimism_discount(gain, 100, 100);
    assert!(
        (result - gain).abs() < 1e-6,
        "Issue #791: All samples improved should give full gain: expected {gain}, got {result}"
    );
}

/// Issue #791: Neuron pessimism should be monotonically non-decreasing.
#[test]
fn neuron_pessimism_monotonically_increasing() {
    let gain = 0.10;
    let mut prev = apply_neuron_pessimism_discount(gain, 0, 100);
    for improved in (5..=100).step_by(5) {
        let current = apply_neuron_pessimism_discount(gain, improved, 100);
        assert!(
            current >= prev - 1e-7,
            "Issue #791: Neuron discount should be monotonically non-decreasing: \
             at {improved}/100, got {current:.6} < previous {prev:.6}"
        );
        prev = current;
    }
}

/// Issue #791: Zero total should give floor discount.
#[test]
fn neuron_pessimism_zero_total_gives_floor() {
    let gain = 0.10;
    let result = apply_neuron_pessimism_discount(gain, 0, 0);
    let expected = gain * NEURON_PESSIMISM_DISCOUNT_FLOOR;
    assert!(
        (result - expected).abs() < 1e-6,
        "Issue #791: Zero total should give floor discount: expected {expected}, got {result}"
    );
}

/// Issue #791: Typical add-neuron candidate at 45% ratio should get a lower
/// score with neuron-specific pessimism than with generic pessimism.
#[test]
fn neuron_pessimism_practical_difference() {
    let gain = 0.02; // Typical small add-neuron gain

    let synapse_discounted = apply_pessimism_discount(gain, 45, 100);
    let neuron_discounted = apply_neuron_pessimism_discount(gain, 45, 100);

    assert!(
        neuron_discounted < synapse_discounted,
        "Issue #791: Neuron-specific discount at 45% ratio should be lower. \
         Neuron={neuron_discounted:.6}, Synapse={synapse_discounted:.6}"
    );

    // The difference should be meaningful (at least 5%)
    let reduction_pct = (synapse_discounted - neuron_discounted) / synapse_discounted * 100.0;
    assert!(
        reduction_pct > 5.0,
        "Issue #791: Neuron discount reduction should be at least 5%. Got {reduction_pct:.1}%"
    );
}

// =============================================================================
// Cross-validation filtering for neuron candidates (Issue #791)
// =============================================================================

/// Issue #791: Cross-validation should detect brittle candidates that perform
/// inconsistently across sample subsets.
#[test]
fn cross_validation_penalises_brittle_neuron_candidates() {
    use neat_ai_discovery::analysis::samples::HelpfulSample;

    // Create brittle samples: first half has strong signal, second half has none.
    // This mimics a candidate that overfits to a specific data subset.
    let mut samples = Vec::new();
    for i in 0..100 {
        if i < 50 {
            // Strong positive correlation in first half
            samples.push(HelpfulSample {
                activation: i as f32 * 0.02,
                avg_error: i as f32 * 0.01,
                target_value: None,
                target_activation: None,
            });
        } else {
            // No meaningful correlation in second half
            samples.push(HelpfulSample {
                activation: 0.5,
                avg_error: if i % 2 == 0 { 0.1 } else { -0.1 },
                target_value: None,
                target_activation: None,
            });
        }
    }

    let config = CrossValidationConfig::default();
    let result = compute_cross_validation_score(&samples, &config);

    assert!(
        result.is_some(),
        "Should have enough samples for cross-validation"
    );
    let cv_result = result.unwrap();

    // Brittle candidates should have a non-zero penalty
    assert!(
        cv_result.brittleness_penalty > 0.0,
        "Issue #791: Brittle candidate should receive a penalty. Got {:.4}",
        cv_result.brittleness_penalty
    );
}

/// Issue #791: Cross-validation should not penalise consistent candidates
/// that perform well across all sample subsets.
#[test]
fn cross_validation_does_not_penalise_consistent_candidates() {
    use neat_ai_discovery::analysis::samples::HelpfulSample;

    // Create consistent samples: strong correlation throughout
    let mut samples = Vec::new();
    for i in 0..100 {
        let activation = (i as f32) * 0.01;
        let error = activation * 0.5 + 0.01; // Consistent linear relationship
        samples.push(HelpfulSample {
            activation,
            avg_error: error,
            target_value: None,
            target_activation: None,
        });
    }

    let config = CrossValidationConfig::default();
    let result = compute_cross_validation_score(&samples, &config);

    assert!(
        result.is_some(),
        "Should have enough samples for cross-validation"
    );
    let cv_result = result.unwrap();

    // Consistent candidates should have low or zero penalty
    assert!(
        cv_result.brittleness_penalty < 0.1,
        "Issue #791: Consistent candidate should have low penalty. Got {:.4}",
        cv_result.brittleness_penalty
    );
}

/// Issue #791: Cross-validation should skip gracefully when there are too
/// few samples for meaningful fold evaluation.
#[test]
fn cross_validation_skips_with_insufficient_samples() {
    use neat_ai_discovery::analysis::samples::HelpfulSample;

    // Only 10 samples — not enough for 5-fold CV with min 15 per fold
    let samples: Vec<HelpfulSample> = (0..10)
        .map(|i| HelpfulSample {
            activation: i as f32 * 0.1,
            avg_error: 0.05,
            target_value: None,
            target_activation: None,
        })
        .collect();

    let config = CrossValidationConfig::default();
    let result = compute_cross_validation_score(&samples, &config);

    assert!(
        result.is_none(),
        "Issue #791: Should return None with insufficient samples"
    );
}
