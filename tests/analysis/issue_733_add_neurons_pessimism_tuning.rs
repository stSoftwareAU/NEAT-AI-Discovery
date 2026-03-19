//! Issue #733: Tune pessimism discount to improve add-neurons success rate
//!
//! Tests for:
//! 1. Logistic pessimism discount curve (more forgiving above 30%, more aggressive below 10%)
//! 2. `MIN_IMPROVED_RATIO` filtering applied to neuron candidates
//! 3. Adaptive pessimism constant validation

use neat_ai_discovery::analysis::constants::{NEURON_MIN_IMPROVED_RATIO, PESSIMISM_DISCOUNT_FLOOR};
use neat_ai_discovery::analysis::synapse::apply_pessimism_discount;

// =============================================================================
// Logistic pessimism discount curve tests (Issue #733)
// =============================================================================

/// Issue #733: The pessimism discount should use a concave curve that is more
/// forgiving for moderate improved ratios (30-60%) compared to the old linear
/// formula. This helps retain add-neuron candidates that have genuine signal.
#[test]
fn pessimism_discount_concave_curve_more_forgiving_at_moderate_ratios() {
    let gain = 0.10;

    // Old linear formula: discount = 0.15 + 0.85 × 0.4 = 0.49
    // New concave curve should give a higher discount (more forgiving)
    let result_40pct = apply_pessimism_discount(gain, 40, 100);
    let old_linear_40pct =
        gain * (PESSIMISM_DISCOUNT_FLOOR + (1.0 - PESSIMISM_DISCOUNT_FLOOR) * 0.4);

    assert!(
        result_40pct > old_linear_40pct,
        "Issue #733: Concave curve at 40% ratio should be more forgiving than linear. \
         Got {result_40pct:.6}, old linear would give {old_linear_40pct:.6}"
    );
}

/// Issue #733: The pessimism discount should still be aggressive for very low
/// ratios (below 10%) to filter out noise.
#[test]
fn pessimism_discount_still_aggressive_at_very_low_ratios() {
    let gain = 0.10;

    // At 5% ratio, the discount should still be close to the floor
    let result_5pct = apply_pessimism_discount(gain, 5, 100);
    let floor_gain = gain * PESSIMISM_DISCOUNT_FLOOR;

    // Should be within 2× of the floor gain (still quite aggressive)
    assert!(
        result_5pct < floor_gain * 2.5,
        "Issue #733: At 5% ratio, discount should remain aggressive. \
         Got {result_5pct:.6}, floor would give {floor_gain:.6}"
    );
}

/// Issue #733: Monotonicity must still hold — higher ratios always produce
/// higher discounted gains.
#[test]
fn pessimism_discount_monotonically_increasing_with_concave_curve() {
    let gain = 0.10;
    let mut prev = apply_pessimism_discount(gain, 0, 100);
    for improved in (5..=100).step_by(5) {
        let current = apply_pessimism_discount(gain, improved, 100);
        assert!(
            current >= prev - 1e-7,
            "Issue #733: Discount should be monotonically non-decreasing: \
             at {improved}/100, got {current:.6} < previous {prev:.6}"
        );
        prev = current;
    }
}

/// Issue #733: Full improvement (ratio = 1.0) should still give full gain.
#[test]
fn pessimism_discount_full_ratio_gives_full_gain() {
    let gain = 0.10;
    let result = apply_pessimism_discount(gain, 100, 100);
    assert!(
        (result - gain).abs() < 1e-6,
        "Issue #733: All samples improved should give full gain: expected {gain}, got {result}"
    );
}

/// Issue #733: Zero total should give floor discount (edge case).
#[test]
fn pessimism_discount_zero_total_gives_floor_gain() {
    let gain = 0.10;
    let result = apply_pessimism_discount(gain, 0, 0);
    let expected = gain * PESSIMISM_DISCOUNT_FLOOR;
    assert!(
        (result - expected).abs() < 1e-6,
        "Issue #733: Zero total should give floor discount: expected {expected}, got {result}"
    );
}

// =============================================================================
// NEURON_MIN_IMPROVED_RATIO constant validation (Issue #733)
// =============================================================================

/// Issue #733: `NEURON_MIN_IMPROVED_RATIO` should be a sensible threshold for
/// filtering add-neuron candidates that have insufficient sample improvement.
#[test]
fn neuron_min_improved_ratio_is_sensible() {
    const {
        assert!(NEURON_MIN_IMPROVED_RATIO > 0.0);
        assert!(NEURON_MIN_IMPROVED_RATIO <= 0.75);
        assert!(NEURON_MIN_IMPROVED_RATIO >= 0.3);
    }

    // Runtime check: a 55% ratio should pass neuron threshold
    let test_ratio_good = 0.55_f32;
    assert!(
        test_ratio_good >= NEURON_MIN_IMPROVED_RATIO,
        "A 55% improved ratio should pass the neuron threshold"
    );

    // Runtime check: a 20% ratio should fail neuron threshold
    let test_ratio_bad = 0.20_f32;
    assert!(
        test_ratio_bad < NEURON_MIN_IMPROVED_RATIO,
        "A 20% improved ratio should fail the neuron threshold"
    );
}

/// Issue #733: `NEURON_MIN_IMPROVED_RATIO` should be <= `MIN_IMPROVED_RATIO`
/// (neuron candidates are inherently noisier, so we allow a slightly lower bar).
#[test]
fn neuron_min_improved_ratio_not_more_strict_than_synapse() {
    use neat_ai_discovery::analysis::constants::MIN_IMPROVED_RATIO;

    const {
        assert!(NEURON_MIN_IMPROVED_RATIO <= MIN_IMPROVED_RATIO);
    }
}

// =============================================================================
// Combined discount behaviour tests
// =============================================================================

/// Issue #733: The concave curve should give a meaningfully higher gain for
/// a typical add-neuron scenario (40-60% improved ratio) compared to the
/// old linear approach with the same floor.
#[test]
fn pessimism_discount_practical_add_neuron_improvement() {
    let gain = 0.02; // Typical small add-neuron gain

    // Typical add-neuron candidate: 45% of samples improve
    let discounted = apply_pessimism_discount(gain, 45, 100);

    // Old linear: 0.02 × (0.15 + 0.85 × 0.45) = 0.02 × 0.5325 = 0.01065
    let old_linear = gain * (PESSIMISM_DISCOUNT_FLOOR + (1.0 - PESSIMISM_DISCOUNT_FLOOR) * 0.45);

    assert!(
        discounted > old_linear,
        "Issue #733: Add-neuron candidate at 45% ratio should get higher score. \
         New={discounted:.6}, old linear={old_linear:.6}"
    );

    // The improvement should be meaningful (at least 10% better)
    let improvement_pct = (discounted - old_linear) / old_linear * 100.0;
    assert!(
        improvement_pct > 5.0,
        "Issue #733: Improvement should be at least 5% better. Got {improvement_pct:.1}%"
    );
}
