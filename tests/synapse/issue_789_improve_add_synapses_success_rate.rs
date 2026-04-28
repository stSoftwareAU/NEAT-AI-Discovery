//! Issue #789: Investigate and improve add-synapses 0% success rate
//!
//! Tests for:
//! 1. Synapse-specific pessimism calibration (more aggressive than both generic and neuron)
//! 2. Raised `MIN_IMPROVED_RATIO` threshold for stricter filtering

use neat_ai_discovery::analysis::constants::{
    MIN_IMPROVED_RATIO, NEURON_PESSIMISM_CURVE_EXPONENT, NEURON_PESSIMISM_DISCOUNT_FLOOR,
    PESSIMISM_CURVE_EXPONENT, PESSIMISM_DISCOUNT_FLOOR, SYNAPSE_PESSIMISM_CURVE_EXPONENT,
    SYNAPSE_PESSIMISM_DISCOUNT_FLOOR,
};
use neat_ai_discovery::analysis::synapse::{
    apply_pessimism_discount, apply_synapse_pessimism_discount,
};

// =============================================================================
// Synapse-specific pessimism constants (Issue #789)
// =============================================================================

/// Issue #789: Synapse-specific pessimism floor should be lower than the generic
/// floor, applying more aggressive base discounting given the 0% success rate.
#[test]
fn synapse_pessimism_floor_more_aggressive_than_generic() {
    const {
        assert!(SYNAPSE_PESSIMISM_DISCOUNT_FLOOR < PESSIMISM_DISCOUNT_FLOOR);
    }
}

/// Issue #789: Synapse-specific pessimism floor should be lower than or equal to
/// the neuron floor, since synapses have a worse success rate (0% vs 15%).
#[test]
fn synapse_pessimism_floor_at_least_as_aggressive_as_neuron() {
    const {
        assert!(SYNAPSE_PESSIMISM_DISCOUNT_FLOOR <= NEURON_PESSIMISM_DISCOUNT_FLOOR);
    }
}

/// Issue #789: Synapse-specific pessimism exponent should be higher than the generic
/// exponent, making the curve less forgiving at moderate ratios.
#[test]
fn synapse_pessimism_exponent_less_forgiving_than_generic() {
    const {
        assert!(SYNAPSE_PESSIMISM_CURVE_EXPONENT > PESSIMISM_CURVE_EXPONENT);
    }
}

/// Issue #789: Synapse-specific pessimism exponent should be higher than or equal to
/// the neuron exponent, since synapses have a worse success rate.
#[test]
fn synapse_pessimism_exponent_at_least_as_strict_as_neuron() {
    const {
        assert!(SYNAPSE_PESSIMISM_CURVE_EXPONENT >= NEURON_PESSIMISM_CURVE_EXPONENT);
    }
}

/// Issue #789: Both synapse pessimism constants should be within valid ranges.
#[test]
fn synapse_pessimism_constants_in_valid_ranges() {
    const {
        assert!(SYNAPSE_PESSIMISM_DISCOUNT_FLOOR > 0.0);
        assert!(SYNAPSE_PESSIMISM_DISCOUNT_FLOOR < 0.5);
        assert!(SYNAPSE_PESSIMISM_CURVE_EXPONENT > 0.0);
        assert!(SYNAPSE_PESSIMISM_CURVE_EXPONENT <= 1.0);
    }
}

// =============================================================================
// Synapse-specific pessimism discount function (Issue #789)
// =============================================================================

/// Issue #789: Synapse pessimism discount should be more aggressive than generic
/// pessimism at all improved ratios between 0% and 100% (exclusive).
#[test]
fn synapse_pessimism_more_aggressive_at_all_moderate_ratios() {
    let gain = 0.10;

    // Test at various ratios from 5% to 95%
    for improved in (5..=95).step_by(5) {
        let generic_result = apply_pessimism_discount(gain, improved, 100);
        let synapse_result = apply_synapse_pessimism_discount(gain, improved, 100, None);

        assert!(
            synapse_result <= generic_result,
            "Issue #789: Synapse pessimism at {improved}% should be <= generic pessimism. \
             Synapse={synapse_result:.6}, Generic={generic_result:.6}"
        );
    }
}

/// Issue #789: Synapse pessimism discount should still give full gain at 100% ratio.
#[test]
fn synapse_pessimism_full_ratio_gives_full_gain() {
    let gain = 0.10;
    let result = apply_synapse_pessimism_discount(gain, 100, 100, None);
    assert!(
        (result - gain).abs() < 1e-6,
        "Issue #789: All samples improved should give full gain: expected {gain}, got {result}"
    );
}

/// Issue #789: Synapse pessimism should be monotonically non-decreasing.
#[test]
fn synapse_pessimism_monotonically_increasing() {
    let gain = 0.10;
    let mut prev = apply_synapse_pessimism_discount(gain, 0, 100, None);
    for improved in (5..=100).step_by(5) {
        let current = apply_synapse_pessimism_discount(gain, improved, 100, None);
        assert!(
            current >= prev - 1e-7,
            "Issue #789: Synapse discount should be monotonically non-decreasing: \
             at {improved}/100, got {current:.6} < previous {prev:.6}"
        );
        prev = current;
    }
}

/// Issue #789: Zero total should give floor discount.
#[test]
fn synapse_pessimism_zero_total_gives_floor() {
    let gain = 0.10;
    let result = apply_synapse_pessimism_discount(gain, 0, 0, None);
    let expected = gain * SYNAPSE_PESSIMISM_DISCOUNT_FLOOR;
    assert!(
        (result - expected).abs() < 1e-6,
        "Issue #789: Zero total should give floor discount: expected {expected}, got {result}"
    );
}

/// Issue #789: Typical synapse candidate at 60% ratio should get a lower
/// score with synapse-specific pessimism than with generic pessimism.
#[test]
fn synapse_pessimism_practical_difference() {
    let gain = 0.02; // Typical small synapse gain

    let generic_discounted = apply_pessimism_discount(gain, 60, 100);
    let synapse_discounted = apply_synapse_pessimism_discount(gain, 60, 100, None);

    assert!(
        synapse_discounted < generic_discounted,
        "Issue #789: Synapse-specific discount at 60% ratio should be lower. \
         Synapse={synapse_discounted:.6}, Generic={generic_discounted:.6}"
    );

    // The difference should be meaningful (at least 5%)
    let reduction_pct = (generic_discounted - synapse_discounted) / generic_discounted * 100.0;
    assert!(
        reduction_pct > 5.0,
        "Issue #789: Synapse discount reduction should be at least 5%. Got {reduction_pct:.1}%"
    );
}

// =============================================================================
// MIN_IMPROVED_RATIO threshold (Issue #789)
// =============================================================================

/// Issue #789: `MIN_IMPROVED_RATIO` should be raised above 0.5 to filter out
/// marginal candidates that consistently fail ablation testing, but not
/// be too strict (still <= 0.75).
#[test]
fn min_improved_ratio_in_range() {
    const {
        assert!(MIN_IMPROVED_RATIO > 0.5);
        assert!(MIN_IMPROVED_RATIO <= 0.75);
    }
}
