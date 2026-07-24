//! Issue #1109: Raise `NEURON_MIN_IMPROVED_RATIO` from 0.4 to 0.55.
//!
//! Production failure-cache data showed neuron
//! candidates with improved ratios of 52-54% (267-274 out of 510) consistently
//! producing negative actual error reductions despite positive predictions.
//! Raising the threshold from 0.4 to 0.55 filters out these near-random
//! candidates that waste evaluation budget.

use neat_ai_discovery::analysis::constants::{MIN_IMPROVED_RATIO, NEURON_MIN_IMPROVED_RATIO};

/// Issue #1109: The neuron improved-ratio threshold must be raised to 0.55.
#[test]
fn neuron_min_improved_ratio_is_raised_to_0_55() {
    const {
        assert!((NEURON_MIN_IMPROVED_RATIO - 0.55_f32).abs() < f32::EPSILON);
    }
}

/// Issue #1109: Candidates with the production failure pattern (52-54% improved)
/// must be rejected by the raised threshold.
#[test]
fn production_failure_pattern_is_rejected() {
    // 267/510 = 0.5235, 274/510 = 0.5372 — the observed failure ratios.
    let failure_ratios = [267.0_f32 / 510.0, 270.0 / 510.0, 274.0 / 510.0];
    for ratio in failure_ratios {
        assert!(
            ratio < NEURON_MIN_IMPROVED_RATIO,
            "Production failure ratio {ratio:.4} should be below the raised threshold \
             {NEURON_MIN_IMPROVED_RATIO}"
        );
    }
}

/// Issue #1109: A candidate with exactly 55% improved must pass.
#[test]
fn exactly_55_percent_still_passes() {
    let ratio = 0.55_f32;
    assert!(
        ratio >= NEURON_MIN_IMPROVED_RATIO,
        "A 55% improved ratio should pass the raised threshold"
    );
}

/// Issue #1109: The neuron threshold remains below the synapse threshold
/// (neuron candidates are inherently noisier and get a small allowance).
#[test]
fn neuron_threshold_stays_below_synapse_threshold() {
    const {
        assert!(NEURON_MIN_IMPROVED_RATIO < MIN_IMPROVED_RATIO);
    }
}
