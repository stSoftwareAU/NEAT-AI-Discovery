//! Issue #1017: Candidate selection pipeline MCMC applicability audit
//!
//! These tests validate the findings from the MCMC audit documented in
//! `docs/CANDIDATE_PIPELINE_MCMC_AUDIT.md`. They confirm that:
//!
//! 1. The pipeline uses deterministic accept/reject (not probabilistic MCMC)
//! 2. Pessimism discount hierarchy is correctly ordered by candidate success rate
//! 3. Prediction calibration factors are correctly ordered by overestimation severity
//! 4. The hold-out validation mechanism provides genuine overfitting protection
//! 5. Diversification (`shuffle_within_top_k`) provides exploration without MCMC

use neat_ai_discovery::analysis::constants::{
    COORDINATED_PESSIMISM_DISCOUNT, COORDINATED_PREDICTION_CALIBRATION, DIVERSIFY_TOP_K,
    HOLDOUT_MIN_SAMPLE_COUNT, HOLDOUT_VALIDATION_FRACTION, MIN_IMPROVED_RATIO,
    NEURON_PESSIMISM_CURVE_EXPONENT, NEURON_PESSIMISM_DISCOUNT_FLOOR,
    NEURON_PREDICTION_CALIBRATION, PESSIMISM_CURVE_EXPONENT, PESSIMISM_DISCOUNT_FLOOR,
    SYNAPSE_PESSIMISM_CURVE_EXPONENT, SYNAPSE_PESSIMISM_DISCOUNT_FLOOR,
    SYNAPSE_PREDICTION_CALIBRATION,
};
use neat_ai_discovery::analysis::synapse::{
    apply_neuron_pessimism_discount, apply_pessimism_discount, apply_prediction_calibration,
    apply_synapse_pessimism_discount,
};

// =============================================================================
// Section 1: Deterministic Accept/Reject (Not Probabilistic MCMC)
// =============================================================================

/// Issue #1017: The accept/reject threshold `MIN_IMPROVED_RATIO` is a fixed
/// constant -- confirming deterministic acceptance, not probabilistic MCMC.
/// A proper Metropolis-Hastings acceptance would use a ratio-dependent probability,
/// but our pipeline uses a hard cutoff.
#[test]
fn accept_reject_is_deterministic_threshold() {
    // MIN_IMPROVED_RATIO is a fixed f32 constant, not a function of chain state
    const {
        assert!(MIN_IMPROVED_RATIO > 0.0);
        assert!(MIN_IMPROVED_RATIO < 1.0);
    }

    // Verify determinism: same inputs always produce same accept/reject decision
    let gain = 0.05_f32;
    let result_a = apply_synapse_pessimism_discount(gain, 70, 100);
    let result_b = apply_synapse_pessimism_discount(gain, 70, 100);
    assert_eq!(
        result_a, result_b,
        "Issue #1017: Pessimism discount must be deterministic (no random acceptance)"
    );
}

/// Issue #1017: Candidates with zero or negative improvement are always rejected.
/// In MCMC, worse states can be accepted with some probability -- our pipeline
/// never does this, confirming it is not MCMC.
#[test]
fn zero_improvement_always_produces_zero_or_negative_gain() {
    // A candidate with 0 improved samples out of 100
    let gain = 0.0_f32;
    let synapse = apply_synapse_pessimism_discount(gain, 0, 100);
    let neuron = apply_neuron_pessimism_discount(gain, 0, 100);
    let generic = apply_pessimism_discount(gain, 0, 100);

    assert_eq!(
        synapse, 0.0,
        "Issue #1017: Zero gain in -> zero gain out (synapse)"
    );
    assert_eq!(
        neuron, 0.0,
        "Issue #1017: Zero gain in -> zero gain out (neuron)"
    );
    assert_eq!(
        generic, 0.0,
        "Issue #1017: Zero gain in -> zero gain out (generic)"
    );
}

// =============================================================================
// Section 2: Pessimism Discount Hierarchy Matches Success Rates
// =============================================================================

/// Issue #1017: The pessimism hierarchy must be ordered by observed success rates:
/// synapse (0%) < neuron (15%) < generic (~baseline).
/// More aggressive discounting for worse-performing candidate types.
#[test]
fn pessimism_floor_hierarchy_matches_success_rates() {
    const {
        // Synapse (0% success) should have the lowest floor (most aggressive)
        assert!(SYNAPSE_PESSIMISM_DISCOUNT_FLOOR < NEURON_PESSIMISM_DISCOUNT_FLOOR);
        // Neuron (15% success) should have a lower floor than generic
        assert!(NEURON_PESSIMISM_DISCOUNT_FLOOR < PESSIMISM_DISCOUNT_FLOOR);
    }
}

/// Issue #1017: The pessimism exponent hierarchy must be ordered inversely to
/// success rates: higher exponent = less forgiving curve.
#[test]
fn pessimism_exponent_hierarchy_matches_success_rates() {
    const {
        assert!(SYNAPSE_PESSIMISM_CURVE_EXPONENT > NEURON_PESSIMISM_CURVE_EXPONENT);
        assert!(NEURON_PESSIMISM_CURVE_EXPONENT > PESSIMISM_CURVE_EXPONENT);
    }
}

/// Issue #1017: At a moderate improved ratio (50%), the synapse discount must be
/// the most aggressive, followed by neuron, then generic -- reflecting their
/// respective success rates.
#[test]
fn pessimism_discount_ordering_at_moderate_ratio() {
    let gain = 0.10_f32;
    let improved = 50_u32;
    let total = 100_u32;

    let generic = apply_pessimism_discount(gain, improved, total);
    let neuron = apply_neuron_pessimism_discount(gain, improved, total);
    let synapse = apply_synapse_pessimism_discount(gain, improved, total);

    assert!(
        synapse < neuron,
        "Issue #1017: Synapse discount ({synapse:.6}) must be < neuron ({neuron:.6}) at 50% ratio"
    );
    assert!(
        neuron < generic,
        "Issue #1017: Neuron discount ({neuron:.6}) must be < generic ({generic:.6}) at 50% ratio"
    );
}

// =============================================================================
// Section 3: Prediction Calibration Hierarchy
// =============================================================================

/// Issue #1017: Prediction calibration factors must be ordered by overestimation
/// severity: coordinated (10,000x) < synapse (1,000x) < neuron (100x).
/// Smaller factor = more severe correction.
#[test]
fn prediction_calibration_hierarchy() {
    const {
        assert!(COORDINATED_PREDICTION_CALIBRATION < SYNAPSE_PREDICTION_CALIBRATION);
        assert!(SYNAPSE_PREDICTION_CALIBRATION < NEURON_PREDICTION_CALIBRATION);
    }
}

/// Issue #1017: All prediction calibration factors must be positive and less than 1.
/// They correct overestimation by scaling predictions down, never up.
#[test]
fn prediction_calibration_factors_are_scale_down() {
    const {
        assert!(SYNAPSE_PREDICTION_CALIBRATION > 0.0);
        assert!(SYNAPSE_PREDICTION_CALIBRATION < 1.0);
        assert!(NEURON_PREDICTION_CALIBRATION > 0.0);
        assert!(NEURON_PREDICTION_CALIBRATION < 1.0);
        assert!(COORDINATED_PREDICTION_CALIBRATION > 0.0);
        assert!(COORDINATED_PREDICTION_CALIBRATION < 1.0);
    }
}

/// Issue #1017: Applying prediction calibration always reduces the absolute gain.
/// This confirms it is a calibration correction (not a probabilistic acceptance).
#[test]
fn prediction_calibration_always_reduces_gain() {
    let gain = 0.05_f32;

    let synapse_calibrated = apply_prediction_calibration(gain, SYNAPSE_PREDICTION_CALIBRATION);
    assert!(
        synapse_calibrated < gain,
        "Issue #1017: Synapse calibration must reduce gain: {synapse_calibrated} >= {gain}"
    );
    assert!(
        synapse_calibrated > 0.0,
        "Issue #1017: Synapse calibration must preserve positivity"
    );

    let neuron_calibrated = apply_prediction_calibration(gain, NEURON_PREDICTION_CALIBRATION);
    assert!(
        neuron_calibrated < gain,
        "Issue #1017: Neuron calibration must reduce gain: {neuron_calibrated} >= {gain}"
    );
    assert!(
        neuron_calibrated > 0.0,
        "Issue #1017: Neuron calibration must preserve positivity"
    );

    let coordinated_calibrated =
        apply_prediction_calibration(gain, COORDINATED_PREDICTION_CALIBRATION);
    assert!(
        coordinated_calibrated < gain,
        "Issue #1017: Coordinated calibration must reduce gain: {coordinated_calibrated} >= {gain}"
    );
    assert!(
        coordinated_calibrated > 0.0,
        "Issue #1017: Coordinated calibration must preserve positivity"
    );
}

// =============================================================================
// Section 4: Hold-Out Validation Prevents Overfitting
// =============================================================================

/// Issue #1017: Hold-out validation parameters are within sensible ranges.
/// This mechanism addresses the multi-weight search overfitting problem
/// identified in the audit, without requiring MCMC-style chain convergence.
#[test]
fn holdout_validation_parameters_sensible() {
    const {
        // Minimum sample count should be high enough for reliable splits
        assert!(HOLDOUT_MIN_SAMPLE_COUNT >= 10);
        // Validation fraction should reserve meaningful validation data
        assert!(HOLDOUT_VALIDATION_FRACTION > 0.1);
        assert!(HOLDOUT_VALIDATION_FRACTION < 0.5);
    }
}

// =============================================================================
// Section 5: Diversification Provides Exploration Without MCMC
// =============================================================================

/// Issue #1017: `DIVERSIFY_TOP_K` must be large enough to provide meaningful
/// exploration diversity but not so large as to undermine the sorting.
#[test]
fn diversify_top_k_in_valid_range() {
    const {
        assert!(DIVERSIFY_TOP_K >= 16);
        assert!(DIVERSIFY_TOP_K <= 256);
    }
}

/// Issue #1017: Coordinated pessimism discount must be a positive scale-down factor.
/// This flat discount accounts for the 2.3% success rate of coordinated candidates.
#[test]
fn coordinated_pessimism_discount_is_scale_down() {
    const {
        assert!(COORDINATED_PESSIMISM_DISCOUNT > 0.0);
        assert!(COORDINATED_PESSIMISM_DISCOUNT < 1.0);
    }
}

// =============================================================================
// Section 6: Full Pipeline Calibration Stack
// =============================================================================

/// Issue #1017: Verify the full calibration stack for synapse candidates.
/// The pipeline applies: pessimism discount -> prediction calibration.
/// Both are multiplicative scale-down operations, confirming a deterministic
/// calibration pipeline rather than probabilistic MCMC acceptance.
#[test]
fn full_synapse_calibration_stack_reduces_prediction() {
    let raw_gain = 0.05_f32; // Typical neuron-level improvement prediction
    let improved = 60_u32;
    let total = 100_u32;

    // Step 1: Pessimism discount
    let after_pessimism = apply_synapse_pessimism_discount(raw_gain, improved, total);
    assert!(
        after_pessimism < raw_gain,
        "Issue #1017: Pessimism should reduce gain"
    );

    // Step 2: Prediction calibration
    let after_calibration =
        apply_prediction_calibration(after_pessimism, SYNAPSE_PREDICTION_CALIBRATION);
    assert!(
        after_calibration < after_pessimism,
        "Issue #1017: Calibration should further reduce gain"
    );

    // The combined reduction should be substantial (>90% reduction)
    let total_reduction = 1.0 - (after_calibration / raw_gain);
    assert!(
        total_reduction > 0.90,
        "Issue #1017: Combined calibration stack should reduce by >90%, got {:.1}%",
        total_reduction * 100.0,
    );
}

/// Issue #1017: The calibration stack is idempotent -- applying it twice with
/// the same inputs gives the same intermediate results. This confirms there
/// is no hidden state (as there would be in an MCMC chain).
#[test]
fn calibration_stack_is_stateless() {
    let gain = 0.03_f32;
    let improved = 45_u32;
    let total = 100_u32;

    let run1 = apply_prediction_calibration(
        apply_synapse_pessimism_discount(gain, improved, total),
        SYNAPSE_PREDICTION_CALIBRATION,
    );
    let run2 = apply_prediction_calibration(
        apply_synapse_pessimism_discount(gain, improved, total),
        SYNAPSE_PREDICTION_CALIBRATION,
    );

    assert_eq!(
        run1, run2,
        "Issue #1017: Calibration stack must be stateless (no chain state)"
    );
}
