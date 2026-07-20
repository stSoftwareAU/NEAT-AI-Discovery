//! Tests for reduced coordinated-structural compound discounting (Issue #1058).
//!
//! GRQ-sampler cache shows coordinated-structural candidates have a ~1.1% success
//! rate (e.g., creature 0e18e62c: 6/389). The three-layer compound discount
//! (per-op × pessimism × calibration) was too aggressive, filtering out viable
//! candidates while remaining poorly calibrated.
//!
//! Issue #1058 replaces the three-layer compound with a single empirical discount
//! per operation count, lowers MIN_COORDINATED_MULTI_OP_GAIN, and removes the
//! separate COORDINATED_PESSIMISM_DISCOUNT (folded into empirical factors).

use neat_ai_discovery::analysis::candidate_aggregation::{
    apply_operation_count_discount, validate_coordinated_candidate_gain,
};
use neat_ai_discovery::analysis::constants::{
    COORDINATED_EMPIRICAL_DISCOUNT_2OPS, COORDINATED_EMPIRICAL_DISCOUNT_3OPS,
    COORDINATED_EMPIRICAL_DISCOUNT_4PLUS_OPS, MIN_COORDINATED_MULTI_OP_GAIN,
};
use neat_ai_discovery::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_single_op_candidate(gain: f32) -> CoordinatedStructuralCandidateJson {
    CoordinatedStructuralCandidateJson {
        remove_neuron_compensation: None,
        constant_neuron_bias_fold: None,
        operations: vec![CoordinatedStructuralOpJson::SetBias {
            neuron_uuid: "neuron-a".to_string(),
            bias: 0.5,
        }],
        expected_creature_score_gain: gain,
        comment: Some("test module".to_string()),
    }
}

fn make_multi_op_candidate(op_count: usize, gain: f32) -> CoordinatedStructuralCandidateJson {
    let mut operations = Vec::with_capacity(op_count);

    operations.push(CoordinatedStructuralOpJson::RemoveSynapse {
        from_neuron_uuid: "source".to_string(),
        to_neuron_uuid: "target".to_string(),
    });

    if op_count >= 2 {
        operations.push(CoordinatedStructuralOpJson::AddNeuron {
            neuron_uuid: "new-hidden".to_string(),
            neuron_type: "hidden".to_string(),
            squash: "LOGISTIC".to_string(),
            bias: 0.0,
            insert_before_neuron_uuid: Some("target".to_string()),
        });
    }

    for i in 2..op_count {
        operations.push(CoordinatedStructuralOpJson::AddSynapse {
            from_neuron_uuid: format!("node-{i}"),
            to_neuron_uuid: "new-hidden".to_string(),
            weight: 0.1,
        });
    }

    CoordinatedStructuralCandidateJson {
        remove_neuron_compensation: None,
        constant_neuron_bias_fold: None,
        operations,
        expected_creature_score_gain: gain,
        comment: Some("test coordinated module".to_string()),
    }
}

// ---------------------------------------------------------------------------
// Empirical discount constants are in valid ranges (Issue #1058)
// ---------------------------------------------------------------------------

#[test]
fn empirical_discount_constants_are_in_valid_range() {
    // All empirical discount factors must be in (0.0, 1.0).
    assert!(COORDINATED_EMPIRICAL_DISCOUNT_2OPS > 0.0);
    assert!(COORDINATED_EMPIRICAL_DISCOUNT_2OPS < 1.0);
    assert!(COORDINATED_EMPIRICAL_DISCOUNT_3OPS > 0.0);
    assert!(COORDINATED_EMPIRICAL_DISCOUNT_3OPS < 1.0);
    assert!(COORDINATED_EMPIRICAL_DISCOUNT_4PLUS_OPS > 0.0);
    assert!(COORDINATED_EMPIRICAL_DISCOUNT_4PLUS_OPS < 1.0);
}

#[test]
fn empirical_discounts_decrease_with_operation_count() {
    // More operations → more aggressive discount (lower factor).
    assert!(
        COORDINATED_EMPIRICAL_DISCOUNT_3OPS < COORDINATED_EMPIRICAL_DISCOUNT_2OPS,
        "3-op discount ({}) should be less than 2-op ({})",
        COORDINATED_EMPIRICAL_DISCOUNT_3OPS,
        COORDINATED_EMPIRICAL_DISCOUNT_2OPS
    );
    assert!(
        COORDINATED_EMPIRICAL_DISCOUNT_4PLUS_OPS < COORDINATED_EMPIRICAL_DISCOUNT_3OPS,
        "4+-op discount ({}) should be less than 3-op ({})",
        COORDINATED_EMPIRICAL_DISCOUNT_4PLUS_OPS,
        COORDINATED_EMPIRICAL_DISCOUNT_3OPS
    );
}

// ---------------------------------------------------------------------------
// Simplified discount model replaces compound (Issue #1058)
// ---------------------------------------------------------------------------

#[test]
fn single_op_candidate_has_no_discount() {
    let candidate = make_single_op_candidate(0.01);
    let discounted = apply_operation_count_discount(&candidate);
    assert!(
        (discounted - 0.01).abs() < 1e-6,
        "single-op candidate should not be discounted, got {discounted}"
    );
}

#[test]
fn two_op_candidate_uses_empirical_factor() {
    let candidate = make_multi_op_candidate(2, 0.01);
    let discounted = apply_operation_count_discount(&candidate);
    let expected = 0.01 * COORDINATED_EMPIRICAL_DISCOUNT_2OPS;
    assert!(
        (discounted - expected).abs() < 1e-6,
        "2-op candidate should use empirical factor: expected {expected}, got {discounted}"
    );
}

#[test]
fn three_op_candidate_uses_empirical_factor() {
    let candidate = make_multi_op_candidate(3, 0.01);
    let discounted = apply_operation_count_discount(&candidate);
    let expected = 0.01 * COORDINATED_EMPIRICAL_DISCOUNT_3OPS;
    assert!(
        (discounted - expected).abs() < 1e-6,
        "3-op candidate should use empirical factor: expected {expected}, got {discounted}"
    );
}

#[test]
fn four_plus_op_candidate_uses_empirical_factor() {
    let candidate = make_multi_op_candidate(4, 0.01);
    let discounted = apply_operation_count_discount(&candidate);
    let expected = 0.01 * COORDINATED_EMPIRICAL_DISCOUNT_4PLUS_OPS;
    assert!(
        (discounted - expected).abs() < 1e-6,
        "4-op candidate should use empirical factor: expected {expected}, got {discounted}"
    );
}

#[test]
fn five_op_candidate_uses_4plus_factor() {
    // 5+ operations should use the same factor as 4+.
    let candidate = make_multi_op_candidate(5, 0.01);
    let discounted = apply_operation_count_discount(&candidate);
    let expected = 0.01 * COORDINATED_EMPIRICAL_DISCOUNT_4PLUS_OPS;
    assert!(
        (discounted - expected).abs() < 1e-6,
        "5-op candidate should use 4+-op factor: expected {expected}, got {discounted}"
    );
}

#[test]
fn discount_monotonically_increases_with_ops() {
    let gain = 0.01;
    let d2 = apply_operation_count_discount(&make_multi_op_candidate(2, gain));
    let d3 = apply_operation_count_discount(&make_multi_op_candidate(3, gain));
    let d4 = apply_operation_count_discount(&make_multi_op_candidate(4, gain));
    let d5 = apply_operation_count_discount(&make_multi_op_candidate(5, gain));

    assert!(d3 < d2, "3-op ({d3}) should be less than 2-op ({d2})");
    assert!(d4 < d3, "4-op ({d4}) should be less than 3-op ({d3})");
    // 4-op and 5-op use the same factor.
    assert!(
        (d4 - d5).abs() < 1e-6,
        "4-op ({d4}) and 5-op ({d5}) should be equal"
    );
}

// ---------------------------------------------------------------------------
// Simplified model is less aggressive than old compound (Issue #1058)
// ---------------------------------------------------------------------------

#[test]
fn simplified_2op_discount_is_less_aggressive_than_old_compound() {
    // Old compound for 2-op: 0.65^1 × 0.15 = 0.0975
    // New empirical factor should be higher (less aggressive).
    let old_compound_2op = 0.65_f32 * 0.15;
    assert!(
        COORDINATED_EMPIRICAL_DISCOUNT_2OPS > old_compound_2op,
        "new 2-op factor ({}) should be less aggressive than old compound ({})",
        COORDINATED_EMPIRICAL_DISCOUNT_2OPS,
        old_compound_2op
    );
}

#[test]
fn simplified_3op_discount_is_less_aggressive_than_old_compound() {
    // Old compound for 3-op: 0.65^2 × 0.15 = 0.0634
    let old_compound_3op = 0.65_f32.powi(2) * 0.15;
    assert!(
        COORDINATED_EMPIRICAL_DISCOUNT_3OPS > old_compound_3op,
        "new 3-op factor ({}) should be less aggressive than old compound ({})",
        COORDINATED_EMPIRICAL_DISCOUNT_3OPS,
        old_compound_3op
    );
}

// ---------------------------------------------------------------------------
// Lowered MIN_COORDINATED_MULTI_OP_GAIN threshold (Issue #1058)
// ---------------------------------------------------------------------------

#[test]
fn min_gain_threshold_is_lower_than_old_value() {
    // Old threshold was 1e-3. New should be lower since calibration already
    // accounts for overestimation.
    assert!(
        MIN_COORDINATED_MULTI_OP_GAIN < 1e-3,
        "threshold ({}) should be lower than old 1e-3",
        MIN_COORDINATED_MULTI_OP_GAIN
    );
    assert!(
        MIN_COORDINATED_MULTI_OP_GAIN > 0.0,
        "threshold must remain positive"
    );
}

#[test]
fn moderate_gain_now_passes_for_2op_candidate() {
    // A gain of 0.005 on a 2-op candidate was borderline under old thresholds.
    // With the less aggressive discount and lower threshold, it should pass.
    let candidate = make_multi_op_candidate(2, 0.005);
    let valid = validate_coordinated_candidate_gain(&candidate);
    assert!(
        valid,
        "moderate gain (0.005) on 2-op candidate should pass with reduced discounting"
    );
}

#[test]
fn tiny_gain_still_rejected_for_multi_op() {
    // Extremely small gains should still be rejected.
    let candidate = make_multi_op_candidate(4, 1e-7);
    let valid = validate_coordinated_candidate_gain(&candidate);
    assert!(!valid, "tiny gain (1e-7) on 4-op candidate should be rejected");
}

#[test]
fn single_op_with_small_positive_gain_passes() {
    let candidate = make_single_op_candidate(1e-6);
    let valid = validate_coordinated_candidate_gain(&candidate);
    assert!(valid, "single-op with small positive gain should pass");
}

// ---------------------------------------------------------------------------
// GRQ-sampler known examples (Issue #1058)
// ---------------------------------------------------------------------------

#[test]
fn grq_sampler_creature_0e18e62c_success_pattern() {
    // Creature 0e18e62c: 6 successes / 389 failures (~1.5% success rate).
    // A typical successful coordinated-structural candidate from this creature
    // was a 2-op candidate. After the simplified discount, such a candidate
    // should retain meaningful gain to be selected.
    let candidate = make_multi_op_candidate(2, 0.008);
    let discounted = apply_operation_count_discount(&candidate);
    assert!(
        discounted > MIN_COORDINATED_MULTI_OP_GAIN,
        "typical 2-op GRQ-sampler candidate (gain=0.008) should pass threshold: \
         discounted={discounted}, threshold={}",
        MIN_COORDINATED_MULTI_OP_GAIN
    );
}

#[test]
fn grq_sampler_aggressive_filtering_of_4op_candidates() {
    // 4-op candidates have near-zero success in production.
    // Even with reduced discounting, small gains on 4-op should be filtered.
    let candidate = make_multi_op_candidate(4, 0.001);
    let discounted = apply_operation_count_discount(&candidate);
    // The gain should be reduced substantially.
    assert!(
        discounted < 0.001,
        "4-op candidate gain should be substantially reduced: got {discounted}"
    );
}
