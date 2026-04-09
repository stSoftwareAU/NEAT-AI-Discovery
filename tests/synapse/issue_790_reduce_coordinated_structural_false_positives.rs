//! Tests for reduced coordinated-structural false positives (Issue #790).
//!
//! GRQ-sampler analysis shows coordinated-structural candidates have a 2.3% success
//! rate (272 / 12,069). These tests verify tighter filtering.
//!
//! Issue #1058: Updated to reflect the simplified empirical discount model that
//! replaces the three-layer compound discount (per-op × pessimism × calibration).

use neat_ai_discovery::analysis::candidate_aggregation::{
    apply_operation_count_discount, validate_coordinated_candidate_gain,
};
use neat_ai_discovery::analysis::constants::{
    COORDINATED_EMPIRICAL_DISCOUNT_2OPS, COORDINATED_EMPIRICAL_DISCOUNT_4PLUS_OPS,
};
use neat_ai_discovery::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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
        operations,
        expected_creature_score_gain: gain,
        comment: Some("test coordinated module".to_string()),
    }
}

// ---------------------------------------------------------------------------
// Operation-count discount with tighter factor (Issue #790)
// ---------------------------------------------------------------------------

#[test]
fn empirical_discount_reduces_4op_candidate() {
    // Issue #1058: 4-op candidate uses the 4+-op empirical factor (0.1).
    // 0.01 × 0.1 = 0.001
    let candidate = make_multi_op_candidate(4, 0.01);
    let discounted = apply_operation_count_discount(&candidate);

    let expected = 0.01 * COORDINATED_EMPIRICAL_DISCOUNT_4PLUS_OPS;
    assert!(
        (discounted - expected).abs() < 1e-6,
        "4-op candidate should use empirical factor: expected {expected}, got {discounted}"
    );
    assert!(discounted > 0.0, "discounted gain should remain positive");
}

#[test]
fn empirical_discount_reduces_2op_candidate() {
    // Issue #1058: 2-op candidate uses the 2-op empirical factor (0.5).
    // 0.01 × 0.5 = 0.005
    let candidate = make_multi_op_candidate(2, 0.01);
    let discounted = apply_operation_count_discount(&candidate);

    let expected = 0.01 * COORDINATED_EMPIRICAL_DISCOUNT_2OPS;
    assert!(
        (discounted - expected).abs() < 1e-6,
        "2-op candidate should use empirical factor: expected {expected}, got {discounted}"
    );
}

#[test]
fn discount_increases_monotonically_with_operation_count() {
    // More operations should always produce a larger discount (lower gain).
    let gain = 0.01;
    let discount_2 = apply_operation_count_discount(&make_multi_op_candidate(2, gain));
    let discount_3 = apply_operation_count_discount(&make_multi_op_candidate(3, gain));
    let discount_4 = apply_operation_count_discount(&make_multi_op_candidate(4, gain));

    assert!(
        discount_3 < discount_2,
        "3-op ({discount_3}) should be less than 2-op ({discount_2})"
    );
    assert!(
        discount_4 < discount_3,
        "4-op ({discount_4}) should be less than 3-op ({discount_3})"
    );
}

// ---------------------------------------------------------------------------
// Minimum gain validation with lowered threshold (Issue #1058)
// ---------------------------------------------------------------------------

#[test]
fn moderate_gain_passes_for_2op_with_lowered_threshold() {
    // Issue #1058: A gain of 5e-4 on a 2-op candidate now passes since
    // the threshold was lowered to 1e-5. Discounted: 5e-4 × 0.5 = 2.5e-4 > 1e-5.
    let candidate = make_multi_op_candidate(2, 5e-4);
    let valid = validate_coordinated_candidate_gain(&candidate);
    assert!(
        valid,
        "moderate gain (5e-4) on 2-op candidate should pass with lowered threshold"
    );
}

#[test]
fn small_gain_passes_for_2op_with_lowered_threshold() {
    // Issue #1058: A gain of 1e-4 on a 2-op candidate now passes.
    // Discounted: 1e-4 × 0.5 = 5e-5 > 1e-5.
    let candidate = make_multi_op_candidate(2, 1e-4);
    let valid = validate_coordinated_candidate_gain(&candidate);
    assert!(
        valid,
        "gain 1e-4 on 2-op should now pass with lowered threshold"
    );
}

#[test]
fn strong_gain_still_passes() {
    // A strong gain of 0.01 on a 2-op candidate should still pass.
    let candidate = make_multi_op_candidate(2, 0.01);
    let valid = validate_coordinated_candidate_gain(&candidate);
    assert!(
        valid,
        "strong gain (0.01) on 2-op candidate should still pass"
    );
}

#[test]
fn tiny_gain_4op_rejected() {
    // Issue #1058: Very small gains on 4-op candidates should still be rejected.
    // 1e-6 × 0.1 = 1e-7 < 1e-5 → rejected.
    let candidate = make_multi_op_candidate(4, 1e-6);
    let valid = validate_coordinated_candidate_gain(&candidate);
    assert!(
        !valid,
        "tiny gain (1e-6) on 4-op candidate should be rejected"
    );
}

// ---------------------------------------------------------------------------
// Simplified discount model replaces compound (Issue #1058)
// ---------------------------------------------------------------------------

#[test]
fn empirical_discount_reduces_coordinated_candidate_gain() {
    // The empirical discount should reduce multi-op candidate gains.
    let candidate = make_multi_op_candidate(2, 0.01);
    let discounted = apply_operation_count_discount(&candidate);
    assert!(
        discounted < 0.01,
        "empirical discount should reduce gain: {discounted} should be < 0.01"
    );
    assert!(discounted > 0.0, "discounted gain should remain positive");
}

#[test]
fn empirical_discount_on_4op_reduces_substantially() {
    // 4-op candidate with gain 0.01 → 0.01 × 0.1 = 0.001.
    let candidate = make_multi_op_candidate(4, 0.01);
    let discounted = apply_operation_count_discount(&candidate);
    assert!(
        discounted < 0.01,
        "4-op discount should substantially reduce gain, got {discounted}"
    );
    assert!(discounted > 0.0, "discounted gain should be positive");
}

#[test]
fn empirical_discount_factors_reduce_gains() {
    // Verify empirical factors actually reduce gains (not amplify them).
    let gains = [0.001_f32, 0.01, 0.1, 1.0];
    for gain in gains {
        let candidate = make_multi_op_candidate(2, gain);
        let discounted = apply_operation_count_discount(&candidate);
        assert!(
            discounted < gain,
            "empirical discount should reduce gain {gain}, got {discounted}"
        );
    }
}
