//! Tests for reduced coordinated-structural false positives (Issue #790).
//!
//! GRQ-sampler analysis shows coordinated-structural candidates have a 2.3% success
//! rate (272 / 12,069). These tests verify tighter filtering:
//!
//! 1. Reduced COORDINATED_OPERATION_DISCOUNT applies steeper per-op discount
//! 2. Raised MIN_COORDINATED_MULTI_OP_GAIN filters near-zero predictions
//! 3. COORDINATED_PESSIMISM_DISCOUNT flat discount applied to all coordinated candidates

mod common;

use neat_ai_discovery::analysis::candidate_aggregation::{
    apply_operation_count_discount, validate_coordinated_candidate_gain,
};
use neat_ai_discovery::analysis::constants::COORDINATED_PESSIMISM_DISCOUNT;
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
fn tighter_operation_discount_reduces_4op_candidate_aggressively() {
    // Issue #790: With the reduced COORDINATED_OPERATION_DISCOUNT, a 4-op candidate
    // should receive substantially more discount than the old 0.8^3 = 0.512.
    // With 0.65, the discount is 0.65^3 ≈ 0.274.
    let candidate = make_multi_op_candidate(4, 0.01);
    let discounted = apply_operation_count_discount(&candidate);

    // Must be well below old discount level (0.01 × 0.512 = 0.00512)
    assert!(
        discounted < 0.004,
        "4-op candidate with tighter discount should be < 0.004, got {discounted}"
    );
    assert!(discounted > 0.0, "discounted gain should remain positive");
}

#[test]
fn tighter_operation_discount_reduces_2op_candidate() {
    // Issue #790: A 2-op candidate gets discount^1 applied to its gain.
    let candidate = make_multi_op_candidate(2, 0.01);
    let discounted = apply_operation_count_discount(&candidate);

    // With 0.65 factor: 0.01 × 0.65 = 0.0065
    assert!(
        discounted < 0.007,
        "2-op candidate with tighter discount should be < 0.007, got {discounted}"
    );
    assert!(
        discounted > 0.005,
        "2-op candidate should still have meaningful gain, got {discounted}"
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
// Minimum gain validation with raised threshold (Issue #790)
// ---------------------------------------------------------------------------

#[test]
fn marginal_gain_rejected_for_multi_op_with_raised_threshold() {
    // A gain of 5e-4 on a 2-op candidate should be rejected after discounting
    // because the discounted value falls below the raised minimum gain threshold.
    let candidate = make_multi_op_candidate(2, 5e-4);
    let valid = validate_coordinated_candidate_gain(&candidate);
    assert!(
        !valid,
        "marginal gain (5e-4) on 2-op candidate should be rejected with raised threshold"
    );
}

#[test]
fn previously_passing_gain_now_rejected() {
    // Under old thresholds (discount=0.8, min=1e-5): gain 1e-4 on 2-op passed.
    // Under tighter thresholds: this should now be rejected.
    let candidate = make_multi_op_candidate(2, 1e-4);
    let valid = validate_coordinated_candidate_gain(&candidate);
    assert!(
        !valid,
        "gain 1e-4 on 2-op should now be rejected with tighter thresholds"
    );
}

#[test]
fn strong_gain_still_passes_with_raised_threshold() {
    // A strong gain of 0.01 on a 2-op candidate should still pass even with
    // tighter thresholds — we only want to filter weak predictions.
    let candidate = make_multi_op_candidate(2, 0.01);
    let valid = validate_coordinated_candidate_gain(&candidate);
    assert!(
        valid,
        "strong gain (0.01) on 2-op candidate should still pass"
    );
}

#[test]
fn very_small_4op_gain_rejected() {
    // A 4-op candidate with gain 0.005 should be rejected because
    // after discounting (0.005 × ~0.274 ≈ 0.00137) it barely exceeds the
    // threshold. But gain 0.002 → 0.002 × 0.274 = 5.48e-4 < 1e-3 should fail.
    let candidate = make_multi_op_candidate(4, 0.002);
    let valid = validate_coordinated_candidate_gain(&candidate);
    assert!(
        !valid,
        "small gain (0.002) on 4-op candidate should be rejected"
    );
}

// ---------------------------------------------------------------------------
// Pessimism discount application (Issue #790)
// ---------------------------------------------------------------------------

#[test]
fn pessimism_discount_reduces_coordinated_candidate_gain() {
    // The flat pessimism discount should reduce coordinated candidate gains
    // substantially, reflecting the 2.3% success rate.
    let gain = 0.01_f32;
    let discounted = gain * COORDINATED_PESSIMISM_DISCOUNT;
    assert!(
        discounted < gain,
        "pessimism discount should reduce gain: {discounted} should be < {gain}"
    );
    assert!(discounted > 0.0, "discounted gain should remain positive");
}

#[test]
fn pessimism_discount_combined_with_operation_discount_filters_aggressively() {
    // Combined effect: a 4-op candidate with gain 0.01
    // After op discount: substantially reduced
    // After pessimism: further reduced to a small fraction of original
    let candidate = make_multi_op_candidate(4, 0.01);
    let after_op_discount = apply_operation_count_discount(&candidate);
    let after_pessimism = after_op_discount * COORDINATED_PESSIMISM_DISCOUNT;

    assert!(
        after_pessimism < 0.001,
        "combined discounts on 4-op candidate should yield < 0.001, got {after_pessimism}"
    );
    assert!(
        after_pessimism > 0.0,
        "combined discounts should still be positive"
    );
}

#[test]
fn pessimism_discount_is_strictly_less_than_one() {
    // Verify the pessimism discount actually reduces gains (not amplifies them).
    // Apply to a runtime-computed value to avoid constant-assertion lint.
    let gains = [0.001_f32, 0.01, 0.1, 1.0];
    for gain in gains {
        let discounted = gain * COORDINATED_PESSIMISM_DISCOUNT;
        assert!(
            discounted < gain,
            "pessimism discount should reduce gain {gain}, got {discounted}"
        );
    }
}
