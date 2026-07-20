//! Tests for improved coordinated-structural success rate (Issue #732).
//!
//! Validates that multi-operation coordinated candidates are properly discounted
//! for compounding prediction uncertainty, and that ensemble boost is capped
//! for complex candidates.

use neat_ai_discovery::analysis::candidate_aggregation::{
    apply_operation_count_discount, validate_coordinated_candidate_gain,
};
use neat_ai_discovery::analysis::ensemble_scoring::apply_ensemble_scoring;
use neat_ai_discovery::analysis::module_weights::ModuleOutcomeTracker;
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

    // Build a realistic multi-operation candidate (RemoveSynapse + AddNeuron + AddSynapse × N)
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
// Operation-count discount tests
// ---------------------------------------------------------------------------

#[test]
fn single_operation_has_no_discount() {
    let candidate = make_single_op_candidate(0.01);
    let discounted = apply_operation_count_discount(&candidate);
    // Single operation should not be discounted
    assert!(
        (discounted - 0.01).abs() < 1e-6,
        "single-op candidate should not be discounted, got {discounted}"
    );
}

#[test]
fn multi_operation_candidate_is_discounted() {
    let candidate = make_multi_op_candidate(4, 0.01);
    let discounted = apply_operation_count_discount(&candidate);
    // 4 operations should receive a meaningful discount
    assert!(
        discounted < 0.01,
        "4-op candidate should be discounted below 0.01, got {discounted}"
    );
    assert!(
        discounted > 0.0,
        "discounted gain should remain positive, got {discounted}"
    );
}

#[test]
fn more_operations_produce_larger_discount() {
    let two_ops = make_multi_op_candidate(2, 0.01);
    let four_ops = make_multi_op_candidate(4, 0.01);

    let discount_2 = apply_operation_count_discount(&two_ops);
    let discount_4 = apply_operation_count_discount(&four_ops);

    assert!(
        discount_4 < discount_2,
        "4-op discount ({discount_4}) should be less than 2-op discount ({discount_2})"
    );
}

// ---------------------------------------------------------------------------
// Minimum gain validation tests
// ---------------------------------------------------------------------------

#[test]
fn tiny_gain_rejected_for_multi_op_candidate() {
    // A very small gain (e.g. 1e-7) on a 4-op candidate should be rejected
    let candidate = make_multi_op_candidate(4, 1e-7);
    let valid = validate_coordinated_candidate_gain(&candidate);
    assert!(!valid, "tiny gain on multi-op candidate should be rejected");
}

#[test]
fn reasonable_gain_accepted_for_multi_op_candidate() {
    // Issue #1058: With simplified empirical factors and lowered threshold,
    // a reasonable gain of 0.01 on a 4-op candidate should pass.
    // 0.01 × 0.1 (4+-op factor) = 0.001 > 1e-5 (threshold).
    let candidate = make_multi_op_candidate(4, 0.01);
    let valid = validate_coordinated_candidate_gain(&candidate);
    assert!(valid, "reasonable gain on multi-op candidate should pass");
}

#[test]
fn single_op_with_small_gain_accepted() {
    // Single-op candidates have lower thresholds
    let candidate = make_single_op_candidate(1e-6);
    let valid = validate_coordinated_candidate_gain(&candidate);
    assert!(
        valid,
        "single-op candidate with small positive gain should pass"
    );
}

// ---------------------------------------------------------------------------
// Ensemble boost capping for multi-operation candidates
// ---------------------------------------------------------------------------

#[test]
fn ensemble_boost_capped_for_multi_operation_candidates() {
    // Two modules agree on a 4-operation coordinated candidate.
    let candidates = vec![make_multi_op_candidate(4, 0.01), {
        let mut c = make_multi_op_candidate(4, 0.012);
        c.comment = Some("another module".to_string());
        c
    }];

    let result = apply_ensemble_scoring(candidates, &ModuleOutcomeTracker::default());

    // The boost for multi-op candidates should be capped lower than for single-op.
    // Single-op 2-module agreement gives ~15% boost (0.012 * 1.15 = 0.0138).
    // Multi-op should get a reduced boost.
    let best = result
        .candidates
        .iter()
        .max_by(|a, b| {
            a.expected_creature_score_gain
                .total_cmp(&b.expected_creature_score_gain)
        })
        .expect("should have candidates");

    // The boosted score should be less than the uncapped single-op boost would give
    let uncapped_single_op_boost = 0.012 * (1.0 + 0.3 * 0.5); // 1.15 × 0.012 = 0.0138
    assert!(
        best.expected_creature_score_gain <= uncapped_single_op_boost,
        "multi-op ensemble boost should be capped, got {} (uncapped would be {})",
        best.expected_creature_score_gain,
        uncapped_single_op_boost
    );
}

#[test]
fn single_op_ensemble_boost_unchanged() {
    // Two modules agree on a single-operation candidate — boost should work normally.
    let candidates = vec![make_single_op_candidate(0.02), {
        let mut c = make_single_op_candidate(0.03);
        c.comment = Some("another module".to_string());
        c
    }];

    let result = apply_ensemble_scoring(candidates, &ModuleOutcomeTracker::default());

    let best = result
        .candidates
        .iter()
        .max_by(|a, b| {
            a.expected_creature_score_gain
                .total_cmp(&b.expected_creature_score_gain)
        })
        .expect("should have candidates");

    // Single-op agreement should still boost above best individual
    assert!(
        best.expected_creature_score_gain > 0.03,
        "single-op ensemble should boost above 0.03, got {}",
        best.expected_creature_score_gain
    );
}
