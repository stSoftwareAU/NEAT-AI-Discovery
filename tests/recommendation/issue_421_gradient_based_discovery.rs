//! Tests for Issue #421: Gradient-based discovery — directional improvement hints.
//!
//! Current discovery analyses correlations between activations and errors but does not
//! explicitly use gradient information. This module computes local gradients
//! (∂error/∂weight) for each synapse and proposes weight adjustments in the
//! error-reducing direction.
//!
//! ## TDD Plan
//! 1. Test local gradient computation accuracy (∂error/∂weight ≈ activation × error)
//! 2. Test high-gradient synapse identification (large |gradient| = high impact)
//! 3. Test gradient-directed weight adjustment proposals
//! 4. Test that low-gradient synapses are NOT flagged
//! 5. Test candidate generation produces valid coordinated candidates
//! 6. Test edge cases: empty records, insufficient samples, non-finite values
//! 7. Test gradient vs correlation — gradient should be more directional
//! 8. Test improvement estimation scales with gradient magnitude

use crate::common::{make_creature, neuron, output, synapse};
use neat_ai_discovery::analysis::recommendation::gradient_discovery::{
    compute_synapse_gradient, detect_gradient_candidates, gradient_candidates_to_coordinated,
};
use neat_ai_discovery::types::DiscoverRecord;

// =============================================================================
// Helper: create records with specified activations and errors
// =============================================================================

fn make_record(neuron_uuid: &str, obs_index: u32, activation: f32, error: f32) -> DiscoverRecord {
    DiscoverRecord {
        obs_index,
        neuron_uuid: neuron_uuid.to_string(),
        value: Some(activation),
        activation,
        errors: vec![error],
    }
}

// =============================================================================
// Test 1: Local gradient computation accuracy
// =============================================================================

/// The local gradient ∂error/∂weight for a synapse should approximate
/// the mean of (source_activation × target_error) across samples.
#[test]
fn test_gradient_computation_basic() {
    // 10 paired samples of source activation and target error
    let activations = [1.0, 0.5, -0.5, 0.8, 0.3, 0.7, -0.3, 0.9, 0.6, 0.4];
    let errors = [0.1, 0.2, -0.1, 0.15, 0.05, 0.12, -0.08, 0.18, 0.1, 0.06];

    let source_records: Vec<DiscoverRecord> = activations
        .iter()
        .enumerate()
        .map(|(i, &act)| make_record("source", i as u32, act, 0.0))
        .collect();
    let target_records: Vec<DiscoverRecord> = errors
        .iter()
        .enumerate()
        .map(|(i, &err)| make_record("target", i as u32, 0.0, err))
        .collect();

    let gradient = compute_synapse_gradient(&source_records, &target_records);

    assert!(gradient.is_some(), "Should compute a gradient");
    let grad = gradient.unwrap();

    // Expected: mean of (activation_i × error_i)
    // = (1.0×0.1 + 0.5×0.2 + (-0.5)×(-0.1) + 0.8×0.15 + 0.3×0.05
    //    + 0.7×0.12 + (-0.3)×(-0.08) + 0.9×0.18 + 0.6×0.1 + 0.4×0.06) / 10
    // = (0.1 + 0.1 + 0.05 + 0.12 + 0.015 + 0.084 + 0.024 + 0.162 + 0.06 + 0.024) / 10
    // = 0.739 / 10 = 0.0739
    assert!(
        (grad - 0.0739).abs() < 0.01,
        "Expected gradient ~0.0739, got {grad}"
    );
}

/// Zero activations should produce zero gradient.
#[test]
fn test_gradient_zero_activations() {
    let source_records: Vec<DiscoverRecord> = (0..10)
        .map(|i| make_record("source", i, 0.0, 0.0))
        .collect();
    let target_records: Vec<DiscoverRecord> = (0..10)
        .map(|i| make_record("target", i, 0.0, 0.5))
        .collect();

    let gradient = compute_synapse_gradient(&source_records, &target_records);

    // With zero activations, gradient should be zero or None
    if let Some(grad) = gradient {
        assert!(
            grad.abs() < 0.001,
            "Zero activations should give zero gradient, got {grad}"
        );
    }
}

// =============================================================================
// Test 2: High-gradient synapse identification
// =============================================================================

/// Synapses with large |gradient| should be identified as candidates.
#[test]
fn test_detects_high_gradient_synapse() {
    let creature = make_creature(
        vec![
            neuron("input-0", "input", "IDENTITY"),
            neuron("input-1", "input", "IDENTITY"),
            output("output-0", "HARD_TANH"),
        ],
        vec![
            synapse("input-0", "output-0", 0.5),
            synapse("input-1", "output-0", 0.1),
        ],
    );

    // input-0: high activation correlated with high error → high gradient
    let input0_records: Vec<DiscoverRecord> = (0..30)
        .map(|i| {
            let act = (i as f32) / 30.0;
            make_record("input-0", i, act, 0.0)
        })
        .collect();

    // input-1: low activation, uncorrelated with error → low gradient
    let input1_records: Vec<DiscoverRecord> = (0..30)
        .map(|i| make_record("input-1", i, 0.01, 0.0))
        .collect();

    // output-0: errors correlated with input-0 activations
    let output_records: Vec<DiscoverRecord> = (0..30)
        .map(|i| {
            let err = (i as f32) / 30.0 * 0.5; // Correlated with input-0
            make_record("output-0", i, 0.0, err)
        })
        .collect();

    let records = vec![
        ("input-0".to_string(), input0_records),
        ("input-1".to_string(), input1_records),
        ("output-0".to_string(), output_records),
    ];

    let candidates = detect_gradient_candidates(&creature, &records);

    assert!(
        !candidates.is_empty(),
        "Should detect at least one gradient candidate"
    );

    // The high-gradient synapse (input-0 → output-0) should be detected
    assert!(
        candidates
            .iter()
            .any(|c| c.from_neuron_uuid == "input-0" && c.to_neuron_uuid == "output-0"),
        "Should detect input-0 → output-0 as high-gradient synapse"
    );
}

// =============================================================================
// Test 3: Gradient-directed weight adjustment proposals
// =============================================================================

/// The proposed weight delta should be in the error-reducing direction
/// (negative gradient = reduce weight along gradient descent direction).
#[test]
fn test_weight_adjustment_direction() {
    let creature = make_creature(
        vec![
            neuron("input-0", "input", "IDENTITY"),
            output("output-0", "HARD_TANH"),
        ],
        vec![synapse("input-0", "output-0", 0.5)],
    );

    // Positive gradient: increasing weight increases error
    // → should propose negative weight delta (reduce weight)
    let input_records: Vec<DiscoverRecord> = (0..30)
        .map(|i| {
            let act = (i as f32 + 1.0) / 10.0; // Positive activations
            make_record("input-0", i, act, 0.0)
        })
        .collect();

    let output_records: Vec<DiscoverRecord> = (0..30)
        .map(|i| {
            let err = (i as f32 + 1.0) / 10.0 * 0.3; // Positive errors, correlated with activation
            make_record("output-0", i, 0.0, err)
        })
        .collect();

    let records = vec![
        ("input-0".to_string(), input_records),
        ("output-0".to_string(), output_records),
    ];

    let candidates = detect_gradient_candidates(&creature, &records);

    assert!(!candidates.is_empty(), "Should detect candidates");

    let c = candidates
        .iter()
        .find(|c| c.from_neuron_uuid == "input-0")
        .expect("Should find input-0 candidate");

    // Gradient is positive (activation × error are same sign),
    // so proposed delta should be negative (gradient descent direction)
    assert!(
        c.proposed_weight_delta < 0.0,
        "Positive gradient should propose negative delta, got {}",
        c.proposed_weight_delta
    );
}

// =============================================================================
// Test 4: Low-gradient synapses are NOT flagged
// =============================================================================

/// Synapses with negligible gradient should not be detected.
#[test]
fn test_low_gradient_not_detected() {
    let creature = make_creature(
        vec![
            neuron("input-0", "input", "IDENTITY"),
            output("output-0", "HARD_TANH"),
        ],
        vec![synapse("input-0", "output-0", 0.5)],
    );

    // Activations and errors are uncorrelated → near-zero gradient
    let input_records: Vec<DiscoverRecord> = (0..30)
        .map(|i| {
            let act = if i % 2 == 0 { 0.5 } else { -0.5 };
            make_record("input-0", i, act, 0.0)
        })
        .collect();

    let output_records: Vec<DiscoverRecord> = (0..30)
        .map(|i| {
            // Errors alternate in opposite pattern to activations → zero mean product
            let err = if i % 2 == 0 { -0.01 } else { 0.01 };
            make_record("output-0", i, 0.0, err)
        })
        .collect();

    let records = vec![
        ("input-0".to_string(), input_records),
        ("output-0".to_string(), output_records),
    ];

    let candidates = detect_gradient_candidates(&creature, &records);

    assert!(
        candidates.is_empty(),
        "Low-gradient synapses should not be detected, got {} candidate(s)",
        candidates.len()
    );
}

// =============================================================================
// Test 5: Candidate conversion to coordinated candidates
// =============================================================================

/// Detected gradient candidates should produce valid coordinated candidates
/// with SetWeight operations and positive expected improvement.
#[test]
fn test_gradient_to_coordinated_candidates() {
    let creature = make_creature(
        vec![
            neuron("input-0", "input", "IDENTITY"),
            output("output-0", "HARD_TANH"),
        ],
        vec![synapse("input-0", "output-0", 0.05)],
    );

    // Strong gradient scenario
    let input_records: Vec<DiscoverRecord> = (0..30)
        .map(|i| {
            let act = (i as f32 + 1.0) / 10.0;
            make_record("input-0", i, act, 0.0)
        })
        .collect();

    let output_records: Vec<DiscoverRecord> = (0..30)
        .map(|i| {
            let err = (i as f32 + 1.0) / 10.0 * 0.3;
            make_record("output-0", i, 0.0, err)
        })
        .collect();

    let records = vec![
        ("input-0".to_string(), input_records),
        ("output-0".to_string(), output_records),
    ];

    let candidates = detect_gradient_candidates(&creature, &records);
    assert!(!candidates.is_empty(), "Should detect candidates");

    let coordinated = gradient_candidates_to_coordinated(&candidates);

    assert!(
        !coordinated.is_empty(),
        "Should produce coordinated candidates"
    );

    for c in &coordinated {
        assert!(
            c.expected_creature_score_gain > 0.0,
            "Expected improvement should be positive, got {}",
            c.expected_creature_score_gain
        );
        assert!(c.comment.is_some(), "Candidate should have a comment");
        assert!(
            !c.operations.is_empty(),
            "Candidate should have at least one operation"
        );
    }
}

// =============================================================================
// Test 6: Edge cases
// =============================================================================

/// Empty records should produce no candidates.
#[test]
fn test_gradient_discovery_empty_records_no_candidates() {
    let creature = make_creature(
        vec![
            neuron("input-0", "input", "IDENTITY"),
            output("output-0", "HARD_TANH"),
        ],
        vec![synapse("input-0", "output-0", 0.5)],
    );

    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![];
    let candidates = detect_gradient_candidates(&creature, &records);

    assert!(
        candidates.is_empty(),
        "Empty records should produce no candidates"
    );
}

/// Insufficient samples should produce no candidates.
#[test]
fn test_gradient_discovery_insufficient_samples_no_candidates() {
    let creature = make_creature(
        vec![
            neuron("input-0", "input", "IDENTITY"),
            output("output-0", "HARD_TANH"),
        ],
        vec![synapse("input-0", "output-0", 0.5)],
    );

    // Only 3 samples — below minimum threshold
    let input_records: Vec<DiscoverRecord> = (0..3)
        .map(|i| make_record("input-0", i, 0.8, 0.0))
        .collect();
    let output_records: Vec<DiscoverRecord> = (0..3)
        .map(|i| make_record("output-0", i, 0.0, 0.5))
        .collect();

    let records = vec![
        ("input-0".to_string(), input_records),
        ("output-0".to_string(), output_records),
    ];

    let candidates = detect_gradient_candidates(&creature, &records);

    assert!(
        candidates.is_empty(),
        "Insufficient samples should produce no candidates"
    );
}

/// Non-finite values should be safely handled.
#[test]
fn test_non_finite_values_handled() {
    let creature = make_creature(
        vec![
            neuron("input-0", "input", "IDENTITY"),
            output("output-0", "HARD_TANH"),
        ],
        vec![synapse("input-0", "output-0", 0.5)],
    );

    let mut input_records: Vec<DiscoverRecord> = (0..20)
        .map(|i| make_record("input-0", i, 0.5, 0.0))
        .collect();
    input_records.push(make_record("input-0", 20, f32::NAN, 0.0));
    input_records.push(make_record("input-0", 21, f32::INFINITY, 0.0));

    let mut output_records: Vec<DiscoverRecord> = (0..20)
        .map(|i| make_record("output-0", i, 0.0, 0.3))
        .collect();
    output_records.push(make_record("output-0", 20, 0.0, f32::NAN));
    output_records.push(make_record("output-0", 21, 0.0, f32::INFINITY));

    let records = vec![
        ("input-0".to_string(), input_records),
        ("output-0".to_string(), output_records),
    ];

    // Should not panic
    let _candidates = detect_gradient_candidates(&creature, &records);
}

// =============================================================================
// Test 7: Gradient provides directional information
// =============================================================================

/// Positive gradient (activations and errors positively correlated) should
/// propose weight decrease. Negative gradient should propose weight increase.
#[test]
fn test_gradient_sign_determines_direction() {
    let creature = make_creature(
        vec![
            neuron("input-0", "input", "IDENTITY"),
            output("output-0", "HARD_TANH"),
        ],
        vec![synapse("input-0", "output-0", 0.05)],
    );

    // Negative gradient: high activation with negative error
    // → should propose positive weight delta (increase weight)
    let input_records: Vec<DiscoverRecord> = (0..30)
        .map(|i| {
            let act = (i as f32 + 1.0) / 10.0;
            make_record("input-0", i, act, 0.0)
        })
        .collect();

    let output_records: Vec<DiscoverRecord> = (0..30)
        .map(|i| {
            let err = -((i as f32 + 1.0) / 10.0 * 0.3); // Negative errors
            make_record("output-0", i, 0.0, err)
        })
        .collect();

    let records = vec![
        ("input-0".to_string(), input_records),
        ("output-0".to_string(), output_records),
    ];

    let candidates = detect_gradient_candidates(&creature, &records);

    if !candidates.is_empty() {
        let c = &candidates[0];
        // Negative gradient → proposed delta should be positive
        assert!(
            c.proposed_weight_delta > 0.0,
            "Negative gradient should propose positive delta, got {}",
            c.proposed_weight_delta
        );
    }
}

// =============================================================================
// Test 8: Improvement estimation scales with gradient magnitude
// =============================================================================

/// Higher gradient magnitude should yield higher estimated improvement.
#[test]
fn test_improvement_scales_with_gradient() {
    let creature = make_creature(
        vec![
            neuron("input-0", "input", "IDENTITY"),
            neuron("input-1", "input", "IDENTITY"),
            output("output-0", "HARD_TANH"),
        ],
        vec![
            synapse("input-0", "output-0", 0.05),
            synapse("input-1", "output-0", 0.05),
        ],
    );

    // input-0: Strong gradient (high activation × high error)
    let input0_records: Vec<DiscoverRecord> = (0..30)
        .map(|i| {
            let act = (i as f32 + 1.0) / 10.0;
            make_record("input-0", i, act, 0.0)
        })
        .collect();

    // input-1: Weak gradient (low activation × low error)
    let input1_records: Vec<DiscoverRecord> = (0..30)
        .map(|i| {
            let act = 0.05 + (i as f32) / 1000.0;
            make_record("input-1", i, act, 0.0)
        })
        .collect();

    // output-0: errors scale with input-0 but not input-1
    let output_records: Vec<DiscoverRecord> = (0..30)
        .map(|i| {
            let err = (i as f32 + 1.0) / 10.0 * 0.3;
            make_record("output-0", i, 0.0, err)
        })
        .collect();

    let records = vec![
        ("input-0".to_string(), input0_records),
        ("input-1".to_string(), input1_records),
        ("output-0".to_string(), output_records),
    ];

    let candidates = detect_gradient_candidates(&creature, &records);

    let strong = candidates.iter().find(|c| c.from_neuron_uuid == "input-0");
    let weak = candidates.iter().find(|c| c.from_neuron_uuid == "input-1");

    if let Some(s) = strong {
        if let Some(w) = weak {
            assert!(
                s.estimated_improvement > w.estimated_improvement,
                "Strong gradient improvement ({}) should exceed weak ({})",
                s.estimated_improvement,
                w.estimated_improvement
            );
        }
        // Even if weak is not detected (below threshold), strong should have improvement
        assert!(
            s.estimated_improvement > 0.0,
            "Strong gradient should have positive improvement"
        );
    } else {
        panic!("Should detect the strong gradient synapse (input-0 → output-0)");
    }
}

// =============================================================================
// Test 9: Candidates sorted by improvement
// =============================================================================

/// Candidates should be sorted by estimated improvement (best first).
#[test]
fn test_gradient_discovery_candidates_sorted_by_improvement() {
    let creature = make_creature(
        vec![
            neuron("input-0", "input", "IDENTITY"),
            neuron("input-1", "input", "IDENTITY"),
            output("output-0", "HARD_TANH"),
        ],
        vec![
            synapse("input-0", "output-0", 0.05),
            synapse("input-1", "output-0", 0.05),
        ],
    );

    let input0_records: Vec<DiscoverRecord> = (0..30)
        .map(|i| {
            let act = (i as f32 + 1.0) / 10.0;
            make_record("input-0", i, act, 0.0)
        })
        .collect();
    let input1_records: Vec<DiscoverRecord> = (0..30)
        .map(|i| {
            let act = (i as f32 + 1.0) / 20.0;
            make_record("input-1", i, act, 0.0)
        })
        .collect();
    let output_records: Vec<DiscoverRecord> = (0..30)
        .map(|i| {
            let err = (i as f32 + 1.0) / 10.0 * 0.4;
            make_record("output-0", i, 0.0, err)
        })
        .collect();

    let records = vec![
        ("input-0".to_string(), input0_records),
        ("input-1".to_string(), input1_records),
        ("output-0".to_string(), output_records),
    ];

    let candidates = detect_gradient_candidates(&creature, &records);

    if candidates.len() >= 2 {
        for i in 1..candidates.len() {
            assert!(
                candidates[i - 1].estimated_improvement >= candidates[i].estimated_improvement,
                "Candidates should be sorted by improvement (descending)"
            );
        }
    }
}
