//! Tests for Issue #644: Synapse weight polarity flip candidate.
//!
//! When a synapse weight and raw gradient have the same sign, gradient descent
//! pushes the weight through zero toward the opposite sign. Many small delta
//! steps are needed. A direct polarity flip (`setWeight` to the negated value)
//! can skip that traversal entirely.
//!
//! Sign logic:
//! - gradient > 0 → ∂error/∂weight > 0 → descent direction is negative
//! - If weight is also positive, descent must cross zero → polarity flip case
//! - Same-sign weight and gradient = polarity mismatch
//!
//! ## TDD Plan
//! 1. Test detection of positive weight + positive gradient (descent crosses zero)
//! 2. Test detection of negative weight + negative gradient (descent crosses zero)
//! 3. Test that opposite-sign weight-gradient pairs are NOT flagged
//! 4. Test that inconsistent gradients are NOT flagged
//! 5. Test candidate produces negated weight via `setWeight`
//! 6. Test that near-zero weights are NOT flagged (no meaningful flip)
//! 7. Test edge cases: empty records, insufficient samples
//! 8. Test candidates are distinct from small-delta gradient proposals
//! 9. Test candidates sorted by improvement

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use crate::common::{make_creature, neuron, output, synapse};
use neat_ai_discovery::analysis::detection::weight_polarity_flip::{
    detect_weight_polarity_flip_candidates, polarity_flip_candidates_to_coordinated,
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
// Test 1: Detects clear polarity mismatch (positive weight, positive gradient)
// =============================================================================

/// When a synapse has positive weight and positive gradient (∂error/∂weight > 0),
/// gradient descent pushes the weight negative through zero.
#[test]
fn test_detects_positive_weight_positive_gradient() {
    let creature = make_creature(
        vec![
            neuron("input-0", "input", "IDENTITY"),
            output("output-0", "HARD_TANH"),
        ],
        vec![synapse("input-0", "output-0", 2.0)], // Positive weight
    );

    // Positive activations × positive errors → positive gradient
    // Descent direction is negative → pushes weight through zero
    let input_records: Vec<DiscoverRecord> = (0..30)
        .map(|i| {
            let act = (i as f32 + 1.0) / 10.0;
            make_record("input-0", i, act, 0.0)
        })
        .collect();

    let output_records: Vec<DiscoverRecord> = (0..30)
        .map(|i| {
            let err = (i as f32 + 1.0) / 10.0 * 0.5;
            make_record("output-0", i, 0.0, err)
        })
        .collect();

    let records = vec![
        ("input-0".to_string(), input_records),
        ("output-0".to_string(), output_records),
    ];

    let candidates = detect_weight_polarity_flip_candidates(&creature, &records);

    assert!(
        !candidates.is_empty(),
        "Should detect polarity flip for positive weight with positive gradient"
    );

    let c = &candidates[0];
    assert_eq!(c.from_neuron_uuid, "input-0");
    assert_eq!(c.to_neuron_uuid, "output-0");
    assert!(c.current_weight > 0.0, "Current weight should be positive");
    assert!(c.mean_gradient > 0.0, "Mean gradient should be positive");
}

// =============================================================================
// Test 2: Detects reverse polarity mismatch (negative weight, negative gradient)
// =============================================================================

/// When a synapse has negative weight and negative gradient (∂error/∂weight < 0),
/// gradient descent pushes the weight positive through zero.
#[test]
fn test_detects_negative_weight_negative_gradient() {
    let creature = make_creature(
        vec![
            neuron("input-0", "input", "IDENTITY"),
            output("output-0", "HARD_TANH"),
        ],
        vec![synapse("input-0", "output-0", -2.0)], // Negative weight
    );

    // Positive activations × negative errors → negative gradient
    // Descent direction is positive → pushes weight through zero
    let input_records: Vec<DiscoverRecord> = (0..30)
        .map(|i| {
            let act = (i as f32 + 1.0) / 10.0;
            make_record("input-0", i, act, 0.0)
        })
        .collect();

    let output_records: Vec<DiscoverRecord> = (0..30)
        .map(|i| {
            let err = -((i as f32 + 1.0) / 10.0 * 0.5);
            make_record("output-0", i, 0.0, err)
        })
        .collect();

    let records = vec![
        ("input-0".to_string(), input_records),
        ("output-0".to_string(), output_records),
    ];

    let candidates = detect_weight_polarity_flip_candidates(&creature, &records);

    assert!(
        !candidates.is_empty(),
        "Should detect polarity flip for negative weight with negative gradient"
    );

    let c = &candidates[0];
    assert!(c.current_weight < 0.0, "Current weight should be negative");
    assert!(c.mean_gradient < 0.0, "Mean gradient should be negative");
}

// =============================================================================
// Test 3: Opposite-sign weight-gradient pairs are NOT flagged
// =============================================================================

/// When weight is positive and gradient is negative, descent pushes weight MORE
/// positive (further from zero). No flip is needed.
#[test]
fn test_opposite_sign_weight_gradient_not_flagged() {
    let creature = make_creature(
        vec![
            neuron("input-0", "input", "IDENTITY"),
            output("output-0", "HARD_TANH"),
        ],
        vec![synapse("input-0", "output-0", 2.0)], // Positive weight
    );

    // Positive activations × negative errors → negative gradient
    // Descent direction is positive → weight increases, moves away from zero
    let input_records: Vec<DiscoverRecord> = (0..30)
        .map(|i| {
            let act = (i as f32 + 1.0) / 10.0;
            make_record("input-0", i, act, 0.0)
        })
        .collect();

    let output_records: Vec<DiscoverRecord> = (0..30)
        .map(|i| {
            let err = -((i as f32 + 1.0) / 10.0 * 0.5);
            make_record("output-0", i, 0.0, err)
        })
        .collect();

    let records = vec![
        ("input-0".to_string(), input_records),
        ("output-0".to_string(), output_records),
    ];

    let candidates = detect_weight_polarity_flip_candidates(&creature, &records);

    assert!(
        candidates.is_empty(),
        "Opposite-sign weight-gradient should not trigger polarity flip, got {} candidate(s)",
        candidates.len()
    );
}

// =============================================================================
// Test 4: Inconsistent gradients are NOT flagged
// =============================================================================

/// When the gradient direction is noisy (low consistency), no polarity flip
/// should be proposed even if the mean gradient has the same sign as the weight.
#[test]
fn test_inconsistent_gradient_not_flagged() {
    let creature = make_creature(
        vec![
            neuron("input-0", "input", "IDENTITY"),
            output("output-0", "HARD_TANH"),
        ],
        vec![synapse("input-0", "output-0", 2.0)],
    );

    // Alternating error signs → gradient products are noisy
    let input_records: Vec<DiscoverRecord> = (0..30)
        .map(|i| make_record("input-0", i, 1.0, 0.0))
        .collect();

    let output_records: Vec<DiscoverRecord> = (0..30)
        .map(|i| {
            let err = if i % 2 == 0 { 0.5 } else { -0.4 };
            make_record("output-0", i, 0.0, err)
        })
        .collect();

    let records = vec![
        ("input-0".to_string(), input_records),
        ("output-0".to_string(), output_records),
    ];

    let candidates = detect_weight_polarity_flip_candidates(&creature, &records);

    assert!(
        candidates.is_empty(),
        "Inconsistent gradient should not trigger polarity flip, got {} candidate(s)",
        candidates.len()
    );
}

// =============================================================================
// Test 5: Candidate produces negated weight via setWeight
// =============================================================================

/// The coordinated candidate should propose `setWeight` with the negated weight.
#[test]
fn test_candidate_produces_negated_weight() {
    let creature = make_creature(
        vec![
            neuron("input-0", "input", "IDENTITY"),
            output("output-0", "HARD_TANH"),
        ],
        vec![synapse("input-0", "output-0", 2.0)],
    );

    let input_records: Vec<DiscoverRecord> = (0..30)
        .map(|i| {
            let act = (i as f32 + 1.0) / 10.0;
            make_record("input-0", i, act, 0.0)
        })
        .collect();

    let output_records: Vec<DiscoverRecord> = (0..30)
        .map(|i| {
            let err = (i as f32 + 1.0) / 10.0 * 0.5;
            make_record("output-0", i, 0.0, err)
        })
        .collect();

    let records = vec![
        ("input-0".to_string(), input_records),
        ("output-0".to_string(), output_records),
    ];

    let candidates = detect_weight_polarity_flip_candidates(&creature, &records);
    assert!(!candidates.is_empty(), "Should detect candidate");

    let coordinated = polarity_flip_candidates_to_coordinated(&candidates);
    assert!(
        !coordinated.is_empty(),
        "Should produce coordinated candidates"
    );

    let coord = &coordinated[0];
    assert_eq!(coord.operations.len(), 1, "Should have one operation");
    assert!(
        coord.expected_creature_score_gain > 0.0,
        "Should have positive expected improvement"
    );

    use neat_ai_discovery::CoordinatedStructuralOpJson;
    match &coord.operations[0] {
        CoordinatedStructuralOpJson::SetWeight {
            from_neuron_uuid,
            to_neuron_uuid,
            weight,
        } => {
            assert_eq!(from_neuron_uuid, "input-0");
            assert_eq!(to_neuron_uuid, "output-0");
            assert!(
                (*weight - (-2.0)).abs() < 0.01,
                "Weight should be negated: expected -2.0, got {weight}"
            );
        }
        other => panic!("Expected SetWeight operation, got {other:?}"),
    }
}

// =============================================================================
// Test 6: Near-zero weights are NOT flagged
// =============================================================================

/// Weights close to zero have no meaningful polarity to flip.
#[test]
fn test_near_zero_weight_not_flagged() {
    let creature = make_creature(
        vec![
            neuron("input-0", "input", "IDENTITY"),
            output("output-0", "HARD_TANH"),
        ],
        vec![synapse("input-0", "output-0", 0.01)],
    );

    let input_records: Vec<DiscoverRecord> = (0..30)
        .map(|i| {
            let act = (i as f32 + 1.0) / 10.0;
            make_record("input-0", i, act, 0.0)
        })
        .collect();

    let output_records: Vec<DiscoverRecord> = (0..30)
        .map(|i| {
            let err = (i as f32 + 1.0) / 10.0 * 0.5;
            make_record("output-0", i, 0.0, err)
        })
        .collect();

    let records = vec![
        ("input-0".to_string(), input_records),
        ("output-0".to_string(), output_records),
    ];

    let candidates = detect_weight_polarity_flip_candidates(&creature, &records);

    assert!(
        candidates.is_empty(),
        "Near-zero weight should not trigger polarity flip, got {} candidate(s)",
        candidates.len()
    );
}

// =============================================================================
// Test 7: Edge cases — empty records, insufficient samples
// =============================================================================

/// Empty records should produce no candidates.
#[test]
fn test_weight_polarity_flip_empty_records_no_candidates() {
    let creature = make_creature(
        vec![
            neuron("input-0", "input", "IDENTITY"),
            output("output-0", "HARD_TANH"),
        ],
        vec![synapse("input-0", "output-0", 2.0)],
    );

    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![];
    let candidates = detect_weight_polarity_flip_candidates(&creature, &records);

    assert!(
        candidates.is_empty(),
        "Empty records should produce no candidates"
    );
}

/// Insufficient samples should produce no candidates.
#[test]
fn test_weight_polarity_flip_insufficient_samples_no_candidates() {
    let creature = make_creature(
        vec![
            neuron("input-0", "input", "IDENTITY"),
            output("output-0", "HARD_TANH"),
        ],
        vec![synapse("input-0", "output-0", 2.0)],
    );

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

    let candidates = detect_weight_polarity_flip_candidates(&creature, &records);

    assert!(
        candidates.is_empty(),
        "Insufficient samples should produce no candidates"
    );
}

// =============================================================================
// Test 8: Candidates are distinct from small-delta gradient proposals
// =============================================================================

/// The polarity flip candidate should propose the full negated weight,
/// not a small delta adjustment like the gradient discovery module.
#[test]
fn test_flip_is_distinct_from_gradient_delta() {
    let creature = make_creature(
        vec![
            neuron("input-0", "input", "IDENTITY"),
            output("output-0", "HARD_TANH"),
        ],
        vec![synapse("input-0", "output-0", 2.0)],
    );

    let input_records: Vec<DiscoverRecord> = (0..30)
        .map(|i| {
            let act = (i as f32 + 1.0) / 10.0;
            make_record("input-0", i, act, 0.0)
        })
        .collect();

    let output_records: Vec<DiscoverRecord> = (0..30)
        .map(|i| {
            let err = (i as f32 + 1.0) / 10.0 * 0.5;
            make_record("output-0", i, 0.0, err)
        })
        .collect();

    let records = vec![
        ("input-0".to_string(), input_records),
        ("output-0".to_string(), output_records),
    ];

    // Get polarity flip candidates
    let flip_candidates = detect_weight_polarity_flip_candidates(&creature, &records);
    assert!(!flip_candidates.is_empty(), "Should detect flip candidate");

    // Get regular gradient candidates
    use neat_ai_discovery::analysis::recommendation::gradient_discovery::detect_gradient_candidates;
    let gradient_candidates = detect_gradient_candidates(&creature, &records);

    let flip_coord = polarity_flip_candidates_to_coordinated(&flip_candidates);
    assert!(!flip_coord.is_empty());

    use neat_ai_discovery::CoordinatedStructuralOpJson;
    if let CoordinatedStructuralOpJson::SetWeight {
        weight: flip_weight,
        ..
    } = &flip_coord[0].operations[0]
    {
        // Flip weight should be -2.0 (negated from +2.0)
        assert!(
            (*flip_weight - (-2.0)).abs() < 0.01,
            "Flip weight should be -2.0, got {flip_weight}"
        );

        // If gradient also detected this synapse, its proposed weight should be different
        if let Some(gc) = gradient_candidates
            .iter()
            .find(|c| c.from_neuron_uuid == "input-0")
        {
            let gradient_new_weight = gc.current_weight + gc.proposed_weight_delta;
            assert!(
                (gradient_new_weight - flip_weight).abs() > 0.1,
                "Flip weight ({flip_weight}) should be distinct from gradient weight ({gradient_new_weight})"
            );
        }
    }
}

// =============================================================================
// Test 9: Candidates sorted by estimated improvement
// =============================================================================

/// Candidates should be sorted by estimated improvement (best first).
#[test]
fn test_weight_polarity_flip_candidates_sorted_by_improvement() {
    let creature = make_creature(
        vec![
            neuron("input-0", "input", "IDENTITY"),
            neuron("input-1", "input", "IDENTITY"),
            output("output-0", "HARD_TANH"),
        ],
        vec![
            synapse("input-0", "output-0", 3.0),
            synapse("input-1", "output-0", 1.5),
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
            let act = (i as f32 + 1.0) / 10.0;
            make_record("input-1", i, act, 0.0)
        })
        .collect();

    let output_records: Vec<DiscoverRecord> = (0..30)
        .map(|i| {
            let err = (i as f32 + 1.0) / 10.0 * 0.5;
            make_record("output-0", i, 0.0, err)
        })
        .collect();

    let records = vec![
        ("input-0".to_string(), input0_records),
        ("input-1".to_string(), input1_records),
        ("output-0".to_string(), output_records),
    ];

    let candidates = detect_weight_polarity_flip_candidates(&creature, &records);

    if candidates.len() >= 2 {
        for i in 1..candidates.len() {
            assert!(
                candidates[i - 1].estimated_improvement >= candidates[i].estimated_improvement,
                "Candidates should be sorted by improvement (descending)"
            );
        }
    }
}
