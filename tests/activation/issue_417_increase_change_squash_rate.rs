//! Tests for Issue #417: Increase change-squash suggestion rate.
//!
//! The change-squash discovery type has an 18.2% success rate (highest among active types)
//! but very low volume — only 11 total samples across all experiments. This issue aims to
//! increase the suggestion rate by:
//!
//! 1. Lowering saturation detection thresholds to catch "near-saturated" neurons
//! 2. Lowering oscillation detection thresholds to catch more oscillating neurons
//! 3. Expanding activation function recommendations (not just IDENTITY/ABSOLUTE)
//! 4. Integrating the proactive activation recommendation engine into the pipeline
//!
//! ## TDD Plan
//! - Test near-saturation detection at lower thresholds (0.85 instead of 0.95)
//! - Test oscillation detection at lower thresholds (0.15 instead of 0.3)
//! - Test expanded activation function recommendations
//! - Test proactive activation recommendation integration

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use neat_ai_discovery::analysis::detection::oscillating_neuron::detect_oscillating_neurons;
use neat_ai_discovery::analysis::detection::saturation::{
    detect_saturated_neurons, saturated_neurons_to_coordinated_candidates,
};
use neat_ai_discovery::analysis::recommendation::activation_recommendation::{
    recommend_activation_function, recommendations_to_coordinated_candidates,
};
use neat_ai_discovery::types::DiscoverRecord;

/// Helper: create a `DiscoverRecord` for a neuron with given activation.
fn record(
    neuron_uuid: &str,
    obs_index: u32,
    activation: f32,
    value: Option<f32>,
) -> DiscoverRecord {
    DiscoverRecord {
        obs_index,
        neuron_uuid: neuron_uuid.to_string(),
        value,
        activation,
        errors: vec![0.01],
    }
}

// =============================================================================
// Near-Saturation Detection Tests (lowered thresholds)
// =============================================================================

/// Test: TANH neuron at 0.90 (near saturation) should now be detected.
///
/// Previously, the threshold was 0.95, which meant neurons at 0.90 were missed.
/// With the lowered threshold of 0.85, these "approaching saturation" neurons
/// should now be caught and recommended for activation function changes.
#[test]
fn test_detects_tanh_near_saturation_at_090() {
    // TANH neuron at 0.90 — approaching saturation but not fully saturated
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            record(
                "hidden-near-sat",
                i,
                0.90 + 0.005 * (i as f32 / 100.0),
                Some(2.0 + i as f32 * 0.01),
            )
        })
        .collect();

    let candidates = detect_saturated_neurons(
        &[("hidden-near-sat".to_string(), "TANH".to_string(), 0.0)],
        &[("hidden-near-sat".to_string(), records)],
    );

    assert!(
        !candidates.is_empty(),
        "Should detect near-saturated TANH neuron at mean activation ~0.90"
    );
    let c = &candidates[0];
    assert_eq!(c.neuron_uuid, "hidden-near-sat");
    assert!(
        c.mean_activation > 0.85,
        "Mean activation should be > 0.85, got {}",
        c.mean_activation
    );
}

/// Test: TANH neuron at negative near-saturation (-0.88) should be detected.
#[test]
fn test_detects_tanh_near_saturation_at_negative_088() {
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            record(
                "hidden-neg-near-sat",
                i,
                -0.88 - 0.005 * (i as f32 / 100.0),
                Some(-1.5 - i as f32 * 0.01),
            )
        })
        .collect();

    let candidates = detect_saturated_neurons(
        &[("hidden-neg-near-sat".to_string(), "TANH".to_string(), 0.0)],
        &[("hidden-neg-near-sat".to_string(), records)],
    );

    assert!(
        !candidates.is_empty(),
        "Should detect near-saturated TANH neuron at mean activation ~-0.88"
    );
}

/// Test: LOGISTIC neuron at 0.92 (near saturation) should be detected.
#[test]
fn test_detects_logistic_near_saturation_at_092() {
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            record(
                "hidden-log-near-sat",
                i,
                0.92 + 0.003 * (i as f32 / 100.0),
                Some(3.0 + i as f32 * 0.01),
            )
        })
        .collect();

    let candidates = detect_saturated_neurons(
        &[(
            "hidden-log-near-sat".to_string(),
            "LOGISTIC".to_string(),
            0.0,
        )],
        &[("hidden-log-near-sat".to_string(), records)],
    );

    assert!(
        !candidates.is_empty(),
        "Should detect near-saturated LOGISTIC neuron at mean activation ~0.92"
    );
}

/// Test: LOGISTIC neuron at 0.08 (near lower saturation) should be detected.
#[test]
fn test_detects_logistic_near_saturation_at_lower_bound() {
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            record(
                "hidden-log-lower-sat",
                i,
                0.08 - 0.003 * (i as f32 / 100.0),
                Some(-2.5 - i as f32 * 0.01),
            )
        })
        .collect();

    let candidates = detect_saturated_neurons(
        &[(
            "hidden-log-lower-sat".to_string(),
            "LOGISTIC".to_string(),
            0.0,
        )],
        &[("hidden-log-lower-sat".to_string(), records)],
    );

    assert!(
        !candidates.is_empty(),
        "Should detect near-saturated LOGISTIC neuron at mean activation ~0.08"
    );
}

/// Test: Neuron at 0.70 should NOT be detected (still well within active range).
#[test]
fn test_does_not_detect_tanh_at_070() {
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            record(
                "hidden-active",
                i,
                0.70 + 0.005 * (i as f32 / 100.0),
                Some(1.0 + i as f32 * 0.01),
            )
        })
        .collect();

    let candidates = detect_saturated_neurons(
        &[("hidden-active".to_string(), "TANH".to_string(), 0.0)],
        &[("hidden-active".to_string(), records)],
    );

    assert!(
        candidates.is_empty(),
        "Should not detect TANH neuron at 0.70 as saturated"
    );
}

/// Test: Near-saturation candidates should have lower estimated improvement
/// than fully-saturated candidates, reflecting the lower severity.
#[test]
fn test_near_saturation_has_lower_improvement_than_full() {
    // Fully saturated at 0.999
    let full_sat_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| record("full-sat", i, 0.999, Some(10.0)))
        .collect();

    // Near-saturated at 0.90
    let near_sat_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| record("near-sat", i, 0.90, Some(2.0)))
        .collect();

    let full_candidates = detect_saturated_neurons(
        &[("full-sat".to_string(), "TANH".to_string(), 0.0)],
        &[("full-sat".to_string(), full_sat_records)],
    );
    let near_candidates = detect_saturated_neurons(
        &[("near-sat".to_string(), "TANH".to_string(), 0.0)],
        &[("near-sat".to_string(), near_sat_records)],
    );

    assert!(
        !full_candidates.is_empty(),
        "Full saturation should be detected"
    );
    assert!(
        !near_candidates.is_empty(),
        "Near saturation should be detected"
    );

    assert!(
        full_candidates[0].estimated_improvement > near_candidates[0].estimated_improvement,
        "Full saturation ({}) should have higher estimated improvement than near-saturation ({})",
        full_candidates[0].estimated_improvement,
        near_candidates[0].estimated_improvement
    );
}

// =============================================================================
// Lower Oscillation Threshold Tests
// =============================================================================

/// Test: Neuron with 20% sign change fraction (previously below 30% threshold)
/// should now be detected as oscillating.
#[test]
fn test_detects_mild_oscillation_at_020_sign_change_fraction() {
    // Create a pattern with ~20% sign changes: mostly positive with occasional negatives
    let mut records = Vec::new();
    for i in 0..100u32 {
        // Every 5th sample is negative, creating ~20% sign change fraction
        let activation = if i % 5 == 0 { -0.5 } else { 0.5 };
        records.push(record("hidden-mild-osc", i, activation, Some(activation)));
    }

    let candidates = detect_oscillating_neurons(
        &[("hidden-mild-osc".to_string(), "TANH".to_string(), 0.0)],
        &[("hidden-mild-osc".to_string(), records)],
    );

    assert!(
        !candidates.is_empty(),
        "Should detect mildly oscillating neuron with ~20% sign change fraction"
    );
}

/// Test: Neuron with 15% minority sign fraction (previously below 20% threshold)
/// should now be detected.
#[test]
fn test_detects_oscillation_with_lower_minority_fraction() {
    // Pattern: 85% positive, 15% negative but with frequent sign changes
    let mut records = Vec::new();
    for i in 0..100u32 {
        // Create alternating pattern in the first 30 records, rest positive
        let activation = if i < 30 && i % 2 == 0 { -0.5 } else { 0.5 };
        records.push(record(
            "hidden-low-minority",
            i,
            activation,
            Some(activation),
        ));
    }

    let candidates = detect_oscillating_neurons(
        &[("hidden-low-minority".to_string(), "TANH".to_string(), 0.0)],
        &[("hidden-low-minority".to_string(), records)],
    );

    assert!(
        !candidates.is_empty(),
        "Should detect oscillation with lower minority sign fraction"
    );
}

/// Test: Completely stable neuron (all positive) should NOT be detected even
/// with lowered thresholds.
#[test]
fn test_does_not_detect_stable_neuron_with_lowered_thresholds() {
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| record("hidden-stable", i, 0.5, Some(0.5)))
        .collect();

    let candidates = detect_oscillating_neurons(
        &[("hidden-stable".to_string(), "TANH".to_string(), 0.0)],
        &[("hidden-stable".to_string(), records)],
    );

    assert!(
        candidates.is_empty(),
        "Stable neuron should not be detected as oscillating"
    );
}

// =============================================================================
// Expanded Activation Function Recommendation Tests
// =============================================================================

/// Test: Saturated TANH neuron should recommend more than just IDENTITY.
/// The recommendation should consider RELU, LEAKYRELU, or other alternatives
/// based on the activation pattern.
#[test]
fn test_saturation_recommends_expanded_activations() {
    // Positively saturated TANH neuron
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| record("hidden-exp-act", i, 0.98, Some(5.0)))
        .collect();

    let candidates = detect_saturated_neurons(
        &[("hidden-exp-act".to_string(), "TANH".to_string(), 0.0)],
        &[("hidden-exp-act".to_string(), records)],
    );

    assert!(!candidates.is_empty(), "Should detect saturated neuron");

    // Convert to coordinated candidates and check that we get candidates
    let coordinated = saturated_neurons_to_coordinated_candidates(&candidates);
    assert!(
        !coordinated.is_empty(),
        "Should produce coordinated structural candidates"
    );

    // At least one candidate should include a changeSquash operation
    let has_change_squash = coordinated.iter().any(|c| {
        c.operations.iter().any(|op| {
            matches!(
                op,
                neat_ai_discovery::CoordinatedStructuralOpJson::ChangeSquash { .. }
            )
        })
    });
    assert!(has_change_squash, "Should include changeSquash operations");
}

/// Test: Oscillating neuron with LOGISTIC should recommend ABSOLUTE not just RELU.
#[test]
fn test_oscillation_recommends_appropriate_activation_for_logistic() {
    let mut records = Vec::new();
    for i in 0..100u32 {
        let activation = if i % 2 == 0 { 0.3 } else { -0.3 };
        records.push(record("hidden-osc-log", i, activation, Some(activation)));
    }

    let candidates = detect_oscillating_neurons(
        &[("hidden-osc-log".to_string(), "LOGISTIC".to_string(), 0.0)],
        &[("hidden-osc-log".to_string(), records)],
    );

    assert!(!candidates.is_empty(), "Should detect oscillating neuron");
    // LOGISTIC is not in the symmetric group, so should recommend RELU
    assert_eq!(
        candidates[0].recommended_squash, "RELU",
        "LOGISTIC oscillation should recommend RELU"
    );
}

// =============================================================================
// Proactive Activation Recommendation Tests
// =============================================================================

/// Test: Proactive activation recommendation should produce changeSquash candidates
/// for neurons where the current activation is suboptimal for the input distribution.
#[test]
fn test_proactive_recommendation_for_suboptimal_activation() {
    // Create a sparse input pattern — many zeros with occasional non-zero values
    // This pattern should recommend RELU over TANH
    let records: Vec<DiscoverRecord> = (0..50)
        .map(|i| {
            let activation = if i % 4 == 0 { 1.0 } else { 0.0 };
            record("hidden-sparse-input", i, activation, Some(activation))
        })
        .collect();

    let recommendation = recommend_activation_function(&records, "TANH");

    assert!(
        recommendation.is_some(),
        "Should produce a recommendation for TANH on sparse inputs"
    );

    let rec = recommendation.unwrap();
    assert_eq!(rec.current_squash, "TANH");
    assert!(
        rec.expected_improvement > 0.0,
        "Expected improvement should be positive"
    );
}

/// Test: Proactive recommendation converts to coordinated structural candidates.
#[test]
fn test_proactive_recommendations_to_coordinated_candidates() {
    let records: Vec<DiscoverRecord> = (0..50)
        .map(|i| {
            let activation = if i % 4 == 0 { 1.0 } else { 0.0 };
            record("hidden-sparse-convert", i, activation, Some(activation))
        })
        .collect();

    let recommendation = recommend_activation_function(&records, "TANH");
    if let Some(rec) = recommendation {
        let candidates = recommendations_to_coordinated_candidates(&[rec]);
        assert!(
            !candidates.is_empty(),
            "Should convert recommendations to coordinated candidates"
        );

        // Check that the candidate has a changeSquash operation
        let has_change_squash = candidates[0].operations.iter().any(|op| {
            matches!(
                op,
                neat_ai_discovery::CoordinatedStructuralOpJson::ChangeSquash { .. }
            )
        });
        assert!(has_change_squash, "Should include changeSquash operation");
    }
}

/// Test: Near-saturation detection produces more candidates than full-saturation-only
/// detection. This validates that the lowered thresholds increase volume.
#[test]
fn test_lowered_thresholds_increase_candidate_volume() {
    // Create several neurons at different saturation levels
    let neurons = vec![
        ("full-sat-1".to_string(), "TANH".to_string(), 0.0f32),
        ("near-sat-1".to_string(), "TANH".to_string(), 0.0f32),
        ("near-sat-2".to_string(), "TANH".to_string(), 0.0f32),
        ("active-1".to_string(), "TANH".to_string(), 0.0f32),
    ];

    let neuron_records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        // Fully saturated at 0.99
        (
            "full-sat-1".to_string(),
            (0..100u32)
                .map(|i| record("full-sat-1", i, 0.99, Some(5.0)))
                .collect(),
        ),
        // Near-saturated at 0.90
        (
            "near-sat-1".to_string(),
            (0..100u32)
                .map(|i| record("near-sat-1", i, 0.90, Some(2.5)))
                .collect(),
        ),
        // Near-saturated at 0.87
        (
            "near-sat-2".to_string(),
            (0..100u32)
                .map(|i| record("near-sat-2", i, 0.87, Some(2.0)))
                .collect(),
        ),
        // Active at 0.50
        (
            "active-1".to_string(),
            (0..100u32)
                .map(|i| record("active-1", i, 0.50, Some(0.5)))
                .collect(),
        ),
    ];

    let candidates = detect_saturated_neurons(&neurons, &neuron_records);

    // With lowered thresholds (0.85), we should detect 3 neurons:
    // full-sat-1 (0.99), near-sat-1 (0.90), near-sat-2 (0.87)
    // But NOT active-1 (0.50)
    assert!(
        candidates.len() >= 3,
        "Should detect at least 3 saturated/near-saturated neurons with lowered thresholds, got {}",
        candidates.len()
    );

    // Verify active neuron is not included
    assert!(
        !candidates.iter().any(|c| c.neuron_uuid == "active-1"),
        "Active neuron at 0.50 should not be detected"
    );
}

/// Test: SOFTSIGN near-saturation detection at 0.85
#[test]
fn test_detects_softsign_near_saturation() {
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| record("hidden-softsign-sat", i, 0.88, Some(7.0)))
        .collect();

    let candidates = detect_saturated_neurons(
        &[(
            "hidden-softsign-sat".to_string(),
            "SOFTSIGN".to_string(),
            0.0,
        )],
        &[("hidden-softsign-sat".to_string(), records)],
    );

    assert!(
        !candidates.is_empty(),
        "Should detect near-saturated SOFTSIGN neuron at 0.88"
    );
}
