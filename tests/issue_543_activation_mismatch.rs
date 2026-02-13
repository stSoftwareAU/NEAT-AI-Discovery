//! Issue #543: Integration tests for activation mismatch detection.
//!
//! Tests the `activation_mismatch` detection module which identifies neurons
//! whose current activation function is poorly suited to their observed
//! activation range, and recommends better-matching alternatives.
//!
//! These tests exercise real detection and conversion functions with test data.

use neat_ai_discovery::analysis::detection::activation_mismatch::{
    ActivationMismatchCandidate, MismatchKind, activation_mismatch_to_coordinated_candidates,
    detect_activation_mismatches,
};
use neat_ai_discovery::types::DiscoverRecord;

// =============================================================================
// Helpers
// =============================================================================

fn make_record(uuid: &str, idx: u32, value: Option<f32>, activation: f32) -> DiscoverRecord {
    DiscoverRecord {
        obs_index: idx,
        neuron_uuid: uuid.to_string(),
        value,
        activation,
        errors: vec![0.05],
    }
}

/// Build hidden neuron tuples in the format expected by detection modules.
fn hidden_neuron(uuid: &str, squash: &str, bias: f32) -> (String, String, f32) {
    (uuid.to_string(), squash.to_string(), bias)
}

// =============================================================================
// RELU with predominantly negative pre-activation values
// =============================================================================

#[test]
fn relu_with_mostly_negative_pre_activation_detected() {
    // A neuron using RELU but whose pre-activation values are mostly negative,
    // meaning the RELU clips most of the information away.
    let neurons = vec![hidden_neuron("hidden-neg", "RELU", 0.0)];

    // 80% of samples have negative pre-activation (value), yielding activation=0
    // 20% have positive, yielding activation=value
    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "hidden-neg".to_string(),
        (0..50)
            .map(|i| {
                let pre_act = if i < 40 {
                    -(i as f32 + 1.0) / 10.0
                } else {
                    (i as f32 - 39.0) / 10.0
                };
                let activation = pre_act.max(0.0);
                make_record("hidden-neg", i, Some(pre_act), activation)
            })
            .collect(),
    )];

    let mismatches = detect_activation_mismatches(&neurons, &records);

    assert!(
        !mismatches.is_empty(),
        "RELU with 80% negative pre-activation should be detected as mismatched"
    );

    let c = &mismatches[0];
    assert_eq!(c.neuron_uuid, "hidden-neg");
    assert_eq!(c.mismatch_kind, MismatchKind::ReluNegativeBias);
    assert!(
        c.estimated_improvement > 0.0,
        "Should have positive estimated improvement"
    );
    assert!(
        c.recommended_squash.is_some(),
        "Should recommend an alternative activation"
    );
}

// =============================================================================
// RELU with balanced pre-activation → NOT mismatched
// =============================================================================

#[test]
fn relu_with_balanced_pre_activation_not_detected() {
    let neurons = vec![hidden_neuron("hidden-bal", "RELU", 0.0)];

    // 50/50 split of negative and positive pre-activations
    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "hidden-bal".to_string(),
        (0..50)
            .map(|i| {
                let pre_act = (i as f32 - 25.0) / 10.0;
                let activation = pre_act.max(0.0);
                make_record("hidden-bal", i, Some(pre_act), activation)
            })
            .collect(),
    )];

    let mismatches = detect_activation_mismatches(&neurons, &records);

    assert!(
        mismatches.is_empty(),
        "RELU with balanced pre-activation should NOT be detected as mismatched"
    );
}

// =============================================================================
// Bounded activation with tightly clustered output (underutilised)
// =============================================================================

#[test]
fn tanh_with_narrow_output_range_detected() {
    // A TANH neuron that only produces activations in a tiny band (e.g. -0.1 to 0.1),
    // meaning it operates entirely in the linear region and would be better as IDENTITY.
    let neurons = vec![hidden_neuron("hidden-narrow", "TANH", 0.0)];

    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "hidden-narrow".to_string(),
        (0..50)
            .map(|i| {
                // Activations clustered near zero: range -0.05 to 0.05
                let activation = (i as f32 - 25.0) / 500.0;
                make_record("hidden-narrow", i, Some(activation * 1.01), activation)
            })
            .collect(),
    )];

    let mismatches = detect_activation_mismatches(&neurons, &records);

    assert!(
        !mismatches.is_empty(),
        "TANH with narrow output range should be detected as underutilised"
    );

    let c = &mismatches[0];
    assert_eq!(c.mismatch_kind, MismatchKind::BoundedUnderutilised);
}

// =============================================================================
// Bounded activation fully utilising its range → NOT mismatched
// =============================================================================

#[test]
fn tanh_with_full_range_not_detected() {
    let neurons = vec![hidden_neuron("hidden-full", "TANH", 0.0)];

    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "hidden-full".to_string(),
        (0..50)
            .map(|i| {
                // Activations spanning -0.9 to 0.9 — good utilisation
                let activation = (i as f32 - 25.0) / 27.0;
                make_record("hidden-full", i, Some(activation * 2.0), activation)
            })
            .collect(),
    )];

    let mismatches = detect_activation_mismatches(&neurons, &records);

    assert!(
        mismatches.is_empty(),
        "TANH utilising full range should NOT be detected as mismatched"
    );
}

// =============================================================================
// Insufficient samples returns empty
// =============================================================================

#[test]
fn insufficient_samples_returns_empty() {
    let neurons = vec![hidden_neuron("hidden-few", "RELU", 0.0)];

    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "hidden-few".to_string(),
        (0..5)
            .map(|i| make_record("hidden-few", i, Some(-1.0), 0.0))
            .collect(),
    )];

    let mismatches = detect_activation_mismatches(&neurons, &records);

    assert!(
        mismatches.is_empty(),
        "Too few samples should return empty results"
    );
}

// =============================================================================
// Neuron not in records returns empty
// =============================================================================

#[test]
fn neuron_without_records_returns_empty() {
    let neurons = vec![hidden_neuron("hidden-missing", "RELU", 0.0)];
    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![];

    let mismatches = detect_activation_mismatches(&neurons, &records);

    assert!(
        mismatches.is_empty(),
        "Neuron without records should return empty"
    );
}

// =============================================================================
// Coordinated candidate conversion
// =============================================================================

#[test]
fn mismatch_candidates_convert_to_coordinated_candidates() {
    let mismatches = vec![ActivationMismatchCandidate {
        neuron_uuid: "hidden-1".to_string(),
        current_squash: "RELU".to_string(),
        recommended_squash: Some("ELU".to_string()),
        mismatch_kind: MismatchKind::ReluNegativeBias,
        clipped_fraction: 0.8,
        estimated_improvement: 0.01,
        reason: "80% of pre-activation values are negative".to_string(),
    }];

    let coordinated = activation_mismatch_to_coordinated_candidates(&mismatches);

    assert_eq!(coordinated.len(), 1);
    assert_eq!(coordinated[0].operations.len(), 1);
    assert!(coordinated[0].expected_creature_score_gain > 0.0);
    assert!(coordinated[0].comment.as_ref().unwrap().contains("543"));
}

// =============================================================================
// IDENTITY is never flagged as mismatched
// =============================================================================

#[test]
fn identity_activation_never_flagged() {
    let neurons = vec![hidden_neuron("hidden-id", "IDENTITY", 0.0)];

    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "hidden-id".to_string(),
        (0..50)
            .map(|i| {
                let activation = (i as f32 - 25.0) / 10.0;
                make_record("hidden-id", i, Some(activation), activation)
            })
            .collect(),
    )];

    let mismatches = detect_activation_mismatches(&neurons, &records);

    assert!(
        mismatches.is_empty(),
        "IDENTITY activation should never be flagged as mismatched"
    );
}

// =============================================================================
// Multiple neurons can produce multiple candidates
// =============================================================================

#[test]
fn multiple_mismatched_neurons_detected() {
    let neurons = vec![
        hidden_neuron("hidden-a", "RELU", 0.0),
        hidden_neuron("hidden-b", "RELU", 0.0),
    ];

    // Both neurons have 90% negative pre-activation
    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        (
            "hidden-a".to_string(),
            (0..50)
                .map(|i| {
                    let pre_act = if i < 45 { -1.0 } else { 0.5 };
                    make_record("hidden-a", i, Some(pre_act), pre_act.max(0.0))
                })
                .collect(),
        ),
        (
            "hidden-b".to_string(),
            (0..50)
                .map(|i| {
                    let pre_act = if i < 45 { -2.0 } else { 1.0 };
                    make_record("hidden-b", i, Some(pre_act), pre_act.max(0.0))
                })
                .collect(),
        ),
    ];

    let mismatches = detect_activation_mismatches(&neurons, &records);

    assert_eq!(
        mismatches.len(),
        2,
        "Both mismatched neurons should be detected"
    );
}
