//! Tests for Issue #356: Detect oscillating neurons for stabilisation candidates.
//!
//! Oscillating neurons have activations that frequently change sign across training
//! samples, indicating the neuron is fighting between contradictory functions.
//! This module detects them and recommends activation function changes.
//!
//! ## TDD Plan
//! 1. Create test network with intentionally oscillating neurons
//! 2. Verify detection identifies them with correct statistics
//! 3. Verify non-oscillating neurons are not flagged
//! 4. Test edge cases: balanced vs unbalanced oscillation, dead neurons
//! 5. Test coordinated structural candidate conversion

use neat_ai_discovery::analysis::oscillating_neuron::{
    detect_oscillating_neurons, oscillating_neurons_to_coordinated_candidates,
};
use neat_ai_discovery::types::DiscoverRecord;

/// Helper: create a DiscoverRecord for a neuron with given activation.
fn record(neuron_uuid: &str, obs_index: u32, activation: f32) -> DiscoverRecord {
    DiscoverRecord {
        obs_index,
        neuron_uuid: neuron_uuid.to_string(),
        value: Some(activation * 0.8),
        activation,
        errors: vec![0.01],
    }
}

/// Test 1: Neuron with alternating positive/negative activations is detected as oscillating.
#[test]
fn test_detects_alternating_activation_neuron() {
    let neurons = vec![("hidden-osc".to_string(), "TANH".to_string(), 0.0)];

    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = if i % 2 == 0 { 0.5 } else { -0.5 };
            record("hidden-osc", i, activation)
        })
        .collect();

    let candidates = detect_oscillating_neurons(&neurons, &[("hidden-osc".to_string(), records)]);

    assert_eq!(candidates.len(), 1, "Should detect one oscillating neuron");
    let c = &candidates[0];
    assert_eq!(c.neuron_uuid, "hidden-osc");
    assert!(
        c.sign_change_fraction > 0.3,
        "Sign change fraction should be high: got {}",
        c.sign_change_fraction
    );
    assert!(
        c.mean_abs_activation > 0.01,
        "Mean abs activation should be meaningful"
    );
}

/// Test 2: Neuron with consistently positive activations is NOT oscillating.
#[test]
fn test_consistently_positive_not_flagged() {
    let neurons = vec![("hidden-pos".to_string(), "RELU".to_string(), 0.0)];

    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = 0.3 + 0.2 * ((i as f32 * 0.1).sin()).abs();
            record("hidden-pos", i, activation)
        })
        .collect();

    let candidates = detect_oscillating_neurons(&neurons, &[("hidden-pos".to_string(), records)]);

    assert!(
        candidates.is_empty(),
        "Consistently positive neuron should not be flagged as oscillating"
    );
}

/// Test 3: Dead neuron (near-zero activation) is NOT flagged as oscillating.
#[test]
fn test_dead_neuron_not_flagged_as_oscillating() {
    let neurons = vec![("hidden-dead".to_string(), "TANH".to_string(), 0.0)];

    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = if i % 2 == 0 { 1e-8 } else { -1e-8 };
            record("hidden-dead", i, activation)
        })
        .collect();

    let candidates = detect_oscillating_neurons(&neurons, &[("hidden-dead".to_string(), records)]);

    assert!(
        candidates.is_empty(),
        "Dead neuron should not be flagged as oscillating"
    );
}

/// Test 4: Neuron with only occasional sign changes is NOT oscillating.
#[test]
fn test_low_sign_change_frequency_not_flagged() {
    let neurons = vec![("hidden-rare".to_string(), "TANH".to_string(), 0.0)];

    // 90 positive, then 10 negative → only 1 sign change
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = if i < 90 { 0.5 } else { -0.5 };
            record("hidden-rare", i, activation)
        })
        .collect();

    let candidates = detect_oscillating_neurons(&neurons, &[("hidden-rare".to_string(), records)]);

    assert!(
        candidates.is_empty(),
        "Low sign change frequency should not be flagged"
    );
}

/// Test 5: Insufficient samples should not trigger detection.
#[test]
fn test_insufficient_samples_not_flagged() {
    let neurons = vec![("hidden-few".to_string(), "TANH".to_string(), 0.0)];

    let records: Vec<DiscoverRecord> = (0..5)
        .map(|i| {
            let activation = if i % 2 == 0 { 0.5 } else { -0.5 };
            record("hidden-few", i, activation)
        })
        .collect();

    let candidates = detect_oscillating_neurons(&neurons, &[("hidden-few".to_string(), records)]);

    assert!(
        candidates.is_empty(),
        "Too few samples should not trigger detection"
    );
}

/// Test 6: Oscillating neuron candidates produce correct coordinated structural operations.
#[test]
fn test_candidates_produce_coordinated_change_squash_operations() {
    let neurons = vec![("hidden-osc".to_string(), "TANH".to_string(), 0.0)];

    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = if i % 2 == 0 { 0.5 } else { -0.5 };
            record("hidden-osc", i, activation)
        })
        .collect();

    let candidates = detect_oscillating_neurons(&neurons, &[("hidden-osc".to_string(), records)]);
    assert!(!candidates.is_empty(), "Should detect oscillating neuron");

    let coordinated = oscillating_neurons_to_coordinated_candidates(&candidates);

    assert_eq!(
        coordinated.len(),
        1,
        "Should produce one coordinated candidate"
    );
    let c = &coordinated[0];
    assert!(
        c.expected_creature_score_gain > 0.0,
        "Expected improvement should be positive"
    );
    assert!(c.comment.is_some(), "Should have a comment");

    // Check that operations include a ChangeSquash
    let ops_json = serde_json::to_string(&c.operations).unwrap();
    assert!(
        ops_json.contains("changeSquash"),
        "Should include changeSquash operation, got: {ops_json}"
    );
}

/// Test 7: Unbalanced oscillation (>80% one sign) is NOT flagged.
#[test]
fn test_unbalanced_oscillation_not_flagged() {
    let neurons = vec![("hidden-unbal".to_string(), "TANH".to_string(), 0.0)];

    // 85% positive, 15% negative — unbalanced
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = if i < 85 {
                if i % 3 == 0 { 0.3 } else { 0.6 }
            } else {
                -0.4
            };
            record("hidden-unbal", i, activation)
        })
        .collect();

    let candidates = detect_oscillating_neurons(&neurons, &[("hidden-unbal".to_string(), records)]);

    assert!(
        candidates.is_empty(),
        "Unbalanced sign distribution should not be flagged as oscillating"
    );
}

/// Test 8: Multiple neurons — only oscillating ones are detected.
#[test]
fn test_mixed_neurons_only_oscillating_detected() {
    let neurons = vec![
        ("osc-1".to_string(), "TANH".to_string(), 0.0),
        ("stable-1".to_string(), "RELU".to_string(), 0.0),
        ("osc-2".to_string(), "IDENTITY".to_string(), 0.0),
    ];

    let osc_records_1: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = if i % 2 == 0 { 0.7 } else { -0.7 };
            record("osc-1", i, activation)
        })
        .collect();

    let stable_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| record("stable-1", i, 0.5 + 0.1 * (i as f32 * 0.1).sin()))
        .collect();

    let osc_records_2: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = if i % 3 == 0 {
                -0.4
            } else if i % 3 == 1 {
                0.4
            } else {
                -0.3
            };
            record("osc-2", i, activation)
        })
        .collect();

    let candidates = detect_oscillating_neurons(
        &neurons,
        &[
            ("osc-1".to_string(), osc_records_1),
            ("stable-1".to_string(), stable_records),
            ("osc-2".to_string(), osc_records_2),
        ],
    );

    // At minimum osc-1 should be detected (perfect alternation)
    assert!(
        !candidates.is_empty(),
        "Should detect at least one oscillating neuron"
    );
    let uuids: Vec<&str> = candidates.iter().map(|c| c.neuron_uuid.as_str()).collect();
    assert!(uuids.contains(&"osc-1"), "Should detect osc-1");
    assert!(
        !uuids.contains(&"stable-1"),
        "Should not detect stable neuron"
    );
}

/// Test 9: Recommended squash is ABSOLUTE for symmetric activation functions.
#[test]
fn test_recommends_absolute_for_symmetric_activations() {
    let neurons = vec![("hidden-osc".to_string(), "TANH".to_string(), 0.0)];

    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = if i % 2 == 0 { 0.5 } else { -0.5 };
            record("hidden-osc", i, activation)
        })
        .collect();

    let candidates = detect_oscillating_neurons(&neurons, &[("hidden-osc".to_string(), records)]);

    assert!(!candidates.is_empty());
    assert_eq!(
        candidates[0].recommended_squash, "ABSOLUTE",
        "Should recommend ABSOLUTE for TANH oscillation"
    );
}
