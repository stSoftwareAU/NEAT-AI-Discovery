//! Tests for Issue #342: Detect saturated neurons for activation function change candidates.
//!
//! Saturated neurons are stuck at their activation ceiling or floor (e.g., TANH outputting
//! ≈ 1.0 regardless of input variation). These neurons pass no gradient information and
//! block learning. This module detects them and recommends activation function changes
//! or bias adjustments.
//!
//! ## TDD Plan
//! 1. Create test network with intentionally saturated TANH neurons
//! 2. Verify detection identifies them with correct statistics
//! 3. Verify non-saturated neurons are not flagged
//! 4. Test activation function recommendation logic
//! 5. Test bias adjustment calculation

use neat_ai_discovery::analysis::saturation::{
    SaturatedNeuronCandidate, detect_saturated_neurons, saturated_neurons_to_coordinated_candidates,
};
use neat_ai_discovery::types::DiscoverRecord;

/// Helper: create a DiscoverRecord for a neuron with given activation.
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

/// Test 1: TANH neuron with mean activation > 0.95 is detected as saturated.
#[test]
fn test_detects_tanh_saturated_at_positive_ceiling() {
    // TANH neuron that is consistently outputting near 1.0
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            record(
                "hidden-17",
                i,
                0.998 + 0.001 * (i as f32 / 100.0),
                Some(5.0 + i as f32 * 0.1),
            )
        })
        .collect();

    let candidates = detect_saturated_neurons(
        &[("hidden-17".to_string(), "TANH".to_string(), 0.0)],
        &[("hidden-17".to_string(), records)],
    );

    assert_eq!(candidates.len(), 1, "Should detect one saturated neuron");
    let c = &candidates[0];
    assert_eq!(c.neuron_uuid, "hidden-17");
    assert_eq!(c.current_squash, "TANH");
    assert!(
        c.mean_activation > 0.95,
        "Mean activation should be > 0.95, got {}",
        c.mean_activation
    );
    assert!(
        c.activation_std_dev < 0.05,
        "Activation std dev should be low, got {}",
        c.activation_std_dev
    );
}

/// Test 2: TANH neuron with mean activation < -0.95 is detected as saturated.
#[test]
fn test_detects_tanh_saturated_at_negative_floor() {
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            record(
                "hidden-22",
                i,
                -0.999 + 0.0005 * (i as f32 / 100.0),
                Some(-6.0),
            )
        })
        .collect();

    let candidates = detect_saturated_neurons(
        &[("hidden-22".to_string(), "TANH".to_string(), 0.0)],
        &[("hidden-22".to_string(), records)],
    );

    assert_eq!(
        candidates.len(),
        1,
        "Should detect saturated neuron at floor"
    );
    let c = &candidates[0];
    assert!(
        c.mean_activation < -0.95,
        "Mean activation should be < -0.95, got {}",
        c.mean_activation
    );
}

/// Test 3: Non-saturated TANH neuron is NOT flagged.
#[test]
fn test_does_not_flag_non_saturated_tanh() {
    // TANH neuron operating in the active region (mean ≈ 0.3)
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = 0.3 + 0.4 * ((i as f32 * 0.1).sin());
            record("hidden-normal", i, activation, Some(0.5))
        })
        .collect();

    let candidates = detect_saturated_neurons(
        &[("hidden-normal".to_string(), "TANH".to_string(), 0.0)],
        &[("hidden-normal".to_string(), records)],
    );

    assert!(
        candidates.is_empty(),
        "Non-saturated neuron should not be flagged"
    );
}

/// Test 4: LOGISTIC neuron saturated at ceiling (output ≈ 1.0).
#[test]
fn test_detects_logistic_saturated_at_ceiling() {
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| record("logistic-1", i, 0.9999, Some(600.0)))
        .collect();

    let candidates = detect_saturated_neurons(
        &[("logistic-1".to_string(), "LOGISTIC".to_string(), 0.0)],
        &[("logistic-1".to_string(), records)],
    );

    assert_eq!(
        candidates.len(),
        1,
        "Should detect saturated LOGISTIC neuron"
    );
    assert_eq!(candidates[0].current_squash, "LOGISTIC");
}

/// Test 5: LOGISTIC neuron saturated at floor (output ≈ 0.0).
#[test]
fn test_detects_logistic_saturated_at_floor() {
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| record("logistic-2", i, 0.0001, Some(-600.0)))
        .collect();

    let candidates = detect_saturated_neurons(
        &[("logistic-2".to_string(), "LOGISTIC".to_string(), 0.0)],
        &[("logistic-2".to_string(), records)],
    );

    assert_eq!(
        candidates.len(),
        1,
        "Should detect LOGISTIC saturated at floor"
    );
}

/// Test 6: HARD_TANH neuron clamped at +1.0.
#[test]
fn test_detects_hard_tanh_clamped() {
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| record("ht-1", i, 1.0, Some(5.0 + i as f32 * 0.5)))
        .collect();

    let candidates = detect_saturated_neurons(
        &[("ht-1".to_string(), "HARD_TANH".to_string(), 0.0)],
        &[("ht-1".to_string(), records)],
    );

    assert_eq!(candidates.len(), 1, "Should detect clamped HARD_TANH");
}

/// Test 7: Unbounded activation (RELU, IDENTITY) should NOT be flagged as saturated.
#[test]
fn test_unbounded_activations_not_flagged() {
    let relu_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| record("relu-1", i, 5.0 + i as f32 * 0.1, Some(5.0)))
        .collect();

    let identity_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| record("identity-1", i, 100.0, Some(100.0)))
        .collect();

    let candidates = detect_saturated_neurons(
        &[
            ("relu-1".to_string(), "RELU".to_string(), 0.0),
            ("identity-1".to_string(), "IDENTITY".to_string(), 0.0),
        ],
        &[
            ("relu-1".to_string(), relu_records),
            ("identity-1".to_string(), identity_records),
        ],
    );

    assert!(
        candidates.is_empty(),
        "Unbounded activations should not be flagged as saturated"
    );
}

/// Test 8: Neuron with too few samples should not be flagged.
#[test]
fn test_insufficient_samples_not_flagged() {
    // Only 5 samples — not enough to be confident about saturation
    let records: Vec<DiscoverRecord> = (0..5)
        .map(|i| record("few-samples", i, 0.999, Some(10.0)))
        .collect();

    let candidates = detect_saturated_neurons(
        &[("few-samples".to_string(), "TANH".to_string(), 0.0)],
        &[("few-samples".to_string(), records)],
    );

    assert!(
        candidates.is_empty(),
        "Too few samples should not trigger detection"
    );
}

/// Test 9: Candidates produce correct coordinated structural operations.
#[test]
fn test_candidates_produce_coordinated_operations() {
    let candidate = SaturatedNeuronCandidate {
        neuron_uuid: "hidden-17".to_string(),
        current_squash: "TANH".to_string(),
        mean_activation: 0.998,
        activation_std_dev: 0.001,
        input_std_dev: 2.5,
        recommended_squash: Some("IDENTITY".to_string()),
        recommended_bias_delta: Some(-3.2),
        estimated_improvement: 0.05,
    };

    let coordinated = saturated_neurons_to_coordinated_candidates(&[candidate]);

    assert!(
        !coordinated.is_empty(),
        "Should produce coordinated candidates"
    );

    // Each saturated neuron should produce at least one coordinated candidate
    let c = &coordinated[0];
    assert!(
        c.expected_creature_score_gain > 0.0,
        "Expected improvement should be positive"
    );
    assert!(
        c.comment.is_some(),
        "Should have a comment explaining the change"
    );

    // Check that operations include a ChangeSquash or SetBias
    let ops_json = serde_json::to_string(&c.operations).unwrap();
    let has_change_squash = ops_json.contains("changeSquash");
    let has_set_bias = ops_json.contains("setBias");
    assert!(
        has_change_squash || has_set_bias,
        "Should include changeSquash or setBias operation, got: {ops_json}"
    );
}

/// Test 10: Mixed neurons — only saturated ones are detected.
#[test]
fn test_mixed_neurons_only_saturated_detected() {
    let saturated_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| record("sat", i, 0.999, Some(10.0)))
        .collect();

    let normal_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = 0.1 + 0.3 * ((i as f32 * 0.2).sin());
            record("norm", i, activation, Some(0.5))
        })
        .collect();

    let candidates = detect_saturated_neurons(
        &[
            ("sat".to_string(), "TANH".to_string(), 0.0),
            ("norm".to_string(), "TANH".to_string(), 0.0),
        ],
        &[
            ("sat".to_string(), saturated_records),
            ("norm".to_string(), normal_records),
        ],
    );

    assert_eq!(
        candidates.len(),
        1,
        "Should detect only the saturated neuron"
    );
    assert_eq!(candidates[0].neuron_uuid, "sat");
}

/// Test 11: Input neurons should not be flagged (they are not computation nodes).
#[test]
fn test_input_neurons_excluded() {
    // Input neurons are excluded by the caller — they should not be passed to detect.
    // But if they are, IDENTITY squash means they are not bounded so not flagged.
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| record("input-0", i, 1.0, Some(1.0)))
        .collect();

    let candidates = detect_saturated_neurons(
        &[("input-0".to_string(), "IDENTITY".to_string(), 0.0)],
        &[("input-0".to_string(), records)],
    );

    assert!(candidates.is_empty(), "Input neurons should not be flagged");
}

/// Test 12: RELU neuron dead at zero (all negative inputs) is detected.
#[test]
fn test_detects_relu_dead_at_zero() {
    // RELU with consistently zero output — dead neuron
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| record("relu-dead", i, 0.0, Some(-5.0 - i as f32 * 0.1)))
        .collect();

    let candidates = detect_saturated_neurons(
        &[("relu-dead".to_string(), "RELU".to_string(), 0.0)],
        &[("relu-dead".to_string(), records)],
    );

    assert_eq!(candidates.len(), 1, "Dead RELU neuron should be detected");
    assert_eq!(candidates[0].neuron_uuid, "relu-dead");
}

/// Test 13: Bias adjustment direction is correct.
#[test]
fn test_bias_adjustment_direction() {
    // Saturated at positive ceiling — bias should be adjusted negatively to move away
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| record("bias-test", i, 0.999, Some(8.0)))
        .collect();

    let candidates = detect_saturated_neurons(
        &[("bias-test".to_string(), "TANH".to_string(), 2.0)],
        &[("bias-test".to_string(), records)],
    );

    assert_eq!(candidates.len(), 1);
    let c = &candidates[0];
    // For positive saturation, bias delta should be negative (move toward active region)
    if let Some(delta) = c.recommended_bias_delta {
        assert!(
            delta < 0.0,
            "Bias delta should be negative for positive saturation, got {delta}"
        );
    }
}
