//! Tests for Issue #358: Oscillating Neuron Detection.
//!
//! Dedicated tests for oscillating neuron detection as a structural discovery method.
//! Oscillating neurons have activations that frequently change sign across training
//! samples, indicating the neuron is fighting between contradictory functions. The
//! detection recommends `changeSquash` coordinated structural candidates to stabilise
//! the neuron's output.
//!
//! ## TDD Plan
//! 1. Verify detection with perfect alternating pattern
//! 2. Verify non-oscillating (stable) neurons are excluded
//! 3. Verify dead neurons are excluded despite sign changes
//! 4. Verify configurable thresholds: sign-change frequency below threshold
//! 5. Verify minimum sample count enforcement
//! 6. Verify coordinated structural candidate conversion (changeSquash operations)
//! 7. Verify unbalanced sign distribution is excluded
//! 8. Verify multiple neurons: only oscillating ones detected
//! 9. Verify recommended squash for symmetric vs asymmetric activation functions
//! 10. Verify bias adjustment recommendations for imbalanced positive/negative fractions
//! 11. Verify candidates are sorted by estimated improvement
//! 12. Verify irregular oscillation patterns (not perfectly alternating)

use neat_ai_discovery::analysis::oscillating_neuron::{
    detect_oscillating_neurons, oscillating_neurons_to_coordinated_candidates,
};
use neat_ai_discovery::types::DiscoverRecord;

/// Helper: create a DiscoverRecord for a neuron with given activation.
fn make_record(neuron_uuid: &str, obs_index: u32, activation: f32) -> DiscoverRecord {
    DiscoverRecord {
        obs_index,
        neuron_uuid: neuron_uuid.to_string(),
        value: Some(activation * 0.8),
        activation,
        errors: vec![0.01],
    }
}

/// Test 1: Perfect alternating pattern is detected as oscillating.
#[test]
fn test_perfect_alternation_detected() {
    let neurons = vec![("osc-a".to_string(), "TANH".to_string(), 0.0)];
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = if i % 2 == 0 { 0.5 } else { -0.5 };
            make_record("osc-a", i, activation)
        })
        .collect();

    let candidates = detect_oscillating_neurons(&neurons, &[("osc-a".to_string(), records)]);

    assert_eq!(candidates.len(), 1, "Should detect one oscillating neuron");
    let c = &candidates[0];
    assert_eq!(c.neuron_uuid, "osc-a");
    // Perfect alternation: sign change fraction should be ~1.0
    assert!(
        c.sign_change_fraction > 0.9,
        "Perfect alternation should have high sign change fraction: got {}",
        c.sign_change_fraction
    );
    assert!(
        (c.positive_fraction - 0.5).abs() < 0.05,
        "Should be roughly 50/50 positive/negative: got {}",
        c.positive_fraction
    );
    assert_eq!(c.sample_count, 100);
}

/// Test 2: Consistently positive neuron is NOT detected.
#[test]
fn test_stable_positive_not_detected() {
    let neurons = vec![("stable-pos".to_string(), "RELU".to_string(), 0.0)];
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = 0.3 + 0.2 * ((i as f32 * 0.1).sin()).abs();
            make_record("stable-pos", i, activation)
        })
        .collect();

    let candidates = detect_oscillating_neurons(&neurons, &[("stable-pos".to_string(), records)]);

    assert!(
        candidates.is_empty(),
        "Consistently positive neuron should not be flagged"
    );
}

/// Test 3: Dead neuron with sign changes is NOT detected.
#[test]
fn test_dead_neuron_excluded() {
    let neurons = vec![("dead-osc".to_string(), "TANH".to_string(), 0.0)];
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = if i % 2 == 0 { 1e-8 } else { -1e-8 };
            make_record("dead-osc", i, activation)
        })
        .collect();

    let candidates = detect_oscillating_neurons(&neurons, &[("dead-osc".to_string(), records)]);

    assert!(
        candidates.is_empty(),
        "Dead neuron with tiny oscillations should not be flagged"
    );
}

/// Test 4: Low sign-change frequency is NOT detected.
#[test]
fn test_low_sign_change_frequency_excluded() {
    let neurons = vec![("low-freq".to_string(), "TANH".to_string(), 0.0)];
    // 90 positive, then 10 negative → only 1 sign change out of 99 transitions
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = if i < 90 { 0.5 } else { -0.5 };
            make_record("low-freq", i, activation)
        })
        .collect();

    let candidates = detect_oscillating_neurons(&neurons, &[("low-freq".to_string(), records)]);

    assert!(
        candidates.is_empty(),
        "Low sign change frequency should not trigger detection"
    );
}

/// Test 5: Insufficient samples (below minimum) are NOT detected.
#[test]
fn test_insufficient_samples_excluded() {
    let neurons = vec![("few".to_string(), "TANH".to_string(), 0.0)];
    let records: Vec<DiscoverRecord> = (0..5)
        .map(|i| {
            let activation = if i % 2 == 0 { 0.5 } else { -0.5 };
            make_record("few", i, activation)
        })
        .collect();

    let candidates = detect_oscillating_neurons(&neurons, &[("few".to_string(), records)]);

    assert!(
        candidates.is_empty(),
        "Too few samples should not trigger detection"
    );
}

/// Test 6: Coordinated structural candidates include changeSquash operation.
#[test]
fn test_coordinated_candidate_has_change_squash() {
    let neurons = vec![("osc-coord".to_string(), "TANH".to_string(), 0.0)];
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = if i % 2 == 0 { 0.6 } else { -0.6 };
            make_record("osc-coord", i, activation)
        })
        .collect();

    let candidates = detect_oscillating_neurons(&neurons, &[("osc-coord".to_string(), records)]);
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
        "Expected gain should be positive"
    );
    assert!(c.comment.is_some(), "Should have explanatory comment");

    let ops_json = serde_json::to_string(&c.operations).unwrap();
    assert!(
        ops_json.contains("changeSquash"),
        "Should include changeSquash operation, got: {ops_json}"
    );
}

/// Test 7: Unbalanced sign distribution (>80% one sign) is NOT detected.
#[test]
fn test_unbalanced_sign_distribution_excluded() {
    let neurons = vec![("unbal".to_string(), "TANH".to_string(), 0.0)];
    // 85% positive with some pattern, 15% negative
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = if i < 85 {
                if i % 3 == 0 { 0.3 } else { 0.6 }
            } else {
                -0.4
            };
            make_record("unbal", i, activation)
        })
        .collect();

    let candidates = detect_oscillating_neurons(&neurons, &[("unbal".to_string(), records)]);

    assert!(
        candidates.is_empty(),
        "Unbalanced sign distribution should not be flagged"
    );
}

/// Test 8: Multiple neurons — only oscillating ones detected.
#[test]
fn test_mixed_neurons_filters_correctly() {
    let neurons = vec![
        ("osc-mix".to_string(), "TANH".to_string(), 0.0),
        ("stable-mix".to_string(), "RELU".to_string(), 0.0),
    ];

    let osc_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = if i % 2 == 0 { 0.7 } else { -0.7 };
            make_record("osc-mix", i, activation)
        })
        .collect();

    let stable_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| make_record("stable-mix", i, 0.5 + 0.1 * (i as f32 * 0.1).sin()))
        .collect();

    let candidates = detect_oscillating_neurons(
        &neurons,
        &[
            ("osc-mix".to_string(), osc_records),
            ("stable-mix".to_string(), stable_records),
        ],
    );

    assert_eq!(
        candidates.len(),
        1,
        "Should detect only the oscillating neuron"
    );
    assert_eq!(candidates[0].neuron_uuid, "osc-mix");
}

/// Test 9: TANH recommends ABSOLUTE; other functions recommend RELU.
#[test]
fn test_recommended_squash_varies_by_activation() {
    // TANH → ABSOLUTE
    let tanh_neurons = vec![("osc-tanh".to_string(), "TANH".to_string(), 0.0)];
    let tanh_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = if i % 2 == 0 { 0.5 } else { -0.5 };
            make_record("osc-tanh", i, activation)
        })
        .collect();

    let tanh_candidates =
        detect_oscillating_neurons(&tanh_neurons, &[("osc-tanh".to_string(), tanh_records)]);
    assert_eq!(tanh_candidates[0].recommended_squash, "ABSOLUTE");

    // LOGISTIC → RELU (non-symmetric)
    let logistic_neurons = vec![("osc-log".to_string(), "LOGISTIC".to_string(), 0.0)];
    let logistic_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = if i % 2 == 0 { 0.5 } else { -0.5 };
            make_record("osc-log", i, activation)
        })
        .collect();

    let logistic_candidates = detect_oscillating_neurons(
        &logistic_neurons,
        &[("osc-log".to_string(), logistic_records)],
    );
    assert_eq!(logistic_candidates[0].recommended_squash, "RELU");

    // IDENTITY → ABSOLUTE (symmetric)
    let identity_neurons = vec![("osc-id".to_string(), "IDENTITY".to_string(), 0.0)];
    let identity_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = if i % 2 == 0 { 0.5 } else { -0.5 };
            make_record("osc-id", i, activation)
        })
        .collect();

    let identity_candidates = detect_oscillating_neurons(
        &identity_neurons,
        &[("osc-id".to_string(), identity_records)],
    );
    assert_eq!(identity_candidates[0].recommended_squash, "ABSOLUTE");
}

/// Test 10: Bias adjustment recommended when positive fraction is imbalanced.
#[test]
fn test_bias_adjustment_for_imbalanced_oscillation() {
    // 65% positive, 35% negative → should recommend negative bias delta
    let neurons = vec![("osc-bias".to_string(), "TANH".to_string(), 0.0)];
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            // Roughly 65 positive, 35 negative with frequent sign changes
            let activation = if (i * 7 + 3) % 100 < 65 { 0.4 } else { -0.4 };
            make_record("osc-bias", i, activation)
        })
        .collect();

    let candidates = detect_oscillating_neurons(&neurons, &[("osc-bias".to_string(), records)]);

    if !candidates.is_empty() {
        let c = &candidates[0];
        // If positive fraction > 0.6, bias delta should be negative
        if c.positive_fraction > 0.6 {
            assert!(
                c.recommended_bias_delta.is_some(),
                "Should recommend bias adjustment for imbalanced oscillation"
            );
            assert!(
                c.recommended_bias_delta.unwrap() < 0.0,
                "Should recommend negative bias delta for positive-heavy oscillation"
            );
        }
    }
}

/// Test 11: Candidates are sorted by estimated improvement (best first).
#[test]
fn test_candidates_sorted_by_improvement() {
    let neurons = vec![
        ("osc-small".to_string(), "TANH".to_string(), 0.0),
        ("osc-large".to_string(), "TANH".to_string(), 0.0),
    ];

    // Small oscillation magnitude
    let small_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = if i % 2 == 0 { 0.1 } else { -0.1 };
            make_record("osc-small", i, activation)
        })
        .collect();

    // Large oscillation magnitude → should have higher estimated improvement
    let large_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = if i % 2 == 0 { 0.9 } else { -0.9 };
            make_record("osc-large", i, activation)
        })
        .collect();

    let candidates = detect_oscillating_neurons(
        &neurons,
        &[
            ("osc-small".to_string(), small_records),
            ("osc-large".to_string(), large_records),
        ],
    );

    assert_eq!(
        candidates.len(),
        2,
        "Should detect both oscillating neurons"
    );
    // Best improvement first
    assert!(
        candidates[0].estimated_improvement >= candidates[1].estimated_improvement,
        "Candidates should be sorted by estimated improvement (best first): {} >= {}",
        candidates[0].estimated_improvement,
        candidates[1].estimated_improvement
    );
    // The larger magnitude neuron should be first
    assert_eq!(
        candidates[0].neuron_uuid, "osc-large",
        "Larger magnitude oscillation should rank higher"
    );
}

/// Test 12: Irregular oscillation pattern (not perfectly alternating) is still detected.
#[test]
fn test_irregular_oscillation_detected() {
    let neurons = vec![("osc-irr".to_string(), "TANH".to_string(), 0.0)];
    // Pattern: +, -, -, +, -, +, +, -, ... (irregular but frequent sign changes)
    let pattern = [0.4, -0.3, -0.5, 0.6, -0.2, 0.3, 0.5, -0.4, 0.7, -0.6];
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| make_record("osc-irr", i, pattern[i as usize % pattern.len()]))
        .collect();

    let candidates = detect_oscillating_neurons(&neurons, &[("osc-irr".to_string(), records)]);

    assert!(
        !candidates.is_empty(),
        "Irregular but frequent oscillation should be detected"
    );
    let c = &candidates[0];
    assert!(
        c.sign_change_fraction > 0.3,
        "Sign change fraction should exceed threshold: got {}",
        c.sign_change_fraction
    );
}

/// Test 13: Empty records produce no candidates.
#[test]
fn test_empty_records_no_candidates() {
    let neurons = vec![("empty".to_string(), "TANH".to_string(), 0.0)];
    let records: Vec<DiscoverRecord> = vec![];

    let candidates = detect_oscillating_neurons(&neurons, &[("empty".to_string(), records)]);

    assert!(
        candidates.is_empty(),
        "Empty records should produce no candidates"
    );
}

/// Test 14: Neuron not in records produces no candidate.
#[test]
fn test_missing_neuron_records_no_candidate() {
    let neurons = vec![("missing".to_string(), "TANH".to_string(), 0.0)];
    // No records for this neuron
    let candidates = detect_oscillating_neurons(&neurons, &[]);

    assert!(
        candidates.is_empty(),
        "Missing records should produce no candidate"
    );
}

/// Test 15: Coordinated candidate includes setBias when bias adjustment recommended.
#[test]
fn test_coordinated_candidate_includes_set_bias_when_imbalanced() {
    let neurons = vec![("osc-bias-op".to_string(), "TANH".to_string(), 0.0)];
    // Create strongly positive-biased oscillation (>60% positive)
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            // Use a pattern where ~70% are positive but with frequent sign changes
            let activation = if (i * 3) % 10 < 7 { 0.5 } else { -0.5 };
            make_record("osc-bias-op", i, activation)
        })
        .collect();

    let candidates = detect_oscillating_neurons(&neurons, &[("osc-bias-op".to_string(), records)]);

    if !candidates.is_empty() && candidates[0].recommended_bias_delta.is_some() {
        let coordinated = oscillating_neurons_to_coordinated_candidates(&candidates);
        assert!(!coordinated.is_empty());

        let ops_json = serde_json::to_string(&coordinated[0].operations).unwrap();
        assert!(
            ops_json.contains("setBias"),
            "Should include setBias when bias adjustment recommended, got: {ops_json}"
        );
    }
}

/// Test 16: Balanced oscillation does NOT include setBias.
#[test]
fn test_balanced_oscillation_no_set_bias() {
    let neurons = vec![("osc-bal".to_string(), "TANH".to_string(), 0.0)];
    // Perfect 50/50 split
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = if i % 2 == 0 { 0.5 } else { -0.5 };
            make_record("osc-bal", i, activation)
        })
        .collect();

    let candidates = detect_oscillating_neurons(&neurons, &[("osc-bal".to_string(), records)]);

    assert!(!candidates.is_empty());
    let c = &candidates[0];
    assert!(
        c.recommended_bias_delta.is_none(),
        "Balanced oscillation should not recommend bias adjustment, got: {:?}",
        c.recommended_bias_delta
    );

    let coordinated = oscillating_neurons_to_coordinated_candidates(&candidates);
    let ops_json = serde_json::to_string(&coordinated[0].operations).unwrap();
    assert!(
        !ops_json.contains("setBias"),
        "Balanced oscillation should not include setBias, got: {ops_json}"
    );
}

/// Test 17: Estimated improvement is positive for detected neurons.
#[test]
fn test_estimated_improvement_positive() {
    let neurons = vec![("osc-imp".to_string(), "TANH".to_string(), 0.0)];
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = if i % 2 == 0 { 0.5 } else { -0.5 };
            make_record("osc-imp", i, activation)
        })
        .collect();

    let candidates = detect_oscillating_neurons(&neurons, &[("osc-imp".to_string(), records)]);

    assert!(!candidates.is_empty());
    for c in &candidates {
        assert!(
            c.estimated_improvement > 0.0,
            "Estimated improvement should be positive: got {}",
            c.estimated_improvement
        );
    }
}

/// Test 18: SOFTSIGN activation recommends ABSOLUTE (symmetric function).
#[test]
fn test_softsign_recommends_absolute() {
    let neurons = vec![("osc-ss".to_string(), "SOFTSIGN".to_string(), 0.0)];
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = if i % 2 == 0 { 0.5 } else { -0.5 };
            make_record("osc-ss", i, activation)
        })
        .collect();

    let candidates = detect_oscillating_neurons(&neurons, &[("osc-ss".to_string(), records)]);

    assert!(!candidates.is_empty());
    assert_eq!(
        candidates[0].recommended_squash, "ABSOLUTE",
        "SOFTSIGN should recommend ABSOLUTE"
    );
}

/// Test 19: Mean absolute activation is correctly computed.
#[test]
fn test_mean_abs_activation_correct() {
    let neurons = vec![("osc-mean".to_string(), "TANH".to_string(), 0.0)];
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = if i % 2 == 0 { 0.6 } else { -0.4 };
            make_record("osc-mean", i, activation)
        })
        .collect();

    let candidates = detect_oscillating_neurons(&neurons, &[("osc-mean".to_string(), records)]);

    assert!(!candidates.is_empty());
    let c = &candidates[0];
    // Expected: (50 * 0.6 + 50 * 0.4) / 100 = 0.5
    assert!(
        (c.mean_abs_activation - 0.5).abs() < 0.01,
        "Mean abs activation should be ~0.5, got {}",
        c.mean_abs_activation
    );
}
