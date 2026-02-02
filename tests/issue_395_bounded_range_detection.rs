//! Tests for Issue #395: Discover bounded ranges of observations and hidden neurons.
//!
//! Observations are finite numbers often normalised to -1..1. A value like -1 may
//! represent "null/invalid" rather than a meaningful low value. This module detects
//! neurons whose activation is concentrated in a sub-range, suggesting that values
//! outside that range are "invalid/null" sentinels. It recommends bias/weight
//! adjustments or activation function changes so the network can use the meaningful
//! range without being polluted by sentinel values.
//!
//! ## TDD Plan
//! 1. Detect neurons operating in a narrow bounded sub-range of their activation
//! 2. Verify sentinel values (e.g., -1) outside the active range are identified
//! 3. Verify neurons using their full range are NOT flagged
//! 4. Verify insufficient samples are rejected
//! 5. Test conversion to coordinated structural candidates
//! 6. Test mixed neurons — only bounded-range ones detected
//! 7. Test that estimated improvement is positive and reasonable

use neat_ai_discovery::analysis::bounded_range::{
    bounded_range_to_coordinated_candidates, detect_bounded_range_neurons, BoundedRangeCandidate,
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

/// Test 1: Neuron with activations clustered in a narrow positive range, plus
/// sentinel values at -1, should be detected as having a bounded range.
#[test]
fn test_detects_narrow_range_with_sentinel() {
    // 80 samples in the "meaningful" range 0.2..0.8
    // 20 samples at -1.0 (sentinel / null indicator)
    let mut records: Vec<DiscoverRecord> = (0..80)
        .map(|i| {
            let activation = 0.2 + 0.6 * (i as f32 / 80.0);
            record("hidden-debt", i, activation, Some(activation))
        })
        .collect();
    for i in 80..100 {
        records.push(record("hidden-debt", i, -1.0, Some(-1.0)));
    }

    let candidates = detect_bounded_range_neurons(
        &[("hidden-debt".to_string(), "TANH".to_string(), 0.0)],
        &[("hidden-debt".to_string(), records)],
    );

    assert_eq!(candidates.len(), 1, "Should detect bounded range neuron");
    let c = &candidates[0];
    assert_eq!(c.neuron_uuid, "hidden-debt");
    assert!(
        c.estimated_improvement > 0.0,
        "Estimated improvement should be positive"
    );
}

/// Test 2: Neuron with activations uniformly spread across the full range
/// should NOT be flagged.
#[test]
fn test_full_range_neuron_not_flagged() {
    // Activations uniformly distributed across -1..1
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = -1.0 + 2.0 * (i as f32 / 99.0);
            record("uniform", i, activation, Some(activation))
        })
        .collect();

    let candidates = detect_bounded_range_neurons(
        &[("uniform".to_string(), "TANH".to_string(), 0.0)],
        &[("uniform".to_string(), records)],
    );

    assert!(
        candidates.is_empty(),
        "Uniformly distributed neuron should not be flagged"
    );
}

/// Test 3: Insufficient samples should not trigger detection.
#[test]
fn test_insufficient_samples_not_flagged() {
    let records: Vec<DiscoverRecord> = (0..5).map(|i| record("few", i, 0.5, Some(0.5))).collect();

    let candidates = detect_bounded_range_neurons(
        &[("few".to_string(), "TANH".to_string(), 0.0)],
        &[("few".to_string(), records)],
    );

    assert!(
        candidates.is_empty(),
        "Too few samples should not trigger detection"
    );
}

/// Test 4: Neuron with sentinel at 0 and meaningful range in 0.5..0.9.
#[test]
fn test_detects_zero_sentinel_with_positive_range() {
    let mut records: Vec<DiscoverRecord> = (0..70)
        .map(|i| {
            let activation = 0.5 + 0.4 * (i as f32 / 70.0);
            record("zero-sentinel", i, activation, Some(activation))
        })
        .collect();
    // 30 samples at 0.0 (sentinel)
    for i in 70..100 {
        records.push(record("zero-sentinel", i, 0.0, Some(0.0)));
    }

    let candidates = detect_bounded_range_neurons(
        &[("zero-sentinel".to_string(), "LOGISTIC".to_string(), 0.0)],
        &[("zero-sentinel".to_string(), records)],
    );

    assert_eq!(
        candidates.len(),
        1,
        "Should detect bounded range with zero sentinel"
    );
}

/// Test 5: Mixed neurons — only the bounded-range one is detected.
#[test]
fn test_mixed_neurons_only_bounded_detected() {
    // Bounded range neuron: cluster at 0.4..0.7, sentinel at -1.0
    let mut bounded_records: Vec<DiscoverRecord> = (0..75)
        .map(|i| {
            let activation = 0.4 + 0.3 * (i as f32 / 75.0);
            record("bounded", i, activation, Some(activation))
        })
        .collect();
    for i in 75..100 {
        bounded_records.push(record("bounded", i, -1.0, Some(-1.0)));
    }

    // Normal neuron: spread across full range
    let normal_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = -0.9 + 1.8 * (i as f32 / 99.0);
            record("normal", i, activation, Some(activation))
        })
        .collect();

    let candidates = detect_bounded_range_neurons(
        &[
            ("bounded".to_string(), "TANH".to_string(), 0.0),
            ("normal".to_string(), "TANH".to_string(), 0.0),
        ],
        &[
            ("bounded".to_string(), bounded_records),
            ("normal".to_string(), normal_records),
        ],
    );

    assert_eq!(
        candidates.len(),
        1,
        "Should detect only the bounded range neuron"
    );
    assert_eq!(candidates[0].neuron_uuid, "bounded");
}

/// Test 6: Candidates produce correct coordinated structural operations.
#[test]
fn test_candidates_produce_coordinated_operations() {
    let candidate = BoundedRangeCandidate {
        neuron_uuid: "hidden-debt".to_string(),
        current_squash: "TANH".to_string(),
        active_range_low: 0.2,
        active_range_high: 0.8,
        sentinel_fraction: 0.2,
        estimated_improvement: 0.01,
    };

    let coordinated = bounded_range_to_coordinated_candidates(&[candidate]);

    assert!(
        !coordinated.is_empty(),
        "Should produce coordinated candidates"
    );

    let c = &coordinated[0];
    assert!(
        c.expected_creature_score_gain > 0.0,
        "Expected improvement should be positive"
    );
    assert!(
        c.comment.is_some(),
        "Should have a comment explaining the change"
    );

    // Should include a bias or weight adjustment operation
    let ops_json = serde_json::to_string(&c.operations).unwrap();
    let has_set_bias = ops_json.contains("setBias");
    let has_change_squash = ops_json.contains("changeSquash");
    assert!(
        has_set_bias || has_change_squash,
        "Should include setBias or changeSquash operation, got: {ops_json}"
    );
}

/// Test 7: Neuron with bimodal distribution (two clusters) is detected.
#[test]
fn test_detects_bimodal_distribution() {
    // Cluster 1: activations around 0.6..0.8
    // Cluster 2: activations at -1.0 (sentinel)
    let mut records: Vec<DiscoverRecord> = (0..60)
        .map(|i| {
            let activation = 0.6 + 0.2 * (i as f32 / 60.0);
            record("bimodal", i, activation, Some(activation))
        })
        .collect();
    for i in 60..100 {
        records.push(record("bimodal", i, -1.0, Some(-1.0)));
    }

    let candidates = detect_bounded_range_neurons(
        &[("bimodal".to_string(), "TANH".to_string(), 0.0)],
        &[("bimodal".to_string(), records)],
    );

    assert_eq!(
        candidates.len(),
        1,
        "Should detect bimodal distribution with sentinel"
    );
}

/// Test 8: Neuron with no records should not panic or be flagged.
#[test]
fn test_no_records_not_flagged() {
    let candidates = detect_bounded_range_neurons(
        &[("missing".to_string(), "TANH".to_string(), 0.0)],
        &[], // no records at all
    );

    assert!(candidates.is_empty(), "No records should not be flagged");
}

/// Test 9: All activations at the same value (constant) should not be flagged
/// as bounded range — that is a dead/constant neuron, handled elsewhere.
#[test]
fn test_constant_activation_not_flagged() {
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| record("constant", i, 0.5, Some(0.5)))
        .collect();

    let candidates = detect_bounded_range_neurons(
        &[("constant".to_string(), "TANH".to_string(), 0.0)],
        &[("constant".to_string(), records)],
    );

    assert!(
        candidates.is_empty(),
        "Constant activation should not be flagged as bounded range"
    );
}

/// Test 10: Sentinel fraction is correctly computed.
#[test]
fn test_sentinel_fraction_computed() {
    // 70 samples in range 0.2..0.8, 30 at -1.0
    let mut records: Vec<DiscoverRecord> = (0..70)
        .map(|i| {
            let activation = 0.2 + 0.6 * (i as f32 / 70.0);
            record("frac-test", i, activation, Some(activation))
        })
        .collect();
    for i in 70..100 {
        records.push(record("frac-test", i, -1.0, Some(-1.0)));
    }

    let candidates = detect_bounded_range_neurons(
        &[("frac-test".to_string(), "TANH".to_string(), 0.0)],
        &[("frac-test".to_string(), records)],
    );

    assert_eq!(candidates.len(), 1);
    let c = &candidates[0];
    // 30 out of 100 samples are sentinels
    assert!(
        c.sentinel_fraction > 0.1,
        "Sentinel fraction should be > 0.1, got {}",
        c.sentinel_fraction
    );
    assert!(
        c.sentinel_fraction < 0.5,
        "Sentinel fraction should be < 0.5, got {}",
        c.sentinel_fraction
    );
}
