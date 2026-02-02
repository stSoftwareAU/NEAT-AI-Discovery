//! Tests for Issue #395: Discover bounded ranges of observations and hidden neurons.
//!
//! Observations are finite numbers often normalised to -1..1, but they do not have
//! a concept of null. A sentinel value (e.g., -1 or 0) is used instead. This module
//! detects neurons where a bounded sub-range of activation values is meaningful while
//! values at extremes act as sentinel/null markers.
//!
//! When a bounded range is detected, the module recommends a coordinated structural
//! candidate that inserts an IF-gated path so the network can treat the meaningful
//! range separately from sentinel values.
//!
//! ## TDD Plan
//! 1. Detect observation with bimodal distribution: meaningful range + sentinel cluster
//! 2. Verify active neurons with uniform spread are NOT flagged
//! 3. Verify neurons with insufficient samples are NOT flagged
//! 4. Verify output neurons are NOT flagged
//! 5. Test sentinel at lower bound (-1 sentinel, meaningful values in 0..1)
//! 6. Test sentinel at upper bound (1 sentinel, meaningful values in -1..0)
//! 7. Test sentinel at zero (0 sentinel, meaningful values away from zero)
//! 8. Test coordinated candidate conversion produces IF-gated structure
//! 9. Test mixed neurons — only bounded-range ones are detected
//! 10. Test that estimated improvement is positive

mod common;

use common::{make_creature, neuron, record, synapse};
use neat_ai_discovery::analysis::bounded_range::{
    bounded_ranges_to_coordinated_candidates, detect_bounded_ranges, BoundedRangeCandidate,
};
use neat_ai_discovery::types::DiscoverRecord;

/// Helper: create records with a sentinel cluster at `sentinel_value` and meaningful
/// values spread uniformly in `[range_lo, range_hi]`.
fn make_bimodal_records(
    uuid: &str,
    sentinel_value: f32,
    sentinel_fraction: f32,
    range_lo: f32,
    range_hi: f32,
    total: u32,
) -> Vec<DiscoverRecord> {
    let sentinel_count = (total as f32 * sentinel_fraction) as u32;
    let mut records = Vec::with_capacity(total as usize);

    for i in 0..sentinel_count {
        records.push(record(uuid, i, sentinel_value, Some(sentinel_value)));
    }

    let meaningful_count = total - sentinel_count;
    for i in 0..meaningful_count {
        let t = i as f32 / meaningful_count.max(1) as f32;
        let activation = range_lo + t * (range_hi - range_lo);
        records.push(record(
            uuid,
            sentinel_count + i,
            activation,
            Some(activation),
        ));
    }

    records
}

/// Test 1: Observation with sentinel cluster at -1 and meaningful values in 0..1 is detected.
#[test]
fn test_detects_sentinel_at_lower_bound() {
    let creature = make_creature(
        vec![
            neuron("obs-debt", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![synapse("obs-debt", "output-1", 0.5)],
    );

    // 30% of samples are sentinel (-1), 70% are meaningful (0..1)
    let records = make_bimodal_records("obs-debt", -1.0, 0.30, 0.0, 1.0, 200);

    let candidates = detect_bounded_ranges(&creature, &[("obs-debt".to_string(), records)]);

    assert_eq!(candidates.len(), 1, "Should detect bounded range neuron");
    let c = &candidates[0];
    assert_eq!(c.neuron_uuid, "obs-debt");
    assert!(
        c.sentinel_value < 0.0,
        "Sentinel should be at the lower bound"
    );
    assert!(
        c.meaningful_range_lo > c.sentinel_value,
        "Meaningful range should be above sentinel"
    );
}

/// Test 2: Neuron with uniform spread across full range is NOT flagged.
#[test]
fn test_uniform_spread_not_flagged() {
    let creature = make_creature(
        vec![
            neuron("obs-normal", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![synapse("obs-normal", "output-1", 0.5)],
    );

    // Uniform activation across -1..1 — no sentinel cluster
    let records: Vec<DiscoverRecord> = (0..200)
        .map(|i| {
            let activation = -1.0 + 2.0 * (i as f32 / 199.0);
            record("obs-normal", i, activation, Some(activation))
        })
        .collect();

    let candidates = detect_bounded_ranges(&creature, &[("obs-normal".to_string(), records)]);

    assert!(
        candidates.is_empty(),
        "Uniform spread should not be flagged as bounded range"
    );
}

/// Test 3: Insufficient samples should not trigger detection.
#[test]
fn test_insufficient_samples_not_flagged() {
    let creature = make_creature(
        vec![
            neuron("obs-few", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![synapse("obs-few", "output-1", 0.5)],
    );

    let records = make_bimodal_records("obs-few", -1.0, 0.30, 0.0, 1.0, 10);

    let candidates = detect_bounded_ranges(&creature, &[("obs-few".to_string(), records)]);

    assert!(
        candidates.is_empty(),
        "Too few samples should not trigger detection"
    );
}

/// Test 4: Output neurons are NOT flagged.
#[test]
fn test_output_neurons_not_flagged() {
    let creature = make_creature(vec![neuron("output-1", "output", "IDENTITY")], vec![]);

    let records = make_bimodal_records("output-1", -1.0, 0.30, 0.0, 1.0, 200);

    let candidates = detect_bounded_ranges(&creature, &[("output-1".to_string(), records)]);

    assert!(
        candidates.is_empty(),
        "Output neurons should not be flagged"
    );
}

/// Test 5: Sentinel at upper bound (1 sentinel, meaningful values in -1..0).
#[test]
fn test_detects_sentinel_at_upper_bound() {
    let creature = make_creature(
        vec![
            neuron("obs-upper", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![synapse("obs-upper", "output-1", 0.5)],
    );

    // 25% sentinel at +1, meaningful values in -1..0
    let records = make_bimodal_records("obs-upper", 1.0, 0.25, -1.0, 0.0, 200);

    let candidates = detect_bounded_ranges(&creature, &[("obs-upper".to_string(), records)]);

    assert_eq!(candidates.len(), 1, "Should detect upper-bound sentinel");
    let c = &candidates[0];
    assert!(
        c.sentinel_value > 0.0,
        "Sentinel should be at the upper bound"
    );
}

/// Test 6: Sentinel at zero (0 sentinel, meaningful values away from zero).
#[test]
fn test_detects_sentinel_at_zero() {
    let creature = make_creature(
        vec![
            neuron("obs-zero", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![synapse("obs-zero", "output-1", 0.5)],
    );

    // 30% sentinel at 0, meaningful values in 0.3..1.0
    let records = make_bimodal_records("obs-zero", 0.0, 0.30, 0.3, 1.0, 200);

    let candidates = detect_bounded_ranges(&creature, &[("obs-zero".to_string(), records)]);

    assert_eq!(candidates.len(), 1, "Should detect zero sentinel");
    let c = &candidates[0];
    assert!(
        (c.sentinel_value).abs() < 0.05,
        "Sentinel should be near zero"
    );
}

/// Test 7: Coordinated candidate produces IF-gated structural operations.
#[test]
fn test_candidates_produce_if_gated_operations() {
    let creature = make_creature(
        vec![
            neuron("obs-debt", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![synapse("obs-debt", "output-1", 0.5)],
    );

    let candidate = BoundedRangeCandidate {
        neuron_uuid: "obs-debt".to_string(),
        sentinel_value: -1.0,
        sentinel_fraction: 0.30,
        meaningful_range_lo: 0.0,
        meaningful_range_hi: 1.0,
        sample_count: 200,
        detection_confidence: 0.85,
        estimated_improvement: 0.005,
        downstream_uuids: vec!["output-1".to_string()],
    };

    let coordinated = bounded_ranges_to_coordinated_candidates(&[candidate], &creature);

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
        "Should have a comment explaining the recommendation"
    );

    // Check that operations include an AddNeuron with IF squash
    let ops_json = serde_json::to_string(&c.operations).unwrap();
    assert!(
        ops_json.contains("addNeuron"),
        "Should include addNeuron operation, got: {ops_json}"
    );
    assert!(
        ops_json.contains("IF"),
        "Should use IF squash function, got: {ops_json}"
    );
}

/// Test 8: Mixed neurons — only bounded-range ones are detected.
#[test]
fn test_mixed_neurons_only_bounded_range_detected() {
    let creature = make_creature(
        vec![
            neuron("obs-bounded", "input", "IDENTITY"),
            neuron("obs-normal", "input", "IDENTITY"),
            neuron("hidden-1", "hidden", "TANH"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("obs-bounded", "hidden-1", 0.5),
            synapse("obs-normal", "hidden-1", 0.3),
            synapse("hidden-1", "output-1", 0.8),
        ],
    );

    // obs-bounded: bimodal with sentinel
    let bounded_records = make_bimodal_records("obs-bounded", -1.0, 0.30, 0.0, 1.0, 200);

    // obs-normal: uniform, no sentinel
    let normal_records: Vec<DiscoverRecord> = (0..200)
        .map(|i| {
            let activation = -1.0 + 2.0 * (i as f32 / 199.0);
            record("obs-normal", i, activation, Some(activation))
        })
        .collect();

    let candidates = detect_bounded_ranges(
        &creature,
        &[
            ("obs-bounded".to_string(), bounded_records),
            ("obs-normal".to_string(), normal_records),
        ],
    );

    assert_eq!(
        candidates.len(),
        1,
        "Only bounded-range neuron should be detected"
    );
    assert_eq!(candidates[0].neuron_uuid, "obs-bounded");
}

/// Test 9: Hidden neuron with bounded range is also detected.
#[test]
fn test_hidden_neuron_with_bounded_range_detected() {
    let creature = make_creature(
        vec![
            neuron("obs-1", "input", "IDENTITY"),
            neuron("hidden-bounded", "hidden", "TANH"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("obs-1", "hidden-bounded", 0.5),
            synapse("hidden-bounded", "output-1", 0.8),
        ],
    );

    // Hidden neuron: 30% at -1 (tanh saturation acting as sentinel), 70% in 0..0.8
    let records = make_bimodal_records("hidden-bounded", -1.0, 0.30, 0.0, 0.8, 200);

    let candidates = detect_bounded_ranges(&creature, &[("hidden-bounded".to_string(), records)]);

    assert_eq!(
        candidates.len(),
        1,
        "Hidden neuron with bounded range should be detected"
    );
    assert_eq!(candidates[0].neuron_uuid, "hidden-bounded");
}

/// Test 10: Estimated improvement is positive for all candidates.
#[test]
fn test_estimated_improvement_positive() {
    let creature = make_creature(
        vec![
            neuron("obs-debt", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![synapse("obs-debt", "output-1", 0.5)],
    );

    let records = make_bimodal_records("obs-debt", -1.0, 0.30, 0.0, 1.0, 200);

    let candidates = detect_bounded_ranges(&creature, &[("obs-debt".to_string(), records)]);

    assert!(!candidates.is_empty());
    for c in &candidates {
        assert!(
            c.estimated_improvement > 0.0,
            "Estimated improvement should be positive"
        );
    }
}

/// Test 11: Sample count is correctly recorded.
#[test]
fn test_sample_count_recorded() {
    let creature = make_creature(
        vec![
            neuron("obs-debt", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![synapse("obs-debt", "output-1", 0.5)],
    );

    let records = make_bimodal_records("obs-debt", -1.0, 0.30, 0.0, 1.0, 300);

    let candidates = detect_bounded_ranges(&creature, &[("obs-debt".to_string(), records)]);

    assert_eq!(candidates.len(), 1);
    assert_eq!(
        candidates[0].sample_count, 300,
        "Sample count should match record count"
    );
}

/// Test 12: Detection confidence increases with more samples.
#[test]
fn test_confidence_increases_with_more_samples() {
    let creature = make_creature(
        vec![
            neuron("obs-debt", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![synapse("obs-debt", "output-1", 0.5)],
    );

    let records_small = make_bimodal_records("obs-debt", -1.0, 0.30, 0.0, 1.0, 50);
    let records_large = make_bimodal_records("obs-debt", -1.0, 0.30, 0.0, 1.0, 500);

    let candidates_small =
        detect_bounded_ranges(&creature, &[("obs-debt".to_string(), records_small)]);
    let candidates_large =
        detect_bounded_ranges(&creature, &[("obs-debt".to_string(), records_large)]);

    assert!(!candidates_small.is_empty(), "Small sample should detect");
    assert!(!candidates_large.is_empty(), "Large sample should detect");
    assert!(
        candidates_large[0].detection_confidence >= candidates_small[0].detection_confidence,
        "More samples should yield equal or higher confidence"
    );
}

/// Test 13: Neurons with no downstream synapses are still detected
/// (input neurons connected to nothing may still be flagged for future use).
#[test]
fn test_neuron_with_downstream_found() {
    let creature = make_creature(
        vec![
            neuron("obs-debt", "input", "IDENTITY"),
            neuron("hidden-1", "hidden", "TANH"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("obs-debt", "hidden-1", 0.5),
            synapse("hidden-1", "output-1", 0.8),
        ],
    );

    let records = make_bimodal_records("obs-debt", -1.0, 0.30, 0.0, 1.0, 200);

    let candidates = detect_bounded_ranges(&creature, &[("obs-debt".to_string(), records)]);

    assert_eq!(candidates.len(), 1);
    assert!(
        !candidates[0].downstream_uuids.is_empty(),
        "Should identify downstream neurons"
    );
}
