//! Tests for Issue #640: Pre-activation distribution shape detector (bimodal neuron splitting).
//!
//! Detects neurons whose pre-activation (`value`) distribution is bimodal or multi-modal,
//! indicating the neuron is serving two distinct input regimes and should be split.
//!
//! This differs from oscillating neuron detection (Issue #358) which analyses post-activation
//! sign changes, and from activation mismatch (Issue #543) which checks RELU negative fraction.
//! Bimodal detection specifically examines the **shape** of the pre-activation distribution.
//!
//! ## TDD Plan
//! 1. Verify bimodal distribution is detected (two well-separated clusters)
//! 2. Verify unimodal distribution is NOT detected
//! 3. Verify neurons with `value: None` are skipped gracefully
//! 4. Verify insufficient samples are excluded
//! 5. Verify coordinated candidate produces `addNeuron` operations
//! 6. Verify multi-modal (3+ modes) is also detected
//! 7. Verify overlapping clusters are NOT detected (insufficient separation)
//! 8. Verify multiple neurons — only bimodal ones detected
//! 9. Verify candidates are sorted by estimated improvement
//! 10. Verify empty records produce no candidates

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use neat_ai_discovery::analysis::detection::bimodal_neuron::{
    bimodal_neurons_to_coordinated_candidates, detect_bimodal_neurons,
};
use neat_ai_discovery::types::DiscoverRecord;

/// Helper: create a `DiscoverRecord` with a specific pre-activation value.
fn make_record(neuron_uuid: &str, obs_index: u32, value: Option<f32>) -> DiscoverRecord {
    DiscoverRecord {
        obs_index,
        neuron_uuid: neuron_uuid.to_string(),
        value,
        activation: value.unwrap_or(0.0).tanh(), // post-activation is irrelevant here
        errors: vec![0.01],
    }
}

/// Test 1: A clearly bimodal distribution (two well-separated clusters) is detected.
#[test]
fn test_bimodal_distribution_detected() {
    let neurons = vec![("bimodal-a".to_string(), "TANH".to_string(), 0.0)];
    // Half the samples cluster around -2.0, half around +2.0
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let value = if i < 50 {
                -2.0 + 0.1 * (i as f32 * 0.1).sin()
            } else {
                2.0 + 0.1 * ((i - 50) as f32 * 0.1).sin()
            };
            make_record("bimodal-a", i, Some(value))
        })
        .collect();

    let candidates = detect_bimodal_neurons(&neurons, &[("bimodal-a".to_string(), records)]);

    assert!(
        !candidates.is_empty(),
        "Should detect bimodal pre-activation distribution"
    );
    let c = &candidates[0];
    assert_eq!(c.neuron_uuid, "bimodal-a");
    assert_eq!(c.sample_count, 100);
    assert!(
        c.bimodality_score > 0.0,
        "Bimodality score should be positive: got {}",
        c.bimodality_score
    );
}

/// Test 2: A unimodal (normal) distribution is NOT detected.
#[test]
fn test_unimodal_distribution_not_detected() {
    let neurons = vec![("unimodal".to_string(), "TANH".to_string(), 0.0)];
    // Linear ramp from 0.8 to 1.2 — clearly unimodal (uniform distribution)
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let value = 0.8 + 0.4 * (i as f32 / 99.0);
            make_record("unimodal", i, Some(value))
        })
        .collect();

    let candidates = detect_bimodal_neurons(&neurons, &[("unimodal".to_string(), records)]);

    assert!(
        candidates.is_empty(),
        "Unimodal distribution should not be flagged as bimodal"
    );
}

/// Test 3: Neurons with `value: None` are skipped gracefully.
#[test]
fn test_none_values_skipped() {
    let neurons = vec![("no-value".to_string(), "TANH".to_string(), 0.0)];
    let records: Vec<DiscoverRecord> = (0..100).map(|i| make_record("no-value", i, None)).collect();

    let candidates = detect_bimodal_neurons(&neurons, &[("no-value".to_string(), records)]);

    assert!(
        candidates.is_empty(),
        "Neurons with None values should not be flagged"
    );
}

/// Test 4: Insufficient samples are excluded.
#[test]
fn test_bimodal_neuron_insufficient_samples_excluded() {
    let neurons = vec![("few".to_string(), "TANH".to_string(), 0.0)];
    let records: Vec<DiscoverRecord> = (0..5)
        .map(|i| {
            let value = if i < 3 { -2.0 } else { 2.0 };
            make_record("few", i, Some(value))
        })
        .collect();

    let candidates = detect_bimodal_neurons(&neurons, &[("few".to_string(), records)]);

    assert!(
        candidates.is_empty(),
        "Too few samples should not trigger detection"
    );
}

/// Test 5: Coordinated candidates produce `addNeuron` operations.
#[test]
fn test_coordinated_candidate_has_add_neuron() {
    let neurons = vec![("bimodal-coord".to_string(), "TANH".to_string(), 0.5)];
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let value = if i < 50 { -3.0 } else { 3.0 };
            make_record("bimodal-coord", i, Some(value))
        })
        .collect();

    let candidates = detect_bimodal_neurons(&neurons, &[("bimodal-coord".to_string(), records)]);
    assert!(!candidates.is_empty(), "Should detect bimodal neuron");

    let coordinated = bimodal_neurons_to_coordinated_candidates(&candidates);

    assert!(
        !coordinated.is_empty(),
        "Should produce coordinated candidates"
    );
    let c = &coordinated[0];
    assert!(
        c.expected_creature_score_gain > 0.0,
        "Expected gain should be positive"
    );
    assert!(c.comment.is_some(), "Should have explanatory comment");

    let ops_json = serde_json::to_string(&c.operations).unwrap();
    assert!(
        ops_json.contains("addNeuron"),
        "Should include addNeuron operation to split the bimodal neuron, got: {ops_json}"
    );
}

/// Test 6: Multi-modal distribution (3+ modes) is also detected.
#[test]
fn test_multimodal_detected() {
    let neurons = vec![("trimodal".to_string(), "TANH".to_string(), 0.0)];
    // Three clusters: around -3.0, 0.0, and +3.0
    let records: Vec<DiscoverRecord> = (0..150)
        .map(|i| {
            let value = if i < 50 {
                -3.0 + 0.05 * (i as f32 * 0.1).sin()
            } else if i < 100 {
                0.0 + 0.05 * ((i - 50) as f32 * 0.1).sin()
            } else {
                3.0 + 0.05 * ((i - 100) as f32 * 0.1).sin()
            };
            make_record("trimodal", i, Some(value))
        })
        .collect();

    let candidates = detect_bimodal_neurons(&neurons, &[("trimodal".to_string(), records)]);

    assert!(
        !candidates.is_empty(),
        "Multi-modal distribution should also be detected"
    );
}

/// Test 7: Overlapping clusters (insufficient separation) are NOT detected.
#[test]
fn test_overlapping_clusters_not_detected() {
    let neurons = vec![("overlap".to_string(), "TANH".to_string(), 0.0)];
    // Two clusters very close together — linear ramps that heavily overlap
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let value = if i < 50 {
                // Range [0.5, 1.5] — linear ramp
                0.5 + 1.0 * (i as f32 / 49.0)
            } else {
                // Range [0.7, 1.7] — linear ramp, heavily overlapping
                0.7 + 1.0 * ((i - 50) as f32 / 49.0)
            };
            make_record("overlap", i, Some(value))
        })
        .collect();

    let candidates = detect_bimodal_neurons(&neurons, &[("overlap".to_string(), records)]);

    assert!(
        candidates.is_empty(),
        "Overlapping clusters should not be flagged as bimodal"
    );
}

/// Test 8: Multiple neurons — only bimodal ones detected.
#[test]
fn test_bimodal_neuron_mixed_neurons_filters_correctly() {
    let neurons = vec![
        ("bimodal-mix".to_string(), "TANH".to_string(), 0.0),
        ("unimodal-mix".to_string(), "RELU".to_string(), 0.0),
    ];

    let bimodal_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let value = if i < 50 { -2.0 } else { 2.0 };
            make_record("bimodal-mix", i, Some(value))
        })
        .collect();

    let unimodal_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            // Linear ramp — clearly unimodal
            let value = 0.3 + 0.4 * (i as f32 / 99.0);
            make_record("unimodal-mix", i, Some(value))
        })
        .collect();

    let candidates = detect_bimodal_neurons(
        &neurons,
        &[
            ("bimodal-mix".to_string(), bimodal_records),
            ("unimodal-mix".to_string(), unimodal_records),
        ],
    );

    assert_eq!(candidates.len(), 1, "Should detect only the bimodal neuron");
    assert_eq!(candidates[0].neuron_uuid, "bimodal-mix");
}

/// Test 9: Candidates are sorted by estimated improvement (best first).
#[test]
fn test_bimodal_neuron_candidates_sorted_by_improvement() {
    let neurons = vec![
        ("bimodal-small".to_string(), "TANH".to_string(), 0.0),
        ("bimodal-large".to_string(), "TANH".to_string(), 0.0),
    ];

    // Small separation
    let small_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let value = if i < 50 { -1.0 } else { 1.0 };
            make_record("bimodal-small", i, Some(value))
        })
        .collect();

    // Large separation → should have higher estimated improvement
    let large_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let value = if i < 50 { -5.0 } else { 5.0 };
            make_record("bimodal-large", i, Some(value))
        })
        .collect();

    let candidates = detect_bimodal_neurons(
        &neurons,
        &[
            ("bimodal-small".to_string(), small_records),
            ("bimodal-large".to_string(), large_records),
        ],
    );

    assert_eq!(candidates.len(), 2, "Should detect both bimodal neurons");
    assert!(
        candidates[0].estimated_improvement >= candidates[1].estimated_improvement,
        "Candidates should be sorted by estimated improvement (best first): {} >= {}",
        candidates[0].estimated_improvement,
        candidates[1].estimated_improvement
    );
    assert_eq!(
        candidates[0].neuron_uuid, "bimodal-large",
        "Larger separation should rank higher"
    );
}

/// Test 10: Empty records produce no candidates.
#[test]
fn test_bimodal_neuron_empty_records_no_candidates() {
    let neurons = vec![("empty".to_string(), "TANH".to_string(), 0.0)];
    let records: Vec<DiscoverRecord> = vec![];

    let candidates = detect_bimodal_neurons(&neurons, &[("empty".to_string(), records)]);

    assert!(
        candidates.is_empty(),
        "Empty records should produce no candidates"
    );
}

/// Test 11: Missing neuron records produce no candidate.
#[test]
fn test_bimodal_neuron_missing_neuron_records_no_candidate() {
    let neurons = vec![("missing".to_string(), "TANH".to_string(), 0.0)];
    let candidates = detect_bimodal_neurons(&neurons, &[]);

    assert!(
        candidates.is_empty(),
        "Missing records should produce no candidate"
    );
}

/// Test 12: Partial None values — uses only available values.
#[test]
fn test_partial_none_values_still_detects() {
    let neurons = vec![("partial".to_string(), "TANH".to_string(), 0.0)];
    // 60 records with values (bimodal), 40 with None
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            if i < 60 {
                let value = if i < 30 { -3.0 } else { 3.0 };
                make_record("partial", i, Some(value))
            } else {
                make_record("partial", i, None)
            }
        })
        .collect();

    let candidates = detect_bimodal_neurons(&neurons, &[("partial".to_string(), records)]);

    assert!(
        !candidates.is_empty(),
        "Should still detect bimodality from available value records"
    );
}

/// Test 13: Estimated improvement is positive for detected neurons.
#[test]
fn test_bimodal_neuron_estimated_improvement_positive() {
    let neurons = vec![("bimodal-imp".to_string(), "TANH".to_string(), 0.0)];
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let value = if i < 50 { -2.0 } else { 2.0 };
            make_record("bimodal-imp", i, Some(value))
        })
        .collect();

    let candidates = detect_bimodal_neurons(&neurons, &[("bimodal-imp".to_string(), records)]);

    assert!(!candidates.is_empty());
    for c in &candidates {
        assert!(
            c.estimated_improvement > 0.0,
            "Estimated improvement should be positive: got {}",
            c.estimated_improvement
        );
    }
}

/// Test 14: Coordinated candidate comment mentions bimodal splitting.
#[test]
fn test_coordinated_candidate_comment() {
    let neurons = vec![("bimodal-comment".to_string(), "TANH".to_string(), 0.0)];
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let value = if i < 50 { -2.0 } else { 2.0 };
            make_record("bimodal-comment", i, Some(value))
        })
        .collect();

    let candidates = detect_bimodal_neurons(&neurons, &[("bimodal-comment".to_string(), records)]);
    assert!(!candidates.is_empty());

    let coordinated = bimodal_neurons_to_coordinated_candidates(&candidates);
    assert!(!coordinated.is_empty());

    let comment = coordinated[0].comment.as_ref().unwrap();
    assert!(
        comment.contains("bimodal") || comment.contains("Bimodal"),
        "Comment should mention bimodal: got {comment}"
    );
}
