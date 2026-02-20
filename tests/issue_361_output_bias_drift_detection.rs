//! Tests for Issue #361: Output Bias Drift Detection.
//!
//! Dedicated tests for output bias drift detection as a structural discovery method.
//! Output neurons with consistent error sign bias (predominantly positive or negative
//! errors) indicate a systematic prediction offset that can be corrected by adjusting
//! the neuron's bias parameter. The detection recommends `setBias` coordinated
//! structural candidates.
//!
//! ## TDD Plan
//! 1. Verify detection of output neuron with predominantly positive errors
//! 2. Verify detection of output neuron with predominantly negative errors
//! 3. Verify balanced errors are not flagged
//! 4. Verify hidden neurons are excluded (only output neurons)
//! 5. Verify input neurons are excluded
//! 6. Verify insufficient samples are excluded
//! 7. Verify noise-level errors (tiny magnitude) are excluded
//! 8. Verify coordinated candidate conversion produces correct SetBias operations
//! 9. Verify multiple outputs: only biased ones detected
//! 10. Verify current bias is correctly recorded in candidate
//! 11. Verify recommended bias delta is negative of mean error
//! 12. Verify positive error fraction is correctly computed
//! 13. Verify candidates are sorted by estimated improvement (best first)
//! 14. Verify estimated improvement is always positive
//! 15. Verify coordinated candidate comment includes diagnostics
//! 16. Verify exactly minimum sample count (20) is accepted
//! 17. Verify nineteen samples (below minimum) is rejected
//! 18. Verify exactly 70% same sign at threshold boundary
//! 19. Verify 69% same sign below threshold is rejected
//! 20. Verify mean error near noise threshold boundary (0.01)
//! 21. Verify empty records produce no candidates
//! 22. Verify empty creature (no neurons) produces no candidates
//! 23. Verify coordinated candidate SetBias value is current_bias + delta
//! 24. Verify each coordinated candidate has exactly one operation
//! 25. Verify empty candidates conversion produces empty results
//! 26. Verify coordinated candidates sorted by expected score gain
//! 27. Verify all-positive errors produce correct statistics
//! 28. Verify all-negative errors produce correct statistics
//! 29. Verify records with empty error vectors are handled

use neat_ai_discovery::analysis::recommendation::output_bias_drift::{
    detect_output_bias_drift, output_bias_drift_to_coordinated_candidates,
};
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

/// Helper: create a DiscoverRecord for a neuron with given activation and errors.
fn make_record(
    neuron_uuid: &str,
    obs_index: u32,
    activation: f32,
    errors: Vec<f32>,
) -> DiscoverRecord {
    DiscoverRecord {
        obs_index,
        neuron_uuid: neuron_uuid.to_string(),
        value: Some(activation * 0.8),
        activation,
        errors,
    }
}

/// Helper: build a minimal creature.
fn make_creature(neurons: Vec<NeuronJson>, synapses: Vec<SynapseJson>) -> CreatureJson {
    CreatureJson {
        neurons,
        synapses,
        input: 1,
        output: 1,
    }
}

/// Helper: build a NeuronJson.
fn neuron(uuid: &str, neuron_type: &str, bias: f32) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
        squash: "IDENTITY".to_string(),
        bias,
    }
}

/// Helper: build a SynapseJson.
fn synapse(from: &str, to: &str, weight: f32) -> SynapseJson {
    SynapseJson {
        from_uuid: from.to_string(),
        to_uuid: to.to_string(),
        weight,
        synapse_type: None,
    }
}

// ---------------------------------------------------------------------------
// Test 1: Output neuron with predominantly positive errors is detected.
// ---------------------------------------------------------------------------
#[test]
fn test_detects_positive_bias_drift() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", 0.0),
            neuron("output-1", "output", 0.0),
        ],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    // 80% positive errors -> output predicting too low
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let error = if i < 80 { 0.3 } else { -0.1 };
            make_record("output-1", i, 0.5, vec![error])
        })
        .collect();

    let candidates = detect_output_bias_drift(&creature, &[("output-1".to_string(), records)]);

    assert_eq!(candidates.len(), 1, "Should detect one biased output");
    let c = &candidates[0];
    assert_eq!(c.neuron_uuid, "output-1");
    assert!(c.mean_error > 0.0, "Mean error should be positive");
    assert!(
        c.positive_error_fraction > 0.7,
        "Positive error fraction should be > 0.7, got {}",
        c.positive_error_fraction
    );
}

// ---------------------------------------------------------------------------
// Test 2: Output neuron with predominantly negative errors is detected.
// ---------------------------------------------------------------------------
#[test]
fn test_detects_negative_bias_drift() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", 0.0),
            neuron("output-1", "output", 0.5),
        ],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    // 85% negative errors -> output predicting too high
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let error = if i < 85 { -0.4 } else { 0.1 };
            make_record("output-1", i, 0.5, vec![error])
        })
        .collect();

    let candidates = detect_output_bias_drift(&creature, &[("output-1".to_string(), records)]);

    assert_eq!(candidates.len(), 1, "Should detect one biased output");
    let c = &candidates[0];
    assert!(c.mean_error < 0.0, "Mean error should be negative");
    assert!(
        c.recommended_bias_delta > 0.0,
        "Should recommend positive bias adjustment for negative errors"
    );
}

// ---------------------------------------------------------------------------
// Test 3: Output neuron with balanced errors is NOT flagged.
// ---------------------------------------------------------------------------
#[test]
fn test_balanced_errors_not_flagged() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", 0.0),
            neuron("output-1", "output", 0.0),
        ],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    // Roughly 50/50 positive and negative errors
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let error = if i % 2 == 0 { 0.2 } else { -0.2 };
            make_record("output-1", i, 0.5, vec![error])
        })
        .collect();

    let candidates = detect_output_bias_drift(&creature, &[("output-1".to_string(), records)]);

    assert!(
        candidates.is_empty(),
        "Balanced errors should not be flagged as bias drift"
    );
}

// ---------------------------------------------------------------------------
// Test 4: Hidden neurons are excluded (only output neurons analysed).
// ---------------------------------------------------------------------------
#[test]
fn test_hidden_neuron_excluded() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", 0.0),
            neuron("hidden-1", "hidden", 0.0),
            neuron("output-1", "output", 0.0),
        ],
        vec![
            synapse("input-1", "hidden-1", 0.5),
            synapse("hidden-1", "output-1", 0.3),
        ],
    );

    // Hidden neuron with biased errors - should NOT be detected
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| make_record("hidden-1", i, 0.5, vec![0.5]))
        .collect();

    let candidates = detect_output_bias_drift(&creature, &[("hidden-1".to_string(), records)]);

    assert!(
        candidates.is_empty(),
        "Hidden neurons should not be flagged"
    );
}

// ---------------------------------------------------------------------------
// Test 5: Input neurons are excluded.
// ---------------------------------------------------------------------------
#[test]
fn test_input_neuron_excluded() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", 0.0),
            neuron("output-1", "output", 0.0),
        ],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    // Input neuron with biased errors - should NOT be detected
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| make_record("input-1", i, 0.5, vec![0.5]))
        .collect();

    let candidates = detect_output_bias_drift(&creature, &[("input-1".to_string(), records)]);

    assert!(candidates.is_empty(), "Input neurons should not be flagged");
}

// ---------------------------------------------------------------------------
// Test 6: Insufficient samples should not trigger detection.
// ---------------------------------------------------------------------------
#[test]
fn test_insufficient_samples_not_flagged() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", 0.0),
            neuron("output-1", "output", 0.0),
        ],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    let records: Vec<DiscoverRecord> = (0..5)
        .map(|i| make_record("output-1", i, 0.5, vec![0.5]))
        .collect();

    let candidates = detect_output_bias_drift(&creature, &[("output-1".to_string(), records)]);

    assert!(
        candidates.is_empty(),
        "Too few samples should not trigger detection"
    );
}

// ---------------------------------------------------------------------------
// Test 7: Very small errors (noise level) are NOT flagged.
// ---------------------------------------------------------------------------
#[test]
fn test_noise_level_errors_not_flagged() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", 0.0),
            neuron("output-1", "output", 0.0),
        ],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    // All positive but tiny errors (noise level)
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| make_record("output-1", i, 0.5, vec![0.001]))
        .collect();

    let candidates = detect_output_bias_drift(&creature, &[("output-1".to_string(), records)]);

    assert!(
        candidates.is_empty(),
        "Noise-level errors should not be flagged"
    );
}

// ---------------------------------------------------------------------------
// Test 8: Coordinated candidate conversion produces correct SetBias operations.
// ---------------------------------------------------------------------------
#[test]
fn test_coordinated_candidate_conversion() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", 0.0),
            neuron("output-1", "output", 0.0),
        ],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let error = if i < 80 { 0.3 } else { -0.1 };
            make_record("output-1", i, 0.5, vec![error])
        })
        .collect();

    let candidates = detect_output_bias_drift(&creature, &[("output-1".to_string(), records)]);
    assert!(!candidates.is_empty(), "Should detect bias drift");

    let coordinated = output_bias_drift_to_coordinated_candidates(&candidates);

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

    // Check that operations include a SetBias
    let ops_json = serde_json::to_string(&c.operations).unwrap();
    assert!(
        ops_json.contains("setBias"),
        "Should include setBias operation, got: {ops_json}"
    );
}

// ---------------------------------------------------------------------------
// Test 9: Multiple output neurons — only biased ones are detected.
// ---------------------------------------------------------------------------
#[test]
fn test_multiple_outputs_only_biased_detected() {
    let creature = CreatureJson {
        neurons: vec![
            neuron("input-1", "input", 0.0),
            neuron("output-1", "output", 0.0),
            neuron("output-2", "output", 0.0),
        ],
        synapses: vec![
            synapse("input-1", "output-1", 0.5),
            synapse("input-1", "output-2", 0.3),
        ],
        input: 1,
        output: 2,
    };

    // output-1: biased (80% positive)
    let output_1_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let error = if i < 80 { 0.3 } else { -0.1 };
            make_record("output-1", i, 0.5, vec![error])
        })
        .collect();

    // output-2: balanced
    let output_2_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let error = if i % 2 == 0 { 0.2 } else { -0.2 };
            make_record("output-2", i, 0.5, vec![error])
        })
        .collect();

    let candidates = detect_output_bias_drift(
        &creature,
        &[
            ("output-1".to_string(), output_1_records),
            ("output-2".to_string(), output_2_records),
        ],
    );

    assert_eq!(candidates.len(), 1, "Should detect only the biased output");
    assert_eq!(candidates[0].neuron_uuid, "output-1");
}

// ---------------------------------------------------------------------------
// Test 10: Current bias is correctly recorded in candidate.
// ---------------------------------------------------------------------------
#[test]
fn test_current_bias_recorded() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", 0.0),
            neuron("output-1", "output", 0.3),
        ],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let error = if i < 80 { 0.3 } else { -0.1 };
            make_record("output-1", i, 0.5, vec![error])
        })
        .collect();

    let candidates = detect_output_bias_drift(&creature, &[("output-1".to_string(), records)]);

    assert_eq!(candidates.len(), 1);
    assert!(
        (candidates[0].current_bias - 0.3).abs() < 1e-6,
        "Current bias should be recorded correctly"
    );
}

// ---------------------------------------------------------------------------
// Test 11: Recommended bias delta is negative of mean error.
// ---------------------------------------------------------------------------
#[test]
fn test_recommended_bias_delta_is_negative_mean_error() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", 0.0),
            neuron("output-1", "output", 0.0),
        ],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    // All positive errors of 0.5
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| make_record("output-1", i, 0.5, vec![0.5]))
        .collect();

    let candidates = detect_output_bias_drift(&creature, &[("output-1".to_string(), records)]);

    assert_eq!(candidates.len(), 1);
    let c = &candidates[0];
    // Mean error = 0.5, so recommended delta should be -0.5
    assert!(
        (c.recommended_bias_delta - (-0.5)).abs() < 1e-6,
        "Recommended bias delta should be -mean_error: got {}",
        c.recommended_bias_delta
    );
    assert!(
        (c.mean_error - 0.5).abs() < 1e-6,
        "Mean error should be 0.5: got {}",
        c.mean_error
    );
}

// ---------------------------------------------------------------------------
// Test 12: Positive error fraction is correctly computed.
// ---------------------------------------------------------------------------
#[test]
fn test_positive_error_fraction_computed_correctly() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", 0.0),
            neuron("output-1", "output", 0.0),
        ],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    // 75 positive, 25 negative
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let error = if i < 75 { 0.3 } else { -0.3 };
            make_record("output-1", i, 0.5, vec![error])
        })
        .collect();

    let candidates = detect_output_bias_drift(&creature, &[("output-1".to_string(), records)]);

    assert_eq!(candidates.len(), 1);
    assert!(
        (candidates[0].positive_error_fraction - 0.75).abs() < 0.01,
        "Positive error fraction should be ~0.75: got {}",
        candidates[0].positive_error_fraction
    );
}

// ---------------------------------------------------------------------------
// Test 13: Candidates are sorted by estimated improvement (best first).
// ---------------------------------------------------------------------------
#[test]
fn test_candidates_sorted_by_estimated_improvement() {
    let creature = CreatureJson {
        neurons: vec![
            neuron("input-1", "input", 0.0),
            neuron("output-1", "output", 0.0),
            neuron("output-2", "output", 0.0),
        ],
        synapses: vec![
            synapse("input-1", "output-1", 0.5),
            synapse("input-1", "output-2", 0.3),
        ],
        input: 1,
        output: 2,
    };

    // output-1: moderate bias (small errors)
    let output_1_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let error = if i < 80 { 0.1 } else { -0.05 };
            make_record("output-1", i, 0.5, vec![error])
        })
        .collect();

    // output-2: strong bias (large errors)
    let output_2_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let error = if i < 90 { 0.8 } else { -0.1 };
            make_record("output-2", i, 0.5, vec![error])
        })
        .collect();

    let candidates = detect_output_bias_drift(
        &creature,
        &[
            ("output-1".to_string(), output_1_records),
            ("output-2".to_string(), output_2_records),
        ],
    );

    assert_eq!(candidates.len(), 2, "Should detect both biased outputs");
    assert!(
        candidates[0].estimated_improvement >= candidates[1].estimated_improvement,
        "First candidate should have higher estimated improvement: {} >= {}",
        candidates[0].estimated_improvement,
        candidates[1].estimated_improvement
    );
}

// ---------------------------------------------------------------------------
// Test 14: Estimated improvement is always positive.
// ---------------------------------------------------------------------------
#[test]
fn test_estimated_improvement_always_positive() {
    let creature = CreatureJson {
        neurons: vec![
            neuron("input-1", "input", 0.0),
            neuron("output-1", "output", 0.0),
            neuron("output-2", "output", 0.0),
        ],
        synapses: vec![
            synapse("input-1", "output-1", 0.5),
            synapse("input-1", "output-2", 0.3),
        ],
        input: 1,
        output: 2,
    };

    // Positive bias
    let output_1_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let error = if i < 80 { 0.3 } else { -0.1 };
            make_record("output-1", i, 0.5, vec![error])
        })
        .collect();

    // Negative bias
    let output_2_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let error = if i < 85 { -0.4 } else { 0.1 };
            make_record("output-2", i, 0.5, vec![error])
        })
        .collect();

    let candidates = detect_output_bias_drift(
        &creature,
        &[
            ("output-1".to_string(), output_1_records),
            ("output-2".to_string(), output_2_records),
        ],
    );

    for c in &candidates {
        assert!(
            c.estimated_improvement > 0.0,
            "Estimated improvement should be positive for neuron {}: got {}",
            c.neuron_uuid,
            c.estimated_improvement
        );
    }
}

// ---------------------------------------------------------------------------
// Test 15: Coordinated candidate comment includes diagnostics.
// ---------------------------------------------------------------------------
#[test]
fn test_coordinated_candidate_comment_includes_diagnostics() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", 0.0),
            neuron("output-1", "output", 0.2),
        ],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let error = if i < 80 { 0.3 } else { -0.1 };
            make_record("output-1", i, 0.5, vec![error])
        })
        .collect();

    let candidates = detect_output_bias_drift(&creature, &[("output-1".to_string(), records)]);
    let coordinated = output_bias_drift_to_coordinated_candidates(&candidates);

    assert_eq!(coordinated.len(), 1);
    let comment = coordinated[0].comment.as_ref().unwrap();

    assert!(
        comment.contains("output-1"),
        "Comment should contain neuron UUID: {comment}"
    );
    assert!(
        comment.contains("mean error"),
        "Comment should contain mean error info: {comment}"
    );
    assert!(
        comment.contains("same sign"),
        "Comment should contain sign bias info: {comment}"
    );
    assert!(
        comment.contains("bias"),
        "Comment should contain bias info: {comment}"
    );
}

// ---------------------------------------------------------------------------
// Test 16: Exactly minimum sample count (20) is accepted.
// ---------------------------------------------------------------------------
#[test]
fn test_exactly_minimum_samples_accepted() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", 0.0),
            neuron("output-1", "output", 0.0),
        ],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    // Exactly 20 samples, all positive errors
    let records: Vec<DiscoverRecord> = (0..20)
        .map(|i| make_record("output-1", i, 0.5, vec![0.5]))
        .collect();

    let candidates = detect_output_bias_drift(&creature, &[("output-1".to_string(), records)]);

    assert_eq!(candidates.len(), 1, "Exactly 20 samples should be accepted");
}

// ---------------------------------------------------------------------------
// Test 17: Nineteen samples (below minimum) is rejected.
// ---------------------------------------------------------------------------
#[test]
fn test_nineteen_samples_below_minimum_rejected() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", 0.0),
            neuron("output-1", "output", 0.0),
        ],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    // 19 samples — just below minimum
    let records: Vec<DiscoverRecord> = (0..19)
        .map(|i| make_record("output-1", i, 0.5, vec![0.5]))
        .collect();

    let candidates = detect_output_bias_drift(&creature, &[("output-1".to_string(), records)]);

    assert!(
        candidates.is_empty(),
        "19 samples should be below minimum and rejected"
    );
}

// ---------------------------------------------------------------------------
// Test 18: Exactly 70% same sign at threshold boundary.
// ---------------------------------------------------------------------------
#[test]
fn test_exactly_seventy_percent_at_threshold() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", 0.0),
            neuron("output-1", "output", 0.0),
        ],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    // 70 positive, 30 negative out of 100
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let error = if i < 70 { 0.3 } else { -0.3 };
            make_record("output-1", i, 0.5, vec![error])
        })
        .collect();

    let candidates = detect_output_bias_drift(&creature, &[("output-1".to_string(), records)]);

    // 70% = 0.7 which is >= MIN_MAJORITY_SIGN_FRACTION (0.7), should be accepted
    assert_eq!(
        candidates.len(),
        1,
        "Exactly 70% same sign should be accepted (at threshold)"
    );
}

// ---------------------------------------------------------------------------
// Test 19: 69% same sign below threshold is rejected.
// ---------------------------------------------------------------------------
#[test]
fn test_sixty_nine_percent_below_threshold_rejected() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", 0.0),
            neuron("output-1", "output", 0.0),
        ],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    // 69 positive, 31 negative out of 100
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let error = if i < 69 { 0.3 } else { -0.3 };
            make_record("output-1", i, 0.5, vec![error])
        })
        .collect();

    let candidates = detect_output_bias_drift(&creature, &[("output-1".to_string(), records)]);

    assert!(
        candidates.is_empty(),
        "69% same sign should be below threshold and rejected"
    );
}

// ---------------------------------------------------------------------------
// Test 20: Mean error near noise threshold boundary (0.01).
// ---------------------------------------------------------------------------
#[test]
fn test_mean_error_at_noise_threshold_boundary() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", 0.0),
            neuron("output-1", "output", 0.0),
        ],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    // Errors just above the noise threshold (0.01) — using 0.011 to avoid f32 rounding
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| make_record("output-1", i, 0.5, vec![0.011]))
        .collect();

    let candidates = detect_output_bias_drift(&creature, &[("output-1".to_string(), records)]);

    // Mean error = 0.011 which is > MIN_MEAN_ERROR_MAGNITUDE (0.01), should be accepted
    assert_eq!(
        candidates.len(),
        1,
        "Mean error just above noise threshold should be accepted"
    );

    // Errors below the noise threshold should be rejected
    let records_below: Vec<DiscoverRecord> = (0..100)
        .map(|i| make_record("output-1", i, 0.5, vec![0.005]))
        .collect();

    let candidates_below =
        detect_output_bias_drift(&creature, &[("output-1".to_string(), records_below)]);

    assert!(
        candidates_below.is_empty(),
        "Mean error below noise threshold should be rejected"
    );
}

// ---------------------------------------------------------------------------
// Test 21: Empty records produce no candidates.
// ---------------------------------------------------------------------------
#[test]
fn test_empty_records_no_candidates() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", 0.0),
            neuron("output-1", "output", 0.0),
        ],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    let candidates = detect_output_bias_drift(&creature, &[]);

    assert!(
        candidates.is_empty(),
        "Empty records should produce no candidates"
    );
}

// ---------------------------------------------------------------------------
// Test 22: Empty creature (no neurons) produces no candidates.
// ---------------------------------------------------------------------------
#[test]
fn test_empty_creature_no_candidates() {
    let creature = CreatureJson {
        neurons: vec![],
        synapses: vec![],
        input: 0,
        output: 0,
    };

    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| make_record("output-1", i, 0.5, vec![0.5]))
        .collect();

    let candidates = detect_output_bias_drift(&creature, &[("output-1".to_string(), records)]);

    assert!(
        candidates.is_empty(),
        "Empty creature should produce no candidates"
    );
}

// ---------------------------------------------------------------------------
// Test 23: Coordinated candidate SetBias value is current_bias + delta.
// ---------------------------------------------------------------------------
#[test]
fn test_set_bias_value_is_current_plus_delta() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", 0.0),
            neuron("output-1", "output", 0.3),
        ],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    // All positive errors of 0.2 -> mean error = 0.2, delta = -0.2
    // Expected new bias = 0.3 + (-0.2) = 0.1
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| make_record("output-1", i, 0.5, vec![0.2]))
        .collect();

    let candidates = detect_output_bias_drift(&creature, &[("output-1".to_string(), records)]);
    assert_eq!(candidates.len(), 1);

    let c = &candidates[0];
    assert!(
        (c.current_bias - 0.3).abs() < 1e-6,
        "Current bias should be 0.3"
    );
    assert!(
        (c.recommended_bias_delta - (-0.2)).abs() < 1e-6,
        "Delta should be -0.2"
    );

    let coordinated = output_bias_drift_to_coordinated_candidates(&candidates);
    assert_eq!(coordinated.len(), 1);

    let ops_json = serde_json::to_string(&coordinated[0].operations).unwrap();
    // The SetBias should set bias to 0.3 + (-0.2) = 0.1
    assert!(
        ops_json.contains("setBias"),
        "Should contain setBias operation"
    );
    // Parse to verify the bias value
    let ops: Vec<serde_json::Value> = serde_json::from_str(&ops_json).unwrap();
    assert_eq!(ops.len(), 1);
    let bias_value = ops[0]["bias"].as_f64().unwrap();
    assert!(
        (bias_value - 0.1).abs() < 1e-4,
        "SetBias should set bias to 0.1 (current 0.3 + delta -0.2): got {bias_value}"
    );
}

// ---------------------------------------------------------------------------
// Test 24: Each coordinated candidate has exactly one operation.
// ---------------------------------------------------------------------------
#[test]
fn test_each_candidate_has_one_operation() {
    let creature = CreatureJson {
        neurons: vec![
            neuron("input-1", "input", 0.0),
            neuron("output-1", "output", 0.0),
            neuron("output-2", "output", 0.0),
        ],
        synapses: vec![
            synapse("input-1", "output-1", 0.5),
            synapse("input-1", "output-2", 0.3),
        ],
        input: 1,
        output: 2,
    };

    // Both outputs biased
    let output_1_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| make_record("output-1", i, 0.5, vec![0.5]))
        .collect();
    let output_2_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| make_record("output-2", i, 0.5, vec![-0.5]))
        .collect();

    let candidates = detect_output_bias_drift(
        &creature,
        &[
            ("output-1".to_string(), output_1_records),
            ("output-2".to_string(), output_2_records),
        ],
    );

    let coordinated = output_bias_drift_to_coordinated_candidates(&candidates);

    for c in &coordinated {
        assert_eq!(
            c.operations.len(),
            1,
            "Each coordinated candidate should have exactly one SetBias operation"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 25: Empty candidates conversion produces empty results.
// ---------------------------------------------------------------------------
#[test]
fn test_empty_candidates_conversion() {
    let coordinated = output_bias_drift_to_coordinated_candidates(&[]);
    assert!(
        coordinated.is_empty(),
        "Empty candidates should produce empty coordinated results"
    );
}

// ---------------------------------------------------------------------------
// Test 26: Coordinated candidates sorted by expected score gain.
// ---------------------------------------------------------------------------
#[test]
fn test_coordinated_candidates_sorted_by_score_gain() {
    let creature = CreatureJson {
        neurons: vec![
            neuron("input-1", "input", 0.0),
            neuron("output-1", "output", 0.0),
            neuron("output-2", "output", 0.0),
        ],
        synapses: vec![
            synapse("input-1", "output-1", 0.5),
            synapse("input-1", "output-2", 0.3),
        ],
        input: 1,
        output: 2,
    };

    // output-1: small bias
    let output_1_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let error = if i < 80 { 0.1 } else { -0.05 };
            make_record("output-1", i, 0.5, vec![error])
        })
        .collect();

    // output-2: large bias
    let output_2_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let error = if i < 90 { 0.8 } else { -0.1 };
            make_record("output-2", i, 0.5, vec![error])
        })
        .collect();

    let candidates = detect_output_bias_drift(
        &creature,
        &[
            ("output-1".to_string(), output_1_records),
            ("output-2".to_string(), output_2_records),
        ],
    );

    let coordinated = output_bias_drift_to_coordinated_candidates(&candidates);

    assert_eq!(
        coordinated.len(),
        2,
        "Should have two coordinated candidates"
    );
    assert!(
        coordinated[0].expected_creature_score_gain >= coordinated[1].expected_creature_score_gain,
        "Coordinated candidates should be sorted by expected score gain (best first)"
    );
}

// ---------------------------------------------------------------------------
// Test 27: All-positive errors produce correct statistics.
// ---------------------------------------------------------------------------
#[test]
fn test_all_positive_errors_statistics() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", 0.0),
            neuron("output-1", "output", 0.0),
        ],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| make_record("output-1", i, 0.5, vec![0.4]))
        .collect();

    let candidates = detect_output_bias_drift(&creature, &[("output-1".to_string(), records)]);

    assert_eq!(candidates.len(), 1);
    let c = &candidates[0];
    assert!(
        (c.positive_error_fraction - 1.0).abs() < 1e-6,
        "All positive errors should give fraction 1.0: got {}",
        c.positive_error_fraction
    );
    assert!(
        (c.mean_error - 0.4).abs() < 1e-6,
        "Mean error should be 0.4: got {}",
        c.mean_error
    );
    assert_eq!(c.sample_count, 100);
}

// ---------------------------------------------------------------------------
// Test 28: All-negative errors produce correct statistics.
// ---------------------------------------------------------------------------
#[test]
fn test_all_negative_errors_statistics() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", 0.0),
            neuron("output-1", "output", 0.0),
        ],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| make_record("output-1", i, 0.5, vec![-0.6]))
        .collect();

    let candidates = detect_output_bias_drift(&creature, &[("output-1".to_string(), records)]);

    assert_eq!(candidates.len(), 1);
    let c = &candidates[0];
    assert!(
        (c.positive_error_fraction - 0.0).abs() < 1e-6,
        "All negative errors should give fraction 0.0: got {}",
        c.positive_error_fraction
    );
    assert!(
        (c.mean_error - (-0.6)).abs() < 1e-6,
        "Mean error should be -0.6: got {}",
        c.mean_error
    );
    assert!(
        c.recommended_bias_delta > 0.0,
        "Recommended delta should be positive for negative errors"
    );
}

// ---------------------------------------------------------------------------
// Test 29: Records with empty error vectors are handled gracefully.
// ---------------------------------------------------------------------------
#[test]
fn test_records_with_empty_errors_handled() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", 0.0),
            neuron("output-1", "output", 0.0),
        ],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    // Mix of records with errors and empty errors
    let records: Vec<DiscoverRecord> = (0..50)
        .map(|i| {
            if i < 30 {
                make_record("output-1", i, 0.5, vec![0.3])
            } else {
                make_record("output-1", i, 0.5, vec![])
            }
        })
        .collect();

    let candidates = detect_output_bias_drift(&creature, &[("output-1".to_string(), records)]);

    // Only 30 records have valid errors, which is >= 20 minimum.
    // All 30 are positive, so 100% same sign -> should be detected.
    assert_eq!(
        candidates.len(),
        1,
        "Should detect bias drift from records with valid errors"
    );
    assert_eq!(
        candidates[0].sample_count, 30,
        "Sample count should reflect only records with valid errors"
    );
}
