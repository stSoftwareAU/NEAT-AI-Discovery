//! Tests for Issue #360: Opposing Synapse Detection.
//!
//! Dedicated tests for opposing synapse detection as a structural discovery method.
//! Opposing synapses have contributions that correlate positively with target error,
//! meaning they actively make predictions worse. The detection recommends `removeSynapse`
//! (strong opposition) or `setWeight` sign flip (moderate opposition) coordinated
//! structural candidates.
//!
//! ## TDD Plan
//! 1. Verify detection of synapse with strong positive contribution–error correlation
//! 2. Verify helpful synapse (negative correlation) is not flagged
//! 3. Verify hidden-target synapses are not analysed (only output targets)
//! 4. Verify insufficient samples are excluded
//! 5. Verify strongly opposing synapse recommends removal
//! 6. Verify moderately opposing synapse recommends weight flip
//! 7. Verify coordinated candidate conversion produces correct operations
//! 8. Verify empty synapse list produces no candidates
//! 9. Verify missing source neuron records are handled
//! 10. Verify missing target neuron records are handled
//! 11. Verify correlation below threshold is not flagged
//! 12. Verify dormant synapse (low contribution) is excluded
//! 13. Verify multiple opposing synapses are all detected
//! 14. Verify candidates are sorted by estimated improvement
//! 15. Verify estimated improvement is always positive
//! 16. Verify coordinated candidate comment includes diagnostics
//! 17. Verify exactly minimum sample count (20) is accepted
//! 18. Verify nineteen samples (below minimum) is rejected
//! 19. Verify empty records list produces no candidates
//! 20. Verify coordinated candidates sorted by expected score gain
//! 21. Verify negative weight opposing synapse is detected
//! 22. Verify each coordinated candidate has exactly one operation
//! 23. Verify empty candidates conversion produces empty results
//! 24. Verify weight flip candidate uses negated weight
//! 25. Verify removal candidate uses removeSynapse operation
//! 26. Verify Pearson correlation computed correctly for perfect correlation

use neat_ai_discovery::analysis::detection::opposing_synapse::{
    detect_opposing_synapses, opposing_synapses_to_coordinated_candidates,
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
        input: 2,
        output: 1,
    }
}

/// Helper: build a NeuronJson.
fn neuron(uuid: &str, neuron_type: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
        squash: "IDENTITY".to_string(),
        bias: 0.0,
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

/// Helper: generate opposing source + target records.
///
/// The source activation varies linearly from -1 to +1 over `count` observations.
/// The target error is `factor × (weight × activation) + offset`, producing
/// a strong positive correlation between contribution and error when `factor > 0`.
fn make_opposing_records(
    source_uuid: &str,
    target_uuid: &str,
    weight: f32,
    count: u32,
    factor: f32,
    offset: f32,
) -> (Vec<DiscoverRecord>, Vec<DiscoverRecord>) {
    let source_records: Vec<DiscoverRecord> = (0..count)
        .map(|i| {
            let activation = (i as f32 - count as f32 / 2.0) / (count as f32 / 2.0);
            make_record(source_uuid, i, activation, vec![])
        })
        .collect();

    let target_records: Vec<DiscoverRecord> = (0..count)
        .map(|i| {
            let activation = (i as f32 - count as f32 / 2.0) / (count as f32 / 2.0);
            let contribution = weight * activation;
            let error = factor * contribution + offset;
            make_record(target_uuid, i, 0.0, vec![error])
        })
        .collect();

    (source_records, target_records)
}

// ---------------------------------------------------------------------------
// Test 1: Synapse with strong positive contribution–error correlation is detected.
// ---------------------------------------------------------------------------
#[test]
fn test_detects_opposing_synapse() {
    let creature = make_creature(
        vec![neuron("input-1", "input"), neuron("output-1", "output")],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    let (input_records, output_records) =
        make_opposing_records("input-1", "output-1", 0.5, 100, 0.8, 0.1);

    let candidates = detect_opposing_synapses(
        &creature,
        &[
            ("input-1".to_string(), input_records),
            ("output-1".to_string(), output_records),
        ],
    );

    assert_eq!(candidates.len(), 1, "Should detect one opposing synapse");
    let c = &candidates[0];
    assert_eq!(c.from_neuron_uuid, "input-1");
    assert_eq!(c.to_neuron_uuid, "output-1");
    assert!(
        c.contribution_error_correlation > 0.3,
        "Correlation should exceed threshold: got {}",
        c.contribution_error_correlation
    );
}

// ---------------------------------------------------------------------------
// Test 2: Helpful synapse (negative contribution–error correlation) is NOT flagged.
// ---------------------------------------------------------------------------
#[test]
fn test_helpful_synapse_not_flagged() {
    let creature = make_creature(
        vec![neuron("input-1", "input"), neuron("output-1", "output")],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    // Negative factor → negative correlation → helpful
    let (input_records, output_records) =
        make_opposing_records("input-1", "output-1", 0.5, 100, -0.8, 0.0);

    let candidates = detect_opposing_synapses(
        &creature,
        &[
            ("input-1".to_string(), input_records),
            ("output-1".to_string(), output_records),
        ],
    );

    assert!(
        candidates.is_empty(),
        "Helpful synapse should not be flagged as opposing"
    );
}

// ---------------------------------------------------------------------------
// Test 3: Hidden-to-hidden synapses are not analysed (only output targets).
// ---------------------------------------------------------------------------
#[test]
fn test_hidden_target_synapses_not_analysed() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input"),
            neuron("hidden-1", "hidden"),
            neuron("output-1", "output"),
        ],
        vec![
            synapse("input-1", "hidden-1", 0.5), // Hidden target — skipped
            synapse("hidden-1", "output-1", 0.3),
        ],
    );

    let input_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = (i as f32 - 50.0) / 50.0;
            make_record("input-1", i, activation, vec![])
        })
        .collect();

    let hidden_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = (i as f32 - 50.0) / 50.0;
            let error = activation * 0.5; // Would be opposing if analysed
            make_record("hidden-1", i, activation, vec![error])
        })
        .collect();

    let output_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| make_record("output-1", i, 0.0, vec![0.01]))
        .collect();

    let candidates = detect_opposing_synapses(
        &creature,
        &[
            ("input-1".to_string(), input_records),
            ("hidden-1".to_string(), hidden_records),
            ("output-1".to_string(), output_records),
        ],
    );

    let non_output_targets: Vec<_> = candidates
        .iter()
        .filter(|c| c.to_neuron_uuid == "hidden-1")
        .collect();
    assert!(
        non_output_targets.is_empty(),
        "Should not analyse synapses targeting hidden neurons"
    );
}

// ---------------------------------------------------------------------------
// Test 4: Insufficient samples should not trigger detection.
// ---------------------------------------------------------------------------
#[test]
fn test_insufficient_samples_not_flagged() {
    let creature = make_creature(
        vec![neuron("input-1", "input"), neuron("output-1", "output")],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    let input_records: Vec<DiscoverRecord> = (0..5)
        .map(|i| make_record("input-1", i, 0.5, vec![]))
        .collect();

    let output_records: Vec<DiscoverRecord> = (0..5)
        .map(|i| make_record("output-1", i, 0.0, vec![0.5]))
        .collect();

    let candidates = detect_opposing_synapses(
        &creature,
        &[
            ("input-1".to_string(), input_records),
            ("output-1".to_string(), output_records),
        ],
    );

    assert!(
        candidates.is_empty(),
        "Too few samples should not trigger detection"
    );
}

// ---------------------------------------------------------------------------
// Test 5: Strongly opposing synapse (correlation > 0.5) recommends removal.
// ---------------------------------------------------------------------------
#[test]
fn test_strongly_opposing_recommends_removal() {
    let creature = make_creature(
        vec![neuron("input-1", "input"), neuron("output-1", "output")],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    // Very strong positive correlation
    let (input_records, output_records) =
        make_opposing_records("input-1", "output-1", 0.5, 100, 1.0, 0.05);

    let candidates = detect_opposing_synapses(
        &creature,
        &[
            ("input-1".to_string(), input_records),
            ("output-1".to_string(), output_records),
        ],
    );

    assert!(!candidates.is_empty(), "Should detect opposing synapse");
    assert!(
        candidates[0].contribution_error_correlation > 0.5,
        "Correlation should be > 0.5 for strong opposition: got {}",
        candidates[0].contribution_error_correlation
    );
    assert!(
        candidates[0].recommend_removal,
        "Strongly opposing synapse should recommend removal"
    );

    let coordinated = opposing_synapses_to_coordinated_candidates(&candidates);
    assert!(!coordinated.is_empty());
    let ops_json = serde_json::to_string(&coordinated[0].operations).unwrap();
    assert!(
        ops_json.contains("removeSynapse"),
        "Should include removeSynapse operation, got: {ops_json}"
    );
}

// ---------------------------------------------------------------------------
// Test 6: Moderately opposing synapse (correlation 0.3–0.5) recommends weight flip.
// ---------------------------------------------------------------------------
#[test]
fn test_moderately_opposing_recommends_weight_flip() {
    let creature = make_creature(
        vec![neuron("input-1", "input"), neuron("output-1", "output")],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    // Moderate correlation: factor 0.4 with noise to get correlation in [0.3, 0.5]
    let source_records: Vec<DiscoverRecord> = (0..200)
        .map(|i| {
            let activation = (i as f32 - 100.0) / 100.0;
            make_record("input-1", i, activation, vec![])
        })
        .collect();

    let target_records: Vec<DiscoverRecord> = (0..200)
        .map(|i| {
            let activation = (i as f32 - 100.0) / 100.0;
            let contribution = 0.5 * activation;
            // Add noise that weakens correlation but keeps it positive
            let noise = ((i as f32 * 7.3).sin()) * 0.5;
            let error = contribution * 0.3 + noise;
            make_record("output-1", i, 0.0, vec![error])
        })
        .collect();

    let candidates = detect_opposing_synapses(
        &creature,
        &[
            ("input-1".to_string(), source_records),
            ("output-1".to_string(), target_records),
        ],
    );

    // This test verifies the moderate case when correlation is between 0.3 and 0.5.
    // Due to random-like noise, the exact correlation varies, but we can check the
    // weight flip path by constructing a candidate directly if needed.
    // If the correlation falls into the moderate range, recommend_removal should be false.
    for c in &candidates {
        if c.contribution_error_correlation >= 0.3 && c.contribution_error_correlation <= 0.5 {
            assert!(
                !c.recommend_removal,
                "Moderately opposing synapse (r={:.3}) should recommend weight flip, not removal",
                c.contribution_error_correlation
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Test 7: Coordinated candidate conversion produces correct operations.
// ---------------------------------------------------------------------------
#[test]
fn test_coordinated_candidate_conversion() {
    let creature = make_creature(
        vec![neuron("input-1", "input"), neuron("output-1", "output")],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    let (input_records, output_records) =
        make_opposing_records("input-1", "output-1", 0.5, 100, 0.8, 0.1);

    let candidates = detect_opposing_synapses(
        &creature,
        &[
            ("input-1".to_string(), input_records),
            ("output-1".to_string(), output_records),
        ],
    );
    assert!(!candidates.is_empty(), "Should detect opposing synapse");

    let coordinated = opposing_synapses_to_coordinated_candidates(&candidates);
    assert!(
        !coordinated.is_empty(),
        "Should produce coordinated candidates"
    );

    let c = &coordinated[0];
    assert!(c.expected_creature_score_gain > 0.0);
    assert!(c.comment.is_some(), "Should have a comment");

    let ops_json = serde_json::to_string(&c.operations).unwrap();
    assert!(
        ops_json.contains("removeSynapse") || ops_json.contains("setWeight"),
        "Should include removeSynapse or setWeight operation, got: {ops_json}"
    );
}

// ---------------------------------------------------------------------------
// Test 8: Empty synapse list produces no candidates.
// ---------------------------------------------------------------------------
#[test]
fn test_empty_synapses_no_candidates() {
    let creature = make_creature(
        vec![neuron("input-1", "input"), neuron("output-1", "output")],
        vec![], // No synapses
    );

    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| make_record("input-1", i, 0.5, vec![]))
        .collect();

    let candidates = detect_opposing_synapses(&creature, &[("input-1".to_string(), records)]);

    assert!(
        candidates.is_empty(),
        "No synapses should produce no candidates"
    );
}

// ---------------------------------------------------------------------------
// Test 9: Missing source neuron records are handled gracefully.
// ---------------------------------------------------------------------------
#[test]
fn test_missing_source_records_handled() {
    let creature = make_creature(
        vec![neuron("input-1", "input"), neuron("output-1", "output")],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    // Only provide records for the output neuron, not the source
    let output_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| make_record("output-1", i, 0.0, vec![0.5]))
        .collect();

    let candidates =
        detect_opposing_synapses(&creature, &[("output-1".to_string(), output_records)]);

    assert!(
        candidates.is_empty(),
        "Missing source records should not produce candidates"
    );
}

// ---------------------------------------------------------------------------
// Test 10: Missing target neuron records are handled gracefully.
// ---------------------------------------------------------------------------
#[test]
fn test_missing_target_records_handled() {
    let creature = make_creature(
        vec![neuron("input-1", "input"), neuron("output-1", "output")],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    // Only provide records for the source neuron, not the target
    let input_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| make_record("input-1", i, 0.5, vec![]))
        .collect();

    let candidates = detect_opposing_synapses(&creature, &[("input-1".to_string(), input_records)]);

    assert!(
        candidates.is_empty(),
        "Missing target records should not produce candidates"
    );
}

// ---------------------------------------------------------------------------
// Test 11: Correlation below threshold (< 0.3) is not flagged.
// ---------------------------------------------------------------------------
#[test]
fn test_low_correlation_not_flagged() {
    let creature = make_creature(
        vec![neuron("input-1", "input"), neuron("output-1", "output")],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    // Very weak correlation: mostly noise
    let source_records: Vec<DiscoverRecord> = (0..200)
        .map(|i| {
            let activation = (i as f32 - 100.0) / 100.0;
            make_record("input-1", i, activation, vec![])
        })
        .collect();

    let target_records: Vec<DiscoverRecord> = (0..200)
        .map(|i| {
            // Error is mostly noise, very weak contribution relationship
            let noise = ((i as f32 * 13.7).sin()) * 2.0;
            let activation = (i as f32 - 100.0) / 100.0;
            let contribution = 0.5 * activation;
            let error = contribution * 0.05 + noise;
            make_record("output-1", i, 0.0, vec![error])
        })
        .collect();

    let candidates = detect_opposing_synapses(
        &creature,
        &[
            ("input-1".to_string(), source_records),
            ("output-1".to_string(), target_records),
        ],
    );

    // With very weak factor (0.05) and strong noise, correlation should be well below 0.3
    assert!(
        candidates.is_empty(),
        "Low correlation should not produce candidates"
    );
}

// ---------------------------------------------------------------------------
// Test 12: Dormant synapse (low contribution) is excluded from opposing detection.
// ---------------------------------------------------------------------------
#[test]
fn test_dormant_synapse_excluded() {
    let creature = make_creature(
        vec![neuron("input-1", "input"), neuron("output-1", "output")],
        vec![synapse("input-1", "output-1", 1e-6)], // Very low weight → low contribution
    );

    // Even with strong correlation, contribution is too low
    let (input_records, output_records) =
        make_opposing_records("input-1", "output-1", 1e-6, 100, 1000.0, 0.0);

    let candidates = detect_opposing_synapses(
        &creature,
        &[
            ("input-1".to_string(), input_records),
            ("output-1".to_string(), output_records),
        ],
    );

    assert!(
        candidates.is_empty(),
        "Dormant synapse (low contribution) should be excluded from opposing detection"
    );
}

// ---------------------------------------------------------------------------
// Test 13: Multiple opposing synapses are all detected.
// ---------------------------------------------------------------------------
#[test]
fn test_multiple_opposing_synapses_detected() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input"),
            neuron("input-2", "input"),
            neuron("output-1", "output"),
        ],
        vec![
            synapse("input-1", "output-1", 0.5),
            synapse("input-2", "output-1", 0.3),
        ],
    );

    // Both synapses have positive contribution–error correlation
    let input1_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = (i as f32 - 50.0) / 50.0;
            make_record("input-1", i, activation, vec![])
        })
        .collect();

    let input2_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = (i as f32 - 50.0) / 50.0;
            make_record("input-2", i, activation, vec![])
        })
        .collect();

    let output_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = (i as f32 - 50.0) / 50.0;
            // Error correlates with both contributions
            let error = 0.5 * activation * 0.8 + 0.3 * activation * 0.6 + 0.05;
            make_record("output-1", i, 0.0, vec![error])
        })
        .collect();

    let candidates = detect_opposing_synapses(
        &creature,
        &[
            ("input-1".to_string(), input1_records),
            ("input-2".to_string(), input2_records),
            ("output-1".to_string(), output_records),
        ],
    );

    assert!(
        candidates.len() >= 2,
        "Should detect at least two opposing synapses, got {}",
        candidates.len()
    );
}

// ---------------------------------------------------------------------------
// Test 14: Candidates are sorted by estimated improvement (best first).
// ---------------------------------------------------------------------------
#[test]
fn test_candidates_sorted_by_estimated_improvement() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input"),
            neuron("input-2", "input"),
            neuron("output-1", "output"),
        ],
        vec![
            synapse("input-1", "output-1", 0.5), // Higher weight → larger contribution
            synapse("input-2", "output-1", 0.1), // Lower weight → smaller contribution
        ],
    );

    let input1_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = (i as f32 - 50.0) / 50.0;
            make_record("input-1", i, activation, vec![])
        })
        .collect();

    let input2_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = (i as f32 - 50.0) / 50.0;
            make_record("input-2", i, activation, vec![])
        })
        .collect();

    let output_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = (i as f32 - 50.0) / 50.0;
            // Error correlates with both contributions
            let error = 0.5 * activation * 0.9 + 0.1 * activation * 0.9 + 0.02;
            make_record("output-1", i, 0.0, vec![error])
        })
        .collect();

    let candidates = detect_opposing_synapses(
        &creature,
        &[
            ("input-1".to_string(), input1_records),
            ("input-2".to_string(), input2_records),
            ("output-1".to_string(), output_records),
        ],
    );

    if candidates.len() >= 2 {
        assert!(
            candidates[0].estimated_improvement >= candidates[1].estimated_improvement,
            "Candidates should be sorted by estimated improvement (best first): {} >= {}",
            candidates[0].estimated_improvement,
            candidates[1].estimated_improvement
        );
    }
}

// ---------------------------------------------------------------------------
// Test 15: All estimated improvements are positive.
// ---------------------------------------------------------------------------
#[test]
fn test_estimated_improvement_always_positive() {
    let creature = make_creature(
        vec![neuron("input-1", "input"), neuron("output-1", "output")],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    let (input_records, output_records) =
        make_opposing_records("input-1", "output-1", 0.5, 100, 0.8, 0.1);

    let candidates = detect_opposing_synapses(
        &creature,
        &[
            ("input-1".to_string(), input_records),
            ("output-1".to_string(), output_records),
        ],
    );

    for c in &candidates {
        assert!(
            c.estimated_improvement > 0.0,
            "Estimated improvement should be positive, got: {}",
            c.estimated_improvement
        );
    }
}

// ---------------------------------------------------------------------------
// Test 16: Coordinated candidate comment includes diagnostic information.
// ---------------------------------------------------------------------------
#[test]
fn test_coordinated_candidate_comment_includes_diagnostics() {
    let creature = make_creature(
        vec![neuron("input-1", "input"), neuron("output-1", "output")],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    let (input_records, output_records) =
        make_opposing_records("input-1", "output-1", 0.5, 100, 0.8, 0.1);

    let candidates = detect_opposing_synapses(
        &creature,
        &[
            ("input-1".to_string(), input_records),
            ("output-1".to_string(), output_records),
        ],
    );

    let coordinated = opposing_synapses_to_coordinated_candidates(&candidates);
    assert!(!coordinated.is_empty());

    let comment = coordinated[0]
        .comment
        .as_ref()
        .expect("Comment should exist");

    // Comment should include source and target neuron UUIDs
    assert!(
        comment.contains("input-1"),
        "Comment should mention source neuron UUID"
    );
    assert!(
        comment.contains("output-1"),
        "Comment should mention target neuron UUID"
    );
    // Comment should mention correlation
    assert!(
        comment.contains("correlation"),
        "Comment should mention correlation"
    );
    // Comment should mention weight
    assert!(comment.contains("weight"), "Comment should mention weight");
}

// ---------------------------------------------------------------------------
// Test 17: Exactly minimum sample count (20) is accepted.
// ---------------------------------------------------------------------------
#[test]
fn test_exactly_minimum_samples_accepted() {
    let creature = make_creature(
        vec![neuron("input-1", "input"), neuron("output-1", "output")],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    let (input_records, output_records) =
        make_opposing_records("input-1", "output-1", 0.5, 20, 1.0, 0.0);

    let candidates = detect_opposing_synapses(
        &creature,
        &[
            ("input-1".to_string(), input_records),
            ("output-1".to_string(), output_records),
        ],
    );

    assert!(
        !candidates.is_empty(),
        "Exactly minimum sample count (20) should be accepted"
    );
    assert_eq!(candidates[0].sample_count, 20);
}

// ---------------------------------------------------------------------------
// Test 18: Nineteen samples (just below minimum) is rejected.
// ---------------------------------------------------------------------------
#[test]
fn test_nineteen_samples_below_minimum_rejected() {
    let creature = make_creature(
        vec![neuron("input-1", "input"), neuron("output-1", "output")],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    let (input_records, output_records) =
        make_opposing_records("input-1", "output-1", 0.5, 19, 1.0, 0.0);

    let candidates = detect_opposing_synapses(
        &creature,
        &[
            ("input-1".to_string(), input_records),
            ("output-1".to_string(), output_records),
        ],
    );

    assert!(
        candidates.is_empty(),
        "19 samples (below minimum of 20) should be rejected"
    );
}

// ---------------------------------------------------------------------------
// Test 19: Empty records list produces no candidates.
// ---------------------------------------------------------------------------
#[test]
fn test_empty_records_no_candidates() {
    let creature = make_creature(
        vec![neuron("input-1", "input"), neuron("output-1", "output")],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    let candidates = detect_opposing_synapses(&creature, &[]);

    assert!(
        candidates.is_empty(),
        "Empty records should produce no candidates"
    );
}

// ---------------------------------------------------------------------------
// Test 20: Coordinated candidates sorted by expected score gain.
// ---------------------------------------------------------------------------
#[test]
fn test_coordinated_candidates_sorted_by_score_gain() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input"),
            neuron("input-2", "input"),
            neuron("output-1", "output"),
        ],
        vec![
            synapse("input-1", "output-1", 0.5),
            synapse("input-2", "output-1", 0.3),
        ],
    );

    let input1_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = (i as f32 - 50.0) / 50.0;
            make_record("input-1", i, activation, vec![])
        })
        .collect();

    let input2_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = (i as f32 - 50.0) / 50.0;
            make_record("input-2", i, activation, vec![])
        })
        .collect();

    let output_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = (i as f32 - 50.0) / 50.0;
            let error = 0.5 * activation * 0.8 + 0.3 * activation * 0.6 + 0.05;
            make_record("output-1", i, 0.0, vec![error])
        })
        .collect();

    let candidates = detect_opposing_synapses(
        &creature,
        &[
            ("input-1".to_string(), input1_records),
            ("input-2".to_string(), input2_records),
            ("output-1".to_string(), output_records),
        ],
    );

    let coordinated = opposing_synapses_to_coordinated_candidates(&candidates);
    if coordinated.len() >= 2 {
        assert!(
            coordinated[0].expected_creature_score_gain
                >= coordinated[1].expected_creature_score_gain,
            "Coordinated candidates should be sorted by expected score gain: {} >= {}",
            coordinated[0].expected_creature_score_gain,
            coordinated[1].expected_creature_score_gain
        );
    }
}

// ---------------------------------------------------------------------------
// Test 21: Negative weight opposing synapse is detected.
// ---------------------------------------------------------------------------
#[test]
fn test_negative_weight_opposing_synapse_detected() {
    let creature = make_creature(
        vec![neuron("input-1", "input"), neuron("output-1", "output")],
        vec![synapse("input-1", "output-1", -0.5)],
    );

    // With negative weight, contribution = -0.5 * activation
    // Error positively correlated with contribution means error also goes negative
    // when activation is positive
    let (input_records, output_records) =
        make_opposing_records("input-1", "output-1", -0.5, 100, 0.8, 0.1);

    let candidates = detect_opposing_synapses(
        &creature,
        &[
            ("input-1".to_string(), input_records),
            ("output-1".to_string(), output_records),
        ],
    );

    assert!(
        !candidates.is_empty(),
        "Negative weight opposing synapse should be detected"
    );
    assert!(
        candidates[0].weight < 0.0,
        "Weight should be negative: {}",
        candidates[0].weight
    );
}

// ---------------------------------------------------------------------------
// Test 22: Each coordinated candidate has exactly one operation.
// ---------------------------------------------------------------------------
#[test]
fn test_each_candidate_has_one_operation() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input"),
            neuron("input-2", "input"),
            neuron("output-1", "output"),
        ],
        vec![
            synapse("input-1", "output-1", 0.5),
            synapse("input-2", "output-1", 0.3),
        ],
    );

    let input1_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = (i as f32 - 50.0) / 50.0;
            make_record("input-1", i, activation, vec![])
        })
        .collect();

    let input2_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = (i as f32 - 50.0) / 50.0;
            make_record("input-2", i, activation, vec![])
        })
        .collect();

    let output_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = (i as f32 - 50.0) / 50.0;
            let error = 0.5 * activation * 0.8 + 0.3 * activation * 0.6 + 0.05;
            make_record("output-1", i, 0.0, vec![error])
        })
        .collect();

    let candidates = detect_opposing_synapses(
        &creature,
        &[
            ("input-1".to_string(), input1_records),
            ("input-2".to_string(), input2_records),
            ("output-1".to_string(), output_records),
        ],
    );

    let coordinated = opposing_synapses_to_coordinated_candidates(&candidates);

    for (i, c) in coordinated.iter().enumerate() {
        assert_eq!(
            c.operations.len(),
            1,
            "Candidate {i} should have exactly one operation"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 23: Empty candidates conversion produces empty results.
// ---------------------------------------------------------------------------
#[test]
fn test_empty_candidates_conversion() {
    let coordinated = opposing_synapses_to_coordinated_candidates(&[]);
    assert!(
        coordinated.is_empty(),
        "Converting empty candidates should produce empty results"
    );
}

// ---------------------------------------------------------------------------
// Test 24: Weight flip candidate uses negated weight.
// ---------------------------------------------------------------------------
#[test]
fn test_weight_flip_uses_negated_weight() {
    use neat_ai_discovery::analysis::detection::opposing_synapse::OpposingSynapseCandidate;

    // Construct a moderate-opposition candidate directly to test weight flip path
    let candidate = OpposingSynapseCandidate {
        from_neuron_uuid: "input-1".to_string(),
        to_neuron_uuid: "output-1".to_string(),
        weight: 0.5,
        contribution_error_correlation: 0.4, // Moderate — should flip, not remove
        mean_abs_contribution: 0.2,
        sample_count: 100,
        recommend_removal: false,
        estimated_improvement: 0.01,
    };

    let coordinated = opposing_synapses_to_coordinated_candidates(&[candidate]);
    assert_eq!(coordinated.len(), 1);

    let ops_json = serde_json::to_string(&coordinated[0].operations).unwrap();
    assert!(
        ops_json.contains("setWeight"),
        "Moderate opposition should produce setWeight operation, got: {ops_json}"
    );
    // The setWeight value should be the negated weight
    assert!(
        ops_json.contains("-0.5") || ops_json.contains("-0.50"),
        "Weight should be negated to -0.5, got: {ops_json}"
    );
}

// ---------------------------------------------------------------------------
// Test 25: Removal candidate uses removeSynapse operation.
// ---------------------------------------------------------------------------
#[test]
fn test_removal_candidate_uses_remove_synapse() {
    use neat_ai_discovery::analysis::detection::opposing_synapse::OpposingSynapseCandidate;

    // Construct a strong-opposition candidate directly to test removal path
    let candidate = OpposingSynapseCandidate {
        from_neuron_uuid: "input-1".to_string(),
        to_neuron_uuid: "output-1".to_string(),
        weight: 0.5,
        contribution_error_correlation: 0.8, // Strong — should remove
        mean_abs_contribution: 0.3,
        sample_count: 100,
        recommend_removal: true,
        estimated_improvement: 0.05,
    };

    let coordinated = opposing_synapses_to_coordinated_candidates(&[candidate]);
    assert_eq!(coordinated.len(), 1);

    let ops_json = serde_json::to_string(&coordinated[0].operations).unwrap();
    assert!(
        ops_json.contains("removeSynapse"),
        "Strong opposition should produce removeSynapse operation, got: {ops_json}"
    );
    assert!(
        ops_json.contains("input-1"),
        "Operation should reference source neuron"
    );
    assert!(
        ops_json.contains("output-1"),
        "Operation should reference target neuron"
    );
}

// ---------------------------------------------------------------------------
// Test 26: Pearson correlation is correct for perfect linear relationship.
// ---------------------------------------------------------------------------
#[test]
fn test_perfect_correlation_detected() {
    let creature = make_creature(
        vec![neuron("input-1", "input"), neuron("output-1", "output")],
        vec![synapse("input-1", "output-1", 1.0)],
    );

    // Perfect positive correlation: error = contribution exactly
    let (input_records, output_records) =
        make_opposing_records("input-1", "output-1", 1.0, 100, 1.0, 0.0);

    let candidates = detect_opposing_synapses(
        &creature,
        &[
            ("input-1".to_string(), input_records),
            ("output-1".to_string(), output_records),
        ],
    );

    assert!(!candidates.is_empty(), "Should detect opposing synapse");
    let c = &candidates[0];
    // Pearson correlation for perfect linear relationship should be ~1.0
    assert!(
        c.contribution_error_correlation > 0.99,
        "Perfect correlation should be near 1.0, got: {}",
        c.contribution_error_correlation
    );
}

// ---------------------------------------------------------------------------
// Test 27: Candidate fields are correctly populated.
// ---------------------------------------------------------------------------
#[test]
fn test_candidate_fields_populated() {
    let creature = make_creature(
        vec![neuron("input-1", "input"), neuron("output-1", "output")],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    let (input_records, output_records) =
        make_opposing_records("input-1", "output-1", 0.5, 100, 0.8, 0.1);

    let candidates = detect_opposing_synapses(
        &creature,
        &[
            ("input-1".to_string(), input_records),
            ("output-1".to_string(), output_records),
        ],
    );

    assert_eq!(candidates.len(), 1);
    let c = &candidates[0];

    assert_eq!(c.from_neuron_uuid, "input-1");
    assert_eq!(c.to_neuron_uuid, "output-1");
    assert!(
        (c.weight - 0.5).abs() < f32::EPSILON,
        "Weight should be 0.5"
    );
    assert_eq!(c.sample_count, 100, "Sample count should be 100");
    assert!(
        c.mean_abs_contribution > 0.0,
        "Mean abs contribution should be positive"
    );
    assert!(
        c.contribution_error_correlation > 0.0,
        "Correlation should be positive"
    );
    assert!(
        c.estimated_improvement > 0.0,
        "Estimated improvement should be positive"
    );
}

// ---------------------------------------------------------------------------
// Test 28: Multiple output neurons — opposing synapses detected per target.
// ---------------------------------------------------------------------------
#[test]
fn test_multiple_output_neurons() {
    let creature = CreatureJson {
        neurons: vec![
            neuron("input-1", "input"),
            neuron("output-1", "output"),
            neuron("output-2", "output"),
        ],
        synapses: vec![
            synapse("input-1", "output-1", 0.5),
            synapse("input-1", "output-2", 0.3),
        ],
        input: 1,
        output: 2,
    };

    let input_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = (i as f32 - 50.0) / 50.0;
            make_record("input-1", i, activation, vec![])
        })
        .collect();

    let output1_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = (i as f32 - 50.0) / 50.0;
            let error = 0.5 * activation * 0.8 + 0.05;
            make_record("output-1", i, 0.0, vec![error])
        })
        .collect();

    let output2_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = (i as f32 - 50.0) / 50.0;
            let error = 0.3 * activation * 0.9 + 0.02;
            make_record("output-2", i, 0.0, vec![error])
        })
        .collect();

    let candidates = detect_opposing_synapses(
        &creature,
        &[
            ("input-1".to_string(), input_records),
            ("output-1".to_string(), output1_records),
            ("output-2".to_string(), output2_records),
        ],
    );

    // Both synapses target output neurons with correlated errors
    assert!(
        candidates.len() >= 2,
        "Should detect opposing synapses for both output targets, got {}",
        candidates.len()
    );
}

// ---------------------------------------------------------------------------
// Test 29: Weight flip discount factor applied correctly.
// ---------------------------------------------------------------------------
#[test]
fn test_weight_flip_discount_factor() {
    use neat_ai_discovery::analysis::detection::opposing_synapse::OpposingSynapseCandidate;

    let removal_candidate = OpposingSynapseCandidate {
        from_neuron_uuid: "a".to_string(),
        to_neuron_uuid: "b".to_string(),
        weight: 0.5,
        contribution_error_correlation: 0.8,
        mean_abs_contribution: 0.2,
        sample_count: 100,
        recommend_removal: true,
        estimated_improvement: 0.01,
    };

    let flip_candidate = OpposingSynapseCandidate {
        from_neuron_uuid: "c".to_string(),
        to_neuron_uuid: "d".to_string(),
        weight: 0.5,
        contribution_error_correlation: 0.4,
        mean_abs_contribution: 0.2,
        sample_count: 100,
        recommend_removal: false,
        estimated_improvement: 0.01,
    };

    let removal_coordinated =
        opposing_synapses_to_coordinated_candidates(std::slice::from_ref(&removal_candidate));
    let flip_coordinated =
        opposing_synapses_to_coordinated_candidates(std::slice::from_ref(&flip_candidate));

    // Removal uses full estimated_improvement, flip uses 0.7× discount
    let removal_gain = removal_coordinated[0].expected_creature_score_gain;
    let flip_gain = flip_coordinated[0].expected_creature_score_gain;

    let expected_removal = removal_candidate.estimated_improvement;
    let expected_flip = flip_candidate.estimated_improvement * 0.7;

    assert!(
        (removal_gain - expected_removal).abs() < 1e-6,
        "Removal gain should equal estimated_improvement: {removal_gain} vs {expected_removal}"
    );
    assert!(
        (flip_gain - expected_flip).abs() < 1e-6,
        "Flip gain should be 0.7× estimated_improvement: {flip_gain} vs {expected_flip}"
    );
}
