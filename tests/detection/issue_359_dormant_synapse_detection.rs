//! Tests for Issue #359: Dormant Synapse Detection.
//!
//! Dedicated tests for dormant synapse detection as a structural discovery method.
//! Dormant synapses have near-zero weights that contribute negligible signal to their
//! target neurons. The detection recommends `removeSynapse` coordinated structural
//! candidates to reduce network complexity.
//!
//! ## TDD Plan
//! 1. Verify detection of synapse with near-zero weight
//! 2. Verify active synapses are not flagged
//! 3. Verify sole connection is protected even if dormant
//! 4. Verify insufficient samples are excluded
//! 5. Verify coordinated candidate conversion produces removeSynapse
//! 6. Verify multiple dormant synapses are all detected
//! 7. Verify `other_fan_in` count is correct
//! 8. Verify empty synapse list produces no candidates
//! 9. Verify missing source neuron records are handled
//! 10. Verify weight exactly at threshold boundary
//! 11. Verify high-activation source with zero-weight synapse
//! 12. Verify low-activation source does not make active synapse dormant
//! 13. Verify candidates are sorted by estimated improvement
//! 14. Verify estimated improvement is always positive
//! 15. Verify coordinated candidate comment includes diagnostics
//! 16. Verify multiple targets: dormant to one, active to another
//! 17. Verify exactly minimum sample count is accepted
//! 18. Verify negative near-zero weight is detected
//! 19. Verify coordinated candidates sorted by expected score gain

use neat_ai_discovery::analysis::detection::dormant_synapse::{
    detect_dormant_synapses, dormant_synapses_to_coordinated_candidates,
};
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

/// Helper: create a `DiscoverRecord` for a neuron with given activation.
fn make_record(neuron_uuid: &str, obs_index: u32, activation: f32) -> DiscoverRecord {
    DiscoverRecord {
        obs_index,
        neuron_uuid: neuron_uuid.to_string(),
        value: Some(activation * 0.8),
        activation,
        errors: vec![0.01],
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

/// Helper: build a `NeuronJson`.
fn neuron(uuid: &str, neuron_type: &str, squash: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
        squash: squash.to_string(),
        bias: 0.0,
    }
}

/// Helper: build a `SynapseJson`.
fn synapse(from: &str, to: &str, weight: f32) -> SynapseJson {
    SynapseJson {
        from_uuid: from.to_string(),
        to_uuid: to.to_string(),
        weight,
        synapse_type: None,
    }
}

/// Helper: generate N records for a neuron with constant activation.
fn make_records(neuron_uuid: &str, count: u32, activation: f32) -> Vec<DiscoverRecord> {
    (0..count)
        .map(|i| make_record(neuron_uuid, i, activation))
        .collect()
}

// ---------------------------------------------------------------------------
// Test 1: Synapse with near-zero weight is detected as dormant.
// ---------------------------------------------------------------------------
#[test]
fn test_dormant_synapse_detects_near_zero_weight_synapse() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("input-2", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "output-1", 1e-6), // Dormant
            synapse("input-2", "output-1", 0.5),  // Active
        ],
    );

    let records_1 = make_records("input-1", 100, 0.5);
    let records_2 = make_records("input-2", 100, 0.5);

    let candidates = detect_dormant_synapses(
        &creature,
        &[
            ("input-1".to_string(), records_1),
            ("input-2".to_string(), records_2),
        ],
    );

    assert_eq!(candidates.len(), 1, "Should detect one dormant synapse");
    let c = &candidates[0];
    assert_eq!(c.from_neuron_uuid, "input-1");
    assert_eq!(c.to_neuron_uuid, "output-1");
    assert!(c.weight.abs() < 1e-4, "Weight should be near zero");
}

// ---------------------------------------------------------------------------
// Test 2: Active synapse with meaningful weight is NOT flagged.
// ---------------------------------------------------------------------------
#[test]
fn test_dormant_synapse_active_synapse_not_flagged() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("input-2", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "output-1", 0.5),
            synapse("input-2", "output-1", 0.3),
        ],
    );

    let records_1 = make_records("input-1", 100, 0.5);
    let records_2 = make_records("input-2", 100, 0.5);

    let candidates = detect_dormant_synapses(
        &creature,
        &[
            ("input-1".to_string(), records_1),
            ("input-2".to_string(), records_2),
        ],
    );

    assert!(
        candidates.is_empty(),
        "Active synapses should not be flagged as dormant"
    );
}

// ---------------------------------------------------------------------------
// Test 3: Sole connection to target is NOT flagged even if dormant.
// ---------------------------------------------------------------------------
#[test]
fn test_dormant_synapse_sole_connection_not_flagged() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "output-1", 1e-6), // Dormant but sole connection
        ],
    );

    let records = make_records("input-1", 100, 0.5);

    let candidates = detect_dormant_synapses(&creature, &[("input-1".to_string(), records)]);

    assert!(
        candidates.is_empty(),
        "Sole connection should not be flagged even if dormant"
    );
}

// ---------------------------------------------------------------------------
// Test 4: Insufficient samples should not trigger detection.
// ---------------------------------------------------------------------------
#[test]
fn test_dormant_synapse_insufficient_samples_not_flagged() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("input-2", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "output-1", 1e-6),
            synapse("input-2", "output-1", 0.5),
        ],
    );

    // Only 5 samples — below the minimum of 20
    let records = make_records("input-1", 5, 0.5);

    let candidates = detect_dormant_synapses(&creature, &[("input-1".to_string(), records)]);

    assert!(
        candidates.is_empty(),
        "Too few samples should not trigger detection"
    );
}

// ---------------------------------------------------------------------------
// Test 5: Dormant synapse candidates produce correct coordinated removal operations.
// ---------------------------------------------------------------------------
#[test]
fn test_dormant_synapse_candidates_produce_coordinated_removal_operations() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("input-2", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "output-1", 1e-6),
            synapse("input-2", "output-1", 0.5),
        ],
    );

    let records_1 = make_records("input-1", 100, 0.5);
    let records_2 = make_records("input-2", 100, 0.5);

    let candidates = detect_dormant_synapses(
        &creature,
        &[
            ("input-1".to_string(), records_1),
            ("input-2".to_string(), records_2),
        ],
    );
    assert!(!candidates.is_empty(), "Should detect dormant synapse");

    let coordinated = dormant_synapses_to_coordinated_candidates(&candidates);

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

    // Check that operations include a removeSynapse
    let ops_json = serde_json::to_string(&c.operations).unwrap();
    assert!(
        ops_json.contains("removeSynapse"),
        "Should include removeSynapse operation, got: {ops_json}"
    );
}

// ---------------------------------------------------------------------------
// Test 6: Multiple dormant synapses are all detected.
// ---------------------------------------------------------------------------
#[test]
fn test_dormant_synapse_multiple_dormant_synapses_detected() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("input-2", "input", "IDENTITY"),
            neuron("input-3", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "output-1", 1e-6), // Dormant
            synapse("input-2", "output-1", 1e-5), // Dormant
            synapse("input-3", "output-1", 0.5),  // Active
        ],
    );

    let records_1 = make_records("input-1", 100, 0.5);
    let records_2 = make_records("input-2", 100, 0.5);
    let records_3 = make_records("input-3", 100, 0.5);

    let candidates = detect_dormant_synapses(
        &creature,
        &[
            ("input-1".to_string(), records_1),
            ("input-2".to_string(), records_2),
            ("input-3".to_string(), records_3),
        ],
    );

    assert_eq!(candidates.len(), 2, "Should detect two dormant synapses");
}

// ---------------------------------------------------------------------------
// Test 7: Other fan-in count is correctly recorded.
// ---------------------------------------------------------------------------
#[test]
fn test_dormant_synapse_other_fan_in_count_correct() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("input-2", "input", "IDENTITY"),
            neuron("input-3", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "output-1", 1e-6), // Dormant
            synapse("input-2", "output-1", 0.5),
            synapse("input-3", "output-1", 0.3),
        ],
    );

    let records_1 = make_records("input-1", 100, 0.5);

    let candidates = detect_dormant_synapses(&creature, &[("input-1".to_string(), records_1)]);

    assert_eq!(candidates.len(), 1);
    assert_eq!(
        candidates[0].other_fan_in, 2,
        "Other fan-in should be 2 (input-2 and input-3)"
    );
}

// ---------------------------------------------------------------------------
// Test 8: Empty synapse list produces no candidates.
// ---------------------------------------------------------------------------
#[test]
fn test_dormant_synapse_empty_synapses_no_candidates() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![], // No synapses
    );

    let records = make_records("input-1", 100, 0.5);

    let candidates = detect_dormant_synapses(&creature, &[("input-1".to_string(), records)]);

    assert!(
        candidates.is_empty(),
        "No synapses should produce no candidates"
    );
}

// ---------------------------------------------------------------------------
// Test 9: Missing source neuron records are handled gracefully.
// ---------------------------------------------------------------------------
#[test]
fn test_dormant_synapse_missing_source_records_handled() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("input-2", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "output-1", 1e-6), // Dormant but no records
            synapse("input-2", "output-1", 0.5),
        ],
    );

    // Only provide records for input-2, not input-1
    let records_2 = make_records("input-2", 100, 0.5);

    let candidates = detect_dormant_synapses(&creature, &[("input-2".to_string(), records_2)]);

    assert!(
        candidates.is_empty(),
        "Missing source records should not produce candidates"
    );
}

// ---------------------------------------------------------------------------
// Test 10: Weight just above threshold boundary is NOT dormant.
// ---------------------------------------------------------------------------
#[test]
fn test_weight_above_threshold_not_dormant() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("input-2", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "output-1", 2e-4), // Just above threshold
            synapse("input-2", "output-1", 0.5),
        ],
    );

    let records_1 = make_records("input-1", 100, 0.5);
    let records_2 = make_records("input-2", 100, 0.5);

    let candidates = detect_dormant_synapses(
        &creature,
        &[
            ("input-1".to_string(), records_1),
            ("input-2".to_string(), records_2),
        ],
    );

    assert!(
        candidates.is_empty(),
        "Weight above threshold should not be flagged as dormant"
    );
}

// ---------------------------------------------------------------------------
// Test 10b: Weight exactly at threshold passes weight check but contribution
// determines outcome.
// ---------------------------------------------------------------------------
#[test]
fn test_weight_at_exact_threshold_passes_weight_check() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("input-2", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "output-1", 1e-4), // Exactly at weight threshold
            synapse("input-2", "output-1", 0.5),
        ],
    );

    // With activation 0.5, contribution = 1e-4 * 0.5 = 5e-5 < 1e-4 threshold
    // Weight check uses strict >, so 1e-4 is NOT > 1e-4 → passes through
    // Contribution 5e-5 < 1e-4 → flagged as dormant
    let records_1 = make_records("input-1", 100, 0.5);
    let records_2 = make_records("input-2", 100, 0.5);

    let candidates = detect_dormant_synapses(
        &creature,
        &[
            ("input-1".to_string(), records_1),
            ("input-2".to_string(), records_2),
        ],
    );

    // Weight exactly at threshold is NOT strictly greater, so the weight check
    // does not skip it. The contribution check then determines the outcome.
    assert_eq!(
        candidates.len(),
        1,
        "Weight exactly at threshold passes through weight check; contribution determines outcome"
    );
}

// ---------------------------------------------------------------------------
// Test 11: High-activation source with zero-weight synapse is dormant.
// ---------------------------------------------------------------------------
#[test]
fn test_zero_weight_high_activation_is_dormant() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("input-2", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "output-1", 0.0), // Zero weight — dormant
            synapse("input-2", "output-1", 0.5),
        ],
    );

    // High activation values, but weight is zero so contribution is zero
    let records_1 = make_records("input-1", 100, 10.0);
    let records_2 = make_records("input-2", 100, 0.5);

    let candidates = detect_dormant_synapses(
        &creature,
        &[
            ("input-1".to_string(), records_1),
            ("input-2".to_string(), records_2),
        ],
    );

    assert_eq!(
        candidates.len(),
        1,
        "Zero-weight synapse should be detected regardless of source activation"
    );
    assert!(
        candidates[0].mean_abs_contribution.abs() < f32::EPSILON,
        "Zero weight should yield zero contribution"
    );
}

// ---------------------------------------------------------------------------
// Test 12: Low activation does NOT make an active-weight synapse dormant.
// ---------------------------------------------------------------------------
#[test]
fn test_low_activation_does_not_make_active_synapse_dormant() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("input-2", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "output-1", 0.5), // Active weight
            synapse("input-2", "output-1", 0.3),
        ],
    );

    // Even with very low activation, the weight is above threshold
    let records_1 = make_records("input-1", 100, 1e-8);
    let records_2 = make_records("input-2", 100, 0.5);

    let candidates = detect_dormant_synapses(
        &creature,
        &[
            ("input-1".to_string(), records_1),
            ("input-2".to_string(), records_2),
        ],
    );

    assert!(
        candidates.is_empty(),
        "Active-weight synapse should not be flagged even with low source activation"
    );
}

// ---------------------------------------------------------------------------
// Test 13: Candidates are sorted by estimated improvement (best first).
// ---------------------------------------------------------------------------
#[test]
fn test_dormant_synapse_candidates_sorted_by_estimated_improvement() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("input-2", "input", "IDENTITY"),
            neuron("input-3", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "output-1", 1e-5), // Dormant, slightly higher contribution
            synapse("input-2", "output-1", 1e-8), // Dormant, lower contribution (better)
            synapse("input-3", "output-1", 0.5),  // Active
        ],
    );

    let records_1 = make_records("input-1", 100, 0.5);
    let records_2 = make_records("input-2", 100, 0.5);
    let records_3 = make_records("input-3", 100, 0.5);

    let candidates = detect_dormant_synapses(
        &creature,
        &[
            ("input-1".to_string(), records_1),
            ("input-2".to_string(), records_2),
            ("input-3".to_string(), records_3),
        ],
    );

    assert_eq!(candidates.len(), 2, "Should detect two dormant synapses");

    // Best improvement should come first
    assert!(
        candidates[0].estimated_improvement >= candidates[1].estimated_improvement,
        "Candidates should be sorted by estimated improvement (best first): {} >= {}",
        candidates[0].estimated_improvement,
        candidates[1].estimated_improvement
    );
}

// ---------------------------------------------------------------------------
// Test 14: All estimated improvements are positive.
// ---------------------------------------------------------------------------
#[test]
fn test_dormant_synapse_estimated_improvement_always_positive() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("input-2", "input", "IDENTITY"),
            neuron("input-3", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "output-1", 1e-6),
            synapse("input-2", "output-1", 1e-7),
            synapse("input-3", "output-1", 0.5),
        ],
    );

    let records_1 = make_records("input-1", 100, 0.5);
    let records_2 = make_records("input-2", 100, 0.5);
    let records_3 = make_records("input-3", 100, 0.5);

    let candidates = detect_dormant_synapses(
        &creature,
        &[
            ("input-1".to_string(), records_1),
            ("input-2".to_string(), records_2),
            ("input-3".to_string(), records_3),
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
// Test 15: Coordinated candidate comment includes diagnostic information.
// ---------------------------------------------------------------------------
#[test]
fn test_dormant_synapse_coordinated_candidate_comment_includes_diagnostics() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("input-2", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "output-1", 1e-6),
            synapse("input-2", "output-1", 0.5),
        ],
    );

    let records_1 = make_records("input-1", 100, 0.5);
    let records_2 = make_records("input-2", 100, 0.5);

    let candidates = detect_dormant_synapses(
        &creature,
        &[
            ("input-1".to_string(), records_1),
            ("input-2".to_string(), records_2),
        ],
    );

    let coordinated = dormant_synapses_to_coordinated_candidates(&candidates);
    assert_eq!(coordinated.len(), 1);

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
    // Comment should mention the weight
    assert!(
        comment.contains("weight"),
        "Comment should mention the weight"
    );
    // Comment should mention sample count
    assert!(
        comment.contains("samples"),
        "Comment should mention sample count"
    );
}

// ---------------------------------------------------------------------------
// Test 16: Synapse dormant to one target but active to another.
// ---------------------------------------------------------------------------
#[test]
fn test_dormant_to_one_target_active_to_another() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("input-2", "input", "IDENTITY"),
            neuron("hidden-1", "hidden", "TANH"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "output-1", 1e-6), // Dormant to output-1
            synapse("input-1", "hidden-1", 0.8),  // Active to hidden-1
            synapse("input-2", "output-1", 0.5),  // Active to output-1
            synapse("input-2", "hidden-1", 0.3),  // Active to hidden-1
        ],
    );

    let records_1 = make_records("input-1", 100, 0.5);
    let records_2 = make_records("input-2", 100, 0.5);

    let candidates = detect_dormant_synapses(
        &creature,
        &[
            ("input-1".to_string(), records_1),
            ("input-2".to_string(), records_2),
        ],
    );

    assert_eq!(
        candidates.len(),
        1,
        "Should detect only the dormant synapse (input-1 → output-1)"
    );
    assert_eq!(candidates[0].from_neuron_uuid, "input-1");
    assert_eq!(candidates[0].to_neuron_uuid, "output-1");
}

// ---------------------------------------------------------------------------
// Test 17: Exactly minimum sample count (20) is accepted.
// ---------------------------------------------------------------------------
#[test]
fn test_dormant_synapse_exactly_minimum_samples_accepted() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("input-2", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "output-1", 1e-6),
            synapse("input-2", "output-1", 0.5),
        ],
    );

    // Exactly 20 samples — the minimum
    let records_1 = make_records("input-1", 20, 0.5);
    let records_2 = make_records("input-2", 20, 0.5);

    let candidates = detect_dormant_synapses(
        &creature,
        &[
            ("input-1".to_string(), records_1),
            ("input-2".to_string(), records_2),
        ],
    );

    assert_eq!(
        candidates.len(),
        1,
        "Exactly minimum sample count should be accepted"
    );
    assert_eq!(candidates[0].sample_count, 20);
}

// ---------------------------------------------------------------------------
// Test 18: Negative near-zero weight is also detected as dormant.
// ---------------------------------------------------------------------------
#[test]
fn test_negative_near_zero_weight_detected() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("input-2", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "output-1", -1e-6), // Negative near-zero
            synapse("input-2", "output-1", 0.5),
        ],
    );

    let records_1 = make_records("input-1", 100, 0.5);
    let records_2 = make_records("input-2", 100, 0.5);

    let candidates = detect_dormant_synapses(
        &creature,
        &[
            ("input-1".to_string(), records_1),
            ("input-2".to_string(), records_2),
        ],
    );

    assert_eq!(
        candidates.len(),
        1,
        "Negative near-zero weight should be detected"
    );
    assert!(
        candidates[0].weight < 0.0,
        "Weight should be negative: {}",
        candidates[0].weight
    );
}

// ---------------------------------------------------------------------------
// Test 19: Coordinated candidates are sorted by expected score gain.
// ---------------------------------------------------------------------------
#[test]
fn test_dormant_synapse_coordinated_candidates_sorted_by_score_gain() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("input-2", "input", "IDENTITY"),
            neuron("input-3", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "output-1", 1e-5), // Dormant
            synapse("input-2", "output-1", 1e-8), // Dormant (lower contribution, better)
            synapse("input-3", "output-1", 0.5),  // Active
        ],
    );

    let records_1 = make_records("input-1", 100, 0.5);
    let records_2 = make_records("input-2", 100, 0.5);
    let records_3 = make_records("input-3", 100, 0.5);

    let candidates = detect_dormant_synapses(
        &creature,
        &[
            ("input-1".to_string(), records_1),
            ("input-2".to_string(), records_2),
            ("input-3".to_string(), records_3),
        ],
    );

    let coordinated = dormant_synapses_to_coordinated_candidates(&candidates);
    assert_eq!(coordinated.len(), 2);

    assert!(
        coordinated[0].expected_creature_score_gain >= coordinated[1].expected_creature_score_gain,
        "Coordinated candidates should be sorted by expected score gain: {} >= {}",
        coordinated[0].expected_creature_score_gain,
        coordinated[1].expected_creature_score_gain
    );
}

// ---------------------------------------------------------------------------
// Test 20: Nineteen samples (just below minimum) is rejected.
// ---------------------------------------------------------------------------
#[test]
fn test_dormant_synapse_nineteen_samples_below_minimum_rejected() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("input-2", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "output-1", 1e-6),
            synapse("input-2", "output-1", 0.5),
        ],
    );

    // 19 samples — one below the minimum of 20
    let records_1 = make_records("input-1", 19, 0.5);
    let records_2 = make_records("input-2", 19, 0.5);

    let candidates = detect_dormant_synapses(
        &creature,
        &[
            ("input-1".to_string(), records_1),
            ("input-2".to_string(), records_2),
        ],
    );

    assert!(
        candidates.is_empty(),
        "19 samples (below minimum of 20) should be rejected"
    );
}

// ---------------------------------------------------------------------------
// Test 21: Empty records list produces no candidates.
// ---------------------------------------------------------------------------
#[test]
fn test_dormant_synapse_empty_records_no_candidates() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("input-2", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "output-1", 1e-6),
            synapse("input-2", "output-1", 0.5),
        ],
    );

    let candidates = detect_dormant_synapses(&creature, &[]);

    assert!(
        candidates.is_empty(),
        "Empty records should produce no candidates"
    );
}

// ---------------------------------------------------------------------------
// Test 22: Mean absolute contribution is correctly computed.
// ---------------------------------------------------------------------------
#[test]
fn test_mean_abs_contribution_correct() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("input-2", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "output-1", 1e-5),
            synapse("input-2", "output-1", 0.5),
        ],
    );

    // Activation of 0.5, weight of 1e-5 → contribution = 0.5e-5 = 5e-6
    let records_1 = make_records("input-1", 100, 0.5);
    let records_2 = make_records("input-2", 100, 0.5);

    let candidates = detect_dormant_synapses(
        &creature,
        &[
            ("input-1".to_string(), records_1),
            ("input-2".to_string(), records_2),
        ],
    );

    assert_eq!(candidates.len(), 1);
    let expected_contribution = (1e-5_f32 * 0.5).abs();
    let tolerance = 1e-8;
    assert!(
        (candidates[0].mean_abs_contribution - expected_contribution).abs() < tolerance,
        "Mean abs contribution should be {expected_contribution:.2e}, got {:.2e}",
        candidates[0].mean_abs_contribution
    );
}

// ---------------------------------------------------------------------------
// Test 23: Varying activations produce correct mean contribution.
// ---------------------------------------------------------------------------
#[test]
fn test_varying_activations_correct_mean_contribution() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("input-2", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "output-1", 1e-5),
            synapse("input-2", "output-1", 0.5),
        ],
    );

    // Alternating activations: 0.0 and 1.0
    let records_1: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = if i % 2 == 0 { 0.0 } else { 1.0 };
            make_record("input-1", i, activation)
        })
        .collect();
    let records_2 = make_records("input-2", 100, 0.5);

    let candidates = detect_dormant_synapses(
        &creature,
        &[
            ("input-1".to_string(), records_1),
            ("input-2".to_string(), records_2),
        ],
    );

    assert_eq!(candidates.len(), 1);
    // Mean contribution: (50 * 0.0 + 50 * 1e-5) / 100 = 5e-6
    let expected = 5e-6_f32;
    let tolerance = 1e-8;
    assert!(
        (candidates[0].mean_abs_contribution - expected).abs() < tolerance,
        "Mean contribution with varying activations should be {expected:.2e}, got {:.2e}",
        candidates[0].mean_abs_contribution
    );
}

// ---------------------------------------------------------------------------
// Test 24: Conversion of empty candidates produces empty results.
// ---------------------------------------------------------------------------
#[test]
fn test_dormant_synapse_empty_candidates_conversion() {
    let coordinated = dormant_synapses_to_coordinated_candidates(&[]);
    assert!(
        coordinated.is_empty(),
        "Converting empty candidates should produce empty results"
    );
}

// ---------------------------------------------------------------------------
// Test 25: Each coordinated candidate has exactly one RemoveSynapse operation.
// ---------------------------------------------------------------------------
#[test]
fn test_each_candidate_has_one_remove_synapse_op() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("input-2", "input", "IDENTITY"),
            neuron("input-3", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "output-1", 1e-6),
            synapse("input-2", "output-1", 1e-7),
            synapse("input-3", "output-1", 0.5),
        ],
    );

    let records_1 = make_records("input-1", 100, 0.5);
    let records_2 = make_records("input-2", 100, 0.5);
    let records_3 = make_records("input-3", 100, 0.5);

    let candidates = detect_dormant_synapses(
        &creature,
        &[
            ("input-1".to_string(), records_1),
            ("input-2".to_string(), records_2),
            ("input-3".to_string(), records_3),
        ],
    );

    let coordinated = dormant_synapses_to_coordinated_candidates(&candidates);

    for (i, c) in coordinated.iter().enumerate() {
        assert_eq!(
            c.operations.len(),
            1,
            "Candidate {i} should have exactly one operation"
        );
        let ops_json = serde_json::to_string(&c.operations).unwrap();
        assert!(
            ops_json.contains("removeSynapse"),
            "Candidate {i} operation should be removeSynapse, got: {ops_json}"
        );
    }
}
