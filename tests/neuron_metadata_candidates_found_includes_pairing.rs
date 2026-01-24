//! Test that `candidates_found` correctly includes paired variants and maintains
//! the invariant `candidates_found >= candidates_returned`.
//!
//! Background: The `pair_extreme_candidates_with_conservative_variants` function
//! can ADD conservative and gentle nudge variants for extreme candidates. For the
//! metric pair to make semantic sense, `candidates_found` must be captured AFTER
//! pairing (so it includes generated variants) but BEFORE truncation.
//!
//! This ensures "found" always implies "at least as many as returned."

use neat_ai_discovery::{
    analysis::utils::pair_extreme_candidates_with_conservative_variants, CandidateNeuronJson,
};

/// Test that `candidates_found` includes paired variants.
///
/// With extreme candidates, the pairing function adds variants. The `candidates_found`
/// metric must capture the count AFTER pairing to maintain the invariant
/// `candidates_found >= candidates_returned`.
#[test]
fn candidates_found_includes_paired_variants() {
    // Create an extreme candidate (incoming_weight > 2.0 or bias > 1.0)
    let extreme_candidate = CandidateNeuronJson {
        source_neuron_uuid: "source-1".to_string(),
        target_neuron_uuid: "target-1".to_string(),
        source_neuron_index: None,
        target_neuron_index: None,
        incoming_weight: 200.0, // EXTREME - way above the 2.0 threshold
        outgoing_weight: 0.1,
        squash: "TANH".to_string(),
        bias: 50.0, // EXTREME - way above the 1.0 threshold
        comment: None,
        target_neuron_impact: 1.0,
        expected_creature_error_reduction: 0.2,
        expected_creature_score_gain: 0.2,
        improved_count: 10,
        total_count: 20,
        target_neuron_stats: None,
        prediction_confidence: 0.8,
        expected_score_gain_confidence_interval: [0.1, 0.3],
    };

    // Simulate CORRECT neuron analysis behaviour:
    // 1. Apply pairing (adds variants)
    // 2. Capture candidates_found AFTER pairing
    // 3. Apply truncation
    // 4. Capture candidates_returned AFTER truncation

    // Step 1: Apply pairing with NO limit (to get all variants)
    let paired = pair_extreme_candidates_with_conservative_variants(
        vec![extreme_candidate],
        None, // No limit - capture all variants
    );

    // Step 2: candidates_found = total after pairing
    let candidates_found = paired.len();

    // Step 3: Truncate (e.g., to max_candidates=2)
    let mut returned = paired;
    returned.truncate(2);

    // Step 4: candidates_returned = count after truncation
    let candidates_returned = returned.len();

    // THE KEY INVARIANT: candidates_found >= candidates_returned
    assert!(
        candidates_found >= candidates_returned,
        "Invariant violated: candidates_found ({candidates_found}) < candidates_returned ({candidates_returned})"
    );

    // Verify actual values
    assert_eq!(
        candidates_found, 3,
        "Expected 3 candidates found (original + conservative + gentle nudge)"
    );
    assert_eq!(
        candidates_returned, 2,
        "Expected 2 candidates returned after truncation"
    );

    // Verify the types of candidates returned (first two of three)
    assert!(
        returned[0]
            .comment
            .as_deref()
            .unwrap_or_default()
            .contains("paired"),
        "Original should be marked as paired"
    );
    assert!(
        returned[1]
            .comment
            .as_deref()
            .unwrap_or_default()
            .contains("Conservative"),
        "Second should be Conservative variant"
    );
}

/// Test that non-extreme candidates maintain the invariant (no variants added).
#[test]
fn non_extreme_candidates_maintain_count_invariant() {
    // Create a non-extreme candidate (incoming_weight <= 2.0 AND bias <= 1.0)
    let normal_candidate = CandidateNeuronJson {
        source_neuron_uuid: "source-1".to_string(),
        target_neuron_uuid: "target-1".to_string(),
        source_neuron_index: None,
        target_neuron_index: None,
        incoming_weight: 1.5, // Normal - below 2.0 threshold
        outgoing_weight: 0.05,
        squash: "TANH".to_string(),
        bias: 0.5, // Normal - below 1.0 threshold
        comment: None,
        target_neuron_impact: 1.0,
        expected_creature_error_reduction: 0.1,
        expected_creature_score_gain: 0.1,
        improved_count: 5,
        total_count: 20,
        target_neuron_stats: None,
        prediction_confidence: 0.8,
        expected_score_gain_confidence_interval: [0.05, 0.15],
    };

    // Apply pairing with no limit
    let paired = pair_extreme_candidates_with_conservative_variants(
        vec![normal_candidate],
        None, // No limit
    );

    let candidates_found = paired.len();
    let candidates_returned = paired.len(); // No truncation

    // For non-extreme candidates, no pairing occurs
    assert_eq!(
        candidates_found, 1,
        "Expected only 1 candidate found (no pairing for non-extreme)"
    );
    assert!(
        candidates_found >= candidates_returned,
        "Invariant: candidates_found >= candidates_returned"
    );

    // The candidate should not have been modified
    assert!(
        paired[0].comment.is_none(),
        "Non-extreme candidate should not be tagged"
    );
}

/// Test with multiple extreme candidates and a low max_candidates limit.
/// This verifies the correct handling when truncation actually occurs.
#[test]
fn truncation_respects_invariant_with_multiple_extreme_candidates() {
    // Create 3 extreme candidates
    let extreme_candidates: Vec<CandidateNeuronJson> = (0..3)
        .map(|i| CandidateNeuronJson {
            source_neuron_uuid: format!("source-{i}"),
            target_neuron_uuid: "target-1".to_string(),
            source_neuron_index: None,
            target_neuron_index: None,
            incoming_weight: 200.0, // EXTREME
            outgoing_weight: 0.1,
            squash: "TANH".to_string(),
            bias: 50.0, // EXTREME
            comment: None,
            target_neuron_impact: 1.0,
            expected_creature_error_reduction: 0.2 - (i as f32 * 0.01), // Decreasing score
            expected_creature_score_gain: 0.2 - (i as f32 * 0.01),
            improved_count: 10,
            total_count: 20,
            target_neuron_stats: None,
            prediction_confidence: 0.8,
            expected_score_gain_confidence_interval: [0.1, 0.3],
        })
        .collect();

    // CORRECT approach: pair first (no limit), then count, then truncate
    let paired = pair_extreme_candidates_with_conservative_variants(extreme_candidates, None);

    // candidates_found = total after pairing (3 originals × 3 variants each = 9)
    let candidates_found = paired.len();

    // Truncate to max_candidates=5
    let mut returned = paired;
    returned.truncate(5);
    let candidates_returned = returned.len();

    // THE KEY INVARIANT
    assert!(
        candidates_found >= candidates_returned,
        "Invariant violated: candidates_found ({candidates_found}) < candidates_returned ({candidates_returned})"
    );

    // Verify actual values
    assert_eq!(
        candidates_found, 9,
        "Expected 9 candidates found (3 extreme × 3 variants each)"
    );
    assert_eq!(
        candidates_returned, 5,
        "Expected 5 candidates returned after truncation"
    );
}
