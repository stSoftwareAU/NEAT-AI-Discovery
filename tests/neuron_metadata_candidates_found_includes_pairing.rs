//! Test demonstrating that neuron analysis `candidates_found` can be less than
//! `candidates_returned` when extreme candidates trigger pairing with conservative
//! and gentle nudge variants.
//!
//! Issue: The test in analyze_all_deadline_prioritises_synapses.rs:326-333 asserts
//! `candidates_found >= candidates_returned`, but this invariant doesn't hold for
//! neuron analysis because `pair_extreme_candidates_with_conservative_variants` can
//! ADD variants, making `candidates_returned` larger than `candidates_found`.
//!
//! This test verifies the correct behaviour: that pairing CAN increase the count.

use neat_ai_discovery::{
    analysis::utils::pair_extreme_candidates_with_conservative_variants, CandidateNeuronJson,
};

/// Test that demonstrates candidates_returned can exceed candidates_found
/// when extreme candidates are paired with safety variants.
#[test]
fn candidates_returned_can_exceed_candidates_found_due_to_pairing() {
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
    };

    // Simulate what happens during neuron analysis:
    // candidates_found is captured BEFORE pairing
    let candidates_found = 1; // Just one original candidate

    // After pairing, we can get up to 3 candidates (original + conservative + gentle nudge)
    let paired = pair_extreme_candidates_with_conservative_variants(
        vec![extreme_candidate],
        Some(10), // High limit to allow all variants
    );

    let candidates_returned = paired.len();

    // THIS IS THE KEY ASSERTION:
    // candidates_returned (3) > candidates_found (1)
    // The old test assertion `candidates_found >= candidates_returned` would FAIL here!
    assert_eq!(
        candidates_returned, 3,
        "Expected 3 candidates (original + conservative + gentle nudge)"
    );
    assert!(
        candidates_returned > candidates_found,
        "candidates_returned ({candidates_returned}) should be > candidates_found ({candidates_found}) when pairing adds variants"
    );

    // Verify the types of candidates returned
    assert!(
        paired[0]
            .comment
            .as_deref()
            .unwrap_or_default()
            .contains("paired"),
        "Original should be marked as paired"
    );
    assert!(
        paired[1]
            .comment
            .as_deref()
            .unwrap_or_default()
            .contains("Conservative"),
        "Second should be Conservative variant"
    );
    assert!(
        paired[2]
            .comment
            .as_deref()
            .unwrap_or_default()
            .contains("Gentle Nudge"),
        "Third should be Gentle Nudge variant"
    );
}

/// Test that non-extreme candidates don't trigger pairing
/// (in this case the invariant DOES hold)
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
    };

    let candidates_found = 1;
    let paired = pair_extreme_candidates_with_conservative_variants(
        vec![normal_candidate],
        Some(10), // High limit
    );
    let candidates_returned = paired.len();

    // For non-extreme candidates, no pairing occurs
    assert_eq!(
        candidates_returned, 1,
        "Expected only 1 candidate (no pairing)"
    );
    assert_eq!(
        candidates_returned, candidates_found,
        "For non-extreme candidates, counts should match"
    );

    // The candidate should not have been modified
    assert!(
        paired[0].comment.is_none(),
        "Non-extreme candidate should not be tagged"
    );
}
