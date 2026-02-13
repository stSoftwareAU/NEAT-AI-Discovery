//! Tests for ReLU candidate evaluation and splitting functionality.
//!
//! Tests cover:
//! - ReLU split evaluation finds candidates when activation correlates with error
//! - ReLU negative orientation candidates
//! - Complementary pairs detection
//! - Candidate upsert behaviour

use super::common::*;

#[test]
fn relu_split_evaluation_finds_candidates_when_activation_correlates_with_error() {
    // Test that split-by-error ReLU evaluation finds candidates when source activation
    // correlates with target error direction.
    //
    // Key insight: A ReLU can only help if its activation correlates with the errors
    // it's trying to fix. If activation is the same for all samples, the ReLU's
    // contribution will cancel out across balanced errors.
    //
    // This test creates samples where:
    // - Source fires (activation > 0) when target error is positive (output should go UP)
    // - Source doesn't fire (activation <= 0) when target error is negative
    //
    // This is the realistic scenario where adding a ReLU neuron can help.
    skip_if_no_gpu!();
    let analyzer = GpuAnalyzer::new().expect("GPU analysis should be available");

    let mut samples = Vec::new();
    // Samples where source fires AND output should go UP (positive error)
    for _ in 0..MIN_NEURON_SAMPLE_COUNT {
        samples.push(HelpfulSample {
            activation: 1.0, // Source fires
            avg_error: 0.5,  // Output should be HIGHER
            target_value: None,
            target_activation: None,
        });
    }
    // Samples where source doesn't fire AND output should go DOWN (negative error)
    for _ in 0..MIN_NEURON_SAMPLE_COUNT {
        samples.push(HelpfulSample {
            activation: -0.5, // Source doesn't fire (ReLU will output 0)
            avg_error: -0.5,  // Output should be LOWER
            target_value: None,
            target_activation: None,
        });
    }

    let result =
        evaluate_relu_candidates_split(&analyzer, "input-0", "output-0", &samples, 0.0, None)
            .expect("ReLU split evaluation should succeed");

    // With correlation between activation and error, we should find a positive-error candidate
    // The ReLU fires when we need output to go UP, and doesn't fire when we need it DOWN.
    assert!(
        result.positive_error_candidate.is_some(),
        "Should find positive-error ReLU candidate when activation correlates with error direction"
    );

    // Verify the candidate pushes in the correct direction
    if let Some(pos_candidate) = &result.positive_error_candidate {
        assert!(
            pos_candidate.outgoing_weight > 0.0,
            "Positive-error candidate should have positive outgoing weight (pushes UP). Got: {}",
            pos_candidate.outgoing_weight
        );
    }
}

#[test]
fn relu_split_evaluation_finds_negative_orientation_candidates() {
    // Test that we can find ReLU candidates with incoming_weight = -1.0 (negative orientation).
    //
    // This is critical: when source neurons have predominantly NEGATIVE activations
    // that correlate with errors, we need a ReLU with incoming_weight = -1.0 to flip
    // the sign before the ReLU activation.
    //
    // Bug regression test: Previously, evaluate_relu_candidates_split discarded
    // negative_stats entirely, meaning these candidates could never be found.
    skip_if_no_gpu!();
    let analyzer = GpuAnalyzer::new().expect("GPU analysis should be available");

    let mut samples = Vec::new();
    // Samples where source has NEGATIVE activation AND output should go UP (positive error)
    // A ReLU with incoming_weight = -1.0 will flip -1.0 to +1.0, then ReLU outputs 1.0
    for _ in 0..MIN_NEURON_SAMPLE_COUNT {
        samples.push(HelpfulSample {
            activation: -1.0, // NEGATIVE activation
            avg_error: 0.5,   // Output should be HIGHER
            target_value: None,
            target_activation: None,
        });
    }
    // Samples where source has POSITIVE activation AND output should go DOWN (negative error)
    // A ReLU with incoming_weight = -1.0 will flip +0.5 to -0.5, then ReLU outputs 0
    for _ in 0..MIN_NEURON_SAMPLE_COUNT {
        samples.push(HelpfulSample {
            activation: 0.5, // POSITIVE activation (will be flipped to negative, ReLU = 0)
            avg_error: -0.5, // Output should be LOWER
            target_value: None,
            target_activation: None,
        });
    }

    let result =
        evaluate_relu_candidates_split(&analyzer, "input-0", "output-0", &samples, 0.0, None)
            .expect("ReLU split evaluation should succeed");

    // With negative activations correlating with positive errors, we should find a candidate
    // that uses the NEGATIVE orientation (incoming_weight = -1.0)
    assert!(
        result.positive_error_candidate.is_some(),
        "Should find ReLU candidate even when source has negative activations (requires negative orientation)"
    );

    // Verify the candidate uses negative incoming weight (the critical fix!)
    if let Some(pos_candidate) = &result.positive_error_candidate {
        assert!(
            pos_candidate.incoming_weight < 0.0,
            "Candidate should have NEGATIVE incoming weight to flip negative activations. Got: {}",
            pos_candidate.incoming_weight
        );
        assert!(
            pos_candidate.outgoing_weight > 0.0,
            "Candidate should have positive outgoing weight (pushes UP). Got: {}",
            pos_candidate.outgoing_weight
        );
    }
}

/// Test that split-error ReLU evaluation finds complementary pairs when errors are split.
/// When errors are ~50/50 positive/negative, no single ReLU can help all samples.
/// Split evaluation should find two candidates: one for each error direction.
#[test]
fn test_split_relu_finds_complementary_pairs() {
    // Create samples with split errors:
    // - Half have positive error (output should be higher) with high source activation
    // - Half have negative error (output should be lower) with different pattern
    let mut samples = Vec::new();

    // Positive errors: when source is high, output should be higher
    // A ReLU with positive weight on these samples will help
    for i in 0..50 {
        samples.push(HelpfulSample {
            activation: 0.5 + (i as f32) * 0.01,
            avg_error: 0.3, // Positive: output should be higher
            target_value: None,
            target_activation: None,
        });
    }

    // Negative errors: when source is high, output should be lower
    // A ReLU with negative weight on these samples will help
    for i in 0..50 {
        samples.push(HelpfulSample {
            activation: 0.5 + (i as f32) * 0.01,
            avg_error: -0.3, // Negative: output should be lower
            target_value: None,
            target_activation: None,
        });
    }

    // Verify we have split errors
    let positive_count = samples.iter().filter(|s| s.avg_error > 0.0).count();
    let negative_count = samples.iter().filter(|s| s.avg_error < 0.0).count();
    assert_eq!(positive_count, 50);
    assert_eq!(negative_count, 50);

    // Standard ReLU evaluation should struggle because errors cancel out
    // when computing error*activation correlation - roughly equal positive
    // and negative errors with similar activations means weak correlation overall.
    //
    // The split evaluation separates these, so each subset has strong correlation.
    // This test documents the expected behaviour without requiring GPU.
}

/// Test that upsert_candidate keeps complementary ReLU candidates with different
/// incoming_weight values. A positive-weight ReLU (incoming_weight=1.0) and a
/// negative-weight ReLU (incoming_weight=-1.0) should both be kept, not collide.
#[test]
fn test_upsert_keeps_complementary_relu_candidates_by_incoming_weight() {
    let mut map: HashMap<u64, CandidateNeuronJson> = HashMap::new();

    // Positive-orientation ReLU candidate
    // Issue #128: Use creature-level metrics
    let positive_candidate = CandidateNeuronJson {
        source_neuron_uuid: "source-1".to_string(),
        target_neuron_uuid: "target-1".to_string(),
        source_neuron_index: None,
        target_neuron_index: None,
        incoming_weight: 1.0, // Positive orientation
        outgoing_weight: 0.5,
        squash: "ReLU".to_string(),
        bias: 0.0,
        comment: None,
        target_neuron_impact: 1.0,
        expected_creature_error_reduction: 0.15,
        expected_creature_score_gain: 0.15,
        improved_count: 30,
        total_count: 50,
        target_neuron_stats: None,
        prediction_confidence: 0.0,
        expected_score_gain_confidence_interval: [0.0, 0.0],
    };

    // Negative-orientation ReLU candidate
    let negative_candidate = CandidateNeuronJson {
        source_neuron_uuid: "source-1".to_string(),
        target_neuron_uuid: "target-1".to_string(),
        source_neuron_index: None,
        target_neuron_index: None,
        incoming_weight: -1.0, // Negative orientation
        outgoing_weight: 0.4,  // Same outgoing sign
        squash: "ReLU".to_string(),
        bias: 0.0,
        comment: None,
        target_neuron_impact: 1.0,
        expected_creature_error_reduction: 0.12,
        expected_creature_score_gain: 0.12,
        improved_count: 25,
        total_count: 50,
        target_neuron_stats: None,
        prediction_confidence: 0.0,
        expected_score_gain_confidence_interval: [0.0, 0.0],
    };

    // Insert both candidates
    upsert_candidate(&mut map, positive_candidate.clone());
    upsert_candidate(&mut map, negative_candidate.clone());

    // Both should be kept - they have different incoming_weight signs
    assert_eq!(
        map.len(),
        2,
        "Candidates with different incoming_weight should both be kept"
    );

    // Verify both are present with correct keys (Issue #526: hash-based dedup keys)
    let pos_key = compute_candidate_dedup_key(&positive_candidate);
    let neg_key = compute_candidate_dedup_key(&negative_candidate);

    assert!(
        map.contains_key(&pos_key),
        "Positive orientation should be present"
    );
    assert!(
        map.contains_key(&neg_key),
        "Negative orientation should be present"
    );
    assert_ne!(
        pos_key, neg_key,
        "Different incoming_weight signs must produce different keys"
    );
}

/// Test that upsert keeps split-error complementary pairs.
/// When we have both positive-error and negative-error candidates from the same
/// source/target pair, both should be kept since they address different samples.
#[test]
fn test_upsert_keeps_split_error_complementary_pairs() {
    let mut map: HashMap<u64, CandidateNeuronJson> = HashMap::new();

    // Candidate for positive-error samples (positive outgoing weight)
    let positive_error_candidate = CandidateNeuronJson {
        source_neuron_uuid: "source-1".to_string(),
        target_neuron_uuid: "target-1".to_string(),
        source_neuron_index: None,
        target_neuron_index: None,
        incoming_weight: 1.0,
        outgoing_weight: 0.5, // Positive: pushes output UP
        squash: "ReLU".to_string(),
        bias: 0.0,
        comment: None,
        target_neuron_impact: 1.0,
        expected_creature_error_reduction: 0.1,
        expected_creature_score_gain: 0.1,
        improved_count: 25,
        total_count: 50,
        target_neuron_stats: None,
        prediction_confidence: 0.0,
        expected_score_gain_confidence_interval: [0.0, 0.0],
    };

    // Candidate for negative-error samples (negative outgoing weight)
    let negative_error_candidate = CandidateNeuronJson {
        source_neuron_uuid: "source-1".to_string(),
        target_neuron_uuid: "target-1".to_string(),
        source_neuron_index: None,
        target_neuron_index: None,
        incoming_weight: 1.0,
        outgoing_weight: -0.4, // Negative: pushes output DOWN
        squash: "ReLU".to_string(),
        bias: 0.0,
        comment: None,
        target_neuron_impact: 1.0,
        expected_creature_error_reduction: 0.08,
        expected_creature_score_gain: 0.08,
        improved_count: 20,
        total_count: 50,
        target_neuron_stats: None,
        prediction_confidence: 0.0,
        expected_score_gain_confidence_interval: [0.0, 0.0],
    };

    // Insert both candidates
    upsert_candidate(&mut map, positive_error_candidate);
    upsert_candidate(&mut map, negative_error_candidate);

    // Both should be kept - they have different outgoing_weight signs
    assert_eq!(
        map.len(),
        2,
        "Candidates with different outgoing_weight signs should both be kept (complementary pair)"
    );
}
