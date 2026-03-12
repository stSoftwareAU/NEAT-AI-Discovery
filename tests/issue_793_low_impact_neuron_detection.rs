//! Tests for Issue #793: Expand remove-low-impact candidate generation.
//!
//! Low-impact neurons have activations slightly above the dead-neuron threshold
//! (1e-6) but still negligible (up to 1e-3). Detecting these neurons and
//! recommending their removal increases the volume of remove-low-impact
//! candidates while maintaining a high success rate.
//!
//! ## TDD Plan
//! 1. Neuron with mean abs activation in the low-impact range is detected
//! 2. Truly dead neurons (below 1e-6) are NOT detected (handled by dead_neuron module)
//! 3. Active neurons (above the low-impact threshold) are NOT detected
//! 4. Confidence scales with sample count (more samples = higher confidence)
//! 5. Confidence scales inversely with activation magnitude (lower = more confident)
//! 6. Output and input neurons are excluded
//! 7. Candidates convert to coordinated structural RemoveNeuron operations
//! 8. Mixed network: only low-impact neurons are detected

mod common;

use common::{make_creature, neuron, record, synapse};
use neat_ai_discovery::analysis::detection::low_impact_neuron::{
    LowImpactNeuronCandidate, detect_low_impact_neurons,
    low_impact_neurons_to_coordinated_candidates,
};
use neat_ai_discovery::types::DiscoverRecord;

/// Test 1: Neuron with mean abs activation in low-impact range (1e-5) is detected.
#[test]
fn test_detects_low_impact_neuron() {
    let creature = make_creature(
        vec![
            neuron("hidden-low", "hidden", "RELU"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![synapse("hidden-low", "output-1", 0.5)],
    );

    // Mean abs activation ~1e-5: above dead threshold (1e-6) but below low-impact ceiling (1e-3)
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| record("hidden-low", i, 1e-5, Some(0.01)))
        .collect();

    let candidates =
        detect_low_impact_neurons(&creature, &[("hidden-low".to_string(), records)], None);

    assert_eq!(candidates.len(), 1, "Should detect one low-impact neuron");
    let c = &candidates[0];
    assert_eq!(c.neuron_uuid, "hidden-low");
    assert!(
        c.mean_abs_activation > 1e-6,
        "Should be above dead threshold"
    );
    assert!(
        c.mean_abs_activation < 1e-3,
        "Should be below low-impact ceiling"
    );
    assert!(
        c.removal_confidence > 0.0,
        "Should have positive confidence"
    );
}

/// Test 2: Truly dead neuron (below 1e-6) is NOT detected — handled by dead_neuron module.
#[test]
fn test_does_not_detect_dead_neurons() {
    let creature = make_creature(
        vec![
            neuron("hidden-dead", "hidden", "RELU"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![synapse("hidden-dead", "output-1", 0.5)],
    );

    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| record("hidden-dead", i, 0.0, Some(-5.0)))
        .collect();

    let candidates =
        detect_low_impact_neurons(&creature, &[("hidden-dead".to_string(), records)], None);

    assert!(
        candidates.is_empty(),
        "Dead neurons should not be flagged by low-impact detector"
    );
}

/// Test 3: Active neuron (above low-impact ceiling) is NOT detected.
#[test]
fn test_does_not_detect_active_neuron() {
    let creature = make_creature(
        vec![
            neuron("hidden-active", "hidden", "TANH"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![synapse("hidden-active", "output-1", 0.5)],
    );

    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = 0.5 + 0.3 * ((i as f32 * 0.1).sin());
            record("hidden-active", i, activation, Some(0.5))
        })
        .collect();

    let candidates =
        detect_low_impact_neurons(&creature, &[("hidden-active".to_string(), records)], None);

    assert!(
        candidates.is_empty(),
        "Active neuron should not be flagged as low-impact"
    );
}

/// Test 4: More samples yield higher confidence.
#[test]
fn test_confidence_increases_with_sample_count() {
    let creature = make_creature(
        vec![
            neuron("hidden-low", "hidden", "RELU"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![synapse("hidden-low", "output-1", 0.5)],
    );

    let small_records: Vec<DiscoverRecord> = (0..30)
        .map(|i| record("hidden-low", i, 5e-5, Some(0.01)))
        .collect();

    let large_records: Vec<DiscoverRecord> = (0..500)
        .map(|i| record("hidden-low", i, 5e-5, Some(0.01)))
        .collect();

    let small_candidates = detect_low_impact_neurons(
        &creature,
        &[("hidden-low".to_string(), small_records)],
        None,
    );
    let large_candidates = detect_low_impact_neurons(
        &creature,
        &[("hidden-low".to_string(), large_records)],
        None,
    );

    assert_eq!(small_candidates.len(), 1);
    assert_eq!(large_candidates.len(), 1);
    assert!(
        large_candidates[0].removal_confidence > small_candidates[0].removal_confidence,
        "More samples should yield higher confidence: {} vs {}",
        large_candidates[0].removal_confidence,
        small_candidates[0].removal_confidence
    );
}

/// Test 5: Lower activation yields higher confidence.
#[test]
fn test_confidence_increases_with_lower_activation() {
    let creature = make_creature(
        vec![
            neuron("hidden-a", "hidden", "RELU"),
            neuron("hidden-b", "hidden", "RELU"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("hidden-a", "output-1", 0.5),
            synapse("hidden-b", "output-1", 0.5),
        ],
    );

    // Neuron A: very low impact (5e-6)
    let records_a: Vec<DiscoverRecord> = (0..100)
        .map(|i| record("hidden-a", i, 5e-6, Some(0.01)))
        .collect();

    // Neuron B: moderate-low impact (5e-4)
    let records_b: Vec<DiscoverRecord> = (0..100)
        .map(|i| record("hidden-b", i, 5e-4, Some(0.01)))
        .collect();

    let candidates = detect_low_impact_neurons(
        &creature,
        &[
            ("hidden-a".to_string(), records_a),
            ("hidden-b".to_string(), records_b),
        ],
        None,
    );

    assert_eq!(candidates.len(), 2, "Both should be detected");
    let c_a = candidates
        .iter()
        .find(|c| c.neuron_uuid == "hidden-a")
        .unwrap();
    let c_b = candidates
        .iter()
        .find(|c| c.neuron_uuid == "hidden-b")
        .unwrap();
    assert!(
        c_a.removal_confidence > c_b.removal_confidence,
        "Lower activation should yield higher confidence: {} vs {}",
        c_a.removal_confidence,
        c_b.removal_confidence
    );
}

/// Test 6: Output and input neurons are excluded.
#[test]
fn test_output_and_input_neurons_excluded() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![],
    );

    let records_input: Vec<DiscoverRecord> = (0..100)
        .map(|i| record("input-1", i, 5e-5, Some(0.01)))
        .collect();
    let records_output: Vec<DiscoverRecord> = (0..100)
        .map(|i| record("output-1", i, 5e-5, Some(0.01)))
        .collect();

    let candidates = detect_low_impact_neurons(
        &creature,
        &[
            ("input-1".to_string(), records_input),
            ("output-1".to_string(), records_output),
        ],
        None,
    );

    assert!(
        candidates.is_empty(),
        "Input and output neurons should not be flagged"
    );
}

/// Test 7: Candidates convert to coordinated structural RemoveNeuron operations.
#[test]
fn test_candidates_produce_coordinated_removal_operations() {
    let candidate = LowImpactNeuronCandidate {
        neuron_uuid: "hidden-low".to_string(),
        mean_abs_activation: 5e-5,
        activation_std_dev: 1e-5,
        sample_count: 200,
        removal_confidence: 0.75,
        estimated_improvement: 0.002,
    };

    let coordinated = low_impact_neurons_to_coordinated_candidates(&[candidate]);

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
    assert!(
        c.comment.is_some(),
        "Should have a comment explaining the removal"
    );

    // Check that operations include a RemoveNeuron
    let ops_json = serde_json::to_string(&c.operations).unwrap();
    assert!(
        ops_json.contains("removeNeuron"),
        "Should include removeNeuron operation, got: {ops_json}"
    );
}

/// Test 8: Mixed network — only low-impact neurons are detected.
#[test]
fn test_mixed_network_only_low_impact_detected() {
    let creature = make_creature(
        vec![
            neuron("dead", "hidden", "RELU"),
            neuron("low-impact", "hidden", "TANH"),
            neuron("active", "hidden", "RELU"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("dead", "output-1", 0.5),
            synapse("low-impact", "output-1", 0.3),
            synapse("active", "output-1", 0.8),
        ],
    );

    // Dead neuron: activation 0.0 (below 1e-6) — should NOT be detected here
    let dead_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| record("dead", i, 0.0, Some(-3.0)))
        .collect();

    // Low-impact neuron: activation 1e-4 — should be detected
    let low_impact_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| record("low-impact", i, 1e-4, Some(1e-5)))
        .collect();

    // Active neuron: activation ~0.5 — should NOT be detected
    let active_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = 0.5 + 0.2 * ((i as f32 * 0.1).sin());
            record("active", i, activation, Some(1.0))
        })
        .collect();

    let candidates = detect_low_impact_neurons(
        &creature,
        &[
            ("dead".to_string(), dead_records),
            ("low-impact".to_string(), low_impact_records),
            ("active".to_string(), active_records),
        ],
        None,
    );

    assert_eq!(
        candidates.len(),
        1,
        "Should detect exactly 1 low-impact neuron"
    );
    assert_eq!(candidates[0].neuron_uuid, "low-impact");
}

/// Test 9: Too few samples should not trigger detection.
#[test]
fn test_insufficient_samples_not_detected() {
    let creature = make_creature(
        vec![
            neuron("hidden-few", "hidden", "RELU"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![synapse("hidden-few", "output-1", 0.5)],
    );

    let records: Vec<DiscoverRecord> = (0..5)
        .map(|i| record("hidden-few", i, 5e-5, Some(0.01)))
        .collect();

    let candidates =
        detect_low_impact_neurons(&creature, &[("hidden-few".to_string(), records)], None);

    assert!(
        candidates.is_empty(),
        "Too few samples should not trigger detection"
    );
}

/// Test 10: Neuron with low mean but high variance is NOT low-impact.
/// If a neuron has sporadic high activations, it may still be useful.
#[test]
fn test_low_mean_high_variance_not_detected() {
    let creature = make_creature(
        vec![
            neuron("hidden-sporadic", "hidden", "RELU"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![synapse("hidden-sporadic", "output-1", 0.5)],
    );

    // Most samples are 0, but 10% have activation = 0.1 → mean_abs ≈ 0.01
    // This is above the low-impact ceiling, so not detected.
    // But even if mean_abs were in range, high variance means it's not consistently low.
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = if i % 10 == 0 { 0.1 } else { 0.0 };
            record("hidden-sporadic", i, activation, Some(0.0))
        })
        .collect();

    let candidates =
        detect_low_impact_neurons(&creature, &[("hidden-sporadic".to_string(), records)], None);

    assert!(
        candidates.is_empty(),
        "Sporadic high activations should prevent low-impact detection"
    );
}
