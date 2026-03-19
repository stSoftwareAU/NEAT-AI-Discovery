//! Tests for Issue #341: Detect dead neurons for removal candidates.
//!
//! Dead neurons always output zero or near-zero activation, wasting computation
//! without contributing to the network's output. This module detects them and
//! recommends their removal.
//!
//! ## TDD Plan
//! 1. Create test network with intentionally dead neurons
//! 2. Verify detection identifies them with correct statistics
//! 3. Verify active neurons are not flagged
//! 4. Test edge cases: rarely active neurons, tiny but nonzero activation
//! 5. Test coordinated structural candidate conversion

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use crate::common::{make_creature, neuron, record, synapse};
use neat_ai_discovery::analysis::detection::dead_neuron::{
    DeadNeuronCandidate, dead_neurons_to_coordinated_candidates, detect_dead_neurons,
};
use neat_ai_discovery::types::DiscoverRecord;

/// Test 1: Neuron with all-zero activations is detected as dead.
#[test]
fn test_detects_all_zero_activation_neuron() {
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

    let candidates = detect_dead_neurons(&creature, &[("hidden-dead".to_string(), records)], None);

    assert_eq!(candidates.len(), 1, "Should detect one dead neuron");
    let c = &candidates[0];
    assert_eq!(c.neuron_uuid, "hidden-dead");
    assert!(
        c.mean_abs_activation < 1e-6,
        "Mean abs activation should be near zero"
    );
    assert!(c.activation_std_dev < 1e-6, "Std dev should be near zero");
    assert!(c.removal_confidence > 0.9, "Confidence should be high");
}

/// Test 2: Neuron with near-zero but nonzero activations is detected.
#[test]
fn test_detects_near_zero_activation_neuron() {
    let creature = make_creature(
        vec![
            neuron("hidden-tiny", "hidden", "TANH"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![synapse("hidden-tiny", "output-1", 0.3)],
    );

    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| record("hidden-tiny", i, 1e-8 * (i as f32), Some(1e-9)))
        .collect();

    let candidates = detect_dead_neurons(&creature, &[("hidden-tiny".to_string(), records)], None);

    assert_eq!(candidates.len(), 1, "Should detect near-zero neuron");
    assert!(candidates[0].mean_abs_activation < 1e-6);
}

/// Test 3: Active neuron is NOT flagged as dead.
#[test]
fn test_does_not_flag_active_neuron() {
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
        detect_dead_neurons(&creature, &[("hidden-active".to_string(), records)], None);

    assert!(
        candidates.is_empty(),
        "Active neuron should not be flagged as dead"
    );
}

/// Test 4: Neuron that is rarely but meaningfully active is NOT flagged.
/// This tests the false-positive prevention: a neuron active on only 5% of samples
/// but with significant activation when active should not be considered dead.
#[test]
fn test_rarely_active_neuron_not_flagged() {
    let creature = make_creature(
        vec![
            neuron("hidden-rare", "hidden", "RELU"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![synapse("hidden-rare", "output-1", 0.5)],
    );

    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            // Active on 5% of samples with significant activation (0.8)
            let activation = if i % 20 == 0 { 0.8 } else { 0.0 };
            record(
                "hidden-rare",
                i,
                activation,
                Some(if i % 20 == 0 { 1.0 } else { -1.0 }),
            )
        })
        .collect();

    let candidates = detect_dead_neurons(&creature, &[("hidden-rare".to_string(), records)], None);

    assert!(
        candidates.is_empty(),
        "Rarely but meaningfully active neuron should not be flagged"
    );
}

/// Test 5: Neuron with constant tiny activation and zero variance is dead.
#[test]
fn test_constant_tiny_activation_is_dead() {
    let creature = make_creature(
        vec![
            neuron("hidden-const", "hidden", "TANH"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![synapse("hidden-const", "output-1", 0.5)],
    );

    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| record("hidden-const", i, 1e-8, Some(1e-9)))
        .collect();

    let candidates = detect_dead_neurons(&creature, &[("hidden-const".to_string(), records)], None);

    assert_eq!(
        candidates.len(),
        1,
        "Constant tiny activation neuron is dead"
    );
    assert!(candidates[0].activation_std_dev < 1e-6);
}

/// Test 6: Output neurons should never be flagged as dead.
#[test]
fn test_dead_neuron_output_neurons_not_flagged() {
    let creature = make_creature(vec![neuron("output-1", "output", "IDENTITY")], vec![]);

    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| record("output-1", i, 0.0, Some(0.0)))
        .collect();

    let candidates = detect_dead_neurons(&creature, &[("output-1".to_string(), records)], None);

    assert!(
        candidates.is_empty(),
        "Output neurons should not be flagged"
    );
}

/// Test 7: Too few samples should not trigger detection.
#[test]
fn test_dead_neuron_insufficient_samples_not_flagged() {
    let creature = make_creature(
        vec![
            neuron("hidden-few", "hidden", "RELU"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![synapse("hidden-few", "output-1", 0.5)],
    );

    let records: Vec<DiscoverRecord> = (0..5)
        .map(|i| record("hidden-few", i, 0.0, Some(-5.0)))
        .collect();

    let candidates = detect_dead_neurons(&creature, &[("hidden-few".to_string(), records)], None);

    assert!(
        candidates.is_empty(),
        "Too few samples should not trigger detection"
    );
}

/// Test 8: Mixed neurons — only dead ones are detected.
#[test]
fn test_mixed_neurons_only_dead_detected() {
    let creature = make_creature(
        vec![
            neuron("dead-1", "hidden", "RELU"),
            neuron("dead-2", "hidden", "TANH"),
            neuron("alive-1", "hidden", "RELU"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("dead-1", "output-1", 0.5),
            synapse("dead-2", "output-1", 0.3),
            synapse("alive-1", "output-1", 0.8),
        ],
    );

    let dead_records_1: Vec<DiscoverRecord> = (0..100)
        .map(|i| record("dead-1", i, 0.0, Some(-3.0)))
        .collect();
    let dead_records_2: Vec<DiscoverRecord> = (0..100)
        .map(|i| record("dead-2", i, 1e-9, Some(1e-10)))
        .collect();
    let alive_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = 0.5 + 0.2 * ((i as f32 * 0.1).sin());
            record("alive-1", i, activation, Some(1.0))
        })
        .collect();

    let candidates = detect_dead_neurons(
        &creature,
        &[
            ("dead-1".to_string(), dead_records_1),
            ("dead-2".to_string(), dead_records_2),
            ("alive-1".to_string(), alive_records),
        ],
        None,
    );

    assert_eq!(candidates.len(), 2, "Should detect exactly 2 dead neurons");
    let uuids: Vec<&str> = candidates.iter().map(|c| c.neuron_uuid.as_str()).collect();
    assert!(uuids.contains(&"dead-1"), "Should detect dead-1");
    assert!(uuids.contains(&"dead-2"), "Should detect dead-2");
}

/// Test 9: Dead neuron candidates produce correct coordinated structural operations.
#[test]
fn test_dead_neuron_candidates_produce_coordinated_removal_operations() {
    let candidate = DeadNeuronCandidate {
        neuron_uuid: "hidden-dead".to_string(),
        mean_abs_activation: 1e-8,
        activation_std_dev: 0.0,
        sample_count: 1000,
        connected_outputs: vec!["output-1".to_string(), "output-3".to_string()],
        removal_confidence: 0.99,
        estimated_improvement: 0.001,
    };

    let coordinated = dead_neurons_to_coordinated_candidates(&[candidate]);

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

/// Test 10: Neuron with negative near-zero activation (e.g., leaky RELU) is detected.
#[test]
fn test_detects_negative_near_zero_activation() {
    let creature = make_creature(
        vec![
            neuron("hidden-neg", "hidden", "LEAKYRELU"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![synapse("hidden-neg", "output-1", 0.5)],
    );

    // LeakyRELU with very small negative outputs (dead-like behaviour)
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| record("hidden-neg", i, -1e-8, Some(-1e-7)))
        .collect();

    let candidates = detect_dead_neurons(&creature, &[("hidden-neg".to_string(), records)], None);

    assert_eq!(
        candidates.len(),
        1,
        "Should detect near-zero negative activation"
    );
}

/// Test 11: Neuron with zero mean but high variance is NOT dead (bipolar activation).
#[test]
fn test_zero_mean_high_variance_not_dead() {
    let creature = make_creature(
        vec![
            neuron("hidden-bipolar", "hidden", "TANH"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![synapse("hidden-bipolar", "output-1", 0.5)],
    );

    // Activation alternates between +0.5 and -0.5 → mean ≈ 0 but high variance
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = if i % 2 == 0 { 0.5 } else { -0.5 };
            record("hidden-bipolar", i, activation, Some(0.0))
        })
        .collect();

    let candidates =
        detect_dead_neurons(&creature, &[("hidden-bipolar".to_string(), records)], None);

    assert!(
        candidates.is_empty(),
        "Zero-mean but high-variance neuron is not dead"
    );
}

/// Test 12: Sample count is correctly recorded in the candidate.
#[test]
fn test_sample_count_recorded() {
    let creature = make_creature(
        vec![
            neuron("hidden-dead", "hidden", "RELU"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![synapse("hidden-dead", "output-1", 0.5)],
    );

    let records: Vec<DiscoverRecord> = (0..200)
        .map(|i| record("hidden-dead", i, 0.0, Some(-5.0)))
        .collect();

    let candidates = detect_dead_neurons(&creature, &[("hidden-dead".to_string(), records)], None);

    assert_eq!(candidates.len(), 1);
    assert_eq!(
        candidates[0].sample_count, 200,
        "Sample count should match record count"
    );
}

/// Test 13: Connected outputs are correctly identified.
#[test]
fn test_connected_outputs_identified() {
    let creature = make_creature(
        vec![
            neuron("hidden-dead", "hidden", "RELU"),
            neuron("hidden-mid", "hidden", "RELU"),
            neuron("output-1", "output", "IDENTITY"),
            neuron("output-2", "output", "IDENTITY"),
        ],
        vec![
            synapse("hidden-dead", "hidden-mid", 0.5),
            synapse("hidden-dead", "output-1", 0.3),
            synapse("hidden-mid", "output-2", 0.4),
        ],
    );

    let dead_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| record("hidden-dead", i, 0.0, Some(-5.0)))
        .collect();
    let mid_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| record("hidden-mid", i, 0.5, Some(1.0)))
        .collect();

    let candidates = detect_dead_neurons(
        &creature,
        &[
            ("hidden-dead".to_string(), dead_records),
            ("hidden-mid".to_string(), mid_records),
        ],
        None,
    );

    assert_eq!(candidates.len(), 1);
    // The dead neuron connects directly to output-1 and indirectly to output-2 via hidden-mid
    assert!(
        !candidates[0].connected_outputs.is_empty(),
        "Should identify connected outputs"
    );
}
