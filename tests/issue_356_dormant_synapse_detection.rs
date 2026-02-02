//! Tests for Issue #356: Detect dormant synapses for removal candidates.
//!
//! Dormant synapses have near-zero weights that contribute negligible signal to their
//! target neurons. This module detects them and recommends removal to reduce complexity.
//!
//! ## TDD Plan
//! 1. Create test network with intentionally dormant synapses
//! 2. Verify detection identifies them with correct statistics
//! 3. Verify active synapses are not flagged
//! 4. Test edge cases: sole connection, zero samples
//! 5. Test coordinated structural candidate conversion

mod common;

use common::{make_creature, neuron, synapse};
use neat_ai_discovery::analysis::dormant_synapse::{
    detect_dormant_synapses, dormant_synapses_to_coordinated_candidates,
};
use neat_ai_discovery::types::DiscoverRecord;

/// Helper: create a DiscoverRecord with value derived from activation.
/// Dormant-synapse tests use `value = activation * 0.8` by convention.
fn record(neuron_uuid: &str, obs_index: u32, activation: f32) -> DiscoverRecord {
    DiscoverRecord {
        obs_index,
        neuron_uuid: neuron_uuid.to_string(),
        value: Some(activation * 0.8),
        activation,
        errors: vec![0.01],
    }
}

/// Test 1: Synapse with near-zero weight is detected as dormant.
#[test]
fn test_detects_near_zero_weight_synapse() {
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

    let records_1: Vec<DiscoverRecord> = (0..100).map(|i| record("input-1", i, 0.5)).collect();
    let records_2: Vec<DiscoverRecord> = (0..100).map(|i| record("input-2", i, 0.5)).collect();

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

/// Test 2: Synapse with meaningful weight is NOT flagged.
#[test]
fn test_active_synapse_not_flagged() {
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

    let records_1: Vec<DiscoverRecord> = (0..100).map(|i| record("input-1", i, 0.5)).collect();
    let records_2: Vec<DiscoverRecord> = (0..100).map(|i| record("input-2", i, 0.5)).collect();

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

/// Test 3: Sole connection to target is NOT flagged even if dormant.
#[test]
fn test_sole_connection_not_flagged() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "output-1", 1e-6), // Dormant but sole connection
        ],
    );

    let records: Vec<DiscoverRecord> = (0..100).map(|i| record("input-1", i, 0.5)).collect();

    let candidates = detect_dormant_synapses(&creature, &[("input-1".to_string(), records)]);

    assert!(
        candidates.is_empty(),
        "Sole connection should not be flagged even if dormant"
    );
}

/// Test 4: Insufficient samples should not trigger detection.
#[test]
fn test_insufficient_samples_not_flagged() {
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

    let records: Vec<DiscoverRecord> = (0..5).map(|i| record("input-1", i, 0.5)).collect();

    let candidates = detect_dormant_synapses(&creature, &[("input-1".to_string(), records)]);

    assert!(
        candidates.is_empty(),
        "Too few samples should not trigger detection"
    );
}

/// Test 5: Dormant synapse candidates produce correct coordinated removal operations.
#[test]
fn test_candidates_produce_coordinated_removal_operations() {
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

    let records_1: Vec<DiscoverRecord> = (0..100).map(|i| record("input-1", i, 0.5)).collect();
    let records_2: Vec<DiscoverRecord> = (0..100).map(|i| record("input-2", i, 0.5)).collect();

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

    // Check that operations include a RemoveSynapse
    let ops_json = serde_json::to_string(&c.operations).unwrap();
    assert!(
        ops_json.contains("removeSynapse"),
        "Should include removeSynapse operation, got: {ops_json}"
    );
}

/// Test 6: Multiple dormant synapses are all detected.
#[test]
fn test_multiple_dormant_synapses_detected() {
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

    let records_1: Vec<DiscoverRecord> = (0..100).map(|i| record("input-1", i, 0.5)).collect();
    let records_2: Vec<DiscoverRecord> = (0..100).map(|i| record("input-2", i, 0.5)).collect();
    let records_3: Vec<DiscoverRecord> = (0..100).map(|i| record("input-3", i, 0.5)).collect();

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

/// Test 7: Other fan-in count is correctly recorded.
#[test]
fn test_other_fan_in_count_correct() {
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

    let records_1: Vec<DiscoverRecord> = (0..100).map(|i| record("input-1", i, 0.5)).collect();

    let candidates = detect_dormant_synapses(&creature, &[("input-1".to_string(), records_1)]);

    assert_eq!(candidates.len(), 1);
    assert_eq!(
        candidates[0].other_fan_in, 2,
        "Other fan-in should be 2 (input-2 and input-3)"
    );
}
