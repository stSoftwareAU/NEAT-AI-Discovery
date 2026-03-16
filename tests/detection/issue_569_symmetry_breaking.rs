//! Tests for Issue #569: Symmetry-breaking discovery — detect and perturb
//! converged duplicate neurons.
//!
//! ## TDD Plan
//! 1. Test detection identifies identical hidden neurons (same squash, close bias, similar weights)
//! 2. Test detection ignores dissimilar neurons (different squash, different bias, different weights)
//! 3. Test edge case: single hidden neuron produces no candidates
//! 4. Test edge case: no hidden neurons produces no candidates
//! 5. Test coordinated candidates include setBias and/or setWeight perturbations
//! 6. Test insufficient samples returns empty
//! 7. Test multiple symmetric pairs each produce candidates

use crate::common::{hidden_with_bias, make_creature, neuron, output, synapse};

use neat_ai_discovery::analysis::detection::symmetry_breaking::{
    detect_symmetric_neurons, symmetric_neurons_to_coordinated_candidates,
};
use neat_ai_discovery::types::DiscoverRecord;

// =============================================================================
// Helpers
// =============================================================================

fn make_record(uuid: &str, idx: u32, value: f32, activation: f32, error: f32) -> DiscoverRecord {
    DiscoverRecord {
        obs_index: idx,
        neuron_uuid: uuid.to_string(),
        value: Some(value),
        activation,
        errors: vec![error],
    }
}

/// Build a creature with two symmetric hidden neurons.
///
/// Network topology:
///   input-0 --w=1.0--> h1 --w=0.5--> output-0
///   input-1 --w=0.8--> h1
///   input-0 --w=1.0--> h2 --w=0.5--> output-0
///   input-1 --w=0.8--> h2
///
/// h1 and h2 have identical squash (TANH), similar bias, and identical incoming weights.
fn symmetric_creature() -> neat_ai_discovery::CreatureJson {
    make_creature(
        vec![
            neuron("input-0", "input", "IDENTITY"),
            neuron("input-1", "input", "IDENTITY"),
            hidden_with_bias("h1", "TANH", 0.5),
            hidden_with_bias("h2", "TANH", 0.5),
            output("output-0", "LOGISTIC"),
        ],
        vec![
            synapse("input-0", "h1", 1.0),
            synapse("input-1", "h1", 0.8),
            synapse("h1", "output-0", 0.5),
            synapse("input-0", "h2", 1.0),
            synapse("input-1", "h2", 0.8),
            synapse("h2", "output-0", 0.5),
        ],
    )
}

/// Build records for two neurons with very similar activations (symmetric behaviour).
fn symmetric_records() -> Vec<(String, Vec<DiscoverRecord>)> {
    let sample_count = 60;
    vec![
        (
            "h1".to_string(),
            (0..sample_count)
                .map(|i| {
                    let value = -1.0 + (i as f32) * 2.0 / sample_count as f32;
                    let activation = value.tanh();
                    make_record("h1", i, value, activation, 0.1)
                })
                .collect(),
        ),
        (
            "h2".to_string(),
            (0..sample_count)
                .map(|i| {
                    // Near-identical activations (symmetric)
                    let value = -1.0 + (i as f32) * 2.0 / sample_count as f32;
                    let activation = value.tanh();
                    make_record("h2", i, value, activation, 0.1)
                })
                .collect(),
        ),
    ]
}

// =============================================================================
// 1. Detects identical hidden neurons
// =============================================================================

#[test]
fn test_detects_identical_symmetric_neurons() {
    let creature = symmetric_creature();
    let records = symmetric_records();

    let detected = detect_symmetric_neurons(&creature, &records);
    assert!(
        !detected.is_empty(),
        "Should detect symmetric pair h1/h2 with identical weights, squash, and bias"
    );

    // Verify the pair involves h1 and h2
    let pair = &detected[0];
    let uuids = [pair.neuron_a_uuid.as_str(), pair.neuron_b_uuid.as_str()];
    assert!(uuids.contains(&"h1"), "Pair should include h1");
    assert!(uuids.contains(&"h2"), "Pair should include h2");
    assert!(
        pair.cosine_similarity > 0.95,
        "Cosine similarity should be > 0.95, got {}",
        pair.cosine_similarity
    );
}

// =============================================================================
// 2. Ignores dissimilar neurons
// =============================================================================

#[test]
fn test_ignores_dissimilar_neurons() {
    // h1 and h2 have very different incoming weights
    let creature = make_creature(
        vec![
            neuron("input-0", "input", "IDENTITY"),
            neuron("input-1", "input", "IDENTITY"),
            hidden_with_bias("h1", "TANH", 0.5),
            hidden_with_bias("h2", "TANH", 0.5),
            output("output-0", "LOGISTIC"),
        ],
        vec![
            synapse("input-0", "h1", 1.0),
            synapse("input-1", "h1", 0.8),
            synapse("h1", "output-0", 0.5),
            // Very different weights for h2
            synapse("input-0", "h2", -0.3),
            synapse("input-1", "h2", 2.0),
            synapse("h2", "output-0", 0.5),
        ],
    );

    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        (
            "h1".to_string(),
            (0..60)
                .map(|i| {
                    let value = -1.0 + (i as f32) * 2.0 / 60.0;
                    make_record("h1", i, value, value.tanh(), 0.1)
                })
                .collect(),
        ),
        (
            "h2".to_string(),
            (0..60)
                .map(|i| {
                    // Different activation pattern due to different weights
                    let value = 1.5 - (i as f32) * 3.0 / 60.0;
                    make_record("h2", i, value, value.tanh(), 0.1)
                })
                .collect(),
        ),
    ];

    let detected = detect_symmetric_neurons(&creature, &records);
    assert!(
        detected.is_empty(),
        "Should not detect symmetric pair when incoming weights are very different"
    );
}

// =============================================================================
// 3. Different activation functions prevent detection
// =============================================================================

#[test]
fn test_different_squash_prevents_detection() {
    // Same weights but different squash functions
    let creature = make_creature(
        vec![
            neuron("input-0", "input", "IDENTITY"),
            neuron("input-1", "input", "IDENTITY"),
            hidden_with_bias("h1", "TANH", 0.5),
            hidden_with_bias("h2", "LOGISTIC", 0.5),
            output("output-0", "LOGISTIC"),
        ],
        vec![
            synapse("input-0", "h1", 1.0),
            synapse("input-1", "h1", 0.8),
            synapse("h1", "output-0", 0.5),
            synapse("input-0", "h2", 1.0),
            synapse("input-1", "h2", 0.8),
            synapse("h2", "output-0", 0.5),
        ],
    );

    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        (
            "h1".to_string(),
            (0..60)
                .map(|i| {
                    let value = -1.0 + (i as f32) * 2.0 / 60.0;
                    make_record("h1", i, value, value.tanh(), 0.1)
                })
                .collect(),
        ),
        (
            "h2".to_string(),
            (0..60)
                .map(|i| {
                    let value = -1.0 + (i as f32) * 2.0 / 60.0;
                    let activation = 1.0 / (1.0 + (-value).exp());
                    make_record("h2", i, value, activation, 0.1)
                })
                .collect(),
        ),
    ];

    let detected = detect_symmetric_neurons(&creature, &records);
    assert!(
        detected.is_empty(),
        "Different squash functions should prevent symmetric detection"
    );
}

// =============================================================================
// 4. Single hidden neuron produces no candidates
// =============================================================================

#[test]
fn test_symmetry_breaking_single_hidden_neuron_no_candidates() {
    let creature = make_creature(
        vec![
            neuron("input-0", "input", "IDENTITY"),
            hidden_with_bias("h1", "TANH", 0.5),
            output("output-0", "LOGISTIC"),
        ],
        vec![
            synapse("input-0", "h1", 1.0),
            synapse("h1", "output-0", 0.5),
        ],
    );

    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "h1".to_string(),
        (0..60)
            .map(|i| {
                let value = -1.0 + (i as f32) * 2.0 / 60.0;
                make_record("h1", i, value, value.tanh(), 0.1)
            })
            .collect(),
    )];

    let detected = detect_symmetric_neurons(&creature, &records);
    assert!(
        detected.is_empty(),
        "Single hidden neuron cannot form a symmetric pair"
    );
}

// =============================================================================
// 5. No hidden neurons produces no candidates
// =============================================================================

#[test]
fn test_symmetry_breaking_no_hidden_neurons_no_candidates() {
    let creature = make_creature(
        vec![
            neuron("input-0", "input", "IDENTITY"),
            output("output-0", "LOGISTIC"),
        ],
        vec![synapse("input-0", "output-0", 1.0)],
    );

    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![];

    let detected = detect_symmetric_neurons(&creature, &records);
    assert!(
        detected.is_empty(),
        "No hidden neurons should produce no candidates"
    );
}

// =============================================================================
// 6. Coordinated candidates include perturbation operations
// =============================================================================

#[test]
fn test_coordinated_candidates_include_perturbations() {
    let creature = symmetric_creature();
    let records = symmetric_records();

    let detected = detect_symmetric_neurons(&creature, &records);
    assert!(!detected.is_empty());

    let coordinated = symmetric_neurons_to_coordinated_candidates(&detected);
    assert!(
        !coordinated.is_empty(),
        "Should produce coordinated candidates for symmetric pair"
    );

    // Verify positive expected gain
    assert!(
        coordinated[0].expected_creature_score_gain > 0.0,
        "Expected gain should be positive"
    );

    // Verify comment references issue #569
    assert!(
        coordinated[0].comment.as_ref().unwrap().contains("569"),
        "Comment should reference issue #569, got: {:?}",
        coordinated[0].comment
    );

    // Verify the candidate includes at least a setBias or setWeight operation
    let ops_json = serde_json::to_string(&coordinated[0].operations).unwrap();
    let has_perturbation = ops_json.contains("setBias")
        || ops_json.contains("setWeight")
        || ops_json.contains("changeSquash");
    assert!(
        has_perturbation,
        "Candidate must include a perturbation operation (setBias, setWeight, or changeSquash), got: {ops_json}"
    );
}

// =============================================================================
// 7. Insufficient samples returns empty
// =============================================================================

#[test]
fn test_symmetry_breaking_insufficient_samples_returns_empty() {
    let creature = symmetric_creature();

    // Only 5 samples — below MIN_DISCOVERY_SAMPLE_COUNT
    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        (
            "h1".to_string(),
            (0..5)
                .map(|i| {
                    let value = -1.0 + (i as f32) * 2.0 / 5.0;
                    make_record("h1", i, value, value.tanh(), 0.1)
                })
                .collect(),
        ),
        (
            "h2".to_string(),
            (0..5)
                .map(|i| {
                    let value = -1.0 + (i as f32) * 2.0 / 5.0;
                    make_record("h2", i, value, value.tanh(), 0.1)
                })
                .collect(),
        ),
    ];

    let detected = detect_symmetric_neurons(&creature, &records);
    assert!(
        detected.is_empty(),
        "Insufficient samples should produce no candidates"
    );
}

// =============================================================================
// 8. Multiple symmetric pairs each produce candidates
// =============================================================================

#[test]
fn test_multiple_symmetric_pairs() {
    // Two distinct pairs: (h1, h2) and (h3, h4)
    let creature = make_creature(
        vec![
            neuron("input-0", "input", "IDENTITY"),
            neuron("input-1", "input", "IDENTITY"),
            hidden_with_bias("h1", "TANH", 0.5),
            hidden_with_bias("h2", "TANH", 0.5),
            hidden_with_bias("h3", "LOGISTIC", 1.0),
            hidden_with_bias("h4", "LOGISTIC", 1.0),
            output("output-0", "LOGISTIC"),
        ],
        vec![
            // Pair 1: h1/h2 with identical weights
            synapse("input-0", "h1", 1.0),
            synapse("input-1", "h1", 0.8),
            synapse("h1", "output-0", 0.5),
            synapse("input-0", "h2", 1.0),
            synapse("input-1", "h2", 0.8),
            synapse("h2", "output-0", 0.5),
            // Pair 2: h3/h4 with identical weights (different from pair 1)
            synapse("input-0", "h3", 0.3),
            synapse("input-1", "h3", -0.7),
            synapse("h3", "output-0", 0.9),
            synapse("input-0", "h4", 0.3),
            synapse("input-1", "h4", -0.7),
            synapse("h4", "output-0", 0.9),
        ],
    );

    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        (
            "h1".to_string(),
            (0..60)
                .map(|i| {
                    let value = -1.0 + (i as f32) * 2.0 / 60.0;
                    make_record("h1", i, value, value.tanh(), 0.1)
                })
                .collect(),
        ),
        (
            "h2".to_string(),
            (0..60)
                .map(|i| {
                    let value = -1.0 + (i as f32) * 2.0 / 60.0;
                    make_record("h2", i, value, value.tanh(), 0.1)
                })
                .collect(),
        ),
        (
            "h3".to_string(),
            (0..60)
                .map(|i| {
                    let value = 0.5 + (i as f32) * 1.0 / 60.0;
                    let activation = 1.0 / (1.0 + (-value).exp());
                    make_record("h3", i, value, activation, 0.1)
                })
                .collect(),
        ),
        (
            "h4".to_string(),
            (0..60)
                .map(|i| {
                    let value = 0.5 + (i as f32) * 1.0 / 60.0;
                    let activation = 1.0 / (1.0 + (-value).exp());
                    make_record("h4", i, value, activation, 0.1)
                })
                .collect(),
        ),
    ];

    let detected = detect_symmetric_neurons(&creature, &records);
    assert!(
        detected.len() >= 2,
        "Should detect at least 2 symmetric pairs, got: {}",
        detected.len()
    );

    let coordinated = symmetric_neurons_to_coordinated_candidates(&detected);
    assert!(
        coordinated.len() >= 2,
        "Should produce at least 2 coordinated candidates, got: {}",
        coordinated.len()
    );
}

// =============================================================================
// 9. Bias difference beyond tolerance prevents detection
// =============================================================================

#[test]
fn test_large_bias_difference_prevents_detection() {
    // Same weights but very different bias values
    let creature = make_creature(
        vec![
            neuron("input-0", "input", "IDENTITY"),
            neuron("input-1", "input", "IDENTITY"),
            hidden_with_bias("h1", "TANH", 0.5),
            hidden_with_bias("h2", "TANH", 5.0), // Very different bias
            output("output-0", "LOGISTIC"),
        ],
        vec![
            synapse("input-0", "h1", 1.0),
            synapse("input-1", "h1", 0.8),
            synapse("h1", "output-0", 0.5),
            synapse("input-0", "h2", 1.0),
            synapse("input-1", "h2", 0.8),
            synapse("h2", "output-0", 0.5),
        ],
    );

    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        (
            "h1".to_string(),
            (0..60)
                .map(|i| {
                    let value = -1.0 + (i as f32) * 2.0 / 60.0;
                    make_record("h1", i, value, value.tanh(), 0.1)
                })
                .collect(),
        ),
        (
            "h2".to_string(),
            (0..60)
                .map(|i| {
                    let value = 4.0 + (i as f32) * 2.0 / 60.0;
                    make_record("h2", i, value, value.tanh(), 0.1)
                })
                .collect(),
        ),
    ];

    let detected = detect_symmetric_neurons(&creature, &records);
    assert!(
        detected.is_empty(),
        "Large bias difference should prevent symmetric detection"
    );
}
