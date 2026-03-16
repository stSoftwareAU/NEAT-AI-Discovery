//! Tests for unbounded activation capping detection (Issue #441).
//!
//! Detects neurons with unbounded activation functions (RELU, IDENTITY, LEAKYRELU, etc.)
//! that are producing high activations ("spiking") and would benefit from being capped
//! with a bounded version (e.g., RELU6).
//!
//! ## The Problem
//!
//! Unbounded activations like RELU can produce arbitrarily high values. When a neuron
//! consistently outputs very high activations, it may be introducing noise into the
//! network. Capping these activations with RELU6 (or similar bounded activation) can
//! reduce this noise and improve the creature's score.
//!
//! ## Detection Criteria
//!
//! A neuron is a candidate for capping if:
//! 1. Uses an unbounded activation (RELU, IDENTITY, LEAKYRELU, SOFTPLUS, ELU, SELU, etc.)
//! 2. Has high max activation (e.g., > 6.0 for RELU → RELU6)
//! 3. Significant fraction of samples exceed the capping threshold
//! 4. Hidden neurons only (output neurons excluded)
//!
//! ## Recommended Actions
//!
//! 1. Change activation from RELU to RELU6 to cap high activations
//! 2. For other unbounded activations, may recommend HARD_TANH or adjust weights

mod common;

use common::{hidden, make_creature, output, record, synapse};
use neat_ai_discovery::analysis::detection::unbounded_capping::{
    detect_unbounded_capping_candidates, unbounded_capping_to_coordinated_candidates,
};

/// Test: RELU neuron with high activations should be detected
#[test]
fn test_relu_spiking_detected() {
    let neurons = vec![
        hidden("hidden-1", "RELU"),
        hidden("hidden-2", "RELU"),
        output("output-0", "IDENTITY"),
    ];

    let creature = make_creature(neurons, vec![synapse("hidden-1", "output-0", 1.0)]);

    let hidden_neurons: Vec<(String, String, f32)> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "hidden")
        .map(|n| (n.uuid.clone(), n.squash.clone(), n.bias))
        .collect();

    // hidden-1: High activations (spiking) - should be detected
    // hidden-2: Normal activations - should NOT be detected
    let neuron_records = vec![
        (
            "hidden-1".to_string(),
            vec![
                record("hidden-1", 0, 10.0, Some(10.0)),  // Very high
                record("hidden-1", 1, 15.0, Some(15.0)),  // Very high
                record("hidden-1", 2, 8.0, Some(8.0)),    // High
                record("hidden-1", 3, 12.0, Some(12.0)),  // Very high
                record("hidden-1", 4, 7.0, Some(7.0)),    // High
                record("hidden-1", 5, 20.0, Some(20.0)),  // Very high
                record("hidden-1", 6, 9.0, Some(9.0)),    // High
                record("hidden-1", 7, 11.0, Some(11.0)),  // Very high
                record("hidden-1", 8, 6.5, Some(6.5)),    // Just above threshold
                record("hidden-1", 9, 14.0, Some(14.0)),  // Very high
                record("hidden-1", 10, 8.0, Some(8.0)),   // High
                record("hidden-1", 11, 7.5, Some(7.5)),   // High
                record("hidden-1", 12, 13.0, Some(13.0)), // Very high
                record("hidden-1", 13, 9.5, Some(9.5)),   // High
                record("hidden-1", 14, 16.0, Some(16.0)), // Very high
                record("hidden-1", 15, 10.5, Some(10.5)), // Very high
                record("hidden-1", 16, 8.0, Some(8.0)),   // High
                record("hidden-1", 17, 11.0, Some(11.0)), // Very high
                record("hidden-1", 18, 7.0, Some(7.0)),   // High
                record("hidden-1", 19, 12.0, Some(12.0)), // Very high
            ],
        ),
        (
            "hidden-2".to_string(),
            vec![
                record("hidden-2", 0, 0.5, Some(0.5)),
                record("hidden-2", 1, 1.0, Some(1.0)),
                record("hidden-2", 2, 0.8, Some(0.8)),
                record("hidden-2", 3, 1.2, Some(1.2)),
                record("hidden-2", 4, 0.3, Some(0.3)),
                record("hidden-2", 5, 2.0, Some(2.0)),
                record("hidden-2", 6, 1.5, Some(1.5)),
                record("hidden-2", 7, 0.7, Some(0.7)),
                record("hidden-2", 8, 1.1, Some(1.1)),
                record("hidden-2", 9, 0.9, Some(0.9)),
                record("hidden-2", 10, 1.3, Some(1.3)),
                record("hidden-2", 11, 0.6, Some(0.6)),
                record("hidden-2", 12, 1.8, Some(1.8)),
                record("hidden-2", 13, 0.4, Some(0.4)),
                record("hidden-2", 14, 2.5, Some(2.5)),
                record("hidden-2", 15, 1.0, Some(1.0)),
                record("hidden-2", 16, 0.8, Some(0.8)),
                record("hidden-2", 17, 1.4, Some(1.4)),
                record("hidden-2", 18, 0.5, Some(0.5)),
                record("hidden-2", 19, 1.6, Some(1.6)),
            ],
        ),
    ];

    let candidates = detect_unbounded_capping_candidates(&hidden_neurons, &neuron_records);

    // Should detect hidden-1 (spiking) but not hidden-2 (normal)
    assert_eq!(
        candidates.len(),
        1,
        "Should detect exactly one spiking neuron"
    );
    assert_eq!(candidates[0].neuron_uuid, "hidden-1");
    assert_eq!(candidates[0].current_squash, "RELU");
    assert_eq!(
        candidates[0].recommended_squash,
        Some("RELU6".to_string()),
        "Should recommend RELU6 for spiking RELU neuron"
    );
    assert!(
        candidates[0].max_activation > 6.0,
        "Max activation should exceed RELU6 cap"
    );
    assert!(
        candidates[0].fraction_above_cap > 0.5,
        "Majority of samples should exceed cap"
    );
}

/// Test: IDENTITY neuron with high activations should be detected
#[test]
fn test_identity_spiking_detected() {
    let neurons = vec![
        hidden("hidden-1", "IDENTITY"),
        output("output-0", "IDENTITY"),
    ];

    let creature = make_creature(neurons, vec![synapse("hidden-1", "output-0", 1.0)]);

    let hidden_neurons: Vec<(String, String, f32)> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "hidden")
        .map(|n| (n.uuid.clone(), n.squash.clone(), n.bias))
        .collect();

    // High positive activations for IDENTITY neuron
    let neuron_records = vec![(
        "hidden-1".to_string(),
        vec![
            record("hidden-1", 0, 5.0, Some(5.0)),
            record("hidden-1", 1, 8.0, Some(8.0)),
            record("hidden-1", 2, 12.0, Some(12.0)),
            record("hidden-1", 3, 7.0, Some(7.0)),
            record("hidden-1", 4, 15.0, Some(15.0)),
            record("hidden-1", 5, 6.0, Some(6.0)),
            record("hidden-1", 6, 9.0, Some(9.0)),
            record("hidden-1", 7, 11.0, Some(11.0)),
            record("hidden-1", 8, 4.0, Some(4.0)),
            record("hidden-1", 9, 10.0, Some(10.0)),
            record("hidden-1", 10, 8.0, Some(8.0)),
            record("hidden-1", 11, 7.0, Some(7.0)),
            record("hidden-1", 12, 13.0, Some(13.0)),
            record("hidden-1", 13, 6.0, Some(6.0)),
            record("hidden-1", 14, 9.0, Some(9.0)),
            record("hidden-1", 15, 5.0, Some(5.0)),
            record("hidden-1", 16, 8.0, Some(8.0)),
            record("hidden-1", 17, 11.0, Some(11.0)),
            record("hidden-1", 18, 7.0, Some(7.0)),
            record("hidden-1", 19, 10.0, Some(10.0)),
        ],
    )];

    let candidates = detect_unbounded_capping_candidates(&hidden_neurons, &neuron_records);

    assert_eq!(candidates.len(), 1, "Should detect IDENTITY spiking neuron");
    assert_eq!(candidates[0].neuron_uuid, "hidden-1");
    // For IDENTITY, recommend HARD_TANH if within [-1, 1] threshold is exceeded
    // Or RELU6 if all positive and high
    assert!(
        candidates[0].recommended_squash.is_some(),
        "Should recommend a bounded activation"
    );
}

/// Test: Output neurons should NOT be detected (even if spiking)
#[test]
fn test_unbounded_capping_output_neurons_excluded() {
    let neurons = vec![hidden("hidden-1", "RELU"), output("output-0", "RELU")];

    let creature = make_creature(neurons, vec![synapse("hidden-1", "output-0", 1.0)]);

    // Only include hidden neurons
    let hidden_neurons: Vec<(String, String, f32)> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "hidden")
        .map(|n| (n.uuid.clone(), n.squash.clone(), n.bias))
        .collect();

    // Hidden neuron has low activations, output neuron would spike but is excluded
    let neuron_records = vec![(
        "hidden-1".to_string(),
        vec![
            record("hidden-1", 0, 0.5, Some(0.5)),
            record("hidden-1", 1, 1.0, Some(1.0)),
            record("hidden-1", 2, 0.8, Some(0.8)),
            record("hidden-1", 3, 1.2, Some(1.2)),
            record("hidden-1", 4, 0.3, Some(0.3)),
            record("hidden-1", 5, 2.0, Some(2.0)),
            record("hidden-1", 6, 1.5, Some(1.5)),
            record("hidden-1", 7, 0.7, Some(0.7)),
            record("hidden-1", 8, 1.1, Some(1.1)),
            record("hidden-1", 9, 0.9, Some(0.9)),
            record("hidden-1", 10, 1.3, Some(1.3)),
            record("hidden-1", 11, 0.6, Some(0.6)),
            record("hidden-1", 12, 1.8, Some(1.8)),
            record("hidden-1", 13, 0.4, Some(0.4)),
            record("hidden-1", 14, 2.5, Some(2.5)),
            record("hidden-1", 15, 1.0, Some(1.0)),
            record("hidden-1", 16, 0.8, Some(0.8)),
            record("hidden-1", 17, 1.4, Some(1.4)),
            record("hidden-1", 18, 0.5, Some(0.5)),
            record("hidden-1", 19, 1.6, Some(1.6)),
        ],
    )];

    let candidates = detect_unbounded_capping_candidates(&hidden_neurons, &neuron_records);

    // hidden-1 has low activations, should not be detected
    assert!(
        candidates.is_empty(),
        "Should not detect low-activation neurons"
    );
}

/// Test: Bounded activations (RELU6, TANH, LOGISTIC) should NOT be detected
#[test]
fn test_bounded_activations_excluded() {
    let neurons = vec![
        hidden("hidden-1", "RELU6"),
        hidden("hidden-2", "TANH"),
        hidden("hidden-3", "LOGISTIC"),
        output("output-0", "IDENTITY"),
    ];

    let creature = make_creature(neurons, vec![synapse("hidden-1", "output-0", 1.0)]);

    let hidden_neurons: Vec<(String, String, f32)> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "hidden")
        .map(|n| (n.uuid.clone(), n.squash.clone(), n.bias))
        .collect();

    // Even with high activations (within their bounds), bounded activations are excluded
    let neuron_records = vec![
        (
            "hidden-1".to_string(),
            (0..20)
                .map(|i| record("hidden-1", i, 6.0, Some(10.0))) // RELU6 capped at 6
                .collect(),
        ),
        (
            "hidden-2".to_string(),
            (0..20)
                .map(|i| record("hidden-2", i, 0.99, Some(5.0))) // TANH saturated
                .collect(),
        ),
        (
            "hidden-3".to_string(),
            (0..20)
                .map(|i| record("hidden-3", i, 0.99, Some(5.0))) // LOGISTIC saturated
                .collect(),
        ),
    ];

    let candidates = detect_unbounded_capping_candidates(&hidden_neurons, &neuron_records);

    assert!(
        candidates.is_empty(),
        "Bounded activations should not be detected for capping"
    );
}

/// Test: Conversion to coordinated structural candidates
#[test]
fn test_unbounded_capping_conversion_to_coordinated_candidates() {
    let neurons = vec![hidden("hidden-1", "RELU"), output("output-0", "IDENTITY")];

    let creature = make_creature(neurons, vec![synapse("hidden-1", "output-0", 1.0)]);

    let hidden_neurons: Vec<(String, String, f32)> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "hidden")
        .map(|n| (n.uuid.clone(), n.squash.clone(), n.bias))
        .collect();

    let neuron_records = vec![(
        "hidden-1".to_string(),
        (0..20)
            .map(|i| record("hidden-1", i, 10.0 + (i as f32), Some(10.0 + (i as f32))))
            .collect(),
    )];

    let candidates = detect_unbounded_capping_candidates(&hidden_neurons, &neuron_records);

    assert!(!candidates.is_empty(), "Should detect spiking neuron");

    let coordinated = unbounded_capping_to_coordinated_candidates(&candidates);

    assert!(
        !coordinated.is_empty(),
        "Should produce coordinated candidates"
    );
    assert!(
        coordinated[0].expected_creature_score_gain > 0.0,
        "Should have positive expected improvement"
    );

    // Check the operation is a ChangeSquash
    assert_eq!(coordinated[0].operations.len(), 1);
    match &coordinated[0].operations[0] {
        neat_ai_discovery::CoordinatedStructuralOpJson::ChangeSquash {
            neuron_uuid,
            squash,
        } => {
            assert_eq!(neuron_uuid, "hidden-1");
            assert_eq!(squash, "RELU6");
        }
        _ => panic!("Expected ChangeSquash operation"),
    }
}

/// Test: Insufficient samples should not trigger detection
#[test]
fn test_unbounded_capping_insufficient_samples_skipped() {
    let neurons = vec![hidden("hidden-1", "RELU"), output("output-0", "IDENTITY")];

    let creature = make_creature(neurons, vec![synapse("hidden-1", "output-0", 1.0)]);

    let hidden_neurons: Vec<(String, String, f32)> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "hidden")
        .map(|n| (n.uuid.clone(), n.squash.clone(), n.bias))
        .collect();

    // Only 5 samples - below minimum threshold
    let neuron_records = vec![(
        "hidden-1".to_string(),
        vec![
            record("hidden-1", 0, 10.0, Some(10.0)),
            record("hidden-1", 1, 15.0, Some(15.0)),
            record("hidden-1", 2, 12.0, Some(12.0)),
            record("hidden-1", 3, 20.0, Some(20.0)),
            record("hidden-1", 4, 8.0, Some(8.0)),
        ],
    )];

    let candidates = detect_unbounded_capping_candidates(&hidden_neurons, &neuron_records);

    assert!(
        candidates.is_empty(),
        "Should not detect with insufficient samples"
    );
}

/// Test: LEAKYRELU spiking should also be detected
#[test]
fn test_leakyrelu_spiking_detected() {
    let neurons = vec![
        hidden("hidden-1", "LEAKYRELU"),
        output("output-0", "IDENTITY"),
    ];

    let creature = make_creature(neurons, vec![synapse("hidden-1", "output-0", 1.0)]);

    let hidden_neurons: Vec<(String, String, f32)> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "hidden")
        .map(|n| (n.uuid.clone(), n.squash.clone(), n.bias))
        .collect();

    let neuron_records = vec![(
        "hidden-1".to_string(),
        (0..20)
            .map(|i| record("hidden-1", i, 10.0 + (i as f32), Some(10.0 + (i as f32))))
            .collect(),
    )];

    let candidates = detect_unbounded_capping_candidates(&hidden_neurons, &neuron_records);

    assert_eq!(candidates.len(), 1, "Should detect LEAKYRELU spiking");
    assert_eq!(candidates[0].current_squash, "LEAKYRELU");
    assert!(
        candidates[0].recommended_squash.is_some(),
        "Should recommend bounded activation"
    );
}

/// Test: Neurons with mostly low activations but occasional spikes should not be detected
#[test]
fn test_occasional_spikes_not_detected() {
    let neurons = vec![hidden("hidden-1", "RELU"), output("output-0", "IDENTITY")];

    let creature = make_creature(neurons, vec![synapse("hidden-1", "output-0", 1.0)]);

    let hidden_neurons: Vec<(String, String, f32)> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "hidden")
        .map(|n| (n.uuid.clone(), n.squash.clone(), n.bias))
        .collect();

    // Most activations are low, only 2 out of 20 are high
    let neuron_records = vec![(
        "hidden-1".to_string(),
        vec![
            record("hidden-1", 0, 0.5, Some(0.5)),
            record("hidden-1", 1, 1.0, Some(1.0)),
            record("hidden-1", 2, 15.0, Some(15.0)), // Spike
            record("hidden-1", 3, 0.8, Some(0.8)),
            record("hidden-1", 4, 1.2, Some(1.2)),
            record("hidden-1", 5, 0.3, Some(0.3)),
            record("hidden-1", 6, 2.0, Some(2.0)),
            record("hidden-1", 7, 1.5, Some(1.5)),
            record("hidden-1", 8, 0.7, Some(0.7)),
            record("hidden-1", 9, 20.0, Some(20.0)), // Spike
            record("hidden-1", 10, 1.1, Some(1.1)),
            record("hidden-1", 11, 0.9, Some(0.9)),
            record("hidden-1", 12, 1.3, Some(1.3)),
            record("hidden-1", 13, 0.6, Some(0.6)),
            record("hidden-1", 14, 1.8, Some(1.8)),
            record("hidden-1", 15, 0.4, Some(0.4)),
            record("hidden-1", 16, 2.5, Some(2.5)),
            record("hidden-1", 17, 1.0, Some(1.0)),
            record("hidden-1", 18, 0.8, Some(0.8)),
            record("hidden-1", 19, 1.4, Some(1.4)),
        ],
    )];

    let candidates = detect_unbounded_capping_candidates(&hidden_neurons, &neuron_records);

    assert!(
        candidates.is_empty(),
        "Should not detect when spikes are occasional (< threshold fraction)"
    );
}
