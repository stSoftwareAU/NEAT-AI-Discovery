//! Tests for gradient flow analysis in focus neuron selection (Issue #206)
//!
//! ## Overview
//!
//! This module tests the gradient flow analysis feature which enhances focus neuron
//! selection by analysing gradient flow through the network to identify neurons with
//! high learning potential.
//!
//! ## Key Metrics
//!
//! 1. **Gradient magnitude**: How much error signal flows through this neuron
//! 2. **Saturation ratio**: Percentage of samples in saturated activation region
//! 3. **Dead neuron ratio**: Percentage of samples with zero gradient (`ReLU` dead zones)
//!
//! ## Test Strategy (TDD)
//!
//! 1. Test saturation detection for TANH near ±1
//! 2. Test dead neuron detection for `ReLU` with negative inputs
//! 3. Test gradient magnitude ranking
//! 4. Verify integration with existing focus selection

#![allow(clippy::cast_possible_truncation)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use neat_ai_discovery::focus::{
    GradientFlowStats, compute_gradient_flow_stats, rank_focus_neurons,
};
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
use tempfile::NamedTempFile;

/// Helper to create a neuron with specified squash function
fn neuron(uuid: &str, neuron_type: &str, squash: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
        squash: squash.to_string(),
        bias: 0.0,
    }
}

/// Helper to create a synapse
fn synapse(from: &str, to: &str, weight: f32) -> SynapseJson {
    SynapseJson {
        from_uuid: from.to_string(),
        to_uuid: to.to_string(),
        weight,
        synapse_type: None,
    }
}

/// Helper to create a simple creature with specified neurons and synapses
fn create_creature(
    neurons: Vec<NeuronJson>,
    synapses: Vec<SynapseJson>,
    input: usize,
) -> CreatureJson {
    let output_count = neurons.iter().filter(|n| n.neuron_type == "output").count();
    CreatureJson {
        neurons,
        synapses,
        input,
        output: output_count,
    }
}

/// Sample data: (value, activation, error) triple.
type Sample = (f32, f32, f32);

/// Neuron sample data: (uuid, samples) pair.
type NeuronSamples<'a> = (&'a str, Vec<Sample>);

/// Create discovery records with specific pre-activation values
///
/// The `value` field represents pre-activation (before squash function).
/// The `activation` field represents post-squash output.
fn create_records_with_values(data: Vec<NeuronSamples<'_>>) -> Vec<DiscoverRecord> {
    let mut records = Vec::new();
    for (uuid, samples) in data {
        for (obs_idx, (value, activation, error)) in samples.into_iter().enumerate() {
            records.push(DiscoverRecord::new(
                obs_idx as u32,
                uuid.to_string(),
                Some(value),
                activation,
                vec![error],
            ));
        }
    }
    records
}

// =============================================================================
// Test: Saturation Detection for TANH
// =============================================================================

/// Test that TANH neurons with activations near ±1 are detected as saturated.
///
/// TANH saturation:
/// - At |input| > 3: output ≈ ±1, derivative ≈ 0
/// - The saturation ratio should be high when most samples have |value| > 3
#[test]
fn test_tanh_saturation_detection_high_saturation() {
    // Create a network with a TANH neuron
    let creature = create_creature(
        vec![
            neuron("hidden-1", "hidden", "TANH"),
            neuron("output-0", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-0", "hidden-1", 1.0),
            synapse("hidden-1", "output-0", 1.0),
        ],
        1,
    );

    // Create records where hidden-1 is heavily saturated (|value| > 3)
    // TANH(5) ≈ 0.9999, TANH(-5) ≈ -0.9999
    let records = create_records_with_values(vec![
        (
            "hidden-1",
            vec![
                (5.0, 0.9999, 0.1),   // Saturated positive
                (-5.0, -0.9999, 0.1), // Saturated negative
                (4.0, 0.9993, 0.1),   // Saturated positive
                (-4.0, -0.9993, 0.1), // Saturated negative
                (0.5, 0.462, 0.1),    // Not saturated
            ],
        ),
        (
            "output-0",
            vec![
                (0.9999, 0.9999, 0.1),
                (-0.9999, -0.9999, 0.1),
                (0.9993, 0.9993, 0.1),
                (-0.9993, -0.9993, 0.1),
                (0.462, 0.462, 0.1),
            ],
        ),
    ]);

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();
    write_records_to_parquet(file_path, &records).unwrap();

    // Compute gradient flow stats
    let stats = compute_gradient_flow_stats(file_path, &creature).unwrap();

    // hidden-1 should have high saturation ratio (4/5 = 80%)
    let hidden_stats = stats.get("hidden-1").expect("hidden-1 stats should exist");
    assert!(
        hidden_stats.saturation_ratio >= 0.7,
        "TANH hidden-1 should have saturation_ratio >= 0.7, got {}",
        hidden_stats.saturation_ratio
    );
}

/// Test that TANH neurons operating in linear region have low saturation.
#[test]
fn test_tanh_saturation_detection_low_saturation() {
    let creature = create_creature(
        vec![
            neuron("hidden-1", "hidden", "TANH"),
            neuron("output-0", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-0", "hidden-1", 1.0),
            synapse("hidden-1", "output-0", 1.0),
        ],
        1,
    );

    // Create records where hidden-1 operates in linear region (|value| < 1)
    let records = create_records_with_values(vec![
        (
            "hidden-1",
            vec![
                (0.1, 0.0997, 0.1),  // Linear region
                (-0.2, -0.197, 0.1), // Linear region
                (0.5, 0.462, 0.1),   // Linear region
                (-0.3, -0.291, 0.1), // Linear region
                (0.8, 0.664, 0.1),   // Slightly non-linear but not saturated
            ],
        ),
        (
            "output-0",
            vec![
                (0.0997, 0.0997, 0.1),
                (-0.197, -0.197, 0.1),
                (0.462, 0.462, 0.1),
                (-0.291, -0.291, 0.1),
                (0.664, 0.664, 0.1),
            ],
        ),
    ]);

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();
    write_records_to_parquet(file_path, &records).unwrap();

    let stats = compute_gradient_flow_stats(file_path, &creature).unwrap();

    let hidden_stats = stats.get("hidden-1").expect("hidden-1 stats should exist");
    assert!(
        hidden_stats.saturation_ratio <= 0.2,
        "TANH hidden-1 in linear region should have saturation_ratio <= 0.2, got {}",
        hidden_stats.saturation_ratio
    );
}

// =============================================================================
// Test: Dead Neuron Detection for ReLU
// =============================================================================

/// Test that `ReLU` neurons with mostly negative inputs are detected as "dead".
///
/// A dead `ReLU`:
/// - Has output = 0 for all negative inputs
/// - Has gradient = 0 for negative inputs (cannot learn)
/// - The `dead_ratio` represents % of samples with zero gradient
#[test]
fn test_relu_dead_neuron_detection_mostly_dead() {
    let creature = create_creature(
        vec![
            neuron("hidden-1", "hidden", "RELU"),
            neuron("output-0", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-0", "hidden-1", 1.0),
            synapse("hidden-1", "output-0", 1.0),
        ],
        1,
    );

    // Create records where hidden-1 has mostly negative inputs (dead zone)
    // ReLU(x) = max(0, x), so negative inputs produce 0 output and 0 gradient
    let records = create_records_with_values(vec![
        (
            "hidden-1",
            vec![
                (-5.0, 0.0, 0.1), // Dead (negative input)
                (-2.0, 0.0, 0.1), // Dead
                (-1.0, 0.0, 0.1), // Dead
                (-0.5, 0.0, 0.1), // Dead
                (0.5, 0.5, 0.1),  // Active
            ],
        ),
        (
            "output-0",
            vec![
                (0.0, 0.0, 0.1),
                (0.0, 0.0, 0.1),
                (0.0, 0.0, 0.1),
                (0.0, 0.0, 0.1),
                (0.5, 0.5, 0.1),
            ],
        ),
    ]);

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();
    write_records_to_parquet(file_path, &records).unwrap();

    let stats = compute_gradient_flow_stats(file_path, &creature).unwrap();

    let hidden_stats = stats.get("hidden-1").expect("hidden-1 stats should exist");
    assert!(
        hidden_stats.dead_ratio >= 0.7,
        "ReLU hidden-1 with mostly negative inputs should have dead_ratio >= 0.7, got {}",
        hidden_stats.dead_ratio
    );
}

/// Test that active `ReLU` neurons have low dead ratio.
#[test]
fn test_relu_active_neuron_detection() {
    let creature = create_creature(
        vec![
            neuron("hidden-1", "hidden", "RELU"),
            neuron("output-0", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-0", "hidden-1", 1.0),
            synapse("hidden-1", "output-0", 1.0),
        ],
        1,
    );

    // Create records where hidden-1 has mostly positive inputs (active)
    let records = create_records_with_values(vec![
        (
            "hidden-1",
            vec![
                (1.0, 1.0, 0.1),  // Active
                (2.0, 2.0, 0.1),  // Active
                (0.5, 0.5, 0.1),  // Active
                (3.0, 3.0, 0.1),  // Active
                (-0.1, 0.0, 0.1), // Dead (but only one sample)
            ],
        ),
        (
            "output-0",
            vec![
                (1.0, 1.0, 0.1),
                (2.0, 2.0, 0.1),
                (0.5, 0.5, 0.1),
                (3.0, 3.0, 0.1),
                (0.0, 0.0, 0.1),
            ],
        ),
    ]);

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();
    write_records_to_parquet(file_path, &records).unwrap();

    let stats = compute_gradient_flow_stats(file_path, &creature).unwrap();

    let hidden_stats = stats.get("hidden-1").expect("hidden-1 stats should exist");
    assert!(
        hidden_stats.dead_ratio <= 0.3,
        "ReLU hidden-1 with mostly positive inputs should have dead_ratio <= 0.3, got {}",
        hidden_stats.dead_ratio
    );
}

// =============================================================================
// Test: LeakyReLU Never Dead
// =============================================================================

/// Test that `LeakyReLU` neurons are never considered "dead" since they always have non-zero gradient.
#[test]
fn test_leaky_relu_never_dead() {
    let creature = create_creature(
        vec![
            neuron("hidden-1", "hidden", "LEAKYRELU"),
            neuron("output-0", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-0", "hidden-1", 1.0),
            synapse("hidden-1", "output-0", 1.0),
        ],
        1,
    );

    // Create records with negative inputs - LeakyReLU still has gradient (0.01)
    let records = create_records_with_values(vec![
        (
            "hidden-1",
            vec![
                (-5.0, -0.05, 0.1), // Leaky output
                (-2.0, -0.02, 0.1), // Leaky output
                (-1.0, -0.01, 0.1), // Leaky output
            ],
        ),
        (
            "output-0",
            vec![
                (-0.05, -0.05, 0.1),
                (-0.02, -0.02, 0.1),
                (-0.01, -0.01, 0.1),
            ],
        ),
    ]);

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();
    write_records_to_parquet(file_path, &records).unwrap();

    let stats = compute_gradient_flow_stats(file_path, &creature).unwrap();

    let hidden_stats = stats.get("hidden-1").expect("hidden-1 stats should exist");
    assert!(
        hidden_stats.dead_ratio == 0.0,
        "LeakyReLU should never have dead neurons, got dead_ratio {}",
        hidden_stats.dead_ratio
    );
}

// =============================================================================
// Test: Gradient Magnitude Calculation
// =============================================================================

/// Test that gradient magnitude is correctly computed from error and activation gradient.
///
/// For a target neuron with squash function f, the gradient magnitude through
/// a neuron should reflect |error × f'(value)|.
#[test]
fn test_gradient_magnitude_calculation() {
    let creature = create_creature(
        vec![
            neuron("hidden-1", "hidden", "TANH"),
            neuron("hidden-2", "hidden", "TANH"),
            neuron("output-0", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-0", "hidden-1", 1.0),
            synapse("input-0", "hidden-2", 1.0),
            synapse("hidden-1", "output-0", 1.0),
            synapse("hidden-2", "output-0", 1.0),
        ],
        1,
    );

    // hidden-1: operates in linear region (high gradient)
    // hidden-2: operates in saturated region (low gradient)
    let records = create_records_with_values(vec![
        (
            "hidden-1",
            vec![
                (0.5, 0.462, 0.5), // TANH'(0.5) ≈ 0.786
                (0.3, 0.291, 0.5), // TANH'(0.3) ≈ 0.915
            ],
        ),
        (
            "hidden-2",
            vec![
                (5.0, 0.9999, 0.5), // TANH'(5.0) ≈ 0.0 (saturated)
                (4.0, 0.9993, 0.5), // TANH'(4.0) ≈ 0.0007
            ],
        ),
        ("output-0", vec![(1.461, 1.461, 0.5), (1.290, 1.290, 0.5)]),
    ]);

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();
    write_records_to_parquet(file_path, &records).unwrap();

    let stats = compute_gradient_flow_stats(file_path, &creature).unwrap();

    let hidden1_stats = stats.get("hidden-1").expect("hidden-1 stats should exist");
    let hidden2_stats = stats.get("hidden-2").expect("hidden-2 stats should exist");

    // hidden-1 should have higher gradient magnitude than hidden-2
    assert!(
        hidden1_stats.avg_gradient_magnitude > hidden2_stats.avg_gradient_magnitude,
        "hidden-1 (linear region) should have higher gradient than hidden-2 (saturated). \
         hidden-1: {}, hidden-2: {}",
        hidden1_stats.avg_gradient_magnitude,
        hidden2_stats.avg_gradient_magnitude
    );
}

// =============================================================================
// Test: LOGISTIC Saturation Detection
// =============================================================================

/// Test saturation detection for LOGISTIC activation function.
///
/// LOGISTIC(x) = 1 / (1 + exp(-x))
/// - At x > 5 or x < -5: output approaches 0 or 1, gradient approaches 0
#[test]
fn test_logistic_saturation_detection() {
    let creature = create_creature(
        vec![
            neuron("hidden-1", "hidden", "LOGISTIC"),
            neuron("output-0", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-0", "hidden-1", 1.0),
            synapse("hidden-1", "output-0", 1.0),
        ],
        1,
    );

    // Create records where hidden-1 is saturated (|value| > 5)
    let records = create_records_with_values(vec![
        (
            "hidden-1",
            vec![
                (10.0, 0.99995, 0.1),  // Saturated high
                (-10.0, 0.00005, 0.1), // Saturated low
                (6.0, 0.9975, 0.1),    // Saturated
                (-6.0, 0.0025, 0.1),   // Saturated
                (0.0, 0.5, 0.1),       // Not saturated (linear region)
            ],
        ),
        (
            "output-0",
            vec![
                (0.99995, 0.99995, 0.1),
                (0.00005, 0.00005, 0.1),
                (0.9975, 0.9975, 0.1),
                (0.0025, 0.0025, 0.1),
                (0.5, 0.5, 0.1),
            ],
        ),
    ]);

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();
    write_records_to_parquet(file_path, &records).unwrap();

    let stats = compute_gradient_flow_stats(file_path, &creature).unwrap();

    let hidden_stats = stats.get("hidden-1").expect("hidden-1 stats should exist");
    assert!(
        hidden_stats.saturation_ratio >= 0.7,
        "LOGISTIC hidden-1 should have saturation_ratio >= 0.7, got {}",
        hidden_stats.saturation_ratio
    );
}

// =============================================================================
// Test: Integration with Focus Neuron Ranking
// =============================================================================

/// Test that gradient flow stats are incorporated into focus neuron ranking.
///
/// Neurons with high error × impact but stuck in saturation should be
/// de-prioritised compared to neurons that can actually learn.
#[test]
fn test_gradient_flow_integration_with_ranking() {
    let creature = create_creature(
        vec![
            neuron("hidden-saturated", "hidden", "TANH"),
            neuron("hidden-active", "hidden", "TANH"),
            neuron("output-0", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-0", "hidden-saturated", 1.0),
            synapse("input-0", "hidden-active", 1.0),
            synapse("hidden-saturated", "output-0", 1.0),
            synapse("hidden-active", "output-0", 1.0),
        ],
        1,
    );

    // Both neurons have same error and impact, but:
    // - hidden-saturated is in saturation (cannot learn)
    // - hidden-active is in linear region (can learn)
    let records = create_records_with_values(vec![
        (
            "hidden-saturated",
            vec![
                (5.0, 0.9999, 0.5),  // Saturated
                (6.0, 0.99998, 0.5), // Saturated
                (4.0, 0.9993, 0.5),  // Saturated
                (5.5, 0.99995, 0.5), // Saturated
                (7.0, 0.99999, 0.5), // Saturated
            ],
        ),
        (
            "hidden-active",
            vec![
                (0.3, 0.291, 0.5),   // Active (gradient ~0.92)
                (0.5, 0.462, 0.5),   // Active (gradient ~0.79)
                (-0.2, -0.197, 0.5), // Active
                (0.1, 0.0997, 0.5),  // Active
                (-0.4, -0.38, 0.5),  // Active
            ],
        ),
        (
            "output-0",
            vec![
                (1.29, 1.29, 0.5),
                (1.46, 1.46, 0.5),
                (0.79, 0.79, 0.5),
                (1.10, 1.10, 0.5),
                (0.62, 0.62, 0.5),
            ],
        ),
    ]);

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();
    write_records_to_parquet(file_path, &records).unwrap();

    // Rank with gradient flow analysis enabled
    let result = rank_focus_neurons(file_path, &creature, None, None).unwrap();

    // Find positions of both neurons
    let saturated_pos = result
        .neurons
        .iter()
        .position(|n| n.neuron_uuid == "hidden-saturated");
    let active_pos = result
        .neurons
        .iter()
        .position(|n| n.neuron_uuid == "hidden-active");

    assert!(
        saturated_pos.is_some() && active_pos.is_some(),
        "Both hidden neurons should be in ranking"
    );

    // hidden-active should rank HIGHER (lower position index) than hidden-saturated
    // because it has higher learning potential (not stuck in saturation)
    assert!(
        active_pos.unwrap() < saturated_pos.unwrap(),
        "hidden-active (rank {}) should rank higher than hidden-saturated (rank {})",
        active_pos.unwrap(),
        saturated_pos.unwrap()
    );
}

/// Test that dead `ReLU` neurons are de-prioritised in ranking.
#[test]
fn test_dead_relu_deprioritised_in_ranking() {
    let creature = create_creature(
        vec![
            neuron("hidden-dead", "hidden", "RELU"),
            neuron("hidden-active", "hidden", "RELU"),
            neuron("output-0", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-0", "hidden-dead", 1.0),
            synapse("input-0", "hidden-active", 1.0),
            synapse("hidden-dead", "output-0", 1.0),
            synapse("hidden-active", "output-0", 1.0),
        ],
        1,
    );

    // Both have same structural impact, but:
    // - hidden-dead always receives negative inputs (dead)
    // - hidden-active receives positive inputs (active)
    let records = create_records_with_values(vec![
        (
            "hidden-dead",
            vec![
                (-2.0, 0.0, 0.5), // Dead
                (-1.0, 0.0, 0.5), // Dead
                (-3.0, 0.0, 0.5), // Dead
                (-0.5, 0.0, 0.5), // Dead
                (-1.5, 0.0, 0.5), // Dead
            ],
        ),
        (
            "hidden-active",
            vec![
                (1.0, 1.0, 0.5), // Active
                (2.0, 2.0, 0.5), // Active
                (0.5, 0.5, 0.5), // Active
                (1.5, 1.5, 0.5), // Active
                (3.0, 3.0, 0.5), // Active
            ],
        ),
        (
            "output-0",
            vec![
                (1.0, 1.0, 0.5),
                (2.0, 2.0, 0.5),
                (0.5, 0.5, 0.5),
                (1.5, 1.5, 0.5),
                (3.0, 3.0, 0.5),
            ],
        ),
    ]);

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();
    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None, None).unwrap();

    let dead_pos = result
        .neurons
        .iter()
        .position(|n| n.neuron_uuid == "hidden-dead");
    let active_pos = result
        .neurons
        .iter()
        .position(|n| n.neuron_uuid == "hidden-active");

    assert!(
        dead_pos.is_some() && active_pos.is_some(),
        "Both hidden neurons should be in ranking"
    );

    // hidden-active should rank higher than hidden-dead
    assert!(
        active_pos.unwrap() < dead_pos.unwrap(),
        "hidden-active (rank {}) should rank higher than hidden-dead (rank {})",
        active_pos.unwrap(),
        dead_pos.unwrap()
    );
}

// =============================================================================
// Test: GradientFlowStats struct fields
// =============================================================================

/// Test that `GradientFlowStats` contains the expected fields.
#[test]
fn test_gradient_flow_stats_struct_fields() {
    let stats = GradientFlowStats {
        avg_gradient_magnitude: 0.5,
        saturation_ratio: 0.2,
        dead_ratio: 0.0,
    };

    assert!((stats.avg_gradient_magnitude - 0.5).abs() < f32::EPSILON);
    assert!((stats.saturation_ratio - 0.2).abs() < f32::EPSILON);
    assert!((stats.dead_ratio - 0.0).abs() < f32::EPSILON);
}

// =============================================================================
// Test: Edge Cases
// =============================================================================

/// Test gradient flow with IDENTITY activation (gradient always 1.0).
#[test]
fn test_identity_always_full_gradient() {
    let creature = create_creature(
        vec![
            neuron("hidden-1", "hidden", "IDENTITY"),
            neuron("output-0", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-0", "hidden-1", 1.0),
            synapse("hidden-1", "output-0", 1.0),
        ],
        1,
    );

    let records = create_records_with_values(vec![
        (
            "hidden-1",
            vec![
                (100.0, 100.0, 0.1), // IDENTITY passes through
                (-100.0, -100.0, 0.1),
                (0.0, 0.0, 0.1),
            ],
        ),
        (
            "output-0",
            vec![(100.0, 100.0, 0.1), (-100.0, -100.0, 0.1), (0.0, 0.0, 0.1)],
        ),
    ]);

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();
    write_records_to_parquet(file_path, &records).unwrap();

    let stats = compute_gradient_flow_stats(file_path, &creature).unwrap();

    let hidden_stats = stats.get("hidden-1").expect("hidden-1 stats should exist");

    // IDENTITY: gradient = 1.0, never saturated, never dead
    assert!(
        hidden_stats.saturation_ratio == 0.0,
        "IDENTITY should never be saturated, got {}",
        hidden_stats.saturation_ratio
    );
    assert!(
        hidden_stats.dead_ratio == 0.0,
        "IDENTITY should never be dead, got {}",
        hidden_stats.dead_ratio
    );
}

/// Test gradient flow with aggregate neurons (MINIMUM/MAXIMUM/IF).
///
/// Aggregate neurons cannot compute scalar gradients - they should
/// be handled gracefully with default values.
#[test]
fn test_aggregate_neurons_handled_gracefully() {
    let creature = create_creature(
        vec![
            neuron("hidden-1", "hidden", "MINIMUM"),
            neuron("output-0", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-0", "hidden-1", 1.0),
            synapse("input-1", "hidden-1", 1.0),
            synapse("hidden-1", "output-0", 1.0),
        ],
        2,
    );

    let records = create_records_with_values(vec![
        (
            "hidden-1",
            vec![
                (0.5, 0.3, 0.1), // MINIMUM selects one input
                (0.8, 0.2, 0.1),
            ],
        ),
        ("output-0", vec![(0.3, 0.3, 0.1), (0.2, 0.2, 0.1)]),
    ]);

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();
    write_records_to_parquet(file_path, &records).unwrap();

    let stats = compute_gradient_flow_stats(file_path, &creature).unwrap();

    // Aggregate neurons should have default/neutral gradient stats
    // (neither saturated nor dead - they just pass through winning input)
    let hidden_stats = stats.get("hidden-1").expect("hidden-1 stats should exist");
    assert!(
        hidden_stats.saturation_ratio <= 0.1,
        "MINIMUM should have low saturation, got {}",
        hidden_stats.saturation_ratio
    );
}
