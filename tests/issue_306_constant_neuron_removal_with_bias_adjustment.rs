//! Issue #306: Constant-value neurons should be removal candidates with bias adjustments.
//!
//! When a hidden neuron has near-constant activation (very low variance), removing it
//! is equivalent to adjusting the biases of all downstream neurons by:
//!   bias_adjustment = synapse_weight * mean_activation
//!
//! This test verifies that such neurons are identified and returned as coordinated
//! structural candidates containing:
//! - RemoveNeuron operation for the constant neuron
//! - SetBias operations for all downstream neurons with adjusted biases

mod common;

use neat_ai_discovery::focus::rank_focus_neurons;
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CoordinatedStructuralOpJson, CreatureJson, NeuronJson, SynapseJson};
use tempfile::NamedTempFile;

/// Helper to create a simple creature with specified neurons and synapses.
fn create_creature(
    neurons: Vec<(&str, &str, f32)>,  // (uuid, type, bias)
    synapses: Vec<(&str, &str, f32)>, // (from, to, weight)
) -> CreatureJson {
    let input_count = neurons.iter().filter(|(_, t, _)| *t == "input").count();
    let output_count = neurons.iter().filter(|(_, t, _)| *t == "output").count();
    CreatureJson {
        neurons: neurons
            .into_iter()
            .filter(|(_, neuron_type, _)| *neuron_type != "input")
            .map(|(uuid, neuron_type, bias)| NeuronJson {
                uuid: uuid.to_string(),
                neuron_type: neuron_type.to_string(),
                squash: "IDENTITY".to_string(),
                bias,
            })
            .collect(),
        synapses: synapses
            .into_iter()
            .map(|(from, to, weight)| SynapseJson {
                from_uuid: from.to_string(),
                to_uuid: to.to_string(),
                weight,
                synapse_type: None,
            })
            .collect(),
        input: input_count,
        output: output_count,
    }
}

/// Helper to create parquet records for neurons with specified errors, activations,
/// and optional variance in the activation.
fn create_records_with_variance(
    neuron_data: Vec<(&str, f32, f32, f32)>, // (uuid, error, mean_activation, variance)
    sample_count: u32,
) -> Vec<DiscoverRecord> {
    let mut records = Vec::new();
    for (uuid, error, mean_activation, variance) in neuron_data {
        let std_dev = variance.sqrt();
        for i in 0..sample_count {
            // Create varying activations based on variance
            // Use a simple pattern: alternate above and below mean
            let activation = if std_dev > 0.0 {
                if i % 2 == 0 {
                    mean_activation + std_dev
                } else {
                    mean_activation - std_dev
                }
            } else {
                mean_activation // Constant activation when variance is 0
            };
            records.push(DiscoverRecord::new(
                i,
                uuid.to_string(),
                Some(activation),
                activation,
                vec![error],
            ));
        }
    }
    records
}

/// Issue #306: A hidden neuron with constant (zero variance) activation should be
/// identified as a candidate for removal with bias adjustments.
///
/// Network:
///   input-0 → constant-hidden (always outputs 1.0) → output-0
///
/// When constant-hidden is removed, output-0's bias should be adjusted by:
///   bias_adjustment = synapse_weight × mean_activation = 0.5 × 1.0 = 0.5
#[test]
fn issue_306_constant_neuron_creates_removal_candidate_with_bias_adjustment() {
    // Create a creature with a hidden neuron that has constant activation
    let creature = create_creature(
        vec![
            ("input-0", "input", 0.0),
            ("constant-hidden", "hidden", 0.0),
            ("output-0", "output", 0.1), // Initial bias
        ],
        vec![
            ("input-0", "constant-hidden", 1.0),
            ("constant-hidden", "output-0", 0.5), // This weight becomes bias adjustment
        ],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // Create records where constant-hidden has ZERO variance (always 1.0)
    // and some error in the output neuron
    let records = create_records_with_variance(
        vec![
            // (uuid, error, mean_activation, variance)
            ("constant-hidden", 0.1, 1.0, 0.0), // Zero variance = constant
            ("output-0", 0.3, 0.5, 0.01),       // Normal variance
        ],
        32,
    );
    write_records_to_parquet(file_path, &records).unwrap();

    // Use a reasonable cost_of_growth to allow removal candidates
    let result = rank_focus_neurons(file_path, &creature, None, Some(0.01)).unwrap();

    // The constant-hidden neuron should be identified as a candidate
    // Check if we have coordinated structural candidates with bias adjustments
    assert!(
        !result.constant_neuron_removals.is_empty(),
        "Expected constant_neuron_removals to be present. \
         Removal candidates: {:?}",
        result
            .removal_candidates
            .iter()
            .map(|c| &c.neuron_uuid)
            .collect::<Vec<_>>()
    );

    let constant_removals = &result.constant_neuron_removals;
    assert!(
        !constant_removals.is_empty(),
        "Expected at least one constant neuron removal candidate"
    );

    // Find the candidate for constant-hidden
    let candidate = constant_removals
        .iter()
        .find(|c| {
            c.operations.iter().any(|op| matches!(
                op,
                CoordinatedStructuralOpJson::RemoveNeuron { neuron_uuid } if neuron_uuid == "constant-hidden"
            ))
        })
        .expect("Should have a removal candidate for constant-hidden");

    // Verify it contains both RemoveNeuron and SetBias operations
    let has_remove = candidate.operations.iter().any(|op| {
        matches!(
            op,
            CoordinatedStructuralOpJson::RemoveNeuron { neuron_uuid } if neuron_uuid == "constant-hidden"
        )
    });
    assert!(
        has_remove,
        "Candidate should include RemoveNeuron operation"
    );

    // Check the SetBias operation
    let set_bias_op = candidate.operations.iter().find_map(|op| {
        if let CoordinatedStructuralOpJson::SetBias { neuron_uuid, bias } = op {
            if neuron_uuid == "output-0" {
                Some(*bias)
            } else {
                None
            }
        } else {
            None
        }
    });
    assert!(
        set_bias_op.is_some(),
        "Candidate should include SetBias operation for output-0"
    );

    let new_bias = set_bias_op.unwrap();
    // Expected: old_bias (0.1) + weight (0.5) × mean_activation (1.0) = 0.6
    let expected_bias = 0.1 + 0.5 * 1.0;
    assert!(
        (new_bias - expected_bias).abs() < 0.01,
        "New bias should be {expected_bias:.3}, got {new_bias:.3}"
    );
}

/// Issue #306: Verify that neurons with non-zero variance are NOT treated as constant.
#[test]
fn issue_306_varying_neuron_not_treated_as_constant() {
    let creature = create_creature(
        vec![
            ("input-0", "input", 0.0),
            ("varying-hidden", "hidden", 0.0),
            ("output-0", "output", 0.0),
        ],
        vec![
            ("input-0", "varying-hidden", 1.0),
            ("varying-hidden", "output-0", 0.5),
        ],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // Create records where varying-hidden has SIGNIFICANT variance
    let records = create_records_with_variance(
        vec![
            ("varying-hidden", 0.1, 0.5, 0.25), // std_dev = 0.5, significant variance
            ("output-0", 0.3, 0.5, 0.01),
        ],
        32,
    );
    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None, Some(0.01)).unwrap();

    // varying-hidden should NOT be in constant_neuron_removals
    let has_varying = result.constant_neuron_removals.iter().any(|c| {
        c.operations.iter().any(|op| {
            matches!(
                op,
                CoordinatedStructuralOpJson::RemoveNeuron { neuron_uuid } if neuron_uuid == "varying-hidden"
            )
        })
    });
    assert!(
        !has_varying,
        "varying-hidden should NOT be in constant_neuron_removals"
    );
}

/// Issue #306: Multiple downstream neurons should all get bias adjustments.
#[test]
fn issue_306_constant_neuron_adjusts_multiple_downstream_biases() {
    // Network: constant-hidden feeds into both output-0 and output-1
    let creature = create_creature(
        vec![
            ("input-0", "input", 0.0),
            ("constant-hidden", "hidden", 0.0),
            ("output-0", "output", 0.1),
            ("output-1", "output", 0.2),
        ],
        vec![
            ("input-0", "constant-hidden", 1.0),
            ("constant-hidden", "output-0", 0.3), // bias adjustment = 0.3 × 2.0 = 0.6
            ("constant-hidden", "output-1", 0.4), // bias adjustment = 0.4 × 2.0 = 0.8
        ],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    let records = create_records_with_variance(
        vec![
            ("constant-hidden", 0.1, 2.0, 0.0), // Zero variance, mean = 2.0
            ("output-0", 0.3, 0.5, 0.01),
            ("output-1", 0.3, 0.5, 0.01),
        ],
        32,
    );
    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None, Some(0.01)).unwrap();

    // Should have constant neuron removal with bias adjustments for both outputs
    assert!(
        !result.constant_neuron_removals.is_empty(),
        "Expected constant_neuron_removals"
    );

    let candidate = result.constant_neuron_removals
        .iter()
        .find(|c| {
            c.operations.iter().any(|op| matches!(
                op,
                CoordinatedStructuralOpJson::RemoveNeuron { neuron_uuid } if neuron_uuid == "constant-hidden"
            ))
        })
        .expect("Should have removal candidate for constant-hidden");

    // Count SetBias operations
    let set_bias_count = candidate
        .operations
        .iter()
        .filter(|op| matches!(op, CoordinatedStructuralOpJson::SetBias { .. }))
        .count();

    assert_eq!(
        set_bias_count, 2,
        "Should have 2 SetBias operations (one for each downstream neuron)"
    );

    // Verify output-0 bias adjustment
    let output_0_bias = candidate.operations.iter().find_map(|op| {
        if let CoordinatedStructuralOpJson::SetBias { neuron_uuid, bias } = op {
            if neuron_uuid == "output-0" {
                Some(*bias)
            } else {
                None
            }
        } else {
            None
        }
    });
    let expected_0 = 0.1 + 0.3 * 2.0; // 0.7
    assert!(
        (output_0_bias.unwrap() - expected_0).abs() < 0.01,
        "output-0 bias should be {expected_0:.3}"
    );

    // Verify output-1 bias adjustment
    let output_1_bias = candidate.operations.iter().find_map(|op| {
        if let CoordinatedStructuralOpJson::SetBias { neuron_uuid, bias } = op {
            if neuron_uuid == "output-1" {
                Some(*bias)
            } else {
                None
            }
        } else {
            None
        }
    });
    let expected_1 = 0.2 + 0.4 * 2.0; // 1.0
    assert!(
        (output_1_bias.unwrap() - expected_1).abs() < 0.01,
        "output-1 bias should be {expected_1:.3}"
    );
}

/// Issue #306: Very low (but non-zero) variance should still be treated as constant.
/// Uses the same threshold as Issue #217 proposes: 1e-10 for "dead" neurons.
#[test]
fn issue_306_near_zero_variance_treated_as_constant() {
    let creature = create_creature(
        vec![
            ("input-0", "input", 0.0),
            ("near-constant", "hidden", 0.0),
            ("output-0", "output", 0.0),
        ],
        vec![
            ("input-0", "near-constant", 1.0),
            ("near-constant", "output-0", 0.5),
        ],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // Create records with very small variance (below threshold)
    let records = create_records_with_variance(
        vec![
            ("near-constant", 0.1, 1.0, 1e-12), // Variance below 1e-10 threshold
            ("output-0", 0.3, 0.5, 0.01),
        ],
        32,
    );
    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None, Some(0.01)).unwrap();

    // near-constant should be in constant_neuron_removals
    assert!(
        !result.constant_neuron_removals.is_empty(),
        "Expected constant_neuron_removals to be present for near-zero variance"
    );

    let has_near_constant = result.constant_neuron_removals.iter().any(|c| {
        c.operations.iter().any(|op| {
            matches!(
                op,
                CoordinatedStructuralOpJson::RemoveNeuron { neuron_uuid } if neuron_uuid == "near-constant"
            )
        })
    });
    assert!(
        has_near_constant,
        "near-constant neuron should be in constant_neuron_removals"
    );
}
