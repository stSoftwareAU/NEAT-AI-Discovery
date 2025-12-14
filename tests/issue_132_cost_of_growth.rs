//! Issue #132: costOfGrowth should be passed from NEAT-AI, not hardcoded.
//!
//! The correct cost of growth is 1e-7 (per Score.ts formula).
//! v0.1.145 incorrectly changed this to 0.01, causing 418 false removal candidates.
//! This test verifies that the costOfGrowth parameter is correctly passed
//! from the input and used in removal candidate filtering.

mod common;

use neat_ai_discovery::focus::rank_focus_neurons;
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
use tempfile::NamedTempFile;

/// Helper to create a simple creature with specified neurons and synapses
fn create_creature(
    neurons: Vec<(&str, &str)>,       // (uuid, type)
    synapses: Vec<(&str, &str, f32)>, // (from, to, weight)
) -> CreatureJson {
    let input_count = neurons.iter().filter(|(_, t)| *t == "input").count();
    let output_count = neurons.iter().filter(|(_, t)| *t == "output").count();
    CreatureJson {
        neurons: neurons
            .into_iter()
            .filter(|(_, neuron_type)| *neuron_type != "input")
            .map(|(uuid, neuron_type)| NeuronJson {
                uuid: uuid.to_string(),
                neuron_type: neuron_type.to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
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

/// Helper to create parquet records for neurons with specified errors AND activations.
fn create_records_with_activation(
    neuron_data: Vec<(&str, f32, f32)>, // (uuid, error, activation)
) -> Vec<DiscoverRecord> {
    neuron_data
        .into_iter()
        .flat_map(|(uuid, error, activation)| {
            vec![
                DiscoverRecord::new(0, uuid.to_string(), Some(0.5), activation, vec![error]),
                DiscoverRecord::new(1, uuid.to_string(), Some(0.5), activation, vec![error]),
            ]
        })
        .collect()
}

/// Issue #132: costOfGrowth should be passable from NEAT-AI.
///
/// When a neuron has activation_weighted_impact between 1e-7 and 0.01:
/// - With costOfGrowth = 0.01 (old default): IS a removal candidate
/// - With costOfGrowth = 1e-7 (typical NEAT-AI value): NOT a removal candidate
#[test]
fn test_cost_of_growth_parameter_affects_removal_candidates() {
    // Create a network where medium-impact neuron has a competing path to output.
    // This gives normalised structural impact < 1.0.
    //
    // Impact calculation (normalised):
    // - medium-impact → output-0: weight 0.001 / total_inbound(1.001) × 1.0 ≈ 0.001
    // - direct → output-0: weight 1.0 / total_inbound(1.001) × 1.0 ≈ 0.999
    //
    // With mean_activation = 0.01:
    // - activation_weighted_impact = 0.001 × 0.01 = 1e-5
    //
    // This is between 1e-7 and 0.01, perfect for testing threshold behaviour.
    let creature = create_creature(
        vec![
            ("input-0", "input"),
            ("input-1", "input"), // Direct input to output
            ("medium-impact", "hidden"),
            ("output-0", "output"),
        ],
        vec![
            ("input-0", "medium-impact", 1.0),
            // Small weight relative to the direct path
            ("medium-impact", "output-0", 0.001),
            // Direct path with large weight dominates
            ("input-1", "output-0", 1.0),
        ],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    let records = create_records_with_activation(vec![
        ("medium-impact", 0.1, 0.01), // activation_weighted_impact ≈ 1e-5
        ("output-0", 0.5, 0.5),
    ]);
    write_records_to_parquet(file_path, &records).unwrap();

    // First, verify the activation_weighted_impact is in the expected range
    let result_check = rank_focus_neurons(file_path, &creature, None, Some(1.0)).unwrap();
    let medium = result_check
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "medium-impact")
        .expect("medium-impact should be in results");

    // Log the actual values for debugging
    eprintln!(
        "medium-impact: structural_impact={:.6e}, mean_activation={:.6e}, activation_weighted_impact={:.6e}",
        medium.impact, medium.mean_activation, medium.activation_weighted_impact
    );

    // The activation_weighted_impact should be between 1e-7 and 0.01
    assert!(
        medium.activation_weighted_impact > 1e-7 && medium.activation_weighted_impact < 0.01,
        "activation_weighted_impact ({:.2e}) should be between 1e-7 and 0.01 for this test to be valid",
        medium.activation_weighted_impact
    );

    // Test with high costOfGrowth (0.01 - the old hardcoded default)
    // Neuron with impact between 1e-7 and 0.01 should be a removal candidate
    let result_high_cost = rank_focus_neurons(file_path, &creature, None, Some(0.01)).unwrap();

    let has_removal_high = result_high_cost
        .removal_candidates
        .iter()
        .any(|c| c.neuron_uuid == "medium-impact");
    assert!(
        has_removal_high,
        "With costOfGrowth=0.01, neuron with impact {:.2e} SHOULD be a removal candidate. \
         Found candidates: {:?}",
        medium.activation_weighted_impact,
        result_high_cost
            .removal_candidates
            .iter()
            .map(|c| (&c.neuron_uuid, c.activation_weighted_impact))
            .collect::<Vec<_>>()
    );

    // Test with low costOfGrowth (1e-7 - typical NEAT-AI value)
    // Neuron with impact > 1e-7 should NOT be a removal candidate
    let result_low_cost = rank_focus_neurons(file_path, &creature, None, Some(1e-7)).unwrap();

    let has_removal_low = result_low_cost
        .removal_candidates
        .iter()
        .any(|c| c.neuron_uuid == "medium-impact");
    assert!(
        !has_removal_low,
        "With costOfGrowth=1e-7, neuron with impact {:.2e} should NOT be a removal candidate \
         (impact > threshold 1e-7). Found candidates: {:?}",
        medium.activation_weighted_impact,
        result_low_cost
            .removal_candidates
            .iter()
            .map(|c| (&c.neuron_uuid, c.activation_weighted_impact))
            .collect::<Vec<_>>()
    );
}

/// Issue #132: The reason message should show the passed costOfGrowth value.
#[test]
fn test_removal_candidate_reason_shows_passed_cost_of_growth() {
    // Create a neuron with very low impact that will definitely be a removal candidate.
    // Use a competing direct path to ensure normalised impact is low.
    let creature = create_creature(
        vec![
            ("input-0", "input"),
            ("input-1", "input"),
            ("low-impact", "hidden"),
            ("output-0", "output"),
        ],
        vec![
            ("input-0", "low-impact", 1.0),
            // Very small weight compared to direct path
            ("low-impact", "output-0", 1e-10),
            // Direct path dominates
            ("input-1", "output-0", 1.0),
        ],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    let records = create_records_with_activation(vec![
        ("low-impact", 0.1, 0.1), // activation_weighted_impact = structural_impact × 0.1
        ("output-0", 0.5, 0.5),
    ]);
    write_records_to_parquet(file_path, &records).unwrap();

    // Use a high costOfGrowth to ensure the neuron IS a removal candidate
    let result = rank_focus_neurons(file_path, &creature, None, Some(0.01)).unwrap();

    let removal = result
        .removal_candidates
        .iter()
        .find(|c| c.neuron_uuid == "low-impact");
    assert!(
        removal.is_some(),
        "Should have a removal candidate. Found candidates: {:?}",
        result
            .removal_candidates
            .iter()
            .map(|c| (&c.neuron_uuid, c.activation_weighted_impact))
            .collect::<Vec<_>>()
    );

    let reason = &removal.unwrap().reason;
    // The reason should contain the costOfGrowth value (1e-02)
    assert!(
        reason.contains("1.00e-2") || reason.contains("1e-02") || reason.contains("0.01"),
        "Reason should mention the costOfGrowth. Got: {reason}",
    );
}

/// Issue #132: When costOfGrowth is not specified, use a sensible default.
#[test]
fn test_cost_of_growth_uses_default_when_not_specified() {
    let creature = create_creature(
        vec![
            ("input-0", "input"),
            ("low-impact", "hidden"),
            ("output-0", "output"),
        ],
        vec![
            ("input-0", "low-impact", 1.0),
            ("low-impact", "output-0", 1e-4),
        ],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    let records = create_records_with_activation(vec![
        ("low-impact", 0.1, 0.01), // activation_weighted_impact ≈ 1e-6
        ("output-0", 0.5, 0.5),
    ]);
    write_records_to_parquet(file_path, &records).unwrap();

    // Call with None for cost_of_growth - should use default
    // The default should be documented in the README
    let result = rank_focus_neurons(file_path, &creature, None, None).unwrap();

    // Just verify it works without error
    assert!(
        !result.neurons.is_empty(),
        "Should have processed neurons with default costOfGrowth"
    );
}

/// REGRESSION TEST Issue #132: Default costOfGrowth MUST be 1e-7, NOT 0.01
///
/// This test exists because v0.1.145 incorrectly changed the default from 1e-7 to 0.01,
/// which caused 418 false removal candidates in production. The previous tests didn't
/// catch this because they only tested the mechanism, not the correctness of the value.
///
/// NEAT-AI's Score.ts formula uses costOfGrowth = 1e-7 per hidden neuron.
/// If the default is wrong (e.g., 0.01), this test WILL FAIL because:
/// - With threshold 0.01: both neurons would be removal candidates (WRONG)
/// - With threshold 1e-7: only the truly negligible neuron is a candidate (CORRECT)
///
/// DO NOT change this test to match a different default. If this test fails,
/// the default costOfGrowth has been incorrectly changed.
#[test]
fn regression_default_cost_of_growth_must_be_1e7_not_001() {
    // Create two neurons with different impacts:
    // - "medium-impact": impact ≈ 1e-5 (between 1e-7 and 0.01)
    // - "negligible": impact ≈ 1e-9 (below 1e-7)
    //
    // If default is WRONG (0.01): BOTH would be removal candidates
    // If default is CORRECT (1e-7): only "negligible" is a candidate
    let creature = create_creature(
        vec![
            ("input-0", "input"),
            ("medium-impact", "hidden"), // Impact ~1e-5: should NOT be removal candidate
            ("negligible", "hidden"),    // Impact ~1e-9: SHOULD be removal candidate
            ("output-0", "output"),
        ],
        vec![
            // medium-impact: small but not negligible connection
            ("input-0", "medium-impact", 1.0),
            ("medium-impact", "output-0", 1e-4), // structural_impact ≈ 1e-4
            // negligible: truly tiny connection
            ("input-0", "negligible", 1.0),
            ("negligible", "output-0", 1e-8), // structural_impact ≈ 1e-8
            // Direct path to output (to dilute impacts)
            ("input-0", "output-0", 1.0),
        ],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // Activations chosen to give desired activation_weighted_impact values
    let records = create_records_with_activation(vec![
        ("medium-impact", 0.1, 0.1), // activation_weighted_impact ≈ 1e-5
        ("negligible", 0.1, 0.1),    // activation_weighted_impact ≈ 1e-9
        ("output-0", 0.5, 0.5),
    ]);
    write_records_to_parquet(file_path, &records).unwrap();

    // Use default costOfGrowth (None)
    let result = rank_focus_neurons(file_path, &creature, None, None).unwrap();

    // Find the neurons
    let medium = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "medium-impact")
        .expect("medium-impact should be in results");
    let negligible = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "negligible")
        .expect("negligible should be in results");

    println!(
        "medium-impact: activation_weighted_impact = {:.2e}",
        medium.activation_weighted_impact
    );
    println!(
        "negligible: activation_weighted_impact = {:.2e}",
        negligible.activation_weighted_impact
    );

    // CRITICAL ASSERTION 1: medium-impact should NOT be a removal candidate
    // If default is 0.01 (WRONG), this would be a candidate and test would fail
    let medium_is_candidate = result
        .removal_candidates
        .iter()
        .any(|c| c.neuron_uuid == "medium-impact");
    assert!(
        !medium_is_candidate,
        "REGRESSION: medium-impact (impact {:.2e}) should NOT be a removal candidate! \
         If this fails, the default costOfGrowth has been incorrectly changed from 1e-7. \
         Removal candidates: {:?}",
        medium.activation_weighted_impact,
        result
            .removal_candidates
            .iter()
            .map(|c| format!("{}:{:.2e}", c.neuron_uuid, c.activation_weighted_impact))
            .collect::<Vec<_>>()
    );

    // CRITICAL ASSERTION 2: negligible SHOULD be a removal candidate
    let negligible_is_candidate = result
        .removal_candidates
        .iter()
        .any(|c| c.neuron_uuid == "negligible");
    assert!(
        negligible_is_candidate,
        "negligible (impact {:.2e}) should be a removal candidate with default threshold 1e-7. \
         Removal candidates: {:?}",
        negligible.activation_weighted_impact,
        result
            .removal_candidates
            .iter()
            .map(|c| format!("{}:{:.2e}", c.neuron_uuid, c.activation_weighted_impact))
            .collect::<Vec<_>>()
    );

    // CRITICAL ASSERTION 3: Verify the impact values are in expected ranges
    // This ensures the test setup is valid
    assert!(
        medium.activation_weighted_impact > 1e-7,
        "Test setup error: medium-impact should have impact > 1e-7 (got {:.2e})",
        medium.activation_weighted_impact
    );
    assert!(
        medium.activation_weighted_impact < 0.01,
        "Test setup error: medium-impact should have impact < 0.01 (got {:.2e})",
        medium.activation_weighted_impact
    );
    assert!(
        negligible.activation_weighted_impact < 1e-7,
        "Test setup error: negligible should have impact < 1e-7 (got {:.2e})",
        negligible.activation_weighted_impact
    );
}
