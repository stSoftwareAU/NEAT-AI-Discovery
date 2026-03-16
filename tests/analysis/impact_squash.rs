//! Tests for squash-aware impact calculation.
//!
//! These tests verify that impact calculation correctly handles special
//! squash functions like STEP, BIPOLAR, MINIMUM, and MAXIMUM where the
//! standard linear model fails.
//!
//! See docs/IMPACT_CALCULATION.md for detailed explanation of the issues.

use neat_ai_discovery::focus::rank_focus_neurons;
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
use tempfile::NamedTempFile;

/// Helper to create a creature with specified neurons (including squash) and synapses
///
/// Note: Input neurons in the `neurons` parameter are used only to count `input`.
/// They are NOT included in `creature.neurons` as per the NEAT-AI data model -
/// input neurons are represented only by the `creature.input` count.
fn create_creature_with_squash(
    neurons: Vec<(&str, &str, &str)>, // (uuid, type, squash)
    synapses: Vec<(&str, &str, f32)>, // (from, to, weight)
) -> CreatureJson {
    let input_count = neurons.iter().filter(|(_, t, _)| *t == "input").count();
    let output_count = neurons.iter().filter(|(_, t, _)| *t == "output").count();
    CreatureJson {
        // Filter out input neurons - they are only represented by creature.input count
        neurons: neurons
            .into_iter()
            .filter(|(_, neuron_type, _)| *neuron_type != "input")
            .map(|(uuid, neuron_type, squash)| NeuronJson {
                uuid: uuid.to_string(),
                neuron_type: neuron_type.to_string(),
                squash: squash.to_string(),
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

/// Helper to create parquet records for neurons with specified errors and activations
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

// =============================================================================
// STEP/BIPOLAR Tests
// =============================================================================

/// Test that tiny weights into STEP neurons are not underestimated.
///
/// Scenario: A neuron with activation 0.000_001 connects via weight 0.000_002
/// to a STEP output neuron. The linear formula calculates impact ≈ 0, but if
/// the STEP neuron is near its threshold, this tiny signal could flip the output.
///
/// NOTE: This test currently documents the EXPECTED behaviour. Until we implement
/// squash-aware impact, it will show the current (incorrect) behaviour.
#[test]
fn test_step_neuron_tiny_weight_impact_not_underestimated() {
    // Network: hidden-a (tiny activation) -> STEP output
    let creature = create_creature_with_squash(
        vec![
            ("input-0", "input", "IDENTITY"),
            ("hidden-a", "hidden", "IDENTITY"),
            ("output-0", "output", "STEP"), // STEP activation!
        ],
        vec![
            ("input-0", "hidden-a", 1.0),
            ("hidden-a", "output-0", 0.000_002), // Tiny weight
        ],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // hidden-a has tiny activation
    let records = create_records_with_activation(vec![
        ("hidden-a", 0.1, 0.000_001), // Tiny activation
        ("output-0", 0.5, 0.5),
    ]);
    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None, None).unwrap();

    // Find hidden-a's impact
    let hidden_a = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "hidden-a")
        .expect("hidden-a should be in results");

    // Current behaviour: impact ≈ 0 (linear calculation: 0.000_002 / 0.000_002 * 1.0 = 1.0 actually)
    // The issue is the activation_weighted_impact which is structural_impact × mean_activation
    // = 1.0 × 0.000_001 = 0.000_001
    //
    // For STEP functions, even this tiny contribution could flip the output if near threshold.
    // We document the current behaviour and note that this should be improved.
    println!(
        "STEP test - hidden-a structural_impact: {}, mean_activation: {}, activation_weighted_impact: {}",
        hidden_a.impact, hidden_a.mean_activation, hidden_a.activation_weighted_impact
    );

    // The structural impact should still be ~1.0 (it's the only synapse to output)
    assert!(
        hidden_a.impact > 0.5,
        "Structural impact should reflect connectivity, got {}",
        hidden_a.impact
    );

    // Document: For STEP neurons, the activation_weighted_impact is potentially misleading
    // because threshold crossings aren't accounted for.
    // This is a known limitation documented in docs/IMPACT_CALCULATION.md
}

/// Test that BIPOLAR neurons also exhibit the threshold sensitivity issue.
#[test]
fn test_bipolar_neuron_threshold_impact() {
    let creature = create_creature_with_squash(
        vec![
            ("input-0", "input", "IDENTITY"),
            ("hidden-a", "hidden", "IDENTITY"),
            ("output-0", "output", "BIPOLAR"), // BIPOLAR activation!
        ],
        vec![
            ("input-0", "hidden-a", 1.0),
            ("hidden-a", "output-0", 0.01), // Small weight
        ],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    let records = create_records_with_activation(vec![
        ("hidden-a", 0.1, 0.5),
        ("output-0", 0.5, 1.0), // BIPOLAR output = 1.0 (positive side)
    ]);
    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None, None).unwrap();

    let hidden_a = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "hidden-a")
        .expect("hidden-a should be in results");

    println!(
        "BIPOLAR test - hidden-a structural_impact: {}, activation_weighted_impact: {}",
        hidden_a.impact, hidden_a.activation_weighted_impact
    );

    // For BIPOLAR, the output change on threshold crossing is 2.0 (-1 to +1)
    // This makes accurate impact even more important.
}

// =============================================================================
// MINIMUM/MAXIMUM Tests
// =============================================================================

/// Test that MINIMUM neurons use activation-based selection probability.
///
/// When activation records are available, we compute which synapse actually
/// "wins" (provides the minimum value) for each observation. The synapse
/// that wins most often gets the highest impact.
///
/// This is more accurate than the old 1/N conservative approach because
/// we're using real data to determine selection probability.
#[test]
fn test_minimum_neuron_selection_impact() {
    // Network: Three hidden neurons feeding into a MINIMUM output
    let creature = create_creature_with_squash(
        vec![
            ("input-0", "input", "IDENTITY"),
            ("hidden-small", "hidden", "IDENTITY"),
            ("hidden-medium", "hidden", "IDENTITY"),
            ("hidden-large", "hidden", "IDENTITY"),
            ("output-0", "output", "MINIMUM"), // MINIMUM activation!
        ],
        vec![
            ("input-0", "hidden-small", 1.0),
            ("input-0", "hidden-medium", 1.0),
            ("input-0", "hidden-large", 1.0),
            // Small weight → contributes small value to MINIMUM
            ("hidden-small", "output-0", 0.1),
            // Medium weight
            ("hidden-medium", "output-0", 1.0),
            // Large weight
            ("hidden-large", "output-0", 10.0),
        ],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // All have similar positive activations, so:
    // hidden-small contributes: 0.5 × 0.1 = 0.05 (smallest → WINS in MINIMUM!)
    // hidden-medium contributes: 0.5 × 1.0 = 0.5
    // hidden-large contributes: 0.5 × 10.0 = 5.0 (largest → loses)
    //
    // With activation-based impact, hidden-small ALWAYS wins because its
    // weighted contribution (0.05) is always the minimum. So it gets 100% impact.
    let records = create_records_with_activation(vec![
        ("hidden-small", 0.1, 0.5),
        ("hidden-medium", 0.1, 0.5),
        ("hidden-large", 0.1, 0.5),
        ("output-0", 0.1, 0.05), // Output = min(0.05, 0.5, 5.0) = 0.05
    ]);
    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None, None).unwrap();

    let small = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "hidden-small")
        .expect("hidden-small should be in results");
    let medium = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "hidden-medium")
        .expect("hidden-medium should be in results");
    let large = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "hidden-large")
        .expect("hidden-large should be in results");

    println!(
        "MINIMUM test - small impact: {}, medium impact: {}, large impact: {}",
        small.impact, medium.impact, large.impact
    );

    // With activation-based impact calculation, hidden-small wins 100% of the time
    // because its weighted contribution (0.05) is always smaller than medium (0.5)
    // and large (5.0). So hidden-small gets 100% of the impact (1.0).
    //
    // This is the CORRECT behaviour - the synapse that actually determines
    // the output should have the most impact!
    let epsilon = 0.01;
    assert!(
        (small.impact - 1.0).abs() < epsilon,
        "hidden-small should have ~100% impact (always wins MINIMUM), got {}",
        small.impact
    );
    assert!(
        medium.impact < epsilon,
        "hidden-medium should have ~0% impact (never wins MINIMUM), got {}",
        medium.impact
    );
    assert!(
        large.impact < epsilon,
        "hidden-large should have ~0% impact (never wins MINIMUM), got {}",
        large.impact
    );
}

/// Test that MAXIMUM neurons use activation-based selection probability.
///
/// When activation records are available, we compute which synapse actually
/// "wins" (provides the maximum value) for each observation. The synapse
/// that wins most often gets the highest impact.
///
/// This is more accurate than the old 1/N conservative approach because
/// we're using real data to determine selection probability.
#[test]
fn test_maximum_neuron_selection_impact() {
    let creature = create_creature_with_squash(
        vec![
            ("input-0", "input", "IDENTITY"),
            ("hidden-small", "hidden", "IDENTITY"),
            ("hidden-large", "hidden", "IDENTITY"),
            ("output-0", "output", "MAXIMUM"), // MAXIMUM activation!
        ],
        vec![
            ("input-0", "hidden-small", 1.0),
            ("input-0", "hidden-large", 1.0),
            ("hidden-small", "output-0", 0.1),
            ("hidden-large", "output-0", 10.0),
        ],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // For MAXIMUM with these weights and activations:
    // hidden-small contributes: 0.5 × 0.1 = 0.05
    // hidden-large contributes: 0.5 × 10.0 = 5.0 (ALWAYS wins MAXIMUM!)
    let records = create_records_with_activation(vec![
        ("hidden-small", 0.1, 0.5),
        ("hidden-large", 0.1, 0.5),
        ("output-0", 0.1, 5.0), // Output = max(0.05, 5.0) = 5.0
    ]);
    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None, None).unwrap();

    let small = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "hidden-small")
        .expect("hidden-small should be in results");
    let large = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "hidden-large")
        .expect("hidden-large should be in results");

    println!(
        "MAXIMUM test - small impact: {}, large impact: {}",
        small.impact, large.impact
    );

    // With activation-based impact calculation, hidden-large wins 100% of the time
    // because its weighted contribution (5.0) is always larger than small (0.05).
    // So hidden-large gets 100% of the impact (1.0).
    //
    // This is the CORRECT behaviour - the synapse that actually determines
    // the output should have the most impact!
    let epsilon = 0.01;
    assert!(
        (large.impact - 1.0).abs() < epsilon,
        "hidden-large should have ~100% impact (always wins MAXIMUM), got {}",
        large.impact
    );
    assert!(
        small.impact < epsilon,
        "hidden-small should have ~0% impact (never wins MAXIMUM), got {}",
        small.impact
    );
}

// =============================================================================
// Interaction Tests
// =============================================================================

/// Test removal candidate detection with STEP output.
///
/// A hidden neuron feeding into a STEP output might be flagged as "low impact"
/// by the current calculation but could actually flip the output.
#[test]
fn test_removal_candidate_step_output_interaction() {
    let creature = create_creature_with_squash(
        vec![
            ("input-0", "input", "IDENTITY"),
            ("hidden-critical", "hidden", "IDENTITY"),
            ("output-0", "output", "STEP"),
        ],
        vec![
            ("input-0", "hidden-critical", 1.0),
            // Tiny weight that could still flip STEP output
            ("hidden-critical", "output-0", 1e-8),
        ],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // hidden-critical has small activation
    let records = create_records_with_activation(vec![
        ("hidden-critical", 0.1, 0.001),
        ("output-0", 0.5, 1.0),
    ]);
    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None, None).unwrap();

    // Check if hidden-critical is flagged as a removal candidate
    let is_removal_candidate = result
        .removal_candidates
        .iter()
        .any(|c| c.neuron_uuid == "hidden-critical");

    let hidden = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "hidden-critical")
        .expect("hidden-critical should be in results");

    println!(
        "STEP removal test - activation_weighted_impact: {}, is_removal_candidate: {}",
        hidden.activation_weighted_impact, is_removal_candidate
    );

    // Document: With tiny weights to STEP outputs, neurons might be incorrectly
    // flagged as removal candidates because the threshold-crossing effect
    // is not accounted for.
}

// =============================================================================
// Documentation Tests
// =============================================================================

/// This test documents the current impact calculation behaviour for various squash
/// functions. It's not meant to pass/fail but to document the current state.
#[test]
fn document_squash_function_impact_accuracy() {
    println!("\n=== IMPACT CALCULATION ACCURACY BY SQUASH FUNCTION ===\n");
    println!("| Squash      | Current Model | Accuracy   | Notes                          |");
    println!("|-------------|---------------|------------|--------------------------------|");
    println!("| IDENTITY    | Linear sum    | ✅ Accurate | Mathematically exact           |");
    println!("| TANH        | Linear sum    | ⚠️ Approx   | Saturation not modelled        |");
    println!("| LOGISTIC    | Linear sum    | ⚠️ Approx   | Saturation not modelled        |");
    println!("| HARD_TANH   | Linear sum    | ⚠️ Approx   | Clamping not modelled          |");
    println!("| STEP        | Linear sum    | ❌ Wrong    | Threshold crossing ignored     |");
    println!("| BIPOLAR     | Linear sum    | ❌ Wrong    | Threshold crossing ignored     |");
    println!("| MINIMUM     | Linear sum    | ❌ Wrong    | Selection not modelled         |");
    println!("| MAXIMUM     | Linear sum    | ⚠️ Approx   | Works by coincidence           |");
    println!("| ReLU        | Linear sum    | ⚠️ Approx   | Zero region not modelled       |");
    println!("\nSee docs/IMPACT_CALCULATION.md for detailed analysis.");
}
