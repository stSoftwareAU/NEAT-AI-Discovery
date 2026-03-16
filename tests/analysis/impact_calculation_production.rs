//! Production-based impact calculation tests (Issue #130 fix)
//!
//! These tests verify the normalised impact calculation introduced in v0.2.1 to
//! fix Issue #130: hidden neurons incorrectly getting impact = 1.0.
//!
//! ## The Problem (Issue #130)
//!
//! The absolute impact calculation:
//!   impact = |weight| × child_impact
//!
//! Could give hidden neurons impact >= 1.0 (same as outputs), which caused
//! predictions to not be discounted, leading to massive overestimation.
//!
//! ## The Fix (v0.2.1)
//!
//! Use normalised impact as documented:
//!   impact = |weight| / total_inbound × child_impact
//!
//! This gives "fraction of influence" (attribution), which:
//! - Is always <= 1.0 for hidden neurons (with competing inputs)
//! - Correctly accounts for dilution when there are many inputs
//! - Matches the documented formula in IMPACT_CALCULATION.md

use crate::common::{hidden, output, synapse};
use neat_ai_discovery::CreatureJson;
use neat_ai_discovery::focus::{compute_impacts_public, rank_focus_neurons};
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use tempfile::NamedTempFile;

/// Production pattern: STEP neuron gets full impact (no normalisation)
///
/// From actual production data where STEP neurons were incorrectly diluted.
/// Any synapse to a STEP neuron could flip the output, so each gets full impact.
#[test]
fn test_step_neuron_uses_full_impact_not_normalised() {
    // A synapse to a STEP output neuron should have full downstream impact
    // because any synapse could cause the threshold to be crossed
    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![
            hidden("candidate", "IDENTITY"),
            output("step-output", "STEP"), // STEP = threshold function
        ],
        synapses: vec![
            synapse("input-0", "candidate", 1.0),
            synapse("candidate", "step-output", 0.001), // Tiny weight
            synapse("input-0", "step-output", 100.0),   // Large competing weight
        ],
    };

    let impacts = compute_impacts_public(&creature);
    let candidate_impact = *impacts.get("candidate").unwrap_or(&0.0);

    // For STEP targets, we use full impact (no normalisation)
    // because any synapse could flip the output
    assert!(
        (candidate_impact - 1.0).abs() < 0.001,
        "STEP target should give full impact (1.0), got {candidate_impact:.4}. \
         Threshold functions don't dilute impact."
    );
}

/// Production pattern: Deep network with competing inputs at each layer.
///
/// With normalised calculation, impact dilutes at each layer where there
/// are competing inputs. This is CORRECT behavior.
#[test]
fn test_deep_network_impact_dilutes_with_competing_inputs() {
    // 3-layer network: candidate → hidden1 → hidden2 → output
    // Each hidden layer has 10 inputs total
    let creature = CreatureJson {
        input: 10,
        output: 1,
        neurons: vec![
            hidden("candidate", "IDENTITY"),
            hidden("hidden1", "IDENTITY"),
            hidden("hidden2", "IDENTITY"),
            output("output-0", "IDENTITY"),
        ],
        synapses: {
            let mut synapses = vec![
                synapse("input-0", "candidate", 1.0),
                synapse("candidate", "hidden1", 0.5),
                synapse("hidden1", "hidden2", 0.5),
                synapse("hidden2", "output-0", 0.5),
            ];
            // Add "diluting" synapses at each layer (9 more inputs to hidden1 and hidden2)
            for i in 1..10 {
                synapses.push(synapse(&format!("input-{i}"), "hidden1", 0.5));
            }
            for i in 1..10 {
                synapses.push(synapse(&format!("input-{i}"), "hidden2", 0.5));
            }
            synapses
        },
    };

    let impacts = compute_impacts_public(&creature);
    let candidate_impact = *impacts.get("candidate").unwrap_or(&0.0);

    // Normalised impact:
    // hidden2 → output: 0.5 / 0.5 = 1.0 (sole connection)
    // hidden1 → hidden2: 0.5 / (0.5 + 9×0.5) = 0.5/5.0 = 0.1
    // candidate → hidden1: 0.5 / (0.5 + 9×0.5) = 0.5/5.0 × 0.1 = 0.01
    //
    // This IS correct: candidate only controls 10% of hidden1's input,
    // and hidden1 only controls 10% of hidden2's input.
    assert!(
        (candidate_impact - 0.01).abs() < 0.01,
        "Deep network with competing inputs should dilute impact to ~0.01, got {candidate_impact:.4}"
    );

    // Impact < 1.0 (Issue #130 fix)
    assert!(
        candidate_impact < 1.0,
        "Hidden neuron with competing inputs should have impact < 1.0"
    );
}

/// Production pattern: Intermediate neuron with many inputs.
///
/// A neuron that only controls a small fraction of a hub's input
/// should have proportionally small impact.
#[test]
fn test_hub_neuron_correctly_dilutes_upstream_impact() {
    let creature = CreatureJson {
        input: 100,
        output: 1,
        neurons: vec![
            hidden("upstream", "IDENTITY"),
            hidden("hub", "IDENTITY"), // Hub with 100 inputs
            output("output-0", "IDENTITY"),
        ],
        synapses: {
            let mut synapses = vec![
                synapse("input-0", "upstream", 1.0),
                synapse("upstream", "hub", 0.5),
                synapse("hub", "output-0", 1.0), // Sole connection to output
            ];
            // Add 99 more inputs to hub (total 100 including upstream)
            for i in 1..100 {
                synapses.push(synapse(&format!("input-{i}"), "hub", 0.5));
            }
            synapses
        },
    };

    let impacts = compute_impacts_public(&creature);
    let upstream_impact = *impacts.get("upstream").unwrap_or(&0.0);

    // Normalised:
    // hub → output: 1.0 / 1.0 × 1.0 = 1.0 (sole connection)
    // upstream → hub: 0.5 / (100 × 0.5) × 1.0 = 0.5/50.0 = 0.01
    //
    // This IS correct: upstream only controls 1% of hub's input.
    assert!(
        (upstream_impact - 0.01).abs() < 0.01,
        "Upstream with 1/100 of hub's input should have impact ~0.01, got {upstream_impact:.4}"
    );
}

/// Verify that neurons with genuinely negligible weights have low impact.
#[test]
fn test_truly_negligible_weight_has_low_impact() {
    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![
            hidden("negligible", "IDENTITY"),
            output("output-0", "IDENTITY"),
        ],
        synapses: vec![
            synapse("input-0", "negligible", 1.0),
            synapse("negligible", "output-0", 1e-8), // Tiny weight
            synapse("input-0", "output-0", 1.0),     // Competing input
        ],
    };

    let impacts = compute_impacts_public(&creature);
    let impact = *impacts.get("negligible").unwrap_or(&0.0);

    // Normalised: 1e-8 / (1e-8 + 1.0) × 1.0 ≈ 1e-8
    // (The 1.0 weight dominates)
    assert!(
        impact < 1e-6,
        "Negligible weight should have negligible impact, got {impact}"
    );
}

/// Multiple outputs: impact sums correctly across paths.
#[test]
fn test_multiple_output_connections_sum_impact() {
    // A neuron connected to 3 outputs should sum the impacts
    let creature = CreatureJson {
        input: 1,
        output: 3,
        neurons: vec![
            hidden("hub", "IDENTITY"),
            output("output-0", "IDENTITY"),
            output("output-1", "IDENTITY"),
            output("output-2", "IDENTITY"),
        ],
        synapses: vec![
            synapse("input-0", "hub", 1.0),
            synapse("hub", "output-0", 1.0), // Sole input to each output
            synapse("hub", "output-1", 1.0),
            synapse("hub", "output-2", 1.0),
        ],
    };

    let impacts = compute_impacts_public(&creature);
    let hub_impact = *impacts.get("hub").unwrap_or(&0.0);

    // Each output contribution: 1.0 / 1.0 × 1.0 = 1.0
    // Total: 3.0 (affects all 3 outputs)
    assert!(
        (hub_impact - 3.0).abs() < 0.01,
        "Hub connected to 3 outputs should have impact ~3.0, got {hub_impact}"
    );
}

/// MINIMUM neuron uses selection-based impact (probability of winning).
#[test]
fn test_minimum_neuron_uses_selection_based_impact() {
    // For MINIMUM targets, impact = 1/N × child_impact (equal probability)
    let creature = CreatureJson {
        input: 3,
        output: 1,
        neurons: vec![
            hidden("candidate", "IDENTITY"),
            output("min-output", "MINIMUM"), // MINIMUM selects one input
        ],
        synapses: vec![
            synapse("input-0", "candidate", 1.0),
            synapse("candidate", "min-output", 1.0),
            synapse("input-1", "min-output", 1.0),
            synapse("input-2", "min-output", 1.0),
        ],
    };

    let impacts = compute_impacts_public(&creature);
    let candidate_impact = *impacts.get("candidate").unwrap_or(&0.0);

    // MINIMUM: each of 3 inputs has 1/3 probability of winning
    // candidate impact = 1/3 × 1.0 ≈ 0.33
    assert!(
        (candidate_impact - 1.0 / 3.0).abs() < 0.01,
        "MINIMUM target should give 1/N impact, got {candidate_impact:.4}"
    );
}

/// Verify activation-weighted impact works correctly with recorded data.
#[test]
fn test_activation_weighted_impact_with_recorded_activations() {
    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    let parquet_path = temp_file.path().to_str().unwrap();

    // Create records with varying activations
    let records = vec![
        // High activation neuron
        DiscoverRecord::new(
            0,
            "high-activation".to_string(),
            Some(1.0),
            10.0, // High activation
            vec![0.1],
        ),
        DiscoverRecord::new(1, "high-activation".to_string(), Some(1.0), 10.0, vec![0.1]),
        DiscoverRecord::new(
            0,
            "low-activation".to_string(),
            Some(1.0),
            0.001, // Low activation
            vec![0.1],
        ),
        DiscoverRecord::new(1, "low-activation".to_string(), Some(1.0), 0.001, vec![0.1]),
        DiscoverRecord::new(0, "output-0".to_string(), Some(0.5), 0.5, vec![0.1]),
        DiscoverRecord::new(1, "output-0".to_string(), Some(0.5), 0.5, vec![0.1]),
    ];

    write_records_to_parquet(parquet_path, &records).expect("Failed to write parquet");

    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![
            hidden("high-activation", "IDENTITY"),
            hidden("low-activation", "IDENTITY"),
            output("output-0", "IDENTITY"),
        ],
        synapses: vec![
            synapse("input-0", "high-activation", 1.0),
            synapse("input-0", "low-activation", 1.0),
            synapse("high-activation", "output-0", 0.01), // Same structural impact
            synapse("low-activation", "output-0", 0.01),  // Same structural impact
        ],
    };

    let result =
        rank_focus_neurons(parquet_path, &creature, None, None).expect("Ranking should succeed");

    // Find the neurons in results
    let high = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "high-activation")
        .expect("high-activation should be ranked");
    let low = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "low-activation")
        .expect("low-activation should be ranked");

    // Both have same structural impact, but high-activation has higher mean_activation
    // activation_weighted_impact = structural × mean_activation
    assert!(
        high.activation_weighted_impact > low.activation_weighted_impact * 100.0,
        "high-activation should have much higher activation_weighted_impact \
        (high: {}, low: {})",
        high.activation_weighted_impact,
        low.activation_weighted_impact
    );
}

/// Test: neuron with many inputs and small output weight.
#[test]
fn test_many_inputs_small_output_weight_has_reasonable_impact() {
    let creature = CreatureJson {
        input: 100,
        output: 1,
        neurons: vec![
            hidden("candidate", "IDENTITY"),
            output("output-0", "IDENTITY"),
        ],
        synapses: {
            let mut synapses = vec![
                synapse("input-0", "candidate", 1.0),
                synapse("candidate", "output-0", 0.001), // Small weight
            ];
            // Add 99 other inputs to output (competing)
            for i in 1..100 {
                synapses.push(synapse(&format!("input-{i}"), "output-0", 0.01));
            }
            synapses
        },
    };

    let impacts = compute_impacts_public(&creature);
    let candidate_impact = *impacts.get("candidate").unwrap_or(&0.0);

    // Normalised: 0.001 / (0.001 + 99×0.01) × 1.0 = 0.001 / 0.991 ≈ 0.001
    assert!(
        candidate_impact > 0.0 && candidate_impact < 0.01,
        "Small weight with competing inputs should have small but positive impact, got {candidate_impact}"
    );
}
