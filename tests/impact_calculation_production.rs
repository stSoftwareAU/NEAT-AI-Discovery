//! Production-based impact calculation tests (Dec 2024)
//!
//! These tests are derived from actual production failures where impact was
//! massively underestimated, causing neurons to be incorrectly marked as
//! "low impact" when removing them actually increased error by up to 20%.
//!
//! ## Production Failure Pattern
//!
//! | Calculated Impact | Actual Error Increase | Underestimation |
//! |-------------------|----------------------|-----------------|
//! | 1.39e-12          | 20% (0.2)            | 145 billion x   |
//! | 4.94e-13          | 0.0054% (5.4e-5)     | 100 million x   |
//! | 1.87e-10          | 0.44% (4.4e-3)       | 24 million x    |
//!
//! ## Root Cause (v0.1.145 fix)
//!
//! The old normalised impact calculation:
//!   impact = |weight| / total_inbound × child_impact
//!
//! Was fundamentally wrong for removal prediction. It answered "what share of
//! blame?" not "what happens when removed?"
//!
//! The fix uses absolute impact:
//!   impact = |weight| × child_impact

mod common;

use neat_ai_discovery::focus::{compute_impacts_public, rank_focus_neurons};
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
use tempfile::NamedTempFile;

/// Create a synapse helper for cleaner test setup
fn synapse(from: &str, to: &str, weight: f32) -> SynapseJson {
    SynapseJson {
        from_uuid: from.to_string(),
        to_uuid: to.to_string(),
        weight,
        synapse_type: None,
    }
}

/// Create a hidden neuron helper
fn hidden(uuid: &str, squash: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: "hidden".to_string(),
        squash: squash.to_string(),
        bias: 0.0,
    }
}

/// Create an output neuron helper
fn output(uuid: &str, squash: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: "output".to_string(),
        squash: squash.to_string(),
        bias: 0.0,
    }
}

/// Production pattern: Hidden neuron with many incoming synapses and small weight to output.
///
/// In production, neurons like this were marked as "low impact" (1e-13) when
/// removing them actually increased error by 0.005%. The old normalised
/// calculation diluted impact by the number of incoming synapses.
#[test]
fn test_many_inputs_small_output_weight_has_reasonable_impact() {
    // Pattern from production failure:
    // - 16 incoming synapses to hidden neuron
    // - 1 outgoing synapse with weight 0.037
    // - Old calculation gave impact ~4.94e-13
    // - Actual removal impact was ~5.4e-5 (100 million x underestimate)
    let creature = CreatureJson {
        input: 16,
        output: 1,
        neurons: vec![hidden("target", "IDENTITY"), output("output-0", "IDENTITY")],
        synapses: {
            let mut synapses = Vec::new();
            // 16 incoming synapses (like production)
            for i in 0..16 {
                synapses.push(synapse(&format!("input-{i}"), "target", 1.0));
            }
            // Small weight to output
            synapses.push(synapse("target", "output-0", 0.037));
            synapses
        },
    };

    let impacts = compute_impacts_public(&creature);
    let target_impact = *impacts.get("target").unwrap_or(&0.0);

    // With absolute impact: 0.037 × 1.0 = 0.037
    // Must be >> 1e-10 (the old underestimate)
    assert!(
        target_impact > 0.01,
        "Target impact should be ~0.037 (absolute weight), got {target_impact:.2e}. \
         Old normalised calculation would give ~1e-13 which was wrong."
    );
}

/// Production pattern: Deep network with multiple intermediate layers.
///
/// The old normalised calculation multiplied fractions at each layer,
/// causing exponential underestimation. A neuron 3 hops from output
/// with 20 synapses at each layer would have impact diluted by:
///   1/20 × 1/20 × 1/20 = 1/8000
#[test]
fn test_deep_network_impact_not_exponentially_diluted() {
    // 3-layer network: candidate → hidden1 → hidden2 → output
    // Each layer has multiple inputs that would dilute normalised impact
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
            // Add "diluting" synapses at each layer
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

    // Absolute impact through the path:
    // candidate → hidden1: 0.5
    // hidden1 → hidden2: 0.5
    // hidden2 → output: 0.5
    // Total: 0.5 × 0.5 × 0.5 = 0.125
    //
    // Old normalised would give:
    // (0.5/5) × (0.5/5) × (0.5) = 0.01 × 0.01 × 0.5 = 0.00005
    assert!(
        candidate_impact > 0.05,
        "Deep network impact should be ~0.125 (weight product), got {candidate_impact:.4}. \
         Old normalised calculation would exponentially dilute this."
    );
}

/// Production pattern: Intermediate neuron with 322 inputs.
///
/// From actual production failure:
/// - Candidate had 1 outgoing synapse (weight 0.037) to intermediate
/// - Intermediate had 322 incoming synapses and connected to output
/// - Old calculation: 0.037/322 ≈ 0.0001 × downstream = ~1e-10
/// - Actual impact was much higher
#[test]
fn test_hub_neuron_with_many_inputs_doesnt_zero_upstream_impact() {
    let creature = CreatureJson {
        input: 100,
        output: 1,
        neurons: vec![
            hidden("upstream", "IDENTITY"),
            hidden("hub", "IDENTITY"), // Like the 322-input neuron in production
            output("output-0", "IDENTITY"),
        ],
        synapses: {
            let mut synapses = vec![
                synapse("input-0", "upstream", 1.0),
                synapse("upstream", "hub", 0.5),
                synapse("hub", "output-0", 1.0),
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

    // Absolute: upstream → hub (0.5) × hub → output (1.0) = 0.5
    // Old normalised: 0.5/50 × 1.0 = 0.01 (50x underestimate!)
    assert!(
        upstream_impact > 0.3,
        "Upstream impact should be ~0.5 (absolute path weight), got {upstream_impact:.4}. \
         The 100 other inputs to hub shouldn't dilute this."
    );
}

/// Verify that neurons with genuinely negligible weights have low impact.
///
/// The fix to absolute impact shouldn't make all impacts large. A neuron
/// with weight 1e-8 to output should still have negligible impact.
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
            synapse("negligible", "output-0", 1e-8),
            synapse("input-0", "output-0", 1.0),
        ],
    };

    let impacts = compute_impacts_public(&creature);
    let impact = *impacts.get("negligible").unwrap_or(&0.0);

    // Absolute: 1e-8 × 1.0 = 1e-8 (genuinely small)
    assert!(
        impact < 1e-6,
        "Neuron with 1e-8 weight should have small impact, got {impact:.2e}"
    );
    assert!(impact > 1e-10, "Impact shouldn't be zero, got {impact:.2e}");
}

/// Test that impact is correctly shared when neuron connects to multiple outputs.
///
/// If a neuron connects to 2 outputs with weight 1.0 each, it should have
/// impact ~2.0 (sum of paths), not diluted.
#[test]
fn test_multiple_output_connections_sum_impact() {
    let creature = CreatureJson {
        input: 1,
        output: 2,
        neurons: vec![
            hidden("multi-output", "IDENTITY"),
            output("output-0", "IDENTITY"),
            output("output-1", "IDENTITY"),
        ],
        synapses: vec![
            synapse("input-0", "multi-output", 1.0),
            synapse("multi-output", "output-0", 1.0),
            synapse("multi-output", "output-1", 1.0),
        ],
    };

    let impacts = compute_impacts_public(&creature);
    let impact = *impacts.get("multi-output").unwrap_or(&0.0);

    // Should sum: 1.0 + 1.0 = 2.0
    assert!(
        (impact - 2.0).abs() < 0.01,
        "Multi-output neuron should have summed impact ~2.0, got {impact}"
    );
}

/// Test with actual activation data - the mean_activation should correctly
/// scale the structural impact for removal candidate detection.
#[test]
fn test_activation_weighted_impact_with_recorded_activations() {
    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![hidden("tested", "IDENTITY"), output("output-0", "IDENTITY")],
        synapses: vec![
            synapse("input-0", "tested", 1.0),
            synapse("tested", "output-0", 0.1), // Small but not negligible
        ],
    };

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // Records with various activations
    let records = vec![
        DiscoverRecord::new(0, "tested".to_string(), Some(0.5), 0.5, vec![0.1]),
        DiscoverRecord::new(1, "tested".to_string(), Some(0.3), 0.3, vec![0.1]),
        DiscoverRecord::new(2, "tested".to_string(), Some(0.7), 0.7, vec![0.1]),
        DiscoverRecord::new(0, "output-0".to_string(), Some(0.5), 0.5, vec![0.1]),
        DiscoverRecord::new(1, "output-0".to_string(), Some(0.5), 0.5, vec![0.1]),
        DiscoverRecord::new(2, "output-0".to_string(), Some(0.5), 0.5, vec![0.1]),
    ];
    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None).unwrap();

    let neuron = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "tested")
        .expect("tested should be in results");

    // structural_impact = 0.1 (weight to output)
    // mean_activation ≈ 0.5 (average of 0.5, 0.3, 0.7)
    // activation_weighted_impact ≈ 0.1 × 0.5 = 0.05
    assert!(
        neuron.impact > 0.05,
        "Structural impact should be ~0.1, got {:.4}",
        neuron.impact
    );
    assert!(
        neuron.mean_activation > 0.3 && neuron.mean_activation < 0.7,
        "Mean activation should be ~0.5, got {:.4}",
        neuron.mean_activation
    );
}

/// Test STEP/BIPOLAR neurons use full child_impact (threshold category).
///
/// For threshold functions, any synapse could flip the output, so we don't
/// normalise by total_inbound.
#[test]
fn test_step_neuron_uses_full_impact_not_normalised() {
    let creature = CreatureJson {
        input: 10,
        output: 1,
        neurons: vec![
            hidden("upstream", "IDENTITY"),
            hidden("step-gate", "STEP"), // Threshold activation
            output("output-0", "IDENTITY"),
        ],
        synapses: {
            let mut synapses = vec![
                synapse("input-0", "upstream", 1.0),
                synapse("upstream", "step-gate", 0.1), // Small weight
                synapse("step-gate", "output-0", 1.0),
            ];
            // Add other inputs to step-gate
            for i in 1..10 {
                synapses.push(synapse(&format!("input-{i}"), "step-gate", 1.0));
            }
            synapses
        },
    };

    let impacts = compute_impacts_public(&creature);
    let upstream_impact = *impacts.get("upstream").unwrap_or(&0.0);

    // For STEP targets, we use full child_impact (not normalised)
    // upstream → step-gate uses child_impact = 1.0 (step-gate → output)
    // So upstream impact should be ~1.0, not 0.1/10 × 1.0 = 0.01
    assert!(
        upstream_impact > 0.5,
        "Upstream to STEP neuron should have high impact (threshold category), got {upstream_impact:.4}"
    );
}

/// Test MINIMUM/MAXIMUM neurons use selection-based impact.
///
/// Only one synapse "wins" at any time, so impact should be probability-weighted.
#[test]
fn test_minimum_neuron_uses_selection_based_impact() {
    let creature = CreatureJson {
        input: 3,
        output: 1,
        neurons: vec![
            hidden("always-wins", "IDENTITY"),
            hidden("min-gate", "MINIMUM"),
            output("output-0", "IDENTITY"),
        ],
        synapses: vec![
            synapse("input-0", "always-wins", 1.0),
            synapse("always-wins", "min-gate", 1.0),
            synapse("input-1", "min-gate", 1.0),
            synapse("input-2", "min-gate", 1.0),
            synapse("min-gate", "output-0", 1.0),
        ],
    };

    let impacts = compute_impacts_public(&creature);
    let impact = *impacts.get("always-wins").unwrap_or(&0.0);

    // Without activation data, MINIMUM uses equal probability: 1/3
    // So impact ≈ 1/3 × 1.0 = 0.33
    assert!(
        impact > 0.2 && impact < 0.5,
        "MINIMUM selection impact should be ~0.33 (1/3 probability), got {impact:.4}"
    );
}
