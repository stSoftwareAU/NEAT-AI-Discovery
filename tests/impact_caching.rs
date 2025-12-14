//! Tests verifying that impact calculation correctly caches results for neurons
//! reached through multiple paths.
//!
//! The impact calculation uses memoization to avoid recalculating impacts for
//! neurons that are reachable through multiple paths from the outputs. This is
//! critical for performance in complex networks with many interconnections.
//!
//! Example network where caching matters:
//! ```text
//!   A ──┬── B ──┐
//!       │      │
//!       └── C ─┴── Output
//! ```
//! Here, the output's impact is computed once and cached, so when computing
//! B's and C's impact, we reuse the cached value rather than recomputing.

use neat_ai_discovery::focus::{compute_impacts_public, rank_focus_neurons};
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
use tempfile::tempdir;

/// Test that impact calculation correctly handles diamond-shaped networks
/// where a neuron is reachable through multiple paths.
#[test]
fn impact_caching_handles_diamond_topology() {
    // Create a diamond topology:
    //   hidden-1 ──┬── hidden-3 ──┐
    //              │              │
    //   hidden-2 ──┴── hidden-4 ─┴── output-0
    let creature = CreatureJson {
        input: 0,
        output: 1,
        neurons: vec![
            NeuronJson {
                uuid: "hidden-1".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "hidden-2".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "hidden-3".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "hidden-4".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "output-0".to_string(),
                neuron_type: "output".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
        ],
        synapses: vec![
            // Diamond from hidden-1 and hidden-2 through hidden-3 and hidden-4
            SynapseJson {
                from_uuid: "hidden-1".to_string(),
                to_uuid: "hidden-3".to_string(),
                weight: 1.0,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "hidden-1".to_string(),
                to_uuid: "hidden-4".to_string(),
                weight: 1.0,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "hidden-2".to_string(),
                to_uuid: "hidden-3".to_string(),
                weight: 1.0,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "hidden-2".to_string(),
                to_uuid: "hidden-4".to_string(),
                weight: 1.0,
                synapse_type: None,
            },
            // Both hidden-3 and hidden-4 connect to output
            SynapseJson {
                from_uuid: "hidden-3".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "hidden-4".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.5,
                synapse_type: None,
            },
        ],
    };

    let impacts = compute_impacts_public(&creature);

    // Output should have impact 1.0
    assert_eq!(
        impacts.get("output-0").copied().unwrap_or(0.0),
        1.0,
        "Output neuron should have impact 1.0"
    );

    // hidden-3 and hidden-4 should have same impact (both connect to output with weight 0.5)
    let impact_3 = impacts.get("hidden-3").copied().unwrap_or(0.0);
    let impact_4 = impacts.get("hidden-4").copied().unwrap_or(0.0);
    assert!(
        (impact_3 - impact_4).abs() < 0.001,
        "hidden-3 ({impact_3}) and hidden-4 ({impact_4}) should have same impact"
    );

    // hidden-1 and hidden-2 should have same impact (symmetric connections)
    let impact_1 = impacts.get("hidden-1").copied().unwrap_or(0.0);
    let impact_2 = impacts.get("hidden-2").copied().unwrap_or(0.0);
    assert!(
        (impact_1 - impact_2).abs() < 0.001,
        "hidden-1 ({impact_1}) and hidden-2 ({impact_2}) should have same impact"
    );

    // hidden-1's impact should be sum of normalised paths through hidden-3 and hidden-4
    // hidden-3 = 0.5 / (0.5+0.5) × 1.0 = 0.5 (shares output with hidden-4)
    // hidden-1 → hidden-3: 1.0 / (1.0+1.0) × 0.5 = 0.25 (shares hidden-3 with hidden-2)
    // hidden-1 → hidden-4: 1.0 / (1.0+1.0) × 0.5 = 0.25 (shares hidden-4 with hidden-2)
    // Total: 0.25 + 0.25 = 0.5
    assert!(
        (impact_1 - 0.5).abs() < 0.001,
        "hidden-1 should have impact ~0.5, got {impact_1}"
    );
}

/// Test that impact calculation correctly handles deep networks
/// without stack overflow or performance degradation.
#[test]
fn impact_caching_handles_deep_networks() {
    // Create a deep chain: hidden-0 -> hidden-1 -> ... -> hidden-9 -> output
    let mut neurons = vec![];
    let mut synapses = vec![];

    for i in 0..10 {
        neurons.push(NeuronJson {
            uuid: format!("hidden-{i}"),
            neuron_type: "hidden".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        });

        if i > 0 {
            synapses.push(SynapseJson {
                from_uuid: format!("hidden-{}", i - 1),
                to_uuid: format!("hidden-{i}"),
                weight: 0.9,
                synapse_type: None,
            });
        }
    }

    neurons.push(NeuronJson {
        uuid: "output-0".to_string(),
        neuron_type: "output".to_string(),
        squash: "IDENTITY".to_string(),
        bias: 0.0,
    });

    synapses.push(SynapseJson {
        from_uuid: "hidden-9".to_string(),
        to_uuid: "output-0".to_string(),
        weight: 1.0,
        synapse_type: None,
    });

    let creature = CreatureJson {
        input: 0,
        output: 1,
        neurons,
        synapses,
    };

    let impacts = compute_impacts_public(&creature);

    // Verify impacts are computed correctly along the chain
    assert_eq!(impacts.get("output-0").copied().unwrap_or(0.0), 1.0);

    // hidden-9 -> output with weight 1.0, and it's the SOLE input to output
    // Normalised impact: 1.0 / 1.0 × 1.0 = 1.0
    assert!(
        (impacts.get("hidden-9").copied().unwrap_or(0.0) - 1.0).abs() < 0.001,
        "hidden-9 should have impact ~1.0"
    );

    // With NORMALISED impact (Issue #130 fix), in a chain of sole connections:
    // Each neuron is the ONLY input to the next, so each has 100% of that neuron's
    // input weight. Therefore, all neurons in a sole-connection chain have impact = 1.0.
    //
    // Example: hidden-8 → hidden-9 (weight 0.9, sole input)
    // Normalised impact = 0.9 / 0.9 × 1.0 = 1.0
    //
    // This is correct! If you're the ONLY input, you have 100% influence.
    // Impact only dilutes when there are COMPETING inputs.
    let impact_0 = impacts.get("hidden-0").copied().unwrap_or(0.0);
    assert!(
        (impact_0 - 1.0).abs() < 0.001,
        "hidden-0 should have impact ~1.0 (sole connection chain), got {impact_0}"
    );

    // All neurons in a sole-connection chain should have the same impact (1.0)
    for i in 0..10 {
        let current_impact = impacts.get(&format!("hidden-{i}")).copied().unwrap_or(0.0);
        assert!(
            (current_impact - 1.0).abs() < 0.001,
            "hidden-{i} should have impact ~1.0 (sole connection), got {current_impact}"
        );
    }
}

/// Test that impact calculation correctly handles cycles without infinite loops.
/// The memoization with cycle detection should return 0 for cyclic paths.
#[test]
fn impact_caching_handles_cycles_gracefully() {
    // Create a cycle: hidden-1 <-> hidden-2, both connecting to output
    let creature = CreatureJson {
        input: 0,
        output: 1,
        neurons: vec![
            NeuronJson {
                uuid: "hidden-1".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "hidden-2".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "output-0".to_string(),
                neuron_type: "output".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
        ],
        synapses: vec![
            // Cycle between hidden-1 and hidden-2
            SynapseJson {
                from_uuid: "hidden-1".to_string(),
                to_uuid: "hidden-2".to_string(),
                weight: 1.0,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "hidden-2".to_string(),
                to_uuid: "hidden-1".to_string(),
                weight: 1.0,
                synapse_type: None,
            },
            // Both connect to output
            SynapseJson {
                from_uuid: "hidden-1".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "hidden-2".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.5,
                synapse_type: None,
            },
        ],
    };

    // This should complete without hanging
    let impacts = compute_impacts_public(&creature);

    // Output should still have impact 1.0
    assert_eq!(impacts.get("output-0").copied().unwrap_or(0.0), 1.0);

    // Hidden neurons should have finite, non-zero impact
    let impact_1 = impacts.get("hidden-1").copied().unwrap_or(0.0);
    let impact_2 = impacts.get("hidden-2").copied().unwrap_or(0.0);

    assert!(
        impact_1.is_finite() && impact_1 > 0.0,
        "hidden-1 should have finite positive impact, got {impact_1}"
    );
    assert!(
        impact_2.is_finite() && impact_2 > 0.0,
        "hidden-2 should have finite positive impact, got {impact_2}"
    );
}

/// Test that focus ranking correctly uses cached impacts for all neurons.
/// This verifies the full integration from parquet reading to impact calculation.
#[test]
fn rank_focus_neurons_uses_cached_impacts() {
    let temp_dir = tempdir().expect("Failed to create temp directory");
    let parquet_path = temp_dir.path().join("records.parquet");
    let parquet_file = parquet_path.to_str().unwrap().to_string();

    // Create records for multiple neurons
    let mut records = Vec::new();
    for obs_index in 0..15u32 {
        for neuron in ["hidden-1", "hidden-2", "output-0"] {
            records.push(DiscoverRecord::new(
                obs_index,
                neuron.to_string(),
                Some(0.5),
                0.5,
                vec![0.1],
            ));
        }
    }

    neat_ai_discovery::parquet_format::write_records_to_parquet(&parquet_file, &records)
        .expect("Failed to write parquet");

    let creature = CreatureJson {
        input: 0,
        output: 1,
        neurons: vec![
            NeuronJson {
                uuid: "hidden-1".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "hidden-2".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "output-0".to_string(),
                neuron_type: "output".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
        ],
        synapses: vec![
            SynapseJson {
                from_uuid: "hidden-1".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 1.0,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "hidden-2".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.5,
                synapse_type: None,
            },
        ],
    };

    let result = rank_focus_neurons(&parquet_file, &creature, None, None)
        .expect("Focus ranking should succeed");

    // All neurons should be processed
    assert_eq!(
        result.processed_neurons, 3,
        "All 3 neurons should be processed"
    );

    // Output should have highest impact (1.0)
    let output_neuron = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "output-0")
        .expect("Output neuron should be in results");
    assert!(
        (output_neuron.impact - 1.0).abs() < 0.001,
        "Output should have impact 1.0, got {}",
        output_neuron.impact
    );

    // hidden-1 should have higher impact than hidden-2 (weight 1.0 vs 0.5)
    let h1 = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "hidden-1")
        .expect("hidden-1 should be in results");
    let h2 = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "hidden-2")
        .expect("hidden-2 should be in results");

    assert!(
        h1.impact > h2.impact,
        "hidden-1 (weight 1.0) should have higher impact than hidden-2 (weight 0.5)"
    );
}
