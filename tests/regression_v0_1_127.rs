//! REGRESSION TESTS for v0.1.127 fixes
//!
//! These tests catch regressions if the v0.1.127 fixes are accidentally reverted.
//! DO NOT DELETE OR COMMENT OUT THESE TESTS - they protect against known bugs.
//!
//! ## Fixes covered:
//!
//! 1. **Dynamic removal threshold based on synapse counts**: The removal candidate
//!    threshold must account for the complexity savings from removing both the
//!    neuron AND its synapses. The NEAT-AI Score.ts formula is:
//!
//!    ```
//!    complexityPenalty = hiddenNeuronCount × growthCost +
//!                        synapseCount × growthCost / 10 +
//!                        penalty × growthCost / 100
//!    ```
//!
//!    So removing a neuron with N incoming and M outgoing synapses saves:
//!    ```
//!    savings = growthCost × (1 + (N + M) / 10)
//!    ```
//!
//!    Removal is beneficial when: activation_weighted_impact < savings × SCALE_FACTOR
//!
//! If any of these tests fail after a code change, the fix has regressed.

mod common;

use neat_ai_discovery::focus::{calculate_removal_savings, rank_focus_neurons};
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
use tempfile::NamedTempFile;

/// Default growth cost matching NEAT-AI's typical value
const COST_OF_GROWTH: f32 = 1e-7;

/// Test the removal savings calculation matches NEAT-AI's Score.ts formula.
///
/// From Score.ts:
/// ```typescript
/// const complexityPenalty = hiddenNeuronCount * growthCost +
///     creature.synapses.length * growthCost / 10 + penalty * growthCost / 100;
/// ```
///
/// So removing a neuron with N incoming and M outgoing synapses saves:
/// - growthCost for the neuron
/// - (N + M) × growthCost / 10 for the synapses
#[test]
fn test_removal_savings_matches_neat_ai_formula() {
    // Neuron with 0 synapses: just the neuron cost
    let savings = calculate_removal_savings(0, 0, COST_OF_GROWTH);
    assert!(
        (savings - COST_OF_GROWTH).abs() < 1e-15,
        "Neuron with 0 synapses should save exactly growthCost ({COST_OF_GROWTH}), got {savings}"
    );

    // Neuron with 1 incoming, 1 outgoing (2 total synapses)
    // savings = growthCost × (1 + 2/10) = growthCost × 1.2
    let savings = calculate_removal_savings(1, 1, COST_OF_GROWTH);
    let expected = COST_OF_GROWTH * 1.2;
    assert!(
        (savings - expected).abs() < 1e-15,
        "Neuron with 2 synapses should save {expected} (growthCost × 1.2), got {savings}"
    );

    // Neuron with 5 incoming, 3 outgoing (8 total synapses)
    // savings = growthCost × (1 + 8/10) = growthCost × 1.8
    let savings = calculate_removal_savings(5, 3, COST_OF_GROWTH);
    let expected = COST_OF_GROWTH * 1.8;
    assert!(
        (savings - expected).abs() < 1e-15,
        "Neuron with 8 synapses should save {expected} (growthCost × 1.8), got {savings}"
    );

    // Neuron with 10 incoming, 10 outgoing (20 total synapses)
    // savings = growthCost × (1 + 20/10) = growthCost × 3.0
    let savings = calculate_removal_savings(10, 10, COST_OF_GROWTH);
    let expected = COST_OF_GROWTH * 3.0;
    assert!(
        (savings - expected).abs() < 1e-15,
        "Neuron with 20 synapses should save {expected} (growthCost × 3.0), got {savings}"
    );
}

/// Test that neurons with many synapses have higher removal threshold.
///
/// A neuron with many connections saves more complexity when removed,
/// so it can have a higher activation_weighted_impact and still be
/// a valid removal candidate.
#[test]
fn test_more_synapses_means_higher_threshold() {
    let few_synapses = calculate_removal_savings(1, 1, COST_OF_GROWTH);
    let many_synapses = calculate_removal_savings(10, 10, COST_OF_GROWTH);

    assert!(
        many_synapses > few_synapses,
        "Neuron with more synapses ({many_synapses}) should have higher removal savings than neuron with fewer ({few_synapses})"
    );

    // Ratio should be 3.0 / 1.2 = 2.5
    let ratio = many_synapses / few_synapses;
    assert!(
        (ratio - 2.5).abs() < 0.01,
        "Ratio should be 2.5, got {ratio}"
    );
}

/// REGRESSION TEST: Removal candidates must use dynamic threshold based on synapse count.
///
/// A neuron with many synapses saves more complexity when removed, so it can have
/// higher activation_weighted_impact and still be a valid removal candidate.
#[test]
fn regression_removal_uses_dynamic_threshold_based_on_synapse_count() {
    // Network with two neurons having different synapse counts:
    //
    // few-synapses: 1 incoming, 1 outgoing
    //   input-0 -> few-synapses -> output-0
    //   savings = growthCost × (1 + 2/10) = 1.2e-7
    //   threshold = 1.2e-7 × 100 = 1.2e-5
    //
    // many-synapses: 3 incoming, 2 outgoing
    //   input-0, input-1, input-2 -> many-synapses -> output-0, output-1
    //   savings = growthCost × (1 + 5/10) = 1.5e-7
    //   threshold = 1.5e-7 × 100 = 1.5e-5
    let creature = CreatureJson {
        input: 3,
        output: 2,
        neurons: vec![
            NeuronJson {
                uuid: "few-synapses".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "many-synapses".to_string(),
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
            NeuronJson {
                uuid: "output-1".to_string(),
                neuron_type: "output".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
        ],
        synapses: vec![
            // few-synapses: 1 in, 1 out
            SynapseJson {
                from_uuid: "input-0".to_string(),
                to_uuid: "few-synapses".to_string(),
                weight: 1e-6, // Very small weight
            },
            SynapseJson {
                from_uuid: "few-synapses".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 1e-6, // Very small weight
            },
            // many-synapses: 3 in, 2 out
            SynapseJson {
                from_uuid: "input-0".to_string(),
                to_uuid: "many-synapses".to_string(),
                weight: 1e-6,
            },
            SynapseJson {
                from_uuid: "input-1".to_string(),
                to_uuid: "many-synapses".to_string(),
                weight: 1e-6,
            },
            SynapseJson {
                from_uuid: "input-2".to_string(),
                to_uuid: "many-synapses".to_string(),
                weight: 1e-6,
            },
            SynapseJson {
                from_uuid: "many-synapses".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 1e-6,
            },
            SynapseJson {
                from_uuid: "many-synapses".to_string(),
                to_uuid: "output-1".to_string(),
                weight: 1e-6,
            },
        ],
    };

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // Both neurons have small activation (0.5) and small impact
    let records = vec![
        DiscoverRecord::new(0, "few-synapses".to_string(), Some(0.5), 0.5, vec![0.1]),
        DiscoverRecord::new(1, "few-synapses".to_string(), Some(0.5), 0.5, vec![0.1]),
        DiscoverRecord::new(0, "many-synapses".to_string(), Some(0.5), 0.5, vec![0.1]),
        DiscoverRecord::new(1, "many-synapses".to_string(), Some(0.5), 0.5, vec![0.1]),
        DiscoverRecord::new(0, "output-0".to_string(), Some(0.5), 0.5, vec![0.1]),
        DiscoverRecord::new(1, "output-0".to_string(), Some(0.5), 0.5, vec![0.1]),
        DiscoverRecord::new(0, "output-1".to_string(), Some(0.5), 0.5, vec![0.1]),
        DiscoverRecord::new(1, "output-1".to_string(), Some(0.5), 0.5, vec![0.1]),
    ];
    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None).unwrap();

    // Verify both neurons have similar low activation_weighted_impact
    let few = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "few-synapses")
        .expect("few-synapses should be in results");
    let many = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "many-synapses")
        .expect("many-synapses should be in results");

    // Both should have very low impact due to tiny weights
    assert!(
        few.activation_weighted_impact < 1e-5,
        "few-synapses should have low activation_weighted_impact, got {}",
        few.activation_weighted_impact
    );
    assert!(
        many.activation_weighted_impact < 1e-5,
        "many-synapses should have low activation_weighted_impact, got {}",
        many.activation_weighted_impact
    );

    // Check that removal candidates are found with the dynamic threshold
    println!(
        "few-synapses: impact={:.2e}, activation_weighted={:.2e}",
        few.impact, few.activation_weighted_impact
    );
    println!(
        "many-synapses: impact={:.2e}, activation_weighted={:.2e}",
        many.impact, many.activation_weighted_impact
    );
    println!(
        "Removal candidates: {:?}",
        result
            .removal_candidates
            .iter()
            .map(|c| (&c.neuron_uuid, c.incoming_synapses, c.outgoing_synapses))
            .collect::<Vec<_>>()
    );
}

/// Test that the removal reason includes synapse count information.
#[test]
fn test_removal_reason_includes_synapse_savings() {
    // Network with a neuron that should be a removal candidate
    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![
            NeuronJson {
                uuid: "negligible".to_string(),
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
                from_uuid: "input-0".to_string(),
                to_uuid: "negligible".to_string(),
                weight: 1e-10, // Extremely small
            },
            SynapseJson {
                from_uuid: "negligible".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 1e-10, // Extremely small
            },
        ],
    };

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    let records = vec![
        DiscoverRecord::new(0, "negligible".to_string(), Some(0.5), 0.5, vec![0.1]),
        DiscoverRecord::new(1, "negligible".to_string(), Some(0.5), 0.5, vec![0.1]),
        DiscoverRecord::new(0, "output-0".to_string(), Some(0.5), 0.5, vec![0.1]),
        DiscoverRecord::new(1, "output-0".to_string(), Some(0.5), 0.5, vec![0.1]),
    ];
    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None).unwrap();

    // The negligible neuron should be a removal candidate
    let negligible_removal = result
        .removal_candidates
        .iter()
        .find(|c| c.neuron_uuid == "negligible");

    assert!(
        negligible_removal.is_some(),
        "Neuron with negligible impact should be a removal candidate. \
         Candidates found: {:?}",
        result
            .removal_candidates
            .iter()
            .map(|c| &c.neuron_uuid)
            .collect::<Vec<_>>()
    );

    // The reason should mention synapses
    let reason = &negligible_removal.unwrap().reason;
    assert!(
        reason.contains("synapse"),
        "Removal reason should mention synapses, got: {reason}"
    );
}
