//! REGRESSION TESTS for v0.1.127+ fixes
//!
//! These tests catch regressions if the v0.1.127+ fixes are accidentally reverted.
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
//! 2. **Unit conversion via fourth root**: activation_weighted_impact is in OUTPUT
//!    units, while savings is in SCORE units (error + complexity). We use a fourth
//!    root to bridge the gap:
//!
//!    ```
//!    threshold = savings^0.25
//!    ```
//!
//!    For savings ≈ 1.5e-7, threshold = (1.5e-7)^0.25 ≈ 6e-3 (0.6%)
//!    Removal is beneficial when: activation_weighted_impact < savings^0.25
//!
//! If any of these tests fail after a code change, the fix has regressed.

use neat_ai_discovery::focus::{calculate_removal_savings, rank_focus_neurons};
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
use tempfile::NamedTempFile;

/// Default growth cost matching NEAT-AI's typical value (1e-7 per Score.ts formula)
/// v0.1.145: Incorrectly changed to 0.01
/// v0.2.3 (Issue #132): Reverted to correct value of 1e-7
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

/// REGRESSION TEST: Removal candidates must use dynamic savings based on synapse count.
///
/// A neuron with many synapses saves more complexity when removed.
/// The criterion is: activation_weighted_impact < savings^0.25.
#[test]
fn regression_removal_uses_dynamic_threshold_based_on_synapse_count() {
    // Network with two neurons having different synapse counts:
    //
    // few-synapses: 1 incoming, 1 outgoing (2 total)
    //   savings = growthCost × (1 + 2/10) = 1.2e-7
    //   threshold = (1.2e-7)^0.25 ≈ 5.9e-3
    //
    // many-synapses: 3 incoming, 2 outgoing (5 total)
    //   savings = growthCost × (1 + 5/10) = 1.5e-7
    //   threshold = (1.5e-7)^0.25 ≈ 6.2e-3
    //
    // To be a removal candidate: activation_weighted_impact < savings^0.25
    // Using tiny weights (1e-8) so both neurons qualify as candidates.
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
                weight: 1e-8,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "few-synapses".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 1e-8, // Tiny compared to other inputs
                synapse_type: None,
            },
            // many-synapses: 3 in, 2 out
            SynapseJson {
                from_uuid: "input-0".to_string(),
                to_uuid: "many-synapses".to_string(),
                weight: 1e-8,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-1".to_string(),
                to_uuid: "many-synapses".to_string(),
                weight: 1e-8,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-2".to_string(),
                to_uuid: "many-synapses".to_string(),
                weight: 1e-8,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "many-synapses".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 1e-8,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "many-synapses".to_string(),
                to_uuid: "output-1".to_string(),
                weight: 1e-8,
                synapse_type: None,
            },
            // Add dominant inputs to outputs so test neurons are a small fraction
            SynapseJson {
                from_uuid: "input-0".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 1.0, // Dominates: 1e-8 / 1.0 ≈ 1e-8 fraction
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-1".to_string(),
                to_uuid: "output-1".to_string(),
                weight: 1.0,
                synapse_type: None,
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

    let result = rank_focus_neurons(file_path, &creature, None, None).unwrap();

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

    // CRITICAL: Both neurons must be found as removal candidates.
    // This is the actual regression check - if zero candidates are found, the feature is broken.
    let few_candidate = result
        .removal_candidates
        .iter()
        .find(|c| c.neuron_uuid == "few-synapses");
    let many_candidate = result
        .removal_candidates
        .iter()
        .find(|c| c.neuron_uuid == "many-synapses");

    assert!(
        few_candidate.is_some(),
        "\n\n\
        ╔══════════════════════════════════════════════════════════════════════════════════╗\n\
        ║  REGRESSION: 'few-synapses' not found as removal candidate!                       ║\n\
        ╠══════════════════════════════════════════════════════════════════════════════════╣\n\
        ║  Neuron has:                                                                      ║\n\
        ║    - impact = {:.2e}                                                              \n\
        ║    - activation_weighted_impact = {:.2e}                                          \n\
        ║                                                                                   ║\n\
        ║  Criterion (1 in + 1 out = 2 synapses):                                           ║\n\
        ║    savings = 1e-7 × 1.2 = 1.2e-7                                                  ║\n\
        ║    activation_weighted_impact < savings should pass                               ║\n\
        ║                                                                                   ║\n\
        ║  Found {} candidates: {:?}                                                        \n\
        ╚══════════════════════════════════════════════════════════════════════════════════╝\n\n",
        few.impact,
        few.activation_weighted_impact,
        result.removal_candidates.len(),
        result
            .removal_candidates
            .iter()
            .map(|c| &c.neuron_uuid)
            .collect::<Vec<_>>()
    );

    assert!(
        many_candidate.is_some(),
        "\n\n\
        ╔══════════════════════════════════════════════════════════════════════════════════╗\n\
        ║  REGRESSION: 'many-synapses' not found as removal candidate!                      ║\n\
        ╠══════════════════════════════════════════════════════════════════════════════════╣\n\
        ║  Neuron has:                                                                      ║\n\
        ║    - impact = {:.2e}                                                              \n\
        ║    - activation_weighted_impact = {:.2e}                                          \n\
        ║                                                                                   ║\n\
        ║  Criterion (3 in + 2 out = 5 synapses):                                           ║\n\
        ║    savings = 1e-7 × 1.5 = 1.5e-7                                                  ║\n\
        ║    activation_weighted_impact < savings should pass                               ║\n\
        ║                                                                                   ║\n\
        ║  Found {} candidates: {:?}                                                        \n\
        ╚══════════════════════════════════════════════════════════════════════════════════╝\n\n",
        many.impact,
        many.activation_weighted_impact,
        result.removal_candidates.len(),
        result
            .removal_candidates
            .iter()
            .map(|c| &c.neuron_uuid)
            .collect::<Vec<_>>()
    );

    // Verify the synapse counts are correct
    let few_candidate = few_candidate.unwrap();
    let many_candidate = many_candidate.unwrap();

    assert_eq!(
        few_candidate.incoming_synapses, 1,
        "few-synapses should have 1 incoming synapse"
    );
    assert_eq!(
        few_candidate.outgoing_synapses, 1,
        "few-synapses should have 1 outgoing synapse"
    );
    assert_eq!(
        many_candidate.incoming_synapses, 3,
        "many-synapses should have 3 incoming synapses"
    );
    assert_eq!(
        many_candidate.outgoing_synapses, 2,
        "many-synapses should have 2 outgoing synapses"
    );

    // Verify the neuron with more synapses has higher removal savings
    // (this is the core of the dynamic threshold feature)
    assert!(
        many_candidate.removal_savings > few_candidate.removal_savings,
        "many-synapses ({} synapses, savings={:.2e}) should have higher removal_savings than \
         few-synapses ({} synapses, savings={:.2e})",
        many_candidate.incoming_synapses + many_candidate.outgoing_synapses,
        many_candidate.removal_savings,
        few_candidate.incoming_synapses + few_candidate.outgoing_synapses,
        few_candidate.removal_savings
    );

    // Verify the savings match the NEAT-AI formula: growthCost × (1 + totalSynapses/10)
    // v0.2.3 (Issue #132): COST_OF_GROWTH is 1e-7 (correct value per Score.ts)
    let expected_few_savings = COST_OF_GROWTH * (1.0 + 2.0 / 10.0); // 1.2e-7
    let expected_many_savings = COST_OF_GROWTH * (1.0 + 5.0 / 10.0); // 1.5e-7

    assert!(
        (few_candidate.removal_savings - expected_few_savings).abs() < 1e-6,
        "few-synapses savings should be {:.2e}, got {:.2e}",
        expected_few_savings,
        few_candidate.removal_savings
    );
    assert!(
        (many_candidate.removal_savings - expected_many_savings).abs() < 1e-6,
        "many-synapses savings should be {:.2e}, got {:.2e}",
        expected_many_savings,
        many_candidate.removal_savings
    );
}

/// REGRESSION TEST: Threshold must use costOfGrowth (1e-7 per Score.ts).
///
/// v0.2.3 (Issue #132): costOfGrowth reverted to correct value of 1e-7
/// (v0.1.145 incorrectly changed it to 0.01)
///
/// Neurons are removal candidates when:
///   activation_weighted_impact < costOfGrowth (1e-7)
///
/// This test creates a neuron with impact ABOVE costOfGrowth that should NOT
/// be a removal candidate.
#[test]
fn regression_threshold_uses_cost_of_growth() {
    // Create a neuron with significant weight to output
    // With NORMALISED impact (v0.2.1+):
    //   structural_impact = 1.0 (sole input to output)
    //   mean_activation = 0.5
    //   activation_weighted_impact = 1.0 × 0.5 = 0.5 >> 1e-7
    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![
            NeuronJson {
                uuid: "high-impact".to_string(),
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
                to_uuid: "high-impact".to_string(),
                weight: 1.0,
                synapse_type: None,
            },
            // Hidden → output with moderate weight
            SynapseJson {
                from_uuid: "high-impact".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.5,
                synapse_type: None,
            },
        ],
    };

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // Records with mean_activation = 0.5
    let records = vec![
        DiscoverRecord::new(0, "high-impact".to_string(), Some(0.5), 0.5, vec![0.1]),
        DiscoverRecord::new(1, "high-impact".to_string(), Some(0.5), 0.5, vec![0.1]),
        DiscoverRecord::new(0, "output-0".to_string(), Some(0.5), 0.5, vec![0.1]),
        DiscoverRecord::new(1, "output-0".to_string(), Some(0.5), 0.5, vec![0.1]),
    ];
    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None, None).unwrap();

    // Find the neuron in ranked neurons
    let neuron = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "high-impact")
        .expect("high-impact should be in ranked neurons");

    // Verify impact is above costOfGrowth (1e-7)
    assert!(
        neuron.activation_weighted_impact > COST_OF_GROWTH,
        "activation_weighted_impact ({:.2e}) should be LARGER than costOfGrowth ({:.0e})",
        neuron.activation_weighted_impact,
        COST_OF_GROWTH
    );

    // Should NOT be a removal candidate (impact > costOfGrowth)
    let candidate = result
        .removal_candidates
        .iter()
        .find(|c| c.neuron_uuid == "high-impact");

    assert!(
        candidate.is_none(),
        "Neuron with activation_weighted_impact ({:.2e}) > costOfGrowth ({:.0e}) should NOT \
         be a removal candidate. Found candidates: {:?}",
        neuron.activation_weighted_impact,
        COST_OF_GROWTH,
        result
            .removal_candidates
            .iter()
            .map(|c| format!("{}: {:.2e}", c.neuron_uuid, c.activation_weighted_impact))
            .collect::<Vec<_>>()
    );
}

/// Test that the removal reason includes synapse count information.
#[test]
fn test_removal_reason_includes_synapse_savings() {
    // Network with a neuron that should be a removal candidate.
    // To achieve truly negligible impact, we need:
    //   - Small weight relative to other inputs to output
    //   - Low activation
    //
    // With normalised impact:
    //   structural_impact = |weight| / total_inbound × downstream_impact
    //   total_inbound to output = 1e-10 + 10.0 ≈ 10.0
    //   structural_impact ≈ 1e-10 / 10.0 × 1.0 = 1e-11
    //   activation_weighted_impact = 1e-11 × 1e-5 = 1e-16 << 1e-7 ✓
    let creature = CreatureJson {
        input: 2,
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
                weight: 1e-10,
                synapse_type: None,
            },
            // Negligible's synapse to output has tiny weight
            SynapseJson {
                from_uuid: "negligible".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 1e-10,
                synapse_type: None,
            },
            // Dominant synapse to output (makes negligible's fraction tiny)
            SynapseJson {
                from_uuid: "input-1".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 10.0,
                synapse_type: None,
            },
        ],
    };

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // Tiny activation ensures activation_weighted_impact << costOfGrowth
    let records = vec![
        DiscoverRecord::new(0, "negligible".to_string(), Some(1e-5), 1e-5, vec![0.1]),
        DiscoverRecord::new(1, "negligible".to_string(), Some(1e-5), 1e-5, vec![0.1]),
        DiscoverRecord::new(0, "output-0".to_string(), Some(0.5), 0.5, vec![0.1]),
        DiscoverRecord::new(1, "output-0".to_string(), Some(0.5), 0.5, vec![0.1]),
    ];
    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None, None).unwrap();

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
            .map(|c| format!("{}: {:.2e}", c.neuron_uuid, c.activation_weighted_impact))
            .collect::<Vec<_>>()
    );

    // The reason should mention synapses
    let reason = &negligible_removal.unwrap().reason;
    assert!(
        reason.contains("synapse"),
        "Removal reason should mention synapses, got: {reason}"
    );
}
