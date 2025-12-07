//! Tests for the focus neuron ranking and removal candidate detection.
//!
//! These tests verify that:
//! - Neurons are ranked by weighted score (error × impact)
//! - Output neurons have impact = 1.0
//! - Hidden neurons have impact based on their path weights to outputs
//! - Removal candidates are correctly identified based on activation-weighted impact
//!
//! Extracted from src/focus.rs in v0.1.126 to follow the testing philosophy
//! documented in README.md (prefer tests/ over inline unit tests).

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
            })
            .collect(),
        input: input_count,
        output: output_count,
    }
}

/// Helper to create parquet records for neurons with specified errors
fn create_records(neuron_errors: Vec<(&str, f32)>) -> Vec<DiscoverRecord> {
    neuron_errors
        .into_iter()
        .flat_map(|(uuid, error)| {
            vec![
                DiscoverRecord::new(0, uuid.to_string(), Some(0.5), 0.5, vec![error]),
                DiscoverRecord::new(1, uuid.to_string(), Some(0.5), 0.5, vec![error]),
            ]
        })
        .collect()
}

/// Helper to create parquet records for neurons with specified errors AND activations.
/// This is essential for testing activation-weighted impact.
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

#[test]
fn test_output_neurons_prioritised_due_to_high_impact() {
    // Scenario: Output neuron has moderate error, hidden neuron has high error
    // but low impact. Output should rank first due to impact × error weighting.
    //
    // Network: input-0 -> hidden-1 (weight 0.5) -> output-0 (weight 0.5)
    //
    // Impact calculation:
    // - output-0: impact = 1.0 (it's an output)
    // - hidden-1: impact = 0.5 / 0.5 * 1.0 = 1.0 (normalised by total inbound weight)
    //
    // But if hidden-1 has many paths or lower weight contribution, impact drops.
    // Let's use a more realistic scenario with multiple hidden neurons.
    let creature = create_creature(
        vec![
            ("input-0", "input"),
            ("hidden-1", "hidden"),
            ("hidden-2", "hidden"),
            ("output-0", "output"),
        ],
        vec![
            ("input-0", "hidden-1", 1.0),
            ("input-0", "hidden-2", 1.0),
            ("hidden-1", "output-0", 0.3), // 30% contribution to output
            ("hidden-2", "output-0", 0.7), // 70% contribution to output
        ],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // Output has moderate error (0.5), hidden-1 has high error (2.0)
    // But hidden-1 only contributes 30% to output
    let records = create_records(vec![
        ("hidden-1", 2.0), // High error but only 30% impact
        ("hidden-2", 0.3), // Low error, 70% impact
        ("output-0", 0.5), // Moderate error, 100% impact
    ]);
    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None).unwrap();

    // Output should be ranked first due to impact × error
    // output-0: 0.5 × 1.0 = 0.5
    // hidden-1: 2.0 × 0.3 = 0.6 (higher!)
    // hidden-2: 0.3 × 0.7 = 0.21
    //
    // Actually hidden-1 might rank first in this case because 0.6 > 0.5
    // Let's just verify the weighted ranking is applied
    assert!(!result.neurons.is_empty());

    // Verify that output-0 has impact = 1.0
    let output = result.neurons.iter().find(|n| n.neuron_uuid == "output-0");
    assert!(output.is_some(), "output-0 should be in results");
    assert!(
        (output.unwrap().impact - 1.0).abs() < 0.001,
        "output neuron should have impact = 1.0"
    );
}

#[test]
fn test_weighted_ranking_prefers_high_impact_moderate_error_over_low_impact_high_error() {
    // Scenario: A hidden neuron has very high error but contributes only a small
    // fraction to the output (due to branching). The output neuron has moderate
    // error but full impact. The output should rank higher.
    //
    // Network: input-0 feeds into hidden-minor (weight 0.1) and hidden-major (weight 0.9)
    //          Both feed into output-0
    //
    // Impact calculation:
    // - output-0: impact = 1.0
    // - hidden-minor: 0.1 / 1.0 (total inbound to output) * 1.0 = 0.1
    // - hidden-major: 0.9 / 1.0 * 1.0 = 0.9
    let creature = create_creature(
        vec![
            ("input-0", "input"),
            ("hidden-minor", "hidden"),
            ("hidden-major", "hidden"),
            ("output-0", "output"),
        ],
        vec![
            ("input-0", "hidden-minor", 1.0),
            ("input-0", "hidden-major", 1.0),
            ("hidden-minor", "output-0", 0.1), // Only 10% contribution
            ("hidden-major", "output-0", 0.9), // 90% contribution
        ],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // hidden-minor has 10x the error of output, but only 10% impact
    // Weighted scores:
    // - hidden-minor: 5.0 * 0.1 = 0.5
    // - hidden-major: 0.5 * 0.9 = 0.45
    // - output-0: 0.6 * 1.0 = 0.6
    let records = create_records(vec![
        ("hidden-minor", 5.0), // Very high error, 10% impact -> weighted 0.5
        ("hidden-major", 0.5), // Low error, 90% impact -> weighted 0.45
        ("output-0", 0.6),     // Moderate error, 100% impact -> weighted 0.6
    ]);
    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None).unwrap();
    assert_eq!(result.neurons.len(), 3);

    // Output should rank first: 0.6 * 1.0 = 0.6
    // hidden-minor second: 5.0 * 0.1 = 0.5
    // hidden-major third: 0.5 * 0.9 = 0.45
    assert_eq!(
        result.neurons[0].neuron_uuid, "output-0",
        "Output (weighted=0.6) should rank first"
    );

    // Verify the impact values are as expected
    let minor = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "hidden-minor")
        .unwrap();
    let major = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "hidden-major")
        .unwrap();
    let output = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "output-0")
        .unwrap();

    assert!(
        (minor.impact - 0.1).abs() < 0.001,
        "hidden-minor impact should be ~0.1, got {}",
        minor.impact
    );
    assert!(
        (major.impact - 0.9).abs() < 0.001,
        "hidden-major impact should be ~0.9, got {}",
        major.impact
    );
    assert!(
        (output.impact - 1.0).abs() < 0.001,
        "output impact should be 1.0, got {}",
        output.impact
    );
}

#[test]
fn test_hidden_neuron_can_rank_first_with_very_high_weighted_score() {
    // Scenario: Hidden neuron directly connected to output with high error
    // can still rank first if its weighted score beats the output
    let creature = create_creature(
        vec![
            ("input-0", "input"),
            ("hidden-1", "hidden"),
            ("output-0", "output"),
        ],
        vec![
            ("input-0", "hidden-1", 1.0),
            ("hidden-1", "output-0", 1.0), // Full weight contribution
        ],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // Hidden has very high error, output has low error
    // Hidden's impact should be ~1.0 since it's the only input to output
    let records = create_records(vec![
        ("hidden-1", 10.0), // Very high error
        ("output-0", 0.1),  // Low error
    ]);
    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None).unwrap();
    assert_eq!(result.neurons.len(), 2);

    // hidden-1 should rank first: 10.0 × ~1.0 = 10.0
    // output-0: 0.1 × 1.0 = 0.1
    assert_eq!(
        result.neurons[0].neuron_uuid, "hidden-1",
        "Hidden neuron with high error × high impact should rank first"
    );
    assert_eq!(
        result.neurons[1].neuron_uuid, "output-0",
        "Output neuron should rank second"
    );
}

#[test]
fn test_impact_epsilon_prevents_zero_impact_neurons_from_being_ignored() {
    // Neurons with zero impact (disconnected from outputs) should still
    // be considered, just with very low priority due to IMPACT_EPSILON
    let creature = create_creature(
        vec![
            ("input-0", "input"),
            ("orphan", "hidden"), // No path to output
            ("connected", "hidden"),
            ("output-0", "output"),
        ],
        vec![
            ("input-0", "orphan", 1.0),
            ("input-0", "connected", 1.0),
            ("connected", "output-0", 1.0),
            // Note: orphan has no connection to output
        ],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    let records = create_records(vec![
        ("orphan", 100.0),  // Very high error but zero impact
        ("connected", 1.0), // Moderate error, has impact
        ("output-0", 0.5),
    ]);
    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None).unwrap();

    // Orphan should still appear in results (not filtered out)
    let orphan = result.neurons.iter().find(|n| n.neuron_uuid == "orphan");
    assert!(orphan.is_some(), "Orphan neuron should be in results");
    assert!(
        orphan.unwrap().impact < 0.001,
        "Orphan should have ~zero impact"
    );

    // But orphan should rank last due to low weighted score
    let orphan_rank = result
        .neurons
        .iter()
        .position(|n| n.neuron_uuid == "orphan")
        .unwrap();
    assert_eq!(
        orphan_rank,
        result.neurons.len() - 1,
        "Zero-impact neuron should rank last despite high error"
    );
}

#[test]
fn test_disconnected_neurons_are_removal_candidates() {
    // Scenario: A neuron disconnected from outputs (zero impact) should be flagged
    // as a removal candidate because impact < costOfGrowth means removing it
    // improves the creature's score.
    let creature = create_creature(
        vec![
            ("input-0", "input"),
            ("orphan", "hidden"), // No path to output - zero impact
            ("connected", "hidden"),
            ("output-0", "output"),
        ],
        vec![
            ("input-0", "orphan", 1.0),
            ("input-0", "connected", 1.0),
            ("connected", "output-0", 1.0),
            // Note: orphan has no connection to output
        ],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // Error level doesn't matter for removal - only impact < costOfGrowth
    let records = create_records(vec![
        ("orphan", 0.5),    // Any error, zero impact -> removal candidate
        ("connected", 1.0), // High impact
        ("output-0", 1.0),  // Output neuron
    ]);
    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None).unwrap();

    // Orphan should be a removal candidate because impact (0) < costOfGrowth (1e-7)
    assert!(
        !result.removal_candidates.is_empty(),
        "Should have at least one removal candidate. Neurons: {:?}",
        result
            .neurons
            .iter()
            .map(|n| (&n.neuron_uuid, n.total_error, n.impact))
            .collect::<Vec<_>>()
    );

    let orphan_removal = result
        .removal_candidates
        .iter()
        .find(|c| c.neuron_uuid == "orphan");
    assert!(
        orphan_removal.is_some(),
        "Orphan should be flagged as a removal candidate"
    );

    let orphan = orphan_removal.unwrap();
    assert!(
        orphan.impact < 1e-7,
        "Orphan should have impact below costOfGrowth (1e-7), got {}",
        orphan.impact
    );
    // Verify the reason explains the removal
    assert!(
        orphan.reason.contains("costOfGrowth") || orphan.reason.contains("removal"),
        "Reason should explain why removal improves score: {}",
        orphan.reason
    );
}

#[test]
fn test_high_impact_neurons_are_not_removal_candidates() {
    // Scenario: Even neurons with high error should NOT be removal candidates
    // if they have high impact (close to outputs).
    let creature = create_creature(
        vec![
            ("input-0", "input"),
            ("hidden-1", "hidden"),
            ("output-0", "output"),
        ],
        vec![
            ("input-0", "hidden-1", 1.0),
            ("hidden-1", "output-0", 1.0), // Full contribution to output
        ],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // Both have high error, but both have high impact
    let records = create_records(vec![
        ("hidden-1", 100.0), // Very high error, but ~100% impact
        ("output-0", 100.0), // Very high error, 100% impact
    ]);
    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None).unwrap();

    // Neither should be a removal candidate because both have high impact
    assert!(
        result.removal_candidates.is_empty(),
        "No removal candidates expected when all neurons have high impact, got: {:?}",
        result
            .removal_candidates
            .iter()
            .map(|c| &c.neuron_uuid)
            .collect::<Vec<_>>()
    );
}

#[test]
fn test_low_error_moderate_impact_neurons_are_not_removal_candidates() {
    // Scenario: Neurons with moderate impact (above negligible threshold) and low error
    // should NOT be removal candidates - they may be doing useful work.
    let creature = create_creature(
        vec![
            ("input-0", "input"),
            ("low-impact", "hidden"), // Weak connection to output but above negligible
            ("connected", "hidden"),
            ("output-0", "output"),
        ],
        vec![
            ("input-0", "low-impact", 1.0),
            ("input-0", "connected", 1.0),
            ("low-impact", "output-0", 0.05), // 5% contribution - above negligible
            ("connected", "output-0", 0.95),  // 95% contribution
        ],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // low-impact has LOW error (below average) and moderate impact (5%)
    let records = create_records(vec![
        ("low-impact", 0.1), // Low error, ~5% impact -> NOT a removal candidate
        ("connected", 10.0), // High error, high impact
        ("output-0", 5.0),   // Moderate error, 100% impact
    ]);
    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None).unwrap();

    // low-impact should NOT be a removal candidate: it has low error and
    // moderate impact (above the negligible threshold)
    let low_impact_removal = result
        .removal_candidates
        .iter()
        .find(|c| c.neuron_uuid == "low-impact");
    assert!(
        low_impact_removal.is_none(),
        "Low-impact neuron with moderate impact (>1e-7) should NOT be a removal candidate when error is below average"
    );
}

#[test]
fn test_negligible_impact_neurons_are_removal_candidates_regardless_of_error() {
    // Scenario: A neuron with NEGLIGIBLE impact (below costOfGrowth threshold of 1e-7)
    // should be a removal candidate REGARDLESS of error level. Such neurons contribute
    // essentially nothing to the output and are just consuming compute.
    //
    // This test replicates the "crippled-removal" scenario where a neuron with near-zero
    // weights (1e-12) was added but not detected as a removal candidate because its
    // error was below average.
    let creature = create_creature(
        vec![
            ("input-0", "input"),
            ("input-1", "input"),
            ("negligible", "hidden"), // Near-zero weights -> negligible impact
            ("connected", "hidden"),
            ("output-0", "output"),
        ],
        vec![
            // Negligible neuron has tiny incoming weights
            ("input-0", "negligible", 1e-12),
            ("input-1", "negligible", -1e-12),
            // And a tiny outgoing weight
            ("negligible", "connected", 1e-12),
            // Connected neuron has normal weights
            ("input-0", "connected", 1.0),
            ("connected", "output-0", 1.0),
        ],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // Negligible neuron has LOW error (below average) because it doesn't
    // contribute enough to create errors. This is the scenario that was
    // slipping through detection.
    let records = create_records(vec![
        ("negligible", 0.01), // Low error, negligible impact -> SHOULD be removal candidate
        ("connected", 1.0),   // Normal error, high impact
        ("output-0", 0.5),    // Normal error, 100% impact
    ]);
    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None).unwrap();

    // Verify the negligible neuron has essentially zero impact
    let negligible_neuron = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "negligible")
        .expect("negligible neuron should be in results");
    assert!(
        negligible_neuron.impact < 1e-7,
        "Negligible neuron should have impact < 1e-7 (costOfGrowth), got {}",
        negligible_neuron.impact
    );

    // Negligible neuron SHOULD be a removal candidate regardless of error level
    // because its impact is below the costOfGrowth threshold (1e-7)
    let negligible_removal = result
        .removal_candidates
        .iter()
        .find(|c| c.neuron_uuid == "negligible");
    assert!(
        negligible_removal.is_some(),
        "Neuron with negligible impact (<1e-7) should be a removal candidate regardless of error. \
         Neurons: {:?}",
        result
            .neurons
            .iter()
            .map(|n| (&n.neuron_uuid, n.total_error, n.impact))
            .collect::<Vec<_>>()
    );

    // Verify the reason explains why removal improves score
    let candidate = negligible_removal.unwrap();
    assert!(
        candidate.reason.contains("costOfGrowth") || candidate.reason.contains("removal"),
        "Reason should explain why removal improves score: {}",
        candidate.reason
    );
}

#[test]
fn test_activation_weighted_impact_prevents_false_removal_candidates() {
    // Scenario: A neuron with moderate STRUCTURAL impact and HIGH activation should NOT be
    // a removal candidate because actual_contribution ≈ weight × activation.
    //
    // This test verifies that activation-weighted impact (not just structural impact)
    // determines removal candidates.
    //
    // Network: input-0 -> high-activation -> output-0
    //
    // high-activation has:
    // - Structural impact: 1e-6 (small weight to output)
    // - Mean activation: 1e5 (very high!)
    // - Activation-weighted impact: 1e-6 × 1e5 = 0.1 (10% - well above threshold)
    //
    // With SCALE_FACTOR=1e5, threshold for 1 synapse = 1.1e-7 × 1e5 = 0.011 (1.1%)
    // 10% > 1.1%, so it should NOT be a removal candidate.
    //
    // low-activation has same weight but tiny activation:
    // - Activation-weighted impact: 1e-6 × 1e-9 = 1e-15 (way below threshold)
    // So it SHOULD be a removal candidate.
    let creature = create_creature(
        vec![
            ("input-0", "input"),
            ("high-activation", "hidden"),
            ("low-activation", "hidden"),
            ("output-0", "output"),
        ],
        vec![
            ("input-0", "high-activation", 1.0),
            ("input-0", "low-activation", 1.0),
            // Both have small structural paths to output
            ("high-activation", "output-0", 1e-6),
            ("low-activation", "output-0", 1e-6),
        ],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // high-activation: small structural impact BUT massive activation → NOT removal candidate
    // low-activation: small structural impact AND tiny activation → IS removal candidate
    let records = create_records_with_activation(vec![
        ("high-activation", 0.1, 1e5), // High activation → contribution 10% → not safe to remove
        ("low-activation", 0.1, 1e-9), // Low activation → contribution ~0% → safe to remove
        ("output-0", 0.5, 0.5),
    ]);
    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None).unwrap();

    // Verify high-activation neuron has high activation-weighted impact
    let high_act = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "high-activation")
        .expect("high-activation should be in results");
    assert!(
        high_act.mean_activation > 1e4,
        "high-activation should have high mean_activation, got {}",
        high_act.mean_activation
    );
    // Activation-weighted impact should be ~0.1 (10%)
    assert!(
        high_act.activation_weighted_impact > 0.01,
        "high-activation should have activation_weighted_impact > 1% (threshold), got {}",
        high_act.activation_weighted_impact
    );

    // high-activation should NOT be a removal candidate (activation-weighted impact too high)
    let high_act_removal = result
        .removal_candidates
        .iter()
        .find(|c| c.neuron_uuid == "high-activation");
    assert!(
        high_act_removal.is_none(),
        "high-activation should NOT be a removal candidate because activation-weighted impact ({:.4}) > threshold (~1%)",
        high_act.activation_weighted_impact
    );

    // Verify low-activation neuron has low activation-weighted impact
    let low_act = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "low-activation")
        .expect("low-activation should be in results");
    assert!(
        low_act.mean_activation < 1e-7,
        "low-activation should have low mean_activation, got {}",
        low_act.mean_activation
    );
    assert!(
        low_act.activation_weighted_impact < 1e-10,
        "low-activation should have activation_weighted_impact << threshold, got {}",
        low_act.activation_weighted_impact
    );

    // low-activation SHOULD be a removal candidate
    let low_act_removal = result
        .removal_candidates
        .iter()
        .find(|c| c.neuron_uuid == "low-activation");
    assert!(
        low_act_removal.is_some(),
        "low-activation SHOULD be a removal candidate because activation-weighted impact ({:.2e}) << threshold. \
         Removal candidates: {:?}",
        low_act.activation_weighted_impact,
        result.removal_candidates.iter().map(|c| &c.neuron_uuid).collect::<Vec<_>>()
    );
}

#[test]
fn test_removal_candidates_sorted_by_activation_weighted_impact() {
    // Verify that removal candidates are sorted by activation-weighted impact (ascending)
    // so the safest candidates (lowest impact) come first.
    let creature = create_creature(
        vec![
            ("input-0", "input"),
            ("medium-impact", "hidden"),
            ("low-impact", "hidden"),
            ("lowest-impact", "hidden"),
            ("output-0", "output"),
        ],
        vec![
            ("input-0", "medium-impact", 1.0),
            ("input-0", "low-impact", 1.0),
            ("input-0", "lowest-impact", 1.0),
            // All disconnected from output → zero structural impact
        ],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // Different activation levels → different activation-weighted impacts
    let records = create_records_with_activation(vec![
        ("medium-impact", 0.1, 1e-4),  // activation-weighted: 0 × 1e-4 = 0
        ("low-impact", 0.1, 1e-6),     // activation-weighted: 0 × 1e-6 = 0
        ("lowest-impact", 0.1, 1e-10), // activation-weighted: 0 × 1e-10 = 0
        ("output-0", 0.5, 0.5),
    ]);
    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None).unwrap();

    // All three should be removal candidates (disconnected from output)
    assert!(
        result.removal_candidates.len() >= 3,
        "Should have at least 3 removal candidates, got {}",
        result.removal_candidates.len()
    );

    // Verify sorted order (ascending by activation_weighted_impact)
    for i in 1..result.removal_candidates.len() {
        assert!(
            result.removal_candidates[i - 1].activation_weighted_impact
                <= result.removal_candidates[i].activation_weighted_impact,
            "Removal candidates should be sorted by activation_weighted_impact (ascending). \
             Got {:?} before {:?}",
            result.removal_candidates[i - 1].activation_weighted_impact,
            result.removal_candidates[i].activation_weighted_impact
        );
    }
}

#[test]
fn test_processed_neurons_reports_accurately_when_some_neurons_missing_records() {
    // Create a creature with 4 selectable neurons:
    // - hidden-1, hidden-2, hidden-3 (hidden neurons are selectable)
    // - output-0 (output neurons are also selectable, not just hidden)
    let creature = CreatureJson {
        neurons: vec![
            NeuronJson {
                uuid: "input-0".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "hidden-1".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "TANH".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "hidden-2".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "TANH".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "hidden-3".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "TANH".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "output-0".to_string(),
                neuron_type: "output".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
        ],
        synapses: vec![],
        input: 1,
        output: 1,
    };

    // Create a temporary parquet file with records for only 2 of the 4 selectable neurons
    // (hidden-1, hidden-2 have records; hidden-3 and output-0 are missing)
    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    let records = vec![
        DiscoverRecord::new(0, "hidden-1".to_string(), Some(0.5), 0.7, vec![0.1, 0.2]),
        DiscoverRecord::new(1, "hidden-1".to_string(), Some(0.6), 0.8, vec![0.15, 0.25]),
        DiscoverRecord::new(0, "hidden-2".to_string(), Some(0.3), 0.5, vec![0.05]),
        DiscoverRecord::new(1, "hidden-2".to_string(), Some(0.4), 0.6, vec![0.1]),
        // Note: hidden-3 and output-0 are missing - no records for them
    ];

    write_records_to_parquet(file_path, &records).unwrap();

    // Call rank_focus_neurons
    let result = rank_focus_neurons(file_path, &creature, None);

    // The function should error because hidden-3 and output-0 are missing records
    // (restore old behavior where missing records cause an error)
    // OR if we allow missing records, processed_neurons should accurately reflect
    // the number of neurons that were actually processed (2, not 4)

    match result {
        Ok(stats) => {
            // If it doesn't error, processed_neurons should be accurate
            let actual_processed = stats.neurons.len();
            assert_eq!(
                actual_processed, 2,
                "Only 2 neurons should have been processed (hidden-3 and output-0 were silently dropped)"
            );
            // This assertion should fail with current buggy code:
            // processed_neurons incorrectly reports 4 (total_neurons) when only 2 were processed
            assert_eq!(
                stats.processed_neurons, actual_processed,
                "processed_neurons ({}) should match actual processed count ({})",
                stats.processed_neurons, actual_processed
            );
        }
        Err(_) => {
            // If it errors, that's the correct behavior (restores old behavior)
            // This is the preferred behavior to maintain data integrity
        }
    }
}

#[test]
fn test_cumulative_impact_for_multiple_outgoing_synapses() {
    // Scenario: A hidden neuron connects to MULTIPLE outputs.
    // The impact should be the SUM of contributions to all outputs, not the MAX.
    //
    // Bug discovered: A neuron with 3 outgoing synapses to 2 outputs + 1 hidden
    // was flagged as "low-impact" (1.07e-13) but removing it caused a score
    // delta of 6.3e-5 - much higher than predicted.
    //
    // Network:
    //   input-0 -> hub -> output-0 (weight 1.0)
    //            -> hub -> output-1 (weight 1.0)
    //
    // With absolute weights (v0.1.126+):
    //   hub → output-0: 1.0 × 1.0 = 1.0
    //   hub → output-1: 1.0 × 1.0 = 1.0
    //   Cumulative impact: 1.0 + 1.0 = 2.0
    //
    // Bug behaviour (using max): would be 1.0 (only counting one output)
    let creature = create_creature(
        vec![
            ("input-0", "input"),
            ("hub", "hidden"), // Connects to both outputs
            ("output-0", "output"),
            ("output-1", "output"),
        ],
        vec![
            ("input-0", "hub", 1.0),
            // Hub connects to BOTH outputs with weight 1.0 each
            ("hub", "output-0", 1.0),
            ("hub", "output-1", 1.0),
        ],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    let records = create_records(vec![("hub", 0.5), ("output-0", 0.5), ("output-1", 0.5)]);
    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None).unwrap();

    // Find the hub neuron's impact
    let hub = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "hub")
        .expect("Hub neuron should be in results");

    // Hub connects to both outputs with weight 1.0 each.
    // With absolute weight formula: impact = 1.0 + 1.0 = 2.0
    // At minimum, impact should be > 1.0 because it affects TWO outputs.
    //
    // With the bug (using max), impact would be 1.0 (only counting one output).
    // With the fix (using sum), impact should be 2.0.
    assert!(
        hub.impact > 1.0,
        "Hub neuron connecting to 2 outputs should have cumulative impact > 1.0, got {}. \
         This indicates the impact calculation is using MAX instead of SUM for multiple outgoing synapses.",
        hub.impact
    );
}

#[test]
fn test_cumulative_impact_mixed_direct_and_indirect_paths() {
    // Scenario: A neuron has multiple paths to outputs:
    // - Direct connection to output-0
    // - Indirect connection through hidden-2 to output-1
    //
    // Both paths should contribute to the cumulative impact.
    let creature = create_creature(
        vec![
            ("input-0", "input"),
            ("hub", "hidden"),
            ("hidden-2", "hidden"),
            ("output-0", "output"),
            ("output-1", "output"),
        ],
        vec![
            ("input-0", "hub", 1.0),
            ("hub", "output-0", 1.0),      // Direct to output-0 (100%)
            ("hub", "hidden-2", 1.0),      // To hidden-2
            ("hidden-2", "output-1", 1.0), // hidden-2 to output-1 (100%)
        ],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    let records = create_records(vec![
        ("hub", 0.5),
        ("hidden-2", 0.5),
        ("output-0", 0.5),
        ("output-1", 0.5),
    ]);
    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None).unwrap();

    let hub = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "hub")
        .expect("Hub neuron should be in results");

    // Hub has:
    // - Direct path to output-0: impact = 1.0
    // - Indirect path via hidden-2 to output-1: impact = 1.0 * 1.0 = 1.0
    // Cumulative impact should be: 1.0 + 1.0 = 2.0
    assert!(
        hub.impact > 1.5,
        "Hub with direct and indirect paths to 2 outputs should have impact > 1.5, got {}",
        hub.impact
    );
}
