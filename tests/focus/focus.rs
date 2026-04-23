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

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use neat_ai_discovery::focus::rank_focus_neurons;
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
use serial_test::serial;
use tempfile::NamedTempFile;

/// RAII guard that disables the Issue #1142 noise floor for tests whose
/// scenarios deliberately use tiny magnitudes (below 1e-5) to exercise
/// the pre-#1142 impact/savings contract. Tests using this guard must be
/// marked `#[serial]` so env var access is not racy.
struct NoiseFloorOffGuard {
    previous: Option<String>,
}

impl NoiseFloorOffGuard {
    fn new() -> Self {
        let key = "NEAT_AI_DISCOVERY_REMOVE_LOW_IMPACT_NOISE_FLOOR";
        let previous = std::env::var(key).ok();
        // SAFETY: serialised via #[serial] — no concurrent env access.
        unsafe {
            std::env::set_var(key, "0");
        }
        Self { previous }
    }
}

impl Drop for NoiseFloorOffGuard {
    fn drop(&mut self) {
        let key = "NEAT_AI_DISCOVERY_REMOVE_LOW_IMPACT_NOISE_FLOOR";
        // SAFETY: serialised via #[serial] — no concurrent env access.
        match &self.previous {
            Some(v) => unsafe { std::env::set_var(key, v) },
            None => unsafe { std::env::remove_var(key) },
        }
    }
}

/// Helper to create a simple creature with specified neurons and synapses
///
/// Note: Input neurons in the `neurons` parameter are used only to count `input`.
/// They are NOT included in `creature.neurons` as per the NEAT-AI data model -
/// input neurons are represented only by the `creature.input` count.
fn create_creature(
    neurons: Vec<(&str, &str)>,       // (uuid, type)
    synapses: Vec<(&str, &str, f32)>, // (from, to, weight)
) -> CreatureJson {
    let input_count = neurons.iter().filter(|(_, t)| *t == "input").count();
    let output_count = neurons.iter().filter(|(_, t)| *t == "output").count();
    CreatureJson {
        // Filter out input neurons - they are only represented by creature.input count
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

    let result = rank_focus_neurons(file_path, &creature, None, None).unwrap();

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

    let result = rank_focus_neurons(file_path, &creature, None, None).unwrap();
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

    let result = rank_focus_neurons(file_path, &creature, None, None).unwrap();
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

    let result = rank_focus_neurons(file_path, &creature, None, None).unwrap();

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
#[serial]
fn test_disconnected_neurons_are_removal_candidates() {
    let _guard = NoiseFloorOffGuard::new();
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

    let result = rank_focus_neurons(file_path, &creature, None, None).unwrap();

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
        "Orphan should have negligible impact (< 1e-7), got {}",
        orphan.impact
    );
    // Verify the reason contains impact info
    assert!(
        orphan.reason.contains("Impact") || orphan.reason.contains("saves"),
        "Reason should contain impact info: {}",
        orphan.reason
    );
}

#[test]
fn test_high_impact_neurons_not_returned_as_removal_candidates() {
    // Scenario: High impact neurons (impact >= costOfGrowth) should NOT be
    // returned as removal candidates - only neurons below threshold qualify.
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

    // Both have high error, but both have high impact (well above costOfGrowth 1e-7)
    let records = create_records(vec![
        ("hidden-1", 100.0), // Very high error, ~100% impact
        ("output-0", 100.0), // Very high error, 100% impact
    ]);
    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None, None).unwrap();

    // High impact neurons should NOT be removal candidates
    // Both hidden-1 and output-0 have activation_weighted_impact >> 0.01
    assert!(
        result.removal_candidates.is_empty(),
        "High impact neurons should NOT be removal candidates. Found: {:?}",
        result
            .removal_candidates
            .iter()
            .map(|c| (&c.neuron_uuid, c.activation_weighted_impact))
            .collect::<Vec<_>>()
    );
}

#[test]
fn test_only_low_impact_neurons_returned_as_removal_candidates() {
    // Scenario: Only neurons with activation_weighted_impact < costOfGrowth (1e-7)
    // are returned as removal candidates. High impact neurons are filtered out.
    let creature = create_creature(
        vec![
            ("input-0", "input"),
            ("low-impact", "hidden"), // Weak connection to output
            ("connected", "hidden"),
            ("output-0", "output"),
        ],
        vec![
            ("input-0", "low-impact", 1.0),
            ("input-0", "connected", 1.0),
            ("low-impact", "output-0", 0.05), // 5% contribution
            ("connected", "output-0", 0.95),  // 95% contribution
        ],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // Both neurons have normal activations (0.5), so their activation_weighted_impact
    // will be structural_impact × 0.5:
    // - low-impact: 0.05 × 0.5 = 0.025 (>> 1e-7, NOT a candidate)
    // - connected: 0.95 × 0.5 = 0.475 (>> 1e-7, NOT a candidate)
    let records = create_records(vec![
        ("low-impact", 0.1), // Low error, ~5% impact × 0.5 activation = 0.025
        ("connected", 10.0), // High error, 95% impact × 0.5 activation = 0.475
        ("output-0", 5.0),   // 100% impact × 0.5 activation = 0.5
    ]);
    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None, None).unwrap();

    // All neurons have impact >> costOfGrowth, so none should be removal candidates
    assert!(
        result.removal_candidates.is_empty(),
        "Neurons with activation_weighted_impact >> costOfGrowth should NOT be removal candidates. Found: {:?}",
        result
            .removal_candidates
            .iter()
            .map(|c| (&c.neuron_uuid, c.activation_weighted_impact))
            .collect::<Vec<_>>()
    );
}

#[test]
#[serial]
fn test_negligible_impact_neurons_sorted_first_as_best_removal_candidates() {
    let _guard = NoiseFloorOffGuard::new();
    // Scenario: A neuron with NEGLIGIBLE impact should be sorted FIRST in the
    // removal candidates list (lowest impact = best candidate for removal).
    //
    // This test replicates the "crippled-removal" scenario where a neuron with near-zero
    // weights (1e-12) should be the top removal candidate.
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
    // contribute enough to create errors.
    let records = create_records(vec![
        ("negligible", 0.01), // Low error, negligible impact
        ("connected", 1.0),   // Normal error, high impact
        ("output-0", 0.5),    // Normal error, 100% impact
    ]);
    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None, None).unwrap();

    // Verify the negligible neuron has essentially zero impact
    let negligible_neuron = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "negligible")
        .expect("negligible neuron should be in results");
    assert!(
        negligible_neuron.impact < 1e-7,
        "Negligible neuron should have negligible impact (< 1e-7), got {}",
        negligible_neuron.impact
    );

    // Negligible neuron should be in removal candidates
    let negligible_removal = result
        .removal_candidates
        .iter()
        .find(|c| c.neuron_uuid == "negligible");
    assert!(
        negligible_removal.is_some(),
        "Neuron with negligible impact should be in removal candidates"
    );

    // Negligible neuron should be sorted FIRST (lowest impact)
    assert_eq!(
        result.removal_candidates[0].neuron_uuid, "negligible",
        "Negligible neuron should be first removal candidate (lowest impact)"
    );

    // Verify the reason contains impact info
    let candidate = negligible_removal.unwrap();
    assert!(
        candidate.reason.contains("Impact") || candidate.reason.contains("saves"),
        "Reason should explain why removal improves score: {}",
        candidate.reason
    );
}

#[test]
#[serial]
fn test_activation_weighted_impact_prevents_false_removal_candidates() {
    let _guard = NoiseFloorOffGuard::new();
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

    let result = rank_focus_neurons(file_path, &creature, None, None).unwrap();

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
        high_act.activation_weighted_impact > 1e-7,
        "high-activation should have activation_weighted_impact > 1e-7 (threshold), got {}",
        high_act.activation_weighted_impact
    );

    // high-activation should NOT be a removal candidate (impact 0.1 >> 1e-7)
    let high_act_removal = result
        .removal_candidates
        .iter()
        .find(|c| c.neuron_uuid == "high-activation");
    assert!(
        high_act_removal.is_none(),
        "high-activation should NOT be a removal candidate (impact {:.2e} >> costOfGrowth 1e-7)",
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
        low_act.activation_weighted_impact < 1e-8,
        "low-activation should have very small activation_weighted_impact, got {}",
        low_act.activation_weighted_impact
    );

    // low-activation SHOULD be a removal candidate (impact 1e-15 << 1e-7)
    let low_act_removal = result
        .removal_candidates
        .iter()
        .find(|c| c.neuron_uuid == "low-activation");
    assert!(
        low_act_removal.is_some(),
        "low-activation SHOULD be a removal candidate (impact {:.2e} < costOfGrowth 1e-7)",
        low_act.activation_weighted_impact
    );

    // Only low-activation should be in removal candidates
    assert_eq!(
        result.removal_candidates.len(),
        1,
        "Only low-activation should be a removal candidate"
    );
}

#[test]
#[serial]
fn test_removal_candidates_sorted_by_activation_weighted_impact() {
    let _guard = NoiseFloorOffGuard::new();
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

    let result = rank_focus_neurons(file_path, &creature, None, None).unwrap();

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
    // Note: Input neurons are NOT included in creature.neurons as per the NEAT-AI data model.
    let creature = CreatureJson {
        neurons: vec![
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
    let result = rank_focus_neurons(file_path, &creature, None, None);

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

    let result = rank_focus_neurons(file_path, &creature, None, None).unwrap();

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
fn test_non_finite_activations_handled_gracefully() {
    // REGRESSION TEST: The refactored mean_absolute_activation_from_records function
    // must handle NaN and Infinity activations without corrupting the ranking.
    //
    // If any record has activation = NaN or Infinity, the sum() would return NaN,
    // which propagates through activation_weighted_impact and corrupts sorting.
    //
    // The old implementation properly filtered out non-finite values using is_finite().
    let creature = create_creature(
        vec![
            ("input-0", "input"),
            ("nan-activation", "hidden"),
            ("inf-activation", "hidden"),
            ("normal", "hidden"),
            ("output-0", "output"),
        ],
        vec![
            ("input-0", "nan-activation", 1.0),
            ("input-0", "inf-activation", 1.0),
            ("input-0", "normal", 1.0),
            ("nan-activation", "output-0", 0.1),
            ("inf-activation", "output-0", 0.1),
            ("normal", "output-0", 0.8),
        ],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // Create records with non-finite activations
    let records = vec![
        // NaN activation record
        DiscoverRecord::new(
            0,
            "nan-activation".to_string(),
            Some(0.5),
            f32::NAN,
            vec![0.1],
        ),
        DiscoverRecord::new(1, "nan-activation".to_string(), Some(0.5), 0.5, vec![0.1]),
        // Infinity activation record
        DiscoverRecord::new(
            0,
            "inf-activation".to_string(),
            Some(0.5),
            f32::INFINITY,
            vec![0.1],
        ),
        DiscoverRecord::new(1, "inf-activation".to_string(), Some(0.5), 0.5, vec![0.1]),
        // Normal activation records
        DiscoverRecord::new(0, "normal".to_string(), Some(0.5), 0.5, vec![0.1]),
        DiscoverRecord::new(1, "normal".to_string(), Some(0.5), 0.5, vec![0.1]),
        // Output records
        DiscoverRecord::new(0, "output-0".to_string(), Some(0.5), 0.5, vec![0.1]),
        DiscoverRecord::new(1, "output-0".to_string(), Some(0.5), 0.5, vec![0.1]),
    ];
    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None, None).unwrap();

    // Verify the ranking is not corrupted by NaN
    assert!(
        !result.neurons.is_empty(),
        "Should have neurons in results despite non-finite activations"
    );

    // Verify no NaN values in activation_weighted_impact
    for neuron in &result.neurons {
        assert!(
            neuron.activation_weighted_impact.is_finite(),
            "activation_weighted_impact for {} should be finite, got {}",
            neuron.neuron_uuid,
            neuron.activation_weighted_impact
        );
        assert!(
            neuron.mean_activation.is_finite(),
            "mean_activation for {} should be finite, got {}",
            neuron.neuron_uuid,
            neuron.mean_activation
        );
    }

    // Verify removal candidates are sorted correctly (not corrupted by NaN)
    for i in 1..result.removal_candidates.len() {
        let prev = result.removal_candidates[i - 1].activation_weighted_impact;
        let curr = result.removal_candidates[i].activation_weighted_impact;
        assert!(
            prev <= curr,
            "Removal candidates should be sorted ascending by impact. \
             Position {}: {} vs position {}: {}",
            i - 1,
            prev,
            i,
            curr
        );
    }

    // Verify the normal neuron has expected values
    let normal = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "normal")
        .expect("normal neuron should be in results");
    assert!(
        (normal.mean_activation - 0.5).abs() < 0.001,
        "normal neuron should have mean_activation = 0.5, got {}",
        normal.mean_activation
    );
}

#[test]
fn test_all_non_finite_activations_returns_zero_mean() {
    // Edge case: ALL activation values are non-finite.
    // The function should return 0.0 (same as empty records).
    let creature = create_creature(
        vec![
            ("input-0", "input"),
            ("all-nan", "hidden"),
            ("output-0", "output"),
        ],
        vec![("input-0", "all-nan", 1.0), ("all-nan", "output-0", 1.0)],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // All records for all-nan neuron have non-finite activations
    let records = vec![
        DiscoverRecord::new(0, "all-nan".to_string(), Some(0.5), f32::NAN, vec![0.1]),
        DiscoverRecord::new(
            1,
            "all-nan".to_string(),
            Some(0.5),
            f32::INFINITY,
            vec![0.1],
        ),
        DiscoverRecord::new(
            2,
            "all-nan".to_string(),
            Some(0.5),
            f32::NEG_INFINITY,
            vec![0.1],
        ),
        DiscoverRecord::new(0, "output-0".to_string(), Some(0.5), 0.5, vec![0.1]),
        DiscoverRecord::new(1, "output-0".to_string(), Some(0.5), 0.5, vec![0.1]),
        DiscoverRecord::new(2, "output-0".to_string(), Some(0.5), 0.5, vec![0.1]),
    ];
    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None, None).unwrap();

    // Find the all-nan neuron
    let all_nan = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "all-nan")
        .expect("all-nan neuron should be in results");

    // When all activations are non-finite, mean_activation should be 0.0
    assert!(
        all_nan.mean_activation.is_finite(),
        "mean_activation should be finite even when all inputs are NaN/Inf, got {}",
        all_nan.mean_activation
    );
    assert_eq!(
        all_nan.mean_activation, 0.0,
        "mean_activation should be 0.0 when all inputs are NaN/Inf, got {}",
        all_nan.mean_activation
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

    let result = rank_focus_neurons(file_path, &creature, None, None).unwrap();

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

/// Issue #117: `expected_error_reduction` should reflect the creature's expected error change,
/// NOT the neuron's error. For removal candidates, this should be based on `activation_weighted_impact`.
///
/// The bug was that `total_error` (the neuron's average error) was being used as
/// `expectedErrorReduction` on the TypeScript side, leading to predictions like 27%
/// when the actual error change was ~0%.
///
/// The fix is to provide `expected_error_reduction` that reflects the ACTUAL expected
/// creature-level error change from removing the neuron. For removal candidates (low-impact
/// neurons), this should be very small - approximately equal to `activation_weighted_impact`.
#[test]
#[serial]
fn test_removal_candidate_expected_error_reduction_is_impact_based_not_neuron_error() {
    let _guard = NoiseFloorOffGuard::new();
    // Scenario: A neuron with HIGH error (0.27 normalised) but NEGLIGIBLE impact (< 1e-7).
    // The bug would predict 27% error reduction, but actual reduction is tiny.
    //
    // Network: input-0 -> negligible -> output-0 (very tiny weights)
    //          input-0 -> output-0 (direct, large weight)
    //
    // Impact calculation (v0.2.3 with 1e-7 threshold):
    // - negligible → output-0: weight 1e-8 / total_inbound(1.0 + 1e-8) × 1.0 ≈ 1e-8
    // - activation_weighted_impact = 1e-8 × 0.5 = 5e-9 < 1e-7 ✓
    let creature = create_creature(
        vec![
            ("input-0", "input"),
            ("negligible", "hidden"), // Negligible impact due to very tiny weights
            ("output-0", "output"),
        ],
        vec![
            // Negligible neuron has very tiny weights (to get impact < 1e-7)
            ("input-0", "negligible", 1e-8),
            ("negligible", "output-0", 1e-8),
            // Direct path dominates
            ("input-0", "output-0", 1.0),
        ],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // Negligible neuron has HIGH error (0.27 or 27% when normalised)
    // but very tiny activation-weighted impact (< 1e-7)
    let records = create_records_with_activation(vec![
        ("negligible", 0.27, 0.5), // High error, moderate activation -> very tiny impact
        ("output-0", 0.5, 0.8),    // Normal error for output
    ]);
    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None, None).unwrap();

    // The negligible neuron should be a removal candidate
    let negligible_removal = result
        .removal_candidates
        .iter()
        .find(|c| c.neuron_uuid == "negligible");
    assert!(
        negligible_removal.is_some(),
        "Negligible neuron should be a removal candidate. All candidates: {:?}",
        result
            .removal_candidates
            .iter()
            .map(|c| &c.neuron_uuid)
            .collect::<Vec<_>>()
    );

    let candidate = negligible_removal.unwrap();

    // CRITICAL: The expected_error_reduction should be tiny (based on impact),
    // NOT the neuron's error (0.27).
    //
    // activation_weighted_impact = structural_impact × mean_activation
    // structural_impact ≈ 1e-6 × 1e-6 = 1e-12 (tiny!)
    // mean_activation = 0.5
    // activation_weighted_impact ≈ 5e-13
    //
    // The expected_error_reduction should be approximately activation_weighted_impact
    // (the actual contribution being removed).
    assert!(
        candidate.expected_error_reduction < 0.01, // Less than 1%
        "expected_error_reduction should be tiny (based on impact), not the neuron's error (0.27). \
         Got: {:.6} ({:.4}%). Neuron error was: {:.2}. Impact: {:.2e}. \
         Bug #117: This would have been ~0.27 (27%) if using neuron error!",
        candidate.expected_error_reduction,
        candidate.expected_error_reduction * 100.0,
        candidate.total_error,
        candidate.activation_weighted_impact
    );

    // The expected_error_reduction should be in the same order of magnitude as impact
    // (within 10x) since removing a low-impact neuron has minimal effect on error.
    let impact_ratio = if candidate.activation_weighted_impact > 1e-10 {
        candidate.expected_error_reduction / candidate.activation_weighted_impact
    } else {
        // For extremely tiny impacts, just verify it's also tiny
        1.0
    };
    assert!(
        impact_ratio < 100.0,
        "expected_error_reduction ({:.2e}) should be within ~100x of activation_weighted_impact ({:.2e}). \
         Ratio: {:.1}x. Bug #117: This ratio would be ~billions if using neuron error!",
        candidate.expected_error_reduction,
        candidate.activation_weighted_impact,
        impact_ratio
    );

    // Verify total_error is high (this is what was incorrectly used before)
    assert!(
        candidate.total_error > 0.1,
        "Neuron should have high error (0.27), got {:.4}",
        candidate.total_error
    );
}

/// Issue #235: Return ALL removal candidates expected to improve the creature's score.
///
/// The only filter should be: "will this candidate improve the creature's score?"
/// A removal improves score when: `removal_savings` > `activation_weighted_impact`
///
/// Previously, we filtered on `activation_weighted_impact` < `cost_of_growth_threshold`,
/// but this missed neurons where `removal_savings` (due to many synapses) exceeds
/// the neuron's contribution even when `activation_weighted_impact` is above threshold.
///
/// Mathematical explanation:
/// - `cost_of_growth_threshold` = 1e-7 (default)
/// - `removal_savings` = `cost_of_growth` * (1 + synapses/10)
/// - For neuron with impact 2e-7 and 20 synapses:
///   - Old filter: 2e-7 < 1e-7 = FALSE → rejected
///   - `removal_savings` = 1e-7 * (1 + 2) = 3e-7
///   - New filter: 3e-7 > 2e-7 = TRUE → accepted (removal improves score!)
///
/// This test creates such a scenario.
#[test]
fn test_issue_235_return_all_removal_candidates_expected_to_improve_score() {
    // Scenario: A neuron has activation_weighted_impact ABOVE costOfGrowth threshold (1e-7)
    // but has many synapses, so its removal_savings exceeds its impact.
    // Such neurons SHOULD be returned as removal candidates.
    //
    // We create "many-synapses" with:
    // - 50 synapses (25 incoming, 25 outgoing)
    // - structural_impact = 0.1 (it contributes 10% via direct path to output)
    // - activation = 1e-5 (very low)
    // - activation_weighted_impact = 0.1 × 1e-5 = 1e-6 (above threshold 1e-7)
    //
    // Removal analysis:
    // - removal_savings = 1e-7 × (1 + 50/10) = 1e-7 × 6 = 6e-7
    // - activation_weighted_impact = 1e-6
    // - Since 6e-7 < 1e-6, removal does NOT improve score for this neuron
    //
    // Let's instead use many MORE synapses to ensure savings > impact:
    // - 100 synapses (50 incoming, 50 outgoing)
    // - removal_savings = 1e-7 × (1 + 100/10) = 1e-7 × 11 = 1.1e-6
    // - activation_weighted_impact = 1e-6 (from above)
    // - Since 1.1e-6 > 1e-6, removal DOES improve score!
    //
    // With old filter: 1e-6 < 1e-7 = FALSE → not returned (BUG!)
    // With new filter: 1.1e-6 > 1e-6 = TRUE → returned (CORRECT!)
    let mut synapses = vec![];
    // Many incoming synapses to many-synapses neuron (50 synapses)
    for i in 0..50 {
        synapses.push((
            format!("input-{i}").leak() as &str,
            "many-synapses",
            0.1 / 50.0, // tiny weights from each
        ));
    }
    // Many outgoing synapses from many-synapses neuron (50 synapses) - but all to same output
    // Each outgoing synapse has tiny weight to keep total contribution small
    for _ in 0..50 {
        // We need to connect to something. Use hidden neurons as intermediaries.
        // Actually simpler: just make the direct path to output with 50 tiny-weight synapses
        // would be invalid (multiple synapses to same target). Instead, use one synapse.
    }
    // Single connection to output with meaningful weight
    synapses.push(("many-synapses", "output-0", 0.1)); // 10% structural impact

    // Add many outgoing synapses to hidden neurons to increase synapse count
    for i in 0..49 {
        synapses.push(("many-synapses", format!("helper-{i}").leak() as &str, 0.001));
        // Helper connects to output
        synapses.push((format!("helper-{i}").leak() as &str, "output-0", 0.001));
    }

    // Also add a neuron with few synapses but same impact (should NOT be returned)
    synapses.push(("input-0", "few-synapses", 0.1));
    synapses.push(("few-synapses", "output-0", 0.1)); // Same 10% impact

    let mut neurons: Vec<(&str, &str)> = vec![("output-0", "output")];
    neurons.push(("many-synapses", "hidden"));
    neurons.push(("few-synapses", "hidden"));
    for i in 0..50 {
        neurons.push((format!("input-{i}").leak() as &str, "input"));
    }
    for i in 0..49 {
        neurons.push((format!("helper-{i}").leak() as &str, "hidden"));
    }

    let creature = create_creature(neurons, synapses);

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // Create records where both neurons have the same activation_weighted_impact
    // but many-synapses has 100 synapses and few-synapses has 2
    let mut records_data: Vec<(&str, f32, f32)> = vec![
        // Both neurons have low activation to keep activation_weighted_impact small
        // but above threshold (1e-7)
        ("many-synapses", 0.1, 1e-5), // activation_weighted_impact ≈ 0.1 × 1e-5 = 1e-6
        ("few-synapses", 0.1, 1e-5),  // same activation_weighted_impact
        ("output-0", 0.5, 0.5),
    ];
    for i in 0..49 {
        records_data.push((format!("helper-{i}").leak() as &str, 0.1, 0.1));
    }

    let records = create_records_with_activation(records_data);
    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None, None).unwrap();

    // Find many-synapses in the results to check its impact
    let many_neuron = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "many-synapses");

    // Verify the many-synapses neuron was analysed
    assert!(
        many_neuron.is_some(),
        "many-synapses should be in neuron results"
    );
    let many = many_neuron.unwrap();

    // Verify activation_weighted_impact is above threshold (so old filter would reject it)
    let cost_of_growth = 1e-7_f32;
    println!(
        "many-synapses: impact={:.2e}, activation_weighted_impact={:.2e}",
        many.impact, many.activation_weighted_impact
    );

    // Count actual synapses for many-synapses
    let incoming = creature
        .synapses
        .iter()
        .filter(|s| s.to_uuid == "many-synapses")
        .count();
    let outgoing = creature
        .synapses
        .iter()
        .filter(|s| s.from_uuid == "many-synapses")
        .count();
    let total_synapses = incoming + outgoing;
    println!(
        "many-synapses: {incoming} incoming, {outgoing} outgoing, {total_synapses} total synapses"
    );

    let removal_savings = cost_of_growth * (1.0 + total_synapses as f32 / 10.0);
    println!(
        "many-synapses: removal_savings={:.2e}, activation_weighted_impact={:.2e}",
        removal_savings, many.activation_weighted_impact
    );

    // The key test: if removal_savings > activation_weighted_impact,
    // then removing this neuron improves score and it SHOULD be returned
    let should_be_candidate = removal_savings > many.activation_weighted_impact;
    println!(
        "Should be candidate? {} (savings > impact: {:.2e} > {:.2e})",
        should_be_candidate, removal_savings, many.activation_weighted_impact
    );

    // Check if many-synapses is in removal candidates
    let many_removal = result
        .removal_candidates
        .iter()
        .find(|c| c.neuron_uuid == "many-synapses");

    if should_be_candidate {
        // Issue #235: This neuron SHOULD be a removal candidate because
        // its removal_savings exceeds its activation_weighted_impact
        assert!(
            many_removal.is_some(),
            "Issue #235: Neuron 'many-synapses' with removal_savings ({:.2e}) > activation_weighted_impact ({:.2e}) \
             SHOULD be a removal candidate but was not returned. \
             This neuron has {} synapses. The old filter (impact < threshold) would reject it \
             because {:.2e} >= {:.2e}, but the new filter (savings > impact) should include it.",
            removal_savings,
            many.activation_weighted_impact,
            total_synapses,
            many.activation_weighted_impact,
            cost_of_growth
        );
    }

    // Also verify few-synapses is NOT a candidate (savings < impact)
    let few_removal = result
        .removal_candidates
        .iter()
        .find(|c| c.neuron_uuid == "few-synapses");
    let few_neuron = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "few-synapses");
    if let Some(few) = few_neuron {
        let few_savings = cost_of_growth * (1.0 + 2.0 / 10.0); // 2 synapses
        if few_savings <= few.activation_weighted_impact {
            assert!(
                few_removal.is_none(),
                "few-synapses with removal_savings ({:.2e}) <= activation_weighted_impact ({:.2e}) \
                 should NOT be a removal candidate",
                few_savings,
                few.activation_weighted_impact
            );
        }
    }
}

/// Issue #235: Direct test - neuron with impact ABOVE threshold but removal still improves score.
///
/// This is the core fix: a neuron with `activation_weighted_impact` = 2e-7 (above threshold 1e-7)
/// but with 20 synapses (`removal_savings` = 3e-7) SHOULD be returned because removal improves score.
///
/// Old behaviour: Only neurons with impact < 1e-7 are returned.
/// New behaviour: Neurons where `removal_savings` > impact are returned.
#[test]
#[serial]
fn test_issue_235_neuron_above_threshold_but_removal_improves_score() {
    let _guard = NoiseFloorOffGuard::new();
    // Create a creature with a hidden neuron that has:
    // - activation_weighted_impact slightly above threshold (2e-7)
    // - Many synapses so removal_savings > impact (20 synapses → 3e-7 savings)
    //
    // Network: input -> above-threshold -> output
    // The "above-threshold" neuron has 20 synapses total to boost removal_savings.
    let mut synapses = vec![];
    // 19 incoming synapses from different inputs with tiny weights
    for i in 0..19 {
        synapses.push((
            format!("input-{i}").leak() as &str,
            "above-threshold",
            1e-8, // tiny weight
        ));
    }
    // Main synapse with meaningful weight to give structural impact
    synapses.push(("input-19", "above-threshold", 1e-6));
    // Single outgoing synapse to output - this gives ~100% impact through this path
    synapses.push(("above-threshold", "output-0", 1.0));

    let mut neurons: Vec<(&str, &str)> =
        vec![("output-0", "output"), ("above-threshold", "hidden")];
    for i in 0..20 {
        neurons.push((format!("input-{i}").leak() as &str, "input"));
    }

    let creature = create_creature(neurons, synapses);

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // Set activation so that:
    // - structural_impact ≈ 1.0 (the neuron is on the main path to output)
    // - activation = 2e-7 (very small)
    // - activation_weighted_impact = 1.0 × 2e-7 = 2e-7 (above threshold 1e-7)
    //
    // With 20 synapses: removal_savings = 1e-7 × (1 + 20/10) = 3e-7
    // Since 3e-7 > 2e-7, removal SHOULD improve score.
    let records = create_records_with_activation(vec![
        ("above-threshold", 0.1, 2e-7), // activation = 2e-7
        ("output-0", 0.5, 0.5),
    ]);
    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None, None).unwrap();

    // Find the neuron in results
    let neuron = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "above-threshold");
    assert!(
        neuron.is_some(),
        "above-threshold should be in neuron results"
    );
    let neuron = neuron.unwrap();

    // Debug output
    println!(
        "above-threshold: impact={:.2e}, activation_weighted_impact={:.2e}",
        neuron.impact, neuron.activation_weighted_impact
    );

    // Verify the neuron has impact ABOVE threshold (so old filter would reject it)
    let cost_of_growth = 1e-7_f32;
    println!(
        "activation_weighted_impact ({:.2e}) vs threshold ({:.2e}): above={}",
        neuron.activation_weighted_impact,
        cost_of_growth,
        neuron.activation_weighted_impact >= cost_of_growth
    );

    // Count synapses
    let incoming = creature
        .synapses
        .iter()
        .filter(|s| s.to_uuid == "above-threshold")
        .count();
    let outgoing = creature
        .synapses
        .iter()
        .filter(|s| s.from_uuid == "above-threshold")
        .count();
    let total_synapses = incoming + outgoing;
    println!(
        "above-threshold: {incoming} incoming, {outgoing} outgoing, {total_synapses} total synapses"
    );

    // Calculate removal_savings
    let removal_savings = cost_of_growth * (1.0 + total_synapses as f32 / 10.0);
    println!(
        "removal_savings ({:.2e}) vs activation_weighted_impact ({:.2e}): improves_score={}",
        removal_savings,
        neuron.activation_weighted_impact,
        removal_savings > neuron.activation_weighted_impact
    );

    // The core assertion: if removal_savings > activation_weighted_impact,
    // then this neuron SHOULD be in removal_candidates
    if removal_savings > neuron.activation_weighted_impact {
        let candidate = result
            .removal_candidates
            .iter()
            .find(|c| c.neuron_uuid == "above-threshold");

        assert!(
            candidate.is_some(),
            "Issue #235: Neuron 'above-threshold' with:\n  \
             - activation_weighted_impact = {:.2e} (ABOVE threshold {:.2e})\n  \
             - removal_savings = {:.2e} (from {} synapses)\n  \
             - removal_savings > impact, so removal IMPROVES score\n\n\
             This neuron SHOULD be in removal_candidates but was NOT returned.\n\
             The old filter (impact < threshold) rejects it, but the new filter \
             (savings > impact) should include it.",
            neuron.activation_weighted_impact,
            cost_of_growth,
            removal_savings,
            total_synapses
        );

        // Verify the reason explains the improvement
        let candidate = candidate.unwrap();
        assert!(
            candidate.reason.contains("saves") || candidate.reason.contains("improve"),
            "Reason should explain why removal improves score: {}",
            candidate.reason
        );
    }
}

/// Issue #235: Verify sensible limits on removal candidates.
///
/// While we return ALL candidates expected to improve score, we should still
/// have sensible limits to prevent overwhelming the caller.
#[test]
#[serial]
fn test_issue_235_removal_candidates_have_sensible_limits() {
    let _guard = NoiseFloorOffGuard::new();
    // Create a creature with many neurons that could be removal candidates
    // to verify we don't return an excessive number.
    let mut neurons: Vec<(&str, &str)> = vec![("output-0", "output")];
    let mut synapses: Vec<(&str, &str, f32)> = vec![];

    // Create 100 hidden neurons, all with negligible impact (disconnected from output)
    for i in 0..100 {
        let uuid: &str = format!("orphan-{i}").leak();
        neurons.push((uuid, "hidden"));
        neurons.push((format!("input-{i}").leak() as &str, "input"));
        // Connect input to orphan, but orphan not connected to output
        synapses.push((format!("input-{i}").leak() as &str, uuid, 0.1));
    }
    // One connected neuron
    neurons.push(("connected", "hidden"));
    synapses.push(("input-0", "connected", 1.0));
    synapses.push(("connected", "output-0", 1.0));

    let creature = create_creature(neurons, synapses);

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    let mut records_data: Vec<(&str, f32, f32)> =
        vec![("connected", 0.1, 0.5), ("output-0", 0.5, 0.5)];
    for i in 0..100 {
        let uuid: &str = format!("orphan-{i}").leak();
        records_data.push((uuid, 0.1, 0.1));
    }

    let records = create_records_with_activation(records_data);
    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None, None).unwrap();

    // Should return all 100 orphan neurons as removal candidates
    // (they all have zero impact and any removal would improve score)
    assert!(
        result.removal_candidates.len() >= 100,
        "Issue #235: Should return all removal candidates expected to improve score. \
         Expected at least 100 orphan neurons, got {}",
        result.removal_candidates.len()
    );

    // All orphans should have negligible activation_weighted_impact
    for candidate in &result.removal_candidates {
        if candidate.neuron_uuid.starts_with("orphan-") {
            assert!(
                candidate.activation_weighted_impact < 1e-6,
                "Orphan {} should have negligible impact, got {:.2e}",
                candidate.neuron_uuid,
                candidate.activation_weighted_impact
            );
        }
    }
}
