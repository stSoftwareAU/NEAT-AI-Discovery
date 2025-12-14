//! Tests for activation-based impact calculation for MINIMUM, MAXIMUM, and IF neurons.
//!
//! These tests verify that when activation records are available, the impact calculation
//! uses actual selection probabilities instead of the conservative 1/N equal probability.
//!
//! Key scenarios:
//! - MIN neuron where one synapse wins 90% → should get ~90% impact
//! - MAX neuron where one synapse wins 90% → should get ~90% impact
//! - IF neuron with known condition/positive/negative branch usage

mod common;

use neat_ai_discovery::focus::rank_focus_neurons;
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
use tempfile::NamedTempFile;

/// Helper to create a creature with specified neurons and synapses.
/// Input neurons are represented only by the creature.input count.
fn create_creature(
    neurons: Vec<(&str, &str, &str)>, // (uuid, type, squash)
    synapses: Vec<(&str, &str, f32, Option<&str>)>, // (from, to, weight, synapse_type)
) -> CreatureJson {
    let input_count = neurons.iter().filter(|(_, t, _)| *t == "input").count();
    let output_count = neurons.iter().filter(|(_, t, _)| *t == "output").count();
    CreatureJson {
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
            .map(|(from, to, weight, synapse_type)| SynapseJson {
                from_uuid: from.to_string(),
                to_uuid: to.to_string(),
                weight,
                synapse_type: synapse_type.map(|s| s.to_string()),
            })
            .collect(),
        input: input_count,
        output: output_count,
    }
}

// =============================================================================
// MINIMUM Tests
// =============================================================================

/// Test that a MINIMUM neuron correctly assigns impact based on actual win statistics.
///
/// Setup:
/// - hidden-dominant has activations that produce the minimum 90% of the time
/// - hidden-rare has activations that produce the minimum 10% of the time
///
/// Expected:
/// - hidden-dominant should have ~90% of the MINIMUM neuron's impact share
/// - hidden-rare should have ~10% of the MINIMUM neuron's impact share
#[test]
fn test_minimum_neuron_with_dominant_synapse_gets_proportional_impact() {
    // Network: Two hidden neurons feeding into a MINIMUM output
    //
    //   input-0 ──► hidden-dominant ─┐
    //                                 ├─► MINIMUM output-0
    //   input-1 ──► hidden-rare ─────┘
    //
    let creature = create_creature(
        vec![
            ("input-0", "input", "IDENTITY"),
            ("input-1", "input", "IDENTITY"),
            ("hidden-dominant", "hidden", "IDENTITY"),
            ("hidden-rare", "hidden", "IDENTITY"),
            ("output-0", "output", "MINIMUM"),
        ],
        vec![
            ("input-0", "hidden-dominant", 1.0, None),
            ("input-1", "hidden-rare", 1.0, None),
            // Both synapses to MINIMUM have same weight
            ("hidden-dominant", "output-0", 1.0, None),
            ("hidden-rare", "output-0", 1.0, None),
        ],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // Create 100 observations where:
    // - hidden-dominant has small activation (0.1) - wins MINIMUM 90% of the time
    // - hidden-rare has large activation (1.0) - wins MINIMUM 10% of the time
    let mut records = Vec::new();
    for i in 0..100u32 {
        if i < 90 {
            // hidden-dominant wins (smaller weighted contribution)
            records.push(DiscoverRecord::new(
                i,
                "hidden-dominant".to_string(),
                Some(0.1),
                0.1, // Small activation → small weighted contribution
                vec![0.1],
            ));
            records.push(DiscoverRecord::new(
                i,
                "hidden-rare".to_string(),
                Some(1.0),
                1.0, // Large activation → large weighted contribution
                vec![0.1],
            ));
        } else {
            // hidden-rare wins (smaller weighted contribution)
            records.push(DiscoverRecord::new(
                i,
                "hidden-dominant".to_string(),
                Some(1.0),
                1.0, // Large activation
                vec![0.1],
            ));
            records.push(DiscoverRecord::new(
                i,
                "hidden-rare".to_string(),
                Some(0.05),
                0.05, // Very small activation → wins MINIMUM
                vec![0.1],
            ));
        }
        // Output record
        records.push(DiscoverRecord::new(
            i,
            "output-0".to_string(),
            Some(0.1),
            0.1,
            vec![0.05],
        ));
    }

    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None, None).unwrap();

    // Find the hidden neurons' impacts
    let dominant = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "hidden-dominant")
        .expect("hidden-dominant should be in results");
    let rare = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "hidden-rare")
        .expect("hidden-rare should be in results");

    println!(
        "MINIMUM test - hidden-dominant impact: {:.4}, hidden-rare impact: {:.4}",
        dominant.impact, rare.impact
    );

    // hidden-dominant should have ~90% of the impact (wins 90% of time)
    // hidden-rare should have ~10% of the impact (wins 10% of time)
    let total_impact = dominant.impact + rare.impact;
    let dominant_share = dominant.impact / total_impact;
    let rare_share = rare.impact / total_impact;

    println!(
        "Impact shares - dominant: {:.2}%, rare: {:.2}%",
        dominant_share * 100.0,
        rare_share * 100.0
    );

    // Allow 5% tolerance for statistical noise
    assert!(
        (dominant_share - 0.90).abs() < 0.05,
        "hidden-dominant should have ~90% impact share, got {:.2}%",
        dominant_share * 100.0
    );
    assert!(
        (rare_share - 0.10).abs() < 0.05,
        "hidden-rare should have ~10% impact share, got {:.2}%",
        rare_share * 100.0
    );
}

// =============================================================================
// MAXIMUM Tests
// =============================================================================

/// Test that a MAXIMUM neuron correctly assigns impact based on actual win statistics.
///
/// Setup:
/// - hidden-dominant has activations that produce the maximum 90% of the time
/// - hidden-rare has activations that produce the maximum 10% of the time
///
/// Expected:
/// - hidden-dominant should have ~90% of the MAXIMUM neuron's impact share
/// - hidden-rare should have ~10% of the MAXIMUM neuron's impact share
#[test]
fn test_maximum_neuron_with_dominant_synapse_gets_proportional_impact() {
    // Network: Two hidden neurons feeding into a MAXIMUM output
    let creature = create_creature(
        vec![
            ("input-0", "input", "IDENTITY"),
            ("input-1", "input", "IDENTITY"),
            ("hidden-dominant", "hidden", "IDENTITY"),
            ("hidden-rare", "hidden", "IDENTITY"),
            ("output-0", "output", "MAXIMUM"),
        ],
        vec![
            ("input-0", "hidden-dominant", 1.0, None),
            ("input-1", "hidden-rare", 1.0, None),
            ("hidden-dominant", "output-0", 1.0, None),
            ("hidden-rare", "output-0", 1.0, None),
        ],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // Create 100 observations where:
    // - hidden-dominant has large activation (1.0) - wins MAXIMUM 90% of the time
    // - hidden-rare has small activation (0.1) - wins MAXIMUM 10% of the time
    let mut records = Vec::new();
    for i in 0..100u32 {
        if i < 90 {
            // hidden-dominant wins (larger weighted contribution)
            records.push(DiscoverRecord::new(
                i,
                "hidden-dominant".to_string(),
                Some(1.0),
                1.0, // Large activation → large weighted contribution
                vec![0.1],
            ));
            records.push(DiscoverRecord::new(
                i,
                "hidden-rare".to_string(),
                Some(0.1),
                0.1, // Small activation
                vec![0.1],
            ));
        } else {
            // hidden-rare wins (larger weighted contribution)
            records.push(DiscoverRecord::new(
                i,
                "hidden-dominant".to_string(),
                Some(0.1),
                0.1, // Small activation
                vec![0.1],
            ));
            records.push(DiscoverRecord::new(
                i,
                "hidden-rare".to_string(),
                Some(2.0),
                2.0, // Very large activation → wins MAXIMUM
                vec![0.1],
            ));
        }
        // Output record
        records.push(DiscoverRecord::new(
            i,
            "output-0".to_string(),
            Some(1.0),
            1.0,
            vec![0.05],
        ));
    }

    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None, None).unwrap();

    let dominant = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "hidden-dominant")
        .expect("hidden-dominant should be in results");
    let rare = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "hidden-rare")
        .expect("hidden-rare should be in results");

    println!(
        "MAXIMUM test - hidden-dominant impact: {:.4}, hidden-rare impact: {:.4}",
        dominant.impact, rare.impact
    );

    let total_impact = dominant.impact + rare.impact;
    let dominant_share = dominant.impact / total_impact;
    let rare_share = rare.impact / total_impact;

    println!(
        "Impact shares - dominant: {:.2}%, rare: {:.2}%",
        dominant_share * 100.0,
        rare_share * 100.0
    );

    // Allow 5% tolerance
    assert!(
        (dominant_share - 0.90).abs() < 0.05,
        "hidden-dominant should have ~90% impact share, got {:.2}%",
        dominant_share * 100.0
    );
    assert!(
        (rare_share - 0.10).abs() < 0.05,
        "hidden-rare should have ~10% impact share, got {:.2}%",
        rare_share * 100.0
    );
}

// =============================================================================
// IF Neuron Tests
// =============================================================================

/// Test that an IF neuron correctly assigns impact based on synapse types.
///
/// Setup:
/// - condition synapse: always active → should have 100% of condition share
/// - positive synapse: active when condition > 0 (70% of observations)
/// - negative synapse: active when condition <= 0 (30% of observations)
///
/// Expected:
/// - hidden-condition: 100% probability (always evaluated)
/// - hidden-positive: ~70% probability (condition positive 70% of time)
/// - hidden-negative: ~30% probability (condition non-positive 30% of time)
#[test]
fn test_if_neuron_synapse_type_based_impact() {
    // Network: Three hidden neurons feeding into an IF output with typed synapses
    let creature = create_creature(
        vec![
            ("input-0", "input", "IDENTITY"),
            ("input-1", "input", "IDENTITY"),
            ("input-2", "input", "IDENTITY"),
            ("hidden-condition", "hidden", "IDENTITY"),
            ("hidden-positive", "hidden", "IDENTITY"),
            ("hidden-negative", "hidden", "IDENTITY"),
            ("output-0", "output", "IF"),
        ],
        vec![
            ("input-0", "hidden-condition", 1.0, None),
            ("input-1", "hidden-positive", 1.0, None),
            ("input-2", "hidden-negative", 1.0, None),
            // Typed synapses to IF neuron
            ("hidden-condition", "output-0", 1.0, Some("condition")),
            ("hidden-positive", "output-0", 1.0, Some("positive")),
            ("hidden-negative", "output-0", 1.0, Some("negative")),
        ],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // Create 100 observations where:
    // - condition is positive 70% of the time
    // - condition is negative/zero 30% of the time
    let mut records = Vec::new();
    for i in 0..100u32 {
        let condition_activation = if i < 70 {
            0.5 // Positive condition → positive branch
        } else {
            -0.5 // Negative condition → negative branch
        };

        records.push(DiscoverRecord::new(
            i,
            "hidden-condition".to_string(),
            Some(condition_activation),
            condition_activation,
            vec![0.1],
        ));
        records.push(DiscoverRecord::new(
            i,
            "hidden-positive".to_string(),
            Some(0.5),
            0.5,
            vec![0.1],
        ));
        records.push(DiscoverRecord::new(
            i,
            "hidden-negative".to_string(),
            Some(0.5),
            0.5,
            vec![0.1],
        ));
        records.push(DiscoverRecord::new(
            i,
            "output-0".to_string(),
            Some(0.5),
            0.5,
            vec![0.05],
        ));
    }

    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None, None).unwrap();

    let condition = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "hidden-condition")
        .expect("hidden-condition should be in results");
    let positive = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "hidden-positive")
        .expect("hidden-positive should be in results");
    let negative = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "hidden-negative")
        .expect("hidden-negative should be in results");

    println!(
        "IF test - condition impact: {:.4}, positive impact: {:.4}, negative impact: {:.4}",
        condition.impact, positive.impact, negative.impact
    );

    // Condition synapses have probability 1.0
    // Positive has ~0.70 probability
    // Negative has ~0.30 probability
    // Total theoretical: 1.0 + 0.70 + 0.30 = 2.0

    // Condition should have significantly higher impact than positive/negative combined
    // because it's always active (probability 1.0)
    assert!(
        condition.impact > positive.impact,
        "Condition synapse should have higher impact than positive (always active). \
         condition: {:.4}, positive: {:.4}",
        condition.impact,
        positive.impact
    );
    assert!(
        condition.impact > negative.impact,
        "Condition synapse should have higher impact than negative (always active). \
         condition: {:.4}, negative: {:.4}",
        condition.impact,
        negative.impact
    );

    // Positive should have higher impact than negative (70% vs 30%)
    assert!(
        positive.impact > negative.impact,
        "Positive branch (70%) should have higher impact than negative (30%). \
         positive: {:.4}, negative: {:.4}",
        positive.impact,
        negative.impact
    );

    // Check approximate ratios
    let total_branch_impact = positive.impact + negative.impact;
    let positive_share = positive.impact / total_branch_impact;
    let negative_share = negative.impact / total_branch_impact;

    println!(
        "Branch impact shares - positive: {:.2}%, negative: {:.2}%",
        positive_share * 100.0,
        negative_share * 100.0
    );

    // Allow 10% tolerance (more lenient for complex IF calculation)
    assert!(
        (positive_share - 0.70).abs() < 0.10,
        "Positive branch should have ~70% of branch impact, got {:.2}%",
        positive_share * 100.0
    );
    assert!(
        (negative_share - 0.30).abs() < 0.10,
        "Negative branch should have ~30% of branch impact, got {:.2}%",
        negative_share * 100.0
    );
}

/// Test that IF neuron falls back to equal probability when no synapse types are set.
#[test]
fn test_if_neuron_without_synapse_types_uses_equal_probability() {
    // Network: Three hidden neurons feeding into an IF output WITHOUT typed synapses
    let creature = create_creature(
        vec![
            ("input-0", "input", "IDENTITY"),
            ("input-1", "input", "IDENTITY"),
            ("input-2", "input", "IDENTITY"),
            ("hidden-a", "hidden", "IDENTITY"),
            ("hidden-b", "hidden", "IDENTITY"),
            ("hidden-c", "hidden", "IDENTITY"),
            ("output-0", "output", "IF"),
        ],
        vec![
            ("input-0", "hidden-a", 1.0, None),
            ("input-1", "hidden-b", 1.0, None),
            ("input-2", "hidden-c", 1.0, None),
            // NO synapse types - should fall back to equal probability
            ("hidden-a", "output-0", 1.0, None),
            ("hidden-b", "output-0", 1.0, None),
            ("hidden-c", "output-0", 1.0, None),
        ],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    let mut records = Vec::new();
    for i in 0..10u32 {
        records.push(DiscoverRecord::new(
            i,
            "hidden-a".to_string(),
            Some(0.5),
            0.5,
            vec![0.1],
        ));
        records.push(DiscoverRecord::new(
            i,
            "hidden-b".to_string(),
            Some(0.5),
            0.5,
            vec![0.1],
        ));
        records.push(DiscoverRecord::new(
            i,
            "hidden-c".to_string(),
            Some(0.5),
            0.5,
            vec![0.1],
        ));
        records.push(DiscoverRecord::new(
            i,
            "output-0".to_string(),
            Some(0.5),
            0.5,
            vec![0.05],
        ));
    }

    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None, None).unwrap();

    let a = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "hidden-a")
        .expect("hidden-a should be in results");
    let b = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "hidden-b")
        .expect("hidden-b should be in results");
    let c = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "hidden-c")
        .expect("hidden-c should be in results");

    println!(
        "IF fallback test - a impact: {:.4}, b impact: {:.4}, c impact: {:.4}",
        a.impact, b.impact, c.impact
    );

    // Without synapse types, all should have approximately equal impact (1/3 each)
    let epsilon = 0.02; // Small tolerance
    assert!(
        (a.impact - b.impact).abs() < epsilon,
        "Without synapse types, all should have equal impact. a: {:.4}, b: {:.4}",
        a.impact,
        b.impact
    );
    assert!(
        (b.impact - c.impact).abs() < epsilon,
        "Without synapse types, all should have equal impact. b: {:.4}, c: {:.4}",
        b.impact,
        c.impact
    );

    // Each should have ~1/3 of the impact
    let expected = 1.0 / 3.0;
    assert!(
        (a.impact - expected).abs() < epsilon,
        "Impact should be ~{:.4}, got {:.4}",
        expected,
        a.impact
    );
}

// =============================================================================
// Regression Tests
// =============================================================================

/// Verify that the old equal-probability behaviour is preserved when no activation data.
/// This test uses compute_impacts_public which doesn't have activation data.
#[test]
fn test_selection_neurons_without_activation_data_use_equal_probability() {
    use neat_ai_discovery::focus::compute_impacts_public;

    // Network with MINIMUM output and 3 inputs
    let creature = create_creature(
        vec![
            ("input-0", "input", "IDENTITY"),
            ("input-1", "input", "IDENTITY"),
            ("input-2", "input", "IDENTITY"),
            ("hidden-a", "hidden", "IDENTITY"),
            ("hidden-b", "hidden", "IDENTITY"),
            ("hidden-c", "hidden", "IDENTITY"),
            ("output-0", "output", "MINIMUM"),
        ],
        vec![
            ("input-0", "hidden-a", 1.0, None),
            ("input-1", "hidden-b", 1.0, None),
            ("input-2", "hidden-c", 1.0, None),
            ("hidden-a", "output-0", 0.1, None), // Different weights
            ("hidden-b", "output-0", 1.0, None),
            ("hidden-c", "output-0", 10.0, None),
        ],
    );

    // Use public API which doesn't have activation data
    let impacts = compute_impacts_public(&creature);

    let a_impact = *impacts.get("hidden-a").unwrap_or(&0.0);
    let b_impact = *impacts.get("hidden-b").unwrap_or(&0.0);
    let c_impact = *impacts.get("hidden-c").unwrap_or(&0.0);

    println!("Equal probability test - a: {a_impact:.4}, b: {b_impact:.4}, c: {c_impact:.4}");

    // Without activation data, all should have equal probability (1/3 each)
    let epsilon = 0.01;
    assert!(
        (a_impact - b_impact).abs() < epsilon,
        "Without activation data, all MINIMUM inputs should have equal impact. a: {a_impact:.4}, b: {b_impact:.4}"
    );
    assert!(
        (b_impact - c_impact).abs() < epsilon,
        "Without activation data, all MINIMUM inputs should have equal impact. b: {b_impact:.4}, c: {c_impact:.4}"
    );
}
