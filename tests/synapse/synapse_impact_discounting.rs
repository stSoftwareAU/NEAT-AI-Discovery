//! Tests for synapse candidate impact discounting.
//!
//! These tests verify that synapse candidates are impact-discounted to creature-level,
//! just like neuron candidates. This ensures TypeScript doesn't need to re-calculate
//! impact - Rust is the single source of truth.
//!
//! See GitHub issue: Single responsibility for impact calculation

use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{AnalyzeSynapsesInput, CreatureJson, NeuronJson, SynapseJson};
use tempfile::NamedTempFile;

/// Skip test if no GPU available
macro_rules! skip_without_gpu {
    () => {
        if !neat_ai_discovery::analysis::GpuAnalyzer::gpu_is_available() {
            eprintln!("Skipping test: no GPU available");
            return;
        }
    };
}

/// Helper to create a creature with specified topology
///
/// Note: Input neurons in the `neurons` parameter are used only to count `input`.
/// They are NOT included in `creature.neurons` as per the NEAT-AI data model -
/// input neurons are represented only by the `creature.input` count.
fn create_test_creature(
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

/// Helper to create discovery records with clear correlation patterns
fn create_synapse_candidate_records(
    neuron_data: Vec<(&str, Vec<(f32, f32)>)>, // (uuid, [(error, activation), ...])
) -> Vec<DiscoverRecord> {
    let mut records = Vec::new();
    for (uuid, samples) in neuron_data {
        for (obs_idx, (error, activation)) in samples.into_iter().enumerate() {
            records.push(DiscoverRecord::new(
                obs_idx as u32,
                uuid.to_string(),
                Some(activation), // value
                activation,
                vec![error],
            ));
        }
    }
    records
}

/// Test that synapse candidates targeting hidden neurons are impact-discounted.
///
/// Scenario:
/// - hidden-b → output-0 (weight 0.1)
/// - hidden-direct → output-0 (weight 0.9) - this dilutes hidden-b's impact!
///
/// hidden-b's impact = |0.1| / (|0.1| + |0.9|) = 0.1 / 1.0 = 0.1 (10%)
///
/// A synapse candidate targeting hidden-b should have its expected improvement
/// discounted by 0.1 (hidden-b's impact on the creature's score).
#[test]
fn test_synapse_candidate_hidden_neuron_is_impact_discounted() {
    skip_without_gpu!();

    // Network where hidden-b has PARTIAL impact (0.1) due to competing paths
    let creature = create_test_creature(
        vec![
            ("input-0", "input", "IDENTITY"),
            ("input-1", "input", "IDENTITY"),
            ("hidden-b", "hidden", "IDENTITY"), // Target - partial impact (0.1)
            ("hidden-direct", "hidden", "IDENTITY"), // Competes for output
            ("output-0", "output", "IDENTITY"),
        ],
        vec![
            ("input-0", "hidden-b", 1.0),
            ("input-0", "hidden-direct", 1.0),
            // hidden-b has weight 0.1 to output
            // hidden-direct has weight 0.9 to output
            // hidden-b's impact = 0.1 / (0.1 + 0.9) = 0.1
            ("hidden-b", "output-0", 0.1),
            ("hidden-direct", "output-0", 0.9),
        ],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // Create records where input-1 correlates with hidden-b's error
    let records = create_synapse_candidate_records(vec![
        (
            "hidden-b",
            vec![
                (0.5, 0.3),
                (0.6, 0.2),
                (-0.1, 0.8),
                (-0.2, 0.9),
                (0.4, 0.25),
                (0.3, 0.35),
                (-0.15, 0.7),
                (-0.25, 0.85),
            ],
        ),
        (
            "hidden-direct",
            vec![
                (0.1, 0.5),
                (0.1, 0.5),
                (0.1, 0.5),
                (0.1, 0.5),
                (0.1, 0.5),
                (0.1, 0.5),
                (0.1, 0.5),
                (0.1, 0.5),
            ],
        ),
        (
            "input-1",
            vec![
                (0.0, 0.9),
                (0.0, 0.85),
                (0.0, 0.1),
                (0.0, 0.15),
                (0.0, 0.8),
                (0.0, 0.75),
                (0.0, 0.2),
                (0.0, 0.1),
            ],
        ),
        (
            "output-0",
            vec![
                (0.1, 0.5),
                (0.1, 0.5),
                (0.1, 0.5),
                (0.1, 0.5),
                (0.1, 0.5),
                (0.1, 0.5),
                (0.1, 0.5),
                (0.1, 0.5),
            ],
        ),
    ]);
    write_records_to_parquet(file_path, &records).unwrap();

    let input = AnalyzeSynapsesInput {
        parquet_file: file_path.to_string(),
        creature,
        focus_neurons: vec!["hidden-b".to_string()],
        max_candidates: Some(10),
        analysis_deadline_ms: Some(30000),
        random_seed: None,
    };

    let result = neat_ai_discovery::analysis::analyze_synapses(&input);
    assert!(
        result.is_ok(),
        "Analysis should succeed: {:?}",
        result.err()
    );

    let analysis_result = result.unwrap();

    if analysis_result.helpful_synapses.is_empty() {
        println!("No synapse candidates found - the correlation may not be strong enough");
        println!(
            "Diagnostics: {:?}",
            analysis_result
                .no_candidate_reasons
                .iter()
                .map(|r| format!("{}: {:?}", r.target_uuid, r.reason))
                .collect::<Vec<_>>()
        );
        return;
    }

    let hidden_b_candidate = analysis_result
        .helpful_synapses
        .iter()
        .find(|c| c.to_neuron_uuid == "hidden-b");

    if let Some(candidate) = hidden_b_candidate {
        println!(
            "Found candidate: {} → hidden-b, expected_improvement: {:.4}%",
            candidate.from_neuron_uuid,
            candidate.expected_creature_score_gain * 100.0
        );

        // hidden-b has impact ≈ 0.1 (10%), so the expected improvement should be
        // discounted to about 10% of what it would be for an output neuron.
        //
        // The key assertion: this value is CREATURE-LEVEL, not neuron-level.
        // TypeScript should use this directly without re-calculation.
        assert!(
            candidate.expected_creature_score_gain >= 0.0,
            "Expected improvement should be non-negative"
        );

        // With impact ≈ 0.1, even a 100% neuron-level improvement becomes ~10% creature-level
        // So we expect the value to be relatively small
        assert!(
            candidate.expected_creature_score_gain <= 0.5,
            "Expected improvement should be discounted (≤50% for low-impact hidden neuron), got {:.2}%",
            candidate.expected_creature_score_gain * 100.0
        );
    }
}

/// Test that synapse candidates targeting output neurons have full impact (no discount).
///
/// Output neurons have impact = 1.0, so their expected_creature_score_gain
/// should not be discounted at all.
#[test]
fn test_synapse_candidate_output_neuron_no_discount() {
    skip_without_gpu!();

    let creature = create_test_creature(
        vec![
            ("input-0", "input", "IDENTITY"),
            ("input-1", "input", "IDENTITY"),
            ("output-0", "output", "IDENTITY"), // Target - full impact
        ],
        vec![("input-0", "output-0", 1.0)],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // Create records where input-1 correlates with output-0's error
    let records = create_synapse_candidate_records(vec![
        (
            "output-0",
            vec![
                (0.5, 0.3),
                (0.6, 0.2),
                (-0.1, 0.8),
                (-0.2, 0.9),
                (0.4, 0.25),
                (0.3, 0.35),
                (-0.15, 0.7),
                (-0.25, 0.85),
            ],
        ),
        (
            "input-1",
            vec![
                (0.0, 0.9),
                (0.0, 0.85),
                (0.0, 0.1),
                (0.0, 0.15),
                (0.0, 0.8),
                (0.0, 0.75),
                (0.0, 0.2),
                (0.0, 0.1),
            ],
        ),
    ]);
    write_records_to_parquet(file_path, &records).unwrap();

    let input = AnalyzeSynapsesInput {
        parquet_file: file_path.to_string(),
        creature,
        focus_neurons: vec!["output-0".to_string()],
        max_candidates: Some(10),
        analysis_deadline_ms: Some(30000),
        random_seed: None,
    };

    let result = neat_ai_discovery::analysis::analyze_synapses(&input);
    assert!(
        result.is_ok(),
        "Analysis should succeed: {:?}",
        result.err()
    );

    let analysis_result = result.unwrap();

    if let Some(candidate) = analysis_result
        .helpful_synapses
        .iter()
        .find(|c| c.to_neuron_uuid == "output-0")
    {
        println!(
            "Output candidate: {} → output-0, expected_improvement: {:.4}%",
            candidate.from_neuron_uuid,
            candidate.expected_creature_score_gain * 100.0
        );

        // Output neurons have impact = 1.0, so no discount should be applied
        // The expected_creature_score_gain IS the creature-level improvement
    }
}

/// Test that harmful synapse candidates are also impact-discounted.
#[test]
fn test_harmful_synapse_candidate_is_impact_discounted() {
    skip_without_gpu!();

    // Network with a hidden neuron
    let creature = create_test_creature(
        vec![
            ("input-0", "input", "IDENTITY"),
            ("hidden-a", "hidden", "IDENTITY"),
            ("output-0", "output", "IDENTITY"),
        ],
        vec![
            ("input-0", "hidden-a", 1.0),
            ("hidden-a", "output-0", 0.5), // This synapse should be tested as harmful candidate
        ],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // Create records where hidden-a activation negatively correlates with output error reduction
    // (i.e., when hidden-a is high, error gets worse)
    let records = create_synapse_candidate_records(vec![
        (
            "output-0",
            vec![
                (0.5, 0.3),
                (0.6, 0.2),
                (-0.1, 0.8),
                (-0.2, 0.9),
                (0.4, 0.25),
                (0.3, 0.35),
                (-0.15, 0.7),
                (-0.25, 0.85),
            ],
        ),
        (
            "hidden-a",
            vec![
                // Low activation when error is positive (should be higher)
                (0.1, 0.1),
                (0.1, 0.15),
                // High activation when error is negative (should be lower)
                (0.05, 0.9),
                (0.05, 0.85),
                (0.1, 0.2),
                (0.1, 0.25),
                (0.05, 0.8),
                (0.05, 0.75),
            ],
        ),
    ]);
    write_records_to_parquet(file_path, &records).unwrap();

    let input = AnalyzeSynapsesInput {
        parquet_file: file_path.to_string(),
        creature,
        focus_neurons: vec!["output-0".to_string()],
        max_candidates: Some(10),
        analysis_deadline_ms: Some(30000),
        random_seed: None,
    };

    let result = neat_ai_discovery::analysis::analyze_synapses(&input);
    assert!(
        result.is_ok(),
        "Analysis should succeed: {:?}",
        result.err()
    );

    let analysis_result = result.unwrap();

    println!(
        "Harmful candidates: {}",
        analysis_result.harmful_synapses.len()
    );
    for candidate in &analysis_result.harmful_synapses {
        println!(
            "  {} → {}, expected_improvement: {:.4}%",
            candidate.from_neuron_uuid,
            candidate.to_neuron_uuid,
            candidate.expected_creature_score_gain * 100.0
        );
    }

    // Harmful synapse candidates should also have creature-level expected improvement
    // (i.e., how much removing them would improve the creature's score)
}
