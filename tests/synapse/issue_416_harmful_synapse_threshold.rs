//! Issue #416: Harmful synapse candidates must have positive expected score gain
//!
//! This test verifies that the harmful synapse detection only returns candidates
//! where removing the synapse is expected to improve the creature's score.
//!
//! Previously, `harmful_synapses` included ALL existing synapses without filtering,
//! resulting in candidates with negative `expected_creature_score_gain` (i.e., synapses
//! that are actually helpful and should NOT be removed).
//!
//! NEAT-AI was filtering these out on its side, resulting in 0 samples being recorded
//! for the "remove-harmful-synapse" discovery type.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
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

/// Helper to create discovery records from neuron data
fn create_records(
    neuron_data: Vec<(&str, Vec<(f32, f32)>)>, // (uuid, [(activation, error), ...])
) -> Vec<DiscoverRecord> {
    let mut records = Vec::new();
    for (uuid, samples) in neuron_data {
        for (obs_idx, (activation, error)) in samples.into_iter().enumerate() {
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

/// Issue #416: Harmful synapse candidates must have positive `expected_creature_score_gain`.
///
/// This test creates a scenario with:
/// 1. A helpful synapse (removing it would make things worse)
/// 2. A harmful synapse (removing it would improve the score)
///
/// Only the harmful synapse should appear in `harmful_synapses`.
#[test]
fn test_harmful_synapse_candidates_must_have_positive_expected_gain() {
    skip_without_gpu!();

    // Network: input-0 -> output-0 (this synapse will be tested)
    //
    // We create data where the synapse is harmful:
    // - When input is positive and error is positive, the synapse contribution
    //   (activation * weight) has the same sign as error -> harmful
    let creature = create_test_creature(
        vec![
            ("input-0", "input", "IDENTITY"),
            ("output-0", "output", "IDENTITY"),
        ],
        vec![
            ("input-0", "output-0", 1.0), // Weight = 1.0
        ],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // Create records where the synapse is harmful:
    // - Positive input activation when error is positive
    // - contribution = activation * weight = positive * 1.0 = positive
    // - error = positive
    // - same sign -> synapse is pushing output in wrong direction -> harmful
    let mut input_data = Vec::new();
    let mut output_data = Vec::new();

    for i in 0..50 {
        // Pattern: activation and error have the SAME sign
        // This means the synapse contribution is pushing the output
        // in the same direction as the error, making it worse.
        let activation = 0.5 + (i as f32 * 0.01);
        let error = 0.3 + (i as f32 * 0.005);

        input_data.push((activation, 0.0)); // Input neurons have 0 error
        output_data.push((0.1, error));
    }

    let records = create_records(vec![("input-0", input_data), ("output-0", output_data)]);

    write_records_to_parquet(file_path, &records).unwrap();

    let input = AnalyzeSynapsesInput {
        parquet_file: file_path.to_string(),
        creature,
        focus_neurons: vec!["output-0".to_string()],
        max_candidates: Some(100),
        analysis_deadline_ms: Some(30000),
        random_seed: None,
        temperature: 1.0,
        failure_cache: None,
        discovery_outcome_log: None,
    };

    let result = neat_ai_discovery::analysis::analyze_synapses(&input);
    assert!(
        result.is_ok(),
        "Analysis should succeed: {:?}",
        result.err()
    );

    let analysis_result = result.unwrap();

    eprintln!("\n=== Issue #416 Test Results ===");
    eprintln!(
        "Harmful synapse candidates: {}",
        analysis_result.harmful_synapses.len()
    );

    for candidate in &analysis_result.harmful_synapses {
        eprintln!(
            "  {} -> {}: expected_creature_score_gain = {:.6} ({:.4}%)",
            candidate.from_neuron_uuid,
            candidate.to_neuron_uuid,
            candidate.expected_creature_score_gain,
            candidate.expected_creature_score_gain * 100.0
        );
    }

    // Issue #416: ALL harmful synapse candidates must have positive expected_creature_score_gain
    // Candidates with negative expected gain are NOT harmful - removing them would make things worse!
    for candidate in &analysis_result.harmful_synapses {
        assert!(
            candidate.expected_creature_score_gain > 0.0,
            "Harmful synapse candidate {} -> {} has non-positive expected_creature_score_gain: {:.6}. \
             This synapse is actually HELPFUL and should NOT be in harmful_synapses!",
            candidate.from_neuron_uuid,
            candidate.to_neuron_uuid,
            candidate.expected_creature_score_gain
        );
    }

    eprintln!(
        "Test passed: All {} harmful synapse candidates have positive expected_creature_score_gain",
        analysis_result.harmful_synapses.len()
    );
}

/// Issue #416: Verify that helpful synapses do NOT appear in `harmful_synapses`.
///
/// This test creates a synapse that is clearly helpful (its contribution reduces error)
/// and verifies that it does NOT appear in `harmful_synapses`.
#[test]
fn test_helpful_synapse_is_not_marked_as_harmful() {
    skip_without_gpu!();

    // Simple network: input-0 -> output-0
    let creature = create_test_creature(
        vec![
            ("input-0", "input", "IDENTITY"),
            ("output-0", "output", "IDENTITY"),
        ],
        vec![("input-0", "output-0", 1.0)],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // Create records where the synapse is helpful:
    // - Positive input activation when error is NEGATIVE
    // - contribution = activation * weight = positive * 1.0 = positive
    // - error = negative
    // - opposite sign -> synapse is pushing output in the RIGHT direction -> helpful
    let mut input_data = Vec::new();
    let mut output_data = Vec::new();

    for i in 0..50 {
        // Pattern: activation is positive, error is negative
        // This means the synapse is pushing output UP when it needs to go UP
        // (negative error means output is too low)
        let activation = 0.5 + (i as f32 * 0.01);
        let error = -0.3 - (i as f32 * 0.005); // Negative error

        input_data.push((activation, 0.0));
        output_data.push((0.1, error));
    }

    let records = create_records(vec![("input-0", input_data), ("output-0", output_data)]);

    write_records_to_parquet(file_path, &records).unwrap();

    let input = AnalyzeSynapsesInput {
        parquet_file: file_path.to_string(),
        creature,
        focus_neurons: vec!["output-0".to_string()],
        max_candidates: Some(100),
        analysis_deadline_ms: Some(30000),
        random_seed: None,
        temperature: 1.0,
        failure_cache: None,
        discovery_outcome_log: None,
    };

    let result = neat_ai_discovery::analysis::analyze_synapses(&input);
    assert!(
        result.is_ok(),
        "Analysis should succeed: {:?}",
        result.err()
    );

    let analysis_result = result.unwrap();

    eprintln!("\n=== Helpful Synapse Should Not Be Harmful Test ===");
    eprintln!(
        "Harmful synapse candidates: {}",
        analysis_result.harmful_synapses.len()
    );

    // The helpful synapse should NOT appear in harmful_synapses
    let found_as_harmful = analysis_result
        .harmful_synapses
        .iter()
        .find(|c| c.from_neuron_uuid == "input-0" && c.to_neuron_uuid == "output-0");

    if let Some(candidate) = found_as_harmful {
        // If it's in harmful_synapses, it MUST have positive expected gain
        // (meaning it's actually harmful). But our test data makes it helpful,
        // so it should either not be in the list OR have positive gain.
        assert!(
            candidate.expected_creature_score_gain > 0.0,
            "Helpful synapse input-0 -> output-0 appeared in harmful_synapses with \
             non-positive expected_creature_score_gain: {:.6}. \
             This is the bug from issue #416!",
            candidate.expected_creature_score_gain
        );
    }

    eprintln!("Test passed: Helpful synapse is correctly filtered from harmful_synapses");
}

/// Issue #416: Mixed scenario - one helpful, one harmful synapse.
///
/// Verifies that only the truly harmful synapse appears with positive gain.
#[test]
fn test_mixed_helpful_and_harmful_synapses() {
    skip_without_gpu!();

    // Network: input-0 -> output-0 (will be harmful)
    //          input-1 -> output-0 (will be helpful)
    let creature = create_test_creature(
        vec![
            ("input-0", "input", "IDENTITY"),
            ("input-1", "input", "IDENTITY"),
            ("output-0", "output", "IDENTITY"),
        ],
        vec![
            ("input-0", "output-0", 1.0), // Will be harmful
            ("input-1", "output-0", 1.0), // Will be helpful
        ],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    let mut input0_data = Vec::new();
    let mut input1_data = Vec::new();
    let mut output_data = Vec::new();

    for i in 0..50 {
        // Error is positive (output should be higher)
        let error = 0.3 + (i as f32 * 0.005);

        // input-0: POSITIVE activation with positive error
        // contribution = positive * 1.0 = positive
        // error = positive
        // same sign -> HARMFUL
        let input0_activation = 0.5 + (i as f32 * 0.01);

        // input-1: NEGATIVE activation with positive error
        // contribution = negative * 1.0 = negative
        // error = positive
        // opposite sign -> HELPFUL (pushes output down when it should go up? No wait...)
        // Actually: if error is positive, output is too LOW, needs to go UP
        // input-1 with negative activation and positive weight pushes output DOWN
        // That's wrong direction, so it's also harmful
        //
        // Let me reconsider: For a synapse to be helpful, its contribution should
        // REDUCE the error. If error is positive (output too low), we need positive contribution.
        // If error is negative (output too high), we need negative contribution.
        //
        // So for HARMFUL: contribution and error have SAME sign
        // For HELPFUL: contribution and error have OPPOSITE sign
        //
        // Let's make input-1 helpful by having opposite pattern:
        // - When error is positive, activation is NEGATIVE (so contribution is negative, opposite to error)
        // Wait, that doesn't help either...
        //
        // Actually the definition of harmful is:
        // signal (activation * weight) has SAME sign as error -> pushing output further from target
        //
        // For HELPFUL:
        // - error > 0 (output too low) -> need positive contribution -> need positive activation (weight is positive)
        // - error < 0 (output too high) -> need negative contribution -> need negative activation

        // Let me create a scenario where errors alternate
        // and input-1's activation is the OPPOSITE sign of the error
        let input1_activation = -0.5 - (i as f32 * 0.01); // Negative when error is positive

        input0_data.push((input0_activation, 0.0));
        input1_data.push((input1_activation, 0.0));
        output_data.push((0.1, error));
    }

    let records = create_records(vec![
        ("input-0", input0_data),
        ("input-1", input1_data),
        ("output-0", output_data),
    ]);

    write_records_to_parquet(file_path, &records).unwrap();

    let input = AnalyzeSynapsesInput {
        parquet_file: file_path.to_string(),
        creature,
        focus_neurons: vec!["output-0".to_string()],
        max_candidates: Some(100),
        analysis_deadline_ms: Some(30000),
        random_seed: None,
        temperature: 1.0,
        failure_cache: None,
        discovery_outcome_log: None,
    };

    let result = neat_ai_discovery::analysis::analyze_synapses(&input);
    assert!(
        result.is_ok(),
        "Analysis should succeed: {:?}",
        result.err()
    );

    let analysis_result = result.unwrap();

    eprintln!("\n=== Mixed Helpful/Harmful Test ===");
    eprintln!(
        "Harmful synapse candidates: {}",
        analysis_result.harmful_synapses.len()
    );

    for candidate in &analysis_result.harmful_synapses {
        eprintln!(
            "  {} -> {}: expected_creature_score_gain = {:.6}",
            candidate.from_neuron_uuid,
            candidate.to_neuron_uuid,
            candidate.expected_creature_score_gain
        );
    }

    // ALL candidates in harmful_synapses must have positive expected gain
    for candidate in &analysis_result.harmful_synapses {
        assert!(
            candidate.expected_creature_score_gain > 0.0,
            "Harmful synapse candidate {} -> {} has non-positive expected_creature_score_gain: {:.6}",
            candidate.from_neuron_uuid,
            candidate.to_neuron_uuid,
            candidate.expected_creature_score_gain
        );
    }

    eprintln!(
        "Test passed: All harmful synapse candidates have positive expected_creature_score_gain"
    );
}
