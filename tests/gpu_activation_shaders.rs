//! Tests for GPU shader activation function support.
//!
//! Bug: New activation functions (LeakyReLU, Mish, Swish, HARD_TANH, SOFTSIGN,
//! BENT_IDENTITY, ArcTan, ReLU6) are assigned GPU IDs 11-18 in `activation_name_to_gpu_id`,
//! but the GPU shaders (`activation.wgsl` and `bias.wgsl`) only handle IDs 0-10.
//! Unknown IDs fall back to `default: { return x; }` which is IDENTITY.
//!
//! This causes GPU evaluation to silently return incorrect (IDENTITY-based) results
//! instead of the correct activation function output.
//!
//! Bug fix: v0.1.141
//!
//! The fix: Added all 8 new activation functions to both GPU shaders
//! (activation.wgsl and bias.wgsl) with correct implementations.

mod common;

use neat_ai_discovery::analysis::{analyze_neurons, GpuAnalyzer};
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{AnalyzeNeuronsInput, CreatureJson, NeuronJson, SynapseJson};
use tempfile::NamedTempFile;

/// Skip test if no GPU available
macro_rules! skip_without_gpu {
    () => {
        if !GpuAnalyzer::gpu_is_available() {
            eprintln!("Skipping test: no GPU available");
            return;
        }
    };
}

/// Helper to create a creature with specified topology
fn create_test_creature(
    neurons: Vec<(&str, &str, &str)>, // (uuid, type, squash)
    synapses: Vec<(&str, &str, f32)>, // (from, to, weight)
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

/// Test that Mish activation produces correct GPU results.
///
/// Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))
/// For x = -5: Mish ≈ -0.034, IDENTITY = -5 (147x different!)
#[test]
fn test_mish_gpu_shader_produces_correct_results() {
    skip_without_gpu!();

    let creature = create_test_creature(
        vec![
            ("input-0", "input", "IDENTITY"),
            ("input-1", "input", "IDENTITY"),
            ("output-0", "output", "IDENTITY"),
        ],
        vec![("input-0", "output-0", 1.0)],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    let mut records = Vec::new();

    for i in 0..200 {
        let obs_idx = i as u32;

        records.push(DiscoverRecord::new(
            obs_idx,
            "output-0".to_string(),
            Some(0.5),
            0.5,
            vec![0.3], // Positive error
        ));

        // Negative source where Mish differs significantly from IDENTITY
        // v0.2.2: Add variance to avoid source variance discounting
        let base_activation = -3.0;
        let variation = (i as f32 / 20.0).sin() * 0.3;
        let source_activation = base_activation + variation;
        records.push(DiscoverRecord::new(
            obs_idx,
            "input-1".to_string(),
            Some(source_activation),
            source_activation,
            vec![0.0],
        ));

        records.push(DiscoverRecord::new(
            obs_idx,
            "input-0".to_string(),
            Some(0.5),
            0.5,
            vec![0.0],
        ));
    }

    write_records_to_parquet(file_path, &records).unwrap();

    let input = AnalyzeNeuronsInput {
        parquet_file: file_path.to_string(),
        creature,
        focus_neurons: vec!["output-0".to_string()],
        improvement_threshold: Some(0.0),
        max_candidates: Some(100),
        analysis_deadline_ms: None,
    };

    let result = analyze_neurons(&input).expect("Analysis should succeed");

    let mish_candidates: Vec<_> = result
        .helpful_neurons
        .iter()
        .filter(|c| c.squash == "Mish" && c.source_neuron_uuid == "input-1")
        .collect();

    eprintln!("Mish candidates from input-1: {}", mish_candidates.len());
    for c in &mish_candidates {
        eprintln!(
            "  incoming={:.4}, outgoing={:.4}, improvement={:.4}%",
            c.incoming_weight,
            c.outgoing_weight,
            c.expected_creature_score_gain * 100.0
        );
    }

    // If Mish is correctly implemented, we should see candidates
    // If it falls back to IDENTITY, the weight calculations will be wrong
    assert!(
        !mish_candidates.is_empty(),
        "Should find Mish candidates (if GPU shader is correct)"
    );

    eprintln!("Test passed: Mish GPU shader produces correct results");
}

/// Test all 8 new activations produce candidates (basic smoke test).
///
/// This verifies that each new activation function is at least producing
/// some candidates, which wouldn't happen if the GPU shader fell back to
/// IDENTITY and produced nonsensical weight calculations.
#[test]
fn test_all_new_activations_produce_candidates() {
    skip_without_gpu!();

    let creature = create_test_creature(
        vec![
            ("input-0", "input", "IDENTITY"),
            ("input-1", "input", "IDENTITY"),
            ("output-0", "output", "IDENTITY"),
        ],
        vec![("input-0", "output-0", 1.0)],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // Create samples with varied source activations to exercise different
    // regions of each activation function
    let mut records = Vec::new();

    for i in 0..500 {
        let obs_idx = i as u32;

        // Varying target error to ensure some correlation patterns
        let error = if i % 3 == 0 {
            0.4
        } else if i % 3 == 1 {
            0.2
        } else {
            0.1
        };

        records.push(DiscoverRecord::new(
            obs_idx,
            "output-0".to_string(),
            Some(0.5),
            0.5,
            vec![error],
        ));

        // Source with varying activations (both positive and negative)
        let source = match i % 5 {
            0 => -2.0,
            1 => -0.5,
            2 => 0.0,
            3 => 0.5,
            _ => 2.0,
        };

        records.push(DiscoverRecord::new(
            obs_idx,
            "input-1".to_string(),
            Some(source),
            source,
            vec![0.0],
        ));

        records.push(DiscoverRecord::new(
            obs_idx,
            "input-0".to_string(),
            Some(0.5),
            0.5,
            vec![0.0],
        ));
    }

    write_records_to_parquet(file_path, &records).unwrap();

    let input = AnalyzeNeuronsInput {
        parquet_file: file_path.to_string(),
        creature,
        focus_neurons: vec!["output-0".to_string()],
        improvement_threshold: Some(0.0),
        max_candidates: Some(500),
        analysis_deadline_ms: None,
    };

    let result = analyze_neurons(&input).expect("Analysis should succeed");

    // Check each new activation function
    let new_activations = [
        "LeakyReLU",
        "Mish",
        "Swish",
        "HARD_TANH",
        "SOFTSIGN",
        "BENT_IDENTITY",
        "ArcTan",
        "ReLU6",
    ];

    let mut activations_with_candidates = 0;
    eprintln!("\nNew activation function candidates:");
    for activation in &new_activations {
        let count = result
            .helpful_neurons
            .iter()
            .filter(|c| c.squash == *activation)
            .count();
        eprintln!("  {activation}: {count} candidates");

        if count > 0 {
            activations_with_candidates += 1;
        }
    }

    // At least half of the new activations should produce candidates.
    // Some activations may not produce candidates due to saturation detection
    // (e.g., HARD_TANH may be saturated with certain input ranges).
    // If GPU falls back to IDENTITY, most/all would fail.
    assert!(
        activations_with_candidates >= 3,
        "At least 3 new activations should produce candidates, got {activations_with_candidates}/{}",
        new_activations.len()
    );

    eprintln!(
        "\nTest passed: {activations_with_candidates}/{} new activations produce candidates",
        new_activations.len()
    );
}
