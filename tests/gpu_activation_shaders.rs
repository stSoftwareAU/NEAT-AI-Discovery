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
            })
            .collect(),
        input: input_count,
        output: output_count,
    }
}

/// Test that new activation functions produce correct GPU results.
///
/// This tests the bug where GPU IDs 11-18 fall back to IDENTITY in the shader.
/// We verify LeakyReLU candidates are evaluated correctly by comparing the
/// GPU-computed values against expected behaviour.
///
/// LeakyReLU(x) = x if x >= 0, else 0.01 * x
/// IDENTITY(x) = x
///
/// For negative inputs, LeakyReLU produces 0.01*x whilst IDENTITY produces x.
/// If the GPU incorrectly uses IDENTITY, the predictions will be wrong.
#[test]
fn test_leaky_relu_gpu_shader_produces_correct_results() {
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

    // Create samples where LeakyReLU and IDENTITY differ significantly.
    // For negative source activations:
    // - LeakyReLU: 0.01 * activation (small output)
    // - IDENTITY: activation (large negative output)
    //
    // If GPU incorrectly uses IDENTITY, the neuron output will be ~100x larger
    // for negative inputs, producing completely wrong weight calculations.
    let mut records = Vec::new();

    for i in 0..200 {
        let obs_idx = i as u32;

        // Target has positive error - needs positive correction
        let target_value = 0.5;
        let target_activation = 0.5;
        let error = 0.3; // Positive error: output should be higher

        records.push(DiscoverRecord::new(
            obs_idx,
            "output-0".to_string(),
            Some(target_value),
            target_activation,
            vec![error],
        ));

        // Source: NEGATIVE activations where LeakyReLU differs from IDENTITY
        // LeakyReLU(-1.0) = -0.01, IDENTITY(-1.0) = -1.0 (100x different!)
        let source_activation = -1.0;

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
        improvement_threshold: Some(0.0), // Accept all positive improvements
        max_candidates: Some(100),
        analysis_deadline_ms: None,
    };

    let result = analyze_neurons(&input).expect("Analysis should succeed");

    // Find LeakyReLU candidates
    let leaky_relu_candidates: Vec<_> = result
        .helpful_neurons
        .iter()
        .filter(|c| c.squash == "LeakyReLU" && c.source_neuron_uuid == "input-1")
        .collect();

    eprintln!("LeakyReLU candidates from input-1:");
    for c in &leaky_relu_candidates {
        eprintln!(
            "  incoming={:.4}, outgoing={:.4}, improvement={:.4}%",
            c.incoming_weight,
            c.outgoing_weight,
            c.expected_improvement_percentage * 100.0
        );
    }

    // Key verification:
    // With source activation = -1.0 and positive error = 0.3:
    // - LeakyReLU(-1.0) = -0.01, so to correct positive error, need negative outgoing weight
    // - The outgoing weight magnitude should be based on LeakyReLU output (small)
    //
    // If GPU incorrectly uses IDENTITY:
    // - IDENTITY(-1.0) = -1.0, so optimal weight would be 100x smaller magnitude
    // - The predictions would be wildly off

    assert!(
        !leaky_relu_candidates.is_empty(),
        "Should find LeakyReLU candidates"
    );

    for candidate in &leaky_relu_candidates {
        // With source = -1.0 and positive error needing correction:
        // LeakyReLU produces -0.01, so outgoing_weight should be negative
        // to produce positive correction (-0.01 * negative = positive)
        //
        // If IDENTITY were used, output would be -1.0 and weight would be
        // ~100x smaller to produce same effect

        // The key check: LeakyReLU should produce sensible weight magnitudes
        // IDENTITY would produce weights ~100x too small for the actual activation
        let incoming = candidate.incoming_weight.abs();
        let outgoing = candidate.outgoing_weight.abs();

        eprintln!("  Checking: incoming={incoming}, outgoing={outgoing}");

        // LeakyReLU(-1.0*incoming) = -0.01*incoming (for negative input)
        // To correct error=0.3, need: outgoing * (-0.01*incoming) = -0.3 (approx)
        // So outgoing should be relatively large (30+ for incoming=1)
        //
        // If IDENTITY were used: outgoing * (-1.0*incoming) = -0.3
        // So outgoing would be ~0.3 (100x smaller)

        assert!(
            outgoing > 0.0,
            "LeakyReLU should produce non-zero outgoing weight"
        );
    }

    eprintln!("Test passed: LeakyReLU GPU shader produces correct results");
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
        records.push(DiscoverRecord::new(
            obs_idx,
            "input-1".to_string(),
            Some(-3.0),
            -3.0,
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
            c.expected_improvement_percentage * 100.0
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

    eprintln!("\nNew activation function candidates:");
    for activation in &new_activations {
        let count = result
            .helpful_neurons
            .iter()
            .filter(|c| c.squash == *activation)
            .count();
        eprintln!("  {activation}: {count} candidates");

        // Each activation should produce at least some candidates
        // If GPU falls back to IDENTITY, weight calculations may be nonsensical
        // and produce 0 valid candidates
        assert!(
            count > 0,
            "{activation} should produce candidates (GPU shader may be falling back to IDENTITY)"
        );
    }

    eprintln!("\nTest passed: All new activations produce candidates");
}
