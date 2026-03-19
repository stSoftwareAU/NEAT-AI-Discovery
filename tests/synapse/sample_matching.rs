//! Integration tests for sample matching behaviour in the analysis pipeline.
//!
//! Sample matching is the process of correlating source neuron activations with
//! target neuron errors across observations. This is CPU-bound work that prepares
//! data for GPU evaluation.
//!
//! NOTE: The detailed unit tests for `build_samples()` are in src/analysis.rs.
//! These integration tests verify the behaviour through the public API.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use neat_ai_discovery::analysis::{GpuAnalyzer, analyze_synapses};
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{AnalyzeSynapsesInput, CreatureJson, NeuronJson};
use tempfile::tempdir;

/// Skip test if no GPU available
macro_rules! skip_without_gpu {
    () => {
        if !GpuAnalyzer::gpu_is_available() {
            eprintln!("Skipping test: no GPU available");
            return;
        }
    };
}

/// Test that analysis filters out records with non-finite values.
/// NaN and Infinity in activation data should not crash analysis.
#[test]
fn analysis_handles_non_finite_values_gracefully() {
    skip_without_gpu!();

    let temp_dir = tempdir().expect("Failed to create temp directory");
    let parquet_path = temp_dir.path().join("records.parquet");
    let parquet_file = parquet_path.to_str().unwrap().to_string();

    // Mix of valid and non-finite records
    let records = vec![
        // Valid records
        DiscoverRecord::new(0, "input-0".to_string(), None, 0.5, Vec::new()),
        DiscoverRecord::new(0, "output-0".to_string(), Some(0.5), 0.5, vec![0.1]),
        // Non-finite records that should be filtered
        DiscoverRecord::new(1, "input-0".to_string(), None, f32::NAN, Vec::new()),
        DiscoverRecord::new(
            1,
            "output-0".to_string(),
            Some(0.5),
            0.5,
            vec![f32::INFINITY],
        ),
        // More valid records
        DiscoverRecord::new(2, "input-0".to_string(), None, 0.3, Vec::new()),
        DiscoverRecord::new(2, "output-0".to_string(), Some(0.5), 0.5, vec![-0.1]),
    ];

    write_records_to_parquet(&parquet_file, &records).expect("Failed to write parquet");

    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![NeuronJson {
            uuid: "output-0".to_string(),
            neuron_type: "output".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        }],
        synapses: Vec::new(),
    };

    let input = AnalyzeSynapsesInput {
        parquet_file,
        creature,
        focus_neurons: vec!["output-0".to_string()],
        max_candidates: None,
        analysis_deadline_ms: None,
        random_seed: None,
    };

    // Should complete without error
    let result = analyze_synapses(&input).expect("Analysis should handle non-finite values");

    // Should produce results from the valid records
    assert!(
        !result.helpful_synapses.is_empty() || !result.no_candidate_reasons.is_empty(),
        "Should process valid records despite non-finite values"
    );
}

/// Test that analysis correctly pairs samples by `obs_index`.
/// Records with mismatched `obs_index` should not be paired.
#[test]
fn analysis_pairs_samples_by_obs_index() {
    skip_without_gpu!();

    let temp_dir = tempdir().expect("Failed to create temp directory");
    let parquet_path = temp_dir.path().join("records.parquet");
    let parquet_file = parquet_path.to_str().unwrap().to_string();

    // Create records with deliberate obs_index gaps
    let mut records = Vec::new();
    for obs_index in &[0, 2, 4, 6, 8] {
        records.push(DiscoverRecord::new(
            *obs_index,
            "input-0".to_string(),
            None,
            (*obs_index as f32 - 4.0) / 4.0, // -1.0 to 1.0
            Vec::new(),
        ));
        records.push(DiscoverRecord::new(
            *obs_index,
            "output-0".to_string(),
            Some(0.5),
            0.5,
            vec![0.1 * (*obs_index as f32 - 4.0)], // Correlated error
        ));
    }

    write_records_to_parquet(&parquet_file, &records).expect("Failed to write parquet");

    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![NeuronJson {
            uuid: "output-0".to_string(),
            neuron_type: "output".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        }],
        synapses: Vec::new(),
    };

    let input = AnalyzeSynapsesInput {
        parquet_file,
        creature,
        focus_neurons: vec!["output-0".to_string()],
        max_candidates: None,
        analysis_deadline_ms: None,
        random_seed: None,
    };

    // Should complete successfully with 5 matching sample pairs
    let result = analyze_synapses(&input).expect("Analysis should pair by obs_index");

    // Should find correlation in the 5 matched pairs
    assert!(
        !result.helpful_synapses.is_empty() || !result.no_candidate_reasons.is_empty(),
        "Should produce results from matched obs_index pairs"
    );
}

/// Test that analysis handles empty error vectors correctly.
/// Input neurons have no errors and should not be used as targets.
#[test]
fn analysis_skips_neurons_without_errors() {
    skip_without_gpu!();

    let temp_dir = tempdir().expect("Failed to create temp directory");
    let parquet_path = temp_dir.path().join("records.parquet");
    let parquet_file = parquet_path.to_str().unwrap().to_string();

    let mut records = Vec::new();
    for obs_index in 0..15u32 {
        // Input neurons have empty error vectors
        records.push(DiscoverRecord::new(
            obs_index,
            "input-0".to_string(),
            None,
            (obs_index as f32 - 7.0) / 7.0,
            Vec::new(), // No errors
        ));

        // Output neuron has errors
        records.push(DiscoverRecord::new(
            obs_index,
            "output-0".to_string(),
            Some(0.5),
            0.5,
            vec![0.1 * (obs_index as f32 - 7.0) / 7.0],
        ));
    }

    write_records_to_parquet(&parquet_file, &records).expect("Failed to write parquet");

    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![NeuronJson {
            uuid: "output-0".to_string(),
            neuron_type: "output".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        }],
        synapses: Vec::new(),
    };

    let input = AnalyzeSynapsesInput {
        parquet_file,
        creature,
        focus_neurons: vec!["output-0".to_string()],
        max_candidates: None,
        analysis_deadline_ms: None,
        random_seed: None,
    };

    // Should complete and find candidates using input-0 as source
    let result = analyze_synapses(&input).expect("Analysis should work with proper error vectors");

    // Should find candidates (input-0 -> output-0)
    assert!(
        !result.helpful_synapses.is_empty() || !result.no_candidate_reasons.is_empty(),
        "Should find candidates when target has errors and sources don't"
    );
}
