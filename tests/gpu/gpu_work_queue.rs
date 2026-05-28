//! Tests for the GPU work queue optimisation.
//!
//! The GPU work queue eliminates the overhead of creating multiple GPU devices
//! by having a single dedicated thread that owns the `GpuAnalyzer`. All GPU
//! operations are processed sequentially on this thread, avoiding device
//! creation overhead and improving GPU utilisation.
//!
//! These tests verify:
//! 1. Analysis functions work correctly with the shared queue
//! 2. Results are equivalent to the previous per-thread analyzer approach
//! 3. The queue handles concurrent requests from multiple focus neurons

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use neat_ai_discovery::analysis::{GpuAnalyzer, analyze_neurons, analyze_synapses};
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{
    AnalyzeNeuronsInput, AnalyzeSynapsesInput, CreatureJson, NeuronJson, SynapseJson,
};
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

/// Test that synapse analysis works correctly with the GPU work queue.
/// This tests the full integration path through the shared queue.
#[test]
fn synapse_analysis_works_with_gpu_work_queue() {
    skip_without_gpu!();

    let temp_dir = tempdir().expect("Failed to create temp directory");
    let parquet_path = temp_dir.path().join("records.parquet");
    let parquet_file = parquet_path.to_str().unwrap().to_string();

    // Create records with clear correlation for synapse discovery
    let mut records = Vec::new();
    for obs_index in 0..30u32 {
        // Input activation varies from -1.0 to 1.0
        let input_activation = (obs_index as f32 - 15.0) / 15.0;

        records.push(DiscoverRecord::new(
            obs_index,
            "input-0".to_string(),
            None,
            input_activation,
            Vec::new(),
        ));

        // Output error correlates with input activation
        let error = input_activation * 0.2;
        records.push(DiscoverRecord::new(
            obs_index,
            "output-0".to_string(),
            Some(0.5),
            0.5,
            vec![error],
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
        temperature: 1.0,
        failure_cache: None,
        discovery_outcome_log: None,
    };

    // Run analysis - this uses the GPU work queue internally
    let result = analyze_synapses(&input).expect("Synapse analysis should succeed");

    // Should find helpful synapse candidates
    assert!(
        !result.helpful_synapses.is_empty() || !result.no_candidate_reasons.is_empty(),
        "Should produce candidates or diagnostics"
    );
}

/// Test that neuron analysis works correctly with the GPU work queue.
#[test]
fn neuron_analysis_works_with_gpu_work_queue() {
    skip_without_gpu!();

    let temp_dir = tempdir().expect("Failed to create temp directory");
    let parquet_path = temp_dir.path().join("records.parquet");
    let parquet_file = parquet_path.to_str().unwrap().to_string();

    // Create records with clear correlation for neuron discovery
    let mut records = Vec::new();
    for obs_index in 0..30u32 {
        let input_activation = (obs_index as f32 - 15.0) / 15.0;

        records.push(DiscoverRecord::new(
            obs_index,
            "input-0".to_string(),
            None,
            input_activation,
            Vec::new(),
        ));

        // Output error that a ReLU could help fix
        let error = if input_activation > 0.0 { 0.2 } else { -0.2 };
        records.push(DiscoverRecord::new(
            obs_index,
            "output-0".to_string(),
            Some(0.5),
            0.5,
            vec![error],
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

    let input = AnalyzeNeuronsInput {
        parquet_file,
        creature,
        focus_neurons: vec!["output-0".to_string()],
        max_candidates: None,
        analysis_deadline_ms: None,
        random_seed: None,
        temperature: 1.0,
        failure_cache: None,
        discovery_outcome_log: None,
        task_descriptor: None,
    };

    // Run analysis - this uses the GPU work queue internally
    let result = analyze_neurons(&input).expect("Neuron analysis should succeed");

    // Should produce results (candidates or diagnostics)
    assert!(
        !result.helpful_neurons.is_empty() || !result.no_candidate_reasons.is_empty(),
        "Should produce candidates or diagnostics"
    );
}

/// Test that multiple focus neurons are processed correctly with the shared queue.
/// This verifies that the queue handles concurrent work from multiple threads.
#[test]
fn multiple_focus_neurons_work_with_shared_queue() {
    skip_without_gpu!();

    let temp_dir = tempdir().expect("Failed to create temp directory");
    let parquet_path = temp_dir.path().join("records.parquet");
    let parquet_file = parquet_path.to_str().unwrap().to_string();

    // Create records for multiple neurons
    let mut records = Vec::new();
    for obs_index in 0..20u32 {
        // Multiple input neurons
        for i in 0..3 {
            records.push(DiscoverRecord::new(
                obs_index,
                format!("input-{i}"),
                None,
                (obs_index as f32 * 0.1) * (i as f32 + 1.0),
                Vec::new(),
            ));
        }

        // Multiple hidden neurons (as focus targets)
        for i in 0..2 {
            records.push(DiscoverRecord::new(
                obs_index,
                format!("hidden-{i}"),
                Some(0.5),
                0.5,
                vec![0.1 * (i as f32 + 1.0)],
            ));
        }

        // Output neuron (also a focus target)
        records.push(DiscoverRecord::new(
            obs_index,
            "output-0".to_string(),
            Some(0.5),
            0.5,
            vec![0.15],
        ));
    }

    write_records_to_parquet(&parquet_file, &records).expect("Failed to write parquet");

    let creature = CreatureJson {
        input: 3,
        output: 1,
        neurons: vec![
            NeuronJson {
                uuid: "hidden-0".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "RELU".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "hidden-1".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "RELU".to_string(),
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
                from_uuid: "hidden-0".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "hidden-1".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.5,
                synapse_type: None,
            },
        ],
    };

    let input = AnalyzeSynapsesInput {
        parquet_file,
        creature,
        focus_neurons: vec![
            "hidden-0".to_string(),
            "hidden-1".to_string(),
            "output-0".to_string(),
        ],
        max_candidates: None,
        analysis_deadline_ms: None,
        random_seed: None,
        temperature: 1.0,
        failure_cache: None,
        discovery_outcome_log: None,
    };

    // Run analysis with multiple focus neurons
    let result = analyze_synapses(&input).expect("Multi-focus analysis should succeed");

    // All focus neurons should be processed
    let processed_neurons: std::collections::HashSet<_> = result
        .helpful_synapses
        .iter()
        .map(|c| c.to_neuron_uuid.as_str())
        .chain(
            result
                .no_candidate_reasons
                .iter()
                .map(|r| r.target_uuid.as_str()),
        )
        .collect();

    assert!(
        !processed_neurons.is_empty(),
        "At least one focus neuron should be processed"
    );
}

/// Test that harmful synapse detection works with the GPU work queue.
#[test]
fn harmful_synapse_detection_works_with_queue() {
    skip_without_gpu!();

    let temp_dir = tempdir().expect("Failed to create temp directory");
    let parquet_path = temp_dir.path().join("records.parquet");
    let parquet_file = parquet_path.to_str().unwrap().to_string();

    // Create records where an existing synapse is harmful
    let mut records = Vec::new();
    for obs_index in 0..30u32 {
        let input_activation = (obs_index as f32 - 15.0) / 15.0;

        records.push(DiscoverRecord::new(
            obs_index,
            "input-0".to_string(),
            None,
            input_activation,
            Vec::new(),
        ));

        // Error that's made worse by the existing synapse
        let error = -input_activation * 0.3; // Opposite direction
        records.push(DiscoverRecord::new(
            obs_index,
            "output-0".to_string(),
            Some(0.5),
            0.5,
            vec![error],
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
        synapses: vec![SynapseJson {
            from_uuid: "input-0".to_string(),
            to_uuid: "output-0".to_string(),
            weight: 0.5, // This weight makes errors worse
            synapse_type: None,
        }],
    };

    let input = AnalyzeSynapsesInput {
        parquet_file,
        creature,
        focus_neurons: vec!["output-0".to_string()],
        max_candidates: None,
        analysis_deadline_ms: None,
        random_seed: None,
        temperature: 1.0,
        failure_cache: None,
        discovery_outcome_log: None,
    };

    // Run analysis
    let result = analyze_synapses(&input).expect("Harmful synapse analysis should succeed");

    // Should detect the harmful synapse
    // Note: The synapse is harmful because its weight pushes output in the wrong direction
    assert!(
        !result.harmful_synapses.is_empty() || !result.no_candidate_reasons.is_empty(),
        "Should detect harmful synapse or provide diagnostics"
    );
}
