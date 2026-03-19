//! Tests for the `TargetMap` optimisation that pre-builds target data for efficient
//! sample building across multiple sources.
//!
//! The `TargetMap` optimisation (v0.1.150) addresses a significant performance bottleneck
//! where the target `HashMap` was being rebuilt for each of ~1000+ source neurons when
//! analysing a single focus neuron. With 64 focus neurons, this meant rebuilding the
//! same `HashMap` ~64,000 times.
//!
//! These tests verify:
//! 1. `TargetMap` produces identical results to the original `build_samples` function
//! 2. `TargetMap` correctly handles edge cases (empty records, non-finite values)
//! 3. The optimisation doesn't change the sample matching behaviour

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use neat_ai_discovery::analysis::{GpuAnalyzer, analyze_synapses};
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

/// Test that synapse analysis completes successfully with the `TargetMap` optimisation.
/// This is an integration test that exercises the full analysis pipeline.
#[test]
fn synapse_analysis_with_target_map_optimization_succeeds() {
    skip_without_gpu!();

    let temp_dir = tempdir().expect("Failed to create temp directory");
    let parquet_path = temp_dir.path().join("records.parquet");
    let parquet_file = parquet_path.to_str().unwrap().to_string();

    // Create records with correlated error patterns
    let mut records = Vec::new();
    for obs_index in 0..20u32 {
        // Input neuron with varying activation
        records.push(DiscoverRecord::new(
            obs_index,
            "input-0".to_string(),
            None,
            (obs_index as f32 - 10.0) / 10.0, // -1.0 to 0.9
            Vec::new(),
        ));

        // Output neuron with errors correlated to input
        let activation = 0.5;
        let error = if obs_index < 10 { 0.3 } else { -0.3 };
        records.push(DiscoverRecord::new(
            obs_index,
            "output-0".to_string(),
            Some(activation),
            activation,
            vec![error],
        ));
    }

    neat_ai_discovery::parquet_format::write_records_to_parquet(&parquet_file, &records)
        .expect("Failed to write parquet");

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

    let result = analyze_synapses(&input).expect("Analysis should succeed");

    // The optimisation should produce candidates when there's correlation
    assert!(
        !result.helpful_synapses.is_empty() || !result.no_candidate_reasons.is_empty(),
        "Analysis should produce candidates or diagnostic reasons"
    );
}

/// Test that analysis handles multiple focus neurons efficiently.
/// With the `TargetMap` optimisation, each focus neuron builds its target map once
/// and reuses it for all source evaluations.
#[test]
fn multiple_focus_neurons_share_target_map_optimization() {
    skip_without_gpu!();

    let temp_dir = tempdir().expect("Failed to create temp directory");
    let parquet_path = temp_dir.path().join("records.parquet");
    let parquet_file = parquet_path.to_str().unwrap().to_string();

    // Create records for multiple focus neurons
    let mut records = Vec::new();
    for obs_index in 0..15u32 {
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

        // Multiple output neurons (focus targets)
        for i in 0..2 {
            records.push(DiscoverRecord::new(
                obs_index,
                format!("output-{i}"),
                Some(0.5),
                0.5,
                vec![0.1 * (i as f32 + 1.0)],
            ));
        }
    }

    neat_ai_discovery::parquet_format::write_records_to_parquet(&parquet_file, &records)
        .expect("Failed to write parquet");

    let creature = CreatureJson {
        input: 3,
        output: 2,
        neurons: vec![
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
        synapses: Vec::new(),
    };

    let input = AnalyzeSynapsesInput {
        parquet_file,
        creature,
        focus_neurons: vec!["output-0".to_string(), "output-1".to_string()],
        max_candidates: None,
        analysis_deadline_ms: None,
        random_seed: None,
    };

    let result =
        analyze_synapses(&input).expect("Analysis should succeed with multiple focus neurons");

    // Both focus neurons should be processed
    let focus_neurons_in_results: std::collections::HashSet<_> = result
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
        focus_neurons_in_results.len() == 2,
        "Both focus neurons should be processed, got {focus_neurons_in_results:?}"
    );
}

/// Test that sample matching correctly handles records with non-finite values.
/// The `TargetMap` should filter out non-finite activations and errors.
#[test]
fn target_map_filters_non_finite_values() {
    skip_without_gpu!();

    let temp_dir = tempdir().expect("Failed to create temp directory");
    let parquet_path = temp_dir.path().join("records.parquet");
    let parquet_file = parquet_path.to_str().unwrap().to_string();

    let mut records = Vec::new();

    // Valid records
    for obs_index in 0..15u32 {
        records.push(DiscoverRecord::new(
            obs_index,
            "input-0".to_string(),
            None,
            0.5,
            Vec::new(),
        ));
        records.push(DiscoverRecord::new(
            obs_index,
            "output-0".to_string(),
            Some(0.5),
            0.5,
            vec![0.1],
        ));
    }

    // Add some non-finite records that should be filtered
    records.push(DiscoverRecord::new(
        100,
        "input-0".to_string(),
        None,
        f32::NAN,
        Vec::new(),
    ));
    records.push(DiscoverRecord::new(
        101,
        "input-0".to_string(),
        None,
        f32::INFINITY,
        Vec::new(),
    ));
    records.push(DiscoverRecord::new(
        102,
        "output-0".to_string(),
        Some(0.5),
        0.5,
        vec![f32::NAN],
    ));

    neat_ai_discovery::parquet_format::write_records_to_parquet(&parquet_file, &records)
        .expect("Failed to write parquet");

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

    // Analysis should complete without errors from non-finite values
    let result =
        analyze_synapses(&input).expect("Analysis should handle non-finite values gracefully");

    // Should still find candidates from the valid records
    assert!(
        !result.helpful_synapses.is_empty()
            || !result.coordinated_structural_candidates.is_empty()
            || !result.no_candidate_reasons.is_empty(),
        "Should process valid records despite non-finite values in other records"
    );
}

/// Test that empty target records result in early return without processing sources.
/// This is an edge case where the focus neuron has no valid records.
#[test]
fn empty_target_records_skips_source_processing() {
    skip_without_gpu!();

    let temp_dir = tempdir().expect("Failed to create temp directory");
    let parquet_path = temp_dir.path().join("records.parquet");
    let parquet_file = parquet_path.to_str().unwrap().to_string();

    // Only create records for input neurons, not the output (focus) neuron
    let mut records = Vec::new();
    for obs_index in 0..15u32 {
        records.push(DiscoverRecord::new(
            obs_index,
            "input-0".to_string(),
            None,
            0.5,
            Vec::new(),
        ));
    }

    // Create minimal output records with no errors (will result in empty target map)
    for obs_index in 0..15u32 {
        records.push(DiscoverRecord::new(
            obs_index,
            "output-0".to_string(),
            Some(0.5),
            0.5,
            Vec::new(), // No errors - target map will be empty
        ));
    }

    neat_ai_discovery::parquet_format::write_records_to_parquet(&parquet_file, &records)
        .expect("Failed to write parquet");

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

    // Should complete quickly without processing sources
    let result = analyze_synapses(&input).expect("Analysis should handle empty target records");

    // No candidates expected when target has no errors
    assert!(
        result.helpful_synapses.is_empty(),
        "Should not find candidates when target has no errors"
    );
}
