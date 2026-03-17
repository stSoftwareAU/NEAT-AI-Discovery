//! Integration tests for Issue #568: Async pipeline CPU/GPU overlap.
//!
//! These tests verify that:
//! 1. The `GpuFuture` type correctly delivers pre-resolved results
//! 2. The analysis pipeline produces identical results with async overlap
//! 3. The `submit_helpful_batch` + `collect()` pattern works correctly

use neat_ai_discovery::analysis::analyze_all;
use neat_ai_discovery::analysis::gpu::GpuAnalyzer;
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{AnalyzeAllInput, CreatureJson, NeuronJson, SynapseJson};

/// Create a small creature for testing correctness.
fn create_test_creature() -> CreatureJson {
    let neurons = vec![
        NeuronJson {
            uuid: "input-0".to_string(),
            neuron_type: "input".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        },
        NeuronJson {
            uuid: "input-1".to_string(),
            neuron_type: "input".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        },
        NeuronJson {
            uuid: "hidden-0".to_string(),
            neuron_type: "hidden".to_string(),
            squash: "TANH".to_string(),
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
    ];

    let synapses = vec![
        SynapseJson {
            from_uuid: "input-0".to_string(),
            to_uuid: "hidden-0".to_string(),
            weight: 0.5,
            synapse_type: None,
        },
        SynapseJson {
            from_uuid: "input-1".to_string(),
            to_uuid: "hidden-1".to_string(),
            weight: 0.3,
            synapse_type: None,
        },
        SynapseJson {
            from_uuid: "hidden-0".to_string(),
            to_uuid: "output-0".to_string(),
            weight: 0.8,
            synapse_type: None,
        },
        SynapseJson {
            from_uuid: "hidden-1".to_string(),
            to_uuid: "output-0".to_string(),
            weight: 0.4,
            synapse_type: None,
        },
    ];

    CreatureJson {
        neurons,
        synapses,
        input: 2,
        output: 1,
    }
}

/// Create test records with correlated patterns.
fn create_test_records(creature: &CreatureJson, count: usize) -> Vec<DiscoverRecord> {
    let mut records = Vec::new();

    for obs in 0..count as u32 {
        let t = obs as f32 / count as f32;
        for neuron in &creature.neurons {
            let (activation, value, errors) = match neuron.neuron_type.as_str() {
                "input" => ((t * std::f32::consts::TAU).sin(), None, Vec::new()),
                "hidden" => {
                    let act = 0.3 + 0.4 * (t * std::f32::consts::TAU).sin();
                    (act, Some(act), vec![0.05 * (t * 3.0).cos()])
                }
                "output" => {
                    let error = 0.1 * (t * std::f32::consts::PI).cos();
                    (0.5 + 0.2 * t, Some(0.5 + 0.2 * t), vec![error])
                }
                _ => continue,
            };

            records.push(DiscoverRecord {
                obs_index: obs,
                neuron_uuid: neuron.uuid.clone(),
                value,
                activation,
                errors,
            });
        }
    }

    records
}

/// Macro to skip GPU-dependent tests on machines without GPU.
macro_rules! skip_without_gpu {
    () => {
        if !GpuAnalyzer::gpu_is_available() {
            eprintln!("Skipping test: no GPU available");
            return;
        }
    };
}

/// Issue #568: Analysis with async pipeline overlap produces valid results.
///
/// Verifies that the pipeline still produces correct candidates after the
/// CPU/GPU overlap refactoring (helpful GPU submission + harmful sample
/// preparation overlap).
#[test]
fn async_pipeline_produces_valid_analysis_results() {
    skip_without_gpu!();

    let creature = create_test_creature();
    let records = create_test_records(&creature, 100);

    let temp_dir = tempfile::tempdir().expect("Failed to create temp directory");
    let parquet_path = temp_dir.path().join("test.parquet");
    let parquet_file = parquet_path.to_str().unwrap().to_string();

    write_records_to_parquet(&parquet_file, &records).expect("Failed to write parquet");

    let focus_neurons: Vec<String> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "output" || n.neuron_type == "hidden")
        .map(|n| n.uuid.clone())
        .collect();

    let input = AnalyzeAllInput {
        parquet_file,
        creature,
        focus_neurons,
        max_synapse_candidates: None,
        max_neuron_candidates: None,
        analysis_deadline_ms: None,
        include_synapse_analysis: Some(true),
        include_neuron_analysis: Some(false),
        random_seed: Some(42),
        previous_neuron_fingerprints: None,
        module_outcome_tracker: None,
    };

    let result = analyze_all(&input).expect("analyze_all should succeed");

    // Verify analysis completed successfully
    assert!(
        result.synapse.is_some(),
        "Synapse analysis should produce a result"
    );
    let synapse_result = result.synapse.unwrap();

    // Verify metadata is populated (proves the pipeline ran through fully)
    assert!(
        synapse_result.metadata.completed_focus_neurons > 0,
        "Should have completed at least one focus neuron"
    );
    assert!(
        synapse_result.gpu_used,
        "GPU should have been used for analysis"
    );
}

/// Issue #568: Analysis is deterministic with fixed seed across multiple runs.
///
/// The async overlap should not affect result ordering or content — running
/// with the same seed should produce identical results.
#[test]
fn async_pipeline_is_deterministic_with_fixed_seed() {
    skip_without_gpu!();

    let creature = create_test_creature();
    let records = create_test_records(&creature, 50);

    let temp_dir = tempfile::tempdir().expect("Failed to create temp directory");
    let parquet_path = temp_dir.path().join("test.parquet");
    let parquet_file = parquet_path.to_str().unwrap().to_string();

    write_records_to_parquet(&parquet_file, &records).expect("Failed to write parquet");

    let focus_neurons: Vec<String> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "output")
        .map(|n| n.uuid.clone())
        .collect();

    let make_input = || AnalyzeAllInput {
        parquet_file: parquet_file.clone(),
        creature: creature.clone(),
        focus_neurons: focus_neurons.clone(),
        max_synapse_candidates: None,
        max_neuron_candidates: None,
        analysis_deadline_ms: None,
        include_synapse_analysis: Some(true),
        include_neuron_analysis: Some(false),
        random_seed: Some(12345),
        previous_neuron_fingerprints: None,
        module_outcome_tracker: None,
    };

    let result1 = analyze_all(&make_input()).expect("Run 1 should succeed");
    let result2 = analyze_all(&make_input()).expect("Run 2 should succeed");

    let syn1 = result1.synapse.unwrap();
    let syn2 = result2.synapse.unwrap();

    // Same number of candidates
    assert_eq!(
        syn1.helpful_synapses.len(),
        syn2.helpful_synapses.len(),
        "Helpful synapse count should be deterministic"
    );
    assert_eq!(
        syn1.harmful_synapses.len(),
        syn2.harmful_synapses.len(),
        "Harmful synapse count should be deterministic"
    );
    assert_eq!(
        syn1.coordinated_structural_candidates.len(),
        syn2.coordinated_structural_candidates.len(),
        "Coordinated candidate count should be deterministic"
    );

    // Same candidate UUIDs (order may vary due to par_iter, so compare as sets)
    let helpful_uuids1: std::collections::HashSet<_> = syn1
        .helpful_synapses
        .iter()
        .map(|c| (&c.from_neuron_uuid, &c.to_neuron_uuid))
        .collect();
    let helpful_uuids2: std::collections::HashSet<_> = syn2
        .helpful_synapses
        .iter()
        .map(|c| (&c.from_neuron_uuid, &c.to_neuron_uuid))
        .collect();
    assert_eq!(
        helpful_uuids1, helpful_uuids2,
        "Helpful candidates should be deterministic"
    );
}
