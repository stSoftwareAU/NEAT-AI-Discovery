//! Tests for lock-free error value collection in neuron analysis (Issue #834).
//!
//! Verifies that error values collected via Rayon fold/reduce produce identical
//! results to the previous Mutex-based approach. The `error_distribution` in
//! `NeuronAnalysisMetadata` must be correctly populated from all focus targets.

use neat_ai_discovery::analysis::{GpuAnalyzer, analyze_neurons};
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{AnalyzeNeuronsInput, CreatureJson, NeuronJson, SynapseJson};
use tempfile::NamedTempFile;

/// Skip test if no GPU available.
macro_rules! skip_without_gpu {
    () => {
        if !GpuAnalyzer::gpu_is_available() {
            eprintln!("Skipping test: no GPU available");
            return;
        }
    };
}

// =============================================================================
// Test: error collection across many focus targets uses lock-free fold/reduce
// =============================================================================

/// Test: With many focus targets (simulating parallel contention), the error
/// distribution is correctly aggregated without mutex contention.
#[test]
fn test_lock_free_error_collection_many_targets() {
    skip_without_gpu!();

    // Create a creature with 4 output neurons to exercise parallel collection
    let creature = CreatureJson {
        neurons: vec![
            NeuronJson {
                uuid: "output-0".to_string(),
                neuron_type: "output".to_string(),
                squash: "HARD_TANH".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "output-1".to_string(),
                neuron_type: "output".to_string(),
                squash: "HARD_TANH".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "output-2".to_string(),
                neuron_type: "output".to_string(),
                squash: "HARD_TANH".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "output-3".to_string(),
                neuron_type: "output".to_string(),
                squash: "HARD_TANH".to_string(),
                bias: 0.0,
            },
        ],
        synapses: vec![
            SynapseJson {
                from_uuid: "input-0".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 1.0,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-0".to_string(),
                to_uuid: "output-1".to_string(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-1".to_string(),
                to_uuid: "output-2".to_string(),
                weight: 0.8,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-1".to_string(),
                to_uuid: "output-3".to_string(),
                weight: 0.3,
                synapse_type: None,
            },
        ],
        input: 2,
        output: 4,
    };

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    let mut records = Vec::new();

    // Each output has distinct error values for verifiable aggregation
    for i in 0..50u32 {
        // output-0: errors = 0.1
        records.push(DiscoverRecord::new(
            i,
            "output-0".to_string(),
            Some(0.0),
            0.0,
            vec![0.1],
        ));
        // output-1: errors = 0.3
        records.push(DiscoverRecord::new(
            i,
            "output-1".to_string(),
            Some(0.0),
            0.0,
            vec![0.3],
        ));
        // output-2: errors = 0.5
        records.push(DiscoverRecord::new(
            i,
            "output-2".to_string(),
            Some(0.0),
            0.0,
            vec![0.5],
        ));
        // output-3: errors = 0.9
        records.push(DiscoverRecord::new(
            i,
            "output-3".to_string(),
            Some(0.0),
            0.0,
            vec![0.9],
        ));

        // Source inputs
        records.push(DiscoverRecord::new(
            i,
            "input-0".to_string(),
            Some(0.5),
            0.5,
            vec![0.0],
        ));
        records.push(DiscoverRecord::new(
            i,
            "input-1".to_string(),
            Some(0.3),
            0.3,
            vec![0.0],
        ));
    }

    write_records_to_parquet(file_path, &records).unwrap();

    let input = AnalyzeNeuronsInput {
        parquet_file: file_path.to_string(),
        creature,
        focus_neurons: vec![
            "output-0".to_string(),
            "output-1".to_string(),
            "output-2".to_string(),
            "output-3".to_string(),
        ],
        max_candidates: Some(20),
        analysis_deadline_ms: None,
        random_seed: Some(42),
        temperature: 1.0,
    };

    let result = analyze_neurons(&input).expect("Neuron analysis should succeed");

    // Error distribution must be populated from all 4 focus targets
    assert!(
        result.metadata.error_distribution.is_some(),
        "Error distribution should be populated from lock-free collection"
    );

    let dist = result.metadata.error_distribution.as_ref().unwrap();

    // 4 targets × 50 observations × 1 error each = 200 total samples
    assert_eq!(
        dist.sample_count, 200,
        "Should have 200 error samples from 4 targets × 50 observations"
    );

    // Mean of (0.1, 0.3, 0.5, 0.9) equally weighted = 0.45
    let expected_mean = 0.45;
    assert!(
        (dist.mean - expected_mean).abs() < 0.01,
        "Mean should be approximately {expected_mean}, got {}",
        dist.mean
    );

    // Min should be 0.1, max should be 0.9
    assert!(
        (dist.min - 0.1).abs() < 0.01,
        "Min should be approximately 0.1, got {}",
        dist.min
    );
    assert!(
        (dist.max - 0.9).abs() < 0.01,
        "Max should be approximately 0.9, got {}",
        dist.max
    );

    eprintln!(
        "Lock-free collection: mean={:.4}, std_dev={:.4}, samples={}, min={:.4}, max={:.4}",
        dist.mean, dist.std_dev, dist.sample_count, dist.min, dist.max
    );
}
