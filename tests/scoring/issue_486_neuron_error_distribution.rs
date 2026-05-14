//! Tests for error distribution computation in neuron analysis (Issue #486).
//!
//! Verifies that `analyze_neurons()` populates the `error_distribution` field
//! in `NeuronAnalysisMetadata` by computing `ErrorDistribution::from_errors()`
//! from the target neuron records' error samples.

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

/// Helper to build a simple creature with 2 inputs, 1 output.
fn simple_creature() -> CreatureJson {
    CreatureJson {
        neurons: vec![NeuronJson {
            uuid: "output-0".to_string(),
            neuron_type: "output".to_string(),
            squash: "HARD_TANH".to_string(),
            bias: 0.0,
        }],
        synapses: vec![SynapseJson {
            from_uuid: "input-0".to_string(),
            to_uuid: "output-0".to_string(),
            weight: 1.0,
            synapse_type: None,
        }],
        input: 2,
        output: 1,
    }
}

// =============================================================================
// Test: neuron analysis populates error_distribution for focus target neurons
// =============================================================================

/// Test: `analyze_neurons` populates the `error_distribution` field in metadata
/// when the focus target neurons have error samples.
#[test]
fn test_neuron_analysis_populates_error_distribution() {
    skip_without_gpu!();

    let creature = simple_creature();

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    let mut records = Vec::new();

    // Create 100 observations with known error pattern:
    // 90 samples with low error (0.1), 10 samples with high error (1.0)
    for i in 0..100u32 {
        let error = if i < 90 { 0.1 } else { 1.0 };

        // Target (output) neuron records with errors
        records.push(DiscoverRecord::new(
            i,
            "output-0".to_string(),
            Some(0.0),
            0.0,
            vec![error],
        ));

        // Source input neurons (needed for the analysis pipeline)
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
        focus_neurons: vec!["output-0".to_string()],
        max_candidates: Some(10),
        analysis_deadline_ms: None,
        random_seed: Some(42),
        temperature: 1.0,
        failure_cache: None,
        discovery_outcome_log: None,
    };

    let result = analyze_neurons(&input).expect("Neuron analysis should succeed");

    // The error_distribution field MUST be populated (this is the Issue #486 fix)
    assert!(
        result.metadata.error_distribution.is_some(),
        "Neuron analysis should populate error_distribution in metadata"
    );

    let dist = result.metadata.error_distribution.as_ref().unwrap();

    // Verify basic statistical properties
    assert!(
        dist.mean > 0.0,
        "Mean error should be positive, got {}",
        dist.mean
    );
    assert!(
        dist.std_dev > 0.0,
        "Std dev should be positive, got {}",
        dist.std_dev
    );
    assert_eq!(dist.percentiles.len(), 5, "Should have 5 percentiles");
    assert!(dist.sample_count > 0, "Sample count should be positive");

    // With 90% at 0.1 and 10% at 1.0, the distribution should be right-skewed
    assert!(
        dist.skewness > 0.0,
        "Distribution with outliers should have positive skewness, got {}",
        dist.skewness
    );

    // Max should be significantly higher than mean due to outliers
    assert!(
        dist.max > dist.mean * 2.0,
        "Max ({}) should be much higher than mean ({}) due to outliers",
        dist.max,
        dist.mean
    );

    eprintln!(
        "Error distribution: mean={:.4}, std_dev={:.4}, skewness={:.4}, kurtosis={:.4}, samples={}",
        dist.mean, dist.std_dev, dist.skewness, dist.kurtosis, dist.sample_count
    );
}

// =============================================================================
// Test: neuron analysis returns None when no error samples exist
// =============================================================================

/// Test: `analyze_neurons` returns `None` for `error_distribution` when
/// target neurons have no error data.
#[test]
fn test_neuron_analysis_no_errors_returns_none() {
    skip_without_gpu!();

    let creature = simple_creature();

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    let mut records = Vec::new();

    // Create records with EMPTY error vectors for target neurons
    for i in 0..50u32 {
        records.push(DiscoverRecord::new(
            i,
            "output-0".to_string(),
            Some(0.0),
            0.0,
            vec![], // No errors
        ));
        records.push(DiscoverRecord::new(
            i,
            "input-0".to_string(),
            Some(0.5),
            0.5,
            vec![],
        ));
        records.push(DiscoverRecord::new(
            i,
            "input-1".to_string(),
            Some(0.3),
            0.3,
            vec![],
        ));
    }

    write_records_to_parquet(file_path, &records).unwrap();

    let input = AnalyzeNeuronsInput {
        parquet_file: file_path.to_string(),
        creature,
        focus_neurons: vec!["output-0".to_string()],
        max_candidates: Some(10),
        analysis_deadline_ms: None,
        random_seed: Some(42),
        temperature: 1.0,
        failure_cache: None,
        discovery_outcome_log: None,
    };

    let result = analyze_neurons(&input).expect("Neuron analysis should succeed");

    // With no errors, distribution should be None
    assert!(
        result.metadata.error_distribution.is_none(),
        "Neuron analysis with no errors should have None error_distribution"
    );
}

// =============================================================================
// Test: error distribution aggregates across multiple focus targets
// =============================================================================

/// Test: When multiple focus neurons are analysed, error distribution is
/// computed from the combined error samples of all focus targets.
#[test]
fn test_neuron_analysis_error_distribution_multiple_targets() {
    skip_without_gpu!();

    // Creature with 2 output neurons (both as focus targets)
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
        ],
        input: 2,
        output: 2,
    };

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    let mut records = Vec::new();

    for i in 0..100u32 {
        // output-0 has errors ~0.2
        records.push(DiscoverRecord::new(
            i,
            "output-0".to_string(),
            Some(0.0),
            0.0,
            vec![0.2],
        ));

        // output-1 has errors ~0.8
        records.push(DiscoverRecord::new(
            i,
            "output-1".to_string(),
            Some(0.0),
            0.0,
            vec![0.8],
        ));

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
        focus_neurons: vec!["output-0".to_string(), "output-1".to_string()],
        max_candidates: Some(10),
        analysis_deadline_ms: None,
        random_seed: Some(42),
        temperature: 1.0,
        failure_cache: None,
        discovery_outcome_log: None,
    };

    let result = analyze_neurons(&input).expect("Neuron analysis should succeed");

    assert!(
        result.metadata.error_distribution.is_some(),
        "Neuron analysis with multiple targets should populate error_distribution"
    );

    let dist = result.metadata.error_distribution.as_ref().unwrap();

    // Combined distribution should have samples from both targets (200 total)
    assert!(
        dist.sample_count >= 100,
        "Should have samples from multiple targets, got {}",
        dist.sample_count
    );

    // Mean should be between 0.2 and 0.8 (combined from both targets)
    assert!(
        dist.mean > 0.1 && dist.mean < 0.9,
        "Combined mean should be between 0.1 and 0.9, got {}",
        dist.mean
    );

    eprintln!(
        "Multi-target distribution: mean={:.4}, samples={}, skewness={:.4}",
        dist.mean, dist.sample_count, dist.skewness
    );
}
