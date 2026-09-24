//! Tests for error distribution analysis (Issue #192).
//!
//! This module tests the error distribution analysis functionality that enables
//! targeted discovery for specific error patterns:
//! - Outlier samples with high error
//! - Bimodal error distributions
//! - Error clusters
//!
//! The analysis computes distribution statistics including percentiles, skewness,
//! and kurtosis to help identify non-uniform error patterns.

#![allow(clippy::cast_precision_loss, clippy::cast_sign_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use neat_ai_discovery::analysis::samples::HelpfulSample;
use neat_ai_discovery::analysis::scoring::error_distribution::ErrorDistribution;
use neat_ai_discovery::analysis::{GpuAnalyzer, analyze_synapses};
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{AnalyzeSynapsesInput, CreatureJson, NeuronJson, SynapseJson};
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

/// Helper to create a basic creature with input and output neurons.
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

// =============================================================================
// ErrorDistribution Unit Tests
// =============================================================================

/// Test: `ErrorDistribution` computes basic statistics correctly.
#[test]
fn test_error_distribution_basic_stats() {
    // Simple uniform distribution: errors from 0.0 to 1.0
    let errors: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
    let samples: Vec<HelpfulSample> = errors
        .iter()
        .map(|&e| HelpfulSample {
            activation: 0.5,
            avg_error: e,
            target_value: None,
            target_activation: None,
        })
        .collect();

    let dist = ErrorDistribution::from_samples(&samples).expect("Should compute distribution");

    // Mean should be ~0.495 (average of 0..99 / 100)
    assert!(
        (dist.mean - 0.495).abs() < 0.01,
        "Mean should be ~0.495, got {}",
        dist.mean
    );

    // Standard deviation should be positive
    assert!(dist.std_dev > 0.0, "Std dev should be positive");

    // Variance should be std_dev^2
    let expected_variance = dist.std_dev * dist.std_dev;
    assert!(
        (dist.variance - expected_variance).abs() < 0.001,
        "Variance should be std_dev^2"
    );
}

/// Test: `ErrorDistribution` computes percentiles correctly.
#[test]
fn test_error_distribution_percentiles() {
    // Linear distribution for predictable percentiles
    let errors: Vec<f32> = (0..100).map(|i| i as f32).collect();
    let samples: Vec<HelpfulSample> = errors
        .iter()
        .map(|&e| HelpfulSample {
            activation: 0.5,
            avg_error: e,
            target_value: None,
            target_activation: None,
        })
        .collect();

    let dist = ErrorDistribution::from_samples(&samples).expect("Should compute distribution");

    // Check percentiles - should be approximately:
    // p10 ≈ 10, p25 ≈ 25, p50 ≈ 50, p75 ≈ 75, p90 ≈ 90
    assert!(
        (dist.percentiles[0] - 10.0).abs() < 2.0,
        "p10 should be ~10, got {}",
        dist.percentiles[0]
    );
    assert!(
        (dist.percentiles[1] - 25.0).abs() < 2.0,
        "p25 should be ~25, got {}",
        dist.percentiles[1]
    );
    assert!(
        (dist.percentiles[2] - 50.0).abs() < 2.0,
        "p50 should be ~50, got {}",
        dist.percentiles[2]
    );
    assert!(
        (dist.percentiles[3] - 75.0).abs() < 2.0,
        "p75 should be ~75, got {}",
        dist.percentiles[3]
    );
    assert!(
        (dist.percentiles[4] - 90.0).abs() < 2.0,
        "p90 should be ~90, got {}",
        dist.percentiles[4]
    );
}

/// Test: `ErrorDistribution` computes skewness correctly.
///
/// Skewness measures the asymmetry of the distribution:
/// - Negative skewness: tail on the left (more high values)
/// - Zero skewness: symmetric (like normal distribution)
/// - Positive skewness: tail on the right (more low values with outliers)
#[test]
fn test_error_distribution_skewness_symmetric() {
    // Symmetric distribution should have ~0 skewness
    let errors: Vec<f32> = (0..100).map(|i| i as f32 - 50.0).collect();
    let samples: Vec<HelpfulSample> = errors
        .iter()
        .map(|&e| HelpfulSample {
            activation: 0.5,
            avg_error: e,
            target_value: None,
            target_activation: None,
        })
        .collect();

    let dist = ErrorDistribution::from_samples(&samples).expect("Should compute distribution");

    assert!(
        dist.skewness.abs() < 0.1,
        "Symmetric distribution should have ~0 skewness, got {}",
        dist.skewness
    );
}

/// Test: `ErrorDistribution` detects positive skewness (right-tailed).
#[test]
fn test_error_distribution_skewness_positive() {
    // Right-skewed distribution: many small values, few large outliers
    let mut errors: Vec<f32> = (0..90).map(|i| i as f32 * 0.1).collect(); // 0-9
    errors.extend(vec![
        50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0,
    ]); // outliers

    let samples: Vec<HelpfulSample> = errors
        .iter()
        .map(|&e| HelpfulSample {
            activation: 0.5,
            avg_error: e,
            target_value: None,
            target_activation: None,
        })
        .collect();

    let dist = ErrorDistribution::from_samples(&samples).expect("Should compute distribution");

    assert!(
        dist.skewness > 0.5,
        "Right-skewed distribution should have positive skewness, got {}",
        dist.skewness
    );
}

/// Test: `ErrorDistribution` computes kurtosis correctly.
///
/// Kurtosis measures the "tailedness" of the distribution:
/// - Low kurtosis (< 3): lighter tails, flatter peak (platykurtic)
/// - Kurtosis = 3: normal distribution (mesokurtic)
/// - High kurtosis (> 3): heavier tails, sharper peak (leptokurtic)
#[test]
fn test_error_distribution_kurtosis() {
    // Uniform distribution should have kurtosis < 3 (platykurtic)
    let errors: Vec<f32> = (0..1000).map(|i| i as f32 / 1000.0).collect();
    let samples: Vec<HelpfulSample> = errors
        .iter()
        .map(|&e| HelpfulSample {
            activation: 0.5,
            avg_error: e,
            target_value: None,
            target_activation: None,
        })
        .collect();

    let dist = ErrorDistribution::from_samples(&samples).expect("Should compute distribution");

    // Uniform distribution has excess kurtosis of -1.2 (or kurtosis ~1.8)
    assert!(
        dist.kurtosis < 2.5,
        "Uniform distribution should have low kurtosis, got {}",
        dist.kurtosis
    );
}

/// Test: `ErrorDistribution` handles empty samples.
#[test]
fn test_error_distribution_empty_samples() {
    let samples: Vec<HelpfulSample> = vec![];
    let dist = ErrorDistribution::from_samples(&samples);
    assert!(dist.is_none(), "Empty samples should return None");
}

/// Test: `ErrorDistribution` handles single sample.
#[test]
fn test_error_distribution_single_sample() {
    let samples = vec![HelpfulSample {
        activation: 0.5,
        avg_error: 1.0,
        target_value: None,
        target_activation: None,
    }];
    let dist = ErrorDistribution::from_samples(&samples);
    // Single sample should return something, but variance/std_dev will be 0
    if let Some(d) = dist {
        assert_eq!(d.mean, 1.0);
        assert_eq!(d.std_dev, 0.0);
    }
}

// =============================================================================
// Integration Tests: Error Distribution in Analysis Output
// =============================================================================

/// Test: Synapse analysis includes error distribution statistics in metadata.
#[test]
fn test_synapse_analysis_includes_error_distribution() {
    skip_without_gpu!();

    let creature = create_test_creature(
        vec![
            ("input-0", "input", "IDENTITY"),
            ("input-1", "input", "IDENTITY"),
            ("output-0", "output", "HARD_TANH"),
        ],
        vec![("input-0", "output-0", 1.0)],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    let mut records = Vec::new();

    // Create samples with known error distribution (outliers at the end)
    for i in 0..100 {
        let obs_idx = i as u32;
        let target_value = 0.0;
        let target_activation = 0.0;

        // Create outlier pattern: 90 samples with small error, 10 with large error
        let error = if i < 90 { 0.1 } else { 1.0 };

        records.push(DiscoverRecord::new(
            obs_idx,
            "output-0".to_string(),
            Some(target_value),
            target_activation,
            vec![error],
        ));

        records.push(DiscoverRecord::new(
            obs_idx,
            "input-1".to_string(),
            Some(0.5),
            0.5,
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

    let input = AnalyzeSynapsesInput {
        parquet_file: file_path.to_string(),
        creature,
        focus_neurons: vec!["output-0".to_string()],
        max_candidates: Some(10),
        analysis_deadline_ms: None,
        random_seed: None,
        temperature: 1.0,
        failure_cache: None,
        discovery_outcome_log: None,
    };

    let result = analyze_synapses(&input).expect("Analysis should succeed");

    // Check that metadata includes error distribution
    assert!(
        result.metadata.error_distribution.is_some(),
        "Synapse analysis should include error distribution in metadata"
    );

    let dist = result.metadata.error_distribution.as_ref().unwrap();

    // Verify basic stats are reasonable
    assert!(dist.mean > 0.0, "Mean error should be positive");
    assert!(dist.std_dev > 0.0, "Std dev should be positive");

    // Verify percentiles are present
    assert_eq!(dist.percentiles.len(), 5, "Should have 5 percentiles");

    // Verify skewness/kurtosis are computed
    // With 90% low error and 10% high error, we expect positive skewness
    eprintln!(
        "Error distribution: mean={:.4}, std_dev={:.4}, skewness={:.4}, kurtosis={:.4}",
        dist.mean, dist.std_dev, dist.skewness, dist.kurtosis
    );
    eprintln!(
        "Percentiles: p10={:.2}, p25={:.2}, p50={:.2}, p75={:.2}, p90={:.2}",
        dist.percentiles[0],
        dist.percentiles[1],
        dist.percentiles[2],
        dist.percentiles[3],
        dist.percentiles[4]
    );

    // With outliers, expect positive skewness (tail on the right)
    assert!(
        dist.skewness > 0.0,
        "Distribution with outliers should have positive skewness, got {}",
        dist.skewness
    );
}

// =============================================================================
// Synthetic Test: Known Outlier Pattern
// =============================================================================

/// Test: Discovery with bimodal error distribution.
///
/// This synthetic test creates a scenario where:
/// - 90% of samples have low error (error ≈ 0.1)
/// - 10% of samples (outliers) have high error (error ≈ 1.0)
/// - The outliers correlate with a specific input
///
/// The error distribution analysis should detect this bimodal pattern.
#[test]
fn test_bimodal_error_pattern_detection() {
    skip_without_gpu!();

    let creature = create_test_creature(
        vec![
            ("input-0", "input", "IDENTITY"),
            ("input-outlier", "input", "IDENTITY"),
            ("output-0", "output", "IDENTITY"),
        ],
        vec![("input-0", "output-0", 1.0)],
    );

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    let mut records = Vec::new();

    for i in 0..1000 {
        let obs_idx = i as u32;
        let is_outlier = i >= 900; // Last 10% are outliers

        // Target error: low for normal samples, high for outliers
        let error = if is_outlier { 0.8 } else { 0.1 };
        let target_activation = 0.5;

        records.push(DiscoverRecord::new(
            obs_idx,
            "output-0".to_string(),
            Some(target_activation),
            target_activation,
            vec![error],
        ));

        // Input-outlier: active only for outlier samples
        let outlier_activation = if is_outlier { 1.0 } else { 0.0 };
        records.push(DiscoverRecord::new(
            obs_idx,
            "input-outlier".to_string(),
            Some(outlier_activation),
            outlier_activation,
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

    let input = AnalyzeSynapsesInput {
        parquet_file: file_path.to_string(),
        creature,
        focus_neurons: vec!["output-0".to_string()],
        max_candidates: Some(10),
        analysis_deadline_ms: None,
        random_seed: None,
        temperature: 1.0,
        failure_cache: None,
        discovery_outcome_log: None,
    };

    let result = analyze_synapses(&input).expect("Analysis should succeed");

    // Verify error distribution is detected
    let dist = result
        .metadata
        .error_distribution
        .as_ref()
        .expect("Should have error distribution");

    // The distribution should show clear bimodality via skewness
    eprintln!("Error distribution stats:");
    eprintln!("  Mean: {:.4}", dist.mean);
    eprintln!("  Std dev: {:.4}", dist.std_dev);
    eprintln!("  Skewness: {:.4}", dist.skewness);
    eprintln!("  Kurtosis: {:.4}", dist.kurtosis);
    eprintln!(
        "  Percentiles: p10={:.2}, p25={:.2}, p50={:.2}, p75={:.2}, p90={:.2}",
        dist.percentiles[0],
        dist.percentiles[1],
        dist.percentiles[2],
        dist.percentiles[3],
        dist.percentiles[4]
    );

    // The skewness should be positive (right-tailed due to outliers)
    assert!(
        dist.skewness > 0.3,
        "Bimodal distribution with outliers should have positive skewness, got {}",
        dist.skewness
    );

    // The distribution should show evidence of outliers - either via skewness,
    // kurtosis, or the spread between percentiles
    let iqr = dist.percentiles[3] - dist.percentiles[1];
    let spread = dist.max - dist.min;
    eprintln!(
        "IQR: {:.4}, Spread: {:.4}, Max: {:.4}",
        iqr, spread, dist.max
    );

    // Max should be significantly higher than the mean due to outliers
    assert!(
        dist.max > dist.mean * 2.0,
        "Max should be much higher than mean due to outliers (max={}, mean={})",
        dist.max,
        dist.mean
    );
}
