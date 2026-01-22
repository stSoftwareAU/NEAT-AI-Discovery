//! Tests for Issue #201: Batch multiple activation function evaluations in single GPU call
//!
//! This module tests the batched activation evaluation feature which reduces GPU
//! round-trips by evaluating multiple activation functions in a single GPU call
//! instead of separate calls per activation type.
//!
//! ## Performance Goal
//! - 10-20% fewer GPU round-trips during neuron analysis
//! - Reduced CPU-GPU synchronisation overhead
//! - Better GPU utilisation (larger, fewer batches)
//!
//! ## Test Strategy (TDD)
//! 1. Create test comparing batched vs sequential results (must be identical)
//! 2. Verify all activation functions work correctly in batched mode
//! 3. Test edge cases (empty samples, single activation, etc.)

mod common;

use neat_ai_discovery::analysis::gpu::{GpuAnalyzer, GpuEvaluator};
use neat_ai_discovery::analysis::samples::HelpfulSample;
use neat_ai_discovery::analysis::{activation_name_to_gpu_id, ACTIVATION_SPECS};

/// Skip test if no GPU available
macro_rules! skip_without_gpu {
    () => {
        if !GpuAnalyzer::gpu_is_available() {
            eprintln!("Skipping test: no GPU available");
            return;
        }
    };
}

/// Helper to create test samples with variance
fn create_test_samples(count: usize) -> Vec<HelpfulSample> {
    (0..count)
        .map(|i| {
            // Create samples with variance to avoid source variance discounting
            let t = i as f32 / count as f32;
            let activation = (t * 10.0 - 5.0) + (t * 7.0).sin() * 0.5;
            let error = (t * 3.0).cos() * 0.3;

            HelpfulSample {
                activation,
                avg_error: error,
                target_value: Some(activation * 0.8),
                target_activation: Some(activation.tanh()),
            }
        })
        .collect()
}

/// Test that batched activation evaluation produces identical results to sequential evaluation.
///
/// This is the core correctness test - batched evaluation must produce exactly the
/// same results as calling evaluate_activation for each activation type separately.
#[test]
fn test_batched_vs_sequential_results_identical() {
    skip_without_gpu!();

    let gpu = GpuAnalyzer::new().expect("GPU initialisation should succeed");
    let samples = create_test_samples(500);

    // Test a subset of activation specs to keep the test fast
    let test_specs = &ACTIVATION_SPECS[..5]; // GELU, ELU, Softplus, LOGISTIC, TANH

    // Collect all activation configs we want to test
    let mut activation_configs: Vec<(u32, f32, f32)> = Vec::new();
    for spec in test_specs {
        let activation_type = activation_name_to_gpu_id(spec.name);
        for &orientation in spec.orientations {
            for &scale in spec.scales.iter().take(3) {
                // Test first 3 scales
                activation_configs.push((activation_type, orientation, scale));
            }
        }
    }

    // Get sequential results (current implementation)
    let mut sequential_results: Vec<(f32, f32, f32, u32)> = Vec::new();
    for &(activation_type, orientation, scale) in &activation_configs {
        let result = gpu
            .evaluate_activation(&samples, activation_type, orientation, scale)
            .expect("Sequential evaluation should succeed");
        sequential_results.push(result);
    }

    // Get batched results (new implementation)
    let batched_results = gpu
        .evaluate_activations_batched(&samples, &activation_configs)
        .expect("Batched evaluation should succeed");

    // Verify results are identical
    assert_eq!(
        sequential_results.len(),
        batched_results.len(),
        "Result counts should match"
    );

    for (i, ((seq, bat), config)) in sequential_results
        .iter()
        .zip(batched_results.iter())
        .zip(activation_configs.iter())
        .enumerate()
    {
        let tolerance = 1e-5;
        assert!(
            (seq.0 - bat.0).abs() < tolerance,
            "sum_activation_sq mismatch at index {i} (config {:?}): sequential={}, batched={}",
            config,
            seq.0,
            bat.0
        );
        assert!(
            (seq.1 - bat.1).abs() < tolerance,
            "sum_error_activation mismatch at index {i} (config {:?}): sequential={}, batched={}",
            config,
            seq.1,
            bat.1
        );
        assert!(
            (seq.2 - bat.2).abs() < tolerance,
            "total_baseline_error_sq mismatch at index {i} (config {:?}): sequential={}, batched={}",
            config,
            seq.2,
            bat.2
        );
        assert_eq!(
            seq.3, bat.3,
            "improved_count mismatch at index {i} (config {config:?})"
        );
    }

    eprintln!(
        "Test passed: {} activation configs produced identical results",
        activation_configs.len()
    );
}

/// Test that all 15 activation specs work correctly in batched mode.
///
/// This verifies that every activation function in ACTIVATION_SPECS is correctly
/// evaluated when using the batched GPU call.
#[test]
fn test_all_activation_specs_in_batched_mode() {
    skip_without_gpu!();

    let gpu = GpuAnalyzer::new().expect("GPU initialisation should succeed");
    let samples = create_test_samples(300);

    // Collect configs for ALL activation specs with a single scale/orientation
    let activation_configs: Vec<(u32, f32, f32)> = ACTIVATION_SPECS
        .iter()
        .map(|spec| {
            let activation_type = activation_name_to_gpu_id(spec.name);
            // Use first orientation and scale
            let orientation = spec.orientations[0];
            let scale = spec.scales[0];
            (activation_type, orientation, scale)
        })
        .collect();

    // Evaluate in batch
    let batched_results = gpu
        .evaluate_activations_batched(&samples, &activation_configs)
        .expect("Batched evaluation should succeed for all activation specs");

    assert_eq!(
        batched_results.len(),
        ACTIVATION_SPECS.len(),
        "Should have result for each activation spec"
    );

    // Verify each result is valid (finite values)
    for (i, (result, spec)) in batched_results
        .iter()
        .zip(ACTIVATION_SPECS.iter())
        .enumerate()
    {
        let (sum_activation_sq, sum_error_activation, total_baseline_error_sq, _improved_count) =
            result;

        assert!(
            sum_activation_sq.is_finite(),
            "Activation {} (index {}) produced non-finite sum_activation_sq",
            spec.name,
            i
        );
        assert!(
            sum_error_activation.is_finite(),
            "Activation {} (index {}) produced non-finite sum_error_activation",
            spec.name,
            i
        );
        assert!(
            total_baseline_error_sq.is_finite(),
            "Activation {} (index {}) produced non-finite total_baseline_error_sq",
            spec.name,
            i
        );
    }

    eprintln!(
        "Test passed: All {} activation specs work correctly in batched mode",
        ACTIVATION_SPECS.len()
    );
}

/// Test batched evaluation with empty samples.
#[test]
fn test_batched_evaluation_empty_samples() {
    skip_without_gpu!();

    let gpu = GpuAnalyzer::new().expect("GPU initialisation should succeed");
    let samples: Vec<HelpfulSample> = Vec::new();

    let activation_configs: Vec<(u32, f32, f32)> = vec![(0, 1.0, 1.0), (1, 1.0, 1.0)];

    let batched_results = gpu
        .evaluate_activations_batched(&samples, &activation_configs)
        .expect("Batched evaluation should handle empty samples");

    // Empty samples should return zero results for each config
    assert_eq!(batched_results.len(), activation_configs.len());
    for (i, result) in batched_results.iter().enumerate() {
        assert_eq!(
            *result,
            (0.0, 0.0, 0.0, 0),
            "Empty samples should produce zero results at index {i}"
        );
    }

    eprintln!("Test passed: Batched evaluation handles empty samples correctly");
}

/// Test batched evaluation with a single activation config.
#[test]
fn test_batched_evaluation_single_config() {
    skip_without_gpu!();

    let gpu = GpuAnalyzer::new().expect("GPU initialisation should succeed");
    let samples = create_test_samples(100);

    let activation_configs: Vec<(u32, f32, f32)> = vec![(5, 1.0, 1.0)]; // TANH

    let batched_results = gpu
        .evaluate_activations_batched(&samples, &activation_configs)
        .expect("Batched evaluation should work with single config");

    let sequential_result = gpu
        .evaluate_activation(&samples, 5, 1.0, 1.0)
        .expect("Sequential evaluation should succeed");

    assert_eq!(batched_results.len(), 1);

    let tolerance = 1e-5;
    assert!(
        (batched_results[0].0 - sequential_result.0).abs() < tolerance,
        "Single config batched should match sequential"
    );
    assert!(
        (batched_results[0].1 - sequential_result.1).abs() < tolerance,
        "Single config batched should match sequential"
    );
    assert!(
        (batched_results[0].2 - sequential_result.2).abs() < tolerance,
        "Single config batched should match sequential"
    );

    eprintln!("Test passed: Batched evaluation works correctly with single config");
}

/// Test batched evaluation with empty activation configs.
#[test]
fn test_batched_evaluation_empty_configs() {
    skip_without_gpu!();

    let gpu = GpuAnalyzer::new().expect("GPU initialisation should succeed");
    let samples = create_test_samples(100);

    let activation_configs: Vec<(u32, f32, f32)> = Vec::new();

    let batched_results = gpu
        .evaluate_activations_batched(&samples, &activation_configs)
        .expect("Batched evaluation should handle empty configs");

    assert!(
        batched_results.is_empty(),
        "Empty configs should produce empty results"
    );

    eprintln!("Test passed: Batched evaluation handles empty configs correctly");
}

/// Test batched evaluation with edge case samples (NaN, infinity).
#[test]
fn test_batched_evaluation_edge_case_samples() {
    skip_without_gpu!();

    let gpu = GpuAnalyzer::new().expect("GPU initialisation should succeed");

    // Mix of valid and invalid samples
    let samples = vec![
        HelpfulSample {
            activation: 1.0,
            avg_error: 0.1,
            target_value: Some(0.8),
            target_activation: Some(0.76),
        },
        HelpfulSample {
            activation: f32::NAN,
            avg_error: 0.1,
            target_value: Some(0.8),
            target_activation: Some(0.76),
        },
        HelpfulSample {
            activation: 2.0,
            avg_error: f32::INFINITY,
            target_value: Some(1.6),
            target_activation: Some(0.96),
        },
        HelpfulSample {
            activation: -1.0,
            avg_error: -0.2,
            target_value: Some(-0.8),
            target_activation: Some(-0.66),
        },
    ];

    let activation_configs: Vec<(u32, f32, f32)> = vec![
        (0, 1.0, 1.0),  // GELU
        (5, -1.0, 2.0), // TANH negative orientation
    ];

    // Should handle edge cases without crashing
    let batched_results = gpu
        .evaluate_activations_batched(&samples, &activation_configs)
        .expect("Batched evaluation should handle edge case samples");

    assert_eq!(batched_results.len(), activation_configs.len());

    // Results should be finite (invalid samples should be skipped by shader)
    for result in &batched_results {
        assert!(
            result.0.is_finite(),
            "sum_activation_sq should be finite even with edge case inputs"
        );
        assert!(
            result.1.is_finite(),
            "sum_error_activation should be finite even with edge case inputs"
        );
        // Note: total_baseline_error_sq may include the valid samples only
        assert!(
            result.2.is_finite(),
            "total_baseline_error_sq should be finite even with edge case inputs"
        );
    }

    eprintln!("Test passed: Batched evaluation handles edge case samples correctly");
}
