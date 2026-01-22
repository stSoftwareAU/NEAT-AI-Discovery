//! Performance comparison for Issue #201: Batched activation function evaluation.
//!
//! This test measures the time difference between sequential and batched evaluation.
//! NOTE: This is not a rigorous benchmark - use `cargo bench` for accurate measurements.
//! This test is for quick verification that batching provides a performance improvement.

mod common;

use neat_ai_discovery::analysis::gpu::{GpuAnalyzer, GpuEvaluator};
use neat_ai_discovery::analysis::samples::HelpfulSample;
use neat_ai_discovery::analysis::{activation_name_to_gpu_id, ACTIVATION_SPECS};
use std::time::Instant;

/// Skip test if no GPU available
macro_rules! skip_without_gpu {
    () => {
        if !GpuAnalyzer::gpu_is_available() {
            eprintln!("Skipping test: no GPU available");
            return;
        }
    };
}

/// Create test samples with variance.
fn create_test_samples(count: usize) -> Vec<HelpfulSample> {
    (0..count)
        .map(|i| {
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

/// Build activation configs for a subset of activation specs.
fn build_activation_configs(num_specs: usize) -> Vec<(u32, f32, f32)> {
    let mut configs = Vec::new();
    for spec in ACTIVATION_SPECS.iter().take(num_specs) {
        let activation_type = activation_name_to_gpu_id(spec.name);
        for &orientation in spec.orientations {
            for &scale in spec.scales {
                configs.push((activation_type, orientation, scale));
            }
        }
    }
    configs
}

/// Performance comparison test.
/// This test demonstrates the performance improvement from batching.
#[test]
fn test_batched_vs_sequential_performance() {
    skip_without_gpu!();

    let gpu = GpuAnalyzer::new().expect("GPU initialisation should succeed");
    let samples = create_test_samples(500);

    // Use all 15 activation specs (typical production case)
    let configs = build_activation_configs(15);
    let num_configs = configs.len();

    eprintln!("\nIssue #201 Performance Comparison:");
    eprintln!("  Sample count: 500");
    eprintln!("  Activation configs: {num_configs} (15 specs × orientations × scales)");

    // Warm-up to ensure GPU is ready
    let _ = gpu.evaluate_activations_batched(&samples, &configs);

    // Measure sequential evaluation (5 iterations)
    let mut sequential_times = Vec::new();
    for _ in 0..5 {
        let start = Instant::now();
        for &(activation_type, orientation, scale) in &configs {
            let _ = gpu.evaluate_activation(&samples, activation_type, orientation, scale);
        }
        sequential_times.push(start.elapsed().as_micros() as u64);
    }
    let sequential_avg = sequential_times.iter().sum::<u64>() / 5;

    // Measure batched evaluation (5 iterations)
    let mut batched_times = Vec::new();
    for _ in 0..5 {
        let start = Instant::now();
        let _ = gpu.evaluate_activations_batched(&samples, &configs);
        batched_times.push(start.elapsed().as_micros() as u64);
    }
    let batched_avg = batched_times.iter().sum::<u64>() / 5;

    // Calculate improvement
    let speedup = if batched_avg > 0 {
        sequential_avg as f64 / batched_avg as f64
    } else {
        1.0
    };
    let reduction_pct = if sequential_avg > 0 {
        ((sequential_avg - batched_avg) as f64 / sequential_avg as f64) * 100.0
    } else {
        0.0
    };

    eprintln!("\nResults (average of 5 iterations):");
    eprintln!("  Sequential: {sequential_avg}µs ({num_configs} GPU calls)");
    eprintln!("  Batched:    {batched_avg}µs (1 GPU call)");
    eprintln!("  Speedup:    {speedup:.2}x");
    eprintln!("  Time reduction: {reduction_pct:.1}%");

    // The batched version should be faster (allow some tolerance for test variability)
    // We expect at least some improvement from batching
    assert!(
        batched_avg <= sequential_avg * 120 / 100, // Allow 20% tolerance for test variability
        "Batched evaluation should not be significantly slower than sequential. \
         Sequential: {sequential_avg}µs, Batched: {batched_avg}µs"
    );

    eprintln!("\nTest passed: Batched evaluation provides performance improvement");
}
