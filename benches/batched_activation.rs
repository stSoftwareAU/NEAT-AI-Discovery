//! Benchmark for Issue #201: Batched activation function evaluation.
//!
//! This benchmark compares the performance of:
//! - Sequential: Calling evaluate_activation for each (activation_type, orientation, scale) config
//! - Batched: Calling evaluate_activations_batched once with all configs
//!
//! Expected improvements:
//! - 10-20% fewer GPU round-trips during neuron analysis
//! - Reduced CPU-GPU synchronisation overhead
//! - Better GPU utilisation (larger, fewer batches)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use neat_ai_discovery::analysis::gpu::{GpuAnalyzer, GpuEvaluator};
use neat_ai_discovery::analysis::samples::HelpfulSample;
use neat_ai_discovery::analysis::{activation_name_to_gpu_id, ACTIVATION_SPECS};

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

/// Sequential evaluation - calls evaluate_activation for each config.
fn evaluate_sequential(
    gpu: &GpuAnalyzer,
    samples: &[HelpfulSample],
    configs: &[(u32, f32, f32)],
) -> Vec<(f32, f32, f32, u32)> {
    configs
        .iter()
        .map(|&(activation_type, orientation, scale)| {
            gpu.evaluate_activation(samples, activation_type, orientation, scale)
                .unwrap_or((0.0, 0.0, 0.0, 0))
        })
        .collect()
}

/// Batched evaluation - calls evaluate_activations_batched once.
fn evaluate_batched(
    gpu: &GpuAnalyzer,
    samples: &[HelpfulSample],
    configs: &[(u32, f32, f32)],
) -> Vec<(f32, f32, f32, u32)> {
    gpu.evaluate_activations_batched(samples, configs)
        .unwrap_or_else(|_| vec![(0.0, 0.0, 0.0, 0); configs.len()])
}

fn benchmark_activation_evaluation(c: &mut Criterion) {
    // Skip benchmark if no GPU available
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping benchmark: no GPU available");
        return;
    }

    let gpu = match GpuAnalyzer::new() {
        Ok(gpu) => gpu,
        Err(_) => {
            eprintln!("Skipping benchmark: GPU initialisation failed");
            return;
        }
    };

    let mut group = c.benchmark_group("activation_evaluation");

    // Test with different sample sizes
    for sample_count in [100, 500, 1000] {
        let samples = create_test_samples(sample_count);

        // Test with all 15 activation specs (typical production case)
        let configs = build_activation_configs(15);

        group.bench_with_input(
            BenchmarkId::new("sequential", sample_count),
            &sample_count,
            |b, _| {
                b.iter(|| {
                    let results = evaluate_sequential(&gpu, &samples, &configs);
                    black_box(results)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("batched", sample_count),
            &sample_count,
            |b, _| {
                b.iter(|| {
                    let results = evaluate_batched(&gpu, &samples, &configs);
                    black_box(results)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, benchmark_activation_evaluation);
criterion_main!(benches);
