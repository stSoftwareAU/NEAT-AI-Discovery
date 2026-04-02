//! Benchmark for Issue #977: GPU queue submission copy overhead.
//!
//! Measures the cost of `to_vec()` copies when submitting work to the GPU queue
//! versus direct `GpuAnalyzer` calls that accept borrowed slices. This quantifies
//! the overhead of the mandatory cross-thread data copy in the queue path.
//!
//! Scaling is measured across sample sizes: 64, 256, 1024, 4096.
//!
//! Skips gracefully on machines without GPU access.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for benchmark computation
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use neat_ai_discovery::analysis::gpu::{GpuAnalyzer, GpuEvaluator};
use neat_ai_discovery::analysis::samples::HelpfulSample;
use std::hint::black_box;

/// Sample sizes to benchmark, matching the range used by the GPU batch size tuning
/// (64–4096 from `NEAT_AI_DISCOVERY_GPU_BATCH_SIZE`).
const SAMPLE_SIZES: &[usize] = &[64, 256, 1024, 4096];

/// Create test samples with realistic variance for benchmarking.
fn create_samples(count: usize) -> Vec<HelpfulSample> {
    (0..count)
        .map(|i| {
            let t = i as f32 / count as f32;
            HelpfulSample {
                activation: (t * 10.0 - 5.0) + (t * 7.0).sin() * 0.5,
                avg_error: (t * 3.0).cos() * 0.3,
                target_value: Some(t * 0.8),
                target_activation: Some(t.tanh()),
            }
        })
        .collect()
}

/// Benchmark direct `GpuAnalyzer` `ReLU` evaluation (no queue, no copy).
///
/// This measures the baseline GPU compute time without any cross-thread
/// copy overhead, establishing a reference for the queue-based path.
fn bench_direct_relu_eval(c: &mut Criterion) {
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping queue_submission_copies benchmarks: no GPU available");
        return;
    }

    let gpu = match GpuAnalyzer::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Skipping queue_submission_copies benchmarks: {e}");
            return;
        }
    };

    let mut group = c.benchmark_group("direct_relu_eval");
    for &size in SAMPLE_SIZES {
        let samples = create_samples(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &samples, |b, s| {
            b.iter(|| {
                let result = gpu.evaluate_relu(black_box(s), black_box(0.0));
                black_box(result).ok();
            });
        });
    }
    group.finish();
}

/// Benchmark the `to_vec()` copy overhead in isolation.
///
/// This measures just the slice-to-vec copy cost, which is the overhead
/// incurred by the queue submission path for cross-thread data transfer.
fn bench_sample_copy_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("sample_copy_overhead");
    for &size in SAMPLE_SIZES {
        let samples = create_samples(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &samples, |b, s| {
            b.iter(|| {
                let copied = black_box(s.as_slice()).to_vec();
                black_box(copied);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_direct_relu_eval, bench_sample_copy_overhead,);
criterion_main!(benches);
