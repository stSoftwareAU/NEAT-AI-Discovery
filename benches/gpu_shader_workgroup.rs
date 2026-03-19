//! Benchmark for Issue #567: GPU compute shader workgroup-level optimisations.
//!
//! Measures GPU shader throughput for the core evaluation pipelines:
//!
//! 1. **Helpful evaluation** — With and without GPU reduction
//! 2. **Harmful evaluation** — With and without GPU reduction
//! 3. **`ReLU` evaluation** — Per-element contribution transfer
//! 4. **Activation evaluation** — Per-element output transfer
//! 5. **Batched activation** — Multiple configs in one command buffer
//!
//! Sample sizes span below and above the `GPU_REDUCTION_THRESHOLD` to
//! compare reduction vs direct transfer performance.
//!
//! Skips gracefully on machines without GPU access.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use neat_ai_discovery::analysis::gpu::{GpuAnalyzer, GpuEvaluator};
use neat_ai_discovery::analysis::samples::HelpfulSample;

/// Sample counts below and above `GPU_REDUCTION_THRESHOLD` (10,000).
const SAMPLE_SIZES: &[usize] = &[1_000, 5_000, 10_000, 50_000, 100_000];

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

/// Benchmark helpful synapse evaluation throughput across sample sizes.
fn bench_helpful_evaluation(c: &mut Criterion) {
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping gpu_shader_workgroup benchmarks: no GPU available");
        return;
    }

    let gpu = match GpuAnalyzer::new() {
        Ok(gpu) => gpu,
        Err(e) => {
            eprintln!("Skipping gpu_shader_workgroup benchmarks: GPU init failed: {e}");
            return;
        }
    };

    let mut group = c.benchmark_group("gpu_shader_helpful");

    for &size in SAMPLE_SIZES {
        let samples = create_samples(size);

        group.bench_with_input(BenchmarkId::new("samples", size), &size, |b, _| {
            b.iter(|| {
                let batch: &[&[HelpfulSample]] = &[&samples];
                let result = gpu.evaluate_helpful_batch(batch);
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark harmful synapse evaluation throughput across sample sizes.
fn bench_harmful_evaluation(c: &mut Criterion) {
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping gpu_shader_workgroup benchmarks: no GPU available");
        return;
    }

    let gpu = match GpuAnalyzer::new() {
        Ok(gpu) => gpu,
        Err(e) => {
            eprintln!("Skipping gpu_shader_workgroup benchmarks: GPU init failed: {e}");
            return;
        }
    };

    let mut group = c.benchmark_group("gpu_shader_harmful");

    for &size in SAMPLE_SIZES {
        let samples = create_samples(size);

        group.bench_with_input(BenchmarkId::new("samples", size), &size, |b, _| {
            b.iter(|| {
                let batch: &[(&[HelpfulSample], f32)] = &[(&samples, 0.5)];
                let result = gpu.evaluate_harmful_batch(batch);
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark `ReLU` evaluation throughput across sample sizes.
fn bench_relu_evaluation(c: &mut Criterion) {
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping gpu_shader_workgroup benchmarks: no GPU available");
        return;
    }

    let gpu = match GpuAnalyzer::new() {
        Ok(gpu) => gpu,
        Err(e) => {
            eprintln!("Skipping gpu_shader_workgroup benchmarks: GPU init failed: {e}");
            return;
        }
    };

    let mut group = c.benchmark_group("gpu_shader_relu");

    for &size in SAMPLE_SIZES {
        let samples = create_samples(size);

        group.bench_with_input(BenchmarkId::new("samples", size), &size, |b, _| {
            b.iter(|| {
                let result = gpu.evaluate_relu(&samples, 0.0);
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark activation evaluation throughput across sample sizes.
fn bench_activation_evaluation(c: &mut Criterion) {
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping gpu_shader_workgroup benchmarks: no GPU available");
        return;
    }

    let gpu = match GpuAnalyzer::new() {
        Ok(gpu) => gpu,
        Err(e) => {
            eprintln!("Skipping gpu_shader_workgroup benchmarks: GPU init failed: {e}");
            return;
        }
    };

    let mut group = c.benchmark_group("gpu_shader_activation");

    for &size in SAMPLE_SIZES {
        let samples = create_samples(size);

        group.bench_with_input(BenchmarkId::new("samples", size), &size, |b, _| {
            b.iter(|| {
                // GELU activation (type 0), positive orientation, scale 1.0
                let result = gpu.evaluate_activation(&samples, 0, 1.0, 1.0);
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark batched activation evaluation throughput.
fn bench_batched_activation(c: &mut Criterion) {
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping gpu_shader_workgroup benchmarks: no GPU available");
        return;
    }

    let gpu = match GpuAnalyzer::new() {
        Ok(gpu) => gpu,
        Err(e) => {
            eprintln!("Skipping gpu_shader_workgroup benchmarks: GPU init failed: {e}");
            return;
        }
    };

    let mut group = c.benchmark_group("gpu_shader_batched_activation");

    // Test with multiple activation configs at 10K samples
    let samples = create_samples(10_000);
    let configs: Vec<(u32, f32, f32)> = vec![
        (0, 1.0, 1.0),  // GELU positive
        (0, -1.0, 1.0), // GELU negative
        (5, 1.0, 1.0),  // TANH positive
        (5, -1.0, 1.0), // TANH negative
        (4, 1.0, 1.0),  // LOGISTIC positive
        (4, -1.0, 1.0), // LOGISTIC negative
    ];

    group.bench_function("6_configs_10k_samples", |b| {
        b.iter(|| {
            let result = gpu.evaluate_activations_batched(&samples, &configs);
            black_box(result)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_helpful_evaluation,
    bench_harmful_evaluation,
    bench_relu_evaluation,
    bench_activation_evaluation,
    bench_batched_activation,
);
criterion_main!(benches);
