//! Benchmark for Issue #529: GPU buffer transfer performance.
//!
//! Measures the critical path of moving data between CPU and GPU memory:
//!
//! 1. **CPU → GPU staging write** — Time to create and fill staging buffers
//! 2. **GPU → CPU readback** — Time to map and read results back from GPU
//! 3. **End-to-end batch cycle** — Full round-trip for a GPU evaluation batch
//! 4. **Batch size scaling** — How transfer time scales across batch sizes
//!
//! Scaling is measured across batch sizes: 64, 256, 1024, 4096.
//!
//! Skips gracefully on machines without GPU access.

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use neat_ai_discovery::analysis::gpu::{GpuAnalyzer, GpuEvaluator};
use neat_ai_discovery::analysis::samples::HelpfulSample;

/// Batch sizes to benchmark, matching the range used by the GPU batch size tuning
/// (64–4096 from NEAT_AI_DISCOVERY_GPU_BATCH_SIZE).
const BATCH_SIZES: &[usize] = &[64, 256, 1024, 4096];

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

/// Benchmark CPU → GPU staging buffer write.
///
/// Measures the time to create staging buffers, fill them with sample data via
/// `create_buffer_init`, and submit them to the GPU. Uses a single batch to
/// isolate buffer creation and upload cost at each batch size.
fn bench_cpu_to_gpu_staging_write(c: &mut Criterion) {
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping gpu_buffer_transfers benchmarks: no GPU available");
        return;
    }

    let gpu = match GpuAnalyzer::new() {
        Ok(gpu) => gpu,
        Err(e) => {
            eprintln!("Skipping gpu_buffer_transfers benchmarks: GPU initialisation failed: {e}");
            return;
        }
    };

    let mut group = c.benchmark_group("gpu_transfer_cpu_to_staging");

    for &batch_size in BATCH_SIZES {
        let samples = create_samples(batch_size);

        group.bench_with_input(
            BenchmarkId::new("samples", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    let batch: &[&[HelpfulSample]] = &[&samples];
                    let result = gpu.evaluate_helpful_batch(batch);
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark GPU → CPU readback via activation evaluation.
///
/// Measures end-to-end time including staging buffer creation, compute dispatch,
/// and readback via `map_async` + polling. Larger batch sizes transfer more data
/// back, isolating readback scaling behaviour.
fn bench_gpu_to_cpu_readback(c: &mut Criterion) {
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping gpu_buffer_transfers benchmarks: no GPU available");
        return;
    }

    let gpu = match GpuAnalyzer::new() {
        Ok(gpu) => gpu,
        Err(e) => {
            eprintln!("Skipping gpu_buffer_transfers benchmarks: GPU initialisation failed: {e}");
            return;
        }
    };

    let mut group = c.benchmark_group("gpu_transfer_readback");

    for &batch_size in BATCH_SIZES {
        let samples = create_samples(batch_size);

        group.bench_with_input(
            BenchmarkId::new("samples", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    // evaluate_activation performs: buffer create → compute → staging copy → readback
                    let result = gpu.evaluate_activation(&samples, 0, 1.0, 1.0);
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark end-to-end batch cycle.
///
/// Measures the full round-trip: CPU → GPU upload, compute dispatch, GPU → CPU
/// readback for a complete helpful synapse evaluation batch with multiple sub-batches.
/// This represents the real-world GPU buffer transfer workload during discovery.
fn bench_end_to_end_batch_cycle(c: &mut Criterion) {
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping gpu_buffer_transfers benchmarks: no GPU available");
        return;
    }

    let gpu = match GpuAnalyzer::new() {
        Ok(gpu) => gpu,
        Err(e) => {
            eprintln!("Skipping gpu_buffer_transfers benchmarks: GPU initialisation failed: {e}");
            return;
        }
    };

    let mut group = c.benchmark_group("gpu_transfer_end_to_end");

    for &batch_size in BATCH_SIZES {
        let samples_a = create_samples(batch_size);
        let samples_b = create_samples(batch_size);
        let samples_c = create_samples(batch_size);

        group.bench_with_input(
            BenchmarkId::new("samples_x3_batches", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    let batch: &[&[HelpfulSample]] = &[&samples_a, &samples_b, &samples_c];
                    let result = gpu.evaluate_helpful_batch(batch);
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark batch size scaling for buffer transfers.
///
/// Directly compares transfer overhead across all batch sizes in a single group,
/// making it easy to spot non-linear scaling (e.g., memory allocation pressure
/// at large sizes or per-call overhead dominating at small sizes).
fn bench_transfer_scaling(c: &mut Criterion) {
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping gpu_buffer_transfers benchmarks: no GPU available");
        return;
    }

    let gpu = match GpuAnalyzer::new() {
        Ok(gpu) => gpu,
        Err(e) => {
            eprintln!("Skipping gpu_buffer_transfers benchmarks: GPU initialisation failed: {e}");
            return;
        }
    };

    let mut group = c.benchmark_group("gpu_transfer_scaling");

    for &batch_size in BATCH_SIZES {
        let samples = create_samples(batch_size);

        group.bench_with_input(
            BenchmarkId::new("batch_size", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    let batch: &[&[HelpfulSample]] = &[&samples];
                    let result = gpu.evaluate_helpful_batch(batch);
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_cpu_to_gpu_staging_write,
    bench_gpu_to_cpu_readback,
    bench_end_to_end_batch_cycle,
    bench_transfer_scaling,
);
criterion_main!(benches);
