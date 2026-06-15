//! Benchmark for Issue #1369: reusing GPU buffers across batch chunks.
//!
//! `evaluate_helpful_batch` splits its input sample sets into chunks of
//! `batch_size` and processes one chunk per command submission. The original
//! code allocated fresh GPU buffers (`create_buffer_init`) and rebuilt bind
//! groups for every sample set in every chunk. This benchmark drives a workload
//! with **many chunks** by using a small batch size, so the per-chunk buffer
//! marshalling cost dominates and any reduction from buffer reuse is visible.
//!
//! Skips gracefully on machines without GPU access.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for sample generation.
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use neat_ai_discovery::analysis::GpuAnalyzer;
use neat_ai_discovery::analysis::samples::HelpfulSample;
use std::hint::black_box;

/// Number of sample sets fed to a single `evaluate_helpful_batch` call.
const NUM_SETS: usize = 96;
/// Samples per set (below the reduction threshold, the common discovery case).
const SET_LEN: usize = 256;
/// Small batch sizes force many chunks, exposing per-chunk allocation overhead.
const BATCH_SIZES: &[usize] = &[2, 4, 8];

fn make_set(len: usize, seed: usize) -> Vec<HelpfulSample> {
    (0..len)
        .map(|i| {
            let t = (i as f32 + seed as f32 * 1.3) / (len as f32 + 1.0);
            HelpfulSample {
                activation: (t * 9.0 - 4.5) + (t * 5.0 + seed as f32).sin() * 0.7,
                avg_error: (t * 3.0).cos() * 0.4 - 0.05,
                target_value: Some(t * 0.6),
                target_activation: Some(t.tanh()),
            }
        })
        .collect()
}

fn bench_chunk_reuse(c: &mut Criterion) {
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping gpu_helpful_chunk_reuse benchmarks: no GPU available");
        return;
    }

    let sets: Vec<Vec<HelpfulSample>> = (0..NUM_SETS).map(|s| make_set(SET_LEN, s)).collect();
    let batch: Vec<&[HelpfulSample]> = sets.iter().map(Vec::as_slice).collect();

    let mut group = c.benchmark_group("gpu_helpful_chunk_reuse");
    for &batch_size in BATCH_SIZES {
        let gpu = match GpuAnalyzer::new_with_batch_size(batch_size) {
            Ok(gpu) => gpu,
            Err(e) => {
                eprintln!("Skipping gpu_helpful_chunk_reuse: GPU init failed: {e}");
                return;
            }
        };
        group.bench_with_input(
            BenchmarkId::new("batch_size", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    let result = gpu.evaluate_helpful_batch(black_box(&batch));
                    black_box(result)
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_chunk_reuse);
criterion_main!(benches);
