//! Benchmark for Issue #1545: squash-aware hidden-target `ACTIVATION_SPECS`
//! pruning.
//!
//! Neuron / add-neuron GPU evaluation previously built the full cross-product
//! of every `ACTIVATION_SPECS` family × orientations × scales for every
//! (source, target) pair. This benchmark compares the batched GPU activation
//! evaluation cost of:
//!
//! - `full`: the full cross-product (pre-#1545 behaviour), and
//! - `cold_start_core`: the cold-start pruned scan set a fresh creature sees.
//!
//! The primary win the issue asks for — "measurable drop in activation GPU
//! configs" — is deterministic and is reported to stderr as the config counts.
//! The batched GPU timings quantify the wall-clock effect on real hardware.

#![allow(clippy::cast_precision_loss)]
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use neat_ai_discovery::analysis::activation::{HiddenScanContext, SquashScanPlan};
use neat_ai_discovery::analysis::gpu::{GpuAnalyzer, GpuEvaluator};
use neat_ai_discovery::analysis::samples::HelpfulSample;
use std::collections::HashSet;
use std::hint::black_box;

/// Create test samples with variance (mirrors `batched_activation.rs`).
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

/// GPU config tuples (`activation_type`, orientation, scale) for a plan.
fn gpu_configs(plan: &SquashScanPlan) -> Vec<(u32, f32, f32)> {
    plan.configs()
        .iter()
        .map(|c| (c.activation_type, c.orientation, c.scale))
        .collect()
}

fn benchmark_pruning(c: &mut Criterion) {
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

    let full = SquashScanPlan::full();
    let history = HashSet::new();
    let cold = SquashScanPlan::for_hidden(
        &HiddenScanContext {
            successful_squashes: &history,
            escalated: false,
        },
        0,
    );

    let full_configs = gpu_configs(&full);
    let cold_configs = gpu_configs(&cold);

    eprintln!(
        "Issue #1545 config counts — full: {} configs, cold-start core: {} configs ({:.1}% reduction)",
        full_configs.len(),
        cold_configs.len(),
        100.0 * (1.0 - cold_configs.len() as f32 / full_configs.len() as f32),
    );

    let mut group = c.benchmark_group("neuron_squash_pruning");

    for sample_count in [500, 1000] {
        let samples = create_test_samples(sample_count);

        group.bench_with_input(
            BenchmarkId::new("full", sample_count),
            &sample_count,
            |b, _| {
                b.iter(|| {
                    let r = gpu
                        .evaluate_activations_batched(&samples, &full_configs)
                        .unwrap_or_default();
                    black_box(r)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("cold_start_core", sample_count),
            &sample_count,
            |b, _| {
                b.iter(|| {
                    let r = gpu
                        .evaluate_activations_batched(&samples, &cold_configs)
                        .unwrap_or_default();
                    black_box(r)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, benchmark_pruning);
criterion_main!(benches);
