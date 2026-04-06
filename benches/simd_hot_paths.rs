//! SIMD baseline micro-benchmarks for hot numerical loops (Issue #1006).
//!
//! Measures the current (scalar) performance of six inner-loop functions
//! identified as SIMD candidates.  These baselines will be compared against
//! future SIMD-optimised implementations.
//!
//! ## SIMD Approach Decision
//!
//! After evaluating the available options we will rely on **compiler
//! auto-vectorisation** as the first step and only reach for explicit SIMD if
//! benchmarks show the compiler fails to vectorise a hot loop.
//!
//! Rationale:
//! - `std::simd` is nightly-only (as of 2026) and the project targets stable Rust.
//! - `std::arch` intrinsics are platform-specific and require unsafe code.
//! - The `wide` crate is stable and portable but limited to a few types.
//! - All six functions iterate over `&[HelpfulSample]` or `&[f32]` with simple
//!   arithmetic — the pattern most amenable to auto-vectorisation with
//!   `-C target-cpu=native` and appropriate data alignment.
//! - No SIMD crate is added to `Cargo.toml` at this stage; a crate will only be
//!   introduced if auto-vectorisation proves insufficient in follow-up work.
//!
//! ## Benchmarked Functions
//!
//! 1. `compute_synapse_improvement_and_count`
//! 2. `compute_relu_improvement_and_count`
//! 3. `compute_activation_improvement_and_count`
//! 4. `ErrorDistribution::from_errors`
//! 5. `compute_source_variance_confidence`
//! 6. `compute_error_variance`

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)] // Intentional numeric casts for GPU/neural network computation (Issue #873)

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use neat_ai_discovery::analysis::activation::tanh_activation;
use neat_ai_discovery::analysis::samples::HelpfulSample;
use neat_ai_discovery::analysis::scoring::confidence::{
    compute_error_variance, compute_source_variance_confidence,
};
use neat_ai_discovery::analysis::scoring::error_distribution::ErrorDistribution;
use neat_ai_discovery::analysis::synapse::{
    compute_activation_improvement_and_count, compute_relu_improvement_and_count,
    compute_synapse_improvement_and_count,
};
use std::hint::black_box;

// =============================================================================
// Sample Sizes
// =============================================================================

/// Production-scale sample sizes used across all benchmarks.
const SAMPLE_SIZES: &[usize] = &[100, 1_000, 10_000];

// =============================================================================
// Test Data Generators
// =============================================================================

/// Generate realistic `HelpfulSample` vectors matching production distributions.
///
/// The generated data includes:
/// - Varied activation ranges (negative, near-zero, positive, large)
/// - A mix of finite and non-finite error values (~2 % `NaN`/`Inf`)
/// - Optional `target_value` and `target_activation` (present on ~80 % of samples)
/// - All-zero activation runs (every 50th sample)
fn generate_helpful_samples(count: usize) -> Vec<HelpfulSample> {
    let mut samples = Vec::with_capacity(count);
    for i in 0..count {
        let phase = i as f32;

        // Activation: mix of ranges with occasional zeros
        let activation = if i % 50 == 0 {
            0.0 // all-zero activation edge case
        } else {
            (phase * 0.1).sin() * 3.0 + (phase * 0.03).cos()
        };

        // Error: mostly finite, with ~2 % non-finite edge cases
        let avg_error = if i % 53 == 0 {
            f32::NAN
        } else if i % 71 == 0 {
            f32::INFINITY
        } else {
            (phase * 0.07).cos() * 0.5
        };

        // Optional target fields present on ~80 % of samples
        let (target_value, target_activation) = if i % 5 == 0 {
            (None, None)
        } else {
            let tv = (phase * 0.05).sin() * 2.0;
            let ta = tv.tanh(); // simulate TANH squash
            (Some(tv), Some(ta))
        };

        samples.push(HelpfulSample {
            activation,
            avg_error,
            target_value,
            target_activation,
        });
    }
    samples
}

/// Generate realistic error slices for `ErrorDistribution::from_errors`.
///
/// Mix of small/large errors, ~2 % non-finite values, and occasional zeros.
fn generate_error_slice(count: usize) -> Vec<f32> {
    let mut errors = Vec::with_capacity(count);
    for i in 0..count {
        let phase = i as f32;
        let error = if i % 53 == 0 {
            f32::NAN
        } else if i % 71 == 0 {
            f32::NEG_INFINITY
        } else if i % 50 == 0 {
            0.0
        } else {
            (phase * 0.13).sin() * 2.0 + (phase * 0.07).cos() * 0.3
        };
        errors.push(error);
    }
    errors
}

/// Compute a realistic baseline error sum-of-squares for a sample set.
fn compute_baseline_error_sq(samples: &[HelpfulSample]) -> f32 {
    samples
        .iter()
        .map(|s| {
            if s.avg_error.is_finite() {
                s.avg_error * s.avg_error
            } else {
                0.0
            }
        })
        .sum()
}

// =============================================================================
// Benchmark: compute_synapse_improvement_and_count
// =============================================================================

fn bench_synapse_improvement(c: &mut Criterion) {
    let mut group = c.benchmark_group("synapse_improvement");

    for &size in SAMPLE_SIZES {
        let samples = generate_helpful_samples(size);
        let baseline = compute_baseline_error_sq(&samples);
        let weight = 0.35;

        // Without target squash (VALUE domain, fast path)
        group.bench_with_input(
            BenchmarkId::new("value_domain", size),
            &(&samples, baseline, weight),
            |b, &(samples, baseline, weight)| {
                b.iter(|| {
                    black_box(compute_synapse_improvement_and_count(
                        black_box(samples),
                        black_box(weight),
                        black_box(baseline),
                        None,
                    ))
                });
            },
        );

        // With TANH target squash (ACTIVATION domain, simulation path)
        group.bench_with_input(
            BenchmarkId::new("tanh_simulation", size),
            &(&samples, baseline, weight),
            |b, &(samples, baseline, weight)| {
                b.iter(|| {
                    black_box(compute_synapse_improvement_and_count(
                        black_box(samples),
                        black_box(weight),
                        black_box(baseline),
                        Some("TANH"),
                    ))
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// Benchmark: compute_relu_improvement_and_count
// =============================================================================

fn bench_relu_improvement(c: &mut Criterion) {
    let mut group = c.benchmark_group("relu_improvement");

    for &size in SAMPLE_SIZES {
        let samples = generate_helpful_samples(size);
        let baseline = compute_baseline_error_sq(&samples);
        let incoming_weight = 0.5;
        let outgoing_weight = 0.3;
        let bias = 0.1;

        // Without target activation function
        group.bench_with_input(
            BenchmarkId::new("no_target_fn", size),
            &(&samples, baseline),
            |b, &(samples, baseline)| {
                b.iter(|| {
                    black_box(compute_relu_improvement_and_count(
                        black_box(samples),
                        black_box(incoming_weight),
                        black_box(outgoing_weight),
                        black_box(bias),
                        black_box(baseline),
                        None,
                    ))
                });
            },
        );

        // With TANH target activation function
        group.bench_with_input(
            BenchmarkId::new("tanh_target", size),
            &(&samples, baseline),
            |b, &(samples, baseline)| {
                b.iter(|| {
                    black_box(compute_relu_improvement_and_count(
                        black_box(samples),
                        black_box(incoming_weight),
                        black_box(outgoing_weight),
                        black_box(bias),
                        black_box(baseline),
                        Some(tanh_activation as fn(f32) -> f32),
                    ))
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// Benchmark: compute_activation_improvement_and_count
// =============================================================================

fn bench_activation_improvement(c: &mut Criterion) {
    let mut group = c.benchmark_group("activation_improvement");

    for &size in SAMPLE_SIZES {
        let samples = generate_helpful_samples(size);
        let baseline = compute_baseline_error_sq(&samples);
        let incoming_weight = 0.4;
        let outgoing_weight = 0.6;
        let bias = -0.2;

        // TANH activation, no target function
        group.bench_with_input(
            BenchmarkId::new("tanh_no_target", size),
            &(&samples, baseline),
            |b, &(samples, baseline)| {
                b.iter(|| {
                    black_box(compute_activation_improvement_and_count(
                        black_box(samples),
                        black_box(incoming_weight),
                        black_box(outgoing_weight),
                        black_box(bias),
                        tanh_activation,
                        black_box(baseline),
                        None,
                    ))
                });
            },
        );

        // TANH activation with TANH target function
        group.bench_with_input(
            BenchmarkId::new("tanh_with_target", size),
            &(&samples, baseline),
            |b, &(samples, baseline)| {
                b.iter(|| {
                    black_box(compute_activation_improvement_and_count(
                        black_box(samples),
                        black_box(incoming_weight),
                        black_box(outgoing_weight),
                        black_box(bias),
                        tanh_activation,
                        black_box(baseline),
                        Some(tanh_activation as fn(f32) -> f32),
                    ))
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// Benchmark: ErrorDistribution::from_errors
// =============================================================================

fn bench_error_distribution(c: &mut Criterion) {
    let mut group = c.benchmark_group("error_distribution");

    for &size in SAMPLE_SIZES {
        let errors = generate_error_slice(size);

        group.bench_with_input(
            BenchmarkId::new("from_errors", size),
            &errors,
            |b, errors| {
                b.iter(|| black_box(ErrorDistribution::from_errors(black_box(errors))));
            },
        );
    }

    group.finish();
}

// =============================================================================
// Benchmark: compute_source_variance_confidence
// =============================================================================

fn bench_source_variance_confidence(c: &mut Criterion) {
    let mut group = c.benchmark_group("source_variance_confidence");

    for &size in SAMPLE_SIZES {
        let samples = generate_helpful_samples(size);

        group.bench_with_input(
            BenchmarkId::new("mixed_activations", size),
            &samples,
            |b, samples| {
                b.iter(|| black_box(compute_source_variance_confidence(black_box(samples))));
            },
        );
    }

    group.finish();
}

// =============================================================================
// Benchmark: compute_error_variance
// =============================================================================

fn bench_error_variance(c: &mut Criterion) {
    let mut group = c.benchmark_group("error_variance");

    for &size in SAMPLE_SIZES {
        let samples = generate_helpful_samples(size);

        group.bench_with_input(
            BenchmarkId::new("mixed_errors", size),
            &samples,
            |b, samples| {
                b.iter(|| black_box(compute_error_variance(black_box(samples))));
            },
        );
    }

    group.finish();
}

// =============================================================================
// Criterion Entry Point
// =============================================================================

criterion_group!(
    benches,
    bench_synapse_improvement,
    bench_relu_improvement,
    bench_activation_improvement,
    bench_error_distribution,
    bench_source_variance_confidence,
    bench_error_variance,
);
criterion_main!(benches);
