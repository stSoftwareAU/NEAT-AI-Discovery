//! Compiler auto-vectorisation audit benchmarks (Issue #1009).
//!
//! Compares the current `AoS` (Array-of-Structures) iteration over `&[HelpfulSample]`
//! against `SoA` (Structure-of-Arrays) field extraction for the hot numerical loops.
//!
//! ## What This Measures
//!
//! 1. **`AoS` vs `SoA` layout** — Does extracting `activation` and `avg_error` into
//!    separate `Vec<f32>` slices before hot loops improve throughput?
//! 2. **`#[inline(always)]` effect** — Do inline hints on the hot functions change
//!    codegen meaningfully?
//! 3. **Field extraction overhead** — Is the cost of `.iter().map().collect()` paid
//!    back by better cache and vectorisation behaviour in the loop?
//!
//! ## Context
//!
//! `HelpfulSample` is 24 bytes (f32 + f32 + Option<f32> + Option<f32>).
//! When the `TargetSimulationMode::None` path only needs `activation` (4 bytes)
//! and `avg_error` (4 bytes), loading 24 bytes per element wastes 67 % of cache
//! lines.  A `SoA` layout (separate `&[f32]` slices) uses every loaded byte.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)] // Intentional numeric casts for GPU/neural network computation (Issue #873)

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use neat_ai_discovery::analysis::samples::HelpfulSample;
use neat_ai_discovery::analysis::scoring::confidence::{
    compute_error_variance, compute_source_variance_confidence,
};
use neat_ai_discovery::analysis::scoring::error_distribution::ErrorDistribution;
use neat_ai_discovery::analysis::synapse::compute_synapse_improvement_and_count;
use std::hint::black_box;

// =============================================================================
// Sample Sizes
// =============================================================================

/// Production-scale sample sizes used across all benchmarks.
const SAMPLE_SIZES: &[usize] = &[100, 1_000, 10_000];

// =============================================================================
// Test Data Generators (shared with simd_hot_paths.rs)
// =============================================================================

/// Generate realistic `HelpfulSample` vectors matching production distributions.
fn generate_helpful_samples(count: usize) -> Vec<HelpfulSample> {
    let mut samples = Vec::with_capacity(count);
    for i in 0..count {
        let phase = i as f32;
        let activation = if i % 50 == 0 {
            0.0
        } else {
            (phase * 0.1).sin() * 3.0 + (phase * 0.03).cos()
        };
        let avg_error = if i % 53 == 0 {
            f32::NAN
        } else if i % 71 == 0 {
            f32::INFINITY
        } else {
            (phase * 0.07).cos() * 0.5
        };
        let (target_value, target_activation) = if i % 5 == 0 {
            (None, None)
        } else {
            let tv = (phase * 0.05).sin() * 2.0;
            let ta = tv.tanh();
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
// `SoA` (Structure-of-Arrays) Reference Implementations
//
// These compute the same results as the `AoS` functions but operate on
// pre-extracted f32 slices for better cache and vectorisation behaviour.
// =============================================================================

/// `SoA` equivalent of `compute_synapse_improvement_and_count` for the
/// `TargetSimulationMode::None` (value-domain) path.
///
/// Operates on pre-extracted activation and error slices.
#[inline(always)]
fn soa_synapse_improvement_value_domain(
    activations: &[f32],
    errors: &[f32],
    weight: f32,
    total_baseline_error_sq: f32,
) -> (f32, u32, u32, u32) {
    const EPSILON: f32 = 1e-10;

    if total_baseline_error_sq <= EPSILON || activations.is_empty() {
        return (0.0, 0, 0, activations.len() as u32);
    }

    let mut new_error_sq_sum = 0.0f32;
    let mut improved_count = 0u32;
    let mut worsened_count = 0u32;

    let len = activations.len().min(errors.len());
    for i in 0..len {
        let contribution = weight * activations[i];
        let baseline_error = errors[i];
        let new_error = baseline_error - contribution;

        if new_error.is_finite() {
            new_error_sq_sum += new_error * new_error;
        }

        if new_error.abs() + EPSILON < baseline_error.abs() {
            improved_count += 1;
        } else if new_error.abs() > baseline_error.abs() + EPSILON {
            worsened_count += 1;
        }
    }

    let improvement = if total_baseline_error_sq > EPSILON {
        (total_baseline_error_sq - new_error_sq_sum) / total_baseline_error_sq
    } else {
        0.0
    };
    let improvement = if improvement.is_finite() {
        improvement
    } else {
        0.0
    };

    (improvement, improved_count, worsened_count, len as u32)
}

/// `SoA` equivalent of `compute_source_variance_confidence`.
///
/// Operates on a pre-extracted activation slice.
#[inline(always)]
fn soa_source_variance_confidence(activations: &[f32]) -> f32 {
    const MIN_CONFIDENT_STD_DEV: f32 = 0.05;

    if activations.len() < 2 {
        return 0.0;
    }

    let mut sum = 0.0f64;
    let mut sq_sum = 0.0f64;
    let mut count = 0u32;

    for &a in activations {
        if a.is_finite() {
            let a64 = a as f64;
            sum += a64;
            sq_sum += a64 * a64;
            count += 1;
        }
    }

    if count < 2 {
        return 0.0;
    }

    let n = count as f64;
    let mean = sum / n;
    let variance = (sq_sum / n) - (mean * mean);
    let std_dev = variance.max(0.0).sqrt() as f32;

    (std_dev / MIN_CONFIDENT_STD_DEV).clamp(0.0, 1.0)
}

/// `SoA` equivalent of `compute_error_variance`.
///
/// Operates on a pre-extracted error slice.
#[inline(always)]
fn soa_error_variance(errors: &[f32]) -> f32 {
    if errors.is_empty() {
        return 0.0;
    }

    let mut sum = 0.0f64;
    let mut sq_sum = 0.0f64;
    let mut count = 0u32;

    for &e in errors {
        if e.is_finite() {
            let e64 = e as f64;
            sum += e64;
            sq_sum += e64 * e64;
            count += 1;
        }
    }

    if count == 0 {
        return 0.0;
    }

    let n = count as f64;
    let mean = sum / n;
    let variance = (sq_sum / n) - (mean * mean);

    variance.max(0.0) as f32
}

// =============================================================================
// Benchmark: `AoS` vs `SoA` for synapse improvement (value-domain path)
// =============================================================================

fn bench_aos_vs_soa_synapse(c: &mut Criterion) {
    let mut group = c.benchmark_group("aos_vs_soa_synapse_improvement");

    for &size in SAMPLE_SIZES {
        let samples = generate_helpful_samples(size);
        let baseline = compute_baseline_error_sq(&samples);
        let weight = 0.35;

        // `AoS`: current implementation (value-domain path, no target squash)
        group.bench_with_input(
            BenchmarkId::new("aos_value_domain", size),
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

        // `SoA`: pre-extract fields then compute
        let activations: Vec<f32> = samples.iter().map(|s| s.activation).collect();
        let errors: Vec<f32> = samples.iter().map(|s| s.avg_error).collect();

        group.bench_with_input(
            BenchmarkId::new("soa_value_domain", size),
            &(&activations, &errors, baseline, weight),
            |b, &(activations, errors, baseline, weight)| {
                b.iter(|| {
                    black_box(soa_synapse_improvement_value_domain(
                        black_box(activations),
                        black_box(errors),
                        black_box(weight),
                        black_box(baseline),
                    ))
                });
            },
        );

        // `SoA` with extraction cost included
        group.bench_with_input(
            BenchmarkId::new("soa_with_extraction", size),
            &(&samples, baseline, weight),
            |b, &(samples, baseline, weight)| {
                b.iter(|| {
                    let activations: Vec<f32> = samples.iter().map(|s| s.activation).collect();
                    let errors: Vec<f32> = samples.iter().map(|s| s.avg_error).collect();
                    black_box(soa_synapse_improvement_value_domain(
                        black_box(&activations),
                        black_box(&errors),
                        black_box(weight),
                        black_box(baseline),
                    ))
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// Benchmark: `AoS` vs `SoA` for source variance confidence
// =============================================================================

fn bench_aos_vs_soa_variance(c: &mut Criterion) {
    let mut group = c.benchmark_group("aos_vs_soa_source_variance");

    for &size in SAMPLE_SIZES {
        let samples = generate_helpful_samples(size);

        // `AoS`: current implementation
        group.bench_with_input(BenchmarkId::new("aos", size), &samples, |b, samples| {
            b.iter(|| black_box(compute_source_variance_confidence(black_box(samples))));
        });

        // `SoA`: pre-extracted activations
        let activations: Vec<f32> = samples.iter().map(|s| s.activation).collect();

        group.bench_with_input(
            BenchmarkId::new("soa_preextracted", size),
            &activations,
            |b, activations| {
                b.iter(|| black_box(soa_source_variance_confidence(black_box(activations))));
            },
        );

        // `SoA` with extraction cost
        group.bench_with_input(
            BenchmarkId::new("soa_with_extraction", size),
            &samples,
            |b, samples| {
                b.iter(|| {
                    let activations: Vec<f32> = samples.iter().map(|s| s.activation).collect();
                    black_box(soa_source_variance_confidence(black_box(&activations)))
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// Benchmark: `AoS` vs `SoA` for error variance
// =============================================================================

fn bench_aos_vs_soa_error_variance(c: &mut Criterion) {
    let mut group = c.benchmark_group("aos_vs_soa_error_variance");

    for &size in SAMPLE_SIZES {
        let samples = generate_helpful_samples(size);

        // `AoS`: current implementation
        group.bench_with_input(BenchmarkId::new("aos", size), &samples, |b, samples| {
            b.iter(|| black_box(compute_error_variance(black_box(samples))));
        });

        // `SoA`: pre-extracted errors
        let errors: Vec<f32> = samples.iter().map(|s| s.avg_error).collect();

        group.bench_with_input(
            BenchmarkId::new("soa_preextracted", size),
            &errors,
            |b, errors| {
                b.iter(|| black_box(soa_error_variance(black_box(errors))));
            },
        );

        // `SoA` with extraction cost
        group.bench_with_input(
            BenchmarkId::new("soa_with_extraction", size),
            &samples,
            |b, samples| {
                b.iter(|| {
                    let errors: Vec<f32> = samples.iter().map(|s| s.avg_error).collect();
                    black_box(soa_error_variance(black_box(&errors)))
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// Benchmark: Error distribution — already operates on &[f32] (SoA baseline)
//
// ErrorDistribution::from_errors already takes &[f32], so this benchmark
// measures whether from_samples (which extracts errors first) adds meaningful
// overhead vs calling from_errors directly on pre-extracted data.
// =============================================================================

fn bench_error_dist_extraction_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("error_dist_extraction_overhead");

    for &size in SAMPLE_SIZES {
        let samples = generate_helpful_samples(size);

        // from_samples: extracts errors, filters non-finite, then computes
        group.bench_with_input(
            BenchmarkId::new("from_samples", size),
            &samples,
            |b, samples| {
                b.iter(|| black_box(ErrorDistribution::from_samples(black_box(samples))));
            },
        );

        // Pre-extracted: errors already in a Vec<f32>
        let errors: Vec<f32> = samples
            .iter()
            .map(|s| s.avg_error)
            .filter(|e| e.is_finite())
            .collect();

        group.bench_with_input(
            BenchmarkId::new("from_errors_preextracted", size),
            &errors,
            |b, errors| {
                b.iter(|| black_box(ErrorDistribution::from_errors(black_box(errors))));
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
    bench_aos_vs_soa_synapse,
    bench_aos_vs_soa_variance,
    bench_aos_vs_soa_error_variance,
    bench_error_dist_extraction_overhead,
);
criterion_main!(benches);
