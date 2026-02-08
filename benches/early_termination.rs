//! Benchmark for Issue #429: Early termination improvements for low-value candidates.
//!
//! Measures the performance of:
//! - Candidate pre-filtering (hierarchical quick filter)
//! - Budget-aware prioritisation (stop when budget exhausted)
//! - Incremental confidence early exit
//! - Cross-module deduplication
//!
//! Run with: cargo bench --bench early_termination

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use neat_ai_discovery::analysis::early_termination::{
    check_batch_early_termination, CandidatePreFilter, EarlyTerminationConfig,
};
use neat_ai_discovery::analysis::samples::HelpfulStats;

/// Create a batch of HelpfulStats with a given positive rate.
fn create_stats_batch(count: usize, positive_rate: f32) -> Vec<HelpfulStats> {
    let total_samples = 500u32;
    let positive = (total_samples as f32 * positive_rate) as u32;
    let negative = total_samples - positive;

    (0..count)
        .map(|_| HelpfulStats {
            positive_count: positive,
            negative_count: negative,
            positive_improvement_sum: positive as f32 * 0.01,
            negative_improvement_sum: negative as f32 * 0.005,
            positive_activation_sum: positive as f32 * 0.5,
            negative_activation_sum: negative as f32 * 0.5,
            error_sq_sum: total_samples as f32 * 0.01,
            activation_sq_sum: total_samples as f32 * 0.25,
            error_activation_sum: total_samples as f32 * 0.05,
            samples_evaluated: total_samples,
            early_terminated: false,
        })
        .collect()
}

/// Create a mixed batch with varying quality candidates.
fn create_mixed_stats_batch(count: usize) -> Vec<HelpfulStats> {
    (0..count)
        .map(|i| {
            let total_samples = 500u32;
            // Mix of strong (90%), marginal (55%), and poor (20%) candidates
            let positive_rate = match i % 3 {
                0 => 0.90,
                1 => 0.55,
                _ => 0.20,
            };
            let positive = (total_samples as f32 * positive_rate) as u32;
            let negative = total_samples - positive;

            HelpfulStats {
                positive_count: positive,
                negative_count: negative,
                positive_improvement_sum: positive as f32 * 0.01,
                negative_improvement_sum: negative as f32 * 0.005,
                positive_activation_sum: positive as f32 * 0.5,
                negative_activation_sum: negative as f32 * 0.5,
                error_sq_sum: total_samples as f32 * 0.01,
                activation_sq_sum: total_samples as f32 * 0.25,
                error_activation_sum: total_samples as f32 * 0.05,
                samples_evaluated: total_samples,
                early_terminated: false,
            }
        })
        .collect()
}

fn bench_batch_early_termination(c: &mut Criterion) {
    let mut group = c.benchmark_group("early_termination");

    let config = EarlyTerminationConfig::default();

    // Benchmark different batch sizes
    for batch_size in [100, 500, 1000, 5000] {
        let stats = create_mixed_stats_batch(batch_size);

        group.bench_with_input(
            BenchmarkId::new("check_batch", batch_size),
            &stats,
            |b, s| b.iter(|| check_batch_early_termination(black_box(s), black_box(&config))),
        );
    }

    group.finish();
}

fn bench_candidate_pre_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("candidate_pre_filter");

    // Benchmark pre-filtering with different batch sizes and quality mixes
    for batch_size in [100, 500, 1000, 5000] {
        let stats = create_mixed_stats_batch(batch_size);

        group.bench_with_input(
            BenchmarkId::new("pre_filter_mixed", batch_size),
            &stats,
            |b, s| {
                b.iter(|| {
                    let filter = CandidatePreFilter::default();
                    filter.filter_batch(black_box(s))
                })
            },
        );

        // Benchmark with mostly poor candidates (common real-world scenario)
        let poor_stats = create_stats_batch(batch_size, 0.20);
        group.bench_with_input(
            BenchmarkId::new("pre_filter_mostly_poor", batch_size),
            &poor_stats,
            |b, s| {
                b.iter(|| {
                    let filter = CandidatePreFilter::default();
                    filter.filter_batch(black_box(s))
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_batch_early_termination,
    bench_candidate_pre_filter
);
criterion_main!(benches);
