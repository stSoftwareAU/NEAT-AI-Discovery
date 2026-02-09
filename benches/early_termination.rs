//! Benchmark for Issue #429: Early termination improvements for low-value candidates.
//!
//! Measures the performance of the new candidate filtering pipeline:
//! - Prefilter: quick ratio-based filtering
//! - Budget-aware evaluation: SPRT with candidate budget
//! - Incremental confidence: SPRT with confidence scoring
//! - Cross-module deduplication: signature-based dedup
//!
//! ## Benchmark Design
//!
//! Each benchmark generates a realistic mix of candidate statistics:
//! - ~20% clearly poor (improvement ratio < 0.25)
//! - ~20% clearly good (improvement ratio > 0.75)
//! - ~60% marginal (ratio between 0.25 and 0.75)
//!
//! This mirrors production workloads where most candidates are marginal.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use neat_ai_discovery::analysis::early_termination::{
    budget_aware_evaluate, check_batch_early_termination, evaluate_with_confidence,
    prefilter_candidates, CandidateSignature, CrossModuleDeduplicator, EarlyTerminationConfig,
};
use neat_ai_discovery::analysis::samples::HelpfulStats;

/// Create a realistic mix of candidate statistics for benchmarking.
fn create_candidate_stats(count: usize) -> Vec<HelpfulStats> {
    let mut stats = Vec::with_capacity(count);
    for i in 0..count {
        let (pos, neg) = match i % 5 {
            0 => (10u32, 90u32), // clearly poor (10% positive)
            4 => (85u32, 15u32), // clearly good (85% positive)
            _ => {
                // marginal: 35–65% positive
                let pos = 35 + ((i * 7) % 31) as u32;
                (pos, 100 - pos)
            }
        };
        stats.push(HelpfulStats {
            positive_count: pos,
            negative_count: neg,
            positive_improvement_sum: pos as f32 * 0.01,
            negative_improvement_sum: -(neg as f32) * 0.01,
            positive_activation_sum: 0.0,
            negative_activation_sum: 0.0,
            error_sq_sum: 0.0,
            activation_sq_sum: 0.0,
            error_activation_sum: 0.0,
            samples_evaluated: 100,
            early_terminated: false,
        });
    }
    stats
}

/// Benchmark: prefilter_candidates at various candidate counts.
fn bench_prefilter(c: &mut Criterion) {
    let candidate_counts = [100, 500, 1000, 5000];

    let mut group = c.benchmark_group("prefilter_candidates");
    for &count in &candidate_counts {
        let stats = create_candidate_stats(count);
        group.bench_with_input(BenchmarkId::from_parameter(count), &stats, |b, stats| {
            b.iter(|| {
                let result = prefilter_candidates(black_box(stats));
                black_box(result);
            });
        });
    }
    group.finish();
}

/// Benchmark: budget_aware_evaluate at various candidate counts.
fn bench_budget_aware(c: &mut Criterion) {
    let candidate_counts = [100, 500, 1000, 5000];
    let config = EarlyTerminationConfig::default();

    let mut group = c.benchmark_group("budget_aware_evaluate");
    for &count in &candidate_counts {
        let stats = create_candidate_stats(count);
        let budget = count / 2; // Budget for half the candidates
        group.bench_with_input(BenchmarkId::from_parameter(count), &stats, |b, stats| {
            b.iter(|| {
                let result = budget_aware_evaluate(black_box(stats), &config, budget);
                black_box(result);
            });
        });
    }
    group.finish();
}

/// Benchmark: evaluate_with_confidence for individual candidates.
fn bench_confidence(c: &mut Criterion) {
    let candidate_counts = [100, 500, 1000, 5000];
    let config = EarlyTerminationConfig::default();

    let mut group = c.benchmark_group("evaluate_with_confidence");
    for &count in &candidate_counts {
        let stats = create_candidate_stats(count);
        group.bench_with_input(BenchmarkId::from_parameter(count), &stats, |b, stats| {
            b.iter(|| {
                for stat in stats {
                    let result = evaluate_with_confidence(black_box(stat), &config);
                    black_box(result);
                }
            });
        });
    }
    group.finish();
}

/// Benchmark: cross-module deduplication at various candidate counts.
fn bench_deduplication(c: &mut Criterion) {
    let candidate_counts = [100, 500, 1000, 5000];

    let mut group = c.benchmark_group("cross_module_dedup");
    for &count in &candidate_counts {
        // Create signatures with ~30% duplicates
        let signatures: Vec<CandidateSignature> = (0..count)
            .map(|i| CandidateSignature {
                source_uuid: format!("input-{}", i % (count * 7 / 10)),
                target_uuid: format!("output-{}", i % 3),
                candidate_type: "addSynapse".to_string(),
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(count),
            &signatures,
            |b, sigs| {
                b.iter(|| {
                    let mut dedup = CrossModuleDeduplicator::new();
                    let mut novel_count = 0usize;
                    for sig in sigs {
                        if dedup.is_novel(black_box(sig)) {
                            dedup.register(sig);
                            novel_count += 1;
                        }
                    }
                    black_box(novel_count);
                });
            },
        );
    }
    group.finish();
}

/// Benchmark: full SPRT (check_batch_early_termination) for comparison baseline.
fn bench_full_sprt_baseline(c: &mut Criterion) {
    let candidate_counts = [100, 500, 1000, 5000];
    let config = EarlyTerminationConfig::default();

    let mut group = c.benchmark_group("full_sprt_baseline");
    for &count in &candidate_counts {
        let stats = create_candidate_stats(count);
        group.bench_with_input(BenchmarkId::from_parameter(count), &stats, |b, stats| {
            b.iter(|| {
                let result = check_batch_early_termination(black_box(stats), &config);
                black_box(result);
            });
        });
    }
    group.finish();
}

/// Benchmark: combined pipeline (prefilter + budget-aware) vs full SPRT.
fn bench_combined_pipeline(c: &mut Criterion) {
    let candidate_counts = [100, 500, 1000, 5000];
    let config = EarlyTerminationConfig::default();

    let mut group = c.benchmark_group("combined_pipeline");
    for &count in &candidate_counts {
        let stats = create_candidate_stats(count);
        let budget = count / 2;

        group.bench_with_input(BenchmarkId::from_parameter(count), &stats, |b, stats| {
            b.iter(|| {
                // Step 1: Quick prefilter
                let prefilter_result = prefilter_candidates(black_box(stats));

                // Step 2: Budget-aware SPRT on remaining candidates
                let remaining: Vec<HelpfulStats> = prefilter_result
                    .continue_indices
                    .iter()
                    .map(|&i| stats[i].clone())
                    .collect();
                let budget_result = budget_aware_evaluate(black_box(&remaining), &config, budget);

                black_box((
                    prefilter_result.accept_indices.len(),
                    prefilter_result.reject_indices.len(),
                    budget_result.accept_indices.len(),
                    budget_result.reject_indices.len(),
                ));
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_prefilter,
    bench_budget_aware,
    bench_confidence,
    bench_deduplication,
    bench_full_sprt_baseline,
    bench_combined_pipeline
);
criterion_main!(benches);
