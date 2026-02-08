//! Benchmark for Issue #429: Early termination improvements for low-value candidates.
//!
//! Measures the overhead and effectiveness of the four new early termination features:
//! 1. Hierarchical candidate pre-filtering
//! 2. Budget-aware prioritisation
//! 3. Incremental confidence checking
//! 4. Cross-module deduplication

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use neat_ai_discovery::analysis::early_termination::{
    BudgetTracker, CandidatePreFilter, CandidatePreFilterConfig, CrossModuleDeduplicator,
    IncrementalConfidenceChecker, SequentialEvaluator,
};

/// Generate test activations with controlled variance.
fn generate_activations(count: usize, variance: f32) -> Vec<f32> {
    (0..count)
        .map(|i| {
            let t = i as f32 / count as f32;
            (t * 2.0 - 1.0) * variance
        })
        .collect()
}

/// Generate errors correlated with activations.
fn generate_correlated_errors(activations: &[f32], correlation: f32) -> Vec<f32> {
    activations
        .iter()
        .enumerate()
        .map(|(i, &a)| a * correlation + (i as f32 % 7.0) * 0.01 * (1.0 - correlation))
        .collect()
}

fn bench_pre_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("pre_filter");

    for &count in &[100, 1_000, 10_000] {
        let activations = generate_activations(count, 1.0);
        let errors = generate_correlated_errors(&activations, 0.5);

        group.bench_with_input(BenchmarkId::new("should_analyse", count), &count, |b, _| {
            let config = CandidatePreFilterConfig::default();
            let pre_filter = CandidatePreFilter::new(config);
            b.iter(|| black_box(pre_filter.should_analyse(&activations, &errors)));
        });

        group.bench_with_input(BenchmarkId::new("variance_check", count), &count, |b, _| {
            let config = CandidatePreFilterConfig::default();
            let pre_filter = CandidatePreFilter::new(config);
            b.iter(|| black_box(pre_filter.passes_variance_check(&activations)));
        });
    }

    group.finish();
}

fn bench_budget_tracker(c: &mut Criterion) {
    c.bench_function("budget_tracker/consume_and_check_1000", |b| {
        b.iter(|| {
            let mut tracker = BudgetTracker::new(1000);
            for i in 0..1000 {
                tracker.consume(1);
                black_box(tracker.should_skip_low_priority(0.01 * i as f32));
            }
        });
    });
}

fn bench_incremental_confidence(c: &mut Criterion) {
    c.bench_function("incremental_confidence/100_candidates", |b| {
        b.iter(|| {
            let mut checker = IncrementalConfidenceChecker::new(0.8);
            for i in 0..100 {
                checker.add_candidate(i as f32 / 100.0);
                black_box(checker.should_stop_generating());
            }
        });
    });
}

fn bench_cross_module_dedup(c: &mut Criterion) {
    let mut group = c.benchmark_group("cross_module_dedup");

    for &count in &[100, 1_000, 10_000] {
        group.bench_with_input(
            BenchmarkId::new("register_and_check", count),
            &count,
            |b, &count| {
                b.iter(|| {
                    let mut dedup = CrossModuleDeduplicator::new();
                    for i in 0..count {
                        let source = format!("source-{}", i % 50);
                        let target = format!("target-{}", i % 20);
                        let gain = (i as f32) * 0.001;
                        black_box(dedup.register_if_unique(&source, &target, gain));
                    }
                });
            },
        );
    }

    group.finish();
}

fn bench_sprt_evaluator(c: &mut Criterion) {
    let mut group = c.benchmark_group("sprt_evaluator");

    // Measure how quickly SPRT reaches a decision for different positive rates
    for &positive_rate in &[0.1, 0.5, 0.9] {
        group.bench_with_input(
            BenchmarkId::new("decision_time", format!("{:.0}pct", positive_rate * 100.0)),
            &positive_rate,
            |b, &rate| {
                b.iter(|| {
                    let mut evaluator = SequentialEvaluator::new(0.01, 0.01, 0.0);
                    for i in 0..10_000u64 {
                        let is_positive = (i as f64 / 10_000.0) < rate;
                        evaluator.add_sample(is_positive);
                        let decision = evaluator.should_stop();
                        if !matches!(
                            decision,
                            neat_ai_discovery::analysis::early_termination::EarlyTerminationDecision::Continue
                        ) {
                            return black_box(evaluator.sample_count());
                        }
                    }
                    black_box(evaluator.sample_count())
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_pre_filter,
    bench_budget_tracker,
    bench_incremental_confidence,
    bench_cross_module_dedup,
    bench_sprt_evaluator,
);
criterion_main!(benches);
