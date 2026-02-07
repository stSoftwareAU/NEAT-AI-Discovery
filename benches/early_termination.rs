//! Benchmark for Issue #429: Early termination improvements for low-value candidates.
//!
//! Measures the performance of the candidate pre-filter across varying
//! candidate counts. The pre-filter applies hierarchical filtering,
//! budget-aware prioritisation, and cross-module deduplication to reduce
//! analysis overhead.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use neat_ai_discovery::analysis::candidate_prefilter::{CandidatePreFilter, PreFilterConfig};
use neat_ai_discovery::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson};

/// Create a batch of candidates with mixed characteristics:
/// - 30% have near-zero gain (will be filtered by min_gain)
/// - 20% are duplicates of earlier pairs (will be filtered by dedup)
/// - 50% are genuine high-gain unique candidates
fn create_mixed_candidates(count: usize) -> Vec<CoordinatedStructuralCandidateJson> {
    let mut candidates = Vec::with_capacity(count);

    for i in 0..count {
        let (from, to, gain) = if i % 10 < 3 {
            // 30%: near-zero gain
            (format!("low-{i}"), "output-0".to_string(), 1e-12_f32)
        } else if i % 10 < 5 {
            // 20%: duplicate pairs (reuse pair from i-3)
            let dup_idx = i.saturating_sub(3);
            (
                format!("src-{dup_idx}"),
                "output-0".to_string(),
                0.1 + (i as f32 * 0.001),
            )
        } else {
            // 50%: genuine unique candidates
            (
                format!("src-{i}"),
                "output-0".to_string(),
                0.1 + (i as f32 * 0.001),
            )
        };

        candidates.push(CoordinatedStructuralCandidateJson {
            operations: vec![CoordinatedStructuralOpJson::AddSynapse {
                from_neuron_uuid: from,
                to_neuron_uuid: to,
                weight: 0.5,
            }],
            expected_creature_score_gain: gain,
            comment: None,
        });
    }

    candidates
}

fn benchmark_prefilter(c: &mut Criterion) {
    let candidate_counts = [50, 200, 500, 1000];

    let mut group = c.benchmark_group("candidate_prefilter");

    for &count in &candidate_counts {
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, &count| {
            let candidates = create_mixed_candidates(count);

            b.iter(|| {
                let config = PreFilterConfig {
                    candidate_budget: 256,
                    dedup_enabled: true,
                    ..PreFilterConfig::default()
                };
                let mut filter = CandidatePreFilter::new(config);
                let result = filter.filter(candidates.clone());
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark simulating multiple discovery modules dispatching candidates
/// through the same pre-filter (cross-module deduplication scenario).
fn benchmark_prefilter_multi_module(c: &mut Criterion) {
    let module_count = 10;
    let candidates_per_module = 50;

    let mut group = c.benchmark_group("candidate_prefilter_multi_module");

    group.bench_function("10_modules_x_50_candidates", |b| {
        // Pre-generate candidate batches for each module
        let batches: Vec<Vec<CoordinatedStructuralCandidateJson>> = (0..module_count)
            .map(|m| {
                (0..candidates_per_module)
                    .map(|i| {
                        // Some overlap across modules: same pair appears in multiple modules
                        let from = if i < 10 {
                            format!("shared-src-{i}") // first 10 are shared across modules
                        } else {
                            format!("mod{m}-src-{i}")
                        };
                        CoordinatedStructuralCandidateJson {
                            operations: vec![CoordinatedStructuralOpJson::AddSynapse {
                                from_neuron_uuid: from,
                                to_neuron_uuid: "output-0".to_string(),
                                weight: 0.5,
                            }],
                            expected_creature_score_gain: 0.1 + (i as f32 * 0.001),
                            comment: None,
                        }
                    })
                    .collect()
            })
            .collect();

        b.iter(|| {
            let config = PreFilterConfig {
                candidate_budget: 256,
                dedup_enabled: true,
                ..PreFilterConfig::default()
            };
            let mut filter = CandidatePreFilter::new(config);

            let mut total_accepted = 0;
            for batch in &batches {
                if filter.budget_exhausted() {
                    filter.record_module_skipped();
                    continue;
                }
                let result = filter.filter(batch.clone());
                total_accepted += result.len();
            }
            black_box(total_accepted);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_prefilter,
    benchmark_prefilter_multi_module
);
criterion_main!(benches);
