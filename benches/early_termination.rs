//! Benchmark for Issue #429: Early termination improvements for low-value candidates.
//!
//! Measures the performance impact of candidate pre-filtering and cross-module
//! deduplication on the discovery pipeline.
//!
//! ## What is Measured
//!
//! - `candidate_prefilter`: Time to filter candidates by improvement threshold and
//!   cross-module deduplication.
//! - `sprt_evaluation`: Time for SPRT batch evaluation (existing early termination).

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use neat_ai_discovery::analysis::candidate_prefilter::{
    deduplicate_candidates, filter_low_value_candidates, CandidatePrefilterConfig,
};
use neat_ai_discovery::CoordinatedStructuralCandidateJson;
use neat_ai_discovery::CoordinatedStructuralOpJson;

/// Create test candidates with varying expected improvements.
fn create_test_candidates(count: usize) -> Vec<CoordinatedStructuralCandidateJson> {
    (0..count)
        .map(|i| {
            let improvement = if i % 10 == 0 {
                // 10% are high-value
                0.05 + (i as f32 * 0.001)
            } else if i % 3 == 0 {
                // ~30% are medium-value
                0.005 + (i as f32 * 0.0001)
            } else {
                // ~60% are low-value
                0.0001 * (i as f32 % 5.0)
            };

            CoordinatedStructuralCandidateJson {
                operations: vec![CoordinatedStructuralOpJson::RemoveSynapse {
                    from_neuron_uuid: format!("source-{i}"),
                    to_neuron_uuid: format!("target-{}", i % 20),
                }],
                expected_creature_score_gain: improvement,
                comment: Some(format!("test candidate {i}")),
            }
        })
        .collect()
}

/// Create test candidates with many duplicates (same target neuron, same operation type).
fn create_duplicate_candidates(count: usize) -> Vec<CoordinatedStructuralCandidateJson> {
    (0..count)
        .map(|i| {
            let target_idx = i % 5; // Only 5 unique targets
            CoordinatedStructuralCandidateJson {
                operations: vec![CoordinatedStructuralOpJson::RemoveSynapse {
                    from_neuron_uuid: format!("source-{i}"),
                    to_neuron_uuid: format!("target-{target_idx}"),
                }],
                expected_creature_score_gain: 0.01 + (i as f32 * 0.0001),
                comment: Some(format!("duplicate candidate {i}")),
            }
        })
        .collect()
}

fn benchmark_candidate_prefilter(c: &mut Criterion) {
    let candidate_counts = [50, 200, 500, 1000];
    let config = CandidatePrefilterConfig::default();

    let mut group = c.benchmark_group("candidate_prefilter");

    for &count in &candidate_counts {
        group.bench_with_input(
            BenchmarkId::new("filter_low_value", count),
            &count,
            |b, &count| {
                let candidates = create_test_candidates(count);
                b.iter(|| {
                    let result = filter_low_value_candidates(black_box(&candidates), &config);
                    black_box(result);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("deduplicate", count),
            &count,
            |b, &count| {
                let candidates = create_duplicate_candidates(count);
                b.iter(|| {
                    let result = deduplicate_candidates(black_box(&candidates), &config);
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, benchmark_candidate_prefilter);
criterion_main!(benches);
