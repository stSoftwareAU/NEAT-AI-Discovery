//! Benchmark for Issue #2355 (CWE-407): `assess_sentinel_cluster` partitioned
//! samples with `(0..n).filter(|i| !sentinel_indices.contains(i))`, an O(n·k)
//! scan. The density rule guarantees `k >= 0.15·n` (`MIN_SENTINEL_FRACTION`),
//! so that was quadratic in the sample count.
//!
//! Criterion reports each size's time; linear growth shows as roughly 4x per
//! 4x step in N (quadratic would be roughly 16x). The N-vs-4N comparison lives
//! here, not in `tests/`, because timing never decides pass/fail in a test
//! (CONTRIBUTING.md, Unit Tests vs Benchmarks). Correctness is pinned by
//! `tests/issue_2355_sentinel_cluster_quadratic.rs`.
//!
//! Run: `cargo bench --bench sentinel_cluster_partition`

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for synthetic activations (Issue #873)

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use neat_ai_discovery::analysis::detection::sentinel_cluster::assess_sentinel_cluster;
use std::hint::black_box;

/// Even indices are pinned at -1.0 with zero error. Odd indices carry useful
/// activations across roughly 0.2..=0.8 with varying errors, so the cluster is
/// accepted (dense, well separated, lower error variance).
fn fixture(n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut acts = Vec::with_capacity(n);
    let mut errs = Vec::with_capacity(n);
    for i in 0..n {
        if i % 2 == 0 {
            acts.push(-1.0);
            errs.push(0.0);
        } else {
            acts.push(0.2 + 0.6 * ((i * 37) % 100) as f32 / 99.0);
            errs.push(if (i / 2) % 2 == 0 { 0.5 } else { -0.5 });
        }
    }
    (acts, errs)
}

fn bench_assess_sentinel_cluster(c: &mut Criterion) {
    let mut group = c.benchmark_group("assess_sentinel_cluster");

    for &n in &[4_000usize, 16_000, 64_000] {
        let (acts, errs) = fixture(n);
        group.bench_with_input(BenchmarkId::new("partition", n), &n, |b, _| {
            b.iter(|| {
                let result =
                    assess_sentinel_cluster(black_box(&acts), black_box(&errs), black_box(-1.0));
                assert!(result.is_some(), "fixture must be accepted");
                black_box(result)
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_assess_sentinel_cluster);
criterion_main!(benches);
