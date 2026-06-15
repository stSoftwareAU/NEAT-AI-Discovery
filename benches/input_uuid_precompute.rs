//! Benchmark for Issue #1368: input-neuron UUID string construction on the
//! per-observation recording hot path.
//!
//! `src/record/processing.rs` builds the fixed set of input UUIDs
//! (`input-0`, `input-1`, …) for every recorded observation. The old code
//! called `format!("input-{i}")` inside the inner per-observation loop, so the
//! same handful of strings were re-allocated `n_records × n_inputs` times. The
//! new code precomputes the UUIDs once per batch and clones the cached `String`
//! per use.
//!
//! These two benchmarks isolate the two strategies so the allocation churn
//! removed by the precompute is directly measurable.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

/// Old strategy: format a fresh `String` for every input on every observation.
fn format_per_observation(n_records: usize, n_inputs: usize) -> usize {
    let mut total = 0usize;
    for _ in 0..n_records {
        for input_index in 0..n_inputs {
            let uuid = format!("input-{input_index}");
            total += black_box(uuid).len();
        }
    }
    total
}

/// New strategy: precompute the UUIDs once, then clone the cached string per use.
fn precompute_once(n_records: usize, n_inputs: usize) -> usize {
    let input_uuids: Vec<String> = (0..n_inputs).map(|i| format!("input-{i}")).collect();
    let mut total = 0usize;
    for _ in 0..n_records {
        for cached in &input_uuids {
            let uuid = cached.clone();
            total += black_box(uuid).len();
        }
    }
    total
}

fn bench_input_uuid_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("input_uuid_precompute");

    // Recording hot path: thousands of observations × tens-to-hundreds of inputs.
    for &(n_records, n_inputs) in &[(2_000usize, 32usize), (2_000, 128), (5_000, 64)] {
        let id = format!("{n_records}rec_{n_inputs}in");

        group.bench_with_input(
            BenchmarkId::new("format_per_observation", &id),
            &(n_records, n_inputs),
            |b, &(r, i)| b.iter(|| black_box(format_per_observation(r, i))),
        );

        group.bench_with_input(
            BenchmarkId::new("precompute_once", &id),
            &(n_records, n_inputs),
            |b, &(r, i)| b.iter(|| black_box(precompute_once(r, i))),
        );
    }

    group.finish();
}

criterion_group!(benches, bench_input_uuid_construction);
criterion_main!(benches);
