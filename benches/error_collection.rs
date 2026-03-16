//! Benchmark: Mutex-based vs fold/reduce error value collection (Issue #834).
//!
//! Compares the old approach (shared `Mutex<Vec<f32>>` with `.lock().extend()`)
//! against the new lock-free approach (Rayon `try_fold`/`try_reduce`).

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use parking_lot::Mutex;
use rayon::prelude::*;
use std::sync::Arc;

/// Simulate the old mutex-based collection pattern.
fn collect_with_mutex(target_count: usize, errors_per_target: usize) -> Vec<f32> {
    let collected = Arc::new(Mutex::new(Vec::<f32>::new()));
    let targets: Vec<usize> = (0..target_count).collect();

    targets.par_iter().for_each(|&i| {
        let errors: Vec<f32> = (0..errors_per_target)
            .map(|j| (i * errors_per_target + j) as f32 * 0.001)
            .collect();
        if !errors.is_empty() {
            collected.lock().extend(errors);
        }
    });

    Arc::try_unwrap(collected)
        .expect("single owner")
        .into_inner()
}

/// Simulate the new fold/reduce collection pattern (Issue #834).
fn collect_with_fold_reduce(target_count: usize, errors_per_target: usize) -> Vec<f32> {
    let targets: Vec<usize> = (0..target_count).collect();

    targets
        .par_iter()
        .fold(Vec::<f32>::new, |mut acc, &i| {
            let errors: Vec<f32> = (0..errors_per_target)
                .map(|j| (i * errors_per_target + j) as f32 * 0.001)
                .collect();
            if !errors.is_empty() {
                acc.extend(errors);
            }
            acc
        })
        .reduce(Vec::new, |mut a, b| {
            a.extend(b);
            a
        })
}

fn bench_error_collection(c: &mut Criterion) {
    let mut group = c.benchmark_group("error_collection");

    for &target_count in &[50, 200, 500] {
        let errors_per_target = 100;

        group.bench_with_input(
            BenchmarkId::new("mutex", target_count),
            &target_count,
            |b, &tc| {
                b.iter(|| collect_with_mutex(tc, errors_per_target));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("fold_reduce", target_count),
            &target_count,
            |b, &tc| {
                b.iter(|| collect_with_fold_reduce(tc, errors_per_target));
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_error_collection);
criterion_main!(benches);
