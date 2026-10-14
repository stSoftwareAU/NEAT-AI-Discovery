//! Regression test for Issue #2355 (CWE-407): `assess_sentinel_cluster`
//! partitioned samples with `(0..n).filter(|i| !sentinel_indices.contains(i))`,
//! an O(n·k) scan. Because the density rule guarantees `k >= 0.15·n`
//! (`MIN_SENTINEL_FRACTION`), that is quadratic in the sample count.
//!
//! The single-pass partition must classify every sample exactly once, in
//! ascending order. The N-vs-4N timing comparison is a Criterion benchmark
//! (`benches/sentinel_cluster_partition.rs`), not a test, because timing never
//! decides pass/fail here (CONTRIBUTING.md, Unit Tests vs Benchmarks).

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for synthetic activations (Issue #873)

use neat_ai_discovery::analysis::detection::sentinel_cluster::assess_sentinel_cluster;

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

#[test]
fn assess_sentinel_cluster_partitions_every_sample_once() {
    for n in [101_usize, 10, 5] {
        let (acts, errs) = fixture(n);
        let cluster = assess_sentinel_cluster(&acts, &errs, -1.0)
            .unwrap_or_else(|| panic!("fixture n={n} must be accepted"));

        let evens: Vec<usize> = (0..n).filter(|i| i % 2 == 0).collect();
        let odds: Vec<usize> = (0..n).filter(|i| i % 2 == 1).collect();
        assert_eq!(cluster.sentinel_indices, evens, "n={n}");
        assert_eq!(cluster.non_sentinel_indices, odds, "n={n}");

        let mut all: Vec<usize> = cluster
            .sentinel_indices
            .iter()
            .chain(&cluster.non_sentinel_indices)
            .copied()
            .collect();
        all.sort_unstable();
        assert_eq!(all, (0..n).collect::<Vec<_>>(), "n={n}");

        let expected = evens.len() as f32 / n as f32;
        assert!((cluster.sentinel_fraction - expected).abs() < 1e-6, "n={n}");
    }
}
