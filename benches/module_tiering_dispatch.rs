//! Benchmark for Issue #1547: creature-scale module tiering during dispatch.
//!
//! Measures the wall-clock of the parallel discovery-detection phase with the
//! full module set versus the tiered set (expensive modules skipped) on a
//! GRQ-scale creature. Expensive modules are modelled with a pairwise
//! `O(hidden^2)` scan — the super-linear cost that motivates tiering — while
//! standard modules do a light `O(hidden)` pass. This isolates the dispatch
//! saving from real detection work, which requires the GRQ production fixtures.
//!
//! Run with: `cargo bench --bench module_tiering_dispatch`

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for benchmarking

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use neat_ai_discovery::analysis::discovery_dispatch::{
    DiscoveryModuleSpec, detect_discovery_modules_parallel,
};
use neat_ai_discovery::analysis::module_tiering::{EXPENSIVE_MODULES, should_skip_module};
use std::hint::black_box;

/// Standard-tier module names used to pad the set to a realistic count.
const STANDARD_MODULES: &[&str] = &[
    "dead neuron detection",
    "dormant synapse detection",
    "gradient-based discovery",
    "saturation detection",
    "bottleneck detection",
    "oscillating neuron detection",
    "noisy synapse detection",
    "bounded range detection",
    "sentinel value gating",
    "dominant input detection",
    "threshold effect detection",
    "sample-weighted discovery",
    "output range compression detection",
    "error plateau detection",
    "low impact neuron detection",
    "restricted range detection",
    "operating point analysis",
    "monotonicity detection",
    "bimodal neuron detection",
    "output conflict detection",
];

/// Light `O(hidden)` work standing in for a standard-tier module.
fn cheap_scan(hidden: usize) -> f64 {
    let mut acc = 0.0_f64;
    for i in 0..hidden {
        acc += (i as f64).sin();
    }
    acc
}

/// Super-linear `O(hidden^2)` pairwise scan standing in for an expensive module.
fn expensive_scan(hidden: usize) -> f64 {
    let mut acc = 0.0_f64;
    for i in 0..hidden {
        for j in (i + 1)..hidden {
            acc += ((i * j) as f64).sqrt();
        }
    }
    acc
}

/// Build the full ~26-module set, expensive modules doing the pairwise scan.
fn build_modules(hidden: usize, tiered: bool, threshold: usize) -> Vec<DiscoveryModuleSpec> {
    let mut modules: Vec<DiscoveryModuleSpec> = Vec::new();

    for &name in EXPENSIVE_MODULES {
        // Tiered runs skip expensive modules on a large creature (no escalation).
        if tiered && should_skip_module(name, hidden, threshold, false) {
            continue;
        }
        modules.push(DiscoveryModuleSpec {
            module_name: name.to_string(),
            phase_name: "bench_expensive",
            max_candidates: 0,
            detect_fn: Box::new(move || {
                black_box(expensive_scan(hidden));
                None
            }),
        });
    }

    for &name in STANDARD_MODULES {
        modules.push(DiscoveryModuleSpec {
            module_name: name.to_string(),
            phase_name: "bench_standard",
            max_candidates: 0,
            detect_fn: Box::new(move || {
                black_box(cheap_scan(hidden));
                None
            }),
        });
    }

    modules
}

fn bench_tiering(c: &mut Criterion) {
    let mut group = c.benchmark_group("module_tiering_dispatch");
    // Keep the pairwise scan tractable while still super-linear; 400 hidden
    // neurons is enough for the expensive scan to dominate the phase.
    let hidden = 400;
    let threshold = 300; // below `hidden`, so tiering engages for the "tiered" arm

    group.bench_with_input(BenchmarkId::new("full_set", hidden), &hidden, |b, &h| {
        b.iter(|| {
            let modules = build_modules(h, false, threshold);
            let results = detect_discovery_modules_parallel(modules, None, None);
            black_box(results.entries.len());
        });
    });

    group.bench_with_input(BenchmarkId::new("tiered_set", hidden), &hidden, |b, &h| {
        b.iter(|| {
            let modules = build_modules(h, true, threshold);
            let results = detect_discovery_modules_parallel(modules, None, None);
            black_box(results.entries.len());
        });
    });

    group.finish();
}

criterion_group!(benches, bench_tiering);
criterion_main!(benches);
