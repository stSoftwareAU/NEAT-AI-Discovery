//! Benchmark for Issue #743: Eliminate intermediate String allocation in
//! deterministic UUID hashing.
//!
//! Measures the performance of `deterministic_coordinated_neuron_uuid()` which
//! generates deterministic UUIDs for coordinated structural candidates.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use criterion::{Criterion, criterion_group, criterion_main};
use neat_ai_discovery::analysis::deterministic_coordinated_neuron_uuid;
use std::hint::black_box;

fn bench_deterministic_uuid(c: &mut Criterion) {
    let mut group = c.benchmark_group("deterministic_uuid");

    // Realistic UUID-like strings
    let sources = [
        "input-a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        "hidden-deadbeef-cafe-babe-face-123456789abc",
        "output-11111111-2222-3333-4444-555555555555",
    ];
    let targets = [
        "hidden-aaaabbbb-cccc-dddd-eeee-ffffffffffff",
        "output-99998888-7777-6666-5555-444433332222",
        "hidden-12345678-abcd-ef01-2345-6789abcdef01",
    ];
    let squashes = ["LOGISTIC", "TANH", "ReLU", "IDENTITY"];

    group.bench_function("10000_iterations", |b| {
        b.iter(|| {
            let mut last = String::new();
            for i in 0..10_000u32 {
                let src = sources[i as usize % sources.len()];
                let tgt = targets[i as usize % targets.len()];
                let sq = squashes[i as usize % squashes.len()];
                let w1 = (i as f32) * 0.001;
                let w2 = (i as f32) * -0.002;
                let bias = (i as f32) * 0.0005;
                last = deterministic_coordinated_neuron_uuid(
                    black_box(src),
                    black_box(tgt),
                    black_box(sq),
                    black_box(w1),
                    black_box(w2),
                    black_box(bias),
                );
            }
            last
        });
    });

    // Single-call benchmark for per-call latency
    group.bench_function("single_call", |b| {
        b.iter(|| {
            deterministic_coordinated_neuron_uuid(
                black_box("input-a1b2c3d4-e5f6-7890-abcd-ef1234567890"),
                black_box("hidden-aaaabbbb-cccc-dddd-eeee-ffffffffffff"),
                black_box("LOGISTIC"),
                black_box(0.5),
                black_box(-0.3),
                black_box(0.1),
            )
        });
    });

    group.finish();
}

criterion_group!(benches, bench_deterministic_uuid);
criterion_main!(benches);
