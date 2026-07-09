//! Benchmark for Issue #1544: CPU pre-reject screen before the helpful GPU submit.
//!
//! Two things are measured:
//!
//! 1. **Screen CPU cost** — the throughput of
//!    [`helpful_candidate_has_no_signal`] over a realistic per-source sample
//!    batch. This must be negligible relative to a GPU submit + sync round-trip
//!    (tens of microseconds to milliseconds) for the pre-reject to pay off.
//! 2. **Helpful GPU-work reduction** — a deterministic count of how many
//!    helpful work items the screen removes on a synthetic GRQ-shaped batch.
//!    The helpful GPU path issues one `dispatch_workgroups` per non-empty work
//!    item (`gpu::helpful_evaluation::evaluate_helpful_batch`, chunked by
//!    `effective_batch_size`), so dropping a work item removes that dispatch
//!    and its buffer write/copy. Any target whose batch empties entirely also
//!    loses a full helpful submit (a `shaderTimings.helpful.calls` decrement).
//!    Printed once at start-up.
//!
//! NOTE: This is a synthetic workload — the production GRQ `network.json` +
//! parquet fixture named in the issue is not available in this environment. The
//! reduction figure below is a mechanical, quality-neutral consequence of the
//! screen (a screened-out candidate has no finite optimal outgoing weight, so
//! the downstream result loop would reject it anyway). The dud fraction is a
//! chosen synthetic mix; the wall-clock / `shaderTimings.helpful.calls` success
//! criteria still need the GRQ fixture on a GPU host to confirm.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for benchmark reporting (Issue #873)
use criterion::Criterion;
use neat_ai_discovery::analysis::samples::HelpfulSample;
use neat_ai_discovery::analysis::synapse::cpu_pre_reject::helpful_candidate_has_no_signal;
use std::hint::black_box;

fn sample(activation: f32, avg_error: f32) -> HelpfulSample {
    HelpfulSample {
        activation,
        avg_error,
        target_value: None,
        target_activation: None,
    }
}

/// Deterministic pseudo-random in [-1, 1) from an index (no `rand` needed).
fn pseudo(i: u64) -> f32 {
    // FNV-1a-ish scramble → map top bits to [-1, 1).
    let mut h = 0xcbf2_9ce4_8422_2325u64 ^ i.wrapping_mul(0x0100_0000_01b3);
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    ((h >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

/// Build one source's samples of a given kind.
///
/// - `dead`: constant-zero activation (unused input observation).
/// - `uncorrelated`: activation varies but has zero covariance with the error.
/// - `signal`: activation correlated with the error (a real candidate).
fn source_samples(kind: u8, seed: u64, n: usize) -> Vec<HelpfulSample> {
    (0..n)
        .map(|j| {
            let idx = seed.wrapping_mul(1_000_003).wrapping_add(j as u64);
            match kind {
                0 => sample(0.0, pseudo(idx)),          // dead: no activation energy
                1 => sample(pseudo(idx), pseudo(!idx)), // uncorrelated act vs error
                _ => {
                    let a = pseudo(idx);
                    sample(a, a * 0.6 + 0.05 * pseudo(!idx)) // correlated + noise
                }
            }
        })
        .collect()
}

/// A synthetic GRQ-shaped batch: a large source fan-in where a substantial
/// fraction are dead/uncorrelated duds (the "many end with
/// `gpu_improved_count == 0`" case the issue targets).
fn grq_shaped_batch(source_count: usize, samples_per_source: usize) -> Vec<Vec<HelpfulSample>> {
    (0..source_count)
        .map(|s| {
            // 40% dead, 30% uncorrelated, 30% signal.
            let kind = match s % 10 {
                0..=3 => 0u8,
                4..=6 => 1u8,
                _ => 2u8,
            };
            source_samples(kind, s as u64, samples_per_source)
        })
        .collect()
}

/// Print the deterministic helpful-shader-call reduction once.
fn print_reduction_analysis() {
    for (sources, spp) in [(2_000usize, 128usize), (5_000, 256)] {
        let batch = grq_shaped_batch(sources, spp);
        let total = batch.len();
        let survivors = batch
            .iter()
            .filter(|s| !helpful_candidate_has_no_signal(s))
            .count();
        let dropped = total - survivors;
        let reduction = dropped as f64 / total as f64 * 100.0;
        eprintln!(
            "[issue-1544] batch: {total} sources × {spp} samples → \
             helpful work items (GPU dispatches): {total} (baseline) vs \
             {survivors} (screened); dropped {dropped} \
             ({reduction:.1}% fewer helpful GPU dispatches)"
        );
    }
}

fn bench_screen(c: &mut Criterion) {
    // Per-source screen cost across a range of sample counts.
    let mut group = c.benchmark_group("cpu_pre_reject_screen");
    for spp in [64usize, 256, 1024] {
        let signal = source_samples(2, 42, spp);
        let dead = source_samples(0, 7, spp);
        group.bench_function(format!("signal_source_{spp}_samples"), |b| {
            b.iter(|| black_box(helpful_candidate_has_no_signal(black_box(&signal))));
        });
        group.bench_function(format!("dead_source_{spp}_samples"), |b| {
            b.iter(|| black_box(helpful_candidate_has_no_signal(black_box(&dead))));
        });
    }
    group.finish();

    // Whole-batch screen cost (what runs once per target before submit).
    let batch = grq_shaped_batch(2_000, 128);
    c.bench_function("cpu_pre_reject_batch_2000x128", |b| {
        b.iter(|| {
            let survivors = batch
                .iter()
                .filter(|s| !helpful_candidate_has_no_signal(black_box(s)))
                .count();
            black_box(survivors)
        });
    });
}

fn main() {
    print_reduction_analysis();
    // Run the criterion timing benchmarks.
    let mut criterion = Criterion::default().configure_from_args();
    bench_screen(&mut criterion);
    criterion.final_summary();
}
