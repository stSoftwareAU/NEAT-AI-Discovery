//! Issue #1929: stale GPU work requests are skipped, and the skip is visible.
//!
//! The loop-level behaviour (never invoking the analyser for an abandoned
//! request, never retrying a device-lost failure for one) is exercised by the
//! crate-internal tests in `src/analysis/gpu/queue/stale_skip_tests.rs`, since
//! `GpuWorkRequest` and the worker loop are `pub(crate)`. What is observable
//! from outside the crate — and what production reads in run telemetry — is the
//! `stale_skipped` counter, covered here.

use neat_ai_discovery::observability::{GpuMetrics, global_gpu_metrics};
use serial_test::serial;

#[test]
fn fresh_metrics_report_no_stale_skips() {
    let metrics = GpuMetrics::new();
    assert_eq!(
        metrics.stale_skipped(),
        0,
        "a healthy run starts with no skipped requests"
    );
}

#[test]
fn stale_skips_accumulate_per_skipped_request() {
    let metrics = GpuMetrics::new();
    for _ in 0..3 {
        metrics.record_stale_skip();
    }
    assert_eq!(
        metrics.stale_skipped(),
        3,
        "each skipped request must be counted exactly once"
    );
}

/// The skip counter is independent of the throughput counters — a queue that
/// only ever discarded abandoned work must not look like it did any.
#[test]
fn stale_skips_do_not_inflate_batch_or_sample_counts() {
    let metrics = GpuMetrics::new();
    metrics.record_stale_skip();
    metrics.record_stale_skip();

    assert_eq!(metrics.stale_skipped(), 2);
    assert_eq!(metrics.batch_count(), 0, "skips are not batches");
    assert_eq!(
        metrics.total_samples_processed(),
        0,
        "skips process no samples"
    );
    assert_eq!(metrics.total_gpu_busy_us(), 0, "skips consume no GPU time");
}

/// The counter is reachable through the global instance the analysis pipeline
/// uses, so a rise is visible in run telemetry without extra plumbing.
#[test]
#[serial]
fn global_metrics_expose_the_stale_skip_counter() {
    let before = global_gpu_metrics().stale_skipped();
    global_gpu_metrics().record_stale_skip();
    assert_eq!(
        global_gpu_metrics().stale_skipped(),
        before + 1,
        "the global counter must record the skip"
    );
}
