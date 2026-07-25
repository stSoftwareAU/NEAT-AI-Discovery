//! Issue #1752 — `mark_analysis_finished` must saturate **atomically**.
//!
//! The original implementation read the counter with a `load` and then
//! decremented with a separate `fetch_sub`. Both operations are individually
//! atomic, but not atomic as a group: two threads racing on a counter of `1`
//! both observe `prev == 1`, both pass the guard, and both decrement, wrapping
//! the `usize` to `usize::MAX`. `is_analysis_active()` would then return `true`
//! forever and the host would never delete the parquet temp directory — the
//! "discovery locked up" symptom Issue #1077's guard exists to prevent.
//!
//! These tests exercise the observable behaviour (the counter value), not the
//! implementation, so they keep working if the decrement is refactored again.

use neat_ai_discovery::cancellation::{
    analysis_active_count, is_analysis_active, mark_analysis_finished, mark_analysis_started,
    reset_analysis_active,
};
use std::sync::Arc;
use std::sync::Barrier;

/// A single unbalanced call must not underflow the counter.
#[test]
fn unbalanced_finish_saturates_at_zero() {
    reset_analysis_active();

    mark_analysis_finished();

    assert_eq!(
        analysis_active_count(),
        0,
        "an unmatched finish must saturate at zero, not wrap"
    );
    assert!(
        !is_analysis_active(),
        "a saturated counter must report no active analysis"
    );
}

/// Concurrent unbalanced calls must not underflow either.
///
/// Each round leaves the counter at `1` and releases `THREADS` finishers at the
/// same instant through a barrier. Exactly one decrement may take effect; the
/// rest must saturate. Against the old load-then-`fetch_sub` implementation this
/// wraps to `usize::MAX` within a few rounds.
#[test]
fn concurrent_unbalanced_finishes_never_underflow() {
    const THREADS: usize = 8;
    const ROUNDS: usize = 2_000;

    for round in 0..ROUNDS {
        reset_analysis_active();
        mark_analysis_started();
        assert_eq!(analysis_active_count(), 1, "round {round} setup");

        let barrier = Arc::new(Barrier::new(THREADS));
        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    mark_analysis_finished();
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("finisher thread must not panic");
        }

        assert_eq!(
            analysis_active_count(),
            0,
            "round {round}: {THREADS} concurrent finishers on a counter of 1 must \
             leave it at zero, never wrapped to usize::MAX"
        );
    }

    reset_analysis_active();
}

/// Balanced concurrent start/finish pairs must still return the counter to zero.
#[test]
fn balanced_concurrent_pairs_return_to_zero() {
    const THREADS: usize = 8;
    const ITERATIONS: usize = 500;

    reset_analysis_active();

    let barrier = Arc::new(Barrier::new(THREADS));
    let handles: Vec<_> = (0..THREADS)
        .map(|_| {
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                for _ in 0..ITERATIONS {
                    mark_analysis_started();
                    mark_analysis_finished();
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("worker thread must not panic");
    }

    assert_eq!(
        analysis_active_count(),
        0,
        "balanced start/finish pairs must leave the counter at zero"
    );
    assert!(!is_analysis_active());
}
