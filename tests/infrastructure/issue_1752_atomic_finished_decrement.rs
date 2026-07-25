//! Tests that `mark_analysis_finished()` decrements atomically (Issue #1752).
//!
//! The saturating decrement must be a single atomic read-modify-write. A
//! separate `load` followed by `fetch_sub` lets two concurrent unbalanced
//! callers both observe `count == 1`, both pass the guard, and both decrement —
//! wrapping the counter to `usize::MAX`. `is_analysis_active()` then returns
//! `true` forever and the host never deletes the parquet temp directory, which
//! is the "discovery locked up" symptom Issue #1077's guard exists to prevent.

use neat_ai_discovery::cancellation::{
    analysis_active_count, mark_analysis_finished, mark_analysis_started, reset_analysis_active,
};
use serial_test::serial;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

/// Unbalanced concurrent finishes must saturate at zero, never wrap.
///
/// One thread issues `ITERATIONS` starts while `FINISHERS` threads spam
/// unmatched finishes. A correct saturating decrement keeps the counter inside
/// `[0, ITERATIONS]` at all times. A non-atomic check-then-decrement lets two
/// finishers both observe `count == 1` and both subtract, wrapping the counter
/// past zero to a value near `usize::MAX` — far outside the sane bound.
#[test]
#[serial]
fn concurrent_unbalanced_finishes_saturate_at_zero() {
    const FINISHERS: usize = 8;
    const ITERATIONS: usize = 200_000;
    /// Any observation above this can only be an underflow wrap.
    const SANE_BOUND: usize = ITERATIONS * 2;

    reset_analysis_active();

    let wrapped = Arc::new(AtomicBool::new(false));

    let starter = {
        let wrapped = Arc::clone(&wrapped);
        thread::spawn(move || {
            for _ in 0..ITERATIONS {
                mark_analysis_started();
                if analysis_active_count() > SANE_BOUND {
                    wrapped.store(true, Ordering::Relaxed);
                }
            }
        })
    };

    let finishers: Vec<_> = (0..FINISHERS)
        .map(|_| {
            let wrapped = Arc::clone(&wrapped);
            thread::spawn(move || {
                for _ in 0..ITERATIONS {
                    mark_analysis_finished();
                    if analysis_active_count() > SANE_BOUND {
                        wrapped.store(true, Ordering::Relaxed);
                    }
                }
            })
        })
        .collect();

    starter.join().expect("starter thread panicked");
    for handle in finishers {
        handle.join().expect("finisher thread panicked");
    }

    let final_count = analysis_active_count();
    reset_analysis_active();

    assert!(
        !wrapped.load(Ordering::Relaxed),
        "counter underflowed: a concurrent finish decremented past zero"
    );
    assert!(
        final_count <= SANE_BOUND,
        "counter must saturate at 0, never wrap below zero (final count {final_count})"
    );
}

/// A single unbalanced finish on an idle counter must leave it at zero.
#[test]
#[serial]
fn unbalanced_finish_on_idle_counter_stays_zero() {
    reset_analysis_active();
    mark_analysis_finished();
    assert_eq!(analysis_active_count(), 0);
    mark_analysis_finished();
    assert_eq!(analysis_active_count(), 0);
}

/// Balanced concurrent start/finish pairs must return the counter to zero.
#[test]
#[serial]
fn balanced_concurrent_start_finish_returns_to_zero() {
    const THREADS: usize = 8;
    const ITERATIONS: usize = 500;

    reset_analysis_active();

    let handles: Vec<_> = (0..THREADS)
        .map(|_| {
            thread::spawn(|| {
                for _ in 0..ITERATIONS {
                    mark_analysis_started();
                    mark_analysis_finished();
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("worker thread panicked");
    }

    assert_eq!(
        analysis_active_count(),
        0,
        "balanced start/finish pairs must leave the counter at 0"
    );
}
