//! Regression test for the panicked-GPU-thread defence (Issue #2361).
//!
//! Before this fix, a panic inside the GPU thread's work loop unwound the
//! thread without ever touching the queued requests behind the one that
//! panicked: every response channel was simply dropped, and the production
//! `GpuWorkQueue` holds the *sending* half of the work channel forever, so no
//! `Disconnected` ever reached a waiting caller either. The caller just sat
//! out its full stall window/batch timeout and was reported as a wedge
//! (`Stalled`), discarding the payload along the way. This test drives the
//! real [`run_guarded_gpu_thread`] guard against a fake device that panics on
//! its first request and asserts that a request queued behind it fails
//! promptly with a typed panic error instead of stalling.

use crossbeam_channel::bounded;
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::fake_evaluator::{FakeEvaluatorFactory, FakeGpuEvaluator, WedgeBehaviour};
use super::scheduling::run_guarded_gpu_thread;
use super::staleness::caller_liveness_pair;
use super::submission::{GpuWaitOutcome, wait_for_gpu_response};
use super::{GpuWorkQueue, GpuWorkRequest};
use crate::analysis::gpu::breaker::{GpuCircuitBreaker, GpuTripReason};
use crate::analysis::gpu::budget::GpuTimeBudget;
use crate::analysis::gpu::heartbeat::GpuHeartbeat;
use crate::analysis::samples::HelpfulSample;

/// The stall window this test configures, matching `wedge_tests`'s
/// [`STALL_WINDOW`](super::wedge_tests) choice — short enough to keep the
/// suite fast, long enough that scheduling jitter cannot be misread as a
/// wedge.
const STALL_WINDOW: Duration = Duration::from_millis(300);

/// Upper bound on the second caller's wait. A prompt panic failure must cost
/// nowhere near the stall window; this is a generous backstop so the harness
/// fails the assertion rather than hanging on a real regression.
const DETECTION_CAP: Duration = Duration::from_secs(3);

fn sample() -> HelpfulSample {
    HelpfulSample {
        activation: 1.0,
        avg_error: 0.5,
        target_value: None,
        target_activation: None,
    }
}

/// Issue #2361: a GPU-thread panic must not strand the requests still queued
/// behind the one that panicked. The guard must catch the panic, trip the
/// breaker with [`GpuTripReason::WorkerPanicked`], drain every queued request
/// with a typed error, and the request whose evaluation actually panicked
/// must fail too (its response sender is dropped mid-unwind).
#[test]
fn queued_request_fails_promptly_when_the_gpu_thread_panics() {
    let (work_tx, work_rx) = bounded::<GpuWorkRequest>(8);
    let heartbeat = Arc::new(GpuHeartbeat::new());
    let evaluator = FakeGpuEvaluator::new(WedgeBehaviour::Panics, Arc::clone(&heartbeat));
    let factory = Arc::new(FakeEvaluatorFactory::new());
    let breaker: &'static GpuCircuitBreaker = Box::leak(Box::new(GpuCircuitBreaker::new()));

    // Both requests are enqueued before the worker starts, so the second is
    // still sitting in the channel when the first panics.
    let (response_tx1, response_rx1) = bounded(1);
    let (guard1, liveness1) = caller_liveness_pair();
    work_tx
        .send_timeout(
            GpuWorkRequest::HelpfulBatch {
                samples: vec![Arc::new(vec![sample()])],
                response_tx: response_tx1,
                budget: GpuTimeBudget::unbounded(),
                liveness: liveness1,
            },
            Duration::from_secs(1),
        )
        .expect("the test work queue must accept the first request");

    let (response_tx2, response_rx2) = bounded(1);
    let (guard2, liveness2) = caller_liveness_pair();
    work_tx
        .send_timeout(
            GpuWorkRequest::HelpfulBatch {
                samples: vec![Arc::new(vec![sample()])],
                response_tx: response_tx2,
                budget: GpuTimeBudget::unbounded(),
                liveness: liveness2,
            },
            Duration::from_secs(1),
        )
        .expect("the test work queue must accept the second request");

    // Keep the work queue's sending half alive for the whole test, mirroring
    // production: `GpuWorkQueue` holds `work_tx` for the queue's lifetime,
    // which is exactly what would otherwise strand queued requests forever.
    let _work_tx = work_tx;

    let drain_rx = work_rx.clone();
    let loop_heartbeat = Arc::clone(&heartbeat);
    let loop_factory = Arc::clone(&factory);
    let worker = std::thread::spawn(move || {
        run_guarded_gpu_thread(&drain_rx, breaker, move || {
            GpuWorkQueue::run_work_loop(evaluator, work_rx, &*loop_factory, &loop_heartbeat);
        });
    });

    // The second caller must fail promptly, not stall.
    let started = Instant::now();
    let outcome = wait_for_gpu_response(&response_rx2, DETECTION_CAP, &heartbeat, STALL_WINDOW);
    let elapsed = started.elapsed();
    drop(guard2);

    match outcome {
        GpuWaitOutcome::Answered(Err(e)) => {
            let msg = e.to_string();
            assert!(
                msg.contains("panicked"),
                "the queued request's error must mention the panic, got: {msg}"
            );
            assert!(
                msg.contains("fake GPU panicked"),
                "the queued request's error must carry the panic payload, got: {msg}"
            );
        }
        GpuWaitOutcome::Answered(Ok(_)) => {
            panic!("a request queued behind a panicking worker must not succeed")
        }
        other => panic!(
            "expected a prompt typed-error answer for the queued request, got a different \
             outcome variant (not Answered): {other:?}"
        ),
    }
    assert!(
        elapsed < STALL_WINDOW,
        "the queued request must fail promptly, not ride out the stall window \
         (took {elapsed:?})"
    );

    // The request that actually panicked must also fail: its response sender
    // is dropped while the worker unwinds, so the channel disconnects.
    let first_outcome = response_rx1.recv_timeout(DETECTION_CAP);
    assert!(
        first_outcome.is_err(),
        "the panicking request's own response channel must disconnect, got: {first_outcome:?}"
    );
    drop(guard1);

    assert_eq!(
        breaker.trip_reason(),
        Some(GpuTripReason::WorkerPanicked),
        "a worker panic must trip the breaker with the panic reason"
    );

    worker
        .join()
        .expect("the guard must catch the panic, so the wrapper thread must not itself panic");
}

/// Issue #2361: even with no request queued behind the panicking one, the
/// breaker must still trip and the guarded thread must still join cleanly.
#[test]
fn panicking_worker_with_empty_queue_still_trips_the_breaker() {
    let (work_tx, work_rx) = bounded::<GpuWorkRequest>(8);
    let heartbeat = Arc::new(GpuHeartbeat::new());
    let evaluator = FakeGpuEvaluator::new(WedgeBehaviour::Panics, Arc::clone(&heartbeat));
    let factory = Arc::new(FakeEvaluatorFactory::new());
    let breaker: &'static GpuCircuitBreaker = Box::leak(Box::new(GpuCircuitBreaker::new()));

    let (response_tx, response_rx) = bounded(1);
    let (guard, liveness) = caller_liveness_pair();
    work_tx
        .send_timeout(
            GpuWorkRequest::HelpfulBatch {
                samples: vec![Arc::new(vec![sample()])],
                response_tx,
                budget: GpuTimeBudget::unbounded(),
                liveness,
            },
            Duration::from_secs(1),
        )
        .expect("the test work queue must accept the only request");

    let _work_tx = work_tx;

    let drain_rx = work_rx.clone();
    let loop_heartbeat = Arc::clone(&heartbeat);
    let loop_factory = Arc::clone(&factory);
    let worker = std::thread::spawn(move || {
        run_guarded_gpu_thread(&drain_rx, breaker, move || {
            GpuWorkQueue::run_work_loop(evaluator, work_rx, &*loop_factory, &loop_heartbeat);
        });
    });

    // The request that actually panicked never gets an answer: its own
    // `response_tx` is dropped while the worker unwinds (before the guard's
    // drain loop even starts), so the channel disconnects rather than
    // delivering an `Err`.
    let outcome = response_rx.recv_timeout(DETECTION_CAP);
    drop(guard);
    assert!(
        outcome.is_err(),
        "the panicking request's own response channel must disconnect, got: {outcome:?}"
    );

    // Issue #2361 follow-up: unlike the paired test above (where the second
    // request's typed error is only sent *after* `trip()`, so observing it
    // serialises the main thread behind the trip), this channel's disconnect
    // and the breaker's trip are two independent writes on the panicking
    // worker thread with no synchronisation between them beyond program
    // order there: the drop that disconnects `response_rx` happens, by
    // construction, during the panic unwind that precedes the `trip()`
    // call. The main thread can therefore observe the disconnect a few
    // instructions before the trip actually lands. Poll briefly for it
    // rather than asserting instantaneously — bounded by the same
    // `DETECTION_CAP` backstop the test already uses to fail loudly on a
    // genuine regression.
    let poll_deadline = Instant::now() + DETECTION_CAP;
    let mut trip_reason = breaker.trip_reason();
    while trip_reason.is_none() && Instant::now() < poll_deadline {
        std::thread::yield_now();
        trip_reason = breaker.trip_reason();
    }
    assert_eq!(
        trip_reason,
        Some(GpuTripReason::WorkerPanicked),
        "a worker panic must trip the breaker even with nothing queued behind it"
    );

    worker
        .join()
        .expect("the guard must catch the panic, so the wrapper thread must not itself panic");
}
