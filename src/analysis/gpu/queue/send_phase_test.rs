//! Regression tests for the Issue #2339 send-phase wedge bound.
//!
//! Before this fix, `send_until_deadline`'s predecessor was a single
//! `send_timeout(request, timeout)` call: a full queue with a silent GPU sat
//! out the *whole* send timeout even though the breaker had already tripped or
//! the heartbeat had already stalled, and then `await_gpu_response` started a
//! **fresh** timeout on top of that, so one caller could pay for two full
//! waits. These tests exercise `send_until_deadline` directly against a
//! capacity-1 test queue (mirroring `submission::tests::test_queue`), plus one
//! full-loop test (mirroring `wedge_tests::WedgedGpu::spawn`) proving a third
//! submitter behind a wedged GPU still returns within the detection cap, and
//! one `GpuFuture::collect` test proving the send and wait phases share one
//! deadline instead of each restarting their own.
//!
//! Every breaker and heartbeat here is test-owned, never the process-wide
//! singleton, except where `GpuFuture::breaker` requires a `'static` reference
//! — that one test leaks its own breaker, exactly as the rest of this module's
//! tests already do, so it still cannot affect any other test in the binary.

use anyhow::Result;
use crossbeam_channel::bounded;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use super::fake_evaluator::{FakeEvaluatorFactory, FakeGpuEvaluator, WedgeBehaviour};
use super::staleness::caller_liveness_pair;
use super::submission::send_until_deadline;
use super::{GpuFuture, GpuWorkQueue, GpuWorkRequest};
use crate::analysis::gpu::breaker::{GpuCircuitBreaker, GpuTripReason};
use crate::analysis::gpu::budget::GpuTimeBudget;
use crate::analysis::gpu::heartbeat::GpuHeartbeat;
use crate::analysis::samples::{HelpfulSample, HelpfulStats};

/// Short, CI-friendly stall window — long enough that a healthy poll never
/// trips it by accident, short enough that a wedge test costs milliseconds.
const STALL_WINDOW: Duration = Duration::from_millis(150);

/// Upper bound on any single detection in this module. Anything approaching it
/// is a regression — it should fail the assertion, not hang the suite.
const DETECTION_CAP: Duration = Duration::from_secs(2);

/// Upper bound on waiting for the fake worker thread to dequeue a request.
const HARNESS_WAIT_CAP: Duration = Duration::from_secs(2);

fn sample() -> HelpfulSample {
    HelpfulSample {
        activation: 1.0,
        avg_error: 0.5,
        target_value: None,
        target_activation: None,
    }
}

fn helpful_request() -> GpuWorkRequest {
    let (response_tx, _response_rx) = bounded(1);
    let (_guard, liveness) = caller_liveness_pair();
    GpuWorkRequest::HelpfulBatch {
        samples: vec![Arc::new(vec![sample()])],
        response_tx,
        budget: GpuTimeBudget::unbounded(),
        liveness,
    }
}

/// A full queue with a silent GPU — nothing ever drains it — must be declared
/// wedged within the stall window, not after sitting out the whole deadline.
#[test]
fn a_full_queue_with_a_silent_gpu_is_declared_wedged_within_the_stall_window() {
    let (work_tx, _work_rx) = bounded::<GpuWorkRequest>(1);
    // Fill the queue so every further send blocks.
    work_tx
        .send_timeout(helpful_request(), Duration::from_secs(1))
        .expect("the first send fills the capacity-1 queue");

    let breaker = GpuCircuitBreaker::new();
    let heartbeat = GpuHeartbeat::new();
    let deadline = Instant::now() + Duration::from_secs(30);

    let started = Instant::now();
    let err = send_until_deadline(
        &work_tx,
        helpful_request(),
        deadline,
        &breaker,
        &heartbeat,
        STALL_WINDOW,
        "helpful batch evaluation",
        30,
    )
    .expect_err("a full queue with no progress must not wait out the whole deadline");
    let elapsed = started.elapsed();

    assert!(
        elapsed >= STALL_WINDOW,
        "the guard must not fire before the window elapses (fired after {elapsed:?})"
    );
    assert!(
        elapsed < DETECTION_CAP,
        "detection must cost the stall window, not the 30s deadline (took {elapsed:?})"
    );
    let msg = format!("{err:#}");
    assert!(
        msg.contains("no GPU-thread progress"),
        "a silent queue must be reported as a heartbeat stall: {msg}"
    );
    assert_eq!(breaker.trip_reason(), Some(GpuTripReason::HeartbeatStall));
}

/// A breaker tripped mid-send — by another caller, or by this one's own
/// earlier poll — must end the send promptly instead of waiting for the next
/// heartbeat check or the deadline.
#[test]
fn a_breaker_tripped_mid_send_ends_the_send_promptly() {
    let (work_tx, _work_rx) = bounded::<GpuWorkRequest>(1);
    work_tx
        .send_timeout(helpful_request(), Duration::from_secs(1))
        .expect("fill the capacity-1 queue");

    let breaker = GpuCircuitBreaker::new();
    let heartbeat = GpuHeartbeat::new();
    // Trip the breaker before the send even starts polling, so the very first
    // `breaker.check()?` after the first poll timeout must catch it.
    breaker.trip(GpuTripReason::AbandonedThread);
    // A long deadline and a wide stall window: only the breaker check can end
    // this send quickly; if it did not, the test would hang into the stall
    // window or the deadline instead of failing fast.
    let deadline = Instant::now() + Duration::from_secs(30);

    let started = Instant::now();
    let err = send_until_deadline(
        &work_tx,
        helpful_request(),
        deadline,
        &breaker,
        &heartbeat,
        Duration::from_secs(10),
        "helpful batch evaluation",
        30,
    )
    .expect_err("a tripped breaker must stop the send");
    let elapsed = started.elapsed();

    assert!(
        format!("{err:#}").contains("GPU circuit breaker tripped"),
        "the send must be ended by the breaker, not by the stall guard or deadline: {err:#}"
    );
    assert!(
        elapsed < DETECTION_CAP,
        "a breaker tripped before the send started must end it in roughly one \
         poll interval, not the 10s stall window (took {elapsed:?})"
    );
}

/// A full queue that is never drained, with no breaker or heartbeat guard to
/// catch it early, still ends promptly at the shared deadline — reported as the
/// same wedged-batch signal a response timeout is.
#[test]
fn a_full_queue_past_the_deadline_trips_the_breaker_as_a_batch_timeout() {
    let (work_tx, _work_rx) = bounded::<GpuWorkRequest>(1);
    work_tx
        .send_timeout(helpful_request(), Duration::from_secs(1))
        .expect("fill the capacity-1 queue");

    let breaker = GpuCircuitBreaker::new();
    let heartbeat = GpuHeartbeat::new();
    // Stall guard disabled (zero window): only the deadline can end this send.
    let deadline = Instant::now() + Duration::from_millis(200);

    let started = Instant::now();
    let err = send_until_deadline(
        &work_tx,
        helpful_request(),
        deadline,
        &breaker,
        &heartbeat,
        Duration::ZERO,
        "helpful batch evaluation",
        30,
    )
    .expect_err("a full queue past the deadline must fail");
    let elapsed = started.elapsed();

    assert!(
        elapsed >= Duration::from_millis(200),
        "the deadline must be honoured, not cut short (took {elapsed:?})"
    );
    assert!(
        elapsed < DETECTION_CAP,
        "the wait must end at the deadline, not run away (took {elapsed:?})"
    );
    let msg = format!("{err:#}");
    assert!(
        msg.contains("GPU work queue full"),
        "a deadline reached with no drain must report the queue-full signature: {msg}"
    );
    assert_eq!(breaker.trip_reason(), Some(GpuTripReason::BatchTimeout));
}

/// A GPU thread that dropped its end of the work channel is reported as a
/// closed channel, not misread as a stall or a timeout.
#[test]
fn a_closed_queue_is_reported_as_channel_closed() {
    let (work_tx, work_rx) = bounded::<GpuWorkRequest>(1);
    drop(work_rx);

    let breaker = GpuCircuitBreaker::new();
    let heartbeat = GpuHeartbeat::new();
    let deadline = Instant::now() + Duration::from_secs(30);

    let err = send_until_deadline(
        &work_tx,
        helpful_request(),
        deadline,
        &breaker,
        &heartbeat,
        STALL_WINDOW,
        "helpful batch evaluation",
        30,
    )
    .expect_err("a closed channel must fail the send");

    assert!(
        format!("{err:#}").contains("GPU work queue channel closed"),
        "a dropped receiver must be reported as closed, not as a stall or timeout: {err:#}"
    );
    assert!(
        !breaker.is_tripped(),
        "a channel closed outright is not the wedged-GPU signal the breaker exists for"
    );
}

/// A queue with room — or one that drains in time — accepts the request
/// normally, without the deadline, breaker or heartbeat guard interfering.
#[test]
fn a_queue_that_drains_accepts_the_request() {
    let (work_tx, work_rx) = bounded::<GpuWorkRequest>(1);
    let breaker = GpuCircuitBreaker::new();
    let heartbeat = GpuHeartbeat::new();
    let deadline = Instant::now() + Duration::from_secs(30);

    send_until_deadline(
        &work_tx,
        helpful_request(),
        deadline,
        &breaker,
        &heartbeat,
        STALL_WINDOW,
        "helpful batch evaluation",
        30,
    )
    .expect("an empty queue must accept the request immediately");

    assert!(
        work_rx.try_recv().is_ok(),
        "the request must actually be enqueued"
    );
    assert!(!breaker.is_tripped());
}

/// A third submitter queued behind a wedged GPU — the production work loop
/// itself is silent — still returns within the detection cap rather than
/// paying for the full batch timeout, mirroring `wedge_tests::WedgedGpu::spawn`.
#[test]
fn a_third_submitter_behind_a_wedged_gpu_returns_within_the_detection_cap() {
    // Capacity 1, matching every other test in this module: the first request
    // is dequeued (and wedges the device inside `respond()`), which is what
    // frees the one buffer slot the second request then fills — only then is
    // the queue actually full ahead of the third submitter this test is about.
    let (work_tx, work_rx) = bounded::<GpuWorkRequest>(1);
    let heartbeat = Arc::new(GpuHeartbeat::new());
    let evaluator = FakeGpuEvaluator::new(WedgeBehaviour::NeverAnswers, Arc::clone(&heartbeat));
    let probe = evaluator.probe();
    let factory = FakeEvaluatorFactory::new();

    let loop_heartbeat = Arc::clone(&heartbeat);
    let worker: JoinHandle<()> = std::thread::spawn(move || {
        GpuWorkQueue::run_work_loop(evaluator, work_rx, &factory, &loop_heartbeat);
    });

    let breaker = GpuCircuitBreaker::new();

    // First submission reaches (and wedges) the device.
    let (first_tx, first_rx) = bounded::<Result<Vec<HelpfulStats>>>(1);
    let (first_guard, first_liveness) = caller_liveness_pair();
    work_tx
        .send_timeout(
            GpuWorkRequest::HelpfulBatch {
                samples: vec![Arc::new(vec![sample()])],
                response_tx: first_tx,
                budget: GpuTimeBudget::unbounded(),
                liveness: first_liveness,
            },
            Duration::from_secs(1),
        )
        .expect("the first request must enqueue");

    // Wait for the worker to actually dequeue and start evaluating the first
    // request — only then is the one buffer slot free again for the second.
    let dequeued_at = Instant::now();
    while probe.calls() < 1 {
        assert!(
            dequeued_at.elapsed() < HARNESS_WAIT_CAP,
            "the worker never picked up the first request"
        );
        std::thread::sleep(Duration::from_millis(5));
    }

    // Second submission fills the now-empty capacity-1 buffer behind the
    // wedged first one.
    let (second_tx, _second_rx) = bounded::<Result<Vec<HelpfulStats>>>(1);
    let (second_guard, second_liveness) = caller_liveness_pair();
    work_tx
        .send_timeout(
            GpuWorkRequest::HelpfulBatch {
                samples: vec![Arc::new(vec![sample()])],
                response_tx: second_tx,
                budget: GpuTimeBudget::unbounded(),
                liveness: second_liveness,
            },
            Duration::from_secs(1),
        )
        .expect("the second request must enqueue behind the first");

    // The third submitter — this test's subject — must not block inside the
    // production work loop at all once the queue is full behind a wedged
    // device; `send_until_deadline` is what bounds its wait.
    let deadline = Instant::now() + Duration::from_secs(30);
    let started = Instant::now();
    let err = send_until_deadline(
        &work_tx,
        helpful_request(),
        deadline,
        &breaker,
        &heartbeat,
        STALL_WINDOW,
        "helpful batch evaluation",
        30,
    )
    .expect_err("a third submitter behind a wedged GPU must not get through");
    let elapsed = started.elapsed();

    assert!(
        elapsed < DETECTION_CAP,
        "the third submitter must be told within the detection cap, not the 30s \
         deadline (took {elapsed:?})"
    );
    assert!(
        format!("{err:#}").contains("no GPU-thread progress")
            || format!("{err:#}").contains("GPU work queue full"),
        "unexpected error for a wedged queue: {err:#}"
    );

    drop(first_guard);
    drop(second_guard);
    probe.release();
    let _ = work_tx.send_timeout(GpuWorkRequest::Shutdown, Duration::from_secs(1));
    worker.join().expect("the fake GPU worker thread panicked");
    let _ = first_rx.try_recv();
}

/// `GpuFuture::collect` must honour the same deadline the send phase used,
/// not restart a fresh timeout at collection time (Issue #2339's `mod.rs`
/// half of the fix).
#[test]
fn collect_honours_the_shared_deadline_rather_than_restarting_it() {
    // `GpuFuture::breaker` is `&'static`; leak a test-owned breaker exactly as
    // every other wedged-GPU test in this module already does, so nothing here
    // touches the process-wide breaker.
    let breaker: &'static GpuCircuitBreaker = Box::leak(Box::new(GpuCircuitBreaker::new()));
    let (_response_tx, response_rx) = bounded::<Result<Vec<HelpfulStats>>>(1);
    let (caller_guard, _liveness) = caller_liveness_pair();

    // Simulate a send phase that already burned most of the deadline before
    // `collect()` is ever called.
    let short_remaining = Duration::from_millis(150);
    let deadline = Instant::now() + short_remaining;

    let future: GpuFuture<Vec<HelpfulStats>> = GpuFuture {
        response_rx,
        deadline,
        timeout_secs: 30,
        caller_guard,
        breaker,
    };

    let started = Instant::now();
    let err = future
        .collect()
        .expect_err("a response that never arrives before the shared deadline must fail");
    let elapsed = started.elapsed();

    assert!(
        elapsed < Duration::from_secs(2),
        "collect() must respect the deadline already set by the send phase, not a \
         fresh 30s timeout (took {elapsed:?})"
    );
    assert!(
        elapsed >= short_remaining.saturating_sub(Duration::from_millis(50)),
        "collect() must wait out at least the remaining budget (took {elapsed:?})"
    );
    let msg = format!("{err:#}");
    assert!(
        msg.contains("timed out after 30s"),
        "the caller's originally configured timeout must still be reported: {msg}"
    );
}
