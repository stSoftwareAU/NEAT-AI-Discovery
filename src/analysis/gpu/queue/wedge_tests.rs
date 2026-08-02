//! Regression tests for the wedged-GPU defences, driven by the deterministic
//! test double in [`super::fake_evaluator`] (Issue #1935).
//!
//! Each test names the sibling of Issue #1926 it guards, so a red test maps
//! straight back to the fix it protects. Everything here runs on a machine with
//! no GPU: the production work loop, the production bounded wait and the
//! production circuit breaker are all real — only the device is fake.
//!
//! **Order independence.** Every test owns an isolated
//! [`GpuCircuitBreaker`], never the process-wide one, and resets it before
//! returning. Nothing here can leak a tripped breaker into another test, which
//! is what makes the suite safe under CI's `--test-threads=2`.
//!
//! **No long sleeps.** Stall windows and budgets are configured in the
//! hundreds of milliseconds, and every timing assertion carries an explicit
//! upper bound, so a harness bug fails the assertion instead of hanging the
//! suite.

use anyhow::Result;
use crossbeam_channel::{Receiver, Sender, bounded};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use super::fake_evaluator::{
    FakeEvaluatorFactory, FakeGpuEvaluator, FakeGpuProbe, HARNESS_HARD_CAP, WedgeBehaviour,
};
use super::staleness::{CallerGuard, caller_liveness_pair};
use super::submission::{GpuWaitOutcome, resolve_gpu_wait, wait_for_gpu_response};
use super::{GpuWorkQueue, GpuWorkRequest};
use crate::analysis::gpu::breaker::{GpuCircuitBreaker, GpuTripReason};
use crate::analysis::gpu::budget::GpuTimeBudget;
use crate::analysis::gpu::device::GPU_BUFFER_MAP_TIMEOUT_MARGIN_SECS;
use crate::analysis::gpu::heartbeat::GpuHeartbeat;
use crate::analysis::samples::{HelpfulSample, HelpfulStats};
use crate::ffi_types::{DiscoveryErrorKind, classify_anyhow_error};

/// The stall window every wedge test configures — short enough to keep the
/// suite fast, and comfortably wider than the 60ms beat interval the
/// slow-but-progressing test uses, so scheduling jitter on a loaded CI runner
/// cannot be misread as a wedge.
const STALL_WINDOW: Duration = Duration::from_millis(300);

/// Upper bound on any single wedge detection. Detection is meant to cost the
/// stall window plus a poll interval; anything approaching this is a
/// regression, and the assertion fails rather than the suite hanging.
const DETECTION_CAP: Duration = Duration::from_secs(2);

/// Stand-in for a run's whole GPU budget. The point of the #1926 breakdown is
/// that a wedged GPU costs a run *seconds*, not the hours it took to walk the
/// process towards the external 3-hour kill.
const SIMULATED_RUN_BUDGET: Duration = Duration::from_secs(3);

fn sample() -> HelpfulSample {
    HelpfulSample {
        activation: 1.0,
        avg_error: 0.5,
        target_value: None,
        target_activation: None,
    }
}

/// A fake GPU device running the production work loop on its own thread, plus
/// a `GpuWorkQueue` wired to the same channel and an isolated breaker.
struct WedgedGpu {
    queue: GpuWorkQueue,
    breaker: &'static GpuCircuitBreaker,
    heartbeat: Arc<GpuHeartbeat>,
    probe: FakeGpuProbe,
    factory: Arc<FakeEvaluatorFactory>,
    work_tx: Sender<GpuWorkRequest>,
    worker: Option<JoinHandle<()>>,
}

impl WedgedGpu {
    /// Spawn the production loop against a fake device with `behaviour`.
    fn spawn(behaviour: WedgeBehaviour) -> Self {
        let (work_tx, work_rx) = bounded::<GpuWorkRequest>(8);
        let heartbeat = Arc::new(GpuHeartbeat::new());
        let evaluator = FakeGpuEvaluator::new(behaviour, Arc::clone(&heartbeat));
        let probe = evaluator.probe();
        let factory = Arc::new(FakeEvaluatorFactory::new());

        let loop_factory = Arc::clone(&factory);
        let loop_heartbeat = Arc::clone(&heartbeat);
        let worker = std::thread::spawn(move || {
            GpuWorkQueue::run_work_loop(evaluator, work_rx, &*loop_factory, &loop_heartbeat);
        });

        // Drop the exit sender at once so `Drop for GpuWorkQueue` takes the
        // disconnected path instead of sitting out `GPU_SHUTDOWN_TIMEOUT_SECS`.
        let (_, exit_rx) = bounded::<()>(1);
        let breaker: &'static GpuCircuitBreaker = Box::leak(Box::new(GpuCircuitBreaker::new()));
        let queue = GpuWorkQueue {
            work_tx: work_tx.clone(),
            thread_handle: None,
            exit_rx,
            deadline: None,
            breaker,
        };

        Self {
            queue,
            breaker,
            heartbeat,
            probe,
            factory,
            work_tx,
            worker: Some(worker),
        }
    }

    /// Enqueue a helpful batch exactly as a live submitter would, returning the
    /// two handles that caller holds.
    fn submit(&self, budget: GpuTimeBudget) -> (Receiver<Result<Vec<HelpfulStats>>>, CallerGuard) {
        let (response_tx, response_rx) = bounded(1);
        let (guard, liveness) = caller_liveness_pair();
        self.work_tx
            .send_timeout(
                GpuWorkRequest::HelpfulBatch {
                    samples: vec![Arc::new(vec![sample()])],
                    response_tx,
                    budget,
                    liveness,
                },
                Duration::from_secs(1),
            )
            .expect("the test work queue must accept the request");
        (response_rx, guard)
    }

    /// Wait for a response the way production does — bounded by the heartbeat —
    /// and resolve the verdict against this queue's breaker.
    fn await_response(
        &self,
        response_rx: &Receiver<Result<Vec<HelpfulStats>>>,
        timeout: Duration,
    ) -> Result<Vec<HelpfulStats>> {
        let outcome = wait_for_gpu_response(response_rx, timeout, &self.heartbeat, STALL_WINDOW);
        resolve_gpu_wait(
            outcome,
            self.breaker,
            "helpful batch evaluation",
            timeout.as_secs(),
        )
    }

    /// Release the fake device, shut the loop down and join the worker thread.
    fn stop(mut self) {
        self.probe.release();
        let _ = self
            .work_tx
            .send_timeout(GpuWorkRequest::Shutdown, Duration::from_secs(1));
        if let Some(worker) = self.worker.take() {
            worker.join().expect("the fake GPU worker thread panicked");
        }
        // Leave no tripped breaker behind, mirroring `reset_gpu_breaker()`.
        self.breaker.reset();
    }
}

/// A bounded budget with roughly `remaining` left, derived the way
/// `submit_helpful_batch` derives one: caller timeout less the safety margin.
/// Returns the budget and the caller timeout it was derived from.
fn budget_with(remaining: Duration) -> (GpuTimeBudget, Duration) {
    let margin = Duration::from_secs(GPU_BUFFER_MAP_TIMEOUT_MARGIN_SECS);
    let elapsed = Duration::from_secs(2);
    let caller_timeout = margin + elapsed + remaining;
    let budget = GpuTimeBudget::from_caller_timeout_at(Instant::now() - elapsed, caller_timeout);
    (budget, caller_timeout)
}

// =============================================================================
// Issue #1928 — the GPU thread cannot outlive a submission's timeout budget
// =============================================================================

/// Issue #1928: a wedged device holding a request must give up when the
/// request's own budget runs out, not when its fixed inner constants do.
///
/// Without the deadline-derived budget the request would carry no caller
/// deadline, the fake would block to [`HARNESS_HARD_CAP`], and the elapsed
/// assertion below would fail — which is the whole point: in production that
/// same gap is the GPU thread outliving its caller and being abandoned.
#[test]
fn a_wedged_request_cannot_outlive_its_time_budget() {
    let gpu = WedgedGpu::spawn(WedgeBehaviour::WedgesUntilBudgetExpires);
    let (budget, caller_timeout) = budget_with(Duration::from_secs(1));

    let started = Instant::now();
    let (response_rx, guard) = gpu.submit(budget);
    let response = response_rx
        .recv_timeout(HARNESS_HARD_CAP + Duration::from_secs(1))
        .expect("the worker must answer, not go silent");
    let elapsed = started.elapsed();
    drop(guard);

    assert!(
        response.is_err(),
        "an exhausted budget must surface as a real error, not a fabricated success"
    );
    assert!(
        elapsed < caller_timeout,
        "the worker must give up before its caller does ({elapsed:?} >= {caller_timeout:?})"
    );
    assert!(
        elapsed < DETECTION_CAP,
        "the worker must stop at the request's budget, not at a fixed constant \
         (took {elapsed:?})"
    );

    let observed = gpu.probe.observations();
    assert_eq!(observed.len(), 1, "exactly one evaluation was attempted");
    assert!(
        observed[0].budget_bounded,
        "the request must reach the device carrying the caller's budget"
    );
    assert!(
        observed[0].budget_remaining < caller_timeout,
        "the inner budget {:?} must be shorter than the caller timeout {caller_timeout:?}",
        observed[0].budget_remaining
    );

    gpu.stop();
}

// =============================================================================
// Issue #1929 — a request whose receiver was dropped is never executed
// =============================================================================

/// Issue #1929: a caller that has already given up must not cost the wedged GPU
/// a single evaluation, and the live request queued behind it must still be
/// served.
#[test]
fn an_abandoned_request_never_reaches_the_wedged_gpu() {
    let gpu = WedgedGpu::spawn(WedgeBehaviour::Completes);

    let (abandoned_rx, abandoned_guard) = gpu.submit(GpuTimeBudget::unbounded());
    drop(abandoned_guard); // the submitter timed out before the worker got to it
    let (live_rx, live_guard) = gpu.submit(GpuTimeBudget::unbounded());

    let live = live_rx
        .recv_timeout(DETECTION_CAP)
        .expect("the live request must still be served")
        .expect("a healthy device answers successfully");
    drop(live_guard);

    assert_eq!(live.len(), 1, "the live caller receives its stats");
    assert_eq!(
        gpu.probe.calls(),
        1,
        "only the live request may reach the device"
    );
    assert!(
        abandoned_rx.try_recv().is_err(),
        "nothing is sent for a request whose caller has gone"
    );

    gpu.stop();
}

// =============================================================================
// Issue #1933 — stall detection fires for "no progress", not for "slow"
// =============================================================================

/// Issue #1933: a device that publishes nothing is declared wedged within the
/// stall window. Issue #1930: that verdict trips the breaker. Issue #1932: the
/// error is typed, so the host is told not to retry.
#[test]
fn a_silent_gpu_is_declared_wedged_within_the_stall_window() {
    let gpu = WedgedGpu::spawn(WedgeBehaviour::NeverAnswers);
    let (response_rx, guard) = gpu.submit(GpuTimeBudget::unbounded());

    let started = Instant::now();
    let err = gpu
        .await_response(&response_rx, Duration::from_secs(30))
        .expect_err("a silent GPU must not be waited on for the whole batch timeout");
    let elapsed = started.elapsed();
    drop(guard);

    assert!(
        elapsed >= STALL_WINDOW,
        "the guard must not fire before the window elapses (fired after {elapsed:?})"
    );
    assert!(
        elapsed < DETECTION_CAP,
        "detection must cost the stall window, not the batch timeout (took {elapsed:?})"
    );
    assert_eq!(
        gpu.breaker.trip_reason(),
        Some(GpuTripReason::HeartbeatStall),
        "the wedge verdict must trip the breaker"
    );
    assert_eq!(
        classify_anyhow_error(&err),
        DiscoveryErrorKind::GpuWedged,
        "a wedged GPU must not be reported as a retryable timeout"
    );
    assert!(!DiscoveryErrorKind::GpuWedged.is_retryable());

    gpu.stop();
}

/// Issue #1933: a device that is merely slow — silent, but answering inside the
/// window — must never be flagged.
#[test]
fn a_device_that_answers_inside_the_window_is_not_flagged() {
    let gpu = WedgedGpu::spawn(WedgeBehaviour::CompletesAfter(Duration::from_millis(60)));
    let (response_rx, guard) = gpu.submit(GpuTimeBudget::unbounded());

    let stats = gpu
        .await_response(&response_rx, Duration::from_secs(30))
        .expect("a device answering inside the window is healthy");
    drop(guard);

    assert_eq!(stats.len(), 1);
    assert!(
        !gpu.breaker.is_tripped(),
        "a healthy device must not trip the breaker"
    );

    gpu.stop();
}

/// Issue #1933: a long evaluation that keeps publishing progress resets the
/// stall clock, so it is not flagged even though it runs well past the window.
#[test]
fn a_slow_but_progressing_gpu_is_never_flagged_as_wedged() {
    let gpu = WedgedGpu::spawn(WedgeBehaviour::BeatsThenCompletes {
        interval: Duration::from_millis(60),
        beats: 5,
    });
    let (response_rx, guard) = gpu.submit(GpuTimeBudget::unbounded());

    let started = Instant::now();
    let stats = gpu
        .await_response(&response_rx, Duration::from_secs(30))
        .expect("a slow but advancing GPU must not be flagged");
    let elapsed = started.elapsed();
    drop(guard);

    assert_eq!(stats.len(), 1);
    assert!(
        elapsed > STALL_WINDOW,
        "the evaluation must outlast the stall window for this to prove anything \
         (took {elapsed:?})"
    );
    assert!(
        !gpu.breaker.is_tripped(),
        "progress must keep resetting the stall clock"
    );

    gpu.stop();
}

/// Issue #1933: the absolute batch timeout stays the backstop. A device that
/// advertises progress forever but never completes still ends the caller's wait
/// — as a plain timeout, not as a stall.
#[test]
fn an_endlessly_progressing_gpu_still_ends_at_the_absolute_timeout() {
    let gpu = WedgedGpu::spawn(WedgeBehaviour::BeatsWithoutCompleting {
        interval: Duration::from_millis(30),
    });
    let (response_rx, guard) = gpu.submit(GpuTimeBudget::unbounded());
    let timeout = Duration::from_millis(400);

    let started = Instant::now();
    let outcome = wait_for_gpu_response(&response_rx, timeout, &gpu.heartbeat, STALL_WINDOW);
    let elapsed = started.elapsed();

    assert!(
        matches!(outcome, GpuWaitOutcome::TimedOut),
        "a progressing device must end at the backstop, not as a stall: {outcome:?}"
    );
    assert!(
        elapsed >= timeout && elapsed < DETECTION_CAP,
        "the backstop must wait out the timeout and no longer (took {elapsed:?})"
    );

    let err = resolve_gpu_wait(outcome, gpu.breaker, "helpful batch evaluation", 300)
        .expect_err("a timed-out wait must fail");
    drop(guard);
    assert_eq!(
        gpu.breaker.trip_reason(),
        Some(GpuTripReason::BatchTimeout),
        "the backstop trips for its own reason, not as a heartbeat stall"
    );
    assert_eq!(classify_anyhow_error(&err), DiscoveryErrorKind::GpuWedged);

    gpu.stop();
}

// =============================================================================
// Issue #1930 — the first wedge stops all further GPU work
// =============================================================================

/// Issue #1930: once the first wedge has tripped the breaker, no later
/// submission may reach the GPU, start another wait, or rebuild the device —
/// which is what stops a run spawning a second GPU thread against dead
/// hardware. (`GpuWorkQueue::new()`'s own refusal is asserted end-to-end in
/// `tests/issue_1935_wedged_gpu_harness.rs`.)
#[test]
fn the_first_wedge_stops_every_later_submission() {
    let gpu = WedgedGpu::spawn(WedgeBehaviour::NeverAnswers);
    let (response_rx, guard) = gpu.submit(GpuTimeBudget::unbounded());
    gpu.await_response(&response_rx, Duration::from_secs(30))
        .expect_err("the first submission wedges");
    drop(guard);

    let started = Instant::now();
    for attempt in 1..=5 {
        let err = gpu
            .queue
            .evaluate_helpful_batch(vec![Arc::new(vec![sample()])], &None)
            .expect_err("every later submission must be refused by the breaker");
        assert!(
            format!("{err:#}").contains("GPU circuit breaker tripped"),
            "attempt {attempt} must fail with the breaker error, got: {err:#}"
        );
    }
    let refusals = started.elapsed();

    assert!(
        refusals < Duration::from_millis(500),
        "refusals must be immediate, five took {refusals:?}"
    );
    assert_eq!(
        gpu.probe.calls(),
        1,
        "no work may reach the device after the wedge"
    );
    assert_eq!(
        gpu.factory.create_count(),
        0,
        "a wedged device must never be rebuilt"
    );

    gpu.stop();
}

/// The whole sequence — wedge, detect, trip, then keep the run going — must fit
/// inside a run budget measured in seconds. Before the #1926 fixes each pass
/// cost a fresh 60–300s wait plus an abandoned GPU thread, which is how a
/// wedged device walked the process towards its external 3-hour kill.
#[test]
fn the_whole_wedge_sequence_fits_inside_a_simulated_run_budget() {
    let started = Instant::now();

    let gpu = WedgedGpu::spawn(WedgeBehaviour::NeverAnswers);
    let (response_rx, guard) = gpu.submit(GpuTimeBudget::unbounded());
    gpu.await_response(&response_rx, Duration::from_secs(300))
        .expect_err("the wedged submission fails");
    drop(guard);

    // Every later analysis pass in the run.
    for _ in 0..20 {
        assert!(
            gpu.queue
                .evaluate_helpful_batch(vec![Arc::new(vec![sample()])], &None)
                .is_err()
        );
    }

    let elapsed = started.elapsed();
    gpu.stop();

    assert!(
        elapsed < SIMULATED_RUN_BUDGET,
        "a wedged GPU must cost the run seconds, not minutes (took {elapsed:?})"
    );
}
