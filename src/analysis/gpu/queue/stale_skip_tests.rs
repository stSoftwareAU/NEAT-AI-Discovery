//! Loop-level tests for stale GPU request skipping (Issue #1929).
//!
//! These drive `GpuWorkQueue::run_work_loop` with a counting stub evaluator, so
//! they assert what the loop does *not* do — never invoke the analyser for an
//! abandoned request, never re-initialise the device for one — on machines with
//! no GPU at all. They live in the crate rather than `tests/gpu/` because
//! `GpuWorkRequest` and the loop are `pub(crate)`.

use anyhow::{Result, anyhow};
use crossbeam_channel::{Receiver, Sender, bounded};
use serial_test::serial;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use super::executor::{EvaluatorFactory, RequestEvaluator};
use super::staleness::{CallerGuard, caller_liveness_pair};
use super::{GpuWorkQueue, GpuWorkRequest};
use crate::analysis::gpu::budget::GpuTimeBudget;
use crate::analysis::gpu::heartbeat::GpuHeartbeat;
use crate::analysis::samples::{
    HarmfulStats, HelpfulSample, HelpfulStats, ReluOrientation, ReluStats,
};
use crate::observability::global_gpu_metrics;

/// What the stub evaluator returns from every evaluation.
#[derive(Clone, Copy, PartialEq, Eq)]
enum StubOutcome {
    Succeed,
    DeviceLost,
}

/// Stub evaluator that counts calls instead of touching a GPU.
///
/// `abandon_on_call` models a caller that times out mid-evaluation: the guard
/// it holds is dropped the first time the loop evaluates anything.
struct CountingEvaluator {
    calls: Arc<AtomicUsize>,
    outcome: StubOutcome,
    abandon_on_call: Mutex<Option<CallerGuard>>,
}

impl CountingEvaluator {
    fn new(outcome: StubOutcome) -> Self {
        Self {
            calls: Arc::new(AtomicUsize::new(0)),
            outcome,
            abandon_on_call: Mutex::new(None),
        }
    }

    /// Hand the evaluator the caller's guard to drop on its first evaluation.
    fn abandoning(outcome: StubOutcome, guard: CallerGuard) -> Self {
        let mut evaluator = Self::new(outcome);
        evaluator.abandon_on_call = Mutex::new(Some(guard));
        evaluator
    }

    /// A handle to the call counter that survives moving the evaluator into
    /// the loop.
    fn counter(&self) -> Arc<AtomicUsize> {
        Arc::clone(&self.calls)
    }

    /// Record one evaluation and produce the configured outcome.
    fn record<T>(&self, value: T) -> Result<T> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        // Drop the caller's guard, simulating a mid-flight timeout.
        drop(
            self.abandon_on_call
                .lock()
                .expect("stub evaluator mutex poisoned")
                .take(),
        );
        match self.outcome {
            StubOutcome::Succeed => Ok(value),
            StubOutcome::DeviceLost => Err(anyhow!("Device is lost")),
        }
    }
}

impl RequestEvaluator for CountingEvaluator {
    fn batch_size(&self) -> usize {
        1024
    }

    fn evaluate_helpful_batch(
        &self,
        _samples_batch: &[&[HelpfulSample]],
        _budget: GpuTimeBudget,
    ) -> Result<Vec<HelpfulStats>> {
        self.record(Vec::new())
    }

    fn evaluate_harmful_batch(
        &self,
        _samples_batch: &[(&[HelpfulSample], f32)],
        _budget: GpuTimeBudget,
    ) -> Result<Vec<HarmfulStats>> {
        self.record(Vec::new())
    }

    fn evaluate_relu(
        &self,
        _samples: &[HelpfulSample],
        _threshold: f32,
        _budget: GpuTimeBudget,
    ) -> Result<(ReluStats, ReluStats, f32)> {
        self.record((
            ReluStats::new(ReluOrientation::Positive),
            ReluStats::new(ReluOrientation::Negative),
            0.0,
        ))
    }

    fn evaluate_activation(
        &self,
        _samples: &[HelpfulSample],
        _activation_type: u32,
        _orientation: f32,
        _scale: f32,
        _budget: GpuTimeBudget,
    ) -> Result<(f32, f32, f32, u32)> {
        self.record((0.0, 0.0, 0.0, 0))
    }

    fn evaluate_activations_batched(
        &self,
        _samples: &[HelpfulSample],
        _activation_configs: &[(u32, f32, f32)],
        _budget: GpuTimeBudget,
    ) -> Result<Vec<(f32, f32, f32, u32)>> {
        self.record(Vec::new())
    }
}

/// Factory that counts re-initialisation attempts and always fails, so a
/// recovery attempt that *is* made is unmistakable in the count.
struct CountingFactory {
    creates: AtomicUsize,
}

impl CountingFactory {
    fn new() -> Self {
        Self {
            creates: AtomicUsize::new(0),
        }
    }

    fn create_count(&self) -> usize {
        self.creates.load(Ordering::Relaxed)
    }
}

impl EvaluatorFactory<CountingEvaluator> for CountingFactory {
    fn create(&self, _batch_size_override: Option<usize>) -> Result<CountingEvaluator> {
        self.creates.fetch_add(1, Ordering::Relaxed);
        Err(anyhow!("stub factory never recovers"))
    }
}

/// A submitted request together with the two handles a live caller holds: its
/// response receiver and its liveness guard.
struct SubmittedRequest {
    request: GpuWorkRequest,
    response_rx: Receiver<Result<Vec<HelpfulStats>>>,
    guard: CallerGuard,
}

/// Build a helpful-batch request as a live submitter would.
fn helpful_request() -> SubmittedRequest {
    helpful_request_with_budget(GpuTimeBudget::unbounded())
}

fn helpful_request_with_budget(budget: GpuTimeBudget) -> SubmittedRequest {
    let (response_tx, response_rx) = bounded(1);
    let (guard, liveness) = caller_liveness_pair();
    SubmittedRequest {
        request: GpuWorkRequest::HelpfulBatch {
            samples: vec![],
            response_tx,
            budget,
            liveness,
        },
        response_rx,
        guard,
    }
}

/// Enqueue a request, failing loudly if the bounded channel cannot take it.
fn enqueue(work_tx: &Sender<GpuWorkRequest>, request: GpuWorkRequest) {
    work_tx
        .send_timeout(request, Duration::from_secs(1))
        .expect("test work queue should accept the request");
}

/// Acceptance criterion 1: a request whose caller has timed out is dropped
/// without invoking the analyser, and the loop goes on to serve the next one.
#[test]
#[serial(gpu_stale_skip)]
fn stale_request_skipped_without_analysis() {
    let (work_tx, work_rx) = bounded::<GpuWorkRequest>(4);

    let stale = helpful_request();
    drop(stale.guard); // the caller timed out before the worker dequeued its work
    let live = helpful_request();

    enqueue(&work_tx, stale.request);
    enqueue(&work_tx, live.request);
    enqueue(&work_tx, GpuWorkRequest::Shutdown);

    let evaluator = CountingEvaluator::new(StubOutcome::Succeed);
    let calls = evaluator.counter();
    let factory = CountingFactory::new();
    GpuWorkQueue::run_work_loop(evaluator, work_rx, &factory, &GpuHeartbeat::new());

    assert_eq!(
        calls.load(Ordering::Relaxed),
        1,
        "analyser must be invoked once — for the live request only"
    );
    assert!(
        live.response_rx.try_recv().is_ok(),
        "the live request queued behind the stale one must still be served"
    );
    assert_eq!(
        factory.create_count(),
        0,
        "no device recovery should be attempted on the happy path"
    );
}

/// Acceptance criterion 2: a device-lost retry is not attempted once the
/// caller's receiver is gone — zero re-initialisations instead of
/// `DEFAULT_GPU_RETRY_LIMIT`.
#[test]
#[serial(gpu_stale_skip)]
fn device_lost_retry_aborts_on_dead_receiver() {
    let (work_tx, work_rx) = bounded::<GpuWorkRequest>(2);

    let submitted = helpful_request();
    enqueue(&work_tx, submitted.request);
    enqueue(&work_tx, GpuWorkRequest::Shutdown);

    // The caller is alive at dequeue time and goes away during evaluation,
    // exactly as a submitter timing out mid-flight would.
    let evaluator = CountingEvaluator::abandoning(StubOutcome::DeviceLost, submitted.guard);
    let calls = evaluator.counter();
    let factory = CountingFactory::new();
    GpuWorkQueue::run_work_loop(evaluator, work_rx, &factory, &GpuHeartbeat::new());

    assert_eq!(
        calls.load(Ordering::Relaxed),
        1,
        "the failed request must not be re-evaluated for an absent caller"
    );
    assert_eq!(
        factory.create_count(),
        0,
        "no GpuAnalyzer re-initialisation should be attempted once the caller is gone"
    );
}

/// A device-lost failure whose caller is still waiting *does* exhaust the retry
/// budget — the abort above must be caused by the dead receiver, not by the
/// skip check swallowing every recovery.
#[test]
#[serial(gpu_stale_skip)]
fn device_lost_retry_still_runs_for_live_caller() {
    let (work_tx, work_rx) = bounded::<GpuWorkRequest>(2);

    let submitted = helpful_request();
    enqueue(&work_tx, submitted.request);
    enqueue(&work_tx, GpuWorkRequest::Shutdown);

    let evaluator = CountingEvaluator::new(StubOutcome::DeviceLost);
    let factory = CountingFactory::new();
    GpuWorkQueue::run_work_loop(evaluator, work_rx, &factory, &GpuHeartbeat::new());

    assert_eq!(
        factory.create_count(),
        crate::config::gpu_retry_limit() as usize,
        "a live caller should still get the full recovery budget"
    );
    let response = submitted
        .response_rx
        .try_recv()
        .expect("the caller must be told recovery failed");
    assert!(
        response.is_err(),
        "exhausted recovery must surface an error"
    );
}

/// Acceptance criterion 4: skipped requests are counted in the GPU metrics.
#[test]
#[serial(gpu_stale_skip)]
fn stale_skip_counted_in_metrics() {
    let before = global_gpu_metrics().stale_skipped();

    let (work_tx, work_rx) = bounded::<GpuWorkRequest>(4);
    for _ in 0..2 {
        let submitted = helpful_request();
        drop(submitted.guard);
        enqueue(&work_tx, submitted.request);
    }
    enqueue(&work_tx, GpuWorkRequest::Shutdown);

    let evaluator = CountingEvaluator::new(StubOutcome::Succeed);
    let calls = evaluator.counter();
    let factory = CountingFactory::new();
    GpuWorkQueue::run_work_loop(evaluator, work_rx, &factory, &GpuHeartbeat::new());

    assert_eq!(
        calls.load(Ordering::Relaxed),
        0,
        "no request should have been run"
    );
    assert_eq!(
        global_gpu_metrics().stale_skipped() - before,
        2,
        "the skip counter must rise by exactly the number of skipped requests"
    );
}

/// A request whose own budget expired in the queue is abandoned without
/// evaluation, and its still-waiting caller is told loudly rather than being
/// left to time out.
#[test]
#[serial(gpu_stale_skip)]
fn expired_budget_request_fails_loudly_without_analysis() {
    let (work_tx, work_rx) = bounded::<GpuWorkRequest>(2);

    let expired = GpuTimeBudget::from_caller_timeout_at(
        Instant::now() - Duration::from_secs(600),
        Duration::from_secs(60),
    );
    let submitted = helpful_request_with_budget(expired);
    enqueue(&work_tx, submitted.request);
    enqueue(&work_tx, GpuWorkRequest::Shutdown);

    let evaluator = CountingEvaluator::new(StubOutcome::Succeed);
    let calls = evaluator.counter();
    let factory = CountingFactory::new();
    GpuWorkQueue::run_work_loop(evaluator, work_rx, &factory, &GpuHeartbeat::new());

    assert_eq!(
        calls.load(Ordering::Relaxed),
        0,
        "an expired request must not reach the analyser"
    );
    let response = submitted
        .response_rx
        .try_recv()
        .expect("caller must receive a response, not silence");
    let err = response.expect_err("an expired budget must surface as an error");
    assert!(
        err.to_string().contains("budget expired"),
        "error should name the expired budget, got: {err}"
    );
}
