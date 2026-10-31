//! Loop-level tests for classifying device loss by source (Issue #2363).
//!
//! wgpu 30 reports a lost device as an opaque error, such as "Error occurred
//! when trying to async map a buffer", which none of the text matchers in
//! `recovery.rs` recognise. The analyser's device-lost callback sets a flag, and
//! `execute_request` routes a failure to recovery when that flag is set. These
//! tests drive `GpuWorkQueue::run_work_loop` with the deterministic fake device
//! so they run on machines with no GPU at all.

use crossbeam_channel::{Sender, bounded};
use serial_test::serial;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use super::executor::{EvaluatorFactory, RequestEvaluator};
use super::fake_evaluator::{FakeEvaluatorFactory, FakeGpuEvaluator, WedgeBehaviour};
use super::{GpuWorkQueue, GpuWorkRequest};
use crate::analysis::gpu::analyzer::GpuAnalyzer;
use crate::analysis::gpu::budget::GpuTimeBudget;
use crate::analysis::gpu::heartbeat::GpuHeartbeat;
use crate::analysis::gpu::none_field_tests::all_none_analyzer;
use crate::analysis::samples::HelpfulSample;
use crate::analysis::samples::{HarmfulStats, HelpfulStats, ReluStats};
use anyhow::Result;
use crossbeam_channel::Receiver;

use super::staleness::caller_liveness_pair;

type ActivationResponse = Receiver<Result<(f32, f32, f32, u32)>>;
type ActivationBatchResponse = Receiver<Result<Vec<(f32, f32, f32, u32)>>>;

/// Enqueue a request, failing loudly if the bounded channel cannot take it.
fn enqueue(work_tx: &Sender<GpuWorkRequest>, request: GpuWorkRequest) {
    work_tx
        .send_timeout(request, Duration::from_secs(1))
        .expect("test work queue should accept the request");
}

/// Build a helpful-batch request, returning its response receiver.
fn helpful_request() -> (GpuWorkRequest, Receiver<Result<Vec<HelpfulStats>>>) {
    let (response_tx, response_rx) = bounded(1);
    let (guard, liveness) = caller_liveness_pair();
    // The caller stays live for the whole test, so no request is skipped as stale.
    std::mem::forget(guard);
    let request = GpuWorkRequest::HelpfulBatch {
        samples: vec![Arc::new(vec![HelpfulSample {
            activation: 1.0,
            avg_error: 0.5,
            target_value: None,
            target_activation: None,
        }])],
        response_tx,
        budget: GpuTimeBudget::unbounded(),
        liveness,
    };
    (request, response_rx)
}

/// Response type of a `ReLU` evaluation request.
type ReluResponse = Result<(ReluStats, ReluStats, f32)>;

/// Build a `ReLU` request, returning its response receiver.
fn relu_request() -> (GpuWorkRequest, Receiver<ReluResponse>) {
    let (response_tx, response_rx) = bounded(1);
    let (guard, liveness) = caller_liveness_pair();
    std::mem::forget(guard);
    let request = GpuWorkRequest::ReluEval {
        samples: vec![HelpfulSample {
            activation: 1.0,
            avg_error: 0.5,
            target_value: None,
            target_activation: None,
        }],
        threshold: 0.0,
        response_tx,
        budget: GpuTimeBudget::unbounded(),
        liveness,
    };
    (request, response_rx)
}

/// Build a harmful-batch request, returning its response receiver.
fn harmful_request() -> (GpuWorkRequest, Receiver<Result<Vec<HarmfulStats>>>) {
    let (response_tx, response_rx) = bounded(1);
    let (guard, liveness) = caller_liveness_pair();
    std::mem::forget(guard);
    let request = GpuWorkRequest::HarmfulBatch {
        samples_with_weights: vec![(
            Arc::new(vec![HelpfulSample {
                activation: 1.0,
                avg_error: 0.5,
                target_value: None,
                target_activation: None,
            }]),
            1.0,
        )],
        response_tx,
        budget: GpuTimeBudget::unbounded(),
        liveness,
    };
    (request, response_rx)
}

/// Build a general activation evaluation request, returning its response receiver.
fn activation_request() -> (GpuWorkRequest, ActivationResponse) {
    let (response_tx, response_rx) = bounded(1);
    let (guard, liveness) = caller_liveness_pair();
    std::mem::forget(guard);
    let request = GpuWorkRequest::ActivationEval {
        samples: vec![HelpfulSample {
            activation: 1.0,
            avg_error: 0.5,
            target_value: None,
            target_activation: None,
        }],
        activation_type: 0,
        orientation: 1.0,
        scale: 1.0,
        response_tx,
        budget: GpuTimeBudget::unbounded(),
        liveness,
    };
    (request, response_rx)
}

/// Build a batched activation evaluation request, returning its response receiver.
fn activation_batch_request() -> (GpuWorkRequest, ActivationBatchResponse) {
    let (response_tx, response_rx) = bounded(1);
    let (guard, liveness) = caller_liveness_pair();
    std::mem::forget(guard);
    let request = GpuWorkRequest::ActivationBatchEval {
        samples: vec![HelpfulSample {
            activation: 1.0,
            avg_error: 0.5,
            target_value: None,
            target_activation: None,
        }],
        activation_configs: vec![(0, 1.0, 1.0)],
        response_tx,
        budget: GpuTimeBudget::unbounded(),
        liveness,
    };
    (request, response_rx)
}

struct FailingAnalyzerFactory(AtomicUsize);

impl EvaluatorFactory<GpuAnalyzer> for FailingAnalyzerFactory {
    fn create(&self, _batch_size_override: Option<usize>) -> Result<GpuAnalyzer> {
        self.0.fetch_add(1, Ordering::SeqCst);
        anyhow::bail!("test device initialisation failed")
    }
}

#[test]
#[serial(gpu_stale_skip)]
fn real_analyzer_flag_routes_failure_to_recovery() {
    for lost in [false, true] {
        let (work_tx, work_rx) = bounded::<GpuWorkRequest>(4);
        let (request, response_rx) = helpful_request();
        enqueue(&work_tx, request);
        enqueue(&work_tx, GpuWorkRequest::Shutdown);

        let analyzer = all_none_analyzer();
        analyzer.device_lost.store(lost, Ordering::SeqCst);
        assert_eq!(RequestEvaluator::device_lost(&analyzer), lost);
        let factory = FailingAnalyzerFactory(AtomicUsize::new(0));
        let heartbeat = GpuHeartbeat::new();
        GpuWorkQueue::run_work_loop(analyzer, work_rx, &factory, &heartbeat);

        let attempts = factory.0.load(Ordering::SeqCst);
        let response = response_rx
            .try_recv()
            .expect("caller must receive a result");
        let err = response.expect_err("missing device must fail");
        let text = format!("{err:#}");
        if lost {
            assert_eq!(attempts, crate::config::gpu_retry_limit() as usize);
            assert!(
                text.contains("GPU device lost and recovery failed"),
                "{text}"
            );
        } else {
            assert_eq!(attempts, 0);
            assert_eq!(text, "GPU device unavailable for error-scoped evaluation");
        }
    }
}

/// Issue #2363: an opaque buffer-mapping error from a device whose
/// device-lost callback has fired must enter recovery, even though its text
/// matches no device-loss pattern. The fake factory never recovers, so the loop
/// retries up to the configured limit and then reports the recovery failure.
#[test]
#[serial(gpu_stale_skip)]
fn flagged_opaque_map_error_enters_recovery() {
    let (work_tx, work_rx) = bounded::<GpuWorkRequest>(4);
    let (request, response_rx) = helpful_request();
    enqueue(&work_tx, request);
    enqueue(&work_tx, GpuWorkRequest::Shutdown);

    let heartbeat = Arc::new(GpuHeartbeat::new());
    let evaluator = FakeGpuEvaluator::new(
        WedgeBehaviour::OpaqueMapErrorAfterDeviceLoss,
        Arc::clone(&heartbeat),
    );
    let factory = FakeEvaluatorFactory::new();
    GpuWorkQueue::run_work_loop(evaluator, work_rx, &factory, &heartbeat);

    assert_eq!(
        factory.create_count(),
        crate::config::gpu_retry_limit() as usize,
        "a flagged device loss must use the full recovery budget"
    );
    let response = response_rx
        .try_recv()
        .expect("the caller must be told recovery failed");
    let err = response.expect_err("exhausted recovery must surface an error");
    let text = format!("{err:#}");
    assert!(
        text.contains("GPU device lost and recovery failed"),
        "expected the recovery-exhausted message, got: {text}"
    );
}

/// Issue #2363: without the device-lost flag, the same opaque error is not a
/// device loss. It must reach the caller unchanged and never trigger a
/// re-initialisation.
#[test]
#[serial(gpu_stale_skip)]
fn unflagged_opaque_map_error_reaches_caller_unchanged() {
    let (work_tx, work_rx) = bounded::<GpuWorkRequest>(4);
    let (request, response_rx) = helpful_request();
    enqueue(&work_tx, request);
    enqueue(&work_tx, GpuWorkRequest::Shutdown);

    let heartbeat = Arc::new(GpuHeartbeat::new());
    let evaluator = FakeGpuEvaluator::new(
        WedgeBehaviour::OpaqueMapErrorHealthy,
        Arc::clone(&heartbeat),
    );
    let factory = FakeEvaluatorFactory::new();
    GpuWorkQueue::run_work_loop(evaluator, work_rx, &factory, &heartbeat);

    assert_eq!(
        factory.create_count(),
        0,
        "an unflagged error must not re-initialise the device"
    );
    let response = response_rx
        .try_recv()
        .expect("the caller must receive the evaluation result");
    let err = response.expect_err("the evaluation failed, so the caller gets an error");
    let text = format!("{err:#}");
    assert!(
        text.contains("async map a buffer"),
        "the original error text must reach the caller, got: {text}"
    );
    assert!(
        !text.contains("recovery failed"),
        "an unflagged error must not be reported as a recovery failure, got: {text}"
    );
}

/// Issue #2363: the flag routes every evaluation kind to recovery, not only the
/// helpful batch path. Here a `ReLU` evaluation whose error text is opaque enters
/// recovery when the device-lost flag is set.
#[test]
#[serial(gpu_stale_skip)]
fn flagged_non_matching_relu_error_enters_recovery() {
    let (work_tx, work_rx) = bounded::<GpuWorkRequest>(4);
    let (request, response_rx) = relu_request();
    enqueue(&work_tx, request);
    enqueue(&work_tx, GpuWorkRequest::Shutdown);

    let heartbeat = Arc::new(GpuHeartbeat::new());
    let evaluator = FakeGpuEvaluator::new(
        WedgeBehaviour::OpaqueMapErrorAfterDeviceLoss,
        Arc::clone(&heartbeat),
    );
    let factory = FakeEvaluatorFactory::new();
    GpuWorkQueue::run_work_loop(evaluator, work_rx, &factory, &heartbeat);

    assert_eq!(
        factory.create_count(),
        crate::config::gpu_retry_limit() as usize,
        "a flagged ReLU failure must enter recovery"
    );
    let response = response_rx
        .try_recv()
        .expect("the caller must be told recovery failed");
    // `ReluStats` does not implement `Debug`, so match rather than `expect_err`.
    let err = match response {
        Ok(_) => panic!("exhausted recovery must surface an error"),
        Err(err) => err,
    };
    assert!(format!("{err:#}").contains("GPU device lost and recovery failed"));
}

/// Issue #2363: an opaque error on the harmful-batch arm enters recovery when the device-lost flag is set.
#[test]
#[serial(gpu_stale_skip)]
fn flagged_opaque_harmful_error_enters_recovery() {
    let (work_tx, work_rx) = bounded::<GpuWorkRequest>(4);
    let (request, response_rx) = harmful_request();
    enqueue(&work_tx, request);
    enqueue(&work_tx, GpuWorkRequest::Shutdown);

    let heartbeat = Arc::new(GpuHeartbeat::new());
    let evaluator = FakeGpuEvaluator::new(
        WedgeBehaviour::OpaqueMapErrorAfterDeviceLoss,
        Arc::clone(&heartbeat),
    );
    let factory = FakeEvaluatorFactory::new();
    GpuWorkQueue::run_work_loop(evaluator, work_rx, &factory, &heartbeat);

    assert_eq!(
        factory.create_count(),
        crate::config::gpu_retry_limit() as usize,
        "a flagged harmful-batch device loss must enter recovery"
    );
    let response = response_rx
        .try_recv()
        .expect("the caller must be told recovery failed");
    // `HarmfulStats` does not implement `Debug`, so match rather than `expect_err`.
    let err = match response {
        Ok(_) => panic!("exhausted recovery must surface an error"),
        Err(err) => err,
    };
    assert!(format!("{err:#}").contains("GPU device lost and recovery failed"));
}

/// Issue #2363: an opaque error on the general activation arm enters recovery when the device-lost flag is set.
#[test]
#[serial(gpu_stale_skip)]
fn flagged_opaque_activation_error_enters_recovery() {
    let (work_tx, work_rx) = bounded::<GpuWorkRequest>(4);
    let (request, response_rx) = activation_request();
    enqueue(&work_tx, request);
    enqueue(&work_tx, GpuWorkRequest::Shutdown);

    let heartbeat = Arc::new(GpuHeartbeat::new());
    let evaluator = FakeGpuEvaluator::new(
        WedgeBehaviour::OpaqueMapErrorAfterDeviceLoss,
        Arc::clone(&heartbeat),
    );
    let factory = FakeEvaluatorFactory::new();
    GpuWorkQueue::run_work_loop(evaluator, work_rx, &factory, &heartbeat);

    assert_eq!(
        factory.create_count(),
        crate::config::gpu_retry_limit() as usize,
        "a flagged activation device loss must enter recovery"
    );
    let response = response_rx
        .try_recv()
        .expect("the caller must be told recovery failed");
    let err = response.expect_err("exhausted recovery must surface an error");
    let text = format!("{err:#}");
    assert!(
        text.contains("GPU device lost and recovery failed"),
        "expected the recovery-exhausted message, got: {text}"
    );
}

/// Issue #2363: an opaque error on the batched activation arm enters recovery when the device-lost flag is set.
#[test]
#[serial(gpu_stale_skip)]
fn flagged_opaque_activation_batch_error_enters_recovery() {
    let (work_tx, work_rx) = bounded::<GpuWorkRequest>(4);
    let (request, response_rx) = activation_batch_request();
    enqueue(&work_tx, request);
    enqueue(&work_tx, GpuWorkRequest::Shutdown);

    let heartbeat = Arc::new(GpuHeartbeat::new());
    let evaluator = FakeGpuEvaluator::new(
        WedgeBehaviour::OpaqueMapErrorAfterDeviceLoss,
        Arc::clone(&heartbeat),
    );
    let factory = FakeEvaluatorFactory::new();
    GpuWorkQueue::run_work_loop(evaluator, work_rx, &factory, &heartbeat);

    assert_eq!(
        factory.create_count(),
        crate::config::gpu_retry_limit() as usize,
        "a flagged batched activation device loss must enter recovery"
    );
    let response = response_rx
        .try_recv()
        .expect("the caller must be told recovery failed");
    let err = response.expect_err("exhausted recovery must surface an error");
    let text = format!("{err:#}");
    assert!(
        text.contains("GPU device lost and recovery failed"),
        "expected the recovery-exhausted message, got: {text}"
    );
}

/// Issue #2363: a device-loss message whose text matches a known pattern enters
/// recovery even when the device-lost callback never fired. This pins the text
/// operand of `routes_to_recovery`.
#[test]
#[serial(gpu_stale_skip)]
fn text_matched_device_loss_enters_recovery_without_flag() {
    let (work_tx, work_rx) = bounded::<GpuWorkRequest>(4);
    let (request, response_rx) = helpful_request();
    enqueue(&work_tx, request);
    enqueue(&work_tx, GpuWorkRequest::Shutdown);

    let heartbeat = Arc::new(GpuHeartbeat::new());
    let evaluator = FakeGpuEvaluator::new(
        WedgeBehaviour::TextDeviceLostUnflagged,
        Arc::clone(&heartbeat),
    );
    let factory = FakeEvaluatorFactory::new();
    GpuWorkQueue::run_work_loop(evaluator, work_rx, &factory, &heartbeat);

    assert_eq!(
        factory.create_count(),
        crate::config::gpu_retry_limit() as usize,
        "a text-matched device loss must enter recovery even without the flag"
    );
    let response = response_rx
        .try_recv()
        .expect("the caller must be told recovery failed");
    let err = response.expect_err("exhausted recovery must surface an error");
    let text = format!("{err:#}");
    assert!(
        text.contains("GPU device lost and recovery failed"),
        "expected the recovery-exhausted message, got: {text}"
    );
}
