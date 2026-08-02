//! GPU execution and result collection
//!
//! This module contains the GPU thread main loop that processes work requests
//! and the `GpuEvaluator` trait implementation for `GpuWorkQueue`.
//!
//! ## Device-lost recovery (Issue #647)
//!
//! When a GPU operation fails with a device-lost error, the loop attempts to
//! re-initialise the `GpuAnalyzer` and retry the failed work item. The retry
//! limit is configurable via the `NEAT_AI_DISCOVERY_GPU_RETRY_LIMIT` environment
//! variable (default: 3).
//!
//! ## Stale request skipping (Issue #1929)
//!
//! A dequeued request whose caller has already gone (dropped receiver) or whose
//! own time budget has expired is skipped before the analyser is touched, and
//! the device-lost retry loop aborts as soon as the caller disappears — no
//! re-initialisation, no back-off sleeps for work nobody will collect. See
//! [`super::staleness`].

#![allow(clippy::cast_possible_truncation)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use anyhow::Result;
use crossbeam_channel::{Receiver, RecvTimeoutError};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use super::executor::{EvaluatorFactory, GpuAnalyzerFactory, RequestEvaluator};
use super::recovery::{
    DEFAULT_BACKOFF_INITIAL_MS, DEFAULT_BACKOFF_MAX_MS, MINIMUM_GPU_BATCH_SIZE, backoff_delay_ms,
    get_gpu_retry_limit, is_device_lost_error, is_memory_exhaustion_error,
};
use super::staleness::{StaleReason, has_live_receiver, stale_reason};
use super::{GpuWorkQueue, GpuWorkRequest};
use crate::analysis::gpu::analyzer::{GpuAnalyzer, GpuEvaluator};
use crate::analysis::samples::{HelpfulSample, ReluStats};
use crate::observability::{global_gpu_metrics, gpu_metrics_enabled};

/// Stall warning threshold in seconds (Issue #953). When a single GPU request
/// takes longer than this, a warning is logged to help diagnose liveness issues
/// before the outer task controller kills the process.
const GPU_REQUEST_STALL_WARN_SECS: u64 = 30;

/// Timeout for `recv_timeout()` in the GPU thread loop (Issue #1082). Long
/// enough to avoid busy-waiting, short enough to detect a dropped sender or
/// cancellation promptly.
const GPU_RECV_TIMEOUT: Duration = Duration::from_secs(30);

/// Return a short label describing the GPU work request variant.
fn request_label(request: &GpuWorkRequest) -> &'static str {
    match request {
        GpuWorkRequest::HelpfulBatch { .. } => "helpful_batch",
        GpuWorkRequest::HarmfulBatch { .. } => "harmful_batch",
        GpuWorkRequest::ReluEval { .. } => "relu_eval",
        GpuWorkRequest::ActivationEval { .. } => "activation_eval",
        GpuWorkRequest::ActivationBatchEval { .. } => "activation_batch_eval",
        GpuWorkRequest::Shutdown => "shutdown",
    }
}

/// Execute a single GPU work request, returning the result via the embedded
/// response channel. Returns `Err` only for device-lost errors that should
/// trigger recovery; normal evaluation errors are sent back to the caller.
fn execute_request<E: RequestEvaluator>(
    analyzer: &E,
    request: &GpuWorkRequest,
    track_metrics: bool,
) -> Result<(), anyhow::Error> {
    match request {
        GpuWorkRequest::HelpfulBatch {
            samples,
            response_tx,
            budget,
            ..
        } => {
            let sample_count: usize = samples.iter().map(|s| s.len()).sum();
            let start = if track_metrics {
                Some(Instant::now())
            } else {
                None
            };

            // Issue #1548: borrow each Arc-shared sample Vec as a slice — the
            // GPU thread never takes ownership, so refcount sharing is safe.
            let samples_refs: Vec<&[HelpfulSample]> =
                samples.iter().map(|s| s.as_slice()).collect();
            let result = analyzer.evaluate_helpful_batch(&samples_refs, *budget);

            if let Some(start) = start {
                let metrics = global_gpu_metrics();
                metrics.record_batch(sample_count);
                metrics.record_gpu_busy_us(start.elapsed().as_micros() as u64);
            }

            if let Err(ref e) = result
                && is_device_lost_error(e)
            {
                return Err(anyhow::anyhow!("{e:#}"));
            }

            if response_tx.send(result).is_err() {
                tracing::trace!("GPU queue: receiver dropped for helpful batch result");
            }
        }
        GpuWorkRequest::HarmfulBatch {
            samples_with_weights,
            response_tx,
            budget,
            ..
        } => {
            let sample_count: usize = samples_with_weights.iter().map(|(v, _)| v.len()).sum();
            let start = if track_metrics {
                Some(Instant::now())
            } else {
                None
            };

            let batch_refs: Vec<(&[HelpfulSample], f32)> = samples_with_weights
                .iter()
                .map(|(samples, weight)| (samples.as_slice(), *weight))
                .collect();
            let result = analyzer.evaluate_harmful_batch(&batch_refs, *budget);

            if let Some(start) = start {
                let metrics = global_gpu_metrics();
                metrics.record_batch(sample_count);
                metrics.record_gpu_busy_us(start.elapsed().as_micros() as u64);
            }

            if let Err(ref e) = result
                && is_device_lost_error(e)
            {
                return Err(anyhow::anyhow!("{e:#}"));
            }

            if response_tx.send(result).is_err() {
                tracing::trace!("GPU queue: receiver dropped for harmful batch result");
            }
        }
        GpuWorkRequest::ReluEval {
            samples,
            threshold,
            response_tx,
            budget,
            ..
        } => {
            let sample_count = samples.len();
            let start = if track_metrics {
                Some(Instant::now())
            } else {
                None
            };

            let result = analyzer.evaluate_relu(samples, *threshold, *budget);

            if let Some(start) = start {
                let metrics = global_gpu_metrics();
                metrics.record_batch(sample_count);
                metrics.record_gpu_busy_us(start.elapsed().as_micros() as u64);
            }

            if let Err(ref e) = result
                && is_device_lost_error(e)
            {
                return Err(anyhow::anyhow!("{e:#}"));
            }

            if response_tx.send(result).is_err() {
                tracing::trace!("GPU queue: receiver dropped for ReLU eval result");
            }
        }
        GpuWorkRequest::ActivationEval {
            samples,
            activation_type,
            orientation,
            scale,
            response_tx,
            budget,
            ..
        } => {
            let sample_count = samples.len();
            let start = if track_metrics {
                Some(Instant::now())
            } else {
                None
            };

            let result = analyzer.evaluate_activation(
                samples,
                *activation_type,
                *orientation,
                *scale,
                *budget,
            );

            if let Some(start) = start {
                let metrics = global_gpu_metrics();
                metrics.record_batch(sample_count);
                metrics.record_gpu_busy_us(start.elapsed().as_micros() as u64);
            }

            if let Err(ref e) = result
                && is_device_lost_error(e)
            {
                return Err(anyhow::anyhow!("{e:#}"));
            }

            if response_tx.send(result).is_err() {
                tracing::trace!("GPU queue: receiver dropped for activation eval result");
            }
        }
        GpuWorkRequest::ActivationBatchEval {
            samples,
            activation_configs,
            response_tx,
            budget,
            ..
        } => {
            let sample_count = samples.len();
            let config_count = activation_configs.len();
            let start = if track_metrics {
                Some(Instant::now())
            } else {
                None
            };

            let result =
                analyzer.evaluate_activations_batched(samples, activation_configs, *budget);

            if let Some(start) = start {
                let metrics = global_gpu_metrics();
                metrics.record_batch(sample_count * config_count);
                metrics.record_gpu_busy_us(start.elapsed().as_micros() as u64);
            }

            if let Err(ref e) = result
                && is_device_lost_error(e)
            {
                return Err(anyhow::anyhow!("{e:#}"));
            }

            if response_tx.send(result).is_err() {
                tracing::trace!("GPU queue: receiver dropped for activation batch eval result");
            }
        }
        GpuWorkRequest::Shutdown => {
            // Handled by caller
        }
    }
    Ok(())
}

/// Discard a request the worker must not spend GPU time on (Issue #1929).
///
/// A dropped receiver is dropped silently — nobody is left to hear about it.
/// An expired budget is reported as a real error so the still-waiting caller
/// fails immediately instead of blocking until its own timeout elapses.
fn skip_stale_request(request: &GpuWorkRequest, reason: StaleReason, label: &str) {
    global_gpu_metrics().record_stale_skip();
    tracing::debug!(
        request_type = label,
        reason = reason.as_str(),
        "GPU queue: skipping stale work item without invoking the analyser"
    );
    if reason == StaleReason::BudgetExpired {
        send_error_to_request(
            request,
            "GPU time budget expired while the request waited in the work queue — \
             abandoned without evaluation to keep the queue moving.",
        );
    }
}

impl GpuWorkQueue {
    /// The main loop for the GPU thread.
    /// Processes work requests until shutdown is requested.
    ///
    /// When a GPU operation fails with a device-lost error, this loop attempts
    /// to re-initialise the `GpuAnalyzer` and retry the failed work item up to
    /// `NEAT_AI_DISCOVERY_GPU_RETRY_LIMIT` times (default: 3). If recovery
    /// fails, the error is propagated back to the caller via the response
    /// channel.
    ///
    /// When `NEAT_AI_DISCOVERY_GPU_METRICS=1` is set, this loop tracks:
    ///   - Batch count and samples processed
    ///   - GPU busy time (time spent executing GPU operations)
    pub(super) fn gpu_thread_loop(analyzer: GpuAnalyzer, work_rx: Receiver<GpuWorkRequest>) {
        Self::run_work_loop(analyzer, work_rx, &GpuAnalyzerFactory);
    }

    /// The loop body, parameterised over the evaluator so it can be driven
    /// without a GPU in tests (Issue #1929).
    pub(super) fn run_work_loop<E: RequestEvaluator, F: EvaluatorFactory<E>>(
        mut analyzer: E,
        work_rx: Receiver<GpuWorkRequest>,
        factory: &F,
    ) {
        let track_metrics = gpu_metrics_enabled();
        let retry_limit = get_gpu_retry_limit();
        let completed_count = AtomicU64::new(0);
        tracing::debug!("GPU thread loop started — waiting for work");

        loop {
            let request = match work_rx.recv_timeout(GPU_RECV_TIMEOUT) {
                Ok(req) => req,
                Err(RecvTimeoutError::Timeout) => {
                    crate::watchdog::beat("gpu-queue-idle");
                    if crate::cancellation::is_cancelled() {
                        tracing::info!("GPU thread exiting — cancellation requested");
                        break;
                    }
                    tracing::debug!("GPU queue: recv timeout — channel idle, continuing");
                    continue;
                }
                Err(RecvTimeoutError::Disconnected) => {
                    tracing::warn!(
                        "GPU queue: sender disconnected without Shutdown — exiting thread"
                    );
                    break;
                }
            };
            if matches!(request, GpuWorkRequest::Shutdown) {
                let total = completed_count.load(Ordering::Relaxed);
                tracing::debug!(
                    total_completed = total,
                    "GPU thread received shutdown request"
                );
                break;
            }

            let label = request_label(&request);
            let request_start = Instant::now();
            tracing::debug!(request_type = label, "GPU queue: dequeued work item");

            // Issue #1929: never hand an abandoned or already-expired request to
            // the analyser — the result could not be delivered anyway, and the
            // GPU time it would consume belongs to live submitters.
            if let Some(reason) = stale_reason(&request) {
                skip_stale_request(&request, reason, label);
                continue;
            }

            match execute_request(&analyzer, &request, track_metrics) {
                Ok(()) => {
                    let elapsed = request_start.elapsed();
                    let count = completed_count.fetch_add(1, Ordering::Relaxed) + 1;

                    // Issue #953: Warn when a single GPU request takes too long,
                    // providing early visibility into potential stalls.
                    if elapsed.as_secs() >= GPU_REQUEST_STALL_WARN_SECS {
                        tracing::warn!(
                            request_type = label,
                            elapsed_secs = elapsed.as_secs(),
                            total_completed = count,
                            "GPU request took longer than {GPU_REQUEST_STALL_WARN_SECS}s — \
                             GPU may be under heavy load or driver is slow",
                            GPU_REQUEST_STALL_WARN_SECS = GPU_REQUEST_STALL_WARN_SECS,
                        );
                    } else {
                        tracing::debug!(
                            request_type = label,
                            elapsed_ms = elapsed.as_millis() as u64,
                            "GPU queue: work item completed"
                        );
                    }
                }
                Err(device_err) => {
                    let is_oom = is_memory_exhaustion_error(&device_err);

                    tracing::warn!(
                        error = %device_err,
                        retry_limit = retry_limit,
                        is_memory_exhaustion = is_oom,
                        "GPU device-lost detected — attempting recovery"
                    );

                    // Issue #1083: On OOM, halve the batch size for retry.
                    // Track the effective batch size so subsequent requests
                    // also use the reduced value.
                    let mut effective_batch_size = analyzer.batch_size();
                    if is_oom {
                        let new_size = effective_batch_size / 2;
                        if new_size < MINIMUM_GPU_BATCH_SIZE {
                            tracing::warn!(
                                current_batch_size = effective_batch_size,
                                minimum_batch_size = MINIMUM_GPU_BATCH_SIZE,
                                "GPU batch size already at or below minimum — \
                                 cannot reduce further, propagating OOM error"
                            );
                            send_error_to_request(
                                &request,
                                &format!(
                                    "GPU memory exhaustion with batch size {effective_batch_size} \
                                     (minimum {MINIMUM_GPU_BATCH_SIZE}) — cannot reduce further: \
                                     {device_err}"
                                ),
                            );
                            continue;
                        }
                        effective_batch_size = new_size;
                        tracing::warn!(
                            previous_batch_size = effective_batch_size * 2,
                            new_batch_size = effective_batch_size,
                            "Reducing GPU batch size due to memory exhaustion — \
                             consider tuning NEAT_AI_DISCOVERY_GPU_BATCH_SIZE"
                        );
                        if track_metrics {
                            let metrics = global_gpu_metrics();
                            metrics.record_batch_size_reduction(effective_batch_size);
                        }
                    }

                    let mut recovered = false;
                    let mut abandoned = false;
                    for attempt in 1..=retry_limit {
                        // Issue #1929: re-check before every attempt. Recovery
                        // costs a device re-initialisation plus a back-off
                        // sleep; none of it is worth spending once the caller
                        // has gone.
                        if !has_live_receiver(&request) {
                            global_gpu_metrics().record_stale_skip();
                            tracing::debug!(
                                request_type = label,
                                attempt = attempt,
                                "GPU recovery abandoned — caller dropped its receiver"
                            );
                            abandoned = true;
                            break;
                        }

                        let delay_ms = backoff_delay_ms(
                            attempt,
                            DEFAULT_BACKOFF_INITIAL_MS,
                            DEFAULT_BACKOFF_MAX_MS,
                        );
                        tracing::warn!(
                            attempt = attempt,
                            max_attempts = retry_limit,
                            backoff_ms = delay_ms,
                            effective_batch_size = effective_batch_size,
                            error = %device_err,
                            "GPU recovery attempt {attempt}/{retry_limit} — \
                             waiting {delay_ms}ms before re-initialising GpuAnalyzer"
                        );
                        std::thread::sleep(Duration::from_millis(delay_ms));

                        let init_result = factory.create(is_oom.then_some(effective_batch_size));

                        match init_result {
                            Ok(new_analyzer) => {
                                analyzer = new_analyzer;
                                tracing::warn!(
                                    attempt = attempt,
                                    batch_size = analyzer.batch_size(),
                                    "GPU device recovered — retrying failed work item"
                                );

                                match execute_request(&analyzer, &request, track_metrics) {
                                    Ok(()) => {
                                        tracing::warn!(
                                            attempt = attempt,
                                            batch_size = analyzer.batch_size(),
                                            "GPU work item succeeded after recovery"
                                        );
                                        recovered = true;
                                        break;
                                    }
                                    Err(retry_err) => {
                                        tracing::warn!(
                                            attempt = attempt,
                                            error = %retry_err,
                                            "GPU work item failed again after recovery"
                                        );
                                    }
                                }
                            }
                            Err(init_err) => {
                                tracing::warn!(
                                    attempt = attempt,
                                    error = %init_err,
                                    "GpuAnalyzer re-initialisation failed"
                                );
                            }
                        }
                    }

                    if !recovered && !abandoned {
                        tracing::warn!(
                            retry_limit = retry_limit,
                            "GPU recovery exhausted all {retry_limit} attempts — \
                             error propagated to caller"
                        );
                        send_error_to_request(
                            &request,
                            &format!(
                                "GPU device lost and recovery failed after \
                                 {retry_limit} attempts: {device_err}"
                            ),
                        );
                    }
                }
            }
        }
    }
}

/// Send an error message back through the request's response channel.
/// This is used when recovery fails and we need to inform the caller.
fn send_error_to_request(request: &GpuWorkRequest, error_msg: &str) {
    match request {
        GpuWorkRequest::HelpfulBatch { response_tx, .. } => {
            if response_tx
                .send(Err(anyhow::anyhow!("{error_msg}")))
                .is_err()
            {
                tracing::trace!("GPU queue: receiver dropped for helpful batch error response");
            }
        }
        GpuWorkRequest::HarmfulBatch { response_tx, .. } => {
            if response_tx
                .send(Err(anyhow::anyhow!("{error_msg}")))
                .is_err()
            {
                tracing::trace!("GPU queue: receiver dropped for harmful batch error response");
            }
        }
        GpuWorkRequest::ReluEval { response_tx, .. } => {
            if response_tx
                .send(Err(anyhow::anyhow!("{error_msg}")))
                .is_err()
            {
                tracing::trace!("GPU queue: receiver dropped for ReLU eval error response");
            }
        }
        GpuWorkRequest::ActivationEval { response_tx, .. } => {
            if response_tx
                .send(Err(anyhow::anyhow!("{error_msg}")))
                .is_err()
            {
                tracing::trace!("GPU queue: receiver dropped for activation eval error response");
            }
        }
        GpuWorkRequest::ActivationBatchEval { response_tx, .. } => {
            if response_tx
                .send(Err(anyhow::anyhow!("{error_msg}")))
                .is_err()
            {
                tracing::trace!(
                    "GPU queue: receiver dropped for activation batch eval error response"
                );
            }
        }
        GpuWorkRequest::Shutdown => {}
    }
}

/// Implementation for shared `GpuWorkQueue`.
///
/// Issue #953: These trait methods now use the queue's `deadline` field (set via
/// `with_deadline()`) to calculate adaptive timeouts. Previously they always
/// passed `None`, giving every call a maximum 5-minute timeout. When many rayon
/// threads were blocked waiting on slow GPU responses simultaneously, the process
/// appeared hung until the outer task controller killed it.
impl GpuEvaluator for GpuWorkQueue {
    fn evaluate_relu(
        &self,
        samples: &[HelpfulSample],
        threshold: f32,
    ) -> Result<(ReluStats, ReluStats, f32)> {
        self.evaluate_relu_gpu(samples, threshold, &self.deadline)
    }

    fn evaluate_activation(
        &self,
        samples: &[HelpfulSample],
        activation_type: u32,
        orientation: f32,
        scale: f32,
    ) -> Result<(f32, f32, f32, u32)> {
        self.evaluate_activation_gpu(samples, activation_type, orientation, scale, &self.deadline)
    }

    fn evaluate_activations_batched(
        &self,
        samples: &[HelpfulSample],
        activation_configs: &[(u32, f32, f32)],
    ) -> Result<Vec<(f32, f32, f32, u32)>> {
        self.evaluate_activations_batched_gpu(samples, activation_configs, &self.deadline)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::gpu::budget::GpuTimeBudget;
    use crate::analysis::samples::{HarmfulStats, HelpfulStats};
    use crossbeam_channel::bounded;

    use super::super::staleness::caller_liveness_pair;

    /// Verify that the GPU thread exits cleanly when the sender is dropped
    /// without sending a `Shutdown` request (Issue #1082). The
    /// `recv_timeout()` detects the disconnected channel and breaks the loop.
    #[test]
    fn test_gpu_thread_exits_when_sender_dropped() {
        if !crate::analysis::gpu::analyzer::GpuAnalyzer::gpu_is_available() {
            eprintln!("Skipping test: no GPU available");
            return;
        }

        let analyzer = match crate::analysis::gpu::analyzer::GpuAnalyzer::new() {
            Ok(a) => a,
            Err(e) => {
                eprintln!("Skipping test: GPU analyser init failed: {e}");
                return;
            }
        };

        let (tx, rx) = bounded::<GpuWorkRequest>(1);
        drop(tx);

        let handle = std::thread::spawn(move || {
            GpuWorkQueue::gpu_thread_loop(analyzer, rx);
        });

        let join_result = handle.join();
        assert!(
            join_result.is_ok(),
            "GPU thread should exit cleanly when sender is dropped"
        );
    }

    /// Verify that `send_error_to_request` does not panic when the receiver has
    /// been dropped (e.g., caller timed out). Each variant is tested.
    #[test]
    fn test_send_error_to_request_handles_dropped_helpful_receiver() {
        let (tx, rx) = bounded::<Result<Vec<HelpfulStats>>>(1);
        drop(rx);
        let request = GpuWorkRequest::HelpfulBatch {
            samples: vec![],
            response_tx: tx,
            budget: GpuTimeBudget::unbounded(),
            liveness: caller_liveness_pair().1,
        };
        // Must not panic — the receiver is gone
        send_error_to_request(&request, "test error");
    }

    #[test]
    fn test_send_error_to_request_handles_dropped_harmful_receiver() {
        let (tx, rx) = bounded::<Result<Vec<HarmfulStats>>>(1);
        drop(rx);
        let request = GpuWorkRequest::HarmfulBatch {
            samples_with_weights: vec![],
            response_tx: tx,
            budget: GpuTimeBudget::unbounded(),
            liveness: caller_liveness_pair().1,
        };
        send_error_to_request(&request, "test error");
    }

    #[test]
    fn test_send_error_to_request_handles_dropped_relu_receiver() {
        let (tx, rx) = bounded::<Result<(ReluStats, ReluStats, f32)>>(1);
        drop(rx);
        let request = GpuWorkRequest::ReluEval {
            samples: vec![],
            threshold: 0.0,
            response_tx: tx,
            budget: GpuTimeBudget::unbounded(),
            liveness: caller_liveness_pair().1,
        };
        send_error_to_request(&request, "test error");
    }

    #[test]
    fn test_send_error_to_request_handles_dropped_activation_receiver() {
        let (tx, rx) = bounded::<Result<(f32, f32, f32, u32)>>(1);
        drop(rx);
        let request = GpuWorkRequest::ActivationEval {
            samples: vec![],
            activation_type: 0,
            orientation: 1.0,
            scale: 1.0,
            response_tx: tx,
            budget: GpuTimeBudget::unbounded(),
            liveness: caller_liveness_pair().1,
        };
        send_error_to_request(&request, "test error");
    }

    #[test]
    fn test_send_error_to_request_handles_dropped_activation_batch_receiver() {
        let (tx, rx) = bounded::<Result<Vec<(f32, f32, f32, u32)>>>(1);
        drop(rx);
        let request = GpuWorkRequest::ActivationBatchEval {
            samples: vec![],
            activation_configs: vec![],
            response_tx: tx,
            budget: GpuTimeBudget::unbounded(),
            liveness: caller_liveness_pair().1,
        };
        send_error_to_request(&request, "test error");
    }

    /// Verify that `send_error_to_request` succeeds when the receiver is still
    /// alive — the error message is delivered correctly.
    #[test]
    fn test_send_error_to_request_delivers_error_when_receiver_alive() {
        let (tx, rx) = bounded::<Result<Vec<HelpfulStats>>>(1);
        let request = GpuWorkRequest::HelpfulBatch {
            samples: vec![],
            response_tx: tx,
            budget: GpuTimeBudget::unbounded(),
            liveness: caller_liveness_pair().1,
        };
        send_error_to_request(&request, "recovery failed");
        let result = rx.recv().expect("should receive error");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("recovery failed"));
    }

    /// Verify that `send_error_to_request` is a no-op for the Shutdown variant.
    #[test]
    fn test_send_error_to_request_noop_for_shutdown() {
        let request = GpuWorkRequest::Shutdown;
        // Must not panic
        send_error_to_request(&request, "should be ignored");
    }

    /// Verify that `request_label` returns correct labels for all variants.
    #[test]
    fn test_request_label_returns_correct_labels() {
        let (tx, _rx) = bounded::<Result<Vec<HelpfulStats>>>(1);
        assert_eq!(
            request_label(&GpuWorkRequest::HelpfulBatch {
                samples: vec![],
                response_tx: tx,
                budget: GpuTimeBudget::unbounded(),
                liveness: caller_liveness_pair().1,
            }),
            "helpful_batch"
        );

        let (tx, _rx) = bounded::<Result<Vec<HarmfulStats>>>(1);
        assert_eq!(
            request_label(&GpuWorkRequest::HarmfulBatch {
                samples_with_weights: vec![],
                response_tx: tx,
                budget: GpuTimeBudget::unbounded(),
                liveness: caller_liveness_pair().1,
            }),
            "harmful_batch"
        );

        let (tx, _rx) = bounded::<Result<(ReluStats, ReluStats, f32)>>(1);
        assert_eq!(
            request_label(&GpuWorkRequest::ReluEval {
                samples: vec![],
                threshold: 0.0,
                response_tx: tx,
                budget: GpuTimeBudget::unbounded(),
                liveness: caller_liveness_pair().1,
            }),
            "relu_eval"
        );

        let (tx, _rx) = bounded::<Result<(f32, f32, f32, u32)>>(1);
        assert_eq!(
            request_label(&GpuWorkRequest::ActivationEval {
                samples: vec![],
                activation_type: 0,
                orientation: 1.0,
                scale: 1.0,
                response_tx: tx,
                budget: GpuTimeBudget::unbounded(),
                liveness: caller_liveness_pair().1,
            }),
            "activation_eval"
        );

        let (tx, _rx) = bounded::<Result<Vec<(f32, f32, f32, u32)>>>(1);
        assert_eq!(
            request_label(&GpuWorkRequest::ActivationBatchEval {
                samples: vec![],
                activation_configs: vec![],
                response_tx: tx,
                budget: GpuTimeBudget::unbounded(),
                liveness: caller_liveness_pair().1,
            }),
            "activation_batch_eval"
        );

        assert_eq!(request_label(&GpuWorkRequest::Shutdown), "shutdown");
    }
}
