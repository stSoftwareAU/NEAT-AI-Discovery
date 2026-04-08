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

#![allow(clippy::cast_possible_truncation)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use anyhow::Result;
use crossbeam_channel::Receiver;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use super::recovery::{
    DEFAULT_BACKOFF_INITIAL_MS, DEFAULT_BACKOFF_MAX_MS, backoff_delay_ms, get_gpu_retry_limit,
    is_device_lost_error,
};
use super::{GpuWorkQueue, GpuWorkRequest};
use crate::analysis::gpu::analyzer::{GpuAnalyzer, GpuEvaluator};
use crate::analysis::samples::{HelpfulSample, ReluStats};
use crate::observability::{global_gpu_metrics, gpu_metrics_enabled};

/// Stall warning threshold in seconds (Issue #953). When a single GPU request
/// takes longer than this, a warning is logged to help diagnose liveness issues
/// before the outer task controller kills the process.
const GPU_REQUEST_STALL_WARN_SECS: u64 = 30;

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
fn execute_request(
    analyzer: &GpuAnalyzer,
    request: &GpuWorkRequest,
    track_metrics: bool,
) -> Result<(), anyhow::Error> {
    match request {
        GpuWorkRequest::HelpfulBatch {
            samples,
            response_tx,
        } => {
            let sample_count: usize = samples.iter().map(std::vec::Vec::len).sum();
            let start = if track_metrics {
                Some(Instant::now())
            } else {
                None
            };

            let samples_refs: Vec<&[HelpfulSample]> =
                samples.iter().map(std::vec::Vec::as_slice).collect();
            let result = analyzer.evaluate_helpful_batch(&samples_refs);

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
            let result = analyzer.evaluate_harmful_batch(&batch_refs);

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
        } => {
            let sample_count = samples.len();
            let start = if track_metrics {
                Some(Instant::now())
            } else {
                None
            };

            let result = analyzer.evaluate_relu_gpu(samples, *threshold);

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
        } => {
            let sample_count = samples.len();
            let start = if track_metrics {
                Some(Instant::now())
            } else {
                None
            };

            let result =
                analyzer.evaluate_activation_gpu(samples, *activation_type, *orientation, *scale);

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
        } => {
            let sample_count = samples.len();
            let config_count = activation_configs.len();
            let start = if track_metrics {
                Some(Instant::now())
            } else {
                None
            };

            let result = analyzer.evaluate_activations_batched_gpu(samples, activation_configs);

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
    pub(super) fn gpu_thread_loop(mut analyzer: GpuAnalyzer, work_rx: Receiver<GpuWorkRequest>) {
        let track_metrics = gpu_metrics_enabled();
        let retry_limit = get_gpu_retry_limit();
        let completed_count = AtomicU64::new(0);
        tracing::debug!("GPU thread loop started — waiting for work");

        while let Ok(request) = work_rx.recv() {
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
                    // Device-lost detected — attempt recovery
                    tracing::warn!(
                        error = %device_err,
                        retry_limit = retry_limit,
                        "GPU device-lost detected — attempting recovery"
                    );

                    let mut recovered = false;
                    for attempt in 1..=retry_limit {
                        let delay_ms = backoff_delay_ms(
                            attempt,
                            DEFAULT_BACKOFF_INITIAL_MS,
                            DEFAULT_BACKOFF_MAX_MS,
                        );
                        tracing::warn!(
                            attempt = attempt,
                            max_attempts = retry_limit,
                            backoff_ms = delay_ms,
                            error = %device_err,
                            "GPU recovery attempt {attempt}/{retry_limit} — \
                             waiting {delay_ms}ms before re-initialising GpuAnalyzer"
                        );
                        std::thread::sleep(Duration::from_millis(delay_ms));

                        match GpuAnalyzer::new() {
                            Ok(new_analyzer) => {
                                analyzer = new_analyzer;
                                tracing::warn!(
                                    attempt = attempt,
                                    "GPU device recovered — retrying failed work item"
                                );

                                // Retry the failed request with the new analyser
                                match execute_request(&analyzer, &request, track_metrics) {
                                    Ok(()) => {
                                        tracing::warn!(
                                            attempt = attempt,
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
                                        // Continue to next attempt
                                    }
                                }
                            }
                            Err(init_err) => {
                                tracing::warn!(
                                    attempt = attempt,
                                    error = %init_err,
                                    "GpuAnalyzer re-initialisation failed"
                                );
                                // Continue to next attempt
                            }
                        }
                    }

                    if !recovered {
                        tracing::warn!(
                            retry_limit = retry_limit,
                            "GPU recovery exhausted all {retry_limit} attempts — \
                             error propagated to caller"
                        );
                        // The error was already sent to the response channel
                        // in execute_request (or the channel was dropped).
                        // Send the error back if the response channel is still open.
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
    use crate::analysis::samples::{HarmfulStats, HelpfulStats};
    use crossbeam_channel::bounded;

    /// Verify that `send_error_to_request` does not panic when the receiver has
    /// been dropped (e.g., caller timed out). Each variant is tested.
    #[test]
    fn test_send_error_to_request_handles_dropped_helpful_receiver() {
        let (tx, rx) = bounded::<Result<Vec<HelpfulStats>>>(1);
        drop(rx);
        let request = GpuWorkRequest::HelpfulBatch {
            samples: vec![],
            response_tx: tx,
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
            }),
            "helpful_batch"
        );

        let (tx, _rx) = bounded::<Result<Vec<HarmfulStats>>>(1);
        assert_eq!(
            request_label(&GpuWorkRequest::HarmfulBatch {
                samples_with_weights: vec![],
                response_tx: tx,
            }),
            "harmful_batch"
        );

        let (tx, _rx) = bounded::<Result<(ReluStats, ReluStats, f32)>>(1);
        assert_eq!(
            request_label(&GpuWorkRequest::ReluEval {
                samples: vec![],
                threshold: 0.0,
                response_tx: tx,
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
            }),
            "activation_eval"
        );

        let (tx, _rx) = bounded::<Result<Vec<(f32, f32, f32, u32)>>>(1);
        assert_eq!(
            request_label(&GpuWorkRequest::ActivationBatchEval {
                samples: vec![],
                activation_configs: vec![],
                response_tx: tx,
            }),
            "activation_batch_eval"
        );

        assert_eq!(request_label(&GpuWorkRequest::Shutdown), "shutdown");
    }
}
