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

use anyhow::Result;
use crossbeam_channel::Receiver;
use std::time::Instant;

use super::recovery::{get_gpu_retry_limit, is_device_lost_error};
use super::{GpuWorkQueue, GpuWorkRequest};
use crate::analysis::gpu::analyzer::{GpuAnalyzer, GpuEvaluator};
use crate::analysis::samples::{HelpfulSample, ReluStats};
use crate::observability::{global_gpu_metrics, gpu_metrics_enabled};

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
            let sample_count: usize = samples.iter().map(|v| v.len()).sum();
            let start = if track_metrics {
                Some(Instant::now())
            } else {
                None
            };

            let samples_refs: Vec<&[HelpfulSample]> =
                samples.iter().map(|v| v.as_slice()).collect();
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

            let _ = response_tx.send(result);
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

            let _ = response_tx.send(result);
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

            let _ = response_tx.send(result);
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

            let _ = response_tx.send(result);
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

            let _ = response_tx.send(result);
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
    /// - Batch count and samples processed
    /// - GPU busy time (time spent executing GPU operations)
    pub(super) fn gpu_thread_loop(mut analyzer: GpuAnalyzer, work_rx: Receiver<GpuWorkRequest>) {
        let track_metrics = gpu_metrics_enabled();
        let retry_limit = get_gpu_retry_limit();

        while let Ok(request) = work_rx.recv() {
            if matches!(request, GpuWorkRequest::Shutdown) {
                break;
            }

            match execute_request(&analyzer, &request, track_metrics) {
                Ok(()) => {
                    // Success — nothing to do
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
                        tracing::warn!(
                            attempt = attempt,
                            max_attempts = retry_limit,
                            "Re-initialising GpuAnalyzer (attempt {attempt}/{retry_limit})"
                        );

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
            let _ = response_tx.send(Err(anyhow::anyhow!("{error_msg}")));
        }
        GpuWorkRequest::HarmfulBatch { response_tx, .. } => {
            let _ = response_tx.send(Err(anyhow::anyhow!("{error_msg}")));
        }
        GpuWorkRequest::ReluEval { response_tx, .. } => {
            let _ = response_tx.send(Err(anyhow::anyhow!("{error_msg}")));
        }
        GpuWorkRequest::ActivationEval { response_tx, .. } => {
            let _ = response_tx.send(Err(anyhow::anyhow!("{error_msg}")));
        }
        GpuWorkRequest::ActivationBatchEval { response_tx, .. } => {
            let _ = response_tx.send(Err(anyhow::anyhow!("{error_msg}")));
        }
        GpuWorkRequest::Shutdown => {}
    }
}

/// Implementation for shared GpuWorkQueue.
/// NOTE: These trait methods use `None` deadline, which gives maximum timeout (5 minutes).
/// For deadline-aware evaluation, use the batch methods directly with an explicit deadline.
impl GpuEvaluator for GpuWorkQueue {
    fn evaluate_relu(
        &self,
        samples: &[HelpfulSample],
        threshold: f32,
    ) -> Result<(ReluStats, ReluStats, f32)> {
        // Clone samples to send to the GPU thread
        // Uses None deadline = maximum timeout (5 minutes)
        self.evaluate_relu_gpu(samples.to_vec(), threshold, &None)
    }

    fn evaluate_activation(
        &self,
        samples: &[HelpfulSample],
        activation_type: u32,
        orientation: f32,
        scale: f32,
    ) -> Result<(f32, f32, f32, u32)> {
        // Uses None deadline = maximum timeout (5 minutes)
        self.evaluate_activation_gpu(samples.to_vec(), activation_type, orientation, scale, &None)
    }

    fn evaluate_activations_batched(
        &self,
        samples: &[HelpfulSample],
        activation_configs: &[(u32, f32, f32)],
    ) -> Result<Vec<(f32, f32, f32, u32)>> {
        // Uses None deadline = maximum timeout (5 minutes)
        self.evaluate_activations_batched_gpu(samples.to_vec(), activation_configs.to_vec(), &None)
    }
}
