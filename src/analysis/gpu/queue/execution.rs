//! GPU execution and result collection
//!
//! This module contains the GPU thread main loop that processes work requests
//! and the `GpuEvaluator` trait implementation for `GpuWorkQueue`.

use anyhow::Result;
use crossbeam_channel::Receiver;
use std::time::Instant;

use super::{GpuWorkQueue, GpuWorkRequest};
use crate::analysis::gpu::analyzer::{GpuAnalyzer, GpuEvaluator};
use crate::analysis::samples::{HelpfulSample, ReluStats};
use crate::observability::{global_gpu_metrics, gpu_metrics_enabled};

impl GpuWorkQueue {
    /// The main loop for the GPU thread.
    /// Processes work requests until shutdown is requested.
    ///
    /// When `NEAT_AI_DISCOVERY_GPU_METRICS=1` is set, this loop tracks:
    /// - Batch count and samples processed
    /// - GPU busy time (time spent executing GPU operations)
    pub(super) fn gpu_thread_loop(analyzer: GpuAnalyzer, work_rx: Receiver<GpuWorkRequest>) {
        let track_metrics = gpu_metrics_enabled();

        while let Ok(request) = work_rx.recv() {
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

                    // Convert Vec<Vec<HelpfulSample>> to &[&[HelpfulSample]] for the API
                    let samples_refs: Vec<&[HelpfulSample]> =
                        samples.iter().map(|v| v.as_slice()).collect();
                    let result = analyzer.evaluate_helpful_batch(&samples_refs);

                    if let Some(start) = start {
                        let metrics = global_gpu_metrics();
                        metrics.record_batch(sample_count);
                        metrics.record_gpu_busy_us(start.elapsed().as_micros() as u64);
                    }

                    // Send result back (ignore send errors - receiver may have dropped)
                    let _ = response_tx.send(result);
                }
                GpuWorkRequest::HarmfulBatch {
                    samples_with_weights,
                    response_tx,
                } => {
                    let sample_count: usize =
                        samples_with_weights.iter().map(|(v, _)| v.len()).sum();

                    let start = if track_metrics {
                        Some(Instant::now())
                    } else {
                        None
                    };

                    // Convert to the format expected by evaluate_harmful_batch
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

                    let result = analyzer.evaluate_relu_gpu(&samples, threshold);

                    if let Some(start) = start {
                        let metrics = global_gpu_metrics();
                        metrics.record_batch(sample_count);
                        metrics.record_gpu_busy_us(start.elapsed().as_micros() as u64);
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

                    let result = analyzer.evaluate_activation_gpu(
                        &samples,
                        activation_type,
                        orientation,
                        scale,
                    );

                    if let Some(start) = start {
                        let metrics = global_gpu_metrics();
                        metrics.record_batch(sample_count);
                        metrics.record_gpu_busy_us(start.elapsed().as_micros() as u64);
                    }

                    let _ = response_tx.send(result);
                }
                GpuWorkRequest::ActivationBatchEval {
                    samples,
                    activation_configs,
                    response_tx,
                } => {
                    // Issue #201: Batched activation evaluation
                    let sample_count = samples.len();
                    let config_count = activation_configs.len();

                    let start = if track_metrics {
                        Some(Instant::now())
                    } else {
                        None
                    };

                    let result =
                        analyzer.evaluate_activations_batched_gpu(&samples, &activation_configs);

                    if let Some(start) = start {
                        let metrics = global_gpu_metrics();
                        // Record total samples processed (samples * configs)
                        metrics.record_batch(sample_count * config_count);
                        metrics.record_gpu_busy_us(start.elapsed().as_micros() as u64);
                    }

                    let _ = response_tx.send(result);
                }
                GpuWorkRequest::Shutdown => {
                    // Clean shutdown requested
                    break;
                }
            }
        }
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
