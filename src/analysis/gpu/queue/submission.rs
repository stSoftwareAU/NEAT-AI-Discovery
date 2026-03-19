//! Work item submission and batching
//!
//! This module contains the methods for submitting GPU work items to the queue,
//! including both synchronous (blocking) and asynchronous (future-based) submission.

use anyhow::{Result, anyhow};
use crossbeam_channel::bounded;
use std::time::Duration;

use super::{GpuFuture, GpuWorkQueue, GpuWorkRequest};
use crate::analysis::samples::{
    HarmfulStats, HelpfulSample, HelpfulStats, ReluOrientation, ReluStats,
};
use crate::analysis::utils::calculate_gpu_batch_timeout;

impl GpuWorkQueue {
    /// Submit a helpful batch to the GPU thread without blocking (Issue #568).
    ///
    /// Returns a `GpuFuture` that can be collected later, allowing the caller
    /// to perform CPU work (e.g., preparing harmful samples) while the GPU
    /// processes the helpful batch.
    pub(crate) fn submit_helpful_batch(
        &self,
        samples: Vec<Vec<HelpfulSample>>,
        deadline: &Option<std::time::SystemTime>,
    ) -> Result<GpuFuture<Vec<HelpfulStats>>> {
        if samples.is_empty() {
            // Return a pre-resolved future with empty results
            let (tx, rx) = bounded(1);
            if tx.send(Ok(Vec::new())).is_err() {
                tracing::trace!("GPU queue: receiver dropped for empty helpful batch");
            }
            return Ok(GpuFuture {
                response_rx: rx,
                timeout: Duration::from_secs(1),
            });
        }

        let (response_tx, response_rx) = bounded(1);
        let timeout = calculate_gpu_batch_timeout(deadline);
        let timeout_secs = timeout.as_secs();

        tracing::debug!(
            batch_count = samples.len(),
            "GPU queue: enqueuing helpful batch"
        );
        match self.work_tx.send_timeout(
            GpuWorkRequest::HelpfulBatch {
                samples,
                response_tx,
            },
            timeout,
        ) {
            Ok(()) => {}
            Err(crossbeam_channel::SendTimeoutError::Timeout(_)) => {
                return Err(anyhow!(
                    "GPU work queue full - send timed out after {timeout_secs}s. \
                     The GPU thread may be hung. Consider restarting the process."
                ));
            }
            Err(crossbeam_channel::SendTimeoutError::Disconnected(_)) => {
                return Err(anyhow!("GPU work queue channel closed"));
            }
        }

        Ok(GpuFuture {
            response_rx,
            timeout,
        })
    }

    /// Submit a batch of helpful evaluations and wait for results.
    ///
    /// This is a synchronous call that blocks until the GPU thread processes
    /// the batch and returns results.
    ///
    /// The `deadline` parameter is used to calculate an adaptive timeout:
    /// - With deadline: uses up to half remaining time (60s-5min)
    /// - Without deadline: uses maximum timeout (5 minutes)
    pub fn evaluate_helpful_batch(
        &self,
        samples: Vec<Vec<HelpfulSample>>,
        deadline: &Option<std::time::SystemTime>,
    ) -> Result<Vec<HelpfulStats>> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }

        // Create a one-shot channel for the response
        let (response_tx, response_rx) = bounded(1);

        // Calculate timeout based on remaining deadline
        let timeout = calculate_gpu_batch_timeout(deadline);
        let timeout_secs = timeout.as_secs();

        tracing::debug!(
            batch_count = samples.len(),
            "GPU queue: enqueuing helpful batch (blocking)"
        );
        // Send the work request with timeout to prevent deadlock if GPU thread is hung
        // If the channel is full (GPU not processing), this will timeout instead of blocking forever
        match self.work_tx.send_timeout(
            GpuWorkRequest::HelpfulBatch {
                samples,
                response_tx,
            },
            timeout,
        ) {
            Ok(()) => {}
            Err(crossbeam_channel::SendTimeoutError::Timeout(_)) => {
                return Err(anyhow!(
                    "GPU work queue full - send timed out after {timeout_secs}s. \
                     The GPU thread may be hung. Consider restarting the process."
                ));
            }
            Err(crossbeam_channel::SendTimeoutError::Disconnected(_)) => {
                return Err(anyhow!("GPU work queue channel closed"));
            }
        }

        // Wait for the response with timeout
        match response_rx.recv_timeout(timeout) {
            Ok(result) => result,
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => Err(anyhow!(
                "GPU helpful batch evaluation timed out after {timeout_secs}s. \
                     The GPU may be unresponsive. Consider reducing batch size or restarting."
            )),
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                Err(anyhow!("GPU response channel closed unexpectedly"))
            }
        }
    }

    /// Submit a batch of harmful evaluations and wait for results.
    ///
    /// The `deadline` parameter is used to calculate an adaptive timeout (60s-5min).
    pub fn evaluate_harmful_batch(
        &self,
        samples_with_weights: Vec<(Vec<HelpfulSample>, f32)>,
        deadline: &Option<std::time::SystemTime>,
    ) -> Result<Vec<HarmfulStats>> {
        if samples_with_weights.is_empty() {
            return Ok(Vec::new());
        }

        let (response_tx, response_rx) = bounded(1);
        let timeout = calculate_gpu_batch_timeout(deadline);
        let timeout_secs = timeout.as_secs();

        tracing::debug!(
            batch_count = samples_with_weights.len(),
            "GPU queue: enqueuing harmful batch"
        );
        // Send with timeout to prevent deadlock if GPU thread is hung
        match self.work_tx.send_timeout(
            GpuWorkRequest::HarmfulBatch {
                samples_with_weights,
                response_tx,
            },
            timeout,
        ) {
            Ok(()) => {}
            Err(crossbeam_channel::SendTimeoutError::Timeout(_)) => {
                return Err(anyhow!(
                    "GPU work queue full - send timed out after {timeout_secs}s. \
                     The GPU thread may be hung. Consider restarting the process."
                ));
            }
            Err(crossbeam_channel::SendTimeoutError::Disconnected(_)) => {
                return Err(anyhow!("GPU work queue channel closed"));
            }
        }

        match response_rx.recv_timeout(timeout) {
            Ok(result) => result,
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => Err(anyhow!(
                "GPU harmful batch evaluation timed out after {timeout_secs}s. \
                     The GPU may be unresponsive. Consider reducing batch size or restarting."
            )),
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                Err(anyhow!("GPU response channel closed unexpectedly"))
            }
        }
    }

    /// Submit a `ReLU` evaluation and wait for results.
    /// Returns (`positive_stats`, `negative_stats`, `baseline_error_sq`).
    ///
    /// The `deadline` parameter is used to calculate an adaptive timeout (60s-5min).
    pub(crate) fn evaluate_relu_gpu(
        &self,
        samples: Vec<HelpfulSample>,
        threshold: f32,
        deadline: &Option<std::time::SystemTime>,
    ) -> Result<(ReluStats, ReluStats, f32)> {
        if samples.is_empty() {
            return Ok((
                ReluStats::new(ReluOrientation::Positive),
                ReluStats::new(ReluOrientation::Negative),
                0.0,
            ));
        }

        let (response_tx, response_rx) = bounded(1);
        let timeout = calculate_gpu_batch_timeout(deadline);
        let timeout_secs = timeout.as_secs();

        tracing::debug!(
            sample_count = samples.len(),
            "GPU queue: enqueuing ReLU eval"
        );
        // Send with timeout to prevent deadlock if GPU thread is hung
        match self.work_tx.send_timeout(
            GpuWorkRequest::ReluEval {
                samples,
                threshold,
                response_tx,
            },
            timeout,
        ) {
            Ok(()) => {}
            Err(crossbeam_channel::SendTimeoutError::Timeout(_)) => {
                return Err(anyhow!(
                    "GPU work queue full - send timed out after {timeout_secs}s. \
                     The GPU thread may be hung. Consider restarting the process."
                ));
            }
            Err(crossbeam_channel::SendTimeoutError::Disconnected(_)) => {
                return Err(anyhow!("GPU work queue channel closed"));
            }
        }

        match response_rx.recv_timeout(timeout) {
            Ok(result) => result,
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => Err(anyhow!(
                "GPU ReLU evaluation timed out after {timeout_secs}s. \
                     The GPU may be unresponsive. Consider reducing batch size or restarting."
            )),
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                Err(anyhow!("GPU response channel closed unexpectedly"))
            }
        }
    }

    /// Submit an activation evaluation and wait for results.
    /// Returns (`sum_activation_sq`, `sum_error_activation`, `total_baseline_error_sq`, `improved_count`).
    ///
    /// The `deadline` parameter is used to calculate an adaptive timeout (60s-5min).
    pub(crate) fn evaluate_activation_gpu(
        &self,
        samples: Vec<HelpfulSample>,
        activation_type: u32,
        orientation: f32,
        scale: f32,
        deadline: &Option<std::time::SystemTime>,
    ) -> Result<(f32, f32, f32, u32)> {
        if samples.is_empty() {
            return Ok((0.0, 0.0, 0.0, 0));
        }

        let (response_tx, response_rx) = bounded(1);
        let timeout = calculate_gpu_batch_timeout(deadline);
        let timeout_secs = timeout.as_secs();

        tracing::debug!(
            sample_count = samples.len(),
            activation_type,
            "GPU queue: enqueuing activation eval"
        );
        // Send with timeout to prevent deadlock if GPU thread is hung
        match self.work_tx.send_timeout(
            GpuWorkRequest::ActivationEval {
                samples,
                activation_type,
                orientation,
                scale,
                response_tx,
            },
            timeout,
        ) {
            Ok(()) => {}
            Err(crossbeam_channel::SendTimeoutError::Timeout(_)) => {
                return Err(anyhow!(
                    "GPU work queue full - send timed out after {timeout_secs}s. \
                     The GPU thread may be hung. Consider restarting the process."
                ));
            }
            Err(crossbeam_channel::SendTimeoutError::Disconnected(_)) => {
                return Err(anyhow!("GPU work queue channel closed"));
            }
        }

        match response_rx.recv_timeout(timeout) {
            Ok(result) => result,
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => Err(anyhow!(
                "GPU activation evaluation timed out after {timeout_secs}s. \
                     The GPU may be unresponsive. Consider reducing batch size or restarting."
            )),
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                Err(anyhow!("GPU response channel closed unexpectedly"))
            }
        }
    }

    /// Submit a batched activation evaluation and wait for results.
    ///
    /// Issue #201: Evaluates multiple activation function configurations in a single
    /// GPU command buffer submission, reducing CPU-GPU round-trips by 10-20%.
    ///
    /// # Arguments
    /// * `samples` - The sample data to evaluate
    /// * `activation_configs` - List of (`activation_type`, orientation, scale) tuples
    /// * `deadline` - Optional deadline for adaptive timeout calculation
    ///
    /// # Returns
    /// Vector of (`sum_activation_sq`, `sum_error_activation`, `total_baseline_error_sq`, `improved_count`)
    /// in the same order as the input configs.
    pub(crate) fn evaluate_activations_batched_gpu(
        &self,
        samples: Vec<HelpfulSample>,
        activation_configs: Vec<(u32, f32, f32)>,
        deadline: &Option<std::time::SystemTime>,
    ) -> Result<Vec<(f32, f32, f32, u32)>> {
        // Handle edge cases
        if activation_configs.is_empty() {
            return Ok(Vec::new());
        }

        if samples.is_empty() {
            // Return zero results for each config
            return Ok(vec![(0.0, 0.0, 0.0, 0); activation_configs.len()]);
        }

        let (response_tx, response_rx) = bounded(1);
        let timeout = calculate_gpu_batch_timeout(deadline);
        let timeout_secs = timeout.as_secs();

        tracing::debug!(
            sample_count = samples.len(),
            config_count = activation_configs.len(),
            "GPU queue: enqueuing activation batch eval"
        );
        // Send with timeout to prevent deadlock if GPU thread is hung
        match self.work_tx.send_timeout(
            GpuWorkRequest::ActivationBatchEval {
                samples,
                activation_configs,
                response_tx,
            },
            timeout,
        ) {
            Ok(()) => {}
            Err(crossbeam_channel::SendTimeoutError::Timeout(_)) => {
                return Err(anyhow!(
                    "GPU work queue full - send timed out after {timeout_secs}s. \
                     The GPU thread may be hung. Consider restarting the process."
                ));
            }
            Err(crossbeam_channel::SendTimeoutError::Disconnected(_)) => {
                return Err(anyhow!("GPU work queue channel closed"));
            }
        }

        match response_rx.recv_timeout(timeout) {
            Ok(result) => result,
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => Err(anyhow!(
                "GPU batched activation evaluation timed out after {timeout_secs}s. \
                     The GPU may be unresponsive. Consider reducing batch size or restarting."
            )),
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                Err(anyhow!("GPU response channel closed unexpectedly"))
            }
        }
    }
}
