//! Work item submission and batching
//!
//! This module contains the methods for submitting GPU work items to the queue,
//! including both synchronous (blocking) and asynchronous (future-based) submission.

use anyhow::{Result, anyhow};
use crossbeam_channel::{Receiver, bounded};
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::staleness::caller_liveness_pair;
use super::{GpuFuture, GpuWorkQueue, GpuWorkRequest};
use crate::analysis::gpu::breaker::{GpuCircuitBreaker, GpuTripReason, gpu_wedged_error};
use crate::analysis::gpu::budget::GpuTimeBudget;
use crate::analysis::gpu::heartbeat::{GpuHeartbeat, HeartbeatWatch, global_gpu_heartbeat};
use crate::analysis::samples::{
    HarmfulStats, HelpfulSample, HelpfulStats, ReluOrientation, ReluStats,
};
use crate::analysis::utils::calculate_gpu_batch_timeout;

/// Error for a work request the GPU thread never accepted (Issue #1930).
///
/// A full queue that will not drain within the caller's whole timeout means the
/// GPU thread is not consuming work, so this trips the breaker. Issue #1932: the
/// returned error is typed, so the host reads it as a wedged GPU rather than as
/// a retryable timeout.
pub fn queue_full_error(breaker: &GpuCircuitBreaker, timeout_secs: u64) -> anyhow::Error {
    breaker.trip(GpuTripReason::BatchTimeout);
    gpu_wedged_error(format!(
        "GPU work queue full - send timed out after {timeout_secs}s; \
         the GPU thread is not consuming work"
    ))
}

/// Error for a submitted batch the GPU thread never answered (Issue #1930).
///
/// This is the wedged-GPU signature the breaker exists to stop repeating: the
/// caller has just burned its full 60–300s wait for nothing. Issue #1932: it is
/// reported as [`DiscoveryError::GpuWedged`](crate::ffi_types::DiscoveryError),
/// never as a timeout the host could retry with a longer deadline.
pub fn batch_timeout_error(
    breaker: &GpuCircuitBreaker,
    operation: &str,
    timeout_secs: u64,
) -> anyhow::Error {
    breaker.trip(GpuTripReason::BatchTimeout);
    gpu_wedged_error(format!("GPU {operation} timed out after {timeout_secs}s"))
}

/// Error for a GPU thread that stopped publishing progress (Issue #1933).
///
/// This is the fast verdict the heartbeat exists for: instead of sitting out the
/// whole 60–300s batch timeout, the submitter declares the device wedged as soon
/// as the GPU thread has been silent for the stall window.
pub fn heartbeat_stall_error(
    breaker: &GpuCircuitBreaker,
    operation: &str,
    idle: Duration,
    window: Duration,
) -> anyhow::Error {
    breaker.trip(GpuTripReason::HeartbeatStall);
    gpu_wedged_error(format!(
        "GPU {operation} abandoned after {idle_secs:.1}s with no GPU-thread progress \
         (stall window {window_secs}s); the GPU is wedged",
        idle_secs = idle.as_secs_f64(),
        window_secs = window.as_secs(),
    ))
}

/// How a bounded wait for a GPU response ended (Issue #1933).
#[derive(Debug)]
pub enum GpuWaitOutcome<T> {
    /// The GPU thread answered — successfully or with an evaluation error.
    Answered(Result<T>),
    /// The GPU thread published no progress for the stall window.
    Stalled {
        /// How long the heartbeat had been silent when the wait gave up.
        idle: Duration,
        /// The configured stall window that was exceeded.
        window: Duration,
    },
    /// The absolute batch timeout expired — the backstop for a heartbeat that
    /// cannot be updated at all.
    TimedOut,
    /// The GPU thread dropped the response sender.
    Disconnected,
}

/// Wait for a GPU response, giving up early once the GPU thread goes silent.
///
/// The wait wakes every [`HeartbeatWatch::poll_interval`] and checks both the
/// response channel and the heartbeat, so a wedged device is detected within the
/// stall window instead of costing the caller its whole `timeout`. The absolute
/// `timeout` remains as the backstop.
pub fn wait_for_gpu_response<T>(
    response_rx: &Receiver<Result<T>>,
    timeout: Duration,
    heartbeat: &GpuHeartbeat,
    stall_window: Duration,
) -> GpuWaitOutcome<T> {
    let deadline = Instant::now() + timeout;
    let mut watch = HeartbeatWatch::new(heartbeat, stall_window);
    let interval = watch.poll_interval();

    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return GpuWaitOutcome::TimedOut;
        }

        match response_rx.recv_timeout(interval.min(remaining)) {
            Ok(result) => return GpuWaitOutcome::Answered(result),
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                if let Some(idle) = watch.stalled_for() {
                    return GpuWaitOutcome::Stalled {
                        idle,
                        window: watch.stall_window(),
                    };
                }
            }
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                return GpuWaitOutcome::Disconnected;
            }
        }
    }
}

/// Resolve a bounded wait into the caller's result, tripping the breaker on
/// either wedged-GPU verdict (Issue #1933).
pub(crate) fn resolve_gpu_wait<T>(
    outcome: GpuWaitOutcome<T>,
    breaker: &GpuCircuitBreaker,
    operation: &str,
    timeout_secs: u64,
) -> Result<T> {
    match outcome {
        GpuWaitOutcome::Answered(result) => result,
        GpuWaitOutcome::Stalled { idle, window } => {
            Err(heartbeat_stall_error(breaker, operation, idle, window))
        }
        GpuWaitOutcome::TimedOut => Err(batch_timeout_error(breaker, operation, timeout_secs)),
        GpuWaitOutcome::Disconnected => Err(anyhow!(
            "GPU response channel closed unexpectedly — \
             the GPU thread may have exited or panicked"
        )),
    }
}

/// Wait for a submitted request on the global heartbeat with the configured
/// stall window (Issue #1933).
pub(crate) fn await_gpu_response<T>(
    response_rx: &Receiver<Result<T>>,
    timeout: Duration,
    breaker: &GpuCircuitBreaker,
    operation: &str,
) -> Result<T> {
    let outcome = wait_for_gpu_response(
        response_rx,
        timeout,
        global_gpu_heartbeat(),
        crate::config::gpu_stall_window(),
    );
    resolve_gpu_wait(outcome, breaker, operation, timeout.as_secs())
}

impl GpuWorkQueue {
    /// Submit a helpful batch to the GPU thread without blocking (Issue #568).
    ///
    /// Returns a `GpuFuture` that can be collected later, allowing the caller
    /// to perform CPU work (e.g., preparing harmful samples) while the GPU
    /// processes the helpful batch.
    pub(crate) fn submit_helpful_batch(
        &self,
        samples: Vec<Arc<Vec<HelpfulSample>>>,
        deadline: &Option<std::time::SystemTime>,
    ) -> Result<GpuFuture<Vec<HelpfulStats>>> {
        // Issue #1930: a wedged GPU answers nothing — fail now instead of
        // handing back a future that can only time out.
        self.breaker.check()?;

        if samples.is_empty() {
            // Return a pre-resolved future with empty results
            let (tx, rx) = bounded(1);
            if tx.send(Ok(Vec::new())).is_err() {
                tracing::trace!("GPU queue: receiver dropped for empty helpful batch");
            }
            let (caller_guard, _) = caller_liveness_pair();
            return Ok(GpuFuture {
                response_rx: rx,
                timeout: Duration::from_secs(1),
                caller_guard,
                breaker: self.breaker,
            });
        }

        let (response_tx, response_rx) = bounded(1);
        let timeout = calculate_gpu_batch_timeout(deadline);
        let timeout_secs = timeout.as_secs();
        // Issue #1928: the worker's inner waits share this request's budget,
        // which expires a safety margin before the caller stops waiting.
        let budget = GpuTimeBudget::from_caller_timeout(timeout);
        // Issue #1929: the guard lives exactly as long as this caller waits, so
        // the worker can tell an abandoned request from a live one.
        let (caller_guard, liveness) = caller_liveness_pair();

        tracing::debug!(
            batch_count = samples.len(),
            "GPU queue: enqueuing helpful batch"
        );
        match self.work_tx.send_timeout(
            GpuWorkRequest::HelpfulBatch {
                samples,
                response_tx,
                budget,
                liveness,
            },
            timeout,
        ) {
            Ok(()) => {}
            Err(crossbeam_channel::SendTimeoutError::Timeout(_)) => {
                return Err(queue_full_error(self.breaker, timeout_secs));
            }
            Err(crossbeam_channel::SendTimeoutError::Disconnected(_)) => {
                return Err(anyhow!("GPU work queue channel closed"));
            }
        }

        Ok(GpuFuture {
            response_rx,
            timeout,
            caller_guard,
            breaker: self.breaker,
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
        samples: Vec<Arc<Vec<HelpfulSample>>>,
        deadline: &Option<std::time::SystemTime>,
    ) -> Result<Vec<HelpfulStats>> {
        // Issue #1930: never start another multi-minute wait on a wedged GPU.
        self.breaker.check()?;

        if samples.is_empty() {
            return Ok(Vec::new());
        }

        // Create a one-shot channel for the response
        let (response_tx, response_rx) = bounded(1);

        // Calculate timeout based on remaining deadline
        let timeout = calculate_gpu_batch_timeout(deadline);
        let timeout_secs = timeout.as_secs();
        // Issue #1928: the worker's inner waits share this request's budget,
        // which expires a safety margin before the caller stops waiting.
        let budget = GpuTimeBudget::from_caller_timeout(timeout);
        // Issue #1929: the guard lives exactly as long as this caller waits, so
        // the worker can tell an abandoned request from a live one.
        let (_caller_guard, liveness) = caller_liveness_pair();

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
                budget,
                liveness,
            },
            timeout,
        ) {
            Ok(()) => {}
            Err(crossbeam_channel::SendTimeoutError::Timeout(_)) => {
                return Err(queue_full_error(self.breaker, timeout_secs));
            }
            Err(crossbeam_channel::SendTimeoutError::Disconnected(_)) => {
                return Err(anyhow!("GPU work queue channel closed"));
            }
        }

        // Issue #1933: bounded wait — a GPU thread that stops publishing
        // progress is declared wedged within the stall window rather than
        // costing this caller the whole batch timeout.
        await_gpu_response(
            &response_rx,
            timeout,
            self.breaker,
            "helpful batch evaluation",
        )
    }

    /// Submit a batch of harmful evaluations and wait for results.
    ///
    /// The `deadline` parameter is used to calculate an adaptive timeout (60s-5min).
    pub fn evaluate_harmful_batch(
        &self,
        samples_with_weights: Vec<(Arc<Vec<HelpfulSample>>, f32)>,
        deadline: &Option<std::time::SystemTime>,
    ) -> Result<Vec<HarmfulStats>> {
        // Issue #1930: never start another multi-minute wait on a wedged GPU.
        self.breaker.check()?;

        if samples_with_weights.is_empty() {
            return Ok(Vec::new());
        }

        let (response_tx, response_rx) = bounded(1);
        let timeout = calculate_gpu_batch_timeout(deadline);
        let timeout_secs = timeout.as_secs();
        // Issue #1928: the worker's inner waits share this request's budget,
        // which expires a safety margin before the caller stops waiting.
        let budget = GpuTimeBudget::from_caller_timeout(timeout);
        // Issue #1929: the guard lives exactly as long as this caller waits, so
        // the worker can tell an abandoned request from a live one.
        let (_caller_guard, liveness) = caller_liveness_pair();

        tracing::debug!(
            batch_count = samples_with_weights.len(),
            "GPU queue: enqueuing harmful batch"
        );
        // Send with timeout to prevent deadlock if GPU thread is hung
        match self.work_tx.send_timeout(
            GpuWorkRequest::HarmfulBatch {
                samples_with_weights,
                response_tx,
                budget,
                liveness,
            },
            timeout,
        ) {
            Ok(()) => {}
            Err(crossbeam_channel::SendTimeoutError::Timeout(_)) => {
                return Err(queue_full_error(self.breaker, timeout_secs));
            }
            Err(crossbeam_channel::SendTimeoutError::Disconnected(_)) => {
                return Err(anyhow!("GPU work queue channel closed"));
            }
        }

        // Issue #1933: bounded wait — a GPU thread that stops publishing
        // progress is declared wedged within the stall window rather than
        // costing this caller the whole batch timeout.
        await_gpu_response(
            &response_rx,
            timeout,
            self.breaker,
            "harmful batch evaluation",
        )
    }

    /// Submit a `ReLU` evaluation and wait for results.
    /// Returns (`positive_stats`, `negative_stats`, `baseline_error_sq`).
    ///
    /// The `deadline` parameter is used to calculate an adaptive timeout (60s-5min).
    pub(crate) fn evaluate_relu_gpu(
        &self,
        samples: &[HelpfulSample],
        threshold: f32,
        deadline: &Option<std::time::SystemTime>,
    ) -> Result<(ReluStats, ReluStats, f32)> {
        // Issue #1930: never start another multi-minute wait on a wedged GPU.
        self.breaker.check()?;

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
        // Issue #1928: the worker's inner waits share this request's budget,
        // which expires a safety margin before the caller stops waiting.
        let budget = GpuTimeBudget::from_caller_timeout(timeout);
        // Issue #1929: the guard lives exactly as long as this caller waits, so
        // the worker can tell an abandoned request from a live one.
        let (_caller_guard, liveness) = caller_liveness_pair();

        tracing::debug!(
            sample_count = samples.len(),
            "GPU queue: enqueuing ReLU eval"
        );
        // Send with timeout to prevent deadlock if GPU thread is hung
        match self.work_tx.send_timeout(
            GpuWorkRequest::ReluEval {
                samples: samples.to_vec(),
                threshold,
                response_tx,
                budget,
                liveness,
            },
            timeout,
        ) {
            Ok(()) => {}
            Err(crossbeam_channel::SendTimeoutError::Timeout(_)) => {
                return Err(queue_full_error(self.breaker, timeout_secs));
            }
            Err(crossbeam_channel::SendTimeoutError::Disconnected(_)) => {
                return Err(anyhow!("GPU work queue channel closed"));
            }
        }

        // Issue #1933: bounded wait — a GPU thread that stops publishing
        // progress is declared wedged within the stall window rather than
        // costing this caller the whole batch timeout.
        await_gpu_response(&response_rx, timeout, self.breaker, "ReLU evaluation")
    }

    /// Submit an activation evaluation and wait for results.
    /// Returns (`sum_activation_sq`, `sum_error_activation`, `total_baseline_error_sq`, `improved_count`).
    ///
    /// The `deadline` parameter is used to calculate an adaptive timeout (60s-5min).
    pub(crate) fn evaluate_activation_gpu(
        &self,
        samples: &[HelpfulSample],
        activation_type: u32,
        orientation: f32,
        scale: f32,
        deadline: &Option<std::time::SystemTime>,
    ) -> Result<(f32, f32, f32, u32)> {
        // Issue #1930: never start another multi-minute wait on a wedged GPU.
        self.breaker.check()?;

        if samples.is_empty() {
            return Ok((0.0, 0.0, 0.0, 0));
        }

        let (response_tx, response_rx) = bounded(1);
        let timeout = calculate_gpu_batch_timeout(deadline);
        let timeout_secs = timeout.as_secs();
        // Issue #1928: the worker's inner waits share this request's budget,
        // which expires a safety margin before the caller stops waiting.
        let budget = GpuTimeBudget::from_caller_timeout(timeout);
        // Issue #1929: the guard lives exactly as long as this caller waits, so
        // the worker can tell an abandoned request from a live one.
        let (_caller_guard, liveness) = caller_liveness_pair();

        tracing::debug!(
            sample_count = samples.len(),
            activation_type,
            "GPU queue: enqueuing activation eval"
        );
        // Send with timeout to prevent deadlock if GPU thread is hung
        match self.work_tx.send_timeout(
            GpuWorkRequest::ActivationEval {
                samples: samples.to_vec(),
                activation_type,
                orientation,
                scale,
                response_tx,
                budget,
                liveness,
            },
            timeout,
        ) {
            Ok(()) => {}
            Err(crossbeam_channel::SendTimeoutError::Timeout(_)) => {
                return Err(queue_full_error(self.breaker, timeout_secs));
            }
            Err(crossbeam_channel::SendTimeoutError::Disconnected(_)) => {
                return Err(anyhow!("GPU work queue channel closed"));
            }
        }

        // Issue #1933: bounded wait — a GPU thread that stops publishing
        // progress is declared wedged within the stall window rather than
        // costing this caller the whole batch timeout.
        await_gpu_response(&response_rx, timeout, self.breaker, "activation evaluation")
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
        samples: &[HelpfulSample],
        activation_configs: &[(u32, f32, f32)],
        deadline: &Option<std::time::SystemTime>,
    ) -> Result<Vec<(f32, f32, f32, u32)>> {
        // Issue #1930: never start another multi-minute wait on a wedged GPU.
        self.breaker.check()?;

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
        // Issue #1928: the worker's inner waits share this request's budget,
        // which expires a safety margin before the caller stops waiting.
        let budget = GpuTimeBudget::from_caller_timeout(timeout);
        // Issue #1929: the guard lives exactly as long as this caller waits, so
        // the worker can tell an abandoned request from a live one.
        let (_caller_guard, liveness) = caller_liveness_pair();

        tracing::debug!(
            sample_count = samples.len(),
            config_count = activation_configs.len(),
            "GPU queue: enqueuing activation batch eval"
        );
        // Send with timeout to prevent deadlock if GPU thread is hung
        match self.work_tx.send_timeout(
            GpuWorkRequest::ActivationBatchEval {
                samples: samples.to_vec(),
                activation_configs: activation_configs.to_vec(),
                response_tx,
                budget,
                liveness,
            },
            timeout,
        ) {
            Ok(()) => {}
            Err(crossbeam_channel::SendTimeoutError::Timeout(_)) => {
                return Err(queue_full_error(self.breaker, timeout_secs));
            }
            Err(crossbeam_channel::SendTimeoutError::Disconnected(_)) => {
                return Err(anyhow!("GPU work queue channel closed"));
            }
        }

        // Issue #1933: bounded wait — a GPU thread that stops publishing
        // progress is declared wedged within the stall window rather than
        // costing this caller the whole batch timeout.
        await_gpu_response(
            &response_rx,
            timeout,
            self.breaker,
            "batched activation evaluation",
        )
    }
}

#[cfg(test)]
mod tests {
    //! Issue #1548: verify the GPU submit path shares `Arc`-wrapped sample
    //! buffers with the queue instead of deep-copying them for ownership.
    use super::*;

    fn sample(v: f32) -> HelpfulSample {
        HelpfulSample {
            activation: v,
            avg_error: 0.0,
            target_value: None,
            target_activation: None,
        }
    }

    /// Build a queue whose work receiver is returned so tests can inspect the
    /// enqueued request. No GPU thread is spawned.
    ///
    /// Issue #1930: each queue gets its **own** circuit breaker, leaked for
    /// `'static`. Tripping the process-wide breaker here would refuse GPU work
    /// for every other test in this binary — including the ones that run a real
    /// analysis — so the tripped path is exercised in isolation instead.
    fn test_queue() -> (GpuWorkQueue, crossbeam_channel::Receiver<GpuWorkRequest>) {
        let (work_tx, work_rx) = bounded::<GpuWorkRequest>(1);
        let (_exit_tx, exit_rx) = bounded::<()>(1);
        let queue = GpuWorkQueue {
            work_tx,
            thread_handle: None,
            exit_rx,
            deadline: None,
            breaker: Box::leak(Box::new(GpuCircuitBreaker::new())),
        };
        (queue, work_rx)
    }

    /// Issue #1548: submitting a helpful batch shares the sample buffer with the
    /// queue via `Arc` (pointer identity preserved) rather than deep-copying it,
    /// and the caller retains read access afterwards.
    #[test]
    fn submit_helpful_batch_shares_samples_without_deep_copy() {
        let (queue, work_rx) = test_queue();

        let samples = Arc::new(vec![sample(1.0), sample(2.0), sample(3.0)]);
        let original_ptr = Arc::as_ptr(&samples);

        // Caller keeps its own handle; the submit batch takes a refcount clone.
        let batch = vec![Arc::clone(&samples)];
        assert_eq!(
            Arc::strong_count(&samples),
            2,
            "submit batch must share the buffer, not copy it"
        );

        let _future = queue
            .submit_helpful_batch(batch, &None)
            .expect("submit should enqueue");

        // The queued request holds the SAME buffer — no deep copy crossed the
        // submit boundary.
        match work_rx.recv().expect("request should be enqueued") {
            GpuWorkRequest::HelpfulBatch {
                samples: queued, ..
            } => {
                assert_eq!(queued.len(), 1, "one work item enqueued");
                assert!(
                    std::ptr::eq(Arc::as_ptr(&queued[0]), original_ptr),
                    "queued samples must be the shared buffer, not a copy"
                );
                assert_eq!(queued[0].len(), 3, "sample contents preserved");
            }
            _ => panic!("expected HelpfulBatch request variant"),
        }

        // Caller can still read its samples after submit (shared, not moved).
        assert_eq!(samples.len(), 3, "caller retains access after submit");
    }

    /// Issue #1548: an empty helpful batch resolves immediately to empty stats
    /// without enqueuing any work.
    #[test]
    fn submit_helpful_batch_empty_is_preresolved() {
        let (queue, _work_rx) = test_queue();
        let future = queue
            .submit_helpful_batch(Vec::new(), &None)
            .expect("empty batch should succeed");
        let stats = future.collect().expect("empty batch resolves");
        assert!(stats.is_empty(), "empty batch yields no stats");
    }

    /// Issue #1548: the harmful batch API accepts `Arc`-shared samples; the
    /// empty-input fast path returns empty stats without a GPU.
    #[test]
    fn evaluate_harmful_batch_empty_returns_empty() {
        let (queue, _work_rx) = test_queue();
        let batch: Vec<(Arc<Vec<HelpfulSample>>, f32)> = Vec::new();
        let stats = queue
            .evaluate_harmful_batch(batch, &None)
            .expect("empty harmful batch should succeed");
        assert!(stats.is_empty(), "empty harmful batch yields no stats");
    }

    /// Issue #1928: a submitted request carries a bounded budget that expires
    /// strictly before the caller's own timeout, so the worker gives up first
    /// and the GPU thread stays joinable.
    #[test]
    fn submitted_request_budget_expires_before_the_caller_timeout() {
        let (queue, work_rx) = test_queue();
        let deadline = Some(std::time::SystemTime::now() + Duration::from_secs(240));
        let caller_timeout = calculate_gpu_batch_timeout(&deadline);

        queue
            .submit_helpful_batch(vec![Arc::new(vec![sample(1.0)])], &deadline)
            .expect("submit should enqueue");

        let GpuWorkRequest::HelpfulBatch { budget, .. } =
            work_rx.recv().expect("request should be enqueued")
        else {
            panic!("expected HelpfulBatch request variant");
        };

        assert!(budget.is_bounded(), "a deadlined request carries a budget");
        assert!(
            budget.remaining() < caller_timeout,
            "worker budget {:?} must expire before the caller timeout {caller_timeout:?}",
            budget.remaining()
        );
        assert!(
            budget.remaining()
                <= caller_timeout
                    - Duration::from_secs(
                        crate::analysis::gpu::device::GPU_BUFFER_MAP_TIMEOUT_MARGIN_SECS
                    ),
            "the safety margin must be deducted from the worker budget"
        );
    }

    /// Issue #1928: without a deadline the request still carries a budget, and
    /// it never exceeds the maximum queue timeout.
    #[test]
    fn submitted_request_without_deadline_stays_within_the_max_timeout() {
        let (queue, work_rx) = test_queue();
        let caller_timeout = calculate_gpu_batch_timeout(&None);

        queue
            .submit_helpful_batch(vec![Arc::new(vec![sample(1.0)])], &None)
            .expect("submit should enqueue");

        let GpuWorkRequest::HelpfulBatch { budget, .. } =
            work_rx.recv().expect("request should be enqueued")
        else {
            panic!("expected HelpfulBatch request variant");
        };

        assert!(budget.remaining() < caller_timeout);
        assert!(
            budget.remaining()
                < Duration::from_secs(crate::analysis::utils::GPU_QUEUE_TIMEOUT_MAX_SECS)
        );
    }

    // =========================================================================
    // Issue #1930 — a tripped circuit breaker short-circuits every entry point
    // =========================================================================

    use std::time::Instant;

    /// Well under the 60s minimum wait each of these calls would otherwise
    /// start, and far above the microseconds a suppressed call actually takes.
    const IMMEDIATE: Duration = Duration::from_secs(1);

    /// Assert a call was suppressed by the breaker. Takes the whole `Result` so
    /// it works for the entry points whose success type is not `Debug`.
    fn assert_suppressed<T>(result: Result<T>, entry_point: &str) {
        match result {
            Ok(_) => panic!("{entry_point} must be suppressed while the breaker is tripped"),
            Err(err) => {
                let msg = format!("{err:#}");
                assert!(
                    msg.contains("GPU circuit breaker tripped"),
                    "{entry_point}: expected the breaker error, got: {msg}"
                );
            }
        }
    }

    /// Every `submit_*`/`evaluate_*` entry point must return the breaker error
    /// immediately once the breaker has tripped — no enqueue, no waiting.
    ///
    /// The queue here has capacity 1 and no consumer, so an un-suppressed call
    /// would block for its full timeout; the deadline bound is what proves the
    /// short circuit.
    #[test]
    fn tripped_breaker_short_circuits_every_entry_point() {
        let (queue, work_rx) = test_queue();
        queue.breaker.trip(GpuTripReason::AbandonedThread);

        let batch = vec![Arc::new(vec![sample(1.0)])];
        let samples = vec![sample(1.0)];

        let started = Instant::now();

        assert_suppressed(
            queue.submit_helpful_batch(batch.clone(), &None),
            "submit_helpful_batch",
        );
        assert_suppressed(
            queue.evaluate_helpful_batch(batch, &None),
            "evaluate_helpful_batch",
        );
        assert_suppressed(
            queue.evaluate_harmful_batch(vec![(Arc::new(samples.clone()), 1.0)], &None),
            "evaluate_harmful_batch",
        );
        assert_suppressed(
            queue.evaluate_relu_gpu(&samples, 0.0, &None),
            "evaluate_relu_gpu",
        );
        assert_suppressed(
            queue.evaluate_activation_gpu(&samples, 0, 1.0, 1.0, &None),
            "evaluate_activation_gpu",
        );
        assert_suppressed(
            queue.evaluate_activations_batched_gpu(&samples, &[(0, 1.0, 1.0)], &None),
            "evaluate_activations_batched_gpu",
        );

        let elapsed = started.elapsed();
        assert!(
            elapsed < IMMEDIATE,
            "suppressed calls must return immediately, took {elapsed:?}"
        );
        assert!(
            work_rx.try_recv().is_err(),
            "a suppressed call must not enqueue GPU work"
        );
    }

    /// The empty-input fast paths are suppressed too: after a trip nothing
    /// reports a clean (zero) result that a caller could mistake for real work.
    #[test]
    fn tripped_breaker_suppresses_the_empty_input_fast_paths() {
        let (queue, _work_rx) = test_queue();
        queue.breaker.trip(GpuTripReason::BatchTimeout);

        assert!(queue.submit_helpful_batch(Vec::new(), &None).is_err());
        assert!(queue.evaluate_helpful_batch(Vec::new(), &None).is_err());
        assert!(queue.evaluate_harmful_batch(Vec::new(), &None).is_err());
        assert!(queue.evaluate_relu_gpu(&[], 0.0, &None).is_err());
        assert!(
            queue
                .evaluate_activations_batched_gpu(&[], &[], &None)
                .is_err()
        );
    }

    /// A batch that times out waiting for the GPU trips the breaker — this is
    /// the wedged-batch signal the breaker exists to remember.
    #[test]
    fn a_batch_response_timeout_trips_the_breaker() {
        let breaker = GpuCircuitBreaker::new();
        assert!(!breaker.is_tripped(), "starts closed");

        let err = batch_timeout_error(&breaker, "helpful batch evaluation", 60);

        assert!(
            format!("{err:#}").contains("timed out after 60s"),
            "the caller still gets the original timeout error"
        );
        assert!(
            breaker.is_tripped(),
            "a batch timeout must trip the breaker"
        );
        assert_eq!(breaker.trip_reason(), Some(GpuTripReason::BatchTimeout));
    }

    // =========================================================================
    // Issue #1933 — the bounded wait detects a wedged GPU within the stall
    // window instead of sitting out the whole batch timeout
    // =========================================================================

    /// A GPU thread that stops publishing progress is declared wedged within
    /// the stall window, not after `calculate_gpu_batch_timeout()`.
    #[test]
    fn stalled_heartbeat_trips_within_window() {
        // The sender stays alive, so only the stall guard can end this wait.
        let (_response_tx, response_rx) = bounded::<Result<Vec<HelpfulStats>>>(1);
        let heartbeat = GpuHeartbeat::new();
        let window = Duration::from_millis(200);
        let batch_timeout = calculate_gpu_batch_timeout(&None);

        let started = Instant::now();
        let outcome = wait_for_gpu_response(&response_rx, batch_timeout, &heartbeat, window);
        let elapsed = started.elapsed();

        match outcome {
            GpuWaitOutcome::Stalled { idle, window: w } => {
                assert!(idle >= window, "the reported idle time covers the window");
                assert_eq!(w, window, "the verdict reports the configured window");
            }
            other => panic!("expected a stalled verdict, got {other:?}"),
        }
        assert!(
            elapsed >= window,
            "the guard must not fire before the window elapses, fired after {elapsed:?}"
        );
        assert!(
            elapsed < Duration::from_secs(5) && elapsed < batch_timeout,
            "detection must cost seconds, not the {batch_timeout:?} batch timeout \
             (took {elapsed:?})"
        );
    }

    /// A slow-but-progressing GPU must never be flagged: a synthetic evaluator
    /// advances the heartbeat at intervals inside the window and still answers.
    #[test]
    fn slow_but_advancing_heartbeat_does_not_trip() {
        let (response_tx, response_rx) = bounded::<Result<Vec<HelpfulStats>>>(1);
        let heartbeat = Arc::new(GpuHeartbeat::new());
        let worker_heartbeat = Arc::clone(&heartbeat);

        // Five slow steps at 60ms — each well inside the 250ms window, and
        // together far longer than the window itself.
        let worker = std::thread::spawn(move || {
            for _ in 0..5 {
                std::thread::sleep(Duration::from_millis(60));
                worker_heartbeat.beat("synthetic_step");
            }
            let _ = response_tx.send(Ok(vec![HelpfulStats::default()]));
        });

        let outcome = wait_for_gpu_response(
            &response_rx,
            Duration::from_secs(30),
            &heartbeat,
            Duration::from_millis(250),
        );
        worker.join().expect("synthetic evaluator thread panicked");

        match outcome {
            GpuWaitOutcome::Answered(Ok(stats)) => assert_eq!(stats.len(), 1),
            other => panic!("a slow but advancing GPU must not be flagged, got {other:?}"),
        }
    }

    /// The absolute timeout stays as the backstop for the case where the
    /// heartbeat itself cannot be updated — here, the guard is disabled.
    #[test]
    fn the_absolute_timeout_remains_the_backstop() {
        let (_response_tx, response_rx) = bounded::<Result<Vec<HelpfulStats>>>(1);
        let heartbeat = GpuHeartbeat::new();
        let timeout = Duration::from_millis(300);

        let started = Instant::now();
        let outcome = wait_for_gpu_response(&response_rx, timeout, &heartbeat, Duration::ZERO);

        assert!(
            matches!(outcome, GpuWaitOutcome::TimedOut),
            "with the guard disabled the wait ends at the absolute timeout"
        );
        assert!(
            started.elapsed() >= timeout,
            "the backstop must wait out the full timeout"
        );
    }

    /// A dropped GPU thread is still reported as a disconnect, not a stall.
    #[test]
    fn a_dropped_sender_is_reported_as_disconnected() {
        let (response_tx, response_rx) = bounded::<Result<Vec<HelpfulStats>>>(1);
        drop(response_tx);
        let heartbeat = GpuHeartbeat::new();

        let outcome = wait_for_gpu_response(
            &response_rx,
            Duration::from_secs(30),
            &heartbeat,
            Duration::from_millis(200),
        );

        assert!(matches!(outcome, GpuWaitOutcome::Disconnected));
    }

    /// A stall verdict is a wedged-GPU verdict: it trips the breaker with its
    /// own reason so the rest of the run stops submitting.
    #[test]
    fn a_heartbeat_stall_trips_the_breaker_as_a_wedged_gpu() {
        let breaker = GpuCircuitBreaker::new();
        assert!(!breaker.is_tripped(), "starts closed");

        let outcome: GpuWaitOutcome<Vec<HelpfulStats>> = GpuWaitOutcome::Stalled {
            idle: Duration::from_secs(31),
            window: Duration::from_secs(30),
        };
        let err = resolve_gpu_wait(outcome, &breaker, "helpful batch evaluation", 300)
            .expect_err("a stalled wait must fail");

        let msg = format!("{err:#}");
        assert!(
            msg.contains("no GPU-thread progress") && msg.contains("stall window 30s"),
            "the error must name the stall window: {msg}"
        );
        assert_eq!(breaker.trip_reason(), Some(GpuTripReason::HeartbeatStall));
    }

    /// A queue that will not accept work within the caller's whole timeout is
    /// the same wedged-GPU signal.
    #[test]
    fn a_full_queue_send_timeout_trips_the_breaker() {
        let breaker = GpuCircuitBreaker::new();
        let err = queue_full_error(&breaker, 60);

        assert!(format!("{err:#}").contains("GPU work queue full"));
        assert_eq!(breaker.trip_reason(), Some(GpuTripReason::BatchTimeout));
    }

    /// After the reset hook the queue submits normally again, so tests that
    /// trip the breaker do not make later ones order-dependent.
    #[test]
    fn reset_restores_normal_submission() {
        let (queue, work_rx) = test_queue();
        queue.breaker.trip(GpuTripReason::AbandonedThread);
        assert!(queue.evaluate_helpful_batch(Vec::new(), &None).is_err());

        queue.breaker.reset();

        queue
            .submit_helpful_batch(vec![Arc::new(vec![sample(1.0)])], &None)
            .expect("submission works again once the breaker is reset");
        assert!(
            work_rx.try_recv().is_ok(),
            "the request is enqueued once more"
        );
    }
}
