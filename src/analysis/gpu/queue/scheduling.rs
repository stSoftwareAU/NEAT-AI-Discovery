//! Work scheduling, initialisation, and shutdown
//!
//! This module handles GPU thread lifecycle management including creation,
//! initialisation with timeout, graceful shutdown, and cleanup via Drop.

use anyhow::{Context, Result, anyhow};
use crossbeam_channel::{Receiver, Sender, bounded};
use std::thread;
use std::time::Duration;

use super::{GpuWorkQueue, GpuWorkRequest};
use crate::analysis::gpu::analyzer::GpuAnalyzer;
use crate::analysis::gpu::breaker::{
    GpuCircuitBreaker, GpuTripReason, global_gpu_breaker, gpu_wedged_error,
};
use crate::analysis::gpu::shaders::{GPU_INIT_TIMEOUT_SECS, GPU_SHUTDOWN_TIMEOUT_SECS};
use crate::analysis::utils::get_work_queue_capacity;

/// Run the GPU thread's body, turning a panic into a loud, prompt failure
/// (Issue #2361).
///
/// Without this guard a panic unwinds out of the thread, the work receiver
/// drops, and every request already queued keeps its `response_tx` alive
/// inside the channel buffer (crossbeam's bounded flavour does not discard
/// buffered messages while a sender — the queue's `work_tx` — lives), so each
/// caller waits out the stall window and is reported as a wedge. On panic this
/// logs the payload, trips the breaker with `WorkerPanicked`, and answers every
/// queued request with an error naming the panic.
pub(super) fn run_guarded_gpu_thread(
    work_rx: &Receiver<GpuWorkRequest>,
    breaker: &GpuCircuitBreaker,
    body: impl FnOnce(),
) {
    if let Err(payload) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
        let msg = panic_payload_message(&*payload);
        tracing::error!("GPU thread panicked: {msg}");
        breaker.trip(GpuTripReason::WorkerPanicked);

        let mut failed = 0usize;
        while let Ok(request) = work_rx.try_recv() {
            fail_queued_request(request, &msg);
            failed += 1;
        }
        if failed > 0 {
            tracing::error!(
                failed_requests = failed,
                "GPU thread panic: failed queued requests that were still waiting"
            );
        }
    }
}

/// Answer a single queued request with an error naming the panic, so its
/// caller fails immediately instead of waiting out the stall window.
///
/// Exhaustive on purpose (no `_` wildcard) so a new `GpuWorkRequest` variant
/// forces this to be updated.
fn fail_queued_request(request: GpuWorkRequest, msg: &str) {
    match request {
        GpuWorkRequest::HelpfulBatch { response_tx, .. } => {
            if response_tx
                .send(Err(anyhow!("GPU thread panicked: {msg}")))
                .is_err()
            {
                tracing::trace!("GPU queue: caller gone before panic response could be sent");
            }
        }
        GpuWorkRequest::HarmfulBatch { response_tx, .. } => {
            if response_tx
                .send(Err(anyhow!("GPU thread panicked: {msg}")))
                .is_err()
            {
                tracing::trace!("GPU queue: caller gone before panic response could be sent");
            }
        }
        GpuWorkRequest::ReluEval { response_tx, .. } => {
            if response_tx
                .send(Err(anyhow!("GPU thread panicked: {msg}")))
                .is_err()
            {
                tracing::trace!("GPU queue: caller gone before panic response could be sent");
            }
        }
        GpuWorkRequest::ActivationEval { response_tx, .. } => {
            if response_tx
                .send(Err(anyhow!("GPU thread panicked: {msg}")))
                .is_err()
            {
                tracing::trace!("GPU queue: caller gone before panic response could be sent");
            }
        }
        GpuWorkRequest::ActivationBatchEval { response_tx, .. } => {
            if response_tx
                .send(Err(anyhow!("GPU thread panicked: {msg}")))
                .is_err()
            {
                tracing::trace!("GPU queue: caller gone before panic response could be sent");
            }
        }
        GpuWorkRequest::Shutdown => {}
    }
}

/// Extract a human-readable message from a caught panic payload.
///
/// Panics raised via `panic!("...")` or `.expect("...")` carry a `&str` or
/// `String` payload; anything else (a custom `panic_any` payload) falls back
/// to a fixed message rather than failing to report at all.
pub(super) fn panic_payload_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

impl GpuWorkQueue {
    /// Create a new GPU work queue with a dedicated GPU thread.
    ///
    /// The GPU thread is spawned immediately and owns the `GpuAnalyzer`.
    /// All GPU operations are processed sequentially on this thread,
    /// eliminating device creation overhead and improving utilisation.
    ///
    /// CRITICAL: The `GpuAnalyzer` is created INSIDE the GPU thread, not before.
    /// wgpu devices have thread-local state that doesn't transfer properly when
    /// moved across threads, causing deadlocks in `device.poll()`.
    pub fn new() -> Result<Self> {
        // Issue #1930: once the GPU has wedged, never spawn another thread against
        // it — a fresh thread means a fresh wgpu device on dead hardware, and the
        // last one is still leaked.
        let breaker = global_gpu_breaker();
        breaker.check()?;

        // Create the channel for sending work to the GPU thread.
        // Capacity is dynamically sized based on available system memory.
        // Lower capacity = more backpressure = less memory usage.
        // This prevents Metal command buffer exhaustion on memory-constrained systems.
        let queue_capacity = get_work_queue_capacity();
        let (work_tx, work_rx): (Sender<GpuWorkRequest>, Receiver<GpuWorkRequest>) =
            bounded(queue_capacity);

        // Channel to receive initialisation result from the GPU thread.
        // This ensures the GpuAnalyzer is created ON the GPU thread, not moved to it.
        let (init_tx, init_rx): (Sender<Result<()>>, Receiver<Result<()>>) = bounded(1);

        // Channel to receive notification when GPU thread exits.
        // This allows Drop to use a timeout instead of blocking forever if the GPU hangs.
        let (exit_tx, exit_rx): (Sender<()>, Receiver<()>) = bounded(1);

        // Spawn dedicated GPU thread - analyzer is created INSIDE this thread
        let drain_rx = work_rx.clone();
        let thread_breaker = breaker;
        let thread_handle = thread::spawn(move || {
            run_guarded_gpu_thread(&drain_rx, thread_breaker, move || {
                tracing::debug!("GPU thread started — initialising GpuAnalyzer");
                // Create analyzer on THIS thread to avoid wgpu thread-local state issues
                match GpuAnalyzer::new() {
                    Ok(analyzer) => {
                        tracing::debug!("GPU thread initialisation succeeded");
                        // Signal successful initialisation
                        if init_tx.send(Ok(())).is_err() {
                            tracing::trace!(
                                "GPU queue: init receiver dropped before success signal"
                            );
                        }
                        // Run the main loop
                        Self::gpu_thread_loop(analyzer, work_rx);
                    }
                    Err(e) => {
                        tracing::debug!("GPU thread initialisation failed");
                        // Signal initialisation failure
                        if init_tx.send(Err(e)).is_err() {
                            tracing::trace!(
                                "GPU queue: init receiver dropped before failure signal"
                            );
                        }
                    }
                }
            });
            // Always signal exit — `run_guarded_gpu_thread` catches a panic from
            // initialisation or the main loop (Issue #2361), so this now runs
            // even when the body above panicked.
            if exit_tx.send(()).is_err() {
                tracing::trace!("GPU queue: exit receiver dropped");
            }
            tracing::debug!("GPU thread exiting");
        });

        // Wait for initialisation to complete with timeout
        let init_timeout = Duration::from_secs(GPU_INIT_TIMEOUT_SECS);
        match init_rx.recv_timeout(init_timeout) {
            Ok(Ok(())) => {} // Success
            Ok(Err(e)) => {
                return Err(e).context("GPU analyser initialisation failed on GPU thread");
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                // Issue #1930: initialisation that never completes means the device
                // is unreachable — do not let the next analysis try again.
                breaker.trip(GpuTripReason::InitTimeout);
                // Issue #1932: typed, so the host sees a wedged GPU rather than
                // a timeout it should retry with a longer deadline.
                return Err(gpu_wedged_error(format!(
                    "GPU initialisation timed out after {GPU_INIT_TIMEOUT_SECS}s"
                )));
            }
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                return Err(anyhow!("GPU thread failed to start (channel disconnected)"));
            }
        }

        Ok(Self {
            work_tx,
            thread_handle: Some(thread_handle),
            exit_rx,
            deadline: None,
            breaker,
        })
    }

    /// Request the GPU thread to shut down.
    /// This should be called before dropping the queue to ensure clean shutdown.
    ///
    /// Uses a timeout to avoid blocking forever if the queue is full and the GPU
    /// thread is hung. If the send times out, the GPU thread is likely unresponsive
    /// and the Drop implementation will handle cleanup via the `exit_rx` timeout.
    pub fn shutdown(&self) {
        // Use timeout to avoid blocking forever if queue is full and GPU thread is hung.
        // 2 seconds is generous - if the GPU thread is responsive, it should drain
        // items much faster. If this times out, proceed to exit_rx timeout in Drop.
        let shutdown_send_timeout = Duration::from_secs(2);
        tracing::debug!("GPU queue: requesting shutdown");
        if self
            .work_tx
            .send_timeout(GpuWorkRequest::Shutdown, shutdown_send_timeout)
            .is_err()
        {
            tracing::trace!("GPU queue: shutdown send failed — GPU thread may be unresponsive");
        }
    }
}

// Note: GPU_SHUTDOWN_TIMEOUT_SECS is imported from gpu/shaders.rs (Issue #277)

impl Drop for GpuWorkQueue {
    fn drop(&mut self) {
        // Request shutdown
        self.shutdown();

        // Wait for GPU thread to exit with timeout
        // This prevents hanging forever if the GPU driver is stuck (e.g., Metal semaphore wait)
        let timeout = Duration::from_secs(GPU_SHUTDOWN_TIMEOUT_SECS);
        match self.exit_rx.recv_timeout(timeout) {
            Ok(()) => {
                // Thread exited cleanly, now safe to join
                if let Some(handle) = self.thread_handle.take() {
                    join_gpu_thread(handle);
                }
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                // GPU thread is stuck (likely in Metal driver)
                // Log warning and abandon the thread - it will be cleaned up on process exit
                tracing::warn!(
                    timeout_secs = GPU_SHUTDOWN_TIMEOUT_SECS,
                    "GPU thread did not exit within timeout — the GPU driver may be hung, \
                     abandoning thread to prevent deadlock. Consider restarting the process."
                );
                // Don't join - the thread is stuck and joining would block forever
                let _ = self.thread_handle.take();
                // Issue #1930: count the leaked thread and trip the breaker, so the
                // next analysis fails fast instead of abandoning another one.
                self.breaker.record_abandoned_thread();
            }
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                // Channel disconnected - thread already exited. `run_guarded_gpu_thread`
                // (Issue #2361) catches a panicking body and still sends on `exit_tx`,
                // so disconnection here means `exit_tx` itself was dropped without
                // sending, not a panic that skipped the signal.
                if let Some(handle) = self.thread_handle.take() {
                    join_gpu_thread(handle);
                }
            }
        }
    }
}

/// Join a GPU thread handle, logging rather than discarding a panic payload
/// (Issue #2361) — the panic itself was already handled by
/// `run_guarded_gpu_thread`, so this only fires for a panic that happened
/// outside that guard (e.g. during unwinding of the guard itself).
fn join_gpu_thread(handle: thread::JoinHandle<()>) {
    if let Err(payload) = handle.join() {
        tracing::error!(
            "GPU thread panicked before shutdown: {}",
            panic_payload_message(&*payload)
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::gpu::breaker::GpuCircuitBreaker;

    #[test]
    fn panic_payload_message_handles_str_payload() {
        let payload: Box<dyn std::any::Any + Send> = Box::new("boom");
        assert_eq!(panic_payload_message(&*payload), "boom");
    }

    #[test]
    fn panic_payload_message_handles_string_payload() {
        let payload: Box<dyn std::any::Any + Send> = Box::new(String::from("boom"));
        assert_eq!(panic_payload_message(&*payload), "boom");
    }

    #[test]
    fn panic_payload_message_handles_non_string_payload() {
        let payload: Box<dyn std::any::Any + Send> = Box::new(42_i32);
        assert_eq!(panic_payload_message(&*payload), "non-string panic payload");
    }

    #[test]
    fn run_guarded_gpu_thread_ok_body_leaves_queue_and_breaker_untouched() {
        let breaker: &'static GpuCircuitBreaker = Box::leak(Box::new(GpuCircuitBreaker::new()));
        let (work_tx, work_rx) = bounded::<GpuWorkRequest>(1);
        work_tx
            .send(GpuWorkRequest::Shutdown)
            .expect("queue has capacity");

        run_guarded_gpu_thread(&work_rx, breaker, || {
            // Body returns normally — no panic.
        });

        assert!(!breaker.is_tripped(), "a clean body must not trip the breaker");
        assert_eq!(
            work_rx.len(),
            1,
            "an untripped guard must not drain the queue"
        );
    }

    #[test]
    fn run_guarded_gpu_thread_drains_shutdown_without_panicking() {
        let breaker: &'static GpuCircuitBreaker = Box::leak(Box::new(GpuCircuitBreaker::new()));
        let (work_tx, work_rx) = bounded::<GpuWorkRequest>(1);
        work_tx
            .send(GpuWorkRequest::Shutdown)
            .expect("queue has capacity");

        run_guarded_gpu_thread(&work_rx, breaker, || {
            panic!("simulated GPU thread panic");
        });

        assert_eq!(
            breaker.trip_reason(),
            Some(GpuTripReason::WorkerPanicked),
            "a panicking body must trip the breaker with WorkerPanicked"
        );
        assert_eq!(
            work_rx.len(),
            0,
            "the guard must drain every queued request after a panic"
        );
    }
}
