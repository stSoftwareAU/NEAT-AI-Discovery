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
use crate::analysis::gpu::breaker::{GpuTripReason, global_gpu_breaker};
use crate::analysis::gpu::shaders::{GPU_INIT_TIMEOUT_SECS, GPU_SHUTDOWN_TIMEOUT_SECS};
use crate::analysis::utils::get_work_queue_capacity;

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
        let thread_handle = thread::spawn(move || {
            tracing::debug!("GPU thread started — initialising GpuAnalyzer");
            // Create analyzer on THIS thread to avoid wgpu thread-local state issues
            match GpuAnalyzer::new() {
                Ok(analyzer) => {
                    tracing::debug!("GPU thread initialisation succeeded");
                    // Signal successful initialisation
                    if init_tx.send(Ok(())).is_err() {
                        tracing::trace!("GPU queue: init receiver dropped before success signal");
                    }
                    // Run the main loop
                    Self::gpu_thread_loop(analyzer, work_rx);
                }
                Err(e) => {
                    tracing::debug!("GPU thread initialisation failed");
                    // Signal initialisation failure
                    if init_tx.send(Err(e)).is_err() {
                        tracing::trace!("GPU queue: init receiver dropped before failure signal");
                    }
                }
            }
            // Always signal exit, even if initialisation failed or loop panicked
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
                return Err(anyhow!(
                    "GPU initialisation timed out after {GPU_INIT_TIMEOUT_SECS}s. \
                     The GPU may be unresponsive or overwhelmed. \
                     Try restarting the process or reducing workload."
                ));
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
                    let _ = handle.join();
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
                // Channel disconnected - thread already exited (possibly via panic)
                if let Some(handle) = self.thread_handle.take() {
                    let _ = handle.join();
                }
            }
        }
    }
}
