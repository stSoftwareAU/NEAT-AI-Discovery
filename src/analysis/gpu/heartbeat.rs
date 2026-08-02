//! GPU-thread liveness heartbeat (Issue #1933).
//!
//! A wedged GPU used to cost a submitter its whole batch timeout — 60–300s of a
//! one-hour run budget — before anything concluded the device was gone, and the
//! verdict was indistinguishable from "the GPU is slow but progressing".
//!
//! The GPU thread now publishes a monotonically increasing progress counter at
//! every observable step (request dequeued, sub-batch submitted, buffer map
//! completed, device poll returning idle, request completed). A submitter waits
//! in a bounded loop and watches that counter: no progress for the configured
//! stall window means the device is wedged and the wait is abandoned in seconds
//! rather than minutes.
//!
//! Crucially, the counter distinguishes *no progress at all* from *slow
//! progress*: a long-running kernel that keeps advancing the counter never trips
//! the guard. Beats are published only when a step actually **completes**, never
//! from inside a poll loop, so a driver spinning forever cannot fake liveness.
//!
//! The absolute batch timeout stays as the backstop for the case where the
//! heartbeat itself cannot be updated.
//!
//! The counter is process-wide, mirroring
//! [`global_gpu_breaker()`](crate::analysis::gpu::breaker::global_gpu_breaker):
//! production runs exactly one GPU thread, and the device helpers that publish
//! the deepest beats are shared by every evaluation path. Tests construct their
//! own [`GpuHeartbeat`] so they never observe another test's progress.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Environment variable configuring the no-progress stall window.
pub const GPU_STALL_WINDOW_ENV: &str = "NEAT_AI_DISCOVERY_GPU_STALL_WINDOW_SECS";

/// Default stall window in seconds — in the spirit of the long-standing
/// `GPU_REQUEST_STALL_WARN_SECS` warning threshold, now a decision rather than
/// an after-the-fact log line.
pub const DEFAULT_GPU_STALL_WINDOW_SECS: u64 = 30;

/// Smallest configurable stall window. Below this a genuinely healthy machine
/// under load would be flagged as wedged.
pub const MIN_GPU_STALL_WINDOW_SECS: u64 = 1;

/// Largest configurable stall window. Above the maximum batch timeout the guard
/// could never fire before the backstop.
pub const MAX_GPU_STALL_WINDOW_SECS: u64 = 600;

/// Step labels published with each beat, used in the trace log and in the
/// stalled-wait diagnostics.
pub const STEP_REQUEST_DEQUEUED: &str = "request_dequeued";
/// A sub-batch command buffer was handed to the GPU queue.
pub const STEP_SUB_BATCH_SUBMITTED: &str = "sub_batch_submitted";
/// One or more staging-buffer mappings completed.
pub const STEP_BUFFER_MAPPED: &str = "buffer_map_completed";
/// `poll_device_until_idle` returned with the device queue drained.
pub const STEP_DEVICE_IDLE: &str = "poll_device_idle";
/// A dequeued request finished and its response was sent.
pub const STEP_REQUEST_COMPLETED: &str = "request_completed";
/// A device-lost recovery attempt started — the thread is alive and working,
/// so recovery must not read as a stall.
pub const STEP_RECOVERY_ATTEMPT: &str = "recovery_attempt";

/// A monotonically increasing count of observable GPU-thread progress steps.
#[derive(Debug)]
pub struct GpuHeartbeat {
    ticks: AtomicU64,
}

impl GpuHeartbeat {
    /// A heartbeat that has published no progress yet.
    pub const fn new() -> Self {
        Self {
            ticks: AtomicU64::new(0),
        }
    }

    /// Publish one step of progress.
    ///
    /// Called only when a step has genuinely completed — never from inside a
    /// polling loop, which would let a wedged driver advertise liveness it does
    /// not have.
    pub fn beat(&self, step: &'static str) {
        let ticks = self.ticks.fetch_add(1, Ordering::Release) + 1;
        tracing::trace!(step, ticks, "GPU heartbeat");
    }

    /// The number of progress steps published so far.
    pub fn ticks(&self) -> u64 {
        self.ticks.load(Ordering::Acquire)
    }
}

impl Default for GpuHeartbeat {
    fn default() -> Self {
        Self::new()
    }
}

/// The heartbeat the production GPU thread publishes to.
pub fn global_gpu_heartbeat() -> &'static GpuHeartbeat {
    static HEARTBEAT: GpuHeartbeat = GpuHeartbeat::new();
    &HEARTBEAT
}

/// Publish a sub-batch submission on the global heartbeat.
pub fn beat_sub_batch_submitted() {
    global_gpu_heartbeat().beat(STEP_SUB_BATCH_SUBMITTED);
}

/// Publish a completed buffer mapping on the global heartbeat.
pub fn beat_buffer_mapped() {
    global_gpu_heartbeat().beat(STEP_BUFFER_MAPPED);
}

/// Publish a drained device queue on the global heartbeat.
pub fn beat_device_idle() {
    global_gpu_heartbeat().beat(STEP_DEVICE_IDLE);
}

/// A submitter's view of the GPU thread's progress.
///
/// Remembers the last tick value it saw and when it saw it, so "no progress for
/// the stall window" is measured from the last *change* rather than from the
/// start of the wait — a slow-but-advancing GPU keeps resetting the clock.
#[derive(Debug)]
pub struct HeartbeatWatch<'a> {
    heartbeat: &'a GpuHeartbeat,
    stall_window: Duration,
    last_ticks: u64,
    last_progress: Instant,
}

impl<'a> HeartbeatWatch<'a> {
    /// Start watching. A zero `stall_window` disables the guard, leaving the
    /// absolute timeout as the only bound.
    pub fn new(heartbeat: &'a GpuHeartbeat, stall_window: Duration) -> Self {
        Self {
            heartbeat,
            stall_window,
            last_ticks: heartbeat.ticks(),
            last_progress: Instant::now(),
        }
    }

    /// How long the GPU thread has published nothing, once that exceeds the
    /// stall window; `None` while it is still making progress.
    pub fn stalled_for(&mut self) -> Option<Duration> {
        let ticks = self.heartbeat.ticks();
        if ticks != self.last_ticks {
            self.last_ticks = ticks;
            self.last_progress = Instant::now();
            return None;
        }
        if self.stall_window.is_zero() {
            return None;
        }
        let idle = self.last_progress.elapsed();
        (idle >= self.stall_window).then_some(idle)
    }

    /// How often the waiter should wake to re-check the heartbeat: a tenth of
    /// the stall window, bounded so a tiny window does not busy-wait and a large
    /// one still notices promptly.
    pub fn poll_interval(&self) -> Duration {
        if self.stall_window.is_zero() {
            return Duration::from_secs(1);
        }
        (self.stall_window / 10).clamp(Duration::from_millis(10), Duration::from_secs(1))
    }

    /// The configured stall window, for the diagnostics on a wedged verdict.
    pub fn stall_window(&self) -> Duration {
        self.stall_window
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beats_advance_the_counter_monotonically() {
        let heartbeat = GpuHeartbeat::new();
        assert_eq!(heartbeat.ticks(), 0, "a fresh heartbeat has no progress");

        heartbeat.beat(STEP_REQUEST_DEQUEUED);
        heartbeat.beat(STEP_SUB_BATCH_SUBMITTED);
        heartbeat.beat(STEP_BUFFER_MAPPED);
        heartbeat.beat(STEP_DEVICE_IDLE);

        assert_eq!(heartbeat.ticks(), 4, "every step publishes one tick");
    }

    #[test]
    fn the_device_step_helpers_publish_on_the_global_heartbeat() {
        let before = global_gpu_heartbeat().ticks();

        beat_sub_batch_submitted();
        beat_buffer_mapped();
        beat_device_idle();

        assert!(
            global_gpu_heartbeat().ticks() >= before + 3,
            "the shared device helpers must advance the global heartbeat"
        );
    }

    #[test]
    fn a_silent_heartbeat_is_reported_as_stalled_after_the_window() {
        let heartbeat = GpuHeartbeat::new();
        let mut watch = HeartbeatWatch::new(&heartbeat, Duration::from_millis(50));

        assert!(watch.stalled_for().is_none(), "not stalled immediately");

        std::thread::sleep(Duration::from_millis(80));

        let stalled = watch.stalled_for().expect("silence past the window stalls");
        assert!(stalled >= Duration::from_millis(50));
    }

    #[test]
    fn slow_progress_keeps_resetting_the_stall_clock() {
        let heartbeat = GpuHeartbeat::new();
        let mut watch = HeartbeatWatch::new(&heartbeat, Duration::from_millis(100));

        // Four beats, each well inside the window: a long kernel that is still
        // advancing must never be flagged.
        for _ in 0..4 {
            std::thread::sleep(Duration::from_millis(30));
            heartbeat.beat(STEP_SUB_BATCH_SUBMITTED);
            assert!(
                watch.stalled_for().is_none(),
                "a slow but advancing heartbeat must not trip the guard"
            );
        }
    }

    #[test]
    fn a_zero_window_disables_the_guard() {
        let heartbeat = GpuHeartbeat::new();
        let mut watch = HeartbeatWatch::new(&heartbeat, Duration::ZERO);

        std::thread::sleep(Duration::from_millis(20));

        assert!(
            watch.stalled_for().is_none(),
            "a zero stall window leaves the absolute timeout as the only bound"
        );
    }

    #[test]
    fn the_poll_interval_stays_within_its_bounds() {
        let heartbeat = GpuHeartbeat::new();

        let tight = HeartbeatWatch::new(&heartbeat, Duration::from_millis(20));
        assert_eq!(
            tight.poll_interval(),
            Duration::from_millis(10),
            "a tiny window must not busy-wait"
        );

        let wide = HeartbeatWatch::new(&heartbeat, Duration::from_secs(300));
        assert_eq!(
            wide.poll_interval(),
            Duration::from_secs(1),
            "a wide window still re-checks every second"
        );

        let default = HeartbeatWatch::new(
            &heartbeat,
            Duration::from_secs(DEFAULT_GPU_STALL_WINDOW_SECS),
        );
        assert_eq!(default.poll_interval(), Duration::from_secs(1));
    }
}
