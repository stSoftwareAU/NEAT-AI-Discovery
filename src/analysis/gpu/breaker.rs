//! Process-wide GPU circuit breaker (Issue #1930).
//!
//! There used to be no memory that the GPU had already wedged: every analysis
//! called [`GpuWorkQueue::new`](crate::analysis::gpu::GpuWorkQueue::new), which
//! spawned a fresh GPU thread with its own `wgpu` device against the same dead
//! hardware, then sat out a full 60–300s batch timeout before failing. A single
//! run could abandon three GPU threads and burn ~30 minutes of wall clock that
//! way.
//!
//! This module is the defence in depth: the first sign that the GPU is wedged
//! trips the breaker, and from then on every GPU queue creation and every
//! submission fails immediately instead of waiting again.
//!
//! The breaker keys off explicit trip sites — [`GpuTripReason`] — never off
//! error-message matching. `is_device_lost_error()` does not recognise the
//! batch-timeout wording ("The GPU may be unresponsive"), so string matching
//! would silently miss the very failure this exists to stop.
//!
//! Recovery is deliberately out of scope: nothing here restarts the process.
//! That stays with the external supervisor. The breaker is one-way for the life
//! of the process, apart from [`GpuCircuitBreaker::reset`], which exists so
//! tests are not order-dependent (mirroring
//! [`reset_cancellation`](crate::cancellation::reset_cancellation)).
//!
//! ## Why the breaker is a value, not just a global
//!
//! Production uses exactly one breaker — [`global_gpu_breaker()`] — which
//! [`GpuWorkQueue`](crate::analysis::gpu::GpuWorkQueue) holds a reference to.
//! Tests can point a queue at an isolated instance instead, so exercising the
//! tripped path does not refuse GPU work for every other test in the process.
//! The abandoned-thread count deliberately stays in the one place production
//! reads it from: [`global_gpu_metrics()`].

use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

use anyhow::Result;

use crate::ffi_types::DiscoveryError;
use crate::observability::global_gpu_metrics;

/// Why the GPU circuit breaker tripped.
///
/// Each variant corresponds to one explicit trip site; the breaker never infers
/// a trip from an error message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuTripReason {
    /// A GPU thread did not exit within `GPU_SHUTDOWN_TIMEOUT_SECS` and was
    /// abandoned by `Drop` — its device, buffer pools and command buffers are
    /// leaked for the life of the process.
    AbandonedThread,
    /// A batch submission timed out: either the work queue never accepted the
    /// request, or the GPU thread never answered it.
    BatchTimeout,
    /// GPU queue creation timed out waiting for the analyser to initialise.
    InitTimeout,
    /// The GPU thread published no progress for the configured stall window
    /// while a submitter was waiting on it (Issue #1933).
    HeartbeatStall,
}

/// Sentinel for "not tripped".
const REASON_UNTRIPPED: u8 = 0;
const REASON_ABANDONED_THREAD: u8 = 1;
const REASON_BATCH_TIMEOUT: u8 = 2;
const REASON_INIT_TIMEOUT: u8 = 3;
const REASON_HEARTBEAT_STALL: u8 = 4;

impl GpuTripReason {
    /// Stable discriminant used for the atomic representation.
    const fn code(self) -> u8 {
        match self {
            Self::AbandonedThread => REASON_ABANDONED_THREAD,
            Self::BatchTimeout => REASON_BATCH_TIMEOUT,
            Self::InitTimeout => REASON_INIT_TIMEOUT,
            Self::HeartbeatStall => REASON_HEARTBEAT_STALL,
        }
    }

    /// Inverse of [`Self::code`]; `None` means the breaker has not tripped.
    const fn from_code(code: u8) -> Option<Self> {
        match code {
            REASON_ABANDONED_THREAD => Some(Self::AbandonedThread),
            REASON_BATCH_TIMEOUT => Some(Self::BatchTimeout),
            REASON_INIT_TIMEOUT => Some(Self::InitTimeout),
            REASON_HEARTBEAT_STALL => Some(Self::HeartbeatStall),
            _ => None,
        }
    }

    /// Human-readable trip reason, carried in the breaker error and the log.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AbandonedThread => {
                "a GPU thread did not exit within the shutdown timeout and was abandoned"
            }
            Self::BatchTimeout => "a GPU batch submission timed out",
            Self::InitTimeout => "GPU initialisation timed out",
            Self::HeartbeatStall => {
                "the GPU thread stopped publishing progress while a submitter waited"
            }
        }
    }
}

impl std::fmt::Display for GpuTripReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A one-way latch that remembers the GPU has wedged.
///
/// The trip reason *is* the tripped flag — a single atomic — so a reader can
/// never observe a trip whose reason has not landed yet.
#[derive(Debug)]
pub struct GpuCircuitBreaker {
    reason: AtomicU8,
    /// Latch for [`Self::warn_analyses_skipped`] — one warn per run, not one
    /// per analysis attempt (Issue #1931).
    skip_warned: AtomicBool,
}

impl GpuCircuitBreaker {
    /// A closed breaker.
    pub const fn new() -> Self {
        Self {
            reason: AtomicU8::new(REASON_UNTRIPPED),
            skip_warned: AtomicBool::new(false),
        }
    }

    /// Returns `true` once the GPU has been declared wedged.
    #[inline]
    pub fn is_tripped(&self) -> bool {
        self.reason.load(Ordering::Acquire) != REASON_UNTRIPPED
    }

    /// The reason this breaker tripped, or `None` while it is closed.
    #[inline]
    pub fn trip_reason(&self) -> Option<GpuTripReason> {
        GpuTripReason::from_code(self.reason.load(Ordering::Acquire))
    }

    /// Trip the breaker, logging **once** at `warn` with the reason.
    ///
    /// The first caller wins: its reason is the one the breaker keeps, and only
    /// it logs at `warn`. Every later trip attempt logs at `debug`, so a wedged
    /// GPU producing failure after failure cannot flood the log.
    pub fn trip(&self, reason: GpuTripReason) {
        match self.reason.compare_exchange(
            REASON_UNTRIPPED,
            reason.code(),
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => {
                tracing::warn!(
                    reason = reason.as_str(),
                    abandoned_gpu_threads = abandoned_gpu_thread_count(),
                    "GPU circuit breaker tripped — no further GPU work will be attempted \
                     in this process. Restart the process to use the GPU again."
                );
            }
            Err(existing) => {
                tracing::debug!(
                    reason = reason.as_str(),
                    original_reason = GpuTripReason::from_code(existing).map(GpuTripReason::as_str),
                    "GPU circuit breaker already tripped — trip suppressed"
                );
            }
        }
    }

    /// Record that a GPU thread was abandoned, and trip the breaker.
    ///
    /// The count is incremented before the trip so the single `warn` reports it.
    pub fn record_abandoned_thread(&self) {
        global_gpu_metrics().record_abandoned_thread();
        self.trip(GpuTripReason::AbandonedThread);
    }

    /// Fail immediately if the GPU has already been declared wedged.
    ///
    /// Call this at the head of every path that would otherwise spawn a GPU
    /// thread or start a multi-minute wait. Suppressed calls log at `debug`
    /// only.
    pub fn check(&self) -> Result<()> {
        match self.trip_reason() {
            None => Ok(()),
            Some(reason) => {
                tracing::debug!(
                    reason = reason.as_str(),
                    "GPU circuit breaker is tripped — refusing GPU work without waiting"
                );
                Err(self.error(reason))
            }
        }
    }

    /// Announce **once per run** that the analyses are skipping their GPU work
    /// (Issue #1931).
    ///
    /// [`Self::trip`] explains why the GPU was declared wedged; this explains
    /// what the analyses did about it. It is separate because the trip happens
    /// once, deep inside the queue, whereas the skip is decided at every
    /// analysis entry point — without this latch a run would emit one identical
    /// warning per analysis attempt.
    ///
    /// There is no CPU fallback for these analyses (Issue #1419), so a skipped
    /// pass is genuinely empty rather than a zero-candidate success.
    pub fn warn_analyses_skipped(&self, reason: GpuTripReason) {
        if self.skip_warned.swap(true, Ordering::AcqRel) {
            tracing::debug!(
                reason = reason.as_str(),
                "GPU analyses skipped — already warned once this run"
            );
            return;
        }
        tracing::warn!(
            reason = reason.as_str(),
            abandoned_gpu_threads = abandoned_gpu_thread_count(),
            "GPU analyses skipped for the rest of this run because the GPU is wedged — \
             there is no CPU fallback, so these results are genuinely empty rather than a \
             zero-candidate success. An external process restart is required to use the \
             GPU again."
        );
    }

    /// The error every suppressed GPU entry point returns: the original trip
    /// reason plus the abandoned-thread count.
    fn error(&self, reason: GpuTripReason) -> anyhow::Error {
        gpu_wedged_error(format!("GPU circuit breaker tripped: {reason}"))
    }

    /// Clear the breaker and the abandoned-thread count (testing only).
    ///
    /// Production never resets the breaker — a wedged GPU stays wedged until
    /// the supervisor restarts the process. This mirrors
    /// [`reset_cancellation`](crate::cancellation::reset_cancellation) so tests
    /// that trip the breaker are not order-dependent.
    pub fn reset(&self) {
        self.reason.store(REASON_UNTRIPPED, Ordering::Release);
        self.skip_warned.store(false, Ordering::Release);
        global_gpu_metrics().reset_abandoned_threads();
    }
}

impl Default for GpuCircuitBreaker {
    fn default() -> Self {
        Self::new()
    }
}

/// Build the typed wedged-GPU error every trip site returns (Issue #1932).
///
/// The typed [`DiscoveryError::GpuWedged`] is what makes the failure
/// non-retryable at the FFI boundary: it classifies as
/// [`DiscoveryErrorKind::GpuWedged`](crate::ffi_types::DiscoveryErrorKind::GpuWedged),
/// so the host stops extending the deadline instead of reading the timeout
/// wording as "retry with longer". The abandoned-thread count is stamped from
/// the same counter [`global_gpu_metrics()`] publishes.
pub fn gpu_wedged_error(detail: impl Into<String>) -> anyhow::Error {
    anyhow::Error::new(DiscoveryError::GpuWedged {
        detail: detail.into(),
        abandoned_threads: abandoned_gpu_thread_count(),
    })
}

/// The one breaker production uses.
static GLOBAL_GPU_BREAKER: GpuCircuitBreaker = GpuCircuitBreaker::new();

/// The process-wide GPU circuit breaker.
///
/// Every `GpuWorkQueue` created by `GpuWorkQueue::new()` consults this one, so
/// a trip anywhere stops GPU work everywhere in the process.
#[inline]
pub fn global_gpu_breaker() -> &'static GpuCircuitBreaker {
    &GLOBAL_GPU_BREAKER
}

/// Number of GPU threads abandoned because they would not exit.
///
/// Read from the same counter that
/// [`report_global_gpu_metrics`](crate::observability::report_global_gpu_metrics)
/// prints, so run telemetry carries the number without extra plumbing.
#[inline]
pub fn abandoned_gpu_thread_count() -> usize {
    global_gpu_metrics().abandoned_threads()
}

/// Returns `true` if the process-wide breaker has tripped.
#[inline]
pub fn is_gpu_breaker_tripped() -> bool {
    global_gpu_breaker().is_tripped()
}

/// The reason the process-wide breaker tripped, or `None`.
#[inline]
pub fn gpu_breaker_trip_reason() -> Option<GpuTripReason> {
    global_gpu_breaker().trip_reason()
}

/// Trip the process-wide breaker.
pub fn trip_gpu_breaker(reason: GpuTripReason) {
    global_gpu_breaker().trip(reason);
}

/// Record an abandoned GPU thread and trip the process-wide breaker.
pub fn record_abandoned_gpu_thread() {
    global_gpu_breaker().record_abandoned_thread();
}

/// Fail immediately if the process-wide breaker has tripped.
pub fn check_gpu_breaker() -> Result<()> {
    global_gpu_breaker().check()
}

/// The reason the analyses must skip their GPU work, or `None` while the GPU is
/// healthy (Issue #1931).
///
/// Announces the skip exactly once per run as a side effect, so every analysis
/// entry point can call this unconditionally without flooding the log.
#[must_use]
pub fn gpu_wedged_skip_reason() -> Option<GpuTripReason> {
    let breaker = global_gpu_breaker();
    let reason = breaker.trip_reason()?;
    breaker.warn_analyses_skipped(reason);
    Some(reason)
}

/// Clear the process-wide breaker (testing only).
pub fn reset_gpu_breaker() {
    global_gpu_breaker().reset();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// These tests use isolated breakers on purpose: tripping the *global* one
    /// here would refuse GPU work for every other test in this binary. The
    /// global wiring is covered in `tests/gpu/issue_1930_gpu_circuit_breaker.rs`,
    /// where every GPU test in the binary is serialised.
    fn breaker() -> GpuCircuitBreaker {
        GpuCircuitBreaker::new()
    }

    #[test]
    fn a_new_breaker_is_closed() {
        let breaker = breaker();
        assert!(!breaker.is_tripped());
        assert!(breaker.trip_reason().is_none());
        assert!(breaker.check().is_ok(), "a closed breaker allows work");
    }

    #[test]
    fn tripping_records_the_reason_and_blocks_work() {
        let breaker = breaker();
        breaker.trip(GpuTripReason::BatchTimeout);

        assert!(breaker.is_tripped());
        assert_eq!(breaker.trip_reason(), Some(GpuTripReason::BatchTimeout));

        let msg = format!("{:#}", breaker.check().expect_err("work is refused"));
        assert!(msg.contains("GPU circuit breaker tripped"), "got: {msg}");
        assert!(
            msg.contains(GpuTripReason::BatchTimeout.as_str()),
            "got: {msg}"
        );
        assert!(msg.contains("abandoned GPU threads:"), "got: {msg}");
    }

    /// The first trip owns the diagnosis — a later, different trip must not
    /// overwrite what actually went wrong first.
    #[test]
    fn the_first_trip_reason_is_kept() {
        let breaker = breaker();
        breaker.trip(GpuTripReason::InitTimeout);
        breaker.trip(GpuTripReason::BatchTimeout);
        breaker.trip(GpuTripReason::AbandonedThread);
        assert_eq!(breaker.trip_reason(), Some(GpuTripReason::InitTimeout));
    }

    /// The count itself is asserted in `tests/gpu/issue_1930_gpu_circuit_breaker.rs`,
    /// where the whole binary is serialised — the shared metrics counter would
    /// race with parallel tests here.
    #[test]
    fn abandoning_a_thread_trips_the_breaker() {
        let breaker = breaker();
        breaker.record_abandoned_thread();
        assert_eq!(breaker.trip_reason(), Some(GpuTripReason::AbandonedThread));
    }

    /// Issue #1932: the suppressed-call error is typed, so the FFI boundary
    /// classifies it as wedged (non-retryable) instead of as a plain timeout.
    #[test]
    fn a_suppressed_call_returns_a_typed_wedged_error() {
        use crate::ffi_types::{DiscoveryErrorKind, classify_anyhow_error};

        let breaker = breaker();
        breaker.trip(GpuTripReason::BatchTimeout);

        let err = breaker.check().expect_err("work is refused");
        assert_eq!(classify_anyhow_error(&err), DiscoveryErrorKind::GpuWedged);
        assert!(!DiscoveryErrorKind::GpuWedged.is_retryable());
    }

    #[test]
    fn reset_restores_the_closed_state() {
        let breaker = breaker();
        breaker.trip(GpuTripReason::AbandonedThread);
        breaker.warn_analyses_skipped(GpuTripReason::AbandonedThread);
        assert!(breaker.is_tripped());

        breaker.reset();

        assert!(!breaker.is_tripped(), "reset must close the breaker");
        assert!(breaker.trip_reason().is_none());
        assert!(breaker.check().is_ok());
        assert!(
            !breaker.skip_warned.load(Ordering::Acquire),
            "reset must re-arm the skip warning so tests are not order-dependent"
        );
    }

    /// The skip warning latches after the first call — the log-level assertion
    /// lives in `tests/issue_1931_gpu_breaker_partial_result.rs`, which captures
    /// the emitted events.
    #[test]
    fn the_skip_warning_latches_after_the_first_call() {
        let breaker = breaker();
        breaker.trip(GpuTripReason::BatchTimeout);

        assert!(!breaker.skip_warned.load(Ordering::Acquire));
        breaker.warn_analyses_skipped(GpuTripReason::BatchTimeout);
        assert!(breaker.skip_warned.load(Ordering::Acquire));
        breaker.warn_analyses_skipped(GpuTripReason::BatchTimeout);
        assert!(breaker.skip_warned.load(Ordering::Acquire));
    }

    /// Reason codes must round-trip through the atomic representation, and the
    /// untripped sentinel must never decode to a reason.
    #[test]
    fn reason_codes_round_trip() {
        for reason in [
            GpuTripReason::AbandonedThread,
            GpuTripReason::BatchTimeout,
            GpuTripReason::InitTimeout,
        ] {
            assert_eq!(GpuTripReason::from_code(reason.code()), Some(reason));
            assert!(!reason.as_str().is_empty());
            assert_eq!(reason.to_string(), reason.as_str());
        }
        assert!(GpuTripReason::from_code(REASON_UNTRIPPED).is_none());
        assert!(GpuTripReason::from_code(u8::MAX).is_none());
    }

    /// Concurrent trips must settle on exactly one reason — the
    /// compare-exchange, not a check-then-set, is what guarantees it.
    #[test]
    fn concurrent_trips_settle_on_one_reason() {
        use std::sync::{Arc, Barrier};

        for _ in 0..200 {
            let breaker = Arc::new(breaker());
            let barrier = Arc::new(Barrier::new(2));

            let handles: Vec<_> = [GpuTripReason::BatchTimeout, GpuTripReason::InitTimeout]
                .into_iter()
                .map(|reason| {
                    let breaker = Arc::clone(&breaker);
                    let barrier = Arc::clone(&barrier);
                    std::thread::spawn(move || {
                        barrier.wait();
                        breaker.trip(reason);
                    })
                })
                .collect();
            for handle in handles {
                handle.join().expect("trip thread panicked");
            }

            let reason = breaker.trip_reason().expect("breaker must be tripped");
            assert!(
                reason == GpuTripReason::BatchTimeout || reason == GpuTripReason::InitTimeout,
                "the winning reason must be one of the two trips"
            );
        }
    }

    /// The free functions must operate on the one breaker production uses —
    /// otherwise a trip in `Drop` would not stop a submission elsewhere. Read
    /// only: tripping the global here would refuse GPU work for the rest of the
    /// binary.
    #[test]
    fn the_free_functions_delegate_to_the_global_breaker() {
        assert!(
            std::ptr::eq(global_gpu_breaker(), global_gpu_breaker()),
            "there is exactly one global breaker"
        );
        assert_eq!(is_gpu_breaker_tripped(), global_gpu_breaker().is_tripped());
        assert_eq!(
            gpu_breaker_trip_reason(),
            global_gpu_breaker().trip_reason()
        );
        assert_eq!(
            check_gpu_breaker().is_ok(),
            global_gpu_breaker().check().is_ok()
        );
    }
}
