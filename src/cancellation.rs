//! Global cancellation signal for graceful SIGTERM shutdown (Issue #1047).
//!
//! Provides an `AtomicBool` flag that the TypeScript host can set via FFI
//! (`cancel_analysis`) when SIGTERM arrives. The analysis pipeline checks
//! this flag at existing deadline-check points so that in-flight work
//! stops promptly without racing against parquet file cleanup.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Global cancellation flag. Set to `true` by `cancel_analysis()`,
/// checked by `is_cancelled()`, and cleared by `reset_cancellation()`.
static CANCELLED: AtomicBool = AtomicBool::new(false);

/// Tracks the number of concurrently active analysis invocations (Issue #1048).
///
/// The TypeScript host can query this via the `is_analysis_active` FFI export
/// to determine whether it is safe to delete the parquet temp directory.
/// Cleanup must wait until this counter reaches zero.
static ANALYSIS_ACTIVE: AtomicUsize = AtomicUsize::new(0);

/// Returns `true` if the host has requested cancellation.
///
/// Called from `deadline_passed()` and the parquet reader's batch loop
/// so that all existing deadline-guarded code paths also respect
/// cancellation.
#[inline]
pub fn is_cancelled() -> bool {
    CANCELLED.load(Ordering::Relaxed)
}

/// Request cancellation of any in-flight analysis.
///
/// This is called via the `cancel_analysis` FFI export when the host
/// process receives SIGTERM. The flag is checked at every point where
/// `deadline_passed()` is called, plus at parquet batch boundaries.
pub fn request_cancellation() {
    CANCELLED.store(true, Ordering::Relaxed);
    tracing::info!("cancellation requested — analysis will stop at the next check point");
}

/// Clear the cancellation flag before starting a new analysis run.
///
/// Must be called at the start of each analysis invocation so that a
/// previous cancellation does not immediately abort the next run.
pub fn reset_cancellation() {
    CANCELLED.store(false, Ordering::Relaxed);
}

// ============================================================================
// Analysis-active tracking (Issue #1048)
// ============================================================================

/// Mark that an analysis invocation has started.
///
/// Called at the entry of `analyze_all` and `rank_focus_neurons_internal`.
/// The host must wait for all active analyses to finish (counter == 0)
/// before deleting the parquet temp directory.
pub fn mark_analysis_started() {
    ANALYSIS_ACTIVE.fetch_add(1, Ordering::Release);
}

/// Mark that an analysis invocation has finished.
///
/// Called when `analyze_all` or `rank_focus_neurons_internal` returns
/// (including error and cancellation paths).
pub fn mark_analysis_finished() {
    // Saturating decrement: avoid underflow if called without a matching start.
    let prev = ANALYSIS_ACTIVE.load(Ordering::Acquire);
    if prev > 0 {
        ANALYSIS_ACTIVE.fetch_sub(1, Ordering::Release);
    }
}

/// Returns `true` if at least one analysis invocation is currently in-flight.
///
/// The TypeScript host should call this (via the `is_analysis_active` FFI
/// export) before deleting the parquet temp directory. If it returns `true`,
/// the host must wait — either by polling or by first calling
/// `cancel_analysis()` and waiting for the FFI call to return.
#[inline]
pub fn is_analysis_active() -> bool {
    ANALYSIS_ACTIVE.load(Ordering::Acquire) > 0
}

/// Returns the current analysis-active count.
///
/// Primarily useful for diagnostics and testing.
pub fn analysis_active_count() -> usize {
    ANALYSIS_ACTIVE.load(Ordering::Acquire)
}

/// Reset the analysis-active counter to zero (for testing only).
pub fn reset_analysis_active() {
    ANALYSIS_ACTIVE.store(0, Ordering::Release);
}

// ============================================================================
// RAII guard for analysis-active tracking (Issue #1077)
// ============================================================================

/// RAII guard that increments the analysis-active counter on creation and
/// decrements it on drop, even if the analysis panics.
///
/// Previously, `mark_analysis_started()` and `mark_analysis_finished()` were
/// called manually before and after each analysis invocation. If the analysis
/// panicked (e.g., from a rayon thread panic propagating through `rayon::join`),
/// `mark_analysis_finished()` was never called. The host process would then
/// wait forever for `is_analysis_active()` to return false, causing the
/// "discovery locked up" symptom described in Issue #1077.
///
/// Using this guard ensures the counter is always decremented via the `Drop`
/// trait, which runs even during stack unwinding from a panic.
pub struct AnalysisActiveGuard {
    _private: (),
}

impl Default for AnalysisActiveGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl AnalysisActiveGuard {
    /// Create a new guard, incrementing the analysis-active counter.
    pub fn new() -> Self {
        mark_analysis_started();
        Self { _private: () }
    }
}

impl Drop for AnalysisActiveGuard {
    fn drop(&mut self) {
        mark_analysis_finished();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    fn test_cancellation_flag_lifecycle() {
        // Start clean
        reset_cancellation();
        assert!(!is_cancelled());

        // Request cancellation
        request_cancellation();
        assert!(is_cancelled());

        // Reset clears the flag
        reset_cancellation();
        assert!(!is_cancelled());
    }

    /// Issue #1077: Verify that `AnalysisActiveGuard` increments on creation
    /// and decrements on drop.
    #[test]
    #[serial]
    fn test_analysis_active_guard_lifecycle() {
        reset_analysis_active();
        assert_eq!(analysis_active_count(), 0);

        {
            let _guard = AnalysisActiveGuard::new();
            assert_eq!(analysis_active_count(), 1);
        }
        // Guard dropped — counter should be back to 0
        assert_eq!(analysis_active_count(), 0);
    }

    /// Issue #1077: Verify that `AnalysisActiveGuard` decrements even when
    /// a panic occurs, preventing the host from waiting forever.
    #[test]
    #[serial]
    fn test_analysis_active_guard_decrements_on_panic() {
        reset_analysis_active();
        assert_eq!(analysis_active_count(), 0);

        let result = std::panic::catch_unwind(|| {
            let _guard = AnalysisActiveGuard::new();
            assert_eq!(analysis_active_count(), 1);
            panic!("simulated analysis panic");
        });

        assert!(result.is_err(), "should have caught the panic");
        // Guard's Drop must have run during unwinding
        assert_eq!(
            analysis_active_count(),
            0,
            "counter must be decremented even after panic"
        );
    }

    /// Issue #1077: Multiple guards can nest correctly.
    #[test]
    #[serial]
    fn test_analysis_active_guard_multiple_guards() {
        reset_analysis_active();

        let _guard1 = AnalysisActiveGuard::new();
        assert_eq!(analysis_active_count(), 1);

        {
            let _guard2 = AnalysisActiveGuard::new();
            assert_eq!(analysis_active_count(), 2);
        }
        assert_eq!(analysis_active_count(), 1);

        drop(_guard1);
        assert_eq!(analysis_active_count(), 0);
    }
}
