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

#[cfg(test)]
mod tests {
    use super::*;

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
}
