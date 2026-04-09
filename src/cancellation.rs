//! Global cancellation signal for graceful SIGTERM shutdown (Issue #1047).
//!
//! Provides an `AtomicBool` flag that the TypeScript host can set via FFI
//! (`cancel_analysis`) when SIGTERM arrives. The analysis pipeline checks
//! this flag at existing deadline-check points so that in-flight work
//! stops promptly without racing against parquet file cleanup.

use std::sync::atomic::{AtomicBool, Ordering};

/// Global cancellation flag. Set to `true` by `cancel_analysis()`,
/// checked by `is_cancelled()`, and cleared by `reset_cancellation()`.
static CANCELLED: AtomicBool = AtomicBool::new(false);

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
