//! RAII-based phase timing for analysis profiling.
//!
//! [`PhaseTimer`] emits a debug event with the duration on drop when timing is
//! enabled, while
//! [`ScopedPhaseTimer`] records into a [`super::profile::ProfileData`] instance
//! for structured JSON output.

#![allow(clippy::cast_possible_truncation)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use std::time::Instant;

use super::{profile::ProfileData, timing_enabled};

// =============================================================================
// PhaseTimer - RAII-based phase timing
// =============================================================================

/// RAII-based phase timer that emits a debug event with the duration on drop
/// when timing is enabled.
///
/// When `NEAT_AI_DISCOVERY_TIMING` is set (any value), dropping a `PhaseTimer`
/// emits a `tracing::debug!` event (message `phase timing`) with the `phase` and
/// `duration` fields. It is hidden under the default `RUST_LOG=warn`; use
/// `RUST_LOG=neat_ai_discovery=debug` to see it.
///
/// ## Example
///
/// ```rust,ignore
/// fn analyze_parallel(...) -> Result<...> {
///     let _total = PhaseTimer::new("total_analysis");
///
///     {
///         let _phase = PhaseTimer::new("parquet_loading");
///         load_parquet(...)?;
///     }
///     // Emits a debug event: phase = parquet_loading, duration = 156ms
///
///     // ...
/// }
/// // Emits a debug event: phase = total_analysis, duration = 1234ms
/// ```
pub struct PhaseTimer {
    phase: &'static str,
    start: Instant,
}

impl PhaseTimer {
    /// Create a new phase timer with the given phase name.
    ///
    /// The timer starts immediately. Duration is measured when the timer is dropped.
    #[inline]
    pub fn new(phase: &'static str) -> Self {
        Self {
            phase,
            start: Instant::now(),
        }
    }

    /// Get the elapsed time since the timer was created.
    #[inline]
    pub fn elapsed_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }
}

impl Drop for PhaseTimer {
    fn drop(&mut self) {
        if timing_enabled() {
            let duration = self.start.elapsed();
            tracing::debug!(phase = self.phase, ?duration, "phase timing");
        }
    }
}

// =============================================================================
// Scoped Phase Timer for Profile Integration
// =============================================================================

/// Scoped phase timer that records to a `ProfileData` instance.
///
/// This is a more structured alternative to `PhaseTimer` that integrates
/// with `ProfileData` for JSON output.
pub struct ScopedPhaseTimer<'a> {
    profile: &'a mut ProfileData,
    phase: String,
    start: Instant,
}

impl<'a> ScopedPhaseTimer<'a> {
    /// Create a new scoped phase timer.
    pub fn new(profile: &'a mut ProfileData, phase: &str) -> Self {
        Self {
            profile,
            phase: phase.to_string(),
            start: Instant::now(),
        }
    }
}

impl Drop for ScopedPhaseTimer<'_> {
    fn drop(&mut self) {
        let duration_ms = self.start.elapsed().as_millis() as u64;
        self.profile.record_phase(&self.phase, duration_ms);

        // Also emit via tracing if timing is enabled
        if timing_enabled() {
            tracing::debug!(phase = %self.phase, duration_ms, "scoped phase timing");
        }
    }
}
