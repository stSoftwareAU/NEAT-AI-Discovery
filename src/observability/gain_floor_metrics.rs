//! Counter for candidates dropped by the absolute minimum-expected-gain
//! floor (Issue #1191).
//!
//! [`GainFloorMetrics`] uses an atomic counter so post-processing in both
//! the neuron and synapse paths can record drops without coordination. A
//! global instance is available via [`global_gain_floor_metrics()`].

use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Thread-safe counter for candidates removed by the
/// `MIN_EXPECTED_CREATURE_SCORE_GAIN` floor (Issue #1191).
///
/// Tracks the total number of add-neuron and add-synapse candidates
/// dropped because their `expected_creature_score_gain` was below the
/// configured floor. Operators read the counter via the global instance to
/// see how many candidates the floor removes per discovery batch.
#[derive(Debug)]
pub struct GainFloorMetrics {
    candidates_below_gain_floor: AtomicUsize,
}

impl GainFloorMetrics {
    /// Create a new counter starting at zero.
    #[must_use]
    pub fn new() -> Self {
        Self {
            candidates_below_gain_floor: AtomicUsize::new(0),
        }
    }

    /// Record that `count` candidates were dropped by the floor.
    #[inline]
    pub fn record_dropped(&self, count: usize) {
        if count == 0 {
            return;
        }
        self.candidates_below_gain_floor
            .fetch_add(count, Ordering::Relaxed);
    }

    /// Read the running total.
    #[inline]
    #[must_use]
    pub fn candidates_below_gain_floor_total(&self) -> usize {
        self.candidates_below_gain_floor.load(Ordering::Relaxed)
    }

    /// Reset the counter to zero (test-only).
    #[cfg(test)]
    pub fn reset(&self) {
        self.candidates_below_gain_floor.store(0, Ordering::Relaxed);
    }
}

impl Default for GainFloorMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Global counter shared across the analysis pipeline (Issue #1191).
static GLOBAL_GAIN_FLOOR_METRICS: OnceLock<GainFloorMetrics> = OnceLock::new();

/// Get the global gain-floor metrics instance.
pub fn global_gain_floor_metrics() -> &'static GainFloorMetrics {
    GLOBAL_GAIN_FLOOR_METRICS.get_or_init(GainFloorMetrics::new)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_dropped_accumulates() {
        let metrics = GainFloorMetrics::new();
        assert_eq!(metrics.candidates_below_gain_floor_total(), 0);
        metrics.record_dropped(3);
        metrics.record_dropped(2);
        assert_eq!(metrics.candidates_below_gain_floor_total(), 5);
    }

    #[test]
    fn record_dropped_zero_is_noop() {
        let metrics = GainFloorMetrics::new();
        metrics.record_dropped(0);
        assert_eq!(metrics.candidates_below_gain_floor_total(), 0);
    }

    #[test]
    fn global_instance_is_shared() {
        // Capture before-state so the test does not assume a fresh global.
        let start = global_gain_floor_metrics().candidates_below_gain_floor_total();
        global_gain_floor_metrics().record_dropped(7);
        assert_eq!(
            global_gain_floor_metrics().candidates_below_gain_floor_total(),
            start + 7
        );
    }
}
