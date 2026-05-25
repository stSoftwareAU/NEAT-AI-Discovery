//! Rolling per-creature failure window for the drought diagnostic (Issue #1274).
//!
//! The [`CandidateOutcomeCache`](super::candidate_cache::CandidateOutcomeCache)
//! tracks which `(source, target, operation)` triples failed and when, so they
//! can be suppressed. That cache is keyed by candidate identity and replaces
//! older entries; it does not preserve the **shape** of recent failures —
//! which module emitted them, which target neuron they pointed at, how many
//! operations they bundled, and whether the predicted gain matched reality.
//!
//! This module fills that gap. [`RecentFailureWindow`] keeps a bounded, FIFO
//! log of the most recent failed candidates, capturing just enough metadata
//! for the drought diagnostic to aggregate dominant-failure patterns. A single
//! FFI payload then surfaces what previously required hand-inspecting the
//! per-failure cache (see Issue #1274 evidence).
//!
//! # Sizing
//!
//! The window holds the last [`DEFAULT_FAILURE_WINDOW_CAPACITY`] failures per
//! creature. New entries push the oldest out. Aggregate computation walks the
//! window once, so it is `O(window_size)` per drought emission — never more
//! than ~50 entries by default.
//!
//! # Aggregate emission rule
//!
//! The aggregate fields populate only once the window holds at least
//! [`MIN_FAILURES_FOR_AGGREGATES`] entries. Smaller samples would produce
//! noisy "dominant" values (a window of two reports a 100 % share regardless
//! of true distribution), so the diagnostic returns `None` / `0.0` instead.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// Default capacity of the rolling failure window.
///
/// The drought operator inspection that motivated this issue covered around
/// 100 failures; 50 is enough to identify a dominant pattern without ballooning
/// the per-creature memory footprint.
pub const DEFAULT_FAILURE_WINDOW_CAPACITY: usize = 50;

/// Minimum number of failures required before dominant-pattern aggregates are
/// reported. Below this, the diagnostic exposes `None` / `0.0`.
pub const MIN_FAILURES_FOR_AGGREGATES: usize = 5;

/// One failure record retained by the rolling window.
///
/// Captures just the fields the drought diagnostic aggregates over. The
/// `module` string follows the same convention as the discovery module names
/// used by [`module_weights`](super::module_weights) (e.g.
/// `"coordinated-structural"`, `"add-neurons"`, `"add-synapses"`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecentFailureRecord {
    /// Discovery module that emitted the candidate (e.g. `"coordinated-structural"`).
    pub module: String,
    /// UUID of the target neuron the candidate operated on.
    pub target_uuid: String,
    /// Number of atomic operations in the candidate. For
    /// `coordinatedStructural` this is the size of the operation group;
    /// for single-op candidates it is `1`.
    pub operation_count: u32,
    /// `expected_creature_score_gain` reported when the candidate was emitted.
    /// Stored as-is so the diagnostic can compute the predicted-vs-actual
    /// gap without re-deriving it.
    pub predicted_gain: f32,
    /// Measured `actualErrorReduction` (or equivalent) when the candidate was
    /// ablated. Negative when the candidate made things worse.
    pub actual_gain: f32,
}

/// Rolling, bounded log of recent failed candidates for a creature.
///
/// The window evicts the oldest entry once `capacity` is reached, so memory
/// is `O(capacity)` regardless of how long the creature runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecentFailureWindow {
    capacity: usize,
    entries: VecDeque<RecentFailureRecord>,
}

impl Default for RecentFailureWindow {
    fn default() -> Self {
        Self::with_capacity(DEFAULT_FAILURE_WINDOW_CAPACITY)
    }
}

impl RecentFailureWindow {
    /// Creates an empty window using the default capacity.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an empty window with a custom capacity.
    ///
    /// A capacity of `0` is coerced to `1` so the window can always accept
    /// at least one record.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            capacity,
            entries: VecDeque::with_capacity(capacity),
        }
    }

    /// Returns the configured capacity (maximum entries retained).
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the current number of records.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when no records are held.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns an iterator over the records, oldest first.
    pub fn iter(&self) -> impl Iterator<Item = &RecentFailureRecord> {
        self.entries.iter()
    }

    /// Pushes a new failure record onto the window, evicting the oldest entry
    /// when the capacity is reached.
    pub fn record(&mut self, record: RecentFailureRecord) {
        if self.entries.len() == self.capacity {
            self.entries.pop_front();
        }
        self.entries.push_back(record);
    }

    /// Computes the dominant-pattern aggregates for the diagnostic.
    ///
    /// Returns `None` when fewer than [`MIN_FAILURES_FOR_AGGREGATES`] entries
    /// are held — the caller should leave the diagnostic's dominant fields
    /// unset/zero in that case.
    #[must_use]
    pub fn aggregates(&self) -> Option<FailureAggregates> {
        if self.entries.len() < MIN_FAILURES_FOR_AGGREGATES {
            return None;
        }

        // The window capacity is bounded (default 50, hard floor 1), so the
        // entry count always fits in an f32 mantissa without precision loss.
        #[allow(clippy::cast_precision_loss)]
        let total = self.entries.len() as f32;

        // Dominant module.
        let (dominant_module, dominant_module_count) =
            mode_string(self.entries.iter().map(|r| r.module.as_str()))
                .unwrap_or_else(|| (String::new(), 0));

        // Dominant target.
        let (dominant_target, dominant_target_count) =
            mode_string(self.entries.iter().map(|r| r.target_uuid.as_str()))
                .unwrap_or_else(|| (String::new(), 0));

        // Dominant op count.
        let dominant_op_count =
            mode_u32(self.entries.iter().map(|r| r.operation_count)).map(|(value, _count)| value);

        // Median of `actual / predicted` ratio. We compute the per-record ratio
        // and then take the median to be robust to outliers (a single
        // 1000x-wrong prediction would dominate the mean).
        let mut ratios: Vec<f32> = self
            .entries
            .iter()
            .filter_map(|r| safe_ratio(r.actual_gain, r.predicted_gain))
            .collect();
        let predicted_vs_actual_gap_p50 = if ratios.is_empty() {
            0.0
        } else {
            // `partial_cmp` is sufficient — NaN is filtered by `safe_ratio`.
            ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            median_sorted(&ratios)
        };

        // Counts are bounded by the window capacity (default 50) so they
        // always fit in an f32 mantissa.
        #[allow(clippy::cast_precision_loss)]
        let dominant_module_share = dominant_module_count as f32 / total;
        #[allow(clippy::cast_precision_loss)]
        let dominant_target_share = dominant_target_count as f32 / total;

        Some(FailureAggregates {
            dominant_module: if dominant_module.is_empty() {
                None
            } else {
                Some(dominant_module)
            },
            dominant_module_share,
            dominant_target_uuid: if dominant_target.is_empty() {
                None
            } else {
                Some(dominant_target)
            },
            dominant_target_share,
            dominant_operation_count: dominant_op_count,
            predicted_vs_actual_gap_p50,
        })
    }
}

/// Result of [`RecentFailureWindow::aggregates`].
///
/// All fields are populated together — the caller either gets the full set or
/// `None`. Field names match the camelCase JSON keys on
/// [`DroughtDiagnostic`](super::drought_diagnostic::DroughtDiagnostic) so the
/// orchestrator can plumb them straight through.
#[derive(Debug, Clone, PartialEq)]
pub struct FailureAggregates {
    pub dominant_module: Option<String>,
    pub dominant_module_share: f32,
    pub dominant_target_uuid: Option<String>,
    pub dominant_target_share: f32,
    pub dominant_operation_count: Option<u32>,
    pub predicted_vs_actual_gap_p50: f32,
}

/// Returns the most-common `&str` and its count. Ties resolve to whichever
/// value `HashMap` iteration encounters first; for our purposes ties on a
/// tiny window are not meaningfully ambiguous.
fn mode_string<'a, I>(values: I) -> Option<(String, u32)>
where
    I: Iterator<Item = &'a str>,
{
    let mut counts: HashMap<&'a str, u32> = HashMap::new();
    for value in values {
        *counts.entry(value).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .max_by_key(|&(_, count)| count)
        .map(|(value, count)| (value.to_string(), count))
}

/// Returns the most-common `u32` and its count.
fn mode_u32<I>(values: I) -> Option<(u32, u32)>
where
    I: Iterator<Item = u32>,
{
    let mut counts: HashMap<u32, u32> = HashMap::new();
    for value in values {
        *counts.entry(value).or_insert(0) += 1;
    }
    counts.into_iter().max_by_key(|&(_, count)| count)
}

/// Computes `actual / predicted` safely. Returns `None` when the denominator
/// is zero or either value is not finite, since the ratio is undefined and
/// would otherwise poison the median.
fn safe_ratio(actual: f32, predicted: f32) -> Option<f32> {
    if !actual.is_finite() || !predicted.is_finite() {
        return None;
    }
    if predicted == 0.0 {
        return None;
    }
    let ratio = actual / predicted;
    if ratio.is_finite() { Some(ratio) } else { None }
}

/// Median of an already-sorted slice. Even-length slices return the mean of
/// the two central values.
fn median_sorted(sorted: &[f32]) -> f32 {
    debug_assert!(!sorted.is_empty());
    let n = sorted.len();
    if n.is_multiple_of(2) {
        let a = sorted[n / 2 - 1];
        let b = sorted[n / 2];
        (a + b) / 2.0
    } else {
        sorted[n / 2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(
        module: &str,
        target: &str,
        op_count: u32,
        predicted: f32,
        actual: f32,
    ) -> RecentFailureRecord {
        RecentFailureRecord {
            module: module.to_string(),
            target_uuid: target.to_string(),
            operation_count: op_count,
            predicted_gain: predicted,
            actual_gain: actual,
        }
    }

    #[test]
    fn empty_window_returns_no_aggregates() {
        let window = RecentFailureWindow::new();
        assert!(window.aggregates().is_none());
    }

    #[test]
    fn partial_window_below_minimum_returns_no_aggregates() {
        let mut window = RecentFailureWindow::new();
        for i in 0..(MIN_FAILURES_FOR_AGGREGATES - 1) {
            window.record(record(
                "coordinated-structural",
                "t1",
                4,
                1e-7,
                -1e-4 - f32::from(u16::try_from(i).unwrap_or(0)) * 1e-5,
            ));
        }
        assert_eq!(window.len(), MIN_FAILURES_FOR_AGGREGATES - 1);
        assert!(window.aggregates().is_none());
    }

    #[test]
    fn full_window_emits_dominant_pattern() {
        // 9× coordinated-structural / 1× add-synapses, all on the same target,
        // mostly 4-op. Mirrors the bcbca347 evidence used by the issue.
        let mut window = RecentFailureWindow::new();
        for _ in 0..9 {
            window.record(record("coordinated-structural", "tgt-A", 4, 5e-7, -5e-4));
        }
        window.record(record("add-synapses", "tgt-A", 1, 1e-6, -1e-5));

        let aggs = window.aggregates().expect("at least 5 entries");
        assert_eq!(
            aggs.dominant_module.as_deref(),
            Some("coordinated-structural")
        );
        assert!((aggs.dominant_module_share - 0.9).abs() < 1e-6);
        assert_eq!(aggs.dominant_target_uuid.as_deref(), Some("tgt-A"));
        assert!((aggs.dominant_target_share - 1.0).abs() < 1e-6);
        assert_eq!(aggs.dominant_operation_count, Some(4));
        // All ratios are actual/predicted = -1000 (-5e-4 / 5e-7) or
        // -10 (-1e-5 / 1e-6); median of 10 values is the average of the
        // 5th and 6th sorted values — both -1000 here.
        assert!(aggs.predicted_vs_actual_gap_p50 < -100.0);
    }

    #[test]
    fn all_same_target_window_reports_full_share() {
        let mut window = RecentFailureWindow::new();
        for _ in 0..5 {
            window.record(record("coordinated-structural", "tgt-X", 4, 1.0, -1.0));
        }
        let aggs = window.aggregates().expect("5 entries");
        assert_eq!(aggs.dominant_target_uuid.as_deref(), Some("tgt-X"));
        assert!((aggs.dominant_target_share - 1.0).abs() < 1e-6);
    }

    #[test]
    fn mixed_modules_window_picks_majority() {
        let mut window = RecentFailureWindow::new();
        for _ in 0..3 {
            window.record(record("add-neurons", "tgt-A", 1, 1.0, -1.0));
        }
        for _ in 0..2 {
            window.record(record("coordinated-structural", "tgt-B", 4, 1.0, -1.0));
        }
        let aggs = window.aggregates().expect("5 entries");
        assert_eq!(aggs.dominant_module.as_deref(), Some("add-neurons"));
        assert!((aggs.dominant_module_share - 0.6).abs() < 1e-6);
        assert_eq!(aggs.dominant_operation_count, Some(1));
    }

    #[test]
    fn capacity_evicts_oldest_entry() {
        let mut window = RecentFailureWindow::with_capacity(3);
        window.record(record("a", "t", 1, 1.0, -1.0));
        window.record(record("b", "t", 1, 1.0, -1.0));
        window.record(record("c", "t", 1, 1.0, -1.0));
        window.record(record("d", "t", 1, 1.0, -1.0));
        assert_eq!(window.len(), 3);
        let modules: Vec<_> = window.iter().map(|r| r.module.as_str()).collect();
        assert_eq!(modules, vec!["b", "c", "d"]);
    }

    #[test]
    fn predicted_vs_actual_gap_handles_zero_predicted() {
        // Zero predicted gain is dropped so the median is computed over
        // valid ratios only.
        let mut window = RecentFailureWindow::new();
        window.record(record("m", "t", 1, 0.0, -1.0));
        window.record(record("m", "t", 1, 1.0, -2.0));
        window.record(record("m", "t", 1, 1.0, -4.0));
        window.record(record("m", "t", 1, 1.0, -6.0));
        window.record(record("m", "t", 1, 1.0, -8.0));
        let aggs = window.aggregates().expect("5 entries");
        // Valid ratios are [-2, -4, -6, -8]; median = -5.
        assert!((aggs.predicted_vs_actual_gap_p50 - -5.0).abs() < 1e-6);
    }

    #[test]
    fn bcbca347_regression_shape() {
        // The drought-investigation evidence from Issue #1274: 91% of
        // attempts are coordinated-structural, all 4-op, all targeting the
        // same output neuron, with predicted gains around the 5e-7 floor and
        // actual deltas roughly 1000x worse in the harmful direction.
        let target = "533d8616-aaaa-bbbb-cccc-000000000000";
        let mut window = RecentFailureWindow::with_capacity(100);
        for _ in 0..91 {
            window.record(record("coordinated-structural", target, 4, 5e-7, -5e-4));
        }
        // The remaining 9% — different modules / op counts.
        for _ in 0..5 {
            window.record(record("add-synapses", target, 1, 1e-6, -1e-5));
        }
        for _ in 0..4 {
            window.record(record("add-neurons", "other-target", 1, 1e-6, -1e-5));
        }

        let aggs = window.aggregates().expect("100 entries");
        assert_eq!(
            aggs.dominant_module.as_deref(),
            Some("coordinated-structural")
        );
        assert!((aggs.dominant_module_share - 0.91).abs() < 1e-6);
        assert_eq!(aggs.dominant_target_uuid.as_deref(), Some(target));
        assert!((aggs.dominant_target_share - 0.96).abs() < 1e-6);
        assert_eq!(aggs.dominant_operation_count, Some(4));
        assert!(aggs.predicted_vs_actual_gap_p50 < -100.0);
    }
}
