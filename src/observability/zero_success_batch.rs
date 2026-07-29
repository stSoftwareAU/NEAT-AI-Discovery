//! Structured zero-success batch summary event (Issue #1194).
//!
//! When a discovery batch finishes with zero accepted candidates, this module
//! produces a structured [`ZeroSuccessBatchSummary`] over the batch's failure
//! cache so operators can detect failure clusters (e.g. three failures
//! targeting the same neuron with the same change type) without having to
//! inspect the cache by hand.
//!
//! The aggregation is intentionally cheap — it walks the cache once with
//! fixed-size accumulators (`HashSet`s for distinct values, `Vec`s sized to the
//! cache length for the median computations). No per-candidate cloning is
//! performed; only the small fields that contribute to the summary are read.
//!
//! ## Emission
//!
//! Use [`emit_zero_success_batch_summary`] from the orchestration layer when
//! the candidate-evaluation loop finishes with zero accepted candidates and a
//! non-empty failure cache is present. The event is logged via
//! `tracing::warn!` with `target = "neat_ai_discovery::observability"` and an
//! `event = "zero_success_batch"` tag so downstream tooling can filter on it.

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)] // Intentional numeric casts for median statistics and wall-clock id

use std::collections::HashSet;

use serde::Serialize;

use crate::analysis::scoring::calibration_correction::FailureCacheEntry;

/// Summary of a discovery batch that produced zero accepted candidates
/// (Issue #1194).
///
/// All numeric aggregates are computed over the failure-cache entries
/// supplied at construction time. Median values use the simple lower / upper
/// midpoint rule: for an even-length sorted vector the mean of the two
/// middle elements is returned. Non-finite or zero `expected_error_reduction`
/// entries are excluded from the ratio computation but counted in the
/// `candidate_count`.
#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ZeroSuccessBatchSummary {
    /// Opaque identifier for this batch — typically the wall-clock instant
    /// (nanoseconds since the Unix epoch) at which the summary was built.
    pub batch_id: u64,
    /// Total number of failure-cache entries representing the batch.
    pub candidate_count: usize,
    /// Number of distinct target neuron UUIDs across the batch
    /// (entries lacking a `target_uuid` are excluded from this count).
    pub distinct_target_uuids: usize,
    /// Number of distinct target activation functions across the batch.
    pub distinct_target_squashes: usize,
    /// Number of distinct change types across the batch.
    pub distinct_change_types: usize,
    /// Number of distinct variant keys across the batch.
    pub distinct_variant_keys: usize,
    /// Minimum `expectedErrorReduction` across the batch (None when empty).
    pub min_expected: Option<f32>,
    /// Median `expectedErrorReduction` across the batch.
    pub median_expected: Option<f32>,
    /// Maximum `expectedErrorReduction` across the batch.
    pub max_expected: Option<f32>,
    /// Minimum `actualErrorReduction` across the batch.
    pub min_actual: Option<f32>,
    /// Median `actualErrorReduction` across the batch.
    pub median_actual: Option<f32>,
    /// Maximum `actualErrorReduction` across the batch.
    pub max_actual: Option<f32>,
    /// `median(actual) / median(expected)`, computed when the median expected
    /// reduction is finite and non-zero. `None` otherwise — including when
    /// the cache is empty or every expected value collapsed to zero.
    pub median_actual_over_expected: Option<f32>,
}

impl ZeroSuccessBatchSummary {
    /// Build a summary from a batch's failure cache.
    ///
    /// `batch_id` is an opaque identifier the caller chooses (typically a
    /// wall-clock nanosecond stamp). The aggregation walks the cache once
    /// and never clones a `FailureCacheEntry`.
    #[must_use]
    pub fn aggregate(batch_id: u64, cache: &[FailureCacheEntry]) -> Self {
        let mut distinct_target_uuids: HashSet<&str> = HashSet::new();
        let mut distinct_target_squashes: HashSet<&str> = HashSet::new();
        let mut distinct_change_types: HashSet<&str> = HashSet::new();
        let mut distinct_variant_keys: HashSet<&str> = HashSet::new();
        // Pre-size to the cache length to avoid reallocations.
        let mut expected: Vec<f32> = Vec::with_capacity(cache.len());
        let mut actual: Vec<f32> = Vec::with_capacity(cache.len());

        for entry in cache {
            distinct_change_types.insert(entry.change_type.as_str());
            if let Some(uuid) = entry.target_uuid.as_deref() {
                distinct_target_uuids.insert(uuid);
            }
            if let Some(squash) = entry.target_squash.as_deref() {
                distinct_target_squashes.insert(squash);
            }
            if let Some(key) = entry.variant_key.as_deref() {
                distinct_variant_keys.insert(key);
            }
            // Skip non-finite aggregates so a single NaN cannot poison the
            // median; the `candidate_count` still reflects the full cache.
            if entry.expected_error_reduction.is_finite() {
                expected.push(entry.expected_error_reduction);
            }
            if entry.actual_error_reduction.is_finite() {
                actual.push(entry.actual_error_reduction);
            }
        }

        let (min_expected, median_expected, max_expected) = min_median_max(&mut expected);
        let (min_actual, median_actual, max_actual) = min_median_max(&mut actual);

        let median_actual_over_expected = match (median_actual, median_expected) {
            (Some(a), Some(e)) if e != 0.0 && e.is_finite() => {
                let ratio = a / e;
                if ratio.is_finite() { Some(ratio) } else { None }
            }
            _ => None,
        };

        Self {
            batch_id,
            candidate_count: cache.len(),
            distinct_target_uuids: distinct_target_uuids.len(),
            distinct_target_squashes: distinct_target_squashes.len(),
            distinct_change_types: distinct_change_types.len(),
            distinct_variant_keys: distinct_variant_keys.len(),
            min_expected,
            median_expected,
            max_expected,
            min_actual,
            median_actual,
            max_actual,
            median_actual_over_expected,
        }
    }

    /// Emit the summary via `tracing::warn!` with structured fields. The
    /// event is filterable by the `event = "zero_success_batch"` tag so
    /// downstream tooling can subscribe without scanning by message text.
    pub fn emit(&self) {
        tracing::warn!(
            target: "neat_ai_discovery::observability",
            event = "zero_success_batch",
            batch_id = self.batch_id,
            candidate_count = self.candidate_count,
            distinct_target_uuids = self.distinct_target_uuids,
            distinct_target_squashes = self.distinct_target_squashes,
            distinct_change_types = self.distinct_change_types,
            distinct_variant_keys = self.distinct_variant_keys,
            min_expected = ?self.min_expected,
            median_expected = ?self.median_expected,
            max_expected = ?self.max_expected,
            min_actual = ?self.min_actual,
            median_actual = ?self.median_actual,
            max_actual = ?self.max_actual,
            median_actual_over_expected = ?self.median_actual_over_expected,
            "discovery batch finished with zero accepted candidates (Issue #1194)"
        );
    }
}

/// Compute (min, median, max) over a slice of floats. The slice is sorted
/// in-place using `partial_cmp` so callers must pass a mutable owned vector.
/// Returns `(None, None, None)` for an empty slice.
fn min_median_max(values: &mut [f32]) -> (Option<f32>, Option<f32>, Option<f32>) {
    if values.is_empty() {
        return (None, None, None);
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let min = values[0];
    let max = values[values.len() - 1];
    let median = if values.len().is_multiple_of(2) {
        (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
    } else {
        values[values.len() / 2]
    };
    (Some(min), Some(median), Some(max))
}

/// Best-effort wall-clock identifier for a batch — nanoseconds since the
/// Unix epoch, falling back to zero if the system clock is before the epoch
/// (which never happens on supported platforms).
#[must_use]
pub fn now_batch_id() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64)
}

/// Emit a [`ZeroSuccessBatchSummary`] when the supplied failure cache is
/// non-empty. Returns `Some(summary)` when an event was emitted, `None`
/// otherwise.
///
/// Callers must check `accepted_candidates == 0` before invoking this
/// function; the helper itself does not gate on the accepted count.
pub fn emit_zero_success_batch_summary(
    cache: &[FailureCacheEntry],
) -> Option<ZeroSuccessBatchSummary> {
    if cache.is_empty() {
        return None;
    }
    let summary = ZeroSuccessBatchSummary::aggregate(now_batch_id(), cache);
    summary.emit();
    Some(summary)
}

/// Conditionally emit a [`ZeroSuccessBatchSummary`] based on the accepted
/// candidate count and the failure cache.
///
/// Returns `Some(summary)` only when `accepted_candidates == 0` *and* the
/// cache is non-empty. This mirrors the orchestration-layer gate so the
/// emission contract is unit-testable without spinning up the GPU pipeline.
pub fn maybe_emit_zero_success_batch_summary(
    accepted_candidates: usize,
    cache: &[FailureCacheEntry],
) -> Option<ZeroSuccessBatchSummary> {
    if accepted_candidates > 0 {
        return None;
    }
    emit_zero_success_batch_summary(cache)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(
        change_type: &str,
        expected: f32,
        actual: f32,
        squash: Option<&str>,
        variant: Option<&str>,
        uuid: Option<&str>,
    ) -> FailureCacheEntry {
        FailureCacheEntry {
            change_type: change_type.to_string(),
            expected_error_reduction: expected,
            actual_error_reduction: actual,
            target_squash: squash.map(str::to_string),
            variant_key: variant.map(str::to_string),
            target_uuid: uuid.map(str::to_string),
            improved_count: None,
            total_count: None,
            age_epochs: None,
        }
    }

    #[test]
    fn aggregate_empty_cache_returns_zero_counts() {
        let summary = ZeroSuccessBatchSummary::aggregate(42, &[]);
        assert_eq!(summary.batch_id, 42);
        assert_eq!(summary.candidate_count, 0);
        assert_eq!(summary.distinct_target_uuids, 0);
        assert_eq!(summary.distinct_change_types, 0);
        assert!(summary.median_expected.is_none());
        assert!(summary.median_actual_over_expected.is_none());
    }

    #[test]
    fn aggregate_distinct_counts_match_set_cardinality() {
        // Three failures targeting the same neuron with the same change type
        // — the textbook cluster from Issue #1189.
        let cache = vec![
            entry(
                "add-neurons",
                0.001,
                -0.5,
                Some("SELU"),
                Some("gentle-nudge"),
                Some("neuron-A"),
            ),
            entry(
                "add-neurons",
                0.002,
                -0.4,
                Some("SELU"),
                Some("gentle-nudge"),
                Some("neuron-A"),
            ),
            entry(
                "add-neurons",
                0.003,
                -0.3,
                Some("SELU"),
                Some("gentle-nudge"),
                Some("neuron-A"),
            ),
        ];
        let summary = ZeroSuccessBatchSummary::aggregate(1, &cache);
        assert_eq!(summary.candidate_count, 3);
        assert_eq!(summary.distinct_target_uuids, 1);
        assert_eq!(summary.distinct_target_squashes, 1);
        assert_eq!(summary.distinct_change_types, 1);
        assert_eq!(summary.distinct_variant_keys, 1);
    }

    #[test]
    fn aggregate_min_median_max_for_odd_length() {
        // Three entries: expected = [0.001, 0.002, 0.003], actual = [-0.5, -0.4, -0.3]
        let cache = vec![
            entry("add-neurons", 0.001, -0.5, None, None, None),
            entry("add-neurons", 0.002, -0.4, None, None, None),
            entry("add-neurons", 0.003, -0.3, None, None, None),
        ];
        let s = ZeroSuccessBatchSummary::aggregate(0, &cache);
        assert!((s.min_expected.unwrap() - 0.001).abs() < 1e-6);
        assert!((s.median_expected.unwrap() - 0.002).abs() < 1e-6);
        assert!((s.max_expected.unwrap() - 0.003).abs() < 1e-6);
        assert!((s.min_actual.unwrap() - (-0.5)).abs() < 1e-6);
        assert!((s.median_actual.unwrap() - (-0.4)).abs() < 1e-6);
        assert!((s.max_actual.unwrap() - (-0.3)).abs() < 1e-6);
        // ratio = -0.4 / 0.002 = -200
        let ratio = s.median_actual_over_expected.unwrap();
        assert!((ratio - (-200.0)).abs() < 1e-3);
    }

    #[test]
    fn aggregate_median_for_even_length_is_midpoint_mean() {
        let cache = vec![
            entry("add-synapses", 1.0, 2.0, None, None, None),
            entry("add-synapses", 3.0, 4.0, None, None, None),
            entry("add-synapses", 5.0, 6.0, None, None, None),
            entry("add-synapses", 7.0, 8.0, None, None, None),
        ];
        let s = ZeroSuccessBatchSummary::aggregate(0, &cache);
        // median expected = (3 + 5) / 2 = 4
        assert!((s.median_expected.unwrap() - 4.0).abs() < 1e-6);
        // median actual = (4 + 6) / 2 = 5
        assert!((s.median_actual.unwrap() - 5.0).abs() < 1e-6);
        // ratio = 5 / 4 = 1.25
        assert!((s.median_actual_over_expected.unwrap() - 1.25).abs() < 1e-6);
    }

    #[test]
    fn aggregate_distinct_counts_ignore_missing_optional_fields() {
        let cache = vec![
            entry("add-neurons", 1.0, 0.5, None, None, None),
            entry("add-synapses", 2.0, 1.0, Some("SELU"), None, Some("n-1")),
        ];
        let s = ZeroSuccessBatchSummary::aggregate(0, &cache);
        assert_eq!(s.distinct_change_types, 2);
        assert_eq!(s.distinct_target_squashes, 1);
        assert_eq!(s.distinct_variant_keys, 0);
        assert_eq!(s.distinct_target_uuids, 1);
    }

    #[test]
    fn aggregate_skips_non_finite_values_for_aggregates() {
        let cache = vec![
            entry("add-neurons", f32::NAN, 0.0, None, None, None),
            entry("add-neurons", 1.0, f32::INFINITY, None, None, None),
            entry("add-neurons", 2.0, 1.0, None, None, None),
        ];
        let s = ZeroSuccessBatchSummary::aggregate(0, &cache);
        // candidate_count counts every entry regardless of finite-ness.
        assert_eq!(s.candidate_count, 3);
        // expected: only [1.0, 2.0] contribute → median 1.5
        assert!((s.median_expected.unwrap() - 1.5).abs() < 1e-6);
        // actual: only [0.0, 1.0] contribute → median 0.5
        assert!((s.median_actual.unwrap() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn aggregate_zero_expected_yields_no_ratio() {
        let cache = vec![
            entry("add-neurons", 0.0, 1.0, None, None, None),
            entry("add-neurons", 0.0, -1.0, None, None, None),
        ];
        let s = ZeroSuccessBatchSummary::aggregate(0, &cache);
        // median expected = 0 → ratio undefined.
        assert!(s.median_actual_over_expected.is_none());
    }

    #[test]
    fn emit_helper_returns_none_for_empty_cache() {
        // No event emitted when there is nothing to summarise.
        let result = emit_zero_success_batch_summary(&[]);
        assert!(result.is_none());
    }

    #[test]
    fn maybe_emit_returns_none_when_at_least_one_candidate_accepted() {
        let cache = vec![entry(
            "add-neurons",
            0.001,
            -0.5,
            Some("SELU"),
            Some("gentle-nudge"),
            Some("neuron-A"),
        )];
        // accepted = 1 ⇒ no event even though the cache is non-empty.
        let result = maybe_emit_zero_success_batch_summary(1, &cache);
        assert!(result.is_none());
    }

    #[test]
    fn maybe_emit_returns_summary_when_zero_accepted_and_cache_non_empty() {
        let cache = vec![entry(
            "add-neurons",
            0.001,
            -0.5,
            Some("SELU"),
            Some("gentle-nudge"),
            Some("neuron-A"),
        )];
        let result = maybe_emit_zero_success_batch_summary(0, &cache);
        let summary = result.expect("zero accepted + non-empty cache emits");
        assert_eq!(summary.candidate_count, 1);
    }

    #[test]
    fn emit_helper_returns_summary_for_non_empty_cache() {
        let cache = vec![entry(
            "add-neurons",
            0.001,
            -0.5,
            Some("SELU"),
            Some("gentle-nudge"),
            Some("neuron-A"),
        )];
        let result = emit_zero_success_batch_summary(&cache);
        let summary = result.expect("non-empty cache must yield a summary");
        assert_eq!(summary.candidate_count, 1);
        assert_eq!(summary.distinct_target_uuids, 1);
    }

    #[test]
    fn summary_serialises_with_camel_case_fields() {
        let cache = vec![entry("add-neurons", 1.0, 0.5, None, None, None)];
        let summary = ZeroSuccessBatchSummary::aggregate(7, &cache);
        let json = serde_json::to_value(&summary).expect("serialise");
        // Spot-check a few field names — `serde(rename_all = "camelCase")`.
        assert_eq!(json["batchId"], 7);
        assert_eq!(json["candidateCount"], 1);
        assert!(json["medianActualOverExpected"].is_number());
    }
}
