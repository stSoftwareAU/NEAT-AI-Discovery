//! Per-class capacity allocation under a `OneHot` task descriptor
//! (Issue #1319).
//!
//! When the producer reports a `OneHot` target topology (e.g.
//! `CATEGORICAL_ERROR`), the output neurons are exchangeable class scores —
//! the cost identity confirms per-class allocation is well-defined. This
//! module skews the growth/candidate budget toward output neurons (classes)
//! with the highest cumulative per-target failure counts, using the
//! existing per-target failure signal already supplied via the
//! `failure_cache` field on the discovery FFI inputs (Issue #1131) and the
//! per-target trackers documented in
//! [`crate::analysis::within_batch_failures`] and
//! [`crate::analysis::target_failure_tracker`].
//!
//! The contract is intentionally narrow:
//!
//! - For non-`OneHot` descriptors (`Independent`, `Margin`, `Simplex`,
//!   `Unknown`, `OTHER`, absent) [`compute_class_failure_counts`] returns
//!   `None`, leaving the legacy allocation path untouched (regression
//!   guard, per the issue acceptance criteria).
//! - [`apply_class_priority_spread`] reorders a gain-sorted candidate list so
//!   the front contains up to `max_distinct` distinct targets ordered by
//!   descending priority (with ties broken by descending gain). The tail
//!   preserves the remaining candidates in their original gain order. This
//!   is a strict generalisation of the legacy `apply_distinct_target_spread`
//!   helper in [`crate::analysis::neuron`] — passing an empty priority map
//!   produces the same no-op behaviour as the legacy spread when there is
//!   no per-class signal yet.
//!
//! Both functions are pure (no I/O, no global state) and operate on plain
//! data, so they are exercised by integration tests in
//! `tests/recommendation/issue_1319_one_hot_per_class_capacity_allocation.rs`.

use std::collections::{HashMap, HashSet};

use crate::CandidateNeuronJson;
use crate::analysis::scoring::calibration_correction::FailureCacheEntry;
use crate::analysis::task_descriptor::{TargetTopology, TaskDescriptor};

/// Aggregate per-output-class failure counts from the supplied failure cache
/// when, and only when, `descriptor.target_topology` is `OneHot`.
///
/// Returns `None` for every other topology — including `Unknown`,
/// `Independent`, `Simplex`, `Margin`, and the neutral / OTHER descriptor —
/// so callers can fall straight through to the legacy allocation path
/// (acceptance criterion: "`OTHER` / `Unknown` / absent ⇒ existing
/// allocation").
///
/// `is_output` is invoked to decide which `target_uuid`s in the cache are
/// output neurons (i.e. classes). Hidden / input / constant targets are
/// skipped — only output-neuron failures form the per-class budget signal.
/// Entries that lack a `target_uuid` (legacy payloads, Issue #1194) are
/// ignored.
///
/// The returned map omits classes with zero recorded failures so callers can
/// cheaply detect "no signal yet" via `is_empty()`.
#[must_use]
pub fn compute_class_failure_counts(
    descriptor: &TaskDescriptor,
    failure_cache: &[FailureCacheEntry],
    is_output: impl Fn(&str) -> bool,
) -> Option<HashMap<String, u32>> {
    if descriptor.target_topology != TargetTopology::OneHot {
        return None;
    }
    let mut counts: HashMap<String, u32> = HashMap::new();
    for entry in failure_cache {
        let Some(target_uuid) = entry.target_uuid.as_deref() else {
            continue;
        };
        if !is_output(target_uuid) {
            continue;
        }
        *counts.entry(target_uuid.to_string()).or_insert(0) = counts
            .get(target_uuid)
            .copied()
            .unwrap_or(0)
            .saturating_add(1);
    }
    Some(counts)
}

/// Reorder a gain-sorted candidate list so the front contains the
/// highest-gain candidate for each of up to `max_distinct` distinct targets,
/// ordered by descending priority (with ties broken by descending gain).
///
/// Preconditions:
///
/// - `candidates` is sorted by `expected_creature_score_gain` descending
///   (NaN-safe via `total_cmp`).
/// - `priority` maps a `target_neuron_uuid` to its priority weight.
///   Targets absent from the map are treated as priority `0`.
///
/// Postconditions:
///
/// - When the candidate pool contains at least `max_distinct` distinct
///   targets, the front of the list contains the highest-gain representative
///   of the `max_distinct` highest-priority targets. The tail preserves the
///   remaining candidates in their original gain order (a stable partition).
/// - When the pool has fewer distinct targets than `max_distinct` requires,
///   the list is left untouched — same fall-through contract as the legacy
///   `apply_distinct_target_spread` helper.
/// - When `priority` is empty, the function behaves identically to the
///   legacy spread (the first occurrence of each distinct target is chosen
///   in gain order, since priority ties break by best-index ascending).
/// - No candidates are added or removed; only the order changes.
pub fn apply_class_priority_spread(
    candidates: &mut Vec<CandidateNeuronJson>,
    priority: &HashMap<String, u32>,
    max_distinct: usize,
) {
    if candidates.len() <= 1 || max_distinct <= 1 {
        return;
    }

    // First occurrence (smallest index) of each distinct target in the
    // gain-sorted list — that is, the highest-gain representative for that
    // target.
    let mut first_index_per_target: HashMap<String, usize> = HashMap::new();
    for (i, c) in candidates.iter().enumerate() {
        first_index_per_target
            .entry(c.target_neuron_uuid.clone())
            .or_insert(i);
    }

    // Fall through when the pool cannot support the requested spread.
    if first_index_per_target.len() < max_distinct {
        return;
    }

    // Order distinct targets by descending priority; ties broken by
    // ascending first-index (i.e. descending gain). This makes the
    // empty-priority case identical to the legacy spread.
    let mut ordered: Vec<(usize, u32)> = first_index_per_target
        .iter()
        .map(|(target, &idx)| (idx, priority.get(target).copied().unwrap_or(0)))
        .collect();
    ordered.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

    let head_indices: Vec<usize> = ordered.iter().take(max_distinct).map(|(i, _)| *i).collect();
    let head_index_set: HashSet<usize> = head_indices.iter().copied().collect();

    // Stable partition: drain candidates into Option slots so we can pull
    // out the head in priority order without disturbing the relative order
    // of the tail.
    let mut slots: Vec<Option<CandidateNeuronJson>> = candidates.drain(..).map(Some).collect();
    let mut head: Vec<CandidateNeuronJson> = Vec::with_capacity(head_indices.len());
    for idx in &head_indices {
        head.push(
            slots[*idx]
                .take()
                .expect("head indices are unique by construction"),
        );
    }
    let mut tail: Vec<CandidateNeuronJson> = Vec::with_capacity(slots.len());
    for (idx, slot) in slots.into_iter().enumerate() {
        if head_index_set.contains(&idx) {
            // Already moved to head.
            debug_assert!(slot.is_none());
            continue;
        }
        if let Some(c) = slot {
            tail.push(c);
        }
    }
    candidates.extend(head);
    candidates.extend(tail);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::scoring::calibration_correction::FailureCacheEntry;

    fn fc(target_uuid: &str) -> FailureCacheEntry {
        FailureCacheEntry {
            change_type: "add-neurons".to_string(),
            expected_error_reduction: 0.1,
            actual_error_reduction: -0.01,
            target_squash: None,
            variant_key: None,
            target_uuid: Some(target_uuid.to_string()),
            improved_count: None,
            total_count: None,
        }
    }

    #[test]
    fn one_hot_aggregates_failures_per_output_class() {
        let descriptor = TaskDescriptor::from_name("CATEGORICAL_ERROR", 3);
        let cache = vec![fc("o-A"), fc("o-A"), fc("o-B")];
        let outputs: HashSet<&str> = ["o-A", "o-B"].into_iter().collect();
        let counts = compute_class_failure_counts(&descriptor, &cache, |u| outputs.contains(u))
            .expect("OneHot must produce a map");
        assert_eq!(counts.get("o-A").copied(), Some(2));
        assert_eq!(counts.get("o-B").copied(), Some(1));
    }

    #[test]
    fn non_one_hot_topologies_return_none() {
        let cache = vec![fc("o-A")];
        for name in [
            "MSE",
            "MAE",
            "MAPE",
            "BINARY_CROSS_ENTROPY",
            "CROSS_ENTROPY",
            "HINGE",
        ] {
            let descriptor = TaskDescriptor::from_name(name, 4);
            assert!(
                compute_class_failure_counts(&descriptor, &cache, |_| true).is_none(),
                "{name} must opt out of per-class allocation",
            );
        }
        assert!(
            compute_class_failure_counts(&TaskDescriptor::neutral(), &cache, |_| true).is_none()
        );
    }

    #[test]
    fn empty_failure_cache_under_one_hot_returns_empty_map() {
        let descriptor = TaskDescriptor::from_name("CATEGORICAL_ERROR", 3);
        let counts = compute_class_failure_counts(&descriptor, &[], |_| true)
            .expect("OneHot must still produce a (possibly empty) map");
        assert!(counts.is_empty());
    }

    #[test]
    fn non_output_targets_are_skipped() {
        let descriptor = TaskDescriptor::from_name("CATEGORICAL_ERROR", 3);
        let cache = vec![fc("o-A"), fc("h-1")];
        let outputs: HashSet<&str> = ["o-A"].into_iter().collect();
        let counts = compute_class_failure_counts(&descriptor, &cache, |u| outputs.contains(u))
            .expect("OneHot must produce a map");
        assert_eq!(counts.get("o-A").copied(), Some(1));
        assert!(!counts.contains_key("h-1"));
    }
}
