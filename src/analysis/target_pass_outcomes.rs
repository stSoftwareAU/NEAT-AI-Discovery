//! Per-target, per-pass outcome accumulation for the global cooldown tracker
//! (Issue #1791).
//!
//! The global [`TargetFailureTracker`](crate::analysis::target_failure_tracker::TargetFailureTracker)
//! is read by both preparation layers but, before this module existed, nothing
//! in `src/` ever wrote to it. Both filters short-circuit on
//! `tracker.is_empty()`, so the cooldown suppression was permanently inert.
//!
//! # Aggregation point
//!
//! The tracker is **per-target, per-pass** — never per-candidate. Ten rejected
//! candidates for one target in one discovery pass are **one** failure, which
//! matches the within-batch semantics of
//! [`WithinBatchFailureTracker`](crate::analysis::within_batch_failures::WithinBatchFailureTracker).
//!
//! The accumulation therefore happens where the per-target verdict already
//! exists — the diagnostics entry each analysis module builds during its
//! parallel evaluation loop (`had_candidate` is set by `mark_candidate_selected`).
//! Each module snapshots its entries into a [`TargetPassOutcome`] list at result
//! finalisation, and `analysis::analyze_all` flushes the **combined** neuron and
//! synapse lists exactly once per pass under a single [`global_tracker`] lock.
//! Flushing per module would double a target's streak growth against a counter
//! that only advances one epoch per pass.
//!
//! # Lock contention
//!
//! The evaluation loops are parallel (rayon); the diagnostics they write to are
//! `DashMap`-backed and lock-free. This module takes the global mutex exactly
//! once per pass, after the parallel work has finished, so it adds no
//! contention to the hot path.
//!
//! ```text
//! parallel eval ──► DashMap diagnostics ──► metadata.target_pass_outcomes
//!                                                     │
//!                                    (one lock, once per pass)
//!                                                     ▼
//!                                            global_tracker()
//! ```

use std::collections::BTreeMap;

use crate::analysis::AnalysisOutcome;
use crate::analysis::target_failure_tracker::global_tracker;

/// One discovery module's verdict for one target neuron over one pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TargetPassOutcome {
    /// The target neuron the verdict applies to.
    pub target_uuid: String,
    /// `true` when the target yielded at least one accepted candidate.
    pub had_candidate: bool,
    /// `true` when the target was genuinely evaluated — at least one source was
    /// tried against it. A target that was filtered out before evaluation
    /// (input / hidden / constant, or dropped by an earlier cooldown) carries no
    /// evidence either way and must not move the streak.
    pub evaluated: bool,
}

impl TargetPassOutcome {
    /// Convenience constructor.
    #[must_use]
    pub fn new(target_uuid: impl Into<String>, had_candidate: bool, evaluated: bool) -> Self {
        Self {
            target_uuid: target_uuid.into(),
            had_candidate,
            evaluated,
        }
    }
}

/// What a flush actually did, for logging and tests.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FlushSummary {
    /// Targets that recorded a success (counter reset, tombstone cleared).
    pub successes: usize,
    /// Targets that recorded a failure (streak extended).
    pub failures: usize,
    /// Targets deliberately not recorded — never evaluated, or the whole pass
    /// was environmentally disabled (Issue #1421).
    pub skipped: usize,
}

/// Merge per-module verdicts into one verdict per target.
///
/// A target is a success when **any** module accepted a candidate for it, and
/// is considered evaluated when **any** module evaluated it. Output is ordered
/// by target UUID so callers and tests see a deterministic sequence.
#[must_use]
pub fn merge_target_pass_outcomes(outcomes: &[TargetPassOutcome]) -> Vec<TargetPassOutcome> {
    let mut merged: BTreeMap<&str, (bool, bool)> = BTreeMap::new();
    for outcome in outcomes {
        let entry = merged
            .entry(outcome.target_uuid.as_str())
            .or_insert((false, false));
        entry.0 |= outcome.had_candidate;
        entry.1 |= outcome.evaluated;
    }
    merged
        .into_iter()
        .map(|(uuid, (had_candidate, evaluated))| {
            TargetPassOutcome::new(uuid, had_candidate, evaluated)
        })
        .collect()
}

/// Flush one pass's per-target outcomes to the process-global tracker.
///
/// Takes the global lock exactly once. `pass_outcome` classifies the pass as a
/// whole: an environmentally-disabled pass (memory-gated, memory-pressure
/// cancelled, or GPU-unavailable) never evaluated anything, so it must not
/// extend any streak — the contract enforced by
/// [`TargetFailureTracker::record_failure_unless_disabled`](crate::analysis::target_failure_tracker::TargetFailureTracker::record_failure_unless_disabled)
/// (Issue #1421). Successes are still recorded on such a pass only if a
/// candidate genuinely surfaced, which cannot happen when the pass was gated
/// before evaluation.
pub fn flush_target_pass_outcomes(
    outcomes: &[TargetPassOutcome],
    pass_outcome: &AnalysisOutcome,
) -> FlushSummary {
    let merged = merge_target_pass_outcomes(outcomes);
    if merged.is_empty() {
        return FlushSummary::default();
    }

    let mut summary = FlushSummary::default();
    let mut tracker = match global_tracker().lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    let epoch = tracker.current_epoch();

    for outcome in &merged {
        if outcome.had_candidate {
            tracker.record_success(&outcome.target_uuid, epoch);
            summary.successes += 1;
        } else if outcome.evaluated {
            if tracker.record_failure_unless_disabled(&outcome.target_uuid, epoch, pass_outcome) {
                summary.failures += 1;
            } else {
                summary.skipped += 1;
            }
        } else {
            summary.skipped += 1;
        }
    }
    drop(tracker);

    tracing::debug!(
        epoch,
        successes = summary.successes,
        failures = summary.failures,
        skipped = summary.skipped,
        "Issue #1791: flushed per-target pass outcomes to the global cooldown tracker"
    );
    summary
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::analysis_outcome::EnvironmentalDisableReason;

    fn completed() -> AnalysisOutcome {
        AnalysisOutcome::Completed { candidates: 1 }
    }

    fn disabled() -> AnalysisOutcome {
        AnalysisOutcome::EnvironmentallyDisabled {
            reason: EnvironmentalDisableReason::MemoryGated,
        }
    }

    #[test]
    fn merge_collapses_duplicate_targets_to_one_verdict() {
        let merged = merge_target_pass_outcomes(&[
            TargetPassOutcome::new("t-1", false, true),
            TargetPassOutcome::new("t-1", false, true),
            TargetPassOutcome::new("t-1", false, true),
        ]);
        assert_eq!(merged.len(), 1, "one target must yield one verdict");
        assert!(!merged[0].had_candidate);
        assert!(merged[0].evaluated);
    }

    #[test]
    fn merge_prefers_success_from_any_module() {
        let merged = merge_target_pass_outcomes(&[
            TargetPassOutcome::new("t-1", false, true),
            TargetPassOutcome::new("t-1", true, true),
        ]);
        assert_eq!(merged.len(), 1);
        assert!(
            merged[0].had_candidate,
            "an accepted candidate in either module is a success for the target"
        );
    }

    #[test]
    fn merge_is_ordered_by_target_uuid() {
        let merged = merge_target_pass_outcomes(&[
            TargetPassOutcome::new("t-b", false, true),
            TargetPassOutcome::new("t-a", false, true),
        ]);
        let uuids: Vec<&str> = merged.iter().map(|o| o.target_uuid.as_str()).collect();
        assert_eq!(uuids, vec!["t-a", "t-b"]);
    }

    #[test]
    fn merge_of_empty_input_is_empty() {
        assert!(merge_target_pass_outcomes(&[]).is_empty());
    }

    #[test]
    fn flush_of_empty_outcomes_touches_nothing() {
        let summary = flush_target_pass_outcomes(&[], &completed());
        assert_eq!(summary, FlushSummary::default());
    }

    #[test]
    fn flush_counts_success_failure_and_unevaluated() {
        // Unique UUIDs: the global tracker is shared across the test binary.
        let outcomes = vec![
            TargetPassOutcome::new("flush-counts-success", true, true),
            TargetPassOutcome::new("flush-counts-failure", false, true),
            TargetPassOutcome::new("flush-counts-unevaluated", false, false),
        ];
        let summary = flush_target_pass_outcomes(&outcomes, &completed());
        assert_eq!(summary.successes, 1);
        assert_eq!(summary.failures, 1);
        assert_eq!(summary.skipped, 1);
    }

    #[test]
    fn flush_on_environmentally_disabled_pass_records_no_failure() {
        let outcomes = vec![TargetPassOutcome::new("flush-env-disabled", false, true)];
        let summary = flush_target_pass_outcomes(&outcomes, &disabled());
        assert_eq!(summary.failures, 0);
        assert_eq!(summary.skipped, 1);

        let tracker = global_tracker()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert!(
            tracker.state("flush-env-disabled").is_none(),
            "a gated pass must not create tracker state"
        );
    }
}
