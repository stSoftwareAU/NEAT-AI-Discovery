//! Environmentally-disabled passes must not corrupt the drought signal
//! (Issue #1421).
//!
//! A discovery pass gated by the memory or GPU checks returns 0 candidates —
//! the same surface as a genuinely-exhausted search. These tests prove that an
//! environmentally-disabled pass is excluded from the drought diagnostic and
//! the per-target cooldown tracker, while a genuine empty pass is still
//! counted.
//!
//! Issue #1792: the `CandidateOutcomeCache::record_unless_disabled` case was
//! dropped with the cache, which was never constructed outside tests.
//! Issue #1793: the `ModuleStarvationTracker` case went the same way — the
//! tracker was never populated in production, so it was deleted rather than
//! wired.

use neat_ai_discovery::analysis::discovery_mode::DiscoveryOutcomeLog;
use neat_ai_discovery::analysis::target_failure_tracker::TargetFailureTracker;
use neat_ai_discovery::analysis::{AnalysisOutcome, EnvironmentalDisableReason, PassOutcomeCounts};

const GATED: AnalysisOutcome = AnalysisOutcome::EnvironmentallyDisabled {
    reason: EnvironmentalDisableReason::GpuUnavailable,
};
const MEM_GATED: AnalysisOutcome = AnalysisOutcome::EnvironmentallyDisabled {
    reason: EnvironmentalDisableReason::MemoryGated,
};
const EMPTY: AnalysisOutcome = AnalysisOutcome::Completed { candidates: 0 };

/// AC3: N consecutive gated passes do not trip the drought diagnostic.
#[test]
fn n_gated_passes_do_not_trip_drought() {
    let mut log = DiscoveryOutcomeLog::default();
    for _ in 0..25 {
        log.record_outcome(&GATED);
    }
    // The drought diagnostic keys off the trailing-failure streak; gated passes
    // never appear in `outcomes`, so the streak stays at zero.
    assert_eq!(log.consecutive_trailing_failures(), 0);
    assert_eq!(log.genuinely_empty_passes(), 0);
    assert_eq!(log.environmentally_disabled_passes, 25);
}

/// AC3: N consecutive gated passes do not trip any target cooldown.
#[test]
fn n_gated_passes_do_not_trip_target_cooldown() {
    let mut tracker = TargetFailureTracker::with_thresholds(3, 100);
    for epoch in 0..25 {
        assert!(!tracker.record_failure_unless_disabled("tgt", epoch, &MEM_GATED));
    }
    assert!(!tracker.is_in_cooldown("tgt", 25));
    assert_eq!(tracker.active_cooldown_count(25), 0);
}

/// Regression guard: a genuine empty pass IS still counted everywhere, so the
/// fix narrows the signal without silencing real droughts.
#[test]
fn genuine_empty_passes_still_count() {
    let mut log = DiscoveryOutcomeLog::default();
    let mut target = TargetFailureTracker::with_thresholds(3, 100);

    for epoch in 0..3 {
        log.record_outcome(&EMPTY);
        assert!(target.record_failure_unless_disabled("tgt", epoch, &EMPTY));
    }

    assert_eq!(log.consecutive_trailing_failures(), 3);
    assert!(target.is_in_cooldown("tgt", 2));
}

/// The discovery summary separates the two categories so operators can tell a
/// "host can't run discovery" drought from a "search exhausted" drought.
#[test]
fn pass_outcome_counts_distinguish_categories() {
    let mut counts = PassOutcomeCounts::default();
    counts.record(&AnalysisOutcome::Completed { candidates: 5 });
    for _ in 0..3 {
        counts.record(&EMPTY);
    }
    for _ in 0..7 {
        counts.record(&GATED);
    }
    assert_eq!(counts.productive, 1);
    assert_eq!(counts.genuinely_empty, 3);
    assert_eq!(counts.environmentally_disabled, 7);
    assert_eq!(counts.total(), 11);
}
