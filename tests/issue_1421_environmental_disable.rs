//! Environmentally-disabled passes must not corrupt the drought signal
//! (Issue #1421).
//!
//! A discovery pass gated by the memory or GPU checks returns 0 candidates —
//! the same surface as a genuinely-exhausted search. These tests prove that an
//! environmentally-disabled pass is excluded from the drought diagnostic, the
//! per-target cooldown tracker, and the per-module starvation tracker, while a
//! genuine empty pass is still counted.

use neat_ai_discovery::analysis::candidate_cache::CandidateOutcomeCache;
use neat_ai_discovery::analysis::discovery_mode::DiscoveryOutcomeLog;
use neat_ai_discovery::analysis::module_starvation_tracker::ModuleStarvationTracker;
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

/// AC3: N consecutive gated passes do not trip module starvation.
#[test]
fn n_gated_passes_do_not_trip_module_starvation() {
    let mut tracker = ModuleStarvationTracker::with_thresholds(3, 100);
    for epoch in 0..25 {
        assert!(!tracker.record_failure_unless_disabled("coordinated-structural", epoch, &GATED));
    }
    assert!(!tracker.is_starved("coordinated-structural", 25));
    assert_eq!(tracker.starved_module_count(25), 0);
}

/// AC3: N consecutive gated passes do not suppress candidates in the cache.
#[test]
fn n_gated_passes_do_not_suppress_candidates() {
    let mut cache = CandidateOutcomeCache::new();
    for epoch in 0..25 {
        assert!(!cache.record_unless_disabled("src", "tgt", "addSynapse", false, epoch, &GATED));
    }
    assert_eq!(cache.len(), 0);
    assert_eq!(cache.suppressed_count(25), 0);
}

/// Regression guard: a genuine empty pass IS still counted everywhere, so the
/// fix narrows the signal without silencing real droughts.
#[test]
fn genuine_empty_passes_still_count() {
    let mut log = DiscoveryOutcomeLog::default();
    let mut target = TargetFailureTracker::with_thresholds(3, 100);
    let mut starvation = ModuleStarvationTracker::with_thresholds(3, 100);

    for epoch in 0..3 {
        log.record_outcome(&EMPTY);
        assert!(target.record_failure_unless_disabled("tgt", epoch, &EMPTY));
        assert!(starvation.record_failure_unless_disabled("m", epoch, &EMPTY));
    }

    assert_eq!(log.consecutive_trailing_failures(), 3);
    assert!(target.is_in_cooldown("tgt", 2));
    assert!(starvation.is_starved("m", 2));
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
