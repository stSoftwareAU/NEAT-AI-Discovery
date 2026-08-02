//! Issue #1781: the fingerprint skip needs an explicit escape hatch and a
//! rejection-breakdown counter so a whole-pass drop is visible.
//!
//! `previous_neuron_fingerprints` covers structure only. During a drought the
//! topology by definition does not change, so the cache says "skip" regardless
//! of freshly recorded data and the pass returns no candidates, no rejection
//! breakdown and no diagnostic. These tests pin the two remedies:
//!
//! 1. `should_bypass_fingerprint_cache` — a drought releases the cache.
//! 2. `fingerprint_unchanged` — a whole-pass drop is counted in the rejection
//!    breakdown and surfaces as the dominant reason of `zeroCandidateSummary`.

use neat_ai_discovery::analysis::candidate_starvation::{
    GenerationSignals, StarvationClass, signals_from_breakdown,
};
use neat_ai_discovery::analysis::diagnostics::RejectionBreakdown;
use neat_ai_discovery::analysis::diagnostics::rejection_reasons::{
    ALL_REJECTION_REASONS, REJECTION_FINGERPRINT_UNCHANGED,
};
use neat_ai_discovery::analysis::discovery_mode::DiscoveryOutcomeLog;
use neat_ai_discovery::analysis::fingerprint_skip_escape::{
    DEFAULT_FINGERPRINT_SKIP_DROUGHT_EPOCHS, should_bypass_fingerprint_cache,
};
use neat_ai_discovery::{EnvironmentalGatesJson, build_zero_candidate_summary};

fn gates() -> EnvironmentalGatesJson {
    EnvironmentalGatesJson {
        memory_budget_exceeded: false,
        memory_pressure_cancelled: false,
        cancelled: false,
        environmentally_disabled: None,
    }
}

/// A log whose tail holds `failures` consecutive empty passes.
fn log_with_trailing_failures(failures: usize) -> DiscoveryOutcomeLog {
    let mut outcomes = vec![true];
    outcomes.extend(std::iter::repeat_n(false, failures));
    DiscoveryOutcomeLog::from_outcomes(outcomes)
}

#[test]
fn no_outcome_log_keeps_the_fingerprint_cache_active() {
    assert!(!should_bypass_fingerprint_cache(
        None,
        DEFAULT_FINGERPRINT_SKIP_DROUGHT_EPOCHS
    ));
}

#[test]
fn healthy_creature_keeps_the_fingerprint_cache_active() {
    let log = DiscoveryOutcomeLog::from_outcomes(vec![true, true, true]);
    assert!(!should_bypass_fingerprint_cache(
        Some(&log),
        DEFAULT_FINGERPRINT_SKIP_DROUGHT_EPOCHS
    ));
}

#[test]
fn short_failure_streak_keeps_the_fingerprint_cache_active() {
    let log = log_with_trailing_failures(
        usize::try_from(DEFAULT_FINGERPRINT_SKIP_DROUGHT_EPOCHS - 1).expect("threshold fits usize"),
    );
    assert!(!should_bypass_fingerprint_cache(
        Some(&log),
        DEFAULT_FINGERPRINT_SKIP_DROUGHT_EPOCHS
    ));
}

#[test]
fn drought_releases_the_fingerprint_cache() {
    let log = log_with_trailing_failures(
        usize::try_from(DEFAULT_FINGERPRINT_SKIP_DROUGHT_EPOCHS).expect("threshold fits usize"),
    );
    assert!(should_bypass_fingerprint_cache(
        Some(&log),
        DEFAULT_FINGERPRINT_SKIP_DROUGHT_EPOCHS
    ));
}

#[test]
fn a_zero_threshold_disables_the_escape_hatch() {
    let log = log_with_trailing_failures(50);
    assert!(!should_bypass_fingerprint_cache(Some(&log), 0));
}

#[test]
fn fingerprint_unchanged_is_a_documented_rejection_reason() {
    assert!(ALL_REJECTION_REASONS.contains(&REJECTION_FINGERPRINT_UNCHANGED));
}

#[test]
fn whole_pass_fingerprint_drop_is_visible_in_the_zero_candidate_summary() {
    // A whole-pass drop has no synapse/neuron metadata at all, so the count has
    // to travel on the pass-level breakdown to reach the operator.
    let mut pass_breakdown = RejectionBreakdown::new();
    pass_breakdown.record_many_u32(REJECTION_FINGERPRINT_UNCHANGED, 6);

    // Issue #1925: the summary also carries the starvation verdict, derived
    // from the same breakdown — a whole-pass fingerprint drop is upstream, so
    // the pass reads as starved rather than over-rejected.
    let signals: GenerationSignals = signals_from_breakdown(&pass_breakdown, 0);
    let summary = build_zero_candidate_summary(
        None,
        None,
        &pass_breakdown,
        gates(),
        signals,
        StarvationClass::CandidateStarved,
    );

    assert_eq!(
        summary.dominant_rejection_reason.as_deref(),
        Some(REJECTION_FINGERPRINT_UNCHANGED),
        "the whole-pass fingerprint drop must name itself as the dominant reason"
    );
    assert_eq!(
        summary
            .rejection_breakdown
            .get(REJECTION_FINGERPRINT_UNCHANGED),
        Some(&6),
        "the skipped focus-neuron count must be reported"
    );
}
