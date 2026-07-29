//! Integration tests for global target-failure tracker population (Issue #1791).
//!
//! The global `TargetFailureTracker` was read by both preparation layers but
//! never written from production code, so `filter_cooldown_targets` was
//! permanently short-circuited by `tracker.is_empty()` and the whole cooldown
//! mechanism was inert. These tests encode the acceptance criteria: a pass whose
//! target produced only rejected candidates extends the streak, an accepted
//! candidate resets it, an environmentally-disabled pass moves nothing, and the
//! cooldown filter genuinely removes a target once the streak crosses the
//! threshold.

use neat_ai_discovery::analysis::AnalysisOutcome;
use neat_ai_discovery::analysis::analysis_outcome::EnvironmentalDisableReason;
use neat_ai_discovery::analysis::target_failure_tracker::{
    filter_cooldown_targets, global_tracker,
};
use neat_ai_discovery::analysis::target_pass_outcomes::{
    TargetPassOutcome, flush_target_pass_outcomes,
};

/// The global tracker is process-wide. Each test uses its own target UUIDs, but
/// the drought-reset tombstone and `clear_cooldown_entries` are global, so the
/// tests in this file serialise on a shared guard.
static TEST_GUARD: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn guard() -> std::sync::MutexGuard<'static, ()> {
    TEST_GUARD
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn completed_pass(candidates: usize) -> AnalysisOutcome {
    AnalysisOutcome::Completed { candidates }
}

fn gated_pass() -> AnalysisOutcome {
    AnalysisOutcome::EnvironmentallyDisabled {
        reason: EnvironmentalDisableReason::MemoryGated,
    }
}

fn consecutive_failures(target: &str) -> u32 {
    let tracker = global_tracker()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    tracker
        .state(target)
        .map_or(0, |state| state.consecutive_failures)
}

fn current_epoch() -> u64 {
    let tracker = global_tracker()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    tracker.current_epoch()
}

fn cooldown_threshold() -> u32 {
    let tracker = global_tracker()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    tracker.cooldown_consecutive_failures()
}

/// Acceptance criterion 1: after a pass where target `T` produced only rejected
/// candidates, the global tracker reports `consecutive_failures >= 1` for `T`.
///
/// This is the canary for the original defect class — a tracker that is read but
/// never written. If a refactor drops the flush wiring, this goes red.
#[test]
fn rejected_only_pass_increments_global_tracker() {
    let _guard = guard();
    let target = "population-rejected-only";
    let before = consecutive_failures(target);

    let summary = flush_target_pass_outcomes(
        &[TargetPassOutcome::new(target, false, true)],
        &completed_pass(0),
    );

    assert_eq!(
        summary.failures, 1,
        "the evaluated-but-empty target must fail"
    );
    assert!(
        consecutive_failures(target) > before,
        "a rejected-only pass must extend the target's streak"
    );
}

/// Acceptance criterion 3 (aggregation): many rejected candidates for one target
/// in one pass are exactly ONE failure, matching the within-batch semantics.
#[test]
fn many_rejections_for_one_target_count_as_one_pass_failure() {
    let _guard = guard();
    let target = "population-one-failure-per-pass";

    // Five modules/candidate evaluations all reporting the same rejected target.
    let outcomes: Vec<TargetPassOutcome> = (0..5)
        .map(|_| TargetPassOutcome::new(target, false, true))
        .collect();
    let summary = flush_target_pass_outcomes(&outcomes, &completed_pass(0));

    assert_eq!(summary.failures, 1);
    assert_eq!(
        consecutive_failures(target),
        1,
        "the tracker is per-target-per-pass, not per-candidate"
    );
}

/// Acceptance criterion 2: after a pass where `T` produced an accepted
/// candidate, its counter is `0` and the drought-reset tombstone is cleared.
///
/// Fails if the `record_success` wiring is lost, which would make cooldowns
/// permanent again.
#[test]
fn accepted_candidate_resets_counter_and_tombstone() {
    let _guard = guard();
    let target = "population-accepted-resets";

    // Build a streak first.
    for _ in 0..3 {
        flush_target_pass_outcomes(
            &[TargetPassOutcome::new(target, false, true)],
            &completed_pass(0),
        );
    }
    assert!(consecutive_failures(target) >= 3);

    // Arm the drought-reset tombstone the way the operator escape hatch does.
    {
        let mut tracker = global_tracker()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let epoch = tracker.current_epoch();
        tracker.clear_cooldown_entries(epoch);
        assert!(
            tracker.drought_reset_tombstone().is_some(),
            "precondition: the tombstone must be set before the success"
        );
    }

    let summary = flush_target_pass_outcomes(
        &[TargetPassOutcome::new(target, true, true)],
        &completed_pass(1),
    );

    assert_eq!(summary.successes, 1);
    assert_eq!(
        consecutive_failures(target),
        0,
        "an accepted candidate must reset the consecutive-failure counter"
    );
    let tracker = global_tracker()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    assert!(
        tracker.drought_reset_tombstone().is_none(),
        "a success must clear the drought-reset tombstone so the lever re-arms"
    );
}

/// Acceptance criterion 3: an environmentally-disabled pass does not increment
/// any target's streak (the Issue #1421 contract).
///
/// Fails if a future change swaps `record_failure_unless_disabled` for a bare
/// `record_failure`.
#[test]
fn env_disabled_pass_does_not_extend_streak() {
    let _guard = guard();
    let target = "population-env-disabled";

    let summary = flush_target_pass_outcomes(
        &[TargetPassOutcome::new(target, false, true)],
        &gated_pass(),
    );

    assert_eq!(summary.failures, 0);
    assert_eq!(summary.skipped, 1);
    assert_eq!(
        consecutive_failures(target),
        0,
        "a memory/GPU-gated pass never evaluated the target"
    );
    let tracker = global_tracker()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    assert!(
        tracker.state(target).is_none(),
        "a gated pass must not create tracker state at all"
    );
}

/// A target the pass never evaluated (filtered out before evaluation) carries no
/// evidence either way and must not move the streak.
#[test]
fn unevaluated_target_does_not_extend_streak() {
    let _guard = guard();
    let target = "population-unevaluated";

    let summary = flush_target_pass_outcomes(
        &[TargetPassOutcome::new(target, false, false)],
        &completed_pass(0),
    );

    assert_eq!(summary.failures, 0);
    assert_eq!(summary.skipped, 1);
    assert_eq!(consecutive_failures(target), 0);
}

/// Acceptance criterion 4: `filter_cooldown_targets` — the function both
/// `apply_target_cooldown` call sites delegate to — actually removes a target
/// once the real, production-populated streak crosses the threshold. Proves the
/// `is_empty()` short-circuit is no longer the permanent path.
#[test]
fn cooldown_filter_removes_target_after_threshold() {
    let _guard = guard();
    let target = "population-cooldown-filter";
    let survivor = "population-cooldown-survivor";
    let threshold = cooldown_threshold();

    for _ in 0..threshold {
        flush_target_pass_outcomes(
            &[TargetPassOutcome::new(target, false, true)],
            &completed_pass(0),
        );
    }
    assert!(consecutive_failures(target) >= threshold);

    let epoch = current_epoch();
    let tracker = global_tracker()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    assert!(
        !tracker.is_empty(),
        "production population means the is_empty() short-circuit no longer fires"
    );

    let mut focus_order = vec![target.to_string(), survivor.to_string()];
    let skipped = filter_cooldown_targets(&mut focus_order, &tracker, epoch);

    assert_eq!(skipped, 1, "the cooled-down target must be dropped");
    assert_eq!(focus_order, vec![survivor.to_string()]);
}
