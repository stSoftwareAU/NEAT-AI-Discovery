//! Operator escape hatch — forced one-shot reset after a configurable drought
//! (Issue #1205).
//!
//! When the `NEAT_AI_DISCOVERY_DROUGHT_RESET_AFTER_EPOCHS` environment variable
//! is set and the rolling outcome log reports at least that many consecutive
//! trailing failures, the orchestrator invokes
//! [`maybe_perform_drought_reset`] to flush the suppression layer that may be
//! blocking new candidates: the [`TargetFailureTracker`] drops all targets that
//! are currently in cooldown (below-threshold tracking is preserved).
//!
//! The tracker carries a `tombstone_reset_epoch` so the same drought streak
//! triggers at most one reset. A subsequent successful outcome (recorded via
//! `TargetFailureTracker::record_success`) clears the tombstone and re-arms the
//! lever for the next future drought.
//!
//! Issue #1792: the reset also used to clear failed `CandidateOutcomeCache`
//! entries, but that cache was never constructed outside tests — the parameter
//! was permanently `None`, so the clearing step never ran. The cache and the
//! parameter have been deleted rather than left advertising work the reset does
//! not do.
//!
//! Issue #1794: a reset that clears nothing is a *no-op*, not a success. It
//! logs at `ERROR` with wording that never claims work was done, names why it
//! was ineffective (input unwired versus wired-but-empty), and — crucially —
//! does not stamp the one-shot tombstone, so an ineffective reset at epoch N no
//! longer blocks an effective one at epoch N+1.

use super::target_failure_tracker::TargetFailureTracker;

/// State of the reset's one clearable input, the [`TargetFailureTracker`]
/// (Issue #1794).
///
/// Distinguishes "not wired into the reset at all" from "wired but had nothing
/// to clear" — very different diagnoses that the old log could not tell apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DroughtResetInputState {
    /// No tracker was supplied (`None`) — the input is not wired in.
    Unwired,
    /// A tracker was supplied but held no active cooldown entries.
    WiredEmpty,
    /// A tracker was supplied and at least one cooldown entry was dropped.
    Cleared,
}

impl DroughtResetInputState {
    /// Stable log-field label for this state.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Unwired => "unwired",
            Self::WiredEmpty => "wired_empty",
            Self::Cleared => "cleared",
        }
    }
}

/// Outcome of a single drought-reset invocation (Issue #1205).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DroughtResetOutcome {
    /// Number of target-failure-tracker entries dropped from active cooldown
    /// (0 when no tracker was supplied).
    pub target_cooldown_cleared: usize,
    /// Why the tracker input did or did not yield work (Issue #1794).
    pub target_tracker_input: DroughtResetInputState,
    /// The epoch at which the reset fired — recorded in both tombstones.
    pub reset_epoch: u64,
    /// Configured `drought_reset_after_epochs` value that armed this reset.
    pub drought_reset_after: u32,
}

impl DroughtResetOutcome {
    /// `true` when the escape hatch fired but flushed no suppression state
    /// (Issue #1794). A no-op outcome leaves the lever armed — no tombstone is
    /// stamped — so callers may treat it as "the drought is not cooldown-bound".
    #[must_use]
    pub fn is_noop(&self) -> bool {
        self.target_cooldown_cleared == 0
    }
}

/// Perform the one-shot drought reset when the trigger conditions hold.
///
/// Returns `Some(outcome)` exactly when the reset fired. Returns `None` when:
/// - The lever is disabled (`drought_reset_after == 0`).
/// - The consecutive-failure streak has not yet crossed `drought_reset_after`.
/// - The tracker already carries a tombstone for the current streak (a reset
///   has fired and no success has re-armed the lever).
///
/// The orchestrator also clears the tombstone when `consecutive_failures == 0`
/// so a successful pass re-arms the lever even if the per-target success path
/// was not exercised.
///
/// Issue #1794: when the reset clears nothing the returned outcome reports
/// [`DroughtResetOutcome::is_noop`], the log line says plainly that nothing was
/// reset, and the tombstone is left unstamped so a later pass in the same
/// streak can still do real work.
pub fn maybe_perform_drought_reset(
    tracker: Option<&mut TargetFailureTracker>,
    consecutive_failures: u32,
    drought_reset_after: u32,
    current_epoch: u64,
) -> Option<DroughtResetOutcome> {
    // Lever disabled or threshold not reached.
    if drought_reset_after == 0 || consecutive_failures < drought_reset_after {
        return None;
    }

    // One-shot semantics: if the tracker has already tombstoned the current
    // streak, do nothing. A successful pass clears the tombstone before this
    // point, so a fresh drought naturally re-fires.
    if tracker
        .as_ref()
        .is_some_and(|t| t.drought_reset_tombstone().is_some())
    {
        return None;
    }

    let (target_cooldown_cleared, target_tracker_input) = match tracker {
        Some(t) => {
            let cleared = t.clear_cooldown_entries(current_epoch);
            if cleared == 0 {
                // Issue #1794: an ineffective reset must not consume the
                // one-shot for the streak. `clear_cooldown_entries` stamps the
                // tombstone unconditionally, so unwind it here.
                t.clear_drought_reset_tombstone();
                (0, DroughtResetInputState::WiredEmpty)
            } else {
                (cleared, DroughtResetInputState::Cleared)
            }
        }
        None => (0, DroughtResetInputState::Unwired),
    };

    if target_cooldown_cleared == 0 {
        tracing::error!(
            reset_name = "drought_escape_hatch_noop",
            noop = true,
            target_cooldown_cleared,
            target_tracker_input = target_tracker_input.label(),
            tombstone_stamped = false,
            consecutive_failures,
            drought_reset_after,
            current_epoch,
            env_var = "NEAT_AI_DISCOVERY_DROUGHT_RESET_AFTER_EPOCHS",
            "Issue #1794: drought escape hatch fired as a NO-OP — nothing was reset (target \
             failure tracker: {}) after {} consecutive empty passes; the one-shot tombstone was \
             not stamped, so the lever stays armed for this streak",
            target_tracker_input.label(),
            consecutive_failures,
        );
    } else {
        tracing::warn!(
            reset_name = "drought_escape_hatch",
            target_cooldown_cleared,
            target_tracker_input = target_tracker_input.label(),
            consecutive_failures,
            drought_reset_after,
            current_epoch,
            env_var = "NEAT_AI_DISCOVERY_DROUGHT_RESET_AFTER_EPOCHS",
            "Issue #1205: drought escape hatch fired — cleared {} active target cooldowns after \
             {} consecutive empty passes",
            target_cooldown_cleared,
            consecutive_failures,
        );
    }

    Some(DroughtResetOutcome {
        target_cooldown_cleared,
        target_tracker_input,
        reset_epoch: current_epoch,
        drought_reset_after,
    })
}

/// Re-arm the lever by clearing the drought-reset tombstone on the tracker
/// (Issue #1205). The orchestrator calls this when the rolling outcome log
/// reports a fresh success (`consecutive_failures == 0`).
pub fn rearm_drought_reset(tracker: Option<&mut TargetFailureTracker>) {
    if let Some(t) = tracker {
        t.clear_drought_reset_tombstone();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn populated_tracker() -> TargetFailureTracker {
        let mut tracker = TargetFailureTracker::with_thresholds(2, 100);
        // Two targets in cooldown.
        tracker.record_failure("A", 0);
        tracker.record_failure("A", 1);
        tracker.record_failure("B", 0);
        tracker.record_failure("B", 1);
        // One target with only one failure -> below threshold, retained.
        tracker.record_failure("C", 0);
        tracker
    }

    #[test]
    fn disabled_lever_never_fires() {
        let mut tracker = populated_tracker();
        let out = maybe_perform_drought_reset(Some(&mut tracker), 100, 0, 5);
        assert!(out.is_none());
        assert_eq!(tracker.len(), 3); // unchanged
    }

    #[test]
    fn below_threshold_does_not_fire() {
        let mut tracker = populated_tracker();
        let out = maybe_perform_drought_reset(Some(&mut tracker), 9, 10, 5);
        assert!(out.is_none());
    }

    #[test]
    fn at_threshold_fires_once_and_tombstones() {
        let mut tracker = populated_tracker();

        let out =
            maybe_perform_drought_reset(Some(&mut tracker), 10, 10, 5).expect("fires at threshold");
        assert_eq!(out.target_cooldown_cleared, 2);
        assert_eq!(out.reset_epoch, 5);

        // Below-threshold tracker entry preserved, cooldown entries gone.
        assert_eq!(tracker.len(), 1);
        assert!(tracker.state("C").is_some());

        assert_eq!(tracker.drought_reset_tombstone(), Some(5));
    }

    #[test]
    fn second_call_within_streak_is_noop() {
        let mut tracker = populated_tracker();

        let _first =
            maybe_perform_drought_reset(Some(&mut tracker), 10, 10, 5).expect("first fires");
        // Re-add a fresh cooldown to prove the second call leaves it untouched.
        tracker.record_failure("D", 6);
        tracker.record_failure("D", 7);

        let second = maybe_perform_drought_reset(Some(&mut tracker), 11, 10, 7);
        assert!(second.is_none(), "second invocation must not re-fire");

        assert!(tracker.state("D").is_some());
    }

    #[test]
    fn rearm_after_success_allows_next_reset() {
        let mut tracker = populated_tracker();

        // First fire.
        let _ = maybe_perform_drought_reset(Some(&mut tracker), 10, 10, 5).expect("first fires");

        // Re-arm via a successful outcome.
        tracker.record_success("good-tgt", 6);
        assert!(tracker.drought_reset_tombstone().is_none());

        // Set up a second drought.
        tracker.record_failure("E", 7);
        tracker.record_failure("E", 8);

        let second = maybe_perform_drought_reset(Some(&mut tracker), 10, 10, 9);
        assert!(second.is_some(), "re-armed lever fires again");
    }

    #[test]
    fn rearm_via_helper_when_outcome_path_unused() {
        let mut tracker = populated_tracker();
        let _ = maybe_perform_drought_reset(Some(&mut tracker), 10, 10, 5);
        rearm_drought_reset(Some(&mut tracker));
        assert!(tracker.drought_reset_tombstone().is_none());
    }

    /// Issue #1790: the orchestrator sources `current_epoch` from the global
    /// tracker's internal counter (`orchestration.rs`). Now that the counter
    /// actually advances, the reset must fire at that non-zero epoch and stamp
    /// the tombstone with it — not with `0`.
    #[test]
    fn reset_at_advanced_internal_epoch_stamps_non_zero_tombstone() {
        let mut tracker = TargetFailureTracker::with_thresholds(2, 100);
        // Three discovery passes' worth of epoch advance, then two failures
        // recorded through the internal-counter path.
        tracker.advance_epoch();
        tracker.advance_epoch();
        tracker.advance_epoch();
        tracker.record_failure_now("A");
        tracker.record_failure_now("A");
        assert!(tracker.is_in_cooldown_now("A"));

        // Mirrors `orchestration.rs`: epoch_for_reset = guard.current_epoch().
        let epoch_for_reset = tracker.current_epoch();
        assert_eq!(epoch_for_reset, 3, "the internal counter must have moved");

        let out = maybe_perform_drought_reset(Some(&mut tracker), 10, 10, epoch_for_reset)
            .expect("fires at the advanced epoch");

        assert_eq!(out.reset_epoch, 3);
        assert_eq!(out.target_cooldown_cleared, 1);
        assert!(!tracker.is_in_cooldown_now("A"));
        assert_eq!(
            tracker.drought_reset_tombstone(),
            Some(3),
            "tombstone must carry the current non-zero epoch"
        );
    }

    /// Issue #1792: with no tracker supplied the lever still fires and simply
    /// reports that it cleared nothing — it never silently reports work it did
    /// not do.
    #[test]
    fn fires_without_tracker_supplied() {
        let out = maybe_perform_drought_reset(None, 10, 10, 5).expect("fires with no tracker");
        assert_eq!(out.target_cooldown_cleared, 0);
        // Issue #1794: an unwired input is reported as such, not as success.
        assert!(out.is_noop());
        assert_eq!(out.target_tracker_input, DroughtResetInputState::Unwired);
    }

    /// Issue #1794: a wired-but-empty tracker is a distinct diagnosis from an
    /// unwired one, and neither may tombstone the streak.
    #[test]
    fn noop_reset_leaves_lever_armed_for_the_streak() {
        let mut tracker = TargetFailureTracker::with_thresholds(2, 100);
        tracker.record_failure("C", 0); // below threshold — nothing in cooldown

        let first = maybe_perform_drought_reset(Some(&mut tracker), 10, 10, 5)
            .expect("fires with an empty tracker");
        assert!(first.is_noop());
        assert_eq!(
            first.target_tracker_input,
            DroughtResetInputState::WiredEmpty
        );
        assert!(
            tracker.drought_reset_tombstone().is_none(),
            "a no-op reset must not consume the one-shot"
        );

        // Cooldowns appear later in the same streak — the lever still fires.
        tracker.record_failure("A", 6);
        tracker.record_failure("A", 7);
        let second = maybe_perform_drought_reset(Some(&mut tracker), 11, 10, 8)
            .expect("still armed within the streak");
        assert_eq!(second.target_cooldown_cleared, 1);
        assert_eq!(second.target_tracker_input, DroughtResetInputState::Cleared);
        assert_eq!(tracker.drought_reset_tombstone(), Some(8));
    }
}
