//! Target-neuron cooldown tracker for repeated discovery failures (Issue #1130).
//!
//! In production discovery-cache analysis, 17 of 18 `add-neurons` failure cache
//! entries in a single batch all targeted the same neuron (`neuron-1063112866`).
//! Different sources, different activations — all
//! failing. Evaluation budget was spent repeatedly probing a target that was
//! clearly not going to yield an improvement.
//!
//! This module records per-target consecutive failures and provides a cooldown
//! check so the preparation layers can skip targets that have failed too many
//! times in a row.
//!
//! # Design
//!
//! - Keyed by `target_neuron_uuid`.
//! - Tracks: consecutive failure count, last failure epoch, last success epoch.
//! - A target enters cooldown after
//!   [`TargetFailureTracker::cooldown_consecutive_failures`] consecutive
//!   failures. It stays in cooldown for
//!   [`TargetFailureTracker::cooldown_epochs`] epochs after the last failure.
//! - A success resets the consecutive failure counter to zero, clearing the
//!   cooldown immediately.
//! - The two thresholds are env-var overridable via
//!   `NEAT_AI_DISCOVERY_TARGET_COOLDOWN_FAILURES` and
//!   `NEAT_AI_DISCOVERY_TARGET_COOLDOWN_EPOCHS`.
//!
//! # Global Tracker
//!
//! A process-global tracker is exposed via [`global_tracker`]. The FFI
//! `record_discovery_result` hook (and analogues inside Rust) should forward
//! per-target pass/fail signals to the global tracker so preparation layers
//! consult a consistent view.
//!
//! # Operator playbook
//!
//! See [`docs/DROUGHT_PLAYBOOK.md`](https://github.com/stSoftwareAU/NEAT-AI-Discovery/blob/main/docs/DROUGHT_PLAYBOOK.md)
//! for the end-to-end drought diagnostic walkthrough — how this tracker, the
//! candidate outcome cache, conservative-mode bias, and post-processing
//! rejection filters interact, plus the operator escape hatch
//! (`NEAT_AI_DISCOVERY_DROUGHT_RESET_AFTER_EPOCHS`) that drops all active
//! cooldown entries in one shot.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use crate::analysis::constants::{
    COOLDOWN_EPOCHS_FLOOR, TARGET_COOLDOWN_CONSECUTIVE_FAILURES, TARGET_COOLDOWN_EPOCHS,
    cooldown_conservative_divisor, cooldown_extended_drought_divisor,
};
use crate::analysis::discovery_mode::{DEFAULT_CONSERVATIVE_MODE_MAX_EPOCHS, DiscoveryMode};
use crate::config::{target_cooldown_consecutive_failures_env, target_cooldown_epochs_env};

/// Per-target failure state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TargetState {
    /// Number of consecutive failures since the last success (or ever).
    pub consecutive_failures: u32,
    /// Epoch at which the most recent failure was recorded. `None` if never.
    pub last_failure_epoch: Option<u64>,
    /// Epoch at which the most recent success (improvement) was recorded.
    pub last_improvement_epoch: Option<u64>,
}

/// Tracks per-target failure streaks to enable cooldown skipping.
#[derive(Debug, Clone, Default)]
pub struct TargetFailureTracker {
    states: HashMap<String, TargetState>,
    /// Cooldown trigger — enter cooldown after this many consecutive failures.
    cooldown_consecutive_failures: u32,
    /// Cooldown duration — stay in cooldown for this many epochs after the
    /// most recent failure.
    cooldown_epochs: u64,
    /// Monotonic epoch counter used by the `_now` convenience methods. Callers
    /// that don't track their own epoch can call [`advance_epoch`] once per
    /// discovery run.
    current_epoch: u64,
    /// Epoch at which the drought-driven one-shot reset last fired (Issue
    /// #1205). `None` until the first reset; cleared by [`record_success`]
    /// so the next future drought can re-arm the lever.
    tombstone_reset_epoch: Option<u64>,
}

impl TargetFailureTracker {
    /// Creates a new tracker using the effective thresholds (env-var override
    /// if set, otherwise the compiled defaults).
    pub fn new() -> Self {
        Self {
            states: HashMap::new(),
            cooldown_consecutive_failures: target_cooldown_consecutive_failures_env()
                .unwrap_or(TARGET_COOLDOWN_CONSECUTIVE_FAILURES),
            cooldown_epochs: target_cooldown_epochs_env().unwrap_or(TARGET_COOLDOWN_EPOCHS),
            current_epoch: 0,
            tombstone_reset_epoch: None,
        }
    }

    /// Creates a new tracker with explicit thresholds (for tests).
    pub fn with_thresholds(cooldown_consecutive_failures: u32, cooldown_epochs: u64) -> Self {
        Self {
            states: HashMap::new(),
            cooldown_consecutive_failures,
            cooldown_epochs,
            current_epoch: 0,
            tombstone_reset_epoch: None,
        }
    }

    /// Returns the internal monotonic epoch counter.
    pub fn current_epoch(&self) -> u64 {
        self.current_epoch
    }

    /// Increments the internal epoch and returns the new value.
    ///
    /// Call this once per discovery run before consulting the tracker so the
    /// cooldown window advances. Externally-provided epochs (via the explicit
    /// `record_failure` / `is_in_cooldown` methods) are unaffected.
    pub fn advance_epoch(&mut self) -> u64 {
        self.current_epoch = self.current_epoch.saturating_add(1);
        self.current_epoch
    }

    /// Records a failure against `target_uuid` using the internal epoch.
    pub fn record_failure_now(&mut self, target_uuid: &str) {
        self.record_failure(target_uuid, self.current_epoch);
    }

    /// Records a success for `target_uuid` using the internal epoch.
    pub fn record_success_now(&mut self, target_uuid: &str) {
        self.record_success(target_uuid, self.current_epoch);
    }

    /// Checks cooldown using the internal epoch.
    pub fn is_in_cooldown_now(&self, target_uuid: &str) -> bool {
        self.is_in_cooldown(target_uuid, self.current_epoch)
    }

    /// Returns the configured consecutive-failure threshold.
    pub fn cooldown_consecutive_failures(&self) -> u32 {
        self.cooldown_consecutive_failures
    }

    /// Returns the configured cooldown duration in epochs.
    pub fn cooldown_epochs(&self) -> u64 {
        self.cooldown_epochs
    }

    /// Returns the number of tracked targets.
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Returns `true` when no targets are tracked.
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    /// Returns the current state for a target, if any.
    pub fn state(&self, target_uuid: &str) -> Option<&TargetState> {
        self.states.get(target_uuid)
    }

    /// Records a discovery failure against `target_uuid` at `epoch`.
    ///
    /// Increments the consecutive-failure counter and updates the last
    /// failure epoch.
    pub fn record_failure(&mut self, target_uuid: &str, epoch: u64) {
        let entry = self.states.entry(target_uuid.to_string()).or_default();
        entry.consecutive_failures = entry.consecutive_failures.saturating_add(1);
        entry.last_failure_epoch = Some(epoch);
    }

    /// Records a per-target failure unless the pass was environmentally
    /// disabled (Issue #1421).
    ///
    /// A memory/GPU-gated pass never evaluated the target, so it is not
    /// evidence of failure and must not extend the cooldown streak. Returns
    /// `true` when the failure was recorded, `false` when it was skipped.
    pub fn record_failure_unless_disabled(
        &mut self,
        target_uuid: &str,
        epoch: u64,
        outcome: &crate::analysis::AnalysisOutcome,
    ) -> bool {
        if outcome.is_environmentally_disabled() {
            return false;
        }
        self.record_failure(target_uuid, epoch);
        true
    }

    /// Records a successful improvement for `target_uuid` at `epoch`.
    ///
    /// Resets the consecutive-failure counter to zero (clearing any active
    /// cooldown) and records the epoch.
    ///
    /// A success also clears the drought-reset tombstone (Issue #1205) so
    /// the next future drought can re-arm the one-shot reset.
    pub fn record_success(&mut self, target_uuid: &str, epoch: u64) {
        let entry = self.states.entry(target_uuid.to_string()).or_default();
        entry.consecutive_failures = 0;
        entry.last_improvement_epoch = Some(epoch);
        self.tombstone_reset_epoch = None;
    }

    /// Returns `true` when `target_uuid` is currently in cooldown at `current_epoch`.
    ///
    /// A target is in cooldown when:
    /// 1. Consecutive failures ≥ [`Self::cooldown_consecutive_failures`], AND
    /// 2. `current_epoch` < `last_failure_epoch + cooldown_epochs`.
    ///
    /// Targets never-seen or below the consecutive-failure threshold are NOT
    /// in cooldown. A prior success (counter reset) also clears cooldown.
    pub fn is_in_cooldown(&self, target_uuid: &str, current_epoch: u64) -> bool {
        let Some(state) = self.states.get(target_uuid) else {
            return false;
        };
        if state.consecutive_failures < self.cooldown_consecutive_failures {
            return false;
        }
        let Some(failure_epoch) = state.last_failure_epoch else {
            return false;
        };
        current_epoch < failure_epoch.saturating_add(self.cooldown_epochs)
    }

    /// Returns the effective cooldown window in epochs for the given discovery
    /// mode and drought state (Issue #1204).
    ///
    /// The base [`Self::cooldown_epochs`] is divided by the conservative
    /// divisor when the pipeline is in [`DiscoveryMode::Conservative`], and by
    /// the extended-drought divisor once `drought_failures` has met or
    /// exceeded [`DEFAULT_CONSERVATIVE_MODE_MAX_EPOCHS`]. Extended drought
    /// applies regardless of `mode`. The effective window is never smaller
    /// than [`COOLDOWN_EPOCHS_FLOOR`].
    #[must_use]
    pub fn effective_cooldown_epochs(&self, mode: DiscoveryMode, drought_failures: u32) -> u64 {
        let divisor = if drought_failures >= DEFAULT_CONSERVATIVE_MODE_MAX_EPOCHS {
            cooldown_extended_drought_divisor()
        } else if mode == DiscoveryMode::Conservative {
            cooldown_conservative_divisor()
        } else {
            1
        }
        .max(1);
        (self.cooldown_epochs / divisor).max(COOLDOWN_EPOCHS_FLOOR)
    }

    /// Returns the effective consecutive-failure trigger for the given
    /// discovery mode and drought state (Issue #1204).
    ///
    /// The configured trigger is raised by `+1` in
    /// [`DiscoveryMode::Conservative`] mode and by `+2` once `drought_failures`
    /// has met or exceeded [`DEFAULT_CONSERVATIVE_MODE_MAX_EPOCHS`] (extended
    /// drought). Saturating add — never overflows.
    #[must_use]
    pub fn effective_consecutive_failures(
        &self,
        mode: DiscoveryMode,
        drought_failures: u32,
    ) -> u32 {
        let bump: u32 = if drought_failures >= DEFAULT_CONSERVATIVE_MODE_MAX_EPOCHS {
            2
        } else if mode == DiscoveryMode::Conservative {
            1
        } else {
            0
        };
        self.cooldown_consecutive_failures.saturating_add(bump)
    }

    /// Adaptive cooldown check using the relaxed thresholds for the given
    /// mode and drought state (Issue #1204).
    ///
    /// Identical contract to [`Self::is_in_cooldown`] but the consecutive
    /// failure trigger and the cooldown window are sourced from
    /// [`Self::effective_consecutive_failures`] and
    /// [`Self::effective_cooldown_epochs`] respectively.
    #[must_use]
    pub fn is_in_cooldown_adaptive(
        &self,
        target_uuid: &str,
        current_epoch: u64,
        mode: DiscoveryMode,
        drought_failures: u32,
    ) -> bool {
        let Some(state) = self.states.get(target_uuid) else {
            return false;
        };
        let effective_trigger = self.effective_consecutive_failures(mode, drought_failures);
        if state.consecutive_failures < effective_trigger {
            return false;
        }
        let Some(failure_epoch) = state.last_failure_epoch else {
            return false;
        };
        let effective_cooldown = self.effective_cooldown_epochs(mode, drought_failures);
        current_epoch < failure_epoch.saturating_add(effective_cooldown)
    }

    /// Returns the number of targets currently in cooldown at `current_epoch`
    /// (Issue #1202).
    ///
    /// Used by the drought diagnostic to show how many target neurons are
    /// being skipped by the per-target failure tracker. Targets whose cooldown
    /// window has elapsed, or whose consecutive-failure count is below the
    /// configured threshold, are not counted.
    #[must_use]
    pub fn active_cooldown_count(&self, current_epoch: u64) -> usize {
        self.states
            .iter()
            .filter(|(_, state)| {
                if state.consecutive_failures < self.cooldown_consecutive_failures {
                    return false;
                }
                let Some(failure_epoch) = state.last_failure_epoch else {
                    return false;
                };
                current_epoch < failure_epoch.saturating_add(self.cooldown_epochs)
            })
            .count()
    }

    /// Returns the epoch at which the drought-driven reset was last fired,
    /// if any (Issue #1205).
    #[must_use]
    pub fn drought_reset_tombstone(&self) -> Option<u64> {
        self.tombstone_reset_epoch
    }

    /// Clear the drought-reset tombstone explicitly (Issue #1205).
    ///
    /// Normal usage relies on [`Self::record_success`] to clear the tombstone
    /// when a target records an improvement. The orchestrator may also call
    /// this directly after a successful pass at the outcome-log level.
    pub fn clear_drought_reset_tombstone(&mut self) {
        self.tombstone_reset_epoch = None;
    }

    /// Drop every target that is currently in cooldown at `current_epoch`
    /// (Issue #1205).
    ///
    /// Operator-controlled escape hatch invoked after a configurable drought.
    /// States whose `consecutive_failures` are below the cooldown threshold
    /// are preserved (they are tracking but not actively suppressing). The
    /// drought-reset tombstone is set to `current_epoch` so the same streak
    /// cannot trigger a second reset; a subsequent [`Self::record_success`]
    /// re-arms the lever.
    ///
    /// Returns the number of cooldown entries removed.
    pub fn clear_cooldown_entries(&mut self, current_epoch: u64) -> usize {
        let cooldown_threshold = self.cooldown_consecutive_failures;
        let cooldown_epochs = self.cooldown_epochs;
        let before = self.states.len();
        self.states.retain(|_, state| {
            if state.consecutive_failures < cooldown_threshold {
                return true;
            }
            let Some(failure_epoch) = state.last_failure_epoch else {
                return true;
            };
            current_epoch >= failure_epoch.saturating_add(cooldown_epochs)
        });
        let removed = before.saturating_sub(self.states.len());
        self.tombstone_reset_epoch = Some(current_epoch);
        removed
    }
}

/// Remove focus targets currently in cooldown.
///
/// Returns the number of targets dropped so callers can emit a diagnostic
/// counter (the `cooldown_skipped` reason-name per Issue #1129's convention).
pub fn filter_cooldown_targets(
    focus_order: &mut Vec<String>,
    tracker: &TargetFailureTracker,
    current_epoch: u64,
) -> u32 {
    let before = focus_order.len();
    focus_order.retain(|target| !tracker.is_in_cooldown(target, current_epoch));
    let after = focus_order.len();
    let skipped = u32::try_from(before.saturating_sub(after)).unwrap_or(u32::MAX);
    if skipped > 0 {
        tracing::info!(
            cooldown_skipped = skipped,
            remaining_targets = after,
            cooldown_epochs = tracker.cooldown_epochs(),
            cooldown_consecutive_failures = tracker.cooldown_consecutive_failures(),
            "Dropped focus targets currently in cooldown"
        );
    }
    skipped
}

/// Remove focus targets currently in cooldown using adaptive thresholds
/// (Issue #1204).
///
/// Same contract as [`filter_cooldown_targets`] but consults
/// [`TargetFailureTracker::is_in_cooldown_adaptive`], so the effective trigger
/// and cooldown window shrink in [`DiscoveryMode::Conservative`] or during an
/// extended drought.
pub fn filter_cooldown_targets_adaptive(
    focus_order: &mut Vec<String>,
    tracker: &TargetFailureTracker,
    current_epoch: u64,
    mode: DiscoveryMode,
    drought_failures: u32,
) -> u32 {
    let before = focus_order.len();
    focus_order.retain(|target| {
        !tracker.is_in_cooldown_adaptive(target, current_epoch, mode, drought_failures)
    });
    let after = focus_order.len();
    let skipped = u32::try_from(before.saturating_sub(after)).unwrap_or(u32::MAX);
    if skipped > 0 {
        tracing::info!(
            cooldown_skipped = skipped,
            remaining_targets = after,
            cooldown_epochs = tracker.effective_cooldown_epochs(mode, drought_failures),
            cooldown_consecutive_failures =
                tracker.effective_consecutive_failures(mode, drought_failures),
            mode = mode.as_str(),
            drought_failures,
            "Dropped focus targets currently in cooldown (adaptive)"
        );
    }
    skipped
}

/// Process-global target failure tracker.
///
/// Preparation layers consult this tracker to skip targets in cooldown. The
/// FFI `record_discovery_result` pathway (or any Rust caller) updates it with
/// per-target pass/fail outcomes so the next discovery run can skip repeat
/// losers.
pub fn global_tracker() -> &'static Mutex<TargetFailureTracker> {
    static TRACKER: OnceLock<Mutex<TargetFailureTracker>> = OnceLock::new();
    TRACKER.get_or_init(|| Mutex::new(TargetFailureTracker::new()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_tracker_is_empty_and_has_defaults() {
        let tracker = TargetFailureTracker::with_thresholds(3, 10);
        assert!(tracker.is_empty());
        assert_eq!(tracker.len(), 0);
        assert_eq!(tracker.cooldown_consecutive_failures(), 3);
        assert_eq!(tracker.cooldown_epochs(), 10);
    }

    /// Issue #1421: environmentally-disabled passes never increment the
    /// per-target cooldown streak.
    #[test]
    fn gated_passes_never_trip_target_cooldown() {
        use crate::analysis::{AnalysisOutcome, EnvironmentalDisableReason};

        let disabled = AnalysisOutcome::EnvironmentallyDisabled {
            reason: EnvironmentalDisableReason::GpuUnavailable,
        };
        let mut tracker = TargetFailureTracker::with_thresholds(3, 10);
        // Far more gated passes than the threshold — none must count.
        for epoch in 0..10 {
            let recorded = tracker.record_failure_unless_disabled("T", epoch, &disabled);
            assert!(!recorded, "gated pass must not be recorded");
        }
        assert!(tracker.state("T").is_none());
        assert!(!tracker.is_in_cooldown("T", 10));
    }

    /// Issue #1421: a genuine empty pass still records and trips cooldown.
    #[test]
    fn genuine_empty_pass_records_via_guard() {
        use crate::analysis::AnalysisOutcome;

        let empty = AnalysisOutcome::Completed { candidates: 0 };
        let mut tracker = TargetFailureTracker::with_thresholds(2, 10);
        assert!(tracker.record_failure_unless_disabled("T", 0, &empty));
        assert!(tracker.record_failure_unless_disabled("T", 1, &empty));
        assert_eq!(tracker.state("T").expect("state").consecutive_failures, 2);
        assert!(tracker.is_in_cooldown("T", 1));
    }

    /// Transition 1: `record_failure` increments the counter.
    #[test]
    fn record_failure_increments_consecutive_counter() {
        let mut tracker = TargetFailureTracker::with_thresholds(3, 10);
        tracker.record_failure("T", 0);
        tracker.record_failure("T", 1);
        let state = tracker.state("T").expect("state must exist");
        assert_eq!(state.consecutive_failures, 2);
        assert_eq!(state.last_failure_epoch, Some(1));
        assert!(state.last_improvement_epoch.is_none());
    }

    /// Transition 2: hitting the consecutive-failure threshold triggers cooldown.
    #[test]
    fn threshold_triggers_cooldown() {
        let mut tracker = TargetFailureTracker::with_thresholds(3, 10);

        // Below threshold -> not in cooldown.
        tracker.record_failure("T", 0);
        tracker.record_failure("T", 1);
        assert!(!tracker.is_in_cooldown("T", 2));

        // At threshold and within window -> in cooldown.
        tracker.record_failure("T", 2);
        assert!(tracker.is_in_cooldown("T", 2));
        assert!(tracker.is_in_cooldown("T", 11));

        // Past the cooldown window -> released.
        assert!(!tracker.is_in_cooldown("T", 12));
    }

    /// Transition 3: a success resets the consecutive-failure counter and
    /// clears cooldown immediately.
    #[test]
    fn success_resets_counter_and_clears_cooldown() {
        let mut tracker = TargetFailureTracker::with_thresholds(3, 10);
        tracker.record_failure("T", 0);
        tracker.record_failure("T", 1);
        tracker.record_failure("T", 2);
        assert!(tracker.is_in_cooldown("T", 2));

        tracker.record_success("T", 3);
        let state = tracker.state("T").expect("state must exist");
        assert_eq!(state.consecutive_failures, 0);
        assert_eq!(state.last_improvement_epoch, Some(3));
        assert!(!tracker.is_in_cooldown("T", 3));
    }

    #[test]
    fn unknown_target_is_not_in_cooldown() {
        let tracker = TargetFailureTracker::with_thresholds(3, 10);
        assert!(!tracker.is_in_cooldown("never-seen", 5));
    }

    #[test]
    fn filter_cooldown_targets_drops_only_cooldown_entries() {
        let mut tracker = TargetFailureTracker::with_thresholds(3, 10);
        // Target A: 3 failures -> cooldown.
        tracker.record_failure("A", 0);
        tracker.record_failure("A", 1);
        tracker.record_failure("A", 2);
        // Target B: only 2 failures -> not in cooldown.
        tracker.record_failure("B", 0);
        tracker.record_failure("B", 1);
        // Target C: untracked -> not in cooldown.
        let mut focus = vec!["A".to_string(), "B".to_string(), "C".to_string()];

        let skipped = filter_cooldown_targets(&mut focus, &tracker, 3);

        assert_eq!(skipped, 1);
        assert_eq!(focus, vec!["B".to_string(), "C".to_string()]);
    }

    #[test]
    fn filter_cooldown_targets_respects_epoch_window() {
        let mut tracker = TargetFailureTracker::with_thresholds(2, 5);
        tracker.record_failure("A", 0);
        tracker.record_failure("A", 1);
        // Within window.
        let mut focus = vec!["A".to_string()];
        assert_eq!(filter_cooldown_targets(&mut focus, &tracker, 3), 1);
        assert!(focus.is_empty());

        // Past window -> no longer skipped.
        let mut focus = vec!["A".to_string()];
        assert_eq!(filter_cooldown_targets(&mut focus, &tracker, 10), 0);
        assert_eq!(focus, vec!["A".to_string()]);
    }

    // -------------------------------------------------------------------
    // Issue #1205 — clear_cooldown_entries + tombstone semantics
    // -------------------------------------------------------------------

    #[test]
    fn clear_cooldown_entries_empty_tracker_returns_zero() {
        let mut tracker = TargetFailureTracker::with_thresholds(3, 10);
        assert_eq!(tracker.clear_cooldown_entries(0), 0);
        // Tombstone is set even when nothing was removed — the lever fired.
        assert_eq!(tracker.drought_reset_tombstone(), Some(0));
    }

    #[test]
    fn clear_cooldown_entries_removes_only_in_cooldown_targets() {
        // cooldown_epochs = 2 so timing differences between targets are
        // visible without overlapping windows.
        let mut tracker = TargetFailureTracker::with_thresholds(3, 2);
        // Target A: 3 failures within window -> currently in cooldown at 12.
        // Last failure at 11, cooldown until 13.
        tracker.record_failure("A", 9);
        tracker.record_failure("A", 10);
        tracker.record_failure("A", 11);
        // Target B: only 2 failures -> below threshold, retained.
        tracker.record_failure("B", 10);
        tracker.record_failure("B", 11);
        // Target C: 3 failures but the cooldown window has elapsed -> retained.
        // Last failure at 2, cooldown until 4, current_epoch is 12.
        tracker.record_failure("C", 0);
        tracker.record_failure("C", 1);
        tracker.record_failure("C", 2);

        let removed = tracker.clear_cooldown_entries(12);
        assert_eq!(removed, 1);
        assert!(tracker.state("A").is_none());
        assert!(tracker.state("B").is_some());
        assert!(tracker.state("C").is_some());
        assert_eq!(tracker.drought_reset_tombstone(), Some(12));
    }

    #[test]
    fn clear_cooldown_entries_all_in_cooldown() {
        let mut tracker = TargetFailureTracker::with_thresholds(2, 100);
        for tgt in ["X", "Y", "Z"] {
            tracker.record_failure(tgt, 0);
            tracker.record_failure(tgt, 1);
        }
        let removed = tracker.clear_cooldown_entries(5);
        assert_eq!(removed, 3);
        assert!(tracker.is_empty());
    }

    #[test]
    fn record_success_clears_tombstone() {
        let mut tracker = TargetFailureTracker::with_thresholds(2, 100);
        tracker.record_failure("T", 0);
        tracker.record_failure("T", 1);
        let _ = tracker.clear_cooldown_entries(2);
        assert_eq!(tracker.drought_reset_tombstone(), Some(2));

        tracker.record_success("Other", 3);
        assert!(tracker.drought_reset_tombstone().is_none());
    }

    #[test]
    fn clear_drought_reset_tombstone_resets() {
        let mut tracker = TargetFailureTracker::with_thresholds(2, 100);
        let _ = tracker.clear_cooldown_entries(7);
        assert_eq!(tracker.drought_reset_tombstone(), Some(7));
        tracker.clear_drought_reset_tombstone();
        assert!(tracker.drought_reset_tombstone().is_none());
    }

    /// Integration-style: a sequence of 3 consecutive failures on target T
    /// causes T to be skipped on the next discovery preparation within the
    /// cooldown window, while a different target is retained.
    #[test]
    fn three_consecutive_failures_skip_target_on_next_run() {
        let mut tracker = TargetFailureTracker::with_thresholds(3, 10);

        // Epochs 0..=2: all failures on target T.
        for epoch in 0..3u64 {
            tracker.record_failure("target-1063112866", epoch);
        }

        // Epoch 3 represents the next discovery run's preparation.
        let mut focus = vec![
            "target-1063112866".to_string(),
            "target-healthy".to_string(),
        ];
        let skipped = filter_cooldown_targets(&mut focus, &tracker, 3);

        assert_eq!(skipped, 1);
        assert_eq!(focus, vec!["target-healthy".to_string()]);
    }
}
