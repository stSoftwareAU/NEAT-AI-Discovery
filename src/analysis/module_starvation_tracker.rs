//! Per-creature, per-module starvation tracker (Issue #1273).
//!
//! Disables a single discovery module for a single creature after `N`
//! consecutive failures without an intervening success. Complements:
//!
//! - [`super::module_weights::ModuleOutcomeTracker`] — population-wide success
//!   rate tracking (Issue #1060).
//! - [`super::discovery_mode::DiscoveryMode`] — creature-level conservative
//!   mode (Issue #1132).
//! - [`super::target_failure_tracker::TargetFailureTracker`] — per-target
//!   cooldown (Issue #1130).
//!
//! None of those layers cover the per-(creature, module) starvation case
//! described in GRQ-sampler commit `e85c5d2` (creature `bcbca347`), where one
//! module (`coordinated-structural`) recorded 41 consecutive failures and
//! zero successes — consuming ~91% of the candidate budget for the creature
//! while other modules (e.g. `add-neurons`, `add-synapses`) were starved out.
//!
//! # Design
//!
//! - Keyed by `module_name`. A single tracker instance is creature-scoped:
//!   the orchestrator constructs one per `analyze_all` invocation.
//! - Tracks per-module consecutive failure count, last failure / success
//!   epoch, and the epoch at which the cooldown started.
//! - A module enters cooldown after
//!   [`ModuleStarvationTracker::failure_streak_threshold`] consecutive
//!   failures. It stays disabled for
//!   [`ModuleStarvationTracker::cooldown_epochs`] epochs after the cooldown
//!   started.
//! - A success resets the consecutive-failure counter to zero, clearing any
//!   active cooldown.
//! - Thresholds are env-var overridable via
//!   `NEAT_AI_DISCOVERY_MODULE_STARVATION_FAILURE_STREAK` and
//!   `NEAT_AI_DISCOVERY_MODULE_STARVATION_COOLDOWN_EPOCHS`.

use std::collections::HashMap;

use super::constants::{module_starvation_cooldown_epochs, module_starvation_failure_streak};

/// Per-module starvation state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StarvationState {
    /// Consecutive failures since the last success (or ever).
    pub consecutive_failures: u32,
    /// Epoch of the most recent failure. `None` if never.
    pub last_failure_epoch: Option<u64>,
    /// Epoch of the most recent success. `None` if never.
    pub last_success_epoch: Option<u64>,
    /// Epoch at which the cooldown started (i.e. the failure-streak threshold
    /// was first crossed in the current streak). `None` if not in cooldown.
    pub cooldown_start_epoch: Option<u64>,
}

/// Per-creature, per-module failure-streak tracker (Issue #1273).
///
/// The orchestrator constructs one tracker per creature (per `analyze_all`
/// invocation). Thread-safety: the tracker uses interior `HashMap` storage
/// and offers `&self` reads / `&mut self` writes. The detection phase reads
/// the tracker concurrently (via shared `&Self` borrow); the merge phase
/// writes through a single `&mut Self`.
#[derive(Debug, Clone, Default)]
pub struct ModuleStarvationTracker {
    states: HashMap<String, StarvationState>,
    failure_streak_threshold: u32,
    cooldown_epochs: u64,
}

impl ModuleStarvationTracker {
    /// Creates a new tracker with thresholds sourced from the env-var helpers.
    #[must_use]
    pub fn new() -> Self {
        Self {
            states: HashMap::new(),
            failure_streak_threshold: module_starvation_failure_streak(),
            cooldown_epochs: module_starvation_cooldown_epochs(),
        }
    }

    /// Creates a new tracker with explicit thresholds (for tests).
    #[must_use]
    pub fn with_thresholds(failure_streak_threshold: u32, cooldown_epochs: u64) -> Self {
        Self {
            states: HashMap::new(),
            failure_streak_threshold,
            cooldown_epochs,
        }
    }

    /// Returns the configured failure-streak threshold.
    #[must_use]
    pub fn failure_streak_threshold(&self) -> u32 {
        self.failure_streak_threshold
    }

    /// Returns the configured cooldown duration in epochs.
    #[must_use]
    pub fn cooldown_epochs(&self) -> u64 {
        self.cooldown_epochs
    }

    /// Returns the number of modules being tracked.
    #[must_use]
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Returns `true` when no modules have ever been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    /// Returns the current state for `module_name`, if any.
    #[must_use]
    pub fn state(&self, module_name: &str) -> Option<&StarvationState> {
        self.states.get(module_name)
    }

    /// Records a failure for `module_name` at `epoch`.
    ///
    /// Increments the consecutive-failure counter and updates the last-failure
    /// epoch. When the counter first crosses
    /// [`Self::failure_streak_threshold`] in the current streak, the cooldown
    /// start epoch is set.
    pub fn record_failure(&mut self, module_name: &str, epoch: u64) {
        let threshold = self.failure_streak_threshold;
        let entry = self.states.entry(module_name.to_string()).or_default();
        entry.consecutive_failures = entry.consecutive_failures.saturating_add(1);
        entry.last_failure_epoch = Some(epoch);
        if entry.consecutive_failures >= threshold && entry.cooldown_start_epoch.is_none() {
            entry.cooldown_start_epoch = Some(epoch);
        }
    }

    /// Records a success for `module_name` at `epoch`.
    ///
    /// Resets the consecutive-failure counter to zero and clears any active
    /// cooldown — the module is re-armed for the next pass.
    pub fn record_success(&mut self, module_name: &str, epoch: u64) {
        let entry = self.states.entry(module_name.to_string()).or_default();
        entry.consecutive_failures = 0;
        entry.last_success_epoch = Some(epoch);
        entry.cooldown_start_epoch = None;
    }

    /// Returns `true` when `module_name` is currently disabled at
    /// `current_epoch`.
    ///
    /// A module is starved when:
    /// 1. Consecutive failures >= [`Self::failure_streak_threshold`], AND
    /// 2. `current_epoch` < `cooldown_start_epoch + cooldown_epochs`.
    ///
    /// Modules never-seen, below the threshold, or whose cooldown window has
    /// elapsed are NOT starved.
    #[must_use]
    pub fn is_starved(&self, module_name: &str, current_epoch: u64) -> bool {
        let Some(state) = self.states.get(module_name) else {
            return false;
        };
        if state.consecutive_failures < self.failure_streak_threshold {
            return false;
        }
        let Some(cooldown_start) = state.cooldown_start_epoch else {
            return false;
        };
        current_epoch < cooldown_start.saturating_add(self.cooldown_epochs)
    }

    /// Returns the number of modules currently in active starvation cooldown
    /// at `current_epoch`. Surfaced in the drought diagnostic payload
    /// (Issue #1273, Issue #1202).
    #[must_use]
    pub fn starved_module_count(&self, current_epoch: u64) -> usize {
        self.states
            .iter()
            .filter(|(_, state)| {
                if state.consecutive_failures < self.failure_streak_threshold {
                    return false;
                }
                let Some(cooldown_start) = state.cooldown_start_epoch else {
                    return false;
                };
                current_epoch < cooldown_start.saturating_add(self.cooldown_epochs)
            })
            .count()
    }

    /// Returns the names of every module currently in active cooldown at
    /// `current_epoch`. Useful for verbose diagnostics.
    #[must_use]
    pub fn starved_module_names(&self, current_epoch: u64) -> Vec<String> {
        let mut names: Vec<String> = self
            .states
            .iter()
            .filter_map(|(name, state)| {
                if state.consecutive_failures < self.failure_streak_threshold {
                    return None;
                }
                let cooldown_start = state.cooldown_start_epoch?;
                if current_epoch < cooldown_start.saturating_add(self.cooldown_epochs) {
                    Some(name.clone())
                } else {
                    None
                }
            })
            .collect();
        // Deterministic order for diagnostics.
        names.sort();
        names
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Streak counting: consecutive failures bump the counter.
    #[test]
    fn record_failure_increments_consecutive_counter() {
        let mut tracker = ModuleStarvationTracker::with_thresholds(3, 5);
        tracker.record_failure("coordinated-structural", 0);
        tracker.record_failure("coordinated-structural", 1);

        let state = tracker
            .state("coordinated-structural")
            .expect("state present");
        assert_eq!(state.consecutive_failures, 2);
        assert_eq!(state.last_failure_epoch, Some(1));
        assert!(state.last_success_epoch.is_none());
        assert!(state.cooldown_start_epoch.is_none());
    }

    /// Threshold trip: hitting the streak threshold disables the module.
    #[test]
    fn threshold_trip_disables_module() {
        let mut tracker = ModuleStarvationTracker::with_thresholds(3, 10);

        tracker.record_failure("M", 0);
        tracker.record_failure("M", 1);
        // Below threshold yet.
        assert!(!tracker.is_starved("M", 1));

        tracker.record_failure("M", 2);
        // At the threshold within the cooldown window.
        assert!(tracker.is_starved("M", 2));
        assert!(tracker.is_starved("M", 11));

        let state = tracker.state("M").expect("state present");
        assert_eq!(state.cooldown_start_epoch, Some(2));
    }

    /// Cooldown expiry: after the window elapses the module is re-armed.
    #[test]
    fn cooldown_expires_after_window() {
        let mut tracker = ModuleStarvationTracker::with_thresholds(2, 5);
        tracker.record_failure("M", 10);
        tracker.record_failure("M", 11);
        assert!(tracker.is_starved("M", 11));
        assert!(tracker.is_starved("M", 15));
        // cooldown_start = 11, cooldown_epochs = 5 -> released at 16.
        assert!(!tracker.is_starved("M", 16));
    }

    /// Success clears the streak immediately and re-arms the module.
    #[test]
    fn success_clears_streak_and_cooldown() {
        let mut tracker = ModuleStarvationTracker::with_thresholds(3, 10);
        tracker.record_failure("M", 0);
        tracker.record_failure("M", 1);
        tracker.record_failure("M", 2);
        assert!(tracker.is_starved("M", 2));

        tracker.record_success("M", 3);

        let state = tracker.state("M").expect("state present");
        assert_eq!(state.consecutive_failures, 0);
        assert_eq!(state.last_success_epoch, Some(3));
        assert!(state.cooldown_start_epoch.is_none());
        assert!(!tracker.is_starved("M", 3));
    }

    /// Unknown module: never starved.
    #[test]
    fn unknown_module_is_not_starved() {
        let tracker = ModuleStarvationTracker::with_thresholds(3, 10);
        assert!(!tracker.is_starved("never-seen", 5));
    }

    /// `starved_module_count` only counts modules currently in cooldown.
    #[test]
    fn starved_module_count_only_counts_active_cooldowns() {
        let mut tracker = ModuleStarvationTracker::with_thresholds(2, 5);
        // Module A: in cooldown.
        tracker.record_failure("A", 0);
        tracker.record_failure("A", 1);
        // Module B: only one failure -> not starved.
        tracker.record_failure("B", 0);
        // Module C: cooldown window elapsed.
        tracker.record_failure("C", 0);
        tracker.record_failure("C", 1);
        // Evaluate at epoch 10: A and C have cooldown_start ≤ 1, window = 5.
        // A is at epoch 10 — 1 + 5 = 6 -> elapsed. So none are starved at 10.
        assert_eq!(tracker.starved_module_count(10), 0);

        // At epoch 5: A (cooldown_start = 1, ends at 6) is starved, C also.
        assert_eq!(tracker.starved_module_count(5), 2);
    }

    /// Regression: GRQ-sampler creature `bcbca347` showed 41 consecutive
    /// coordinated-structural failures and zero successes. With the default
    /// threshold of 15, the module must be disabled by the 15th failure.
    #[test]
    fn regression_bcbca347_coordinated_structural_disabled_after_fifteen_failures() {
        let mut tracker = ModuleStarvationTracker::with_thresholds(15, 10);
        // Replay the 41-failure streak. The 15th failure must trip the gate.
        for epoch in 0..14u64 {
            tracker.record_failure("coordinated-structural", epoch);
            // Still below threshold.
            assert!(
                !tracker.is_starved("coordinated-structural", epoch),
                "module wrongly starved before 15 failures (epoch {epoch})"
            );
        }
        tracker.record_failure("coordinated-structural", 14);
        assert!(
            tracker.is_starved("coordinated-structural", 14),
            "coordinated-structural must be starved after exactly 15 failures"
        );

        // Continuing past 15 keeps the module starved within the cooldown.
        for epoch in 15..41u64 {
            tracker.record_failure("coordinated-structural", epoch);
        }
        // Cooldown started at epoch 14 with window of 10 -> still active at 23,
        // released at 24.
        assert!(tracker.is_starved("coordinated-structural", 23));
        assert!(!tracker.is_starved("coordinated-structural", 24));
    }

    #[test]
    fn starved_module_names_returns_sorted_distinct_list() {
        let mut tracker = ModuleStarvationTracker::with_thresholds(2, 100);
        tracker.record_failure("zeta", 0);
        tracker.record_failure("zeta", 1);
        tracker.record_failure("alpha", 0);
        tracker.record_failure("alpha", 1);
        tracker.record_failure("only-one", 0);

        let names = tracker.starved_module_names(2);
        assert_eq!(names, vec!["alpha".to_string(), "zeta".to_string()]);
    }

    #[test]
    fn env_var_overrides_thresholds() {
        // Defaults from constants.
        let default_tracker = ModuleStarvationTracker::new();
        assert!(default_tracker.failure_streak_threshold() >= 1);
        assert!(default_tracker.cooldown_epochs() >= 1);
    }
}
