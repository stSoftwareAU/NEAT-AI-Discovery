//! Tests for Issue #1204: adaptive target-cooldown relaxation during drought.
//!
//! Relax `TargetFailureTracker` cooldown thresholds when the rolling outcome
//! log signals a drought so previously-failing targets re-enter the focus
//! list sooner. Complements the candidate-cache staleness window relaxation
//! (Issue #1203).
//!
//! ## Key Behaviours Verified
//!
//! - Normal mode keeps the full cooldown window and configured trigger.
//! - Conservative mode halves the effective cooldown and raises the trigger
//!   by 1.
//! - Extended drought (`drought_failures >= conservative_mode_max_epochs`)
//!   quarters the cooldown with a floor of 2 and raises the trigger by 2.
//! - Env-var overrides for both divisors are parsed and respected.
//! - Integration: a target with 3 consecutive failures at epoch 0 is in
//!   cooldown at epoch 15 under Normal mode (30-epoch cooldown) but cleared
//!   at epoch 15 in Conservative mode.

#![allow(clippy::cast_sign_loss)]

use neat_ai_discovery::analysis::constants::{
    COOLDOWN_CONSERVATIVE_DIVISOR, COOLDOWN_EPOCHS_FLOOR, COOLDOWN_EXTENDED_DROUGHT_DIVISOR,
    cooldown_conservative_divisor, cooldown_extended_drought_divisor,
};
use neat_ai_discovery::analysis::discovery_mode::{
    DEFAULT_CONSERVATIVE_MODE_MAX_EPOCHS, DiscoveryMode,
};
use neat_ai_discovery::analysis::target_failure_tracker::{
    TargetFailureTracker, filter_cooldown_targets_adaptive,
};
use serial_test::serial;

/// Guard that restores or removes an env var on drop so a test does not leak
/// configuration into siblings that share the process.
struct EnvGuard {
    key: &'static str,
    prior: Option<String>,
}

impl EnvGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let prior = std::env::var(key).ok();
        // SAFETY: tests are marked #[serial] so no concurrent set/unset races.
        unsafe { std::env::set_var(key, value) };
        Self { key, prior }
    }

    fn unset(key: &'static str) -> Self {
        let prior = std::env::var(key).ok();
        // SAFETY: tests are marked #[serial] so no concurrent set/unset races.
        unsafe { std::env::remove_var(key) };
        Self { key, prior }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        // SAFETY: tests are marked #[serial] so no concurrent set/unset races.
        unsafe {
            match &self.prior {
                Some(v) => std::env::set_var(self.key, v),
                None => std::env::remove_var(self.key),
            }
        }
    }
}

// =============================================================================
// effective_cooldown_epochs — three regimes
// =============================================================================

#[test]
#[serial]
fn normal_mode_keeps_full_cooldown_window() {
    let _g1 = EnvGuard::unset("NEAT_AI_DISCOVERY_COOLDOWN_CONSERVATIVE_DIVISOR");
    let _g2 = EnvGuard::unset("NEAT_AI_DISCOVERY_COOLDOWN_EXTENDED_DROUGHT_DIVISOR");

    let tracker = TargetFailureTracker::with_thresholds(3, 30);
    let effective = tracker.effective_cooldown_epochs(DiscoveryMode::Normal, 0);
    assert_eq!(effective, 30);
}

#[test]
#[serial]
fn conservative_mode_halves_cooldown_window() {
    let _g1 = EnvGuard::unset("NEAT_AI_DISCOVERY_COOLDOWN_CONSERVATIVE_DIVISOR");
    let _g2 = EnvGuard::unset("NEAT_AI_DISCOVERY_COOLDOWN_EXTENDED_DROUGHT_DIVISOR");

    let tracker = TargetFailureTracker::with_thresholds(3, 30);
    let effective = tracker.effective_cooldown_epochs(DiscoveryMode::Conservative, 5);
    assert_eq!(effective, 30 / COOLDOWN_CONSERVATIVE_DIVISOR);
}

#[test]
#[serial]
fn extended_drought_quarters_cooldown_window() {
    let _g1 = EnvGuard::unset("NEAT_AI_DISCOVERY_COOLDOWN_CONSERVATIVE_DIVISOR");
    let _g2 = EnvGuard::unset("NEAT_AI_DISCOVERY_COOLDOWN_EXTENDED_DROUGHT_DIVISOR");

    let tracker = TargetFailureTracker::with_thresholds(3, 40);
    let effective = tracker
        .effective_cooldown_epochs(DiscoveryMode::Normal, DEFAULT_CONSERVATIVE_MODE_MAX_EPOCHS);
    assert_eq!(effective, 40 / COOLDOWN_EXTENDED_DROUGHT_DIVISOR);
}

#[test]
#[serial]
fn extended_drought_honours_floor_of_two() {
    let _g1 = EnvGuard::unset("NEAT_AI_DISCOVERY_COOLDOWN_CONSERVATIVE_DIVISOR");
    let _g2 = EnvGuard::unset("NEAT_AI_DISCOVERY_COOLDOWN_EXTENDED_DROUGHT_DIVISOR");

    // Tiny base cooldown (4) divided by extended-drought divisor (4) → 1,
    // which is below the floor of 2 → clamped to 2.
    let tracker = TargetFailureTracker::with_thresholds(3, 4);
    let effective = tracker.effective_cooldown_epochs(
        DiscoveryMode::Normal,
        DEFAULT_CONSERVATIVE_MODE_MAX_EPOCHS + 10,
    );
    assert_eq!(effective, COOLDOWN_EPOCHS_FLOOR);
    assert_eq!(COOLDOWN_EPOCHS_FLOOR, 2);
}

// =============================================================================
// effective_consecutive_failures — three regimes
// =============================================================================

#[test]
#[serial]
fn normal_mode_keeps_configured_failure_trigger() {
    let _g1 = EnvGuard::unset("NEAT_AI_DISCOVERY_COOLDOWN_CONSERVATIVE_DIVISOR");
    let _g2 = EnvGuard::unset("NEAT_AI_DISCOVERY_COOLDOWN_EXTENDED_DROUGHT_DIVISOR");

    let tracker = TargetFailureTracker::with_thresholds(3, 30);
    let effective = tracker.effective_consecutive_failures(DiscoveryMode::Normal, 0);
    assert_eq!(effective, 3);
}

#[test]
#[serial]
fn conservative_mode_raises_failure_trigger_by_one() {
    let _g1 = EnvGuard::unset("NEAT_AI_DISCOVERY_COOLDOWN_CONSERVATIVE_DIVISOR");
    let _g2 = EnvGuard::unset("NEAT_AI_DISCOVERY_COOLDOWN_EXTENDED_DROUGHT_DIVISOR");

    let tracker = TargetFailureTracker::with_thresholds(3, 30);
    let effective = tracker.effective_consecutive_failures(DiscoveryMode::Conservative, 5);
    assert_eq!(effective, 4);
}

#[test]
#[serial]
fn extended_drought_raises_failure_trigger_by_two() {
    let _g1 = EnvGuard::unset("NEAT_AI_DISCOVERY_COOLDOWN_CONSERVATIVE_DIVISOR");
    let _g2 = EnvGuard::unset("NEAT_AI_DISCOVERY_COOLDOWN_EXTENDED_DROUGHT_DIVISOR");

    let tracker = TargetFailureTracker::with_thresholds(3, 30);
    let effective = tracker.effective_consecutive_failures(
        DiscoveryMode::Normal,
        DEFAULT_CONSERVATIVE_MODE_MAX_EPOCHS,
    );
    assert_eq!(effective, 5);
}

#[test]
#[serial]
fn effective_failures_saturate_at_u32_max() {
    let tracker = TargetFailureTracker::with_thresholds(u32::MAX, 30);
    // Saturating-add: u32::MAX + 2 stays at u32::MAX.
    let effective = tracker.effective_consecutive_failures(
        DiscoveryMode::Normal,
        DEFAULT_CONSERVATIVE_MODE_MAX_EPOCHS + 1,
    );
    assert_eq!(effective, u32::MAX);
}

// =============================================================================
// Env-var overrides
// =============================================================================

#[test]
#[serial]
fn env_var_overrides_cooldown_conservative_divisor() {
    let _g_extended = EnvGuard::unset("NEAT_AI_DISCOVERY_COOLDOWN_EXTENDED_DROUGHT_DIVISOR");
    let _g = EnvGuard::set("NEAT_AI_DISCOVERY_COOLDOWN_CONSERVATIVE_DIVISOR", "5");
    assert_eq!(cooldown_conservative_divisor(), 5);

    let tracker = TargetFailureTracker::with_thresholds(3, 100);
    let effective = tracker.effective_cooldown_epochs(DiscoveryMode::Conservative, 0);
    assert_eq!(effective, 100 / 5);
}

#[test]
#[serial]
fn env_var_overrides_cooldown_extended_drought_divisor() {
    let _g_cons = EnvGuard::unset("NEAT_AI_DISCOVERY_COOLDOWN_CONSERVATIVE_DIVISOR");
    let _g = EnvGuard::set("NEAT_AI_DISCOVERY_COOLDOWN_EXTENDED_DROUGHT_DIVISOR", "10");
    assert_eq!(cooldown_extended_drought_divisor(), 10);

    let tracker = TargetFailureTracker::with_thresholds(3, 100);
    let effective = tracker
        .effective_cooldown_epochs(DiscoveryMode::Normal, DEFAULT_CONSERVATIVE_MODE_MAX_EPOCHS);
    assert_eq!(effective, 100 / 10);
}

#[test]
#[serial]
fn unparsable_cooldown_env_vars_fall_back_to_defaults() {
    let _g1 = EnvGuard::set(
        "NEAT_AI_DISCOVERY_COOLDOWN_CONSERVATIVE_DIVISOR",
        "not-a-number",
    );
    let _g2 = EnvGuard::set(
        "NEAT_AI_DISCOVERY_COOLDOWN_EXTENDED_DROUGHT_DIVISOR",
        "also-bogus",
    );
    assert_eq!(
        cooldown_conservative_divisor(),
        COOLDOWN_CONSERVATIVE_DIVISOR
    );
    assert_eq!(
        cooldown_extended_drought_divisor(),
        COOLDOWN_EXTENDED_DROUGHT_DIVISOR,
    );
}

#[test]
#[serial]
fn zero_cooldown_divisor_env_var_clamped_to_floor() {
    let _g = EnvGuard::set("NEAT_AI_DISCOVERY_COOLDOWN_CONSERVATIVE_DIVISOR", "0");
    assert!(cooldown_conservative_divisor() >= 1);

    let tracker = TargetFailureTracker::with_thresholds(3, 30);
    // Should not panic and should produce a sensible window >= floor.
    let effective = tracker.effective_cooldown_epochs(DiscoveryMode::Conservative, 0);
    assert!(effective >= COOLDOWN_EPOCHS_FLOOR);
}

// =============================================================================
// Integration: a previously-cooldown target is released earlier in Conservative mode
// =============================================================================

#[test]
#[serial]
fn target_in_cooldown_under_normal_is_released_in_conservative_at_same_epoch() {
    let _g1 = EnvGuard::unset("NEAT_AI_DISCOVERY_COOLDOWN_CONSERVATIVE_DIVISOR");
    let _g2 = EnvGuard::unset("NEAT_AI_DISCOVERY_COOLDOWN_EXTENDED_DROUGHT_DIVISOR");

    // Acceptance criterion: a target with 3 consecutive failures at epoch 0
    // is in cooldown at epoch 15 under Normal mode (30-epoch cooldown) but
    // cleared at epoch 15 in Conservative mode (effective cooldown = 15).
    let mut tracker = TargetFailureTracker::with_thresholds(3, 30);
    tracker.record_failure("target-T", 0);
    tracker.record_failure("target-T", 0);
    tracker.record_failure("target-T", 0);

    // Normal mode at epoch 15: still in cooldown (15 < 0 + 30).
    assert!(tracker.is_in_cooldown_adaptive("target-T", 15, DiscoveryMode::Normal, 0,));

    // Conservative mode at epoch 15: released (15 < 0 + 15 is false).
    assert!(!tracker.is_in_cooldown_adaptive("target-T", 15, DiscoveryMode::Conservative, 5,));
}

#[test]
#[serial]
fn filter_cooldown_targets_adaptive_drops_only_under_normal() {
    let _g1 = EnvGuard::unset("NEAT_AI_DISCOVERY_COOLDOWN_CONSERVATIVE_DIVISOR");
    let _g2 = EnvGuard::unset("NEAT_AI_DISCOVERY_COOLDOWN_EXTENDED_DROUGHT_DIVISOR");

    // Build a tracker with three failures at epoch 0 on target-T.
    let mut tracker = TargetFailureTracker::with_thresholds(3, 30);
    for _ in 0..3 {
        tracker.record_failure("target-T", 0);
    }

    // Normal mode at epoch 15: target is dropped.
    let mut focus = vec!["target-T".to_string(), "target-healthy".to_string()];
    let skipped =
        filter_cooldown_targets_adaptive(&mut focus, &tracker, 15, DiscoveryMode::Normal, 0);
    assert_eq!(skipped, 1);
    assert_eq!(focus, vec!["target-healthy".to_string()]);

    // Conservative mode at epoch 15: target is kept (effective cooldown = 15).
    let mut focus = vec!["target-T".to_string(), "target-healthy".to_string()];
    let skipped =
        filter_cooldown_targets_adaptive(&mut focus, &tracker, 15, DiscoveryMode::Conservative, 5);
    assert_eq!(skipped, 0);
    assert_eq!(
        focus,
        vec!["target-T".to_string(), "target-healthy".to_string()]
    );
}

#[test]
#[serial]
fn raised_trigger_under_drought_keeps_borderline_targets_out_of_cooldown() {
    let _g1 = EnvGuard::unset("NEAT_AI_DISCOVERY_COOLDOWN_CONSERVATIVE_DIVISOR");
    let _g2 = EnvGuard::unset("NEAT_AI_DISCOVERY_COOLDOWN_EXTENDED_DROUGHT_DIVISOR");

    // Configured trigger = 3 failures. Effective trigger in Conservative mode
    // is 4, so a target with exactly 3 failures is *not* in cooldown under
    // Conservative mode but *is* under Normal mode.
    let mut tracker = TargetFailureTracker::with_thresholds(3, 30);
    tracker.record_failure("target-T", 0);
    tracker.record_failure("target-T", 0);
    tracker.record_failure("target-T", 0);

    assert!(tracker.is_in_cooldown_adaptive("target-T", 5, DiscoveryMode::Normal, 0));
    assert!(!tracker.is_in_cooldown_adaptive("target-T", 5, DiscoveryMode::Conservative, 5));
}
