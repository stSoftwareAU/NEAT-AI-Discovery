//! Issue #1422 — drought escape hatch armed by default + effective
//! drought-mitigation config snapshot.
//!
//! Verifies that:
//! - With no env overrides, the operator escape hatch is armed
//!   (`drought_reset_after_epochs()` returns the intended default).
//! - An explicit `0` deliberately disables the lever.
//! - A positive override is honoured; an unparsable value falls back to the
//!   armed default.
//! - `DroughtMitigationConfig::from_env()` snapshots every lever, reflecting
//!   both defaults and env overrides.

use neat_ai_discovery::analysis::constants::{
    MODULE_STARVATION_FAILURE_STREAK, STALENESS_CONSERVATIVE_DIVISOR,
    STALENESS_EXTENDED_DROUGHT_DIVISOR, TARGET_COOLDOWN_CONSECUTIVE_FAILURES,
    TARGET_COOLDOWN_EPOCHS,
};
use neat_ai_discovery::analysis::discovery_mode::{
    DEFAULT_CONSERVATIVE_GAIN_MULTIPLIER, DEFAULT_CONSERVATIVE_MODE_MAX_EPOCHS,
    DEFAULT_LOW_SUCCESS_RATE_THRESHOLD,
};
use neat_ai_discovery::config::{
    DEFAULT_DROUGHT_LOG_THRESHOLD, DEFAULT_DROUGHT_RESET_AFTER_EPOCHS, DroughtMitigationConfig,
    drought_reset_after_epochs,
};
use serial_test::serial;

/// RAII guard that restores an env var to its prior value on drop, so tests
/// that mutate process-global env state do not leak into one another.
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

const RESET_KEY: &str = "NEAT_AI_DISCOVERY_DROUGHT_RESET_AFTER_EPOCHS";

// =============================================================================
// Escape hatch armed by default (acceptance criterion)
// =============================================================================

#[test]
#[serial]
fn escape_hatch_armed_by_default_when_unset() {
    let _g = EnvGuard::unset(RESET_KEY);
    assert_eq!(
        drought_reset_after_epochs(),
        Some(DEFAULT_DROUGHT_RESET_AFTER_EPOCHS),
        "with no env override the escape hatch must be armed at the default threshold"
    );
}

#[test]
#[serial]
fn explicit_zero_disables_escape_hatch() {
    let _g = EnvGuard::set(RESET_KEY, "0");
    assert_eq!(
        drought_reset_after_epochs(),
        None,
        "an explicit 0 is the deliberate opt-out"
    );
}

#[test]
#[serial]
fn positive_override_is_honoured() {
    let _g = EnvGuard::set(RESET_KEY, "7");
    assert_eq!(drought_reset_after_epochs(), Some(7));
}

#[test]
#[serial]
fn unparsable_value_falls_back_to_armed_default() {
    let _g = EnvGuard::set(RESET_KEY, "not-a-number");
    assert_eq!(
        drought_reset_after_epochs(),
        Some(DEFAULT_DROUGHT_RESET_AFTER_EPOCHS),
        "a garbage value must not silently disable the escape hatch"
    );
}

#[test]
#[serial]
fn whitespace_padded_override_is_parsed() {
    let _g = EnvGuard::set(RESET_KEY, "  12  ");
    assert_eq!(drought_reset_after_epochs(), Some(12));
}

// =============================================================================
// Effective-config snapshot
// =============================================================================

#[test]
#[serial]
fn from_env_reports_compiled_defaults_when_unset() {
    // Unset every drought-mitigation lever so the snapshot reflects defaults.
    let _guards = [
        EnvGuard::unset(RESET_KEY),
        EnvGuard::unset("NEAT_AI_DISCOVERY_DROUGHT_LOG_THRESHOLD"),
        EnvGuard::unset("NEAT_AI_DISCOVERY_LOW_SUCCESS_RATE_THRESHOLD"),
        EnvGuard::unset("NEAT_AI_DISCOVERY_CONSERVATIVE_MODE_MAX_EPOCHS"),
        EnvGuard::unset("NEAT_AI_DISCOVERY_CONSERVATIVE_GAIN_MULTIPLIER"),
        EnvGuard::unset("NEAT_AI_DISCOVERY_TARGET_COOLDOWN_FAILURES"),
        EnvGuard::unset("NEAT_AI_DISCOVERY_TARGET_COOLDOWN_EPOCHS"),
        EnvGuard::unset("NEAT_AI_DISCOVERY_STALENESS_CONSERVATIVE_DIVISOR"),
        EnvGuard::unset("NEAT_AI_DISCOVERY_STALENESS_EXTENDED_DROUGHT_DIVISOR"),
        EnvGuard::unset("NEAT_AI_DISCOVERY_MODULE_STARVATION_FAILURE_STREAK"),
    ];

    let cfg = DroughtMitigationConfig::from_env();

    assert_eq!(
        cfg.drought_reset_after_epochs,
        Some(DEFAULT_DROUGHT_RESET_AFTER_EPOCHS)
    );
    assert_eq!(cfg.drought_log_threshold, DEFAULT_DROUGHT_LOG_THRESHOLD);
    assert_eq!(
        cfg.low_success_rate_threshold,
        DEFAULT_LOW_SUCCESS_RATE_THRESHOLD
    );
    assert_eq!(
        cfg.conservative_mode_max_epochs,
        DEFAULT_CONSERVATIVE_MODE_MAX_EPOCHS
    );
    assert_eq!(
        cfg.conservative_gain_multiplier,
        DEFAULT_CONSERVATIVE_GAIN_MULTIPLIER
    );
    assert_eq!(
        cfg.target_cooldown_failures,
        TARGET_COOLDOWN_CONSECUTIVE_FAILURES
    );
    assert_eq!(cfg.target_cooldown_epochs, TARGET_COOLDOWN_EPOCHS);
    assert_eq!(
        cfg.staleness_conservative_divisor,
        STALENESS_CONSERVATIVE_DIVISOR
    );
    assert_eq!(
        cfg.staleness_extended_drought_divisor,
        STALENESS_EXTENDED_DROUGHT_DIVISOR
    );
    assert_eq!(
        cfg.module_starvation_failure_streak,
        MODULE_STARVATION_FAILURE_STREAK
    );

    // Default lever renders as its numeric threshold, not "disabled".
    assert_eq!(
        cfg.drought_reset_display(),
        DEFAULT_DROUGHT_RESET_AFTER_EPOCHS.to_string()
    );
}

#[test]
#[serial]
fn from_env_reflects_overrides() {
    let _guards = [
        EnvGuard::set(RESET_KEY, "0"),
        EnvGuard::set("NEAT_AI_DISCOVERY_DROUGHT_LOG_THRESHOLD", "9"),
        EnvGuard::set("NEAT_AI_DISCOVERY_TARGET_COOLDOWN_FAILURES", "4"),
        EnvGuard::set("NEAT_AI_DISCOVERY_TARGET_COOLDOWN_EPOCHS", "25"),
        EnvGuard::set("NEAT_AI_DISCOVERY_MODULE_STARVATION_FAILURE_STREAK", "30"),
    ];

    let cfg = DroughtMitigationConfig::from_env();

    assert_eq!(cfg.drought_reset_after_epochs, None);
    assert_eq!(cfg.drought_log_threshold, 9);
    assert_eq!(cfg.target_cooldown_failures, 4);
    assert_eq!(cfg.target_cooldown_epochs, 25);
    assert_eq!(cfg.module_starvation_failure_streak, 30);

    // Disabled lever renders as "disabled" for the structured log line.
    assert_eq!(cfg.drought_reset_display(), "disabled");
}
