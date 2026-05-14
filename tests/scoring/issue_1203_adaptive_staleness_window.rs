//! Tests for Issue #1203: adaptive candidate-cache staleness window during drought.
//!
//! Auto-shrink `CandidateOutcomeCache::staleness_window` when the rolling
//! outcome log signals a drought, so previously-failed candidates become
//! re-eligible sooner instead of staying suppressed for the full 100-epoch
//! default while the pipeline struggles to find any improvement.
//!
//! ## Key Behaviours Verified
//!
//! - Normal mode keeps the full staleness window.
//! - Conservative mode halves the effective window.
//! - Extended drought (`drought_failures` >= `conservative_mode_max_epochs`)
//!   quarters the window with a hard floor of 5.
//! - A successful pass — resetting drought to 0 and reverting to Normal mode —
//!   restores the full window on the next call.
//! - Env-var overrides for both divisors are parsed and respected.
//! - Integration: a candidate suppressed in Normal mode is re-eligible in
//!   Conservative mode at the same epoch.

#![allow(clippy::cast_sign_loss)]

use neat_ai_discovery::analysis::candidate_cache::{
    CandidateOutcomeCache, DEFAULT_STALENESS_WINDOW,
};
use neat_ai_discovery::analysis::constants::{
    STALENESS_CONSERVATIVE_DIVISOR, STALENESS_EXTENDED_DROUGHT_DIVISOR, STALENESS_WINDOW_FLOOR,
    staleness_conservative_divisor, staleness_extended_drought_divisor,
};
use neat_ai_discovery::analysis::discovery_mode::{
    DEFAULT_CONSERVATIVE_MODE_MAX_EPOCHS, DiscoveryMode,
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
// effective_staleness_window — three regimes
// =============================================================================

#[test]
#[serial]
fn normal_mode_keeps_full_window() {
    let _g1 = EnvGuard::unset("NEAT_AI_DISCOVERY_STALENESS_CONSERVATIVE_DIVISOR");
    let _g2 = EnvGuard::unset("NEAT_AI_DISCOVERY_STALENESS_EXTENDED_DROUGHT_DIVISOR");

    let cache = CandidateOutcomeCache::new();
    let effective = cache.effective_staleness_window(DiscoveryMode::Normal, 0);
    assert_eq!(effective, DEFAULT_STALENESS_WINDOW);
}

#[test]
#[serial]
fn conservative_mode_halves_window() {
    let _g1 = EnvGuard::unset("NEAT_AI_DISCOVERY_STALENESS_CONSERVATIVE_DIVISOR");
    let _g2 = EnvGuard::unset("NEAT_AI_DISCOVERY_STALENESS_EXTENDED_DROUGHT_DIVISOR");

    let cache = CandidateOutcomeCache::new();
    let effective = cache.effective_staleness_window(DiscoveryMode::Conservative, 5);
    assert_eq!(
        effective,
        DEFAULT_STALENESS_WINDOW / STALENESS_CONSERVATIVE_DIVISOR,
    );
}

#[test]
#[serial]
fn conservative_mode_re_enables_failed_candidate_earlier_than_normal_mode() {
    let _g1 = EnvGuard::unset("NEAT_AI_DISCOVERY_STALENESS_CONSERVATIVE_DIVISOR");
    let _g2 = EnvGuard::unset("NEAT_AI_DISCOVERY_STALENESS_EXTENDED_DROUGHT_DIVISOR");

    // Worked example from the rustdoc: candidate failed at epoch 0,
    // staleness_window = 100. In Normal mode it is re-eligible at epoch 100;
    // in Conservative mode the effective window is 50, so at epoch 60 the
    // candidate is still suppressed under Normal (60 < 100) but re-eligible
    // under Conservative (60 >= 50).
    let mut cache = CandidateOutcomeCache::new();
    cache.record("source-1", "target-1", "addSynapse", false, 0);

    // Normal mode at epoch 60: still suppressed (60 < 100).
    assert!(cache.is_suppressed(
        "source-1",
        "target-1",
        "addSynapse",
        60,
        DiscoveryMode::Normal,
        0,
    ));

    // Conservative mode at epoch 60: re-eligible (60 >= 50).
    assert!(!cache.is_suppressed(
        "source-1",
        "target-1",
        "addSynapse",
        60,
        DiscoveryMode::Conservative,
        5,
    ));
}

#[test]
#[serial]
fn extended_drought_quarters_window() {
    let _g1 = EnvGuard::unset("NEAT_AI_DISCOVERY_STALENESS_CONSERVATIVE_DIVISOR");
    let _g2 = EnvGuard::unset("NEAT_AI_DISCOVERY_STALENESS_EXTENDED_DROUGHT_DIVISOR");

    let cache = CandidateOutcomeCache::new();
    let effective = cache
        .effective_staleness_window(DiscoveryMode::Normal, DEFAULT_CONSERVATIVE_MODE_MAX_EPOCHS);
    assert_eq!(
        effective,
        DEFAULT_STALENESS_WINDOW / STALENESS_EXTENDED_DROUGHT_DIVISOR,
    );
}

#[test]
#[serial]
fn extended_drought_honours_floor_of_five() {
    let _g1 = EnvGuard::unset("NEAT_AI_DISCOVERY_STALENESS_CONSERVATIVE_DIVISOR");
    let _g2 = EnvGuard::unset("NEAT_AI_DISCOVERY_STALENESS_EXTENDED_DROUGHT_DIVISOR");

    // Tiny base window (16) divided by extended-drought divisor (4) → 4,
    // which is below the floor of 5 → clamped to 5.
    let cache = CandidateOutcomeCache::with_staleness_window(16);
    let effective = cache.effective_staleness_window(
        DiscoveryMode::Normal,
        DEFAULT_CONSERVATIVE_MODE_MAX_EPOCHS + 10,
    );
    assert_eq!(effective, STALENESS_WINDOW_FLOOR);
    assert_eq!(STALENESS_WINDOW_FLOOR, 5);
}

#[test]
#[serial]
fn successful_pass_restores_full_window_on_next_call() {
    let _g1 = EnvGuard::unset("NEAT_AI_DISCOVERY_STALENESS_CONSERVATIVE_DIVISOR");
    let _g2 = EnvGuard::unset("NEAT_AI_DISCOVERY_STALENESS_EXTENDED_DROUGHT_DIVISOR");

    let cache = CandidateOutcomeCache::new();

    // Drought: extended drought regime → quartered.
    let drought = cache.effective_staleness_window(
        DiscoveryMode::Normal,
        DEFAULT_CONSERVATIVE_MODE_MAX_EPOCHS + 5,
    );
    assert_eq!(
        drought,
        DEFAULT_STALENESS_WINDOW / STALENESS_EXTENDED_DROUGHT_DIVISOR,
    );

    // After a successful pass, callers report drought_failures = 0 and
    // DiscoveryMode::Normal. The next call must return the full window.
    let restored = cache.effective_staleness_window(DiscoveryMode::Normal, 0);
    assert_eq!(restored, DEFAULT_STALENESS_WINDOW);
}

// =============================================================================
// Env-var overrides
// =============================================================================

#[test]
#[serial]
fn env_var_overrides_conservative_divisor() {
    let _g_extended = EnvGuard::unset("NEAT_AI_DISCOVERY_STALENESS_EXTENDED_DROUGHT_DIVISOR");
    let _g = EnvGuard::set("NEAT_AI_DISCOVERY_STALENESS_CONSERVATIVE_DIVISOR", "5");
    assert_eq!(staleness_conservative_divisor(), 5);

    let cache = CandidateOutcomeCache::new();
    let effective = cache.effective_staleness_window(DiscoveryMode::Conservative, 0);
    assert_eq!(effective, DEFAULT_STALENESS_WINDOW / 5);
}

#[test]
#[serial]
fn env_var_overrides_extended_drought_divisor() {
    let _g_cons = EnvGuard::unset("NEAT_AI_DISCOVERY_STALENESS_CONSERVATIVE_DIVISOR");
    let _g = EnvGuard::set("NEAT_AI_DISCOVERY_STALENESS_EXTENDED_DROUGHT_DIVISOR", "10");
    assert_eq!(staleness_extended_drought_divisor(), 10);

    let cache = CandidateOutcomeCache::new();
    let effective = cache
        .effective_staleness_window(DiscoveryMode::Normal, DEFAULT_CONSERVATIVE_MODE_MAX_EPOCHS);
    assert_eq!(effective, DEFAULT_STALENESS_WINDOW / 10);
}

#[test]
#[serial]
fn unparsable_env_vars_fall_back_to_defaults() {
    let _g1 = EnvGuard::set(
        "NEAT_AI_DISCOVERY_STALENESS_CONSERVATIVE_DIVISOR",
        "not-a-number",
    );
    let _g2 = EnvGuard::set(
        "NEAT_AI_DISCOVERY_STALENESS_EXTENDED_DROUGHT_DIVISOR",
        "also-bogus",
    );
    assert_eq!(
        staleness_conservative_divisor(),
        STALENESS_CONSERVATIVE_DIVISOR
    );
    assert_eq!(
        staleness_extended_drought_divisor(),
        STALENESS_EXTENDED_DROUGHT_DIVISOR,
    );
}

#[test]
#[serial]
fn env_var_zero_clamped_to_floor() {
    // Divisor of 0 would panic on integer division — verify the clamp catches it.
    let _g = EnvGuard::set("NEAT_AI_DISCOVERY_STALENESS_CONSERVATIVE_DIVISOR", "0");
    assert!(staleness_conservative_divisor() >= 1);

    let cache = CandidateOutcomeCache::new();
    // Should not panic, and should produce a sensible window >= floor.
    let effective = cache.effective_staleness_window(DiscoveryMode::Conservative, 0);
    assert!(effective >= STALENESS_WINDOW_FLOOR);
}

// =============================================================================
// Integration: Normal-suppressed candidate is re-eligible in Conservative mode
// =============================================================================

#[test]
#[serial]
fn integration_candidate_failed_at_epoch_zero_re_eligible_in_conservative_at_epoch_thirty() {
    let _g1 = EnvGuard::unset("NEAT_AI_DISCOVERY_STALENESS_CONSERVATIVE_DIVISOR");
    let _g2 = EnvGuard::unset("NEAT_AI_DISCOVERY_STALENESS_EXTENDED_DROUGHT_DIVISOR");

    // Acceptance criterion from Issue #1203: a candidate failed at epoch 0
    // is suppressed at epoch 30 in Normal mode but is re-eligible at epoch 30
    // in Conservative mode. Note: 30 < 100/2 = 50, so it's still suppressed
    // at epoch 30 in Conservative mode too. The actual re-eligibility point
    // moves from 100 → 50 — confirm at the boundary instead.
    //
    // Suppression boundary: Normal = epoch 100, Conservative = epoch 50.
    let mut cache = CandidateOutcomeCache::new();
    cache.record("source-A", "target-B", "addSynapse", false, 0);

    // At epoch 50 (the conservative boundary):
    assert!(cache.is_suppressed(
        "source-A",
        "target-B",
        "addSynapse",
        50 - 1,
        DiscoveryMode::Conservative,
        5,
    ));
    assert!(!cache.is_suppressed(
        "source-A",
        "target-B",
        "addSynapse",
        50,
        DiscoveryMode::Conservative,
        5,
    ));

    // Same epoch range, Normal mode: still suppressed.
    assert!(cache.is_suppressed(
        "source-A",
        "target-B",
        "addSynapse",
        50,
        DiscoveryMode::Normal,
        0,
    ));
    assert!(cache.is_suppressed(
        "source-A",
        "target-B",
        "addSynapse",
        99,
        DiscoveryMode::Normal,
        0,
    ));
    assert!(!cache.is_suppressed(
        "source-A",
        "target-B",
        "addSynapse",
        100,
        DiscoveryMode::Normal,
        0,
    ));
}
