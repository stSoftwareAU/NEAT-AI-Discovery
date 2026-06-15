//! Issue #1375: Wall-clock budget on focus ranking with graceful fallback.
//!
//! Focus ranking previously had no wall-clock bound — in the #1373 incident it
//! ran for 1h 11m and blew the whole 3h discovery budget. These tests cover the
//! `NEAT_AI_DISCOVERY_FOCUS_RANKING_BUDGET_MS` accessor that configures the new
//! safety net. The abort behaviour itself is exercised by the in-crate unit
//! tests (`src/focus/tests.rs`) using an injected slow record provider.

use neat_ai_discovery::config::{
    DEFAULT_FOCUS_RANKING_BUDGET_MS, FOCUS_RANKING_BUDGET_MAX_MS, FOCUS_RANKING_BUDGET_MIN_MS,
    focus_ranking_budget_ms,
};
use serial_test::serial;

const BUDGET_ENV: &str = "NEAT_AI_DISCOVERY_FOCUS_RANKING_BUDGET_MS";

/// RAII guard that sets/restores `BUDGET_ENV` for a single serialised test.
struct EnvVarGuard {
    previous: Option<String>,
}

impl EnvVarGuard {
    fn set(value: &str) -> Self {
        let previous = std::env::var(BUDGET_ENV).ok();
        // SAFETY: Serialised via #[serial] — no concurrent env access.
        unsafe { std::env::set_var(BUDGET_ENV, value) };
        Self { previous }
    }

    fn unset() -> Self {
        let previous = std::env::var(BUDGET_ENV).ok();
        // SAFETY: Serialised via #[serial] — no concurrent env access.
        unsafe { std::env::remove_var(BUDGET_ENV) };
        Self { previous }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        match &self.previous {
            // SAFETY: Serialised via #[serial] — no concurrent env access.
            Some(v) => unsafe { std::env::set_var(BUDGET_ENV, v) },
            // SAFETY: Serialised via #[serial] — no concurrent env access.
            None => unsafe { std::env::remove_var(BUDGET_ENV) },
        }
    }
}

#[test]
#[serial]
fn unset_budget_defaults_to_a_few_minutes() {
    let _guard = EnvVarGuard::unset();
    assert_eq!(
        focus_ranking_budget_ms(),
        Some(DEFAULT_FOCUS_RANKING_BUDGET_MS)
    );
    // Sanity: the default is a couple of minutes, not unbounded.
    assert_eq!(DEFAULT_FOCUS_RANKING_BUDGET_MS, 120_000);
}

#[test]
#[serial]
fn explicit_budget_is_honoured() {
    let _guard = EnvVarGuard::set("5000");
    assert_eq!(focus_ranking_budget_ms(), Some(5_000));
}

#[test]
#[serial]
fn zero_budget_disables_the_bound() {
    let _guard = EnvVarGuard::set("0");
    assert_eq!(
        focus_ranking_budget_ms(),
        None,
        "0 must opt out of the wall-clock bound (fully unbounded)"
    );
}

#[test]
#[serial]
fn below_min_budget_clamps_up_to_min() {
    // Issue #1385: a misconfigured tiny value must not abort every run after a
    // few milliseconds — it is clamped up to the supported minimum.
    let _guard = EnvVarGuard::set("5");
    assert_eq!(focus_ranking_budget_ms(), Some(FOCUS_RANKING_BUDGET_MIN_MS));
    assert_eq!(FOCUS_RANKING_BUDGET_MIN_MS, 1_000);
}

#[test]
#[serial]
fn above_max_budget_clamps_down_to_max() {
    // Issue #1385: a huge value must not restore the unbounded behaviour
    // Issue #1375 prevented — it is clamped down to the supported maximum.
    let _guard = EnvVarGuard::set("999999999");
    assert_eq!(focus_ranking_budget_ms(), Some(FOCUS_RANKING_BUDGET_MAX_MS));
    assert_eq!(FOCUS_RANKING_BUDGET_MAX_MS, 3_600_000);
}

#[test]
#[serial]
fn min_boundary_is_used_verbatim() {
    let _guard = EnvVarGuard::set("1000");
    assert_eq!(focus_ranking_budget_ms(), Some(FOCUS_RANKING_BUDGET_MIN_MS));
}

#[test]
#[serial]
fn max_boundary_is_used_verbatim() {
    let _guard = EnvVarGuard::set("3600000");
    assert_eq!(focus_ranking_budget_ms(), Some(FOCUS_RANKING_BUDGET_MAX_MS));
}

#[test]
#[serial]
fn invalid_budget_falls_back_to_default() {
    let _guard = EnvVarGuard::set("not-a-number");
    assert_eq!(
        focus_ranking_budget_ms(),
        Some(DEFAULT_FOCUS_RANKING_BUDGET_MS)
    );
}

#[test]
#[serial]
fn empty_budget_falls_back_to_default() {
    let _guard = EnvVarGuard::set("   ");
    assert_eq!(
        focus_ranking_budget_ms(),
        Some(DEFAULT_FOCUS_RANKING_BUDGET_MS)
    );
}
