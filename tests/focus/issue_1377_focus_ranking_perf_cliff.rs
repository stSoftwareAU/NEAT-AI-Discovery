//! Issue #1377: Surface focus-ranking mode + Rust elapsed in the timing breakdown.
//!
//! Diagnosing #1373 required manually correlating a Rust `WARN` with the
//! TypeScript per-phase timing summary because nothing attributed the
//! 71-minute `Focus selection` figure to the lazy Rust path. These tests cover
//! the two pure, testable seams added here:
//!
//! 1. `lazy_pass_exceeds_perf_cliff` — the perf-cliff threshold decision that
//!    gates the explicit perf-cliff `WARN` emitted when a *lazy* ranking pass
//!    runs longer than the configured threshold.
//! 2. `focus_ranking_perf_cliff_ms` — the env-var accessor configuring that
//!    threshold.
//!
//! The log-string side effects (perf-cliff warning + enriched lazy-selection
//! log) are observability-only and exercised indirectly through the decision
//! helper, mirroring the #1375/#1376 testing approach.

use neat_ai_discovery::config::{DEFAULT_FOCUS_RANKING_PERF_CLIFF_MS, focus_ranking_perf_cliff_ms};
use neat_ai_discovery::focus::{FocusLoadingMode, lazy_pass_exceeds_perf_cliff};
use serial_test::serial;

const CLIFF_ENV: &str = "NEAT_AI_DISCOVERY_FOCUS_RANKING_PERF_CLIFF_MS";

/// RAII guard that sets/restores `CLIFF_ENV` for a single serialised test.
struct EnvVarGuard {
    previous: Option<String>,
}

impl EnvVarGuard {
    fn set(value: &str) -> Self {
        let previous = std::env::var(CLIFF_ENV).ok();
        // SAFETY: Serialised via #[serial] — no concurrent env access.
        unsafe { std::env::set_var(CLIFF_ENV, value) };
        Self { previous }
    }

    fn unset() -> Self {
        let previous = std::env::var(CLIFF_ENV).ok();
        // SAFETY: Serialised via #[serial] — no concurrent env access.
        unsafe { std::env::remove_var(CLIFF_ENV) };
        Self { previous }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        match &self.previous {
            // SAFETY: Serialised via #[serial] — no concurrent env access.
            Some(v) => unsafe { std::env::set_var(CLIFF_ENV, v) },
            // SAFETY: Serialised via #[serial] — no concurrent env access.
            None => unsafe { std::env::remove_var(CLIFF_ENV) },
        }
    }
}

// ---------------------------------------------------------------------------
// lazy_pass_exceeds_perf_cliff — pure threshold decision
// ---------------------------------------------------------------------------

#[test]
fn lazy_pass_over_threshold_trips_the_cliff() {
    // 70s lazy pass against a 60s threshold → perf cliff.
    assert!(lazy_pass_exceeds_perf_cliff(
        FocusLoadingMode::Lazy,
        70_000,
        60_000
    ));
}

#[test]
fn lazy_pass_at_threshold_trips_the_cliff() {
    // Boundary: elapsed == threshold counts as exceeding (>=).
    assert!(lazy_pass_exceeds_perf_cliff(
        FocusLoadingMode::Lazy,
        60_000,
        60_000
    ));
}

#[test]
fn lazy_pass_under_threshold_does_not_trip() {
    assert!(!lazy_pass_exceeds_perf_cliff(
        FocusLoadingMode::Lazy,
        59_999,
        60_000
    ));
}

#[test]
fn preload_pass_never_trips_the_cliff() {
    // Preload is the fast path — even a very slow preload is not the lazy
    // perf cliff this warning is about.
    assert!(!lazy_pass_exceeds_perf_cliff(
        FocusLoadingMode::Preload,
        u128::from(u64::MAX),
        60_000
    ));
}

#[test]
fn zero_threshold_disables_the_cliff() {
    // 0 opts out of the perf-cliff warning entirely.
    assert!(!lazy_pass_exceeds_perf_cliff(
        FocusLoadingMode::Lazy,
        u128::from(u64::MAX),
        0
    ));
}

// ---------------------------------------------------------------------------
// focus_ranking_perf_cliff_ms — env-var accessor
// ---------------------------------------------------------------------------

#[test]
#[serial]
fn unset_threshold_defaults_to_sixty_seconds() {
    let _guard = EnvVarGuard::unset();
    assert_eq!(
        focus_ranking_perf_cliff_ms(),
        DEFAULT_FOCUS_RANKING_PERF_CLIFF_MS
    );
    assert_eq!(DEFAULT_FOCUS_RANKING_PERF_CLIFF_MS, 60_000);
}

#[test]
#[serial]
fn explicit_threshold_is_honoured() {
    let _guard = EnvVarGuard::set("30000");
    assert_eq!(focus_ranking_perf_cliff_ms(), 30_000);
}

#[test]
#[serial]
fn zero_threshold_is_honoured_as_opt_out() {
    let _guard = EnvVarGuard::set("0");
    assert_eq!(
        focus_ranking_perf_cliff_ms(),
        0,
        "0 must opt out of the perf-cliff warning"
    );
}

#[test]
#[serial]
fn invalid_threshold_falls_back_to_default() {
    let _guard = EnvVarGuard::set("not-a-number");
    assert_eq!(
        focus_ranking_perf_cliff_ms(),
        DEFAULT_FOCUS_RANKING_PERF_CLIFF_MS
    );
}

#[test]
#[serial]
fn empty_threshold_falls_back_to_default() {
    let _guard = EnvVarGuard::set("   ");
    assert_eq!(
        focus_ranking_perf_cliff_ms(),
        DEFAULT_FOCUS_RANKING_PERF_CLIFF_MS
    );
}
