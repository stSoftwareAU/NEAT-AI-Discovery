//! Issue #1408: Reserve a guaranteed minimum analysis budget so focus selection
//! and parquet loading cannot starve synapse/neuron analysis.
//!
//! These tests cover the two `NEAT_AI_DISCOVERY_ANALYSIS_RESERVE_*` config
//! accessors that tune the reserve. The reserve arithmetic itself (effective
//! reserve, loading-deadline curtailment, fail-fast shortfall) is exercised by
//! the in-crate unit tests in `src/analysis/utils/deadline_tests.rs`.

use neat_ai_discovery::config::{
    ANALYSIS_RESERVE_FRACTION_MAX, ANALYSIS_RESERVE_MAX_MS, DEFAULT_ANALYSIS_RESERVE_FRACTION,
    DEFAULT_ANALYSIS_RESERVE_MS, analysis_reserve_fraction, analysis_reserve_ms,
};
use serial_test::serial;

const RESERVE_MS_ENV: &str = "NEAT_AI_DISCOVERY_ANALYSIS_RESERVE_MS";
const RESERVE_FRACTION_ENV: &str = "NEAT_AI_DISCOVERY_ANALYSIS_RESERVE_FRACTION";

/// RAII guard that sets/restores a single env var for a serialised test.
struct EnvVarGuard {
    name: &'static str,
    previous: Option<String>,
}

impl EnvVarGuard {
    fn set(name: &'static str, value: &str) -> Self {
        let previous = std::env::var(name).ok();
        // SAFETY: Serialised via #[serial] — no concurrent env access.
        unsafe { std::env::set_var(name, value) };
        Self { name, previous }
    }

    fn unset(name: &'static str) -> Self {
        let previous = std::env::var(name).ok();
        // SAFETY: Serialised via #[serial] — no concurrent env access.
        unsafe { std::env::remove_var(name) };
        Self { name, previous }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        match &self.previous {
            // SAFETY: Serialised via #[serial] — no concurrent env access.
            Some(v) => unsafe { std::env::set_var(self.name, v) },
            // SAFETY: Serialised via #[serial] — no concurrent env access.
            None => unsafe { std::env::remove_var(self.name) },
        }
    }
}

#[test]
#[serial]
fn unset_reserve_ms_defaults_to_sixty_seconds() {
    let _guard = EnvVarGuard::unset(RESERVE_MS_ENV);
    assert_eq!(analysis_reserve_ms(), DEFAULT_ANALYSIS_RESERVE_MS);
    assert_eq!(DEFAULT_ANALYSIS_RESERVE_MS, 60_000);
}

#[test]
#[serial]
fn reserve_ms_zero_disables_reserve() {
    let _guard = EnvVarGuard::set(RESERVE_MS_ENV, "0");
    assert_eq!(analysis_reserve_ms(), 0);
}

#[test]
#[serial]
fn reserve_ms_custom_value_is_honoured() {
    let _guard = EnvVarGuard::set(RESERVE_MS_ENV, "90000");
    assert_eq!(analysis_reserve_ms(), 90_000);
}

#[test]
#[serial]
fn reserve_ms_clamped_to_maximum() {
    let _guard = EnvVarGuard::set(RESERVE_MS_ENV, "9999999999");
    assert_eq!(analysis_reserve_ms(), ANALYSIS_RESERVE_MAX_MS);
}

#[test]
#[serial]
fn reserve_ms_invalid_falls_back_to_default() {
    let _guard = EnvVarGuard::set(RESERVE_MS_ENV, "not-a-number");
    assert_eq!(analysis_reserve_ms(), DEFAULT_ANALYSIS_RESERVE_MS);
}

#[test]
#[serial]
fn unset_reserve_fraction_defaults_to_half() {
    let _guard = EnvVarGuard::unset(RESERVE_FRACTION_ENV);
    assert!((analysis_reserve_fraction() - DEFAULT_ANALYSIS_RESERVE_FRACTION).abs() < f64::EPSILON);
    assert!((DEFAULT_ANALYSIS_RESERVE_FRACTION - 0.5).abs() < f64::EPSILON);
}

#[test]
#[serial]
fn reserve_fraction_custom_value_is_honoured() {
    let _guard = EnvVarGuard::set(RESERVE_FRACTION_ENV, "0.25");
    assert!((analysis_reserve_fraction() - 0.25).abs() < f64::EPSILON);
}

#[test]
#[serial]
fn reserve_fraction_clamped_to_maximum() {
    let _guard = EnvVarGuard::set(RESERVE_FRACTION_ENV, "0.99");
    assert!((analysis_reserve_fraction() - ANALYSIS_RESERVE_FRACTION_MAX).abs() < f64::EPSILON);
}

#[test]
#[serial]
fn reserve_fraction_invalid_falls_back_to_default() {
    let _guard = EnvVarGuard::set(RESERVE_FRACTION_ENV, "abc");
    assert!((analysis_reserve_fraction() - DEFAULT_ANALYSIS_RESERVE_FRACTION).abs() < f64::EPSILON);
}

#[test]
#[serial]
fn reserve_fraction_non_positive_falls_back_to_default() {
    let _guard = EnvVarGuard::set(RESERVE_FRACTION_ENV, "0.0");
    assert!((analysis_reserve_fraction() - DEFAULT_ANALYSIS_RESERVE_FRACTION).abs() < f64::EPSILON);
}
