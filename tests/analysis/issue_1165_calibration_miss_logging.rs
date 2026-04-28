//! Integration tests for Issue #1165: structured prediction-vs-actual
//! calibration mismatch logging and counters.
//!
//! These tests cover the public surface of:
//!
//! * [`neat_ai_discovery::config::calibration_miss_threshold`] — the env-var
//!   gate that controls the warn threshold.
//! * The `calibrationMissCount` field in the analysis JSON output (asserted
//!   indirectly via the public summary type that backs it — see the unit
//!   tests in `src/analysis/diagnostics/mcmc_diagnostics.rs` for the
//!   tracker-level coverage).

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use neat_ai_discovery::config::calibration_miss_threshold;
use serial_test::serial;

/// Acceptance: the threshold defaults to 10.0 when the env-var is unset, and
/// is overridable to a finite value greater than 1.0. Invalid values fall
/// back to the default so a malformed env-var cannot silence the channel.
#[test]
#[serial]
fn calibration_miss_threshold_env_var_overrides_default() {
    // SAFETY: This test is `#[serial]` so no other thread mutates env vars.
    unsafe {
        std::env::remove_var("NEAT_AI_DISCOVERY_CALIBRATION_MISS_THRESHOLD");
    }
    assert!((calibration_miss_threshold() - 10.0).abs() < 1e-6);

    // Override with a valid value.
    unsafe {
        std::env::set_var("NEAT_AI_DISCOVERY_CALIBRATION_MISS_THRESHOLD", "25.5");
    }
    assert!((calibration_miss_threshold() - 25.5).abs() < 1e-3);

    // Malformed -> default.
    unsafe {
        std::env::set_var(
            "NEAT_AI_DISCOVERY_CALIBRATION_MISS_THRESHOLD",
            "not-a-number",
        );
    }
    assert!((calibration_miss_threshold() - 10.0).abs() < 1e-6);

    // <=1.0 is rejected (would silence the check).
    unsafe {
        std::env::set_var("NEAT_AI_DISCOVERY_CALIBRATION_MISS_THRESHOLD", "0.5");
    }
    assert!((calibration_miss_threshold() - 10.0).abs() < 1e-6);

    unsafe {
        std::env::remove_var("NEAT_AI_DISCOVERY_CALIBRATION_MISS_THRESHOLD");
    }
}
