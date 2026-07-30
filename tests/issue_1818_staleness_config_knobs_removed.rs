//! Issue #1818 — the adaptive staleness window (#1203) lived entirely on the
//! `CandidateOutcomeCache` deleted by #1792, but its configuration surface
//! survived and controlled nothing.
//!
//! Decision (A): delete. #1792 established that no writer for per-candidate
//! history exists — the caller-supplied `failureCache` carries failures only and
//! no source UUID — so there is no surface to re-point the divisors at.
//!
//! These tests are the regression guard that the orphaned knobs do not return:
//!
//! 1. Setting `NEAT_AI_DISCOVERY_STALENESS_*` changes nothing an operator can
//!    observe in the effective drought-mitigation config.
//! 2. `DroughtMitigationConfig` carries no staleness fields (exhaustive
//!    destructuring fails to compile if one is re-added, which also stops the
//!    startup config log from advertising them).
//! 3. No `NEAT_AI_DISCOVERY_STALENESS_*` variable is documented anywhere.

use neat_ai_discovery::config::DroughtMitigationConfig;
use serial_test::serial;

const CONSERVATIVE_KEY: &str = "NEAT_AI_DISCOVERY_STALENESS_CONSERVATIVE_DIVISOR";
const EXTENDED_KEY: &str = "NEAT_AI_DISCOVERY_STALENESS_EXTENDED_DROUGHT_DIVISOR";

/// RAII guard restoring an env var to its prior value on drop.
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

/// The operator-visible failure mode: tuning either divisor used to be echoed
/// back in the startup config log while changing no behaviour whatsoever. With
/// the knobs gone the snapshot is byte-identical either way.
#[test]
#[serial]
fn staleness_env_vars_do_not_change_the_effective_config() {
    let baseline = {
        let _g1 = EnvGuard::unset(CONSERVATIVE_KEY);
        let _g2 = EnvGuard::unset(EXTENDED_KEY);
        DroughtMitigationConfig::from_env()
    };

    let overridden = {
        let _g1 = EnvGuard::set(CONSERVATIVE_KEY, "7");
        let _g2 = EnvGuard::set(EXTENDED_KEY, "13");
        DroughtMitigationConfig::from_env()
    };

    assert_eq!(
        baseline, overridden,
        "NEAT_AI_DISCOVERY_STALENESS_* must not appear in the effective config — \
         the adaptive staleness window was deleted with the cache (Issue #1792)"
    );
}

/// Exhaustive destructuring — re-adding a staleness field to the snapshot (and
/// therefore to the startup config log line) breaks this pattern at compile time.
#[test]
#[serial]
fn config_snapshot_carries_no_staleness_fields() {
    let DroughtMitigationConfig {
        drought_reset_after_epochs: _,
        drought_log_threshold: _,
        low_success_rate_threshold: _,
        conservative_mode_max_epochs: _,
        conservative_gain_multiplier: _,
        target_cooldown_failures: _,
        target_cooldown_epochs: _,
        drought_alarm_epochs: _,
        remove_neuron_drought_factor: _,
    } = DroughtMitigationConfig::from_env();
}

/// A knob that controls nothing must not be advertised to operators.
#[test]
fn no_staleness_variable_is_documented() {
    const DOCS: &[(&str, &str)] = &[
        (
            "docs/CONFIGURATION.md",
            include_str!("../docs/CONFIGURATION.md"),
        ),
        (
            "docs/DROUGHT_PLAYBOOK.md",
            include_str!("../docs/DROUGHT_PLAYBOOK.md"),
        ),
        ("README.md", include_str!("../README.md")),
        ("AGENTS.md", include_str!("../AGENTS.md")),
    ];

    for (name, body) in DOCS {
        assert!(
            !body.contains("NEAT_AI_DISCOVERY_STALENESS_"),
            "{name} still documents a NEAT_AI_DISCOVERY_STALENESS_* knob that controls nothing"
        );
    }
}
