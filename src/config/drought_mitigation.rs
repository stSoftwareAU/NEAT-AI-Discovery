//! Startup logging of the effective drought-mitigation configuration
//! (Issue #1422).
//!
//! The adaptive-mitigation stack (conservative mode #1132, adaptive staleness
//! #1203, adaptive cooldown #1204, target cooldown #1130, drought reset #1205)
//! is tuned by a spread of environment variables.
//! Their *effective* values were previously invisible at runtime, so a drought
//! could not be diagnosed without reading the source. This module snapshots
//! every lever and emits a single structured log line at startup so a drought
//! is diagnosable from one log entry.

/// Snapshot of every effective drought-mitigation lever (Issue #1422).
///
/// Each field is the *resolved* value — env-var override applied, otherwise the
/// compiled default. `from_env` reads the environment once so the snapshot is a
/// faithful picture of what the running pipeline will use.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DroughtMitigationConfig {
    /// Consecutive empty passes before the one-shot escape hatch fires, or
    /// `None` when deliberately disabled (Issue #1205, #1422).
    pub drought_reset_after_epochs: Option<u32>,
    /// Consecutive trailing failures at which the drought diagnostic warns
    /// (Issue #1202).
    pub drought_log_threshold: u32,
    /// Rolling success-rate threshold below which conservative mode engages
    /// (Issue #1132).
    pub low_success_rate_threshold: f32,
    /// Max consecutive failed passes before conservative mode is abandoned
    /// (Issue #1132).
    pub conservative_mode_max_epochs: u32,
    /// Multiplier applied to the coordinated minimum expected gain in
    /// conservative mode (Issue #1132).
    pub conservative_gain_multiplier: f32,
    /// Consecutive target-neuron failures before cooldown (Issue #1130).
    pub target_cooldown_failures: u32,
    /// Cooldown duration in epochs for skipped targets (Issue #1130).
    pub target_cooldown_epochs: u64,
    /// Divisor applied to the staleness window in conservative mode
    /// (Issue #1203).
    pub staleness_conservative_divisor: u64,
    /// Divisor applied to the staleness window during an extended drought
    /// (Issue #1203).
    pub staleness_extended_drought_divisor: u64,
    /// Epochs-since-last-acceptance at which the creature-level drought alarm
    /// fires, or `None` when deliberately disabled (Issue #1424).
    pub drought_alarm_epochs: Option<u32>,
    /// Multiplier applied to single-op remove-neuron candidates during a
    /// search-exhaustion drought; `1.0` disables the deprioritisation
    /// (Issue #1448).
    pub remove_neuron_drought_factor: f32,
}

impl DroughtMitigationConfig {
    /// Resolve every lever from the environment (override-or-default).
    #[must_use]
    pub fn from_env() -> Self {
        use crate::analysis::constants::{
            TARGET_COOLDOWN_CONSECUTIVE_FAILURES, TARGET_COOLDOWN_EPOCHS,
            staleness_conservative_divisor, staleness_extended_drought_divisor,
        };

        Self {
            drought_reset_after_epochs: super::drought_reset_after_epochs(),
            drought_log_threshold: super::drought_log_threshold(),
            low_success_rate_threshold: super::low_success_rate_threshold(),
            conservative_mode_max_epochs: super::conservative_mode_max_epochs(),
            conservative_gain_multiplier: super::conservative_gain_multiplier(),
            target_cooldown_failures: super::target_cooldown_consecutive_failures_env()
                .unwrap_or(TARGET_COOLDOWN_CONSECUTIVE_FAILURES),
            target_cooldown_epochs: super::target_cooldown_epochs_env()
                .unwrap_or(TARGET_COOLDOWN_EPOCHS),
            staleness_conservative_divisor: staleness_conservative_divisor(),
            staleness_extended_drought_divisor: staleness_extended_drought_divisor(),
            drought_alarm_epochs: super::drought_alarm_epochs(),
            remove_neuron_drought_factor: super::remove_neuron_drought_factor(),
        }
    }

    /// Render the creature-level drought-alarm lever for logging: the
    /// threshold, or `"disabled"` when the operator has opted out.
    #[must_use]
    pub fn drought_alarm_display(&self) -> String {
        self.drought_alarm_epochs
            .map_or_else(|| "disabled".to_string(), |n| n.to_string())
    }

    /// Render the drought-reset lever for logging: the threshold, or
    /// `"disabled"` when the operator has opted out.
    #[must_use]
    pub fn drought_reset_display(&self) -> String {
        self.drought_reset_after_epochs
            .map_or_else(|| "disabled".to_string(), |n| n.to_string())
    }
}

/// Emit a single structured `info` line enumerating every effective
/// drought-mitigation lever (Issue #1422).
///
/// Called once at library startup so an in-progress drought is diagnosable from
/// one greppable log entry without reading source.
pub fn log_effective_drought_mitigation_config() {
    let cfg = DroughtMitigationConfig::from_env();
    tracing::info!(
        drought_reset_after_epochs = cfg.drought_reset_display().as_str(),
        drought_log_threshold = cfg.drought_log_threshold,
        low_success_rate_threshold = cfg.low_success_rate_threshold,
        conservative_mode_max_epochs = cfg.conservative_mode_max_epochs,
        conservative_gain_multiplier = cfg.conservative_gain_multiplier,
        target_cooldown_failures = cfg.target_cooldown_failures,
        target_cooldown_epochs = cfg.target_cooldown_epochs,
        staleness_conservative_divisor = cfg.staleness_conservative_divisor,
        staleness_extended_drought_divisor = cfg.staleness_extended_drought_divisor,
        drought_alarm_epochs = cfg.drought_alarm_display().as_str(),
        remove_neuron_drought_factor = cfg.remove_neuron_drought_factor,
        "Issue #1422: effective drought-mitigation config"
    );
}
