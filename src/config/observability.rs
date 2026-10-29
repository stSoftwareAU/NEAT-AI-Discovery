//! Observability configuration — timing, profiling, and GPU metrics.

use std::sync::OnceLock;

use super::helpers::parse_env;

/// Check if phase timing output is enabled (cached).
///
/// Presence-only: any set value enables it, `0` included. The output is
/// `tracing::debug!` events, so it is hidden under the default `RUST_LOG=warn`.
pub fn timing() -> bool {
    static VAL: OnceLock<bool> = OnceLock::new();
    *VAL.get_or_init(|| std::env::var("NEAT_AI_DISCOVERY_TIMING").is_ok())
}

/// Profile output mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProfileMode {
    /// No profiling output.
    #[default]
    None,
    /// Output structured profile as JSON, in a `tracing::info!` event.
    Json,
}

/// Get the profile output mode (cached).
///
/// Set `NEAT_AI_DISCOVERY_PROFILE=json` (case-insensitive) to enable JSON
/// profiling; any other value disables it. The output is a `tracing::info!`
/// event, so it is hidden under the default `RUST_LOG=warn`.
pub fn profile_mode() -> ProfileMode {
    static VAL: OnceLock<ProfileMode> = OnceLock::new();
    *VAL.get_or_init(|| {
        match std::env::var("NEAT_AI_DISCOVERY_PROFILE")
            .as_deref()
            .unwrap_or("")
            .to_lowercase()
            .as_str()
        {
            "json" => ProfileMode::Json,
            _ => ProfileMode::None,
        }
    })
}

/// Check if GPU metrics output is enabled (cached).
///
/// Presence-only: any set value enables it, `0` included. The output is a
/// `tracing::info!` event, so it is hidden under the default `RUST_LOG=warn`.
pub fn gpu_metrics() -> bool {
    static VAL: OnceLock<bool> = OnceLock::new();
    *VAL.get_or_init(|| std::env::var("NEAT_AI_DISCOVERY_GPU_METRICS").is_ok())
}

/// Whether a candidate-reconciliation mismatch should trip a `debug_assert!`
/// (Issue #1802).
///
/// Defaults to **on for debug builds** — so a new unaccounted drop path fails
/// the PR that introduces it under `cargo test` — and off for release builds,
/// where assertions are compiled out and the warn-only posture applies anyway.
/// Either way the `unaccounted_drop` rejection-breakdown entry and the
/// `tracing::warn!` are emitted, so a mismatch is never silent.
///
/// Set `NEAT_AI_DISCOVERY_STRICT_CANDIDATE_RECONCILIATION=0` (or `false`/`no`)
/// to force warn-only, or `=1` (or `true`/`yes`) to force the check on. The
/// default is on for debug builds and off for release builds.
///
/// Deliberately **not** cached in a `OnceLock`: it is read twice per discovery
/// pass, and caching would stop tests toggling it. Prefer
/// [`crate::analysis::candidate_reconciliation::StrictModeGuard`] over mutating
/// the environment.
pub fn strict_candidate_reconciliation() -> bool {
    super::helpers::parse_optional_bool_env("NEAT_AI_DISCOVERY_STRICT_CANDIDATE_RECONCILIATION")
        .unwrap_or(cfg!(debug_assertions))
}

/// Threshold for prediction-vs-actual calibration mismatch logging
/// (Issue #1165).
///
/// A failure-cache entry's `actual / expected` ratio triggers a structured
/// `tracing::warn!` when `|ratio| > threshold` or `|ratio| < 1 / threshold`.
/// Default `10.0` (matches the historical "10× off" rule of thumb in Issue
/// #1160). Override via `NEAT_AI_DISCOVERY_CALIBRATION_MISS_THRESHOLD`.
///
/// Values that fail to parse, are non-finite, or `<= 1.0` fall back to the
/// default so the log channel cannot be silenced by a malformed value.
pub fn calibration_miss_threshold() -> f32 {
    parse_env::<f32>("NEAT_AI_DISCOVERY_CALIBRATION_MISS_THRESHOLD")
        .filter(|v| v.is_finite() && *v > 1.0)
        .unwrap_or(
            crate::analysis::diagnostics::mcmc_diagnostics::DEFAULT_CALIBRATION_MISS_THRESHOLD,
        )
}
