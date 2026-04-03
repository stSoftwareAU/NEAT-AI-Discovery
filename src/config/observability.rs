//! Observability configuration — timing, profiling, and GPU metrics.

use std::sync::OnceLock;

/// Check if phase timing output is enabled (cached).
///
/// Set `NEAT_AI_DISCOVERY_TIMING=1` to enable.
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
    /// Output structured profile as JSON to stderr.
    Json,
}

/// Get the profile output mode (cached).
///
/// Set `NEAT_AI_DISCOVERY_PROFILE=json` to enable JSON profiling.
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
/// Set `NEAT_AI_DISCOVERY_GPU_METRICS=1` to enable.
pub fn gpu_metrics() -> bool {
    static VAL: OnceLock<bool> = OnceLock::new();
    *VAL.get_or_init(|| std::env::var("NEAT_AI_DISCOVERY_GPU_METRICS").is_ok())
}
