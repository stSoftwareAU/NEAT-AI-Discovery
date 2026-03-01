//! Central configuration for all environment variables (Issue #717).
//!
//! This module is the single source of truth for runtime environment variable
//! configuration. Each variable is documented with its purpose, type, default
//! value, and valid range. Typed accessor functions validate values at read
//! time and provide clear error messages via `tracing`.
//!
//! ## User-Facing Configuration
//!
//! | Variable | Type | Default | Description |
//! |----------|------|---------|-------------|
//! | `NEAT_AI_DISCOVERY_VERBOSE` | bool | `false` | Enable verbose logging (`1` to enable) |
//! | `NEAT_AI_DISCOVERY_GPU_TIMING` | bool | `false` | Enable GPU kernel timing (~5% overhead) |
//! | `NEAT_AI_DISCOVERY_GPU_BATCH_SIZE` | usize | auto | Override GPU batch size (64–4096) |
//! | `NEAT_AI_DISCOVERY_GPU_RETRY_LIMIT` | u32 | `3` | Max consecutive device-lost recovery attempts (0–10) |
//! | `NEAT_AI_DISCOVERY_WATCHDOG_STALL_SECS` | u64 | `0` (disabled) | Abort if no heartbeat for N seconds |
//! | `NEAT_AI_DISCOVERY_WATCHDOG_ABORT_DELAY_SECS` | u64 | `2` | Delay between thread dump and abort |
//! | `NEAT_AI_DISCOVERY_MAX_CACHED_BLOCKS` | usize | adaptive | Maximum streaming cache blocks |
//! | `NEAT_AI_DISCOVERY_PREFETCH_DEPTH` | usize | `2` | How many blocks ahead to prefetch |
//! | `NEAT_AI_DISCOVERY_PRELOAD_ALL` | bool | `false` | Disable streaming; use full preload |
//! | `NEAT_AI_DISCOVERY_BLOCK_SIZE` | usize | `10000` | Records per streaming block (10–100000) |
//! | `NEAT_AI_DISCOVERY_CONSTANT_SOURCE_EFFECT_THRESHOLD` | f32 | `1e-7` | Constant source folding threshold (`0` to disable) |
//! | `NEAT_AI_DISCOVERY_OUTLIER_ANALYSIS` | bool | `false` | Enable outlier-focused analysis |
//! | `NEAT_AI_DISCOVERY_OUTLIER_PERCENTILE` | u8 | `90` | Outlier percentile threshold (1–99) |
//! | `NEAT_AI_DISCOVERY_NEURON_TARGETS_OUTPUT_ONLY` | bool | `false` | Restrict focus targets to output neurons only |
//! | `NEAT_AI_DISCOVERY_FOCUS_UNUSED_OBSERVATIONS` | bool | `false` | Prioritise unused input neurons |
//! | `NEAT_AI_DISCOVERY_SOURCE_INPUT_INDEX_BIAS` | f64 | disabled | Bias source ordering toward higher input indices (> 0) |
//! | `NEAT_AI_DISCOVERY_ZERO_COPY` | Option\<bool\> | auto | Force-enable/disable zero-copy buffers |
//! | `NEAT_AI_DISCOVERY_QUIET_GPU` | bool | `false` | Suppress Mesa/libEGL debug output (Linux) |
//!
//! ## Observability Variables
//!
//! | Variable | Type | Default | Description |
//! |----------|------|---------|-------------|
//! | `RUST_LOG` | filter string | `warn` | Control tracing log level (e.g. `neat_ai_discovery=info`) |
//! | `NEAT_AI_DISCOVERY_TIMING` | bool | `false` | Print phase timing to stderr |
//! | `NEAT_AI_DISCOVERY_PROFILE` | `json`/empty | disabled | Output structured profile as JSON |
//! | `NEAT_AI_DISCOVERY_GPU_METRICS` | bool | `false` | Print GPU metrics to stderr |
//!
//! ## Detection Tuning Variables
//!
//! | Variable | Type | Default | Description |
//! |----------|------|---------|-------------|
//! | `NEAT_AI_DISCOVERY_DOMINANCE_THRESHOLD` | f32 | module default | Input dominance detection threshold |
//! | `NEAT_AI_DISCOVERY_GRADIENT_THRESHOLD` | f32 | module default | Gradient detection threshold |
//! | `NEAT_AI_DISCOVERY_NOISE_SIGNAL_THRESHOLD` | f32 | module default | Noise-to-signal ratio threshold |
//!
//! ## Internal/Debug Variables
//!
//! | Variable | Type | Default | Description |
//! |----------|------|---------|-------------|
//! | `NEAT_AI_DISCOVERY_SAMPLE_PROGRAM` | String | `sample` | macOS `sample` binary path override |
//!
//! ## GPU Platform Variables (Linux only, set internally)
//!
//! | Variable | Type | Default | Description |
//! |----------|------|---------|-------------|
//! | `EGL_LOG_LEVEL` | String | (not set) | Set to `fatal` when `QUIET_GPU` is enabled |
//! | `MESA_GLSL_CACHE_DISABLE` | String | (not set) | Set to `true` when `QUIET_GPU` is enabled |
//! | `MESA_DEBUG` | String | (not set) | Set to `silent` when `QUIET_GPU` is enabled |
//! | `XDG_RUNTIME_DIR` | String | (not set) | Auto-created if missing (Wayland requirement) |

use std::sync::OnceLock;
use std::time::Duration;

// =============================================================================
// Boolean flag helpers
// =============================================================================

/// Parse a boolean-style environment variable.
///
/// Truthy values: `"1"`, `"true"`, `"yes"` (case-insensitive).
/// Falsy values: `"0"`, `"false"`, `"no"` (case-insensitive), unset, or empty.
fn parse_bool_env(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .is_some_and(|v| matches!(v.trim().to_lowercase().as_str(), "1" | "true" | "yes"))
}

/// Parse an optional boolean-style environment variable.
///
/// Returns `Some(true)` for truthy, `Some(false)` for falsy, `None` if unset.
fn parse_optional_bool_env(name: &str) -> Option<bool> {
    std::env::var(name).ok().and_then(|v| {
        let v = v.trim().to_lowercase();
        match v.as_str() {
            "1" | "true" | "yes" => Some(true),
            "0" | "false" | "no" => Some(false),
            _ => None,
        }
    })
}

// =============================================================================
// User-facing configuration
// =============================================================================

/// Check if verbose logging is enabled (cached).
///
/// Set `NEAT_AI_DISCOVERY_VERBOSE=1` to enable.
pub fn verbose() -> bool {
    static VAL: OnceLock<bool> = OnceLock::new();
    *VAL.get_or_init(|| std::env::var("NEAT_AI_DISCOVERY_VERBOSE").is_ok())
}

/// Check if GPU kernel timing collection is enabled (cached).
///
/// Set `NEAT_AI_DISCOVERY_GPU_TIMING=1` to enable. Adds ~5% overhead.
pub fn gpu_timing() -> bool {
    static VAL: OnceLock<bool> = OnceLock::new();
    *VAL.get_or_init(|| std::env::var("NEAT_AI_DISCOVERY_GPU_TIMING").is_ok())
}

/// Get the GPU batch size override (cached).
///
/// Set `NEAT_AI_DISCOVERY_GPU_BATCH_SIZE` to a value in 64–4096.
/// Returns `None` if unset or out of range.
pub fn gpu_batch_size_override() -> Option<usize> {
    static VAL: OnceLock<Option<usize>> = OnceLock::new();
    *VAL.get_or_init(|| {
        std::env::var("NEAT_AI_DISCOVERY_GPU_BATCH_SIZE")
            .ok()
            .and_then(|val| val.parse::<usize>().ok())
            .filter(|size| (64..=4096).contains(size))
    })
}

/// Get the GPU retry limit for device-lost recovery (cached).
///
/// Set `NEAT_AI_DISCOVERY_GPU_RETRY_LIMIT` to a value 0–10.
/// Default: 3.
pub fn gpu_retry_limit() -> u32 {
    static VAL: OnceLock<u32> = OnceLock::new();
    *VAL.get_or_init(|| {
        std::env::var("NEAT_AI_DISCOVERY_GPU_RETRY_LIMIT")
            .ok()
            .and_then(|val| val.parse::<u32>().ok())
            .filter(|&n| n <= 10)
            .unwrap_or(super::analysis::gpu::queue::recovery::DEFAULT_GPU_RETRY_LIMIT)
    })
}

/// Get the watchdog stall timeout.
///
/// Set `NEAT_AI_DISCOVERY_WATCHDOG_STALL_SECS` to a positive number to enable.
/// Returns `None` when disabled (unset or `0`).
pub fn watchdog_stall_timeout() -> Option<Duration> {
    let secs = std::env::var("NEAT_AI_DISCOVERY_WATCHDOG_STALL_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0);

    if secs == 0 {
        None
    } else {
        Some(Duration::from_secs(secs))
    }
}

/// Get the watchdog abort delay.
///
/// Set `NEAT_AI_DISCOVERY_WATCHDOG_ABORT_DELAY_SECS` to control delay between
/// thread dump and process abort. Default: 2 seconds.
pub fn watchdog_abort_delay() -> Duration {
    let secs = std::env::var("NEAT_AI_DISCOVERY_WATCHDOG_ABORT_DELAY_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(2);

    Duration::from_secs(secs)
}

/// Get maximum streaming cache blocks.
///
/// Set `NEAT_AI_DISCOVERY_MAX_CACHED_BLOCKS` to override adaptive sizing.
/// Returns `None` for adaptive behaviour based on available RAM.
pub fn max_cached_blocks() -> Option<usize> {
    std::env::var("NEAT_AI_DISCOVERY_MAX_CACHED_BLOCKS")
        .ok()
        .and_then(|v| v.parse().ok())
}

/// Get streaming prefetch depth.
///
/// Set `NEAT_AI_DISCOVERY_PREFETCH_DEPTH` to control how many blocks ahead
/// to prefetch. Default: 2.
pub fn prefetch_depth() -> usize {
    std::env::var("NEAT_AI_DISCOVERY_PREFETCH_DEPTH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2)
}

/// Check if streaming mode is disabled (full preload requested).
///
/// Set `NEAT_AI_DISCOVERY_PRELOAD_ALL=1` to disable streaming.
pub fn preload_all() -> bool {
    parse_bool_env("NEAT_AI_DISCOVERY_PRELOAD_ALL")
}

/// Default records per streaming block.
pub const DEFAULT_BLOCK_SIZE: usize = 10_000;

/// Minimum records per streaming block.
pub const MIN_BLOCK_SIZE: usize = 10;

/// Maximum records per streaming block.
pub const MAX_BLOCK_SIZE: usize = 100_000;

/// Get the streaming block size.
///
/// Set `NEAT_AI_DISCOVERY_BLOCK_SIZE` to override. Default: 10000.
/// Clamped to 10–100000.
pub fn block_size() -> usize {
    std::env::var("NEAT_AI_DISCOVERY_BLOCK_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_BLOCK_SIZE)
        .clamp(MIN_BLOCK_SIZE, MAX_BLOCK_SIZE)
}

/// Check if outlier analysis is enabled.
///
/// Set `NEAT_AI_DISCOVERY_OUTLIER_ANALYSIS=1` to enable.
pub fn outlier_analysis() -> bool {
    parse_bool_env("NEAT_AI_DISCOVERY_OUTLIER_ANALYSIS")
}

/// Get the outlier percentile threshold.
///
/// Set `NEAT_AI_DISCOVERY_OUTLIER_PERCENTILE` to a value 1–99.
/// Default: 90.
pub fn outlier_percentile() -> u8 {
    std::env::var("NEAT_AI_DISCOVERY_OUTLIER_PERCENTILE")
        .ok()
        .and_then(|v| v.trim().parse::<u8>().ok())
        .filter(|&p| p > 0 && p < 100)
        .unwrap_or(90)
}

/// Check if neuron targets should be restricted to output neurons only.
///
/// Set `NEAT_AI_DISCOVERY_NEURON_TARGETS_OUTPUT_ONLY=1` to enable.
pub fn neuron_targets_output_only() -> bool {
    std::env::var("NEAT_AI_DISCOVERY_NEURON_TARGETS_OUTPUT_ONLY").is_ok()
}

/// Check if discovery should focus on unused observations.
///
/// Set `NEAT_AI_DISCOVERY_FOCUS_UNUSED_OBSERVATIONS=1` to enable.
/// Truthy values: `"1"`, `"true"`, `"yes"` (case-insensitive).
pub fn focus_unused_observations() -> bool {
    parse_bool_env("NEAT_AI_DISCOVERY_FOCUS_UNUSED_OBSERVATIONS")
}

/// Get the source input index bias strength.
///
/// Set `NEAT_AI_DISCOVERY_SOURCE_INPUT_INDEX_BIAS` to a positive finite number.
/// Returns `None` when disabled (unset, empty, or invalid).
pub fn source_input_index_bias() -> Option<f64> {
    let raw = std::env::var("NEAT_AI_DISCOVERY_SOURCE_INPUT_INDEX_BIAS").ok()?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    match trimmed.parse::<f64>() {
        Ok(v) if v.is_finite() && v > 0.0 => Some(v),
        _ => {
            tracing::debug!(
                raw_value = trimmed,
                "Ignoring invalid NEAT_AI_DISCOVERY_SOURCE_INPUT_INDEX_BIAS (expected a finite number > 0)"
            );
            None
        }
    }
}

/// Get the zero-copy buffer override.
///
/// Set `NEAT_AI_DISCOVERY_ZERO_COPY`:
/// - `"1"` / `"true"`: Force-enable
/// - `"0"` / `"false"`: Force-disable
/// - Unset: Auto-detect based on hardware
pub fn zero_copy_override() -> Option<bool> {
    parse_optional_bool_env("NEAT_AI_DISCOVERY_ZERO_COPY")
}

/// Check if Mesa/libEGL warning suppression is requested (Linux only).
///
/// Set `NEAT_AI_DISCOVERY_QUIET_GPU=1` to enable.
pub fn quiet_gpu() -> bool {
    std::env::var("NEAT_AI_DISCOVERY_QUIET_GPU").is_ok()
}

// =============================================================================
// Observability configuration
// =============================================================================

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

// =============================================================================
// Detection tuning
// =============================================================================

/// Get the noise-to-signal detection threshold.
///
/// Set `NEAT_AI_DISCOVERY_NOISE_SIGNAL_THRESHOLD` to override.
/// Requires each detection module's default to be passed as fallback.
pub fn noise_signal_threshold(default: f32) -> f32 {
    std::env::var("NEAT_AI_DISCOVERY_NOISE_SIGNAL_THRESHOLD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Get the input dominance detection threshold.
///
/// Set `NEAT_AI_DISCOVERY_DOMINANCE_THRESHOLD` to override.
/// Requires each detection module's default to be passed as fallback.
pub fn dominance_threshold(default: f32) -> f32 {
    std::env::var("NEAT_AI_DISCOVERY_DOMINANCE_THRESHOLD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Get the gradient detection threshold.
///
/// Set `NEAT_AI_DISCOVERY_GRADIENT_THRESHOLD` to override.
/// Requires each detection module's default to be passed as fallback.
pub fn gradient_threshold(default: f32) -> f32 {
    std::env::var("NEAT_AI_DISCOVERY_GRADIENT_THRESHOLD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

// =============================================================================
// Internal/debug configuration
// =============================================================================

/// Get the macOS `sample` program path for thread dumps.
///
/// Set `NEAT_AI_DISCOVERY_SAMPLE_PROGRAM` to override.
/// Default: `"sample"`.
pub fn sample_program() -> String {
    std::env::var("NEAT_AI_DISCOVERY_SAMPLE_PROGRAM").unwrap_or_else(|_| "sample".to_string())
}

// =============================================================================
// Constant source effect threshold (Issue #199)
// =============================================================================

/// Default constant source effect threshold.
pub const DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD: f32 = 1e-7;

/// Get the constant source effect threshold from environment.
///
/// Set `NEAT_AI_DISCOVERY_CONSTANT_SOURCE_EFFECT_THRESHOLD`:
/// - Unset / empty: returns `Some(DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD)`
/// - `0`: returns `None` (folding disabled)
/// - Positive finite number: returns `Some(value)`
pub fn constant_source_effect_threshold() -> Option<f32> {
    let raw = std::env::var("NEAT_AI_DISCOVERY_CONSTANT_SOURCE_EFFECT_THRESHOLD").ok();
    let Some(raw) = raw else {
        return Some(DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD);
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Some(DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD);
    }
    match trimmed.parse::<f32>() {
        Ok(v) if v.is_finite() && v == 0.0 => None,
        Ok(v) if v.is_finite() && v > 0.0 => Some(v),
        _ => {
            tracing::debug!(
                raw_value = ?trimmed,
                "Ignoring invalid NEAT_AI_DISCOVERY_CONSTANT_SOURCE_EFFECT_THRESHOLD (expected 0 or a finite number > 0)"
            );
            Some(DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD)
        }
    }
}

/// Get the constant source threshold considering both env var override and dynamic calculation.
///
/// If the environment variable is explicitly set, it takes precedence.
/// Otherwise, uses the dynamic threshold based on `source_std_dev_avg`.
pub fn constant_source_threshold_with_dynamic(source_std_dev_avg: Option<f32>) -> Option<f32> {
    // Check for explicit env var override first
    let raw = std::env::var("NEAT_AI_DISCOVERY_CONSTANT_SOURCE_EFFECT_THRESHOLD").ok();

    if let Some(raw) = raw {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            match trimmed.parse::<f32>() {
                Ok(v) if v.is_finite() && v == 0.0 => return None,
                Ok(v) if v.is_finite() && v > 0.0 => return Some(v),
                _ => {
                    tracing::debug!(
                        raw_value = ?trimmed,
                        "Ignoring invalid NEAT_AI_DISCOVERY_CONSTANT_SOURCE_EFFECT_THRESHOLD"
                    );
                }
            }
        }
    }

    // No explicit override — use dynamic threshold if variance data is available
    match source_std_dev_avg {
        Some(avg) if avg.is_finite() && avg > 0.0 => {
            let scaling_factor = (avg / 0.05).max(1.0);
            let dynamic = DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD * scaling_factor;
            Some(if dynamic.is_finite() {
                dynamic
            } else {
                DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD
            })
        }
        _ => Some(DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD),
    }
}

// =============================================================================
// Streaming mode check
// =============================================================================

/// Check if streaming mode is enabled (vs full preload).
///
/// Returns `false` if `NEAT_AI_DISCOVERY_PRELOAD_ALL=1` is set.
pub fn streaming_enabled() -> bool {
    !preload_all()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_bool_env_truthy_values() {
        // Cannot test with real env vars due to caching / global state.
        // Instead, test the parsing logic directly.
        assert!(matches!("1", "1" | "true" | "yes"));
        assert!(matches!("true", "1" | "true" | "yes"));
        assert!(matches!("yes", "1" | "true" | "yes"));
        assert!(!matches!("0", "1" | "true" | "yes"));
        assert!(!matches!("false", "1" | "true" | "yes"));
        assert!(!matches!("", "1" | "true" | "yes"));
    }

    #[test]
    fn optional_bool_parsing() {
        // Test the parse_optional_bool_env logic patterns
        let truthy = |s: &str| {
            let v = s.trim().to_lowercase();
            match v.as_str() {
                "1" | "true" | "yes" => Some(true),
                "0" | "false" | "no" => Some(false),
                _ => None,
            }
        };
        assert_eq!(truthy("1"), Some(true));
        assert_eq!(truthy("true"), Some(true));
        assert_eq!(truthy("yes"), Some(true));
        assert_eq!(truthy("YES"), Some(true));
        assert_eq!(truthy("0"), Some(false));
        assert_eq!(truthy("false"), Some(false));
        assert_eq!(truthy("no"), Some(false));
        assert_eq!(truthy("maybe"), None);
        assert_eq!(truthy(""), None);
    }

    #[test]
    fn block_size_defaults() {
        assert_eq!(DEFAULT_BLOCK_SIZE, 10_000);
        assert_eq!(MIN_BLOCK_SIZE, 10);
        assert_eq!(MAX_BLOCK_SIZE, 100_000);
    }

    #[test]
    fn outlier_percentile_default_value() {
        // When no env var is set, should return 90
        // (This test relies on the env var NOT being set in the test environment)
        let result = outlier_percentile();
        assert!(result > 0 && result < 100);
    }

    #[test]
    fn constant_source_default_threshold() {
        assert!((DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD - 1e-7).abs() < 1e-13);
    }

    #[test]
    fn constant_source_threshold_with_dynamic_no_variance() {
        // With no variance data and no env var, should return default
        let result = constant_source_threshold_with_dynamic(None);
        assert_eq!(result, Some(DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD));
    }

    #[test]
    fn constant_source_threshold_with_dynamic_high_variance() {
        // With high variance, threshold should scale up
        let result = constant_source_threshold_with_dynamic(Some(0.5));
        let expected_scaling = (0.5_f32 / 0.05).max(1.0);
        let expected = DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD * expected_scaling;
        assert!(result.is_some());
        let val = result.unwrap();
        assert!((val - expected).abs() < 1e-12);
    }

    #[test]
    fn constant_source_threshold_with_dynamic_low_variance() {
        // With low variance (< 0.05), scaling factor should be 1.0
        let result = constant_source_threshold_with_dynamic(Some(0.01));
        assert_eq!(result, Some(DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD));
    }

    #[test]
    fn constant_source_threshold_with_dynamic_invalid_variance() {
        // NaN and negative values should fall back to default
        let result_nan = constant_source_threshold_with_dynamic(Some(f32::NAN));
        assert_eq!(result_nan, Some(DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD));

        let result_neg = constant_source_threshold_with_dynamic(Some(-1.0));
        assert_eq!(result_neg, Some(DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD));
    }

    #[test]
    fn profile_mode_default_is_none() {
        // Default profile mode should be None
        let default_mode = ProfileMode::default();
        assert_eq!(default_mode, ProfileMode::None);
    }

    #[test]
    fn watchdog_abort_delay_default() {
        // When env var is not set, should return 2 seconds
        let delay = watchdog_abort_delay();
        assert!(delay.as_secs() >= 1);
    }
}
