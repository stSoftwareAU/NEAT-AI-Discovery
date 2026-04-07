//! User-facing configuration accessors.
//!
//! These functions expose runtime settings that users control via environment
//! variables — GPU tuning, streaming, outlier analysis, watchdog, etc.

use std::sync::OnceLock;
use std::time::Duration;

use super::helpers::{parse_bool_env, parse_optional_bool_env};

// =============================================================================
// Core user-facing accessors
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
            .unwrap_or(crate::analysis::gpu::queue::recovery::DEFAULT_GPU_RETRY_LIMIT)
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

/// Check if streaming mode is enabled (vs full preload).
///
/// Returns `false` if `NEAT_AI_DISCOVERY_PRELOAD_ALL=1` is set.
pub fn streaming_enabled() -> bool {
    !preload_all()
}

/// Get the Metropolis-Hastings temperature for probabilistic acceptance (Issue #1018).
///
/// Set `NEAT_AI_DISCOVERY_MH_TEMPERATURE` to a positive finite number to enable
/// probabilistic acceptance of marginal synapse candidates. When unset,
/// deterministic threshold-based acceptance is used (existing behaviour).
///
/// Returns `None` when disabled (unset, empty, or invalid).
pub fn mh_temperature() -> Option<f32> {
    static VAL: OnceLock<Option<f32>> = OnceLock::new();
    *VAL.get_or_init(|| {
        let raw = std::env::var("NEAT_AI_DISCOVERY_MH_TEMPERATURE").ok()?;
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return None;
        }
        match trimmed.parse::<f32>() {
            Ok(v) if v.is_finite() && v > 0.0 => Some(v),
            _ => {
                tracing::debug!(
                    raw_value = trimmed,
                    "Ignoring invalid NEAT_AI_DISCOVERY_MH_TEMPERATURE (expected a finite number > 0)"
                );
                None
            }
        }
    })
}

/// Get the macOS `sample` program path for thread dumps.
///
/// Set `NEAT_AI_DISCOVERY_SAMPLE_PROGRAM` to override.
/// Default: `"sample"`.
pub fn sample_program() -> String {
    std::env::var("NEAT_AI_DISCOVERY_SAMPLE_PROGRAM").unwrap_or_else(|_| "sample".to_string())
}
