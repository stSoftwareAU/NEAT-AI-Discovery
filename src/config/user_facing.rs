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

/// Maximum valid source input index bias.
pub const MAX_SOURCE_INPUT_INDEX_BIAS: f64 = 10.0;

/// Get the source input index bias strength.
///
/// Set `NEAT_AI_DISCOVERY_SOURCE_INPUT_INDEX_BIAS` to a positive finite number
/// up to 10.0. Returns `None` when disabled (unset, empty, out of range, or
/// invalid).
pub fn source_input_index_bias() -> Option<f64> {
    let raw = std::env::var("NEAT_AI_DISCOVERY_SOURCE_INPUT_INDEX_BIAS").ok()?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    match trimmed.parse::<f64>() {
        Ok(v) if v.is_finite() && v > 0.0 && v <= MAX_SOURCE_INPUT_INDEX_BIAS => Some(v),
        _ => {
            tracing::debug!(
                raw_value = trimmed,
                "Ignoring invalid NEAT_AI_DISCOVERY_SOURCE_INPUT_INDEX_BIAS \
                 (expected a finite number in 0.0–{MAX_SOURCE_INPUT_INDEX_BIAS})"
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

/// Minimum valid Metropolis-Hastings temperature.
pub const MIN_MH_TEMPERATURE: f32 = 0.01;

/// Maximum valid Metropolis-Hastings temperature.
pub const MAX_MH_TEMPERATURE: f32 = 5.0;

/// Get the Metropolis-Hastings temperature for probabilistic acceptance (Issue #1018).
///
/// Set `NEAT_AI_DISCOVERY_MH_TEMPERATURE` to a value in 0.01–5.0 to enable
/// probabilistic acceptance of marginal synapse candidates. When unset,
/// deterministic threshold-based acceptance is used (existing behaviour).
///
/// Returns `None` when disabled (unset, empty, out of range, or invalid).
pub fn mh_temperature() -> Option<f32> {
    static VAL: OnceLock<Option<f32>> = OnceLock::new();
    *VAL.get_or_init(|| {
        let raw = std::env::var("NEAT_AI_DISCOVERY_MH_TEMPERATURE").ok()?;
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return None;
        }
        match trimmed.parse::<f32>() {
            Ok(v) if v.is_finite() && (MIN_MH_TEMPERATURE..=MAX_MH_TEMPERATURE).contains(&v) => {
                Some(v)
            }
            _ => {
                tracing::debug!(
                    raw_value = trimmed,
                    "Ignoring invalid NEAT_AI_DISCOVERY_MH_TEMPERATURE \
                     (expected a finite number in {MIN_MH_TEMPERATURE}–{MAX_MH_TEMPERATURE})"
                );
                None
            }
        }
    })
}

/// Check if the batch-successful discovery module is enabled (Issue #1059).
///
/// Disabled by default — zero production successes across all creatures in
/// GRQ-sampler evidence. Set `NEAT_AI_DISCOVERY_BATCH_SUCCESSFUL=1` to
/// re-enable for experimentation.
pub fn batch_successful_enabled() -> bool {
    parse_bool_env("NEAT_AI_DISCOVERY_BATCH_SUCCESSFUL")
}

/// Get the focus ranking memory budget in megabytes (Issue #1172).
///
/// Set `NEAT_AI_DISCOVERY_FOCUS_RANKING_MEMORY_BUDGET_MB` to cap the eager
/// pre-load size used by [`crate::focus::rank_focus_neurons`] and
/// [`crate::focus::rank_focus_neurons_with_history`]. When set, the projected
/// in-memory size of the parquet file is compared against this budget and
/// lazy mode is selected when the projection exceeds it (with a structured
/// `info` log instead of a `WARN`). When unset, the existing
/// `check_memory_for_parquet` heuristic is used (auto-detect plus `WARN` on
/// fallback) to preserve behaviour on big hosts.
///
/// Returns `None` when the variable is unset, empty, non-numeric, or zero.
pub fn focus_ranking_memory_budget_mb() -> Option<u64> {
    let raw = std::env::var("NEAT_AI_DISCOVERY_FOCUS_RANKING_MEMORY_BUDGET_MB").ok()?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    match trimmed.parse::<u64>() {
        Ok(0) => {
            tracing::debug!(
                raw_value = trimmed,
                "Ignoring NEAT_AI_DISCOVERY_FOCUS_RANKING_MEMORY_BUDGET_MB=0 (must be > 0)"
            );
            None
        }
        Ok(v) => Some(v),
        Err(_) => {
            tracing::debug!(
                raw_value = trimmed,
                "Ignoring invalid NEAT_AI_DISCOVERY_FOCUS_RANKING_MEMORY_BUDGET_MB \
                 (expected a positive integer in megabytes)"
            );
            None
        }
    }
}

/// Default safety margin (in megabytes) reserved from OS-available memory
/// before the focus ranker pre-loads a parquet file (Issue #1376).
pub const DEFAULT_FOCUS_RANKING_MEMORY_MARGIN_MB: u64 = 1024;

/// Get the focus ranking memory safety margin in megabytes (Issue #1376).
///
/// When no explicit `NEAT_AI_DISCOVERY_FOCUS_RANKING_MEMORY_BUDGET_MB` is set,
/// the eager-vs-lazy decision is based on **real OS-available memory** minus
/// this margin. The margin reserves headroom for GPU buffers, the system, and
/// allocator slack so a pre-load that *just* fits does not push the host into
/// swap.
///
/// Override with `NEAT_AI_DISCOVERY_FOCUS_RANKING_MEMORY_MARGIN_MB`. Falls back
/// to [`DEFAULT_FOCUS_RANKING_MEMORY_MARGIN_MB`] when the variable is unset,
/// empty, or non-numeric. A value of `0` is honoured (no margin reserved).
pub fn focus_ranking_memory_margin_mb() -> u64 {
    let Ok(raw) = std::env::var("NEAT_AI_DISCOVERY_FOCUS_RANKING_MEMORY_MARGIN_MB") else {
        return DEFAULT_FOCUS_RANKING_MEMORY_MARGIN_MB;
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return DEFAULT_FOCUS_RANKING_MEMORY_MARGIN_MB;
    }
    match trimmed.parse::<u64>() {
        Ok(v) => v,
        Err(_) => {
            tracing::debug!(
                raw_value = trimmed,
                "Ignoring invalid NEAT_AI_DISCOVERY_FOCUS_RANKING_MEMORY_MARGIN_MB \
                 (expected a non-negative integer in megabytes)"
            );
            DEFAULT_FOCUS_RANKING_MEMORY_MARGIN_MB
        }
    }
}

/// Default wall-clock budget (in milliseconds) for a single focus-ranking run
/// (Issue #1375). Two minutes mirrors the per-chunk Rust FFI analysis budget.
/// A pathological focus-ranking run that exceeds this aborts gracefully so the
/// TypeScript caller can fall back to its instant local ranking path instead of
/// blowing the whole discovery wall-clock budget.
pub const DEFAULT_FOCUS_RANKING_BUDGET_MS: u64 = 120_000;

/// Lower bound (in milliseconds) for the focus-ranking wall-clock budget
/// (Issue #1385). A positive override below this is clamped up so a
/// misconfigured tiny value (e.g. `5`) cannot silently abort every run after a
/// few milliseconds and degrade every creature to the fallback path. `0`
/// remains the explicit opt-out and bypasses the clamp.
pub const FOCUS_RANKING_BUDGET_MIN_MS: u64 = 1_000;

/// Upper bound (in milliseconds) for the focus-ranking wall-clock budget
/// (Issue #1385). A positive override above this is clamped down so a huge
/// value (e.g. `999999999`) cannot effectively restore the unbounded
/// behaviour Issue #1375 set out to prevent. One hour mirrors the discovery
/// wall-clock budget.
pub const FOCUS_RANKING_BUDGET_MAX_MS: u64 = 3_600_000;

/// Grace period (in milliseconds) added on top of the focus-ranking budget
/// before an in-flight pass is forced to abort (Issue #1375). Mirrors the
/// per-chunk FFI "grace 1s" allowance so a check that lands mid-operation does
/// not abort a run that was about to finish anyway.
pub const FOCUS_RANKING_BUDGET_GRACE_MS: u64 = 1_000;

/// Get the focus-ranking wall-clock budget in milliseconds (Issue #1375).
///
/// Focus ranking previously had **no** wall-clock bound: in the #1373 incident
/// it ran for 1h 11m and contributed to the whole discovery task overrunning
/// its 3h budget and being killed. This budget gives focus ranking the same
/// safety net the per-chunk FFI analysis already enforces.
///
/// Override with `NEAT_AI_DISCOVERY_FOCUS_RANKING_BUDGET_MS`:
/// - Unset / empty / non-numeric: returns [`DEFAULT_FOCUS_RANKING_BUDGET_MS`].
/// - `0`: disables the budget (returns `None`, fully unbounded — opt-out).
/// - Any positive integer: clamped to
///   `[FOCUS_RANKING_BUDGET_MIN_MS, FOCUS_RANKING_BUDGET_MAX_MS]` (Issue #1385).
///
/// Returns `None` only when the budget is explicitly disabled with `0`.
///
/// ## Relationship to the shared discovery deadline (Issue #1407)
///
/// This budget is no longer an *independent* window. When the caller supplies
/// the shared absolute discovery deadline (`analysisDeadlineMs`) to
/// `rank_focus_neurons`, focus selection aborts at whichever is **sooner**: the
/// shared deadline or this wall-clock budget. The budget therefore acts purely
/// as a safety net that caps a pathological ranking run; it cannot extend focus
/// selection past the shared discovery deadline that the synapse/neuron
/// analysis phase also bills against. When no shared deadline is supplied, the
/// budget behaves exactly as before.
pub fn focus_ranking_budget_ms() -> Option<u64> {
    let Ok(raw) = std::env::var("NEAT_AI_DISCOVERY_FOCUS_RANKING_BUDGET_MS") else {
        return Some(DEFAULT_FOCUS_RANKING_BUDGET_MS);
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Some(DEFAULT_FOCUS_RANKING_BUDGET_MS);
    }
    match trimmed.parse::<u64>() {
        Ok(0) => {
            tracing::debug!(
                "Focus-ranking wall-clock budget disabled via \
                 NEAT_AI_DISCOVERY_FOCUS_RANKING_BUDGET_MS=0"
            );
            None
        }
        Ok(v) => {
            let clamped = v.clamp(FOCUS_RANKING_BUDGET_MIN_MS, FOCUS_RANKING_BUDGET_MAX_MS);
            if clamped != v {
                tracing::debug!(
                    requested_ms = v,
                    clamped_ms = clamped,
                    min_ms = FOCUS_RANKING_BUDGET_MIN_MS,
                    max_ms = FOCUS_RANKING_BUDGET_MAX_MS,
                    "Clamped NEAT_AI_DISCOVERY_FOCUS_RANKING_BUDGET_MS to the \
                     supported range"
                );
            }
            Some(clamped)
        }
        Err(_) => {
            tracing::debug!(
                raw_value = trimmed,
                "Ignoring invalid NEAT_AI_DISCOVERY_FOCUS_RANKING_BUDGET_MS \
                 (expected a non-negative integer in milliseconds)"
            );
            Some(DEFAULT_FOCUS_RANKING_BUDGET_MS)
        }
    }
}

// ============================================================================
// Reserved analysis budget (Issue #1408)
// ============================================================================

/// Default reserved minimum analysis window: 60 seconds.
///
/// Guarantees synapse/neuron analysis at least this slice of the shared
/// discovery deadline so focus selection and parquet loading cannot starve it
/// (Issue #1408). Matches the 60s "only N seconds remaining" warning threshold
/// already emitted by the analysis logger.
pub const DEFAULT_ANALYSIS_RESERVE_MS: u64 = 60_000;

/// Maximum reserved analysis window: 1 hour (matches the maximum timeout).
pub const ANALYSIS_RESERVE_MAX_MS: u64 = 3_600_000;

/// Default fraction of the remaining discovery window reserved for analysis.
///
/// The effective reserve is `min(reserve_ms, remaining * fraction)`, so this
/// caps the reserve on small budgets and stops it starving focus/parquet to
/// zero. A balanced 0.5 splits a tight window evenly between loading and
/// analysis.
pub const DEFAULT_ANALYSIS_RESERVE_FRACTION: f64 = 0.5;

/// Maximum reserve fraction. Capped below 1.0 so loading always keeps a slice.
pub const ANALYSIS_RESERVE_FRACTION_MAX: f64 = 0.9;

/// Reserved minimum analysis window in milliseconds (Issue #1408).
///
/// Override with `NEAT_AI_DISCOVERY_ANALYSIS_RESERVE_MS`:
/// - Unset / empty / non-numeric: returns [`DEFAULT_ANALYSIS_RESERVE_MS`].
/// - `0`: disables the reserve (opt-out — restores pre-#1408 behaviour where
///   focus/parquet may consume the whole window).
/// - Any positive integer: clamped to `[1, ANALYSIS_RESERVE_MAX_MS]` ms.
pub fn analysis_reserve_ms() -> u64 {
    let Ok(raw) = std::env::var("NEAT_AI_DISCOVERY_ANALYSIS_RESERVE_MS") else {
        return DEFAULT_ANALYSIS_RESERVE_MS;
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return DEFAULT_ANALYSIS_RESERVE_MS;
    }
    match trimmed.parse::<u64>() {
        Ok(0) => {
            tracing::debug!(
                "Analysis reserve disabled via NEAT_AI_DISCOVERY_ANALYSIS_RESERVE_MS=0"
            );
            0
        }
        Ok(v) => v.clamp(1, ANALYSIS_RESERVE_MAX_MS),
        Err(_) => {
            tracing::debug!(
                raw_value = trimmed,
                "Ignoring invalid NEAT_AI_DISCOVERY_ANALYSIS_RESERVE_MS \
                 (expected a non-negative integer in milliseconds)"
            );
            DEFAULT_ANALYSIS_RESERVE_MS
        }
    }
}

/// Fraction of the remaining discovery window reserved for analysis (Issue #1408).
///
/// Override with `NEAT_AI_DISCOVERY_ANALYSIS_RESERVE_FRACTION`:
/// - Unset / empty / non-finite / out of range: returns
///   [`DEFAULT_ANALYSIS_RESERVE_FRACTION`].
/// - Any finite value in `(0.0, ANALYSIS_RESERVE_FRACTION_MAX]`: clamped to that
///   range.
pub fn analysis_reserve_fraction() -> f64 {
    let Ok(raw) = std::env::var("NEAT_AI_DISCOVERY_ANALYSIS_RESERVE_FRACTION") else {
        return DEFAULT_ANALYSIS_RESERVE_FRACTION;
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return DEFAULT_ANALYSIS_RESERVE_FRACTION;
    }
    match trimmed.parse::<f64>() {
        Ok(v) if v.is_finite() && v > 0.0 => v.min(ANALYSIS_RESERVE_FRACTION_MAX),
        _ => {
            tracing::debug!(
                raw_value = trimmed,
                "Ignoring invalid NEAT_AI_DISCOVERY_ANALYSIS_RESERVE_FRACTION \
                 (expected a finite number in 0.0–{ANALYSIS_RESERVE_FRACTION_MAX})"
            );
            DEFAULT_ANALYSIS_RESERVE_FRACTION
        }
    }
}

/// Default perf-cliff threshold (in milliseconds) for a lazy focus-ranking
/// pass (Issue #1377). When a *lazy* pass runs longer than this, a single,
/// clearly-labelled perf-cliff `WARN` is emitted naming the neuron count and
/// projected dataset size, so the #1373-style "lazy ranking ran for over an
/// hour" cliff is one log line instead of a forensic exercise. 60 seconds is
/// well below the #1375 wall-clock budget (120s) so the warning fires before
/// the budget aborts the run.
pub const DEFAULT_FOCUS_RANKING_PERF_CLIFF_MS: u64 = 60_000;

/// Get the lazy focus-ranking perf-cliff threshold in milliseconds (Issue #1377).
///
/// A lazy ranking pass exceeding this threshold emits one explicit perf-cliff
/// `WARN`. The fast preload path never trips it.
///
/// Override with `NEAT_AI_DISCOVERY_FOCUS_RANKING_PERF_CLIFF_MS`:
/// - Unset / empty / non-numeric: returns [`DEFAULT_FOCUS_RANKING_PERF_CLIFF_MS`].
/// - `0`: disables the perf-cliff warning (opt-out).
/// - Any positive integer: that many milliseconds.
pub fn focus_ranking_perf_cliff_ms() -> u64 {
    let Ok(raw) = std::env::var("NEAT_AI_DISCOVERY_FOCUS_RANKING_PERF_CLIFF_MS") else {
        return DEFAULT_FOCUS_RANKING_PERF_CLIFF_MS;
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return DEFAULT_FOCUS_RANKING_PERF_CLIFF_MS;
    }
    match trimmed.parse::<u64>() {
        Ok(v) => v,
        Err(_) => {
            tracing::debug!(
                raw_value = trimmed,
                "Ignoring invalid NEAT_AI_DISCOVERY_FOCUS_RANKING_PERF_CLIFF_MS \
                 (expected a non-negative integer in milliseconds)"
            );
            DEFAULT_FOCUS_RANKING_PERF_CLIFF_MS
        }
    }
}

/// Default streaming session TTL in seconds (1 hour).
pub const DEFAULT_SESSION_TTL_SECS: u64 = 3600;

/// Minimum streaming session TTL in seconds (60 seconds).
pub const MIN_SESSION_TTL_SECS: u64 = 60;

/// Maximum streaming session TTL in seconds (24 hours).
pub const MAX_SESSION_TTL_SECS: u64 = 86400;

/// Get the streaming session TTL (time-to-live) in seconds.
///
/// Set `NEAT_AI_DISCOVERY_SESSION_TTL_SECS` to control how long orphaned
/// streaming sessions are kept before automatic cleanup. Sessions older than
/// this threshold are removed at the start of each new `start_discovery_session()`
/// call.
///
/// Default: 3600 (1 hour). Clamped to 60–86400.
pub fn session_ttl_secs() -> u64 {
    std::env::var("NEAT_AI_DISCOVERY_SESSION_TTL_SECS")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(DEFAULT_SESSION_TTL_SECS)
        .clamp(MIN_SESSION_TTL_SECS, MAX_SESSION_TTL_SECS)
}

/// Default maximum wall-clock minutes for total discovery time (Issue #1098).
pub const DEFAULT_MAX_WALL_CLOCK_MINUTES: u64 = 20;

/// Minimum valid wall-clock cap in minutes.
pub const MIN_WALL_CLOCK_MINUTES: u64 = 1;

/// Maximum valid wall-clock cap in minutes.
pub const MAX_WALL_CLOCK_MINUTES: u64 = 120;

/// Get the maximum wall-clock cap for total discovery time in minutes (Issue #1098).
///
/// Set `NEAT_AI_DISCOVERY_MAX_WALL_CLOCK_MINUTES` to control the overall
/// wall-clock cap for discovery (recording + analysis combined). This prevents
/// total discovery from exceeding the configured limit regardless of how
/// recording and analysis budgets are split.
///
/// Default: 20 minutes. Clamped to 1–120.
pub fn max_wall_clock_minutes() -> u64 {
    std::env::var("NEAT_AI_DISCOVERY_MAX_WALL_CLOCK_MINUTES")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(DEFAULT_MAX_WALL_CLOCK_MINUTES)
        .clamp(MIN_WALL_CLOCK_MINUTES, MAX_WALL_CLOCK_MINUTES)
}

/// Get the macOS `sample` program path for thread dumps.
///
/// Set `NEAT_AI_DISCOVERY_SAMPLE_PROGRAM` to override.
/// Default: `"sample"`.
pub fn sample_program() -> String {
    std::env::var("NEAT_AI_DISCOVERY_SAMPLE_PROGRAM").unwrap_or_else(|_| "sample".to_string())
}

// =============================================================================
// Issue #1132 — creature-level discovery strategy adaptation
// =============================================================================

/// Rolling success-rate threshold below which conservative mode engages
/// (Issue #1132).
///
/// Set `NEAT_AI_DISCOVERY_LOW_SUCCESS_RATE_THRESHOLD` to override. Values
/// outside (0.0, 1.0] are ignored and the default is used.
///
/// Default:
/// [`crate::analysis::discovery_mode::DEFAULT_LOW_SUCCESS_RATE_THRESHOLD`]
/// (0.2).
pub fn low_success_rate_threshold() -> f32 {
    std::env::var("NEAT_AI_DISCOVERY_LOW_SUCCESS_RATE_THRESHOLD")
        .ok()
        .and_then(|v| v.trim().parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v > 0.0 && *v <= 1.0)
        .unwrap_or(crate::analysis::discovery_mode::DEFAULT_LOW_SUCCESS_RATE_THRESHOLD)
}

/// Maximum number of consecutive failed passes before conservative mode is
/// abandoned as ineffective (Issue #1132).
///
/// Set `NEAT_AI_DISCOVERY_CONSERVATIVE_MODE_MAX_EPOCHS` to override.
///
/// Default:
/// [`crate::analysis::discovery_mode::DEFAULT_CONSERVATIVE_MODE_MAX_EPOCHS`]
/// (20).
pub fn conservative_mode_max_epochs() -> u32 {
    std::env::var("NEAT_AI_DISCOVERY_CONSERVATIVE_MODE_MAX_EPOCHS")
        .ok()
        .and_then(|v| v.trim().parse::<u32>().ok())
        .filter(|v| *v >= 1)
        .unwrap_or(crate::analysis::discovery_mode::DEFAULT_CONSERVATIVE_MODE_MAX_EPOCHS)
}

/// Multiplier applied to `COORDINATED_MIN_EXPECTED_GAIN` when conservative
/// mode is active (Issue #1132).
///
/// Set `NEAT_AI_DISCOVERY_CONSERVATIVE_GAIN_MULTIPLIER` to override. Values
/// below `1.0` are clamped to `1.0` so the floor is never relaxed below the
/// base constant.
///
/// Default:
/// [`crate::analysis::discovery_mode::DEFAULT_CONSERVATIVE_GAIN_MULTIPLIER`]
/// (10.0).
pub fn conservative_gain_multiplier() -> f32 {
    let raw = std::env::var("NEAT_AI_DISCOVERY_CONSERVATIVE_GAIN_MULTIPLIER")
        .ok()
        .and_then(|v| v.trim().parse::<f32>().ok())
        .filter(|v| v.is_finite())
        .unwrap_or(crate::analysis::discovery_mode::DEFAULT_CONSERVATIVE_GAIN_MULTIPLIER);
    raw.max(1.0)
}

/// Default consecutive-trailing-failure threshold above which the drought
/// diagnostic warn log fires (Issue #1202).
pub const DEFAULT_DROUGHT_LOG_THRESHOLD: u32 = 5;

/// Consecutive trailing failure count at which the drought diagnostic
/// `tracing::warn!` event is emitted and `droughtDiagnostic` populated on the
/// FFI metadata (Issue #1202).
///
/// Set `NEAT_AI_DISCOVERY_DROUGHT_LOG_THRESHOLD` to a positive integer to
/// override. Values that fail to parse, are zero, or are otherwise invalid
/// fall back to [`DEFAULT_DROUGHT_LOG_THRESHOLD`] (5).
pub fn drought_log_threshold() -> u32 {
    std::env::var("NEAT_AI_DISCOVERY_DROUGHT_LOG_THRESHOLD")
        .ok()
        .and_then(|v| v.trim().parse::<u32>().ok())
        .filter(|v| *v >= 1)
        .unwrap_or(DEFAULT_DROUGHT_LOG_THRESHOLD)
}

/// Operator escape hatch — force a one-shot reset of the candidate cache
/// failed entries and target cooldown tracker after this many consecutive
/// empty discovery passes (Issue #1205).
///
/// Set `NEAT_AI_DISCOVERY_DROUGHT_RESET_AFTER_EPOCHS` to a positive integer
/// to enable. Returns `None` when unset, zero, or invalid — meaning the
/// escape hatch is off.
///
/// The reset clears all failed `CandidateOutcomeCache` outcomes (preserving
/// successes and source-type stats) and all `TargetFailureTracker` entries
/// currently in cooldown. The reset fires at most once per consecutive
/// failure streak; a successful pass re-arms the lever.
pub fn drought_reset_after_epochs() -> Option<u32> {
    std::env::var("NEAT_AI_DISCOVERY_DROUGHT_RESET_AFTER_EPOCHS")
        .ok()
        .and_then(|v| v.trim().parse::<u32>().ok())
        .filter(|v| *v >= 1)
}
