//! Utility functions for analysis module
//!
//! This module contains helper functions for memory checks, deadline handling,
//! activation functions, and other utilities used across analysis modules.
//!
//! ## Module Structure
//!
//! - `memory` - Memory detection and system requirements checking (Issue #267)
//! - `platform` - Platform-specific setup (Linux XDG, Mesa warnings) (Issue #267)
//! - `deadline` - Deadline handling and logging utilities (Issue #268)
//! - `variant_generation` - Parameterised candidate variant generation (Issue #806)

pub mod deadline;
pub mod lock_contention;
pub mod memory;
pub mod platform;
pub mod variant_generation;

// ============================================================================
// Mutex helpers — graceful handling of poisoned mutexes (Issue #525)
// ============================================================================

use parking_lot::Mutex;

/// Lock a mutex, returning the guard wrapped in `Ok`.
///
/// `parking_lot::Mutex` does not poison on thread panic, so this helper always
/// succeeds. The `context` parameter is used as a diagnostic label for lock
/// contention tracing when verbose mode is enabled (Issue #837).
pub fn lock_or_bail<'a, T>(
    mutex: &'a Mutex<T>,
    context: &str,
) -> anyhow::Result<parking_lot::MutexGuard<'a, T>> {
    Ok(lock_contention::traced_lock_default(mutex, context))
}

/// Consume a mutex and return its inner value.
///
/// `parking_lot::Mutex` does not poison, so this always succeeds. The
/// `_context` parameter is retained for API compatibility.
pub fn into_inner_or_bail<T>(mutex: Mutex<T>, _context: &str) -> anyhow::Result<T> {
    Ok(mutex.into_inner())
}

// Re-export key memory functions for convenience
pub use memory::{
    DEFAULT_GPU_BATCH_SIZE, HIGH_PERF_GPU_BATCH_SIZE, LOW_MEMORY_GPU_BATCH_SIZE, MemoryPressure,
    MemoryTier, PARQUET_MEMORY_MULTIPLIER, bytes_to_mb_ceil, cap_gpu_batch_size_by_bytes,
    categorise_memory_pressure, categorise_memory_tier, check_memory_budget_exceeded,
    check_memory_for_parquet, check_memory_pressure_and_cancel, check_system_memory_requirements,
    detect_memory_pressure, detect_memory_tier, estimate_parquet_in_memory_bytes, get_memory_info,
    get_work_queue_capacity, get_work_queue_capacity_for_tier, is_memory_budget_exceeded,
    parquet_preload_fits_available, validate_parquet_memory_requirements,
    would_cancel_for_memory_pressure,
};

// Re-export platform setup functions
pub use platform::{ensure_xdg_runtime_dir, suppress_mesa_warnings_if_requested};

// Re-export deadline handling functions (Issue #268)
pub use deadline::{
    DEFAULT_DURATION_MS, GPU_QUEUE_TIMEOUT_MAX_SECS, GPU_QUEUE_TIMEOUT_MIN_SECS, MAX_DURATION_MS,
    MIN_DURATION_MS, OrderedNeuron, YEAR_2000_MS, build_deadline, calculate_effective_timeout_ms,
    calculate_gpu_batch_timeout, cap_deadline_to_wall_clock, deadline_passed,
    deadline_to_absolute_ms, derive_seed, focus_unused_observations_from_env, log_analysis_start,
    log_analysis_timeout, order_eligible_sources, order_focus_targets, parse_input_index,
    shuffle_slice, shuffle_within_top_k, source_input_index_bias_from_env,
};

// Re-export deadline override for tests
#[cfg(test)]
pub use deadline::deadline_override;

// Re-export lock contention tracing functions (Issue #837)
pub use lock_contention::{DEFAULT_LOCK_WAIT_THRESHOLD, traced_lock, traced_lock_default};

// Re-export variant generation functions for backward compatibility (Issue #806)
pub(crate) use variant_generation::sensible_bias_abs_max_for_squash;
pub use variant_generation::{
    filter_candidates_to_sensible_ranges, pair_coordinated_structural_with_weight_variants,
    pair_extreme_candidates_with_conservative_variants,
    pair_extreme_candidates_with_conservative_variants_calibrated,
    pair_synapse_candidates_with_weight_variants,
    pair_synapse_candidates_with_weight_variants_calibrated,
};

/// Check if verbose logging is enabled. Result is cached for performance.
/// Set `NEAT_AI_DISCOVERY_VERBOSE=1` to enable verbose logging.
/// This is public so it can be used by focus.rs and other modules.
///
/// Delegates to [`crate::config::verbose()`].
pub fn verbose_enabled() -> bool {
    crate::config::verbose()
}

/// Check if GPU timing is enabled. Result is cached for performance.
/// Set `NEAT_AI_DISCOVERY_GPU_TIMING=1` to enable GPU timing collection.
/// This adds ~5% overhead when enabled but provides detailed timing breakdown.
///
/// Delegates to [`crate::config::gpu_timing()`].
pub fn gpu_timing_enabled() -> bool {
    crate::config::gpu_timing()
}
