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
pub mod memory;
pub mod platform;
pub mod variant_generation;

// ============================================================================
// Mutex helpers — graceful handling of poisoned mutexes (Issue #525)
// ============================================================================

use std::sync::Mutex;

/// Lock a mutex, returning an `anyhow::Error` instead of panicking if poisoned.
///
/// In an FFI library a panic unwinds into the calling process (Deno / NEAT-AI)
/// and causes an unexpected abort. This helper converts a `PoisonError` into a
/// recoverable `anyhow::Error` that propagates to the JSON `success: false`
/// boundary.
///
/// The `context` parameter is included in the error message for diagnostics.
pub fn lock_or_bail<'a, T>(
    mutex: &'a Mutex<T>,
    context: &str,
) -> anyhow::Result<std::sync::MutexGuard<'a, T>> {
    mutex.lock().map_err(|_| {
        anyhow::anyhow!("Mutex poisoned ({context}): a thread panicked while holding this lock")
    })
}

/// Consume a mutex and return its inner value, or an error if poisoned.
///
/// Equivalent to `Mutex::into_inner().unwrap()` but returns an error instead
/// of panicking.
pub fn into_inner_or_bail<T>(mutex: Mutex<T>, context: &str) -> anyhow::Result<T> {
    mutex.into_inner().map_err(|_| {
        anyhow::anyhow!(
            "Mutex poisoned on into_inner ({context}): a thread panicked while holding this lock"
        )
    })
}

// Re-export key memory functions for convenience
pub use memory::{
    DEFAULT_GPU_BATCH_SIZE, HIGH_PERF_GPU_BATCH_SIZE, LOW_MEMORY_GPU_BATCH_SIZE, MemoryPressure,
    MemoryTier, cap_gpu_batch_size_by_bytes, categorise_memory_pressure, categorise_memory_tier,
    check_memory_for_parquet, check_system_memory_requirements, detect_memory_pressure,
    detect_memory_tier, get_memory_info, get_work_queue_capacity, get_work_queue_capacity_for_tier,
    validate_parquet_memory_requirements,
};

// Re-export platform setup functions
pub use platform::{ensure_xdg_runtime_dir, suppress_mesa_warnings_if_requested};

// Re-export deadline handling functions (Issue #268)
pub use deadline::{
    DEFAULT_DURATION_MS, GPU_QUEUE_TIMEOUT_MAX_SECS, GPU_QUEUE_TIMEOUT_MIN_SECS, MAX_DURATION_MS,
    MIN_DURATION_MS, OrderedNeuron, YEAR_2000_MS, build_deadline, calculate_effective_timeout_ms,
    calculate_gpu_batch_timeout, deadline_passed, derive_seed, focus_unused_observations_from_env,
    log_analysis_start, log_analysis_timeout, order_eligible_sources, order_focus_targets,
    parse_input_index, shuffle_slice, shuffle_within_top_k, source_input_index_bias_from_env,
};

// Re-export deadline override for tests
#[cfg(test)]
pub use deadline::deadline_override;

// Re-export variant generation functions for backward compatibility (Issue #806)
pub(crate) use variant_generation::sensible_bias_abs_max_for_squash;
pub use variant_generation::{
    filter_candidates_to_sensible_ranges, pair_coordinated_structural_with_weight_variants,
    pair_extreme_candidates_with_conservative_variants,
    pair_synapse_candidates_with_weight_variants,
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
