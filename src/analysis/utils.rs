//! Utility functions for analysis module
//!
//! This module contains helper functions for memory checks, deadline handling,
//! activation functions, and other utilities used across analysis modules.

/// Check if verbose logging is enabled. Result is cached for performance.
/// Set `NEAT_AI_DISCOVERY_VERBOSE=1` to enable verbose logging.
/// This is public so it can be used by focus.rs and other modules.
pub fn verbose_enabled() -> bool {
    use std::sync::OnceLock;
    static VERBOSE: OnceLock<bool> = OnceLock::new();
    *VERBOSE.get_or_init(|| std::env::var("NEAT_AI_DISCOVERY_VERBOSE").is_ok())
}

// TODO: Move other utility functions from impl.rs here:
// - check_memory_for_parquet
// - get_memory_info (platform-specific)
// - parse_vm_stat_line, parse_vm_stat_page_size (macOS)
// - parse_meminfo_line (Linux)
// - detect_system_resources
// - build_deadline, deadline_passed, calculate_effective_timeout_ms
// - log_analysis_start, log_analysis_timeout
// - wait_for_buffer_map, wait_for_buffer_maps_batch
// - calculate_gpu_batch_timeout
// - All activation functions (identity_activation, tanh_activation, etc.)
// - is_threshold_activation, has_sufficient_output_variance
// - activation_name_to_gpu_id, ACTIVATION_SPECS
