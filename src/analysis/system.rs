//! System utilities module
//!
//! This module provides a unified interface for memory detection, system requirements
//! checking, and GPU performance utilities. It serves as a facade that consolidates
//! re-exports from the underlying implementation modules.
//!
//! ## Module Organisation (Issue #239)
//!
//! The actual implementation lives in specialised modules:
//! - `utils/memory.rs` - Platform-specific memory detection (macOS, Linux)
//! - `gpu/device.rs` - GPU performance tier and unified memory detection
//! - `gpu/analyzer.rs` - GPU batch size configuration
//!
//! This facade module provides backward-compatible re-exports so callers can access
//! all system utilities from `crate::analysis::system`.
//!
//! ## Contents
//!
//! ### Memory Detection
//! - `get_memory_info()` - Platform-specific memory detection (macOS `vm_stat`, Linux /proc/meminfo)
//! - `MemoryTier` - Memory classification enum (Low/Standard/High)
//! - `detect_memory_tier()` - Cached memory tier detection
//! - `categorise_memory_tier()` - Pure function for testing
//!
//! ### Parquet Memory Validation
//! - `check_memory_for_parquet()` - Validate memory before loading parquet files
//! - `validate_parquet_memory_requirements()` - Testable pure function
//!
//! ### System Requirements
//! - `check_system_memory_requirements()` - Validate minimum system requirements
//!
//! ### GPU Performance
//! - `GpuPerformanceTier` - GPU performance classification (High/Standard/Unknown)
//! - `detect_gpu_tier()` - Detect GPU tier from adapter info
//! - `detect_unified_memory()` - Detect unified memory architecture (Apple Silicon)
//!
//! ### GPU Batch Size
//! - `DEFAULT_GPU_BATCH_SIZE` - Standard batch size (512)
//! - `HIGH_PERF_GPU_BATCH_SIZE` - High-performance batch size (1024)
//! - `LOW_MEMORY_GPU_BATCH_SIZE` - Low-memory batch size (256)
//! - `cap_gpu_batch_size_by_bytes()` - Cap batch size based on memory constraints
//! - `get_work_queue_capacity()` - Get work queue capacity based on memory tier
//! - `get_work_queue_capacity_for_tier()` - Get capacity for a specific tier
//!
//! ### Platform Setup
//! - `setup_gpu_environment()` - Safe, thread-guarded GPU environment setup
//! - `ensure_xdg_runtime_dir()` - Ensure `XDG_RUNTIME_DIR` is set (Linux, `unsafe`)
//! - `suppress_mesa_warnings_if_requested()` - Suppress Mesa GPU warnings (Linux, `unsafe`)

// =============================================================================
// Memory Detection Re-exports
// =============================================================================

pub use crate::analysis::utils::memory::{
    // GPU batch size constants
    DEFAULT_GPU_BATCH_SIZE,
    HIGH_PERF_GPU_BATCH_SIZE,
    LOW_MEMORY_GPU_BATCH_SIZE,
    MemoryTier,
    // GPU batch size utilities
    cap_gpu_batch_size_by_bytes,
    // Memory tier classification
    categorise_memory_tier,
    // Parquet memory validation
    check_memory_for_parquet,
    // System requirements checking
    check_system_memory_requirements,
    detect_memory_tier,
    // Platform-specific memory detection
    get_memory_info,
    get_work_queue_capacity,
    get_work_queue_capacity_for_tier,
    validate_parquet_memory_requirements,
};

// Platform-specific parsing functions (for testing)
#[cfg(target_os = "macos")]
pub use crate::analysis::utils::memory::{parse_vm_stat_line, parse_vm_stat_page_size};

#[cfg(target_os = "linux")]
pub use crate::analysis::utils::memory::parse_meminfo_line;

// =============================================================================
// Platform Setup Re-exports
// =============================================================================

pub use crate::analysis::utils::platform::{
    GpuEnvSetup, ensure_xdg_runtime_dir, setup_gpu_environment, suppress_mesa_warnings_if_requested,
};

// =============================================================================
// GPU Performance Re-exports
// =============================================================================

pub use crate::analysis::gpu::device::{
    GpuPerformanceTier,
    // GPU performance tier
    detect_gpu_tier,
    // Unified memory detection
    detect_unified_memory,
};

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_tier_re_exports_accessible() {
        // Verify memory tier types are accessible via this facade
        let _tier = MemoryTier::Standard;
        assert!(matches!(
            categorise_memory_tier(10.0 * 1024.0 * 1024.0 * 1024.0),
            MemoryTier::Standard
        ));
    }

    #[test]
    fn test_gpu_performance_tier_re_exports_accessible() {
        // Verify GPU tier types are accessible via this facade
        let _tier = GpuPerformanceTier::High;
        assert_ne!(GpuPerformanceTier::High, GpuPerformanceTier::Standard);
    }

    #[test]
    fn test_batch_size_constants_accessible() {
        // Verify batch size constants are accessible via this facade
        // Use const assertions to validate at compile time
        const _: () = assert!(DEFAULT_GPU_BATCH_SIZE > 0);
        const _: () = assert!(HIGH_PERF_GPU_BATCH_SIZE >= DEFAULT_GPU_BATCH_SIZE);
        const _: () = assert!(LOW_MEMORY_GPU_BATCH_SIZE <= DEFAULT_GPU_BATCH_SIZE);

        // Runtime check to ensure constants are reachable
        let _default = DEFAULT_GPU_BATCH_SIZE;
        let _high = HIGH_PERF_GPU_BATCH_SIZE;
        let _low = LOW_MEMORY_GPU_BATCH_SIZE;
    }

    #[test]
    fn test_work_queue_capacity_accessible() {
        // Verify work queue capacity functions are accessible
        let low = get_work_queue_capacity_for_tier(MemoryTier::Low);
        let high = get_work_queue_capacity_for_tier(MemoryTier::High);
        assert!(high >= low);
    }
}
