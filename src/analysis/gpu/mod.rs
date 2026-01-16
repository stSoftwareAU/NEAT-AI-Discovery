//! GPU infrastructure module
//!
//! This module contains GPU-related code including device management, GpuAnalyzer,
//! GpuWorkQueue, and GPU evaluation functions.
//!
//! ## Module Structure (Issue #272, #273, #274)
//!
//! ```text
//! src/analysis/gpu/
//! ├── mod.rs          <- This file: module router and re-exports
//! ├── device.rs       <- GPU device management (Issue #272)
//! ├── analyzer.rs     <- GpuAnalyzer struct and GpuEvaluator trait (Issue #273)
//! ├── queue.rs        <- GpuWorkQueue struct and thread management (Issue #274)
//! └── pipelines.rs    <- Pipeline builders (future)
//! ```
//!
//! ## Refactoring Progress
//!
//! - [x] device.rs - GPU device initialisation, detection, buffer management (Issue #272)
//! - [x] analyzer.rs - GpuAnalyzer struct, GpuEvaluator trait, pipeline builders (Issue #273)
//! - [x] queue.rs - GpuWorkQueue struct and implementation (Issue #274)
//! - [ ] pipelines.rs - Shared pipeline builder utilities

pub mod analyzer;
pub mod device;
pub mod queue;

// Re-export device module contents for backwards compatibility
pub use device::{
    create_wgpu_instance_safely, detect_gpu_tier, detect_unified_memory, get_adapter_info_internal,
    no_gpu_result, poll_device_until_idle, wait_for_buffer_map, wait_for_buffer_maps_batch,
    GpuAvailabilityResult, GpuPerformanceTier, GPU_BUFFER_MAP_TIMEOUT_MARGIN_SECS,
    GPU_BUFFER_MAP_TIMEOUT_SECS, GPU_INIT_TIMEOUT_SECS,
};

// Re-export GPU_QUEUE_TIMEOUT_MAX_SECS from device (which gets it from utils)
pub use device::GPU_QUEUE_TIMEOUT_MAX_SECS;

// Re-export analyzer module contents
pub use analyzer::{GpuAnalyzer, GpuEvaluator, GPU_MAX_BATCH_ALLOC_BYTES};

// Re-export queue module contents (Issue #274)
pub use queue::GpuWorkQueue;

// Re-export test helper function for batch size tests
#[cfg(test)]
pub use analyzer::get_batch_size_for_tier;

/// Check if the current GPU supports unified memory architecture.
///
/// On unified memory systems (Apple Silicon), CPU and GPU share the same
/// physical memory, enabling zero-copy buffer sharing.
///
/// Returns `true` for:
/// - Apple Silicon Macs (M1/M2/M3/M4)
///
/// Returns `false` for:
/// - Discrete GPUs (separate VRAM)
/// - No GPU available
///
/// This is a convenience wrapper around `GpuAnalyzer::supports_unified_memory()`.
pub fn supports_unified_memory() -> bool {
    GpuAnalyzer::supports_unified_memory()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports_are_accessible() {
        // Verify that all re-exported types are accessible
        let _tier = GpuPerformanceTier::Standard;
        let _result = GpuAvailabilityResult {
            available: false,
            reason: None,
            is_error: false,
        };

        // Verify constants are accessible using const assertions
        const _: () = assert!(GPU_BUFFER_MAP_TIMEOUT_SECS > 0);
        const _: () = assert!(GPU_INIT_TIMEOUT_SECS > 0);
    }

    #[test]
    fn test_gpu_analyzer_is_exported() {
        // Verify GpuAnalyzer type is accessible via the gpu module
        // We can't create an instance without GPU, but we can reference the type
        fn _takes_analyzer(_: &GpuAnalyzer) {}

        // Verify GpuEvaluator trait is accessible
        fn _takes_evaluator<T: GpuEvaluator>(_: &T) {}
    }
}
