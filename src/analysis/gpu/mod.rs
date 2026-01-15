//! GPU infrastructure module
//!
//! This module contains GPU-related code including device management, GpuAnalyzer,
//! GpuWorkQueue, and GPU evaluation functions.
//!
//! ## Module Structure (Issue #272)
//!
//! ```text
//! src/analysis/gpu/
//! ├── mod.rs          <- This file: module router and re-exports
//! ├── device.rs       <- GPU device management (Issue #272)
//! ├── analyzer.rs     <- GpuAnalyzer struct (future)
//! ├── queue.rs        <- GpuWorkQueue (future)
//! ├── evaluator.rs    <- GpuEvaluator trait (future)
//! └── pipelines.rs    <- Pipeline builders (future)
//! ```
//!
//! ## Refactoring Progress
//!
//! - [x] device.rs - GPU device initialisation, detection, buffer management (Issue #272)
//! - [ ] analyzer.rs - GpuAnalyzer struct and implementation
//! - [ ] queue.rs - GpuWorkQueue struct and implementation
//! - [ ] evaluator.rs - GpuEvaluator trait
//! - [ ] pipelines.rs - Pipeline builders

pub mod device;

// Re-export device module contents for backwards compatibility
pub use device::{
    create_wgpu_instance_safely, detect_gpu_tier, detect_unified_memory, get_adapter_info_internal,
    no_gpu_result, poll_device_until_idle, wait_for_buffer_map, wait_for_buffer_maps_batch,
    GpuAvailabilityResult, GpuPerformanceTier, GPU_BUFFER_MAP_TIMEOUT_MARGIN_SECS,
    GPU_BUFFER_MAP_TIMEOUT_SECS, GPU_INIT_TIMEOUT_SECS,
};

// Re-export GPU_QUEUE_TIMEOUT_MAX_SECS from device (which gets it from utils)
pub use device::GPU_QUEUE_TIMEOUT_MAX_SECS;

// Temporarily re-export from implementation until full refactoring is complete
use super::implementation;

/// GPU analyzer for performing GPU-accelerated analysis operations.
/// Temporarily re-exported from implementation until refactoring is complete.
pub use implementation::GpuAnalyzer;

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
}
