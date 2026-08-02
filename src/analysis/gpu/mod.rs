//! GPU infrastructure module
//!
//! This module contains GPU-related code including device management, `GpuAnalyzer`,
//! `GpuWorkQueue`, and GPU evaluation functions.
//!
//! ## Module Structure (Issue #272, #273, #274, #277, #520)
//!
//! ```text
//! src/analysis/gpu/
//! ├── mod.rs                    <- This file: module router and re-exports
//! ├── shaders.rs                <- GPU shader constants and references (Issue #277)
//! ├── device.rs                 <- GPU device management (Issue #272)
//! ├── analyzer.rs               <- Core GpuAnalyzer struct, initialisation, shared logic
//! ├── helpful_evaluation.rs     <- Helpful synapse GPU evaluation (Issue #520)
//! ├── harmful_evaluation.rs     <- Harmful synapse GPU evaluation (Issue #520)
//! ├── relu_evaluation.rs        <- ReLU activation GPU evaluation (Issue #520)
//! ├── activation_evaluation.rs  <- Activation function GPU evaluation (Issue #520)
//! ├── bias_evaluation.rs        <- Bias GPU evaluation (Issue #520)
//! ├── budget.rs                 <- Per-request GPU time budget (Issue #1928)
//! ├── breaker.rs                <- Process-wide GPU circuit breaker (Issue #1930)
//! ├── pipeline_builder.rs      <- Shared compute pipeline builder (Issue #978)
//! ├── queue/                    <- GPU work queue (Issue #274, #608)
//! │   ├── mod.rs                <- Public API, re-exports, queue types
//! │   ├── submission.rs         <- Work item submission and batching
//! │   ├── execution.rs          <- GPU execution and result collection
//! │   └── scheduling.rs         <- Work scheduling and prioritisation
//! ```
//!
//! ## Refactoring Progress
//!
//! - [x] shaders.rs - GPU shader constants and references (Issue #277)
//! - [x] device.rs - GPU device initialisation, detection, buffer management (Issue #272)
//! - [x] analyzer.rs - `GpuAnalyzer` struct, `GpuEvaluator` trait (Issue #273)
//! - [x] queue/ - `GpuWorkQueue` sub-modules (Issue #274, #608)
//! - [x] Per-evaluation modules split from analyzer.rs (Issue #520)

pub mod activation_evaluation;
pub mod analyzer;
pub mod bias_evaluation;
pub mod breaker;
pub mod budget;
pub mod device;
pub mod harmful_evaluation;
pub mod helpful_evaluation;
pub(crate) mod pipeline_builder;
pub mod queue;
pub mod relu_evaluation;
pub mod shaders;

// Re-export device module contents for backwards compatibility
pub use device::{
    GPU_BUFFER_MAP_TIMEOUT_MARGIN_SECS, GPU_BUFFER_MAP_TIMEOUT_SECS, GPU_INIT_TIMEOUT_SECS,
    GpuAvailabilityResult, GpuPerformanceTier, create_wgpu_instance_safely, detect_gpu_tier,
    detect_unified_memory, get_adapter_info_internal, no_gpu_result, poll_device_until_idle,
    wait_for_buffer_map, wait_for_buffer_maps_batch,
};

// Re-export GPU_QUEUE_TIMEOUT_MAX_SECS from device (which gets it from utils)
pub use device::GPU_QUEUE_TIMEOUT_MAX_SECS;

// Re-export the per-request GPU time budget (Issue #1928)
pub use budget::GpuTimeBudget;

// Re-export the process-wide GPU circuit breaker (Issue #1930)
pub use breaker::{
    GpuCircuitBreaker, GpuTripReason, abandoned_gpu_thread_count, check_gpu_breaker,
    global_gpu_breaker, gpu_breaker_trip_reason, is_gpu_breaker_tripped,
    record_abandoned_gpu_thread, reset_gpu_breaker, trip_gpu_breaker,
};

// Re-export analyzer module contents
pub use analyzer::{GPU_MAX_BATCH_ALLOC_BYTES, GpuAnalyzer, GpuEvaluator};

// Re-export queue module contents (Issue #274, #647)
pub use queue::GpuWorkQueue;
pub use queue::recovery::{
    DEFAULT_BACKOFF_INITIAL_MS, DEFAULT_BACKOFF_MAX_MS, DEFAULT_GPU_RETRY_LIMIT,
    GPU_RETRY_LIMIT_ENV, MINIMUM_GPU_BATCH_SIZE, backoff_delay_ms, get_gpu_retry_limit,
    is_device_lost_error, is_memory_exhaustion_error,
};

// Re-export shader module contents (Issue #277)
pub use shaders::{
    ACTIVATION_REDUCE_SHADER, ACTIVATION_SHADER, BIAS_SHADER,
    GPU_INIT_TIMEOUT_SECS as SHADER_GPU_INIT_TIMEOUT_SECS, GPU_SHUTDOWN_TIMEOUT_SECS,
    HARMFUL_SHADER, HELPFUL_SHADER, MIN_NEURON_SAMPLE_COUNT, RELU_SHADER, WORKGROUP_SIZE,
};

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

    #[test]
    fn test_shader_constants_are_exported() {
        // Verify shader constants are accessible via the gpu module (Issue #277)
        // Note: We check for meaningful content rather than just non-empty,
        // since these are compile-time constants and clippy flags empty checks.
        assert!(HELPFUL_SHADER.contains("@compute"));
        assert!(HARMFUL_SHADER.contains("@compute"));
        assert!(RELU_SHADER.contains("@compute"));
        assert!(ACTIVATION_SHADER.contains("@compute"));
        assert!(BIAS_SHADER.contains("@compute"));

        // Verify workgroup size matches shaders
        assert_eq!(WORKGROUP_SIZE, 256);

        // Verify timing constants are accessible
        const _: () = assert!(SHADER_GPU_INIT_TIMEOUT_SECS > 0);
        const _: () = assert!(GPU_SHUTDOWN_TIMEOUT_SECS > 0);

        // Verify sample count constant
        assert_eq!(MIN_NEURON_SAMPLE_COUNT, 10);
    }
}
