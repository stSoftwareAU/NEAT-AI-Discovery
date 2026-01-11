//! GPU infrastructure module
//!
//! This module contains GPU-related code including GpuAnalyzer, GpuWorkQueue,
//! and GPU evaluation functions.

// TODO: Move GPU infrastructure from impl.rs here:
// - GpuAnalyzer struct and implementation
// - GpuWorkQueue struct and implementation
// - GpuWorkRequest enum
// - GpuEvaluator trait
// - GPU evaluation functions (evaluate_relu_gpu, evaluate_activation_gpu, etc.)
// - GPU buffer mapping helpers
// - GPU shader constants

// Temporarily re-export from implementation until refactoring is complete
use super::implementation;

/// GPU analyzer for performing GPU-accelerated analysis operations.
/// Temporarily re-exported from implementation until refactoring is complete.
pub use implementation::GpuAnalyzer;

/// Result of GPU availability check with detailed diagnostics.
/// Temporarily re-exported from implementation until refactoring is complete.
pub use implementation::GpuAvailabilityResult;

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
