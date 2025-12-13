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
