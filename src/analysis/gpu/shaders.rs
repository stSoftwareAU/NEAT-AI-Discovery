//! GPU Shader Constants and References
//!
//! This module centralises all GPU shader code and related constants used by the
//! NEAT-AI-Discovery library. By grouping these constants together, we:
//!
//! 1. **Centralise GPU configuration** - All shader-related constants in one place
//! 2. **Simplify threshold tuning** - Easy to find and modify values
//! 3. **Document relationships** - Constants are documented with their purpose
//! 4. **Prepare for extensibility** - Future shader hot-reloading could be added here
//!
//! ## Module Structure
//!
//! ```text
//! src/analysis/gpu/shaders.rs
//! ├── Shader source references (include_str! macros)
//! ├── Workgroup configuration constants
//! └── GPU timing constants
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use crate::analysis::gpu::shaders::{HELPFUL_SHADER, WORKGROUP_SIZE};
//!
//! // Create shader module
//! let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
//!     label: Some("helpful"),
//!     source: wgpu::ShaderSource::Wgsl(HELPFUL_SHADER.into()),
//! });
//!
//! // Dispatch workgroups
//! let workgroups = (sample_count as u32).div_ceil(WORKGROUP_SIZE);
//! ```

// =============================================================================
// Shader Source References
// =============================================================================

/// Helpful synapse analysis shader.
///
/// Computes statistics for potential new synapses that would reduce target neuron error.
/// Uses parallel reduction to compute sum of (error × activation) and sum of (activation²).
pub const HELPFUL_SHADER: &str = include_str!("../../shaders/helpful.wgsl");

/// Harmful synapse analysis shader.
///
/// Computes statistics for existing synapses that may be increasing error.
/// Identifies synapses that could be candidates for removal or weight adjustment.
pub const HARMFUL_SHADER: &str = include_str!("../../shaders/harmful.wgsl");

/// ReLU activation analysis shader.
///
/// Evaluates ReLU activation candidates by computing statistics for samples
/// above and below the threshold. Used for add-neuron analysis with ReLU activation.
pub const RELU_SHADER: &str = include_str!("../../shaders/relu.wgsl");

/// General activation function analysis shader.
///
/// Evaluates various activation functions (TANH, HARD_TANH, LOGISTIC, etc.)
/// for add-neuron candidates. Supports orientation and scale parameters.
pub const ACTIVATION_SHADER: &str = include_str!("../../shaders/activation.wgsl");

/// Bias optimisation shader.
///
/// Evaluates multiple bias candidate values in parallel to find the optimal
/// bias for a new neuron. Uses GPU-accelerated error computation.
pub const BIAS_SHADER: &str = include_str!("../../shaders/bias.wgsl");

// =============================================================================
// Workgroup Configuration
// =============================================================================

/// Workgroup size for GPU compute shaders.
///
/// This value **must match** the `@workgroup_size` declarations in the WGSL shader files.
/// Currently all shaders use `@workgroup_size(256)`.
///
/// ## Why 256?
///
/// - **Apple Silicon optimal**: Divisible by SIMD width (32), good occupancy
/// - **Efficient scheduling**: Allows wavefront scheduling on M1/M2/M3/M4 GPUs
/// - **Power of 2**: Simplifies workgroup count calculations
/// - **Memory alignment**: Aligns well with typical buffer sizes
///
/// ## Valid Range
///
/// - Minimum: 64 (too small wastes GPU parallelism)
/// - Maximum: 1024 (limited by GPU hardware)
/// - Recommended: 256 (balanced for Apple Silicon and discrete GPUs)
pub const WORKGROUP_SIZE: u32 = 256;

// =============================================================================
// GPU Timing Constants
// =============================================================================

/// Timeout in seconds for GPU device initialisation.
///
/// GPU initialisation can be slow on some systems, especially when:
/// - Driver needs to compile shaders
/// - System is under memory pressure
/// - Multiple GPUs are being probed
///
/// ## Valid Range
///
/// - Minimum: 5 (too short may cause false timeouts)
/// - Maximum: 60 (too long delays error detection)
/// - Default: 30 (reasonable for most systems)
///
/// Note: This constant is also available from `gpu/device.rs` for backwards
/// compatibility. Both locations reference the same value.
pub const GPU_INIT_TIMEOUT_SECS: u64 = 30;

/// Timeout in seconds for graceful GPU thread shutdown.
///
/// When the `GpuWorkQueue` is dropped, it signals the GPU thread to exit and
/// waits for it to complete. This timeout prevents indefinite hangs if the
/// GPU thread is stuck.
///
/// ## Valid Range
///
/// - Minimum: 5 (allow time for cleanup)
/// - Maximum: 30 (don't block process exit too long)
/// - Default: 10 (balance between cleanup and responsiveness)
pub const GPU_SHUTDOWN_TIMEOUT_SECS: u64 = 10;

// =============================================================================
// Analysis Threshold Constants
// =============================================================================

/// Minimum number of samples required for valid neuron analysis.
///
/// Statistical analysis requires sufficient samples to produce reliable results.
/// With fewer samples, variance estimates become unreliable and candidates
/// may be based on noise rather than signal.
///
/// ## Valid Range
///
/// - Minimum: 5 (absolute minimum for any statistics)
/// - Maximum: 100 (too high excludes valid neurons)
/// - Default: 10 (balance between reliability and coverage)
///
/// ## Usage
///
/// This constant is used in:
/// - GPU shader validation (bias shader minimum sample count)
/// - Synapse candidate filtering
/// - Add-neuron candidate filtering
pub const MIN_NEURON_SAMPLE_COUNT: usize = 10;

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shader_sources_are_not_empty() {
        // Verify all shaders have content
        assert!(
            !HELPFUL_SHADER.is_empty(),
            "HELPFUL_SHADER should not be empty"
        );
        assert!(
            !HARMFUL_SHADER.is_empty(),
            "HARMFUL_SHADER should not be empty"
        );
        assert!(!RELU_SHADER.is_empty(), "RELU_SHADER should not be empty");
        assert!(
            !ACTIVATION_SHADER.is_empty(),
            "ACTIVATION_SHADER should not be empty"
        );
        assert!(!BIAS_SHADER.is_empty(), "BIAS_SHADER should not be empty");
    }

    #[test]
    fn test_shader_sources_contain_workgroup_size() {
        // Verify shaders declare the correct workgroup size
        let expected_workgroup = format!("@workgroup_size({WORKGROUP_SIZE})");

        assert!(
            HELPFUL_SHADER.contains(&expected_workgroup),
            "HELPFUL_SHADER should declare @workgroup_size({WORKGROUP_SIZE})"
        );
        assert!(
            HARMFUL_SHADER.contains(&expected_workgroup),
            "HARMFUL_SHADER should declare @workgroup_size({WORKGROUP_SIZE})"
        );
        assert!(
            RELU_SHADER.contains(&expected_workgroup),
            "RELU_SHADER should declare @workgroup_size({WORKGROUP_SIZE})"
        );
        assert!(
            ACTIVATION_SHADER.contains(&expected_workgroup),
            "ACTIVATION_SHADER should declare @workgroup_size({WORKGROUP_SIZE})"
        );
        assert!(
            BIAS_SHADER.contains(&expected_workgroup),
            "BIAS_SHADER should declare @workgroup_size({WORKGROUP_SIZE})"
        );
    }

    #[test]
    fn test_workgroup_size_is_valid() {
        // Workgroup size must be a power of 2 between 64 and 1024
        // Using const blocks to satisfy clippy::assertions_on_constants
        const _: () = assert!(WORKGROUP_SIZE >= 64, "Workgroup size too small");
        const _: () = assert!(WORKGROUP_SIZE <= 1024, "Workgroup size too large");
        const _: () = assert!(
            WORKGROUP_SIZE.is_power_of_two(),
            "Workgroup size should be a power of 2"
        );
    }

    #[test]
    fn test_gpu_timeouts_are_valid() {
        // Compile-time assertions for timeout validity
        const _: () = assert!(GPU_INIT_TIMEOUT_SECS >= 5, "Init timeout too short");
        const _: () = assert!(GPU_INIT_TIMEOUT_SECS <= 60, "Init timeout too long");
        const _: () = assert!(GPU_SHUTDOWN_TIMEOUT_SECS >= 5, "Shutdown timeout too short");
        const _: () = assert!(GPU_SHUTDOWN_TIMEOUT_SECS <= 30, "Shutdown timeout too long");
    }

    #[test]
    fn test_min_neuron_sample_count_is_valid() {
        // Minimum sample count must be reasonable
        // Using const blocks to satisfy clippy::assertions_on_constants
        const _: () = assert!(
            MIN_NEURON_SAMPLE_COUNT >= 5,
            "Minimum sample count too low for reliable statistics"
        );
        const _: () = assert!(
            MIN_NEURON_SAMPLE_COUNT <= 100,
            "Minimum sample count too high, would exclude valid neurons"
        );
    }

    #[test]
    fn test_shader_wgsl_syntax_basics() {
        // Basic syntax verification - shaders should have struct and fn declarations
        for (name, shader) in [
            ("helpful", HELPFUL_SHADER),
            ("harmful", HARMFUL_SHADER),
            ("relu", RELU_SHADER),
            ("activation", ACTIVATION_SHADER),
            ("bias", BIAS_SHADER),
        ] {
            assert!(
                shader.contains("struct") || shader.contains("fn "),
                "{name} shader should contain struct or fn declarations"
            );
            assert!(
                shader.contains("@compute"),
                "{name} shader should contain @compute decorator"
            );
        }
    }
}
