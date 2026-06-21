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

/// `ReLU` activation analysis shader.
///
/// Evaluates `ReLU` activation candidates by computing statistics for samples
/// above and below the threshold. Used for add-neuron analysis with `ReLU` activation.
pub const RELU_SHADER: &str = include_str!("../../shaders/relu.wgsl");

/// General activation function analysis shader.
///
/// Evaluates various activation functions (TANH, `HARD_TANH`, LOGISTIC, etc.)
/// for add-neuron candidates. Supports orientation and scale parameters.
pub const ACTIVATION_SHADER: &str = include_str!("../../shaders/activation.wgsl");

/// Bias optimisation shader.
///
/// Evaluates multiple bias candidate values in parallel to find the optimal
/// bias for a new neuron. Uses GPU-accelerated error computation.
pub const BIAS_SHADER: &str = include_str!("../../shaders/bias.wgsl");

/// Helpful contribution reduction shader (Issue #218).
///
/// Performs parallel tree reduction within workgroups to aggregate `HelpfulContribution`
/// data on the GPU. This reduces GPU→CPU data transfer by ~250× for large sample counts.
///
/// For 100K samples: 4.8MB → 18.8KB transfer
pub const HELPFUL_REDUCE_SHADER: &str = include_str!("../../shaders/helpful_reduce.wgsl");

/// Harmful contribution reduction shader (Issue #218).
///
/// Performs parallel tree reduction within workgroups to aggregate `HarmfulContribution`
/// data on the GPU. This reduces GPU→CPU data transfer by ~250× for large sample counts.
///
/// For 100K samples: 1.6MB → 6.3KB transfer
pub const HARMFUL_REDUCE_SHADER: &str = include_str!("../../shaders/harmful_reduce.wgsl");

/// `ReLU` contribution reduction shader (Issue #567).
///
/// Performs parallel tree reduction within workgroups to aggregate `ReluContribution`
/// data on the GPU. This reduces GPU→CPU data transfer by ~255× for large sample counts.
///
/// For 100K samples: 4.0MB → 15.6KB transfer
pub const RELU_REDUCE_SHADER: &str = include_str!("../../shaders/relu_reduce.wgsl");

/// Activation output reduction shader (Issue #567).
///
/// Performs parallel tree reduction within workgroups to aggregate `ActivationOutput`
/// data on the GPU. This reduces GPU→CPU data transfer by ~255× for large sample counts.
///
/// For 100K samples: 2.8MB → 10.9KB transfer
pub const ACTIVATION_REDUCE_SHADER: &str = include_str!("../../shaders/activation_reduce.wgsl");

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

// MIN_NEURON_SAMPLE_COUNT moved to constants.rs (Issue #424)
pub use crate::analysis::constants::MIN_NEURON_SAMPLE_COUNT;

// =============================================================================
// GPU Reduction Configuration (Issue #218)
// =============================================================================

/// Minimum sample count threshold for using GPU workgroup reduction.
///
/// For small sample counts, the overhead of a second shader pass may not be
/// worthwhile. This threshold determines when to use reduction vs direct
/// CPU aggregation.
///
/// ## Analysis
///
/// With workgroup size 256:
/// - Below threshold: Transfer all contributions, reduce on CPU
/// - At/above threshold: Run reduction shader, transfer partial sums only
///
/// The break-even point depends on:
/// - GPU dispatch overhead (~10-50μs per dispatch)
/// - Memory bandwidth (GPU→CPU transfer cost)
/// - CPU reduction cost
///
/// ## Valid Range
///
/// - Minimum: 256 (one workgroup - no benefit from reduction)
/// - Maximum: 50,000 (issue mentions 50K+ as benefiting)
/// - Default: 10,000 (conservative to ensure reduction helps)
///
/// For 10K samples:
/// - Without reduction: 10K × 48 bytes = 480KB transfer
/// - With reduction: 40 workgroups × 48 bytes = 1.9KB transfer
pub const GPU_REDUCTION_THRESHOLD: usize = 10_000;

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Every embedded shader, paired with a human-readable name for assertions.
    const ALL_SHADERS: &[(&str, &str)] = &[
        ("helpful", HELPFUL_SHADER),
        ("harmful", HARMFUL_SHADER),
        ("relu", RELU_SHADER),
        ("activation", ACTIVATION_SHADER),
        ("bias", BIAS_SHADER),
        ("helpful_reduce", HELPFUL_REDUCE_SHADER),
        ("harmful_reduce", HARMFUL_REDUCE_SHADER),
        ("relu_reduce", RELU_REDUCE_SHADER),
        ("activation_reduce", ACTIVATION_REDUCE_SHADER),
    ];

    /// Parse and validate a WGSL shader with naga, returning the compiled
    /// module. Panics with a descriptive message if the shader does not parse
    /// or fails validation — this "does it compile" property is behavioural
    /// (a caller observes a broken shader as a failed pipeline), unlike a
    /// substring grep of the source text.
    fn validate_wgsl(name: &str, source: &str) -> naga::Module {
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|e| panic!("{name} shader is not valid WGSL: {e}"));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{name} shader failed WGSL validation: {e:?}"));
        module
    }

    #[test]
    fn test_shader_sources_are_not_empty() {
        // Verify all shaders have content
        // Note: We check for meaningful content rather than just non-empty,
        // since these are compile-time constants and clippy flags empty checks.
        // The presence of expected WGSL syntax confirms the shaders are loaded.
        assert!(
            HELPFUL_SHADER.contains("@compute"),
            "HELPFUL_SHADER should contain @compute decorator"
        );
        assert!(
            HARMFUL_SHADER.contains("@compute"),
            "HARMFUL_SHADER should contain @compute decorator"
        );
        assert!(
            RELU_SHADER.contains("@compute"),
            "RELU_SHADER should contain @compute decorator"
        );
        assert!(
            ACTIVATION_SHADER.contains("@compute"),
            "ACTIVATION_SHADER should contain @compute decorator"
        );
        assert!(
            BIAS_SHADER.contains("@compute"),
            "BIAS_SHADER should contain @compute decorator"
        );
        assert!(
            HELPFUL_REDUCE_SHADER.contains("@compute"),
            "HELPFUL_REDUCE_SHADER should contain @compute decorator"
        );
        assert!(
            HARMFUL_REDUCE_SHADER.contains("@compute"),
            "HARMFUL_REDUCE_SHADER should contain @compute decorator"
        );
        assert!(
            RELU_REDUCE_SHADER.contains("@compute"),
            "RELU_REDUCE_SHADER should contain @compute decorator"
        );
        assert!(
            ACTIVATION_REDUCE_SHADER.contains("@compute"),
            "ACTIVATION_REDUCE_SHADER should contain @compute decorator"
        );
    }

    #[test]
    fn test_compute_entry_points_declare_workgroup_size() {
        // WHAT-test: every compute entry point must declare the configured
        // workgroup size. Reads the validated module's structured
        // `workgroup_size` instead of grepping for the formatted
        // "@workgroup_size(N)" text, so it is immune to whitespace and
        // formatting changes (e.g. "@workgroup_size( 256 )") yet still fails
        // if a shader genuinely declares the wrong size.
        for (name, shader) in ALL_SHADERS {
            let module = validate_wgsl(name, shader);
            let compute_points: Vec<_> = module
                .entry_points
                .iter()
                .filter(|ep| ep.stage == naga::ShaderStage::Compute)
                .collect();
            assert!(
                !compute_points.is_empty(),
                "{name} shader should expose a @compute entry point"
            );
            for ep in compute_points {
                assert_eq!(
                    ep.workgroup_size,
                    [WORKGROUP_SIZE, 1, 1],
                    "{name} entry point '{}' should declare workgroup size {WORKGROUP_SIZE}",
                    ep.name
                );
            }
        }
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
    fn test_shaders_are_valid_wgsl() {
        // WHAT-test: each shader must parse and validate as WGSL via naga (the
        // same front-end wgpu uses) and expose at least one @compute entry
        // point. Unlike a substring grep for "fn " or "struct", this actually
        // fails when a shader is malformed, and it survives any internal
        // rename, reformat, or rewrite that keeps the shader compilable.
        for (name, shader) in ALL_SHADERS {
            let module = validate_wgsl(name, shader);
            let has_compute = module
                .entry_points
                .iter()
                .any(|ep| ep.stage == naga::ShaderStage::Compute);
            assert!(
                has_compute,
                "{name} shader should expose a @compute entry point"
            );
        }
    }

    #[test]
    fn test_gpu_reduction_threshold_is_valid() {
        // Reduction threshold must be reasonable
        // Using const blocks to satisfy clippy::assertions_on_constants
        const _: () = assert!(
            GPU_REDUCTION_THRESHOLD >= 256,
            "Reduction threshold too low (no benefit below one workgroup)"
        );
        const _: () = assert!(
            GPU_REDUCTION_THRESHOLD <= 100_000,
            "Reduction threshold too high (would miss optimisation opportunities)"
        );
    }

    // NOTE (Issue #1467): the former HOW-tests
    // `test_reduction_shaders_contain_required_functions` and
    // `test_reduction_shaders_use_shared_memory` were removed. They grepped the
    // reduce shaders' source for internal helper names (`add_contributions`,
    // `zero_contribution`, ...) and the `var<workgroup>` memory strategy —
    // implementation details that a behaviour-preserving refactor (renaming a
    // helper, switching reduction strategy) would break for no reason. The
    // observable behaviour they were standing in for — that the reduce shaders
    // produce correct aggregated statistics — is verified end-to-end by
    // `tests/gpu/gpu_workgroup_reduction.rs` (`test_helpful_reduction_correctness`,
    // `test_harmful_reduction_correctness`, etc.), and the reduce shaders'
    // WGSL validity plus workgroup size are now covered by
    // `test_shaders_are_valid_wgsl` and
    // `test_compute_entry_points_declare_workgroup_size` above.
}
