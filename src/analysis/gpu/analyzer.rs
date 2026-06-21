//! GPU Analyzer core module — `GpuAnalyzer` struct, `GpuEvaluator` trait,
//! initialisation, and shared helper functions.
//!
//! Per-evaluation pipeline builders and batch methods are in sibling modules
//! (Issue #520): `helpful_evaluation`, `harmful_evaluation`, `relu_evaluation`,
//! `activation_evaluation`, `bias_evaluation`.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use anyhow::Result;
use std::time::Duration;

use crate::analysis::gpu::device::{
    GpuAvailabilityResult, GpuPerformanceTier, create_wgpu_instance_safely, detect_gpu_tier,
    detect_unified_memory, get_adapter_info_internal, no_gpu_result, poll_device_until_idle,
};

use crate::analysis::gpu::shaders::GPU_INIT_TIMEOUT_SECS;

use crate::analysis::samples::{HelpfulSample, ReluStats};

use crate::analysis::utils::{
    DEFAULT_GPU_BATCH_SIZE, HIGH_PERF_GPU_BATCH_SIZE, LOW_MEMORY_GPU_BATCH_SIZE, MemoryTier,
    check_system_memory_requirements, detect_memory_tier, ensure_xdg_runtime_dir, get_memory_info,
    suppress_mesa_warnings_if_requested,
};

/// Maximum allocation size (256 MB) for a single GPU batch operation.
/// Caps batch size dynamically to avoid excessive Metal command buffer pressure.
pub const GPU_MAX_BATCH_ALLOC_BYTES: usize = 256 * 1024 * 1024;

/// Central GPU orchestrator for analysis operations.
///
/// `GpuAnalyzer` owns the wgpu device and queue, creates and caches compute pipelines,
/// and implements batch evaluation for all analysis types. It can be used directly or
/// via `GpuWorkQueue` wrapper for concurrent access.
///
/// Pipeline creation is expensive (~100ms per device), so pipelines are built once
/// during construction and reused for all subsequent evaluations.
pub struct GpuAnalyzer {
    pub(super) device: Option<wgpu::Device>,
    pub(super) queue: Option<wgpu::Queue>,
    pub(super) helpful_layout: Option<wgpu::BindGroupLayout>,
    pub(super) helpful_pipeline: Option<wgpu::ComputePipeline>,
    pub(super) harmful_layout: Option<wgpu::BindGroupLayout>,
    pub(super) harmful_pipeline: Option<wgpu::ComputePipeline>,
    pub(super) relu_layout: Option<wgpu::BindGroupLayout>,
    pub(super) relu_pipeline: Option<wgpu::ComputePipeline>,
    pub(super) activation_layout: Option<wgpu::BindGroupLayout>,
    pub(super) activation_pipeline: Option<wgpu::ComputePipeline>,
    pub(super) bias_layout: Option<wgpu::BindGroupLayout>,
    pub(super) bias_pipeline: Option<wgpu::ComputePipeline>,
    /// Reduction pipeline for `HelpfulContribution` aggregation (Issue #218)
    pub(super) helpful_reduce_layout: Option<wgpu::BindGroupLayout>,
    pub(super) helpful_reduce_pipeline: Option<wgpu::ComputePipeline>,
    /// Reduction pipeline for `HarmfulContribution` aggregation (Issue #218)
    pub(super) harmful_reduce_layout: Option<wgpu::BindGroupLayout>,
    pub(super) harmful_reduce_pipeline: Option<wgpu::ComputePipeline>,
    /// Reduction pipeline for `ActivationOutput` aggregation (Issue #567)
    pub(super) activation_reduce_layout: Option<wgpu::BindGroupLayout>,
    pub(super) activation_reduce_pipeline: Option<wgpu::ComputePipeline>,
    /// Optimised GPU batch size based on detected hardware.
    /// Higher values improve GPU utilisation on high-performance hardware.
    pub(super) batch_size: usize,
}

/// Trait for GPU-based evaluation operations.
///
/// This allows helper functions to work with either a direct `GpuAnalyzer`
/// or a shared `GpuWorkQueue` without code duplication.
pub trait GpuEvaluator {
    /// Evaluate `ReLU` activation for neuron candidates.
    /// Returns (`positive_stats`, `negative_stats`, `baseline_error_sq`).
    fn evaluate_relu(
        &self,
        samples: &[HelpfulSample],
        threshold: f32,
    ) -> Result<(ReluStats, ReluStats, f32)>;

    /// Evaluate general activation function for neuron candidates.
    /// Returns (`sum_activation_sq`, `sum_error_activation`, `total_baseline_error_sq`, `improved_count`).
    fn evaluate_activation(
        &self,
        samples: &[HelpfulSample],
        activation_type: u32,
        orientation: f32,
        scale: f32,
    ) -> Result<(f32, f32, f32, u32)>;

    /// Batch-evaluate multiple activation functions in a single GPU call (Issue #201).
    /// Returns Vec of (`sum_activation_sq`, `sum_error_activation`, `total_baseline_error_sq`, `improved_count`).
    fn evaluate_activations_batched(
        &self,
        samples: &[HelpfulSample],
        activation_configs: &[(u32, f32, f32)],
    ) -> Result<Vec<(f32, f32, f32, u32)>>;
}

/// Implementation for direct `GpuAnalyzer` access.
impl GpuEvaluator for GpuAnalyzer {
    fn evaluate_relu(
        &self,
        samples: &[HelpfulSample],
        threshold: f32,
    ) -> Result<(ReluStats, ReluStats, f32)> {
        self.evaluate_relu_gpu(samples, threshold)
    }

    fn evaluate_activation(
        &self,
        samples: &[HelpfulSample],
        activation_type: u32,
        orientation: f32,
        scale: f32,
    ) -> Result<(f32, f32, f32, u32)> {
        self.evaluate_activation_gpu(samples, activation_type, orientation, scale)
    }

    fn evaluate_activations_batched(
        &self,
        samples: &[HelpfulSample],
        activation_configs: &[(u32, f32, f32)],
    ) -> Result<Vec<(f32, f32, f32, u32)>> {
        self.evaluate_activations_batched_gpu(samples, activation_configs)
    }
}

/// Check minimum system requirements. Returns `Some(result)` if NOT met, `None` if OK.
/// On macOS, memory failure is an error; on Linux, graceful disable (Issue #326).
fn check_minimum_system_requirements() -> Option<GpuAvailabilityResult> {
    let (available, total) = get_memory_info();

    // Use the utility function from the memory module
    if let Some(reason) = check_system_memory_requirements(available, total) {
        let available_gb = available as f64 / (1024.0 * 1024.0 * 1024.0);
        let total_gb = total as f64 / (1024.0 * 1024.0 * 1024.0);
        tracing::warn!(
            available_gb = format_args!("{available_gb:.2}"),
            total_gb = format_args!("{total_gb:.1}"),
            "Memory check failed — discovery disabled"
        );

        // On macOS, memory failure should be treated as an error because:
        // 1. macOS always has working GPU (Metal)
        // 2. macOS can quickly reclaim cached memory
        // 3. We want callers to know this is a memory issue, not "no GPU"
        // Issue #326: Prevent misleading "no usable GPU" messages on Mac.
        #[cfg(target_os = "macos")]
        let is_error = true;
        #[cfg(not(target_os = "macos"))]
        let is_error = false; // Graceful disable on Linux (may not have GPU)

        return Some(GpuAvailabilityResult {
            available: false,
            reason: Some(reason),
            is_error,
        });
    }

    // Requirements met
    None
}

/// Adjust batch size based on both GPU tier and memory availability.
fn get_adjusted_batch_size(gpu_tier: GpuPerformanceTier) -> usize {
    // Check for explicit override first
    if let Some(size) = get_batch_size_override() {
        return size;
    }

    let base_size = match gpu_tier {
        GpuPerformanceTier::High => HIGH_PERF_GPU_BATCH_SIZE,
        GpuPerformanceTier::Standard | GpuPerformanceTier::Unknown => DEFAULT_GPU_BATCH_SIZE,
    };

    // Reduce batch size if memory is constrained
    // Standard memory tier should also reduce batch size to prevent Metal command buffer exhaustion
    match detect_memory_tier() {
        MemoryTier::Low => LOW_MEMORY_GPU_BATCH_SIZE.min(base_size),
        MemoryTier::Standard => DEFAULT_GPU_BATCH_SIZE.min(base_size), // Use 512 max, not 1024
        MemoryTier::High => base_size,
    }
}

/// Get cached batch size override from environment variable.
///
/// Delegates to [`crate::config::gpu_batch_size_override()`].
fn get_batch_size_override() -> Option<usize> {
    crate::config::gpu_batch_size_override()
}

/// Get optimised GPU batch size based on detected GPU tier (without memory adjustment).
/// Used by tests. Production code uses `get_adjusted_batch_size()` which also considers memory.
#[cfg(test)]
pub fn get_batch_size_for_tier(tier: GpuPerformanceTier) -> usize {
    // Check for explicit override first (cached)
    if let Some(size) = get_batch_size_override() {
        return size;
    }

    match tier {
        GpuPerformanceTier::High => HIGH_PERF_GPU_BATCH_SIZE,
        GpuPerformanceTier::Standard | GpuPerformanceTier::Unknown => DEFAULT_GPU_BATCH_SIZE,
    }
}

/// Log GPU adapter info once per process for diagnostic purposes.
/// This helps users understand what hardware is being used and the selected batch size.
fn log_gpu_info_once(
    adapter_info: &wgpu::AdapterInfo,
    tier: GpuPerformanceTier,
    batch_size: usize,
) {
    use std::sync::OnceLock;
    static LOGGED: OnceLock<bool> = OnceLock::new();

    LOGGED.get_or_init(|| {
        // Always log GPU info to help diagnose performance issues
        let tier_str = match tier {
            GpuPerformanceTier::High => "high-performance",
            GpuPerformanceTier::Standard => "standard",
            GpuPerformanceTier::Unknown => "unknown",
        };

        let device_type = match adapter_info.device_type {
            wgpu::DeviceType::DiscreteGpu => "discrete",
            wgpu::DeviceType::IntegratedGpu => "integrated",
            wgpu::DeviceType::VirtualGpu => "virtual",
            wgpu::DeviceType::Cpu => "CPU",
            wgpu::DeviceType::Other => "other",
        };

        tracing::info!(
            gpu_name = %adapter_info.name,
            device_type = device_type,
            backend = %format!("{:?}", adapter_info.backend).to_lowercase(),
            tier = tier_str,
            batch_size = batch_size,
            "GPU device initialised"
        );

        // Provide tuning hints at debug level
        tracing::debug!(
            "GPU tuning: set NEAT_AI_DISCOVERY_GPU_BATCH_SIZE=N to override (64-4096) — \
             higher values improve GPU utilisation on powerful hardware"
        );

        true
    });
}

impl GpuAnalyzer {
    /// Cached probe for whether a usable GPU device is available.
    pub fn gpu_is_available() -> bool {
        use std::sync::OnceLock;
        static GPU_AVAILABLE: OnceLock<bool> = OnceLock::new();
        *GPU_AVAILABLE.get_or_init(|| Self::check_gpu_availability().available)
    }

    /// Check GPU availability with detailed diagnostics and error classification.
    pub fn check_gpu_availability() -> GpuAvailabilityResult {
        // Check minimum system requirements FIRST before any GPU operations.
        // This prevents hangs on very old/constrained machines by disabling discovery early.
        if let Some(result) = check_minimum_system_requirements() {
            return result;
        }

        // Suppress Mesa/libEGL warnings if requested (must be called before GPU init)
        suppress_mesa_warnings_if_requested();

        // Set XDG_RUNTIME_DIR if not already set (required by wgpu on Linux/Wayland)
        // Uses Once internally for thread-safe one-time initialisation
        ensure_xdg_runtime_dir();

        // Use safe instance creation to avoid panics from EGL/GL backend probing on Linux
        let Some(instance) = create_wgpu_instance_safely() else {
            return no_gpu_result("wgpu instance creation failed (GPU backend unavailable)");
        };
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }));

        let adapter = match adapter {
            Ok(adapter) => adapter,
            Err(_) => return no_gpu_result("No GPU adapter found"),
        };

        let device_result = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("NEAT-AI Discovery GPU probe device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        }));

        match device_result {
            Ok(_) => GpuAvailabilityResult {
                available: true,
                reason: None,
                is_error: false,
            },
            Err(e) => no_gpu_result(&format!("GPU device creation failed: {e}")),
        }
    }

    /// Check if the GPU supports unified memory (Apple Silicon / integrated GPU).
    pub fn supports_unified_memory() -> bool {
        use std::sync::OnceLock;
        static UNIFIED_MEMORY: OnceLock<bool> = OnceLock::new();

        *UNIFIED_MEMORY.get_or_init(|| {
            let Some(info) = get_adapter_info_internal() else {
                return false;
            };
            detect_unified_memory(&info)
        })
    }

    /// Get GPU adapter info, or `None` if no GPU is available.
    pub fn get_adapter_info() -> Option<crate::analysis::shared::GpuAdapterInfo> {
        use std::sync::OnceLock;
        static ADAPTER_INFO: OnceLock<Option<crate::analysis::shared::GpuAdapterInfo>> =
            OnceLock::new();

        ADAPTER_INFO
            .get_or_init(|| {
                let raw_info = get_adapter_info_internal()?;
                let has_unified_memory = detect_unified_memory(&raw_info);
                let config = crate::analysis::shared::ZeroCopyBufferConfig::from_env();
                let zero_copy_enabled = config.enabled_with_hardware(has_unified_memory);

                Some(crate::analysis::shared::GpuAdapterInfo {
                    name: raw_info.name,
                    device_type: raw_info.device_type.into(),
                    has_unified_memory,
                    zero_copy_enabled,
                })
            })
            .clone()
    }

    /// Check if this `GpuAnalyzer` has a valid GPU device (always `true` after `new()`).
    pub fn has_gpu(&self) -> bool {
        self.device.is_some()
    }

    /// Create a new `GpuAnalyzer` with all pipelines initialised (~100ms; reuse the instance).
    pub fn new() -> Result<Self> {
        // Suppress Mesa/libEGL warnings if requested (must be called before GPU init)
        suppress_mesa_warnings_if_requested();

        // Set XDG_RUNTIME_DIR if not already set (required by wgpu on Linux/Wayland)
        // Uses Once internally for thread-safe one-time initialisation
        ensure_xdg_runtime_dir();

        // Use safe instance creation to avoid panics from EGL/GL backend probing on Linux
        let instance = create_wgpu_instance_safely().ok_or_else(|| {
            anyhow::anyhow!(
                "wgpu instance creation failed (GPU backend unavailable). \
                 Discovery requires GPU acceleration."
            )
        })?;
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }));

        let adapter = match adapter {
            Ok(adapter) => adapter,
            Err(_) => {
                // GPU is required - return an error instead of a CPU-only analyzer.
                // TypeScript calls check_gpu_available() before discovery, but the GPU
                // could become unavailable due to race conditions or resource exhaustion.
                anyhow::bail!(
                    "GPU adapter not available. Discovery requires GPU acceleration. \
                     This may indicate a transient GPU resource issue - consider retrying."
                );
            }
        };

        // Detect GPU tier and system resources for auto-tuning
        let adapter_info = adapter.get_info();
        let gpu_tier = detect_gpu_tier(&adapter_info);
        // Batch size is adjusted based on BOTH GPU tier AND available memory
        let batch_size = get_adjusted_batch_size(gpu_tier);

        // Log GPU info once per process (helps diagnose performance issues)
        log_gpu_info_once(&adapter_info, gpu_tier, batch_size);

        let (device, queue) =
            match pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("NEAT-AI Discovery GPU device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            })) {
                Ok(result) => result,
                Err(e) => {
                    // GPU is required - return an error instead of a CPU-only analyzer.
                    // TypeScript calls check_gpu_available() before discovery, but the GPU
                    // could become unavailable due to race conditions or resource exhaustion.
                    anyhow::bail!(
                        "GPU device creation failed: {e}. Discovery requires GPU acceleration. \
                     This may indicate a transient GPU resource issue - consider retrying."
                    );
                }
            };

        let (helpful_layout, helpful_pipeline) =
            Self::build_helpful_pipeline(&device, "helpful-synapse-pipeline");
        let (harmful_layout, harmful_pipeline) =
            Self::build_harmful_pipeline(&device, "harmful-synapse-pipeline");
        let (relu_layout, relu_pipeline) = Self::build_relu_pipeline(&device, "relu-pipeline");
        let (activation_layout, activation_pipeline) =
            Self::build_activation_pipeline(&device, "activation-pipeline");
        let (bias_layout, bias_pipeline) = Self::build_bias_pipeline(&device, "bias-pipeline");
        // Issue #218: Build reduction pipelines for GPU-side aggregation
        let (helpful_reduce_layout, helpful_reduce_pipeline) =
            Self::build_helpful_reduce_pipeline(&device, "helpful-reduce-pipeline");
        let (harmful_reduce_layout, harmful_reduce_pipeline) =
            Self::build_harmful_reduce_pipeline(&device, "harmful-reduce-pipeline");
        // Issue #567: Build activation reduction pipeline
        let (activation_reduce_layout, activation_reduce_pipeline) =
            Self::build_activation_reduce_pipeline(&device, "activation-reduce-pipeline");

        // CRITICAL: Warm up the GPU by polling to ensure all pipeline creation work is complete.
        //
        // We intentionally avoid `Maintain::Wait` here because it can block forever if the
        // GPU driver wedges on a specific machine. If this doesn't settle quickly, treat it
        // as a GPU initialisation failure and return an error so callers can restart/skip.
        poll_device_until_idle(
            &device,
            Duration::from_secs(GPU_INIT_TIMEOUT_SECS),
            "GPU warm-up after pipeline creation",
        )?;

        Ok(Self {
            device: Some(device),
            queue: Some(queue),
            helpful_layout: Some(helpful_layout),
            helpful_pipeline: Some(helpful_pipeline),
            harmful_layout: Some(harmful_layout),
            harmful_pipeline: Some(harmful_pipeline),
            relu_layout: Some(relu_layout),
            relu_pipeline: Some(relu_pipeline),
            activation_layout: Some(activation_layout),
            activation_pipeline: Some(activation_pipeline),
            bias_layout: Some(bias_layout),
            bias_pipeline: Some(bias_pipeline),
            helpful_reduce_layout: Some(helpful_reduce_layout),
            helpful_reduce_pipeline: Some(helpful_reduce_pipeline),
            harmful_reduce_layout: Some(harmful_reduce_layout),
            harmful_reduce_pipeline: Some(harmful_reduce_pipeline),
            activation_reduce_layout: Some(activation_reduce_layout),
            activation_reduce_pipeline: Some(activation_reduce_pipeline),
            batch_size,
        })
    }

    /// Create a new `GpuAnalyzer` with all pipelines initialised, overriding
    /// the auto-detected batch size (Issue #1083).
    ///
    /// Used by the GPU recovery loop to re-initialise with a reduced batch size
    /// after memory exhaustion.
    pub fn new_with_batch_size(batch_size_override: usize) -> Result<Self> {
        let mut analyzer = Self::new()?;
        analyzer.batch_size = batch_size_override;
        Ok(analyzer)
    }

    /// Return the current effective batch size.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tautological constant-pin tests removed (Issue #1469):
    // `test_workgroup_size_constant`, `test_min_neuron_sample_count` and
    // `test_gpu_max_batch_alloc_bytes` only re-asserted each constant's own
    // literal — they could never catch a bug, only flag a deliberate retune.
    // The WORKGROUP_SIZE invariants (power-of-two, within hardware limits) are
    // already guarded at compile time in `shaders.rs`; the batch-size tiering
    // behaviour is covered by `test_batch_size_for_tier` below.

    #[test]
    fn test_batch_size_for_tier() {
        let high = get_batch_size_for_tier(GpuPerformanceTier::High);
        let standard = get_batch_size_for_tier(GpuPerformanceTier::Standard);
        let unknown = get_batch_size_for_tier(GpuPerformanceTier::Unknown);

        assert!(high >= standard);
        assert_eq!(standard, unknown);
    }
}
