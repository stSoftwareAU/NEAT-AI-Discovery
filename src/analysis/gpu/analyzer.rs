//! GPU Analyzer module
//!
//! This module contains the `GpuAnalyzer` struct and `GpuEvaluator` trait extracted
//! from the monolithic implementation.rs file (Issue #273).
//!
//! ## Contents
//!
//! - `GpuAnalyzer` - Central GPU orchestrator for analysis operations
//! - `GpuEvaluator` - Trait abstracting GPU evaluation operations
//! - Pipeline builders for each evaluation type (helpful, harmful, ReLU, activation, bias)
//! - Batch evaluation methods for GPU-accelerated analysis
//!
//! ## Dependencies
//!
//! **Incoming dependencies (modules that use analyzer):**
//! - GPU work queue (wraps analyzer)
//! - Synapse analysis (helpful/harmful evaluation)
//! - Neuron analysis (ReLU/activation evaluation)
//!
//! **Outgoing dependencies (analyzer needs these):**
//! - GPU device management (`gpu/device.rs`)
//! - Sample structures (`samples.rs`)
//! - Shader constants
//! - wgpu crate

use anyhow::{Context, Result};
use bytemuck::Zeroable;
use std::sync::mpsc;
use std::time::Duration;
use wgpu::util::DeviceExt;

// Import device management functions
use crate::analysis::gpu::device::{
    GPU_BUFFER_MAP_TIMEOUT_SECS, GpuAvailabilityResult, GpuPerformanceTier,
    create_wgpu_instance_safely, detect_gpu_tier, detect_unified_memory, get_adapter_info_internal,
    no_gpu_result, poll_device_until_idle, wait_for_buffer_map, wait_for_buffer_maps_batch,
};

// Import shader constants (Issue #277)
use crate::analysis::gpu::shaders::{
    ACTIVATION_SHADER, BIAS_SHADER, GPU_INIT_TIMEOUT_SECS, GPU_REDUCTION_THRESHOLD,
    HARMFUL_REDUCE_SHADER, HARMFUL_SHADER, HELPFUL_REDUCE_SHADER, HELPFUL_SHADER,
    MIN_NEURON_SAMPLE_COUNT, RELU_SHADER, WORKGROUP_SIZE,
};

// Import sample data structures
use crate::analysis::samples::{
    ActivationOutput, ActivationUniforms, BiasResult, BiasUniforms, EPSILON, GpuHelpfulSample,
    HarmfulContribution, HarmfulStats, HarmfulUniforms, HelpfulContribution, HelpfulSample,
    HelpfulStats, HelpfulUniforms, ReductionUniforms, ReluContribution, ReluOrientation, ReluStats,
    ReluUniforms,
};

// Import utility functions
use crate::analysis::utils::{
    DEFAULT_GPU_BATCH_SIZE, HIGH_PERF_GPU_BATCH_SIZE, LOW_MEMORY_GPU_BATCH_SIZE, MemoryTier,
    cap_gpu_batch_size_by_bytes, check_system_memory_requirements, detect_memory_tier,
    ensure_xdg_runtime_dir, get_memory_info, suppress_mesa_warnings_if_requested, verbose_enabled,
};

// =============================================================================
// Constants
// =============================================================================

// Note: WORKGROUP_SIZE, MIN_NEURON_SAMPLE_COUNT, and shader sources have been moved
// to gpu/shaders.rs (Issue #277) for centralised GPU configuration.

/// Maximum allocation size for a single GPU batch operation.
///
/// When a single operation requires very large buffers (e.g., samples in the thousands),
/// we dynamically cap the batch size to avoid allocating hundreds of MB in a single chunk.
/// This improves Metal command buffer pooling and prevents memory pressure.
///
/// Empirical values: Each operation creates ~3 buffers:
/// - Sample buffer (GpuHelpfulSample)
/// - Contribution buffer (HelpfulContribution / HarmfulContribution)
/// - Staging buffer (same size as contribution buffer)
///
/// This cap is conservative by design; it trades a bit of peak throughput for stability on
/// Apple Silicon (and makes time-bounded runs far more reliable).
pub const GPU_MAX_BATCH_ALLOC_BYTES: usize = 256 * 1024 * 1024; // 256MB

// =============================================================================
// GpuAnalyzer Struct
// =============================================================================

/// Central GPU orchestrator for analysis operations.
///
/// `GpuAnalyzer` owns the wgpu device and queue, creates and caches compute pipelines,
/// and implements batch evaluation for all analysis types. It can be used directly or
/// via `GpuWorkQueue` wrapper for concurrent access.
///
/// Pipeline creation is expensive (~100ms per device), so pipelines are built once
/// during construction and reused for all subsequent evaluations.
pub struct GpuAnalyzer {
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
    helpful_layout: Option<wgpu::BindGroupLayout>,
    helpful_pipeline: Option<wgpu::ComputePipeline>,
    harmful_layout: Option<wgpu::BindGroupLayout>,
    harmful_pipeline: Option<wgpu::ComputePipeline>,
    relu_layout: Option<wgpu::BindGroupLayout>,
    relu_pipeline: Option<wgpu::ComputePipeline>,
    activation_layout: Option<wgpu::BindGroupLayout>,
    activation_pipeline: Option<wgpu::ComputePipeline>,
    bias_layout: Option<wgpu::BindGroupLayout>,
    bias_pipeline: Option<wgpu::ComputePipeline>,
    /// Reduction pipeline for HelpfulContribution aggregation (Issue #218)
    helpful_reduce_layout: Option<wgpu::BindGroupLayout>,
    helpful_reduce_pipeline: Option<wgpu::ComputePipeline>,
    /// Reduction pipeline for HarmfulContribution aggregation (Issue #218)
    harmful_reduce_layout: Option<wgpu::BindGroupLayout>,
    harmful_reduce_pipeline: Option<wgpu::ComputePipeline>,
    /// Optimised GPU batch size based on detected hardware.
    /// Higher values improve GPU utilisation on high-performance hardware.
    batch_size: usize,
}

// =============================================================================
// GpuEvaluator Trait
// =============================================================================

/// Trait for GPU-based evaluation operations.
///
/// This allows helper functions to work with either a direct `GpuAnalyzer`
/// or a shared `GpuWorkQueue` without code duplication.
pub trait GpuEvaluator {
    /// Evaluate ReLU activation for neuron candidates.
    /// Returns (positive_stats, negative_stats, baseline_error_sq).
    fn evaluate_relu(
        &self,
        samples: &[HelpfulSample],
        threshold: f32,
    ) -> Result<(ReluStats, ReluStats, f32)>;

    /// Evaluate general activation function for neuron candidates.
    /// Returns (sum_activation_sq, sum_error_activation, total_baseline_error_sq, improved_count).
    fn evaluate_activation(
        &self,
        samples: &[HelpfulSample],
        activation_type: u32,
        orientation: f32,
        scale: f32,
    ) -> Result<(f32, f32, f32, u32)>;

    /// Batch-evaluate multiple activation functions in a single GPU call.
    ///
    /// Issue #201: Reduces GPU round-trips by evaluating multiple activation
    /// function configurations in one command buffer submission instead of
    /// separate calls per activation type.
    ///
    /// # Arguments
    /// * `samples` - The sample data to evaluate
    /// * `activation_configs` - List of (activation_type, orientation, scale) tuples
    ///
    /// # Returns
    /// Vector of (sum_activation_sq, sum_error_activation, total_baseline_error_sq, improved_count)
    /// in the same order as the input configs.
    ///
    /// # Performance
    /// - 10-20% fewer GPU round-trips during neuron analysis
    /// - Reduced CPU-GPU synchronisation overhead
    /// - Better GPU utilisation (larger, fewer batches)
    fn evaluate_activations_batched(
        &self,
        samples: &[HelpfulSample],
        activation_configs: &[(u32, f32, f32)],
    ) -> Result<Vec<(f32, f32, f32, u32)>>;
}

/// Implementation for direct GpuAnalyzer access.
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

// =============================================================================
// Helper Functions
// =============================================================================

/// Check if the system meets minimum requirements for GPU discovery.
///
/// Very old machines with insufficient memory cannot reliably run GPU operations
/// without risking hangs from memory pressure. This check runs before GPU
/// initialisation to allow discovery to be disabled gracefully.
///
/// # Platform Differences
///
/// - **macOS**: Memory check failure is treated as an error (`is_error: true`) because
///   macOS should always have working GPU (Metal) and sufficient reclaimable memory.
///   This ensures callers don't just report "no GPU" when the issue is memory.
/// - **Linux**: Memory check failure is graceful (`is_error: false`) because
///   headless servers may genuinely lack GPU hardware.
///
/// Issue #326: On macOS, memory check failure was misleadingly reported as "no usable GPU".
///
/// Returns `Some(GpuAvailabilityResult)` if requirements are NOT met (discovery disabled).
/// Returns `None` if requirements ARE met (continue with GPU check).
fn check_minimum_system_requirements() -> Option<GpuAvailabilityResult> {
    let (available, total) = get_memory_info();

    // Use the utility function from the memory module
    if let Some(reason) = check_system_memory_requirements(available, total) {
        let available_gb = available as f64 / (1024.0 * 1024.0 * 1024.0);
        let total_gb = total as f64 / (1024.0 * 1024.0 * 1024.0);
        eprintln!(
            "[NEAT-AI-Discovery] Memory check failed: {available_gb:.2}GB available / {total_gb:.1}GB total. Discovery disabled."
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
/// Returns None if not set or invalid.
fn get_batch_size_override() -> Option<usize> {
    use std::sync::OnceLock;
    static OVERRIDE: OnceLock<Option<usize>> = OnceLock::new();
    *OVERRIDE.get_or_init(|| {
        std::env::var("NEAT_AI_DISCOVERY_GPU_BATCH_SIZE")
            .ok()
            .and_then(|val| val.parse::<usize>().ok())
            .filter(|size| (64..=4096).contains(size))
    })
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

        eprintln!(
            "[NEAT-AI-Discovery] GPU: {} ({} {}) | Tier: {} | Batch size: {}",
            adapter_info.name,
            device_type,
            format!("{:?}", adapter_info.backend).to_lowercase(),
            tier_str,
            batch_size
        );

        // Provide tuning hints for verbose mode
        if verbose_enabled() {
            eprintln!(
                "[NEAT-AI-Discovery][verbose] GPU tuning: Set NEAT_AI_DISCOVERY_GPU_BATCH_SIZE=N to override (64-4096). \
                 Higher values improve GPU utilisation on powerful hardware."
            );
        }

        true
    });
}

// =============================================================================
// GpuAnalyzer Implementation
// =============================================================================

impl GpuAnalyzer {
    /// Lightweight probe to determine whether a usable GPU device is available.
    ///
    /// This is intended for callers (via FFI) that want to decide whether to
    /// enable the Rust discovery extension at all. It deliberately avoids
    /// falling back to CPU - if the adapter or device cannot be created, the
    /// probe reports `false`.
    ///
    /// Platform-specific behaviour:
    /// - **macOS**: GPU should always be available (Metal). Missing GPU is an error.
    /// - **Linux**: GPU may not be available on headless servers without GPU hardware
    ///   or proper permissions. Missing GPU gracefully disables discovery.
    ///
    /// **Note**: Result is cached for consistency. Creating wgpu instances is expensive
    /// and can give inconsistent results under parallel load (e.g., CI environments).
    pub fn gpu_is_available() -> bool {
        use std::sync::OnceLock;
        static GPU_AVAILABLE: OnceLock<bool> = OnceLock::new();
        *GPU_AVAILABLE.get_or_init(|| Self::check_gpu_availability().available)
    }

    /// Check GPU availability with detailed diagnostics.
    ///
    /// Returns availability status, reason, and whether it's an error condition.
    /// On macOS, missing GPU is treated as an error (Metal should always work).
    /// On Linux, missing GPU gracefully disables discovery (common on headless servers).
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

        let Some(adapter) = adapter else {
            return no_gpu_result("No GPU adapter found");
        };

        let device_result = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("NEAT-AI Discovery GPU probe device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        ));

        match device_result {
            Ok(_) => GpuAvailabilityResult {
                available: true,
                reason: None,
                is_error: false,
            },
            Err(e) => no_gpu_result(&format!("GPU device creation failed: {e}")),
        }
    }

    /// Check if the GPU supports unified memory architecture.
    ///
    /// On unified memory systems (Apple Silicon), CPU and GPU share the same
    /// physical memory, enabling zero-copy buffer sharing.
    ///
    /// Returns `true` if:
    /// - GPU name contains "Apple" (Apple Silicon Macs)
    /// - GPU is an integrated GPU (Intel/AMD integrated)
    ///
    /// Returns `false` if:
    /// - Discrete GPU (separate VRAM)
    /// - GPU not available
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

    /// Get information about the GPU adapter.
    ///
    /// Returns `None` if no GPU is available.
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

    /// Check if this GpuAnalyzer has a valid GPU device.
    ///
    /// Since `GpuAnalyzer::new()` now requires a GPU device (it returns an error
    /// if the GPU is unavailable), this method always returns `true` for a
    /// successfully created GpuAnalyzer instance.
    ///
    /// This method exists for backwards compatibility with code that previously
    /// checked `device.is_some()` before performing GPU operations.
    pub fn has_gpu(&self) -> bool {
        self.device.is_some()
    }

    /// Create a new GpuAnalyzer with all pipelines initialised.
    ///
    /// This is an expensive operation (~100ms) as it creates the GPU device and
    /// compiles all compute shaders. The analyzer should be created once and reused.
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
            Some(adapter) => adapter,
            None => {
                // GPU is required - return an error instead of a CPU-only analyzer.
                // TypeScript calls check_gpu_available() before discovery, but the GPU
                // could become unavailable due to race conditions or resource exhaustion.
                // Returning an error here prevents panics in GPU methods that .expect() on device.
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

        let (device, queue) = match pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("NEAT-AI Discovery GPU device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        )) {
            Ok(result) => result,
            Err(e) => {
                // GPU is required - return an error instead of a CPU-only analyzer.
                // TypeScript calls check_gpu_available() before discovery, but the GPU
                // could become unavailable due to race conditions or resource exhaustion.
                // Returning an error here prevents panics in GPU methods that .expect() on device.
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
            batch_size,
        })
    }

    // =========================================================================
    // Pipeline Builders
    // =========================================================================

    fn build_helpful_pipeline(
        device: &wgpu::Device,
        label: &str,
    ) -> (wgpu::BindGroupLayout, wgpu::ComputePipeline) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("helpful-synapse-shader"),
            source: wgpu::ShaderSource::Wgsl(HELPFUL_SHADER.into()),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("helpful-synapse-bind-group"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        (layout, pipeline)
    }

    fn build_harmful_pipeline(
        device: &wgpu::Device,
        label: &str,
    ) -> (wgpu::BindGroupLayout, wgpu::ComputePipeline) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("harmful-synapse-shader"),
            source: wgpu::ShaderSource::Wgsl(HARMFUL_SHADER.into()),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("harmful-synapse-bind-group"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        (layout, pipeline)
    }

    fn build_relu_pipeline(
        device: &wgpu::Device,
        label: &str,
    ) -> (wgpu::BindGroupLayout, wgpu::ComputePipeline) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("relu-shader"),
            source: wgpu::ShaderSource::Wgsl(RELU_SHADER.into()),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("relu-bind-group"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        (layout, pipeline)
    }

    fn build_activation_pipeline(
        device: &wgpu::Device,
        label: &str,
    ) -> (wgpu::BindGroupLayout, wgpu::ComputePipeline) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("activation-shader"),
            source: wgpu::ShaderSource::Wgsl(ACTIVATION_SHADER.into()),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("activation-bind-group"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        (layout, pipeline)
    }

    fn build_bias_pipeline(
        device: &wgpu::Device,
        label: &str,
    ) -> (wgpu::BindGroupLayout, wgpu::ComputePipeline) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bias-shader"),
            source: wgpu::ShaderSource::Wgsl(BIAS_SHADER.into()),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bias-bind-group"),
            entries: &[
                // Binding 0: samples (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 1: bias_candidates (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 2: results (read-write)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 3: uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        (layout, pipeline)
    }

    /// Build the helpful contribution reduction pipeline (Issue #218).
    ///
    /// This pipeline aggregates HelpfulContribution data on the GPU using parallel
    /// tree reduction within workgroups, reducing GPU→CPU transfer by ~250×.
    fn build_helpful_reduce_pipeline(
        device: &wgpu::Device,
        label: &str,
    ) -> (wgpu::BindGroupLayout, wgpu::ComputePipeline) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("helpful-reduce-shader"),
            source: wgpu::ShaderSource::Wgsl(HELPFUL_REDUCE_SHADER.into()),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("helpful-reduce-bind-group"),
            entries: &[
                // Binding 0: contributions (read-only input)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 1: partial_sums (read-write output)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 2: uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        (layout, pipeline)
    }

    /// Build the harmful contribution reduction pipeline (Issue #218).
    ///
    /// This pipeline aggregates HarmfulContribution data on the GPU using parallel
    /// tree reduction within workgroups, reducing GPU→CPU transfer by ~250×.
    fn build_harmful_reduce_pipeline(
        device: &wgpu::Device,
        label: &str,
    ) -> (wgpu::BindGroupLayout, wgpu::ComputePipeline) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("harmful-reduce-shader"),
            source: wgpu::ShaderSource::Wgsl(HARMFUL_REDUCE_SHADER.into()),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("harmful-reduce-bind-group"),
            entries: &[
                // Binding 0: contributions (read-only input)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 1: partial_sums (read-write output)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 2: uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        (layout, pipeline)
    }

    // =========================================================================
    // Batch Evaluation Methods
    // =========================================================================

    /// Batch evaluate multiple harmful synapse operations to improve GPU utilisation.
    /// Each entry is (samples, weight) pair. Returns stats in the same order as input.
    ///
    /// This reduces CPU-GPU round trips by submitting multiple GPU dispatches in a single
    /// command buffer, significantly improving throughput for harmful synapse analysis.
    ///
    /// For sample sets with >= GPU_REDUCTION_THRESHOLD samples, uses GPU-side
    /// workgroup reduction to minimise data transfer (Issue #218).
    pub fn evaluate_harmful_batch(
        &self,
        samples_batch: &[(&[HelpfulSample], f32)],
    ) -> Result<Vec<HarmfulStats>> {
        if samples_batch.is_empty() {
            return Ok(Vec::new());
        }

        // GPU is always required - discovery should be disabled without GPU
        let device = self
            .device
            .as_ref()
            .expect("GPU device must be available - discovery should be disabled without GPU");
        let queue = self
            .queue
            .as_ref()
            .context("GPU queue not initialised for batched harmful analysis")?;
        let harmful_layout = self
            .harmful_layout
            .as_ref()
            .context("GPU layout not initialised for batched harmful analysis")?;
        let harmful_pipeline = self
            .harmful_pipeline
            .as_ref()
            .context("GPU pipeline not initialised for batched harmful analysis")?;
        // Issue #218: Get reduction pipeline for large sample sets
        let harmful_reduce_layout = self
            .harmful_reduce_layout
            .as_ref()
            .context("GPU harmful reduce layout not initialised")?;
        let harmful_reduce_pipeline = self
            .harmful_reduce_pipeline
            .as_ref()
            .context("GPU harmful reduce pipeline not initialised")?;

        // Process in batches to avoid excessive memory usage
        // Apple Silicon optimisation: Use single encoder per batch to reduce Metal driver overhead
        let mut all_results = Vec::with_capacity(samples_batch.len());

        let max_sample_len = samples_batch
            .iter()
            .map(|(samples, _)| samples.len())
            .max()
            .unwrap_or(0);
        let bytes_per_sample = std::mem::size_of::<GpuHelpfulSample>()
            + (2 * std::mem::size_of::<HarmfulContribution>());
        let effective_batch_size = cap_gpu_batch_size_by_bytes(
            self.batch_size,
            max_sample_len,
            bytes_per_sample,
            GPU_MAX_BATCH_ALLOC_BYTES,
        );

        if verbose_enabled() && effective_batch_size < self.batch_size {
            let bytes_per_op = max_sample_len.saturating_mul(bytes_per_sample);
            let approx_mb = (bytes_per_op as f64 / (1024.0 * 1024.0)).max(0.0);
            eprintln!(
                "[NEAT-AI-Discovery][verbose] Capping GPU harmful batch size from {} to {} due to large sample count. \
                 max_sample_len={}, approx_buffers_per_op\u{2248}{:.1}MB, cap={}MB",
                self.batch_size,
                effective_batch_size,
                max_sample_len,
                approx_mb,
                GPU_MAX_BATCH_ALLOC_BYTES / (1024 * 1024)
            );
        }

        for batch_chunk in samples_batch.chunks(effective_batch_size) {
            let mut empty_flags = Vec::with_capacity(batch_chunk.len());
            let mut batch_staging_buffers = Vec::new();
            let mut batch_contribution_sizes = Vec::new();
            let mut batch_contributions_buffers = Vec::new();
            // Issue #218: Track whether each sample set uses reduction
            let mut uses_reduction_flags: Vec<bool> = Vec::with_capacity(batch_chunk.len());

            // Single encoder for entire batch - reduces Metal driver overhead on Apple Silicon
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("harmful-command-encoder-batch"),
            });

            // Prepare all operations in this batch
            for (samples, weight) in batch_chunk {
                if samples.is_empty() {
                    empty_flags.push(true);
                    uses_reduction_flags.push(false);
                    continue;
                }
                empty_flags.push(false);

                let gpu_samples: Vec<GpuHelpfulSample> = samples
                    .iter()
                    .copied()
                    .map(GpuHelpfulSample::from)
                    .collect();
                let contributions_zeroed = vec![HarmfulContribution::zeroed(); samples.len()];

                let sample_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("harmful-samples-buffer-batch"),
                    contents: bytemuck::cast_slice(&gpu_samples),
                    usage: wgpu::BufferUsages::STORAGE,
                });

                let contributions_buffer =
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("harmful-contributions-buffer-batch"),
                        contents: bytemuck::cast_slice(&contributions_zeroed),
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST,
                    });

                let uniforms = HarmfulUniforms {
                    length: samples.len() as u32,
                    pad0: 0,
                    epsilon: EPSILON,
                    weight: *weight,
                };
                let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("harmful-uniform-buffer-batch"),
                    contents: bytemuck::bytes_of(&uniforms),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: harmful_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: sample_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: contributions_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: uniform_buffer.as_entire_binding(),
                        },
                    ],
                    label: Some("harmful-bind-group-batch"),
                });

                // Add compute pass for per-sample contribution calculation
                {
                    let mut compute_pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("harmful-compute-pass-batch"),
                            timestamp_writes: None,
                        });
                    compute_pass.set_pipeline(harmful_pipeline);
                    compute_pass.set_bind_group(0, &bind_group, &[]);
                    let workgroups = (samples.len() as u32).div_ceil(WORKGROUP_SIZE);
                    compute_pass.dispatch_workgroups(workgroups.max(1), 1, 1);
                }

                // Issue #218: Use reduction for large sample counts
                let use_reduction = samples.len() >= GPU_REDUCTION_THRESHOLD;
                uses_reduction_flags.push(use_reduction);

                if use_reduction {
                    // Calculate number of workgroups for reduction
                    let num_workgroups = (samples.len() as u32).div_ceil(WORKGROUP_SIZE);
                    let partial_sums_size = (std::mem::size_of::<HarmfulContribution>()
                        * num_workgroups as usize)
                        as u64;

                    // Create partial sums buffer for reduction output
                    let partial_sums_zeroed =
                        vec![HarmfulContribution::zeroed(); num_workgroups as usize];
                    let partial_sums_buffer =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("harmful-partial-sums-buffer"),
                            contents: bytemuck::cast_slice(&partial_sums_zeroed),
                            usage: wgpu::BufferUsages::STORAGE
                                | wgpu::BufferUsages::COPY_SRC
                                | wgpu::BufferUsages::COPY_DST,
                        });

                    // Create reduction uniforms
                    let reduction_uniforms = ReductionUniforms {
                        contribution_count: samples.len() as u32,
                        pad0: 0,
                        pad1: 0,
                        pad2: 0,
                    };
                    let reduction_uniform_buffer =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("harmful-reduction-uniform-buffer"),
                            contents: bytemuck::bytes_of(&reduction_uniforms),
                            usage: wgpu::BufferUsages::UNIFORM,
                        });

                    // Create reduction bind group
                    let reduction_bind_group =
                        device.create_bind_group(&wgpu::BindGroupDescriptor {
                            layout: harmful_reduce_layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: contributions_buffer.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: partial_sums_buffer.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: reduction_uniform_buffer.as_entire_binding(),
                                },
                            ],
                            label: Some("harmful-reduction-bind-group"),
                        });

                    // Add reduction compute pass
                    {
                        let mut compute_pass =
                            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                label: Some("harmful-reduction-compute-pass"),
                                timestamp_writes: None,
                            });
                        compute_pass.set_pipeline(harmful_reduce_pipeline);
                        compute_pass.set_bind_group(0, &reduction_bind_group, &[]);
                        compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
                    }

                    // Staging buffer for partial sums (much smaller than full contributions)
                    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("harmful-staging-buffer-reduced"),
                        size: partial_sums_size,
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });

                    batch_contributions_buffers.push(partial_sums_buffer);
                    batch_staging_buffers.push(staging_buffer);
                    batch_contribution_sizes.push(partial_sums_size);
                } else {
                    // Original path: transfer all contributions to CPU
                    let contribution_size =
                        (std::mem::size_of::<HarmfulContribution>() * samples.len()) as u64;
                    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("harmful-staging-buffer-batch"),
                        size: contribution_size,
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });

                    batch_contributions_buffers.push(contributions_buffer);
                    batch_staging_buffers.push(staging_buffer);
                    batch_contribution_sizes.push(contribution_size);
                }
            }

            // Add all buffer copies after compute passes (better GPU scheduling)
            for (i, staging_buffer) in batch_staging_buffers.iter().enumerate() {
                let contribution_size = batch_contribution_sizes[i];
                encoder.copy_buffer_to_buffer(
                    &batch_contributions_buffers[i],
                    0,
                    staging_buffer,
                    0,
                    contribution_size,
                );
            }

            // Submit single command buffer for entire batch - reduces Metal driver overhead
            if !batch_staging_buffers.is_empty() {
                queue.submit(Some(encoder.finish()));
            }

            // OPTIMISATION: Map ALL buffers first, then poll ONCE for all.
            // This reduces GPU-CPU round trips compared to mapping each buffer individually.
            let mut map_receivers = Vec::with_capacity(batch_staging_buffers.len());
            for staging_buffer in &batch_staging_buffers {
                let buffer_slice = staging_buffer.slice(..);
                let (sender, receiver) = mpsc::channel();
                buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                    sender
                        .send(result)
                        .expect("Failed to send map_async result");
                });
                map_receivers.push(receiver);
            }

            // Event-driven wait: poll non-blocking, check all callback channels
            wait_for_buffer_maps_batch(device, &map_receivers, GPU_BUFFER_MAP_TIMEOUT_SECS)
                .context("Harmful batch buffer mapping failed")?;

            // Process results - maintain order with empty flags
            // Note: wait_for_buffer_maps_batch already verified all buffers are mapped
            // Issue #218: Filter uses_reduction_flags to only include non-empty samples
            let non_empty_reduction_flags: Vec<bool> = uses_reduction_flags
                .iter()
                .zip(empty_flags.iter())
                .filter(|&(_, &empty)| !empty)
                .map(|(&reduce, _)| reduce)
                .collect();

            let mut buffer_idx = 0;
            for is_empty in empty_flags {
                if is_empty {
                    all_results.push(HarmfulStats::default());
                } else {
                    // Buffer is already mapped and verified by wait_for_buffer_maps_batch
                    let staging_buffer = &batch_staging_buffers[buffer_idx];
                    let buffer_slice = staging_buffer.slice(..);
                    let data = buffer_slice.get_mapped_range();
                    let contributions: &[HarmfulContribution] = bytemuck::cast_slice(&data);

                    let mut stats = HarmfulStats::default();
                    // Both reduction and non-reduction paths produce HarmfulContribution,
                    // we just iterate over fewer elements when using reduction
                    for contribution in contributions {
                        stats.harmful_count += contribution.harmful_flag;
                        stats.helpful_count += contribution.helpful_flag;
                        stats.harmful_error_sum += contribution.error_magnitude;
                    }

                    // Log reduction usage for verbose output
                    if verbose_enabled()
                        && non_empty_reduction_flags
                            .get(buffer_idx)
                            .copied()
                            .unwrap_or(false)
                    {
                        let num_partial_sums = contributions.len();
                        eprintln!(
                            "[NEAT-AI-Discovery][verbose] GPU reduction (harmful): transferred {num_partial_sums} partial sums instead of full contributions"
                        );
                    }

                    drop(data);
                    staging_buffer.unmap();

                    all_results.push(stats);
                    buffer_idx += 1;
                }
            }

            // CRITICAL: Ensure Metal releases command buffers before creating new ones.
            // Without this, we can exhaust Metal's command buffer pool when processing
            // many batches, causing the GPU thread to hang in semaphore_wait_trap.
            //
            // Avoid an unbounded wait - bail out with an error if the driver is wedged.
            poll_device_until_idle(
                device,
                Duration::from_secs(5),
                "post-harmful-batch command buffer release",
            )?;
        }

        Ok(all_results)
    }

    /// GPU-accelerated ReLU evaluation.
    ///
    /// Evaluates ReLU activation for neuron candidates, returning statistics for
    /// both positive and negative orientations plus baseline error.
    pub fn evaluate_relu_gpu(
        &self,
        samples: &[HelpfulSample],
        threshold: f32,
    ) -> Result<(ReluStats, ReluStats, f32)> {
        if samples.is_empty() {
            return Ok((
                ReluStats::new(ReluOrientation::Positive),
                ReluStats::new(ReluOrientation::Negative),
                0.0,
            ));
        }

        // GPU is always required - discovery should be disabled without GPU
        let device = self
            .device
            .as_ref()
            .expect("GPU device must be available - discovery should be disabled without GPU");
        let queue = self
            .queue
            .as_ref()
            .context("GPU queue not initialised for ReLU analysis")?;
        let relu_layout = self
            .relu_layout
            .as_ref()
            .context("GPU ReLU layout not initialised")?;
        let relu_pipeline = self
            .relu_pipeline
            .as_ref()
            .context("GPU ReLU pipeline not initialised")?;

        let gpu_samples: Vec<GpuHelpfulSample> = samples
            .iter()
            .copied()
            .map(GpuHelpfulSample::from)
            .collect();
        let contributions_zeroed = vec![ReluContribution::zeroed(); samples.len()];

        let sample_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("relu-samples-buffer"),
            contents: bytemuck::cast_slice(&gpu_samples),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let contributions_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("relu-contributions-buffer"),
            contents: bytemuck::cast_slice(&contributions_zeroed),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let uniforms = ReluUniforms {
            length: samples.len() as u32,
            threshold,
            epsilon: EPSILON,
            pad0: 0.0,
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("relu-uniform-buffer"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: relu_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sample_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: contributions_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
            label: Some("relu-bind-group"),
        });

        let contribution_size = (std::mem::size_of::<ReluContribution>() * samples.len()) as u64;
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("relu-staging-buffer"),
            size: contribution_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("relu-command-encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("relu-compute-pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(relu_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (samples.len() as u32).div_ceil(WORKGROUP_SIZE);
            compute_pass.dispatch_workgroups(workgroups.max(1), 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &contributions_buffer,
            0,
            &staging_buffer,
            0,
            contribution_size,
        );

        queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender
                .send(result)
                .expect("Failed to send map_async result");
        });

        // Event-driven wait: poll non-blocking, check callback channel
        wait_for_buffer_map(device, &receiver, GPU_BUFFER_MAP_TIMEOUT_SECS)
            .context("ReLU buffer mapping failed")?;

        let data = buffer_slice.get_mapped_range();
        let contributions: &[ReluContribution] = bytemuck::cast_slice(&data);

        let mut positive_stats = ReluStats::new(ReluOrientation::Positive);
        let mut negative_stats = ReluStats::new(ReluOrientation::Negative);
        let mut total_baseline_error_sq = 0.0;

        // Accumulate statistics from GPU contributions
        for (idx, contribution) in contributions.iter().enumerate() {
            if idx < samples.len() {
                total_baseline_error_sq += contribution.error_sq;

                // For positive ReLU, accumulate activation_sq and error_activation
                if contribution.positive_count > 0 {
                    positive_stats.activation_sq_sum += contribution.positive_activation_sq;
                    positive_stats.error_activation_sum += contribution.positive_error_activation;
                    // Reconstruct the activation and error for the samples vector
                    let activation = (contribution.positive_activation_sq).sqrt();
                    let error = if activation > EPSILON {
                        contribution.positive_error_activation / activation
                    } else {
                        0.0
                    };
                    positive_stats.samples.push((activation, error));
                }

                // For negative ReLU, accumulate activation_sq and error_activation
                if contribution.negative_count > 0 {
                    negative_stats.activation_sq_sum += contribution.negative_activation_sq;
                    negative_stats.error_activation_sum += contribution.negative_error_activation;
                    // Reconstruct the activation and error for the samples vector
                    let activation = (contribution.negative_activation_sq).sqrt();
                    let error = if activation > EPSILON {
                        contribution.negative_error_activation / activation
                    } else {
                        0.0
                    };
                    negative_stats.samples.push((activation, error));
                }
            }
        }

        drop(data);
        staging_buffer.unmap();

        Ok((positive_stats, negative_stats, total_baseline_error_sq))
    }

    /// GPU-accelerated general activation evaluation.
    ///
    /// Returns (sum_activation_sq, sum_error_activation, total_baseline_error_sq, improved_count).
    pub fn evaluate_activation_gpu(
        &self,
        samples: &[HelpfulSample],
        activation_type: u32,
        orientation: f32,
        scale: f32,
    ) -> Result<(f32, f32, f32, u32)> {
        // Returns: (sum_activation_sq, sum_error_activation, total_baseline_error_sq, improved_count)
        if samples.is_empty() {
            return Ok((0.0, 0.0, 0.0, 0));
        }

        // GPU is always required - discovery should be disabled without GPU
        let device = self
            .device
            .as_ref()
            .expect("GPU device must be available - discovery should be disabled without GPU");
        let queue = self
            .queue
            .as_ref()
            .context("GPU queue not initialised for activation analysis")?;
        let activation_layout = self
            .activation_layout
            .as_ref()
            .context("GPU activation layout not initialised")?;
        let activation_pipeline = self
            .activation_pipeline
            .as_ref()
            .context("GPU activation pipeline not initialised")?;

        let gpu_samples: Vec<GpuHelpfulSample> = samples
            .iter()
            .copied()
            .map(GpuHelpfulSample::from)
            .collect();
        let outputs_zeroed = vec![ActivationOutput::zeroed(); samples.len()];

        let sample_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("activation-samples-buffer"),
            contents: bytemuck::cast_slice(&gpu_samples),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let outputs_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("activation-outputs-buffer"),
            contents: bytemuck::cast_slice(&outputs_zeroed),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let uniforms = ActivationUniforms {
            sample_count: samples.len() as u32,
            orientation,
            scale,
            activation_type,
            epsilon: EPSILON,
            pad0: 0.0,
            pad1: 0.0,
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("activation-uniform-buffer"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: activation_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sample_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: outputs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
            label: Some("activation-bind-group"),
        });

        let output_size = (std::mem::size_of::<ActivationOutput>() * samples.len()) as u64;
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("activation-staging-buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("activation-command-encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("activation-compute-pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(activation_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (samples.len() as u32).div_ceil(WORKGROUP_SIZE);
            compute_pass.dispatch_workgroups(workgroups.max(1), 1, 1);
        }

        encoder.copy_buffer_to_buffer(&outputs_buffer, 0, &staging_buffer, 0, output_size);

        queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender
                .send(result)
                .expect("Failed to send map_async result");
        });

        // Event-driven wait: poll non-blocking, check callback channel
        wait_for_buffer_map(device, &receiver, GPU_BUFFER_MAP_TIMEOUT_SECS)
            .context("Activation buffer mapping failed")?;

        let data = buffer_slice.get_mapped_range();
        let outputs: &[ActivationOutput] = bytemuck::cast_slice(&data);

        let mut sum_activation_sq = 0.0;
        let mut sum_error_activation = 0.0;
        let mut total_baseline_error_sq = 0.0;

        for (idx, output) in outputs.iter().enumerate() {
            if idx < samples.len() {
                let sample = &samples[idx];
                if sample.avg_error.is_finite() {
                    total_baseline_error_sq += sample.avg_error * sample.avg_error;
                }
                if output.valid > 0 {
                    sum_activation_sq += output.output_sq;
                    sum_error_activation += output.error_output;
                }
            }
        }

        // Note: improved_count is not calculated here because it requires weight validation
        // that happens in the caller. The caller will calculate improved_count after
        // validating and clamping the outgoing_weight.

        drop(data);
        staging_buffer.unmap();

        Ok((
            sum_activation_sq,
            sum_error_activation,
            total_baseline_error_sq,
            0, // improved_count calculated by caller after weight validation
        ))
    }

    /// GPU-accelerated batched activation evaluation.
    ///
    /// Issue #201: Evaluates multiple activation function configurations in a single
    /// GPU command buffer submission, reducing CPU-GPU round-trips by 10-20%.
    ///
    /// Instead of submitting separate GPU operations for each (activation_type, orientation, scale)
    /// combination, this method:
    /// 1. Uploads sample data once
    /// 2. Creates all compute passes in a single command buffer
    /// 3. Maps all result buffers together with a single `device.poll(Wait)`
    /// 4. Processes all results in a single CPU pass
    ///
    /// # Arguments
    /// * `samples` - The sample data to evaluate (uploaded once)
    /// * `activation_configs` - List of (activation_type, orientation, scale) tuples
    ///
    /// # Returns
    /// Vector of (sum_activation_sq, sum_error_activation, total_baseline_error_sq, improved_count)
    /// in the same order as the input configs.
    pub fn evaluate_activations_batched_gpu(
        &self,
        samples: &[HelpfulSample],
        activation_configs: &[(u32, f32, f32)],
    ) -> Result<Vec<(f32, f32, f32, u32)>> {
        // Handle edge cases
        if activation_configs.is_empty() {
            return Ok(Vec::new());
        }

        if samples.is_empty() {
            // Return zero results for each config
            return Ok(vec![(0.0, 0.0, 0.0, 0); activation_configs.len()]);
        }

        // GPU is always required - discovery should be disabled without GPU
        let device = self
            .device
            .as_ref()
            .expect("GPU device must be available - discovery should be disabled without GPU");
        let queue = self
            .queue
            .as_ref()
            .context("GPU queue not initialised for batched activation analysis")?;
        let activation_layout = self
            .activation_layout
            .as_ref()
            .context("GPU activation layout not initialised")?;
        let activation_pipeline = self
            .activation_pipeline
            .as_ref()
            .context("GPU activation pipeline not initialised")?;

        // Pre-convert samples to GPU format once (shared across all configs)
        let gpu_samples: Vec<GpuHelpfulSample> = samples
            .iter()
            .copied()
            .map(GpuHelpfulSample::from)
            .collect();

        // Pre-compute baseline error (same for all configs)
        let total_baseline_error_sq: f32 = samples
            .iter()
            .filter(|s| s.avg_error.is_finite())
            .map(|s| s.avg_error * s.avg_error)
            .sum();

        // Create a single sample buffer (shared across all compute passes)
        let sample_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("batched-activation-samples-buffer"),
            contents: bytemuck::cast_slice(&gpu_samples),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Calculate sizes
        let outputs_zeroed = vec![ActivationOutput::zeroed(); samples.len()];
        let output_size = (std::mem::size_of::<ActivationOutput>() * samples.len()) as u64;

        // Create output buffers, staging buffers, uniform buffers, and bind groups for each config
        let mut output_buffers = Vec::with_capacity(activation_configs.len());
        let mut staging_buffers = Vec::with_capacity(activation_configs.len());
        let mut bind_groups = Vec::with_capacity(activation_configs.len());

        for (config_idx, &(activation_type, orientation, scale)) in
            activation_configs.iter().enumerate()
        {
            // Output buffer for this config
            let outputs_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("batched-activation-outputs-buffer-{config_idx}")),
                contents: bytemuck::cast_slice(&outputs_zeroed),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

            // Staging buffer for this config
            let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("batched-activation-staging-buffer-{config_idx}")),
                size: output_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            // Uniform buffer for this config
            let uniforms = ActivationUniforms {
                sample_count: samples.len() as u32,
                orientation,
                scale,
                activation_type,
                epsilon: EPSILON,
                pad0: 0.0,
                pad1: 0.0,
            };
            let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("batched-activation-uniform-buffer-{config_idx}")),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            // Bind group for this config (shares sample_buffer)
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: activation_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: sample_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: outputs_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                ],
                label: Some(&format!("batched-activation-bind-group-{config_idx}")),
            });

            output_buffers.push(outputs_buffer);
            staging_buffers.push(staging_buffer);
            bind_groups.push(bind_group);
        }

        // Create a SINGLE command encoder for ALL compute passes
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("batched-activation-command-encoder"),
        });

        // Issue all compute passes in a single command buffer
        let workgroups = (samples.len() as u32).div_ceil(WORKGROUP_SIZE);
        for (config_idx, bind_group) in bind_groups.iter().enumerate() {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("batched-activation-compute-pass-{config_idx}")),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(activation_pipeline);
            compute_pass.set_bind_group(0, bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroups.max(1), 1, 1);
        }

        // Copy all output buffers to staging buffers
        for (output_buffer, staging_buffer) in output_buffers.iter().zip(staging_buffers.iter()) {
            encoder.copy_buffer_to_buffer(output_buffer, 0, staging_buffer, 0, output_size);
        }

        // Submit the SINGLE command buffer with ALL operations
        queue.submit(Some(encoder.finish()));

        // Issue map_async for ALL staging buffers BEFORE polling
        // This allows the GPU to process the map requests in parallel
        let receivers: Vec<_> = staging_buffers
            .iter()
            .map(|staging_buffer| {
                let buffer_slice = staging_buffer.slice(..);
                let (sender, receiver) = mpsc::channel();
                buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                    sender
                        .send(result)
                        .expect("Failed to send map_async result");
                });
                receiver
            })
            .collect();

        // Wait for ALL buffers to be mapped with a single polling loop
        wait_for_buffer_maps_batch(device, &receivers, GPU_BUFFER_MAP_TIMEOUT_SECS)
            .context("Batched activation buffer mapping failed")?;

        // Process all results
        let mut results = Vec::with_capacity(activation_configs.len());
        for staging_buffer in &staging_buffers {
            let buffer_slice = staging_buffer.slice(..);
            let data = buffer_slice.get_mapped_range();
            let outputs: &[ActivationOutput] = bytemuck::cast_slice(&data);

            let mut sum_activation_sq = 0.0f32;
            let mut sum_error_activation = 0.0f32;

            for (idx, output) in outputs.iter().enumerate() {
                if idx < samples.len() && output.valid > 0 {
                    sum_activation_sq += output.output_sq;
                    sum_error_activation += output.error_output;
                }
            }

            drop(data);
            staging_buffer.unmap();

            results.push((
                sum_activation_sq,
                sum_error_activation,
                total_baseline_error_sq,
                0, // improved_count calculated by caller after weight validation
            ));
        }

        Ok(results)
    }

    /// GPU-accelerated bias grid search.
    /// Tests all bias values in parallel and returns the optimal bias.
    pub fn evaluate_bias_gpu(
        &self,
        samples: &[HelpfulSample],
        incoming_weight: f32,
        outgoing_weight: f32,
        activation_type: u32,
        bias_range: (f32, f32, f32),
    ) -> Result<f32> {
        // Returns: optimal bias value
        if samples.is_empty() {
            return Ok(0.0);
        }

        let (min_bias, max_bias, step) = bias_range;

        // GPU is always required - discovery should be disabled without GPU
        let device = self
            .device
            .as_ref()
            .expect("GPU device must be available - discovery should be disabled without GPU");
        let queue = self
            .queue
            .as_ref()
            .context("GPU queue not initialised for bias analysis")?;
        let bias_layout = self
            .bias_layout
            .as_ref()
            .context("GPU bias layout not initialised")?;
        let bias_pipeline = self
            .bias_pipeline
            .as_ref()
            .context("GPU bias pipeline not initialised")?;

        // Generate bias candidates
        let num_steps = ((max_bias - min_bias) / step).ceil() as i32 + 1;
        let bias_candidates: Vec<f32> = (0..num_steps)
            .map(|i| min_bias + (i as f32 * step).min(max_bias - min_bias))
            .collect();

        if bias_candidates.is_empty() {
            return Ok(0.0);
        }

        // Prepare GPU buffers
        let gpu_samples: Vec<GpuHelpfulSample> = samples
            .iter()
            .copied()
            .map(GpuHelpfulSample::from)
            .collect();
        let results_zeroed = vec![BiasResult::zeroed(); bias_candidates.len()];

        let sample_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bias-samples-buffer"),
            contents: bytemuck::cast_slice(&gpu_samples),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let bias_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bias-candidates-buffer"),
            contents: bytemuck::cast_slice(&bias_candidates),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let results_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bias-results-buffer"),
            contents: bytemuck::cast_slice(&results_zeroed),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let uniforms = BiasUniforms {
            sample_count: samples.len() as u32,
            bias_count: bias_candidates.len() as u32,
            incoming_weight,
            outgoing_weight,
            activation_type,
            epsilon: EPSILON,
            min_sample_count: MIN_NEURON_SAMPLE_COUNT as u32,
            pad0: 0,
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bias-uniform-buffer"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: bias_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sample_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bias_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: results_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
            label: Some("bias-bind-group"),
        });

        let output_size = (std::mem::size_of::<BiasResult>() * bias_candidates.len()) as u64;
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bias-staging-buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("bias-command-encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bias-compute-pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(bias_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (bias_candidates.len() as u32).div_ceil(WORKGROUP_SIZE);
            compute_pass.dispatch_workgroups(workgroups.max(1), 1, 1);
        }

        encoder.copy_buffer_to_buffer(&results_buffer, 0, &staging_buffer, 0, output_size);

        queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender
                .send(result)
                .expect("Failed to send map_async result");
        });

        // Event-driven wait: poll non-blocking, check callback channel
        wait_for_buffer_map(device, &receiver, GPU_BUFFER_MAP_TIMEOUT_SECS)
            .context("Bias buffer mapping failed")?;

        let data = buffer_slice.get_mapped_range();
        let results: &[BiasResult] = bytemuck::cast_slice(&data);

        // Find bias with best error reduction
        let mut best_bias = 0.0;
        let mut best_error_reduction = f32::NEG_INFINITY;

        for result in results {
            if result.valid_sample_count >= MIN_NEURON_SAMPLE_COUNT as u32
                && result.error_reduction > best_error_reduction
            {
                best_error_reduction = result.error_reduction;
                best_bias = result.bias_value;
            }
        }

        drop(data);
        staging_buffer.unmap();

        Ok(best_bias)
    }

    /// Batch evaluate multiple helpful operations to improve GPU utilisation.
    /// Returns a vector of stats in the same order as the input samples.
    ///
    /// For sample sets with >= GPU_REDUCTION_THRESHOLD samples, uses GPU-side
    /// workgroup reduction to minimise data transfer (Issue #218).
    pub fn evaluate_helpful_batch(
        &self,
        samples_batch: &[&[HelpfulSample]],
    ) -> Result<Vec<HelpfulStats>> {
        if samples_batch.is_empty() {
            return Ok(Vec::new());
        }

        // GPU is always required - discovery should be disabled without GPU
        let device = self
            .device
            .as_ref()
            .expect("GPU device must be available - discovery should be disabled without GPU");
        let queue = self
            .queue
            .as_ref()
            .context("GPU queue not initialised for batched helpful analysis")?;
        let helpful_layout = self
            .helpful_layout
            .as_ref()
            .context("GPU layout not initialised for batched helpful analysis")?;
        let helpful_pipeline = self
            .helpful_pipeline
            .as_ref()
            .context("GPU pipeline not initialised for batched helpful analysis")?;
        // Issue #218: Get reduction pipeline for large sample sets
        let helpful_reduce_layout = self
            .helpful_reduce_layout
            .as_ref()
            .context("GPU helpful reduce layout not initialised")?;
        let helpful_reduce_pipeline = self
            .helpful_reduce_pipeline
            .as_ref()
            .context("GPU helpful reduce pipeline not initialised")?;

        // Process in batches to avoid excessive memory usage
        // Apple Silicon optimisation: Use single encoder per batch to reduce Metal driver overhead
        let mut all_results = Vec::with_capacity(samples_batch.len());

        let max_sample_len = samples_batch.iter().map(|s| s.len()).max().unwrap_or(0);
        let bytes_per_sample = std::mem::size_of::<GpuHelpfulSample>()
            + (2 * std::mem::size_of::<HelpfulContribution>());
        let effective_batch_size = cap_gpu_batch_size_by_bytes(
            self.batch_size,
            max_sample_len,
            bytes_per_sample,
            GPU_MAX_BATCH_ALLOC_BYTES,
        );

        if verbose_enabled() && effective_batch_size < self.batch_size {
            let bytes_per_op = max_sample_len.saturating_mul(bytes_per_sample);
            let approx_mb = (bytes_per_op as f64 / (1024.0 * 1024.0)).max(0.0);
            eprintln!(
                "[NEAT-AI-Discovery][verbose] Capping GPU helpful batch size from {} to {} due to large sample count. \
                 max_sample_len={}, approx_buffers_per_op\u{2248}{:.1}MB, cap={}MB",
                self.batch_size,
                effective_batch_size,
                max_sample_len,
                approx_mb,
                GPU_MAX_BATCH_ALLOC_BYTES / (1024 * 1024)
            );
        }

        for batch_chunk in samples_batch.chunks(effective_batch_size) {
            let mut empty_flags = Vec::with_capacity(batch_chunk.len());
            let mut batch_staging_buffers = Vec::new();
            let mut batch_contribution_sizes = Vec::new();
            let mut batch_contributions_buffers = Vec::new();
            // Issue #218: Track whether each sample set uses reduction
            let mut uses_reduction_flags: Vec<bool> = Vec::with_capacity(batch_chunk.len());

            // Single encoder for entire batch - reduces Metal driver overhead on Apple Silicon
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("helpful-command-encoder-batch"),
            });

            // Prepare all operations in this batch
            for samples in batch_chunk {
                if samples.is_empty() {
                    empty_flags.push(true);
                    uses_reduction_flags.push(false);
                    continue;
                }
                empty_flags.push(false);

                let gpu_samples: Vec<GpuHelpfulSample> = samples
                    .iter()
                    .copied()
                    .map(GpuHelpfulSample::from)
                    .collect();
                let contributions_zeroed = vec![HelpfulContribution::zeroed(); samples.len()];

                let sample_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("helpful-samples-buffer-batch"),
                    contents: bytemuck::cast_slice(&gpu_samples),
                    usage: wgpu::BufferUsages::STORAGE,
                });

                let contributions_buffer =
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("helpful-contributions-buffer-batch"),
                        contents: bytemuck::cast_slice(&contributions_zeroed),
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST,
                    });

                let uniforms = HelpfulUniforms {
                    length: samples.len() as u32,
                    pad0: 0,
                    epsilon: EPSILON,
                    pad1: 0.0,
                };
                let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("helpful-uniform-buffer-batch"),
                    contents: bytemuck::bytes_of(&uniforms),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: helpful_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: sample_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: contributions_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: uniform_buffer.as_entire_binding(),
                        },
                    ],
                    label: Some("helpful-bind-group-batch"),
                });

                // Add compute pass for per-sample contribution calculation
                {
                    let mut compute_pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("helpful-compute-pass-batch"),
                            timestamp_writes: None,
                        });
                    compute_pass.set_pipeline(helpful_pipeline);
                    compute_pass.set_bind_group(0, &bind_group, &[]);
                    let workgroups = (samples.len() as u32).div_ceil(WORKGROUP_SIZE);
                    compute_pass.dispatch_workgroups(workgroups.max(1), 1, 1);
                }

                // Issue #218: Use reduction for large sample counts
                let use_reduction = samples.len() >= GPU_REDUCTION_THRESHOLD;
                uses_reduction_flags.push(use_reduction);

                if use_reduction {
                    // Calculate number of workgroups for reduction
                    let num_workgroups = (samples.len() as u32).div_ceil(WORKGROUP_SIZE);
                    let partial_sums_size = (std::mem::size_of::<HelpfulContribution>()
                        * num_workgroups as usize)
                        as u64;

                    // Create partial sums buffer for reduction output
                    let partial_sums_zeroed =
                        vec![HelpfulContribution::zeroed(); num_workgroups as usize];
                    let partial_sums_buffer =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("helpful-partial-sums-buffer"),
                            contents: bytemuck::cast_slice(&partial_sums_zeroed),
                            usage: wgpu::BufferUsages::STORAGE
                                | wgpu::BufferUsages::COPY_SRC
                                | wgpu::BufferUsages::COPY_DST,
                        });

                    // Create reduction uniforms
                    let reduction_uniforms = ReductionUniforms {
                        contribution_count: samples.len() as u32,
                        pad0: 0,
                        pad1: 0,
                        pad2: 0,
                    };
                    let reduction_uniform_buffer =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("helpful-reduction-uniform-buffer"),
                            contents: bytemuck::bytes_of(&reduction_uniforms),
                            usage: wgpu::BufferUsages::UNIFORM,
                        });

                    // Create reduction bind group
                    let reduction_bind_group =
                        device.create_bind_group(&wgpu::BindGroupDescriptor {
                            layout: helpful_reduce_layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: contributions_buffer.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: partial_sums_buffer.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: reduction_uniform_buffer.as_entire_binding(),
                                },
                            ],
                            label: Some("helpful-reduction-bind-group"),
                        });

                    // Add reduction compute pass
                    {
                        let mut compute_pass =
                            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                label: Some("helpful-reduction-compute-pass"),
                                timestamp_writes: None,
                            });
                        compute_pass.set_pipeline(helpful_reduce_pipeline);
                        compute_pass.set_bind_group(0, &reduction_bind_group, &[]);
                        compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
                    }

                    // Staging buffer for partial sums (much smaller than full contributions)
                    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("helpful-staging-buffer-reduced"),
                        size: partial_sums_size,
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });

                    batch_contributions_buffers.push(partial_sums_buffer);
                    batch_staging_buffers.push(staging_buffer);
                    batch_contribution_sizes.push((partial_sums_size, num_workgroups as usize));
                } else {
                    // Original path: transfer all contributions to CPU
                    let contribution_size =
                        (std::mem::size_of::<HelpfulContribution>() * samples.len()) as u64;
                    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("helpful-staging-buffer-batch"),
                        size: contribution_size,
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });

                    batch_contributions_buffers.push(contributions_buffer);
                    batch_staging_buffers.push(staging_buffer);
                    batch_contribution_sizes.push((contribution_size, samples.len()));
                }
            }

            // Add all buffer copies after compute passes (better GPU scheduling)
            for (i, staging_buffer) in batch_staging_buffers.iter().enumerate() {
                let (contribution_size, _) = batch_contribution_sizes[i];
                encoder.copy_buffer_to_buffer(
                    &batch_contributions_buffers[i],
                    0,
                    staging_buffer,
                    0,
                    contribution_size,
                );
            }

            // Submit single command buffer for entire batch - reduces Metal driver overhead
            if !batch_staging_buffers.is_empty() {
                queue.submit(Some(encoder.finish()));
            }

            // OPTIMISATION: Map ALL buffers first, then poll ONCE for all.
            // This reduces GPU-CPU round trips compared to mapping each buffer individually.
            let mut map_receivers = Vec::with_capacity(batch_staging_buffers.len());
            for staging_buffer in &batch_staging_buffers {
                let buffer_slice = staging_buffer.slice(..);
                let (sender, receiver) = mpsc::channel();
                buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                    sender
                        .send(result)
                        .expect("Failed to send map_async result");
                });
                map_receivers.push(receiver);
            }

            // Event-driven wait: poll non-blocking, check all callback channels
            wait_for_buffer_maps_batch(device, &map_receivers, GPU_BUFFER_MAP_TIMEOUT_SECS)
                .context("Helpful batch buffer mapping failed")?;

            // Now read all the mapped data (buffers are already mapped)
            let mut batch_results = Vec::with_capacity(batch_contribution_sizes.len());
            // Buffer mappings already verified by wait_for_buffer_maps_batch
            // Issue #218: Filter uses_reduction_flags to only include non-empty samples
            let non_empty_reduction_flags: Vec<bool> = uses_reduction_flags
                .iter()
                .zip(empty_flags.iter())
                .filter(|&(_, &empty)| !empty)
                .map(|(&reduce, _)| reduce)
                .collect();

            for (i, staging_buffer) in batch_staging_buffers.iter().enumerate() {
                let buffer_slice = staging_buffer.slice(..);
                let data = buffer_slice.get_mapped_range();
                let contributions: &[HelpfulContribution] = bytemuck::cast_slice(&data);

                let mut stats = HelpfulStats::default();
                // Both reduction and non-reduction paths produce HelpfulContribution,
                // we just iterate over fewer elements when using reduction
                for contribution in contributions {
                    stats.positive_count += contribution.positive_flag;
                    stats.negative_count += contribution.negative_flag;
                    stats.positive_improvement_sum += contribution.positive_improvement;
                    stats.negative_improvement_sum += contribution.negative_improvement;
                    stats.positive_activation_sum += contribution.positive_activation;
                    stats.negative_activation_sum += contribution.negative_activation;
                    stats.error_sq_sum += contribution.error_squared;
                    stats.activation_sq_sum += contribution.activation_squared;
                    stats.error_activation_sum += contribution.error_activation;
                }

                // Log reduction usage for verbose output
                if verbose_enabled() && non_empty_reduction_flags.get(i).copied().unwrap_or(false) {
                    let (_, count) = batch_contribution_sizes[i];
                    eprintln!(
                        "[NEAT-AI-Discovery][verbose] GPU reduction: transferred {count} partial sums instead of full contributions"
                    );
                }

                drop(data);
                staging_buffer.unmap();

                batch_results.push(stats);
            }

            let merged_results = Self::merge_batch_results(&empty_flags, batch_results);
            all_results.extend(merged_results);

            // CRITICAL: Ensure Metal releases command buffers before creating new ones.
            // Without this, we can exhaust Metal's command buffer pool when processing
            // many batches, causing the GPU thread to hang in semaphore_wait_trap.
            //
            // Avoid an unbounded wait - bail out with an error if the driver is wedged.
            poll_device_until_idle(
                device,
                Duration::from_secs(5),
                "post-helpful-batch command buffer release",
            )?;
        }

        Ok(all_results)
    }

    /// Merge batch results, inserting default stats for empty sample sets.
    ///
    /// This is a helper for batch processing that maintains ordering when some
    /// input sample sets are empty and were skipped during GPU processing.
    pub fn merge_batch_results(
        empty_flags: &[bool],
        computed_stats: Vec<HelpfulStats>,
    ) -> Vec<HelpfulStats> {
        let expected_non_empty = empty_flags.iter().filter(|flag| !**flag).count();
        debug_assert_eq!(
            expected_non_empty,
            computed_stats.len(),
            "Computed stats should match number of non-empty sample sets"
        );

        let mut results = Vec::with_capacity(empty_flags.len());
        let mut stats_iter = computed_stats.into_iter();

        for &is_empty in empty_flags {
            if is_empty {
                results.push(HelpfulStats::default());
            } else if let Some(stats) = stats_iter.next() {
                results.push(stats);
            } else {
                // Safety guard: if counts mismatch, preserve ordering by inserting default.
                results.push(HelpfulStats::default());
            }
        }

        results
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workgroup_size_constant() {
        // Verify workgroup size is 256 as expected by shaders
        assert_eq!(WORKGROUP_SIZE, 256);
    }

    #[test]
    fn test_min_neuron_sample_count() {
        // Verify minimum sample count is reasonable
        assert_eq!(MIN_NEURON_SAMPLE_COUNT, 10);
    }

    #[test]
    fn test_gpu_max_batch_alloc_bytes() {
        // Verify max batch allocation is 256MB
        assert_eq!(GPU_MAX_BATCH_ALLOC_BYTES, 256 * 1024 * 1024);
    }

    #[test]
    fn test_batch_size_for_tier() {
        // Test that batch sizes are returned correctly for each tier
        let high = get_batch_size_for_tier(GpuPerformanceTier::High);
        let standard = get_batch_size_for_tier(GpuPerformanceTier::Standard);
        let unknown = get_batch_size_for_tier(GpuPerformanceTier::Unknown);

        // High tier should have larger batch size
        assert!(high >= standard);
        // Standard and unknown should be the same
        assert_eq!(standard, unknown);
    }

    #[test]
    fn test_merge_batch_results_empty() {
        let empty_flags: Vec<bool> = vec![];
        let computed: Vec<HelpfulStats> = vec![];
        let result = GpuAnalyzer::merge_batch_results(&empty_flags, computed);
        assert!(result.is_empty());
    }

    #[test]
    fn test_merge_batch_results_all_empty() {
        let empty_flags = vec![true, true, true];
        let computed: Vec<HelpfulStats> = vec![];
        let result = GpuAnalyzer::merge_batch_results(&empty_flags, computed);
        assert_eq!(result.len(), 3);
        for stats in result {
            assert_eq!(stats.positive_count, 0);
        }
    }

    #[test]
    fn test_merge_batch_results_mixed() {
        let empty_flags = vec![true, false, true, false];
        let computed = vec![
            HelpfulStats {
                positive_count: 5,
                ..Default::default()
            },
            HelpfulStats {
                positive_count: 10,
                ..Default::default()
            },
        ];
        let result = GpuAnalyzer::merge_batch_results(&empty_flags, computed);
        assert_eq!(result.len(), 4);
        assert_eq!(result[0].positive_count, 0); // empty
        assert_eq!(result[1].positive_count, 5); // first computed
        assert_eq!(result[2].positive_count, 0); // empty
        assert_eq!(result[3].positive_count, 10); // second computed
    }
}
