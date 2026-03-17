//! Shared types and structures used across analysis modules

use crate::{
    CandidateNeuronJson, CandidateSynapseJson, CoordinatedStructuralCandidateJson,
    SynapseWeightUpdateCandidateJson,
};
use std::collections::HashMap;

// Re-import ErrorDistribution for metadata (Issue #192)
use super::scoring::error_distribution::ErrorDistribution;

// =============================================================================
// GPU Timing Types (Issue #195)
// =============================================================================

/// Timing statistics for a single shader type.
///
/// Collects call counts and timing data for performance diagnostics.
#[derive(Debug, Clone, Default)]
pub struct ShaderTiming {
    /// Number of times this shader was executed.
    pub calls: u32,
    /// Total execution time in milliseconds.
    pub total_ms: f64,
    /// Average execution time per call in milliseconds.
    pub avg_ms: f64,
}

/// GPU-side timing breakdown.
///
/// Tracks time spent in GPU operations including shader execution
/// and buffer transfers.
#[derive(Debug, Clone, Default)]
pub struct GpuTimingBreakdown {
    /// Total time spent in shader execution (all shaders combined) in milliseconds.
    pub shader_execution_ms: f64,
    /// Total time spent in buffer mapping/transfers in milliseconds.
    pub buffer_transfer_ms: f64,
    /// Per-shader timing statistics.
    /// Keys are shader names: "helpful", "harmful", "relu", "activation", "bias"
    pub shader_timings: HashMap<String, ShaderTiming>,
}

/// CPU-side timing breakdown.
///
/// Tracks time spent in CPU operations during analysis.
#[derive(Debug, Clone, Default)]
pub struct CpuTimingBreakdown {
    /// Time spent building samples for GPU evaluation in milliseconds.
    pub sample_building_ms: f64,
    /// Time spent processing results from GPU in milliseconds.
    pub result_processing_ms: f64,
}

/// Complete timing data for an analysis run.
///
/// This is only populated when `NEAT_AI_DISCOVERY_GPU_TIMING=1` is set.
/// Provides detailed timing breakdown for performance diagnostics.
#[derive(Debug, Clone, Default)]
pub struct AnalysisTiming {
    /// Total wall-clock time for the analysis in milliseconds.
    pub total_analysis_ms: f64,
    /// GPU-side timing breakdown.
    pub gpu: GpuTimingBreakdown,
    /// CPU-side timing breakdown.
    pub cpu: CpuTimingBreakdown,
}

// =============================================================================
// Timing Collector (Issue #195)
// =============================================================================

use parking_lot::Mutex;
use std::sync::atomic::Ordering;
use std::time::Instant;

/// Thread-safe timing collector for GPU kernel profiling.
///
/// This collector aggregates timing data from multiple threads (parallel focus neuron processing)
/// and provides a consolidated view of GPU and CPU timing.
///
/// Only collects timing when `NEAT_AI_DISCOVERY_GPU_TIMING=1` is set.
#[derive(Debug)]
pub struct TimingCollector {
    enabled: bool,
    start_time: Instant,
    /// Per-shader timing data: (name -> (calls, total_ns))
    shader_timings: Mutex<HashMap<String, (u32, u64)>>,
    /// Buffer transfer time in nanoseconds
    buffer_transfer_ns: AtomicU64,
    /// Sample building time in nanoseconds
    sample_building_ns: AtomicU64,
    /// Result processing time in nanoseconds
    result_processing_ns: AtomicU64,
}

use std::sync::atomic::AtomicU64;

impl TimingCollector {
    /// Create a new timing collector.
    ///
    /// If `enabled` is false, all timing operations are no-ops.
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            start_time: Instant::now(),
            shader_timings: Mutex::new(HashMap::new()),
            buffer_transfer_ns: AtomicU64::new(0),
            sample_building_ns: AtomicU64::new(0),
            result_processing_ns: AtomicU64::new(0),
        }
    }

    /// Check if timing collection is enabled.
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Record a shader execution.
    pub fn record_shader(&self, shader_name: &str, duration_ns: u64) {
        if !self.enabled {
            return;
        }
        let mut timings = super::utils::lock_contention::traced_lock_default(
            &self.shader_timings,
            "shader_timings",
        );
        let entry = timings.entry(shader_name.to_string()).or_insert((0, 0));
        entry.0 += 1;
        entry.1 += duration_ns;
    }

    /// Record buffer transfer time.
    pub fn record_buffer_transfer(&self, duration_ns: u64) {
        if !self.enabled {
            return;
        }
        self.buffer_transfer_ns
            .fetch_add(duration_ns, Ordering::Relaxed);
    }

    /// Record sample building time.
    pub fn record_sample_building(&self, duration_ns: u64) {
        if !self.enabled {
            return;
        }
        self.sample_building_ns
            .fetch_add(duration_ns, Ordering::Relaxed);
    }

    /// Record result processing time.
    pub fn record_result_processing(&self, duration_ns: u64) {
        if !self.enabled {
            return;
        }
        self.result_processing_ns
            .fetch_add(duration_ns, Ordering::Relaxed);
    }

    /// Finalize and return the collected timing data.
    ///
    /// Returns `None` if timing is disabled.
    pub fn finalize(&self) -> Option<AnalysisTiming> {
        if !self.enabled {
            return None;
        }

        let total_analysis_ms = self.start_time.elapsed().as_secs_f64() * 1000.0;

        let shader_timings_lock = super::utils::lock_contention::traced_lock_default(
            &self.shader_timings,
            "shader_timings_finalize",
        );
        let mut shader_timings = HashMap::new();
        let mut total_shader_ns: u64 = 0;

        for (name, (calls, total_ns)) in shader_timings_lock.iter() {
            let total_ms = *total_ns as f64 / 1_000_000.0;
            let avg_ms = if *calls > 0 {
                total_ms / (*calls as f64)
            } else {
                0.0
            };
            shader_timings.insert(
                name.clone(),
                ShaderTiming {
                    calls: *calls,
                    total_ms,
                    avg_ms,
                },
            );
            total_shader_ns += total_ns;
        }

        let buffer_transfer_ns = self.buffer_transfer_ns.load(Ordering::Relaxed);
        let sample_building_ns = self.sample_building_ns.load(Ordering::Relaxed);
        let result_processing_ns = self.result_processing_ns.load(Ordering::Relaxed);

        Some(AnalysisTiming {
            total_analysis_ms,
            gpu: GpuTimingBreakdown {
                shader_execution_ms: total_shader_ns as f64 / 1_000_000.0,
                buffer_transfer_ms: buffer_transfer_ns as f64 / 1_000_000.0,
                shader_timings,
            },
            cpu: CpuTimingBreakdown {
                sample_building_ms: sample_building_ns as f64 / 1_000_000.0,
                result_processing_ms: result_processing_ns as f64 / 1_000_000.0,
            },
        })
    }
}

impl Default for TimingCollector {
    fn default() -> Self {
        Self::new(false)
    }
}

/// RAII guard for timing a scope.
///
/// Records the duration when dropped.
pub struct TimingScope<'a> {
    collector: &'a TimingCollector,
    shader_name: Option<String>,
    category: TimingCategory,
    start: Instant,
}

/// Category of timing to record.
#[derive(Debug, Clone, Copy)]
pub enum TimingCategory {
    Shader,
    BufferTransfer,
    SampleBuilding,
    ResultProcessing,
}

impl<'a> TimingScope<'a> {
    /// Create a new timing scope for a shader.
    pub fn shader(collector: &'a TimingCollector, shader_name: &str) -> Self {
        Self {
            collector,
            shader_name: Some(shader_name.to_string()),
            category: TimingCategory::Shader,
            start: Instant::now(),
        }
    }

    /// Create a new timing scope for buffer transfers.
    pub fn buffer_transfer(collector: &'a TimingCollector) -> Self {
        Self {
            collector,
            shader_name: None,
            category: TimingCategory::BufferTransfer,
            start: Instant::now(),
        }
    }

    /// Create a new timing scope for sample building.
    pub fn sample_building(collector: &'a TimingCollector) -> Self {
        Self {
            collector,
            shader_name: None,
            category: TimingCategory::SampleBuilding,
            start: Instant::now(),
        }
    }

    /// Create a new timing scope for result processing.
    pub fn result_processing(collector: &'a TimingCollector) -> Self {
        Self {
            collector,
            shader_name: None,
            category: TimingCategory::ResultProcessing,
            start: Instant::now(),
        }
    }
}

impl Drop for TimingScope<'_> {
    fn drop(&mut self) {
        if !self.collector.is_enabled() {
            return;
        }
        let duration_ns = self.start.elapsed().as_nanos() as u64;
        match self.category {
            TimingCategory::Shader => {
                if let Some(name) = &self.shader_name {
                    self.collector.record_shader(name, duration_ns);
                }
            }
            TimingCategory::BufferTransfer => {
                self.collector.record_buffer_transfer(duration_ns);
            }
            TimingCategory::SampleBuilding => {
                self.collector.record_sample_building(duration_ns);
            }
            TimingCategory::ResultProcessing => {
                self.collector.record_result_processing(duration_ns);
            }
        }
    }
}

/// Metadata about synapse analysis for diagnostics and observability.
///
/// This metadata helps callers understand:
/// - Whether prediction accuracy is degraded (missing `value` data)
/// - Whether candidates were truncated by `maxCandidates`
/// - What fraction of the analysis budget was consumed
#[derive(Debug, Default, Clone)]
pub struct SynapseAnalysisMetadata {
    /// Whether `target_value` (pre-activation) was available in the recorded data.
    ///
    /// If `false`, the saturation-aware simulation path cannot be used and predictions
    /// may be less accurate for saturating activation functions like `HARD_TANH`.
    pub target_value_available: bool,

    /// Whether saturation-aware simulation was used for at least one candidate.
    ///
    /// This is `true` when both:
    /// 1. `target_value` is available, AND
    /// 2. The target neuron has a supported saturating activation (e.g., HARD_TANH)
    ///
    /// If `false` but the target has a saturating activation, predictions may invert.
    pub saturation_aware_simulation_used: bool,

    /// Total number of synapse candidates found during analysis (before truncation).
    pub candidates_found: usize,

    /// Number of synapse candidates returned to caller (after `maxCandidates` truncation).
    pub candidates_returned: usize,

    /// True when analysis hit its deadline and returned partial results.
    pub timed_out: bool,

    /// Number of focus neurons completed before returning.
    ///
    /// This allows callers to validate long-run coverage under repeated deadline-constrained runs.
    pub completed_focus_neurons: usize,

    /// Total focus neurons requested for this analysis invocation.
    pub total_focus_neurons: usize,

    /// Range of input indices that were observed with non-empty record sets during analysis.
    ///
    /// This helps diagnose whether new observation inputs (eg `input-1486+`) are present in the
    /// Parquet and are being considered by discovery.
    pub input_index_min_seen_with_records: Option<usize>,
    pub input_index_max_seen_with_records: Option<usize>,

    /// GPU timing data for performance diagnostics (Issue #195).
    ///
    /// Only populated when `NEAT_AI_DISCOVERY_GPU_TIMING=1` is set.
    /// When `None`, timing collection was disabled (default).
    pub timing: Option<AnalysisTiming>,

    /// Information about the GPU adapter used (Issue #228).
    ///
    /// Includes unified memory detection and zero-copy status.
    pub gpu_info: Option<GpuAdapterInfo>,

    /// Error distribution statistics for the target neurons (Issue #192).
    ///
    /// Provides percentiles, skewness, kurtosis and other distribution metrics
    /// to enable targeted discovery for specific error patterns like outliers.
    pub error_distribution: Option<ErrorDistribution>,

    /// Per-module discovery statistics for the current run (Issue #485).
    ///
    /// Reports how many candidates each discovery module produced, enabling
    /// operators to see which modules are most effective.
    pub discovery_module_stats: Vec<super::module_weights::DiscoveryModuleStatsJson>,
}

/// Metadata about neuron analysis for diagnostics and observability.
#[derive(Debug, Default, Clone)]
pub struct NeuronAnalysisMetadata {
    /// Total number of neuron candidates found during analysis (before truncation).
    pub candidates_found: usize,

    /// Number of neuron candidates returned to caller (after `maxCandidates` truncation).
    pub candidates_returned: usize,

    /// True when analysis hit its deadline and returned partial results.
    pub timed_out: bool,

    /// Number of focus neurons completed before returning.
    pub completed_focus_neurons: usize,

    /// Total focus neurons requested for this analysis invocation.
    pub total_focus_neurons: usize,

    /// GPU timing data for performance diagnostics (Issue #195).
    ///
    /// Only populated when `NEAT_AI_DISCOVERY_GPU_TIMING=1` is set.
    /// When `None`, timing collection was disabled (default).
    pub timing: Option<AnalysisTiming>,

    /// Information about the GPU adapter used (Issue #228).
    ///
    /// Includes unified memory detection and zero-copy status.
    pub gpu_info: Option<GpuAdapterInfo>,

    /// Error distribution statistics for the target neurons (Issue #192).
    ///
    /// Provides percentiles, skewness, kurtosis and other distribution metrics
    /// to enable targeted discovery for specific error patterns like outliers.
    pub error_distribution: Option<ErrorDistribution>,
}

/// Result of synapse analysis
#[derive(Debug)]
pub struct AnalyzeSynapsesResult {
    pub helpful_synapses: Vec<CandidateSynapseJson>,
    pub harmful_synapses: Vec<CandidateSynapseJson>,
    pub synapse_weight_updates: Vec<SynapseWeightUpdateCandidateJson>,
    /// Coordinated (grouped) candidates produced from synapse analysis (Issue #165).
    pub coordinated_structural_candidates: Vec<CoordinatedStructuralCandidateJson>,
    /// Candidate clusters for redundancy reduction (Issue #224).
    ///
    /// Groups similar candidates by target neuron, source type, and improvement
    /// similarity so the controller can test a representative first and skip
    /// redundant ablation tests.
    pub candidate_clusters: Vec<super::candidate_clustering::CandidateClusterJson>,
    pub gpu_used: bool,
    pub no_candidate_reasons: Vec<SynapseNoCandidateSummary>,
    /// Metadata about the analysis run for diagnostics (v0.2.17+).
    pub metadata: SynapseAnalysisMetadata,
}

/// Result of neuron analysis
#[derive(Debug)]
pub struct AnalyzeNeuronsResult {
    pub helpful_neurons: Vec<CandidateNeuronJson>,
    pub gpu_used: bool,
    pub no_candidate_reasons: Vec<NeuronNoCandidateSummary>,
    /// Metadata about the analysis run for diagnostics (v0.2.17+).
    pub metadata: NeuronAnalysisMetadata,
}

/// Combined result of both synapse and neuron analysis
#[derive(Debug)]
pub struct AnalyzeAllResult {
    pub synapse: Option<AnalyzeSynapsesResult>,
    pub neuron: Option<AnalyzeNeuronsResult>,
    /// Current neuron fingerprints for incremental analysis (Issue #490).
    ///
    /// Callers should store these and pass them back on the next run.
    pub neuron_fingerprints:
        Option<std::collections::HashMap<String, super::neuron_fingerprint::NeuronFingerprint>>,
    /// Number of focus neurons skipped due to unchanged fingerprints (Issue #490).
    pub fingerprint_cache_hits: usize,
    /// Number of focus neurons analysed (changed or new fingerprints) (Issue #490).
    pub fingerprint_cache_misses: usize,
    /// Updated module outcome tracker for persistence (Issue #792).
    ///
    /// Contains the tracker passed in (or a default), updated with candidate counts
    /// from this run. Callers should persist this and pass it back on subsequent runs.
    pub module_outcome_tracker: super::module_weights::ModuleOutcomeTracker,
}

/// Reason why no synapse candidate was found for a target neuron
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SynapseNoCandidateReason {
    NoEligibleSources,
    NoDiagnostics,
    NoSamples,
    ZeroImprovement,
    BelowThreshold,
}

/// Detailed information about why a synapse candidate was rejected
#[derive(Debug, Clone)]
pub struct SynapseNoCandidateDetail {
    pub source_uuid: Option<String>,
    pub sample_count: Option<usize>,
    pub source_record_count: Option<usize>,
    pub improved_count: Option<u32>,
    pub worsened_count: Option<u32>,
    pub expected_improvement: Option<f32>,
    pub threshold: Option<f32>,
    pub suggested_weight: Option<f32>,
}

/// Summary of why no synapse candidate was found for a target neuron
#[derive(Debug, Clone)]
pub struct SynapseNoCandidateSummary {
    pub target_uuid: String,
    pub reason: SynapseNoCandidateReason,
    pub evaluated_candidates: u32,
    pub candidates_with_samples: u32,
    pub target_record_count: usize,
    pub detail: Option<SynapseNoCandidateDetail>,
}

/// Reason why no neuron candidate was found for a target neuron
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NeuronNoCandidateReason {
    NoEligibleSources,
    NoDiagnostics,
    NoSamples,
    NotEnoughActivations,
    WeightDegenerate,
    BelowThreshold,
    /// Hidden neurons are filtered out from add-neuron analysis because their
    /// backpropagated errors don't reliably translate to output error reduction.
    HiddenNeuronFiltered,
    /// Input neurons are filtered out from add-neuron analysis because they're
    /// observation sources, not computation nodes - they have no activation function
    /// or error to reduce.
    InputNeuronFiltered,
    /// Constant neurons are filtered out from add-neuron analysis because they
    /// don't receive inputs - they always output a fixed value regardless of
    /// network state, so adding a connection to them has no effect.
    ConstantNeuronFiltered,
}

/// Detailed information about why a neuron candidate was rejected
#[derive(Debug, Clone)]
pub struct NeuronNoCandidateDetail {
    pub source_uuid: Option<String>,
    pub orientation: Option<String>,
    pub sample_count: Option<usize>,
    pub improved_count: Option<u32>,
    pub worsened_count: Option<u32>,
    pub expected_improvement: Option<f32>,
    pub threshold: Option<f32>,
    pub outgoing_weight: Option<f32>,
}

/// Summary of why no neuron candidate was found for a target neuron
#[derive(Debug, Clone)]
pub struct NeuronNoCandidateSummary {
    pub target_uuid: String,
    pub reason: NeuronNoCandidateReason,
    pub evaluated_sources: u32,
    pub sources_with_samples: u32,
    pub target_record_count: usize,
    pub detail: Option<NeuronNoCandidateDetail>,
}

// =============================================================================
// GPU Info and Zero-Copy Types (Issue #228)
// =============================================================================

/// Information about the GPU adapter being used.
///
/// Provides details about the GPU hardware and its capabilities,
/// particularly for unified memory detection (Apple Silicon).
#[derive(Debug, Clone)]
pub struct GpuAdapterInfo {
    /// Human-readable name of the GPU (e.g., "Apple M4 Pro").
    pub name: String,
    /// Device type (discrete, integrated, software, etc.).
    pub device_type: GpuDeviceType,
    /// Whether the GPU has unified memory architecture.
    ///
    /// On unified memory systems (Apple Silicon), CPU and GPU share the same
    /// physical memory, enabling zero-copy buffer sharing.
    pub has_unified_memory: bool,
    /// Whether zero-copy buffer sharing is currently enabled.
    ///
    /// This may differ from `has_unified_memory` if the user has explicitly
    /// disabled zero-copy via environment variable.
    pub zero_copy_enabled: bool,
}

/// GPU device type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuDeviceType {
    /// Discrete GPU (separate VRAM, e.g., NVIDIA/AMD cards).
    Discrete,
    /// Integrated GPU (shares system memory, e.g., Intel integrated).
    Integrated,
    /// Software/CPU-based rendering (fallback).
    Software,
    /// Virtual GPU (e.g., cloud instances).
    Virtual,
    /// Other/unknown device type.
    Other,
}

impl From<wgpu::DeviceType> for GpuDeviceType {
    fn from(device_type: wgpu::DeviceType) -> Self {
        match device_type {
            wgpu::DeviceType::DiscreteGpu => GpuDeviceType::Discrete,
            wgpu::DeviceType::IntegratedGpu => GpuDeviceType::Integrated,
            wgpu::DeviceType::Cpu => GpuDeviceType::Software,
            wgpu::DeviceType::VirtualGpu => GpuDeviceType::Virtual,
            wgpu::DeviceType::Other => GpuDeviceType::Other,
        }
    }
}

/// Configuration for zero-copy buffer sharing.
///
/// Zero-copy buffer sharing eliminates unnecessary CPU-GPU data copies on
/// unified memory architectures like Apple Silicon. This configuration
/// allows fine-grained control over when zero-copy is used.
#[derive(Debug, Clone)]
pub struct ZeroCopyBufferConfig {
    /// Whether zero-copy is explicitly enabled or disabled via environment variable.
    ///
    /// - `Some(true)`: Force-enabled via NEAT_AI_DISCOVERY_ZERO_COPY=1
    /// - `Some(false)`: Force-disabled via NEAT_AI_DISCOVERY_ZERO_COPY=0
    /// - `None`: Auto-detect based on unified memory support
    force_enabled: Option<bool>,
    /// Number of buffers in the ring buffer for pipelining.
    ///
    /// Triple buffering (3) is the default, allowing one buffer for CPU writes,
    /// one for GPU reads, and one in flight.
    buffer_count: usize,
}

impl Default for ZeroCopyBufferConfig {
    fn default() -> Self {
        Self {
            force_enabled: None,
            buffer_count: 3, // Triple buffering
        }
    }
}

impl ZeroCopyBufferConfig {
    /// Create configuration from environment variables.
    ///
    /// Delegates to [`crate::config::zero_copy_override()`].
    pub fn from_env() -> Self {
        Self {
            force_enabled: crate::config::zero_copy_override(),
            buffer_count: 3,
        }
    }

    /// Whether zero-copy should be enabled for the given hardware.
    ///
    /// Uses force setting if present, otherwise auto-detects based on
    /// unified memory support.
    pub fn enabled(&self) -> bool {
        // Will be computed with actual hardware check
        self.force_enabled.unwrap_or(false)
    }

    /// Whether zero-copy was explicitly enabled or disabled.
    pub fn force_enabled(&self) -> Option<bool> {
        self.force_enabled
    }

    /// Number of buffers in the ring buffer.
    pub fn buffer_count(&self) -> usize {
        self.buffer_count
    }

    /// Check if enabled with hardware detection.
    pub fn enabled_with_hardware(&self, has_unified_memory: bool) -> bool {
        self.force_enabled.unwrap_or(has_unified_memory)
    }
}
