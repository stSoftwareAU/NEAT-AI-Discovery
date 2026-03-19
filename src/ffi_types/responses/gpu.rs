//! GPU-related JSON types for the FFI boundary.
//!
//! Contains [`CheckGpuOutput`], GPU timing JSON types, and adapter info JSON
//! types, plus conversion helpers from internal analysis types.

use serde::Serialize;

use crate::analysis;
use crate::ffi_types::DiscoveryErrorKind;

// =============================================================================
// GPU Info JSON Types (Issue #228)
// =============================================================================

/// JSON representation of GPU adapter information.
#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct GpuAdapterInfoJson {
    /// Human-readable name of the GPU (e.g., "Apple M4 Pro").
    pub name: String,
    /// Whether the GPU has unified memory architecture.
    pub unified_memory: bool,
    /// Whether zero-copy buffer sharing is currently enabled.
    pub zero_copy_enabled: bool,
}

// =============================================================================
// GPU Timing JSON Types (Issue #195)
// =============================================================================

/// JSON representation of per-shader timing statistics.
#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ShaderTimingJson {
    /// Number of times this shader was executed.
    pub calls: u32,
    /// Total execution time in milliseconds.
    pub total_ms: f64,
    /// Average execution time per call in milliseconds.
    pub avg_ms: f64,
}

/// JSON representation of GPU-side timing breakdown.
#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct GpuTimingBreakdownJson {
    /// Total time spent in shader execution (all shaders combined) in milliseconds.
    pub shader_execution_ms: f64,
    /// Total time spent in buffer mapping/transfers in milliseconds.
    pub buffer_transfer_ms: f64,
    /// Per-shader timing statistics.
    /// Keys are shader names: "helpful", "harmful", "relu", "activation", "bias"
    #[serde(skip_serializing_if = "std::collections::HashMap::is_empty")]
    pub shader_timings: std::collections::HashMap<String, ShaderTimingJson>,
}

/// JSON representation of CPU-side timing breakdown.
#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct CpuTimingBreakdownJson {
    /// Time spent building samples for GPU evaluation in milliseconds.
    pub sample_building_ms: f64,
    /// Time spent processing results from GPU in milliseconds.
    pub result_processing_ms: f64,
}

/// JSON representation of complete timing data for an analysis run.
///
/// Only populated when `NEAT_AI_DISCOVERY_GPU_TIMING=1` is set.
#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct AnalysisTimingJson {
    /// Total wall-clock time for the analysis in milliseconds.
    pub total_analysis_ms: f64,
    /// GPU-side timing breakdown.
    pub gpu: GpuTimingBreakdownJson,
    /// CPU-side timing breakdown.
    pub cpu: CpuTimingBreakdownJson,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CheckGpuOutput {
    pub success: bool,
    pub gpu_available: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Structured error classification for retry decisions (Issue #651).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_kind: Option<DiscoveryErrorKind>,
    /// Whether this error is typically worth retrying (Issue #651).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retryable: Option<bool>,
}

// ============================================================================
// Conversion helpers
// ============================================================================

/// Convert GPU adapter info to JSON representation.
pub(crate) fn gpu_info_to_json(info: &analysis::shared::GpuAdapterInfo) -> GpuAdapterInfoJson {
    GpuAdapterInfoJson {
        name: info.name.clone(),
        unified_memory: info.has_unified_memory,
        zero_copy_enabled: info.zero_copy_enabled,
    }
}

/// Convert internal timing data to JSON representation.
pub(crate) fn timing_to_json(timing: &analysis::shared::AnalysisTiming) -> AnalysisTimingJson {
    AnalysisTimingJson {
        total_analysis_ms: timing.total_analysis_ms,
        gpu: GpuTimingBreakdownJson {
            shader_execution_ms: timing.gpu.shader_execution_ms,
            buffer_transfer_ms: timing.gpu.buffer_transfer_ms,
            shader_timings: timing
                .gpu
                .shader_timings
                .iter()
                .map(|(name, st)| {
                    (
                        name.clone(),
                        ShaderTimingJson {
                            calls: st.calls,
                            total_ms: st.total_ms,
                            avg_ms: st.avg_ms,
                        },
                    )
                })
                .collect(),
        },
        cpu: CpuTimingBreakdownJson {
            sample_building_ms: timing.cpu.sample_building_ms,
            result_processing_ms: timing.cpu.result_processing_ms,
        },
    }
}
