//! FFI response structs (output to NEAT-AI).
//!
//! All `*Output` types and supporting diagnostic/metadata types used at the
//! FFI boundary for serialising JSON responses.

use serde::Serialize;

use super::DiscoveryErrorKind;
use crate::analysis;

use super::{
    CandidateNeuronJson, CandidateSynapseJson, CoordinatedStructuralCandidateJson,
    RankedNeuronJson, RemovalCandidateJson, SynapseWeightUpdateCandidateJson,
};

/// JSON output from record_discovery function
#[derive(Debug, Serialize)]
pub struct RecordDiscoveryOutput {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temp_dir: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Structured error classification for retry decisions (Issue #651).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_kind: Option<DiscoveryErrorKind>,
    /// Whether this error is typically worth retrying (Issue #651).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retryable: Option<bool>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AnalyzeParallelOutput {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub helpful_synapses: Option<Vec<CandidateSynapseJson>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub harmful_synapses: Option<Vec<CandidateSynapseJson>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub synapse_diagnostics: Option<Vec<SynapseDiagnosticJson>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub synapse_gpu_used: Option<bool>,
    /// Synapse analysis metadata for observability (v0.2.17+).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub synapse_metadata: Option<SynapseAnalysisMetadataJson>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub helpful_neurons: Option<Vec<CandidateNeuronJson>>,
    /// Candidates to update weights on existing synapses (v0.2.18+).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub synapse_weight_updates: Option<Vec<SynapseWeightUpdateCandidateJson>>,
    /// Coordinated (grouped) structural candidates (v0.2.18+).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub coordinated_structural_candidates: Option<Vec<CoordinatedStructuralCandidateJson>>,
    /// Candidate clusters for redundancy reduction (Issue #224).
    ///
    /// Groups similar candidates by target neuron so the controller can test a
    /// representative first and skip redundant ablation tests.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidate_clusters: Option<Vec<analysis::candidate_clustering::CandidateClusterJson>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub neuron_diagnostics: Option<Vec<NeuronDiagnosticJson>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub neuron_gpu_used: Option<bool>,
    /// Neuron analysis metadata for observability (v0.2.17+).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub neuron_metadata: Option<NeuronAnalysisMetadataJson>,
    /// Current neuron fingerprints for incremental analysis (Issue #490).
    ///
    /// Callers should store these and pass them back as `previousNeuronFingerprints`
    /// on the next discovery run to enable incremental analysis.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub neuron_fingerprints:
        Option<std::collections::HashMap<String, analysis::neuron_fingerprint::NeuronFingerprint>>,
    /// Number of focus neurons skipped due to unchanged fingerprints (Issue #490).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fingerprint_cache_hits: Option<usize>,
    /// Number of focus neurons analysed (changed or new fingerprints) (Issue #490).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fingerprint_cache_misses: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Structured error classification for retry decisions (Issue #651).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_kind: Option<DiscoveryErrorKind>,
    /// Whether this error is typically worth retrying (Issue #651).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retryable: Option<bool>,
}

// Coordinated structural candidates are now produced inside synapse analysis and surfaced via
// `AnalyzeSynapsesResult.coordinated_structural_candidates` (Issue #165).

/// JSON representation of synapse analysis metadata.
///
/// Surfaces diagnostic information to help callers understand:
/// - Whether prediction accuracy is degraded (missing `value` data)
/// - Whether candidates were truncated by `maxCandidates`
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SynapseAnalysisMetadataJson {
    /// Whether `targetValue` (pre-activation) was available in the recorded data.
    pub target_value_available: bool,
    /// Whether saturation-aware simulation was used for at least one candidate.
    pub saturation_aware_simulation_used: bool,
    /// Total number of synapse candidates found during analysis (before truncation).
    pub candidates_found: usize,
    /// Number of synapse candidates returned to caller (after `maxCandidates` truncation).
    pub candidates_returned: usize,
    /// True when analysis hit its deadline and returned partial results.
    pub timed_out: bool,
    /// Number of focus neurons completed before returning.
    pub completed_focus_neurons: usize,
    /// Total focus neurons requested for this analysis invocation.
    pub total_focus_neurons: usize,
    /// Minimum input index observed with non-empty records (eg 0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_index_min_seen_with_records: Option<usize>,
    /// Maximum input index observed with non-empty records (eg 1555).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_index_max_seen_with_records: Option<usize>,
    /// GPU timing data for performance diagnostics (Issue #195).
    /// Only present when `NEAT_AI_DISCOVERY_GPU_TIMING=1` is set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timing: Option<AnalysisTimingJson>,
    /// Information about the GPU adapter used (Issue #228).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_info: Option<GpuAdapterInfoJson>,
    /// Per-discovery-module statistics for the current run (Issue #485).
    /// Reports how many candidates each module produced and historical success rates.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub discovery_module_stats: Vec<analysis::module_weights::DiscoveryModuleStatsJson>,
}

/// JSON representation of neuron analysis metadata.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NeuronAnalysisMetadataJson {
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
    /// Only present when `NEAT_AI_DISCOVERY_GPU_TIMING=1` is set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timing: Option<AnalysisTimingJson>,
    /// Information about the GPU adapter used (Issue #228).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_info: Option<GpuAdapterInfoJson>,
}

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

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GetVersionOutput {
    pub success: bool,
    pub version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Structured error classification for retry decisions (Issue #651).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_kind: Option<DiscoveryErrorKind>,
    /// Whether this error is typically worth retrying (Issue #651).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retryable: Option<bool>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RankFocusNeuronsOutput {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub neurons: Option<Vec<RankedNeuronJson>>,
    /// Neurons with high error but very low impact - candidates for removal
    #[serde(skip_serializing_if = "Option::is_none")]
    pub removal_candidates: Option<Vec<RemovalCandidateJson>>,
    /// Issue #306: Coordinated structural candidates for removing constant-value neurons.
    /// When a hidden neuron has near-zero activation variance (constant output), it can be
    /// removed and its effect folded into bias adjustments for downstream neurons.
    /// Each candidate contains:
    /// - A RemoveNeuron operation for the constant neuron
    /// - SetBias operations for all downstream neurons with adjusted biases
    #[serde(skip_serializing_if = "Option::is_none")]
    pub constant_neuron_removals: Option<Vec<CoordinatedStructuralCandidateJson>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_error: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub processed_neurons: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_neurons: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Structured error classification for retry decisions (Issue #651).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_kind: Option<DiscoveryErrorKind>,
    /// Whether this error is typically worth retrying (Issue #651).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retryable: Option<bool>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MergeParquetOutput {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_file: Option<String>,
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
// Visualisation Snapshot Export API Types
// ============================================================================

/// JSON output from export_visualisation_snapshot function
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportVisualisationSnapshotOutput {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub out_file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stats: Option<ExportVisualisationStats>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Structured error classification for retry decisions (Issue #651).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_kind: Option<DiscoveryErrorKind>,
    /// Whether this error is typically worth retrying (Issue #651).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retryable: Option<bool>,
}

/// Statistics from the export operation
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportVisualisationStats {
    pub obs_count: usize,
    pub neuron_count: usize,
    pub synapse_count: usize,
    pub output_count: usize,
}

// ============================================================================
// Read discovery records types
// ============================================================================

/// JSON output from read_discovery_records function
#[derive(Debug, Serialize)]
pub struct ReadDiscoveryOutput {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub records: Option<Vec<DiscoverRecordJson>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Structured error classification for retry decisions (Issue #651).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_kind: Option<DiscoveryErrorKind>,
    /// Whether this error is typically worth retrying (Issue #651).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retryable: Option<bool>,
}

/// JSON representation of DiscoverRecord for serialization
#[derive(Debug, Serialize)]
pub struct DiscoverRecordJson {
    pub obs_index: u32,
    pub neuron_uuid: String,
    pub value: Option<f32>,
    pub activation: f32,
    pub errors: Vec<f32>,
}

// ============================================================================
// Calibration Summary (Issue #605)
// ============================================================================

/// JSON output from get_calibration_summary function.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CalibrationSummaryOutput {
    pub success: bool,
    /// Calibration summary entries, one per module/candidate-type combination.
    pub calibration_summary: Vec<crate::discovery_history::CalibrationSummaryEntry>,
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
// Diagnostic types
// ============================================================================

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SynapseDiagnosticJson {
    pub target_neuron_uuid: String,
    pub reason: SynapseDiagnosticReasonJson,
    pub evaluated_candidates: u32,
    pub candidates_with_samples: u32,
    pub target_record_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<SynapseDiagnosticDetailJson>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SynapseDiagnosticReasonJson {
    NoEligibleSources,
    NoDiagnostics,
    NoSamples,
    ZeroImprovement,
    BelowThreshold,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SynapseDiagnosticDetailJson {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_neuron_uuid: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_record_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub improved_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worsened_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_creature_error_reduction: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suggested_weight: Option<f32>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NeuronDiagnosticJson {
    pub target_neuron_uuid: String,
    pub reason: NeuronDiagnosticReasonJson,
    pub evaluated_sources: u32,
    pub sources_with_samples: u32,
    pub target_record_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<NeuronDiagnosticDetailJson>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NeuronDiagnosticReasonJson {
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

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NeuronDiagnosticDetailJson {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_neuron_uuid: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub orientation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub improved_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worsened_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_creature_error_reduction: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outgoing_weight: Option<f32>,
}

// ============================================================================
// Conversion helpers
// ============================================================================

/// Convert GPU adapter info to JSON representation.
pub(crate) fn gpu_info_to_json(info: &analysis::GpuAdapterInfo) -> GpuAdapterInfoJson {
    GpuAdapterInfoJson {
        name: info.name.clone(),
        unified_memory: info.has_unified_memory,
        zero_copy_enabled: info.zero_copy_enabled,
    }
}

/// Convert internal timing data to JSON representation.
pub(crate) fn timing_to_json(timing: &analysis::AnalysisTiming) -> AnalysisTimingJson {
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

pub(crate) fn synapse_diagnostics_json(
    summaries: &[analysis::SynapseNoCandidateSummary],
) -> Option<Vec<SynapseDiagnosticJson>> {
    if summaries.is_empty() {
        return None;
    }
    Some(
        summaries
            .iter()
            .map(|summary| SynapseDiagnosticJson {
                target_neuron_uuid: summary.target_uuid.clone(),
                reason: match summary.reason {
                    analysis::SynapseNoCandidateReason::NoEligibleSources => {
                        SynapseDiagnosticReasonJson::NoEligibleSources
                    }
                    analysis::SynapseNoCandidateReason::NoDiagnostics => {
                        SynapseDiagnosticReasonJson::NoDiagnostics
                    }
                    analysis::SynapseNoCandidateReason::NoSamples => {
                        SynapseDiagnosticReasonJson::NoSamples
                    }
                    analysis::SynapseNoCandidateReason::ZeroImprovement => {
                        SynapseDiagnosticReasonJson::ZeroImprovement
                    }
                    analysis::SynapseNoCandidateReason::BelowThreshold => {
                        SynapseDiagnosticReasonJson::BelowThreshold
                    }
                },
                evaluated_candidates: summary.evaluated_candidates,
                candidates_with_samples: summary.candidates_with_samples,
                target_record_count: summary.target_record_count,
                detail: summary
                    .detail
                    .as_ref()
                    .map(|detail| SynapseDiagnosticDetailJson {
                        source_neuron_uuid: detail.source_uuid.clone(),
                        sample_count: detail.sample_count,
                        source_record_count: detail.source_record_count,
                        improved_count: detail.improved_count,
                        worsened_count: detail.worsened_count,
                        expected_creature_error_reduction: detail.expected_improvement,
                        threshold: detail.threshold,
                        suggested_weight: detail.suggested_weight,
                    }),
            })
            .collect(),
    )
}

pub(crate) fn neuron_diagnostics_json(
    summaries: &[analysis::NeuronNoCandidateSummary],
) -> Option<Vec<NeuronDiagnosticJson>> {
    if summaries.is_empty() {
        return None;
    }
    Some(
        summaries
            .iter()
            .map(|summary| NeuronDiagnosticJson {
                target_neuron_uuid: summary.target_uuid.clone(),
                reason: match summary.reason {
                    analysis::NeuronNoCandidateReason::NoEligibleSources => {
                        NeuronDiagnosticReasonJson::NoEligibleSources
                    }
                    analysis::NeuronNoCandidateReason::NoDiagnostics => {
                        NeuronDiagnosticReasonJson::NoDiagnostics
                    }
                    analysis::NeuronNoCandidateReason::NoSamples => {
                        NeuronDiagnosticReasonJson::NoSamples
                    }
                    analysis::NeuronNoCandidateReason::NotEnoughActivations => {
                        NeuronDiagnosticReasonJson::NotEnoughActivations
                    }
                    analysis::NeuronNoCandidateReason::WeightDegenerate => {
                        NeuronDiagnosticReasonJson::WeightDegenerate
                    }
                    analysis::NeuronNoCandidateReason::BelowThreshold => {
                        NeuronDiagnosticReasonJson::BelowThreshold
                    }
                    analysis::NeuronNoCandidateReason::HiddenNeuronFiltered => {
                        NeuronDiagnosticReasonJson::HiddenNeuronFiltered
                    }
                    analysis::NeuronNoCandidateReason::InputNeuronFiltered => {
                        NeuronDiagnosticReasonJson::InputNeuronFiltered
                    }
                    analysis::NeuronNoCandidateReason::ConstantNeuronFiltered => {
                        NeuronDiagnosticReasonJson::ConstantNeuronFiltered
                    }
                },
                evaluated_sources: summary.evaluated_sources,
                sources_with_samples: summary.sources_with_samples,
                target_record_count: summary.target_record_count,
                detail: summary
                    .detail
                    .as_ref()
                    .map(|detail| NeuronDiagnosticDetailJson {
                        source_neuron_uuid: detail.source_uuid.clone(),
                        orientation: detail.orientation.clone(),
                        sample_count: detail.sample_count,
                        improved_count: detail.improved_count,
                        worsened_count: detail.worsened_count,
                        expected_creature_error_reduction: detail.expected_improvement,
                        threshold: detail.threshold,
                        outgoing_weight: detail.outgoing_weight,
                    }),
            })
            .collect(),
    )
}
