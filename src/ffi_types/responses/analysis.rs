//! Analysis output types and diagnostic JSON structures for the FFI boundary.
//!
//! Contains [`AnalyzeParallelOutput`], metadata JSON types, and diagnostic
//! types for synapse and neuron analysis results.

use serde::Serialize;

use super::gpu::{AnalysisTimingJson, GpuAdapterInfoJson};
use crate::analysis;
use crate::ffi_types::DiscoveryErrorKind;

use crate::{
    CandidateNeuronJson, CandidateSynapseJson, CoordinatedStructuralCandidateJson,
    SynapseWeightUpdateCandidateJson,
};

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
    /// Updated module outcome tracker for persistence across runs (Issue #792).
    ///
    /// Contains historical per-module success rates updated with candidate counts
    /// from this run. Callers should persist this and pass it back as
    /// `moduleOutcomeTracker` on the next discovery run.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub module_outcome_tracker: Option<analysis::module_weights::ModuleOutcomeTracker>,
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
