//! Analysis metadata and result types.
//!
//! Contains the result structures ([`AnalyzeSynapsesResult`],
//! [`AnalyzeNeuronsResult`], [`AnalyzeAllResult`]) and their associated metadata
//! and diagnostic types for synapse and neuron analysis pipelines.

use crate::{
    CandidateNeuronJson, CandidateSynapseJson, CoordinatedStructuralCandidateJson,
    SynapseWeightUpdateCandidateJson,
};

use super::gpu_info::GpuAdapterInfo;
use super::timing::AnalysisTiming;
use crate::analysis::scoring::error_distribution::ErrorDistribution;

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
    /// 2. The target neuron has a supported saturating activation (e.g., `HARD_TANH`)
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
    pub discovery_module_stats: Vec<crate::analysis::module_weights::DiscoveryModuleStatsJson>,

    /// MCMC diagnostics: acceptance rates, proposal quality, diversity (Issue #1021).
    pub mcmc_diagnostics:
        Option<crate::analysis::diagnostics::mcmc_diagnostics::McmcDiagnosticsSummary>,
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
    pub candidate_clusters: Vec<crate::analysis::candidate_clustering::CandidateClusterJson>,
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
    /// Whether the analysis was cut short because the memory budget was
    /// approached or exceeded (Issue #1028). `false` when no budget is set.
    pub memory_budget_exceeded: bool,
    /// Whether the analysis was cancelled by the host via `cancel_analysis()`
    /// (Issue #1047). When `true`, the results are partial but valid.
    pub cancelled: bool,
    /// Whether the cancellation was specifically triggered by CRITICAL memory
    /// pressure (Issue #1099). When `true`, the host should take additional
    /// recovery actions such as clearing WASM caches and discovery buffers.
    pub memory_pressure_cancelled: bool,
    /// Current neuron fingerprints for incremental analysis (Issue #490).
    ///
    /// Callers should store these and pass them back on the next run.
    pub neuron_fingerprints: Option<
        std::collections::HashMap<String, crate::analysis::neuron_fingerprint::NeuronFingerprint>,
    >,
    /// Number of focus neurons skipped due to unchanged fingerprints (Issue #490).
    pub fingerprint_cache_hits: usize,
    /// Number of focus neurons analysed (changed or new fingerprints) (Issue #490).
    pub fingerprint_cache_misses: usize,
    /// Updated module outcome tracker for persistence (Issue #792).
    ///
    /// Contains the tracker passed in (or a default), updated with candidate counts
    /// from this run. Callers should persist this and pass it back on subsequent runs.
    pub module_outcome_tracker: crate::analysis::module_weights::ModuleOutcomeTracker,
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
