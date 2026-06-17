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

/// FFI response payload returned by
/// [`crate::ffi_internal::analyze_parallel_internal`] — the helpful/harmful
/// synapse and neuron candidates plus observability metadata serialised to
/// the TypeScript host.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AnalyzeParallelOutput {
    pub success: bool,
    /// FFI schema version so callers can reject stale cached payloads (Issue #952).
    pub schema_version: String,
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
    /// Whether the analysis was cut short because the Rust-side memory usage
    /// approached or exceeded the configured `maxAnalysisMemoryMb` (Issue #1028).
    ///
    /// When `true`, the results are partial — some focus neurons or post-processing
    /// steps may have been skipped. Always `false` when no budget is set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_budget_exceeded: Option<bool>,
    /// Whether the analysis was cancelled by the host via `cancel_analysis()`
    /// (Issue #1047). When `true`, the result is partial but valid — the host
    /// requested graceful shutdown (e.g. SIGTERM).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cancelled: Option<bool>,
    /// Whether the cancellation was specifically triggered by CRITICAL memory
    /// pressure (Issue #1099). When `true`, the host should take additional
    /// recovery actions such as clearing WASM caches and discovery buffers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_pressure_cancelled: Option<bool>,
    /// When set, this pass was gated by a host-environment check (memory budget,
    /// memory pressure, or missing GPU) and never evaluated the creature
    /// (Issue #1421). Such a pass returns 0 candidates but is NOT evidence of
    /// search exhaustion — the host must exclude it from drought / target
    /// cooldown / module starvation accounting.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub environmentally_disabled: Option<analysis::EnvironmentalDisableReason>,
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
    /// True when synapse analysis was curtailed by the deadline: it timed out
    /// **and** left at least one focus neuron unanalysed (Issue #1409).
    ///
    /// Lets the GRQ layer detect synapse starvation programmatically without
    /// recomputing `timedOut && completedFocusNeurons < totalFocusNeurons`.
    pub starved: bool,
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
    /// MCMC diagnostics: acceptance rates, proposal quality, diversity (Issue #1021).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mcmc_diagnostics: Option<McmcDiagnosticsJson>,
    /// Structured rejection-reason breakdown (Issue #1129).
    ///
    /// Keyed by the stable reason names in
    /// `analysis::diagnostics::rejection_reasons`. Populated even when
    /// `candidatesReturned == 0`.
    #[serde(skip_serializing_if = "std::collections::HashMap::is_empty")]
    pub rejection_breakdown: std::collections::HashMap<String, u32>,
    /// One-sentence summary naming the dominant rejection reason (Issue #1129).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_level_summary: Option<String>,
    /// Per-change-type calibration correction factors derived from the
    /// failure cache (Issue #1131).
    ///
    /// Keyed by the stable change-type identifiers (`add-neurons`,
    /// `add-synapses`, `coordinated-structural`, ...). Values are scalar
    /// correction factors that multiplied the compiled
    /// `*_PREDICTION_CALIBRATION` constants for this run. Empty when no
    /// failure cache was supplied.
    #[serde(skip_serializing_if = "std::collections::HashMap::is_empty")]
    pub calibration_corrections: std::collections::HashMap<String, f32>,
    /// Current creature-level discovery mode (Issue #1132).
    ///
    /// `"normal"` under ordinary operation, `"conservative"` when the rolling
    /// success rate has fallen below the configured threshold and the
    /// pipeline has biased the candidate mix away from high-failure-rate
    /// structural modules.
    pub discovery_mode: analysis::discovery_mode::DiscoveryMode,
    /// Rolling success rate over the most recent
    /// [`analysis::discovery_mode::ROLLING_WINDOW`] discovery passes
    /// (Issue #1132).
    pub rolling_success_rate: f32,
    /// Drought diagnostic payload (Issue #1202).
    ///
    /// Populated only when the trailing-failure streak crosses the configured
    /// `NEAT_AI_DISCOVERY_DROUGHT_LOG_THRESHOLD` (default 5). Carries the same
    /// suppression-layer signals as the structured warn log so the controller
    /// can react without scraping logs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub drought_diagnostic: Option<analysis::drought_diagnostic::DroughtDiagnostic>,
    /// Creature-level drought alarm payload (Issue #1424).
    ///
    /// Populated only on the single pass where the creature's epochs-since-
    /// last-acceptance crosses `NEAT_AI_DISCOVERY_DROUGHT_ALARM_EPOCHS`
    /// (default 100). Carries the creature uuid, epochs since the last
    /// acceptance, and the environmental-vs-search-exhaustion classification so
    /// the controller can raise a "weeks-long drought" alert without scraping
    /// logs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub creature_drought_alarm: Option<analysis::creature_drought_alarm::CreatureDroughtAlarm>,
}

/// MCMC diagnostics summary for the analysis output JSON (Issue #1021).
///
/// Reports acceptance rates per candidate type, proposal quality distribution,
/// and source/target diversity metrics for chain mixing analysis.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct McmcDiagnosticsJson {
    /// Synapse candidate acceptance rate.
    pub synapse_acceptance: AcceptanceRateJson,
    /// Neuron candidate acceptance rate.
    pub neuron_acceptance: AcceptanceRateJson,
    /// Coordinated candidate acceptance rate.
    pub coordinated_acceptance: AcceptanceRateJson,
    /// Total candidates proposed across all types.
    pub total_proposed: u32,
    /// Total candidates accepted across all types.
    pub total_accepted: u32,
    /// Overall acceptance rate (accepted / proposed).
    pub overall_acceptance_rate: f32,
    /// Distribution of improvement values among accepted candidates.
    /// Only present when verbose mode is enabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proposal_quality: Option<ProposalQualityJson>,
    /// Source/target diversity among evaluated vs accepted candidates.
    /// Only present when verbose mode is enabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diversity: Option<DiversityMetricJson>,
    /// Total prediction-vs-actual calibration mismatches detected in the
    /// failure cache (Issue #1165). Surfaced so operators can see the rate at
    /// a glance.
    pub calibration_miss_count: u32,
    /// Per-entry calibration mismatch records (Issue #1165). Only present in
    /// verbose mode and capped at the internal `CALIBRATION_MISS_VEC_CAP`
    /// (currently 1000) to bound memory.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub calibration_misses: Option<Vec<CalibrationMissEntryJson>>,
}

/// JSON form of a single calibration mismatch (Issue #1165).
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CalibrationMissEntryJson {
    pub change_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_squash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub variant_key: Option<String>,
    pub expected: f32,
    pub actual: f32,
    pub ratio: f32,
}

/// Acceptance rate for a single candidate type (Issue #1021).
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AcceptanceRateJson {
    pub proposed: u32,
    pub accepted: u32,
    pub rate: f32,
}

/// Distribution statistics for improvement values (Issue #1021).
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProposalQualityJson {
    pub count: usize,
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub median: f32,
}

/// Source/target diversity metrics (Issue #1021).
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DiversityMetricJson {
    pub evaluated_unique_sources: usize,
    pub evaluated_unique_targets: usize,
    pub accepted_unique_sources: usize,
    pub accepted_unique_targets: usize,
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
    /// True when neuron analysis was curtailed by the deadline: it timed out
    /// **and** left at least one focus neuron unanalysed (Issue #1409).
    ///
    /// Lets the GRQ layer detect neuron starvation programmatically without
    /// recomputing `timedOut && completedFocusNeurons < totalFocusNeurons`.
    pub starved: bool,
    /// GPU timing data for performance diagnostics (Issue #195).
    /// Only present when `NEAT_AI_DISCOVERY_GPU_TIMING=1` is set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timing: Option<AnalysisTimingJson>,
    /// Information about the GPU adapter used (Issue #228).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_info: Option<GpuAdapterInfoJson>,
    /// Structured rejection-reason breakdown (Issue #1129).
    #[serde(skip_serializing_if = "std::collections::HashMap::is_empty")]
    pub rejection_breakdown: std::collections::HashMap<String, u32>,
    /// One-sentence summary naming the dominant rejection reason (Issue #1129).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_level_summary: Option<String>,
    /// Per-change-type calibration correction factors (Issue #1131). See
    /// `SynapseAnalysisMetadataJson::calibration_corrections` for full docs.
    #[serde(skip_serializing_if = "std::collections::HashMap::is_empty")]
    pub calibration_corrections: std::collections::HashMap<String, f32>,
    /// Current creature-level discovery mode (Issue #1132). See
    /// `SynapseAnalysisMetadataJson::discovery_mode` for full docs.
    pub discovery_mode: analysis::discovery_mode::DiscoveryMode,
    /// Rolling success rate over the most recent discovery passes
    /// (Issue #1132).
    pub rolling_success_rate: f32,
    /// Drought diagnostic payload (Issue #1202). See
    /// `SynapseAnalysisMetadataJson::drought_diagnostic` for full docs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub drought_diagnostic: Option<analysis::drought_diagnostic::DroughtDiagnostic>,
    /// Creature-level drought alarm payload (Issue #1424). See
    /// `SynapseAnalysisMetadataJson::creature_drought_alarm` for full docs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub creature_drought_alarm: Option<analysis::creature_drought_alarm::CreatureDroughtAlarm>,
}

// ============================================================================
// Diagnostic types
// ============================================================================

/// Per-target diagnostic surfaced in
/// [`AnalyzeParallelOutput::synapse_diagnostics`] when synapse analysis
/// produced no candidate for a focus neuron, explaining why.
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

/// Stable reason code carried by [`SynapseDiagnosticJson::reason`] explaining
/// why synapse analysis produced no candidate for a given focus neuron.
#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SynapseDiagnosticReasonJson {
    NoEligibleSources,
    NoDiagnostics,
    NoSamples,
    ZeroImprovement,
    BelowThreshold,
    /// Target neuron has zero activation records in the Parquet file
    /// (recording may have timed out) (Issue #1101).
    NoTargetRecords,
}

/// Optional detail attached to [`SynapseDiagnosticJson::detail`] — extra
/// per-source observability for the closest-rejected synapse candidate.
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

/// Per-target diagnostic surfaced in
/// [`AnalyzeParallelOutput::neuron_diagnostics`] when neuron analysis produced
/// no candidate for a focus neuron, explaining why.
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

/// Stable reason code carried by [`NeuronDiagnosticJson::reason`] explaining
/// why neuron analysis produced no candidate for a given focus neuron.
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

/// Optional detail attached to [`NeuronDiagnosticJson::detail`] — extra
/// per-source observability for the closest-rejected neuron candidate.
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
// MCMC diagnostics conversion (Issue #1021)
// ============================================================================

/// Convert internal MCMC diagnostics summary to JSON output format.
pub(crate) fn mcmc_to_json(
    summary: &crate::analysis::diagnostics::mcmc_diagnostics::McmcDiagnosticsSummary,
) -> McmcDiagnosticsJson {
    McmcDiagnosticsJson {
        synapse_acceptance: acceptance_snapshot_to_json(&summary.synapse_acceptance),
        neuron_acceptance: acceptance_snapshot_to_json(&summary.neuron_acceptance),
        coordinated_acceptance: acceptance_snapshot_to_json(&summary.coordinated_acceptance),
        total_proposed: summary.total_proposed,
        total_accepted: summary.total_accepted,
        overall_acceptance_rate: summary.overall_acceptance_rate(),
        proposal_quality: summary
            .proposal_quality
            .as_ref()
            .map(|q| ProposalQualityJson {
                count: q.count,
                min: q.min,
                max: q.max,
                mean: q.mean,
                median: q.median,
            }),
        diversity: summary.diversity.as_ref().map(|d| DiversityMetricJson {
            evaluated_unique_sources: d.evaluated_unique_sources,
            evaluated_unique_targets: d.evaluated_unique_targets,
            accepted_unique_sources: d.accepted_unique_sources,
            accepted_unique_targets: d.accepted_unique_targets,
        }),
        calibration_miss_count: summary.calibration_miss_count,
        calibration_misses: summary.calibration_misses.as_ref().map(|misses| {
            misses
                .iter()
                .map(|m| CalibrationMissEntryJson {
                    change_type: m.change_type.clone(),
                    target_squash: m.target_squash.clone(),
                    variant_key: m.variant_key.clone(),
                    expected: m.expected,
                    actual: m.actual,
                    ratio: m.ratio,
                })
                .collect()
        }),
    }
}

fn acceptance_snapshot_to_json(
    snap: &crate::analysis::diagnostics::mcmc_diagnostics::AcceptanceSnapshot,
) -> AcceptanceRateJson {
    AcceptanceRateJson {
        proposed: snap.proposed,
        accepted: snap.accepted,
        rate: snap.rate(),
    }
}
