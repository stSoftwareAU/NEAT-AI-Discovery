//! FFI response structs (output to NEAT-AI).
//!
//! All `*Output` types and supporting diagnostic/metadata types used at the
//! FFI boundary for serialising JSON responses.
//!
//! Organised into focused submodules:
//! - [`analysis`] — Analysis output, metadata, and diagnostic types
//! - [`gpu`] — GPU check output and timing/adapter JSON types
//! - [`export`] — Export, merge, read, and calibration output types

pub mod analysis;
pub mod export;
pub mod gpu;

// Re-export all public types for backward compatibility.
pub use analysis::*;
pub use export::*;
pub use gpu::*;

use serde::Serialize;

use super::DiscoveryErrorKind;

use super::{CoordinatedStructuralCandidateJson, RankedNeuronJson, RemovalCandidateJson};

use crate::analysis as analysis_mod;

/// JSON output from `record_discovery` function
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RecordDiscoveryOutput {
    pub success: bool,
    /// FFI schema version so callers can reject stale cached payloads (Issue #952).
    pub schema_version: String,
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

/// FFI response payload returned by the `get_version` FFI entry point —
/// the library crate version plus the FFI schema version.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GetVersionOutput {
    pub success: bool,
    pub version: String,
    /// FFI schema version so callers can reject stale cached payloads (Issue #952).
    pub schema_version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Structured error classification for retry decisions (Issue #651).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_kind: Option<DiscoveryErrorKind>,
    /// Whether this error is typically worth retrying (Issue #651).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retryable: Option<bool>,
}

/// Diversity-aware focus-selection diagnostics (Issue #1445).
///
/// Surfaces the final focus set the exploit/explore allocator chose over the
/// impact-ranked list, plus the concentration metrics and the allocation
/// diagnostics that motivated it (Issue #1662). Serialised as `focusSelection`
/// on [`RankFocusNeuronsOutput`].
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FocusSelectionJson {
    /// Selected neuron uuids, in selection order (exploitation head first, then
    /// the exploration picks) — the focus set the caller should analyse.
    pub selected: Vec<String>,
    /// Concentration ratio (max weight ÷ sum) of the **raw** roulette weights
    /// over the ranked pool. The diagnostic that exposes single-target
    /// collapse — ~0.985 on the plateaued-creature fixture.
    pub raw_weight_concentration_ratio: f32,
    /// Concentration ratio of the **selected** set's weights. Lower than the raw
    /// ratio because the exploration quota spreads budget.
    pub weight_concentration_ratio: f32,
    /// Slots filled by ranking/history exploitation (the strongest neurons).
    pub exploitation_count: usize,
    /// Slots filled by deterministic exploration rotation over the eligible tail.
    pub exploration_count: usize,
    /// The monotonic per-creature cursor that seeded exploration this pass.
    pub exploration_cursor: u64,
    /// Total eligible candidate-producing neurons available for selection.
    pub eligible_pool_size: usize,
    /// Best-effort cumulative coverage across cursors `0..=explorationCursor`.
    pub cumulative_coverage: usize,
    /// Whether drought widened the exploration quota this pass.
    pub drought_active: bool,
    /// Number of candidates considered (== `eligiblePoolSize`). Kept for logging
    /// compatibility.
    pub pool_size: usize,
}

/// FFI response payload returned by the `rank_focus_neurons` FFI entry
/// point — ranked neurons, removal candidates, coordinated structural
/// candidates, and observability fields for the chosen loading mode.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RankFocusNeuronsOutput {
    pub success: bool,
    /// FFI schema version so callers can reject stale cached payloads (Issue #952).
    pub schema_version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub neurons: Option<Vec<RankedNeuronJson>>,
    /// Diversity-aware focus selection over the ranked neurons (Issue #1445).
    /// Carries the chosen focus set, the raw vs effective concentration ratios,
    /// and whether the diversity floor or drought rotation fired. Omitted on
    /// error paths and when no ranking pass ran.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub focus_selection: Option<FocusSelectionJson>,
    /// Neurons with high error but very low impact - candidates for removal
    #[serde(skip_serializing_if = "Option::is_none")]
    pub removal_candidates: Option<Vec<RemovalCandidateJson>>,
    /// Issue #306: Coordinated structural candidates for removing constant-value neurons.
    /// When a hidden neuron has near-zero activation variance (constant output), it can be
    /// removed and its effect folded into bias adjustments for downstream neurons.
    /// Each candidate contains:
    /// - A `RemoveNeuron` operation for the constant neuron
    /// - `SetBias` operations for all downstream neurons with adjusted biases
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
    /// Aggregate rejection counts keyed by stable reason name (Issue #1142).
    ///
    /// Reuses the Issue #1129 rejection-reason vocabulary so FFI consumers can
    /// merge these counts with the `analyze_all` `metadata.rejection_breakdown`
    /// maps. Currently populated with
    /// [`REJECTION_REMOVAL_BELOW_NOISE_FLOOR`][r] counts for
    /// removal candidates dropped by the noise-floor gate; empty maps are
    /// omitted from the serialised JSON.
    ///
    /// [r]: crate::analysis::diagnostics::rejection_reasons::REJECTION_REMOVAL_BELOW_NOISE_FLOOR
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rejection_breakdown: Option<std::collections::HashMap<String, u32>>,
    /// Record loading mode chosen by the focus ranker (Issue #1172).
    /// One of `"preload"` or `"lazy"`. Omitted when no ranking pass ran
    /// (e.g. validation errors).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loading_mode: Option<String>,
    /// Reason lazy mode was selected, if any (Issue #1172). One of `"none"`,
    /// `"budget"`, or `"memory_pressure"`. Omitted when no ranking pass ran.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lazy_reason: Option<String>,
    /// Configured `NEAT_AI_DISCOVERY_FOCUS_RANKING_MEMORY_BUDGET_MB` value
    /// when set (Issue #1172). Omitted when no explicit budget was configured.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget_mb: Option<u64>,
    /// Projected in-memory size of the parquet pre-load in megabytes
    /// (Issue #1172). Omitted on error paths.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub projected_mb: Option<u64>,
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

pub(crate) fn synapse_diagnostics_json(
    summaries: &[analysis_mod::shared::SynapseNoCandidateSummary],
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
                    analysis_mod::shared::SynapseNoCandidateReason::NoEligibleSources => {
                        SynapseDiagnosticReasonJson::NoEligibleSources
                    }
                    analysis_mod::shared::SynapseNoCandidateReason::NoDiagnostics => {
                        SynapseDiagnosticReasonJson::NoDiagnostics
                    }
                    analysis_mod::shared::SynapseNoCandidateReason::NoSamples => {
                        SynapseDiagnosticReasonJson::NoSamples
                    }
                    analysis_mod::shared::SynapseNoCandidateReason::ZeroImprovement => {
                        SynapseDiagnosticReasonJson::ZeroImprovement
                    }
                    analysis_mod::shared::SynapseNoCandidateReason::BelowThreshold => {
                        SynapseDiagnosticReasonJson::BelowThreshold
                    }
                    analysis_mod::shared::SynapseNoCandidateReason::NoTargetRecords => {
                        SynapseDiagnosticReasonJson::NoTargetRecords
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
    summaries: &[analysis_mod::shared::NeuronNoCandidateSummary],
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
                    analysis_mod::shared::NeuronNoCandidateReason::NoEligibleSources => {
                        NeuronDiagnosticReasonJson::NoEligibleSources
                    }
                    analysis_mod::shared::NeuronNoCandidateReason::NoDiagnostics => {
                        NeuronDiagnosticReasonJson::NoDiagnostics
                    }
                    analysis_mod::shared::NeuronNoCandidateReason::NoSamples => {
                        NeuronDiagnosticReasonJson::NoSamples
                    }
                    analysis_mod::shared::NeuronNoCandidateReason::NotEnoughActivations => {
                        NeuronDiagnosticReasonJson::NotEnoughActivations
                    }
                    analysis_mod::shared::NeuronNoCandidateReason::WeightDegenerate => {
                        NeuronDiagnosticReasonJson::WeightDegenerate
                    }
                    analysis_mod::shared::NeuronNoCandidateReason::BelowThreshold => {
                        NeuronDiagnosticReasonJson::BelowThreshold
                    }
                    analysis_mod::shared::NeuronNoCandidateReason::HiddenNeuronFiltered => {
                        NeuronDiagnosticReasonJson::HiddenNeuronFiltered
                    }
                    analysis_mod::shared::NeuronNoCandidateReason::InputNeuronFiltered => {
                        NeuronDiagnosticReasonJson::InputNeuronFiltered
                    }
                    analysis_mod::shared::NeuronNoCandidateReason::ConstantNeuronFiltered => {
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
