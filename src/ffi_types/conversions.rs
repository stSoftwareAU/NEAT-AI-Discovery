//! Conversion helpers between internal analysis types and FFI JSON types.

use crate::analysis;

use super::diagnostics::{
    AnalysisTimingJson, CpuTimingBreakdownJson, GpuAdapterInfoJson, GpuTimingBreakdownJson,
    NeuronDiagnosticDetailJson, NeuronDiagnosticJson, NeuronDiagnosticReasonJson, ShaderTimingJson,
    SynapseDiagnosticDetailJson, SynapseDiagnosticJson, SynapseDiagnosticReasonJson,
};

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
