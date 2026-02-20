//! Internal business-logic functions for analysis FFI entry points.

use anyhow::Result;

use crate::ffi_types::*;
use crate::{analysis, focus};

pub fn analyze_parallel_internal(input_json: &str) -> Result<String> {
    let input: AnalyzeParallelInput = match serde_json::from_str::<AnalyzeParallelInput>(input_json)
    {
        Ok(value) => value,
        Err(e) => {
            let typed = DiscoveryError::InvalidInput {
                detail: format!("Failed to parse input JSON: {e}"),
            };
            let kind = typed.error_kind();
            let output = AnalyzeParallelOutput {
                success: false,
                helpful_synapses: None,
                harmful_synapses: None,
                synapse_diagnostics: None,
                synapse_gpu_used: None,
                synapse_metadata: None,
                helpful_neurons: None,
                synapse_weight_updates: None,
                coordinated_structural_candidates: None,
                candidate_clusters: None,
                neuron_diagnostics: None,
                neuron_gpu_used: None,
                neuron_metadata: None,
                neuron_fingerprints: None,
                fingerprint_cache_hits: None,
                fingerprint_cache_misses: None,
                error: Some(typed.to_string()),
                error_kind: Some(kind),
                retryable: Some(kind.is_retryable()),
            };
            return Ok(serde_json::to_string(&output)?);
        }
    };

    let combined_input = build_analyze_all_input_from_parallel(input);

    match analysis::analyze_all(&combined_input) {
        Ok(result) => {
            let synapse = result.synapse;
            let neuron = result.neuron;
            let synapse_weight_updates = synapse.as_ref().and_then(|s| {
                if s.synapse_weight_updates.is_empty() {
                    None
                } else {
                    Some(s.synapse_weight_updates.clone())
                }
            });

            let coordinated_structural_candidates = synapse.as_ref().and_then(|s| {
                if s.coordinated_structural_candidates.is_empty() {
                    None
                } else {
                    Some(s.coordinated_structural_candidates.clone())
                }
            });

            let candidate_clusters = synapse.as_ref().and_then(|s| {
                if s.candidate_clusters.is_empty() {
                    None
                } else {
                    Some(s.candidate_clusters.clone())
                }
            });

            let output = AnalyzeParallelOutput {
                success: true,
                helpful_synapses: synapse.as_ref().map(|s| s.helpful_synapses.clone()),
                harmful_synapses: synapse.as_ref().map(|s| s.harmful_synapses.clone()),
                synapse_diagnostics: synapse
                    .as_ref()
                    .and_then(|s| synapse_diagnostics_json(s.no_candidate_reasons.as_slice())),
                synapse_gpu_used: synapse.as_ref().map(|s| s.gpu_used),
                synapse_metadata: synapse.as_ref().map(|s| SynapseAnalysisMetadataJson {
                    target_value_available: s.metadata.target_value_available,
                    saturation_aware_simulation_used: s.metadata.saturation_aware_simulation_used,
                    candidates_found: s.metadata.candidates_found,
                    candidates_returned: s.metadata.candidates_returned,
                    timed_out: s.metadata.timed_out,
                    completed_focus_neurons: s.metadata.completed_focus_neurons,
                    total_focus_neurons: s.metadata.total_focus_neurons,
                    input_index_min_seen_with_records: s.metadata.input_index_min_seen_with_records,
                    input_index_max_seen_with_records: s.metadata.input_index_max_seen_with_records,
                    timing: s.metadata.timing.as_ref().map(timing_to_json),
                    gpu_info: s.metadata.gpu_info.as_ref().map(gpu_info_to_json),
                    discovery_module_stats: s.metadata.discovery_module_stats.clone(),
                }),
                helpful_neurons: neuron.as_ref().map(|n| n.helpful_neurons.clone()),
                synapse_weight_updates,
                coordinated_structural_candidates,
                candidate_clusters,
                neuron_diagnostics: neuron
                    .as_ref()
                    .and_then(|n| neuron_diagnostics_json(n.no_candidate_reasons.as_slice())),
                neuron_gpu_used: neuron.as_ref().map(|n| n.gpu_used),
                neuron_metadata: neuron.as_ref().map(|n| NeuronAnalysisMetadataJson {
                    candidates_found: n.metadata.candidates_found,
                    candidates_returned: n.metadata.candidates_returned,
                    timed_out: n.metadata.timed_out,
                    completed_focus_neurons: n.metadata.completed_focus_neurons,
                    total_focus_neurons: n.metadata.total_focus_neurons,
                    timing: n.metadata.timing.as_ref().map(timing_to_json),
                    gpu_info: n.metadata.gpu_info.as_ref().map(gpu_info_to_json),
                }),
                neuron_fingerprints: result.neuron_fingerprints,
                fingerprint_cache_hits: if result.fingerprint_cache_hits > 0 {
                    Some(result.fingerprint_cache_hits)
                } else {
                    None
                },
                fingerprint_cache_misses: if result.fingerprint_cache_misses > 0 {
                    Some(result.fingerprint_cache_misses)
                } else {
                    None
                },
                error: None,
                error_kind: None,
                retryable: None,
            };
            Ok(serde_json::to_string(&output)?)
        }
        Err(e) => {
            let (err_msg, error_kind, retryable) = error_fields_from_anyhow(&e);
            let output = AnalyzeParallelOutput {
                success: false,
                helpful_synapses: None,
                harmful_synapses: None,
                synapse_diagnostics: None,
                synapse_gpu_used: None,
                synapse_metadata: None,
                helpful_neurons: None,
                synapse_weight_updates: None,
                coordinated_structural_candidates: None,
                candidate_clusters: None,
                neuron_diagnostics: None,
                neuron_gpu_used: None,
                neuron_metadata: None,
                neuron_fingerprints: None,
                fingerprint_cache_hits: None,
                fingerprint_cache_misses: None,
                error: Some(err_msg),
                error_kind,
                retryable,
            };
            Ok(serde_json::to_string(&output)?)
        }
    }
}

pub(crate) fn build_analyze_all_input_from_parallel(
    input: AnalyzeParallelInput,
) -> AnalyzeAllInput {
    AnalyzeAllInput {
        parquet_file: input.parquet_file,
        creature: input.creature,
        focus_neurons: input.focus_neurons,
        max_synapse_candidates: input.max_synapse_candidates,
        max_neuron_candidates: input.max_neuron_candidates,
        analysis_deadline_ms: input.analysis_deadline_ms,
        include_synapse_analysis: Some(true),
        include_neuron_analysis: Some(true),
        random_seed: input.random_seed,
        previous_neuron_fingerprints: input.previous_neuron_fingerprints,
    }
}

pub fn rank_focus_neurons_internal(input_json: &str) -> Result<String> {
    let input: RankFocusNeuronsInput = match serde_json::from_str(input_json) {
        Ok(value) => value,
        Err(e) => {
            let typed = DiscoveryError::InvalidInput {
                detail: format!("Failed to parse input JSON: {e}"),
            };
            let kind = typed.error_kind();
            let output = RankFocusNeuronsOutput {
                success: false,
                neurons: None,
                removal_candidates: None,
                constant_neuron_removals: None,
                max_output_error: None,
                processed_neurons: None,
                total_neurons: None,
                duration_ms: None,
                error: Some(typed.to_string()),
                error_kind: Some(kind),
                retryable: Some(kind.is_retryable()),
            };
            return Ok(serde_json::to_string(&output)?);
        }
    };

    match focus::rank_focus_neurons(
        &input.parquet_file,
        &input.creature,
        input.max_results,
        input.cost_of_growth,
    ) {
        Ok(stats) => {
            let neurons: Vec<RankedNeuronJson> = stats
                .neurons
                .into_iter()
                .map(|neuron| RankedNeuronJson {
                    neuron_uuid: neuron.neuron_uuid,
                    total_error: neuron.total_error,
                    impact: neuron.impact,
                    mean_activation: neuron.mean_activation,
                    activation_weighted_impact: neuron.activation_weighted_impact,
                })
                .collect();
            let removal_candidates: Vec<RemovalCandidateJson> = stats
                .removal_candidates
                .into_iter()
                .map(|c| RemovalCandidateJson {
                    neuron_uuid: c.neuron_uuid,
                    total_error: c.total_error,
                    impact: c.impact,
                    mean_activation: c.mean_activation,
                    activation_weighted_impact: c.activation_weighted_impact,
                    incoming_synapses: c.incoming_synapses,
                    outgoing_synapses: c.outgoing_synapses,
                    removal_savings: c.removal_savings,
                    expected_error_reduction: c.expected_error_reduction,
                    reason: c.reason,
                })
                .collect();
            let (error_kind, retryable) = no_error_fields();
            let output = RankFocusNeuronsOutput {
                success: true,
                neurons: Some(neurons),
                removal_candidates: if removal_candidates.is_empty() {
                    None
                } else {
                    Some(removal_candidates)
                },
                // Issue #306: Return constant neuron removal candidates
                constant_neuron_removals: if stats.constant_neuron_removals.is_empty() {
                    None
                } else {
                    Some(stats.constant_neuron_removals)
                },
                max_output_error: Some(stats.max_output_error),
                processed_neurons: Some(stats.processed_neurons),
                total_neurons: Some(stats.total_neurons),
                duration_ms: Some(stats.duration_ms.min(u64::MAX as u128) as u64),
                error: None,
                error_kind,
                retryable,
            };
            Ok(serde_json::to_string(&output)?)
        }
        Err(e) => {
            let (err_msg, error_kind, retryable) = error_fields_from_anyhow(&e);
            let output = RankFocusNeuronsOutput {
                success: false,
                neurons: None,
                removal_candidates: None,
                constant_neuron_removals: None,
                max_output_error: None,
                processed_neurons: None,
                total_neurons: None,
                duration_ms: None,
                error: Some(err_msg),
                error_kind,
                retryable,
            };
            Ok(serde_json::to_string(&output)?)
        }
    }
}

/// Returns a calibration summary from discovery history (Issue #605).
///
/// Takes JSON input containing a serialised `DiscoveryHistory` and returns
/// calibration metrics (MAE, bias, calibration factor) per module/candidate type.
pub fn get_calibration_summary_internal(input_json: &str) -> Result<String> {
    let input: CalibrationSummaryInput = match serde_json::from_str(input_json) {
        Ok(input) => input,
        Err(e) => {
            let typed = DiscoveryError::InvalidInput {
                detail: format!("Failed to parse input JSON: {e}"),
            };
            let kind = typed.error_kind();
            let output = CalibrationSummaryOutput {
                success: false,
                calibration_summary: vec![],
                error: Some(typed.to_string()),
                error_kind: Some(kind),
                retryable: Some(kind.is_retryable()),
            };
            return Ok(serde_json::to_string(&output)?);
        }
    };

    let history: crate::discovery_history::DiscoveryHistory =
        match serde_json::from_str(&input.discovery_history) {
            Ok(h) => h,
            Err(e) => {
                let typed = DiscoveryError::InvalidInput {
                    detail: format!("Failed to parse discovery history: {e}"),
                };
                let kind = typed.error_kind();
                let output = CalibrationSummaryOutput {
                    success: false,
                    calibration_summary: vec![],
                    error: Some(typed.to_string()),
                    error_kind: Some(kind),
                    retryable: Some(kind.is_retryable()),
                };
                return Ok(serde_json::to_string(&output)?);
            }
        };

    let summary = history.calibration_summary();
    let (error_kind, retryable) = no_error_fields();
    let output = CalibrationSummaryOutput {
        success: true,
        calibration_summary: summary,
        error: None,
        error_kind,
        retryable,
    };
    Ok(serde_json::to_string(&output)?)
}
