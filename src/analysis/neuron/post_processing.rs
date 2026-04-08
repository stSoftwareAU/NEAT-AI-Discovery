//! Neuron analysis post-processing — impact discounting, sorting, filtering,
//! and result assembly.
//!
//! Extracted from neuron.rs as part of issue #598.

use crate::{AnalyzeNeuronsInput, CandidateNeuronJson};
use anyhow::Result;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::analysis::cache::RecordCache;
use crate::analysis::diagnostics::{NeuronDiagnostics, compute_impact_scores_for_discounting};
use crate::analysis::gpu::GpuAnalyzer;
use crate::analysis::shared::AnalyzeNeuronsResult;
use crate::analysis::synapse::{apply_neuron_pessimism_discount, apply_prediction_calibration};
use crate::analysis::utils::{
    lock_or_bail, log_analysis_timeout, shuffle_within_top_k, verbose_enabled,
};

/// Parameters for building the final neuron analysis result.
pub(crate) struct NeuronResultParams<'a> {
    pub analysis_timed_out: &'a Arc<AtomicBool>,
    pub helpful_map: &'a Arc<Mutex<HashMap<u64, CandidateNeuronJson>>>,
    pub completed_count: &'a Arc<std::sync::atomic::AtomicUsize>,
    pub total_focus_count: usize,
    pub original_focus_count: usize,
    pub order_map: &'a Arc<HashMap<super::preparation::SharedUuid, usize>>,
    pub neuron_type_map: &'a HashMap<super::preparation::SharedUuid, String>,
    pub input: &'a AnalyzeNeuronsInput,
    pub cache: &'a Arc<RecordCache>,
    pub error_values_for_distribution: &'a [f32],
    pub timing_collector: &'a Arc<crate::analysis::shared::TimingCollector>,
    pub diagnostics: &'a Arc<NeuronDiagnostics>,
    pub gpu_used: bool,
}

/// Build the final neuron analysis result from the collected candidates.
///
/// This applies impact-based discounting, pessimism discount, filtering,
/// sorting, and assembles the result metadata.
pub(crate) fn build_neuron_results(
    params: &NeuronResultParams<'_>,
) -> Result<AnalyzeNeuronsResult> {
    let analysis_timed_out = params.analysis_timed_out.load(Ordering::Relaxed);
    let helpful_map = lock_or_bail(params.helpful_map, "helpful_map")?.clone();

    // Log timeout with completion stats (always visible, not just verbose)
    if analysis_timed_out {
        let completed = params
            .completed_count
            .load(std::sync::atomic::Ordering::Relaxed);
        log_analysis_timeout("neuron", completed, params.total_focus_count);
    }

    let mut helpful_results: Vec<CandidateNeuronJson> = helpful_map.into_values().collect();

    // Issue #128: Apply impact-based discounting and set creature-level metrics.
    apply_impact_discounting(
        &mut helpful_results,
        params.order_map,
        params.neuron_type_map,
        params.input,
        params.cache,
    );

    // Issue #557: Filter out candidates with non-positive expected_creature_score_gain.
    helpful_results.retain(|c| c.expected_creature_score_gain > 0.0);

    // Sort by expected creature score gain (highest first) - Issue #128
    helpful_results.sort_by(|a, b| {
        b.expected_creature_score_gain
            .total_cmp(&a.expected_creature_score_gain)
    });

    // Production experiment: pair "extreme" candidates with a conservative variant.
    helpful_results = crate::analysis::utils::pair_extreme_candidates_with_conservative_variants(
        helpful_results,
        None,
    );

    // Production guard rail (Dec 2025): only return candidates within sensible parameter ranges.
    helpful_results = crate::analysis::utils::filter_candidates_to_sensible_ranges(helpful_results);

    // Deadline coverage (Jan 2026): diversify within the top-K
    if params.input.analysis_deadline_ms.is_some() {
        use crate::analysis::constants::DIVERSIFY_TOP_K;
        shuffle_within_top_k(
            helpful_results.as_mut_slice(),
            params.input.random_seed,
            "neuron:candidates:top_k",
            DIVERSIFY_TOP_K,
        );
    }

    // Track candidates_found AFTER pairing but BEFORE truncation.
    let candidates_found = helpful_results.len();

    // Apply max_candidates limit (truncation)
    if let Some(limit) = params.input.max_candidates {
        helpful_results.truncate(limit);
    }

    let candidates_returned = helpful_results.len();

    let no_candidate_reasons = params.diagnostics.no_candidate_summaries();
    params.diagnostics.emit_logs();

    // Issue #486 / #192: Compute error distribution from collected target neuron error samples.
    // Issue #834: Error values are now collected lock-free via Rayon fold/reduce.
    let error_distribution =
        crate::analysis::scoring::error_distribution::ErrorDistribution::from_errors(
            params.error_values_for_distribution,
        );

    Ok(AnalyzeNeuronsResult {
        helpful_neurons: helpful_results,
        gpu_used: params.gpu_used,
        no_candidate_reasons,
        metadata: crate::analysis::shared::NeuronAnalysisMetadata {
            candidates_found,
            candidates_returned,
            timed_out: analysis_timed_out,
            completed_focus_neurons: params
                .completed_count
                .load(std::sync::atomic::Ordering::Relaxed),
            total_focus_neurons: params.original_focus_count,
            timing: params.timing_collector.finalize(),
            gpu_info: GpuAnalyzer::get_adapter_info(),
            error_distribution,
        },
    })
}

/// Apply impact-based discounting to neuron candidates.
///
/// Output neurons have impact = 1.0 (no discount).
/// Hidden neurons have impact in [0, 1] based on their weighted paths to outputs.
fn apply_impact_discounting(
    helpful_results: &mut [CandidateNeuronJson],
    order_map_arc: &Arc<HashMap<super::preparation::SharedUuid, usize>>,
    neuron_type_map: &HashMap<super::preparation::SharedUuid, String>,
    input: &AnalyzeNeuronsInput,
    cache: &Arc<RecordCache>,
) {
    let impact_scores = compute_impact_scores_for_discounting(&input.creature, cache.as_ref());
    for candidate in helpful_results.iter_mut() {
        candidate.source_neuron_index = order_map_arc
            .get(candidate.source_neuron_uuid.as_str())
            .copied();
        candidate.target_neuron_index = order_map_arc
            .get(candidate.target_neuron_uuid.as_str())
            .copied();

        let is_hidden = neuron_type_map
            .get(candidate.target_neuron_uuid.as_str())
            .is_none_or(|t| t != "output");

        let impact = if is_hidden {
            if let Some(&impact) = impact_scores.get(&candidate.target_neuron_uuid) {
                impact.clamp(0.0, 1.0)
            } else {
                // No impact score means disconnected from outputs - heavy discount
                0.1
            }
        } else {
            // Output neuron - full impact
            1.0
        };

        // Update creature-level metrics
        candidate.target_neuron_impact = impact;
        let original = candidate.expected_creature_error_reduction;
        candidate.expected_creature_error_reduction *= impact;
        candidate.expected_creature_score_gain = candidate.expected_creature_error_reduction;

        // Issue #791: Apply neuron-specific pessimism discount based on improved sample ratio.
        // Neuron candidates have a 15% success rate (vs higher synapse rates), so they
        // use more aggressive discounting parameters (lower floor, higher exponent).
        candidate.expected_creature_score_gain = apply_neuron_pessimism_discount(
            candidate.expected_creature_score_gain,
            candidate.improved_count,
            candidate.total_count,
        );

        // Issue #891: Apply neuron prediction calibration to correct ~100× overestimation.
        // Applied after pessimism discount to scale the final prediction closer to
        // observed actual gains, improving cross-type candidate ranking.
        candidate.expected_creature_score_gain = apply_prediction_calibration(
            candidate.expected_creature_score_gain,
            crate::analysis::constants::NEURON_PREDICTION_CALIBRATION,
        );

        if verbose_enabled() && is_hidden {
            tracing::trace!(
                target_uuid = %&candidate.target_neuron_uuid[..12.min(candidate.target_neuron_uuid.len())],
                impact = format_args!("{impact:.3}"),
                original_pct = format_args!("{:.4}", original * 100.0),
                discounted_pct = format_args!("{:.4}", candidate.expected_creature_score_gain * 100.0),
                "Neuron candidate impact discount applied"
            );
        }
    }
}
