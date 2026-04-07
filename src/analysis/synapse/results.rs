//! Synapse analysis result finalisation
//!
//! Assembles the final `AnalyzeSynapsesResult` from merged per-target results,
//! applying post-processing (impact discounting, sorting, diversification) and
//! building metadata.

use crate::AnalyzeSynapsesInput;
use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::analysis::diagnostics::TargetDiagnostics;
use crate::analysis::shared::AnalyzeSynapsesResult;
use crate::analysis::utils::log_analysis_timeout;

use super::metadata::MergedResults;
use super::post_processing;
use super::structural_patterns;
use crate::analysis::cache::RecordCache;

/// Parameters for the result finalisation phase.
pub(super) struct FinaliseParams<'a> {
    pub collectors: MergedResults,
    pub completed_count: Arc<AtomicUsize>,
    pub total_focus_count: usize,
    pub diagnostics: Arc<TargetDiagnostics>,
    pub timing_collector: Arc<crate::analysis::shared::TimingCollector>,
    pub input: &'a AnalyzeSynapsesInput,
    pub cache: Arc<RecordCache>,
    pub order_map: &'a HashMap<String, usize>,
    /// Issue #1021: MCMC diagnostics summary for inclusion in metadata.
    pub mcmc_summary: crate::analysis::diagnostics::mcmc_diagnostics::McmcDiagnosticsSummary,
}

/// Collect results from merged state, apply post-processing, and build the final output.
pub(super) fn finalise_synapse_results(
    params: FinaliseParams<'_>,
) -> Result<AnalyzeSynapsesResult> {
    let c = params.collectors;
    let analysis_timed_out = c.analysis_timed_out;
    let mut helpful_results = c.helpful_results;
    let mut harmful_results = c.harmful_results;
    let mut coordinated_structural_results = c.coordinated_structural_results;
    let error_values = c.error_values_for_distribution;

    if analysis_timed_out {
        let completed = params.completed_count.load(Ordering::Relaxed);
        log_analysis_timeout("synapse", completed, params.total_focus_count);
    }

    // Collapse 1-in/1-out hidden neurons into direct synapses (Issue #425)
    let collapse_candidates =
        structural_patterns::detect_collapsible_hidden_neurons(params.input, params.cache.as_ref());
    coordinated_structural_results.extend(collapse_candidates);

    // Post-processing: impact discounting, sorting, diversification, truncation
    let pp_metrics = post_processing::apply_post_processing(
        &mut helpful_results,
        &mut harmful_results,
        &mut coordinated_structural_results,
        params.input,
        params.cache.as_ref(),
        params.order_map,
    );

    let no_candidate_reasons = params.diagnostics.no_candidate_summaries();
    params.diagnostics.emit_logs();

    // Build metadata from atomics
    let m = &c.metadata;
    let saw_any_input = m.seen_any_input_with_records.load(Ordering::Relaxed);
    let input_min = m.input_min_with_records.load(Ordering::Relaxed);
    let input_max = m.input_max_with_records.load(Ordering::Relaxed);

    let metadata = post_processing::build_metadata(&post_processing::MetadataParams {
        target_value_seen: m.target_value_seen.load(Ordering::Relaxed),
        saturation_aware_used: m.saturation_aware_used.load(Ordering::Relaxed),
        candidates_found: pp_metrics.candidates_found,
        candidates_returned: pp_metrics.candidates_returned,
        analysis_timed_out,
        completed_focus_neurons: params.completed_count.load(Ordering::Relaxed),
        total_focus_neurons: params.total_focus_count,
        saw_any_input,
        input_min,
        input_max,
        error_values: &error_values,
        timing_collector: &params.timing_collector,
        mcmc_summary: Some(params.mcmc_summary),
    });

    Ok(AnalyzeSynapsesResult {
        helpful_synapses: helpful_results,
        harmful_synapses: harmful_results,
        synapse_weight_updates: Vec::new(),
        coordinated_structural_candidates: coordinated_structural_results,
        candidate_clusters: Vec::new(),
        gpu_used: true,
        no_candidate_reasons,
        metadata,
    })
}
