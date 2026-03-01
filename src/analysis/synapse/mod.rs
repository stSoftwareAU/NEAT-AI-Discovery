//! Synapse analysis module
//!
//! This module contains the complete synapse analysis pipeline — identifying
//! beneficial new synapses, harmful existing synapses, and coordinated
//! structural changes that would reduce creature error.
//!
//! ## Key Functions
//!
//! - `analyze_synapses` — Public entry point for synapse analysis
//! - `analyze_synapses_with_cache` — Internal implementation with shared cache
//! - `analyze_synapses_with_cache_impl` — Core analysis engine (GPU-accelerated)
//! - `compute_synapse_improvement_and_count` — Core improvement calculation
//!
//! ## Sample Locality Optimisation (Issue #221)
//!
//! When analysing multiple source neurons for the same target, sources that share
//! the same obs_indices can benefit from batched sample building. This reduces
//! sample building overhead by up to 100x for creatures where input neurons share
//! the same observation indices.
//!
//! ## Module Structure (Issue #482)
//!
//! - `scoring` — Improvement calculation, saturation-aware simulation, boosting
//! - `candidate_generation` — Sample building, locality grouping, ordered neurons
//! - `gpu_evaluation` — ReLU/activation evaluation, batched GPU processing
//! - `filtering` — Candidate filtering, deduplication, truncation
//! - `target_analysis` — Per-target analysis loop (helpful, harmful, coordinated)
//! - `structural_patterns` — Coordinated structural discovery (noisy vs trusted, collapse hidden)
//! - `post_processing` — Impact discounting, sorting, diversification, metadata

mod candidate_generation;
mod filtering;
mod gpu_evaluation;
pub mod post_processing;
mod preparation;
mod scoring;
mod structural_patterns;
mod target_analysis;

// Re-export items used by code outside this module (analysis/mod.rs, neuron.rs, implementation_tests).
pub use scoring::{apply_pessimism_discount, apply_source_type_boost, apply_target_type_boost};

pub(crate) use candidate_generation::{
    MIN_GROUP_SIZE_FOR_LOCALITY, build_ordered_neurons, build_samples_for_locality_group,
    group_sources_by_locality,
};

pub(crate) use filtering::{
    ReplaceSynapseParams, deterministic_coordinated_neuron_uuid,
    expected_gain_replace_synapse_with_hidden_neuron, truncate_combined_synapse_candidate_sets,
};

pub(crate) use gpu_evaluation::{
    evaluate_all_activation_specs_batched, evaluate_relu_candidates_split,
};

pub(crate) use scoring::upsert_candidate;

#[cfg(test)]
pub(crate) use scoring::compute_candidate_dedup_key;

#[cfg(test)]
pub(crate) use scoring::compute_synapse_improvement_and_count;

// Test-only re-exports used by implementation_tests
#[cfg(test)]
pub(crate) use candidate_generation::build_samples;
#[cfg(test)]
pub(crate) use gpu_evaluation::{ActivationEvalParams, evaluate_activation_candidate};
#[cfg(test)]
pub(crate) use scoring::{
    compute_net_improvement_with_squash, compute_relu_improvement_and_count,
    compute_synapse_improvement_with_target_squash, count_improved_samples,
};

// =============================================================================
// Imports for the core pipeline
// =============================================================================

use crate::{AnalyzeSynapsesInput, CandidateSynapseJson};
use anyhow::Result;

use crate::analysis::shared::AnalyzeSynapsesResult;

use crate::analysis::diagnostics::{TargetDiagnostics, require_unique_focus};

use crate::analysis::utils::{
    build_deadline, deadline_passed, lock_or_bail, log_analysis_start, log_analysis_timeout,
    order_focus_targets,
};

use crate::analysis::gpu::{GpuAnalyzer, GpuWorkQueue};

use super::cache::RecordCache;

use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// =============================================================================
// Core Implementation
// =============================================================================

/// Internal implementation of synapse analysis with cache.
/// This is called from the public API functions below.
///
/// The `gpu_queue` parameter is mandatory - callers should create it once and reuse it
/// across multiple calls for better performance (avoids ~100ms initialisation overhead).
pub(crate) fn analyze_synapses_with_cache_impl(
    input: &AnalyzeSynapsesInput,
    cache: Arc<RecordCache>,
    gpu_queue: Arc<GpuWorkQueue>,
) -> Result<AnalyzeSynapsesResult> {
    // Phase 1: Build creature lookup maps
    let lookups = preparation::build_creature_lookups(input);

    // Phase 2: Focus target and diagnostics setup
    let unique_focus = require_unique_focus(&input.focus_neurons, "analyse_synapses")?;
    let diagnostics = Arc::new(TargetDiagnostics::new(&unique_focus));
    let timing_collector = Arc::new(super::shared::TimingCollector::new(
        super::utils::gpu_timing_enabled(),
    ));
    let deadline = build_deadline(input.analysis_deadline_ms);

    let mut focus_order: Vec<String> = unique_focus.iter().map(|s| (*s).clone()).collect();
    let focus_neuron_type_map: HashMap<&str, &str> = input
        .creature
        .neurons
        .iter()
        .map(|n| (n.uuid.as_str(), n.neuron_type.as_str()))
        .collect();
    order_focus_targets(&mut focus_order, input.random_seed, &focus_neuron_type_map);
    log_analysis_start(
        "synapse",
        input.analysis_deadline_ms,
        focus_order.len(),
        &focus_order,
    );

    let total_focus_count = focus_order.len();
    let completed_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    assert!(
        GpuAnalyzer::gpu_is_available(),
        "Discovery logic called without GPU - check_gpu_available should have prevented this"
    );

    // Phase 3: Shared result collections for parallel processing
    let collectors = SharedResultCollectors::new();

    // Phase 4: Compute constant-source threshold
    let constant_source_effect_threshold =
        preparation::compute_constant_source_threshold_from_cache(input, cache.as_ref());

    // Phase 5: Build shared context for per-target analysis
    let ctx = Arc::new(target_analysis::TargetAnalysisContext {
        ordered_neurons: Arc::new(lookups.ordered_neurons),
        order_map: Arc::new(lookups.order_map),
        neuron_index: Arc::new(lookups.neuron_index),
        existing_synapses: Arc::new(lookups.existing_synapses),
        existing_synapse_weights: Arc::new(lookups.existing_synapse_weights),
        synapses_by_target: Arc::new(lookups.synapses_by_target),
        neuron_squash_map: Arc::new(lookups.neuron_squash_map),
        neuron_type_map: Arc::new(lookups.neuron_type_map),
        input_neuron_uuids: Arc::new(lookups.input_neuron_uuids),
        used_inputs: Arc::new(lookups.used_inputs),
        neuron_bias_map: Arc::new(lookups.neuron_bias_map),
        constant_source_effect_threshold,
        diagnostics: diagnostics.clone(),
        timing_collector: timing_collector.clone(),
        deadline,
        threshold: 0.0,
    });

    // Phase 6: Process each focus neuron in parallel
    focus_order
        .par_iter()
        .try_for_each(|target_uuid| -> Result<()> {
            if *lock_or_bail(&collectors.analysis_timed_out, "analysis_timed_out")?
                || deadline_passed(&deadline)
            {
                *lock_or_bail(&collectors.analysis_timed_out, "analysis_timed_out")? = true;
                return Ok(());
            }

            let target_results = target_analysis::analyse_single_target(
                target_uuid,
                input,
                cache.as_ref(),
                &gpu_queue,
                &ctx,
            )?;

            collectors.merge_target_results(target_results)?;

            let completed =
                completed_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            crate::watchdog::beat(format!(
                "synapse analysis → completed {completed}/{total_focus_count} (last target {target_uuid})"
            ));
            Ok(())
        })?;

    // Phase 7: Collect results and build final output
    finalise_synapse_results(&FinaliseParams {
        collectors,
        completed_count,
        total_focus_count,
        diagnostics,
        timing_collector,
        input,
        cache,
        order_map: &ctx.order_map,
    })
}

/// Shared mutable state collected during parallel target analysis.
struct SharedResultCollectors {
    helpful_results: Arc<Mutex<Vec<CandidateSynapseJson>>>,
    harmful_results: Arc<Mutex<Vec<CandidateSynapseJson>>>,
    coordinated_structural_results: Arc<Mutex<Vec<crate::CoordinatedStructuralCandidateJson>>>,
    helpful_fallback: Arc<Mutex<Option<CandidateSynapseJson>>>,
    analysis_timed_out: Arc<Mutex<bool>>,
    error_values_for_distribution: Arc<Mutex<Vec<f32>>>,
    metadata_target_value_seen: Arc<std::sync::atomic::AtomicBool>,
    metadata_saturation_aware_used: Arc<std::sync::atomic::AtomicBool>,
    metadata_seen_any_input_with_records: Arc<std::sync::atomic::AtomicBool>,
    metadata_input_min_with_records: Arc<std::sync::atomic::AtomicUsize>,
    metadata_input_max_with_records: Arc<std::sync::atomic::AtomicUsize>,
}

impl SharedResultCollectors {
    fn new() -> Self {
        Self {
            helpful_results: Arc::new(Mutex::new(Vec::new())),
            harmful_results: Arc::new(Mutex::new(Vec::new())),
            coordinated_structural_results: Arc::new(Mutex::new(Vec::new())),
            helpful_fallback: Arc::new(Mutex::new(None)),
            analysis_timed_out: Arc::new(Mutex::new(false)),
            error_values_for_distribution: Arc::new(Mutex::new(Vec::new())),
            metadata_target_value_seen: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            metadata_saturation_aware_used: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            metadata_seen_any_input_with_records: Arc::new(std::sync::atomic::AtomicBool::new(
                false,
            )),
            metadata_input_min_with_records: Arc::new(std::sync::atomic::AtomicUsize::new(
                usize::MAX,
            )),
            metadata_input_max_with_records: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }

    /// Merge a single target's results into the shared collections.
    fn merge_target_results(
        &self,
        target_results: target_analysis::TargetAnalysisResults,
    ) -> Result<()> {
        if !target_results.helpful.is_empty() {
            lock_or_bail(&self.helpful_results, "helpful_results")?.extend(target_results.helpful);
        }
        if !target_results.harmful.is_empty() {
            lock_or_bail(&self.harmful_results, "harmful_results")?.extend(target_results.harmful);
        }
        if !target_results.coordinated.is_empty() {
            lock_or_bail(
                &self.coordinated_structural_results,
                "coordinated_structural_results",
            )?
            .extend(target_results.coordinated);
        }
        if !target_results.error_values.is_empty() {
            lock_or_bail(
                &self.error_values_for_distribution,
                "error_values_for_distribution",
            )?
            .extend(target_results.error_values);
        }
        if target_results.target_value_seen {
            self.metadata_target_value_seen
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }
        if target_results.saturation_aware_used {
            self.metadata_saturation_aware_used
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }
        if target_results.input_metadata.seen_any {
            self.metadata_seen_any_input_with_records
                .store(true, std::sync::atomic::Ordering::Relaxed);
            let _ = self.metadata_input_min_with_records.fetch_min(
                target_results.input_metadata.min_index,
                std::sync::atomic::Ordering::Relaxed,
            );
            let _ = self.metadata_input_max_with_records.fetch_max(
                target_results.input_metadata.max_index,
                std::sync::atomic::Ordering::Relaxed,
            );
        }
        Ok(())
    }
}

/// Parameters for the result finalisation phase.
struct FinaliseParams<'a> {
    collectors: SharedResultCollectors,
    completed_count: Arc<std::sync::atomic::AtomicUsize>,
    total_focus_count: usize,
    diagnostics: Arc<TargetDiagnostics>,
    timing_collector: Arc<super::shared::TimingCollector>,
    input: &'a AnalyzeSynapsesInput,
    cache: Arc<RecordCache>,
    order_map: &'a HashMap<String, usize>,
}

/// Collect results from shared state, apply post-processing, and build the final output.
fn finalise_synapse_results(params: &FinaliseParams<'_>) -> Result<AnalyzeSynapsesResult> {
    let c = &params.collectors;
    let analysis_timed_out = *lock_or_bail(&c.analysis_timed_out, "analysis_timed_out")?;
    let mut helpful_results =
        std::mem::take(&mut *lock_or_bail(&c.helpful_results, "helpful_results")?);
    let mut harmful_results =
        std::mem::take(&mut *lock_or_bail(&c.harmful_results, "harmful_results")?);
    let mut coordinated_structural_results = std::mem::take(&mut *lock_or_bail(
        &c.coordinated_structural_results,
        "coordinated_structural_results",
    )?);
    let mut helpful_fallback = lock_or_bail(&c.helpful_fallback, "helpful_fallback")?.take();

    if analysis_timed_out {
        let completed = params
            .completed_count
            .load(std::sync::atomic::Ordering::Relaxed);
        log_analysis_timeout("synapse", completed, params.total_focus_count);
    }

    if helpful_results.is_empty()
        && let Some(candidate) = helpful_fallback.take()
    {
        params
            .diagnostics
            .mark_candidate_selected(&candidate.to_neuron_uuid);
        helpful_results.push(candidate);
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

    // Build metadata
    let saw_any_input = c
        .metadata_seen_any_input_with_records
        .load(std::sync::atomic::Ordering::Relaxed);
    let input_min = c
        .metadata_input_min_with_records
        .load(std::sync::atomic::Ordering::Relaxed);
    let input_max = c
        .metadata_input_max_with_records
        .load(std::sync::atomic::Ordering::Relaxed);

    let error_vec = std::mem::take(&mut *lock_or_bail(
        &c.error_values_for_distribution,
        "error_values_for_distribution",
    )?);

    let metadata = post_processing::build_metadata(&post_processing::MetadataParams {
        target_value_seen: c
            .metadata_target_value_seen
            .load(std::sync::atomic::Ordering::Relaxed),
        saturation_aware_used: c
            .metadata_saturation_aware_used
            .load(std::sync::atomic::Ordering::Relaxed),
        candidates_found: pp_metrics.candidates_found,
        candidates_returned: pp_metrics.candidates_returned,
        analysis_timed_out,
        completed_focus_neurons: params
            .completed_count
            .load(std::sync::atomic::Ordering::Relaxed),
        total_focus_neurons: params.total_focus_count,
        saw_any_input,
        input_min,
        input_max,
        error_values: &error_vec,
        timing_collector: &params.timing_collector,
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

// =============================================================================
// Public API
// =============================================================================

/// Analyze synapses for a given input.
/// This is the public entry point for synapse analysis.
pub fn analyze_synapses(input: &AnalyzeSynapsesInput) -> Result<AnalyzeSynapsesResult> {
    // Validate focus_neurons before expensive pre-loading
    require_unique_focus(&input.focus_neurons, "Synapse analysis")?;

    // Pre-load all records for faster analysis (1 scan vs ~2000 scans)
    let cache = Arc::new(RecordCache::new_adaptive(&input.parquet_file)?);
    analyze_synapses_with_cache(input, cache)
}

/// Internal synapse analysis with shared cache.
/// This is called by analyze_all to share the cache between synapse and neuron analysis.
pub(crate) fn analyze_synapses_with_cache(
    input: &AnalyzeSynapsesInput,
    cache: Arc<RecordCache>,
) -> Result<AnalyzeSynapsesResult> {
    // Create GPU queue for this analysis
    let gpu_queue = Arc::new(GpuWorkQueue::new()?);
    analyze_synapses_with_cache_impl(input, cache, gpu_queue)
}

/// Test-only helper for benchmarks that need to reuse GPU queue.
/// This allows benchmarks to avoid GPU initialisation overhead across iterations.
///
/// Note: This is public for use in external test files (tests/ directory).
pub fn analyze_synapses_with_cache_and_gpu_queue(
    input: &AnalyzeSynapsesInput,
    cache: Arc<RecordCache>,
    gpu_queue: Arc<GpuWorkQueue>,
) -> Result<AnalyzeSynapsesResult> {
    analyze_synapses_with_cache_impl(input, cache, gpu_queue)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::candidate_generation::compute_obs_index_overlap;
    use super::scoring::weight_sign;
    use super::*;
    use crate::analysis::samples::{EPSILON, HelpfulSample};
    use crate::analysis::scoring::weights::MAX_OUTGOING_WEIGHT;
    use std::collections::HashSet;

    #[test]
    fn test_weight_sign() {
        assert_eq!(weight_sign(1.0), 1);
        assert_eq!(weight_sign(-1.0), -1);
        assert_eq!(weight_sign(0.0), 0);
        assert_eq!(weight_sign(0.5), 1);
        assert_eq!(weight_sign(-0.5), -1);
    }

    #[test]
    fn test_compute_obs_index_overlap() {
        let a: HashSet<u32> = [1, 2, 3, 4, 5].into_iter().collect();
        let b: HashSet<u32> = [1, 2, 3, 4, 5].into_iter().collect();
        assert!((compute_obs_index_overlap(&a, &b) - 1.0).abs() < 0.001);

        let c: HashSet<u32> = [1, 2, 3].into_iter().collect();
        let d: HashSet<u32> = [4, 5, 6].into_iter().collect();
        assert!((compute_obs_index_overlap(&c, &d) - 0.0).abs() < 0.001);

        let e: HashSet<u32> = [1, 2, 3, 4].into_iter().collect();
        let f: HashSet<u32> = [3, 4, 5, 6].into_iter().collect();
        assert!((compute_obs_index_overlap(&e, &f) - 0.5).abs() < 0.001);

        let empty: HashSet<u32> = HashSet::new();
        assert!((compute_obs_index_overlap(&a, &empty) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_target_simulation_mode_none_without_data() {
        use crate::analysis::activation::{TargetSimulationMode, get_target_simulation_mode};
        let samples = vec![HelpfulSample {
            activation: 1.0,
            avg_error: 0.1,
            target_value: None,
            target_activation: None,
        }];

        let mode = get_target_simulation_mode(&samples, Some("HARD_TANH"));
        assert!(matches!(mode, TargetSimulationMode::None));
    }

    #[test]
    fn test_target_simulation_mode_full_with_data() {
        use crate::analysis::activation::{TargetSimulationMode, get_target_simulation_mode};
        let samples = vec![HelpfulSample {
            activation: 1.0,
            avg_error: 0.1,
            target_value: Some(0.5),
            target_activation: Some(0.5),
        }];

        let mode = get_target_simulation_mode(&samples, Some("HARD_TANH"));
        assert!(matches!(mode, TargetSimulationMode::Full(_)));
    }

    #[test]
    fn test_target_simulation_mode_approximate() {
        use crate::analysis::activation::{TargetSimulationMode, get_target_simulation_mode};
        let samples = vec![HelpfulSample {
            activation: 1.0,
            avg_error: 0.1,
            target_value: None,
            target_activation: Some(0.5),
        }];

        let mode = get_target_simulation_mode(&samples, Some("HARD_TANH"));
        assert!(matches!(
            mode,
            TargetSimulationMode::ApproximateValueFromActivation(_)
        ));
    }

    // =========================================================================
    // Issue #413: Add-synapse prediction accuracy tests
    // =========================================================================

    /// Helper: compute optimal weight using least-squares formula (same as production).
    fn compute_linear_optimal_weight(samples: &[HelpfulSample]) -> f32 {
        let sum_ea: f32 = samples.iter().map(|s| s.avg_error * s.activation).sum();
        let sum_aa: f32 = samples.iter().map(|s| s.activation * s.activation).sum();
        let raw = sum_ea / (sum_aa + EPSILON);
        raw.clamp(-MAX_OUTGOING_WEIGHT, MAX_OUTGOING_WEIGHT)
    }

    /// Issue #413: Positive correlation should produce positive improvement.
    #[test]
    fn test_issue_413_positive_correlation_gives_positive_improvement() {
        let samples: Vec<HelpfulSample> = (0..100)
            .map(|i| {
                let x = (i as f32 - 50.0) / 50.0;
                HelpfulSample {
                    activation: x,
                    avg_error: 0.05 * x + 0.001 * (i as f32 * 0.1).sin(),
                    target_value: None,
                    target_activation: None,
                }
            })
            .collect();

        let baseline_sq: f32 = samples.iter().map(|s| s.avg_error * s.avg_error).sum();
        let weight = compute_linear_optimal_weight(&samples);

        let (improvement, improved, _worsened, total) =
            compute_synapse_improvement_and_count(&samples, weight, baseline_sq, None);

        assert!(
            improvement > 0.0,
            "Positive correlation should give positive improvement, got {improvement}"
        );
        assert!(
            improved > total / 2,
            "More than half of samples should be improved: {improved}/{total}"
        );
    }

    /// Issue #413: Negative correlation should produce positive improvement
    /// with negative weight.
    #[test]
    fn test_issue_413_negative_correlation_gives_positive_improvement() {
        let samples: Vec<HelpfulSample> = (0..100)
            .map(|i| {
                let x = (i as f32 - 50.0) / 50.0;
                HelpfulSample {
                    activation: x,
                    avg_error: -0.05 * x + 0.001 * (i as f32 * 0.1).sin(),
                    target_value: None,
                    target_activation: None,
                }
            })
            .collect();

        let baseline_sq: f32 = samples.iter().map(|s| s.avg_error * s.avg_error).sum();
        let weight = compute_linear_optimal_weight(&samples);
        assert!(
            weight < 0.0,
            "Anti-correlated source should have negative weight"
        );

        let (improvement, improved, _worsened, total) =
            compute_synapse_improvement_and_count(&samples, weight, baseline_sq, None);

        assert!(
            improvement > 0.0,
            "Anti-correlated source with negative weight should give positive improvement, got {improvement}"
        );
        assert!(improved > total / 2);
    }

    /// Issue #413: HARD_TANH target near saturation should NOT produce inverted
    /// predictions. This is the core bug.
    #[test]
    fn test_issue_413_hard_tanh_saturated_no_inversion() {
        let samples: Vec<HelpfulSample> = (0..100)
            .map(|i| {
                let x = (i as f32) / 100.0; // 0 to 1
                let target_value = 0.9 + 0.05 * x;
                let target_activation = target_value.clamp(-1.0, 1.0);
                HelpfulSample {
                    activation: x,
                    avg_error: 0.05,
                    target_value: Some(target_value),
                    target_activation: Some(target_activation),
                }
            })
            .collect();

        let baseline_sq: f32 = samples.iter().map(|s| s.avg_error * s.avg_error).sum();
        let weight = compute_linear_optimal_weight(&samples);

        // With saturation-aware simulation, prediction should NOT be inverted
        let (improvement, _improved, _worsened, _total) =
            compute_synapse_improvement_and_count(&samples, weight, baseline_sq, Some("HARD_TANH"));

        assert!(
            improvement >= -EPSILON,
            "Issue #413: Saturated HARD_TANH target should NOT produce inverted prediction. \
             Got improvement={improvement}, weight={weight}"
        );
    }

    /// Issue #413: Deeply saturated target should not produce inverted predictions.
    #[test]
    fn test_issue_413_deeply_saturated_not_inverted() {
        let samples: Vec<HelpfulSample> = (0..100)
            .map(|i| {
                let x = (i as f32 - 50.0) / 50.0;
                HelpfulSample {
                    activation: x,
                    avg_error: -0.5,
                    target_value: Some(2.0),      // Way beyond HARD_TANH
                    target_activation: Some(1.0), // Clamped
                }
            })
            .collect();

        let baseline_sq: f32 = samples.iter().map(|s| s.avg_error * s.avg_error).sum();
        let weight = compute_linear_optimal_weight(&samples);

        let (improvement, _improved, _worsened, _total) =
            compute_synapse_improvement_and_count(&samples, weight, baseline_sq, Some("HARD_TANH"));

        assert!(
            improvement >= -EPSILON,
            "Issue #413: Deeply saturated target should not invert. Got {improvement}"
        );
    }

    /// Issue #413: Prediction signs should be consistent between linear and
    /// saturation-aware models when target is in the linear region.
    #[test]
    fn test_issue_413_prediction_sign_consistency_in_linear_region() {
        let samples: Vec<HelpfulSample> = (0..100)
            .map(|i| {
                let x = (i as f32 - 50.0) / 50.0;
                let target_value = 0.3 * x; // Well within [-1, 1]
                HelpfulSample {
                    activation: x,
                    avg_error: 0.05 * x,
                    target_value: Some(target_value),
                    target_activation: Some(target_value.clamp(-1.0, 1.0)),
                }
            })
            .collect();

        let baseline_sq: f32 = samples.iter().map(|s| s.avg_error * s.avg_error).sum();
        let weight = compute_linear_optimal_weight(&samples);

        let (linear_imp, _, _, _) =
            compute_synapse_improvement_and_count(&samples, weight, baseline_sq, None);
        let (sat_imp, _, _, _) =
            compute_synapse_improvement_and_count(&samples, weight, baseline_sq, Some("HARD_TANH"));

        assert!(
            linear_imp > 0.0,
            "Linear model should be positive: {linear_imp}"
        );
        assert!(
            sat_imp > 0.0,
            "Saturation model should agree in linear region: {sat_imp}"
        );
    }

    /// Issue #413: Weight search over multiple candidates should find non-negative
    /// improvement for saturating targets, even when the linear-model weight fails.
    #[test]
    fn test_issue_413_weight_search_finds_non_negative_for_saturated_target() {
        let samples: Vec<HelpfulSample> = (0..100)
            .map(|i| {
                let x = (i as f32) / 100.0;
                let target_value = 0.95 + 0.04 * x;
                HelpfulSample {
                    activation: x,
                    avg_error: 0.02 * x,
                    target_value: Some(target_value),
                    target_activation: Some(target_value.clamp(-1.0, 1.0)),
                }
            })
            .collect();

        let baseline_sq: f32 = samples.iter().map(|s| s.avg_error * s.avg_error).sum();
        let base_weight = compute_linear_optimal_weight(&samples);

        // Search over scaled weights (same approach as add-neuron path)
        let scales: [f32; 9] = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, -0.5, -1.0];
        let mut best_improvement = f32::NEG_INFINITY;

        for &scale in &scales {
            let w = (base_weight * scale).clamp(-MAX_OUTGOING_WEIGHT, MAX_OUTGOING_WEIGHT);
            if w.abs() <= EPSILON {
                continue;
            }
            let (imp, _, _, _) =
                compute_synapse_improvement_and_count(&samples, w, baseline_sq, Some("HARD_TANH"));
            if imp > best_improvement {
                best_improvement = imp;
            }
        }

        assert!(
            best_improvement >= -EPSILON,
            "Issue #413: Weight search should find non-negative improvement. Best={best_improvement}"
        );
    }

    /// Issue #413: TANH target saturation should not produce inverted predictions.
    #[test]
    fn test_issue_413_tanh_saturation_not_inverted() {
        let samples: Vec<HelpfulSample> = (0..100)
            .map(|i| {
                let x = (i as f32 - 50.0) / 50.0;
                let target_value = 2.0 * x;
                HelpfulSample {
                    activation: x,
                    avg_error: 0.03 * x,
                    target_value: Some(target_value),
                    target_activation: Some(target_value.tanh()),
                }
            })
            .collect();

        let baseline_sq: f32 = samples.iter().map(|s| s.avg_error * s.avg_error).sum();
        let improvement = compute_synapse_improvement_with_target_squash(
            &samples,
            0.03,
            baseline_sq,
            Some("TANH"),
        );

        assert!(
            improvement >= -EPSILON,
            "Issue #413: TANH saturation should not invert prediction. Got {improvement}"
        );
    }

    /// Issue #413: Uncorrelated source should give near-zero improvement.
    #[test]
    fn test_issue_413_uncorrelated_source_near_zero_improvement() {
        let samples: Vec<HelpfulSample> = (0..100)
            .map(|i| {
                let x = (i as f32 - 50.0) / 50.0;
                let error = 0.1 * ((i as f32 * 7.3).sin());
                HelpfulSample {
                    activation: x,
                    avg_error: error,
                    target_value: None,
                    target_activation: None,
                }
            })
            .collect();

        let baseline_sq: f32 = samples.iter().map(|s| s.avg_error * s.avg_error).sum();
        let weight = compute_linear_optimal_weight(&samples);

        let (improvement, _, _, _) =
            compute_synapse_improvement_and_count(&samples, weight, baseline_sq, None);

        assert!(
            improvement.abs() < 0.05,
            "Uncorrelated source should give near-zero improvement, got {improvement}"
        );
    }

    // =========================================================================
    // Issue #730: Multi-weight search and improved ratio tests
    // =========================================================================

    /// Issue #730: Multi-weight search should find improvement >= single weight
    /// for noisy data where the optimal weight overshoots.
    #[test]
    fn test_issue_730_multi_weight_search_finds_better_improvement() {
        // Samples where outliers pull the optimal weight too high
        let samples: Vec<HelpfulSample> = (0..100)
            .map(|i| {
                let x = (i as f32 - 50.0) / 50.0;
                let noise = ((i as f32 * 3.7).sin()) * 0.08;
                let error = if i % 25 == 0 {
                    0.2 * x // Outlier: strong correlation
                } else {
                    0.01 * x + noise // Weak correlation + noise
                };
                HelpfulSample {
                    activation: x,
                    avg_error: error,
                    target_value: None,
                    target_activation: None,
                }
            })
            .collect();

        let baseline_sq: f32 = samples.iter().map(|s| s.avg_error * s.avg_error).sum();
        let optimal_weight = compute_linear_optimal_weight(&samples);

        // Single weight evaluation
        let (single_imp, _, _, _) =
            compute_synapse_improvement_and_count(&samples, optimal_weight, baseline_sq, None);

        // Multi-weight search
        let scales: [f32; 9] = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, -0.5, -1.0];
        let mut best_improvement = f32::NEG_INFINITY;

        for &scale in &scales {
            let w = (optimal_weight * scale).clamp(-MAX_OUTGOING_WEIGHT, MAX_OUTGOING_WEIGHT);
            if w.abs() <= EPSILON {
                continue;
            }
            let (imp, _, _, _) =
                compute_synapse_improvement_and_count(&samples, w, baseline_sq, None);
            if imp > best_improvement {
                best_improvement = imp;
            }
        }

        assert!(
            best_improvement >= single_imp,
            "Multi-weight search should find improvement >= single weight. \
             Single={single_imp:.6}, Best={best_improvement:.6}"
        );
    }

    /// Issue #730: Multi-weight search should find candidates with better
    /// improved/worsened ratio than single weight for noisy data.
    #[test]
    fn test_issue_730_multi_weight_improves_ratio() {
        let samples: Vec<HelpfulSample> = (0..100)
            .map(|i| {
                let x = (i as f32 - 50.0) / 50.0;
                let error = 0.005 * x; // Very small true correlation
                HelpfulSample {
                    activation: x,
                    avg_error: error,
                    target_value: None,
                    target_activation: None,
                }
            })
            .collect();

        let baseline_sq: f32 = samples.iter().map(|s| s.avg_error * s.avg_error).sum();
        let optimal_weight = compute_linear_optimal_weight(&samples);

        // Single weight
        let (_, single_improved, single_worsened, _) =
            compute_synapse_improvement_and_count(&samples, optimal_weight, baseline_sq, None);

        // Multi-weight: find best by improvement
        let scales: [f32; 9] = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, -0.5, -1.0];
        let mut best_imp = f32::NEG_INFINITY;
        let mut best_improved = 0u32;
        let mut best_worsened = 0u32;

        for &scale in &scales {
            let w = (optimal_weight * scale).clamp(-MAX_OUTGOING_WEIGHT, MAX_OUTGOING_WEIGHT);
            if w.abs() <= EPSILON {
                continue;
            }
            let (imp, improved, worsened, _) =
                compute_synapse_improvement_and_count(&samples, w, baseline_sq, None);
            if imp > best_imp {
                best_imp = imp;
                best_improved = improved;
                best_worsened = worsened;
            }
        }

        // Best multi-weight candidate should have at least as good a ratio
        let single_net = single_improved as i32 - single_worsened as i32;
        let best_net = best_improved as i32 - best_worsened as i32;

        assert!(
            best_net >= single_net,
            "Multi-weight should find better or equal net improvement. \
             Single: {single_improved}-{single_worsened}={single_net}, \
             Best: {best_improved}-{best_worsened}={best_net}"
        );
    }

    /// Issue #730: Candidates with more worsened than improved samples should
    /// have non-positive improvement (these should be filtered in production).
    #[test]
    fn test_issue_730_worsened_exceeds_improved_poor_score() {
        // Random noise — no real correlation
        let samples: Vec<HelpfulSample> = (0..100)
            .map(|i| {
                let x = (i as f32 - 50.0) / 50.0;
                let error = 0.1 * ((i as f32 * 7.3).sin());
                HelpfulSample {
                    activation: x,
                    avg_error: error,
                    target_value: None,
                    target_activation: None,
                }
            })
            .collect();

        let baseline_sq: f32 = samples.iter().map(|s| s.avg_error * s.avg_error).sum();

        // For uncorrelated data with various weights, when worsened > improved
        // the overall improvement should be low
        let test_weights = [0.01f32, 0.05, 0.1, -0.01, -0.05, -0.1];
        for &w in &test_weights {
            let (improvement, improved, worsened, _) =
                compute_synapse_improvement_and_count(&samples, w, baseline_sq, None);

            if worsened > improved {
                assert!(
                    improvement < 0.01,
                    "With worsened ({worsened}) > improved ({improved}), \
                     improvement should be very small, got {improvement:.6} for weight={w}"
                );
            }
        }
    }
}

// Issue #425: Implementation tests moved from implementation.rs as part of the refactoring.
// These test the core synapse analysis pipeline (GPU batch evaluation, diagnostics, etc.).
#[cfg(test)]
#[path = "../implementation_tests/mod.rs"]
mod implementation_tests;
