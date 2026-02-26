//! Neuron analysis module
//!
//! This module contains functions for analysing neuron candidates - identifying
//! beneficial new neurons that would reduce error.
//!
//! Note: This module was extracted from implementation.rs as part of issue #185
//! (Complete refactoring of implementation.rs monolith).
//!
//! Split into sub-modules as part of issue #598:
//! - `preparation` — Focus target filtering, neuron type maps, source ordering
//! - `evaluation` — GPU-based candidate evaluation (ReLU, activation specs)
//! - `post_processing` — Impact discounting, sorting, filtering, result assembly

mod evaluation;
mod post_processing;
mod preparation;

use crate::{AnalyzeNeuronsInput, CandidateNeuronJson};
use anyhow::Result;

// Import shared types from the analysis module structure
use crate::analysis::shared::{AnalyzeNeuronsResult, TimingScope};

// Import utilities
use crate::analysis::utils::{
    build_deadline, deadline_passed, lock_or_bail, log_analysis_start, shuffle_slice,
    verbose_enabled,
};

// Import diagnostics and rejection tracking (Issue #271)
use crate::analysis::diagnostics::{NeuronDiagnostics, TargetMap, require_unique_focus};

// Import GPU infrastructure (Issue #272, #273, #274)
use crate::analysis::gpu::{GpuAnalyzer, GpuWorkQueue};

// Import RecordCache from cache module (Issue #185)
use super::cache::RecordCache;

// Import shared helper functions from synapse module
use super::synapse::{
    MIN_GROUP_SIZE_FOR_LOCALITY, build_ordered_neurons, build_samples_for_locality_group,
    group_sources_by_locality,
};

use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Analyze neurons for a given input.
/// This is the public entry point for neuron analysis.
pub fn analyze_neurons(input: &AnalyzeNeuronsInput) -> Result<AnalyzeNeuronsResult> {
    // Validate focus_neurons before expensive pre-loading
    require_unique_focus(&input.focus_neurons, "Neuron analysis")?;

    // Pre-load all records for faster analysis (1 scan vs ~2000 scans)
    let cache = Arc::new(RecordCache::new_adaptive(&input.parquet_file)?);
    analyze_neurons_with_cache(input, cache)
}

/// Internal neuron analysis function that accepts a pre-built cache.
/// This allows sharing the cache between synapse and neuron analysis
/// in `analyze_all`.
pub(crate) fn analyze_neurons_with_cache(
    input: &AnalyzeNeuronsInput,
    cache: Arc<RecordCache>,
) -> Result<AnalyzeNeuronsResult> {
    // v0.1.134: Return ALL positive improvements.
    // NEAT-AI applies the cost-of-growth gate during evaluation.
    let threshold = 0.0;
    let ordered_neurons = build_ordered_neurons(&input.creature);

    // Build lookup maps and filter focus targets
    let prep = preparation::prepare_neuron_analysis(input, &ordered_neurons, &cache)?;

    // If no output neurons remain after filtering, return early with empty results
    if let Some(early_return) = prep.early_return {
        return Ok(early_return);
    }

    let mut focus_order = prep.focus_order;

    let helpful_map = Arc::new(Mutex::new(HashMap::<u64, CandidateNeuronJson>::new()));

    // Issue #216: NeuronDiagnostics uses DashMap internally for lock-free concurrent access.
    // No Mutex wrapper needed - the struct handles concurrency internally.
    let diagnostics = Arc::new(NeuronDiagnostics::new(&prep.unique_focus));

    // GPU timing collector (Issue #195)
    // Only collects timing data when NEAT_AI_DISCOVERY_GPU_TIMING=1 is set
    let timing_collector = Arc::new(super::shared::TimingCollector::new(
        super::utils::gpu_timing_enabled(),
    ));

    let deadline = build_deadline(input.analysis_deadline_ms);
    // GPU is always required - TypeScript layer calls check_gpu_available() and skips
    // discovery entirely on machines without GPU. Reaching here without GPU is a bug.
    assert!(
        GpuAnalyzer::gpu_is_available(),
        "Discovery logic called without GPU - check_gpu_available should have prevented this"
    );
    let gpu_used = true;
    let analysis_timed_out = Arc::new(Mutex::new(false));

    // Mark skipped neurons in diagnostics so they appear with the correct reason
    // instead of misleading reasons like NoEligibleSources.
    // Issue #216: Direct method calls - no lock needed with DashMap-based diagnostics.
    for input_uuid in &prep.skipped_input {
        diagnostics.mark_input_filtered(input_uuid);
    }
    for hidden_uuid in &prep.skipped_hidden {
        diagnostics.mark_hidden_filtered(hidden_uuid);
    }
    for constant_uuid in &prep.skipped_constant {
        diagnostics.mark_constant_filtered(constant_uuid);
    }

    // Log threshold-crossing neurons for visibility
    if verbose_enabled() && !prep.threshold_targets.is_empty() {
        tracing::debug!(
            count = prep.threshold_targets.len(),
            sample = ?prep.threshold_targets.iter().take(5).collect::<Vec<_>>(),
            "Using threshold-crossing model for STEP/BIPOLAR neurons"
        );
    }

    shuffle_slice(&mut focus_order, input.random_seed, "neuron:focus_order");

    // Log analysis start with timeout duration and randomised order
    log_analysis_start(
        "neuron",
        input.analysis_deadline_ms,
        focus_order.len(),
        &focus_order,
    );

    // Track completed focus neurons for timeout logging
    let total_focus_count = focus_order.len();
    let completed_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    let focus_order_arc = Arc::new(focus_order);
    let ordered_neurons_arc = Arc::new(ordered_neurons);
    let order_map_arc = Arc::new(prep.order_map);
    let neuron_squash_map_arc = Arc::new(prep.neuron_squash_map);

    let used_inputs_arc = Arc::new(prep.used_inputs);

    // Issue #486 / #192: Collect error values from focus target neurons for distribution analysis.
    let error_values_for_distribution = Arc::new(Mutex::new(Vec::<f32>::new()));

    // Create a shared GPU work queue ONCE before the parallel loop.
    // This eliminates the overhead of creating multiple GPU devices (one per thread).
    // All GPU operations are processed by a single dedicated thread, improving utilisation.
    // CRITICAL: The GpuAnalyzer is created INSIDE the GPU thread to avoid wgpu deadlocks.
    let gpu_queue = Arc::new(GpuWorkQueue::new()?);

    // Process each focus neuron in parallel. Deadline checks happen at the start of
    // each focus target so that once analysis for a neuron begins, we prefer to
    // complete its upstream evaluation rather than abandoning it mid-stream. This
    // gives us "vertical" timeout behaviour where some neurons complete fully even
    // if later targets are skipped when the deadline is reached.
    focus_order_arc
        .par_iter()
        .try_for_each(|target_uuid| -> Result<()> {
            if *lock_or_bail(&analysis_timed_out, "analysis_timed_out")? || deadline_passed(&deadline) {
                *lock_or_bail(&analysis_timed_out, "analysis_timed_out")? = true;
                return Ok(());
            }

            crate::watchdog::beat(format!(
                "neuron analysis → processing target {target_uuid}"
            ));

            // Use the shared GPU work queue instead of creating a new GpuAnalyzer per thread.
            // This eliminates device creation overhead and improves GPU utilisation.
            let gpu = &*gpu_queue;

            let target_records_arc = match cache.get(target_uuid.as_str()) {
                Ok(records) => records,
                Err(err) => {
                    if cfg!(debug_assertions) {
                        tracing::error!(target_uuid = %target_uuid, error = %err, "Failed to load target neuron records");
                    }
                    return Ok(());
                }
            };
            if target_records_arc.is_empty() {
                diagnostics.set_target_record_count(target_uuid, 0);
                return Ok(());
            }
            let target_records = target_records_arc.as_ref();
            diagnostics.set_target_record_count(target_uuid, target_records.len());

            // Issue #486 / #192: Collect error values for distribution analysis
            {
                let errors: Vec<f32> = target_records
                    .iter()
                    .flat_map(|r| r.errors.iter().filter(|e| e.is_finite()).copied())
                    .collect();
                if !errors.is_empty() {
                    error_values_for_distribution
                        .lock()
                        .map_err(|_| anyhow::anyhow!("Mutex poisoned (error_values_for_distribution)"))?
                        .extend(errors);
                }
            }

            // Log target neuron obs_index range for debugging sample matching
            if verbose_enabled() && !target_records.is_empty() {
                let first_obs = target_records.first().map_or(0, |r| r.obs_index);
                let last_obs = target_records.last().map_or(0, |r| r.obs_index);
                let has_errors = target_records.iter().any(|r| !r.errors.is_empty());
                tracing::trace!(
                    target_uuid = %target_uuid,
                    record_count = target_records.len(),
                    first_obs_index = first_obs,
                    last_obs_index = last_obs,
                    has_errors,
                    "Target neuron record details"
                );
            }

            // STEP/BIPOLAR are discrete targets. Add-neuron discovery for these targets was
            // removed as dead code; add-synapse is the intended mechanism.
            let is_threshold_target = neuron_squash_map_arc
                .get(target_uuid)
                .is_some_and(|squash| crate::analysis::activation::is_threshold_activation(squash));

            let target_index = match order_map_arc.get(target_uuid.as_str()) {
                Some(index) => *index,
                None => return Ok(()),
            };

            // Phase 1: Pre-filter sources and collect their records
            let sources_to_process = preparation::load_source_records(
                target_uuid,
                target_index,
                &ordered_neurons_arc,
                input,
                &used_inputs_arc,
                &cache,
                &deadline,
                &analysis_timed_out,
                &diagnostics,
            )?;

            // Check if timed out during pre-filtering
            if *lock_or_bail(&analysis_timed_out, "analysis_timed_out")? {
                return Ok(());
            }

            // Phase 2: Build samples in parallel using CPU
            let target_map = TargetMap::from_records(target_records);
            let target_map_ref = &target_map;

            // Issue #221: Sample Locality Optimisation
            let work_results: Vec<evaluation::NeuronWorkResult> = {
                let _timing = TimingScope::sample_building(&timing_collector);

                let locality_groups = group_sources_by_locality(&sources_to_process);

                if verbose_enabled() && sources_to_process.len() >= MIN_GROUP_SIZE_FOR_LOCALITY {
                    let group_sizes: Vec<usize> = locality_groups.iter().map(|g| g.sources.len()).collect();
                    let max_group = group_sizes.iter().max().copied().unwrap_or(0);
                    let avg_group = if !group_sizes.is_empty() {
                        group_sizes.iter().sum::<usize>() as f32 / group_sizes.len() as f32
                    } else {
                        0.0
                    };
                    tracing::debug!(
                        target_uuid = %target_uuid,
                        source_count = sources_to_process.len(),
                        locality_group_count = locality_groups.len(),
                        max_group_size = max_group,
                        avg_group_size = format_args!("{avg_group:.1}"),
                        "Neuron analysis locality grouping"
                    );
                }

                locality_groups
                    .par_iter()
                    .flat_map(|group| {
                        let group_results = build_samples_for_locality_group(group, target_map_ref);
                        group_results
                            .into_iter()
                            .map(|(source_uuid, samples, _record_count)| evaluation::NeuronWorkResult {
                                source_uuid,
                                samples,
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect()
            };

            // Phase 3: Batch diagnostics updates for sample building results
            for result in &work_results {
                diagnostics.record_candidate_attempt(target_uuid, !result.samples.is_empty());
                if result.samples.is_empty() {
                    diagnostics.record_no_samples(target_uuid, &result.source_uuid);
                }
            }

            // Phase 4: GPU evaluation
            if !is_threshold_target {
                let eval_ctx = evaluation::NeuronEvalContext {
                    gpu,
                    neuron_squash_map: &neuron_squash_map_arc,
                    timing_collector: &timing_collector,
                    diagnostics: &diagnostics,
                    helpful_map: &helpful_map,
                    threshold,
                };
                evaluation::evaluate_neuron_candidates(
                    &work_results,
                    target_uuid,
                    &eval_ctx,
                    &deadline,
                    &analysis_timed_out,
                )?;
            }

            // Track completion of this focus neuron for timeout reporting.
            let completed =
                completed_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            crate::watchdog::beat(format!(
                "neuron analysis → completed {completed}/{total_focus_count} (last target {target_uuid})"
            ));
            Ok(())
        })?;

    // Post-processing: impact discounting, sorting, filtering, result assembly
    let result_params = post_processing::NeuronResultParams {
        analysis_timed_out: &analysis_timed_out,
        helpful_map: &helpful_map,
        completed_count: &completed_count,
        total_focus_count,
        original_focus_count: prep.original_focus_count,
        order_map: &order_map_arc,
        neuron_type_map: &prep.neuron_type_map,
        input,
        cache: &cache,
        error_values_for_distribution: &error_values_for_distribution,
        timing_collector: &timing_collector,
        diagnostics: &diagnostics,
        gpu_used,
    };
    post_processing::build_neuron_results(&result_params)
}
