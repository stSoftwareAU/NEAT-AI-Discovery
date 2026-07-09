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
//! - `evaluation` — GPU-based candidate evaluation (`ReLU`, activation specs)
//! - `post_processing` — Impact discounting, sorting, filtering, result assembly

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
mod evaluation;
pub(crate) mod post_processing;
pub(crate) mod preparation;

use crate::{AnalyzeNeuronsInput, CandidateNeuronJson};
use anyhow::{Context, Result};

// Import shared types from the analysis module structure
use crate::analysis::shared::{AnalyzeNeuronsResult, TimingScope};

// Import utilities
use crate::analysis::utils::{
    build_deadline, deadline_passed, log_analysis_start, shuffle_slice, verbose_enabled,
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

use parking_lot::Mutex;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Build the squash-aware activation scan plan for a neuron-analysis phase
/// (Issue #1545).
///
/// Add-neuron candidates always insert a **hidden** neuron between a source and
/// a target, so the scan is gated by [`HiddenScanContext`]:
///
/// - When pruning is disabled ([`crate::config::hidden_squash_prune_enabled`]
///   is `false`) the full [`SquashScanPlan::full`] is used — regression-safe.
/// - The creature's currently-adopted squash families seed the history-aware
///   widening (a family the creature already uses has demonstrably succeeded).
/// - A drought (trailing-failure streak at or above
///   [`crate::config::drought_log_threshold`]) escalates the scan back to the
///   full set so pruning can never permanently starve a plateaued creature.
fn build_neuron_scan_plan(
    input: &AnalyzeNeuronsInput,
) -> crate::analysis::activation::SquashScanPlan {
    use crate::analysis::activation::{HiddenScanContext, SquashScanPlan};

    if !crate::config::hidden_squash_prune_enabled() {
        return SquashScanPlan::full();
    }

    // Squash families the creature already uses count as non-zero historical
    // success for this creature (Issue #1545). Deserialisation uppercases the
    // names; scan matching is case-insensitive.
    let successful_squashes: std::collections::HashSet<String> = input
        .creature
        .neurons
        .iter()
        .map(|n| n.squash.clone())
        .collect();

    // Escalate to the full set while in a drought so a plateaued creature widens
    // its search rather than re-treading the pruned core forever.
    let escalated = input.discovery_outcome_log.as_ref().is_some_and(|log| {
        log.consecutive_trailing_failures() >= crate::config::drought_log_threshold()
    });

    let ctx = HiddenScanContext {
        successful_squashes: &successful_squashes,
        escalated,
    };
    SquashScanPlan::for_hidden(&ctx, crate::config::max_activation_configs_per_target())
}

/// Analyze neurons for a given input.
/// This is the public entry point for neuron analysis.
pub fn analyze_neurons(input: &AnalyzeNeuronsInput) -> Result<AnalyzeNeuronsResult> {
    // Validate focus_neurons before expensive pre-loading
    require_unique_focus(&input.focus_neurons, "Neuron analysis")
        .context("neuron analysis input validation failed")?;

    // Pre-load all records for faster analysis (1 scan vs ~2000 scans)
    let cache = Arc::new(
        RecordCache::new_adaptive(&input.parquet_file)
            .context("failed to load parquet record cache for neuron analysis")?,
    );
    analyze_neurons_with_cache(input, cache)
}

/// Internal neuron analysis function that accepts a pre-built cache.
/// This allows sharing the cache between synapse and neuron analysis
/// in `analyze_all`.
pub(crate) fn analyze_neurons_with_cache(
    input: &AnalyzeNeuronsInput,
    cache: Arc<RecordCache>,
) -> Result<AnalyzeNeuronsResult> {
    let deadline = build_deadline(input.analysis_deadline_ms);
    let gpu_queue = Arc::new(
        GpuWorkQueue::new()
            .context("failed to create GPU work queue for neuron analysis")?
            .with_deadline(deadline),
    );
    analyze_neurons_with_cache_and_gpu_queue(input, cache, gpu_queue)
}

/// Neuron analysis with a shared GPU work queue (Issue #1002).
///
/// This variant accepts an externally created `GpuWorkQueue`, allowing the
/// caller to share a single GPU thread between synapse and neuron analyses
/// when they run concurrently.
pub fn analyze_neurons_with_cache_and_gpu_queue(
    input: &AnalyzeNeuronsInput,
    cache: Arc<RecordCache>,
    gpu_queue: Arc<GpuWorkQueue>,
) -> Result<AnalyzeNeuronsResult> {
    // v0.1.134: Return ALL positive improvements.
    // NEAT-AI applies the cost-of-growth gate during evaluation.
    let threshold = 0.0;
    let ordered_neurons = build_ordered_neurons(&input.creature);

    // Build lookup maps and filter focus targets
    let prep = preparation::prepare_neuron_analysis(input, &ordered_neurons, &cache)
        .context("failed to prepare neuron analysis")?;

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
    // GPU is required for analysis. Return a structured error instead of panicking
    // so the TypeScript layer receives a JSON response it can handle gracefully.
    if !GpuAnalyzer::gpu_is_available() {
        return Err(crate::ffi_types::DiscoveryError::GpuUnavailable {
            reason: "No compatible GPU adapter found on this system".to_string(),
        }
        .into());
    }
    let gpu_used = true;
    let analysis_timed_out = Arc::new(AtomicBool::new(false));

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

    // Issue #1130: surface the number of targets dropped for cooldown so
    // long-run observability can track the savings. `filter_cooldown_targets`
    // already emits an info-level log on non-zero counts; this line ties the
    // counter to the neuron analysis phase for downstream diagnostics.
    if prep.cooldown_skipped > 0 {
        tracing::debug!(
            phase = "neuron",
            cooldown_skipped = prep.cooldown_skipped,
            remaining_targets = focus_order.len(),
            "Neuron analysis preparation dropped targets in cooldown (Issue #1130)"
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

    // Issue #1164: within-batch target-failure short-circuit. Lives for the
    // duration of this orchestration call only. Shared across rayon workers
    // so different parallel target evaluations contribute to a single view.
    let within_batch_failures =
        Arc::new(crate::analysis::within_batch_failures::WithinBatchFailureTracker::new());

    let focus_order_arc = Arc::new(focus_order);
    let ordered_neurons_arc = Arc::new(ordered_neurons);
    let order_map_arc = Arc::new(prep.order_map);
    let neuron_squash_map_arc = Arc::new(prep.neuron_squash_map);

    let used_inputs_arc = Arc::new(prep.used_inputs);

    // Issue #1545: build the squash-aware activation scan plan once for the
    // whole phase. Hidden add-neuron targets scan only the pruned core /
    // history-widened squash set unless the search is escalated (drought /
    // novelty), which restores the full set.
    let scan_plan = build_neuron_scan_plan(input);

    // Issue #486 / #192: Error values collected lock-free via Rayon fold/reduce (Issue #834).

    // Process each focus neuron in parallel. Deadline checks happen at the start of
    // each focus target so that once analysis for a neuron begins, we prefer to
    // complete its upstream evaluation rather than abandoning it mid-stream. This
    // gives us "vertical" timeout behaviour where some neurons complete fully even
    // if later targets are skipped when the deadline is reached.
    // Issue #834: Use Rayon fold/reduce to collect error values lock-free.
    // Each thread accumulates its own Vec<f32>, merged after the parallel loop.
    let error_values_for_distribution: Vec<f32> = focus_order_arc
        .par_iter()
        .try_fold(
            Vec::<f32>::new,
            |mut error_acc, target_uuid| -> Result<Vec<f32>> {
            if analysis_timed_out.load(Ordering::Relaxed) || deadline_passed(&deadline) {
                analysis_timed_out.store(true, Ordering::Relaxed);
                return Ok(error_acc);
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
                    return Ok(error_acc);
                }
            };
            if target_records_arc.is_empty() {
                diagnostics.set_target_record_count(target_uuid, 0);
                return Ok(error_acc);
            }
            let target_records = target_records_arc.as_ref();
            diagnostics.set_target_record_count(target_uuid, target_records.len());

            // Issue #486 / #192: Collect error values for distribution analysis (lock-free)
            {
                let errors: Vec<f32> = target_records
                    .iter()
                    .flat_map(|r| r.errors.iter().filter(|e| e.is_finite()).copied())
                    .collect();
                if !errors.is_empty() {
                    error_acc.extend(errors);
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

            // Issue #1111: Compute target saturation before candidate generation.
            // This detects targets operating near activation bounds and adjusts
            // candidate generation accordingly.
            let target_saturation = neuron_squash_map_arc
                .get(target_uuid.as_str())
                .map_or(preparation::TargetSaturationInfo::NOT_SATURATED, |squash| {
                    preparation::compute_target_saturation(target_records, squash)
                });

            // STEP/BIPOLAR are discrete targets. Add-neuron discovery for these targets was
            // removed as dead code; add-synapse is the intended mechanism.
            let is_threshold_target = neuron_squash_map_arc
                .get(target_uuid.as_str())
                .is_some_and(|squash| crate::analysis::activation::is_threshold_activation(squash));

            let target_index = match order_map_arc.get(target_uuid.as_str()) {
                Some(index) => *index,
                None => return Ok(error_acc),
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
            )
            .with_context(|| {
                format!("failed to load source records for neuron {target_uuid}")
            })?;

            // Check if timed out during pre-filtering
            if analysis_timed_out.load(Ordering::Relaxed) {
                return Ok(error_acc);
            }

            // Phase 2: Build samples in parallel using CPU
            let target_map = TargetMap::from_records(target_records);
            let target_map_ref = &target_map;

            // Issue #221: Sample Locality Optimisation
            let work_results: Vec<evaluation::NeuronWorkResult<'_>> = {
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
                    diagnostics.record_no_samples(target_uuid, result.source_uuid);
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
                    scan_plan: &scan_plan,
                    target_saturation,
                    within_batch_failures: &within_batch_failures,
                };
                evaluation::evaluate_neuron_candidates(
                    &work_results,
                    target_uuid,
                    &eval_ctx,
                    &deadline,
                    &analysis_timed_out,
                )
                .with_context(|| {
                    format!("failed during GPU evaluation for neuron {target_uuid}")
                })?;
            }

            // Track completion of this focus neuron for timeout reporting.
            let completed =
                completed_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            crate::watchdog::beat(format!(
                "neuron analysis → completed {completed}/{total_focus_count} (last target {target_uuid})"
            ));
            Ok(error_acc)
        })
        .try_reduce(Vec::new, |mut a, b| {
            a.extend(b);
            Ok(a)
        })?;

    // Issue #1164: surface the within-batch short-circuit savings.
    let within_batch_skips = within_batch_failures.skip_count();
    if within_batch_skips > 0 {
        tracing::info!(
            phase = "neuron",
            within_batch_skipped = within_batch_skips,
            failed_targets = within_batch_failures.failed_target_count(),
            failure_limit = within_batch_failures.failure_limit(),
            "Short-circuited same-target candidates after within-batch failure (Issue #1164)"
        );
    }

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
        error_values_for_distribution: &error_values_for_distribution[..],
        timing_collector: &timing_collector,
        diagnostics: &diagnostics,
        gpu_used,
    };
    post_processing::build_neuron_results(&result_params)
}
