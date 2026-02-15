//! Neuron analysis module
//!
//! This module contains functions for analysing neuron candidates - identifying
//! beneficial new neurons that would reduce error.
//!
//! Note: This module was extracted from implementation.rs as part of issue #185
//! (Complete refactoring of implementation.rs monolith).

use crate::types::DiscoverRecord;
use crate::{AnalyzeNeuronsInput, CandidateNeuronJson};
use anyhow::Result;

// Import shared types from the analysis module structure
use crate::analysis::shared::{AnalyzeNeuronsResult, TimingScope};

// Import activation functions from the dedicated activation module (Issue #266, #238)
use crate::analysis::activation::is_threshold_activation;

// Import utilities
use crate::analysis::utils::{
    OrderedNeuron, build_deadline, deadline_passed, lock_or_bail, log_analysis_start,
    log_analysis_timeout, order_eligible_sources, parse_input_index, shuffle_slice,
    shuffle_within_top_k, verbose_enabled,
};

// Import sample data structures (Issue #269)
use crate::analysis::samples::{EPSILON, HelpfulSample, compute_source_variance_discount};

// Import pessimism discount (Issue #506)
use crate::analysis::synapse::apply_pessimism_discount;

// Import diagnostics and rejection tracking (Issue #271)
use crate::analysis::diagnostics::{
    FocusTargetFilterResult, NeuronDiagnostics, TargetMap, compute_impact_scores_for_discounting,
    filter_focus_targets_for_neuron_analysis, require_unique_focus,
};

// Import GPU infrastructure (Issue #272, #273, #274)
use crate::analysis::gpu::{GpuAnalyzer, GpuWorkQueue};

// Import shared types for results
use crate::analysis::shared::{NeuronNoCandidateReason, NeuronNoCandidateSummary};

// Import RecordCache from cache module (Issue #185)
use super::cache::RecordCache;

// Import shared helper functions from synapse module
// These functions are used by both synapse and neuron analysis
// Issue #201: evaluate_all_activation_specs_batched replaces the loop over evaluate_activation_candidate
use super::synapse::{
    MIN_GROUP_SIZE_FOR_LOCALITY, build_ordered_neurons, build_samples_for_locality_group,
    evaluate_all_activation_specs_batched, evaluate_relu_candidates_split,
    group_sources_by_locality, upsert_candidate,
};

use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
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

    // Build a lookup map for neuron squash functions to identify discrete targets
    let neuron_squash_map: HashMap<String, String> = input
        .creature
        .neurons
        .iter()
        .map(|n| (n.uuid.clone(), n.squash.clone()))
        .collect();

    // Build a comprehensive lookup map for ALL neuron UUIDs to their types.
    // This includes: input neurons (from creature.input count) and all neurons from
    // creature.neurons (hidden, output, constant). If a UUID is not in this map,
    // it's an invalid UUID (bug in the caller).
    //
    // Only output neurons should be targets for add-neuron candidates because:
    // - Output neuron errors directly affect creature score
    // - Hidden neuron errors are backpropagated approximations that don't correlate
    //   reliably with actual output error reduction
    // - Input neurons are observation sources, not computation nodes
    let mut neuron_type_map: HashMap<String, String> = HashMap::new();

    // Add input neurons (they're not in creature.neurons, only represented by creature.input count)
    for input_index in 0..input.creature.input {
        neuron_type_map.insert(format!("input-{input_index}"), "input".to_string());
    }

    // Add all neurons from creature.neurons (hidden, output, constant)
    for neuron in &input.creature.neurons {
        neuron_type_map.insert(neuron.uuid.clone(), neuron.neuron_type.clone());
    }

    // Log creature configuration for debugging data issues
    if verbose_enabled() {
        let non_input_count = input.creature.neurons.len();
        let total_neurons = input.creature.input + non_input_count;
        tracing::debug!(
            input_neurons = input.creature.input,
            input_max_index = input.creature.input.saturating_sub(1),
            non_input_count,
            total_neurons,
            "Neuron analysis creature config"
        );

        // Verify input neurons exist in parquet by checking a sample
        if input.creature.input > 0 {
            match cache.get("input-0") {
                Ok(records) => {
                    tracing::debug!(
                        neuron = "input-0",
                        record_count = records.len(),
                        "Parquet data check"
                    );
                    if !records.is_empty() {
                        let first = &records[0];
                        let last = &records[records.len() - 1];
                        tracing::debug!(
                            neuron = "input-0",
                            first_obs_index = first.obs_index,
                            last_obs_index = last.obs_index,
                            first_activation = format_args!("{:.4}", first.activation),
                            "Parquet data check obs_index range"
                        );
                    }
                }
                Err(err) => {
                    tracing::debug!(
                        neuron = "input-0",
                        error = %err,
                        "Parquet data check failed"
                    );
                }
            }

            // Also check a middle input neuron
            let mid_input = input.creature.input / 2;
            let mid_uuid = format!("input-{mid_input}");
            match cache.get(&mid_uuid) {
                Ok(records) => {
                    tracing::debug!(
                        neuron = %mid_uuid,
                        record_count = records.len(),
                        "Parquet data check"
                    );
                }
                Err(err) => {
                    tracing::debug!(
                        neuron = %mid_uuid,
                        error = %err,
                        "Parquet data check failed"
                    );
                }
            }
        }
    }

    let order_map: HashMap<String, usize> = ordered_neurons
        .iter()
        .map(|neuron| (neuron.uuid.clone(), neuron.index))
        .collect();

    let unique_focus = require_unique_focus(&input.focus_neurons, "analyse_neurons")?;

    let helpful_map = Arc::new(Mutex::new(HashMap::<u64, CandidateNeuronJson>::new()));

    // Issue #216: NeuronDiagnostics uses DashMap internally for lock-free concurrent access.
    // No Mutex wrapper needed - the struct handles concurrency internally.
    let diagnostics = Arc::new(NeuronDiagnostics::new(&unique_focus));

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

    // Filter focus neurons to valid add-neuron targets.
    //
    // Rationale: Add-neuron candidates predict error reduction at the target neuron.
    // For OUTPUT neurons, this directly corresponds to creature score improvement.
    // For HIDDEN neurons, the backpropagated error is an approximation that doesn't
    // reliably translate to actual output error reduction - we've observed 100%
    // failure rates when targeting hidden neurons.
    // For INPUT neurons, they are observation sources, not computation nodes - they
    // have no activation function or error to reduce.
    //
    // Non-output neurons are skipped here but still tracked in diagnostics so the
    // caller knows they were received but filtered out (with the correct reason).
    let original_focus_count = unique_focus.len();
    // Optional production experiment (29-Dec-2025): allow callers to force output-only
    // focus targets for add-neuron analysis.
    //
    // This is disabled by default for backwards compatibility (existing regression tests
    // and older pipelines expect hidden targets to be analysed with impact discounting).
    //
    // Enable by setting `NEAT_AI_DISCOVERY_NEURON_TARGETS_OUTPUT_ONLY=1`.
    let output_only_targets = std::env::var("NEAT_AI_DISCOVERY_NEURON_TARGETS_OUTPUT_ONLY").is_ok();
    let FocusTargetFilterResult {
        mut focus_order,
        skipped_hidden,
        skipped_input,
        skipped_constant,
        threshold_targets,
    } = filter_focus_targets_for_neuron_analysis(
        &unique_focus,
        &neuron_type_map,
        &neuron_squash_map,
        output_only_targets,
    );

    // Log when non-output neurons are filtered out
    let total_skipped = skipped_hidden.len() + skipped_input.len() + skipped_constant.len();
    if total_skipped > 0 {
        // Build a summary of skipped neuron types
        let mut skipped_parts: Vec<String> = Vec::new();
        if !skipped_input.is_empty() {
            skipped_parts.push(format!(
                "Input: {:?}",
                skipped_input.iter().take(5).collect::<Vec<_>>()
            ));
        }
        if !skipped_hidden.is_empty() {
            skipped_parts.push(format!(
                "Hidden: {:?}",
                skipped_hidden.iter().take(5).collect::<Vec<_>>()
            ));
        }
        if !skipped_constant.is_empty() {
            skipped_parts.push(format!(
                "Constant: {:?}",
                skipped_constant.iter().take(5).collect::<Vec<_>>()
            ));
        }
        tracing::info!(
            total_skipped,
            skipped_detail = %skipped_parts.join(". "),
            remaining_targets = focus_order.len(),
            "Filtered neuron(s) from add-neuron analysis"
        );
    }

    // If no output neurons remain after filtering, return early with empty results
    if focus_order.is_empty() {
        tracing::warn!(
            original_focus_count,
            "No output neurons in focus list — add-neuron candidates can only target output neurons"
        );
        // Build no_candidate_reasons with correct reason for each neuron type
        let mut no_candidate_reasons: Vec<NeuronNoCandidateSummary> = Vec::new();
        for uuid in &skipped_input {
            no_candidate_reasons.push(NeuronNoCandidateSummary {
                target_uuid: uuid.clone(),
                reason: NeuronNoCandidateReason::InputNeuronFiltered,
                evaluated_sources: 0,
                sources_with_samples: 0,
                target_record_count: 0,
                detail: None,
            });
        }
        for uuid in &skipped_hidden {
            no_candidate_reasons.push(NeuronNoCandidateSummary {
                target_uuid: uuid.clone(),
                reason: NeuronNoCandidateReason::HiddenNeuronFiltered,
                evaluated_sources: 0,
                sources_with_samples: 0,
                target_record_count: 0,
                detail: None,
            });
        }
        for uuid in &skipped_constant {
            no_candidate_reasons.push(NeuronNoCandidateSummary {
                target_uuid: uuid.clone(),
                reason: NeuronNoCandidateReason::ConstantNeuronFiltered,
                evaluated_sources: 0,
                sources_with_samples: 0,
                target_record_count: 0,
                detail: None,
            });
        }
        return Ok(AnalyzeNeuronsResult {
            helpful_neurons: Vec::new(),
            gpu_used: true,
            no_candidate_reasons,
            metadata: super::shared::NeuronAnalysisMetadata {
                candidates_found: 0,
                candidates_returned: 0,
                timed_out: false,
                completed_focus_neurons: 0,
                total_focus_neurons: original_focus_count,
                timing: None,
                gpu_info: GpuAnalyzer::get_adapter_info(),
                error_distribution: None,
            },
        });
    }

    // Mark skipped neurons in diagnostics so they appear with the correct reason
    // instead of misleading reasons like NoEligibleSources.
    // This is the normal flow case where some output neurons exist.
    // Issue #216: Direct method calls - no lock needed with DashMap-based diagnostics.
    for input_uuid in &skipped_input {
        diagnostics.mark_input_filtered(input_uuid);
    }
    for hidden_uuid in &skipped_hidden {
        diagnostics.mark_hidden_filtered(hidden_uuid);
    }
    for constant_uuid in &skipped_constant {
        diagnostics.mark_constant_filtered(constant_uuid);
    }

    // Log threshold-crossing neurons for visibility
    if verbose_enabled() && !threshold_targets.is_empty() {
        tracing::debug!(
            count = threshold_targets.len(),
            sample = ?threshold_targets.iter().take(5).collect::<Vec<_>>(),
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
    let order_map_arc = Arc::new(order_map);
    let neuron_squash_map_arc = Arc::new(neuron_squash_map);

    // Issue #182: Build a set of "used" input neurons (those with at least one outgoing synapse)
    // for the focus_unused_observations feature. Unused inputs will be prioritised in source ordering.
    let used_inputs: HashSet<String> = input
        .creature
        .synapses
        .iter()
        .filter(|s| parse_input_index(&s.from_uuid).is_some())
        .map(|s| s.from_uuid.clone())
        .collect();
    let used_inputs_arc = Arc::new(used_inputs);

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
                let first_obs = target_records.first().map(|r| r.obs_index).unwrap_or(0);
                let last_obs = target_records.last().map(|r| r.obs_index).unwrap_or(0);
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
                .map(|squash| is_threshold_activation(squash))
                .unwrap_or(false);

            let target_index = match order_map_arc.get(target_uuid.as_str()) {
                Some(index) => *index,
                None => return Ok(()),
            };

            let mut eligible_sources: Vec<&OrderedNeuron> = ordered_neurons_arc
                .iter()
                .filter(|neuron| neuron.index < target_index)
                .collect();
            // Source ordering:
            // - Candidates must remain forward-only (index < target_index).
            // - Under timeouts we want coverage over time, so we fully shuffle ALL
            //   eligible sources (inputs + hidden + constants).
            // - If NEAT_AI_DISCOVERY_FOCUS_UNUSED_OBSERVATIONS=1, unused inputs are
            //   prioritised to the front (Issue #182).
            //
            // If `random_seed` is provided, the shuffle is deterministic for a given
            // target UUID; otherwise it's non-deterministic.
            let context = format!("neuron:eligible_sources:{target_uuid}");
            order_eligible_sources(
                &mut eligible_sources,
                input.random_seed,
                &context,
                input.creature.input,
                Some(&*used_inputs_arc),
            );

            // Track total eligible sources for diagnostics
            let total_eligible = eligible_sources.len() as u32;
            diagnostics.set_total_eligible_sources(target_uuid, total_eligible);

            // Log focus neuron details for debugging
            if verbose_enabled() && total_eligible == 0 {
                tracing::debug!(
                    target_uuid = %target_uuid,
                    target_index,
                    creature_input = ordered_neurons_arc.len().saturating_sub(input.creature.neurons.len()),
                    max_input_index = ordered_neurons_arc.len().saturating_sub(input.creature.neurons.len()).saturating_sub(1),
                    "Target has 0 eligible upstream sources — possible creature configuration mismatch"
                );
            }

            // Phase 1: Pre-filter sources and collect their records (with deadline checks)
            // This mirrors the synapse analysis approach for better parallelism
            let mut sources_to_process: Vec<(&OrderedNeuron, Arc<Vec<DiscoverRecord>>)> =
                Vec::with_capacity(eligible_sources.len());
            let mut empty_record_sources: Vec<String> = Vec::new();
            let mut load_failure_count = 0u32;

            for source in &eligible_sources {
                // Check deadline during pre-filtering
                if deadline_passed(&deadline) {
                    *lock_or_bail(&analysis_timed_out, "analysis_timed_out")? = true;
                    break;
                }
                let source_uuid = source.uuid.as_str();
                match cache.get(source_uuid) {
                    Ok(records) => {
                        if !records.is_empty() {
                            sources_to_process.push((source, records));
                        } else {
                            empty_record_sources.push(source_uuid.to_string());
                        }
                    }
                    Err(err) => {
                        load_failure_count += 1;
                        if verbose_enabled() {
                            tracing::debug!(
                                source_uuid,
                                target_uuid = %target_uuid,
                                error = %err,
                                "Failed to load source neuron records"
                            );
                        }
                    }
                }
            }

            // Record load failures in diagnostics (no lock needed with DashMap - Issue #216)
            for _ in 0..load_failure_count {
                diagnostics.record_load_failure(target_uuid);
            }

            // Log summary of source loading results for debugging
            let sources_checked = sources_to_process.len() + empty_record_sources.len() + load_failure_count as usize;
            let timed_out_during_loading = *lock_or_bail(&analysis_timed_out, "analysis_timed_out")?;
            if verbose_enabled() && (sources_to_process.is_empty() || load_failure_count > 0 || !empty_record_sources.is_empty() || timed_out_during_loading) {
                let sources_with_records = sources_to_process.len();
                let empty_count = empty_record_sources.len();
                if timed_out_during_loading && sources_checked == 0 {
                    tracing::debug!(
                        target_uuid = %target_uuid,
                        total_eligible,
                        "Source loading timed out before any eligible sources could be checked"
                    );
                } else if timed_out_during_loading {
                    tracing::debug!(
                        target_uuid = %target_uuid,
                        sources_checked,
                        total_eligible,
                        sources_with_records,
                        empty_count,
                        load_failure_count,
                        "Source loading timed out after partial check"
                    );
                } else {
                    tracing::debug!(
                        target_uuid = %target_uuid,
                        total_eligible,
                        sources_with_records,
                        empty_count,
                        load_failure_count,
                        "Source loading summary"
                    );
                }
            }

            // Batch diagnostics for empty record sources (no lock needed with DashMap - Issue #216)
            for source_uuid in &empty_record_sources {
                diagnostics.record_candidate_attempt(target_uuid, false);
                diagnostics.record_no_samples(target_uuid, source_uuid);
            }

            // Check if timed out during pre-filtering
            if *lock_or_bail(&analysis_timed_out, "analysis_timed_out")? {
                return Ok(());
            }

            // Phase 2: Build samples in parallel using CPU (much faster than sequential GPU calls)
            // OPTIMIZATION: Pre-build target map ONCE, reuse for all sources.
            // This avoids rebuilding the HashMap for each of ~1000+ source neurons.
            let target_map = TargetMap::from_records(target_records);
            let target_map_ref = &target_map;

            // Even if target_map is empty, we continue to record diagnostics
            // about what sources were evaluated.
            struct NeuronWorkResult {
                source_uuid: String,
                samples: Vec<HelpfulSample>,
            }

            // Issue #221: Sample Locality Optimisation
            // Group sources by obs_index overlap to reduce redundant sample building.
            // Sources with ≥80% obs_index overlap share sample building in a single pass.
            let work_results: Vec<NeuronWorkResult> = {
                let _timing = TimingScope::sample_building(&timing_collector);

                // Group sources by sample locality
                let locality_groups = group_sources_by_locality(&sources_to_process);

                // Log locality grouping stats if verbose
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

                // Build samples for each group (groups with multiple sources use batched building)
                locality_groups
                    .par_iter()
                    .flat_map(|group| {
                        let group_results = build_samples_for_locality_group(group, target_map_ref);
                        group_results
                            .into_iter()
                            .map(|(source_uuid, samples, _record_count)| NeuronWorkResult {
                                source_uuid,
                                samples,
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect()
            };

            // Phase 3: Batch diagnostics updates for sample building results (no lock needed with DashMap - Issue #216)
            for result in &work_results {
                diagnostics.record_candidate_attempt(target_uuid, !result.samples.is_empty());
                if result.samples.is_empty() {
                    diagnostics.record_no_samples(target_uuid, &result.source_uuid);
                }
            }

            // Phase 4: Process evaluations - GPU work is done here.
            // Filter to only sources with samples, then evaluate.
            //
            // NOTE: We intentionally skip STEP/BIPOLAR targets here; synapse analysis
            // is the supported discovery mechanism for discrete targets.
            //
            // Important: We still count the target as "completed" for progress reporting.
            if !is_threshold_target {
                // Standard continuous activation path
                for result in work_results {
                    // Check deadline before each evaluation batch
                    if deadline_passed(&deadline) {
                        *lock_or_bail(&analysis_timed_out, "analysis_timed_out")? = true;
                        break;
                    }

                    if result.samples.is_empty() {
                        continue;
                    }

                    // Get target_squash for accurate HARD_TANH modelling
                    let target_squash = neuron_squash_map_arc.get(target_uuid).map(|s| s.as_str());

                    // Issue #130 (v0.2.2): Compute source variance discount.
                    // If source activation has low variance, predictions are unreliable.
                    let source_variance_discount = compute_source_variance_discount(&result.samples);
                    if source_variance_discount <= EPSILON {
                        // Source is constant - skip evaluation entirely
                        continue;
                    }

                    // ReLU evaluation: split by TARGET neuron's error sign.
                    //
                    // ReLU can only push output in ONE direction (based on outgoing weight sign),
                    // so we evaluate two candidates separately:
                    // - Positive-error ReLU: optimised for samples where output should be HIGHER
                    // - Negative-error ReLU: optimised for samples where output should be LOWER
                    //
                    // Each candidate's weight is computed from its error subset, then NET
                    // improvement is calculated across ALL samples. This is the correct
                    // approach for directional activation functions like ReLU.
                    //
                    // NOTE: We don't use "averaging over all samples" because when errors are
                    // split ~50/50, the average cancels out and no candidate is found.
                    let split_result = {
                        let _timing = TimingScope::shader(&timing_collector, "relu");
                        evaluate_relu_candidates_split(
                            gpu,
                            &result.source_uuid,
                            target_uuid,
                            &result.samples,
                            threshold,
                            target_squash,
                        )?
                    };

                    if let Some(mut candidate) = split_result.positive_error_candidate {
                        // Issue #130: Apply source variance discount
                        candidate.expected_creature_error_reduction *= source_variance_discount;
                        candidate.expected_creature_score_gain *= source_variance_discount;

                        if verbose_enabled() {
                            tracing::trace!(
                                direction = "push UP",
                                source_uuid = %result.source_uuid,
                                target_uuid = %target_uuid,
                                improvement_pct = format_args!("{:.2}", candidate.expected_creature_score_gain * 100.0),
                                variance_discount = format_args!("{:.2}", source_variance_discount),
                                "ReLU candidate identified"
                            );
                        }
                        // Issue #216: Direct method call - no lock needed with DashMap-based diagnostics
                        diagnostics.mark_candidate_selected(target_uuid);
                        let mut map = lock_or_bail(&helpful_map, "helpful_map")?;
                        upsert_candidate(&mut map, candidate);
                    }

                    if let Some(mut candidate) = split_result.negative_error_candidate {
                        // Issue #130: Apply source variance discount
                        candidate.expected_creature_error_reduction *= source_variance_discount;
                        candidate.expected_creature_score_gain *= source_variance_discount;

                        if verbose_enabled() {
                            tracing::trace!(
                                direction = "push DOWN",
                                source_uuid = %result.source_uuid,
                                target_uuid = %target_uuid,
                                improvement_pct = format_args!("{:.2}", candidate.expected_creature_score_gain * 100.0),
                                variance_discount = format_args!("{:.2}", source_variance_discount),
                                "ReLU candidate identified"
                            );
                        }
                        // Issue #216: Direct method call - no lock needed with DashMap-based diagnostics
                        diagnostics.mark_candidate_selected(target_uuid);
                        let mut map = lock_or_bail(&helpful_map, "helpful_map")?;
                        upsert_candidate(&mut map, candidate);
                    }

                    // Issue #201: Evaluate all activation specs in a single batched GPU call
                    // This reduces GPU round-trips by 10-20% compared to sequential evaluation
                    let batched_candidates = {
                        let _timing = TimingScope::shader(&timing_collector, "activation");
                        evaluate_all_activation_specs_batched(
                            gpu,
                            &result.source_uuid,
                            target_uuid,
                            &result.samples,
                            threshold,
                            target_squash,
                        )?
                    };

                    for mut candidate in batched_candidates {
                        // Issue #130: Apply source variance discount
                        candidate.expected_creature_error_reduction *= source_variance_discount;
                        candidate.expected_creature_score_gain *= source_variance_discount;

                        // Issue #216: Direct method call - no lock needed with DashMap-based diagnostics
                        diagnostics.mark_candidate_selected(target_uuid);
                        let mut map = lock_or_bail(&helpful_map, "helpful_map")?;
                        upsert_candidate(&mut map, candidate);
                    }
                }
            }

            // Track completion of this focus neuron for timeout reporting.
            let completed =
                completed_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            crate::watchdog::beat(format!(
                "neuron analysis → completed {completed}/{total_focus_count} (last target {target_uuid})"
            ));
            Ok(())
        })?;

    let analysis_timed_out = *lock_or_bail(&analysis_timed_out, "analysis_timed_out")?;
    let helpful_map = lock_or_bail(&helpful_map, "helpful_map")?.clone();

    // Log timeout with completion stats (always visible, not just verbose)
    if analysis_timed_out {
        let completed = completed_count.load(std::sync::atomic::Ordering::Relaxed);
        log_analysis_timeout("neuron", completed, total_focus_count);
    }

    let mut helpful_results: Vec<CandidateNeuronJson> = helpful_map.into_values().collect();

    // Issue #128: Apply impact-based discounting and set creature-level metrics.
    // Output neurons have impact = 1.0 (no discount).
    // Hidden neurons have impact in [0, 1] based on their weighted paths to outputs.
    let impact_scores = compute_impact_scores_for_discounting(&input.creature, cache.as_ref());
    for candidate in &mut helpful_results {
        candidate.source_neuron_index = order_map_arc.get(&candidate.source_neuron_uuid).copied();
        candidate.target_neuron_index = order_map_arc.get(&candidate.target_neuron_uuid).copied();

        let is_hidden = neuron_type_map
            .get(&candidate.target_neuron_uuid)
            .map(|t| t != "output")
            .unwrap_or(true); // Default to true if type unknown (treat as hidden)

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

        // Issue #506: Apply pessimism discount based on improved sample ratio.
        candidate.expected_creature_score_gain = apply_pessimism_discount(
            candidate.expected_creature_score_gain,
            candidate.improved_count,
            candidate.total_count,
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

    // Issue #557: Filter out candidates with non-positive expected_creature_score_gain.
    // After impact discounting, some candidates may have zero or negative gain and
    // would waste the evaluation budget if returned.
    helpful_results.retain(|c| c.expected_creature_score_gain > 0.0);

    // Sort by expected creature score gain (highest first) - Issue #128
    helpful_results.sort_by(|a, b| {
        b.expected_creature_score_gain
            .total_cmp(&a.expected_creature_score_gain)
    });

    // Production experiment: pair "extreme" candidates with a conservative variant.
    // Pass None for limit here - we'll truncate separately so that candidates_found
    // correctly includes generated variants.
    helpful_results = crate::analysis::utils::pair_extreme_candidates_with_conservative_variants(
        helpful_results,
        None, // No limit - truncate separately after capturing candidates_found
    );

    // Production guard rail (Dec 2025): only return candidates within sensible parameter ranges.
    // This avoids wasting the evaluation budget on absurd bias/weight configurations.
    helpful_results = crate::analysis::utils::filter_candidates_to_sensible_ranges(helpful_results);

    // Deadline coverage (Jan 2026): when deadline-constrained, diversify within the top-K so
    // repeated runs explore different high-quality candidates over time (helps with failure caches).
    if input.analysis_deadline_ms.is_some() {
        use super::constants::DIVERSIFY_TOP_K;
        shuffle_within_top_k(
            helpful_results.as_mut_slice(),
            input.random_seed,
            "neuron:candidates:top_k",
            DIVERSIFY_TOP_K,
        );
    }

    // Track candidates_found AFTER pairing but BEFORE truncation.
    // This ensures candidates_found >= candidates_returned always holds, which is
    // the expected semantic for this metric pair ("found" >= "returned").
    let candidates_found = helpful_results.len();

    // Apply max_candidates limit (truncation)
    if let Some(limit) = input.max_candidates {
        helpful_results.truncate(limit);
    }

    // Track candidates_returned AFTER truncation
    let candidates_returned = helpful_results.len();

    let no_candidate_reasons = diagnostics.no_candidate_summaries();
    diagnostics.emit_logs();

    // Issue #486 / #192: Compute error distribution from collected target neuron error samples.
    let error_distribution = {
        let error_values = std::mem::take(&mut *lock_or_bail(
            &error_values_for_distribution,
            "error_values_for_distribution",
        )?);
        super::error_distribution::ErrorDistribution::from_errors(&error_values)
    };

    Ok(AnalyzeNeuronsResult {
        helpful_neurons: helpful_results,
        gpu_used,
        no_candidate_reasons,
        metadata: super::shared::NeuronAnalysisMetadata {
            candidates_found,
            candidates_returned,
            timed_out: analysis_timed_out,
            completed_focus_neurons: completed_count.load(std::sync::atomic::Ordering::Relaxed),
            // Total focus neurons requested for this invocation (pre-filter).
            //
            // Note: `total_focus_count` is the post-filter eligible output-neuron count, which can
            // differ from the requested focus list. We keep the "requested" semantics so callers
            // can track long-run coverage consistently across early/normal return paths.
            total_focus_neurons: original_focus_count,
            timing: timing_collector.finalize(),
            gpu_info: GpuAnalyzer::get_adapter_info(),
            error_distribution,
        },
    })
}
