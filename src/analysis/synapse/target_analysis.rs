//! Per-target synapse analysis
//!
//! This module contains the analysis logic for a single focus (target) neuron:
//! - Validate target neuron eligibility
//! - Filter and order eligible source neurons
//! - Build samples via locality grouping
//! - Evaluate helpful synapse candidates via GPU batching
//! - Evaluate harmful synapse candidates via GPU batching
//! - Detect epistatic, synergistic, and redundant path patterns
//!
//! Extracted from mod.rs as part of Issue #482.

use crate::analysis::activation::{get_target_simulation_fn, is_saturating_target};
use crate::analysis::confidence::compute_confidence_metrics;
use crate::analysis::diagnostics::{TargetDiagnostics, TargetMap, ThresholdContext};
use crate::analysis::epistatic::{
    SourceContribution, build_source_contribution, deduplicate_by_dominant_neuron,
    deduplicate_synergistic_by_dominant_neuron, detect_epistatic_pairs,
    detect_synergistic_candidates, epistatic_pairs_to_coordinated_candidates,
    filter_interfering_epistatic_pairs, filter_interfering_synergistic_candidates,
    synergistic_to_coordinated_candidates,
};
use crate::analysis::gpu::GpuWorkQueue;
use crate::analysis::redundant_path::{
    ExistingPathContribution, detect_redundant_paths, redundant_paths_to_coordinated_candidates,
};
use crate::analysis::samples::{EPSILON, HelpfulSample, HelpfulStats, NeuronStats};
use crate::analysis::shared::TimingScope;
use crate::analysis::utils::{
    OrderedNeuron, deadline_passed, order_eligible_sources, parse_input_index, verbose_enabled,
};
use crate::analysis::weights::{
    MAX_OUTGOING_WEIGHT, calculate_optimal_outgoing_weight, clamp_weight_update_delta,
};
use crate::intern::NeuronIndex;
use crate::types::DiscoverRecord;
use crate::{AnalyzeSynapsesInput, CandidateSynapseJson, SynapseJson};
use anyhow::{Result, anyhow};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::SystemTime;

use super::candidate_generation::{
    MIN_GROUP_SIZE_FOR_LOCALITY, build_samples_for_locality_group, group_sources_by_locality,
};
use super::scoring::compute_synapse_improvement_and_count;
use crate::analysis::cache::RecordCache;

/// Shared context for per-target analysis, built once in the main pipeline.
pub(crate) struct TargetAnalysisContext {
    pub ordered_neurons: Arc<Vec<OrderedNeuron>>,
    pub order_map: Arc<HashMap<String, usize>>,
    pub neuron_index: Arc<NeuronIndex>,
    pub existing_synapses: Arc<HashSet<(u32, u32)>>,
    pub existing_synapse_weights: Arc<HashMap<(u32, u32), f32>>,
    pub synapses_by_target: Arc<HashMap<u32, Vec<SynapseJson>>>,
    pub neuron_squash_map: Arc<HashMap<String, String>>,
    pub neuron_type_map: Arc<HashMap<String, String>>,
    pub input_neuron_uuids: Arc<HashSet<String>>,
    pub used_inputs: Arc<HashSet<String>>,
    pub neuron_bias_map: Arc<HashMap<String, f32>>,
    pub constant_source_effect_threshold: Option<f32>,
    pub diagnostics: Arc<TargetDiagnostics>,
    pub timing_collector: Arc<crate::analysis::shared::TimingCollector>,
    pub deadline: Option<SystemTime>,
    pub threshold: f32,
}

/// Results from analysing a single target neuron.
pub(crate) struct TargetAnalysisResults {
    pub helpful: Vec<CandidateSynapseJson>,
    pub harmful: Vec<CandidateSynapseJson>,
    pub coordinated: Vec<crate::CoordinatedStructuralCandidateJson>,
    /// Error values collected from target records for distribution analysis.
    pub error_values: Vec<f32>,
    /// Whether any samples had target_value data available.
    pub target_value_seen: bool,
    /// Whether saturation-aware simulation was used for any candidate.
    pub saturation_aware_used: bool,
    /// Input neuron index tracking for metadata.
    pub input_metadata: InputMetadata,
}

/// Metadata about input neurons with records, for inclusion in analysis metadata.
pub(crate) struct InputMetadata {
    pub seen_any: bool,
    pub min_index: usize,
    pub max_index: usize,
}

/// Work item for helpful synapse evaluation via GPU.
struct HelpfulWork {
    source_uuid: String,
    target_uuid: String,
    samples: Vec<HelpfulSample>,
    /// Existing synapse weight (when the synapse already exists).
    existing_weight: Option<f32>,
}

/// Tracking result from sample building for a single source neuron.
struct SourceWorkResult {
    work: Option<HelpfulWork>,
    had_samples: bool,
    source_uuid: String,
    record_count: usize,
}

/// Existing edge metadata for weight-update candidates.
struct ExistingSourceToProcess<'a> {
    source: &'a OrderedNeuron,
    records: Arc<Vec<DiscoverRecord>>,
    old_weight: f32,
}

/// Analyse a single target neuron: find helpful, harmful, and coordinated candidates.
///
/// This function encapsulates the entire per-target loop body that was previously
/// inline in `analyze_synapses_with_cache_impl`.
pub(crate) fn analyse_single_target(
    target_uuid: &str,
    input: &AnalyzeSynapsesInput,
    cache: &RecordCache,
    gpu: &GpuWorkQueue,
    ctx: &TargetAnalysisContext,
) -> Result<TargetAnalysisResults> {
    let mut results = TargetAnalysisResults {
        helpful: Vec::new(),
        harmful: Vec::new(),
        coordinated: Vec::new(),
        error_values: Vec::new(),
        target_value_seen: false,
        saturation_aware_used: false,
        input_metadata: InputMetadata {
            seen_any: false,
            min_index: usize::MAX,
            max_index: 0,
        },
    };

    crate::watchdog::beat(format!(
        "synapse analysis → processing target {target_uuid}"
    ));

    let target_records_arc = cache.get(target_uuid)?;
    if target_records_arc.is_empty() {
        ctx.diagnostics.set_target_record_count(target_uuid, 0);
        return Ok(results);
    }
    let target_records = target_records_arc.as_ref();
    ctx.diagnostics
        .set_target_record_count(target_uuid, target_records.len());

    // Issue #192: Collect error values for distribution analysis
    {
        let errors: Vec<f32> = target_records
            .iter()
            .flat_map(|r| r.errors.iter().filter(|e| e.is_finite()).copied())
            .collect();
        if !errors.is_empty() {
            results.error_values = errors;
        }
    }

    let target_index = match ctx.order_map.get(target_uuid) {
        Some(index) => *index,
        None => {
            if verbose_enabled() {
                tracing::debug!(
                    target_uuid = target_uuid,
                    "Target not found in creature neuron order map (neuron may not exist in creature definition). Skipping."
                );
            }
            return Ok(results);
        }
    };

    // Early validation: skip input and constant neurons
    let target_neuron_type = ctx
        .neuron_type_map
        .get(target_uuid)
        .ok_or_else(|| {
            anyhow!(
                "Invalid target neuron UUID '{}': not found in neuron type map. \
                This indicates a serious data integrity bug. All valid neurons must be in the type map \
                (input neurons: input-0..input-{}, or neurons from creature.neurons array).",
                target_uuid,
                ctx.input_neuron_uuids.len().saturating_sub(1)
            )
        })?;

    let input_count = ctx.input_neuron_uuids.len();
    let is_input_neuron = target_neuron_type == "input";
    let is_constant_neuron = target_neuron_type == "constant";

    if is_input_neuron || is_constant_neuron {
        return Ok(results);
    }

    // Filter eligible sources
    let mut eligible_sources: Vec<&OrderedNeuron> = ctx
        .ordered_neurons
        .iter()
        .filter(|neuron| {
            neuron.index < target_index
                && {
                    match ctx.neuron_type_map.get(&neuron.uuid) {
                        Some(neuron_type) => neuron_type != "constant",
                        None => {
                            tracing::warn!(
                                neuron_uuid = %neuron.uuid,
                                "Invalid neuron UUID found in ordered_neurons. \
                                Not found in comprehensive neuron type map. This indicates a serious data integrity bug."
                            );
                            false
                        }
                    }
                }
        })
        .collect();

    let total_eligible = eligible_sources.len() as u32;

    if total_eligible == 0 {
        let neurons_before_index = ctx
            .ordered_neurons
            .iter()
            .filter(|n| n.index < target_index)
            .count();
        let constants_before_index = ctx
            .ordered_neurons
            .iter()
            .filter(|n| {
                n.index < target_index
                    && ctx
                        .neuron_type_map
                        .get(&n.uuid)
                        .map(|t| t == "constant")
                        .unwrap_or(false)
            })
            .count();
        let input_neurons_before_index = ctx
            .ordered_neurons
            .iter()
            .filter(|n| n.index < target_index && ctx.input_neuron_uuids.contains(&n.uuid))
            .count();

        tracing::warn!(
            target_uuid = target_uuid,
            target_neuron_type = target_neuron_type,
            target_index = target_index,
            input_count = input_count,
            neurons_before_target = neurons_before_index,
            constants_before_target = constants_before_index,
            input_neurons_before_target = input_neurons_before_index,
            "BUG: Target has no eligible upstream neurons. This should not happen for hidden/output neurons with index >= creature.input."
        );

        return Ok(results);
    }

    let input_neuron_count = eligible_sources
        .iter()
        .filter(|neuron| ctx.input_neuron_uuids.contains(&neuron.uuid))
        .count() as u32;
    ctx.diagnostics
        .set_total_eligible_sources(target_uuid, total_eligible);
    ctx.diagnostics
        .set_input_neuron_count(target_uuid, input_neuron_count);

    let context = format!("synapse:eligible_sources:{target_uuid}");
    order_eligible_sources(
        &mut eligible_sources,
        input.random_seed,
        &context,
        input.creature.input,
        Some(&*ctx.used_inputs),
    );

    // Pre-filter sources and collect their records
    let mut already_connected_count = 0u32;
    let mut load_failure_count = 0u32;
    let mut empty_record_sources: Vec<String> = Vec::new();

    let mut sources_to_process: Vec<(&OrderedNeuron, Arc<Vec<DiscoverRecord>>)> =
        Vec::with_capacity(eligible_sources.len());
    let mut existing_sources_to_process: Vec<ExistingSourceToProcess> =
        Vec::with_capacity(eligible_sources.len());

    for source in &eligible_sources {
        if deadline_passed(&ctx.deadline) {
            break;
        }
        let source_uuid = source.uuid.as_str();

        let source_idx = ctx.neuron_index.get_index(source_uuid);
        let target_idx = ctx.neuron_index.get_index(target_uuid);
        let is_connected = match (source_idx, target_idx) {
            (Some(s), Some(t)) => ctx.existing_synapses.contains(&(s, t)),
            _ => false,
        };
        if is_connected {
            already_connected_count += 1;
        };
        let existing_weight = if is_connected {
            match (source_idx, target_idx) {
                (Some(s), Some(t)) => ctx.existing_synapse_weights.get(&(s, t)).copied(),
                _ => None,
            }
        } else {
            None
        };

        match cache.get(source_uuid) {
            Ok(records) => {
                if !records.is_empty() {
                    if let Some(input_index) = parse_input_index(source_uuid) {
                        results.input_metadata.seen_any = true;
                        if input_index < results.input_metadata.min_index {
                            results.input_metadata.min_index = input_index;
                        }
                        if input_index > results.input_metadata.max_index {
                            results.input_metadata.max_index = input_index;
                        }
                    }
                    if let Some(old_weight) = existing_weight {
                        existing_sources_to_process.push(ExistingSourceToProcess {
                            source,
                            records,
                            old_weight,
                        });
                    } else if !is_connected {
                        sources_to_process.push((source, records));
                    }
                } else {
                    empty_record_sources.push(source_uuid.to_string());
                    let is_input = ctx.input_neuron_uuids.contains(source_uuid);
                    if verbose_enabled() && !is_input {
                        tracing::debug!(
                            source_uuid = source_uuid,
                            target_uuid = target_uuid,
                            "Source has no records in parquet file."
                        );
                    }
                }
            }
            Err(err) => {
                load_failure_count += 1;
                if verbose_enabled() {
                    tracing::debug!(
                        source_uuid = source_uuid,
                        target_uuid = target_uuid,
                        error = %err,
                        "Failed to load records for source."
                    );
                }
            }
        };
    }

    // Log empty input neuron summary
    let empty_input_neuron_count = empty_record_sources
        .iter()
        .filter(|uuid| ctx.input_neuron_uuids.contains(uuid.as_str()))
        .count();
    let empty_non_input_count = empty_record_sources.len() - empty_input_neuron_count;

    if verbose_enabled() && empty_input_neuron_count > 0 {
        tracing::debug!(
            target_uuid = target_uuid,
            empty_input_neuron_count = empty_input_neuron_count,
            total_input_neurons = ctx.input_neuron_uuids.len(),
            empty_non_input_count = empty_non_input_count,
            "Input neurons have no records in parquet file. This may indicate incomplete parquet data."
        );
    }

    // Update diagnostics
    if already_connected_count > 0 || load_failure_count > 0 || !empty_record_sources.is_empty() {
        for _ in 0..already_connected_count {
            ctx.diagnostics.record_already_connected(target_uuid);
        }
        for _ in 0..load_failure_count {
            ctx.diagnostics.record_load_failure(target_uuid);
        }
        for source_uuid in &empty_record_sources {
            ctx.diagnostics.record_candidate_attempt(target_uuid, false);
            ctx.diagnostics
                .record_no_samples(target_uuid, source_uuid, 0);
        }
    }

    // Build target map once, reuse for all sources
    let target_map = TargetMap::from_records(target_records);
    let target_map_ref = &target_map;

    // Noisy vs trusted input detection (Issue #165)
    let target_idx_for_synapses = ctx.neuron_index.get_index(target_uuid);
    if let Some(existing) = target_idx_for_synapses.and_then(|idx| ctx.synapses_by_target.get(&idx))
        && let Some(candidate) = super::structural_patterns::detect_noisy_vs_trusted(
            target_uuid,
            existing,
            cache,
            target_map_ref,
            &ctx.neuron_squash_map,
        )
    {
        results.coordinated.push(candidate);
    }

    // Issue #221: Sample Locality Optimisation
    let source_results: Vec<SourceWorkResult> = {
        let _timing = TimingScope::sample_building(&ctx.timing_collector);

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
                target_uuid = target_uuid,
                source_count = sources_to_process.len(),
                group_count = locality_groups.len(),
                max_group_size = max_group,
                avg_group_size = format_args!("{avg_group:.1}"),
                "Sources grouped into locality groups."
            );
        }

        locality_groups
            .par_iter()
            .flat_map(|group| {
                let group_results = build_samples_for_locality_group(group, target_map_ref);
                group_results
                    .into_iter()
                    .map(|(source_uuid, samples, record_count)| {
                        let had_samples = !samples.is_empty();
                        let work = if had_samples {
                            Some(HelpfulWork {
                                source_uuid: source_uuid.clone(),
                                target_uuid: target_uuid.to_string(),
                                samples,
                                existing_weight: None,
                            })
                        } else {
                            None
                        };
                        SourceWorkResult {
                            work,
                            had_samples,
                            source_uuid,
                            record_count,
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    };

    // Extract work batch and batch diagnostics updates
    let mut helpful_work_batch: Vec<HelpfulWork> = Vec::new();
    let mut diagnostics_updates: Vec<(String, String, bool, usize)> = Vec::new();

    for result in source_results {
        if let Some(work) = result.work {
            helpful_work_batch.push(work);
        }
        diagnostics_updates.push((
            target_uuid.to_string(),
            result.source_uuid,
            result.had_samples,
            result.record_count,
        ));
    }

    for (target, source, had_samples, record_count) in diagnostics_updates {
        ctx.diagnostics
            .record_candidate_attempt(&target, had_samples);
        if !had_samples {
            ctx.diagnostics
                .record_no_samples(&target, &source, record_count);
        }
    }

    // Append existing edges for weight-update evaluation
    let mut existing_path_contributions: Vec<ExistingPathContribution> = Vec::new();
    if !existing_sources_to_process.is_empty() {
        // Build contributions first (owns the samples), then create work items from them.
        existing_path_contributions = existing_sources_to_process
            .par_iter()
            .filter_map(|item| {
                let from_records = item.records.as_ref();
                let samples = target_map_ref.build_samples_from(from_records);
                if samples.is_empty() {
                    return None;
                }
                Some(ExistingPathContribution {
                    source_uuid: item.source.uuid.clone(),
                    existing_weight: item.old_weight,
                    samples,
                })
            })
            .collect();

        // Create work items by cloning samples from contributions (single clone instead of double).
        helpful_work_batch.extend(existing_path_contributions.iter().map(|c| HelpfulWork {
            source_uuid: c.source_uuid.clone(),
            target_uuid: target_uuid.to_string(),
            samples: c.samples.clone(), // Clone required: GPU queue takes ownership
            existing_weight: Some(c.existing_weight),
        }));
    }

    // Issue #568: Overlap CPU analysis with GPU computation.
    // Submit the helpful GPU batch non-blocking, then prepare harmful samples
    // (CPU work) while the GPU processes the helpful batch.
    let helpful_future = if !helpful_work_batch.is_empty() {
        Some(submit_helpful_gpu_work(
            &helpful_work_batch,
            gpu,
            ctx,
            &mut results,
        )?)
    } else {
        None
    };

    // While GPU processes helpful batch, prepare harmful samples on CPU
    let harmful_prep = if !deadline_passed(&ctx.deadline) {
        let target_idx_for_harmful = ctx.neuron_index.get_index(target_uuid);
        target_idx_for_harmful
            .and_then(|idx| ctx.synapses_by_target.get(&idx))
            .map(|existing| prepare_harmful_samples(existing, cache, target_map_ref))
    } else {
        None
    };

    // Collect helpful GPU results and process them
    if let Some(future) = helpful_future {
        collect_and_process_helpful_results(
            future,
            &helpful_work_batch,
            target_uuid,
            cache,
            ctx,
            &existing_path_contributions,
            &mut results,
        )?;
    }

    // Process harmful synapses (GPU submission + result processing)
    if let Some(harmful_work) = harmful_prep
        && !harmful_work.is_empty()
    {
        process_harmful_batch_from_prepared(
            target_uuid,
            &harmful_work,
            gpu,
            ctx,
            cache,
            &mut results,
        )?;
    }

    Ok(results)
}

/// Issue #568: Submit helpful GPU work non-blocking.
///
/// Clones sample data for the GPU, tracks metadata, and submits the batch.
/// Returns a `GpuFuture` that can be collected after overlapping CPU work.
fn submit_helpful_gpu_work(
    helpful_work_batch: &[HelpfulWork],
    gpu: &GpuWorkQueue,
    ctx: &TargetAnalysisContext,
    results: &mut TargetAnalysisResults,
) -> Result<crate::analysis::gpu::queue::GpuFuture<Vec<HelpfulStats>>> {
    let helpful_samples: Vec<Vec<HelpfulSample>> = helpful_work_batch
        .iter()
        .map(|w| w.samples.clone()) // Clone required: GPU queue takes ownership of sample data
        .collect();

    // Track metadata
    for samples in &helpful_samples {
        if samples.iter().any(|s| s.target_value.is_some()) {
            results.target_value_seen = true;
            break;
        }
    }

    let _timing = TimingScope::shader(&ctx.timing_collector, "helpful");
    gpu.submit_helpful_batch(helpful_samples, &ctx.deadline)
}

/// Issue #568: Collect helpful GPU results and process them into candidates.
fn collect_and_process_helpful_results(
    future: crate::analysis::gpu::queue::GpuFuture<Vec<HelpfulStats>>,
    helpful_work_batch: &[HelpfulWork],
    target_uuid: &str,
    cache: &RecordCache,
    ctx: &TargetAnalysisContext,
    existing_path_contributions: &[ExistingPathContribution],
    results: &mut TargetAnalysisResults,
) -> Result<()> {
    let helpful_stats_batch = future.collect()?;

    let mut candidates_to_add = Vec::new();
    let mut coordinated_to_add = Vec::new();
    let mut diagnostics_zero_improvements: Vec<(&str, &str, usize, u32, u32)> = Vec::new();
    let mut diagnostics_below_threshold: Vec<(&str, &str, ThresholdContext)> = Vec::new();
    let mut diagnostics_selected: Vec<&str> = Vec::new();
    let mut source_contributions: Vec<SourceContribution> = Vec::new();

    {
        let _timing = TimingScope::result_processing(&ctx.timing_collector);
        for (work, stats) in helpful_work_batch.iter().zip(helpful_stats_batch.iter()) {
            let positive_is_better = stats.positive_count >= stats.negative_count;
            let gpu_improved_count = if positive_is_better {
                stats.positive_count
            } else {
                stats.negative_count
            };
            if gpu_improved_count == 0 {
                diagnostics_zero_improvements.push((
                    work.target_uuid.as_str(),
                    work.source_uuid.as_str(),
                    work.samples.len(),
                    stats.positive_count,
                    stats.negative_count,
                ));
                continue;
            }

            let total_count = work.samples.len() as u32;
            if total_count == 0 {
                continue;
            }

            let weight = match calculate_optimal_outgoing_weight(
                stats.error_activation_sum,
                stats.activation_sq_sum,
                1.0,
            ) {
                Some(w) => w,
                None => continue,
            };

            let target_squash = ctx
                .neuron_squash_map
                .get(&work.target_uuid)
                .map(|s| s.as_str());

            if get_target_simulation_fn(&work.samples, target_squash).is_some() {
                results.saturation_aware_used = true;
            }

            let baseline_error_sq = stats.error_sq_sum;

            let (applied_weight, neuron_error_improvement, improved_count, worsened_count) =
                if let Some(old_weight) = work.existing_weight {
                    let Some((_new_weight, delta_weight)) =
                        clamp_weight_update_delta(old_weight, weight)
                    else {
                        continue;
                    };
                    let (improvement, improved, worsened, _) =
                        compute_synapse_improvement_and_count(
                            &work.samples,
                            delta_weight,
                            baseline_error_sq,
                            target_squash,
                        );
                    (delta_weight, improvement, improved, worsened)
                } else if is_saturating_target(&work.samples, target_squash) {
                    // Issue #413: Search over scaled weights for saturating targets
                    let weight_candidates: [f32; 9] = [
                        weight * 0.1,
                        weight * 0.25,
                        weight * 0.5,
                        weight * 0.75,
                        weight,
                        weight * 1.5,
                        weight * 2.0,
                        -weight * 0.5,
                        -weight,
                    ];

                    let mut best_weight = weight;
                    let mut best_improvement = f32::NEG_INFINITY;
                    let mut best_improved = 0u32;
                    let mut best_worsened = 0u32;

                    for &w in &weight_candidates {
                        let clamped = w.clamp(-MAX_OUTGOING_WEIGHT, MAX_OUTGOING_WEIGHT);
                        if clamped.abs() <= EPSILON {
                            continue;
                        }
                        let (imp, improved, worsened, _) = compute_synapse_improvement_and_count(
                            &work.samples,
                            clamped,
                            baseline_error_sq,
                            target_squash,
                        );
                        if imp > best_improvement {
                            best_improvement = imp;
                            best_weight = clamped;
                            best_improved = improved;
                            best_worsened = worsened;
                        }
                    }
                    (best_weight, best_improvement, best_improved, best_worsened)
                } else {
                    let (improvement, improved, worsened, _) =
                        compute_synapse_improvement_and_count(
                            &work.samples,
                            weight,
                            baseline_error_sq,
                            target_squash,
                        );
                    (weight, improvement, improved, worsened)
                };

            // Issue #202: Track source contribution for epistatic pair detection
            if work.existing_weight.is_none() {
                source_contributions.push(build_source_contribution(
                    &work.source_uuid,
                    work.samples.clone(), // Clone required: SourceContribution takes ownership
                    *stats,
                    applied_weight,
                    neuron_error_improvement,
                ));
            }

            if neuron_error_improvement <= 0.0 {
                continue;
            }

            if neuron_error_improvement <= ctx.threshold {
                diagnostics_below_threshold.push((
                    work.target_uuid.as_str(),
                    work.source_uuid.as_str(),
                    ThresholdContext {
                        sample_count: work.samples.len(),
                        expected_improvement: neuron_error_improvement,
                        threshold: ctx.threshold,
                        improved_count,
                        worsened_count,
                        weight: applied_weight,
                    },
                ));
            }

            let target_stats = cache
                .get(&work.target_uuid)
                .ok()
                .and_then(|records| NeuronStats::from_records(records.as_ref()))
                .map(|s| s.to_json());
            if let Some(old_weight) = work.existing_weight {
                let Some((new_weight, delta_weight)) =
                    clamp_weight_update_delta(old_weight, weight)
                else {
                    continue;
                };
                coordinated_to_add.push(crate::CoordinatedStructuralCandidateJson {
                    operations: vec![crate::CoordinatedStructuralOpJson::SetWeight {
                        from_neuron_uuid: work.source_uuid.clone(),
                        to_neuron_uuid: work.target_uuid.clone(),
                        weight: new_weight,
                    }],
                    expected_creature_score_gain: neuron_error_improvement,
                    comment: Some(format!(
                        "Adjust synapse weight: old={old_weight:.6}, new={new_weight:.6}, delta={delta_weight:.6}"
                    )),
                });
            } else {
                diagnostics_selected.push(work.target_uuid.as_str());
                // Issue #178: constant source folding into setBias
                if let Some(threshold) = ctx.constant_source_effect_threshold {
                    let mut act_min = f32::INFINITY;
                    let mut act_max = f32::NEG_INFINITY;
                    let mut act_sum = 0.0f64;
                    let mut act_count: u32 = 0;
                    for s in &work.samples {
                        if s.activation.is_finite() {
                            act_min = act_min.min(s.activation);
                            act_max = act_max.max(s.activation);
                            act_sum += s.activation as f64;
                            act_count += 1;
                        }
                    }

                    if act_count > 0 {
                        let mean_activation = (act_sum / act_count as f64) as f32;
                        let activation_range = (act_max - act_min).abs();
                        let effect_range = applied_weight.abs() * activation_range;

                        if mean_activation.is_finite()
                            && activation_range.is_finite()
                            && effect_range.is_finite()
                            && effect_range <= threshold
                        {
                            let old_bias = ctx
                                .neuron_bias_map
                                .get(&work.target_uuid)
                                .copied()
                                .unwrap_or(0.0);
                            let new_bias = old_bias + (applied_weight * mean_activation);
                            if new_bias.is_finite() {
                                coordinated_to_add.push(
                                    crate::CoordinatedStructuralCandidateJson {
                                        operations: vec![
                                            crate::CoordinatedStructuralOpJson::SetBias {
                                                neuron_uuid: work.target_uuid.clone(),
                                                bias: new_bias,
                                            },
                                        ],
                                        expected_creature_score_gain: neuron_error_improvement,
                                        comment: Some(format!(
                                            "Fold constant source into setBias: old_bias={old_bias:.6}, new_bias={new_bias:.6}, weight={applied_weight:.6}, mean_act={mean_activation:.6}, act_range={activation_range:.6e}, effect_range={effect_range:.6e}"
                                        )),
                                    },
                                );
                                continue;
                            }
                        }
                    }
                }

                let confidence_metrics =
                    compute_confidence_metrics(&work.samples, neuron_error_improvement, None);
                candidates_to_add.push(CandidateSynapseJson {
                    from_neuron_uuid: work.source_uuid.clone(),
                    to_neuron_uuid: work.target_uuid.clone(),
                    from_neuron_index: None,
                    to_neuron_index: None,
                    weight: applied_weight,
                    target_neuron_impact: 1.0,
                    expected_creature_error_reduction: neuron_error_improvement,
                    expected_creature_score_gain: neuron_error_improvement,
                    improved_count,
                    total_count,
                    target_neuron_stats: target_stats,
                    outlier_reduction_info: None,
                    prediction_confidence: confidence_metrics.prediction_confidence,
                    expected_score_gain_confidence_interval: confidence_metrics
                        .expected_score_gain_confidence_interval,
                    comment: None,
                });
            }
        } // End timing scope for result processing
    }

    // Apply diagnostics updates
    for (target, source, sample_count, pos, neg) in diagnostics_zero_improvements {
        ctx.diagnostics
            .record_zero_improvement(target, source, sample_count, pos, neg);
    }
    for (target, source, context) in diagnostics_below_threshold {
        ctx.diagnostics
            .record_below_threshold(target, source, context);
    }
    for target in diagnostics_selected {
        ctx.diagnostics.mark_candidate_selected(target);
    }

    results.helpful.extend(candidates_to_add);
    results.coordinated.extend(coordinated_to_add);

    // Issue #202: Detect epistatic neuron pairs
    if verbose_enabled() {
        tracing::debug!(
            target_uuid = target_uuid,
            source_contribution_count = source_contributions.len(),
            "Collected source contributions for epistatic detection."
        );
    }
    if source_contributions.len() >= 2 {
        let target_is_output = ctx
            .neuron_type_map
            .get(target_uuid)
            .map(|t| t == "output")
            .unwrap_or(false);
        let target_impact = if target_is_output { 1.0 } else { 0.5 };

        let epistatic_pairs =
            detect_epistatic_pairs(target_uuid, &source_contributions, target_impact);

        if !epistatic_pairs.is_empty() {
            let filtered_pairs =
                filter_interfering_epistatic_pairs(epistatic_pairs, &source_contributions);

            // Issue #509: Deduplicate pairs sharing a dominant neuron
            let deduped_pairs = deduplicate_by_dominant_neuron(filtered_pairs);

            if !deduped_pairs.is_empty() {
                let epistatic_candidates =
                    epistatic_pairs_to_coordinated_candidates(&deduped_pairs);
                if !epistatic_candidates.is_empty() {
                    results.coordinated.extend(epistatic_candidates);

                    if verbose_enabled() {
                        tracing::trace!(
                            target_uuid = target_uuid,
                            epistatic_pair_count = deduped_pairs.len(),
                            "Found epistatic pair(s) for target."
                        );
                    }
                }
            }
        }

        // Issue #189: Detect synergistic candidates via residual analysis
        let synergistic_candidates =
            detect_synergistic_candidates(target_uuid, &source_contributions, target_impact);

        if !synergistic_candidates.is_empty() {
            let filtered_synergistic = filter_interfering_synergistic_candidates(
                synergistic_candidates,
                &source_contributions,
            );

            // Issue #509: Deduplicate candidates sharing a dominant (primary) neuron
            let deduped_synergistic =
                deduplicate_synergistic_by_dominant_neuron(filtered_synergistic);

            if !deduped_synergistic.is_empty() {
                let synergistic_coordinated =
                    synergistic_to_coordinated_candidates(&deduped_synergistic);
                if !synergistic_coordinated.is_empty() {
                    results.coordinated.extend(synergistic_coordinated);

                    if verbose_enabled() {
                        tracing::trace!(
                            target_uuid = target_uuid,
                            synergistic_candidate_count = deduped_synergistic.len(),
                            "Found synergistic candidate(s) for target."
                        );
                    }
                }
            }
        }
    }

    // Issue #164: Detect redundant paths
    if existing_path_contributions.len() >= 2 {
        let target_is_output = ctx
            .neuron_type_map
            .get(target_uuid)
            .map(|t| t == "output")
            .unwrap_or(false);
        let target_impact = if target_is_output { 1.0 } else { 0.5 };

        let redundant_paths =
            detect_redundant_paths(target_uuid, existing_path_contributions, target_impact);

        if !redundant_paths.is_empty() {
            let redundant_coordinated = redundant_paths_to_coordinated_candidates(&redundant_paths);
            if !redundant_coordinated.is_empty() {
                results.coordinated.extend(redundant_coordinated);

                if verbose_enabled() {
                    tracing::trace!(
                        target_uuid = target_uuid,
                        redundant_path_count = redundant_paths.len(),
                        "Found redundant path(s) for pruning on target."
                    );
                }
            }
        }
    }

    Ok(())
}

/// Pre-built harmful synapse work item for CPU/GPU overlap (Issue #568).
struct PreparedHarmfulWork {
    from_uuid: String,
    to_uuid: String,
    weight: f32,
    samples: Vec<HelpfulSample>,
}

/// Issue #568: Prepare harmful synapse samples on CPU (no GPU needed).
///
/// This function loads records from cache and builds samples, which is pure CPU work.
/// It is called while the helpful GPU batch is being processed, overlapping CPU and GPU.
fn prepare_harmful_samples(
    existing_synapses: &[SynapseJson],
    cache: &RecordCache,
    target_map_ref: &TargetMap,
) -> Vec<PreparedHarmfulWork> {
    let mut harmful_work = Vec::with_capacity(existing_synapses.len());

    for synapse in existing_synapses {
        let from_records_arc = match cache.get(&synapse.from_uuid) {
            Ok(records) => records,
            Err(_) => continue,
        };
        if from_records_arc.is_empty() {
            continue;
        }
        let from_records = from_records_arc.as_ref();
        let samples = target_map_ref.build_samples_from(from_records);
        if samples.is_empty() {
            continue;
        }

        harmful_work.push(PreparedHarmfulWork {
            from_uuid: synapse.from_uuid.clone(),
            to_uuid: synapse.to_uuid.clone(),
            weight: synapse.weight,
            samples,
        });
    }

    harmful_work
}

/// Issue #568: Process pre-built harmful samples via GPU evaluation.
fn process_harmful_batch_from_prepared(
    target_uuid: &str,
    harmful_work: &[PreparedHarmfulWork],
    gpu: &GpuWorkQueue,
    ctx: &TargetAnalysisContext,
    cache: &RecordCache,
    results: &mut TargetAnalysisResults,
) -> Result<()> {
    let batch_input: Vec<(Vec<HelpfulSample>, f32)> = harmful_work
        .iter()
        .map(|w| (w.samples.clone(), w.weight)) // Clone required: GPU queue takes ownership
        .collect();

    let batch_stats = {
        let _timing = TimingScope::shader(&ctx.timing_collector, "harmful");
        gpu.evaluate_harmful_batch(batch_input, &ctx.deadline)?
    };

    let mut harmful_candidates = Vec::with_capacity(batch_stats.len());
    let target_stats = cache
        .get(target_uuid)
        .ok()
        .and_then(|records| NeuronStats::from_records(records.as_ref()))
        .map(|s| s.to_json());

    for (work, stats) in harmful_work.iter().zip(batch_stats.iter()) {
        let total_count = work.samples.len() as u32;
        if total_count == 0 {
            continue;
        }

        let neuron_error_improvement =
            (stats.harmful_count as f32 - stats.helpful_count as f32) / total_count as f32;

        if neuron_error_improvement <= 0.0 {
            continue;
        }

        let confidence_metrics =
            compute_confidence_metrics(&work.samples, neuron_error_improvement, None);
        harmful_candidates.push(CandidateSynapseJson {
            from_neuron_uuid: work.from_uuid.clone(),
            to_neuron_uuid: work.to_uuid.clone(),
            from_neuron_index: None,
            to_neuron_index: None,
            weight: work.weight,
            target_neuron_impact: 1.0,
            expected_creature_error_reduction: neuron_error_improvement,
            expected_creature_score_gain: neuron_error_improvement,
            improved_count: stats.harmful_count,
            total_count,
            target_neuron_stats: target_stats,
            outlier_reduction_info: None,
            prediction_confidence: confidence_metrics.prediction_confidence,
            expected_score_gain_confidence_interval: confidence_metrics
                .expected_score_gain_confidence_interval,
            comment: None,
        });
    }

    results.harmful.extend(harmful_candidates);
    Ok(())
}
