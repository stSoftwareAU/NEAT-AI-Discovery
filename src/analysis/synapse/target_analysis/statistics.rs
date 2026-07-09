//! Statistical calculations for target analysis
//!
//! This module handles source filtering, record loading, diagnostics tracking,
//! and sample building orchestration for per-target synapse analysis.
//!
//! Extracted from `target_analysis.rs` as part of Issue #599.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use crate::SynapseJson;
use crate::analysis::cache::RecordCache;
use crate::analysis::diagnostics::TargetMap;
use crate::analysis::samples::HelpfulSample;
use crate::analysis::shared::TimingScope;
use crate::analysis::utils::{OrderedNeuron, deadline_passed, parse_input_index, verbose_enabled};
use crate::types::DiscoverRecord;
use rayon::prelude::*;
use std::sync::Arc;

use super::{
    ExistingSourceToProcess, HelpfulWork, InputMetadata, SourceWorkResult, TargetAnalysisContext,
};
use crate::analysis::synapse::candidate_generation::{
    MIN_GROUP_SIZE_FOR_LOCALITY, build_samples_for_locality_group, group_sources_by_locality,
};

/// Pre-filter eligible source neurons and collect their records from cache.
///
/// Returns the new-synapse sources, existing-weight sources, and updated input metadata.
#[allow(clippy::type_complexity)]
pub(crate) fn filter_and_load_sources<'a>(
    eligible_sources: &[&'a OrderedNeuron],
    target_uuid: &str,
    cache: &RecordCache,
    ctx: &TargetAnalysisContext,
    input_metadata: &mut InputMetadata,
) -> (
    Vec<(&'a OrderedNeuron, Arc<Vec<DiscoverRecord>>)>,
    Vec<ExistingSourceToProcess<'a>>,
) {
    let mut already_connected_count = 0u32;
    let mut load_failure_count = 0u32;
    let mut empty_record_sources: Vec<String> = Vec::new();

    let mut sources_to_process: Vec<(&OrderedNeuron, Arc<Vec<DiscoverRecord>>)> =
        Vec::with_capacity(eligible_sources.len());
    let mut existing_sources_to_process: Vec<ExistingSourceToProcess> =
        Vec::with_capacity(eligible_sources.len());

    for source in eligible_sources {
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
                        input_metadata.seen_any = true;
                        if input_index < input_metadata.min_index {
                            input_metadata.min_index = input_index;
                        }
                        if input_index > input_metadata.max_index {
                            input_metadata.max_index = input_index;
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

    (sources_to_process, existing_sources_to_process)
}

/// Build helpful synapse work items from source neurons using locality grouping.
///
/// Groups sources by observation index overlap and builds samples in parallel,
/// then collects work items and diagnostics updates.
pub(crate) fn build_helpful_work_items(
    sources_to_process: &[(&OrderedNeuron, Arc<Vec<DiscoverRecord>>)],
    target_uuid: &str,
    target_map_ref: &TargetMap,
    ctx: &TargetAnalysisContext,
) -> Vec<HelpfulWork> {
    let source_results: Vec<SourceWorkResult<'_>> = {
        let _timing = TimingScope::sample_building(&ctx.timing_collector);

        let locality_groups = group_sources_by_locality(sources_to_process);

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
                                source_uuid: source_uuid.to_string(),
                                target_uuid: target_uuid.to_string(),
                                // Issue #1548: Arc-wrap once so the GPU submit
                                // path clones a refcount, not the sample Vec.
                                samples: Arc::new(samples),
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

    // Extract work batch and update diagnostics directly (Issue #808:
    // use &str source_uuid to avoid intermediate String allocations)
    let mut helpful_work_batch: Vec<HelpfulWork> = Vec::new();

    for result in source_results {
        ctx.diagnostics
            .record_candidate_attempt(target_uuid, result.had_samples);
        if !result.had_samples {
            ctx.diagnostics
                .record_no_samples(target_uuid, result.source_uuid, result.record_count);
        }
        if let Some(work) = result.work {
            helpful_work_batch.push(work);
        }
    }

    helpful_work_batch
}

/// Build work items from existing synapse edges for weight-update evaluation.
///
/// Returns the path contributions and appends work items to the helpful batch.
pub(crate) fn build_existing_edge_work(
    existing_sources_to_process: &[ExistingSourceToProcess],
    target_uuid: &str,
    target_map_ref: &TargetMap,
    helpful_work_batch: &mut Vec<HelpfulWork>,
) -> Vec<crate::analysis::detection::redundant_path::ExistingPathContribution> {
    use crate::analysis::detection::redundant_path::ExistingPathContribution;

    let existing_path_contributions: Vec<ExistingPathContribution> = existing_sources_to_process
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
    // Issue #1548: the contribution keeps its own copy for redundant-path
    // detection, so one deep clone is unavoidable here; wrapping in Arc lets
    // the subsequent GPU submit share the buffer without re-cloning.
    helpful_work_batch.extend(existing_path_contributions.iter().map(|c| HelpfulWork {
        source_uuid: c.source_uuid.clone(),
        target_uuid: target_uuid.to_string(),
        samples: Arc::new(c.samples.clone()),
        existing_weight: Some(c.existing_weight),
    }));

    existing_path_contributions
}

/// Prepare harmful synapse samples on CPU (no GPU needed).
///
/// This function loads records from cache and builds samples, which is pure CPU work.
/// It is called while the helpful GPU batch is being processed, overlapping CPU and GPU.
/// (Issue #568)
pub(crate) fn prepare_harmful_samples<'a>(
    existing_synapses: &[&'a SynapseJson],
    cache: &RecordCache,
    target_map_ref: &TargetMap,
) -> Vec<PreparedHarmfulWork<'a>> {
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
            from_uuid: synapse.from_uuid.as_str(),
            to_uuid: synapse.to_uuid.as_str(),
            weight: synapse.weight,
            // Issue #1548: Arc-wrap so the harmful GPU submit shares the buffer
            // rather than deep-copying it for queue ownership.
            samples: Arc::new(samples),
        });
    }

    harmful_work
}

/// Pre-built harmful synapse work item for CPU/GPU overlap (Issue #568).
///
/// Uses `&str` references to synapse UUIDs from `ctx.synapses_by_target`
/// to avoid cloning in the per-target hot path (Issue #808).
pub(crate) struct PreparedHarmfulWork<'a> {
    pub from_uuid: &'a str,
    pub to_uuid: &'a str,
    pub weight: f32,
    /// Issue #1548: `Arc`-shared so the GPU submit path takes a refcount clone
    /// instead of deep-copying the sample `Vec` for queue ownership.
    pub samples: Arc<Vec<HelpfulSample>>,
}
