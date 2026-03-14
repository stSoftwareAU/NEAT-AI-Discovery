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
//! Split into sub-modules as part of Issue #599.
//!
//! ## Sub-modules
//!
//! - `evaluation` — GPU work submission, result collection, helpful/harmful processing
//! - `candidate_selection` — Epistatic, synergistic, and redundant path detection
//! - `statistics` — Source filtering, record loading, sample building orchestration

mod candidate_selection;
mod evaluation;
mod statistics;

use crate::analysis::cache::RecordCache;
use crate::analysis::diagnostics::TargetMap;
use crate::analysis::gpu::GpuWorkQueue;
use crate::analysis::samples::HelpfulSample;
use crate::analysis::utils::{
    OrderedNeuron, deadline_passed, order_eligible_sources, verbose_enabled,
};
use crate::intern::NeuronIndex;
use crate::types::DiscoverRecord;
use crate::{AnalyzeSynapsesInput, CandidateSynapseJson, SynapseJson};
use anyhow::{Result, anyhow};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::SystemTime;

/// Shared context for per-target analysis, built once in the main pipeline.
///
/// Maps that reference input data use `&str` to avoid cloning (Issue #808).
/// The lifetime `'a` is tied to the `AnalyzeSynapsesInput` reference.
pub(crate) struct TargetAnalysisContext<'a> {
    pub ordered_neurons: Arc<Vec<OrderedNeuron>>,
    pub order_map: Arc<HashMap<String, usize>>,
    pub neuron_index: Arc<NeuronIndex>,
    pub existing_synapses: Arc<HashSet<(u32, u32)>>,
    pub existing_synapse_weights: Arc<HashMap<(u32, u32), f32>>,
    pub synapses_by_target: Arc<HashMap<u32, Vec<SynapseJson>>>,
    pub neuron_squash_map: Arc<HashMap<&'a str, &'a str>>,
    pub neuron_type_map: Arc<HashMap<String, String>>,
    pub input_neuron_uuids: Arc<HashSet<String>>,
    pub used_inputs: Arc<HashSet<&'a str>>,
    pub neuron_bias_map: Arc<HashMap<&'a str, f32>>,
    pub constant_source_effect_threshold: Option<f32>,
    pub diagnostics: Arc<crate::analysis::diagnostics::TargetDiagnostics>,
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
///
/// Uses `&str` for `source_uuid` to avoid cloning UUIDs in the hot path
/// (Issue #808). The reference points to `OrderedNeuron.uuid` which lives
/// in `ctx.ordered_neurons` (Arc'd) for the duration of analysis.
struct SourceWorkResult<'a> {
    work: Option<HelpfulWork>,
    had_samples: bool,
    source_uuid: &'a str,
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
                        .is_some_and(|t| t == "constant")
            })
            .count();
        let input_neurons_before_index = ctx
            .ordered_neurons
            .iter()
            .filter(|n| n.index < target_index && ctx.input_neuron_uuids.contains(&n.uuid))
            .count();

        tracing::error!(
            target_uuid = target_uuid,
            target_neuron_type = target_neuron_type,
            target_index = target_index,
            input_count = input_count,
            neurons_before_target = neurons_before_index,
            constants_before_target = constants_before_index,
            input_neurons_before_target = input_neurons_before_index,
            "Target has no eligible upstream neurons — this should not happen for \
             hidden/output neurons with index >= creature.input"
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
    let (sources_to_process, existing_sources_to_process) = statistics::filter_and_load_sources(
        &eligible_sources,
        target_uuid,
        cache,
        ctx,
        &mut results.input_metadata,
    );

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

    // Issue #221: Sample Locality Optimisation — build helpful work items
    let mut helpful_work_batch =
        statistics::build_helpful_work_items(&sources_to_process, target_uuid, target_map_ref, ctx);

    // Append existing edges for weight-update evaluation
    let existing_path_contributions = statistics::build_existing_edge_work(
        &existing_sources_to_process,
        target_uuid,
        target_map_ref,
        &mut helpful_work_batch,
    );

    // Issue #568: Overlap CPU analysis with GPU computation.
    // Submit the helpful GPU batch non-blocking, then prepare harmful samples
    // (CPU work) while the GPU processes the helpful batch.
    let helpful_future = if !helpful_work_batch.is_empty() {
        Some(evaluation::submit_helpful_gpu_work(
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
            .map(|existing| statistics::prepare_harmful_samples(existing, cache, target_map_ref))
    } else {
        None
    };

    // Collect helpful GPU results and process them
    if let Some(future) = helpful_future {
        evaluation::collect_and_process_helpful_results(
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
        evaluation::process_harmful_batch_from_prepared(
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
