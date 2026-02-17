//! Neuron analysis preparation — focus target filtering, neuron type maps,
//! source ordering, and record loading.
//!
//! Extracted from neuron.rs as part of issue #598.

use crate::AnalyzeNeuronsInput;
use crate::types::DiscoverRecord;
use anyhow::Result;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use crate::analysis::cache::RecordCache;
use crate::analysis::diagnostics::{
    FocusTargetFilterResult, NeuronDiagnostics, filter_focus_targets_for_neuron_analysis,
    require_unique_focus,
};
use crate::analysis::gpu::GpuAnalyzer;
use crate::analysis::shared::{
    AnalyzeNeuronsResult, NeuronNoCandidateReason, NeuronNoCandidateSummary,
};
use crate::analysis::utils::{
    OrderedNeuron, deadline_passed, lock_or_bail, order_eligible_sources, parse_input_index,
    verbose_enabled,
};

/// Result of the preparation phase, containing all maps and filtered targets
/// needed by the main analysis loop.
pub(crate) struct NeuronPreparation<'a> {
    pub neuron_squash_map: HashMap<String, String>,
    pub neuron_type_map: HashMap<String, String>,
    pub order_map: HashMap<String, usize>,
    pub focus_order: Vec<String>,
    pub unique_focus: Vec<&'a String>,
    pub original_focus_count: usize,
    pub skipped_hidden: Vec<String>,
    pub skipped_input: Vec<String>,
    pub skipped_constant: Vec<String>,
    pub threshold_targets: Vec<String>,
    pub used_inputs: HashSet<String>,
    /// If set, the caller should return this immediately (early exit path).
    pub early_return: Option<AnalyzeNeuronsResult>,
}

/// Prepare all lookup maps, filter focus targets, and determine if an early
/// return is needed (e.g., no output neurons remain after filtering).
pub(crate) fn prepare_neuron_analysis<'a>(
    input: &'a AnalyzeNeuronsInput,
    ordered_neurons: &[OrderedNeuron],
    cache: &Arc<RecordCache>,
) -> Result<NeuronPreparation<'a>> {
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
        log_creature_config(input, cache);
    }

    let order_map: HashMap<String, usize> = ordered_neurons
        .iter()
        .map(|neuron| (neuron.uuid.clone(), neuron.index))
        .collect();

    let unique_focus = require_unique_focus(&input.focus_neurons, "analyse_neurons")?;

    // Filter focus neurons to valid add-neuron targets.
    let original_focus_count = unique_focus.len();
    let output_only_targets = std::env::var("NEAT_AI_DISCOVERY_NEURON_TARGETS_OUTPUT_ONLY").is_ok();
    let FocusTargetFilterResult {
        focus_order,
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
    log_skipped_neurons(
        &skipped_hidden,
        &skipped_input,
        &skipped_constant,
        &focus_order,
    );

    // If no output neurons remain after filtering, return early with empty results
    let early_return = if focus_order.is_empty() {
        Some(build_empty_result(
            original_focus_count,
            &skipped_input,
            &skipped_hidden,
            &skipped_constant,
        ))
    } else {
        None
    };

    // Issue #182: Build a set of "used" input neurons (those with at least one outgoing synapse)
    let used_inputs: HashSet<String> = input
        .creature
        .synapses
        .iter()
        .filter(|s| parse_input_index(&s.from_uuid).is_some())
        .map(|s| s.from_uuid.clone())
        .collect();

    Ok(NeuronPreparation {
        neuron_squash_map,
        neuron_type_map,
        order_map,
        focus_order,
        unique_focus,
        original_focus_count,
        skipped_hidden,
        skipped_input,
        skipped_constant,
        threshold_targets,
        used_inputs,
        early_return,
    })
}

/// Load source neuron records for a given target, filtering by eligibility
/// and checking deadlines during the process.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn load_source_records<'a>(
    target_uuid: &str,
    target_index: usize,
    ordered_neurons_arc: &'a Arc<Vec<OrderedNeuron>>,
    input: &AnalyzeNeuronsInput,
    used_inputs_arc: &Arc<HashSet<String>>,
    cache: &Arc<RecordCache>,
    deadline: &Option<SystemTime>,
    analysis_timed_out: &Arc<Mutex<bool>>,
    diagnostics: &Arc<NeuronDiagnostics>,
) -> Result<Vec<(&'a OrderedNeuron, Arc<Vec<DiscoverRecord>>)>> {
    let mut eligible_sources: Vec<&OrderedNeuron> = ordered_neurons_arc
        .iter()
        .filter(|neuron| neuron.index < target_index)
        .collect();

    // Source ordering with shuffle for coverage under timeouts
    let context = format!("neuron:eligible_sources:{target_uuid}");
    order_eligible_sources(
        &mut eligible_sources,
        input.random_seed,
        &context,
        input.creature.input,
        Some(&**used_inputs_arc),
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

    // Pre-filter sources and collect their records (with deadline checks)
    let mut sources_to_process: Vec<(&OrderedNeuron, Arc<Vec<DiscoverRecord>>)> =
        Vec::with_capacity(eligible_sources.len());
    let mut empty_record_sources: Vec<String> = Vec::new();
    let mut load_failure_count = 0u32;

    for source in &eligible_sources {
        if deadline_passed(deadline) {
            *lock_or_bail(analysis_timed_out, "analysis_timed_out")? = true;
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

    // Record load failures in diagnostics
    for _ in 0..load_failure_count {
        diagnostics.record_load_failure(target_uuid);
    }

    // Log summary of source loading results for debugging
    let sources_checked =
        sources_to_process.len() + empty_record_sources.len() + load_failure_count as usize;
    let timed_out_during_loading = *lock_or_bail(analysis_timed_out, "analysis_timed_out")?;
    if verbose_enabled()
        && (sources_to_process.is_empty()
            || load_failure_count > 0
            || !empty_record_sources.is_empty()
            || timed_out_during_loading)
    {
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

    // Batch diagnostics for empty record sources
    for source_uuid in &empty_record_sources {
        diagnostics.record_candidate_attempt(target_uuid, false);
        diagnostics.record_no_samples(target_uuid, source_uuid);
    }

    Ok(sources_to_process)
}

/// Log creature configuration for debugging data issues.
fn log_creature_config(input: &AnalyzeNeuronsInput, cache: &Arc<RecordCache>) {
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

/// Log when non-output neurons are filtered out.
fn log_skipped_neurons(
    skipped_hidden: &[String],
    skipped_input: &[String],
    skipped_constant: &[String],
    focus_order: &[String],
) {
    let total_skipped = skipped_hidden.len() + skipped_input.len() + skipped_constant.len();
    if total_skipped > 0 {
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
}

/// Build an empty result for the early return path when no output neurons remain.
fn build_empty_result(
    original_focus_count: usize,
    skipped_input: &[String],
    skipped_hidden: &[String],
    skipped_constant: &[String],
) -> AnalyzeNeuronsResult {
    tracing::warn!(
        original_focus_count,
        "No output neurons in focus list — add-neuron candidates can only target output neurons"
    );
    let mut no_candidate_reasons: Vec<NeuronNoCandidateSummary> = Vec::new();
    for uuid in skipped_input {
        no_candidate_reasons.push(NeuronNoCandidateSummary {
            target_uuid: uuid.clone(),
            reason: NeuronNoCandidateReason::InputNeuronFiltered,
            evaluated_sources: 0,
            sources_with_samples: 0,
            target_record_count: 0,
            detail: None,
        });
    }
    for uuid in skipped_hidden {
        no_candidate_reasons.push(NeuronNoCandidateSummary {
            target_uuid: uuid.clone(),
            reason: NeuronNoCandidateReason::HiddenNeuronFiltered,
            evaluated_sources: 0,
            sources_with_samples: 0,
            target_record_count: 0,
            detail: None,
        });
    }
    for uuid in skipped_constant {
        no_candidate_reasons.push(NeuronNoCandidateSummary {
            target_uuid: uuid.clone(),
            reason: NeuronNoCandidateReason::ConstantNeuronFiltered,
            evaluated_sources: 0,
            sources_with_samples: 0,
            target_record_count: 0,
            detail: None,
        });
    }
    AnalyzeNeuronsResult {
        helpful_neurons: Vec::new(),
        gpu_used: true,
        no_candidate_reasons,
        metadata: crate::analysis::shared::NeuronAnalysisMetadata {
            candidates_found: 0,
            candidates_returned: 0,
            timed_out: false,
            completed_focus_neurons: 0,
            total_focus_neurons: original_focus_count,
            timing: None,
            gpu_info: GpuAnalyzer::get_adapter_info(),
            error_distribution: None,
        },
    }
}
