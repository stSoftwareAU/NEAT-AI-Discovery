//! Neuron analysis preparation — focus target filtering, neuron type maps,
//! source ordering, record loading, and target saturation pre-check.
//!
//! Extracted from neuron.rs as part of issue #598.
//! Target saturation pre-check added as part of issue #1111.

#![allow(clippy::cast_possible_truncation)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for neural network computation (Issue #873)
use crate::AnalyzeNeuronsInput;
use crate::types::DiscoverRecord;
use anyhow::Result;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::SystemTime;

/// Type alias for shared UUID keys used across preparation maps.
/// Using `Arc<str>` avoids full string allocation when the same UUID
/// is inserted into multiple hash maps (Issue #1036).
pub(crate) type SharedUuid = Arc<str>;

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
    OrderedNeuron, deadline_passed, order_eligible_sources, parse_input_index, verbose_enabled,
};

/// Result of the preparation phase, containing all maps and filtered targets
/// needed by the main analysis loop.
pub(crate) struct NeuronPreparation<'a> {
    pub neuron_squash_map: HashMap<SharedUuid, String>,
    pub neuron_type_map: HashMap<SharedUuid, String>,
    pub order_map: HashMap<SharedUuid, usize>,
    pub focus_order: Vec<String>,
    pub unique_focus: Vec<&'a String>,
    pub original_focus_count: usize,
    pub skipped_hidden: Vec<String>,
    pub skipped_input: Vec<String>,
    pub skipped_constant: Vec<String>,
    pub threshold_targets: Vec<String>,
    pub used_inputs: HashSet<String>,
    /// Number of focus targets dropped because they are in cooldown (Issue #1130).
    pub cooldown_skipped: u32,
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
    // Issue #1036: Build lookup maps using Arc<str> keys so the same UUID string
    // is shared across all three maps via cheap reference-count increments instead
    // of full String allocations.
    let neuron_squash_map: HashMap<SharedUuid, String> = input
        .creature
        .neurons
        .iter()
        .map(|n| {
            let key: SharedUuid = Arc::from(n.uuid.as_str());
            (key, n.squash.clone())
        })
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
    let mut neuron_type_map: HashMap<SharedUuid, String> =
        HashMap::with_capacity(input.creature.input + input.creature.neurons.len());

    // Add input neurons (they're not in creature.neurons, only represented by creature.input count)
    for input_index in 0..input.creature.input {
        let key: SharedUuid = Arc::from(format!("input-{input_index}").as_str());
        neuron_type_map.insert(key, "input".to_string());
    }

    // Add all neurons from creature.neurons (hidden, output, constant)
    // Reuse the Arc<str> from neuron_squash_map where possible for zero-cost sharing.
    for neuron in &input.creature.neurons {
        let key: SharedUuid = if let Some((existing_key, _)) =
            neuron_squash_map.get_key_value(neuron.uuid.as_str())
        {
            Arc::clone(existing_key)
        } else {
            Arc::from(neuron.uuid.as_str())
        };
        neuron_type_map.insert(key, neuron.neuron_type.clone());
    }

    // Log creature configuration for debugging data issues
    if verbose_enabled() {
        log_creature_config(input, cache);
    }

    // Share Arc<str> keys from existing maps where possible.
    let order_map: HashMap<SharedUuid, usize> = ordered_neurons
        .iter()
        .map(|neuron| {
            let key: SharedUuid = if let Some((existing_key, _)) =
                neuron_squash_map.get_key_value(neuron.uuid.as_str())
            {
                Arc::clone(existing_key)
            } else if let Some((existing_key, _)) =
                neuron_type_map.get_key_value(neuron.uuid.as_str())
            {
                Arc::clone(existing_key)
            } else {
                Arc::from(neuron.uuid.as_str())
            };
            (key, neuron.index)
        })
        .collect();

    let unique_focus = require_unique_focus(&input.focus_neurons, "analyse_neurons")?;

    // Filter focus neurons to valid add-neuron targets.
    let original_focus_count = unique_focus.len();
    let output_only_targets = crate::config::neuron_targets_output_only();
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
    log_skipped_neurons(
        &skipped_hidden,
        &skipped_input,
        &skipped_constant,
        &focus_order,
    );

    // Issue #1130: drop targets currently in cooldown after focus filtering.
    let cooldown_skipped = apply_target_cooldown(&mut focus_order);

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
        cooldown_skipped,
        early_return,
    })
}

/// Drop focus targets in cooldown via the global target-failure tracker
/// (Issue #1130). Returns the number of targets removed so callers can include
/// it in diagnostics alongside the `cooldown_skipped` reason-name convention
/// from Issue #1129.
fn apply_target_cooldown(focus_order: &mut Vec<String>) -> u32 {
    use crate::analysis::target_failure_tracker::{filter_cooldown_targets, global_tracker};

    let tracker_lock = match global_tracker().lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    if tracker_lock.is_empty() {
        // Nothing to skip — avoid unnecessary work in the common case.
        return 0;
    }
    let current_epoch = tracker_lock.current_epoch();
    filter_cooldown_targets(focus_order, &tracker_lock, current_epoch)
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
    analysis_timed_out: &Arc<AtomicBool>,
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
            analysis_timed_out.store(true, Ordering::Relaxed);
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
    let timed_out_during_loading = analysis_timed_out.load(Ordering::Relaxed);
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
    // Issue #1129: populate rejection breakdown from the neuron pre-filter
    // reasons so callers can see why zero candidates were returned.
    let mut rejection_breakdown = crate::analysis::diagnostics::RejectionBreakdown::new();
    rejection_breakdown.record_many_u32(
        crate::analysis::diagnostics::rejection_reasons::REJECTION_INPUT_NEURON_FILTERED,
        u32::try_from(skipped_input.len()).unwrap_or(u32::MAX),
    );
    rejection_breakdown.record_many_u32(
        crate::analysis::diagnostics::rejection_reasons::REJECTION_HIDDEN_NEURON_FILTERED,
        u32::try_from(skipped_hidden.len()).unwrap_or(u32::MAX),
    );
    rejection_breakdown.record_many_u32(
        crate::analysis::diagnostics::rejection_reasons::REJECTION_CONSTANT_NEURON_FILTERED,
        u32::try_from(skipped_constant.len()).unwrap_or(u32::MAX),
    );
    let top_level_summary = crate::analysis::diagnostics::rejection_reasons::top_level_summary(
        &rejection_breakdown,
        None,
    );

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
            rejection_breakdown,
            top_level_summary,
            calibration_corrections: std::collections::HashMap::new(),
            // Issue #1132: populated by orchestration once the outcome log is decided.
            discovery_mode: crate::analysis::discovery_mode::DiscoveryMode::Normal,
            rolling_success_rate: 1.0,
        },
    }
}

// =============================================================================
// Target saturation pre-check (Issue #1111)
// =============================================================================

/// Threshold for the fraction of activation output range that is covered.
/// When a target neuron's observed activation range covers more than this
/// fraction of the activation function's output range, it is flagged as
/// near-saturated.
const TARGET_SATURATION_RANGE_THRESHOLD: f32 = 0.90;

/// Information about a target neuron's saturation state (Issue #1111).
///
/// Computed from the target neuron's recorded activation min/max and its
/// activation function's output bounds. Used to adjust candidate generation
/// for targets operating near activation limits.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TargetSaturationInfo {
    /// Whether the target neuron is near saturation (activation range covers
    /// >`TARGET_SATURATION_RANGE_THRESHOLD` of the output range).
    pub is_near_saturated: bool,
    /// Saturation factor (0.0 = not saturated, 1.0 = fully saturated).
    /// Measures how much of the output range is already consumed.
    pub saturation_factor: f32,
}

impl TargetSaturationInfo {
    /// A non-saturated target.
    pub const NOT_SATURATED: Self = Self {
        is_near_saturated: false,
        saturation_factor: 0.0,
    };

    /// Whether a candidate should be rejected because the target neuron is
    /// already saturated (Issue #1143).
    ///
    /// The decision depends **only** on the target's observed activation
    /// distribution relative to its squash bounds — it is deliberately
    /// independent of the candidate/intermediate squash, so the indirect
    /// add-neuron path (where a new intermediate neuron feeds into a
    /// saturated output) is gated just as thoroughly as the direct path.
    #[must_use]
    pub fn rejects_candidates(&self) -> bool {
        self.is_near_saturated
    }
}

/// Returns the output range `(min, max)` for a bounded activation function.
///
/// Returns `None` for unbounded activations (IDENTITY, RELU, ELU, etc.)
/// where saturation is not applicable.
fn bounded_output_range(squash: &str) -> Option<(f32, f32)> {
    match squash {
        "HARD_TANH" | "CLIPPED" => Some((-1.0, 1.0)),
        "TANH" | "BIPOLAR_SIGMOID" => Some((-1.0, 1.0)),
        "LOGISTIC" => Some((0.0, 1.0)),
        "SOFTSIGN" => Some((-1.0, 1.0)),
        "ARCTAN" | "ArcTan" => {
            let half_pi = std::f32::consts::FRAC_PI_2;
            Some((-half_pi, half_pi))
        }
        "RELU6" => Some((0.0, 6.0)),
        "BIPOLAR" | "STEP" => Some((-1.0, 1.0)),
        _ => None, // Unbounded: IDENTITY, RELU, GELU, ELU, Softplus, Mish, etc.
    }
}

/// Returns `true` if a candidate activation function's output range compounds
/// clipping with the target's bounded activation (Issue #1111).
///
/// For example, `ABSOLUTE` feeding into `HARD_TANH`: `ABSOLUTE` outputs [0, ∞), but
/// `HARD_TANH` clips to [-1, 1]. The positive-only output of `ABSOLUTE` means
/// only [0, 1] is useful, halving the effective target range and worsening
/// saturation.
pub(crate) fn compounds_target_clipping(candidate_squash: &str, target_squash: &str) -> bool {
    let Some((target_min, _)) = bounded_output_range(target_squash) else {
        return false;
    };
    // ABSOLUTE outputs [0, ∞) — for targets with negative lower bound,
    // the positive-only output wastes the negative half of the target range.
    if candidate_squash == "ABSOLUTE" && target_min < 0.0 {
        return true;
    }
    false
}

/// Compute the saturation state of a target neuron from its recorded data.
///
/// Examines `activation_min` and `activation_max` from the target's records
/// and compares them against the activation function's output bounds to
/// determine whether the target is operating near saturation.
///
/// Reuses the `HARD_TANH` saturation threshold (0.95) from
/// `detection/saturation.rs` where applicable.
pub(crate) fn compute_target_saturation(
    target_records: &[DiscoverRecord],
    target_squash: &str,
) -> TargetSaturationInfo {
    let Some((out_min, out_max)) = bounded_output_range(target_squash) else {
        return TargetSaturationInfo::NOT_SATURATED;
    };

    if target_records.is_empty() {
        return TargetSaturationInfo::NOT_SATURATED;
    }

    // Compute observed activation min/max from records
    let mut act_min = f32::INFINITY;
    let mut act_max = f32::NEG_INFINITY;
    let mut valid_count = 0u32;

    for record in target_records {
        let a = record.activation;
        if a.is_finite() {
            if a < act_min {
                act_min = a;
            }
            if a > act_max {
                act_max = a;
            }
            valid_count += 1;
        }
    }

    if valid_count < 2 {
        return TargetSaturationInfo::NOT_SATURATED;
    }

    let output_range = out_max - out_min;
    if output_range <= 0.0 {
        return TargetSaturationInfo::NOT_SATURATED;
    }

    let observed_range = act_max - act_min;
    let range_coverage = observed_range / output_range;

    // Saturation factor: how much of the output range is consumed (clamped to [0, 1])
    let saturation_factor = range_coverage.clamp(0.0, 1.0);
    let is_near_saturated = saturation_factor > TARGET_SATURATION_RANGE_THRESHOLD;

    if is_near_saturated {
        tracing::debug!(
            target_squash,
            act_min = format_args!("{act_min:.4}"),
            act_max = format_args!("{act_max:.4}"),
            out_min = format_args!("{out_min:.4}"),
            out_max = format_args!("{out_max:.4}"),
            range_coverage = format_args!("{range_coverage:.4}"),
            saturation_factor = format_args!("{saturation_factor:.4}"),
            "Target neuron near saturation — adjusting candidate generation"
        );
    }

    TargetSaturationInfo {
        is_near_saturated,
        saturation_factor,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that `Arc<str>` map keys support lookup via `&str` (Issue #1036).
    #[test]
    fn test_shared_uuid_map_lookup() {
        let mut map: HashMap<SharedUuid, String> = HashMap::new();
        let key: SharedUuid = Arc::from("output-abc-123");
        map.insert(Arc::clone(&key), "TANH".to_string());

        // Lookup via &str (the pattern used in analysis hot paths)
        assert_eq!(map.get("output-abc-123").map(String::as_str), Some("TANH"));

        // Lookup via &String
        let query = "output-abc-123".to_string();
        assert_eq!(map.get(query.as_str()).map(String::as_str), Some("TANH"));

        // Lookup for missing key
        assert!(!map.contains_key("missing-uuid"));
    }

    /// Verify that `Arc<str>` keys are shared across maps (Issue #1036).
    /// The same Arc pointer should be reused when building multiple maps
    /// from the same neuron UUIDs.
    #[test]
    fn test_shared_uuid_arc_reuse() {
        let key1: SharedUuid = Arc::from("hidden-neuron-1");
        let key2 = Arc::clone(&key1);

        // Both references point to the same allocation
        assert!(Arc::ptr_eq(&key1, &key2));

        // Insert into separate maps — both use the same backing allocation
        let mut map_a: HashMap<SharedUuid, usize> = HashMap::new();
        let mut map_b: HashMap<SharedUuid, String> = HashMap::new();
        map_a.insert(Arc::clone(&key1), 42);
        map_b.insert(Arc::clone(&key1), "output".to_string());

        assert_eq!(map_a.get("hidden-neuron-1"), Some(&42));
        assert_eq!(
            map_b.get("hidden-neuron-1").map(String::as_str),
            Some("output")
        );
    }

    // =========================================================================
    // Target saturation pre-check tests (Issue #1111)
    // =========================================================================

    /// Helper to create a `DiscoverRecord` with a given activation value.
    fn make_record(activation: f32) -> DiscoverRecord {
        DiscoverRecord {
            obs_index: 0,
            neuron_uuid: "test".to_string(),
            value: None,
            activation,
            errors: vec![0.1],
        }
    }

    /// `HARD_TANH` target with activations spanning [-1, 1] triggers the
    /// saturation pre-check (full range coverage).
    #[test]
    fn test_hard_tanh_saturated_target_triggers_precheck() {
        let records: Vec<DiscoverRecord> = vec![
            make_record(-1.0),
            make_record(-0.5),
            make_record(0.0),
            make_record(0.5),
            make_record(1.0),
        ];
        let info = compute_target_saturation(&records, "HARD_TANH");
        assert!(
            info.is_near_saturated,
            "HARD_TANH at full range should be near-saturated"
        );
        assert!(
            (info.saturation_factor - 1.0).abs() < 0.01,
            "Saturation factor should be ~1.0, got {}",
            info.saturation_factor
        );
    }

    /// IDENTITY target (unbounded) does NOT trigger the pre-check.
    #[test]
    fn test_identity_target_not_saturated() {
        let records: Vec<DiscoverRecord> =
            vec![make_record(-100.0), make_record(0.0), make_record(100.0)];
        let info = compute_target_saturation(&records, "IDENTITY");
        assert!(
            !info.is_near_saturated,
            "IDENTITY target should never be flagged as near-saturated"
        );
        assert!(
            info.saturation_factor < 0.01,
            "IDENTITY saturation factor should be ~0.0, got {}",
            info.saturation_factor
        );
    }

    /// TANH target with narrow activation range (e.g., [-0.3, 0.3]) is not saturated.
    #[test]
    fn test_tanh_narrow_range_not_saturated() {
        let records: Vec<DiscoverRecord> =
            vec![make_record(-0.3), make_record(0.0), make_record(0.3)];
        let info = compute_target_saturation(&records, "TANH");
        assert!(
            !info.is_near_saturated,
            "TANH with narrow range should not be near-saturated"
        );
        assert!(
            info.saturation_factor < TARGET_SATURATION_RANGE_THRESHOLD,
            "Saturation factor {} should be below {}",
            info.saturation_factor,
            TARGET_SATURATION_RANGE_THRESHOLD
        );
    }

    /// LOGISTIC target spanning [0.02, 0.98] covers 96% of [0, 1] — saturated.
    #[test]
    fn test_logistic_near_saturated() {
        let records: Vec<DiscoverRecord> =
            vec![make_record(0.02), make_record(0.5), make_record(0.98)];
        let info = compute_target_saturation(&records, "LOGISTIC");
        assert!(
            info.is_near_saturated,
            "LOGISTIC spanning 96% of range should be near-saturated"
        );
        assert!(
            info.saturation_factor > 0.90,
            "Saturation factor should be >0.90, got {}",
            info.saturation_factor
        );
    }

    /// Empty records return non-saturated.
    #[test]
    fn test_empty_records_not_saturated() {
        let info = compute_target_saturation(&[], "HARD_TANH");
        assert!(!info.is_near_saturated);
        assert!(info.saturation_factor < 0.01);
    }

    /// `ABSOLUTE` feeding into `HARD_TANH` compounds clipping.
    #[test]
    fn test_absolute_compounds_hard_tanh_clipping() {
        assert!(compounds_target_clipping("ABSOLUTE", "HARD_TANH"));
        assert!(compounds_target_clipping("ABSOLUTE", "TANH"));
        assert!(!compounds_target_clipping("IDENTITY", "HARD_TANH"));
        // LOGISTIC has min=0 (non-negative), so ABSOLUTE doesn't compound
        assert!(!compounds_target_clipping("ABSOLUTE", "LOGISTIC"));
        // Unbounded target — no clipping to compound
        assert!(!compounds_target_clipping("ABSOLUTE", "IDENTITY"));
    }

    /// Verify `bounded_output_range` returns correct ranges for known activations.
    #[test]
    fn test_bounded_output_range() {
        assert_eq!(bounded_output_range("HARD_TANH"), Some((-1.0, 1.0)));
        assert_eq!(bounded_output_range("CLIPPED"), Some((-1.0, 1.0)));
        assert_eq!(bounded_output_range("TANH"), Some((-1.0, 1.0)));
        assert_eq!(bounded_output_range("LOGISTIC"), Some((0.0, 1.0)));
        assert_eq!(bounded_output_range("RELU6"), Some((0.0, 6.0)));
        assert!(bounded_output_range("IDENTITY").is_none());
        assert!(bounded_output_range("RELU").is_none());
        assert!(bounded_output_range("GELU").is_none());
    }

    /// RELU target (unbounded) returns not-saturated.
    #[test]
    fn test_relu_target_not_saturated() {
        let records = vec![make_record(0.0), make_record(5.0), make_record(10.0)];
        let info = compute_target_saturation(&records, "RELU");
        assert!(!info.is_near_saturated);
    }

    // =========================================================================
    // Issue #1143: Gate rejects every candidate when the target is saturated,
    // independent of the candidate/intermediate squash. Verifies the fix for
    // the indirect add-neuron path (saturated HARD_TANH output accepting
    // ArcTan / BENT_IDENTITY intermediates).
    // =========================================================================

    /// Helper: saturated `HARD_TANH` target matching the GRQ-sampler evidence.
    fn saturated_hard_tanh() -> TargetSaturationInfo {
        let records: Vec<DiscoverRecord> = vec![
            make_record(-1.0),
            make_record(-0.5),
            make_record(0.0),
            make_record(0.5),
            make_record(1.0),
        ];
        let info = compute_target_saturation(&records, "HARD_TANH");
        assert!(info.is_near_saturated, "fixture must be saturated");
        info
    }

    /// (a) Saturated `HARD_TANH` + `ArcTan` intermediate → rejected.
    #[test]
    fn test_saturated_target_rejects_arctan_intermediate() {
        let info = saturated_hard_tanh();
        assert!(
            info.rejects_candidates(),
            "saturated HARD_TANH target must reject candidates regardless of the \
             ArcTan intermediate squash (Issue #1143)"
        );
    }

    /// (b) Saturated `HARD_TANH` + any intermediate → rejected.
    ///
    /// The gate must be independent of the candidate squash — verify across
    /// a representative set of intermediates including `BENT_IDENTITY`,
    /// `IDENTITY`, `RELU`, `TANH`, and `GELU`.
    #[test]
    fn test_saturated_target_rejects_all_intermediates() {
        let info = saturated_hard_tanh();
        // The gate itself is squash-agnostic. Enumerate the squashes that the
        // activation-spec batch evaluator considers to make the intent
        // explicit — every one must be rejected.
        let intermediates = [
            "ArcTan",
            "BENT_IDENTITY",
            "IDENTITY",
            "RELU",
            "TANH",
            "LOGISTIC",
            "GELU",
            "SOFTSIGN",
        ];
        for squash in intermediates {
            assert!(
                info.rejects_candidates(),
                "saturated target must reject candidate with intermediate squash {squash}"
            );
        }
    }

    /// (c) Non-saturated target + any intermediate → kept.
    #[test]
    fn test_non_saturated_target_keeps_candidates() {
        // TANH target covering only [-0.3, 0.3] — nowhere near bounds.
        let records: Vec<DiscoverRecord> =
            vec![make_record(-0.3), make_record(0.0), make_record(0.3)];
        let info = compute_target_saturation(&records, "TANH");
        assert!(!info.is_near_saturated);
        let intermediates = ["ArcTan", "BENT_IDENTITY", "IDENTITY", "RELU"];
        for squash in intermediates {
            assert!(
                !info.rejects_candidates(),
                "non-saturated target must keep candidate with intermediate {squash}"
            );
        }
    }

    /// Unbounded IDENTITY target never triggers the gate.
    #[test]
    fn test_unbounded_target_never_rejects() {
        let records = vec![make_record(-1000.0), make_record(0.0), make_record(1000.0)];
        let info = compute_target_saturation(&records, "IDENTITY");
        assert!(
            !info.rejects_candidates(),
            "unbounded targets cannot saturate and must never gate candidates"
        );
    }

    /// The `NOT_SATURATED` sentinel never rejects.
    #[test]
    fn test_not_saturated_sentinel_keeps_candidates() {
        assert!(!TargetSaturationInfo::NOT_SATURATED.rejects_candidates());
    }
}
