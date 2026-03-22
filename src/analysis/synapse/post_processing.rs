//! Post-processing for synapse analysis results
//!
//! This module handles impact-based discounting, sorting, diversification,
//! truncation, and metadata assembly for synapse analysis candidates.
//! Extracted from mod.rs as part of Issue #482.

use crate::CandidateSynapseJson;
use crate::analysis::diagnostics::compute_impact_scores_for_discounting;
use crate::analysis::utils::{shuffle_within_top_k, verbose_enabled};
use std::collections::HashMap;

use super::filtering::truncate_combined_synapse_candidate_sets;
use super::scoring::{
    apply_prediction_calibration, apply_source_type_boost, apply_synapse_pessimism_discount,
    apply_target_type_boost,
};
use crate::analysis::cache::RecordCache;
use crate::analysis::samples::EPSILON;

/// Scale a raw neuron-level prediction by the target neuron's fraction of total creature error.
///
/// Issue #730: Raw improvement predictions measure error reduction at a single target neuron,
/// but these do not translate directly to creature-level score changes. A synapse improving
/// one neuron's error by 5% does not mean 5% creature improvement — it depends on what
/// fraction of total creature error that neuron contributes.
///
/// ## Formula
///
/// ```text
/// error_fraction = target_error_sq / total_error_sq
/// scaled_prediction = raw_prediction × error_fraction
/// ```
///
/// ## Arguments
///
/// * `raw_prediction` — Neuron-level improvement (fraction of target error reduced)
/// * `target_error_sq` — Sum of squared errors for the target neuron
/// * `total_error_sq` — Sum of squared errors across all focus neurons
pub fn scale_by_error_fraction(
    raw_prediction: f32,
    target_error_sq: f32,
    total_error_sq: f32,
) -> f32 {
    if total_error_sq <= EPSILON {
        return 0.0;
    }
    let fraction = (target_error_sq / total_error_sq).clamp(0.0, 1.0);
    raw_prediction * fraction
}

/// Compute sum of squared errors for each neuron in the creature from the cache.
///
/// Issue #730: Used to determine what fraction of total creature error each target
/// neuron contributes, enabling creature-level prediction calibration.
fn compute_neuron_error_sq_map(
    input: &crate::AnalyzeSynapsesInput,
    cache: &RecordCache,
) -> HashMap<String, f32> {
    let mut error_sq_map = HashMap::new();
    for neuron in &input.creature.neurons {
        if let Ok(records) = cache.get(&neuron.uuid) {
            let error_sq: f32 = records
                .iter()
                .flat_map(|r| r.errors.iter())
                .filter(|e| e.is_finite())
                .map(|e| e * e)
                .sum();
            if error_sq > EPSILON {
                error_sq_map.insert(neuron.uuid.clone(), error_sq);
            }
        }
    }
    error_sq_map
}

/// Apply impact-based discounting to a single helpful synapse candidate.
///
/// Updates `target_neuron_impact`, `expected_creature_error_reduction`,
/// and `expected_creature_score_gain` based on the target neuron's distance
/// from outputs. Also applies creature-level error fraction scaling (Issue #730)
/// and source-type and target-type boosts (Issues #467, #468).
fn apply_impact_to_helpful(
    candidate: &mut CandidateSynapseJson,
    impact_scores: &HashMap<String, f32>,
    neuron_type_map: &HashMap<String, String>,
    order_map: &HashMap<String, usize>,
    target_error_sq: f32,
    total_error_sq: f32,
) {
    // Populate indices for debugging/analysis (consistent with CandidateNeuronJson).
    candidate.from_neuron_index = order_map.get(&candidate.from_neuron_uuid).copied();
    candidate.to_neuron_index = order_map.get(&candidate.to_neuron_uuid).copied();

    let is_hidden = neuron_type_map
        .get(&candidate.to_neuron_uuid)
        .is_none_or(|t| t != "output"); // Default to hidden if type unknown

    let impact = if is_hidden {
        if let Some(&impact) = impact_scores.get(&candidate.to_neuron_uuid) {
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

    // Issue #730: Scale by target neuron's fraction of total creature error.
    // This converts neuron-level improvement to creature-level improvement.
    candidate.expected_creature_error_reduction = scale_by_error_fraction(
        candidate.expected_creature_error_reduction,
        target_error_sq,
        total_error_sq,
    );

    candidate.expected_creature_error_reduction *= impact;
    candidate.expected_creature_score_gain = candidate.expected_creature_error_reduction;

    // Issue #789: Apply synapse-specific pessimism discount based on improved sample ratio.
    // Raw improvement percentages are neuron-level estimates that do not generalise
    // directly to creature-level score gains (18,500× over-estimation in production).
    // Synapse candidates use the most aggressive discounting because the multi-weight
    // search (9 variants) creates selection bias that overfits to sample data,
    // contributing to the 0% success rate observed in GRQ-sampler cache data.
    candidate.expected_creature_score_gain = apply_synapse_pessimism_discount(
        candidate.expected_creature_score_gain,
        candidate.improved_count,
        candidate.total_count,
    );

    // Issue #467: Apply source-type prioritisation boost for input-neuron sources.
    candidate.expected_creature_score_gain = apply_source_type_boost(
        candidate.expected_creature_score_gain,
        &candidate.from_neuron_uuid,
    );

    // Issue #468: Apply target-type prioritisation boost for existing hidden targets.
    candidate.expected_creature_score_gain = apply_target_type_boost(
        candidate.expected_creature_score_gain,
        &candidate.to_neuron_uuid,
        neuron_type_map,
    );

    // Issue #891: Apply synapse prediction calibration to correct ~1,000× overestimation.
    // Applied after pessimism discount and type boosts to scale the final prediction
    // closer to observed actual gains, improving cross-type candidate ranking.
    candidate.expected_creature_score_gain = apply_prediction_calibration(
        candidate.expected_creature_score_gain,
        crate::analysis::constants::SYNAPSE_PREDICTION_CALIBRATION,
    );

    if verbose_enabled() && is_hidden {
        tracing::debug!(
            to_neuron_uuid = &candidate.to_neuron_uuid[..12.min(candidate.to_neuron_uuid.len())],
            impact = format_args!("{impact:.3}"),
            original_pct = format_args!("{:.4}", original * 100.0),
            discounted_pct = format_args!("{:.4}", candidate.expected_creature_score_gain * 100.0),
            "Synapse candidate impact discounting applied"
        );
    }
}

/// Apply impact-based discounting to a single harmful synapse candidate.
fn apply_impact_to_harmful(
    candidate: &mut CandidateSynapseJson,
    impact_scores: &HashMap<String, f32>,
    neuron_type_map: &HashMap<String, String>,
    order_map: &HashMap<String, usize>,
) {
    candidate.from_neuron_index = order_map.get(&candidate.from_neuron_uuid).copied();
    candidate.to_neuron_index = order_map.get(&candidate.to_neuron_uuid).copied();

    let is_hidden = neuron_type_map
        .get(&candidate.to_neuron_uuid)
        .is_none_or(|t| t != "output");

    let impact = if is_hidden {
        if let Some(&impact) = impact_scores.get(&candidate.to_neuron_uuid) {
            impact.clamp(0.0, 1.0)
        } else {
            0.1
        }
    } else {
        1.0
    };

    candidate.target_neuron_impact = impact;
    candidate.expected_creature_error_reduction *= impact;
    candidate.expected_creature_score_gain = candidate.expected_creature_error_reduction;

    // Issue #789: Apply synapse-specific pessimism discount based on improved sample ratio.
    candidate.expected_creature_score_gain = apply_synapse_pessimism_discount(
        candidate.expected_creature_score_gain,
        candidate.improved_count,
        candidate.total_count,
    );

    // Issue #891: Apply synapse prediction calibration to harmful candidates.
    candidate.expected_creature_score_gain = apply_prediction_calibration(
        candidate.expected_creature_score_gain,
        crate::analysis::constants::SYNAPSE_PREDICTION_CALIBRATION,
    );
}

/// Apply impact-based discounting and pessimism discount to a coordinated structural candidate.
///
/// Uses the last operation's target neuron UUID to determine impact, since multi-op
/// groups ultimately adjust the inputs of a target neuron.
///
/// Issue #790: Also applies `COORDINATED_PESSIMISM_DISCOUNT` — a flat multiplicative
/// discount to account for the 2.3% success rate of coordinated-structural candidates.
/// Unlike synapse/neuron candidates which have per-sample improved ratios, coordinated
/// candidates combine multiple operations whose predictions compound optimistically.
fn apply_impact_to_coordinated(
    candidate: &mut crate::CoordinatedStructuralCandidateJson,
    impact_scores: &HashMap<String, f32>,
    neuron_type_map: &HashMap<String, String>,
) {
    use crate::analysis::constants::COORDINATED_PESSIMISM_DISCOUNT;

    let target_uuid = candidate
        .operations
        .iter()
        .rev()
        .map(|op| match op {
            crate::CoordinatedStructuralOpJson::AddSynapse { to_neuron_uuid, .. } => {
                to_neuron_uuid.as_str()
            }
            crate::CoordinatedStructuralOpJson::RemoveSynapse { to_neuron_uuid, .. } => {
                to_neuron_uuid.as_str()
            }
            crate::CoordinatedStructuralOpJson::SetWeight { to_neuron_uuid, .. } => {
                to_neuron_uuid.as_str()
            }
            crate::CoordinatedStructuralOpJson::ChangeSquash { neuron_uuid, .. } => {
                neuron_uuid.as_str()
            }
            crate::CoordinatedStructuralOpJson::SetBias { neuron_uuid, .. } => neuron_uuid.as_str(),
            crate::CoordinatedStructuralOpJson::AddNeuron { neuron_uuid, .. } => {
                neuron_uuid.as_str()
            }
            crate::CoordinatedStructuralOpJson::RemoveNeuron { neuron_uuid } => {
                neuron_uuid.as_str()
            }
        })
        .next()
        .unwrap_or("");

    let is_hidden = neuron_type_map
        .get(target_uuid)
        .is_none_or(|t| t != "output");
    let impact = if is_hidden {
        impact_scores
            .get(target_uuid)
            .copied()
            .unwrap_or(0.1)
            .clamp(0.0, 1.0)
    } else {
        1.0
    };

    candidate.expected_creature_score_gain *= impact;

    // Issue #790: Apply coordinated-specific pessimism discount.
    // Coordinated-structural candidates have a 2.3% success rate with near-negligible
    // actual gains, indicating predictions are wildly over-estimated.
    candidate.expected_creature_score_gain *= COORDINATED_PESSIMISM_DISCOUNT;

    // Issue #891: Apply coordinated prediction calibration to correct ~10,000× overestimation.
    candidate.expected_creature_score_gain = apply_prediction_calibration(
        candidate.expected_creature_score_gain,
        crate::analysis::constants::COORDINATED_PREDICTION_CALIBRATION,
    );
}

/// Apply impact discounting, sorting, diversification, and truncation to all candidate sets.
///
/// This is the main post-processing entry point, called after the parallel analysis loop
/// completes and after structural pattern discovery (collapse hidden neurons).
pub(crate) fn apply_post_processing(
    helpful_results: &mut Vec<CandidateSynapseJson>,
    harmful_results: &mut Vec<CandidateSynapseJson>,
    coordinated_structural_results: &mut Vec<crate::CoordinatedStructuralCandidateJson>,
    input: &crate::AnalyzeSynapsesInput,
    cache: &RecordCache,
    order_map: &HashMap<String, usize>,
) -> PostProcessingMetrics {
    // Issue #128: Apply impact-based discounting and set creature-level metrics.
    let impact_scores = compute_impact_scores_for_discounting(&input.creature, cache);
    let neuron_type_map: HashMap<String, String> = input
        .creature
        .neurons
        .iter()
        .map(|n| (n.uuid.clone(), n.neuron_type.clone()))
        .collect();

    // Issue #730: Compute per-neuron and total error for creature-level calibration.
    let error_sq_map = compute_neuron_error_sq_map(input, cache);
    let total_error_sq: f32 = error_sq_map.values().sum();

    // Apply impact discounting to helpful synapse candidates
    for candidate in helpful_results.iter_mut() {
        let target_error_sq = error_sq_map
            .get(&candidate.to_neuron_uuid)
            .copied()
            .unwrap_or(0.0);
        apply_impact_to_helpful(
            candidate,
            &impact_scores,
            &neuron_type_map,
            order_map,
            target_error_sq,
            total_error_sq,
        );
    }

    // Apply impact discounting to harmful synapse candidates
    for candidate in harmful_results.iter_mut() {
        apply_impact_to_harmful(candidate, &impact_scores, &neuron_type_map, order_map);
    }

    // Apply impact discounting to coordinated candidates
    for candidate in coordinated_structural_results.iter_mut() {
        apply_impact_to_coordinated(candidate, &impact_scores, &neuron_type_map);
    }

    // Issue #557: Filter out candidates with non-positive expected_creature_score_gain.
    // After impact discounting, some candidates may have zero or negative gain and
    // would waste the evaluation budget if returned.
    helpful_results.retain(|c| c.expected_creature_score_gain > 0.0);
    harmful_results.retain(|c| c.expected_creature_score_gain > 0.0);
    coordinated_structural_results.retain(|c| c.expected_creature_score_gain > 0.0);

    // Sort all candidate lists by expected_creature_score_gain (highest first)
    helpful_results.sort_by(|a, b| {
        b.expected_creature_score_gain
            .total_cmp(&a.expected_creature_score_gain)
    });
    harmful_results.sort_by(|a, b| {
        b.expected_creature_score_gain
            .total_cmp(&a.expected_creature_score_gain)
    });
    coordinated_structural_results.sort_by(|a, b| {
        b.expected_creature_score_gain
            .total_cmp(&a.expected_creature_score_gain)
    });

    // Issue #513: Generate weight variants for helpful synapse candidates.
    // Each candidate gets conservative (0.5×), gentle-nudge (0.25×), and micro-nudge (0.1×)
    // weight variants. This maximises the pay-off from the expensive discovery process.
    *helpful_results = crate::analysis::utils::pair_synapse_candidates_with_weight_variants(
        std::mem::take(helpful_results),
        input.max_candidates,
    );

    // Issue #510: Generate conservative weight variants for coordinated-structural candidates.
    // AddSynapse weights are scaled to 0.2× (conservative), 0.1× (gentle nudge), 0.05× (micro-nudge).
    *coordinated_structural_results =
        crate::analysis::utils::pair_coordinated_structural_with_weight_variants(
            std::mem::take(coordinated_structural_results),
            input.max_candidates,
        );

    // Deadline coverage: diversify within the top-K for exploration diversity
    if input.analysis_deadline_ms.is_some() {
        use crate::analysis::constants::DIVERSIFY_TOP_K;
        shuffle_within_top_k(
            helpful_results.as_mut_slice(),
            input.random_seed,
            "synapse:helpful_candidates:top_k",
            DIVERSIFY_TOP_K,
        );
        shuffle_within_top_k(
            harmful_results.as_mut_slice(),
            input.random_seed,
            "synapse:harmful_candidates:top_k",
            DIVERSIFY_TOP_K,
        );
        shuffle_within_top_k(
            coordinated_structural_results.as_mut_slice(),
            input.random_seed,
            "synapse:coordinated_structural_candidates:top_k",
            DIVERSIFY_TOP_K,
        );
    }

    // Track candidates_found before truncation
    let candidates_found =
        helpful_results.len() + harmful_results.len() + coordinated_structural_results.len();

    if let Some(limit) = input.max_candidates {
        let (h1, h2, c) = truncate_combined_synapse_candidate_sets(
            std::mem::take(helpful_results),
            std::mem::take(harmful_results),
            std::mem::take(coordinated_structural_results),
            limit,
            input.analysis_deadline_ms.is_some(),
        );
        *helpful_results = h1;
        *harmful_results = h2;
        *coordinated_structural_results = c;
    }

    let candidates_returned =
        helpful_results.len() + harmful_results.len() + coordinated_structural_results.len();

    PostProcessingMetrics {
        candidates_found,
        candidates_returned,
    }
}

/// Metrics returned from post-processing for inclusion in analysis metadata.
pub(crate) struct PostProcessingMetrics {
    pub candidates_found: usize,
    pub candidates_returned: usize,
}

/// Parameters for building analysis metadata.
pub(crate) struct MetadataParams<'a> {
    pub target_value_seen: bool,
    pub saturation_aware_used: bool,
    pub candidates_found: usize,
    pub candidates_returned: usize,
    pub analysis_timed_out: bool,
    pub completed_focus_neurons: usize,
    pub total_focus_neurons: usize,
    pub saw_any_input: bool,
    pub input_min: usize,
    pub input_max: usize,
    pub error_values: &'a [f32],
    pub timing_collector: &'a crate::analysis::shared::TimingCollector,
}

/// Build the analysis metadata from collected atomic flags and timing data.
pub(crate) fn build_metadata(
    params: &MetadataParams<'_>,
) -> crate::analysis::shared::SynapseAnalysisMetadata {
    use crate::analysis::gpu::GpuAnalyzer;

    let error_distribution =
        crate::analysis::scoring::error_distribution::ErrorDistribution::from_errors(
            params.error_values,
        );

    crate::analysis::shared::SynapseAnalysisMetadata {
        target_value_available: params.target_value_seen,
        saturation_aware_simulation_used: params.saturation_aware_used,
        candidates_found: params.candidates_found,
        candidates_returned: params.candidates_returned,
        timed_out: params.analysis_timed_out,
        completed_focus_neurons: params.completed_focus_neurons,
        total_focus_neurons: params.total_focus_neurons,
        input_index_min_seen_with_records: if params.saw_any_input {
            Some(params.input_min)
        } else {
            None
        },
        input_index_max_seen_with_records: if params.saw_any_input {
            Some(params.input_max)
        } else {
            None
        },
        timing: params.timing_collector.finalize(),
        gpu_info: GpuAnalyzer::get_adapter_info(),
        error_distribution,
        discovery_module_stats: Vec::new(),
    }
}
