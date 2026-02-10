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

use crate::intern::NeuronIndex;
use crate::types::DiscoverRecord;
use crate::{AnalyzeSynapsesInput, CandidateNeuronJson, CandidateSynapseJson, SynapseJson};
use anyhow::{Result, anyhow};

// Import shared types from the new module structure
use crate::analysis::shared::{AnalyzeSynapsesResult, TimingScope};

// Import activation functions from the dedicated activation module (Issue #266, #238)
use crate::analysis::activation::{
    ActivationCandidateSpec, TargetSimulationMode, absolute_activation, activation_name_to_gpu_id,
    arctan_activation, bent_identity_activation, bipolar_activation, clipped_activation,
    elu_activation, gelu_activation, get_target_simulation_fn, get_target_simulation_mode,
    hard_tanh_activation, has_sufficient_output_variance, identity_activation,
    is_saturating_target, logistic_activation, mish_activation, relu6_activation,
    softplus_activation, softsign_activation, tanh_activation,
};

// Import deadline handling and logging utilities from dedicated module (Issue #268)
use crate::analysis::utils::{
    OrderedNeuron, build_deadline, deadline_passed, log_analysis_start, log_analysis_timeout,
    order_eligible_sources, order_focus_targets, parse_input_index, shuffle_within_top_k,
    verbose_enabled,
};

// Import sample data structures from dedicated module (Issue #269)
use crate::analysis::samples::{
    EPSILON, HelpfulSample, NeuronStats, compute_source_std_dev, get_constant_source_threshold,
};

// Import confidence interval calculations (Issue #194)
use crate::analysis::confidence::compute_confidence_metrics;

// Import weight calculation functions from dedicated module (Issue #270)
use crate::analysis::weights::{
    MAX_OUTGOING_WEIGHT, calculate_optimal_bias, calculate_optimal_identity_outgoing_and_bias,
    calculate_optimal_outgoing_weight, clamp_weight_update_delta,
    coordinated_structural_activation_delta,
};

// Import diagnostics and rejection tracking from dedicated module (Issue #271)
use crate::analysis::diagnostics::{
    TargetDiagnostics, TargetMap, ThresholdContext, compute_impact_scores_for_discounting,
    require_unique_focus,
};

// Import epistatic pair detection module (Issue #202), synergistic discovery (Issue #189),
// and interference filtering (Issue #415)
use crate::analysis::epistatic::{
    SourceContribution, build_source_contribution, detect_epistatic_pairs,
    detect_synergistic_candidates, epistatic_pairs_to_coordinated_candidates,
    filter_interfering_epistatic_pairs, filter_interfering_synergistic_candidates,
    synergistic_to_coordinated_candidates,
};

// Import redundant path pruning module (Issue #164)
use crate::analysis::redundant_path::{
    ExistingPathContribution, detect_redundant_paths, redundant_paths_to_coordinated_candidates,
};

// Import GPU infrastructure from dedicated modules (Issue #272, #273, #274)
use crate::analysis::gpu::{GpuAnalyzer, GpuEvaluator, GpuWorkQueue};

// Import RecordCache from cache module (Issue #185)
use super::cache::RecordCache;

use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

// =============================================================================
// Constants
// =============================================================================

// MIN_NEURON_SAMPLE_COUNT moved to constants.rs (Issue #424)
use super::constants::MIN_NEURON_SAMPLE_COUNT;

// INPUT_SOURCE_BOOST for source-type prioritisation (Issue #467)
use super::constants::INPUT_SOURCE_BOOST;

// EXISTING_HIDDEN_TARGET_BOOST for target-type prioritisation (Issue #468)
use super::constants::EXISTING_HIDDEN_TARGET_BOOST;

// Note: MIN_NEURON_OUTPUT_STD_DEV has been moved to the activation module
// as part of Issue #238. It is used by has_sufficient_output_variance.

// =============================================================================
// Source-Type Prioritisation (Issue #467)
// =============================================================================

/// Applies source-type boost to a candidate's expected score gain.
///
/// Input neurons as synapse sources have a 36.2% success rate compared to
/// 2.8–3.3% for hidden neurons (GRQ-sampler data). This function applies
/// [`INPUT_SOURCE_BOOST`] as a multiplier when the source neuron is an input
/// neuron (UUID matches `input-N` pattern).
///
/// Hidden and output neurons receive no boost (multiplier = 1.0).
pub fn apply_source_type_boost(gain: f32, source_uuid: &str) -> f32 {
    if parse_input_index(source_uuid).is_some() {
        gain * INPUT_SOURCE_BOOST as f32
    } else {
        gain
    }
}

// =============================================================================
// Target-Type Prioritisation (Issue #468)
// =============================================================================

/// Applies target-type boost to a candidate's expected score gain.
///
/// Existing hidden neurons as targets have a 31.4% success rate compared to
/// 5.3–5.4% for output or discovery-hidden neurons (GRQ-sampler data). This
/// function applies [`EXISTING_HIDDEN_TARGET_BOOST`] as a multiplier when the
/// target neuron is an existing hidden neuron.
///
/// Output, input, constant, and unknown neurons receive no boost (multiplier = 1.0).
pub fn apply_target_type_boost(
    gain: f32,
    target_uuid: &str,
    neuron_type_map: &HashMap<String, String>,
) -> f32 {
    if neuron_type_map
        .get(target_uuid)
        .map(|t| t == "hidden")
        .unwrap_or(false)
    {
        gain * EXISTING_HIDDEN_TARGET_BOOST as f32
    } else {
        gain
    }
}

// =============================================================================
// Sample Locality Grouping (Issue #221)
// =============================================================================

/// Minimum number of sources in a group to make shared sample building worthwhile.
/// Below this threshold, the overhead of grouping exceeds the benefit.
pub(crate) const MIN_GROUP_SIZE_FOR_LOCALITY: usize = 3;

/// Minimum overlap fraction required to group sources together.
/// Sources are grouped if they share at least this fraction of their obs_indices.
const MIN_LOCALITY_OVERLAP: f32 = 0.8;

/// Represents a group of sources with similar obs_index coverage.
/// Sources in the same group can share sample building overhead.
pub(crate) struct SampleLocalityGroup<'a> {
    /// The source neurons in this group
    pub(crate) sources: Vec<(&'a OrderedNeuron, Arc<Vec<DiscoverRecord>>)>,
    /// Representative obs_indices for this group (from the first source)
    pub(crate) _representative_indices: HashSet<u32>,
}

/// Extract obs_indices from source records.
pub(crate) fn extract_obs_indices(records: &[DiscoverRecord]) -> HashSet<u32> {
    records
        .iter()
        .filter(|r| r.activation.is_finite())
        .map(|r| r.obs_index)
        .collect()
}

/// Compute the overlap fraction between two sets of obs_indices.
/// Returns a value in [0, 1] representing what fraction of the smaller set
/// is contained in the larger set.
pub(crate) fn compute_obs_index_overlap(a: &HashSet<u32>, b: &HashSet<u32>) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let intersection_size = a.intersection(b).count();
    let min_size = a.len().min(b.len());
    intersection_size as f32 / min_size as f32
}

/// Group sources by sample locality for efficient batch processing.
///
/// Sources with high obs_index overlap (≥80%) are grouped together so that
/// sample building can be done in a single pass through the target data
/// rather than N separate passes.
///
/// # Issue #221: Sample Locality for Correlated Source Neurons
///
/// Expected benefits for typical creatures:
/// - 100 sources, same obs_indices: 1 group (100x reduction in target lookups)
/// - 100 sources, 80% overlap: ~5 groups (20x reduction)
/// - 100 sources, no overlap: 100 groups (no change)
pub(crate) fn group_sources_by_locality<'a>(
    sources: &[(&'a OrderedNeuron, Arc<Vec<DiscoverRecord>>)],
) -> Vec<SampleLocalityGroup<'a>> {
    if sources.len() < MIN_GROUP_SIZE_FOR_LOCALITY {
        // Not enough sources to benefit from grouping
        return sources
            .iter()
            .map(|(neuron, records)| {
                let indices = extract_obs_indices(records);
                SampleLocalityGroup {
                    sources: vec![(neuron, Arc::clone(records))],
                    _representative_indices: indices,
                }
            })
            .collect();
    }

    // Extract obs_indices for each source (done once, reused for grouping)
    let source_indices: Vec<HashSet<u32>> = sources
        .iter()
        .map(|(_, records)| extract_obs_indices(records))
        .collect();

    let mut groups: Vec<SampleLocalityGroup<'a>> = Vec::new();
    let mut assigned: Vec<bool> = vec![false; sources.len()];

    for i in 0..sources.len() {
        if assigned[i] {
            continue;
        }

        let my_indices = &source_indices[i];
        if my_indices.is_empty() {
            // Source has no valid indices - put it in its own group
            assigned[i] = true;
            groups.push(SampleLocalityGroup {
                sources: vec![(sources[i].0, Arc::clone(&sources[i].1))],
                _representative_indices: my_indices.clone(),
            });
            continue;
        }

        // Start a new group with this source
        let mut group_sources = vec![(sources[i].0, Arc::clone(&sources[i].1))];
        assigned[i] = true;

        // Find all other sources with high overlap
        for j in (i + 1)..sources.len() {
            if assigned[j] {
                continue;
            }

            let other_indices = &source_indices[j];
            let overlap = compute_obs_index_overlap(my_indices, other_indices);

            if overlap >= MIN_LOCALITY_OVERLAP {
                group_sources.push((sources[j].0, Arc::clone(&sources[j].1)));
                assigned[j] = true;
            }
        }

        groups.push(SampleLocalityGroup {
            sources: group_sources,
            _representative_indices: my_indices.clone(),
        });
    }

    groups
}

/// Build samples for all sources in a locality group efficiently.
///
/// This uses `TargetMap::build_samples_for_group` to build samples for all
/// sources in a single pass through the target data.
pub(crate) fn build_samples_for_locality_group(
    group: &SampleLocalityGroup<'_>,
    target_map: &TargetMap,
) -> Vec<(String, Vec<HelpfulSample>, usize)> {
    if group.sources.len() == 1 {
        // Single source - use standard path (no overhead)
        let (source, records) = &group.sources[0];
        let samples = target_map.build_samples_from(records);
        return vec![(source.uuid.clone(), samples, records.len())];
    }

    // Multiple sources - use batched sample building
    let source_records: Vec<(&str, &[DiscoverRecord])> = group
        .sources
        .iter()
        .map(|(source, records)| (source.uuid.as_str(), records.as_slice()))
        .collect();

    let samples_batch = target_map.build_samples_for_group(&source_records);

    group
        .sources
        .iter()
        .zip(samples_batch)
        .map(|((source, records), samples)| (source.uuid.clone(), samples, records.len()))
        .collect()
}

// =============================================================================
// Synapse Improvement Calculation
// =============================================================================

// Note: Target simulation functions (TargetSimulationMode, get_target_activation_fn,
// get_target_simulation_fn, get_target_simulation_mode, can_use_hard_tanh) have been
// moved to the activation module as part of Issue #238.

/// Computes the sign of a weight as an i8 for use in the candidate key.
/// Returns 1 for positive weights, -1 for negative, and 0 for zero (though this
/// shouldn't happen in practice).
pub(crate) fn weight_sign(weight: f32) -> i8 {
    if weight > 0.0 {
        1
    } else if weight < 0.0 {
        -1
    } else {
        0
    }
}

/// Returns true if this add-neuron candidate is "extreme" enough to warrant a conservative pair.
///
/// We intentionally base this on incoming weight and bias (not outgoing), because outgoing
/// weight is already clamped and ReLU candidates commonly use outgoing=0.1 by design.
pub(crate) fn upsert_candidate(
    map: &mut HashMap<(String, String, String, i8, i8), CandidateNeuronJson>,
    candidate: CandidateNeuronJson,
) {
    use std::collections::hash_map::Entry;

    // Key includes signs of BOTH incoming_weight AND outgoing_weight so that:
    // 1. Different ReLU orientations (incoming_weight ±1) are kept separately
    // 2. Split-error complementary pairs (same incoming_weight, opposite outgoing_weight)
    //    are also kept separately - one pushes output UP, one pushes DOWN
    let key = (
        candidate.source_neuron_uuid.clone(),
        candidate.target_neuron_uuid.clone(),
        candidate.squash.clone(),
        weight_sign(candidate.incoming_weight),
        weight_sign(candidate.outgoing_weight),
    );

    match map.entry(key) {
        Entry::Occupied(mut entry) => {
            // Issue #128: Compare by expected creature score gain
            if candidate.expected_creature_score_gain > entry.get().expected_creature_score_gain {
                entry.insert(candidate);
            }
        }
        Entry::Vacant(entry) => {
            entry.insert(candidate);
        }
    }
}

/// Result from ReLU evaluation (split by target error sign)
pub(crate) struct SplitReluResult {
    /// Candidate for samples with positive error (output should be higher)
    pub(crate) positive_error_candidate: Option<CandidateNeuronJson>,
    /// Candidate for samples with negative error (output should be lower)
    pub(crate) negative_error_candidate: Option<CandidateNeuronJson>,
}

// Note: has_sufficient_output_variance has been moved to the activation module
// as part of Issue #238. It is now imported from crate::analysis::activation.

/// Combined computation of improvement and count for ReLU candidates.
/// Single pass over samples for better cache efficiency.
///
/// When `target_activation_fn` is Some, simulates the target neuron's actual activation
/// function for more accurate improvement estimates. Otherwise falls back to linear approximation.
///
/// IMPORTANT: The `bias` parameter is critical for accurate predictions. It shifts the ReLU
/// activation threshold, affecting which samples produce non-zero output.
///
/// CRITICAL DOMAIN FIX (v0.1.120): When using target_activation_fn simulation,
/// both baseline and new error must be computed in ACTIVATION domain.
///
/// Returns (improvement_percentage, improved_count, total_count)
pub(crate) fn compute_relu_improvement_and_count(
    samples: &[HelpfulSample],
    incoming_weight: f32,
    outgoing_weight: f32,
    bias: f32,
    total_baseline_error_sq: f32,
    target_activation_fn: Option<fn(f32) -> f32>,
) -> (f32, u32, u32) {
    if total_baseline_error_sq <= EPSILON || samples.is_empty() {
        return (0.0, 0, samples.len() as u32);
    }

    let mut baseline_error_sq_sum = 0.0f32; // ACTIVATION domain when simulating
    let mut new_error_sq_sum = 0.0f32;
    let mut improved_count = 0u32;
    let mut worsened_count = 0u32;

    for sample in samples.iter() {
        let pre_activation = incoming_weight * sample.activation + bias;
        let relu_output = pre_activation.max(0.0);
        let contribution = outgoing_weight * relu_output;

        let (baseline_error, new_error) = if let Some(target_fn) = target_activation_fn {
            // CRITICAL: Use ACTIVATION domain for BOTH baseline and new error.
            let target_value = unsafe { sample.target_value.unwrap_unchecked() };
            let target_activation = unsafe { sample.target_activation.unwrap_unchecked() };
            let desired_value = target_value + sample.avg_error;
            let expected = target_fn(desired_value);

            // Baseline error in ACTIVATION domain
            let baseline_err = expected - target_activation;

            // New error in ACTIVATION domain
            let new_input = target_value + contribution;
            let new_err = expected - target_fn(new_input);

            (baseline_err, new_err)
        } else {
            // Linear approximation: both errors in VALUE domain
            let baseline_err = sample.avg_error;
            let new_err = sample.avg_error - contribution;

            (baseline_err, new_err)
        };

        if baseline_error.is_finite() {
            baseline_error_sq_sum += baseline_error * baseline_error;
        }
        if new_error.is_finite() {
            new_error_sq_sum += new_error * new_error;
        }

        // Sample is improved if |new_error| < |baseline_error| (consistent domain)
        if new_error.abs() + EPSILON < baseline_error.abs() {
            improved_count += 1;
        } else if new_error.abs() > baseline_error.abs() + EPSILON {
            worsened_count += 1;
        }
    }

    // Use computed ACTIVATION domain baseline when simulating, else use passed VALUE domain
    let effective_baseline = if target_activation_fn.is_some() {
        baseline_error_sq_sum
    } else {
        total_baseline_error_sq
    };

    let improvement = if effective_baseline > EPSILON {
        (effective_baseline - new_error_sq_sum) / effective_baseline
    } else {
        0.0
    };
    let improvement = if improvement.is_finite() {
        improvement
    } else {
        0.0
    };

    let total_count = samples.len() as u32;
    let _ = worsened_count; // Kept for potential future use
    (improvement, improved_count, total_count)
}

/// Combined computation of improvement and count for activation candidates.
/// Single pass over samples for better cache efficiency.
///
/// When `target_activation_fn` is Some, simulates the target neuron's actual activation
/// function for more accurate improvement estimates. Otherwise falls back to linear approximation.
///
/// CRITICAL DOMAIN FIX (v0.1.120): When using target_activation_fn simulation,
/// both baseline and new error must be computed in ACTIVATION domain.
///
/// Returns (improvement_percentage, improved_count, total_count)
pub(crate) fn compute_activation_improvement_and_count(
    samples: &[HelpfulSample],
    incoming_weight: f32,
    outgoing_weight: f32,
    bias: f32,
    activation_fn: fn(f32) -> f32,
    total_baseline_error_sq: f32,
    target_activation_fn: Option<fn(f32) -> f32>,
) -> (f32, u32, u32) {
    if total_baseline_error_sq <= EPSILON || samples.is_empty() {
        return (0.0, 0, samples.len() as u32);
    }

    let mut baseline_error_sq_sum = 0.0f32; // ACTIVATION domain when simulating
    let mut new_error_sq_sum = 0.0f32;
    let mut improved_count = 0u32;
    let total_count = samples.len() as u32;

    for sample in samples {
        let pre_activation = incoming_weight * sample.activation + bias;
        let neuron_output = activation_fn(pre_activation);
        let contribution = outgoing_weight * neuron_output;

        let (baseline_error, new_error) = if let Some(target_fn) = target_activation_fn {
            // CRITICAL: Use ACTIVATION domain for BOTH baseline and new error.
            let target_value = unsafe { sample.target_value.unwrap_unchecked() };
            let target_activation = unsafe { sample.target_activation.unwrap_unchecked() };
            let desired_value = target_value + sample.avg_error;
            let expected = target_fn(desired_value);

            // Baseline error in ACTIVATION domain
            let baseline_err = expected - target_activation;

            // New error in ACTIVATION domain
            let new_input = target_value + contribution;
            let new_err = expected - target_fn(new_input);

            (baseline_err, new_err)
        } else {
            // Linear approximation: both errors in VALUE domain
            (sample.avg_error, sample.avg_error - contribution)
        };

        if baseline_error.is_finite() {
            baseline_error_sq_sum += baseline_error * baseline_error;
        }
        if new_error.is_finite() {
            new_error_sq_sum += new_error * new_error;
        }

        // Sample is improved if |new_error| < |baseline_error| (consistent domain)
        if new_error.abs() + EPSILON < baseline_error.abs() {
            improved_count += 1;
        }
    }

    // Use computed ACTIVATION domain baseline when simulating, else use passed VALUE domain
    let effective_baseline = if target_activation_fn.is_some() {
        baseline_error_sq_sum
    } else {
        total_baseline_error_sq
    };

    let improvement = if effective_baseline > EPSILON {
        (effective_baseline - new_error_sq_sum) / effective_baseline
    } else {
        0.0
    };
    let improvement = if improvement.is_finite() {
        improvement
    } else {
        0.0
    };

    (improvement, improved_count, total_count)
}

/// Wrapper for tests - computes improvement only.
/// NOTE: For ReLU candidates, bias affects which samples activate. Pass the actual bias
/// that will be used with the new neuron for accurate predictions.
#[cfg(test)]
pub(crate) fn compute_net_improvement_with_squash(
    samples: &[HelpfulSample],
    incoming_weight: f32,
    outgoing_weight: f32,
    bias: f32,
    total_baseline_error_sq: f32,
    target_squash: Option<&str>,
) -> f32 {
    let target_activation_fn = get_target_simulation_fn(samples, target_squash);
    let (improvement, _, _) = compute_relu_improvement_and_count(
        samples,
        incoming_weight,
        outgoing_weight,
        bias,
        total_baseline_error_sq,
        target_activation_fn,
    );
    improvement
}

/// Compute synapse improvement accounting for target neuron's activation function.
///
/// For direct synapse connections (source → target), the contribution is `weight × source_activation`.
/// This function simulates the target's activation function to predict accurate improvement,
/// avoiding overprediction near saturation for HARD_TANH, TANH, LOGISTIC, etc.
///
/// Returns improvement_percentage only. Used in tests; production uses compute_synapse_improvement_and_count.
#[cfg(test)]
pub(crate) fn compute_synapse_improvement_with_target_squash(
    samples: &[HelpfulSample],
    weight: f32,
    total_baseline_error_sq: f32,
    target_squash: Option<&str>,
) -> f32 {
    if total_baseline_error_sq <= EPSILON || samples.is_empty() {
        return 0.0;
    }

    let target_sim = get_target_simulation_mode(samples, target_squash);

    let mut baseline_error_sq_sum = 0.0f32; // ACTIVATION domain when simulating
    let mut new_error_sq_sum = 0.0f32;

    for sample in samples {
        // Direct synapse contribution: weight × source_activation
        let contribution = weight * sample.activation;

        let new_error = match target_sim {
            TargetSimulationMode::None => {
                // Linear approximation - assumes contribution directly reduces error (VALUE domain).
                sample.avg_error - contribution
            }
            TargetSimulationMode::Full(target_fn) => {
                // Saturation-aware model (ACTIVATION domain).
                let target_value = sample.target_value.unwrap();
                let target_activation = sample.target_activation.unwrap();
                let desired_value = target_value + sample.avg_error;
                let expected = target_fn(desired_value);

                let baseline_err = expected - target_activation;
                if baseline_err.is_finite() {
                    baseline_error_sq_sum += baseline_err * baseline_err;
                }

                let new_input = target_value + contribution;
                expected - target_fn(new_input)
            }
            TargetSimulationMode::ApproximateValueFromActivation(target_fn) => {
                // Saturation-aware model (ACTIVATION domain), approximating missing target_value.
                let target_activation = sample.target_activation.unwrap();
                let target_value = sample.target_value.unwrap_or(target_activation);
                let desired_value = target_value + sample.avg_error;
                let expected = target_fn(desired_value);

                let baseline_err = expected - target_activation;
                if baseline_err.is_finite() {
                    baseline_error_sq_sum += baseline_err * baseline_err;
                }

                let new_input = target_value + contribution;
                expected - target_fn(new_input)
            }
        };

        if new_error.is_finite() {
            new_error_sq_sum += new_error * new_error;
        }
    }

    let effective_baseline = match target_sim {
        TargetSimulationMode::None => total_baseline_error_sq,
        _ => baseline_error_sq_sum,
    };
    if effective_baseline <= EPSILON {
        return 0.0;
    }

    let improvement = (effective_baseline - new_error_sq_sum) / effective_baseline;
    if improvement.is_finite() {
        improvement
    } else {
        0.0
    }
}

/// Compute improvement, improved count, and worsened count for synapse candidates.
/// All counts use the same saturation-aware methodology for consistency.
///
/// CRITICAL DOMAIN FIX (v0.1.120): When using target_activation_fn simulation,
/// both baseline and new error must be computed in ACTIVATION domain. The passed-in
/// total_baseline_error_sq is in VALUE domain, so we compute our own ACTIVATION
/// domain baseline when simulating.
///
/// Returns (improvement_percentage, improved_count, worsened_count, total_count)
pub(crate) fn compute_synapse_improvement_and_count(
    samples: &[HelpfulSample],
    weight: f32,
    total_baseline_error_sq: f32,
    target_squash: Option<&str>,
) -> (f32, u32, u32, u32) {
    if total_baseline_error_sq <= EPSILON || samples.is_empty() {
        return (0.0, 0, 0, samples.len() as u32);
    }

    let target_sim = get_target_simulation_mode(samples, target_squash);

    let mut baseline_error_sq_sum = 0.0f32; // ACTIVATION domain when simulating
    let mut new_error_sq_sum = 0.0f32;
    let mut improved_count = 0u32;
    let mut worsened_count = 0u32;
    let total_count = samples.len() as u32;

    for sample in samples {
        let contribution = weight * sample.activation;

        let (baseline_error, new_error) = match target_sim {
            TargetSimulationMode::None => {
                // Linear approximation: both errors in VALUE domain.
                (sample.avg_error, sample.avg_error - contribution)
            }
            TargetSimulationMode::Full(target_fn) => {
                // CRITICAL: Use ACTIVATION domain for BOTH baseline and new error.
                // avg_error is in VALUE domain, but MSE is measured in ACTIVATION domain.
                let target_value = sample.target_value.unwrap();
                let target_activation = sample.target_activation.unwrap();
                let desired_value = target_value + sample.avg_error;
                let expected = target_fn(desired_value);

                let baseline_err = expected - target_activation;
                let new_input = target_value + contribution;
                let new_err = expected - target_fn(new_input);

                (baseline_err, new_err)
            }
            TargetSimulationMode::ApproximateValueFromActivation(target_fn) => {
                // As above, but approximate missing target_value from the observed activation.
                let target_activation = sample.target_activation.unwrap();
                let target_value = sample.target_value.unwrap_or(target_activation);
                let desired_value = target_value + sample.avg_error;
                let expected = target_fn(desired_value);

                let baseline_err = expected - target_activation;
                let new_input = target_value + contribution;
                let new_err = expected - target_fn(new_input);

                (baseline_err, new_err)
            }
        };

        if baseline_error.is_finite() {
            baseline_error_sq_sum += baseline_error * baseline_error;
        }
        if new_error.is_finite() {
            new_error_sq_sum += new_error * new_error;
        }

        // Count improved/worsened samples using consistent domain comparison
        if new_error.abs() + EPSILON < baseline_error.abs() {
            improved_count += 1;
        } else if new_error.abs() > baseline_error.abs() + EPSILON {
            worsened_count += 1;
        }
    }

    // Use computed ACTIVATION domain baseline when simulating, else use passed VALUE domain.
    let effective_baseline = match target_sim {
        TargetSimulationMode::None => total_baseline_error_sq,
        _ => baseline_error_sq_sum,
    };

    let improvement = if effective_baseline > EPSILON {
        (effective_baseline - new_error_sq_sum) / effective_baseline
    } else {
        0.0
    };
    let improvement = if improvement.is_finite() {
        improvement
    } else {
        0.0
    };

    (improvement, improved_count, worsened_count, total_count)
}

/// Wrapper for tests - counts improved samples only.
#[cfg(test)]
pub(crate) fn count_improved_samples(
    samples: &[HelpfulSample],
    incoming_weight: f32,
    outgoing_weight: f32,
    bias: f32,
    target_squash: Option<&str>,
) -> (u32, u32) {
    let total_baseline_error_sq: f32 = samples.iter().map(|s| s.avg_error * s.avg_error).sum();
    let target_activation_fn = get_target_simulation_fn(samples, target_squash);
    let (_, improved, total) = compute_relu_improvement_and_count(
        samples,
        incoming_weight,
        outgoing_weight,
        bias,
        total_baseline_error_sq,
        target_activation_fn,
    );
    (improved, total)
}

// =============================================================================
// ReLU Candidate Evaluation
// =============================================================================

/// Evaluate ReLU candidates by splitting samples based on TARGET neuron's error sign.
///
/// This is the PRIMARY approach for ReLU evaluation. It finds candidates for both directions:
/// - **Positive-error samples** (output should be HIGHER): compute weight that pushes UP
/// - **Negative-error samples** (output should be LOWER): compute weight that pushes DOWN
///
/// For each direction:
/// 1. Compute optimal weight from the error subset
/// 2. Evaluate NET improvement across ALL samples
/// 3. Return candidate if it passes threshold
///
/// This is the correct approach for directional activations like ReLU because:
/// - ReLU can only push output in ONE direction (based on outgoing weight sign)
/// - Averaging over all samples cancels out when errors are split ~50/50
/// - We evaluate source activations as-is (we don't care how they were calculated)
pub(crate) fn evaluate_relu_candidates_split<G: GpuEvaluator>(
    gpu: &G,
    source_uuid: &str,
    target_uuid: &str,
    samples: &[HelpfulSample],
    threshold: f32,
    target_squash: Option<&str>,
) -> Result<SplitReluResult> {
    // Split samples by error sign
    let positive_error_samples: Vec<HelpfulSample> = samples
        .iter()
        .filter(|s| s.avg_error > EPSILON)
        .copied()
        .collect();

    let negative_error_samples: Vec<HelpfulSample> = samples
        .iter()
        .filter(|s| s.avg_error < -EPSILON)
        .copied()
        .collect();

    let mut result = SplitReluResult {
        positive_error_candidate: None,
        negative_error_candidate: None,
    };

    // Compute total baseline error across ALL samples (for net improvement calculation)
    let total_baseline_error_sq: f32 = samples.iter().map(|s| s.avg_error * s.avg_error).sum();

    if total_baseline_error_sq <= EPSILON {
        return Ok(result);
    }

    // Get target activation function for accurate simulation (ReLU, HARD_TANH, etc.)
    let target_activation_fn = get_target_simulation_fn(samples, target_squash);

    // For positive errors (output should be higher), compute optimal weight from subset
    // then evaluate the NET effect across ALL samples.
    // We evaluate BOTH ReLU orientations (positive and negative incoming weight) and pick best.
    if positive_error_samples.len() >= MIN_NEURON_SAMPLE_COUNT {
        let (positive_stats, negative_stats, pos_baseline_error_sq) =
            gpu.evaluate_relu(&positive_error_samples, threshold)?;

        // Try both orientations and pick the best
        let orientations = [positive_stats, negative_stats];
        let mut best_candidate: Option<CandidateNeuronJson> = None;
        let mut best_improvement = threshold;

        for stats in orientations {
            if let Some(mut candidate) = stats.evaluate(
                source_uuid,
                target_uuid,
                threshold,
                pos_baseline_error_sq,
                &positive_error_samples,
            ) {
                // Compute net improvement across ALL samples (single pass)
                // CRITICAL: Include candidate.bias for accurate prediction
                let (net_improvement, improved, total) = compute_relu_improvement_and_count(
                    samples,
                    candidate.incoming_weight,
                    candidate.outgoing_weight,
                    candidate.bias,
                    total_baseline_error_sq,
                    target_activation_fn,
                );
                candidate.improved_count = improved;
                candidate.total_count = total;

                // Issue #128: Update creature-level metrics
                if net_improvement > best_improvement {
                    candidate.expected_creature_error_reduction = net_improvement;
                    candidate.expected_creature_score_gain = net_improvement;
                    best_improvement = net_improvement;
                    best_candidate = Some(candidate);
                }
            }
        }
        result.positive_error_candidate = best_candidate;
    }

    // For negative errors (output should be lower).
    // We evaluate BOTH ReLU orientations (positive and negative incoming weight) and pick best.
    if negative_error_samples.len() >= MIN_NEURON_SAMPLE_COUNT {
        let (positive_stats, negative_stats, neg_baseline_error_sq) =
            gpu.evaluate_relu(&negative_error_samples, threshold)?;

        // Try both orientations and pick the best
        let orientations = [positive_stats, negative_stats];
        let mut best_candidate: Option<CandidateNeuronJson> = None;
        let mut best_improvement = threshold;

        for stats in orientations {
            if let Some(mut candidate) = stats.evaluate(
                source_uuid,
                target_uuid,
                threshold,
                neg_baseline_error_sq,
                &negative_error_samples,
            ) {
                // Compute net improvement across ALL samples (single pass)
                // CRITICAL: Include candidate.bias for accurate prediction
                let (net_improvement, improved, total) = compute_relu_improvement_and_count(
                    samples,
                    candidate.incoming_weight,
                    candidate.outgoing_weight,
                    candidate.bias,
                    total_baseline_error_sq,
                    target_activation_fn,
                );
                candidate.improved_count = improved;
                candidate.total_count = total;

                // Issue #128: Update creature-level metrics
                if net_improvement > best_improvement {
                    candidate.expected_creature_error_reduction = net_improvement;
                    candidate.expected_creature_score_gain = net_improvement;
                    best_improvement = net_improvement;
                    best_candidate = Some(candidate);
                }
            }
        }
        result.negative_error_candidate = best_candidate;
    }

    Ok(result)
}

// =============================================================================
// Activation Candidate Evaluation
// =============================================================================

/// Helper for split-error evaluation: compute optimal weight from subset, evaluate on all samples.
///
/// This is the core of the split-error fix for non-ReLU activations. By computing
/// the optimal weight from a specific error subset (positive or negative), we get
/// a weight that's tuned to help that subset. We then evaluate the NET improvement
/// across ALL samples to ensure the candidate doesn't hurt the other subset more
/// than it helps the target subset.
#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_activation_for_subset<G: GpuEvaluator>(
    gpu: &G,
    source_uuid: &str,
    target_uuid: &str,
    subset_samples: &[HelpfulSample], // Used to compute optimal weight
    all_samples: &[HelpfulSample],    // Used to compute net improvement
    spec: &ActivationCandidateSpec,
    target_squash: Option<&str>,
    total_baseline_error_sq: f32,
    target_activation_fn: Option<fn(f32) -> f32>,
) -> Result<Option<CandidateNeuronJson>> {
    if subset_samples.len() < MIN_NEURON_SAMPLE_COUNT {
        return Ok(None);
    }

    let activation_type = activation_name_to_gpu_id(spec.name);

    let mut best_candidate: Option<CandidateNeuronJson> = None;
    // v0.1.136: Fixed threshold bug - use 0.0 instead of threshold.
    // The calling code in evaluate_activation_candidate handles threshold vs fallback
    // logic. If we initialise to threshold here, candidates with 0 < improvement <= threshold
    // are silently dropped, breaking the fallback mechanism for split-error evaluation.
    let mut best_net_improvement = 0.0;

    for &orientation in spec.orientations {
        for &scale in spec.scales {
            let incoming_weight = orientation * scale;

            // Compute optimal weight from SUBSET samples using GPU
            let (sum_activation_sq, sum_error_activation) = match gpu.evaluate_activation(
                subset_samples,
                activation_type,
                orientation,
                scale,
            ) {
                Ok(result) => (result.0, result.1),
                Err(_) => {
                    // Fall back to CPU on GPU error
                    let mut sum_act_sq = 0.0;
                    let mut sum_err_act = 0.0;
                    for sample in subset_samples {
                        let pre_activation = incoming_weight * sample.activation;
                        let output = (spec.activation)(pre_activation);
                        if output.is_finite() {
                            sum_act_sq += output * output;
                            sum_err_act += output * sample.avg_error;
                        }
                    }
                    (sum_act_sq, sum_err_act)
                }
            };

            let (outgoing_weight, optimal_bias) = if spec.name == "IDENTITY" {
                match calculate_optimal_identity_outgoing_and_bias(subset_samples, incoming_weight)
                {
                    Some((w, b)) => (w, b),
                    None => continue,
                }
            } else {
                // Use shared weight calculation with ratio validation
                let outgoing_weight = match calculate_optimal_outgoing_weight(
                    sum_error_activation,
                    sum_activation_sq,
                    incoming_weight,
                ) {
                    Some(w) => w,
                    None => continue, // Skip if weight is invalid or ratio too small
                };

                // Calculate optimal bias from subset
                let optimal_bias = calculate_optimal_bias(
                    subset_samples,
                    incoming_weight,
                    outgoing_weight,
                    spec.activation,
                    spec.name,
                    None,
                    target_squash,
                );
                (outgoing_weight, optimal_bias)
            };

            // Issue #123: Check for saturation - reject if neuron output is nearly constant.
            if !has_sufficient_output_variance(
                all_samples,
                incoming_weight,
                optimal_bias,
                spec.activation,
            ) {
                continue;
            }

            // CRITICAL: Evaluate NET improvement across ALL samples
            let (net_improvement, improved_count, total_count) =
                compute_activation_improvement_and_count(
                    all_samples,
                    incoming_weight,
                    outgoing_weight,
                    optimal_bias,
                    spec.activation,
                    total_baseline_error_sq,
                    target_activation_fn,
                );

            // Only consider candidates with positive NET improvement
            if net_improvement <= 0.0 {
                continue;
            }

            // Apply validity filters
            let absolute_improvement = net_improvement * total_baseline_error_sq;
            if absolute_improvement < 0.001 {
                continue;
            }

            if spec.name == "IDENTITY" && optimal_bias.abs() < 0.01 {
                continue;
            }

            // Track best candidate
            if net_improvement > best_net_improvement {
                // Guard rail: do not return candidates with absurd bias values.
                let bias_abs_max =
                    crate::analysis::utils::sensible_bias_abs_max_for_squash(spec.name);
                if !optimal_bias.is_finite() || optimal_bias.abs() > bias_abs_max {
                    continue;
                }

                best_net_improvement = net_improvement;

                let target_stats = NeuronStats::from_samples(all_samples).map(|s| s.to_json());
                // Issue #128: Use creature-level metrics
                // Issue #194: Compute confidence metrics for this prediction
                let confidence_metrics = compute_confidence_metrics(
                    all_samples,
                    net_improvement,
                    None, // R² not available for neuron candidates
                );
                best_candidate = Some(CandidateNeuronJson {
                    source_neuron_uuid: source_uuid.to_string(),
                    target_neuron_uuid: target_uuid.to_string(),
                    source_neuron_index: None, // Set during impact discounting
                    target_neuron_index: None, // Set during impact discounting
                    incoming_weight,
                    outgoing_weight,
                    squash: spec.name.to_string(),
                    bias: optimal_bias,
                    comment: None,
                    target_neuron_impact: 1.0,
                    expected_creature_error_reduction: net_improvement,
                    expected_creature_score_gain: net_improvement,
                    improved_count,
                    total_count,
                    target_neuron_stats: target_stats,
                    prediction_confidence: confidence_metrics.prediction_confidence,
                    expected_score_gain_confidence_interval: confidence_metrics
                        .expected_score_gain_confidence_interval,
                });
            }
        }
    }

    Ok(best_candidate)
}

/// Evaluate activation candidates for a given source-target pair.
///
/// This function handles both split-error evaluation and fallback to all-samples
/// evaluation when appropriate.
#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_activation_candidate<G: GpuEvaluator>(
    gpu: &G,
    source_uuid: &str,
    target_uuid: &str,
    samples: &[HelpfulSample],
    threshold: f32,
    spec: &ActivationCandidateSpec,
    target_squash: Option<&str>,
) -> Result<Option<CandidateNeuronJson>> {
    if samples.len() < MIN_NEURON_SAMPLE_COUNT {
        return Ok(None);
    }

    let activation_type = activation_name_to_gpu_id(spec.name);

    let mut best_candidate: Option<CandidateNeuronJson> = None;
    let mut best_score = threshold;
    let mut fallback_candidate: Option<CandidateNeuronJson> = None;
    let mut fallback_score = f32::MIN;

    // v0.1.135: Split-error evaluation for all activations (not just ReLU).
    let positive_error_samples: Vec<HelpfulSample> = samples
        .iter()
        .filter(|s| s.avg_error > EPSILON)
        .copied()
        .collect();

    let negative_error_samples: Vec<HelpfulSample> = samples
        .iter()
        .filter(|s| s.avg_error < -EPSILON)
        .copied()
        .collect();

    // Compute total_baseline_error_sq across ALL samples (for net improvement)
    let mut total_baseline_error_sq = 0.0;
    for sample in samples {
        if sample.avg_error.is_finite() {
            total_baseline_error_sq += sample.avg_error * sample.avg_error;
        }
    }

    // Get target activation function for net improvement calculation
    let target_activation_fn = get_target_simulation_fn(samples, target_squash);

    // v0.1.136: Track whether split-error evaluation was properly attempted.
    let positive_subset_valid = positive_error_samples.len() >= MIN_NEURON_SAMPLE_COUNT;
    let negative_subset_valid = negative_error_samples.len() >= MIN_NEURON_SAMPLE_COUNT;
    let split_error_attempted = positive_subset_valid && negative_subset_valid;

    // Evaluate candidates from BOTH error subsets
    for error_samples in [&positive_error_samples, &negative_error_samples] {
        if error_samples.len() < MIN_NEURON_SAMPLE_COUNT {
            continue;
        }

        // Compute baseline for this subset (used for weight calculation)
        let subset_baseline_sq: f32 = error_samples
            .iter()
            .map(|s| s.avg_error * s.avg_error)
            .sum();

        if subset_baseline_sq <= EPSILON {
            continue;
        }

        if let Some(candidate) = evaluate_activation_for_subset(
            gpu,
            source_uuid,
            target_uuid,
            error_samples, // Compute weight from subset
            samples,       // Evaluate improvement on ALL samples
            spec,
            target_squash,
            total_baseline_error_sq,
            target_activation_fn,
        )? {
            // Track best and fallback candidates from split evaluation
            if candidate.expected_creature_score_gain > best_score {
                best_score = candidate.expected_creature_score_gain;
                best_candidate = Some(candidate.clone());
            }
            if candidate.expected_creature_score_gain > fallback_score
                && candidate.expected_creature_score_gain > 0.0
            {
                fallback_score = candidate.expected_creature_score_gain;
                fallback_candidate = Some(candidate);
            }
        }
    }

    // If split-error evaluation found candidates, return the best
    if best_candidate.is_some() || fallback_candidate.is_some() {
        return Ok(best_candidate.or(fallback_candidate));
    }

    // v0.1.136: If split-error evaluation was properly attempted but found NOTHING,
    // don't fall back to all-samples.
    if split_error_attempted {
        return Ok(None);
    }

    // Fall back to original ALL-samples evaluation ONLY for cases where errors
    // aren't clearly split
    for &orientation in spec.orientations {
        for &scale in spec.scales {
            let incoming_weight = orientation * scale;
            let (
                sum_activation_sq,
                sum_error_activation,
                gpu_baseline_sq,
                _gpu_improved_count,
                gpu_succeeded,
            ) = match gpu.evaluate_activation(samples, activation_type, orientation, scale) {
                Ok(result) => (result.0, result.1, result.2, result.3, true),
                Err(_) => {
                    // Fall back to CPU if GPU fails
                    let mut sum_activation_sq = 0.0;
                    let mut sum_error_activation = 0.0;
                    for sample in samples {
                        let pre_activation = incoming_weight * sample.activation;
                        let output = (spec.activation)(pre_activation);
                        if output.is_finite() {
                            sum_activation_sq += output * output;
                            sum_error_activation += output * sample.avg_error;
                        }
                    }
                    (
                        sum_activation_sq,
                        sum_error_activation,
                        total_baseline_error_sq,
                        0,
                        false,
                    )
                }
            };

            // Use GPU baseline if GPU succeeded, otherwise use CPU baseline
            let baseline_sq = if gpu_succeeded {
                gpu_baseline_sq
            } else {
                total_baseline_error_sq
            };

            let total_count = samples.len() as u32;
            if total_count == 0 {
                continue;
            }

            // For non-linear targets, search for best outgoing_weight
            let target_activation_fn = get_target_simulation_fn(samples, target_squash);
            let (outgoing_weight, optimal_bias, neuron_error_improvement, final_improved_count) =
                if spec.name == "IDENTITY" {
                    let (outgoing_weight, optimal_bias) =
                        match calculate_optimal_identity_outgoing_and_bias(samples, incoming_weight)
                        {
                            Some((w, b)) => (w, b),
                            None => continue,
                        };

                    let (improvement, improved_count, _) = compute_activation_improvement_and_count(
                        samples,
                        incoming_weight,
                        outgoing_weight,
                        optimal_bias,
                        spec.activation,
                        baseline_sq,
                        target_activation_fn,
                    );

                    (outgoing_weight, optimal_bias, improvement, improved_count)
                } else if target_activation_fn.is_some() {
                    // Use shared weight calculation with validation.
                    let base_weight = match calculate_optimal_outgoing_weight(
                        sum_error_activation,
                        sum_activation_sq,
                        incoming_weight,
                    ) {
                        Some(w) => w,
                        None => continue,
                    };

                    // Weight candidates: base weight and scaled versions
                    let weight_candidates: [f32; 9] = [
                        base_weight * 0.1,
                        base_weight * 0.25,
                        base_weight * 0.5,
                        base_weight * 0.75,
                        base_weight,
                        base_weight * 1.5,
                        base_weight * 2.0,
                        -base_weight * 0.5,
                        -base_weight,
                    ];

                    let mut best_weight = base_weight;
                    let mut best_bias = 0.0f32;
                    let mut best_improvement = f32::NEG_INFINITY;
                    let mut best_improved_count = 0u32;

                    for &weight in &weight_candidates {
                        // Clamp scaled weights to ensure they stay within bounds
                        let clamped_weight =
                            weight.clamp(-MAX_OUTGOING_WEIGHT, MAX_OUTGOING_WEIGHT);
                        if clamped_weight.abs() <= EPSILON {
                            continue;
                        }

                        let bias = calculate_optimal_bias(
                            samples,
                            incoming_weight,
                            clamped_weight,
                            spec.activation,
                            spec.name,
                            None,
                            target_squash,
                        );

                        // Single pass for improvement and count with target simulation
                        let (improvement, improved, _) = compute_activation_improvement_and_count(
                            samples,
                            incoming_weight,
                            clamped_weight,
                            bias,
                            spec.activation,
                            baseline_sq,
                            target_activation_fn,
                        );

                        if improvement > best_improvement {
                            best_improvement = improvement;
                            best_weight = clamped_weight;
                            best_bias = bias;
                            best_improved_count = improved;
                        }
                    }

                    (
                        best_weight,
                        best_bias,
                        best_improvement,
                        best_improved_count,
                    )
                } else {
                    // For linear targets or when target data unavailable, use the base weight.
                    let base_weight = match calculate_optimal_outgoing_weight(
                        sum_error_activation,
                        sum_activation_sq,
                        incoming_weight,
                    ) {
                        Some(w) => w,
                        None => continue,
                    };

                    let optimal_bias = calculate_optimal_bias(
                        samples,
                        incoming_weight,
                        base_weight,
                        spec.activation,
                        spec.name,
                        None,
                        target_squash,
                    );

                    // CRITICAL FIX: Recompute optimal weight WITH the bias included.
                    let mut sum_activation_sq_with_bias = 0.0f32;
                    let mut sum_error_activation_with_bias = 0.0f32;
                    for sample in samples.iter() {
                        let pre_activation = incoming_weight * sample.activation + optimal_bias;
                        let output = (spec.activation)(pre_activation);
                        if output.is_finite() {
                            sum_activation_sq_with_bias += output * output;
                            sum_error_activation_with_bias += output * sample.avg_error;
                        }
                    }
                    // Use shared function for bias-adjusted weight calculation
                    let outgoing_weight = calculate_optimal_outgoing_weight(
                        sum_error_activation_with_bias,
                        sum_activation_sq_with_bias,
                        incoming_weight,
                    )
                    .unwrap_or(base_weight);

                    let (improvement, improved_count, _) = compute_activation_improvement_and_count(
                        samples,
                        incoming_weight,
                        outgoing_weight,
                        optimal_bias,
                        spec.activation,
                        baseline_sq,
                        None, // Linear approximation
                    );

                    (outgoing_weight, optimal_bias, improvement, improved_count)
                };

            // Skip invalid weights
            if outgoing_weight.abs() <= EPSILON {
                continue;
            }

            // FUNDAMENTAL VALIDITY FILTERS

            // Issue #123: Check for saturation
            if !has_sufficient_output_variance(
                samples,
                incoming_weight,
                optimal_bias,
                spec.activation,
            ) {
                continue;
            }

            // Require minimum ABSOLUTE error reduction
            let absolute_improvement = neuron_error_improvement * baseline_sq;
            if absolute_improvement < 0.001 {
                continue;
            }

            // IDENTITY requires meaningful bias
            if spec.name == "IDENTITY" && optimal_bias.abs() < 0.01 {
                continue;
            }

            // Guard rail: do not return candidates with absurd bias values.
            let bias_abs_max = crate::analysis::utils::sensible_bias_abs_max_for_squash(spec.name);
            if !optimal_bias.is_finite() || optimal_bias.abs() > bias_abs_max {
                continue;
            }

            // Track best candidate
            if neuron_error_improvement > best_score {
                best_score = neuron_error_improvement;

                let target_stats = NeuronStats::from_samples(samples).map(|s| s.to_json());
                // Issue #194: Compute confidence metrics for this prediction
                let confidence_metrics = compute_confidence_metrics(
                    samples,
                    neuron_error_improvement,
                    None, // R² not available for neuron candidates
                );
                best_candidate = Some(CandidateNeuronJson {
                    source_neuron_uuid: source_uuid.to_string(),
                    target_neuron_uuid: target_uuid.to_string(),
                    source_neuron_index: None,
                    target_neuron_index: None,
                    incoming_weight,
                    outgoing_weight,
                    squash: spec.name.to_string(),
                    bias: optimal_bias,
                    comment: None,
                    target_neuron_impact: 1.0,
                    expected_creature_error_reduction: neuron_error_improvement,
                    expected_creature_score_gain: neuron_error_improvement,
                    improved_count: final_improved_count,
                    total_count,
                    target_neuron_stats: target_stats,
                    prediction_confidence: confidence_metrics.prediction_confidence,
                    expected_score_gain_confidence_interval: confidence_metrics
                        .expected_score_gain_confidence_interval,
                });
            }

            // Track fallback (positive improvement but below threshold)
            if neuron_error_improvement > fallback_score && neuron_error_improvement > 0.0 {
                fallback_score = neuron_error_improvement;

                let target_stats = NeuronStats::from_samples(samples).map(|s| s.to_json());
                // Issue #194: Compute confidence metrics for this prediction
                let confidence_metrics = compute_confidence_metrics(
                    samples,
                    neuron_error_improvement,
                    None, // R² not available for neuron candidates
                );
                fallback_candidate = Some(CandidateNeuronJson {
                    source_neuron_uuid: source_uuid.to_string(),
                    target_neuron_uuid: target_uuid.to_string(),
                    source_neuron_index: None,
                    target_neuron_index: None,
                    incoming_weight,
                    outgoing_weight,
                    squash: spec.name.to_string(),
                    bias: optimal_bias,
                    comment: None,
                    target_neuron_impact: 1.0,
                    expected_creature_error_reduction: neuron_error_improvement,
                    expected_creature_score_gain: neuron_error_improvement,
                    improved_count: final_improved_count,
                    total_count,
                    target_neuron_stats: target_stats,
                    prediction_confidence: confidence_metrics.prediction_confidence,
                    expected_score_gain_confidence_interval: confidence_metrics
                        .expected_score_gain_confidence_interval,
                });
            }
        }
    }

    Ok(best_candidate.or(fallback_candidate))
}

// =============================================================================
// Build Samples (test helper)
// =============================================================================

/// Build samples for testing. In production, use TargetMap::from_records() and
/// TargetMap::build_samples_from() for better performance when processing
/// multiple sources against the same target.
#[cfg(test)]
pub(crate) fn build_samples(
    target_records: &[DiscoverRecord],
    from_records: &[DiscoverRecord],
) -> Vec<HelpfulSample> {
    if target_records.is_empty() || from_records.is_empty() {
        return Vec::new();
    }

    // Build map from obs_index to target data (error, value, activation)
    let target_map = TargetMap::from_records(target_records);

    if target_map.map.is_empty() {
        return Vec::new();
    }

    target_map.build_samples_from(from_records)
}

// =============================================================================
// Build Ordered Neurons
// =============================================================================

/// Build an ordered list of neurons for the creature.
/// This includes input neurons (named "input-0", "input-1", etc.) followed by
/// all neurons from the creature definition.
pub(crate) fn build_ordered_neurons(creature: &crate::CreatureJson) -> Vec<OrderedNeuron> {
    let mut ordered = Vec::with_capacity(creature.input + creature.neurons.len());

    for input_index in 0..creature.input {
        ordered.push(OrderedNeuron {
            uuid: format!("input-{input_index}"),
            index: input_index,
        });
    }

    for (offset, neuron) in creature.neurons.iter().enumerate() {
        ordered.push(OrderedNeuron {
            uuid: neuron.uuid.clone(),
            index: creature.input + offset,
        });
    }

    ordered
}

// =============================================================================
// Coordinated Structural Helpers
// =============================================================================

/// Compute expected gain for a coordinated "replace synapse with neuron" group.
///
/// This models the coordinated operation sequence:
/// 1) remove direct synapse (source -> target)
/// 2) add hidden neuron with (source -> newNeuron) and (newNeuron -> target)
#[allow(clippy::too_many_arguments)]
pub(crate) fn expected_gain_replace_synapse_with_hidden_neuron(
    cache: &RecordCache,
    source_uuid: &str,
    target_uuid: &str,
    old_weight: f32,
    incoming_weight: f32,
    outgoing_weight: f32,
    bias: f32,
    squash: &str,
) -> Option<f32> {
    let from_records_arc = cache.get(source_uuid).ok()?;
    let target_records_arc = cache.get(target_uuid).ok()?;
    if from_records_arc.is_empty() || target_records_arc.is_empty() {
        return None;
    }

    let target_map = TargetMap::from_records(target_records_arc.as_ref());
    if target_map.map.is_empty() {
        return None;
    }

    let mut samples = target_map.build_samples_from(from_records_arc.as_ref());
    if samples.is_empty() {
        return None;
    }

    // Adjust baseline errors for the removal of the existing direct synapse.
    for s in &mut samples {
        s.avg_error += old_weight * s.activation;
    }

    let total_baseline_error_sq: f32 = samples.iter().map(|s| s.avg_error * s.avg_error).sum();
    if total_baseline_error_sq <= EPSILON {
        return None;
    }

    if squash.eq_ignore_ascii_case("ReLU") {
        let (improvement, _improved, _total) = compute_relu_improvement_and_count(
            samples.as_slice(),
            incoming_weight,
            outgoing_weight,
            bias,
            total_baseline_error_sq,
            None, // linear domain
        );
        return Some(improvement);
    }

    // Map squash names to activation functions
    let activation_fn: fn(f32) -> f32 = match squash {
        "GELU" => gelu_activation,
        "ELU" => elu_activation,
        "Softplus" => softplus_activation,
        "LOGISTIC" => logistic_activation,
        "TANH" => tanh_activation,
        "IDENTITY" => identity_activation,
        "BIPOLAR" => bipolar_activation,
        "CLIPPED" => clipped_activation,
        "ABSOLUTE" => absolute_activation,
        "Mish" => mish_activation,
        "HARD_TANH" => hard_tanh_activation,
        "Softsign" => softsign_activation,
        "BentIdentity" => bent_identity_activation,
        "Arctan" => arctan_activation,
        "ReLU6" => relu6_activation,
        _ => return None,
    };

    let (improvement, _improved, _total) = compute_activation_improvement_and_count(
        samples.as_slice(),
        incoming_weight,
        outgoing_weight,
        bias,
        activation_fn,
        total_baseline_error_sq,
        None, // linear domain
    );
    Some(improvement)
}

// =============================================================================
// Batched Activation Evaluation (Issue #201)
// =============================================================================

use crate::analysis::activation::ACTIVATION_SPECS;

/// Evaluate all activation specs for a (source, target) pair using batched GPU evaluation.
///
/// Issue #201: This function reduces GPU round-trips by 10-20% by evaluating multiple
/// activation function configurations in a single GPU command buffer submission.
///
/// Instead of calling the GPU separately for each (activation_type, orientation, scale)
/// combination, this function:
/// 1. Collects all configs across all activation specs
/// 2. Calls `evaluate_activations_batched` once
/// 3. Processes all results and returns the best candidate for each spec
///
/// # Arguments
/// * `gpu` - The GPU evaluator
/// * `source_uuid` - Source neuron UUID
/// * `target_uuid` - Target neuron UUID
/// * `samples` - The sample data
/// * `threshold` - Minimum improvement threshold
/// * `target_squash` - Target neuron's activation function (for accurate simulation)
///
/// # Returns
/// Vector of best candidates found (one per activation spec that produced a valid candidate).
#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_all_activation_specs_batched<G: GpuEvaluator>(
    gpu: &G,
    source_uuid: &str,
    target_uuid: &str,
    samples: &[HelpfulSample],
    threshold: f32,
    target_squash: Option<&str>,
) -> Result<Vec<CandidateNeuronJson>> {
    if samples.len() < MIN_NEURON_SAMPLE_COUNT {
        return Ok(Vec::new());
    }

    let total_baseline_error_sq: f32 = samples
        .iter()
        .filter(|s| s.avg_error.is_finite())
        .map(|s| s.avg_error * s.avg_error)
        .sum();

    if total_baseline_error_sq <= EPSILON {
        return Ok(Vec::new());
    }

    // Build list of all (spec_idx, activation_type, orientation, scale) combinations
    let mut configs: Vec<(usize, u32, f32, f32)> = Vec::new();
    for (spec_idx, spec) in ACTIVATION_SPECS.iter().enumerate() {
        let activation_type = activation_name_to_gpu_id(spec.name);
        for &orientation in spec.orientations {
            for &scale in spec.scales {
                configs.push((spec_idx, activation_type, orientation, scale));
            }
        }
    }

    // Extract just the GPU configs (activation_type, orientation, scale)
    let gpu_configs: Vec<(u32, f32, f32)> = configs
        .iter()
        .map(|&(_, activation_type, orientation, scale)| (activation_type, orientation, scale))
        .collect();

    // Call batched GPU evaluation
    let gpu_results = match gpu.evaluate_activations_batched(samples, &gpu_configs) {
        Ok(results) => results,
        Err(_) => {
            // Fall back to sequential evaluation if batched fails
            return evaluate_all_activation_specs_sequential(
                gpu,
                source_uuid,
                target_uuid,
                samples,
                threshold,
                target_squash,
            );
        }
    };

    // Get target activation function for accurate simulation
    let target_activation_fn = get_target_simulation_fn(samples, target_squash);

    // Process results and find best candidate for each spec
    let mut best_candidates: Vec<Option<(CandidateNeuronJson, f32)>> =
        vec![None; ACTIVATION_SPECS.len()];

    for (idx, &(spec_idx, _activation_type, orientation, scale)) in configs.iter().enumerate() {
        let spec = &ACTIVATION_SPECS[spec_idx];
        let (sum_activation_sq, sum_error_activation, _gpu_baseline_sq, _improved_count) =
            gpu_results[idx];

        let incoming_weight = orientation * scale;

        // Calculate optimal outgoing weight
        let (outgoing_weight, optimal_bias) = if spec.name == "IDENTITY" {
            match calculate_optimal_identity_outgoing_and_bias(samples, incoming_weight) {
                Some((w, b)) => (w, b),
                None => continue,
            }
        } else {
            let outgoing_weight = match calculate_optimal_outgoing_weight(
                sum_error_activation,
                sum_activation_sq,
                incoming_weight,
            ) {
                Some(w) => w,
                None => continue,
            };

            let optimal_bias = calculate_optimal_bias(
                samples,
                incoming_weight,
                outgoing_weight,
                spec.activation,
                spec.name,
                None,
                target_squash,
            );
            (outgoing_weight, optimal_bias)
        };

        // Check for saturation
        if !has_sufficient_output_variance(samples, incoming_weight, optimal_bias, spec.activation)
        {
            continue;
        }

        // Compute improvement
        let (net_improvement, improved_count, total_count) =
            compute_activation_improvement_and_count(
                samples,
                incoming_weight,
                outgoing_weight,
                optimal_bias,
                spec.activation,
                total_baseline_error_sq,
                target_activation_fn,
            );

        if net_improvement <= threshold {
            continue;
        }

        // Apply validity filters
        let absolute_improvement = net_improvement * total_baseline_error_sq;
        if absolute_improvement < 0.001 {
            continue;
        }

        if spec.name == "IDENTITY" && optimal_bias.abs() < 0.01 {
            continue;
        }

        // Guard rail: do not return candidates with absurd bias values
        let bias_abs_max = crate::analysis::utils::sensible_bias_abs_max_for_squash(spec.name);
        if !optimal_bias.is_finite() || optimal_bias.abs() > bias_abs_max {
            continue;
        }

        // Update best candidate for this spec if this is better
        let current_best = &best_candidates[spec_idx];
        if current_best.is_none() || net_improvement > current_best.as_ref().unwrap().1 {
            let target_neuron_stats = NeuronStats::from_samples(samples).map(|s| s.to_json());
            // Issue #194: Compute confidence metrics for this prediction
            let confidence_metrics = compute_confidence_metrics(
                samples,
                net_improvement,
                None, // R² not available for neuron candidates
            );
            let candidate = CandidateNeuronJson {
                source_neuron_uuid: source_uuid.to_string(),
                target_neuron_uuid: target_uuid.to_string(),
                source_neuron_index: None,
                target_neuron_index: None,
                incoming_weight,
                outgoing_weight,
                squash: spec.name.to_string(),
                bias: optimal_bias,
                comment: None,
                target_neuron_impact: 1.0,
                expected_creature_error_reduction: net_improvement,
                expected_creature_score_gain: net_improvement,
                improved_count,
                total_count,
                target_neuron_stats,
                prediction_confidence: confidence_metrics.prediction_confidence,
                expected_score_gain_confidence_interval: confidence_metrics
                    .expected_score_gain_confidence_interval,
            };
            best_candidates[spec_idx] = Some((candidate, net_improvement));
        }
    }

    // Collect non-None candidates
    let result: Vec<CandidateNeuronJson> = best_candidates
        .into_iter()
        .filter_map(|opt| opt.map(|(candidate, _)| candidate))
        .collect();

    Ok(result)
}

/// Sequential fallback for when batched evaluation fails.
fn evaluate_all_activation_specs_sequential<G: GpuEvaluator>(
    gpu: &G,
    source_uuid: &str,
    target_uuid: &str,
    samples: &[HelpfulSample],
    threshold: f32,
    target_squash: Option<&str>,
) -> Result<Vec<CandidateNeuronJson>> {
    let mut results = Vec::new();
    for spec in ACTIVATION_SPECS.iter() {
        if let Some(candidate) = evaluate_activation_candidate(
            gpu,
            source_uuid,
            target_uuid,
            samples,
            threshold,
            spec,
            target_squash,
        )? {
            results.push(candidate);
        }
    }
    Ok(results)
}

/// Deterministic UUID generator for coordinated structural `addNeuron` operations.
pub(crate) fn deterministic_coordinated_neuron_uuid(
    source_uuid: &str,
    target_uuid: &str,
    squash: &str,
    incoming_weight: f32,
    outgoing_weight: f32,
    bias: f32,
) -> String {
    let key = format!(
        "replace-synapse-with-neuron|{source_uuid}|{target_uuid}|{squash}|{incoming_weight:.6}|{outgoing_weight:.6}|{bias:.6}"
    );
    // FNV-1a 64-bit
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in key.as_bytes() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("coordinated-hidden-{hash:016x}")
}

// =============================================================================
// Truncate Combined Synapse Candidate Sets
// =============================================================================

/// Truncate synapse candidate buckets to a global cap, preserving ordering semantics.
pub(crate) fn truncate_combined_synapse_candidate_sets(
    helpful: Vec<CandidateSynapseJson>,
    harmful: Vec<CandidateSynapseJson>,
    coordinated: Vec<crate::CoordinatedStructuralCandidateJson>,
    limit: usize,
    diversify: bool,
) -> (
    Vec<CandidateSynapseJson>,
    Vec<CandidateSynapseJson>,
    Vec<crate::CoordinatedStructuralCandidateJson>,
) {
    if limit == 0 {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    // If we're already under the global cap, keep the caller's ordering exactly.
    let total = helpful.len() + harmful.len() + coordinated.len();
    if total <= limit {
        return (helpful, harmful, coordinated);
    }

    // In diversified mode we intentionally preserve the per-bucket ordering
    if diversify {
        use std::collections::VecDeque;

        let mut helpful_q: VecDeque<CandidateSynapseJson> = VecDeque::from(helpful);
        let mut harmful_q: VecDeque<CandidateSynapseJson> = VecDeque::from(harmful);
        let mut coordinated_q: VecDeque<crate::CoordinatedStructuralCandidateJson> =
            VecDeque::from(coordinated);

        let mut helpful_out = Vec::new();
        let mut harmful_out = Vec::new();
        let mut coordinated_out = Vec::new();

        let mut returned = 0usize;
        while returned < limit {
            let mut progressed = false;

            if let Some(c) = helpful_q.pop_front() {
                helpful_out.push(c);
                returned += 1;
                progressed = true;
                if returned >= limit {
                    break;
                }
            }
            if let Some(c) = harmful_q.pop_front() {
                harmful_out.push(c);
                returned += 1;
                progressed = true;
                if returned >= limit {
                    break;
                }
            }
            if let Some(c) = coordinated_q.pop_front() {
                coordinated_out.push(c);
                returned += 1;
                progressed = true;
                if returned >= limit {
                    break;
                }
            }

            if !progressed {
                break;
            }
        }

        return (helpful_out, harmful_out, coordinated_out);
    }

    enum Any {
        Helpful(CandidateSynapseJson),
        Harmful(CandidateSynapseJson),
        Coordinated(crate::CoordinatedStructuralCandidateJson),
    }

    fn score(item: &Any) -> f32 {
        match item {
            Any::Helpful(c) => c.expected_creature_score_gain,
            Any::Harmful(c) => c.expected_creature_score_gain,
            Any::Coordinated(c) => c.expected_creature_score_gain,
        }
    }

    let mut combined: Vec<Any> = Vec::with_capacity(total);
    combined.extend(helpful.into_iter().map(Any::Helpful));
    combined.extend(harmful.into_iter().map(Any::Harmful));
    combined.extend(coordinated.into_iter().map(Any::Coordinated));

    combined.sort_by(|a, b| score(b).total_cmp(&score(a)));
    combined.truncate(limit);

    let mut helpful_out = Vec::new();
    let mut harmful_out = Vec::new();
    let mut coordinated_out = Vec::new();
    for item in combined {
        match item {
            Any::Helpful(c) => helpful_out.push(c),
            Any::Harmful(c) => harmful_out.push(c),
            Any::Coordinated(c) => coordinated_out.push(c),
        }
    }

    (helpful_out, harmful_out, coordinated_out)
}

// =============================================================================
// Core Implementation (Issue #425: moved from implementation.rs)
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
    let ordered_neurons = build_ordered_neurons(&input.creature);
    let order_map: HashMap<String, usize> = ordered_neurons
        .iter()
        .map(|neuron| (neuron.uuid.clone(), neuron.index))
        .collect();

    // Issue #210: Use NeuronIndex to intern UUID strings, reducing memory allocations.
    // Instead of cloning UUID strings for each synapse (72+ bytes per entry for String pairs),
    // we use u32 indices (8 bytes per entry) for ~89% memory reduction.
    let mut neuron_index = NeuronIndex::with_capacity(
        input.creature.neurons.len() + input.creature.input + input.creature.synapses.len() / 10,
    );

    // Pre-intern all neuron UUIDs (inputs + neurons from creature)
    for i in 0..input.creature.input {
        neuron_index.intern(&format!("input-{i}"));
    }
    for neuron in &input.creature.neurons {
        neuron_index.intern(&neuron.uuid);
    }

    // Build existing_synapses using interned indices instead of String clones
    let existing_synapses: HashSet<(u32, u32)> = input
        .creature
        .synapses
        .iter()
        .map(|synapse| {
            (
                neuron_index.intern(&synapse.from_uuid),
                neuron_index.intern(&synapse.to_uuid),
            )
        })
        .collect();

    // Build existing_synapse_weights using interned indices
    let existing_synapse_weights: HashMap<(u32, u32), f32> = input
        .creature
        .synapses
        .iter()
        .map(|synapse| {
            (
                (
                    neuron_index.intern(&synapse.from_uuid),
                    neuron_index.intern(&synapse.to_uuid),
                ),
                synapse.weight,
            )
        })
        .collect();

    // Build synapses_by_target using interned index as key
    let synapses_by_target: HashMap<u32, Vec<SynapseJson>> = input
        .creature
        .synapses
        .iter()
        .map(|synapse| (neuron_index.intern(&synapse.to_uuid), synapse.clone()))
        .fold(HashMap::new(), |mut acc, (key, val)| {
            acc.entry(key).or_default().push(val);
            acc
        });

    // Build a lookup map for neuron squash functions to identify discrete targets
    let neuron_squash_map: HashMap<String, String> = input
        .creature
        .neurons
        .iter()
        .map(|n| (n.uuid.clone(), n.squash.clone()))
        .collect();

    let unique_focus = require_unique_focus(&input.focus_neurons, "analyse_synapses")?;

    // Issue #216: TargetDiagnostics uses DashMap internally for lock-free concurrent access.
    // No Mutex wrapper needed - the struct handles concurrency internally.
    let diagnostics = Arc::new(TargetDiagnostics::new(&unique_focus));

    // GPU timing collector (Issue #195)
    // Only collects timing data when NEAT_AI_DISCOVERY_GPU_TIMING=1 is set
    let timing_collector = Arc::new(super::shared::TimingCollector::new(
        super::utils::gpu_timing_enabled(),
    ));

    let deadline = build_deadline(input.analysis_deadline_ms);
    // Randomise the focus neuron order so that repeated runs with timeouts will
    // eventually cover all neurons. Convert to owned strings, shuffle, then use.
    // STEP/BIPOLAR neurons are now included - both get proper simulation functions
    // that accurately predict output flips when synapse contributions cross the threshold.
    let mut focus_order: Vec<String> = unique_focus.iter().map(|s| (*s).clone()).collect();

    // Issue #468: Build a neuron type map for target-type prioritisation.
    // Existing hidden neurons as targets have a 31.4% success rate vs 5.3–5.4%
    // for output neurons, so we evaluate hidden targets first under deadline pressure.
    let focus_neuron_type_map: HashMap<String, String> = input
        .creature
        .neurons
        .iter()
        .map(|n| (n.uuid.clone(), n.neuron_type.clone()))
        .collect();

    // Issue #468: Order focus targets so existing hidden neurons are evaluated first.
    // Each partition is shuffled independently for exploration diversity.
    order_focus_targets(&mut focus_order, input.random_seed, &focus_neuron_type_map);

    // Log analysis start with timeout duration and randomised order
    log_analysis_start(
        "synapse",
        input.analysis_deadline_ms,
        focus_order.len(),
        &focus_order,
    );

    // Track completed focus neurons for timeout logging
    let total_focus_count = focus_order.len();
    let completed_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    // v0.1.134: Return ALL positive improvements.
    // NEAT-AI applies the cost-of-growth gate during evaluation.
    let threshold = 0.0;

    // Collect all helpful evaluation work first for batching
    struct HelpfulWork {
        source_uuid: String,
        target_uuid: String,
        samples: Vec<HelpfulSample>,
        /// Existing synapse weight (when the synapse already exists).
        ///
        /// When set, we propose a delta-based weight update rather than adding a new synapse.
        existing_weight: Option<f32>,
    }

    // GPU is always required - TypeScript layer calls check_gpu_available() and skips
    // discovery entirely on machines without GPU. Reaching here without GPU is a bug.
    assert!(
        GpuAnalyzer::gpu_is_available(),
        "Discovery logic called without GPU - check_gpu_available should have prevented this"
    );
    let gpu_used = true;

    let helpful_results = Arc::new(Mutex::new(Vec::<CandidateSynapseJson>::new()));
    let harmful_results = Arc::new(Mutex::new(Vec::<CandidateSynapseJson>::new()));
    let coordinated_structural_results = Arc::new(Mutex::new(Vec::<
        crate::CoordinatedStructuralCandidateJson,
    >::new()));
    let helpful_fallback = Arc::new(Mutex::new(Option::<CandidateSynapseJson>::None));
    let analysis_timed_out = Arc::new(Mutex::new(false));

    // Metadata tracking for observability (v0.2.17+)
    // These track whether target_value was available and whether saturation-aware simulation was used
    let metadata_target_value_seen = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let metadata_saturation_aware_used = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let metadata_seen_any_input_with_records = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let metadata_input_min_with_records = Arc::new(std::sync::atomic::AtomicUsize::new(usize::MAX));
    let metadata_input_max_with_records = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    // Issue #192: Collect error values for error distribution analysis
    // We collect errors from all target neurons to compute aggregate distribution statistics
    let error_values_for_distribution = Arc::new(Mutex::new(Vec::<f32>::new()));

    let focus_order_arc = Arc::new(focus_order);
    let ordered_neurons_arc = Arc::new(ordered_neurons);
    let existing_synapses_arc = Arc::new(existing_synapses);
    let existing_synapse_weights_arc = Arc::new(existing_synapse_weights);
    let synapses_by_target_arc = Arc::new(synapses_by_target);
    let order_map_arc = Arc::new(order_map);
    let neuron_squash_map_arc = Arc::new(neuron_squash_map);
    // Issue #210: Share neuron index for interned UUID lookups across parallel threads
    let neuron_index_arc = Arc::new(neuron_index);

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

    // Map of neuron UUID -> bias, used when folding constant-source synapses into `setBias`.
    // Inputs are not present here (they have no bias).
    let neuron_bias_map: HashMap<String, f32> = input
        .creature
        .neurons
        .iter()
        .map(|n| (n.uuid.clone(), n.bias))
        .collect();
    let neuron_bias_map_arc = Arc::new(neuron_bias_map);

    // Issue #199: Compute source variance profile for dynamic constant-source threshold.
    // We sample source neurons to compute an average standard deviation, which is used
    // to scale the threshold for folding constant sources into setBias operations.
    // This captures more coordinated candidates in creatures where "constant" is relative.
    let source_std_dev_avg: Option<f32> = {
        // Sample input neurons to compute average std dev
        let mut std_dev_sum = 0.0f64;
        let mut std_dev_count = 0u32;
        let max_samples = input.creature.input.min(50); // Sample up to 50 input neurons

        for input_idx in 0..max_samples {
            let input_uuid = format!("input-{input_idx}");
            if let Ok(records) = cache.get(&input_uuid)
                && records.len() >= 2
            {
                let std_dev = compute_source_std_dev(&records);
                if std_dev.is_finite() {
                    std_dev_sum += std_dev as f64;
                    std_dev_count += 1;
                }
            }
        }

        if std_dev_count > 0 {
            let avg = (std_dev_sum / std_dev_count as f64) as f32;
            if verbose_enabled() {
                eprintln!(
                    "[NEAT-AI-Discovery][verbose] Source variance profile: avg_std_dev={avg:.4} (sampled {std_dev_count} sources)"
                );
            }
            Some(avg)
        } else {
            None
        }
    };

    // Issue #178, #199: threshold for folding constant-ish sources into bias adjustments.
    // Now uses dynamic threshold based on source variance profile (Issue #199).
    let constant_source_effect_threshold = get_constant_source_threshold(source_std_dev_avg);

    // Build a comprehensive map of ALL neuron UUIDs to their types
    // This includes: input neurons, and all neurons from creature.neurons (hidden, output, constant)
    // If a UUID is not in this map, it's an invalid UUID (bug)
    let mut neuron_type_map: HashMap<String, String> = HashMap::new();

    // Add input neurons (they're not in creature.neurons, only represented by creature.input count)
    for input_index in 0..input.creature.input {
        neuron_type_map.insert(format!("input-{input_index}"), "input".to_string());
    }

    // Add all neurons from creature.neurons (hidden, output, constant)
    for neuron in &input.creature.neurons {
        neuron_type_map.insert(neuron.uuid.clone(), neuron.neuron_type.clone());
    }

    let neuron_type_map_arc = Arc::new(neuron_type_map);

    // Keep input neuron UUIDs set for quick checks (backwards compatibility)
    let input_neuron_uuids: HashSet<String> = (0..input.creature.input)
        .map(|i| format!("input-{i}"))
        .collect();
    let input_neuron_uuids_arc = Arc::new(input_neuron_uuids);

    // Use the provided GPU work queue.
    // This eliminates the overhead of creating multiple GPU devices (one per thread).
    // All GPU operations are processed by a single dedicated thread, improving utilisation.
    // CRITICAL: The GpuAnalyzer is created INSIDE the GPU thread to avoid wgpu deadlocks.

    // Process each focus neuron in parallel
    focus_order_arc
        .par_iter()
        .try_for_each(|target_uuid| -> Result<()> {
            if *analysis_timed_out.lock().expect("Mutex poisoned") || deadline_passed(&deadline) {
                *analysis_timed_out.lock().expect("Mutex poisoned") = true;
                return Ok(());
            }

            crate::watchdog::beat(format!(
                "synapse analysis → processing target {target_uuid}"
            ));

            // Use the shared GPU work queue instead of creating a new GpuAnalyzer per thread.
            // This eliminates device creation overhead and improves GPU utilisation.
            let gpu = &*gpu_queue;

            let target_records_arc = cache.get(target_uuid.as_str())?;
            if target_records_arc.is_empty() {
                diagnostics.set_target_record_count(target_uuid, 0);
                return Ok(());
            }
            let target_records = target_records_arc.as_ref();
            diagnostics.set_target_record_count(target_uuid, target_records.len());

            // Issue #192: Collect error values for distribution analysis
            // Extract errors from target records and add to shared collection
            {
                let errors: Vec<f32> = target_records
                    .iter()
                    .flat_map(|r| r.errors.iter().filter(|e| e.is_finite()).copied())
                    .collect();
                if !errors.is_empty() {
                    let mut error_vec = error_values_for_distribution
                        .lock()
                        .expect("Mutex poisoned: error_values_for_distribution");
                    error_vec.extend(errors);
                }
            }

            let target_index = match order_map_arc.get(target_uuid.as_str()) {
                Some(index) => *index,
                None => {
                    // Target neuron not found in order map - this indicates a data integrity issue
                    // This should never happen for valid hidden/output neurons
                    if verbose_enabled() {
                        eprintln!(
                            "[NEAT-AI-Discovery][verbose] Target {target_uuid} not found in creature neuron order map (neuron may not exist in creature definition). Skipping."
                        );
                    }
                    return Ok(());
                }
            };

            // Early validation: skip input and constant neurons (they have no upstream sources)
            // Validate target UUID exists in comprehensive neuron type map
            let target_neuron_type = neuron_type_map_arc.get(target_uuid.as_str())
                .ok_or_else(|| anyhow!(
                    "Invalid target neuron UUID '{}': not found in neuron type map. \
                    This indicates a serious data integrity bug. All valid neurons must be in the type map \
                    (input neurons: input-0..input-{}, or neurons from creature.neurons array).",
                    target_uuid,
                    input_neuron_uuids_arc.len().saturating_sub(1)
                ))?;

            let input_count = input_neuron_uuids_arc.len();
            let is_input_neuron = target_neuron_type == "input";
            let is_constant_neuron = target_neuron_type == "constant";

            // Skip actual input/constant neurons (expected - they have no upstream sources)
            // Only skip by UUID check, not by index, to avoid incorrectly skipping hidden neurons
            // that might have been assigned incorrect indices due to ordering bugs
            if is_input_neuron || is_constant_neuron {
                return Ok(());
            }

            // Filter eligible sources: must have index < target_index and not be a constant
            // All neurons should be in the comprehensive neuron_type_map
            let mut eligible_sources: Vec<&OrderedNeuron> = ordered_neurons_arc
                .iter()
                .filter(|neuron| {
                    neuron.index < target_index
                        && {
                            // Look up neuron type - if missing, it's a serious bug
                            match neuron_type_map_arc.get(&neuron.uuid) {
                                Some(neuron_type) => {
                                    // Valid neuron - exclude constants, include everything else (input, hidden, output)
                                    neuron_type != "constant"
                                }
                                None => {
                                    // Invalid UUID - serious data integrity bug
                                    eprintln!(
                                        "[NEAT-AI-Discovery] ERROR: Invalid neuron UUID '{}' found in ordered_neurons. \
                                        Not found in comprehensive neuron type map. This indicates a serious data integrity bug.",
                                        neuron.uuid
                                    );
                                    false // Exclude invalid neurons
                                }
                            }
                        }
                })
                .collect();

            // Track total eligible sources before filtering
            let total_eligible = eligible_sources.len() as u32;

            // For hidden/output neurons with index >= input_count, there should always be at least the input neurons as eligible sources
            // If total_eligible == 0, this indicates a serious bug
            if total_eligible == 0 {
                // This should be impossible - we've already filtered out input/constant neurons
                // Log detailed diagnostics to help debug
                let neurons_before_index = ordered_neurons_arc
                    .iter()
                    .filter(|n| n.index < target_index)
                    .count();
                let constants_before_index = ordered_neurons_arc
                    .iter()
                    .filter(|n| {
                        n.index < target_index
                            && neuron_type_map_arc
                                .get(&n.uuid)
                                .map(|t| t == "constant")
                                .unwrap_or(false)
                    })
                    .count();
                let input_neurons_before_index = ordered_neurons_arc
                    .iter()
                    .filter(|n| n.index < target_index && input_neuron_uuids_arc.contains(&n.uuid))
                    .count();

                eprintln!(
                    "[NEAT-AI-Discovery] BUG: Target {target_uuid} (type: {target_neuron_type}, index: {target_index}) has no eligible upstream neurons. \
                    creature.input: {input_count}, neurons before target: {neurons_before_index}, constants before target: {constants_before_index}, \
                    input neurons before target: {input_neurons_before_index}. This should not happen for hidden/output neurons with index >= creature.input."
                );

                // Still skip to avoid crashing, but log the bug
                return Ok(());
            }
            // Count how many eligible sources are input neurons
            let input_neuron_count = eligible_sources
                .iter()
                .filter(|neuron| input_neuron_uuids_arc.contains(&neuron.uuid))
                .count() as u32;
            diagnostics.set_total_eligible_sources(target_uuid, total_eligible);
            diagnostics.set_input_neuron_count(target_uuid, input_neuron_count);

            let context = format!("synapse:eligible_sources:{target_uuid}");
            order_eligible_sources(
                &mut eligible_sources,
                input.random_seed,
                &context,
                input.creature.input,
                Some(&*used_inputs_arc),
            );

            // Improved GPU utilisation: Build samples on CPU in parallel, then batch GPU evaluation
            // This avoids the GPU sync overhead of calling build_samples_gpu for each source.
            // The heavy computation is in evaluate_helpful_batch which is properly batched.
            struct SourceWorkResult {
                work: Option<HelpfulWork>,
                had_samples: bool,
                source_uuid: String,
                record_count: usize,
            }

            // Pre-filter sources and collect their records (cache is thread-safe)
            // Track already-connected, load-failure, and empty-record counts separately
            let mut already_connected_count = 0u32;
            let mut load_failure_count = 0u32;
            let mut empty_record_sources: Vec<String> = Vec::new();
            struct ExistingSourceToProcess<'a> {
                source: &'a OrderedNeuron,
                records: Arc<Vec<DiscoverRecord>>,
                old_weight: f32,
            }

            // Note: This vector is for *add-synapse* candidates only.
            // Existing edges are intentionally excluded to preserve the original
            // `NoEligibleSources` diagnostics semantics (tests rely on this).
            let mut sources_to_process: Vec<(&OrderedNeuron, Arc<Vec<DiscoverRecord>>)> =
                Vec::with_capacity(eligible_sources.len());

            // Existing edges are evaluated separately as weight-update candidates.
            let mut existing_sources_to_process: Vec<ExistingSourceToProcess> =
                Vec::with_capacity(eligible_sources.len());

            for source in &eligible_sources {
                if deadline_passed(&deadline) {
                    *analysis_timed_out.lock().expect("Mutex poisoned") = true;
                    break;
                }
                let source_uuid = source.uuid.as_str();

                // Issue #210: Use interned indices for efficient lookup (avoids String allocations)
                let source_idx = neuron_index_arc.get_index(source_uuid);
                let target_idx = neuron_index_arc.get_index(target_uuid.as_str());
                let is_connected = match (source_idx, target_idx) {
                    (Some(s), Some(t)) => existing_synapses_arc.contains(&(s, t)),
                    _ => false, // UUID not interned means it's not in the creature
                };
                if is_connected {
                    already_connected_count += 1;
                };
                let existing_weight = if is_connected {
                    match (source_idx, target_idx) {
                        (Some(s), Some(t)) => existing_synapse_weights_arc.get(&(s, t)).copied(),
                        _ => None,
                    }
                } else {
                    None
                };

                match cache.get(source_uuid) {
                    Ok(records) => {
                        if !records.is_empty() {
                            if let Some(input_index) = parse_input_index(source_uuid) {
                                metadata_seen_any_input_with_records.store(
                                    true,
                                    std::sync::atomic::Ordering::Relaxed,
                                );
                                // Update min/max atomically (best-effort).
                                let _ = metadata_input_min_with_records.fetch_min(
                                    input_index,
                                    std::sync::atomic::Ordering::Relaxed,
                                );
                                let _ = metadata_input_max_with_records.fetch_max(
                                    input_index,
                                    std::sync::atomic::Ordering::Relaxed,
                                );
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
                            // Empty records - track for diagnostics
                            empty_record_sources.push(source_uuid.to_string());
                            let is_input_neuron = input_neuron_uuids_arc.contains(source_uuid);
                            // Log non-input neurons with empty records (input neurons are logged as summary below)
                            if verbose_enabled() && !is_input_neuron {
                                eprintln!(
                                    "[NEAT-AI-Discovery][verbose] Source {source_uuid} (target {target_uuid}) has no records in parquet file."
                                );
                            }
                        }
                    }
                    Err(err) => {
                        load_failure_count += 1;
                        if verbose_enabled() {
                            eprintln!(
                                "[NEAT-AI-Discovery][verbose] Failed to load records for source {source_uuid} (target {target_uuid}): {err}"
                            );
                        }
                    }
                };
            }

            // Count how many empty record sources are input neurons (helps diagnose parquet data issues)
            let empty_input_neuron_count = empty_record_sources
                .iter()
                .filter(|uuid| input_neuron_uuids_arc.contains(uuid.as_str()))
                .count();
            let empty_non_input_count = empty_record_sources.len() - empty_input_neuron_count;

            // Log summary if many input neurons have empty records (indicates data issue)
            if verbose_enabled() && empty_input_neuron_count > 0 {
                eprintln!(
                    "[NEAT-AI-Discovery][verbose] Target {target_uuid}: {} of {} input neurons have no records in parquet file (plus {} non-input sources). This may indicate incomplete parquet data.",
                    empty_input_neuron_count,
                    input_neuron_uuids_arc.len(),
                    empty_non_input_count
                );
            }

            // Update diagnostics for already-connected, load failures, and empty records
            // Issue #216: Direct method calls - no lock needed with DashMap-based diagnostics
            if already_connected_count > 0
                || load_failure_count > 0
                || !empty_record_sources.is_empty()
            {
                for _ in 0..already_connected_count {
                    diagnostics.record_already_connected(target_uuid);
                }
                for _ in 0..load_failure_count {
                    diagnostics.record_load_failure(target_uuid);
                }
                // Record diagnostics for sources with empty records (matches old sequential behaviour)
                for source_uuid in &empty_record_sources {
                    diagnostics.record_candidate_attempt(target_uuid, false);
                    diagnostics.record_no_samples(target_uuid, source_uuid, 0);
                }
            }

            // Build samples on CPU (fast hashmap matching, no GPU sync overhead)
            // OPTIMIZATION: Pre-build target map ONCE, reuse for all sources.
            // This avoids rebuilding the HashMap for each of ~1000+ source neurons.
            let target_map = TargetMap::from_records(target_records);
            let target_map_ref = &target_map;

            // ================================================================
            // Coordinated Structural Discovery (Issue #165): noisy vs trusted
            // ================================================================
            //
            // This targets the "simple case" described in the docs:
            // - Two input signals with the same mean and the same starting synapse weight.
            // - One signal is much noisier (higher activation variance).
            //
            // The intended coordinated fix is:
            // - remove the noisy synapse completely
            // - remove the trusted synapse
            // - add the trusted synapse back with a higher weight (typically doubled)
            //
            // This is designed for large-scale feature inputs (eg market data), where variance
            // differences can reflect noise rather than signal.
            let coordinated_candidate = (|| -> Option<crate::CoordinatedStructuralCandidateJson> {
                fn activation_mean_and_variance(records: &[DiscoverRecord]) -> Option<(f32, f32)> {
                    let mut n = 0.0f32;
                    let mut sum = 0.0f32;
                    let mut sum_sq = 0.0f32;
                    for r in records {
                        if r.activation.is_finite() {
                            n += 1.0;
                            sum += r.activation;
                            sum_sq += r.activation * r.activation;
                        }
                    }
                    if n <= 0.0 {
                        return None;
                    }
                    let mean = sum / n;
                    let var = (sum_sq / n) - (mean * mean);
                    Some((mean, var.max(0.0)))
                }

                fn activation_map(records: &[DiscoverRecord]) -> HashMap<u32, f32> {
                    let mut map = HashMap::with_capacity(records.len());
                    for r in records {
                        if r.activation.is_finite() {
                            map.insert(r.obs_index, r.activation);
                        }
                    }
                    map
                }

                // Issue #210: Use interned index for synapses_by_target lookup
                let target_idx_for_synapses = neuron_index_arc.get_index(target_uuid.as_str())?;
                let existing = synapses_by_target_arc.get(&target_idx_for_synapses)?;

                #[derive(Clone)]
                struct IncomingInput {
                    from_uuid: String,
                    weight: f32,
                    mean: f32,
                    var: f32,
                }

                let mut incoming_inputs: Vec<IncomingInput> = Vec::new();
                for syn in existing.iter() {
                    if !syn.from_uuid.starts_with("input-") {
                        continue;
                    }
                    let Ok(from_records_arc) = cache.get(&syn.from_uuid) else {
                        continue;
                    };
                    if from_records_arc.is_empty() {
                        continue;
                    }
                    let Some((mean, var)) = activation_mean_and_variance(from_records_arc.as_ref())
                    else {
                        continue;
                    };
                    incoming_inputs.push(IncomingInput {
                        from_uuid: syn.from_uuid.clone(),
                        weight: syn.weight,
                        mean,
                        var,
                    });
                }

                if incoming_inputs.len() < 2 || target_map_ref.map.is_empty() {
                    None
                } else {
                    // Strict matching for the simple-case test: same weights and same means.
                    const WEIGHT_EPS: f32 = 1e-6;
                    const MEAN_EPS: f32 = 1e-3;
                    const MIN_VAR_RATIO: f32 = 10.0;

                    let target_squash = neuron_squash_map_arc
                        .get(target_uuid.as_str())
                        .map(|s| s.as_str());

                    let mut best: Option<(IncomingInput, IncomingInput, f32)> = None; // (noisy, trusted, gain)

                    for i in 0..incoming_inputs.len() {
                        for j in (i + 1)..incoming_inputs.len() {
                            let a = incoming_inputs[i].clone();
                            let b = incoming_inputs[j].clone();

                            if (a.weight - b.weight).abs() > WEIGHT_EPS {
                                continue;
                            }
                            if (a.mean - b.mean).abs() > MEAN_EPS {
                                continue;
                            }

                            let (noisy, trusted) = if a.var >= b.var { (a, b) } else { (b, a) };
                            let ratio = noisy.var / trusted.var.max(EPSILON);
                            if ratio < MIN_VAR_RATIO {
                                continue;
                            }

                            let Ok(noisy_records_arc) = cache.get(&noisy.from_uuid) else {
                                continue;
                            };
                            let Ok(trusted_records_arc) = cache.get(&trusted.from_uuid) else {
                                continue;
                            };

                            let noisy_map = activation_map(noisy_records_arc.as_ref());
                            let trusted_map = activation_map(trusted_records_arc.as_ref());

                            let mut delta_samples: Vec<HelpfulSample> =
                                Vec::with_capacity(target_map_ref.map.len());
                            for (obs_index, target) in target_map_ref.map.iter() {
                                let Some(noisy_act) = noisy_map.get(obs_index) else {
                                    continue;
                                };
                                let Some(trusted_act) = trusted_map.get(obs_index) else {
                                    continue;
                                };
                                let Some(activation) = coordinated_structural_activation_delta(
                                    *trusted_act,
                                    *noisy_act,
                                    noisy.weight,
                                    trusted.weight,
                                ) else {
                                    continue;
                                };
                                if !activation.is_finite() || !target.avg_error.is_finite() {
                                    continue;
                                }
                                delta_samples.push(HelpfulSample {
                                    activation,
                                    avg_error: target.avg_error,
                                    target_value: target.value,
                                    target_activation: Some(target.activation),
                                });
                            }

                            if delta_samples.is_empty() {
                                continue;
                            }

                            let baseline_sq: f32 = delta_samples
                                .iter()
                                .map(|s| s.avg_error * s.avg_error)
                                .sum();

                            // Move the noisy weight onto the trusted input:
                            // Δoutput = w_noisy * (trusted - noisy)
                            let moved_weight = noisy.weight;
                            let (improvement, _, _, _) = compute_synapse_improvement_and_count(
                                delta_samples.as_slice(),
                                moved_weight,
                                baseline_sq,
                                target_squash,
                            );

                            if improvement <= 0.0 {
                                continue;
                            }

                            match &best {
                                Some((_, _, best_gain)) if *best_gain >= improvement => {}
                                _ => best = Some((noisy, trusted, improvement)),
                            }
                        }
                    }

                    best.map(|(noisy, trusted, gain)| {
                        let new_weight = trusted.weight + noisy.weight;
                        crate::CoordinatedStructuralCandidateJson {
                            operations: vec![
                                crate::CoordinatedStructuralOpJson::RemoveSynapse {
                                    from_neuron_uuid: noisy.from_uuid,
                                    to_neuron_uuid: target_uuid.to_string(),
                                },
                                crate::CoordinatedStructuralOpJson::RemoveSynapse {
                                    from_neuron_uuid: trusted.from_uuid.clone(),
                                    to_neuron_uuid: target_uuid.to_string(),
                                },
                                crate::CoordinatedStructuralOpJson::AddSynapse {
                                    from_neuron_uuid: trusted.from_uuid,
                                    to_neuron_uuid: target_uuid.to_string(),
                                    weight: new_weight,
                                },
                            ],
                            expected_creature_score_gain: gain,
                            comment: Some(
                                "Coordinated: prune noisy input (high variance), strengthen trusted input"
                                    .to_string(),
                            ),
                        }
                    })
                }
            })();

            if let Some(candidate) = coordinated_candidate {
                let mut results = coordinated_structural_results
                    .lock()
                    .expect("Mutex poisoned: coordinated_structural_results");
                results.push(candidate);
            }

            // Even if target_map is empty, we continue to record diagnostics
            // about what sources were evaluated (important for debugging).
            //
            // Issue #221: Sample Locality Optimisation
            // Group sources by obs_index overlap to reduce redundant sample building.
            // Sources with ≥80% obs_index overlap share sample building in a single pass.
            let source_results: Vec<SourceWorkResult> = {
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
                    eprintln!(
                        "[NEAT-AI-Discovery][verbose] Target {}: {} sources grouped into {} locality groups (max={}, avg={:.1})",
                        target_uuid,
                        sources_to_process.len(),
                        locality_groups.len(),
                        max_group,
                        avg_group
                    );
                }

                // Build samples for each group (groups with multiple sources use batched building)
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

            // Apply all diagnostics updates directly (no lock needed with DashMap - Issue #216)
            for (target, source, had_samples, record_count) in diagnostics_updates {
                diagnostics.record_candidate_attempt(&target, had_samples);
                if !had_samples {
                    diagnostics.record_no_samples(&target, &source, record_count);
                }
            }

            // Append existing edges for weight-update evaluation.
            //
            // Important: we do NOT record these as "candidate attempts" in TargetDiagnostics because
            // `no_candidate_reasons` is reporting add-synapse eligibility (existing edges are not
            // eligible for add-synapse). This preserves the historical semantics and unit tests.
            //
            // Issue #164: Also collect ExistingPathContribution for redundant path detection.
            let mut existing_path_contributions: Vec<ExistingPathContribution> = Vec::new();
            if !existing_sources_to_process.is_empty() {
                let existing_work: Vec<HelpfulWork> = existing_sources_to_process
                    .par_iter()
                    .filter_map(|item| {
                        let source_uuid = item.source.uuid.as_str();
                        let from_records = item.records.as_ref();
                        let samples = target_map_ref.build_samples_from(from_records);
                        if samples.is_empty() {
                            return None;
                        }
                        Some(HelpfulWork {
                            source_uuid: source_uuid.to_string(),
                            target_uuid: target_uuid.to_string(),
                            samples,
                            existing_weight: Some(item.old_weight),
                        })
                    })
                    .collect();

                // Issue #164: Collect existing path contributions for redundant path detection
                for work in &existing_work {
                    existing_path_contributions.push(ExistingPathContribution {
                        source_uuid: work.source_uuid.clone(),
                        existing_weight: work.existing_weight.unwrap_or(0.0),
                        samples: work.samples.clone(),
                    });
                }

                helpful_work_batch.extend(existing_work);
            }

            // Process helpful work in batches for better GPU utilization
            // Vertical timeout: Complete all GPU batch processing for the current focus neuron
            if !helpful_work_batch.is_empty() {
                // Clone samples for the GPU queue (queue takes ownership)
                let helpful_samples: Vec<Vec<HelpfulSample>> = helpful_work_batch
                    .iter()
                    .map(|w| w.samples.clone())
                    .collect();

                // Track metadata: check if any samples have target_value data
                // This is used to determine if saturation-aware simulation was possible.
                for samples in &helpful_samples {
                    if samples.iter().any(|s| s.target_value.is_some()) {
                        metadata_target_value_seen.store(true, std::sync::atomic::Ordering::Relaxed);
                        break;
                    }
                }

                let helpful_stats_batch = {
                    let _timing = TimingScope::shader(&timing_collector, "helpful");
                    gpu.evaluate_helpful_batch(helpful_samples, &deadline)?
                };

                // Process results - collect all updates first, then apply in batches (reduces mutex contention)
                let mut candidates_to_add = Vec::new();
                let mut coordinated_to_add = Vec::new();
                let mut diagnostics_zero_improvements = Vec::new();
                let mut diagnostics_below_threshold = Vec::new();
                let mut diagnostics_selected = Vec::new();

                // Issue #202: Track source contributions for epistatic pair detection
                let mut source_contributions: Vec<SourceContribution> = Vec::new();

                {
                    let _timing = TimingScope::result_processing(&timing_collector);
                    for (work, stats) in helpful_work_batch.iter().zip(helpful_stats_batch.iter()) {
                    let positive_is_better = stats.positive_count >= stats.negative_count;
                    let gpu_improved_count = if positive_is_better {
                        stats.positive_count
                    } else {
                        stats.negative_count
                    };
                    if gpu_improved_count == 0 {
                        diagnostics_zero_improvements.push((
                            work.target_uuid.clone(),
                            work.source_uuid.clone(),
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

                    // Use shared weight calculation (synapse = direct connection, so incoming_weight = 1.0)
                    // The shared function ensures consistent weight calculation across synapse and neuron analysis
                    let weight = match calculate_optimal_outgoing_weight(
                        stats.error_activation_sum,
                        stats.activation_sq_sum,
                        1.0, // Synapses are direct connections, no intermediate neuron
                    ) {
                        Some(w) => w,
                        None => continue, // Skip if weight is invalid
                    };

                    // Get target's squash function for saturation-aware improvement calculation.
                    // For saturating activations (HARD_TANH, TANH, LOGISTIC, etc.), the linear model
                    // overpredicts improvement near saturation. Using the actual activation function
                    // gives accurate predictions that match real-world results.
                    let target_squash = neuron_squash_map_arc
                        .get(&work.target_uuid)
                        .map(|s| s.as_str());

                    // Track metadata: check if saturation-aware simulation is used for this candidate.
                    // get_target_simulation_fn returns Some when:
                    // 1. The target squash is a supported saturating activation, AND
                    // 2. All samples have target_value/target_activation data
                    if get_target_simulation_fn(&work.samples, target_squash).is_some() {
                        metadata_saturation_aware_used.store(true, std::sync::atomic::Ordering::Relaxed);
                    }

                    // Compute expected improvement using the linear error model.
                    // This works for ALL squash types because we're measuring actual errors
                    // from recordings, not predicting theoretical errors. The correlation
                    // between source activation and target error determines improvement.
                    // Issue #128: This is neuron-level improvement - impact discounting converts to creature-level.
                    let baseline_error_sq = stats.error_sq_sum;

                    // IMPORTANT (3-Jan-2026):
                    // For weight updates, we must compute expected improvement using the *effective*
                    // (clamped) delta weight, not the proposed delta weight. Otherwise, expected gains
                    // are overstated and candidates are mis-prioritised.
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
                            // Issue #413: For saturating target activations (HARD_TANH, TANH,
                            // LOGISTIC, etc.), the linear-model weight can overshoot into
                            // saturation, causing inverted predictions. Search over scaled
                            // weights to find the best candidate, matching the approach used
                            // by add-neuron candidates (synapse.rs evaluate_activation_candidate).
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
                                let clamped =
                                    w.clamp(-MAX_OUTGOING_WEIGHT, MAX_OUTGOING_WEIGHT);
                                if clamped.abs() <= EPSILON {
                                    continue;
                                }
                                let (imp, improved, worsened, _) =
                                    compute_synapse_improvement_and_count(
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
                    // Collect ALL sources (including non-positive improvements) because
                    // epistatic pairs may have low individual improvements but high combined
                    if work.existing_weight.is_none() {
                        source_contributions.push(build_source_contribution(
                            &work.source_uuid,
                            work.samples.clone(),
                            stats.clone(),
                            applied_weight,
                            neuron_error_improvement,
                        ));
                    }

                    // Accept all positive improvements as candidates (not just those above threshold)
                    // Only reject if improvement is non-positive (<= 0.0)
                    if neuron_error_improvement <= 0.0 {
                        // Skip non-positive improvements
                        continue;
                    }

                    // If positive but below threshold, still accept as candidate but log for diagnostics
                    if neuron_error_improvement <= threshold {
                        diagnostics_below_threshold.push((
                            work.target_uuid.clone(),
                            work.source_uuid.clone(),
                            ThresholdContext {
                                sample_count: work.samples.len(),
                                expected_improvement: neuron_error_improvement,
                                threshold,
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
                        // Issue #180 (9-Jan-2026): Weight update using setWeight operation.
                        // Previously used remove+add pattern; now use a single setWeight op
                        // for simplicity and directness.
                        let Some((new_weight, delta_weight)) =
                            clamp_weight_update_delta(old_weight, weight)
                        else {
                            continue;
                        };
                        // NOTE: We keep the computed improvement based on the effective (clamped)
                        // delta (`delta_weight`) but apply the absolute `new_weight` in the op.
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
                        diagnostics_selected.push(work.target_uuid.clone());
                        // Issue #178 (7-Jan-2026): If the source activation is constant/near-constant,
                        // an add-synapse behaves like a bias shift on the target. Prefer `setBias`
                        // to avoid paying complexity cost for what is effectively a constant offset.
                        if let Some(threshold) = constant_source_effect_threshold {
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
                                    let old_bias = neuron_bias_map_arc
                                        .get(&work.target_uuid)
                                        .copied()
                                        .unwrap_or(0.0);
                                    let new_bias = old_bias + (applied_weight * mean_activation);
                                    if new_bias.is_finite() {
                                        coordinated_to_add.push(crate::CoordinatedStructuralCandidateJson {
                                            operations: vec![crate::CoordinatedStructuralOpJson::SetBias {
                                                neuron_uuid: work.target_uuid.clone(),
                                                bias: new_bias,
                                            }],
                                            expected_creature_score_gain: neuron_error_improvement,
                                            comment: Some(format!(
                                                "Fold constant source into setBias: old_bias={old_bias:.6}, new_bias={new_bias:.6}, weight={applied_weight:.6}, mean_act={mean_activation:.6}, act_range={activation_range:.6e}, effect_range={effect_range:.6e}"
                                            )),
                                        });
                                        continue;
                                    }
                                }
                            }
                        }

                        // Issue #128: Use creature-level metrics (impact discounting applied later)
                        // Issue #194: Compute confidence metrics for this prediction
                        let confidence_metrics = compute_confidence_metrics(
                            &work.samples,
                            neuron_error_improvement,
                            None, // R² not available for synapse candidates
                        );
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
                            outlier_reduction_info: None, // Set during outlier analysis pass if enabled (Issue #192)
                            prediction_confidence: confidence_metrics.prediction_confidence,
                            expected_score_gain_confidence_interval: confidence_metrics.expected_score_gain_confidence_interval,
                        });
                    }
                    } // End timing scope for result processing
                }

                // Apply all diagnostics updates directly (no lock needed with DashMap - Issue #216)
                for (target, source, sample_count, pos, neg) in diagnostics_zero_improvements {
                    diagnostics.record_zero_improvement(&target, &source, sample_count, pos, neg);
                }
                for (target, source, context) in diagnostics_below_threshold {
                    diagnostics.record_below_threshold(&target, &source, context);
                }
                for target in diagnostics_selected {
                    diagnostics.mark_candidate_selected(&target);
                }
                if !candidates_to_add.is_empty() {
                    let mut results = helpful_results
                        .lock()
                        .expect("Mutex poisoned: helpful_results");
                    results.extend(candidates_to_add);
                }
                if !coordinated_to_add.is_empty() {
                    let mut results = coordinated_structural_results
                        .lock()
                        .expect("Mutex poisoned: coordinated_structural_results");
                    results.extend(coordinated_to_add);
                }

                // Issue #202: Detect epistatic neuron pairs for this target
                // Epistatic pairs are sources where neither improves individually, but
                // both together could improve the target due to complementary patterns.
                if verbose_enabled() {
                    eprintln!(
                        "[NEAT-AI-Discovery][verbose] Target {target_uuid}: collected {} source contributions for epistatic detection",
                        source_contributions.len()
                    );
                }
                if source_contributions.len() >= 2 {
                    // Get target neuron impact for discounting
                    let target_is_output = neuron_type_map_arc
                        .get(target_uuid.as_str())
                        .map(|t| t == "output")
                        .unwrap_or(false);
                    let target_impact = if target_is_output {
                        1.0
                    } else {
                        // Use a default impact for hidden neurons (will be recalculated later)
                        0.5
                    };

                    let epistatic_pairs = detect_epistatic_pairs(
                        target_uuid.as_str(),
                        &source_contributions,
                        target_impact,
                    );

                    if !epistatic_pairs.is_empty() {
                        // Issue #415: Filter out interfering pairs before converting to candidates
                        let filtered_pairs =
                            filter_interfering_epistatic_pairs(epistatic_pairs, &source_contributions);

                        if !filtered_pairs.is_empty() {
                            let epistatic_candidates =
                                epistatic_pairs_to_coordinated_candidates(&filtered_pairs);
                            if !epistatic_candidates.is_empty() {
                                let mut results = coordinated_structural_results
                                    .lock()
                                    .expect("Mutex poisoned: coordinated_structural_results");
                                results.extend(epistatic_candidates);

                                if verbose_enabled() {
                                    eprintln!(
                                        "[NEAT-AI-Discovery][verbose] Target {target_uuid}: found {} epistatic pair(s)",
                                        filtered_pairs.len()
                                    );
                                }
                            }
                        }
                    }

                    // Issue #189: Detect synergistic candidates via residual analysis
                    // This detects XOR-like patterns where:
                    // - Neither source alone provides strong improvement
                    // - Together they reduce error better than either alone
                    let synergistic_candidates = detect_synergistic_candidates(
                        target_uuid.as_str(),
                        &source_contributions,
                        target_impact,
                    );

                    if !synergistic_candidates.is_empty() {
                        // Issue #415: Filter out interfering candidates before converting
                        let filtered_synergistic = filter_interfering_synergistic_candidates(
                            synergistic_candidates,
                            &source_contributions,
                        );

                        if !filtered_synergistic.is_empty() {
                            let synergistic_coordinated =
                                synergistic_to_coordinated_candidates(&filtered_synergistic);
                            if !synergistic_coordinated.is_empty() {
                                let mut results = coordinated_structural_results
                                    .lock()
                                    .expect("Mutex poisoned: coordinated_structural_results");
                                results.extend(synergistic_coordinated);

                                if verbose_enabled() {
                                    eprintln!(
                                        "[NEAT-AI-Discovery][verbose] Target {target_uuid}: found {} synergistic candidate(s)",
                                        filtered_synergistic.len()
                                    );
                                }
                            }
                        }
                    }
                }

                // Issue #164: Detect redundant paths feeding the same target.
                // Two existing synapses with highly correlated activations are redundant –
                // prune the weaker path and renormalise the survivor's weight.
                if existing_path_contributions.len() >= 2 {
                    let target_is_output = neuron_type_map_arc
                        .get(target_uuid.as_str())
                        .map(|t| t == "output")
                        .unwrap_or(false);
                    let target_impact = if target_is_output {
                        1.0
                    } else {
                        0.5
                    };

                    let redundant_paths = detect_redundant_paths(
                        target_uuid.as_str(),
                        &existing_path_contributions,
                        target_impact,
                    );

                    if !redundant_paths.is_empty() {
                        let redundant_coordinated =
                            redundant_paths_to_coordinated_candidates(&redundant_paths);
                        if !redundant_coordinated.is_empty() {
                            let mut results = coordinated_structural_results
                                .lock()
                                .expect("Mutex poisoned: coordinated_structural_results");
                            results.extend(redundant_coordinated);

                            if verbose_enabled() {
                                eprintln!(
                                    "[NEAT-AI-Discovery][verbose] Target {target_uuid}: found {} redundant path(s) for pruning",
                                    redundant_paths.len()
                                );
                            }
                        }
                    }
                }
            }

            // Process harmful synapses for this target - BATCHED GPU evaluation for better utilisation
            // Vertical timeout: Complete all harmful synapse processing for the current focus neuron
            if !*analysis_timed_out.lock().expect("Mutex poisoned") {
                // Issue #210: Use interned index for synapses_by_target lookup
                let target_idx_for_harmful = neuron_index_arc.get_index(target_uuid.as_str());
                if let Some(existing) = target_idx_for_harmful.and_then(|idx| synapses_by_target_arc.get(&idx)) {
                    // Phase 1: Build all samples on CPU (fast, parallel-friendly)
                    // Reuse the target_map we already built for helpful synapse processing
                    struct HarmfulWork {
                        synapse: SynapseJson,
                        samples: Vec<HelpfulSample>,
                    }
                    let mut harmful_work: Vec<HarmfulWork> = Vec::with_capacity(existing.len());

                    for synapse in existing {
                        let from_records_arc = match cache.get(&synapse.from_uuid) {
                            Ok(records) => records,
                            Err(_) => continue,
                        };
                        if from_records_arc.is_empty() {
                            continue;
                        }
                        let from_records = from_records_arc.as_ref();
                        // Use pre-built target map - avoids rebuilding HashMap for each synapse
                        let samples = target_map_ref.build_samples_from(from_records);
                        if samples.is_empty() {
                            continue;
                        }

                        harmful_work.push(HarmfulWork {
                            synapse: synapse.clone(),
                            samples,
                        });
                    }

                    // Phase 2: Batch GPU evaluation (single submission for all synapses)
                    if !harmful_work.is_empty() {
                        // Clone samples for the GPU queue (queue takes ownership)
                        let batch_input: Vec<(Vec<HelpfulSample>, f32)> = harmful_work
                            .iter()
                            .map(|w| (w.samples.clone(), w.synapse.weight))
                            .collect();

                        let batch_stats = {
                            let _timing = TimingScope::shader(&timing_collector, "harmful");
                            gpu.evaluate_harmful_batch(batch_input, &deadline)?
                        };

                        // Phase 3: Process results
                        let mut harmful_candidates = Vec::with_capacity(batch_stats.len());
                        let target_stats = cache
                            .get(target_uuid.as_str())
                            .ok()
                            .and_then(|records| NeuronStats::from_records(records.as_ref()))
                            .map(|s| s.to_json());

                        for (work, stats) in harmful_work.iter().zip(batch_stats.iter()) {
                            let total_count = work.samples.len() as u32;
                            if total_count == 0 {
                                continue;
                            }

                            // Issue #128: This is neuron-level - impact discounting converts to creature-level
                            let neuron_error_improvement = (stats.harmful_count as f32
                                - stats.helpful_count as f32)
                                / total_count as f32;

                            // Issue #416: Only include candidates where removing the synapse
                            // would improve the score (positive expected_creature_score_gain).
                            // Candidates with non-positive gain are synapses that are actually
                            // helpful - they should NOT be in harmful_synapses.
                            if neuron_error_improvement <= 0.0 {
                                continue;
                            }

                            // Issue #128: Use creature-level metrics (impact discounting applied later)
                            // Issue #194: Compute confidence metrics for this prediction
                            let confidence_metrics = compute_confidence_metrics(
                                &work.samples,
                                neuron_error_improvement,
                                None, // R² not available for synapse candidates
                            );
                            harmful_candidates.push(CandidateSynapseJson {
                                from_neuron_uuid: work.synapse.from_uuid.clone(),
                                to_neuron_uuid: work.synapse.to_uuid.clone(),
                                from_neuron_index: None,
                                to_neuron_index: None,
                                weight: work.synapse.weight,
                                target_neuron_impact: 1.0,
                                expected_creature_error_reduction: neuron_error_improvement,
                                expected_creature_score_gain: neuron_error_improvement,
                                improved_count: stats.harmful_count,
                                total_count,
                                target_neuron_stats: target_stats.clone(),
                                outlier_reduction_info: None, // Set during outlier analysis pass if enabled (Issue #192)
                                prediction_confidence: confidence_metrics.prediction_confidence,
                                expected_score_gain_confidence_interval: confidence_metrics.expected_score_gain_confidence_interval,
                            });
                        }

                        // Batch push all harmful candidates (single lock)
                        if !harmful_candidates.is_empty() {
                            let mut results = harmful_results
                                .lock()
                                .expect("Mutex poisoned: harmful_results");
                            results.extend(harmful_candidates);
                        }
                    }
                }
            }

            // Track completion of this focus neuron for timeout reporting.
            let completed =
                completed_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            crate::watchdog::beat(format!(
                "synapse analysis → completed {completed}/{total_focus_count} (last target {target_uuid})"
            ));
            Ok(())
        })?;

    let analysis_timed_out = *analysis_timed_out.lock().expect("Mutex poisoned");
    let mut helpful_results = helpful_results.lock().expect("Mutex poisoned").clone();
    let mut harmful_results = harmful_results.lock().expect("Mutex poisoned").clone();
    let mut coordinated_structural_results = coordinated_structural_results
        .lock()
        .expect("Mutex poisoned")
        .clone();
    let mut helpful_fallback = helpful_fallback.lock().expect("Mutex poisoned").take();

    // Log timeout with completion stats (always visible, not just verbose)
    if analysis_timed_out {
        let completed = completed_count.load(std::sync::atomic::Ordering::Relaxed);
        log_analysis_timeout("synapse", completed, total_focus_count);
    }

    if helpful_results.is_empty()
        && let Some(candidate) = helpful_fallback.take()
    {
        // Issue #216: Direct method call - no lock needed with DashMap-based diagnostics
        diagnostics.mark_candidate_selected(&candidate.to_neuron_uuid);
        helpful_results.push(candidate);
    }

    // Coordinated structural discovery (7-Jan-2026): collapse a simple hidden neuron into a single synapse.
    //
    // This is the "reverse" of synapse→neuron insertion: when a hidden neuron forms a simple 1-in/1-out
    // chain (a → h → b), we can propose removing that neuron and replacing the chain with a direct
    // synapse (a → b). This must be applied atomically, so it is emitted as a coordinated-structural
    // candidate group.
    //
    // Initial scope: only hidden neurons with exactly one incoming and one outgoing synapse.
    // This keeps the candidate safe and deterministic; broader graph rewrites can be added later.
    {
        // Build incoming/outgoing synapse lists per neuron.
        let mut incoming: HashMap<String, Vec<SynapseJson>> = HashMap::new();
        let mut outgoing: HashMap<String, Vec<SynapseJson>> = HashMap::new();
        for s in &input.creature.synapses {
            incoming
                .entry(s.to_uuid.clone())
                .or_default()
                .push(s.clone());
            outgoing
                .entry(s.from_uuid.clone())
                .or_default()
                .push(s.clone());
        }

        // Quick neuron-type lookup (only creature.neurons; inputs are not here).
        let neuron_type_map_local: HashMap<String, String> = input
            .creature
            .neurons
            .iter()
            .map(|n| (n.uuid.clone(), n.neuron_type.clone()))
            .collect();

        // Precompute existing direct synapses so we don't propose duplicates.
        let mut existing_edges: HashSet<(String, String)> = HashSet::new();
        for s in &input.creature.synapses {
            existing_edges.insert((s.from_uuid.clone(), s.to_uuid.clone()));
        }

        for neuron in &input.creature.neurons {
            if neuron.neuron_type != "hidden" {
                continue;
            }
            let h = neuron.uuid.as_str();
            let Some(ins) = incoming.get(h) else { continue };
            let Some(outs) = outgoing.get(h) else {
                continue;
            };
            if ins.len() != 1 || outs.len() != 1 {
                continue;
            }

            let a_syn = &ins[0];
            let b_syn = &outs[0];
            let a = a_syn.from_uuid.as_str();
            let b = b_syn.to_uuid.as_str();

            // Skip degenerate / non-actionable cases.
            if a == b || a == h || b == h {
                continue;
            }
            if existing_edges.contains(&(a.to_string(), b.to_string())) {
                // A direct synapse already exists; collapsing would need additional ops (future work).
                continue;
            }

            // Ensure the target exists (either input-* or a neuron) so the op is not stale.
            let target_is_known = b.starts_with("input-") || neuron_type_map_local.contains_key(b);
            if !target_is_known {
                continue;
            }

            // Build samples: correlate a's activation to b's adjusted error after removing h→b.
            let Ok(a_records) = cache.get(a) else {
                continue;
            };
            let Ok(h_records) = cache.get(h) else {
                continue;
            };
            let Ok(b_records) = cache.get(b) else {
                continue;
            };
            if a_records.is_empty() || h_records.is_empty() || b_records.is_empty() {
                continue;
            }

            let target_map_b = TargetMap::from_records(b_records.as_ref());
            if target_map_b.map.is_empty() {
                continue;
            }
            let build_act_map = |records: &[DiscoverRecord]| -> HashMap<u32, f32> {
                let mut map: HashMap<u32, f32> = HashMap::with_capacity(records.len());
                for r in records {
                    if r.activation.is_finite() {
                        map.insert(r.obs_index, r.activation);
                    }
                }
                map
            };
            let a_map = build_act_map(a_records.as_ref());
            let h_map = build_act_map(h_records.as_ref());

            let mut samples: Vec<HelpfulSample> = Vec::with_capacity(target_map_b.map.len());
            for (obs_index, target) in target_map_b.map.iter() {
                let Some(a_act) = a_map.get(obs_index) else {
                    continue;
                };
                let Some(h_act) = h_map.get(obs_index) else {
                    continue;
                };
                if !a_act.is_finite() || !h_act.is_finite() || !target.avg_error.is_finite() {
                    continue;
                }
                let adjusted_error = target.avg_error + b_syn.weight * (*h_act);
                if !adjusted_error.is_finite() {
                    continue;
                }
                samples.push(HelpfulSample {
                    activation: *a_act,
                    avg_error: adjusted_error,
                    target_value: None,
                    target_activation: None,
                });
            }

            if samples.len() < MIN_NEURON_SAMPLE_COUNT {
                continue;
            }

            let mut sum_act_sq = 0.0f32;
            let mut sum_err_act = 0.0f32;
            let mut baseline_sq = 0.0f32;
            for s in &samples {
                sum_act_sq += s.activation * s.activation;
                sum_err_act += s.activation * s.avg_error;
                baseline_sq += s.avg_error * s.avg_error;
            }
            if baseline_sq <= EPSILON {
                continue;
            }

            let Some(weight) = calculate_optimal_outgoing_weight(sum_err_act, sum_act_sq, 1.0)
            else {
                continue;
            };

            let (improvement, improved, worsened, total) =
                compute_synapse_improvement_and_count(&samples, weight, baseline_sq, None);
            let _ = (improved, worsened, total);
            if improvement <= 0.0 {
                continue;
            }

            coordinated_structural_results.push(crate::CoordinatedStructuralCandidateJson {
                operations: vec![
                    crate::CoordinatedStructuralOpJson::RemoveSynapse {
                        from_neuron_uuid: a.to_string(),
                        to_neuron_uuid: h.to_string(),
                    },
                    crate::CoordinatedStructuralOpJson::RemoveSynapse {
                        from_neuron_uuid: h.to_string(),
                        to_neuron_uuid: b.to_string(),
                    },
                    crate::CoordinatedStructuralOpJson::RemoveNeuron {
                        neuron_uuid: h.to_string(),
                    },
                    crate::CoordinatedStructuralOpJson::AddSynapse {
                        from_neuron_uuid: a.to_string(),
                        to_neuron_uuid: b.to_string(),
                        weight,
                    },
                ],
                expected_creature_score_gain: improvement,
                comment: Some(
                    "Coordinated collapse: remove 1-in/1-out hidden neuron and add bypass synapse"
                        .to_string(),
                ),
            });
        }
    }

    // Issue #128: Apply impact-based discounting and set creature-level metrics.
    // Output neurons have impact = 1.0 (no discount).
    // Hidden neurons have impact in [0, 1] based on their weighted paths to outputs.
    let impact_scores = compute_impact_scores_for_discounting(&input.creature, cache.as_ref());
    let neuron_type_map: HashMap<String, String> = input
        .creature
        .neurons
        .iter()
        .map(|n| (n.uuid.clone(), n.neuron_type.clone()))
        .collect();

    // Apply impact discounting to helpful synapse candidates
    for candidate in &mut helpful_results {
        // Populate indices for debugging/analysis (consistent with CandidateNeuronJson).
        candidate.from_neuron_index = order_map_arc.get(&candidate.from_neuron_uuid).copied();
        candidate.to_neuron_index = order_map_arc.get(&candidate.to_neuron_uuid).copied();

        let is_hidden = neuron_type_map
            .get(&candidate.to_neuron_uuid)
            .map(|t| t != "output")
            .unwrap_or(true); // Default to hidden if type unknown

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
        candidate.expected_creature_error_reduction *= impact;
        candidate.expected_creature_score_gain = candidate.expected_creature_error_reduction;

        // Issue #467: Apply source-type prioritisation boost for input-neuron sources.
        // Input neurons have a 36.2% success rate vs 2.8–3.3% for hidden neurons.
        candidate.expected_creature_score_gain = apply_source_type_boost(
            candidate.expected_creature_score_gain,
            &candidate.from_neuron_uuid,
        );

        // Issue #468: Apply target-type prioritisation boost for existing hidden targets.
        // Existing hidden neurons as targets have a 31.4% success rate vs 5.3–5.4%
        // for output or discovery-hidden neurons.
        candidate.expected_creature_score_gain = apply_target_type_boost(
            candidate.expected_creature_score_gain,
            &candidate.to_neuron_uuid,
            &neuron_type_map,
        );

        if verbose_enabled() && is_hidden {
            eprintln!(
                "[NEAT-AI-Discovery][verbose] Synapse candidate → {} impact {:.3}: \
                {:.4}% → {:.4}%",
                &candidate.to_neuron_uuid[..12.min(candidate.to_neuron_uuid.len())],
                impact,
                original * 100.0,
                candidate.expected_creature_score_gain * 100.0
            );
        }
    }

    // Apply impact discounting to harmful synapse candidates (same logic)
    for candidate in &mut harmful_results {
        // Populate indices for debugging/analysis (consistent with CandidateNeuronJson).
        candidate.from_neuron_index = order_map_arc.get(&candidate.from_neuron_uuid).copied();
        candidate.to_neuron_index = order_map_arc.get(&candidate.to_neuron_uuid).copied();

        let is_hidden = neuron_type_map
            .get(&candidate.to_neuron_uuid)
            .map(|t| t != "output")
            .unwrap_or(true);

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
    }

    // Apply impact discounting to coordinated candidates (based on target neuron UUID).
    for candidate in &mut coordinated_structural_results {
        // Heuristic: use the last op that targets a concrete neuron, so multi-op groups
        // (remove+add synapses, add/remove neuron, etc.) get discounted by the final target.
        // This aligns with coordinated groups that ultimately adjust the inputs of a target neuron.
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
                crate::CoordinatedStructuralOpJson::SetBias { neuron_uuid, .. } => {
                    neuron_uuid.as_str()
                }
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
            .map(|t| t != "output")
            .unwrap_or(true);
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
    }

    // Note: helpful_fallback does NOT need separate discounting here.
    // If helpful_results was empty, the fallback was already moved into it via .take()
    // and gets discounted in the loop above. If helpful_results was NOT
    // empty, the fallback is intentionally not returned (we have better candidates).

    helpful_results.sort_by(|a, b| {
        // Issue #128: Sort by expected creature score gain (highest first)
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

    // Deadline coverage (Jan 2026): diversify within the top-K so repeated runs explore different
    // high-quality candidates over time (helps with failure caches and avoids category starvation).
    if input.analysis_deadline_ms.is_some() {
        use super::constants::DIVERSIFY_TOP_K;
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

    // Track candidates_found before truncation for metadata
    let candidates_found =
        helpful_results.len() + harmful_results.len() + coordinated_structural_results.len();

    if let Some(limit) = input.max_candidates {
        let (h1, h2, c) = truncate_combined_synapse_candidate_sets(
            std::mem::take(&mut helpful_results),
            std::mem::take(&mut harmful_results),
            std::mem::take(&mut coordinated_structural_results),
            limit,
            input.analysis_deadline_ms.is_some(),
        );
        helpful_results = h1;
        harmful_results = h2;
        coordinated_structural_results = c;
    }

    // Track candidates_returned after truncation
    let candidates_returned =
        helpful_results.len() + harmful_results.len() + coordinated_structural_results.len();

    let no_candidate_reasons = diagnostics.no_candidate_summaries();
    diagnostics.emit_logs();

    // Build metadata for observability (v0.2.17+)
    // Note: target_value_available and saturation_aware_simulation_used are tracked
    // during the inner analysis loop via atomic flags.
    let saw_any_input =
        metadata_seen_any_input_with_records.load(std::sync::atomic::Ordering::Relaxed);
    let input_min = metadata_input_min_with_records.load(std::sync::atomic::Ordering::Relaxed);
    let input_max = metadata_input_max_with_records.load(std::sync::atomic::Ordering::Relaxed);

    // Issue #192: Compute error distribution from collected error values
    let error_distribution = {
        let error_vec = error_values_for_distribution
            .lock()
            .expect("Mutex poisoned: error_values_for_distribution");
        super::error_distribution::ErrorDistribution::from_errors(&error_vec)
    };

    let metadata = super::shared::SynapseAnalysisMetadata {
        target_value_available: metadata_target_value_seen
            .load(std::sync::atomic::Ordering::Relaxed),
        saturation_aware_simulation_used: metadata_saturation_aware_used
            .load(std::sync::atomic::Ordering::Relaxed),
        candidates_found,
        candidates_returned,
        timed_out: analysis_timed_out,
        completed_focus_neurons: completed_count.load(std::sync::atomic::Ordering::Relaxed),
        total_focus_neurons: total_focus_count,
        input_index_min_seen_with_records: if saw_any_input { Some(input_min) } else { None },
        input_index_max_seen_with_records: if saw_any_input { Some(input_max) } else { None },
        timing: timing_collector.finalize(),
        gpu_info: GpuAnalyzer::get_adapter_info(),
        error_distribution,
    };

    Ok(AnalyzeSynapsesResult {
        helpful_synapses: helpful_results,
        harmful_synapses: harmful_results,
        // Weight updates are represented as remove+add coordinated candidates for KISS.
        // This keeps NEAT-AI's apply/ablate pipeline limited to existing structural ops.
        synapse_weight_updates: Vec::new(),
        coordinated_structural_candidates: coordinated_structural_results,
        candidate_clusters: Vec::new(), // populated by analyze_all post-processing (Issue #224)
        gpu_used,
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
    use super::*;

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
}

// Issue #425: Implementation tests moved from implementation.rs as part of the refactoring.
// These test the core synapse analysis pipeline (GPU batch evaluation, diagnostics, etc.).
#[cfg(test)]
#[path = "implementation_tests/mod.rs"]
mod implementation_tests;
