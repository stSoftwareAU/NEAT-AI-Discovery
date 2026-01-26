//! Synapse analysis module
//!
//! This module contains functions for analysing synapse candidates - identifying
//! beneficial new synapses that would reduce error.
//!
//! **Extracted from implementation.rs as part of Issue #275**
//!
//! Note: Some functions are extracted here but still have copies in implementation.rs
//! until the full migration is complete. These will be used when implementation.rs
//! is deprecated and neuron analysis is also extracted.
//!
//! ## Key Functions
//!
//! - `analyze_synapses` - Public entry point for synapse analysis
//! - `analyze_synapses_with_cache` - Internal implementation with shared cache
//! - `compute_synapse_improvement_and_count` - Core improvement calculation
//!
//! ## Sample Locality Optimisation (Issue #221)
//!
//! When analysing multiple source neurons for the same target, sources that share
//! the same obs_indices can benefit from batched sample building. This reduces
//! sample building overhead by up to 100x for creatures where input neurons share
//! the same observation indices.

// Allow dead code warnings for functions that are extracted here but still have
// copies in implementation.rs until the full migration is complete.
#![allow(dead_code)]

use crate::types::DiscoverRecord;
use crate::{AnalyzeSynapsesInput, CandidateNeuronJson, CandidateSynapseJson};
use anyhow::Result;

// Import shared types from the new module structure
use crate::analysis::shared::AnalyzeSynapsesResult;

// Import activation functions from the dedicated activation module (Issue #266, #238)
use crate::analysis::activation::{
    absolute_activation, activation_name_to_gpu_id, arctan_activation, bent_identity_activation,
    bipolar_activation, clipped_activation, elu_activation, gelu_activation,
    get_target_simulation_fn, get_target_simulation_mode, hard_tanh_activation,
    has_sufficient_output_variance, identity_activation, logistic_activation, mish_activation,
    relu6_activation, softplus_activation, softsign_activation, tanh_activation,
    ActivationCandidateSpec, TargetSimulationMode,
};

// Import deadline handling and logging utilities from dedicated module (Issue #268)
use crate::analysis::utils::OrderedNeuron;

// Import sample data structures from dedicated module (Issue #269)
use crate::analysis::samples::{HelpfulSample, NeuronStats, EPSILON};

// Import weight calculation functions from dedicated module (Issue #270)
use crate::analysis::weights::{
    calculate_optimal_bias, calculate_optimal_identity_outgoing_and_bias,
    calculate_optimal_outgoing_weight, MAX_OUTGOING_WEIGHT,
};

// Import diagnostics and rejection tracking from dedicated module (Issue #271)
use crate::analysis::diagnostics::{require_unique_focus, TargetMap};

// Import GPU infrastructure from dedicated modules (Issue #272, #273, #274)
use crate::analysis::gpu::{GpuEvaluator, GpuWorkQueue};

// Import RecordCache from cache module (Issue #185)
use super::cache::RecordCache;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

// =============================================================================
// Constants
// =============================================================================

const MIN_NEURON_SAMPLE_COUNT: usize = 10;

// Note: MIN_NEURON_OUTPUT_STD_DEV has been moved to the activation module
// as part of Issue #238. It is used by has_sufficient_output_variance.

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
                });
            }

            // Track fallback (positive improvement but below threshold)
            if neuron_error_improvement > fallback_score && neuron_error_improvement > 0.0 {
                fallback_score = neuron_error_improvement;

                let target_stats = NeuronStats::from_samples(samples).map(|s| s.to_json());
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

    combined.sort_by(|a, b| {
        score(b)
            .partial_cmp(&score(a))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
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
    // Delegate to implementation.rs for now - the main function is too large to move in one go
    // This will be fully migrated in a follow-up
    super::implementation::analyze_synapses_with_cache_impl(input, cache, gpu_queue)
}

/// Test-only helper for benchmarks that need to reuse GPU queue.
/// This allows benchmarks to avoid GPU initialization overhead across iterations.
///
/// Note: This is public for use in external test files (tests/ directory).
pub fn analyze_synapses_with_cache_and_gpu_queue(
    input: &AnalyzeSynapsesInput,
    cache: Arc<RecordCache>,
    gpu_queue: Arc<super::gpu::GpuWorkQueue>,
) -> Result<AnalyzeSynapsesResult> {
    super::implementation::analyze_synapses_with_cache_impl(input, cache, gpu_queue)
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
}
