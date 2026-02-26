//! GPU-accelerated candidate evaluation
//!
//! This module contains ReLU and activation candidate evaluation functions
//! that use GPU compute shaders for efficient batch processing.

use crate::CandidateNeuronJson;
use crate::analysis::activation::{
    ACTIVATION_SPECS, ActivationCandidateSpec, activation_name_to_gpu_id, get_target_simulation_fn,
    has_sufficient_output_variance,
};
use crate::analysis::gpu::GpuEvaluator;
use crate::analysis::samples::{EPSILON, HelpfulSample, NeuronStats};
use crate::analysis::scoring::confidence::compute_confidence_metrics;
use crate::analysis::scoring::weights::{
    MAX_OUTGOING_WEIGHT, calculate_optimal_bias, calculate_optimal_identity_outgoing_and_bias,
    calculate_optimal_outgoing_weight,
};
use anyhow::Result;

use super::scoring::{
    compute_activation_improvement_and_count, compute_relu_improvement_and_count,
};

// MIN_NEURON_SAMPLE_COUNT moved to constants.rs (Issue #424)
use crate::analysis::constants::MIN_NEURON_SAMPLE_COUNT;

// =============================================================================
// ReLU Candidate Evaluation
// =============================================================================

/// Result from ReLU evaluation (split by target error sign)
pub(crate) struct SplitReluResult {
    /// Candidate for samples with positive error (output should be higher)
    pub(crate) positive_error_candidate: Option<CandidateNeuronJson>,
    /// Candidate for samples with negative error (output should be lower)
    pub(crate) negative_error_candidate: Option<CandidateNeuronJson>,
}

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

/// Parameters for evaluating an activation function candidate from an error
/// subset, computing net improvement across all samples.
pub(crate) struct SubsetEvalParams<'a> {
    pub source_uuid: &'a str,
    pub target_uuid: &'a str,
    pub subset_samples: &'a [HelpfulSample],
    pub all_samples: &'a [HelpfulSample],
    pub spec: &'a ActivationCandidateSpec,
    pub target_squash: Option<&'a str>,
    pub total_baseline_error_sq: f32,
    pub target_activation_fn: Option<fn(f32) -> f32>,
}

/// Helper for split-error evaluation: compute optimal weight from subset, evaluate on all samples.
///
/// This is the core of the split-error fix for non-ReLU activations. By computing
/// the optimal weight from a specific error subset (positive or negative), we get
/// a weight that's tuned to help that subset. We then evaluate the NET improvement
/// across ALL samples to ensure the candidate doesn't hurt the other subset more
/// than it helps the target subset.
pub(crate) fn evaluate_activation_for_subset<G: GpuEvaluator>(
    gpu: &G,
    params: &SubsetEvalParams<'_>,
) -> Result<Option<CandidateNeuronJson>> {
    if params.subset_samples.len() < MIN_NEURON_SAMPLE_COUNT {
        return Ok(None);
    }

    let spec = params.spec;
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
                params.subset_samples,
                activation_type,
                orientation,
                scale,
            ) {
                Ok(result) => (result.0, result.1),
                Err(_) => {
                    // Fall back to CPU on GPU error
                    let mut sum_act_sq = 0.0;
                    let mut sum_err_act = 0.0;
                    for sample in params.subset_samples {
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
                match calculate_optimal_identity_outgoing_and_bias(
                    params.subset_samples,
                    incoming_weight,
                ) {
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
                    params.subset_samples,
                    incoming_weight,
                    outgoing_weight,
                    spec.activation,
                    spec.name,
                    None,
                    params.target_squash,
                );
                (outgoing_weight, optimal_bias)
            };

            // Issue #123: Check for saturation - reject if neuron output is nearly constant.
            if !has_sufficient_output_variance(
                params.all_samples,
                incoming_weight,
                optimal_bias,
                spec.activation,
            ) {
                continue;
            }

            // CRITICAL: Evaluate NET improvement across ALL samples
            let (net_improvement, improved_count, total_count) =
                compute_activation_improvement_and_count(
                    params.all_samples,
                    incoming_weight,
                    outgoing_weight,
                    optimal_bias,
                    spec.activation,
                    params.total_baseline_error_sq,
                    params.target_activation_fn,
                );

            // Only consider candidates with positive NET improvement
            if net_improvement <= 0.0 {
                continue;
            }

            // Apply validity filters
            let absolute_improvement = net_improvement * params.total_baseline_error_sq;
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

                let target_stats =
                    NeuronStats::from_samples(params.all_samples).map(|s| s.to_json());
                // Issue #128: Use creature-level metrics
                // Issue #194: Compute confidence metrics for this prediction
                let confidence_metrics = compute_confidence_metrics(
                    params.all_samples,
                    net_improvement,
                    None, // R² not available for neuron candidates
                );
                best_candidate = Some(CandidateNeuronJson {
                    source_neuron_uuid: params.source_uuid.to_string(),
                    target_neuron_uuid: params.target_uuid.to_string(),
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

/// Parameters for evaluating activation candidates for a source-target pair.
pub(crate) struct ActivationEvalParams<'a> {
    pub source_uuid: &'a str,
    pub target_uuid: &'a str,
    pub samples: &'a [HelpfulSample],
    pub threshold: f32,
    pub spec: &'a ActivationCandidateSpec,
    pub target_squash: Option<&'a str>,
}

/// Evaluate activation candidates for a given source-target pair.
///
/// This function handles both split-error evaluation and fallback to all-samples
/// evaluation when appropriate.
pub(crate) fn evaluate_activation_candidate<G: GpuEvaluator>(
    gpu: &G,
    params: &ActivationEvalParams<'_>,
) -> Result<Option<CandidateNeuronJson>> {
    let samples = params.samples;
    let spec = params.spec;

    if samples.len() < MIN_NEURON_SAMPLE_COUNT {
        return Ok(None);
    }

    let activation_type = activation_name_to_gpu_id(spec.name);

    let mut best_candidate: Option<CandidateNeuronJson> = None;
    let mut best_score = params.threshold;
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
    let target_activation_fn = get_target_simulation_fn(samples, params.target_squash);

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

        let subset_params = SubsetEvalParams {
            source_uuid: params.source_uuid,
            target_uuid: params.target_uuid,
            subset_samples: error_samples,
            all_samples: samples,
            spec,
            target_squash: params.target_squash,
            total_baseline_error_sq,
            target_activation_fn,
        };
        if let Some(candidate) = evaluate_activation_for_subset(gpu, &subset_params)? {
            let gain = candidate.expected_creature_score_gain;
            // Track best (above threshold) and fallback (above 0) candidates.
            if gain > best_score {
                best_score = gain;
                best_candidate = Some(candidate);
            } else if gain > fallback_score && gain > 0.0 {
                fallback_score = gain;
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
            let target_activation_fn = get_target_simulation_fn(samples, params.target_squash);
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
                            params.target_squash,
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
                        params.target_squash,
                    );

                    // CRITICAL FIX: Recompute optimal weight WITH the bias included.
                    let mut sum_activation_sq_with_bias = 0.0f32;
                    let mut sum_error_activation_with_bias = 0.0f32;
                    for sample in samples {
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
                    source_neuron_uuid: params.source_uuid.to_string(),
                    target_neuron_uuid: params.target_uuid.to_string(),
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
                    source_neuron_uuid: params.source_uuid.to_string(),
                    target_neuron_uuid: params.target_uuid.to_string(),
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
// Batched Activation Evaluation (Issue #201)
// =============================================================================

/// Evaluate all activation specs for a (source, target) pair using batched GPU evaluation.
///
/// Issue #201: This function reduces GPU round-trips by 10-20% by evaluating multiple
/// activation function configurations in a single GPU command buffer submission.
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
    for spec in &ACTIVATION_SPECS {
        let eval_params = ActivationEvalParams {
            source_uuid,
            target_uuid,
            samples,
            threshold,
            spec,
            target_squash,
        };
        if let Some(candidate) = evaluate_activation_candidate(gpu, &eval_params)? {
            results.push(candidate);
        }
    }
    Ok(results)
}
