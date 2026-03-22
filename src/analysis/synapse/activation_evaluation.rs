//! Activation function candidate evaluation
//!
//! Evaluates non-ReLU activation candidates for source-target pairs.
//! Handles split-error evaluation (computing optimal weights from positive/negative
//! error subsets) and falls back to all-samples evaluation when needed.

#![allow(clippy::cast_possible_truncation)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use crate::CandidateNeuronJson;
use crate::analysis::activation::{
    ActivationCandidateSpec, activation_name_to_gpu_id, get_target_simulation_fn,
    has_sufficient_output_variance,
};
use crate::analysis::gpu::GpuEvaluator;
use crate::analysis::samples::{EPSILON, HelpfulSample, NeuronStats};
use crate::analysis::scoring::confidence::compute_confidence_metrics;
use crate::analysis::scoring::weights::{
    calculate_activation_aware_outgoing_weight, calculate_optimal_bias,
    calculate_optimal_identity_outgoing_and_bias, max_outgoing_weight_for_activation,
};
use anyhow::Result;

use super::activation_subset_evaluation::{SubsetEvalParams, evaluate_activation_for_subset};
use super::scoring::compute_activation_improvement_and_count;

use crate::analysis::constants::MIN_NEURON_SAMPLE_COUNT;

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
                    // Issue #905: Use activation-aware weight calculation
                    let base_weight = match calculate_activation_aware_outgoing_weight(
                        sum_error_activation,
                        sum_activation_sq,
                        incoming_weight,
                        spec.name,
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

                    // Issue #893: Hold-out validation to combat overfitting from the
                    // 9-variant search. Select weight on training samples, report
                    // improvement on held-out validation samples.
                    use crate::analysis::synapse::holdout_validation::{
                        baseline_error_sq as compute_baseline, collect_samples,
                        split_samples_holdout,
                    };

                    if let Some(split) =
                        split_samples_holdout(samples, params.source_uuid, params.target_uuid)
                    {
                        // Phase 1: Select best weight+bias using training samples only
                        let train_samples = collect_samples(&split.train);
                        let train_baseline = compute_baseline(&split.train);

                        let mut best_weight = base_weight;
                        let mut best_bias = 0.0f32;
                        let mut best_train_improvement = f32::NEG_INFINITY;

                        let max_out = max_outgoing_weight_for_activation(spec.name);
                        for &weight in &weight_candidates {
                            let clamped_weight = weight.clamp(-max_out, max_out);
                            if clamped_weight.abs() <= EPSILON {
                                continue;
                            }

                            let bias = calculate_optimal_bias(
                                &train_samples,
                                incoming_weight,
                                clamped_weight,
                                spec.activation,
                                spec.name,
                                None,
                                params.target_squash,
                            );

                            let (improvement, _, _) = compute_activation_improvement_and_count(
                                &train_samples,
                                incoming_weight,
                                clamped_weight,
                                bias,
                                spec.activation,
                                train_baseline,
                                target_activation_fn,
                            );

                            if improvement > best_train_improvement {
                                best_train_improvement = improvement;
                                best_weight = clamped_weight;
                                best_bias = bias;
                            }
                        }

                        // Phase 2: Report improvement on validation samples only
                        let validate_samples = collect_samples(&split.validate);
                        let validate_baseline = compute_baseline(&split.validate);

                        let (val_improvement, val_improved, _) =
                            compute_activation_improvement_and_count(
                                &validate_samples,
                                incoming_weight,
                                best_weight,
                                best_bias,
                                spec.activation,
                                validate_baseline,
                                target_activation_fn,
                            );

                        (best_weight, best_bias, val_improvement, val_improved)
                    } else {
                        // Fallback: below hold-out threshold, use all samples
                        let mut best_weight = base_weight;
                        let mut best_bias = 0.0f32;
                        let mut best_improvement = f32::NEG_INFINITY;
                        let mut best_improved_count = 0u32;

                        let max_out_fb = max_outgoing_weight_for_activation(spec.name);
                        for &weight in &weight_candidates {
                            let clamped_weight = weight.clamp(-max_out_fb, max_out_fb);
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

                            let (improvement, improved, _) =
                                compute_activation_improvement_and_count(
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
                    }
                } else {
                    // Issue #905: Use activation-aware weight calculation
                    let base_weight = match calculate_activation_aware_outgoing_weight(
                        sum_error_activation,
                        sum_activation_sq,
                        incoming_weight,
                        spec.name,
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
                    // Issue #905: Use activation-aware function for bias-adjusted weight
                    let outgoing_weight = calculate_activation_aware_outgoing_weight(
                        sum_error_activation_with_bias,
                        sum_activation_sq_with_bias,
                        incoming_weight,
                        spec.name,
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
