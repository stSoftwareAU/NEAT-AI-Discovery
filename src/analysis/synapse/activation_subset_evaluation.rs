//! Activation subset evaluation
//!
//! Evaluates activation function candidates from a specific error subset
//! (positive or negative), then computes net improvement across all samples.
//! This is the core of the split-error approach for non-ReLU activations.

use crate::CandidateNeuronJson;
use crate::analysis::activation::{
    ActivationCandidateSpec, activation_name_to_gpu_id, has_sufficient_output_variance,
};
use crate::analysis::gpu::GpuEvaluator;
use crate::analysis::samples::{HelpfulSample, NeuronStats};
use crate::analysis::scoring::confidence::compute_confidence_metrics;
use crate::analysis::scoring::weights::{
    calculate_optimal_bias, calculate_optimal_identity_outgoing_and_bias,
    calculate_optimal_outgoing_weight,
};
use anyhow::Result;

use super::scoring::compute_activation_improvement_and_count;

use crate::analysis::constants::MIN_NEURON_SAMPLE_COUNT;

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
