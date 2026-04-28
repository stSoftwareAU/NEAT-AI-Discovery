//! GPU-accelerated candidate evaluation — batched orchestration
//!
//! This module orchestrates batched GPU evaluation of activation candidates
//! and re-exports the specialised evaluation sub-modules:
//!
//! - `relu_evaluation` — `ReLU` candidate evaluation (split by error sign)
//! - `activation_evaluation` — Non-ReLU activation candidate evaluation
//! - `activation_subset_evaluation` — Split-error subset evaluation

// Re-export from sibling modules (declared in mod.rs)
pub(crate) use super::activation_evaluation::{
    ActivationEvalParams, evaluate_activation_candidate,
};
pub(crate) use super::relu_evaluation::evaluate_relu_candidates_split;

use crate::CandidateNeuronJson;
use crate::analysis::activation::{
    ACTIVATION_SPECS, activation_name_to_gpu_id, get_target_simulation_fn,
    has_sufficient_output_variance,
};
use crate::analysis::gpu::GpuEvaluator;
use crate::analysis::samples::{EPSILON, HelpfulSample, NeuronStats};
use crate::analysis::scoring::confidence::compute_confidence_metrics;
use crate::analysis::scoring::weights::{
    calculate_activation_aware_outgoing_weight, calculate_optimal_bias,
    calculate_optimal_identity_outgoing_and_bias,
};
use anyhow::Result;

use super::scoring::compute_activation_improvement_and_count;

use crate::analysis::constants::MIN_NEURON_SAMPLE_COUNT;

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
            // Issue #905: Use activation-aware weight calculation to avoid
            // systematically rejecting non-linear candidates
            let outgoing_weight = match calculate_activation_aware_outgoing_weight(
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
        let (net_improvement, improved_count, total_count, magnitude_ratio) =
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
                // Issue #1161: magnitude-weighted ratio for downstream pessimism discounting.
                improvement_magnitude_ratio: Some(magnitude_ratio),
                target_neuron_stats,
                prediction_confidence: confidence_metrics.prediction_confidence,
                expected_score_gain_confidence_interval: confidence_metrics
                    .expected_score_gain_confidence_interval,
                target_saturation_factor: None,
                variant_key: None,
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
