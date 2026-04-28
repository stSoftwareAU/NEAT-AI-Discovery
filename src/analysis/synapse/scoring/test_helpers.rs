//! Test-only helper functions for improvement calculation
//!
//! These wrappers provide simpler interfaces to improvement functions for use in
//! integration tests and implementation tests.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use crate::analysis::activation::{
    TargetSimulationMode, get_target_simulation_fn, get_target_simulation_mode,
};
use crate::analysis::samples::{EPSILON, HelpfulSample};

use super::improvement::compute_relu_improvement_and_count;

/// Wrapper for tests - computes improvement only.
/// NOTE: For `ReLU` candidates, bias affects which samples activate. Pass the actual bias
/// that will be used with the new neuron for accurate predictions.
pub(crate) fn compute_net_improvement_with_squash(
    samples: &[HelpfulSample],
    incoming_weight: f32,
    outgoing_weight: f32,
    bias: f32,
    total_baseline_error_sq: f32,
    target_squash: Option<&str>,
) -> f32 {
    let target_activation_fn = get_target_simulation_fn(samples, target_squash);
    let (improvement, _, _, _) = compute_relu_improvement_and_count(
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
/// avoiding overprediction near saturation for `HARD_TANH`, TANH, LOGISTIC, etc.
///
/// Returns `improvement_percentage` only. Used in tests; production uses `compute_synapse_improvement_and_count`.
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
            TargetSimulationMode::ApproximateValueFromActivation {
                activation_fn: target_fn,
                inverse_fn,
            } => {
                // Saturation-aware model (ACTIVATION domain), approximating missing target_value
                // using the inverse function (Issue #906).
                let target_activation = sample.target_activation.unwrap();
                let target_value = sample
                    .target_value
                    .unwrap_or_else(|| inverse_fn(target_activation));
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

/// Wrapper for tests - counts improved samples only.
pub(crate) fn count_improved_samples(
    samples: &[HelpfulSample],
    incoming_weight: f32,
    outgoing_weight: f32,
    bias: f32,
    target_squash: Option<&str>,
) -> (u32, u32) {
    let total_baseline_error_sq: f32 = samples.iter().map(|s| s.avg_error * s.avg_error).sum();
    let target_activation_fn = get_target_simulation_fn(samples, target_squash);
    let (_, improved, total, _) = compute_relu_improvement_and_count(
        samples,
        incoming_weight,
        outgoing_weight,
        bias,
        total_baseline_error_sq,
        target_activation_fn,
    );
    (improved, total)
}
