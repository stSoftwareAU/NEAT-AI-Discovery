//! `ReLU` candidate evaluation
//!
//! This module evaluates `ReLU` activation candidates by splitting samples based
//! on the target neuron's error sign — finding candidates for both positive-error
//! and negative-error subsets.

use crate::CandidateNeuronJson;
use crate::analysis::gpu::GpuEvaluator;
use crate::analysis::samples::{EPSILON, HelpfulSample};
use anyhow::Result;

use super::scoring::compute_relu_improvement_and_count;

// MIN_NEURON_SAMPLE_COUNT moved to constants.rs (Issue #424)
use crate::analysis::constants::MIN_NEURON_SAMPLE_COUNT;

use crate::analysis::activation::get_target_simulation_fn;

/// Result from `ReLU` evaluation (split by target error sign)
pub(crate) struct SplitReluResult {
    /// Candidate for samples with positive error (output should be higher)
    pub(crate) positive_error_candidate: Option<CandidateNeuronJson>,
    /// Candidate for samples with negative error (output should be lower)
    pub(crate) negative_error_candidate: Option<CandidateNeuronJson>,
}

/// Evaluate `ReLU` candidates by splitting samples based on TARGET neuron's error sign.
///
/// This is the PRIMARY approach for `ReLU` evaluation. It finds candidates for both directions:
/// - **Positive-error samples** (output should be HIGHER): compute weight that pushes UP
/// - **Negative-error samples** (output should be LOWER): compute weight that pushes DOWN
///
/// For each direction:
/// 1. Compute optimal weight from the error subset
/// 2. Evaluate NET improvement across ALL samples
/// 3. Return candidate if it passes threshold
///
/// This is the correct approach for directional activations like `ReLU` because:
/// - `ReLU` can only push output in ONE direction (based on outgoing weight sign)
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
                let (net_improvement, improved, total, magnitude_ratio) =
                    compute_relu_improvement_and_count(
                        samples,
                        candidate.incoming_weight,
                        candidate.outgoing_weight,
                        candidate.bias,
                        total_baseline_error_sq,
                        target_activation_fn,
                    );
                candidate.improved_count = improved;
                candidate.total_count = total;
                // Issue #1161: persist magnitude-weighted ratio for pessimism discounting.
                candidate.improvement_magnitude_ratio = Some(magnitude_ratio);

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
                let (net_improvement, improved, total, magnitude_ratio) =
                    compute_relu_improvement_and_count(
                        samples,
                        candidate.incoming_weight,
                        candidate.outgoing_weight,
                        candidate.bias,
                        total_baseline_error_sq,
                        target_activation_fn,
                    );
                candidate.improved_count = improved;
                candidate.total_count = total;
                // Issue #1161: persist magnitude-weighted ratio for pessimism discounting.
                candidate.improvement_magnitude_ratio = Some(magnitude_ratio);

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
