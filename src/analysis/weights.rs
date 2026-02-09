//! Weight calculation functions for NEAT-AI Discovery.
//!
//! This module contains the core weight computation logic used for synapse and
//! neuron candidate evaluation. These functions implement the "what weight should
//! this connection have?" question that is critical for prediction accuracy.
//!
//! ## Key Functions
//!
//! - `calculate_optimal_outgoing_weight` - Compute optimal weight from samples using
//!   least squares formula
//! - `calculate_optimal_identity_outgoing_and_bias` - Joint weight/bias calculation
//!   for IDENTITY neurons (affine correction fitting)
//! - `calculate_optimal_bias` - Find optimal bias via grid search (CPU or GPU)
//!
//! ## Weight Utility Functions
//!
//! - `clamp_weight_update_delta` - Constrain weight updates to valid ranges
//! - `coordinated_structural_activation_delta` - Compute delta for coordinated
//!   structural candidates
//!
//! ## Design Notes
//!
//! - Outgoing weights are clamped to [-0.1, 0.1] (tightened in v0.1.138)
//! - Weight ratio validation ensures incoming/outgoing ratio >= 50 for reliable
//!   predictions (based on successful discovery analysis)
//! - Bias-aware calculation recomputes weights after bias optimisation

use crate::analysis::activation::{activation_name_to_gpu_id, get_bias_range, get_bias_values};
use crate::analysis::gpu::GpuAnalyzer;
use crate::analysis::observation_range::ObservationRangeResult;
use crate::analysis::samples::{EPSILON, HelpfulSample};

// =============================================================================
// Constants
// =============================================================================

/// Maximum allowed outgoing weight for add-neuron and add-synapse candidates.
///
/// Based on analysis of production discoveries (v0.1.138):
/// - ALL successful discoveries have |outgoing_weight| < 0.05
/// - 36% of failures have |outgoing_weight| > 0.05 (up to 50!)
///
/// Using 0.1 provides some margin while eliminating clearly bad candidates.
/// The new neuron should contribute a SMALL correction, not dominate the network.
pub const MAX_OUTGOING_WEIGHT: f32 = 0.1;

/// Minimum incoming/outgoing weight ratio for reliable predictions.
///
/// Based on successful discovery analysis:
/// - Successful discoveries have ratio 71x to 104,000x
/// - Failures often have nearly equal weights (ratio < 10x)
///
/// We require ratio >= 50 when incoming weight > 1.0.
const MIN_WEIGHT_RATIO: f32 = 50.0;

// MIN_NEURON_SAMPLE_COUNT moved to constants.rs (Issue #424)
use super::constants::MIN_NEURON_SAMPLE_COUNT;

// =============================================================================
// Weight Calculation Functions
// =============================================================================

/// Calculate optimal outgoing weight for add-synapse or add-neuron candidates.
///
/// This is the shared weight calculation function used by both synapse and neuron
/// analysis to ensure consistent behaviour and maintainability (DRY principle).
///
/// The formula used is the standard least squares optimal weight:
/// ```text
/// w = Σ(error × activation) / Σ(activation²)
/// ```
///
/// # Arguments
/// * `sum_error_activation` - Σ(error × activation) from samples
/// * `sum_activation_sq` - Σ(activation²) from samples
/// * `incoming_weight` - For neurons: the incoming weight; for synapses: use 1.0
///
/// # Returns
/// * `Some(weight)` - Optimal outgoing weight, clamped to [-MAX_OUTGOING_WEIGHT, MAX_OUTGOING_WEIGHT]
/// * `None` - If weight cannot be computed (insufficient activation, invalid result, or
///   weight ratio too small for reliable prediction)
///
/// # Weight Ratio Validation
/// For add-neuron candidates where incoming_weight > 1.0, we validate that the
/// incoming/outgoing ratio is at least MIN_WEIGHT_RATIO. This is based on analysis
/// showing successful discoveries have much larger incoming than outgoing weights
/// (ratio 71x to 104,000x), while failures often have nearly equal weights.
pub fn calculate_optimal_outgoing_weight(
    sum_error_activation: f32,
    sum_activation_sq: f32,
    incoming_weight: f32,
) -> Option<f32> {
    // Need sufficient activation energy to compute meaningful weight
    if sum_activation_sq <= EPSILON {
        return None;
    }

    // Compute raw optimal weight using least squares formula
    let raw_weight = sum_error_activation / (sum_activation_sq + EPSILON);

    // Reject invalid weights
    if !raw_weight.is_finite() || raw_weight.abs() <= EPSILON {
        return None;
    }

    // Clamp to tight range based on successful discovery analysis
    let clamped = raw_weight.clamp(-MAX_OUTGOING_WEIGHT, MAX_OUTGOING_WEIGHT);

    // For add-neuron candidates with non-trivial incoming weights, validate ratio
    // This catches cases where the computed weight is too large relative to incoming
    if incoming_weight.abs() > 1.0 {
        let ratio = incoming_weight.abs() / (clamped.abs() + EPSILON);
        if ratio < MIN_WEIGHT_RATIO {
            // Weight ratio too small - this configuration is unreliable
            // Skip rather than returning a weight that's likely to fail
            return None;
        }
    }

    Some(clamped)
}

/// Special-case optimisation for IDENTITY candidates: fit an affine correction.
///
/// For IDENTITY, the new neuron's output is linear in the source activation:
/// `output = incoming_weight * activation + bias`.
///
/// The generic search path computes `outgoing_weight` without bias, then searches bias with that
/// fixed outgoing weight. For complement-like shapes (eg `1 - x`) this can miss the optimum
/// because the best `outgoing_weight` depends on the bias/intercept.
///
/// We address that by directly fitting a 2-parameter model:
/// `avg_error ≈ outgoing_weight * (incoming_weight * activation) + intercept`,
/// then converting `intercept` to `bias` via `bias = intercept / outgoing_weight`.
pub fn calculate_optimal_identity_outgoing_and_bias(
    samples: &[HelpfulSample],
    incoming_weight: f32,
) -> Option<(f32, f32)> {
    let mut n: f32 = 0.0;
    let mut sum_a = 0.0f32;
    let mut sum_aa = 0.0f32;
    let mut sum_u = 0.0f32;
    let mut sum_uu = 0.0f32;
    let mut sum_e = 0.0f32;
    let mut sum_eu = 0.0f32;

    for sample in samples {
        if !sample.activation.is_finite() || !sample.avg_error.is_finite() {
            continue;
        }
        sum_a += sample.activation;
        sum_aa += sample.activation * sample.activation;
        let u = incoming_weight * sample.activation;
        n += 1.0;
        sum_u += u;
        sum_uu += u * u;
        sum_e += sample.avg_error;
        sum_eu += sample.avg_error * u;
    }

    if n <= 0.0 {
        return None;
    }

    // If the source activation has low variance, explicitly fitting an intercept (bias)
    // tends to overfit constant-ish sources. We already apply variance discounting later,
    // but keeping IDENTITY bias at 0.0 here avoids inflating the *raw* prediction before
    // the discount is applied (see Issue #130 tests).
    //
    // Use the same threshold as compute_source_variance_discount().
    use super::constants::MIN_SOURCE_STD_DEV;
    let mean_a = sum_a / n;
    let var_a = (sum_aa / n) - (mean_a * mean_a);
    let std_dev_a = var_a.max(0.0).sqrt();
    if std_dev_a < MIN_SOURCE_STD_DEV {
        let outgoing_weight = calculate_optimal_outgoing_weight(sum_eu, sum_uu, incoming_weight)?;
        return Some((outgoing_weight, 0.0));
    }

    // Solve normal equations for `e ≈ w*u + c` (w = outgoing_weight, c = intercept).
    // If the determinant is ~0, fall back to the no-intercept weight and compute the best intercept.
    let det = sum_uu * n - sum_u * sum_u;

    let outgoing_weight_raw = if det.abs() > EPSILON {
        (sum_eu * n - sum_e * sum_u) / det
    } else {
        sum_eu / (sum_uu + EPSILON)
    };

    if !outgoing_weight_raw.is_finite() || outgoing_weight_raw.abs() <= EPSILON {
        return None;
    }

    let outgoing_weight = outgoing_weight_raw.clamp(-MAX_OUTGOING_WEIGHT, MAX_OUTGOING_WEIGHT);

    // Apply the same reliability guard as the generic path.
    if incoming_weight.abs() > 1.0 {
        let ratio = incoming_weight.abs() / (outgoing_weight.abs() + EPSILON);
        if ratio < MIN_WEIGHT_RATIO {
            return None;
        }
    }

    // Best intercept for fixed outgoing_weight (least squares).
    let intercept = (sum_e - outgoing_weight * sum_u) / n;
    let bias = intercept / outgoing_weight;

    if !bias.is_finite() {
        return None;
    }

    // Guard rail: absurd IDENTITY biases are almost always brittle in production.
    let bias_abs_max = crate::analysis::utils::sensible_bias_abs_max_for_squash("IDENTITY");
    if bias.abs() > bias_abs_max {
        return None;
    }

    Some((outgoing_weight, bias))
}

/// Calculate optimal bias for a neuron candidate using grid search.
///
/// This function finds the bias value that maximises error reduction when combined
/// with the given weights and activation function. It tests multiple bias values
/// across an activation-function-specific range and selects the one that gives
/// the best improvement.
///
/// Uses GPU-accelerated parallel search when analyzer is provided and GPU is available,
/// otherwise falls back to CPU sequential search.
///
/// # Arguments
/// * `samples` - Training samples (source activations and target errors)
/// * `incoming_weight` - Weight from source to new neuron
/// * `outgoing_weight` - Weight from new neuron to target
/// * `activation_fn` - Activation function to apply
/// * `squash` - Activation function name (for bias range selection and GPU)
/// * `analyzer` - Optional GPU analyzer for accelerated search
/// * `target_squash` - Optional target neuron's squash function (for saturation-aware models)
///
/// # Returns
/// Optimal bias value that maximises error reduction
pub fn calculate_optimal_bias(
    samples: &[HelpfulSample],
    incoming_weight: f32,
    outgoing_weight: f32,
    activation_fn: fn(f32) -> f32,
    squash: &str,
    analyzer: Option<&GpuAnalyzer>,
    target_squash: Option<&str>,
) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }

    let bias_range = get_bias_range(squash);

    // Try GPU-accelerated search first if analyzer available
    if let Some(gpu_analyzer) = analyzer
        && gpu_analyzer.has_gpu()
    {
        let activation_type = activation_name_to_gpu_id(squash);
        if let Ok(optimal_bias) = gpu_analyzer.evaluate_bias_gpu(
            samples,
            incoming_weight,
            outgoing_weight,
            activation_type,
            bias_range,
        ) {
            // Guard rail: only accept biases within sensible ranges.
            let bias_abs_max = crate::analysis::utils::sensible_bias_abs_max_for_squash(squash);
            if optimal_bias.is_finite() && optimal_bias.abs() <= bias_abs_max {
                return optimal_bias;
            }
        }
        // If GPU fails, fall through to CPU
    }

    // Use log-spaced bias values for efficient search
    // Evolution will fine-tune the exact value after discovery
    let bias_values = get_bias_values(squash);

    // Calculate baseline error (no new neuron)
    let mut total_baseline_error_sq = 0.0;
    for sample in samples {
        if sample.avg_error.is_finite() {
            total_baseline_error_sq += sample.avg_error * sample.avg_error;
        }
    }

    if total_baseline_error_sq <= EPSILON {
        return 0.0;
    }

    let mut best_bias = 0.0;
    let mut best_error_reduction = f32::NEG_INFINITY;

    // Check if we can use HARD_TANH model (aka CLIPPED; need target_value for all samples).
    let use_hard_tanh = matches!(
        target_squash,
        Some(s) if s.eq_ignore_ascii_case("HARD_TANH") || s.eq_ignore_ascii_case("CLIPPED")
    ) && samples
        .iter()
        .all(|s| s.target_value.is_some() && s.target_activation.is_some());

    // Search over log-spaced bias values
    for &bias in &bias_values {
        // Calculate error with this bias
        let mut total_new_error_sq = 0.0;
        let mut valid_samples = 0;

        for sample in samples {
            if !sample.avg_error.is_finite() || !sample.activation.is_finite() {
                continue;
            }

            // Calculate new neuron's activation with bias
            let pre_activation = incoming_weight * sample.activation + bias;
            let new_neuron_activation = activation_fn(pre_activation);

            if !new_neuron_activation.is_finite() {
                continue;
            }

            // Calculate new error at target neuron
            let correction = outgoing_weight * new_neuron_activation;
            let new_error = if use_hard_tanh {
                // HARD_TANH model: account for target neuron's clamping
                // CRITICAL: avg_error is in VALUE domain (targetValue - currentValue from TypeScript)
                // So we compute desired_value = target_value + avg_error, then squash to get expected activation
                let target_value = sample.target_value.unwrap();
                let desired_value = target_value + sample.avg_error;
                let expected = hard_tanh(desired_value);
                let new_input = target_value + correction;
                let new_output = hard_tanh(new_input);
                expected - new_output
            } else {
                // Linear model
                sample.avg_error - correction
            };

            if new_error.is_finite() {
                total_new_error_sq += new_error * new_error;
                valid_samples += 1;
            }
        }

        // Only consider if we have valid samples
        if valid_samples < MIN_NEURON_SAMPLE_COUNT {
            continue;
        }

        // Calculate error reduction (positive is good)
        let error_reduction = total_baseline_error_sq - total_new_error_sq;

        if error_reduction > best_error_reduction {
            best_error_reduction = error_reduction;
            best_bias = bias;
        }
    }

    best_bias
}

// =============================================================================
// Weight Utility Functions
// =============================================================================

/// Clamp a proposed synapse weight delta against `MAX_OUTGOING_WEIGHT`.
///
/// Weight update candidates are represented as a *delta* applied to an existing synapse. If the
/// resulting `new_weight` is clamped, the *effective* delta differs from the proposed delta.
///
/// # Arguments
/// * `old_weight` - Current weight of the synapse
/// * `proposed_delta_weight` - Proposed weight change
///
/// # Returns
/// * `Some((new_weight, delta_weight))` - When the effective delta is meaningful
/// * `None` - When the effective delta is too small (below EPSILON)
pub fn clamp_weight_update_delta(
    old_weight: f32,
    proposed_delta_weight: f32,
) -> Option<(f32, f32)> {
    let new_weight =
        (old_weight + proposed_delta_weight).clamp(-MAX_OUTGOING_WEIGHT, MAX_OUTGOING_WEIGHT);
    let delta_weight = new_weight - old_weight;
    if delta_weight.abs() <= EPSILON {
        None
    } else {
        Some((new_weight, delta_weight))
    }
}

/// Compute activation delta for coordinated structural candidates.
///
/// This function calculates the activation delta needed when replacing a noisy
/// synapse with a trusted one in a coordinated structural operation.
///
/// # Arguments
/// * `trusted_activation` - Activation from the trusted source neuron
/// * `noisy_activation` - Activation from the noisy source neuron (being removed)
/// * `noisy_weight` - Weight of the noisy synapse (being removed)
/// * `trusted_weight` - Current weight of the trusted synapse
///
/// # Returns
/// * `Some(delta)` - The activation delta needed for the coordinated candidate
/// * `None` - If noisy_weight is effectively zero (cannot compute scale)
///
/// # Notes (7-Jan-2026)
/// We intentionally do **not** clamp the trusted weight here. The coordinated candidate is
/// derived from existing synapse weights, and NEAT-AI will validate the full ablation on the
/// complete training set. Clamping here changes the candidate semantics and can suppress
/// valid coordinated candidates.
pub fn coordinated_structural_activation_delta(
    trusted_activation: f32,
    noisy_activation: f32,
    noisy_weight: f32,
    trusted_weight: f32,
) -> Option<f32> {
    if noisy_weight.abs() <= EPSILON {
        return None;
    }

    let new_trusted_weight = trusted_weight + noisy_weight;
    let delta_trusted_weight = new_trusted_weight - trusted_weight;

    let scale = delta_trusted_weight / noisy_weight;
    Some(scale * trusted_activation - noisy_activation)
}

// =============================================================================
// Range-Aware Weight Computation (Issue #402)
// =============================================================================

// DEFAULT_SENTINEL_TOLERANCE uses SENTINEL_TOLERANCE from constants.rs (Issue #424)
pub use super::constants::SENTINEL_TOLERANCE as DEFAULT_SENTINEL_TOLERANCE;

/// Compute `sum_error_activation` and `sum_activation_sq` after excluding samples
/// whose source activation is at a sentinel value.
///
/// This is the pre-filter step described in Issue #402. It accepts observation range
/// metadata from `detect_observation_ranges()` (Issue #398) and removes sentinel
/// samples before accumulating the sums that feed into
/// `calculate_optimal_outgoing_weight()`.
///
/// # Arguments
/// * `samples` - The full set of helpful samples (source activation + target error).
/// * `range` - Observation range metadata identifying sentinel values and effective range.
/// * `sentinel_tolerance` - Tolerance for matching activations to sentinel values.
///
/// # Returns
/// A tuple of `(sum_error_activation, sum_activation_sq, effective_count)` computed
/// only from samples that are **not** at a sentinel value.
pub fn compute_range_aware_sums(
    samples: &[HelpfulSample],
    range: &ObservationRangeResult,
    sentinel_tolerance: f32,
) -> (f32, f32, usize) {
    let mut sum_error_activation: f32 = 0.0;
    let mut sum_activation_sq: f32 = 0.0;
    let mut count: usize = 0;

    for sample in samples {
        if !sample.activation.is_finite() || !sample.avg_error.is_finite() {
            continue;
        }

        // Check whether this sample is at a sentinel value
        let is_sentinel = range
            .sentinel_values
            .iter()
            .any(|&sv| (sample.activation - sv).abs() <= sentinel_tolerance);

        if is_sentinel {
            continue;
        }

        sum_error_activation += sample.avg_error * sample.activation;
        sum_activation_sq += sample.activation * sample.activation;
        count += 1;
    }

    (sum_error_activation, sum_activation_sq, count)
}

/// Compute an optimal outgoing weight after filtering out sentinel samples.
///
/// This is a wrapper around `calculate_optimal_outgoing_weight()` that first
/// excludes samples where the source observation is at a detected sentinel value.
/// The core weight computation function is not modified (DRY principle).
///
/// # Arguments
/// * `samples` - The full set of helpful samples.
/// * `range` - Observation range metadata from `detect_observation_ranges()` (#398).
/// * `incoming_weight` - For neurons: the incoming weight; for synapses: use 1.0.
/// * `sentinel_tolerance` - Tolerance for matching activations to sentinel values.
///
/// # Returns
/// * `Some(weight)` - Optimal weight computed from non-sentinel samples only.
/// * `None` - If insufficient non-sentinel samples or weight cannot be computed.
pub fn calculate_range_aware_weight(
    samples: &[HelpfulSample],
    range: &ObservationRangeResult,
    incoming_weight: f32,
    sentinel_tolerance: f32,
) -> Option<f32> {
    let (sum_ea, sum_aa, _count) = compute_range_aware_sums(samples, range, sentinel_tolerance);

    calculate_optimal_outgoing_weight(sum_ea, sum_aa, incoming_weight)
}

// =============================================================================
// Internal Helper Functions
// =============================================================================

/// Apply HARD_TANH activation function (clamp to [-1, 1])
#[inline(always)]
fn hard_tanh(x: f32) -> f32 {
    x.clamp(-1.0, 1.0)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // calculate_optimal_outgoing_weight tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_optimal_weight_returns_none_for_insufficient_activation() {
        let result = calculate_optimal_outgoing_weight(1.0, 0.0, 1.0);
        assert!(
            result.is_none(),
            "Should return None when sum_activation_sq is zero"
        );
    }

    #[test]
    fn test_optimal_weight_returns_none_for_very_small_activation() {
        let result = calculate_optimal_outgoing_weight(1.0, EPSILON * 0.5, 1.0);
        assert!(
            result.is_none(),
            "Should return None when sum_activation_sq is below EPSILON"
        );
    }

    #[test]
    fn test_optimal_weight_returns_none_for_non_finite_results() {
        let result = calculate_optimal_outgoing_weight(f32::INFINITY, 1.0, 1.0);
        assert!(result.is_none(), "Should return None for infinite input");

        let result = calculate_optimal_outgoing_weight(f32::NAN, 1.0, 1.0);
        assert!(result.is_none(), "Should return None for NaN input");
    }

    #[test]
    fn test_optimal_weight_returns_none_for_near_zero_weights() {
        // Very small sum_error_activation relative to sum_activation_sq
        let result = calculate_optimal_outgoing_weight(EPSILON * 0.1, 100.0, 1.0);
        assert!(
            result.is_none(),
            "Should return None when computed weight is near zero"
        );
    }

    #[test]
    fn test_optimal_weight_is_clamped_to_max_outgoing_weight() {
        // Large positive weight: sum_error_activation=10, sum_activation_sq=1 => raw_weight=10
        let result = calculate_optimal_outgoing_weight(10.0, 1.0, 1.0);
        assert!(result.is_some());
        let weight = result.unwrap();
        assert!(
            (weight - MAX_OUTGOING_WEIGHT).abs() < EPSILON,
            "Weight {weight} should be clamped to MAX_OUTGOING_WEIGHT {MAX_OUTGOING_WEIGHT}"
        );

        // Large negative weight
        let result = calculate_optimal_outgoing_weight(-10.0, 1.0, 1.0);
        assert!(result.is_some());
        let weight = result.unwrap();
        assert!(
            (weight - (-MAX_OUTGOING_WEIGHT)).abs() < EPSILON,
            "Weight {} should be clamped to -MAX_OUTGOING_WEIGHT {}",
            weight,
            -MAX_OUTGOING_WEIGHT
        );
    }

    #[test]
    fn test_optimal_weight_accounts_for_incoming_weight() {
        // With incoming_weight=10, raw_weight=1.0, clamped=0.1
        // ratio = 10/0.1 = 100 >= 50 (MIN_WEIGHT_RATIO), should pass
        let result = calculate_optimal_outgoing_weight(1.0, 1.0, 10.0);
        assert!(
            result.is_some(),
            "incoming_weight=10 should have ratio >= 50"
        );

        // With incoming_weight=2, raw_weight=1.0, clamped=0.1
        // ratio = 2/0.1 = 20 < 50 (MIN_WEIGHT_RATIO), should be rejected
        let result = calculate_optimal_outgoing_weight(1.0, 1.0, 2.0);
        assert!(
            result.is_none(),
            "incoming_weight=2 with clamped weight=0.1 has ratio=20 < MIN_WEIGHT_RATIO"
        );

        // With incoming_weight=1.0 (not > 1), ratio check is skipped
        let result = calculate_optimal_outgoing_weight(1.0, 1.0, 1.0);
        assert!(result.is_some(), "incoming_weight=1.0 skips ratio check");
    }

    #[test]
    fn test_optimal_weight_normal_calculation() {
        // Normal case: small weight within range
        // sum_error_activation=0.5, sum_activation_sq=10 => raw_weight=0.05
        let result = calculate_optimal_outgoing_weight(0.5, 10.0, 1.0);
        assert!(result.is_some());
        let weight = result.unwrap();
        assert!(
            (weight - 0.05).abs() < 0.001,
            "Expected weight ~0.05, got {weight}"
        );
    }

    #[test]
    fn test_optimal_weight_ratio_validation() {
        // With large incoming_weight (100), the outgoing weight must be small enough
        // for ratio >= MIN_WEIGHT_RATIO (50)
        // If raw_weight would be 0.1, we need 100/0.1 = 1000 >= 50, so it should pass
        let sum_error_activation = 10.0;
        let sum_activation_sq = 100.0;
        let incoming_weight = 100.0;
        let result = calculate_optimal_outgoing_weight(
            sum_error_activation,
            sum_activation_sq,
            incoming_weight,
        );

        // raw_weight = 10/100 = 0.1 => clamped to 0.1
        // ratio = 100/0.1 = 1000 >= 50, should pass
        assert!(result.is_some());
        let weight = result.unwrap();
        assert!(
            weight.abs() <= MAX_OUTGOING_WEIGHT,
            "Weight should be within MAX_OUTGOING_WEIGHT"
        );
    }

    #[test]
    fn test_optimal_weight_rejects_poor_ratio() {
        // Large raw weight that would be clamped to MAX_OUTGOING_WEIGHT
        // With incoming_weight=10, ratio = 10/0.1 = 100 >= 50, should pass
        let result = calculate_optimal_outgoing_weight(10.0, 1.0, 10.0);
        assert!(
            (result.unwrap().abs() - MAX_OUTGOING_WEIGHT).abs() < EPSILON,
            "Large raw weight should be clamped to MAX_OUTGOING_WEIGHT"
        );
    }

    // -------------------------------------------------------------------------
    // calculate_optimal_identity_outgoing_and_bias tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_identity_returns_none_for_empty_samples() {
        let samples: Vec<HelpfulSample> = vec![];
        let result = calculate_optimal_identity_outgoing_and_bias(&samples, 1.0);
        assert!(result.is_none());
    }

    #[test]
    fn test_identity_returns_none_for_invalid_samples() {
        let samples = vec![
            HelpfulSample {
                activation: f32::NAN,
                avg_error: 0.5,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 1.0,
                avg_error: f32::NAN,
                target_value: None,
                target_activation: None,
            },
        ];
        let result = calculate_optimal_identity_outgoing_and_bias(&samples, 1.0);
        assert!(result.is_none());
    }

    #[test]
    fn test_identity_with_low_variance_returns_zero_bias() {
        // All samples have same activation (low variance)
        let samples = vec![
            HelpfulSample {
                activation: 0.5,
                avg_error: 0.1,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.5,
                avg_error: 0.2,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.5,
                avg_error: 0.15,
                target_value: None,
                target_activation: None,
            },
        ];
        let result = calculate_optimal_identity_outgoing_and_bias(&samples, 1.0);
        if let Some((_, bias)) = result {
            assert!(
                bias.abs() < 0.001,
                "Low variance source should have bias=0, got {bias}"
            );
        }
    }

    #[test]
    fn test_identity_computes_valid_weight_and_bias() {
        // Create samples with varying activation and correlated error
        let samples = vec![
            HelpfulSample {
                activation: 0.0,
                avg_error: 0.05,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 0.5,
                avg_error: 0.025,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 1.0,
                avg_error: 0.0,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -0.5,
                avg_error: 0.075,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: -1.0,
                avg_error: 0.1,
                target_value: None,
                target_activation: None,
            },
        ];
        let result = calculate_optimal_identity_outgoing_and_bias(&samples, 1.0);
        assert!(result.is_some(), "Should compute valid weight and bias");
        let (weight, bias) = result.unwrap();
        assert!(weight.is_finite(), "Weight should be finite");
        assert!(bias.is_finite(), "Bias should be finite");
        assert!(
            weight.abs() <= MAX_OUTGOING_WEIGHT,
            "Weight should be within MAX_OUTGOING_WEIGHT"
        );
    }

    // -------------------------------------------------------------------------
    // calculate_optimal_bias tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_optimal_bias_returns_zero_for_empty_samples() {
        let samples: Vec<HelpfulSample> = vec![];
        let bias = calculate_optimal_bias(&samples, 1.0, 0.05, |x| x, "IDENTITY", None, None);
        assert!(
            bias.abs() < EPSILON,
            "Empty samples should return bias=0, got {bias}"
        );
    }

    #[test]
    fn test_optimal_bias_returns_zero_for_zero_baseline_error() {
        // All samples have zero error
        let samples = vec![
            HelpfulSample {
                activation: 0.5,
                avg_error: 0.0,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 1.0,
                avg_error: 0.0,
                target_value: None,
                target_activation: None,
            },
        ];
        let bias = calculate_optimal_bias(&samples, 1.0, 0.05, |x| x, "IDENTITY", None, None);
        assert!(
            bias.abs() < EPSILON,
            "Zero baseline error should return bias=0, got {bias}"
        );
    }

    #[test]
    fn test_optimal_bias_with_tanh() {
        // Create samples with error that could be corrected by bias shift
        let samples: Vec<HelpfulSample> = (0..20)
            .map(|i| {
                let activation = (i as f32 - 10.0) / 10.0; // -1 to 1
                HelpfulSample {
                    activation,
                    avg_error: 0.1, // Constant positive error
                    target_value: None,
                    target_activation: None,
                }
            })
            .collect();

        let tanh_fn = |x: f32| x.tanh();
        let bias = calculate_optimal_bias(&samples, 1.0, 0.05, tanh_fn, "TANH", None, None);

        // Should find some bias value (exact value depends on grid search)
        assert!(bias.is_finite(), "Bias should be finite");
    }

    #[test]
    fn test_optimal_bias_with_relu() {
        // Create samples with error pattern suited for ReLU
        let samples: Vec<HelpfulSample> = (0..20)
            .map(|i| {
                let activation = (i as f32 - 10.0) / 10.0; // -1 to 1
                HelpfulSample {
                    activation,
                    avg_error: if activation > 0.0 { 0.1 } else { -0.1 },
                    target_value: None,
                    target_activation: None,
                }
            })
            .collect();

        let relu_fn = |x: f32| x.max(0.0);
        let bias = calculate_optimal_bias(&samples, 1.0, 0.05, relu_fn, "ReLU", None, None);

        assert!(bias.is_finite(), "Bias should be finite");
    }

    // -------------------------------------------------------------------------
    // clamp_weight_update_delta tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_clamp_delta_returns_none_for_tiny_delta() {
        let result = clamp_weight_update_delta(0.05, EPSILON * 0.5);
        assert!(result.is_none(), "Tiny delta should return None");
    }

    #[test]
    fn test_clamp_delta_within_bounds() {
        let result = clamp_weight_update_delta(0.0, 0.05);
        assert!(result.is_some());
        let (new_weight, delta) = result.unwrap();
        assert!((new_weight - 0.05).abs() < EPSILON);
        assert!((delta - 0.05).abs() < EPSILON);
    }

    #[test]
    fn test_clamp_delta_exceeds_upper_bound() {
        // Start at 0.05, try to add 0.1 => clamped to 0.1
        let result = clamp_weight_update_delta(0.05, 0.1);
        assert!(result.is_some());
        let (new_weight, delta) = result.unwrap();
        assert!((new_weight - MAX_OUTGOING_WEIGHT).abs() < EPSILON);
        assert!((delta - 0.05).abs() < EPSILON); // Effective delta is 0.05
    }

    #[test]
    fn test_clamp_delta_exceeds_lower_bound() {
        // Start at -0.05, try to subtract 0.1 => clamped to -0.1
        let result = clamp_weight_update_delta(-0.05, -0.1);
        assert!(result.is_some());
        let (new_weight, delta) = result.unwrap();
        assert!((new_weight - (-MAX_OUTGOING_WEIGHT)).abs() < EPSILON);
        assert!((delta - (-0.05)).abs() < EPSILON); // Effective delta is -0.05
    }

    #[test]
    fn test_clamp_delta_already_at_max() {
        // Already at MAX_OUTGOING_WEIGHT, positive delta should be ineffective
        let result = clamp_weight_update_delta(MAX_OUTGOING_WEIGHT, 0.05);
        assert!(result.is_none(), "Delta at max should have no effect");
    }

    // -------------------------------------------------------------------------
    // coordinated_structural_activation_delta tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_coordinated_delta_returns_none_for_zero_noisy_weight() {
        let result = coordinated_structural_activation_delta(1.0, 0.5, 0.0, 0.5);
        assert!(result.is_none(), "Zero noisy weight should return None");

        let result = coordinated_structural_activation_delta(1.0, 0.5, EPSILON * 0.5, 0.5);
        assert!(
            result.is_none(),
            "Near-zero noisy weight should return None"
        );
    }

    #[test]
    fn test_coordinated_delta_computes_correctly() {
        // trusted_activation=1.0, noisy_activation=0.5, noisy_weight=0.2, trusted_weight=0.3
        // new_trusted_weight = 0.3 + 0.2 = 0.5
        // delta_trusted_weight = 0.5 - 0.3 = 0.2
        // scale = 0.2 / 0.2 = 1.0
        // result = 1.0 * 1.0 - 0.5 = 0.5
        let result = coordinated_structural_activation_delta(1.0, 0.5, 0.2, 0.3);
        assert!(result.is_some());
        let delta = result.unwrap();
        assert!(
            (delta - 0.5).abs() < 0.001,
            "Expected delta ~0.5, got {delta}"
        );
    }

    #[test]
    fn test_coordinated_delta_with_negative_weights() {
        // Test with negative weights
        let result = coordinated_structural_activation_delta(1.0, 0.5, -0.2, -0.3);
        assert!(result.is_some());
        // new_trusted_weight = -0.3 + (-0.2) = -0.5
        // delta_trusted_weight = -0.5 - (-0.3) = -0.2
        // scale = -0.2 / -0.2 = 1.0
        // result = 1.0 * 1.0 - 0.5 = 0.5
        let delta = result.unwrap();
        assert!(
            (delta - 0.5).abs() < 0.001,
            "Expected delta ~0.5, got {delta}"
        );
    }

    // -------------------------------------------------------------------------
    // hard_tanh tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_hard_tanh_clamps_correctly() {
        assert!((hard_tanh(0.5) - 0.5).abs() < EPSILON);
        assert!((hard_tanh(-0.5) - (-0.5)).abs() < EPSILON);
        assert!((hard_tanh(2.0) - 1.0).abs() < EPSILON);
        assert!((hard_tanh(-2.0) - (-1.0)).abs() < EPSILON);
        assert!((hard_tanh(1.0) - 1.0).abs() < EPSILON);
        assert!((hard_tanh(-1.0) - (-1.0)).abs() < EPSILON);
    }
}
