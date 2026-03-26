//! Saturation-aware gain estimation for non-linear squash functions.
//!
//! Estimates the combined gain of routing multiple inputs through a non-linear
//! squash function, accounting for diminished returns in saturated regimes.

use crate::CandidateSynapseJson;
use crate::activations::apply_scalar_squash;

use crate::analysis::constants::{
    COMPRESSION_MIN_BENEFIT_RATIO, COMPRESSION_SATURATION_THRESHOLD, MIN_COMPRESSED_SOURCES,
};

/// Estimate the combined gain of routing multiple inputs through a non-linear
/// squash function, accounting for saturation effects.
///
/// For each candidate, computes `squash(w_i * input_proxy)` individually and
/// `squash(sum(w_i * input_proxy))` combined. The proxy input is 1.0 (unit
/// activation), so weights directly determine the pre-activation magnitude.
///
/// Returns `None` if the combined gain does not exceed the best individual
/// gain by `COMPRESSION_MIN_BENEFIT_RATIO`.
pub(super) fn estimate_nonlinear_gain(
    candidates: &[CandidateSynapseJson],
    squash_name: &str,
) -> Option<f32> {
    if candidates.len() < MIN_COMPRESSED_SOURCES {
        return None;
    }

    // Compute individual squash outputs: |squash(w_i)| as proxy for individual contribution.
    let mut individual_gains: Vec<f32> = Vec::with_capacity(candidates.len());
    let mut combined_pre_activation: f32 = 0.0;

    for c in candidates {
        let individual_output = apply_scalar_squash(squash_name, c.weight)?;
        individual_gains.push(individual_output.abs() * c.expected_creature_score_gain);
        combined_pre_activation += c.weight;
    }

    let best_individual = individual_gains.iter().copied().fold(0.0_f32, f32::max);

    if best_individual <= 0.0 {
        return None;
    }

    // Compute combined squash output.
    let combined_output = apply_scalar_squash(squash_name, combined_pre_activation)?;

    // Estimate combined gain: ratio of combined activation to sum of individual activations,
    // scaled by the sum of individual gains.
    let individual_activation_sum: f32 = candidates
        .iter()
        .filter_map(|c| apply_scalar_squash(squash_name, c.weight).map(f32::abs))
        .sum();

    let combined_gain = if individual_activation_sum > 1e-10 {
        let activation_ratio = combined_output.abs() / individual_activation_sum;
        let gain_sum: f32 = candidates
            .iter()
            .map(|c| c.expected_creature_score_gain)
            .sum();
        gain_sum * activation_ratio
    } else {
        return None;
    };

    // Apply saturation discount if combined activation is near squash bounds.
    let saturation_limit = match squash_name {
        "TANH" => 1.0_f32,
        "GELU" => combined_pre_activation.abs().max(1.0), // GELU is unbounded positive
        _ => 1.0,
    };

    let saturation_fraction = combined_output.abs() / saturation_limit;
    let discounted_gain = if saturation_fraction > COMPRESSION_SATURATION_THRESHOLD {
        // Diminished returns in saturated regime.
        let excess = saturation_fraction - COMPRESSION_SATURATION_THRESHOLD;
        let discount = 1.0 - (excess / (1.0 - COMPRESSION_SATURATION_THRESHOLD)).min(0.9);
        combined_gain * discount
    } else {
        combined_gain
    };

    // Benefit ratio check: combined must beat best individual by the required margin.
    if discounted_gain < best_individual * COMPRESSION_MIN_BENEFIT_RATIO {
        return None;
    }

    Some(discounted_gain)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candidate(from: &str, to: &str, weight: f32, gain: f32) -> CandidateSynapseJson {
        CandidateSynapseJson {
            from_neuron_uuid: from.to_string(),
            to_neuron_uuid: to.to_string(),
            from_neuron_index: None,
            to_neuron_index: None,
            weight,
            target_neuron_impact: 1.0,
            expected_creature_error_reduction: gain,
            expected_creature_score_gain: gain,
            improved_count: 80,
            total_count: 100,
            target_neuron_stats: None,
            outlier_reduction_info: None,
            prediction_confidence: 0.8,
            expected_score_gain_confidence_interval: [gain * 0.5, gain * 1.5],
            comment: None,
        }
    }

    #[test]
    fn test_nonlinear_benefit_ratio_filtering() {
        // Very similar individual gains — non-linear combining may not meet
        // the 1.05 benefit ratio threshold.
        let gain = estimate_nonlinear_gain(
            &[
                make_candidate("input-a", "output-1", 3.0, 0.05),
                make_candidate("input-b", "output-1", 3.0, 0.05),
            ],
            "TANH",
        );

        // With saturated TANH (w=3.0), combining adds minimal benefit.
        // The function should return None because the benefit ratio is not met.
        assert!(
            gain.is_none(),
            "Saturated TANH inputs should fail benefit ratio check"
        );
    }
}
