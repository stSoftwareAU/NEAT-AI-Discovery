//! Oscillating neuron detection module (Issue #358).
//!
//! Identifies hidden neurons whose activations oscillate between positive and negative
//! values across training samples, indicating the neuron is fighting between two
//! contradictory functions. Oscillating neurons may benefit from an activation function
//! change or bias adjustment to stabilise their output.
//!
//! ## Detection Criteria
//!
//! A neuron is "oscillating" if:
//! 1. **Sign changes**: The activation crosses zero frequently (more than a minimum fraction
//!    of samples show sign changes).
//! 2. **Balanced signs**: Both positive and negative activations appear in substantial
//!    proportions (neither dominates overwhelmingly).
//! 3. **Meaningful magnitude**: The mean absolute activation is above a minimum threshold
//!    (distinguishing from dead neurons).
//! 4. **Only hidden neurons**: Input and output neurons are excluded.
//!
//! ## Recommended Actions
//!
//! When oscillation is detected, we recommend:
//! 1. **Change activation function**: Switch to ABSOLUTE or RELU to stabilise sign.
//! 2. **Adjust bias**: Shift the operating point to favour the dominant sign.
//!
//! These are emitted as `CoordinatedStructuralCandidateJson` with `ChangeSquash` and/or
//! `SetBias` operations.

use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson};

/// Minimum samples required for reliable oscillation detection.
const MIN_SAMPLES_FOR_OSCILLATION: usize = 20;

/// Minimum fraction of consecutive sample pairs that must show a sign change
/// to consider the neuron oscillating.
const MIN_SIGN_CHANGE_FRACTION: f32 = 0.3;

/// Minimum fraction of samples on the minority sign side.
/// If 90%+ are one sign, it is not truly oscillating — it is biased.
const MIN_MINORITY_SIGN_FRACTION: f32 = 0.2;

/// Minimum mean absolute activation to distinguish from dead neurons.
const MIN_MEAN_ABS_ACTIVATION: f32 = 0.01;

/// Result of detecting an oscillating neuron.
#[derive(Debug, Clone)]
pub struct OscillatingNeuronCandidate {
    /// UUID of the oscillating neuron.
    pub neuron_uuid: String,
    /// Current activation function of the neuron.
    pub current_squash: String,
    /// Fraction of consecutive sample pairs with sign changes.
    pub sign_change_fraction: f32,
    /// Fraction of samples with positive activation.
    pub positive_fraction: f32,
    /// Mean absolute activation across all samples.
    pub mean_abs_activation: f32,
    /// Number of samples analysed.
    pub sample_count: usize,
    /// Recommended new activation function (e.g., "ABSOLUTE", "RELU").
    pub recommended_squash: String,
    /// Recommended bias adjustment.
    pub recommended_bias_delta: Option<f32>,
    /// Estimated creature score improvement from fixing oscillation.
    pub estimated_improvement: f32,
}

/// Detect oscillating neurons from their recorded activations.
///
/// # Arguments
/// * `neurons` - List of `(neuron_uuid, squash, bias)` tuples for hidden neurons to check.
/// * `neuron_records` - List of `(neuron_uuid, records)` tuples with the recorded activations.
///
/// # Returns
/// A list of `OscillatingNeuronCandidate` for neurons that are oscillating,
/// sorted by estimated improvement (best first).
pub fn detect_oscillating_neurons(
    neurons: &[(String, String, f32)],
    neuron_records: &[(String, Vec<DiscoverRecord>)],
) -> Vec<OscillatingNeuronCandidate> {
    let mut candidates = Vec::new();

    // Build a map from uuid to records for quick lookup
    let records_map: std::collections::HashMap<&str, &Vec<DiscoverRecord>> = neuron_records
        .iter()
        .map(|(uuid, records)| (uuid.as_str(), records))
        .collect();

    for (uuid, squash, bias) in neurons {
        let Some(records) = records_map.get(uuid.as_str()) else {
            continue;
        };

        if records.len() < MIN_SAMPLES_FOR_OSCILLATION {
            continue;
        }

        let n = records.len() as f32;

        // Compute mean absolute activation
        let sum_abs_activation: f32 = records.iter().map(|r| r.activation.abs()).sum();
        let mean_abs_activation = sum_abs_activation / n;

        // Skip dead or near-dead neurons
        if mean_abs_activation < MIN_MEAN_ABS_ACTIVATION {
            continue;
        }

        // Count positive and negative activations
        let positive_count = records.iter().filter(|r| r.activation > 0.0).count();
        let negative_count = records.iter().filter(|r| r.activation < 0.0).count();
        let nonzero_count = positive_count + negative_count;

        if nonzero_count == 0 {
            continue;
        }

        let positive_fraction = positive_count as f32 / n;
        let minority_fraction = positive_fraction.min(1.0 - positive_fraction);

        // Both signs must appear in substantial proportion
        if minority_fraction < MIN_MINORITY_SIGN_FRACTION {
            continue;
        }

        // Count sign changes between consecutive samples (sorted by obs_index)
        let mut sorted_records: Vec<&DiscoverRecord> = records.iter().collect();
        sorted_records.sort_by_key(|r| r.obs_index);

        let mut sign_changes: usize = 0;
        for window in sorted_records.windows(2) {
            let prev_sign = window[0].activation >= 0.0;
            let curr_sign = window[1].activation >= 0.0;
            if prev_sign != curr_sign {
                sign_changes += 1;
            }
        }

        let sign_change_fraction = sign_changes as f32 / (records.len() - 1) as f32;

        if sign_change_fraction < MIN_SIGN_CHANGE_FRACTION {
            continue;
        }

        // Determine recommendation
        let recommended_squash = recommend_squash_for_oscillation(squash);
        let recommended_bias_delta = recommend_bias_for_oscillation(positive_fraction, *bias);

        // Estimate improvement: more oscillation and higher activation = more potential gain
        let oscillation_severity = sign_change_fraction * mean_abs_activation;
        let estimated_improvement = oscillation_severity * 0.01;

        candidates.push(OscillatingNeuronCandidate {
            neuron_uuid: uuid.clone(),
            current_squash: squash.clone(),
            sign_change_fraction,
            positive_fraction,
            mean_abs_activation,
            sample_count: records.len(),
            recommended_squash,
            recommended_bias_delta,
            estimated_improvement,
        });
    }

    // Sort by estimated improvement (best first)
    candidates.sort_by(|a, b| {
        b.estimated_improvement
            .partial_cmp(&a.estimated_improvement)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    candidates
}

/// Recommend an activation function to stabilise an oscillating neuron.
///
/// For neurons that oscillate between positive and negative, ABSOLUTE captures
/// the magnitude regardless of sign. For neurons with bounded oscillation, RELU
/// clips the negative side.
fn recommend_squash_for_oscillation(current_squash: &str) -> String {
    let upper = current_squash.to_ascii_uppercase();
    match upper.as_str() {
        // For symmetric functions that naturally produce oscillation, use ABSOLUTE
        "TANH" | "IDENTITY" | "SOFTSIGN" | "ARCTAN" | "HARD_TANH" => "ABSOLUTE".to_string(),
        // For other functions, RELU clips negative side
        _ => "RELU".to_string(),
    }
}

/// Recommend a bias adjustment for an oscillating neuron.
///
/// If activations are more positive than negative, shift bias to reduce positive dominance
/// (and vice versa), helping to centre the neuron's operating point.
fn recommend_bias_for_oscillation(positive_fraction: f32, _current_bias: f32) -> Option<f32> {
    // If substantially more positive than negative, shift negative
    if positive_fraction > 0.6 {
        Some(-0.1)
    } else if positive_fraction < 0.4 {
        Some(0.1)
    } else {
        None // Already fairly balanced
    }
}

/// Convert oscillating neuron candidates into coordinated structural candidates.
///
/// Each oscillating neuron produces a `ChangeSquash` coordinated candidate,
/// optionally combined with a `SetBias` adjustment.
pub fn oscillating_neurons_to_coordinated_candidates(
    candidates: &[OscillatingNeuronCandidate],
) -> Vec<CoordinatedStructuralCandidateJson> {
    let mut results = Vec::with_capacity(candidates.len());

    for c in candidates {
        let mut operations = vec![CoordinatedStructuralOpJson::ChangeSquash {
            neuron_uuid: c.neuron_uuid.clone(),
            squash: c.recommended_squash.clone(),
        }];

        if let Some(delta) = c.recommended_bias_delta {
            operations.push(CoordinatedStructuralOpJson::SetBias {
                neuron_uuid: c.neuron_uuid.clone(),
                bias: delta,
            });
        }

        results.push(CoordinatedStructuralCandidateJson {
            operations,
            expected_creature_score_gain: c.estimated_improvement,
            comment: Some(format!(
                "Oscillating neuron {}: {} (sign change fraction {:.2}, positive fraction {:.2}, mean abs activation {:.4}) → change to {} to stabilise output",
                c.neuron_uuid, c.current_squash, c.sign_change_fraction, c.positive_fraction, c.mean_abs_activation, c.recommended_squash
            )),
        });
    }

    // Sort by expected improvement (best first)
    results.sort_by(|a, b| {
        b.expected_creature_score_gain
            .partial_cmp(&a.expected_creature_score_gain)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    results
}
