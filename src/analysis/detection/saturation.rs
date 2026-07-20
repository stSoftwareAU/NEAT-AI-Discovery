//! Saturated neuron detection module (Issue #342).
//!
//! Identifies neurons that are permanently saturated (stuck at activation ceiling or floor)
//! and recommends activation function changes or bias adjustments. Saturated neurons pass
//! no gradient information and block learning in their region of the network.
//!
//! See `docs/DISCOVERY_TYPES.md` § "Saturated Neuron Detection" for full documentation.
//!
//! ## Detection Criteria
//!
//! A neuron is "saturated" if:
//! 1. **Activation near bounds**: For TANH, mean activation > 0.95 or < -0.95 across samples.
//! 2. **Low relative variance**: Input varies but output doesn't (activation function is
//!    squashing all variation).
//! 3. **Uses a bounded activation**: Only bounded activations (TANH, LOGISTIC, `HARD_TANH`, etc.)
//!    can saturate. Unbounded activations (RELU, IDENTITY) are excluded, except RELU dead-zone
//!    detection (all activations at zero).
//!
//! ## Recommended Actions
//!
//! When saturation is detected, we recommend:
//! 1. **Change activation function**: Switch from TANH to IDENTITY to restore signal flow.
//! 2. **Adjust bias**: Shift bias to move the neuron's operating point away from saturation.
//!
//! These are emitted as `CoordinatedStructuralCandidateJson` with `ChangeSquash` and/or
//! `SetBias` operations.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use super::activation_properties::{can_have_dead_zone, is_bounded_squash};
use super::helpers::{build_record_map, compute_activation_stats, sort_candidates_by_score_gain};
use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson};

// MIN_SAMPLES_FOR_SATURATION moved to constants.rs (Issue #424)
use crate::analysis::constants::MIN_DISCOVERY_SAMPLE_COUNT as MIN_SAMPLES_FOR_SATURATION;

/// Activation threshold for bounded functions to consider the neuron saturated.
///
/// Issue #417: Lowered from 0.95 to 0.85 to catch "near-saturated" neurons earlier.
/// The change-squash discovery type has an 18.2% success rate but very low volume.
/// Neurons approaching saturation (0.85–0.95) still have reduced gradient flow and
/// benefit from activation function changes before they become fully saturated.
const TANH_SATURATION_THRESHOLD: f32 = 0.85;

/// LOGISTIC saturation thresholds: output near 0 or 1.
///
/// Issue #417: Lowered upper from 0.95 to 0.90, raised lower from 0.05 to 0.10
/// to catch neurons approaching logistic saturation earlier.
const LOGISTIC_UPPER_THRESHOLD: f32 = 0.90;
const LOGISTIC_LOWER_THRESHOLD: f32 = 0.10;

/// `HARD_TANH` / CLIPPED saturation threshold (clamped at exactly ±1.0).
///
/// Issue #417: Lowered from 0.99 to 0.95 to detect near-saturation.
const HARD_TANH_SATURATION_THRESHOLD: f32 = 0.95;

/// Maximum activation standard deviation to confirm saturation.
/// If output varies significantly, the neuron is not truly saturated.
///
/// Issue #417: Raised from 0.05 to 0.08 to accommodate near-saturated neurons
/// which may have slightly more output variance than fully saturated neurons.
const MAX_ACTIVATION_STD_DEV: f32 = 0.08;

/// RELU dead-zone: mean activation is 0.0 (or very close).
const RELU_DEAD_THRESHOLD: f32 = 1e-6;

/// Result of detecting a saturated neuron.
#[derive(Debug, Clone)]
pub struct SaturatedNeuronCandidate {
    /// UUID of the saturated neuron.
    pub neuron_uuid: String,
    /// Current activation function of the neuron.
    pub current_squash: String,
    /// Mean activation across all samples.
    pub mean_activation: f32,
    /// Standard deviation of activation across samples.
    pub activation_std_dev: f32,
    /// Standard deviation of input (pre-activation value) across samples.
    pub input_std_dev: f32,
    /// Recommended new activation function (e.g., "IDENTITY").
    pub recommended_squash: Option<String>,
    /// Recommended bias adjustment to move away from saturation.
    pub recommended_bias_delta: Option<f32>,
    /// Estimated improvement from fixing the saturation.
    pub estimated_improvement: f32,
}

/// Detect saturated neurons from their recorded activations.
///
/// # Arguments
/// * `neurons` - List of `(neuron_uuid, squash, bias)` tuples for hidden neurons to check.
/// * `neuron_records` - List of `(neuron_uuid, records)` tuples with the recorded activations.
///
/// # Returns
/// A list of `SaturatedNeuronCandidate` for neurons that are saturated.
pub fn detect_saturated_neurons(
    neurons: &[(String, String, f32)],
    neuron_records: &[(String, impl AsRef<[DiscoverRecord]>)],
) -> Vec<SaturatedNeuronCandidate> {
    let mut candidates = Vec::with_capacity(neurons.len());

    // Build a map from uuid to records for quick lookup
    let records_map = build_record_map(neuron_records);

    for (uuid, squash, bias) in neurons {
        let Some(records) = records_map.get(uuid.as_str()) else {
            continue;
        };

        if records.len() < MIN_SAMPLES_FOR_SATURATION {
            continue;
        }

        let is_bounded = is_bounded_squash(squash);
        let is_relu_family = can_have_dead_zone(squash);

        if !is_bounded && !is_relu_family {
            continue;
        }

        // Compute activation statistics (Issue #941: shared helper)
        let act_stats = compute_activation_stats(records);
        let mean_activation = act_stats.mean;
        let activation_std_dev = act_stats.std_dev;

        // Compute input (pre-activation) statistics
        let input_std_dev = compute_input_std_dev(records);

        // Check saturation for bounded activations
        if is_bounded {
            let saturated = check_bounded_saturation(squash, mean_activation, activation_std_dev);
            if saturated {
                let (recommended_squash, recommended_bias_delta) =
                    recommend_fix(squash, mean_activation, *bias);

                // Estimate improvement: saturated neurons waste gradient capacity.
                // The improvement is proportional to how severely saturated the neuron is.
                let saturation_severity = compute_saturation_severity(squash, mean_activation);
                let estimated_improvement = saturation_severity * 0.01;

                candidates.push(SaturatedNeuronCandidate {
                    neuron_uuid: uuid.clone(),
                    current_squash: squash.clone(),
                    mean_activation,
                    activation_std_dev,
                    input_std_dev,
                    recommended_squash,
                    recommended_bias_delta,
                    estimated_improvement,
                });
            }
        }

        // Check RELU dead zone
        if is_relu_family
            && mean_activation.abs() < RELU_DEAD_THRESHOLD
            && activation_std_dev < RELU_DEAD_THRESHOLD
        {
            let recommended_bias_delta = Some(0.1_f32); // Small positive bias to revive the neuron

            candidates.push(SaturatedNeuronCandidate {
                neuron_uuid: uuid.clone(),
                current_squash: squash.clone(),
                mean_activation,
                activation_std_dev,
                input_std_dev,
                recommended_squash: Some("IDENTITY".to_string()),
                recommended_bias_delta,
                estimated_improvement: 0.005,
            });
        }
    }

    // Sort by estimated improvement (best first)
    candidates.sort_by(|a, b| b.estimated_improvement.total_cmp(&a.estimated_improvement));

    candidates
}

/// Check if a bounded activation is saturated based on its mean activation and std dev.
fn check_bounded_saturation(squash: &str, mean_activation: f32, activation_std_dev: f32) -> bool {
    if activation_std_dev > MAX_ACTIVATION_STD_DEV {
        return false; // Output still varies — not truly saturated
    }

    match squash {
        "TANH" | "BIPOLAR_SIGMOID" => mean_activation.abs() > TANH_SATURATION_THRESHOLD,
        "LOGISTIC" => {
            !(LOGISTIC_LOWER_THRESHOLD..=LOGISTIC_UPPER_THRESHOLD).contains(&mean_activation)
        }
        "HARD_TANH" | "CLIPPED" => mean_activation.abs() > HARD_TANH_SATURATION_THRESHOLD,
        "BIPOLAR" | "STEP" => {
            // These are inherently binary; saturation means always same output
            activation_std_dev < 1e-6
        }
        "SOFTSIGN" | "ISRU" | "ARCTAN" => {
            // These saturate more slowly, but with very high input they approach bounds
            mean_activation.abs() > TANH_SATURATION_THRESHOLD
        }
        "RELU6" => mean_activation > 5.9 || mean_activation.abs() < RELU_DEAD_THRESHOLD,
        _ => false,
    }
}

/// Compute severity of saturation (0.0 to 1.0).
fn compute_saturation_severity(squash: &str, mean_activation: f32) -> f32 {
    match squash {
        "TANH" | "BIPOLAR_SIGMOID" | "SOFTSIGN" | "ISRU" | "ARCTAN" => {
            // How far past the threshold are we? (range: 0.95 to 1.0 → 0.0 to 1.0)
            let excess = (mean_activation.abs() - TANH_SATURATION_THRESHOLD).max(0.0);
            (excess / (1.0 - TANH_SATURATION_THRESHOLD)).min(1.0)
        }
        "LOGISTIC" => {
            if mean_activation > 0.5 {
                let excess = (mean_activation - LOGISTIC_UPPER_THRESHOLD).max(0.0);
                (excess / (1.0 - LOGISTIC_UPPER_THRESHOLD)).min(1.0)
            } else {
                let excess = (LOGISTIC_LOWER_THRESHOLD - mean_activation).max(0.0);
                (excess / LOGISTIC_LOWER_THRESHOLD).min(1.0)
            }
        }
        "HARD_TANH" | "CLIPPED" => {
            if mean_activation.abs() >= 1.0 {
                1.0
            } else {
                let excess = (mean_activation.abs() - HARD_TANH_SATURATION_THRESHOLD).max(0.0);
                (excess / (1.0 - HARD_TANH_SATURATION_THRESHOLD)).min(1.0)
            }
        }
        _ => 0.5, // Default moderate severity
    }
}

/// Recommend a fix for a saturated neuron.
///
/// Returns `(recommended_squash, recommended_bias_delta)`.
fn recommend_fix(
    squash: &str,
    mean_activation: f32,
    current_bias: f32,
) -> (Option<String>, Option<f32>) {
    // Recommend IDENTITY as the replacement for heavily saturated bounded activations.
    // IDENTITY restores full signal flow without bounds.
    let recommended_squash = match squash {
        "TANH" | "LOGISTIC" | "HARD_TANH" | "CLIPPED" | "BIPOLAR_SIGMOID" | "SOFTSIGN" | "ISRU"
        | "ARCTAN" => Some("IDENTITY".to_string()),
        _ => None,
    };

    // Compute bias adjustment: move the operating point away from saturation.
    // For positive saturation, reduce bias (shift input toward negative).
    // For negative saturation, increase bias (shift input toward positive).
    let recommended_bias_delta = match squash {
        "TANH" | "HARD_TANH" | "CLIPPED" | "BIPOLAR_SIGMOID" | "SOFTSIGN" | "ISRU" | "ARCTAN" => {
            if mean_activation > 0.0 {
                // Positive saturation: shift input negative
                Some(-current_bias.abs().max(1.0))
            } else {
                // Negative saturation: shift input positive
                Some(current_bias.abs().max(1.0))
            }
        }
        "LOGISTIC" => {
            if mean_activation > 0.5 {
                Some(-current_bias.abs().max(1.0))
            } else {
                Some(current_bias.abs().max(1.0))
            }
        }
        _ => None,
    };

    (recommended_squash, recommended_bias_delta)
}

/// Compute the standard deviation of pre-activation (value) inputs.
fn compute_input_std_dev(records: &[DiscoverRecord]) -> f32 {
    let values: Vec<f32> = records.iter().filter_map(|r| r.value).collect();
    if values.len() < 2 {
        return 0.0;
    }

    let n = values.len() as f32;
    let mean: f32 = values.iter().sum::<f32>() / n;
    let variance: f32 = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    variance.sqrt()
}

/// Convert saturated neuron candidates into coordinated structural candidates.
///
/// Each saturated neuron may produce one or two candidates:
/// 1. A `ChangeSquash` operation (if a new activation is recommended)
/// 2. A `SetBias` operation (if a bias adjustment is recommended)
///
/// These are combined into a single coordinated candidate per neuron so the
/// changes are applied atomically.
pub fn saturated_neurons_to_coordinated_candidates(
    candidates: &[SaturatedNeuronCandidate],
) -> Vec<CoordinatedStructuralCandidateJson> {
    let mut results = Vec::with_capacity(candidates.len() * 2);

    for c in candidates {
        // Candidate 1: Change activation function (primary recommendation)
        if let Some(ref new_squash) = c.recommended_squash {
            let mut operations = vec![CoordinatedStructuralOpJson::ChangeSquash {
                neuron_uuid: c.neuron_uuid.clone(),
                squash: new_squash.clone(),
            }];

            // If we also have a bias adjustment, include it in the same coordinated group
            if let Some(delta) = c.recommended_bias_delta {
                operations.push(CoordinatedStructuralOpJson::SetBias {
                    neuron_uuid: c.neuron_uuid.clone(),
                    bias: delta, // Note: this is the delta; the caller should add to current bias
                });
            }

            results.push(CoordinatedStructuralCandidateJson {
                remove_neuron_compensation: None,
                operations,
                expected_creature_score_gain: c.estimated_improvement,
                comment: Some(format!(
                    "Saturated neuron {}: {} (mean activation {:.3}, std dev {:.4}) → change to {} to restore signal flow",
                    c.neuron_uuid, c.current_squash, c.mean_activation, c.activation_std_dev, new_squash
                )),
            });
        }

        // Candidate 2: Bias adjustment only (alternative to changing activation)
        if let Some(delta) = c.recommended_bias_delta {
            results.push(CoordinatedStructuralCandidateJson {
                remove_neuron_compensation: None,
                operations: vec![CoordinatedStructuralOpJson::SetBias {
                    neuron_uuid: c.neuron_uuid.clone(),
                    bias: delta,
                }],
                expected_creature_score_gain: c.estimated_improvement * 0.5, // Less confident than squash change
                comment: Some(format!(
                    "Saturated neuron {}: {} (mean activation {:.3}) → adjust bias by {:.3} to move to active region",
                    c.neuron_uuid, c.current_squash, c.mean_activation, delta
                )),
            });
        }
    }

    // Sort by expected improvement (best first) (Issue #941: shared helper)
    sort_candidates_by_score_gain(&mut results);

    results
}
