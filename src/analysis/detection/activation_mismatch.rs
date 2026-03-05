//! Activation mismatch detection module (Issue #543).
//!
//! Detects neurons whose current activation function is poorly matched to their
//! observed input/output patterns, and recommends better alternatives.
//!
//! ## Key differences from `activation_recommendation.rs`
//!
//! `activation_recommendation.rs` (Issue #417) proactively analyses input
//! distributions and recommends optimal activations based on distribution shape.
//! This module detects **structural mismatches** where the activation function
//! actively wastes information:
//!
//! 1. **RELU with negative bias**: The neuron's pre-activation values are
//!    predominantly negative, so RELU clips most of the signal to zero.
//!    Switching to ELU or IDENTITY would preserve the information.
//!
//! 2. **Bounded activation underutilisation**: A bounded activation (TANH,
//!    LOGISTIC) is used but the neuron only operates in a tiny fraction of
//!    its output range, meaning the non-linear properties are wasted.
//!    IDENTITY would be equally effective and computationally cheaper.
//!
//! ## Detection Criteria
//!
//! ### RELU Negative Bias (`ReluNegativeBias`)
//! - Pre-activation (`value`) data available for ≥ `MIN_SAMPLES` records
//! - ≥ `RELU_NEGATIVE_FRACTION_THRESHOLD` of pre-activation values are negative
//! - Recommends ELU (preserves negative information with smooth curve)
//!
//! ### Bounded Underutilisation (`BoundedUnderutilised`)
//! - Activation function has bounded output range (TANH ∈ \[-1,1\], LOGISTIC ∈ \[0,1\])
//! - Observed activation range covers < `UTILISATION_THRESHOLD` of the theoretical range
//! - Recommends IDENTITY (no bounds, preserves full signal)

use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson};

use crate::analysis::constants::MIN_DISCOVERY_SAMPLE_COUNT as MIN_SAMPLES;

/// Fraction of pre-activation values that must be negative to flag RELU mismatch.
const RELU_NEGATIVE_FRACTION_THRESHOLD: f32 = 0.70;

/// Maximum fraction of the bounded activation's theoretical range that the
/// neuron actually uses before being considered underutilised.
const UTILISATION_THRESHOLD: f32 = 0.15;

/// Classification of activation mismatch type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MismatchKind {
    /// RELU-family activation with predominantly negative pre-activation values.
    ReluNegativeBias,
    /// Bounded activation (TANH, LOGISTIC, etc.) using a tiny fraction of its range.
    BoundedUnderutilised,
}

/// Result of detecting an activation mismatch for a single neuron.
#[derive(Debug, Clone)]
pub struct ActivationMismatchCandidate {
    /// UUID of the mismatched neuron.
    pub neuron_uuid: String,
    /// Current activation function.
    pub current_squash: String,
    /// Recommended replacement activation function.
    pub recommended_squash: Option<String>,
    /// Type of mismatch detected.
    pub mismatch_kind: MismatchKind,
    /// Fraction of samples that are clipped or wasted by the current activation.
    pub clipped_fraction: f32,
    /// Estimated improvement from switching activation.
    pub estimated_improvement: f32,
    /// Human-readable explanation.
    pub reason: String,
}

/// Returns whether a squash function belongs to the RELU family (clips at zero).
fn is_relu_family(squash: &str) -> bool {
    matches!(squash, "RELU" | "RELU6")
}

/// Returns the theoretical output range `(min, max)` for bounded activations.
/// Returns `None` for unbounded activations.
fn bounded_range(squash: &str) -> Option<(f32, f32)> {
    match squash {
        "TANH" | "HARD_TANH" | "CLIPPED" | "BIPOLAR" | "BIPOLAR_SIGMOID" => Some((-1.0, 1.0)),
        "LOGISTIC" => Some((0.0, 1.0)),
        "SOFTSIGN" | "ARCTAN" | "ISRU" => Some((-1.0, 1.0)),
        "RELU6" => Some((0.0, 6.0)),
        _ => None,
    }
}

/// Returns whether a squash function is effectively unbounded and should not
/// be flagged as mismatched.
fn is_skip_squash(squash: &str) -> bool {
    matches!(
        squash,
        "IDENTITY" | "ELU" | "SELU" | "LEAKYRELU" | "GELU" | "MISH" | "SOFTPLUS" | "BENTIDENTITY"
    )
}

/// Detect activation mismatches from recorded neuron data.
///
/// # Arguments
/// * `neurons` - List of `(uuid, squash, bias)` tuples for hidden neurons.
/// * `neuron_records` - List of `(uuid, records)` tuples with recorded samples.
///
/// # Returns
/// A list of `ActivationMismatchCandidate` for neurons with detected mismatches,
/// sorted by estimated improvement (descending).
pub fn detect_activation_mismatches(
    neurons: &[(String, String, f32)],
    neuron_records: &[(String, Vec<DiscoverRecord>)],
) -> Vec<ActivationMismatchCandidate> {
    let records_map: std::collections::HashMap<&str, &Vec<DiscoverRecord>> = neuron_records
        .iter()
        .map(|(uuid, records)| (uuid.as_str(), records))
        .collect();

    let mut candidates = Vec::with_capacity(neurons.len());

    for (uuid, squash, _bias) in neurons {
        // Skip activations that are already unbounded / not susceptible
        if is_skip_squash(squash) {
            continue;
        }

        let Some(records) = records_map.get(uuid.as_str()) else {
            continue;
        };

        if records.len() < MIN_SAMPLES {
            continue;
        }

        // Check RELU family for negative bias mismatch
        if is_relu_family(squash)
            && let Some(candidate) = check_relu_negative_bias(uuid, squash, records)
        {
            candidates.push(candidate);
            continue; // Only report one mismatch per neuron
        }

        // Check bounded activations for underutilisation
        if let Some(range) = bounded_range(squash)
            && let Some(candidate) = check_bounded_underutilisation(uuid, squash, records, range)
        {
            candidates.push(candidate);
        }
    }

    candidates.sort_by(|a, b| b.estimated_improvement.total_cmp(&a.estimated_improvement));
    candidates
}

/// Check if a RELU-family neuron has predominantly negative pre-activation values.
fn check_relu_negative_bias(
    uuid: &str,
    squash: &str,
    records: &[DiscoverRecord],
) -> Option<ActivationMismatchCandidate> {
    // We need pre-activation values (stored in `value` field)
    let pre_activations: Vec<f32> = records.iter().filter_map(|r| r.value).collect();

    if pre_activations.len() < MIN_SAMPLES {
        return None;
    }

    let negative_count = pre_activations.iter().filter(|&&v| v < 0.0).count();
    let negative_fraction = negative_count as f32 / pre_activations.len() as f32;

    if negative_fraction < RELU_NEGATIVE_FRACTION_THRESHOLD {
        return None;
    }

    // Estimate improvement: proportional to how much information is being clipped
    let estimated_improvement = (negative_fraction - 0.5) * 0.02;

    Some(ActivationMismatchCandidate {
        neuron_uuid: uuid.to_string(),
        current_squash: squash.to_string(),
        recommended_squash: Some("ELU".to_string()),
        mismatch_kind: MismatchKind::ReluNegativeBias,
        clipped_fraction: negative_fraction,
        estimated_improvement,
        reason: format!(
            "{:.0}% of pre-activation values are negative, clipped to zero by {}",
            negative_fraction * 100.0,
            squash,
        ),
    })
}

/// Check if a bounded activation neuron uses only a tiny fraction of its range.
fn check_bounded_underutilisation(
    uuid: &str,
    squash: &str,
    records: &[DiscoverRecord],
    theoretical_range: (f32, f32),
) -> Option<ActivationMismatchCandidate> {
    let activations: Vec<f32> = records.iter().map(|r| r.activation).collect();

    let observed_min = activations.iter().copied().fold(f32::INFINITY, f32::min);
    let observed_max = activations
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);

    let observed_range = observed_max - observed_min;
    let theoretical = theoretical_range.1 - theoretical_range.0;

    if theoretical <= 0.0 {
        return None;
    }

    let utilisation = observed_range / theoretical;

    if utilisation >= UTILISATION_THRESHOLD {
        return None;
    }

    let estimated_improvement = (UTILISATION_THRESHOLD - utilisation) * 0.01;

    Some(ActivationMismatchCandidate {
        neuron_uuid: uuid.to_string(),
        current_squash: squash.to_string(),
        recommended_squash: Some("IDENTITY".to_string()),
        mismatch_kind: MismatchKind::BoundedUnderutilised,
        clipped_fraction: 1.0 - utilisation,
        estimated_improvement,
        reason: format!(
            "{} output range utilised only {:.1}% — operating in linear region, IDENTITY would be equally effective",
            squash,
            utilisation * 100.0,
        ),
    })
}

/// Convert activation mismatch candidates to coordinated structural candidates.
///
/// Each candidate becomes a `ChangeSquash` operation recommending a better
/// activation function for the neuron.
pub fn activation_mismatch_to_coordinated_candidates(
    candidates: &[ActivationMismatchCandidate],
) -> Vec<CoordinatedStructuralCandidateJson> {
    candidates
        .iter()
        .filter_map(|c| {
            let recommended = c.recommended_squash.as_ref()?;

            Some(CoordinatedStructuralCandidateJson {
                operations: vec![CoordinatedStructuralOpJson::ChangeSquash {
                    neuron_uuid: c.neuron_uuid.clone(),
                    squash: recommended.clone(),
                }],
                expected_creature_score_gain: c.estimated_improvement,
                comment: Some(format!(
                    "Activation mismatch (Issue #543): {} → {}. {}",
                    c.current_squash, recommended, c.reason,
                )),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_record(uuid: &str, idx: u32, value: Option<f32>, activation: f32) -> DiscoverRecord {
        DiscoverRecord {
            obs_index: idx,
            neuron_uuid: uuid.to_string(),
            value,
            activation,
            errors: vec![0.05],
        }
    }

    #[test]
    fn test_is_relu_family() {
        assert!(is_relu_family("RELU"));
        assert!(is_relu_family("RELU6")); // Issue #753: squash pre-normalised to uppercase
        assert!(!is_relu_family("TANH"));
        assert!(!is_relu_family("IDENTITY"));
    }

    #[test]
    fn test_bounded_range_known_activations() {
        assert_eq!(bounded_range("TANH"), Some((-1.0, 1.0)));
        assert_eq!(bounded_range("LOGISTIC"), Some((0.0, 1.0)));
        assert_eq!(bounded_range("RELU6"), Some((0.0, 6.0)));
        assert_eq!(bounded_range("IDENTITY"), None);
        assert_eq!(bounded_range("RELU"), None);
    }

    #[test]
    fn test_skip_squash() {
        assert!(is_skip_squash("IDENTITY"));
        assert!(is_skip_squash("ELU"));
        assert!(is_skip_squash("GELU"));
        assert!(!is_skip_squash("RELU"));
        assert!(!is_skip_squash("TANH"));
    }

    #[test]
    fn test_relu_negative_bias_detection() {
        let records: Vec<DiscoverRecord> = (0..50)
            .map(|i| {
                let pre = if i < 40 { -1.0 } else { 0.5 };
                make_record("h1", i, Some(pre), pre.max(0.0))
            })
            .collect();

        let result = check_relu_negative_bias("h1", "RELU", &records);
        assert!(result.is_some());
        let c = result.unwrap();
        assert!(c.clipped_fraction > 0.7);
    }

    #[test]
    fn test_bounded_underutilisation_detection() {
        let records: Vec<DiscoverRecord> = (0..50)
            .map(|i| {
                let activation = (i as f32 - 25.0) / 500.0; // range ~0.1
                make_record("h2", i, None, activation)
            })
            .collect();

        let result = check_bounded_underutilisation("h2", "TANH", &records, (-1.0, 1.0));
        assert!(result.is_some());
    }
}
