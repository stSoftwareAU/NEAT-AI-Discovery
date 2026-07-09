//! Unbounded activation capping detection module (Issue #441).
//!
//! Identifies neurons with unbounded activation functions (RELU, IDENTITY, LEAKYRELU, etc.)
//! that are producing high activations ("spiking") and would benefit from being capped
//! with a bounded version (e.g., RELU → RELU6).
//!
//! See `docs/DISCOVERY_TYPES.md` § "Unbounded Capping Detection" for full documentation.
//!
//! ## The Problem
//!
//! Unbounded activations like RELU can produce arbitrarily high values. When a neuron
//! consistently outputs very high activations, it may be introducing noise into the
//! network. Capping these activations with RELU6 (or similar bounded activation) can
//! reduce this noise and improve the creature's score.
//!
//! ## Detection Criteria
//!
//! A neuron is a candidate for capping if:
//! 1. **Uses an unbounded activation**: RELU, IDENTITY, LEAKYRELU, SOFTPLUS, ELU, SELU, etc.
//! 2. **Has high activations**: Max activation exceeds the capping threshold (e.g., > 6.0).
//! 3. **Consistent spiking**: Significant fraction of samples exceed the threshold.
//! 4. **Hidden neurons only**: Output neurons are excluded.
//!
//! ## Recommended Actions
//!
//! When unbounded capping is detected, we recommend:
//! 1. **Change RELU → RELU6**: Cap activations at 6.0.
//! 2. **Change LEAKYRELU → RELU6**: Cap activations (loses negative leak, but caps positive).
//! 3. **Change IDENTITY → `HARD_TANH`**: Cap activations at ±1.0.
//! 4. **Optionally adjust weights**: Scale down incoming weights to reduce activation magnitude.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use super::helpers::build_record_map;
use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson};

// MIN_SAMPLES_FOR_DETECTION moved to constants.rs (Issue #424)
use crate::analysis::constants::MIN_DISCOVERY_SAMPLE_COUNT as MIN_SAMPLES_FOR_DETECTION;

/// Default capping threshold for RELU → RELU6 transition.
/// Activations above this value would be clipped by RELU6.
const RELU6_CAP_THRESHOLD: f32 = 6.0;

/// Minimum fraction of samples that must exceed the cap threshold to trigger detection.
/// We require at least 30% of samples to be above threshold to avoid false positives
/// from occasional spikes.
const MIN_FRACTION_ABOVE_CAP: f32 = 0.3;

/// Result of detecting an unbounded capping candidate.
#[derive(Debug, Clone)]
pub struct UnboundedCappingCandidate {
    /// UUID of the neuron.
    pub neuron_uuid: String,
    /// Current activation function of the neuron.
    pub current_squash: String,
    /// Maximum activation observed across all samples.
    pub max_activation: f32,
    /// Mean activation across all samples.
    pub mean_activation: f32,
    /// Fraction of samples that exceed the capping threshold.
    pub fraction_above_cap: f32,
    /// Number of samples analysed.
    pub sample_count: usize,
    /// Recommended new activation function (e.g., "RELU6").
    pub recommended_squash: Option<String>,
    /// Estimated creature score improvement from capping.
    pub estimated_improvement: f32,
}

/// Returns whether a squash function is unbounded (can produce arbitrarily high values).
fn is_unbounded_squash(squash: &str) -> bool {
    matches!(
        squash,
        "RELU"
            | "IDENTITY"
            | "LEAKYRELU"
            | "SOFTPLUS"
            | "ELU"
            | "SELU"
            | "SWISH"
            | "MISH"
            | "GELU"
            | "EXPONENTIAL"
            | "SQUARE"
            | "CUBE"
    )
}

/// Returns the appropriate capping threshold for a given activation function.
fn get_capping_threshold(squash: &str) -> f32 {
    match squash {
        // ReLU family: use RELU6 threshold
        "RELU" | "LEAKYRELU" | "ELU" | "SELU" | "GELU" | "SWISH" | "MISH" | "SOFTPLUS" => {
            RELU6_CAP_THRESHOLD
        }
        // Identity: use HARD_TANH threshold
        "IDENTITY" => 1.0,
        // Others: use a generous default
        _ => RELU6_CAP_THRESHOLD,
    }
}

/// Returns the recommended bounded activation function for a given unbounded activation.
fn recommend_bounded_squash(squash: &str, mean_activation: f32) -> Option<String> {
    match squash {
        // ReLU family → RELU6
        "RELU" | "LEAKYRELU" | "ELU" | "SELU" | "GELU" | "SWISH" | "MISH" | "SOFTPLUS" => {
            Some("RELU6".to_string())
        }
        // Identity with high positive activations → RELU6
        // Identity with mixed activations → HARD_TANH
        "IDENTITY" => {
            if mean_activation > 0.0 {
                Some("RELU6".to_string())
            } else {
                Some("HARD_TANH".to_string())
            }
        }
        // Exponential is a special case - very aggressive growth
        "EXPONENTIAL" => Some("SOFTPLUS".to_string()),
        // Square/Cube → RELU6 to cap growth
        "SQUARE" | "CUBE" => Some("RELU6".to_string()),
        _ => None,
    }
}

/// Detect neurons with unbounded activations that would benefit from capping.
///
/// # Arguments
/// * `neurons` - List of `(neuron_uuid, squash, bias)` tuples for hidden neurons to check.
/// * `neuron_records` - List of `(neuron_uuid, records)` tuples with the recorded activations.
///
/// # Returns
/// A list of `UnboundedCappingCandidate` for neurons that would benefit from capping,
/// sorted by estimated improvement (best first).
pub fn detect_unbounded_capping_candidates(
    neurons: &[(String, String, f32)],
    neuron_records: &[(String, impl AsRef<[DiscoverRecord]>)],
) -> Vec<UnboundedCappingCandidate> {
    let mut candidates = Vec::with_capacity(neurons.len());

    // Build a map from uuid to records for quick lookup
    let records_map = build_record_map(neuron_records);

    for (uuid, squash, _bias) in neurons {
        // Skip bounded activations
        if !is_unbounded_squash(squash) {
            continue;
        }

        let Some(records) = records_map.get(uuid.as_str()) else {
            continue;
        };

        if records.len() < MIN_SAMPLES_FOR_DETECTION {
            continue;
        }

        let n = records.len() as f32;
        let cap_threshold = get_capping_threshold(squash);

        // Compute activation statistics
        let sum_activation: f32 = records.iter().map(|r| r.activation).sum();
        let mean_activation = sum_activation / n;

        let max_activation = records
            .iter()
            .map(|r| r.activation)
            .fold(f32::NEG_INFINITY, f32::max);

        // Count samples exceeding the cap threshold
        let count_above_cap = records
            .iter()
            .filter(|r| r.activation > cap_threshold)
            .count();
        let fraction_above_cap = count_above_cap as f32 / n;

        // Skip if not enough samples exceed the threshold
        if fraction_above_cap < MIN_FRACTION_ABOVE_CAP {
            continue;
        }

        // Skip if max activation doesn't exceed threshold
        if max_activation <= cap_threshold {
            continue;
        }

        let recommended_squash = recommend_bounded_squash(squash, mean_activation);

        // Estimate improvement based on how much is being "lost" to high activations.
        // Higher fraction above cap and higher excess = higher potential improvement.
        let excess_ratio = (max_activation - cap_threshold) / cap_threshold;
        let estimated_improvement = fraction_above_cap * excess_ratio.min(1.0) * 0.01;

        candidates.push(UnboundedCappingCandidate {
            neuron_uuid: uuid.clone(),
            current_squash: squash.clone(),
            max_activation,
            mean_activation,
            fraction_above_cap,
            sample_count: records.len(),
            recommended_squash,
            estimated_improvement,
        });
    }

    // Sort by estimated improvement (best first)
    candidates.sort_by(|a, b| b.estimated_improvement.total_cmp(&a.estimated_improvement));

    candidates
}

/// Convert unbounded capping candidates into coordinated structural candidates.
///
/// Each candidate produces a `ChangeSquash` operation to switch to the bounded activation.
pub fn unbounded_capping_to_coordinated_candidates(
    candidates: &[UnboundedCappingCandidate],
) -> Vec<CoordinatedStructuralCandidateJson> {
    let mut results = Vec::with_capacity(candidates.len());

    for c in candidates {
        let Some(ref new_squash) = c.recommended_squash else {
            continue;
        };

        results.push(CoordinatedStructuralCandidateJson {
            operations: vec![CoordinatedStructuralOpJson::ChangeSquash {
                neuron_uuid: c.neuron_uuid.clone(),
                squash: new_squash.clone(),
            }],
            expected_creature_score_gain: c.estimated_improvement,
            comment: Some(format!(
                "Unbounded capping: {} {} (max activation {:.2}, {:.0}% above cap {:.1}) → change to {} to reduce noise",
                c.neuron_uuid,
                c.current_squash,
                c.max_activation,
                c.fraction_above_cap * 100.0,
                get_capping_threshold(&c.current_squash),
                new_squash
            )),
        });
    }

    // Sort by expected improvement (best first)
    results.sort_by(|a, b| {
        b.expected_creature_score_gain
            .total_cmp(&a.expected_creature_score_gain)
    });

    results
}
