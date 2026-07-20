//! Gradient-based discovery module (Issue #421).
//!
//! Computes local gradients (∂error/∂weight) for each synapse and identifies
//! synapses where small weight changes would have a large impact on error.
//! Proposes gradient-directed weight adjustments in the error-reducing direction.
//!
//! See `docs/DISCOVERY_TYPES.md` § "Gradient-Based Synapse Adjustment" for full
//! documentation.
//!
//! ## Detection Criteria
//!
//! A synapse is a gradient candidate if:
//! 1. **High gradient magnitude**: The absolute mean gradient (|∂error/∂weight|)
//!    exceeds a minimum threshold, indicating the synapse has significant
//!    error-reduction potential.
//! 2. **Sufficient samples**: Enough recorded samples for statistical reliability.
//! 3. **Gradient consistency**: The gradient sign is consistent across samples
//!    (measured by the ratio of mean gradient to standard deviation).
//!
//! ## Recommended Actions
//!
//! When a high-gradient synapse is detected, we recommend:
//! - **`SetWeight`**: Adjust the weight by a small step in the gradient descent
//!   direction (`proposed_delta` = -`learning_rate` × gradient).
//!
//! These are emitted as `CoordinatedStructuralCandidateJson` with `SetWeight`
//! operations.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use std::collections::HashMap;

use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson};

// MIN_SAMPLES_FOR_GRADIENT uses MIN_NEURON_SAMPLE_COUNT (Issue #424)
use crate::analysis::constants::MIN_NEURON_SAMPLE_COUNT as MIN_SAMPLES_FOR_GRADIENT;

/// Maximum absolute weight for gradient-based synapse adjustments.
///
/// This is distinct from `MAX_OUTGOING_WEIGHT` (which constrains add-neuron
/// candidate outgoing weights). Synapse weight adjustments operate on existing
/// synapse weights, which can be much larger.
const MAX_GRADIENT_ADJUSTED_WEIGHT: f32 = 10.0;

/// Minimum absolute gradient to consider a synapse as a candidate.
/// Below this, the weight change would have negligible error impact.
const MIN_GRADIENT_MAGNITUDE: f32 = 0.01;

/// Minimum gradient consistency (|mean| / `std_dev`) to ensure the gradient
/// direction is reliable, not just noise.
const MIN_GRADIENT_CONSISTENCY: f32 = 0.3;

/// Learning rate for proposed weight adjustments.
/// Conservative to avoid overshooting — NEAT-AI will validate via ablation.
const LEARNING_RATE: f32 = 0.1;

/// Result of detecting a gradient-based synapse candidate.
#[derive(Debug, Clone)]
pub struct GradientCandidate {
    /// UUID of the source neuron.
    pub from_neuron_uuid: String,
    /// UUID of the target neuron.
    pub to_neuron_uuid: String,
    /// Current weight of the synapse.
    pub current_weight: f32,
    /// Mean gradient (∂error/∂weight) across samples.
    pub mean_gradient: f32,
    /// Standard deviation of the per-sample gradients.
    pub gradient_std_dev: f32,
    /// Gradient consistency ratio (|mean| / `std_dev`).
    pub gradient_consistency: f32,
    /// Proposed weight change (negative gradient direction).
    pub proposed_weight_delta: f32,
    /// Number of samples used for gradient computation.
    pub sample_count: usize,
    /// Estimated creature score improvement from this adjustment.
    pub estimated_improvement: f32,
}

/// Compute the local gradient (∂error/∂weight) for a synapse.
///
/// The gradient of the error with respect to a synapse weight is approximated by
/// the mean of `source_activation × target_error` across matched observations.
///
/// # Arguments
/// * `source_records` - Records for the source neuron (provides activations).
/// * `target_records` - Records for the target neuron (provides errors).
///
/// # Returns
/// * `Some(gradient)` - Mean gradient across paired observations.
/// * `None` - If insufficient paired samples or non-finite result.
pub fn compute_synapse_gradient(
    source_records: &[DiscoverRecord],
    target_records: &[DiscoverRecord],
) -> Option<f32> {
    // Build obs_index → error lookup for target
    let target_error_map: HashMap<u32, f32> = target_records
        .iter()
        .filter_map(|r| r.errors.first().map(|&e| (r.obs_index, e)))
        .collect();

    let mut sum_gradient = 0.0_f32;
    let mut count = 0_usize;

    for source in source_records {
        if !source.activation.is_finite() {
            continue;
        }

        if let Some(&error) = target_error_map.get(&source.obs_index) {
            if !error.is_finite() {
                continue;
            }
            // ∂error/∂weight ≈ source_activation × target_error
            sum_gradient += source.activation * error;
            count += 1;
        }
    }

    if count < MIN_SAMPLES_FOR_GRADIENT {
        return None;
    }

    let mean = sum_gradient / count as f32;
    if !mean.is_finite() {
        return None;
    }

    Some(mean)
}

/// Detect gradient-based synapse adjustment candidates.
///
/// For each synapse targeting an output neuron, computes the local gradient and
/// identifies synapses where a small weight adjustment would significantly reduce
/// error.
///
/// # Arguments
/// * `creature` - The creature's network topology (neurons and synapses).
/// * `neuron_records` - List of `(neuron_uuid, records)` tuples with recorded activations.
///
/// # Returns
/// A list of `GradientCandidate` for high-gradient synapses, sorted by estimated
/// improvement (best first).
pub fn detect_gradient_candidates(
    creature: &CreatureJson,
    neuron_records: &[(String, impl AsRef<[DiscoverRecord]>)],
) -> Vec<GradientCandidate> {
    // Build records lookup
    let records_map: HashMap<&str, &[DiscoverRecord]> = neuron_records
        .iter()
        .map(|(uuid, records)| (uuid.as_str(), records.as_ref()))
        .collect();

    // Build obs_index → error lookup for each neuron
    let target_error_map: HashMap<&str, HashMap<u32, f32>> = neuron_records
        .iter()
        .map(|(uuid, records)| {
            let records = records.as_ref();
            let obs_map: HashMap<u32, f32> = records
                .iter()
                .filter_map(|r| {
                    r.errors
                        .first()
                        .filter(|e| e.is_finite())
                        .map(|&e| (r.obs_index, e))
                })
                .collect();
            (uuid.as_str(), obs_map)
        })
        .collect();

    // Identify output neuron UUIDs (primary targets for gradient analysis)
    let output_uuids: std::collections::HashSet<&str> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "output")
        .map(|n| n.uuid.as_str())
        .collect();

    let mut candidates = Vec::new();

    for synapse in &creature.synapses {
        // Only analyse synapses targeting output neurons (where error is directly measured)
        if !output_uuids.contains(synapse.to_uuid.as_str()) {
            continue;
        }

        // Get source neuron records
        let Some(source_records) = records_map.get(synapse.from_uuid.as_str()) else {
            continue;
        };

        if source_records.len() < MIN_SAMPLES_FOR_GRADIENT {
            continue;
        }

        // Get target error map
        let Some(target_obs_map) = target_error_map.get(synapse.to_uuid.as_str()) else {
            continue;
        };

        // Compute per-sample gradients
        let mut gradients: Vec<f32> = Vec::new();

        for source in *source_records {
            if !source.activation.is_finite() {
                continue;
            }
            if let Some(&error) = target_obs_map.get(&source.obs_index) {
                gradients.push(source.activation * error);
            }
        }

        if gradients.len() < MIN_SAMPLES_FOR_GRADIENT {
            continue;
        }

        let n = gradients.len() as f32;
        let mean_gradient = gradients.iter().sum::<f32>() / n;

        if !mean_gradient.is_finite() {
            continue;
        }

        // Compute standard deviation for consistency check
        let variance = gradients
            .iter()
            .map(|&g| {
                let diff = g - mean_gradient;
                diff * diff
            })
            .sum::<f32>()
            / n;
        let std_dev = variance.sqrt();

        // Check gradient magnitude threshold
        if mean_gradient.abs() < MIN_GRADIENT_MAGNITUDE {
            continue;
        }

        // Check gradient consistency (signal-to-noise ratio)
        let consistency = if std_dev > 1e-12 {
            mean_gradient.abs() / std_dev
        } else {
            // Perfect consistency (all gradients identical)
            f32::MAX
        };

        if consistency < MIN_GRADIENT_CONSISTENCY {
            continue;
        }

        // Propose weight delta in gradient descent direction
        let raw_delta = -LEARNING_RATE * mean_gradient;
        let new_weight = (synapse.weight + raw_delta)
            .clamp(-MAX_GRADIENT_ADJUSTED_WEIGHT, MAX_GRADIENT_ADJUSTED_WEIGHT);
        let effective_delta = new_weight - synapse.weight;

        // Skip if effective delta is negligible
        if effective_delta.abs() < 1e-8 {
            continue;
        }

        // Estimate improvement: gradient magnitude × proposed step size
        // Larger gradients with consistent direction yield better improvement estimates
        let estimated_improvement =
            mean_gradient.abs() * effective_delta.abs() * consistency.min(3.0) * 0.01;

        candidates.push(GradientCandidate {
            from_neuron_uuid: synapse.from_uuid.clone(),
            to_neuron_uuid: synapse.to_uuid.clone(),
            current_weight: synapse.weight,
            mean_gradient,
            gradient_std_dev: std_dev,
            gradient_consistency: consistency,
            proposed_weight_delta: effective_delta,
            sample_count: gradients.len(),
            estimated_improvement,
        });
    }

    // Sort by estimated improvement (best first)
    candidates.sort_by(|a, b| b.estimated_improvement.total_cmp(&a.estimated_improvement));

    candidates
}

/// Convert gradient candidates into coordinated structural candidates.
///
/// Each gradient candidate produces a `SetWeight` operation adjusting the weight
/// by the proposed delta in the gradient descent direction.
pub fn gradient_candidates_to_coordinated(
    candidates: &[GradientCandidate],
) -> Vec<CoordinatedStructuralCandidateJson> {
    let mut results = Vec::with_capacity(candidates.len());

    for c in candidates {
        let new_weight = c.current_weight + c.proposed_weight_delta;

        results.push(CoordinatedStructuralCandidateJson {
            remove_neuron_compensation: None,
            operations: vec![CoordinatedStructuralOpJson::SetWeight {
                from_neuron_uuid: c.from_neuron_uuid.clone(),
                to_neuron_uuid: c.to_neuron_uuid.clone(),
                weight: new_weight,
            }],
            expected_creature_score_gain: c.estimated_improvement,
            comment: Some(format!(
                "Gradient-guided weight adjustment {} \u{2192} {}: gradient {:.4}, consistency {:.2}, weight {:.4} \u{2192} {:.4} (delta {:.4})",
                c.from_neuron_uuid,
                c.to_neuron_uuid,
                c.mean_gradient,
                c.gradient_consistency,
                c.current_weight,
                new_weight,
                c.proposed_weight_delta,
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
