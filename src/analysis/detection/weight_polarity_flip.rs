//! Weight polarity flip detection module (Issue #644).
//!
//! Detects synapses where the gradient sign is consistently opposite to the
//! current weight sign. When the gradient magnitude is large relative to the
//! weight, a direct sign inversion (`setWeight` to the negated value) can skip
//! the many small delta steps needed to cross zero.
//!
//! ## Detection Criteria
//!
//! A synapse is a polarity flip candidate when:
//! 1. **Sign disagreement**: The weight sign is opposite to the gradient descent
//!    direction (i.e., weight and gradient have the same sign — gradient descent
//!    would push weight through zero).
//! 2. **High gradient consistency**: The gradient direction is reliable across
//!    samples (|mean| / `std_dev` exceeds threshold).
//! 3. **Significant weight magnitude**: The weight is far enough from zero that
//!    flipping the sign represents a meaningful structural change.
//! 4. **Gradient magnitude relative to weight**: The gradient is large enough
//!    relative to the weight that many small steps would be needed to cross zero.
//!
//! ## Recommended Actions
//!
//! For each detected synapse, the module produces a `setWeight` candidate with
//! the negated weight value. This is distinct from the small-delta adjustments
//! proposed by `gradient_discovery.rs`.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use std::collections::HashMap;

use super::helpers::build_record_map;
use crate::analysis::constants::MIN_NEURON_SAMPLE_COUNT as MIN_SAMPLES_FOR_GRADIENT;
use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson};

/// Minimum absolute weight to consider for polarity flip.
/// Weights near zero have no meaningful polarity to invert.
const MIN_WEIGHT_MAGNITUDE: f32 = 0.1;

/// Minimum gradient consistency (|mean| / `std_dev`) for reliable direction.
/// Higher than `gradient_discovery` because a polarity flip is a large change
/// and requires stronger evidence.
const MIN_FLIP_CONSISTENCY: f32 = 0.5;

/// Minimum absolute gradient magnitude.
/// The gradient must be strong enough to indicate a clear directional signal.
const MIN_GRADIENT_MAGNITUDE: f32 = 0.01;

/// Minimum ratio of gradient magnitude to weight magnitude.
/// Ensures the gradient is large relative to the weight, meaning many small
/// steps would be needed to cross zero without a direct flip.
const MIN_GRADIENT_WEIGHT_RATIO: f32 = 0.05;

/// A detected polarity flip candidate.
#[derive(Debug, Clone)]
pub struct PolarityFlipCandidate {
    /// UUID of the source neuron.
    pub from_neuron_uuid: String,
    /// UUID of the target neuron.
    pub to_neuron_uuid: String,
    /// Current synapse weight.
    pub current_weight: f32,
    /// Mean gradient (∂error/∂weight) across samples.
    pub mean_gradient: f32,
    /// Standard deviation of per-sample gradients.
    pub gradient_std_dev: f32,
    /// Gradient consistency ratio (|mean| / `std_dev`).
    pub gradient_consistency: f32,
    /// Number of samples used for gradient computation.
    pub sample_count: usize,
    /// Estimated improvement from flipping the weight polarity.
    pub estimated_improvement: f32,
}

/// Detect synapses where a weight polarity flip would be beneficial.
///
/// Scans synapses targeting output neurons, computes per-sample gradients, and
/// identifies cases where the gradient consistently points opposite to the
/// current weight sign — indicating that the weight should cross zero.
///
/// # Arguments
/// * `creature` - The creature topology (neurons and synapses).
/// * `neuron_records` - Slice of `(uuid, records)` pairs with observation data.
///
/// # Returns
/// Vector of polarity flip candidates, sorted by estimated improvement (best first).
pub fn detect_weight_polarity_flip_candidates(
    creature: &CreatureJson,
    neuron_records: &[(String, impl AsRef<[DiscoverRecord]>)],
) -> Vec<PolarityFlipCandidate> {
    // Build records lookup
    let records_map = build_record_map(neuron_records);

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

    // Identify output neuron UUIDs
    let output_uuids: std::collections::HashSet<&str> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "output")
        .map(|n| n.uuid.as_str())
        .collect();

    let mut candidates = Vec::with_capacity(creature.synapses.len());

    for synapse in &creature.synapses {
        // Only analyse synapses targeting output neurons
        if !output_uuids.contains(synapse.to_uuid.as_str()) {
            continue;
        }

        // Skip near-zero weights — no meaningful polarity to flip
        if synapse.weight.abs() < MIN_WEIGHT_MAGNITUDE {
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

        if !mean_gradient.is_finite() || mean_gradient.abs() < MIN_GRADIENT_MAGNITUDE {
            continue;
        }

        // Compute standard deviation
        let variance = gradients
            .iter()
            .map(|&g| {
                let diff = g - mean_gradient;
                diff * diff
            })
            .sum::<f32>()
            / n;
        let std_dev = variance.sqrt();

        // Check gradient consistency
        let consistency = if std_dev > 1e-12 {
            mean_gradient.abs() / std_dev
        } else {
            f32::MAX
        };

        if consistency < MIN_FLIP_CONSISTENCY {
            continue;
        }

        // Check sign disagreement: weight and gradient have the same sign means
        // gradient descent (which moves in -gradient direction) pushes weight
        // through zero. This is the polarity flip case.
        let weight_sign = synapse.weight.signum();
        let gradient_sign = mean_gradient.signum();

        if weight_sign != gradient_sign {
            // Weight and gradient have opposite signs — gradient descent pushes
            // weight further from zero in the same direction. No flip needed.
            continue;
        }

        // Check gradient magnitude relative to weight
        let gradient_weight_ratio = mean_gradient.abs() / synapse.weight.abs();
        if gradient_weight_ratio < MIN_GRADIENT_WEIGHT_RATIO {
            continue;
        }

        // Estimate improvement: proportional to gradient magnitude, weight magnitude,
        // and consistency. A large gradient on a large weight with high consistency
        // suggests a strong polarity mismatch.
        let flipped_weight = -synapse.weight;
        let weight_change = (flipped_weight - synapse.weight).abs();
        let estimated_improvement =
            mean_gradient.abs() * weight_change * consistency.min(3.0) * 0.005;

        if estimated_improvement <= 0.0 {
            continue;
        }

        candidates.push(PolarityFlipCandidate {
            from_neuron_uuid: synapse.from_uuid.clone(),
            to_neuron_uuid: synapse.to_uuid.clone(),
            current_weight: synapse.weight,
            mean_gradient,
            gradient_std_dev: std_dev,
            gradient_consistency: consistency,
            sample_count: gradients.len(),
            estimated_improvement,
        });
    }

    // Sort by estimated improvement (best first)
    candidates.sort_by(|a, b| b.estimated_improvement.total_cmp(&a.estimated_improvement));

    candidates
}

/// Convert polarity flip candidates into coordinated structural candidates.
///
/// Each candidate produces a `SetWeight` operation with the negated weight value.
pub fn polarity_flip_candidates_to_coordinated(
    candidates: &[PolarityFlipCandidate],
) -> Vec<CoordinatedStructuralCandidateJson> {
    let mut results = Vec::with_capacity(candidates.len());

    for c in candidates {
        let negated_weight = -c.current_weight;

        results.push(CoordinatedStructuralCandidateJson {
            operations: vec![CoordinatedStructuralOpJson::SetWeight {
                from_neuron_uuid: c.from_neuron_uuid.clone(),
                to_neuron_uuid: c.to_neuron_uuid.clone(),
                weight: negated_weight,
            }],
            expected_creature_score_gain: c.estimated_improvement,
            comment: Some(format!(
                "Polarity flip {} \u{2192} {}: gradient {:.4} opposes weight {:.4}, \
                 consistency {:.2} ({} samples). Flip to {:.4}. (Issue #644)",
                c.from_neuron_uuid,
                c.to_neuron_uuid,
                c.mean_gradient,
                c.current_weight,
                c.gradient_consistency,
                c.sample_count,
                negated_weight,
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
