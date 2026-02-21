//! Weight magnitude reset detection module (Issue #550).
//!
//! Identifies synapses that are stuck in a local minimum — the current weight
//! is locally optimal but a dramatically different weight would be globally better.
//! Incremental weight adjustments cannot cross the error barrier between the local
//! and global optimum, so we generate exploratory `setWeight` candidates with
//! large magnitude changes (sign flip, order of magnitude shift) to escape the
//! local basin.
//!
//! ## Detection Criteria
//!
//! A synapse is "stuck" when:
//! 1. The target neuron has persistently high error (mean abs error above threshold)
//! 2. The error is tightly clustered (low coefficient of variation — plateau-like)
//! 3. The source neuron has active (non-negligible) activations, meaning the synapse
//!    contributes meaningfully to the target
//!
//! ## Recommended Actions
//!
//! For each stuck synapse, the module generates multiple `setWeight` candidates
//! with exploratory weights — sign flips, zero, and scaled magnitudes — to probe
//! beyond the local basin. The NEAT-AI validation framework naturally handles
//! this by only accepting changes that improve the score.

use std::collections::HashMap;

use crate::analysis::constants::MIN_DISCOVERY_SAMPLE_COUNT as MIN_SAMPLES;
use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson};

/// Minimum mean absolute error on the target neuron to consider a synapse stuck.
/// Below this, the synapse is performing well enough.
const MIN_STUCK_ERROR: f32 = 0.1;

/// Maximum coefficient of variation (std_dev / mean) for the target error.
/// Low CV means the error is tightly clustered — a plateau in the error landscape.
const MAX_ERROR_CV: f32 = 0.4;

/// Minimum mean absolute activation of the source neuron.
/// The source must be actively contributing signal for the synapse to matter.
const MIN_SOURCE_ACTIVATION: f32 = 0.01;

/// A detected stuck synapse candidate.
#[derive(Debug, Clone)]
pub struct StuckSynapseCandidate {
    /// UUID of the source neuron.
    pub from_neuron_uuid: String,
    /// UUID of the target neuron.
    pub to_neuron_uuid: String,
    /// Current synapse weight.
    pub current_weight: f32,
    /// Mean absolute error of the target neuron.
    pub target_mean_error: f32,
    /// Coefficient of variation of the target error.
    pub target_error_cv: f32,
    /// Mean absolute activation of the source neuron.
    pub source_mean_activation: f32,
    /// Estimated improvement from resetting the weight.
    pub estimated_improvement: f32,
    /// Exploratory weight values to try.
    pub exploratory_weights: Vec<f32>,
}

/// Generate exploratory weight candidates for a stuck synapse.
///
/// Produces weights that are dramatically different from the current weight:
/// - Sign flip (negative of current weight)
/// - Zero (remove the synapse's contribution entirely)
/// - Scaled magnitudes (2×, 0.5×, 0.1× the current weight)
/// - Fixed exploratory values (-1.0, 1.0)
fn generate_exploratory_weights(current_weight: f32) -> Vec<f32> {
    let mut weights = Vec::new();

    // Sign flip — reverse the synapse's direction
    let flipped = -current_weight;
    if flipped.is_finite() && flipped.abs() > 1e-6 {
        weights.push(flipped);
    }

    // Zero — completely silence this synapse
    weights.push(0.0);

    // Doubled magnitude
    let doubled = current_weight * 2.0;
    if doubled.is_finite() {
        weights.push(doubled);
    }

    // Half magnitude
    let halved = current_weight * 0.5;
    if halved.is_finite() {
        weights.push(halved);
    }

    // Tenth magnitude (order of magnitude reduction)
    let tenth = current_weight * 0.1;
    if tenth.is_finite() {
        weights.push(tenth);
    }

    // Fixed exploratory values if they differ meaningfully from current
    for &fixed in &[-1.0_f32, 1.0] {
        if (fixed - current_weight).abs() > 0.2 {
            weights.push(fixed);
        }
    }

    // Deduplicate: remove weights that are too close to each other
    weights.sort_by(f32::total_cmp);
    weights.dedup_by(|a, b| (*a - *b).abs() < 1e-4);

    // Remove the current weight if it accidentally ended up in the list
    weights.retain(|w| (w - current_weight).abs() > 1e-4);

    weights
}

/// Detect synapses stuck in a local minimum that would benefit from a weight
/// magnitude reset.
///
/// # Arguments
/// * `creature` - The creature topology (neurons and synapses).
/// * `neuron_records` - Slice of `(uuid, records)` pairs with observation data.
///
/// # Returns
/// Vector of stuck synapse candidates, sorted by estimated improvement (best first).
pub fn detect_stuck_synapse_weight_resets(
    creature: &CreatureJson,
    neuron_records: &[(String, Vec<DiscoverRecord>)],
) -> Vec<StuckSynapseCandidate> {
    // Build records lookup
    let records_map: HashMap<&str, &Vec<DiscoverRecord>> = neuron_records
        .iter()
        .map(|(uuid, records)| (uuid.as_str(), records))
        .collect();

    let mut candidates = Vec::new();

    for syn in &creature.synapses {
        // Get source neuron records
        let Some(source_records) = records_map.get(syn.from_uuid.as_str()) else {
            continue;
        };

        if source_records.len() < MIN_SAMPLES {
            continue;
        }

        // Get target neuron records
        let Some(target_records) = records_map.get(syn.to_uuid.as_str()) else {
            continue;
        };

        if target_records.len() < MIN_SAMPLES {
            continue;
        }

        // Check source activation — must be actively contributing
        let source_mean_abs_activation: f32 = source_records
            .iter()
            .map(|r| r.activation.abs())
            .sum::<f32>()
            / source_records.len() as f32;

        if source_mean_abs_activation < MIN_SOURCE_ACTIVATION {
            continue;
        }

        // Compute target error statistics
        let target_errors: Vec<f32> = target_records
            .iter()
            .map(|r| r.errors.first().copied().unwrap_or(0.0).abs())
            .collect();

        let n = target_errors.len() as f32;
        let mean_error: f32 = target_errors.iter().sum::<f32>() / n;

        // Skip if error is already low (converged)
        if mean_error < MIN_STUCK_ERROR {
            continue;
        }

        // Compute coefficient of variation
        let variance: f32 = target_errors
            .iter()
            .map(|e| (e - mean_error).powi(2))
            .sum::<f32>()
            / n;
        let std_dev = variance.sqrt();
        let cv = if mean_error > 1e-6 {
            std_dev / mean_error
        } else {
            f32::INFINITY
        };

        // Stuck = high error + low variance (plateau in error landscape)
        if cv > MAX_ERROR_CV {
            continue;
        }

        // Compute sensitivity: how much the synapse contributes to the target
        // sensitivity ≈ mean(|source_activation × weight|)
        let sensitivity = source_mean_abs_activation * syn.weight.abs();

        // Generate exploratory weights
        let exploratory_weights = generate_exploratory_weights(syn.weight);
        if exploratory_weights.is_empty() {
            continue;
        }

        // Estimated improvement: proportional to error magnitude, plateau tightness,
        // and synapse sensitivity
        let plateau_tightness = (1.0 - cv / MAX_ERROR_CV).max(0.0);
        let sensitivity_factor = sensitivity.min(1.0);
        let estimated_improvement = mean_error * plateau_tightness * sensitivity_factor * 0.15;

        if estimated_improvement <= 0.0 {
            continue;
        }

        candidates.push(StuckSynapseCandidate {
            from_neuron_uuid: syn.from_uuid.clone(),
            to_neuron_uuid: syn.to_uuid.clone(),
            current_weight: syn.weight,
            target_mean_error: mean_error,
            target_error_cv: cv,
            source_mean_activation: source_mean_abs_activation,
            estimated_improvement,
            exploratory_weights,
        });
    }

    // Sort by estimated improvement descending
    candidates.sort_by(|a, b| {
        b.estimated_improvement
            .partial_cmp(&a.estimated_improvement)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    candidates
}

/// Convert stuck synapse candidates to coordinated structural candidates.
///
/// Each stuck synapse produces multiple `setWeight` candidates — one per
/// exploratory weight value — so the NEAT-AI validation framework can test
/// which dramatic weight change (if any) escapes the local minimum.
pub fn stuck_synapses_to_coordinated_candidates(
    candidates: &[StuckSynapseCandidate],
) -> Vec<CoordinatedStructuralCandidateJson> {
    let mut results = Vec::new();

    for c in candidates {
        for &new_weight in &c.exploratory_weights {
            let direction = if (new_weight - c.current_weight).abs() < 1e-6 {
                "unchanged".to_string()
            } else if new_weight.signum() != c.current_weight.signum()
                && c.current_weight.abs() > 1e-6
            {
                "sign flip".to_string()
            } else if new_weight.abs() < 1e-6 {
                "zero reset".to_string()
            } else {
                let ratio = new_weight / c.current_weight;
                format!("{ratio:.1}× scale")
            };

            results.push(CoordinatedStructuralCandidateJson {
                operations: vec![CoordinatedStructuralOpJson::SetWeight {
                    from_neuron_uuid: c.from_neuron_uuid.clone(),
                    to_neuron_uuid: c.to_neuron_uuid.clone(),
                    weight: new_weight,
                }],
                expected_creature_score_gain: c.estimated_improvement,
                comment: Some(format!(
                    "Stuck synapse {} → {}: weight {:.3} → {:.3} ({direction}). \
                     Target mean error: {:.3}, CV: {:.3}. \
                     Exploratory reset to escape local minimum. (Issue #550)",
                    c.from_neuron_uuid,
                    c.to_neuron_uuid,
                    c.current_weight,
                    new_weight,
                    c.target_mean_error,
                    c.target_error_cv,
                )),
            });
        }
    }

    results
}
