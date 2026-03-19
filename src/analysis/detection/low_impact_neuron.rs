//! Low-impact neuron detection module (Issue #793).
//!
//! Identifies hidden neurons whose activations are consistently near-zero but
//! above the dead-neuron threshold (1e-6). These neurons have negligible impact
//! on the network's output and are good candidates for removal.
//!
//! This module complements `dead_neuron.rs` (Issue #341) by catching neurons in
//! the "twilight zone" between truly dead (< 1e-6) and meaningfully active
//! (> 1e-3). The dead-neuron detector uses a very tight threshold; this module
//! broadens the removal pool with a tiered confidence approach.
//!
//! ## Detection Criteria
//!
//! A neuron is "low-impact" if:
//! 1. **Mean absolute activation** is between the dead threshold (1e-6) and
//!    the low-impact ceiling (1e-3).
//! 2. **Low activation variance** — the neuron is consistently near-zero, not
//!    sporadically spiking.
//! 3. **Only hidden neurons** — output and input neurons are excluded.
//! 4. **Sufficient samples** — at least `MIN_DISCOVERY_SAMPLE_COUNT` records.
//!
//! ## Confidence Scoring
//!
//! Removal confidence is computed from three factors:
//! - **Activation proximity** — lower mean abs activation = higher confidence
//! - **Variance consistency** — lower std dev relative to mean = higher confidence
//! - **Sample sufficiency** — more samples = higher confidence (plateaus at 500)

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use super::helpers::build_record_map;

use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson};

use super::topology_cache::CreatureTopologyCache;

use crate::analysis::constants::MIN_DISCOVERY_SAMPLE_COUNT;

/// Lower bound: activations at or below this are handled by `dead_neuron.rs`.
const DEAD_THRESHOLD: f32 = 1e-6;

/// Upper bound: activations above this are considered meaningfully active.
const LOW_IMPACT_CEILING: f32 = 1e-3;

/// Maximum activation standard deviation relative to mean for low-impact status.
/// If the neuron's output varies too much, it may still be contributing on some
/// samples even if the mean is low.
const MAX_RELATIVE_STD_DEV: f32 = 5.0;

/// Absolute ceiling on standard deviation. Even if the relative std dev is low,
/// a high absolute std dev suggests the neuron is not consistently near-zero.
const MAX_ABSOLUTE_STD_DEV: f32 = 1e-3;

/// Base estimated improvement for removing a low-impact neuron.
/// Lower than dead-neuron removal because there is a small chance the neuron
/// contributes marginally.
const BASE_IMPROVEMENT: f32 = 0.002;

/// Result of detecting a low-impact neuron.
#[derive(Debug, Clone)]
pub struct LowImpactNeuronCandidate {
    /// UUID of the low-impact neuron.
    pub neuron_uuid: String,
    /// Mean absolute activation across all samples.
    pub mean_abs_activation: f32,
    /// Standard deviation of activation across samples.
    pub activation_std_dev: f32,
    /// Number of samples analysed.
    pub sample_count: usize,
    /// Confidence that removing this neuron is safe (0.0 to 1.0).
    pub removal_confidence: f32,
    /// Estimated creature score improvement from removing this neuron.
    pub estimated_improvement: f32,
}

/// Detect low-impact neurons from the creature topology and recorded activations.
///
/// Low-impact neurons have mean absolute activations between the dead threshold
/// (1e-6) and the low-impact ceiling (1e-3), with consistently low variance.
///
/// # Arguments
/// * `creature` - The creature's network topology (neurons and synapses).
/// * `neuron_records` - List of `(neuron_uuid, records)` tuples with recorded activations.
/// * `topo` - Optional pre-computed topology cache (Issue #754).
///
/// # Returns
/// A list of `LowImpactNeuronCandidate` sorted by removal confidence (highest first).
pub fn detect_low_impact_neurons(
    creature: &CreatureJson,
    neuron_records: &[(String, Vec<DiscoverRecord>)],
    topo: Option<&CreatureTopologyCache>,
) -> Vec<LowImpactNeuronCandidate> {
    let local_cache;
    let topo = match topo {
        Some(t) => t,
        None => {
            local_cache = CreatureTopologyCache::new(creature);
            &local_cache
        }
    };

    let records_map = build_record_map(neuron_records);

    let mut candidates = Vec::with_capacity(topo.hidden_uuids.len());

    for uuid in &topo.hidden_uuids {
        let Some(records) = records_map.get(uuid.as_str()) else {
            continue;
        };

        if records.len() < MIN_DISCOVERY_SAMPLE_COUNT {
            continue;
        }

        let n = records.len() as f32;

        // Compute mean absolute activation
        let sum_abs_activation: f32 = records.iter().map(|r| r.activation.abs()).sum();
        let mean_abs_activation = sum_abs_activation / n;

        // Must be above dead threshold and below low-impact ceiling
        if mean_abs_activation <= DEAD_THRESHOLD || mean_abs_activation >= LOW_IMPACT_CEILING {
            continue;
        }

        // Compute activation standard deviation
        let mean_activation: f32 = records.iter().map(|r| r.activation).sum::<f32>() / n;
        let variance: f32 = records
            .iter()
            .map(|r| {
                let diff = r.activation - mean_activation;
                diff * diff
            })
            .sum::<f32>()
            / n;
        let activation_std_dev = variance.sqrt();

        // Reject if variance is too high (neuron may still be useful on some samples)
        if activation_std_dev >= MAX_ABSOLUTE_STD_DEV {
            continue;
        }

        // Also reject if relative std dev is too high (noisy relative to mean)
        if mean_abs_activation > 0.0
            && activation_std_dev / mean_abs_activation > MAX_RELATIVE_STD_DEV
        {
            continue;
        }

        let confidence = compute_low_impact_confidence(mean_abs_activation, activation_std_dev, n);

        let estimated_improvement = BASE_IMPROVEMENT * confidence;

        candidates.push(LowImpactNeuronCandidate {
            neuron_uuid: uuid.clone(),
            mean_abs_activation,
            activation_std_dev,
            sample_count: records.len(),
            removal_confidence: confidence,
            estimated_improvement,
        });
    }

    // Sort by removal confidence (highest first)
    candidates.sort_by(|a, b| b.removal_confidence.total_cmp(&a.removal_confidence));

    candidates
}

/// Compute removal confidence for a low-impact neuron.
///
/// Three factors contribute:
/// - **Activation proximity** (40%): how close to the dead threshold (lower = safer)
/// - **Variance consistency** (30%): how consistent the low activation is
/// - **Sample sufficiency** (30%): how many samples confirm the low impact
fn compute_low_impact_confidence(
    mean_abs_activation: f32,
    activation_std_dev: f32,
    sample_count: f32,
) -> f32 {
    // Activation proximity: log-scale distance from dead threshold to ceiling.
    // Closer to dead threshold → higher factor.
    let log_range = (LOW_IMPACT_CEILING / DEAD_THRESHOLD).ln();
    let log_position = (mean_abs_activation / DEAD_THRESHOLD).ln();
    let activation_factor = 1.0 - (log_position / log_range).clamp(0.0, 1.0);

    // Variance consistency: lower std dev relative to ceiling → higher factor.
    let variance_factor = 1.0 - (activation_std_dev / MAX_ABSOLUTE_STD_DEV).clamp(0.0, 1.0);

    // Sample sufficiency: more samples → higher confidence, plateaus at 500.
    let sample_factor = (sample_count / 500.0).min(1.0);

    // Weighted combination
    let raw = activation_factor * 0.4 + variance_factor * 0.3 + sample_factor * 0.3;

    // Scale to [0.3, 0.9] range — lower than dead-neuron confidence since
    // there is more uncertainty when the neuron is not completely dead.
    0.3 + raw * 0.6
}

/// Convert low-impact neuron candidates into coordinated structural candidates.
///
/// Each low-impact neuron produces a `RemoveNeuron` coordinated candidate.
/// NEAT-AI validates the removal through ablation testing before applying.
pub fn low_impact_neurons_to_coordinated_candidates(
    candidates: &[LowImpactNeuronCandidate],
) -> Vec<CoordinatedStructuralCandidateJson> {
    let mut results = Vec::with_capacity(candidates.len());

    for c in candidates {
        results.push(CoordinatedStructuralCandidateJson {
            operations: vec![CoordinatedStructuralOpJson::RemoveNeuron {
                neuron_uuid: c.neuron_uuid.clone(),
            }],
            expected_creature_score_gain: c.estimated_improvement,
            comment: Some(format!(
                "[#793] Low-impact neuron {}: mean abs activation {:.2e}, \
                 std dev {:.2e}, {} samples, confidence {:.2} \
                 — remove to reduce wasted computation",
                c.neuron_uuid,
                c.mean_abs_activation,
                c.activation_std_dev,
                c.sample_count,
                c.removal_confidence
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
