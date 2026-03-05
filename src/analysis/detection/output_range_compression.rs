//! Output neuron range compression detection module (Issue #645).
//!
//! Detects output neurons operating in a compressed sub-range of their
//! activation function's theoretical output domain. For example, a TANH
//! output neuron (range [-1, 1]) whose activations cluster in [0.3, 0.7]
//! is using only 20% of the available range, reducing dynamic resolution
//! and making weight adjustments less precise.
//!
//! ## Relationship to Other Detection Modules
//!
//! - **Restricted range** (Issue #399): hidden neurons with compressed range.
//! - **Output squash mismatch** (Issue #546): wrong activation function type
//!   on output neurons (e.g., LOGISTIC for symmetric targets).
//! - **This module** (Issue #645): output neurons using the *correct* type of
//!   activation but operating in a compressed sub-range.
//!
//! ## Detection Criteria
//!
//! An output neuron has "range compression" if:
//! 1. It uses a bounded squash function (TANH, LOGISTIC, etc.).
//! 2. `range_utilisation = (activation_max - activation_min) / theoretical_range`
//!    is below a configurable threshold (default: 40%).
//! 3. The neuron is not dead (observed range is not near zero).
//! 4. The neuron is not saturated (not at the activation bounds).
//! 5. Sufficient samples are available (≥ 20).
//!
//! ## Recommended Actions
//!
//! - `changeSquash` — switch to an activation function whose range better
//!   matches the target distribution.
//! - `setBias` + `setWeight` — coordinated rescaling to recentre and expand
//!   the output pathway.

use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson};

use crate::analysis::constants::MIN_DISCOVERY_SAMPLE_COUNT as MIN_SAMPLES;

/// Minimum observed range (activation_max - activation_min) to distinguish
/// from dead neurons. Below this, the neuron is considered dead, not compressed.
const MIN_OBSERVED_RANGE: f32 = 0.01;

/// Distance from the theoretical bounds within which a neuron is considered
/// saturated rather than compressed.
const SATURATION_MARGIN: f32 = 0.10;

/// Configuration for output range compression detection.
#[derive(Debug, Clone)]
pub struct OutputRangeCompressionConfig {
    /// Maximum range utilisation below which a neuron is flagged (default: 0.40 = 40%).
    pub utilisation_threshold: f32,
    /// Minimum samples required for detection.
    pub min_samples: usize,
}

impl Default for OutputRangeCompressionConfig {
    fn default() -> Self {
        Self {
            utilisation_threshold: 0.40,
            min_samples: MIN_SAMPLES,
        }
    }
}

/// Result of detecting an output neuron with compressed range.
#[derive(Debug, Clone)]
pub struct OutputRangeCompressionNeuron {
    /// UUID of the output neuron with the compressed range.
    pub neuron_uuid: String,
    /// Current squash (activation) function.
    pub squash: String,
    /// Current bias of the neuron.
    pub bias: f32,
    /// Minimum observed activation across samples.
    pub activation_min: f32,
    /// Maximum observed activation across samples.
    pub activation_max: f32,
    /// Theoretical range of the squash function (e.g., 2.0 for TANH).
    pub theoretical_range: f32,
    /// Ratio of observed range to theoretical range (0.0 to 1.0).
    pub range_utilisation: f32,
    /// Number of samples analysed.
    pub sample_count: usize,
}

/// Returns the theoretical output range `(min, max)` for bounded squash functions.
///
/// Returns `None` for unbounded activations (IDENTITY, RELU, etc.) since they
/// have no fixed output bounds and range compression detection does not apply.
fn theoretical_bounds(squash: &str) -> Option<(f32, f32)> {
    match squash {
        "TANH" | "HARD_TANH" | "CLIPPED" | "BIPOLAR_SIGMOID" | "SOFTSIGN" | "ISRU" => {
            Some((-1.0, 1.0))
        }
        "LOGISTIC" => Some((0.0, 1.0)),
        "BIPOLAR" | "STEP" => Some((-1.0, 1.0)),
        "ARCTAN" => Some((-std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2)),
        "RELU6" => Some((0.0, 6.0)),
        _ => None,
    }
}

/// Detect output neurons with compressed activation ranges.
///
/// Analyses output neurons to find those where the observed activation range
/// is a small fraction of the theoretical range of their squash function.
///
/// # Arguments
/// * `creature` - The creature's network topology.
/// * `neuron_records` - List of `(neuron_uuid, records)` tuples with recorded activations.
/// * `config` - Detection configuration (thresholds).
///
/// # Returns
/// A list of `OutputRangeCompressionNeuron` sorted by range utilisation (lowest first).
pub fn detect_output_range_compression(
    creature: &CreatureJson,
    neuron_records: &[(String, Vec<DiscoverRecord>)],
    config: &OutputRangeCompressionConfig,
) -> Vec<OutputRangeCompressionNeuron> {
    // Build map from UUID to neuron info (only output neurons)
    let output_neurons: Vec<(&str, &str, f32)> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "output")
        .map(|n| (n.uuid.as_str(), n.squash.as_str(), n.bias))
        .collect();

    let neuron_map: std::collections::HashMap<&str, (&str, f32)> = output_neurons
        .iter()
        .map(|&(uuid, squash, bias)| (uuid, (squash, bias)))
        .collect();

    let mut results = Vec::with_capacity(neuron_records.len());

    for (uuid, records) in neuron_records {
        // Only consider output neurons
        let Some(&(squash, bias)) = neuron_map.get(uuid.as_str()) else {
            continue;
        };

        // Skip if insufficient samples
        if records.len() < config.min_samples {
            continue;
        }

        // Skip unbounded activations
        let Some((theoretical_min, theoretical_max)) = theoretical_bounds(squash) else {
            continue;
        };
        let theoretical_range = theoretical_max - theoretical_min;

        // Compute observed activation range
        let mut act_min = f32::INFINITY;
        let mut act_max = f32::NEG_INFINITY;
        for r in records {
            act_min = act_min.min(r.activation);
            act_max = act_max.max(r.activation);
        }

        let observed_range = act_max - act_min;

        // Exclude dead neurons (near-zero range)
        if observed_range < MIN_OBSERVED_RANGE {
            continue;
        }

        // Exclude saturated neurons (at bounds)
        let near_lower = act_min < (theoretical_min + SATURATION_MARGIN);
        let near_upper = act_max > (theoretical_max - SATURATION_MARGIN);
        if near_lower || near_upper {
            continue;
        }

        // Compute range utilisation
        let range_utilisation = observed_range / theoretical_range;

        if range_utilisation >= config.utilisation_threshold {
            continue;
        }

        results.push(OutputRangeCompressionNeuron {
            neuron_uuid: uuid.clone(),
            squash: squash.to_string(),
            bias,
            activation_min: act_min,
            activation_max: act_max,
            theoretical_range,
            range_utilisation,
            sample_count: records.len(),
        });
    }

    // Sort by range utilisation (lowest first — worst offenders first)
    results.sort_by(|a, b| a.range_utilisation.total_cmp(&b.range_utilisation));

    results
}

/// Convert output range compression detections into coordinated structural candidates.
///
/// Each compressed output neuron may produce multiple candidates:
/// 1. **changeSquash** — switch to an activation whose range better matches
///    the observed output distribution (e.g., LOGISTIC for [0, 1] targets).
/// 2. **setBias + setWeight** — coordinated rescaling to recentre and expand
///    the output pathway through the activation function.
pub fn output_range_compression_to_coordinated_candidates(
    detected: &[OutputRangeCompressionNeuron],
    creature: &CreatureJson,
) -> Vec<CoordinatedStructuralCandidateJson> {
    let mut results = Vec::new();

    for n in detected {
        let base_improvement = (1.0 - n.range_utilisation) * 0.005;

        // Candidate 1: Change squash to a better-fitting function
        let recommended_squash = recommend_squash(n);
        results.push(CoordinatedStructuralCandidateJson {
            operations: vec![CoordinatedStructuralOpJson::ChangeSquash {
                neuron_uuid: n.neuron_uuid.clone(),
                squash: recommended_squash.clone(),
            }],
            expected_creature_score_gain: base_improvement,
            comment: Some(format!(
                "Output range compression on {}: {} using {:.0}% of range [{:.3}, {:.3}] → change to {recommended_squash} (Issue #645)",
                n.neuron_uuid,
                n.squash,
                n.range_utilisation * 100.0,
                n.activation_min,
                n.activation_max,
            )),
        });

        // Candidate 2: Coordinated setBias + setWeight to recentre and rescale
        let incoming_synapses: Vec<&crate::SynapseJson> = creature
            .synapses
            .iter()
            .filter(|s| s.to_uuid == n.neuron_uuid)
            .collect();

        if !incoming_synapses.is_empty() {
            let observed_centre = (n.activation_min + n.activation_max) / 2.0;
            let (theoretical_min, theoretical_max) =
                theoretical_bounds(&n.squash).unwrap_or((-1.0, 1.0));
            let theoretical_centre = (theoretical_min + theoretical_max) / 2.0;
            let bias_delta = theoretical_centre - observed_centre;
            let new_bias = n.bias + bias_delta;

            // Scale factor to expand the observed range toward ~80% utilisation
            let target_utilisation = 0.80;
            let scale_factor = target_utilisation / n.range_utilisation.max(0.01);

            let mut ops: Vec<CoordinatedStructuralOpJson> = Vec::new();

            // Rescale incoming weights
            for s in &incoming_synapses {
                ops.push(CoordinatedStructuralOpJson::SetWeight {
                    from_neuron_uuid: s.from_uuid.clone(),
                    to_neuron_uuid: s.to_uuid.clone(),
                    weight: s.weight * scale_factor,
                });
            }

            // Adjust bias to recentre
            ops.push(CoordinatedStructuralOpJson::SetBias {
                neuron_uuid: n.neuron_uuid.clone(),
                bias: new_bias,
            });

            results.push(CoordinatedStructuralCandidateJson {
                operations: ops,
                expected_creature_score_gain: base_improvement * 0.7,
                comment: Some(format!(
                    "Output range compression on {}: {} using {:.0}% → rescale weights by {:.2}× and adjust bias from {:.3} to {:.3} (Issue #645)",
                    n.neuron_uuid,
                    n.squash,
                    n.range_utilisation * 100.0,
                    scale_factor,
                    n.bias,
                    new_bias,
                )),
            });
        }
    }

    // Sort by expected improvement (best first)
    results.sort_by(|a, b| {
        b.expected_creature_score_gain
            .total_cmp(&a.expected_creature_score_gain)
    });

    results
}

/// Recommend an alternative squash function based on the observed activation range.
fn recommend_squash(n: &OutputRangeCompressionNeuron) -> String {
    let all_positive = n.activation_min >= 0.0;
    let all_in_unit = n.activation_min >= 0.0 && n.activation_max <= 1.0;

    if all_in_unit {
        // Activations fit in [0, 1] — LOGISTIC is a natural fit
        "LOGISTIC".to_string()
    } else if all_positive {
        // Activations are positive but extend beyond 1 — LOGISTIC still good
        "LOGISTIC".to_string()
    } else {
        // Activations span both positive and negative — TANH is standard
        "TANH".to_string()
    }
}
