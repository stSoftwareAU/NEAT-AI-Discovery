//! Weight coherence validation for NEAT-AI Discovery.
//!
//! Part of the "Brilliant but Brittle" initiative (Issue #432, #437). This module validates
//! that proposed weight configurations are coherent with network structure and won't create
//! brittleness.
//!
//! ## Coherence Checks
//!
//! 1. **Incoherent weight ratios**: Large incoming weights with tiny outgoing weights indicate
//!    inefficient amplification/attenuation patterns that make the network brittle.
//!
//! 2. **Near-constant output paths**: Weights that cause neurons to produce near-constant
//!    output regardless of inputs create brittle "constant offset" behaviour.
//!
//! 3. **Symmetric weight cancellation**: Opposite weights from correlated inputs can cancel
//!    meaningful signal, causing instability when correlations shift.
//!
//! ## Configuration
//!
//! Detection thresholds can be configured via `WeightCoherenceConfig`:
//! - `max_weight_ratio`: Maximum allowed incoming/outgoing weight ratio (default: 100.0)
//! - `min_activation_variance`: Minimum variance to consider output non-constant (default: 0.01)
//! - `min_correlation_for_cancellation`: Minimum correlation to flag symmetric cancellation (default: 0.8)

use super::activation_properties::is_saturating_squash;
use super::topology_cache::CreatureTopologyCache;
use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson};
use std::collections::HashMap;

// =============================================================================
// Constants
// =============================================================================

/// Default maximum incoming/outgoing weight ratio before flagging as incoherent.
const DEFAULT_MAX_WEIGHT_RATIO: f32 = 100.0;

/// Default minimum activation variance to consider non-constant.
const DEFAULT_MIN_ACTIVATION_VARIANCE: f32 = 0.01;

/// Default minimum correlation to flag symmetric cancellation.
const DEFAULT_MIN_CORRELATION: f32 = 0.8;

/// Minimum samples required for statistical reliability.
const MIN_SAMPLE_COUNT: usize = 20;

/// Small epsilon for numerical stability.
const EPSILON: f32 = 1e-9;

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for weight coherence validation.
#[derive(Debug, Clone)]
pub struct WeightCoherenceConfig {
    /// Maximum allowed incoming/outgoing weight ratio.
    pub max_weight_ratio: f32,
    /// Minimum activation variance to consider output non-constant.
    pub min_activation_variance: f32,
    /// Minimum correlation between inputs to flag symmetric cancellation.
    pub min_correlation_for_cancellation: f32,
    /// Minimum samples required for detection.
    pub min_samples: usize,
}

impl Default for WeightCoherenceConfig {
    fn default() -> Self {
        Self {
            max_weight_ratio: DEFAULT_MAX_WEIGHT_RATIO,
            min_activation_variance: DEFAULT_MIN_ACTIVATION_VARIANCE,
            min_correlation_for_cancellation: DEFAULT_MIN_CORRELATION,
            min_samples: MIN_SAMPLE_COUNT,
        }
    }
}

// =============================================================================
// Candidate Types
// =============================================================================

/// Candidate for incoherent weight ratio detection.
#[derive(Debug, Clone)]
pub struct IncoherentWeightRatioCandidate {
    /// UUID of the neuron with incoherent weight ratio.
    pub neuron_uuid: String,
    /// Sum of absolute incoming weights.
    pub incoming_weight_sum: f32,
    /// Sum of absolute outgoing weights.
    pub outgoing_weight_sum: f32,
    /// Ratio of incoming to outgoing weights.
    pub incoming_outgoing_ratio: f32,
    /// Recommended outgoing weight to restore coherence.
    pub recommended_outgoing_weight: f32,
    /// Number of samples analysed.
    pub sample_count: usize,
    /// Estimated improvement from fixing this issue.
    pub estimated_improvement: f32,
}

/// Candidate for near-constant output path detection.
#[derive(Debug, Clone)]
pub struct NearConstantPathCandidate {
    /// UUID of the neuron with near-constant output.
    pub neuron_uuid: String,
    /// Variance of activation values (very low indicates constant).
    pub activation_variance: f32,
    /// Mean activation value.
    pub mean_activation: f32,
    /// Weight that's causing the constant output (e.g., very large incoming).
    pub causing_weight: f32,
    /// Recommended action: "setBias" or "setWeight".
    pub recommended_action: String,
    /// Number of samples analysed.
    pub sample_count: usize,
    /// Estimated improvement from fixing this issue.
    pub estimated_improvement: f32,
}

/// Candidate for symmetric weight cancellation detection.
#[derive(Debug, Clone)]
pub struct SymmetricCancellationCandidate {
    /// UUID of first source neuron.
    pub source1_neuron_uuid: String,
    /// UUID of second source neuron.
    pub source2_neuron_uuid: String,
    /// UUID of target neuron where cancellation occurs.
    pub target_neuron_uuid: String,
    /// Weight from source1 to target.
    pub weight1: f32,
    /// Weight from source2 to target.
    pub weight2: f32,
    /// Correlation between source activations.
    pub correlation: f32,
    /// Ratio of signal cancellation (0 = no cancellation, 1 = complete).
    pub cancellation_ratio: f32,
    /// Recommended action: "setWeight" or "removeSynapse".
    pub recommended_action: String,
    /// Number of samples analysed.
    pub sample_count: usize,
    /// Estimated improvement from fixing this issue.
    pub estimated_improvement: f32,
}

// =============================================================================
// Detection Functions
// =============================================================================

/// Detect neurons with incoherent incoming/outgoing weight ratios.
///
/// A neuron receiving large incoming weights but outputting via tiny weights is doing
/// significant computation that barely affects the network output. This is inefficient
/// and creates brittleness when small weight changes have disproportionate effects.
///
/// # Arguments
/// * `creature` - The creature JSON containing network topology
/// * `records` - Vector of (neuron_uuid, records) tuples with activation data
/// * `config` - Configuration for detection thresholds
///
/// # Returns
/// Vector of candidates identifying neurons with incoherent weight ratios.
pub fn detect_incoherent_weight_ratios(
    creature: &CreatureJson,
    records: &[(String, Vec<DiscoverRecord>)],
    config: &WeightCoherenceConfig,
    topo: Option<&CreatureTopologyCache>,
) -> Vec<IncoherentWeightRatioCandidate> {
    let mut candidates = Vec::with_capacity(creature.neurons.len());

    // Use shared topology cache or build locally for backward compatibility.
    let local_cache;
    let topo = match topo {
        Some(t) => t,
        None => {
            local_cache = CreatureTopologyCache::new(creature);
            &local_cache
        }
    };

    // Build records lookup
    let records_map: HashMap<String, &Vec<DiscoverRecord>> = records
        .iter()
        .map(|(uuid, recs)| (uuid.clone(), recs))
        .collect();

    for neuron_uuid in &topo.hidden_uuids {
        // Get records for this neuron
        let Some(neuron_records) = records_map.get(neuron_uuid) else {
            continue;
        };

        if neuron_records.len() < config.min_samples {
            continue;
        }

        // Calculate incoming and outgoing weight sums from topology cache
        let incoming_sum: f32 = topo
            .fan_in_for(neuron_uuid)
            .iter()
            .filter_map(|from_uuid| topo.synapse_weight(from_uuid, neuron_uuid))
            .map(f32::abs)
            .sum();

        let outgoing_sum: f32 = topo
            .fan_out_for(neuron_uuid)
            .iter()
            .filter_map(|to_uuid| topo.synapse_weight(neuron_uuid, to_uuid))
            .map(f32::abs)
            .sum();

        // Skip if no meaningful weights
        if incoming_sum <= EPSILON || outgoing_sum <= EPSILON {
            continue;
        }

        let ratio = incoming_sum / outgoing_sum;

        if ratio > config.max_weight_ratio {
            // Calculate recommended outgoing weight to bring ratio down
            let recommended_outgoing = incoming_sum / config.max_weight_ratio;

            // Estimate improvement based on how far out of range we are
            let severity = (ratio / config.max_weight_ratio).ln().max(0.0);
            let estimated_improvement = 0.01 * severity;

            candidates.push(IncoherentWeightRatioCandidate {
                neuron_uuid: neuron_uuid.clone(),
                incoming_weight_sum: incoming_sum,
                outgoing_weight_sum: outgoing_sum,
                incoming_outgoing_ratio: ratio,
                recommended_outgoing_weight: recommended_outgoing.min(0.1), // Cap at sensible max
                sample_count: neuron_records.len(),
                estimated_improvement,
            });
        }
    }

    // Sort by estimated improvement (best first)
    candidates.sort_by(|a, b| b.estimated_improvement.total_cmp(&a.estimated_improvement));

    candidates
}

/// Detect neurons with near-constant output regardless of input variation.
///
/// When weights cause a neuron to produce nearly constant output (e.g., TANH saturated
/// at ±1), the neuron adds no meaningful information and creates brittleness.
///
/// # Arguments
/// * `creature` - The creature JSON containing network topology
/// * `records` - Vector of (neuron_uuid, records) tuples with activation data
/// * `config` - Configuration for detection thresholds
///
/// # Returns
/// Vector of candidates identifying neurons with near-constant output.
pub fn detect_near_constant_paths(
    creature: &CreatureJson,
    records: &[(String, Vec<DiscoverRecord>)],
    config: &WeightCoherenceConfig,
    topo: Option<&CreatureTopologyCache>,
) -> Vec<NearConstantPathCandidate> {
    let mut candidates = Vec::with_capacity(creature.neurons.len());

    // Use shared topology cache or build locally for backward compatibility.
    let local_cache;
    let topo = match topo {
        Some(t) => t,
        None => {
            local_cache = CreatureTopologyCache::new(creature);
            &local_cache
        }
    };

    // Build records lookup
    let records_map: HashMap<String, &Vec<DiscoverRecord>> = records
        .iter()
        .map(|(uuid, recs)| (uuid.clone(), recs))
        .collect();

    // Iterate hidden neurons directly to avoid building a separate squash lookup map.
    for neuron in &creature.neurons {
        if !topo.hidden_uuids.contains(&neuron.uuid) {
            continue;
        }
        let neuron_uuid = &neuron.uuid;
        let squash = neuron.squash.as_str();
        let Some(neuron_records) = records_map.get(neuron_uuid) else {
            continue;
        };

        if neuron_records.len() < config.min_samples {
            continue;
        }

        // Calculate activation variance
        let activations: Vec<f32> = neuron_records
            .iter()
            .filter(|r| r.activation.is_finite())
            .map(|r| r.activation)
            .collect();

        if activations.len() < config.min_samples {
            continue;
        }

        let mean = activations.iter().sum::<f32>() / activations.len() as f32;
        let variance =
            activations.iter().map(|a| (a - mean).powi(2)).sum::<f32>() / activations.len() as f32;

        if variance < config.min_activation_variance {
            // Find the largest incoming weight that might cause saturation
            let causing_weight = topo
                .fan_in_for(neuron_uuid)
                .iter()
                .filter_map(|from_uuid| topo.synapse_weight(from_uuid, neuron_uuid))
                .map(f32::abs)
                .max_by(f32::total_cmp)
                .unwrap_or(0.0);

            // Recommend setBias for saturating activations
            let recommended_action = if is_saturating_squash(squash) && mean.abs() > 0.9 {
                "setBias"
            } else {
                "setWeight"
            };

            // Estimate improvement based on how constant the output is
            let constancy = 1.0 - (variance / config.min_activation_variance).min(1.0);
            let estimated_improvement = 0.008 * constancy;

            candidates.push(NearConstantPathCandidate {
                neuron_uuid: neuron_uuid.clone(),
                activation_variance: variance,
                mean_activation: mean,
                causing_weight,
                recommended_action: recommended_action.to_string(),
                sample_count: activations.len(),
                estimated_improvement,
            });
        }
    }

    // Sort by estimated improvement (best first)
    candidates.sort_by(|a, b| b.estimated_improvement.total_cmp(&a.estimated_improvement));

    candidates
}

/// Detect symmetric weights that cancel meaningful signal.
///
/// When two inputs to a neuron have opposite-sign weights of similar magnitude
/// and the inputs are highly correlated, the signals tend to cancel out.
/// This creates brittleness as small correlation changes can flip the behaviour.
///
/// # Arguments
/// * `creature` - The creature JSON containing network topology
/// * `records` - Vector of (neuron_uuid, records) tuples with activation data
/// * `config` - Configuration for detection thresholds
///
/// # Returns
/// Vector of candidates identifying symmetric weight cancellation.
pub fn detect_symmetric_cancellation(
    creature: &CreatureJson,
    records: &[(String, Vec<DiscoverRecord>)],
    config: &WeightCoherenceConfig,
    topo: Option<&CreatureTopologyCache>,
) -> Vec<SymmetricCancellationCandidate> {
    let mut candidates = Vec::with_capacity(creature.synapses.len());

    // Use shared topology cache or build locally for backward compatibility.
    let local_cache;
    let topo = match topo {
        Some(t) => t,
        None => {
            local_cache = CreatureTopologyCache::new(creature);
            &local_cache
        }
    };

    // Build records lookup indexed by obs_index for correlation calculation
    let records_map: HashMap<String, &Vec<DiscoverRecord>> = records
        .iter()
        .map(|(uuid, recs)| (uuid.clone(), recs))
        .collect();

    // Iterate fan-in directly to avoid building a separate synapses-by-target map.
    for (target_uuid, from_uuids) in &topo.fan_in {
        let incoming_synapses: Vec<(&str, f32)> = from_uuids
            .iter()
            .filter_map(|from_uuid| {
                topo.synapse_weight(from_uuid, target_uuid)
                    .map(|weight| (from_uuid.as_str(), weight))
            })
            .collect();
        if incoming_synapses.len() < 2 {
            continue;
        }

        // Check all pairs of incoming synapses
        for i in 0..incoming_synapses.len() {
            for j in (i + 1)..incoming_synapses.len() {
                let (source1_uuid, weight1) = &incoming_synapses[i];
                let (source2_uuid, weight2) = &incoming_synapses[j];

                // Check if weights have opposite signs and similar magnitude
                if weight1.signum() == weight2.signum() {
                    continue; // Same sign, no cancellation
                }

                let mag_ratio = weight1.abs().min(weight2.abs())
                    / weight1.abs().max(weight2.abs()).max(EPSILON);

                if mag_ratio < 0.5 {
                    continue; // Magnitudes too different for significant cancellation
                }

                // Get records for both sources
                let Some(records1) = records_map.get(*source1_uuid) else {
                    continue;
                };
                let Some(records2) = records_map.get(*source2_uuid) else {
                    continue;
                };

                // Calculate correlation between source activations
                let correlation = calculate_correlation(records1, records2, config.min_samples);

                if let Some(corr) = correlation
                    && corr.abs() >= config.min_correlation_for_cancellation
                {
                    // Calculate cancellation ratio
                    let cancellation_ratio = mag_ratio * corr.abs();

                    if cancellation_ratio > 0.5 {
                        // Significant cancellation detected
                        let estimated_improvement = 0.01 * cancellation_ratio;

                        candidates.push(SymmetricCancellationCandidate {
                            source1_neuron_uuid: source1_uuid.to_string(),
                            source2_neuron_uuid: source2_uuid.to_string(),
                            target_neuron_uuid: target_uuid.clone(),
                            weight1: *weight1,
                            weight2: *weight2,
                            correlation: corr,
                            cancellation_ratio,
                            recommended_action: "setWeight".to_string(),
                            sample_count: records1.len().min(records2.len()),
                            estimated_improvement,
                        });
                    }
                }
            }
        }
    }

    // Sort by estimated improvement (best first)
    candidates.sort_by(|a, b| b.estimated_improvement.total_cmp(&a.estimated_improvement));

    candidates
}

// =============================================================================
// Candidate Conversion Functions
// =============================================================================

/// Convert incoherent weight ratio candidates to coordinated structural candidates.
pub fn incoherent_ratios_to_coordinated_candidates(
    candidates: &[IncoherentWeightRatioCandidate],
) -> Vec<CoordinatedStructuralCandidateJson> {
    let mut result: Vec<CoordinatedStructuralCandidateJson> = candidates
        .iter()
        .map(|c| {
            CoordinatedStructuralCandidateJson {
                operations: vec![CoordinatedStructuralOpJson::SetWeight {
                    from_neuron_uuid: c.neuron_uuid.clone(),
                    to_neuron_uuid: "".to_string(), // Will be filled by NEAT-AI based on topology
                    weight: c.recommended_outgoing_weight,
                }],
                expected_creature_score_gain: c.estimated_improvement,
                comment: Some(format!(
                    "Weight coherence: ratio {:.1}x exceeds threshold, recommend balancing outgoing weight (incoming_sum={:.3}, outgoing_sum={:.3})",
                    c.incoming_outgoing_ratio, c.incoming_weight_sum, c.outgoing_weight_sum
                )),
            }
        })
        .collect();

    // Sort by improvement (best first)
    result.sort_by(|a, b| {
        b.expected_creature_score_gain
            .total_cmp(&a.expected_creature_score_gain)
    });

    result
}

/// Convert near-constant path candidates to coordinated structural candidates.
pub fn near_constant_paths_to_coordinated_candidates(
    candidates: &[NearConstantPathCandidate],
) -> Vec<CoordinatedStructuralCandidateJson> {
    let mut result: Vec<CoordinatedStructuralCandidateJson> = candidates
        .iter()
        .map(|c| {
            let operation = if c.recommended_action == "setBias" {
                CoordinatedStructuralOpJson::SetBias {
                    neuron_uuid: c.neuron_uuid.clone(),
                    bias: -c.mean_activation * 0.5, // Shift operating point away from saturation
                }
            } else {
                CoordinatedStructuralOpJson::SetWeight {
                    from_neuron_uuid: "".to_string(), // Will be filled by NEAT-AI
                    to_neuron_uuid: c.neuron_uuid.clone(),
                    weight: c.causing_weight * 0.1, // Reduce the causing weight
                }
            };

            CoordinatedStructuralCandidateJson {
                operations: vec![operation],
                expected_creature_score_gain: c.estimated_improvement,
                comment: Some(format!(
                    "Near-constant path: variance {:.6} < threshold, mean activation {:.3}, recommend {}",
                    c.activation_variance, c.mean_activation, c.recommended_action
                )),
            }
        })
        .collect();

    result.sort_by(|a, b| {
        b.expected_creature_score_gain
            .total_cmp(&a.expected_creature_score_gain)
    });

    result
}

/// Convert symmetric cancellation candidates to coordinated structural candidates.
pub fn symmetric_cancellation_to_coordinated_candidates(
    candidates: &[SymmetricCancellationCandidate],
) -> Vec<CoordinatedStructuralCandidateJson> {
    let mut result: Vec<CoordinatedStructuralCandidateJson> = candidates
        .iter()
        .map(|c| {
            // Recommend reducing the smaller magnitude weight
            let (from_uuid, new_weight) = if c.weight1.abs() < c.weight2.abs() {
                (c.source1_neuron_uuid.clone(), c.weight1 * 0.5)
            } else {
                (c.source2_neuron_uuid.clone(), c.weight2 * 0.5)
            };

            CoordinatedStructuralCandidateJson {
                operations: vec![CoordinatedStructuralOpJson::SetWeight {
                    from_neuron_uuid: from_uuid,
                    to_neuron_uuid: c.target_neuron_uuid.clone(),
                    weight: new_weight,
                }],
                expected_creature_score_gain: c.estimated_improvement,
                comment: Some(format!(
                    "Symmetric cancellation: correlation {:.3}, cancellation ratio {:.3}, recommend reducing weight imbalance",
                    c.correlation, c.cancellation_ratio
                )),
            }
        })
        .collect();

    result.sort_by(|a, b| {
        b.expected_creature_score_gain
            .total_cmp(&a.expected_creature_score_gain)
    });

    result
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Calculate Pearson correlation between two sets of records.
///
/// Returns None if there are insufficient paired samples.
fn calculate_correlation(
    records1: &[DiscoverRecord],
    records2: &[DiscoverRecord],
    min_samples: usize,
) -> Option<f32> {
    // Build lookup by obs_index, filtering non-finite activations
    let lookup1: HashMap<u32, f32> = records1
        .iter()
        .filter(|r| r.activation.is_finite())
        .map(|r| (r.obs_index, r.activation))
        .collect();

    let lookup2: HashMap<u32, f32> = records2
        .iter()
        .filter(|r| r.activation.is_finite())
        .map(|r| (r.obs_index, r.activation))
        .collect();

    // Check sufficient shared samples before delegating
    let shared_count = lookup1.keys().filter(|k| lookup2.contains_key(k)).count();
    if shared_count < min_samples {
        return None;
    }

    Some(super::stats::pearson_correlation_hashmaps(
        &lookup1,
        &lookup2,
        min_samples,
    ))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default_values() {
        let config = WeightCoherenceConfig::default();
        assert_eq!(config.max_weight_ratio, DEFAULT_MAX_WEIGHT_RATIO);
        assert_eq!(
            config.min_activation_variance,
            DEFAULT_MIN_ACTIVATION_VARIANCE
        );
        assert_eq!(
            config.min_correlation_for_cancellation,
            DEFAULT_MIN_CORRELATION
        );
        assert_eq!(config.min_samples, MIN_SAMPLE_COUNT);
    }

    #[test]
    fn test_is_saturating_squash() {
        assert!(is_saturating_squash("TANH"));
        assert!(is_saturating_squash("tanh"));
        assert!(is_saturating_squash("LOGISTIC"));
        assert!(is_saturating_squash("HARD_TANH"));
        assert!(!is_saturating_squash("RELU"));
        assert!(!is_saturating_squash("IDENTITY"));
    }

    #[test]
    fn test_correlation_identical_signals() {
        let records1: Vec<DiscoverRecord> = (0..50)
            .map(|i| DiscoverRecord {
                obs_index: i,
                neuron_uuid: "a".to_string(),
                value: Some(i as f32 * 0.1),
                activation: i as f32 * 0.1,
                errors: vec![0.0],
            })
            .collect();

        let records2 = records1.clone();

        let corr = calculate_correlation(&records1, &records2, 10);
        assert!(corr.is_some());
        assert!(
            (corr.unwrap() - 1.0).abs() < 0.001,
            "Identical signals should have correlation ~1.0"
        );
    }

    #[test]
    fn test_correlation_opposite_signals() {
        let records1: Vec<DiscoverRecord> = (0..50)
            .map(|i| DiscoverRecord {
                obs_index: i,
                neuron_uuid: "a".to_string(),
                value: Some(i as f32 * 0.1),
                activation: i as f32 * 0.1,
                errors: vec![0.0],
            })
            .collect();

        let records2: Vec<DiscoverRecord> = (0..50)
            .map(|i| DiscoverRecord {
                obs_index: i,
                neuron_uuid: "b".to_string(),
                value: Some(-(i as f32 * 0.1)),
                activation: -(i as f32 * 0.1),
                errors: vec![0.0],
            })
            .collect();

        let corr = calculate_correlation(&records1, &records2, 10);
        assert!(corr.is_some());
        assert!(
            (corr.unwrap() + 1.0).abs() < 0.001,
            "Opposite signals should have correlation ~-1.0"
        );
    }

    #[test]
    fn test_correlation_insufficient_samples() {
        let records1: Vec<DiscoverRecord> = (0..5)
            .map(|i| DiscoverRecord {
                obs_index: i,
                neuron_uuid: "a".to_string(),
                value: Some(i as f32),
                activation: i as f32,
                errors: vec![0.0],
            })
            .collect();

        let records2 = records1.clone();

        let corr = calculate_correlation(&records1, &records2, 10);
        assert!(
            corr.is_none(),
            "Should return None for insufficient samples"
        );
    }
}
