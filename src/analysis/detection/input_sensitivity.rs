//! Input sensitivity analysis module (Issue #435).
//!
//! Part of the "Brilliant but Brittle" initiative (Issue #432). This module analyses
//! how sensitive predictions are to small changes in input observations, identifying
//! inputs with excessive leverage that contribute to brittle predictions.
//!
//! See `docs/DISCOVERY_TYPES.md` for full documentation.
//!
//! ## Detection Criteria
//!
//! ### Dominant Inputs
//! An input neuron has excessive leverage if:
//! 1. **High sensitivity**: Small changes in the input cause disproportionate output changes.
//! 2. **High leverage ratio**: The input's contribution to output variance exceeds expected.
//! 3. **Weight amplification**: Large weights amplify input variance into output variance.
//!
//! ### Threshold Effects
//! A threshold effect exists when:
//! 1. **Steep gradient**: The activation gradient is very large in a region.
//! 2. **Threshold proximity**: Inputs operate near the steep region.
//! 3. **Prediction flip**: Small input changes cause large output changes.
//!
//! ## Recommended Actions
//!
//! - `setWeight`: Reduce weight of excessive sensitivity connections
//! - `addNeuron`: Add dampening/smoothing neuron to reduce sharp transitions
//! - `setBias`: Shift operating point away from threshold regions

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use std::collections::{HashMap, HashSet};

use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson};

// MIN_SAMPLES_FOR_DETECTION moved to constants.rs (Issue #424)
use crate::analysis::constants::MIN_DISCOVERY_SAMPLE_COUNT as MIN_SAMPLES_FOR_DETECTION;

/// Default threshold for dominance ratio to flag an input.
const DEFAULT_DOMINANCE_THRESHOLD: f32 = 2.0;

/// Default threshold for gradient magnitude to flag threshold effects.
const DEFAULT_GRADIENT_THRESHOLD: f32 = 10.0;

/// Minimum variance to consider an input non-constant.
const MIN_VARIANCE: f32 = 1e-8;

/// Weight reduction factor for setWeight recommendations.
const WEIGHT_REDUCTION_FACTOR: f32 = 0.3;

/// Configuration for input sensitivity analysis.
#[derive(Debug, Clone)]
pub struct InputSensitivityConfig {
    /// Threshold for leverage ratio to flag an input as dominant.
    pub dominance_threshold: f32,
    /// Threshold for gradient magnitude to flag threshold effects.
    pub gradient_threshold: f32,
    /// Minimum samples required for detection.
    pub min_samples: usize,
}

impl Default for InputSensitivityConfig {
    fn default() -> Self {
        Self {
            dominance_threshold: dominance_threshold_from_env(),
            gradient_threshold: gradient_threshold_from_env(),
            min_samples: MIN_SAMPLES_FOR_DETECTION,
        }
    }
}

/// Get dominance threshold from environment variable or default.
fn dominance_threshold_from_env() -> f32 {
    crate::config::dominance_threshold(DEFAULT_DOMINANCE_THRESHOLD)
}

/// Get gradient threshold from environment variable or default.
fn gradient_threshold_from_env() -> f32 {
    crate::config::gradient_threshold(DEFAULT_GRADIENT_THRESHOLD)
}

/// Candidate for a dominant input with excessive leverage.
#[derive(Debug, Clone)]
pub struct DominantInputCandidate {
    /// UUID of the dominant input neuron.
    pub input_neuron_uuid: String,
    /// UUID of the target neuron affected by this input.
    pub target_neuron_uuid: String,
    /// Normalised sensitivity score (higher = more sensitive).
    pub sensitivity_score: f32,
    /// Leverage ratio: input's variance contribution vs expected.
    pub leverage_ratio: f32,
    /// Current weight of the connection.
    pub current_weight: f32,
    /// Recommended weight to reduce sensitivity.
    pub recommended_weight: f32,
    /// Number of samples analysed.
    pub sample_count: usize,
    /// Estimated creature score improvement.
    pub estimated_improvement: f32,
}

/// Candidate for a threshold effect where small changes flip predictions.
#[derive(Debug, Clone)]
pub struct ThresholdEffectCandidate {
    /// UUID of the input neuron near the threshold.
    pub input_neuron_uuid: String,
    /// UUID of the intermediate neuron with the threshold (if any).
    pub intermediate_neuron_uuid: Option<String>,
    /// UUID of the target neuron affected.
    pub target_neuron_uuid: String,
    /// Magnitude of the gradient in the threshold region.
    pub gradient_magnitude: f32,
    /// How close the operating point is to the threshold (0-1).
    pub threshold_proximity: f32,
    /// Recommended action: "addNeuron", "setBias", or "setWeight".
    pub recommended_action: String,
    /// Number of samples analysed.
    pub sample_count: usize,
    /// Estimated creature score improvement.
    pub estimated_improvement: f32,
}

use super::helpers::build_record_map;
use super::stats::{compute_mean, compute_variance};

/// Compute covariance between two equal-length slices.
fn compute_covariance(x: &[f32], y: &[f32]) -> f32 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }
    let n = x.len() as f32;
    let mean_x = compute_mean(x);
    let mean_y = compute_mean(y);
    x.iter()
        .zip(y.iter())
        .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
        .sum::<f32>()
        / n
}

/// Detect input neurons with disproportionate leverage on predictions.
///
/// # Arguments
/// * `creature` - The creature's network topology.
/// * `neuron_records` - List of `(neuron_uuid, records)` tuples with recorded data.
/// * `config` - Configuration for sensitivity thresholds.
///
/// # Returns
/// A list of `DominantInputCandidate` for inputs with excessive leverage,
/// sorted by estimated improvement (best first).
pub fn detect_dominant_inputs(
    creature: &CreatureJson,
    neuron_records: &[(String, Vec<DiscoverRecord>)],
    config: &InputSensitivityConfig,
) -> Vec<DominantInputCandidate> {
    // Build records lookup
    let records_map = build_record_map(neuron_records);

    // Identify input neurons
    let input_uuids: HashSet<&str> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "input")
        .map(|n| n.uuid.as_str())
        .collect();

    // Identify output neurons
    let output_uuids: HashSet<&str> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "output")
        .map(|n| n.uuid.as_str())
        .collect();

    // Build synapse lookup: from_uuid -> [(to_uuid, weight)]
    let mut synapse_map: HashMap<&str, Vec<(&str, f32)>> = HashMap::new();
    for syn in &creature.synapses {
        synapse_map
            .entry(syn.from_uuid.as_str())
            .or_default()
            .push((syn.to_uuid.as_str(), syn.weight));
    }

    let mut candidates = Vec::with_capacity(input_uuids.len());

    // For each input neuron, analyse its sensitivity to each connected output
    for input_uuid in &input_uuids {
        let Some(input_records) = records_map.get(*input_uuid) else {
            continue;
        };

        if input_records.len() < config.min_samples {
            continue;
        }

        // Extract input activations
        let input_activations: Vec<f32> = input_records.iter().map(|r| r.activation).collect();
        let input_variance = compute_variance(&input_activations);

        // Skip constant inputs (zero variance)
        if input_variance < MIN_VARIANCE {
            continue;
        }

        // Get connected outputs (direct or via hidden)
        let Some(connections) = synapse_map.get(*input_uuid) else {
            continue;
        };

        for &(target_uuid, weight) in connections {
            // Focus on connections to outputs or high-impact hidden neurons
            let is_output = output_uuids.contains(target_uuid);
            if !is_output {
                // For hidden neurons, check if they connect to outputs
                let Some(hidden_connections) = synapse_map.get(target_uuid) else {
                    continue;
                };
                if !hidden_connections
                    .iter()
                    .any(|(to, _)| output_uuids.contains(to))
                {
                    continue;
                }
            }

            // Get target records
            let Some(target_records) = records_map.get(target_uuid) else {
                continue;
            };

            if target_records.len() < config.min_samples {
                continue;
            }

            // Build matched pairs by obs_index
            let target_map: HashMap<u32, f32> = target_records
                .iter()
                .filter_map(|r| r.errors.first().map(|&e| (r.obs_index, e)))
                .collect();

            let mut matched_inputs = Vec::new();
            let mut matched_errors = Vec::new();

            for record in *input_records {
                if let Some(&error) = target_map.get(&record.obs_index) {
                    matched_inputs.push(record.activation);
                    matched_errors.push(error);
                }
            }

            if matched_inputs.len() < config.min_samples {
                continue;
            }

            // Compute sensitivity metrics
            let input_var = compute_variance(&matched_inputs);
            let error_var = compute_variance(&matched_errors);

            if input_var < MIN_VARIANCE || error_var < MIN_VARIANCE {
                continue;
            }

            // Compute leverage: how much does this input contribute to error variance?
            // Sensitivity = |covariance(input, error)| / sqrt(input_var * error_var)
            let covariance = compute_covariance(&matched_inputs, &matched_errors);
            let correlation = covariance / (input_var.sqrt() * error_var.sqrt());

            // Leverage ratio considers weight amplification
            // Higher weight = higher leverage
            let weight_factor = weight.abs();
            let leverage_ratio = weight_factor * correlation.abs() * (input_var / error_var).sqrt();

            // Sensitivity score normalised for comparison
            let sensitivity_score = leverage_ratio * weight_factor;

            // Only flag if above threshold
            if sensitivity_score <= config.dominance_threshold {
                continue;
            }

            // Compute recommended weight to bring sensitivity below threshold
            let target_sensitivity = config.dominance_threshold * 0.8;
            let recommended_weight = if sensitivity_score > 0.0 {
                weight * (target_sensitivity / sensitivity_score).min(WEIGHT_REDUCTION_FACTOR)
            } else {
                weight * WEIGHT_REDUCTION_FACTOR
            };

            // Estimated improvement
            let sample_factor = (matched_inputs.len() as f32 / 1000.0).min(1.0);
            let sensitivity_factor = ((sensitivity_score - config.dominance_threshold)
                / config.dominance_threshold)
                .min(1.0);
            let estimated_improvement = 0.001 * (sample_factor * 0.3 + sensitivity_factor * 0.7);

            candidates.push(DominantInputCandidate {
                input_neuron_uuid: input_uuid.to_string(),
                target_neuron_uuid: target_uuid.to_string(),
                sensitivity_score,
                leverage_ratio,
                current_weight: weight,
                recommended_weight,
                sample_count: matched_inputs.len(),
                estimated_improvement,
            });
        }
    }

    // Sort by estimated improvement (best first)
    candidates.sort_by(|a, b| b.estimated_improvement.total_cmp(&a.estimated_improvement));

    candidates
}

/// Detect threshold effects where small input changes cause large output changes.
///
/// # Arguments
/// * `creature` - The creature's network topology.
/// * `neuron_records` - List of `(neuron_uuid, records)` tuples with recorded data.
/// * `config` - Configuration for sensitivity thresholds.
///
/// # Returns
/// A list of `ThresholdEffectCandidate` for inputs near threshold regions,
/// sorted by estimated improvement (best first).
pub fn detect_threshold_effects(
    creature: &CreatureJson,
    neuron_records: &[(String, Vec<DiscoverRecord>)],
    config: &InputSensitivityConfig,
) -> Vec<ThresholdEffectCandidate> {
    // Build records lookup
    let records_map = build_record_map(neuron_records);

    // Identify input neurons
    let input_uuids: HashSet<&str> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "input")
        .map(|n| n.uuid.as_str())
        .collect();

    // Build neuron info lookup
    let neuron_info: HashMap<&str, (&str, f32)> = creature
        .neurons
        .iter()
        .map(|n| (n.uuid.as_str(), (n.squash.as_str(), n.bias)))
        .collect();

    // Build synapse lookup: from_uuid -> [(to_uuid, weight)]
    let mut synapse_map: HashMap<&str, Vec<(&str, f32)>> = HashMap::new();
    for syn in &creature.synapses {
        synapse_map
            .entry(syn.from_uuid.as_str())
            .or_default()
            .push((syn.to_uuid.as_str(), syn.weight));
    }

    let mut candidates = Vec::with_capacity(input_uuids.len());

    // Check each input -> hidden path for threshold effects
    for input_uuid in &input_uuids {
        let Some(input_records) = records_map.get(*input_uuid) else {
            continue;
        };

        if input_records.len() < config.min_samples {
            continue;
        }

        let Some(connections) = synapse_map.get(*input_uuid) else {
            continue;
        };

        for &(hidden_uuid, weight) in connections {
            // Look for hidden neurons with steep activation functions
            let Some(&(squash, bias)) = neuron_info.get(hidden_uuid) else {
                continue;
            };

            // Check if this is a steep activation region
            // Steep activations: TANH, LOGISTIC near threshold with large weight
            // Squash strings are normalised to uppercase at load time (Issue #753),
            // so no allocation needed here (Issue #771).
            let is_steep_activation =
                matches!(squash, "TANH" | "LOGISTIC" | "HARD_TANH" | "SOFTSIGN");

            if !is_steep_activation {
                continue;
            }

            let Some(hidden_records) = records_map.get(hidden_uuid) else {
                continue;
            };

            if hidden_records.len() < config.min_samples {
                continue;
            }

            // Build matched pairs
            let hidden_value_map: HashMap<u32, f32> = hidden_records
                .iter()
                .filter_map(|r| r.value.map(|v| (r.obs_index, v)))
                .collect();

            let hidden_activation_map: HashMap<u32, f32> = hidden_records
                .iter()
                .map(|r| (r.obs_index, r.activation))
                .collect();

            let mut matched_values = Vec::new();
            let mut matched_activations = Vec::new();

            for record in *input_records {
                if let (Some(&value), Some(&activation)) = (
                    hidden_value_map.get(&record.obs_index),
                    hidden_activation_map.get(&record.obs_index),
                ) {
                    matched_values.push(value);
                    matched_activations.push(activation);
                }
            }

            if matched_values.len() < config.min_samples {
                continue;
            }

            // Estimate gradient magnitude from value-activation pairs
            // Sort by value and compute finite differences
            let mut pairs: Vec<(f32, f32)> = matched_values
                .iter()
                .zip(matched_activations.iter())
                .map(|(&v, &a)| (v, a))
                .collect();
            pairs.sort_by(|a, b| a.0.total_cmp(&b.0));

            // Compute maximum gradient in the data
            let mut max_gradient: f32 = 0.0;
            for window in pairs.windows(2) {
                let dv = window[1].0 - window[0].0;
                let da = window[1].1 - window[0].1;
                if dv.abs() > 1e-6 {
                    let gradient = (da / dv).abs();
                    max_gradient = max_gradient.max(gradient);
                }
            }

            // Amplify by weight
            let effective_gradient = max_gradient * weight.abs();

            if effective_gradient < config.gradient_threshold {
                continue;
            }

            // Compute threshold proximity
            // For TANH: threshold is around value = 0
            // For LOGISTIC: threshold is around value = 0
            let mean_value = compute_mean(&matched_values);
            // Squash strings are normalised to uppercase at load time (Issue #753, #771).
            let threshold_proximity = match squash {
                "TANH" | "HARD_TANH" | "SOFTSIGN" => {
                    // Threshold at 0, steepest when |value| < 1
                    1.0 / (1.0 + mean_value.abs())
                }
                "LOGISTIC" => {
                    // Threshold at 0, steepest when |value| < 2
                    1.0 / (1.0 + (mean_value - bias).abs())
                }
                _ => 0.5,
            };

            // Determine recommended action
            let recommended_action = if threshold_proximity > 0.5 {
                // Near threshold - add dampening neuron
                "addNeuron".to_string()
            } else if mean_value.abs() < 1.0 {
                // Operating near threshold - shift bias
                "setBias".to_string()
            } else {
                // Reduce weight to lower gradient
                "setWeight".to_string()
            };

            // Estimated improvement
            let sample_factor = (matched_values.len() as f32 / 1000.0).min(1.0);
            let gradient_factor = ((effective_gradient - config.gradient_threshold)
                / config.gradient_threshold)
                .min(1.0);
            let estimated_improvement = 0.001 * (sample_factor * 0.3 + gradient_factor * 0.7);

            // Find target output
            let target_uuid = synapse_map
                .get(hidden_uuid)
                .and_then(|conns| conns.first())
                .map_or_else(|| "output-0".to_string(), |(to, _)| to.to_string());

            candidates.push(ThresholdEffectCandidate {
                input_neuron_uuid: input_uuid.to_string(),
                intermediate_neuron_uuid: Some(hidden_uuid.to_string()),
                target_neuron_uuid: target_uuid,
                gradient_magnitude: effective_gradient,
                threshold_proximity,
                recommended_action,
                sample_count: matched_values.len(),
                estimated_improvement,
            });
        }
    }

    // Sort by estimated improvement (best first)
    candidates.sort_by(|a, b| b.estimated_improvement.total_cmp(&a.estimated_improvement));

    candidates
}

/// Convert dominant input candidates to coordinated structural candidates.
///
/// Each dominant input produces a `SetWeight` coordinated candidate to reduce
/// the input's excessive leverage.
pub fn dominant_inputs_to_coordinated_candidates(
    candidates: &[DominantInputCandidate],
) -> Vec<CoordinatedStructuralCandidateJson> {
    let mut results = Vec::with_capacity(candidates.len());

    for c in candidates {
        results.push(CoordinatedStructuralCandidateJson {
            operations: vec![CoordinatedStructuralOpJson::SetWeight {
                from_neuron_uuid: c.input_neuron_uuid.clone(),
                to_neuron_uuid: c.target_neuron_uuid.clone(),
                weight: c.recommended_weight,
            }],
            expected_creature_score_gain: c.estimated_improvement,
            comment: Some(format!(
                "Dominant input {} → {}: sensitivity {:.2}, leverage {:.2}, weight {:.3} → {:.3} to reduce brittleness ({} samples)",
                c.input_neuron_uuid, c.target_neuron_uuid, c.sensitivity_score, c.leverage_ratio,
                c.current_weight, c.recommended_weight, c.sample_count
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

/// Convert threshold effect candidates to coordinated structural candidates.
///
/// Each threshold effect produces an `AddNeuron`, `SetBias`, or `SetWeight`
/// coordinated candidate depending on the recommended action.
pub fn threshold_effects_to_coordinated_candidates(
    candidates: &[ThresholdEffectCandidate],
) -> Vec<CoordinatedStructuralCandidateJson> {
    let mut results = Vec::with_capacity(candidates.len());

    for c in candidates {
        let operations = match c.recommended_action.as_str() {
            "addNeuron" => {
                // Add a dampening neuron between input and the steep neuron
                let new_uuid =
                    format!("dampening-{}-{}", c.input_neuron_uuid, c.target_neuron_uuid);
                vec![CoordinatedStructuralOpJson::AddNeuron {
                    neuron_uuid: new_uuid.clone(),
                    neuron_type: "hidden".to_string(),
                    squash: "IDENTITY".to_string(), // IDENTITY for linear dampening
                    bias: 0.0,
                    insert_before_neuron_uuid: c.intermediate_neuron_uuid.clone(),
                }]
            }
            "setBias" => {
                // Shift bias to move operating point away from threshold
                let bias_shift = if c.threshold_proximity > 0.5 {
                    0.5
                } else {
                    -0.5
                };
                vec![CoordinatedStructuralOpJson::SetBias {
                    neuron_uuid: c
                        .intermediate_neuron_uuid
                        .clone()
                        .unwrap_or_else(|| c.target_neuron_uuid.clone()),
                    bias: bias_shift,
                }]
            }
            _ => {
                // Default: setWeight to reduce gradient
                vec![CoordinatedStructuralOpJson::SetWeight {
                    from_neuron_uuid: c.input_neuron_uuid.clone(),
                    to_neuron_uuid: c
                        .intermediate_neuron_uuid
                        .clone()
                        .unwrap_or_else(|| c.target_neuron_uuid.clone()),
                    weight: c.gradient_magnitude * WEIGHT_REDUCTION_FACTOR,
                }]
            }
        };

        results.push(CoordinatedStructuralCandidateJson {
            operations,
            expected_creature_score_gain: c.estimated_improvement,
            comment: Some(format!(
                "Threshold effect {} → {}: gradient {:.2}, threshold proximity {:.2} → {} to reduce brittleness ({} samples)",
                c.input_neuron_uuid,
                c.intermediate_neuron_uuid.as_deref().unwrap_or(&c.target_neuron_uuid),
                c.gradient_magnitude, c.threshold_proximity, c.recommended_action, c.sample_count
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
