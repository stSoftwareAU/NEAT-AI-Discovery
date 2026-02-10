//! High noise-to-signal ratio detection module (Issue #434).
//!
//! Part of the "Brilliant but Brittle" initiative (Issue #432). This module identifies
//! neurons and synapses with high noise-to-signal ratios that contribute to brittle
//! predictions when bad or missing observations wildly affect outputs.
//!
//! See `docs/DISCOVERY_TYPES.md` § "Noise-to-Signal Detection" for full documentation.
//!
//! ## Detection Criteria
//!
//! ### Noisy Neurons
//! A neuron has high noise-to-signal ratio if:
//! 1. **Low activation variance**: The neuron's activation varies little across samples,
//!    indicating it doesn't respond to meaningful input patterns.
//! 2. **High error variance**: The neuron's error varies significantly, indicating
//!    unpredictable contribution to network output.
//! 3. **Poor correlation**: Activation changes don't correlate with error reduction.
//!
//! ### Noisy Synapses
//! A synapse amplifies noise if:
//! 1. **Large weight**: Amplifies variance from upstream neurons.
//! 2. **Noisy source**: Source neuron has high variance but poor error correlation.
//! 3. **Low signal contribution**: The synapse contributes more variance than signal.
//!
//! ## Recommended Actions
//!
//! - `removeNeuron`: For hidden neurons with poor signal-to-noise ratio
//! - `removeSynapse`: For synapses that amplify noise without signal benefit
//! - `setWeight`: To reduce weight of partially useful but noisy connections

use std::collections::HashSet;
use std::env;

use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson};

// MIN_SAMPLES_FOR_DETECTION moved to constants.rs (Issue #424)
use super::constants::MIN_DISCOVERY_SAMPLE_COUNT as MIN_SAMPLES_FOR_DETECTION;

/// Default threshold for noise-to-signal ratio to flag a neuron.
/// A ratio > 1.0 means noise dominates signal.
const DEFAULT_NOISE_SIGNAL_THRESHOLD: f32 = 2.0;

/// Minimum activation variance to consider a neuron active.
/// Below this, the neuron is effectively constant and should be handled elsewhere.
const MIN_ACTIVATION_VARIANCE: f32 = 1e-8;

/// Minimum weight magnitude to consider a synapse for noise amplification.
/// Synapses with smaller weights contribute negligible noise.
const MIN_WEIGHT_FOR_NOISE_CHECK: f32 = 0.1;

/// Maximum weight reduction factor for setWeight recommendations.
const WEIGHT_REDUCTION_FACTOR: f32 = 0.5;

/// Result of detecting a noisy neuron.
#[derive(Debug, Clone)]
pub struct NoisyNeuronCandidate {
    /// UUID of the noisy neuron.
    pub neuron_uuid: String,
    /// Ratio of error variance to activation variance.
    /// Values > 1.0 indicate noise dominates signal.
    pub noise_to_signal_ratio: f32,
    /// Variance of the neuron's activation across samples.
    pub activation_variance: f32,
    /// Variance of the neuron's error across samples.
    pub error_variance: f32,
    /// Number of samples analysed.
    pub sample_count: usize,
    /// Estimated creature score improvement from removing this neuron.
    pub estimated_improvement: f32,
}

/// Result of detecting a noisy synapse.
#[derive(Debug, Clone)]
pub struct NoisySynapseCandidate {
    /// UUID of the source neuron.
    pub from_neuron_uuid: String,
    /// UUID of the target neuron.
    pub to_neuron_uuid: String,
    /// Current weight of the synapse.
    pub weight: f32,
    /// Estimated noise contribution through this synapse.
    pub noise_contribution: f32,
    /// Estimated signal contribution through this synapse.
    pub signal_contribution: f32,
    /// Number of samples analysed.
    pub sample_count: usize,
    /// Estimated creature score improvement from the recommended action.
    pub estimated_improvement: f32,
    /// Recommended action: "removeSynapse" or "setWeight".
    pub recommended_action: String,
}

/// Get the noise-to-signal threshold from environment variable or default.
fn noise_signal_threshold_from_env() -> f32 {
    env::var("NEAT_AI_DISCOVERY_NOISE_SIGNAL_THRESHOLD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_NOISE_SIGNAL_THRESHOLD)
}

/// Compute variance of a slice of f32 values.
fn compute_variance(values: &[f32]) -> f32 {
    if values.len() < 2 {
        return 0.0;
    }
    let n = values.len() as f32;
    let mean: f32 = values.iter().sum::<f32>() / n;
    values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n
}

/// Compute the mean of a slice of f32 values.
fn compute_mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f32>() / values.len() as f32
}

/// Detect neurons with high noise-to-signal ratio.
///
/// # Arguments
/// * `creature` - The creature's network topology (neurons and synapses).
/// * `neuron_records` - List of `(neuron_uuid, records)` tuples with recorded data.
///
/// # Returns
/// A list of `NoisyNeuronCandidate` for neurons with poor signal-to-noise,
/// sorted by estimated improvement (best first).
pub fn detect_noisy_neurons(
    creature: &CreatureJson,
    neuron_records: &[(String, Vec<DiscoverRecord>)],
) -> Vec<NoisyNeuronCandidate> {
    let threshold = noise_signal_threshold_from_env();

    // Identify hidden neurons only (inputs and outputs should not be removed for noise)
    let hidden_uuids: HashSet<&str> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "hidden")
        .map(|n| n.uuid.as_str())
        .collect();

    let mut candidates = Vec::new();

    for (uuid, records) in neuron_records {
        // Only consider hidden neurons
        if !hidden_uuids.contains(uuid.as_str()) {
            continue;
        }

        // Require minimum samples for statistical reliability
        if records.len() < MIN_SAMPLES_FOR_DETECTION {
            continue;
        }

        // Extract activations and errors
        let activations: Vec<f32> = records.iter().map(|r| r.activation).collect();
        let errors: Vec<f32> = records
            .iter()
            .flat_map(|r| r.errors.first().copied())
            .collect();

        if errors.len() < MIN_SAMPLES_FOR_DETECTION {
            continue;
        }

        // Compute variances
        let activation_variance = compute_variance(&activations);
        let error_variance = compute_variance(&errors);

        // Skip neurons with negligible activation variance (constant neurons)
        if activation_variance < MIN_ACTIVATION_VARIANCE {
            continue;
        }

        // Compute noise-to-signal ratio
        // Higher ratio means error variance dominates activation variance
        let noise_to_signal_ratio = error_variance / activation_variance;

        // Only flag if above threshold
        if noise_to_signal_ratio <= threshold {
            continue;
        }

        // Estimated improvement based on noise ratio and sample count
        // More samples and higher ratio = higher confidence in improvement
        let sample_factor = (records.len() as f32 / 1000.0).min(1.0);
        let ratio_factor = ((noise_to_signal_ratio - threshold) / threshold).min(1.0);
        let estimated_improvement = 0.001 * (sample_factor * 0.3 + ratio_factor * 0.7);

        candidates.push(NoisyNeuronCandidate {
            neuron_uuid: uuid.clone(),
            noise_to_signal_ratio,
            activation_variance,
            error_variance,
            sample_count: records.len(),
            estimated_improvement,
        });
    }

    // Sort by estimated improvement (best first)
    candidates.sort_by(|a, b| b.estimated_improvement.total_cmp(&a.estimated_improvement));

    candidates
}

/// Detect synapses that amplify noise from upstream neurons.
///
/// # Arguments
/// * `creature` - The creature's network topology (neurons and synapses).
/// * `neuron_records` - List of `(neuron_uuid, records)` tuples with recorded data.
///
/// # Returns
/// A list of `NoisySynapseCandidate` for synapses that amplify noise,
/// sorted by estimated improvement (best first).
pub fn detect_noisy_synapses(
    creature: &CreatureJson,
    neuron_records: &[(String, Vec<DiscoverRecord>)],
) -> Vec<NoisySynapseCandidate> {
    use std::collections::HashMap;

    // Build records lookup
    let records_map: HashMap<&str, &Vec<DiscoverRecord>> = neuron_records
        .iter()
        .map(|(uuid, records)| (uuid.as_str(), records))
        .collect();

    let mut candidates = Vec::new();

    for synapse in &creature.synapses {
        // Only consider synapses with meaningful weight
        if synapse.weight.abs() < MIN_WEIGHT_FOR_NOISE_CHECK {
            continue;
        }

        // Get source and target records
        let Some(source_records) = records_map.get(synapse.from_uuid.as_str()) else {
            continue;
        };
        let Some(target_records) = records_map.get(synapse.to_uuid.as_str()) else {
            continue;
        };

        // Require minimum samples
        if source_records.len() < MIN_SAMPLES_FOR_DETECTION
            || target_records.len() < MIN_SAMPLES_FOR_DETECTION
        {
            continue;
        }

        // Extract source activations
        let source_activations: Vec<f32> = source_records.iter().map(|r| r.activation).collect();
        let source_variance = compute_variance(&source_activations);

        // Extract target errors (aligned by obs_index)
        let target_error_map: HashMap<u32, f32> = target_records
            .iter()
            .filter_map(|r| r.errors.first().map(|&e| (r.obs_index, e)))
            .collect();

        // Match source activations with target errors by obs_index
        let mut matched_pairs: Vec<(f32, f32)> = Vec::new();
        for record in source_records.iter() {
            if let Some(&error) = target_error_map.get(&record.obs_index) {
                matched_pairs.push((record.activation, error));
            }
        }

        if matched_pairs.len() < MIN_SAMPLES_FOR_DETECTION {
            continue;
        }

        // Compute noise contribution: |weight| × source_variance
        // This is the variance amplification through the synapse
        let noise_contribution = synapse.weight.abs() * source_variance.sqrt();

        // Compute signal contribution using correlation
        // If source activation correlates negatively with target error,
        // the synapse is helping reduce error (positive signal)
        let source_mean = compute_mean(&matched_pairs.iter().map(|(a, _)| *a).collect::<Vec<_>>());
        let error_mean = compute_mean(&matched_pairs.iter().map(|(_, e)| *e).collect::<Vec<_>>());

        let mut covariance = 0.0f32;
        for &(activation, error) in &matched_pairs {
            covariance += (activation - source_mean) * (error - error_mean);
        }
        covariance /= matched_pairs.len() as f32;

        // Signal contribution is based on how well the synapse could reduce error
        // Negative correlation (activation up → error down) is good signal
        let signal_contribution = (-covariance * synapse.weight.abs()).max(0.0);

        // Only flag if noise significantly exceeds signal
        if noise_contribution <= signal_contribution * 2.0 {
            continue;
        }

        // Determine recommended action
        // If signal contribution is non-zero, recommend weight reduction
        // Otherwise recommend removal
        let recommended_action = if signal_contribution > 0.01 {
            "setWeight".to_string()
        } else {
            "removeSynapse".to_string()
        };

        // Estimated improvement
        let noise_ratio = if signal_contribution > 1e-8 {
            noise_contribution / signal_contribution
        } else {
            noise_contribution * 10.0 // Heavily penalise zero signal
        };
        let sample_factor = (matched_pairs.len() as f32 / 1000.0).min(1.0);
        let estimated_improvement = 0.001 * sample_factor * (1.0 - 1.0 / (1.0 + noise_ratio));

        candidates.push(NoisySynapseCandidate {
            from_neuron_uuid: synapse.from_uuid.clone(),
            to_neuron_uuid: synapse.to_uuid.clone(),
            weight: synapse.weight,
            noise_contribution,
            signal_contribution,
            sample_count: matched_pairs.len(),
            estimated_improvement,
            recommended_action,
        });
    }

    // Sort by estimated improvement (best first)
    candidates.sort_by(|a, b| b.estimated_improvement.total_cmp(&a.estimated_improvement));

    candidates
}

/// Convert noisy neuron candidates into coordinated structural candidates.
///
/// Each noisy neuron produces a `RemoveNeuron` coordinated candidate.
/// The NEAT-AI controller will validate the removal through ablation testing.
pub fn noisy_neurons_to_coordinated_candidates(
    candidates: &[NoisyNeuronCandidate],
) -> Vec<CoordinatedStructuralCandidateJson> {
    let mut results = Vec::with_capacity(candidates.len());

    for c in candidates {
        results.push(CoordinatedStructuralCandidateJson {
            operations: vec![CoordinatedStructuralOpJson::RemoveNeuron {
                neuron_uuid: c.neuron_uuid.clone(),
            }],
            expected_creature_score_gain: c.estimated_improvement,
            comment: Some(format!(
                "High noise-to-signal neuron {}: ratio {:.2}, activation var {:.2e}, error var {:.2e}, {} samples → remove to reduce brittleness",
                c.neuron_uuid, c.noise_to_signal_ratio, c.activation_variance, c.error_variance, c.sample_count
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

/// Convert noisy synapse candidates into coordinated structural candidates.
///
/// Each noisy synapse produces either a `RemoveSynapse` or `SetWeight` coordinated candidate,
/// depending on whether the synapse has any useful signal contribution.
pub fn noisy_synapses_to_coordinated_candidates(
    candidates: &[NoisySynapseCandidate],
) -> Vec<CoordinatedStructuralCandidateJson> {
    let mut results = Vec::with_capacity(candidates.len());

    for c in candidates {
        let operations = if c.recommended_action == "setWeight" {
            // Reduce weight to dampen noise while preserving some signal
            let reduced_weight = c.weight * WEIGHT_REDUCTION_FACTOR;
            vec![CoordinatedStructuralOpJson::SetWeight {
                from_neuron_uuid: c.from_neuron_uuid.clone(),
                to_neuron_uuid: c.to_neuron_uuid.clone(),
                weight: reduced_weight,
            }]
        } else {
            vec![CoordinatedStructuralOpJson::RemoveSynapse {
                from_neuron_uuid: c.from_neuron_uuid.clone(),
                to_neuron_uuid: c.to_neuron_uuid.clone(),
            }]
        };

        results.push(CoordinatedStructuralCandidateJson {
            operations,
            expected_creature_score_gain: c.estimated_improvement,
            comment: Some(format!(
                "Noise-amplifying synapse {} → {}: weight {:.3}, noise {:.3}, signal {:.3}, {} samples → {} to reduce brittleness",
                c.from_neuron_uuid, c.to_neuron_uuid, c.weight, c.noise_contribution, c.signal_contribution, c.sample_count, c.recommended_action
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
