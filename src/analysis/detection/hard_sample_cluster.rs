//! Hard sample clustering detector (Issue #642).
//!
//! Identifies groups of observations (`obs_index` values) that are consistently
//! high-error across all output neurons, indicating a systematic structural gap.
//! The network lacks capacity to handle that region of the input space.
//!
//! ## Approach
//!
//! 1. **Aggregate per-observation error**: For each `obs_index`, compute the mean
//!    absolute error across all output neurons.
//! 2. **Identify hard observations**: Observations with mean error above a threshold
//!    (based on the overall error distribution) are classified as "hard".
//! 3. **Analyse activation patterns**: Compare input activations between hard and easy
//!    observations to find which inputs discriminate the two groups.
//! 4. **Produce structural candidates**: `addNeuron` + `addSynapse` candidates targeting
//!    the dominant input regions that predict hard samples.
//!
//! ## Difference from `sample_weighted.rs`
//!
//! `sample_weighted.rs` analyses error per neuron independently. This module joins
//! error data **across neurons** by `obs_index` to find observations that are
//! systematically hard for the entire network.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use std::collections::{HashMap, HashSet};

use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson};

use super::helpers::build_record_map;
use crate::analysis::constants::MIN_DISCOVERY_SAMPLE_COUNT;

// =============================================================================
// Constants
// =============================================================================

/// Minimum hard-to-easy error ratio to report a cluster.
const DEFAULT_MIN_HARD_EASY_RATIO: f32 = 2.0;

/// Minimum number of hard observations to form a cluster.
const DEFAULT_MIN_HARD_OBS: usize = 5;

/// Minimum absolute difference in mean activation between hard and easy groups
/// for an input to be considered "dominant".
const DOMINANT_INPUT_ACTIVATION_DIFF: f32 = 0.1;

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for hard sample cluster detection.
#[derive(Debug, Clone)]
pub struct HardSampleClusterConfig {
    /// Minimum hard-to-easy error ratio to report a cluster.
    pub min_hard_easy_ratio: f32,
    /// Minimum number of hard observations required.
    pub min_hard_obs: usize,
    /// Minimum total observations for reliable analysis.
    pub min_samples: usize,
}

impl Default for HardSampleClusterConfig {
    fn default() -> Self {
        Self {
            min_hard_easy_ratio: DEFAULT_MIN_HARD_EASY_RATIO,
            min_hard_obs: DEFAULT_MIN_HARD_OBS,
            min_samples: MIN_DISCOVERY_SAMPLE_COUNT,
        }
    }
}

// =============================================================================
// Detection Types
// =============================================================================

/// A cluster of observations that are consistently hard across all outputs.
#[derive(Debug, Clone)]
pub struct HardSampleCluster {
    /// Observation indices in the hard cluster.
    pub hard_obs_indices: Vec<u32>,
    /// Mean absolute error across all outputs for hard observations.
    pub mean_error: f32,
    /// Mean absolute error for easy observations.
    pub easy_mean_error: f32,
    /// Ratio of hard mean error to easy mean error.
    pub hard_to_easy_ratio: f32,
    /// Input neuron UUIDs whose activations differ most between hard and easy groups.
    pub dominant_input_uuids: Vec<String>,
    /// Number of output neurons analysed.
    pub output_neuron_count: usize,
    /// Estimated creature score improvement from addressing this cluster.
    pub estimated_improvement: f32,
}

// =============================================================================
// Core Detection
// =============================================================================

/// Detect hard sample clusters across all output neurons.
///
/// For each `obs_index`, aggregates the mean absolute error across all output neurons.
/// Observations with error above the median + 1 std dev are classified as "hard".
/// If the hard-to-easy ratio exceeds the configured threshold, a cluster is reported.
///
/// # Arguments
/// * `creature` - The creature's network topology.
/// * `neuron_records` - List of `(neuron_uuid, records)` tuples.
/// * `config` - Detection configuration.
///
/// # Returns
/// A list of `HardSampleCluster` sorted by estimated improvement (best first).
pub fn detect_hard_sample_clusters(
    creature: &CreatureJson,
    neuron_records: &[(String, impl AsRef<[DiscoverRecord]>)],
    config: &HardSampleClusterConfig,
) -> Vec<HardSampleCluster> {
    if neuron_records.is_empty() {
        return Vec::new();
    }

    // Identify output neuron UUIDs
    let output_uuids: HashSet<&str> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "output")
        .map(|n| n.uuid.as_str())
        .collect();

    // Identify input neuron UUIDs
    let input_uuids: HashSet<&str> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "input")
        .map(|n| n.uuid.as_str())
        .collect();

    // Build records lookup
    let records_map = build_record_map(neuron_records);

    // Step 1: Aggregate per-observation mean absolute error across all output neurons
    let obs_errors = aggregate_obs_errors(&output_uuids, &records_map);
    if obs_errors.len() < config.min_samples {
        return Vec::new();
    }

    // Step 2: Compute threshold for hard observations
    let errors: Vec<f32> = obs_errors.values().copied().collect();
    let mean_error = errors.iter().sum::<f32>() / errors.len() as f32;
    let variance = errors
        .iter()
        .map(|&e| (e - mean_error).powi(2))
        .sum::<f32>()
        / errors.len() as f32;
    let std_dev = variance.sqrt();

    // Threshold: observations with error above mean + 1 std dev are "hard"
    let threshold = mean_error + std_dev;

    // Step 3: Split into hard and easy observations
    let mut hard_obs: Vec<u32> = Vec::new();
    let mut easy_obs: Vec<u32> = Vec::new();
    let mut hard_error_sum = 0.0_f32;
    let mut easy_error_sum = 0.0_f32;

    for (&obs_idx, &error) in &obs_errors {
        if error > threshold {
            hard_obs.push(obs_idx);
            hard_error_sum += error;
        } else {
            easy_obs.push(obs_idx);
            easy_error_sum += error;
        }
    }

    if hard_obs.len() < config.min_hard_obs {
        return Vec::new();
    }

    let hard_mean = hard_error_sum / hard_obs.len() as f32;
    let easy_mean = if easy_obs.is_empty() {
        0.0
    } else {
        easy_error_sum / easy_obs.len() as f32
    };

    let ratio = if easy_mean > f32::EPSILON {
        hard_mean / easy_mean
    } else if hard_mean > f32::EPSILON {
        hard_mean / f32::EPSILON
    } else {
        1.0
    };

    if ratio < config.min_hard_easy_ratio {
        return Vec::new();
    }

    // Step 4: Find dominant input neurons
    let hard_set: HashSet<u32> = hard_obs.iter().copied().collect();
    let easy_set: HashSet<u32> = easy_obs.iter().copied().collect();
    let dominant_inputs = find_dominant_inputs(&input_uuids, &records_map, &hard_set, &easy_set);

    // Sort hard obs for deterministic output
    hard_obs.sort();

    let output_neuron_count = output_uuids.len();

    // Estimate improvement: proportional to error reduction potential
    let hard_fraction = hard_obs.len() as f32 / obs_errors.len() as f32;
    let estimated_improvement =
        (hard_mean - easy_mean) * hard_fraction * (output_neuron_count as f32).sqrt() * 0.01;

    let cluster = HardSampleCluster {
        hard_obs_indices: hard_obs,
        mean_error: hard_mean,
        easy_mean_error: easy_mean,
        hard_to_easy_ratio: ratio,
        dominant_input_uuids: dominant_inputs,
        output_neuron_count,
        estimated_improvement: estimated_improvement.max(f32::EPSILON),
    };

    vec![cluster]
}

/// Aggregate per-observation mean absolute error across all output neurons.
///
/// For each `obs_index`, computes the mean of the mean-absolute-errors from
/// every output neuron that has a record at that index.
fn aggregate_obs_errors(
    output_uuids: &HashSet<&str>,
    records_map: &HashMap<&str, &[DiscoverRecord]>,
) -> HashMap<u32, f32> {
    // obs_index -> (sum_of_mean_abs_error, count_of_neurons)
    let mut obs_aggregated: HashMap<u32, (f32, u32)> = HashMap::new();

    for &uuid in output_uuids {
        if let Some(records) = records_map.get(uuid) {
            for r in *records {
                let abs_error = if r.errors.is_empty() {
                    0.0
                } else {
                    let mean =
                        r.errors.iter().map(|e| e.abs()).sum::<f32>() / r.errors.len() as f32;
                    if mean.is_finite() { mean } else { 0.0 }
                };

                let entry = obs_aggregated.entry(r.obs_index).or_insert((0.0, 0));
                entry.0 += abs_error;
                entry.1 += 1;
            }
        }
    }

    // Convert to mean error per observation
    obs_aggregated
        .into_iter()
        .map(|(obs_idx, (sum, count))| {
            let mean = if count > 0 { sum / count as f32 } else { 0.0 };
            (obs_idx, mean)
        })
        .collect()
}

/// Find input neurons whose mean activation differs most between hard and easy groups.
fn find_dominant_inputs(
    input_uuids: &HashSet<&str>,
    records_map: &HashMap<&str, &[DiscoverRecord]>,
    hard_obs: &HashSet<u32>,
    easy_obs: &HashSet<u32>,
) -> Vec<String> {
    let mut scored_inputs: Vec<(String, f32)> = Vec::new();

    for &input_uuid in input_uuids {
        let Some(records) = records_map.get(input_uuid) else {
            continue;
        };

        let mut hard_sum = 0.0_f32;
        let mut hard_count = 0_u32;
        let mut easy_sum = 0.0_f32;
        let mut easy_count = 0_u32;

        for r in *records {
            if hard_obs.contains(&r.obs_index) {
                hard_sum += r.activation;
                hard_count += 1;
            } else if easy_obs.contains(&r.obs_index) {
                easy_sum += r.activation;
                easy_count += 1;
            }
        }

        if hard_count == 0 || easy_count == 0 {
            continue;
        }

        let hard_mean = hard_sum / hard_count as f32;
        let easy_mean = easy_sum / easy_count as f32;
        let diff = (hard_mean - easy_mean).abs();

        if diff >= DOMINANT_INPUT_ACTIVATION_DIFF {
            scored_inputs.push((input_uuid.to_string(), diff));
        }
    }

    // Sort by activation difference (largest first)
    scored_inputs.sort_by(|a, b| b.1.total_cmp(&a.1));

    scored_inputs.into_iter().map(|(uuid, _)| uuid).collect()
}

// =============================================================================
// Candidate Conversion
// =============================================================================

/// Generate a deterministic UUID for a hard-sample hidden neuron.
fn hard_sample_neuron_uuid(cluster_index: usize, hard_obs_count: usize) -> String {
    let key = format!("hard-sample-cluster|{cluster_index}|{hard_obs_count}");
    // FNV-1a 64-bit
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in key.as_bytes() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("hsc-{hash:016x}")
}

/// Convert hard sample clusters into coordinated structural candidates.
///
/// Each cluster produces a candidate that adds a hidden neuron connecting
/// dominant inputs to all output neurons, increasing capacity for the
/// hard sample region.
pub fn hard_sample_clusters_to_coordinated_candidates(
    clusters: &[HardSampleCluster],
    creature: &CreatureJson,
) -> Vec<CoordinatedStructuralCandidateJson> {
    let first_output_uuid = creature
        .neurons
        .iter()
        .find(|n| n.neuron_type == "output")
        .map(|n| n.uuid.clone());

    let output_uuids: Vec<&str> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "output")
        .map(|n| n.uuid.as_str())
        .collect();

    let mut results = Vec::new();

    for (idx, cluster) in clusters.iter().enumerate() {
        let new_uuid = hard_sample_neuron_uuid(idx, cluster.hard_obs_indices.len());

        let mut operations = Vec::new();

        // Add hidden neuron before first output
        operations.push(CoordinatedStructuralOpJson::AddNeuron {
            neuron_uuid: new_uuid.clone(),
            neuron_type: "hidden".to_string(),
            squash: "TANH".to_string(),
            bias: 0.0,
            insert_before_neuron_uuid: first_output_uuid.clone(),
        });

        // Connect dominant inputs to the new neuron
        let input_sources: &[String] = if cluster.dominant_input_uuids.is_empty() {
            &[]
        } else {
            &cluster.dominant_input_uuids
        };

        for input_uuid in input_sources {
            operations.push(CoordinatedStructuralOpJson::AddSynapse {
                from_neuron_uuid: input_uuid.clone(),
                to_neuron_uuid: new_uuid.clone(),
                weight: 0.5,
            });
        }

        // Fallback: if no dominant inputs, connect from first input
        if input_sources.is_empty()
            && let Some(first_input) = creature.neurons.iter().find(|n| n.neuron_type == "input")
        {
            operations.push(CoordinatedStructuralOpJson::AddSynapse {
                from_neuron_uuid: first_input.uuid.clone(),
                to_neuron_uuid: new_uuid.clone(),
                weight: 0.5,
            });
        }

        // Connect new neuron to all outputs
        for &output_uuid in &output_uuids {
            operations.push(CoordinatedStructuralOpJson::AddSynapse {
                from_neuron_uuid: new_uuid.clone(),
                to_neuron_uuid: output_uuid.to_string(),
                weight: 0.1,
            });
        }

        let input_list = if cluster.dominant_input_uuids.is_empty() {
            "auto-selected".to_string()
        } else {
            cluster.dominant_input_uuids.join(", ")
        };

        results.push(CoordinatedStructuralCandidateJson {
            remove_neuron_compensation: None,
        constant_neuron_bias_fold: None,
            operations,
            expected_creature_score_gain: cluster.estimated_improvement,
            comment: Some(format!(
                "Hard sample cluster: {} hard observations (mean error {:.3}, easy mean {:.3}, ratio {:.1}x, {} outputs) → add hidden neuron from [{}]",
                cluster.hard_obs_indices.len(),
                cluster.mean_error,
                cluster.easy_mean_error,
                cluster.hard_to_easy_ratio,
                cluster.output_neuron_count,
                input_list,
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
