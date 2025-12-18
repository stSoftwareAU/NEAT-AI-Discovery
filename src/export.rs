//! Visualisation snapshot export module
//!
//! Exports a debug-friendly JSON snapshot of a creature and its recorded
//! activations/errors for use with the NEAT-AI-Explore visualiser.
//!
//! This is an optional debug tool that does not affect existing analysis behaviour.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufWriter;

use crate::focus::compute_impacts_public;
use crate::parquet_format::read_records_from_parquet_with_limit;
use crate::types::DiscoverRecord;
use crate::CreatureJson;

/// Snapshot metadata
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SnapshotMeta {
    pub exported_at: String,
    pub discovery_version: String,
    pub parquet_file: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub notes: Option<String>,
}

/// Per-neuron stats (aggregates)
#[derive(Debug, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct NeuronRecordingStats {
    pub mean_activation: f32,
    pub activation_variance: f32,
    pub activation_min: f32,
    pub activation_max: f32,
    /// Arithmetic mean of errors (can be misleading as +/- cancel out)
    pub mean_error: f32,
    /// Mean Absolute Error - used in focus neuron ranking
    pub mean_absolute_error: f32,
    /// Mean Squared Error - matches production creature scoring
    pub mean_squared_error: f32,
    pub error_variance: f32,
    pub error_min: f32,
    pub error_max: f32,
    pub record_count: usize,
}

/// Per-neuron recorded data (columnar format)
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NeuronRecording {
    /// Activation values per obsIndex (aligned with obsIndices array)
    pub activation: Vec<f32>,
    /// Value (pre-activation) per obsIndex; null encoded as JSON null
    pub value: Vec<Option<f32>>,
    /// Errors per obsIndex (each element is a vector of error values)
    pub errors: Vec<Vec<f32>>,
    /// Aggregated stats
    pub stats: NeuronRecordingStats,
}

/// Per-synapse stats (aggregates)
#[derive(Debug, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct SynapseStats {
    pub mean_contribution: f32,
    pub contribution_variance: f32,
    pub contribution_min: f32,
    pub contribution_max: f32,
}

/// Per-synapse derived data (columnar format)
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SynapseDerived {
    pub from_uuid: String,
    pub to_uuid: String,
    pub weight: f32,
    /// Contribution = fromActivation * weight per obsIndex
    pub contribution: Vec<f32>,
    /// Aggregated stats
    pub stats: SynapseStats,
}

/// Reconstruction check results for a single neuron
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReconstructionCheck {
    pub neuron_uuid: String,
    pub squash: String,
    pub bias: f32,
    /// max|recordedValue - reconstructedValue|
    pub max_value_delta: f32,
    /// max|recordedActivation - reconstructedActivation|
    pub max_activation_delta: f32,
    /// mean|valueDelta|
    pub mean_value_delta: f32,
    /// mean|activationDelta|
    pub mean_activation_delta: f32,
    /// Top-K worst obsIndex + deltas
    pub worst_samples: Vec<ReconstructionSample>,
}

/// A single sample's reconstruction delta
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReconstructionSample {
    pub obs_index: u32,
    pub recorded_value: Option<f32>,
    pub reconstructed_value: f32,
    pub value_delta: f32,
    pub recorded_activation: f32,
    pub reconstructed_activation: f32,
    pub activation_delta: f32,
}

/// Derived data section
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DerivedData {
    /// Impact scores keyed by neuron UUID
    pub impacts_by_neuron_uuid: HashMap<String, f32>,
    /// Synapse-level derived data keyed by "fromUuid->toUuid"
    pub synapses: HashMap<String, SynapseDerived>,
    /// Reconstruction checks per neuron (non-input neurons only)
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub reconstruction_checks: Option<Vec<ReconstructionCheck>>,
}

/// Recording data section
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecordingData {
    /// Ordered obsIndex values
    pub obs_indices: Vec<u32>,
    /// Per-neuron recorded data keyed by neuron UUID
    pub neurons: HashMap<String, NeuronRecording>,
}

/// The full snapshot structure written to JSON
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VisualisationSnapshot {
    pub meta: SnapshotMeta,
    pub creature: CreatureJson,
    pub recording: RecordingData,
    pub derived: DerivedData,
}

/// Export options
pub struct ExportOptions {
    pub include_per_synapse_series: bool,
    pub include_reconstruction_checks: bool,
    pub max_obs: Option<u32>,
    pub top_k_worst_samples: usize,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            include_per_synapse_series: true,
            include_reconstruction_checks: true,
            max_obs: None,
            top_k_worst_samples: 20,
        }
    }
}

/// Export result statistics
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportStats {
    pub obs_count: usize,
    pub neuron_count: usize,
    pub synapse_count: usize,
    pub output_count: usize,
}

/// Apply a squash function by name
fn apply_squash(squash: &str, value: f32) -> f32 {
    match squash.to_uppercase().as_str() {
        "IDENTITY" => value,
        "TANH" => value.tanh(),
        "LOGISTIC" | "SIGMOID" => 1.0 / (1.0 + (-value).exp()),
        "RELU" => value.max(0.0),
        "LEAKYRELU" => {
            if value >= 0.0 {
                value
            } else {
                0.01 * value
            }
        }
        "STEP" => {
            if value > 0.0 {
                1.0
            } else {
                0.0
            }
        }
        "BIPOLAR" => {
            if value > 0.0 {
                1.0
            } else {
                -1.0
            }
        }
        "HARD_TANH" | "CLIPPED" => value.clamp(-1.0, 1.0),
        "SOFTSIGN" => value / (1.0 + value.abs()),
        "SOFTPLUS" => (1.0 + value.exp()).ln(),
        "ELU" => {
            if value >= 0.0 {
                value
            } else {
                value.exp() - 1.0
            }
        }
        "SELU" => {
            let alpha = 1.673_263_2_f32;
            let scale = 1.050_701_f32;
            if value >= 0.0 {
                scale * value
            } else {
                scale * alpha * (value.exp() - 1.0)
            }
        }
        "GELU" => {
            // Approximate GELU
            let cdf = 0.5 * (1.0 + (0.797_884_6_f32 * (value + 0.044715 * value.powi(3))).tanh());
            value * cdf
        }
        "MISH" => value * ((1.0 + value.exp()).ln()).tanh(),
        "SWISH" => value / (1.0 + (-value).exp()),
        "ARCTAN" => value.atan(),
        "BENT_IDENTITY" => ((value * value + 1.0).sqrt() - 1.0) / 2.0 + value,
        "RELU6" => value.clamp(0.0, 6.0),
        "GAUSSIAN" => (-value * value).exp(),
        "SINE" | "SIN" => value.sin(),
        "COSINE" | "COS" => value.cos(),
        "ABSOLUTE" | "ABS" => value.abs(),
        "INVERSE" => -value,
        "COMPLEMENT" => 1.0 - value,
        // Default to identity for unknown
        _ => value,
    }
}

/// Compute stats from a slice of f32 values
fn compute_stats(values: &[f32]) -> (f32, f32, f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let n = values.len() as f32;
    let sum: f32 = values.iter().filter(|v| v.is_finite()).sum();
    let mean = sum / n;
    let variance: f32 = values
        .iter()
        .filter(|v| v.is_finite())
        .map(|v| (v - mean).powi(2))
        .sum::<f32>()
        / n;
    let min = values
        .iter()
        .filter(|v| v.is_finite())
        .cloned()
        .fold(f32::INFINITY, f32::min);
    let max = values
        .iter()
        .filter(|v| v.is_finite())
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    (mean, variance, min, max)
}

/// Export a visualisation snapshot to JSON
pub fn export_visualisation_snapshot(
    parquet_file: &str,
    creature: &CreatureJson,
    out_file: &str,
    options: &ExportOptions,
) -> Result<ExportStats> {
    // Read records from Parquet (with optional limit for large files)
    let all_records = read_records_from_parquet_with_limit(parquet_file, options.max_obs)
        .with_context(|| format!("Failed to read Parquet file: {parquet_file}"))?;

    // Group records by neuron UUID and collect unique obsIndices
    let mut obs_indices_set = std::collections::HashSet::new();
    let mut records_by_neuron: HashMap<String, Vec<&DiscoverRecord>> = HashMap::new();

    for record in &all_records {
        obs_indices_set.insert(record.obs_index);
        records_by_neuron
            .entry(record.neuron_uuid.clone())
            .or_default()
            .push(record);
    }

    // Sort and optionally limit obsIndices
    let mut obs_indices: Vec<u32> = obs_indices_set.into_iter().collect();
    obs_indices.sort_unstable();

    if let Some(max_obs) = options.max_obs {
        obs_indices.truncate(max_obs as usize);
    }

    // Build obs_index -> position map for alignment
    let obs_index_pos: HashMap<u32, usize> = obs_indices
        .iter()
        .enumerate()
        .map(|(i, &o)| (o, i))
        .collect();

    // Build neuron recordings (columnar)
    let mut neurons_recording: HashMap<String, NeuronRecording> = HashMap::new();

    for (neuron_uuid, records) in &records_by_neuron {
        let n = obs_indices.len();
        let mut activation = vec![0.0f32; n];
        let mut value: Vec<Option<f32>> = vec![None; n];
        let mut errors: Vec<Vec<f32>> = vec![Vec::new(); n];

        for record in records {
            if let Some(&pos) = obs_index_pos.get(&record.obs_index) {
                activation[pos] = record.activation;
                value[pos] = record.value;
                errors[pos] = record.errors.clone();
            }
        }

        // Compute stats
        let (mean_act, var_act, min_act, max_act) = compute_stats(&activation);

        // Flatten errors for stats
        let flat_errors: Vec<f32> = errors.iter().flat_map(|e| e.iter().cloned()).collect();
        let (mean_err, var_err, min_err, max_err) = compute_stats(&flat_errors);

        // Compute Mean Absolute Error (MAE) - used in focus neuron ranking
        let mean_absolute_error = if flat_errors.is_empty() {
            0.0
        } else {
            flat_errors.iter().map(|e| e.abs()).sum::<f32>() / flat_errors.len() as f32
        };

        // Compute Mean Squared Error (MSE) - matches production creature scoring
        let mean_squared_error = if flat_errors.is_empty() {
            0.0
        } else {
            flat_errors.iter().map(|e| e * e).sum::<f32>() / flat_errors.len() as f32
        };

        let stats = NeuronRecordingStats {
            mean_activation: mean_act,
            activation_variance: var_act,
            activation_min: min_act,
            activation_max: max_act,
            mean_error: mean_err,
            mean_absolute_error,
            mean_squared_error,
            error_variance: var_err,
            error_min: min_err,
            error_max: max_err,
            record_count: records.len(),
        };

        neurons_recording.insert(
            neuron_uuid.clone(),
            NeuronRecording {
                activation,
                value,
                errors,
                stats,
            },
        );
    }

    // Add input neurons (input-0, input-1, ...) to recording if they exist in parquet
    // (They may not have explicit recordings - inputs come from observations)

    // Build synapse map for lookup
    let mut synapse_map: HashMap<String, &crate::SynapseJson> = HashMap::new();
    for syn in &creature.synapses {
        let key = format!("{}→{}", syn.from_uuid, syn.to_uuid);
        synapse_map.insert(key, syn);
    }

    // Create a creature with synthetic input neurons added for impact calculation.
    // Input neurons are implied by creature.input count but not in neurons array.
    // The impact calculation iterates over neurons, so we need to add them explicitly.
    let creature_with_inputs = {
        let mut c = creature.clone();
        for i in 0..creature.input {
            let uuid = format!("input-{i}");
            // Only add if not already present
            if !c.neurons.iter().any(|n| n.uuid == uuid) {
                c.neurons.push(crate::NeuronJson {
                    uuid,
                    neuron_type: "input".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 0.0,
                });
            }
        }
        c
    };

    // Compute impacts using existing focus module (with input neurons included)
    let impacts = compute_impacts_public(&creature_with_inputs);
    let impacts_by_neuron_uuid: HashMap<String, f32> = impacts;

    // Build synapse derived data
    let mut synapses_derived: HashMap<String, SynapseDerived> = HashMap::new();

    if options.include_per_synapse_series {
        for syn in &creature.synapses {
            let key = format!("{}→{}", syn.from_uuid, syn.to_uuid);

            // Get from_neuron activations
            let contribution: Vec<f32> =
                if let Some(from_recording) = neurons_recording.get(&syn.from_uuid) {
                    from_recording
                        .activation
                        .iter()
                        .map(|&a| a * syn.weight)
                        .collect()
                } else {
                    // From neuron might be an input neuron not in recordings
                    vec![0.0; obs_indices.len()]
                };

            let (mean_contrib, var_contrib, min_contrib, max_contrib) =
                compute_stats(&contribution);

            synapses_derived.insert(
                key,
                SynapseDerived {
                    from_uuid: syn.from_uuid.clone(),
                    to_uuid: syn.to_uuid.clone(),
                    weight: syn.weight,
                    contribution,
                    stats: SynapseStats {
                        mean_contribution: mean_contrib,
                        contribution_variance: var_contrib,
                        contribution_min: min_contrib,
                        contribution_max: max_contrib,
                    },
                },
            );
        }
    }

    // Build reconstruction checks
    let reconstruction_checks = if options.include_reconstruction_checks {
        let mut checks = Vec::new();

        // Build neuron lookup
        let _neuron_map: HashMap<&str, &crate::NeuronJson> = creature
            .neurons
            .iter()
            .map(|n| (n.uuid.as_str(), n))
            .collect();

        // Build inbound synapses map
        let mut inbound_synapses: HashMap<&str, Vec<&crate::SynapseJson>> = HashMap::new();
        for syn in &creature.synapses {
            inbound_synapses
                .entry(syn.to_uuid.as_str())
                .or_default()
                .push(syn);
        }

        for neuron in &creature.neurons {
            // Skip input/constant neurons - they don't have reconstruction
            if neuron.neuron_type == "input" || neuron.neuron_type == "constant" {
                continue;
            }

            let Some(recording) = neurons_recording.get(&neuron.uuid) else {
                continue;
            };

            let inbound = inbound_synapses
                .get(neuron.uuid.as_str())
                .cloned()
                .unwrap_or_default();

            let mut worst_samples: Vec<ReconstructionSample> = Vec::new();
            let mut sum_value_delta = 0.0f32;
            let mut sum_act_delta = 0.0f32;
            let mut max_value_delta = 0.0f32;
            let mut max_act_delta = 0.0f32;
            let mut count = 0usize;

            for (pos, &obs_index) in obs_indices.iter().enumerate() {
                // Reconstruct value = bias + sum(from_activation * weight)
                let mut reconstructed_value = neuron.bias;
                for syn in &inbound {
                    if let Some(from_recording) = neurons_recording.get(&syn.from_uuid) {
                        reconstructed_value += from_recording.activation[pos] * syn.weight;
                    }
                }

                let reconstructed_activation = apply_squash(&neuron.squash, reconstructed_value);

                let recorded_value = recording.value[pos];
                let recorded_activation = recording.activation[pos];

                let value_delta = recorded_value
                    .map(|rv| (rv - reconstructed_value).abs())
                    .unwrap_or(0.0);
                let activation_delta = (recorded_activation - reconstructed_activation).abs();

                if value_delta.is_finite() {
                    sum_value_delta += value_delta;
                    max_value_delta = max_value_delta.max(value_delta);
                }
                if activation_delta.is_finite() {
                    sum_act_delta += activation_delta;
                    max_act_delta = max_act_delta.max(activation_delta);
                }
                count += 1;

                // Collect sample for potential worst-K
                worst_samples.push(ReconstructionSample {
                    obs_index,
                    recorded_value,
                    reconstructed_value,
                    value_delta,
                    recorded_activation,
                    reconstructed_activation,
                    activation_delta,
                });
            }

            // Sort by activation_delta descending and keep top-K
            worst_samples.sort_by(|a, b| {
                b.activation_delta
                    .partial_cmp(&a.activation_delta)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            worst_samples.truncate(options.top_k_worst_samples);

            let mean_value_delta = if count > 0 {
                sum_value_delta / count as f32
            } else {
                0.0
            };
            let mean_act_delta = if count > 0 {
                sum_act_delta / count as f32
            } else {
                0.0
            };

            checks.push(ReconstructionCheck {
                neuron_uuid: neuron.uuid.clone(),
                squash: neuron.squash.clone(),
                bias: neuron.bias,
                max_value_delta,
                max_activation_delta: max_act_delta,
                mean_value_delta,
                mean_activation_delta: mean_act_delta,
                worst_samples,
            });
        }

        Some(checks)
    } else {
        None
    };

    // Build the snapshot
    let snapshot = VisualisationSnapshot {
        meta: SnapshotMeta {
            exported_at: chrono_lite_now(),
            discovery_version: env!("CARGO_PKG_VERSION").to_string(),
            parquet_file: parquet_file.to_string(),
            notes: None,
        },
        creature: creature.clone(),
        recording: RecordingData {
            obs_indices,
            neurons: neurons_recording,
        },
        derived: DerivedData {
            impacts_by_neuron_uuid,
            synapses: synapses_derived,
            reconstruction_checks,
        },
    };

    // Compute stats
    let stats = ExportStats {
        obs_count: snapshot.recording.obs_indices.len(),
        neuron_count: snapshot.recording.neurons.len(),
        synapse_count: creature.synapses.len(),
        output_count: creature.output,
    };

    // Write to file
    let file = File::create(out_file)
        .with_context(|| format!("Failed to create output file: {out_file}"))?;
    let writer = BufWriter::new(file);
    serde_json::to_writer(writer, &snapshot)
        .with_context(|| format!("Failed to write JSON to: {out_file}"))?;

    Ok(stats)
}

/// Simple ISO 8601 timestamp without external chrono dependency
fn chrono_lite_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = duration.as_secs();

    // Convert to date/time components (UTC)
    let days = secs / 86400;
    let time_secs = secs % 86400;
    let hours = time_secs / 3600;
    let minutes = (time_secs % 3600) / 60;
    let seconds = time_secs % 60;

    // Simple year/month/day calculation (good enough for logging)
    let mut year = 1970;
    let mut remaining_days = days as i64;

    loop {
        let days_in_year = if is_leap_year(year) { 366 } else { 365 };
        if remaining_days < days_in_year {
            break;
        }
        remaining_days -= days_in_year;
        year += 1;
    }

    let days_in_months: [i64; 12] = if is_leap_year(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    let mut month = 1;
    for &days_in_month in &days_in_months {
        if remaining_days < days_in_month {
            break;
        }
        remaining_days -= days_in_month;
        month += 1;
    }
    let day = remaining_days + 1;

    format!("{year:04}{month:02}{day:02}T{hours:02}{minutes:02}{seconds:02}Z")
}

fn is_leap_year(year: i64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_squash_identity() {
        assert!((apply_squash("IDENTITY", 0.5) - 0.5).abs() < 1e-6);
        assert!((apply_squash("IDENTITY", -1.0) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_apply_squash_tanh() {
        assert!((apply_squash("TANH", 0.0) - 0.0).abs() < 1e-6);
        assert!((apply_squash("TANH", 1.0) - 1.0_f32.tanh()).abs() < 1e-6);
    }

    #[test]
    fn test_apply_squash_relu() {
        assert!((apply_squash("RELU", 1.0) - 1.0).abs() < 1e-6);
        assert!((apply_squash("RELU", -1.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_apply_squash_hard_tanh() {
        assert!((apply_squash("HARD_TANH", 0.5) - 0.5).abs() < 1e-6);
        assert!((apply_squash("HARD_TANH", 2.0) - 1.0).abs() < 1e-6);
        assert!((apply_squash("HARD_TANH", -2.0) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compute_stats_empty() {
        let (mean, var, min, max) = compute_stats(&[]);
        assert_eq!(mean, 0.0);
        assert_eq!(var, 0.0);
        assert_eq!(min, 0.0);
        assert_eq!(max, 0.0);
    }

    #[test]
    fn test_compute_stats_single() {
        let (mean, var, min, max) = compute_stats(&[5.0]);
        assert!((mean - 5.0).abs() < 1e-6);
        assert!((var - 0.0).abs() < 1e-6);
        assert!((min - 5.0).abs() < 1e-6);
        assert!((max - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_stats_multiple() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (mean, var, min, max) = compute_stats(&values);
        assert!((mean - 3.0).abs() < 1e-6);
        assert!((var - 2.0).abs() < 1e-6); // variance of [1,2,3,4,5] = 2
        assert!((min - 1.0).abs() < 1e-6);
        assert!((max - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_chrono_lite_now_format() {
        let timestamp = chrono_lite_now();
        // Should be in format YYYYMMDDTHHMMSSZ
        assert_eq!(timestamp.len(), 16);
        assert!(timestamp.ends_with('Z'));
        assert!(timestamp.contains('T'));
    }
}
