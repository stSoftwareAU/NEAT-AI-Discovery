//! Main visualisation snapshot export pipeline.
//!
//! Reads Parquet recordings, builds per-neuron and per-synapse columnar data,
//! computes impact scores and reconstruction checks, then writes the full
//! snapshot to a JSON file.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss
)] // Intentional numeric casts for GPU/neural network computation (Issue #873)

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};

use crate::CreatureJson;
use crate::focus::compute_impacts_public;
use crate::parquet_format::read_records_from_parquet_with_limit;
use crate::types::DiscoverRecord;

use super::stats::{apply_squash, compute_stats, json_safe_f32};
use super::timestamp::chrono_lite_now;
use super::types::*;

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

    // Issue #1869: the dense grid below is O(neurons × observations) regardless
    // of how sparse the recording is — bound it before the first allocation.
    super::dense_bound::ensure_dense_snapshot_fits(records_by_neuron.len(), obs_indices.len())?;

    // Build neuron recordings (columnar)
    let mut neurons_recording: HashMap<String, NeuronRecording> = HashMap::new();

    for (neuron_uuid, records) in &records_by_neuron {
        let n = obs_indices.len();
        let mut activation = vec![0.0f32; n];
        let mut value: Vec<Option<f32>> = vec![None; n];
        let mut errors: Vec<Vec<f32>> = vec![Vec::new(); n];

        for record in records {
            if let Some(&pos) = obs_index_pos.get(&record.obs_index) {
                // Sanitise non-finite values so the exported JSON remains valid and
                // downstream consumers don't see unexpected nulls in float arrays.
                activation[pos] = if record.activation.is_finite() {
                    record.activation
                } else {
                    0.0
                };

                value[pos] = record.value.filter(|v| v.is_finite());

                // We intentionally drop NaN/±Infinity error values. These represent
                // corrupt data and will otherwise poison aggregates (MAE/MSE) and/or
                // serialise as nulls inside float arrays.
                errors[pos] = record
                    .errors
                    .iter()
                    .copied()
                    .filter(|e| e.is_finite())
                    .collect();
            }
        }

        // Compute stats
        let (mean_act, var_act, min_act, max_act) = compute_stats(&activation);

        // Flatten errors for stats
        let flat_errors: Vec<f32> = errors.iter().flat_map(|e| e.iter().copied()).collect();
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

            let safe_weight = json_safe_f32(syn.weight);

            // Get from_neuron activations
            let contribution: Vec<f32> =
                if let Some(from_recording) = neurons_recording.get(&syn.from_uuid) {
                    from_recording
                        .activation
                        .iter()
                        .map(|&a| json_safe_f32(a * safe_weight))
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
                    weight: safe_weight,
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
                let mut reconstructed_value = json_safe_f32(neuron.bias);
                for syn in &inbound {
                    if let Some(from_recording) = neurons_recording.get(&syn.from_uuid) {
                        reconstructed_value += json_safe_f32(
                            from_recording.activation[pos] * json_safe_f32(syn.weight),
                        );
                    }
                }

                reconstructed_value = json_safe_f32(reconstructed_value);
                let reconstructed_activation =
                    json_safe_f32(apply_squash(&neuron.squash, reconstructed_value));

                let recorded_value = recording.value[pos];
                let recorded_activation = recording.activation[pos];

                let value_delta = json_safe_f32(
                    recorded_value.map_or(0.0, |rv| (rv - reconstructed_value).abs()),
                );
                let activation_delta =
                    json_safe_f32((recorded_activation - reconstructed_activation).abs());

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
            worst_samples.sort_by(|a, b| b.activation_delta.total_cmp(&a.activation_delta));
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
    write_snapshot_json(file, &snapshot, out_file)?;

    Ok(stats)
}

/// Serialise `snapshot` to `writer` as JSON and flush explicitly.
///
/// `BufWriter`'s `Drop` flushes any buffered bytes but **discards** the I/O
/// error, so relying on drop-flush can return `Ok` while the buffered tail of
/// the JSON is silently lost (disk full, quota exceeded, I/O error). We flush
/// explicitly and propagate the error so a truncated snapshot fails loudly
/// instead of masquerading as success (Issue #1750).
fn write_snapshot_json<W: Write>(
    writer: W,
    snapshot: &VisualisationSnapshot,
    out_file: &str,
) -> Result<()> {
    let mut writer = BufWriter::new(writer);
    serde_json::to_writer(&mut writer, snapshot)
        .with_context(|| format!("Failed to write JSON to: {out_file}"))?;
    writer
        .flush()
        .with_context(|| format!("Failed to flush JSON snapshot to: {out_file}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{self, Write};

    /// A `Write` that buffers small writes fine but always fails on `flush`,
    /// mimicking a disk-full / quota-exceeded error surfacing only when the
    /// buffered tail is flushed to the underlying file.
    struct FlushFailsWriter;

    impl Write for FlushFailsWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            // Accept the bytes so `serde_json::to_writer` succeeds; the failure
            // must surface at flush time, not during serialisation.
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Err(io::Error::other("simulated flush failure"))
        }
    }

    fn minimal_snapshot() -> VisualisationSnapshot {
        VisualisationSnapshot {
            meta: SnapshotMeta {
                exported_at: "1970-01-01T00:00:00Z".to_string(),
                discovery_version: "test".to_string(),
                parquet_file: "test.parquet".to_string(),
                notes: None,
            },
            creature: CreatureJson {
                neurons: vec![],
                synapses: vec![],
                input: 0,
                output: 0,
            },
            recording: RecordingData {
                obs_indices: vec![],
                neurons: HashMap::new(),
            },
            derived: DerivedData {
                impacts_by_neuron_uuid: HashMap::new(),
                synapses: HashMap::new(),
                reconstruction_checks: None,
            },
        }
    }

    /// Regression test for Issue #1750: a flush error must propagate as `Err`
    /// rather than being swallowed by `BufWriter::drop`. Against the unfixed
    /// code (which dropped the writer without an explicit flush) this returned
    /// `Ok` and lost the buffered tail silently.
    #[test]
    fn write_snapshot_json_propagates_flush_error() {
        let snapshot = minimal_snapshot();
        let result = write_snapshot_json(FlushFailsWriter, &snapshot, "out.json");
        assert!(
            result.is_err(),
            "a flush failure must surface as Err, not be swallowed",
        );
        let msg = format!("{:#}", result.unwrap_err());
        assert!(
            msg.contains("Failed to flush JSON snapshot to: out.json"),
            "error should name the flush failure and target file, got: {msg}",
        );
    }

    /// A writer with no I/O errors serialises and flushes successfully.
    #[test]
    fn write_snapshot_json_succeeds_on_healthy_writer() {
        let snapshot = minimal_snapshot();
        let mut buf: Vec<u8> = Vec::new();
        write_snapshot_json(&mut buf, &snapshot, "out.json").expect("healthy writer must succeed");
        // Round-trips back to a snapshot, proving the full JSON was written.
        let parsed: VisualisationSnapshot =
            serde_json::from_slice(&buf).expect("written JSON must deserialise");
        assert_eq!(parsed.meta.discovery_version, "test");
    }
}
