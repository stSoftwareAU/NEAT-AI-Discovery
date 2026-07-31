//! Record building from training data.

use anyhow::{Context, Result};

use crate::RecordDiscoveryInput;
use crate::parquet_format::ParquetRecordWriter;
use crate::types::DiscoverRecord;

/// Process training data and write discovery records to Parquet.
///
/// Iterates through training records, builds `DiscoverRecord` batches from
/// pre-computed neuron data, and writes them atomically per observation.
pub fn process_training_data(
    input: &RecordDiscoveryInput,
    obs_indices: &[u32],
    non_input_neuron_count: usize,
    writer: &mut ParquetRecordWriter,
) -> Result<()> {
    let mut wrote_any_records = false;

    // Precompute the fixed input-neuron UUID strings once for the whole batch
    // (`input-0`, `input-1`, …) instead of re-allocating them per observation
    // (Issue #1368). Size to the widest training record so every per-observation
    // index is an in-bounds cache lookup followed by a single clone.
    let max_inputs = input
        .training_data
        .iter()
        .map(|record| record.input.len())
        .max()
        .unwrap_or(0)
        .max(input.creature.input);
    let input_uuids: Vec<String> = (0..max_inputs)
        .map(|index| format!("input-{index}"))
        .collect();

    // Issue #1867: checked once up front — `creature.input` is caller-supplied
    // and a wrapped sum would size every per-observation batch wrongly.
    let records_per_sample = non_input_neuron_count
        .checked_add(input.creature.input)
        .ok_or_else(|| anyhow::anyhow!("Discovery records per sample would overflow usize"))?;

    for (relative_idx, training_record) in input.training_data.iter().enumerate() {
        let obs_index_u32 = obs_indices[relative_idx];

        let mut batch_records = Vec::with_capacity(records_per_sample);

        // Use pre-computed neuron_data if available (from TypeScript)
        // Otherwise, we would need to activate the creature here (not implemented)
        if let Some(neuron_data) = &training_record.neuron_data {
            // Process each neuron from pre-computed data
            for neuron_info in neuron_data {
                // Skip input neurons and non-existent neurons (match TypeScript behaviour)
                let neuron = match input
                    .creature
                    .neurons
                    .iter()
                    .find(|n| n.uuid == neuron_info.neuron_uuid)
                {
                    Some(n) => n,
                    None => continue, // Skip non-existent neurons to prevent invalid discovery data
                };

                if neuron.neuron_type == "input" {
                    continue;
                }

                let record = DiscoverRecord::new(
                    obs_index_u32,
                    neuron_info.neuron_uuid.clone(),
                    neuron_info.value,
                    neuron_info.activation,
                    neuron_info.errors.clone(),
                );

                batch_records.push(record);
            }
        } else {
            // No pre-computed data - this should not happen in normal operation
            // TypeScript should always provide neuron_data
            return Err(anyhow::anyhow!(
                "No pre-computed neuron_data provided. TypeScript must compute activations and errors before calling Rust."
            ));
        }

        // Record input neuron activations for GPU-assisted analysis. UUIDs are
        // looked up from the precomputed cache rather than formatted per record.
        for (input_index, value) in training_record.input.iter().enumerate() {
            let record = DiscoverRecord::new(
                obs_index_u32,
                input_uuids[input_index].clone(),
                Some(*value),
                *value,
                Vec::new(),
            );

            batch_records.push(record);
        }

        if !batch_records.is_empty() {
            writer
                .write_records(&batch_records)
                .context("Failed to write discovery batch to Parquet")?;
            wrote_any_records = true;
        }
    }

    if !wrote_any_records {
        return Err(anyhow::anyhow!(
            "No discovery records were generated from the training data"
        ));
    }

    Ok(())
}
