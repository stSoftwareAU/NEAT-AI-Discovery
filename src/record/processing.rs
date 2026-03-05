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

    for (relative_idx, training_record) in input.training_data.iter().enumerate() {
        let obs_index_u32 = obs_indices[relative_idx];

        let mut batch_records = Vec::with_capacity(non_input_neuron_count + input.creature.input);

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

        // Record input neuron activations for GPU-assisted analysis
        for (input_index, value) in training_record.input.iter().enumerate() {
            let input_uuid = format!("input-{input_index}");

            let record =
                DiscoverRecord::new(obs_index_u32, input_uuid, Some(*value), *value, Vec::new());

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
