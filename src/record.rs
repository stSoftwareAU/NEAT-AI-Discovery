//! Record discovery data logic

use anyhow::{Context, Result};
use std::fs;
use std::path::Path;

use crate::parquet_format::write_records_to_parquet;
use crate::types::DiscoverRecord;
use crate::RecordDiscoveryInput;

/// Result of recording discovery data
#[derive(Debug)]
pub struct RecordResult {
    pub temp_dir: String,
    pub file: String,
}

/// Record discovery data from training records
///
/// This function processes training data and records neuron activations and errors.
/// For each training record, it collects all neuron data atomically and writes to Parquet.
/// Since the training dataset is already randomized, parallelization is allowed,
/// but each training record must be processed atomically (all neurons written together).
pub fn record_discovery_data(input: &RecordDiscoveryInput) -> Result<RecordResult> {
    // Create temp directory
    let temp_dir = Path::new(&input.temp_dir);
    let temp_dir_str = input.temp_dir.clone();
    fs::create_dir_all(temp_dir)
        .with_context(|| format!("Failed to create temp directory: {temp_dir_str}"))?;

    // Collect all discovery records
    // For now, we'll process sequentially to ensure atomicity
    // TODO: Add parallel processing with proper synchronization
    let mut all_records = Vec::new();

    for (obs_index, _training_record) in input.training_data.iter().enumerate() {
        // For each training record, we need to:
        // 1. Activate creature with training_record.input
        // 2. Get all neuron activations and errors
        // 3. Collect all neuron data atomically

        // TODO: This is a placeholder - actual implementation needs to:
        // - Activate the creature (requires Creature implementation or FFI)
        // - Get neuron activations and errors
        // - For now, we'll create placeholder records

        // Process each neuron
        for neuron in &input.creature.neurons {
            // Skip input neurons for now (match TypeScript behavior)
            if neuron.neuron_type == "input" {
                continue;
            }

            // Placeholder: In real implementation, these would come from creature activation
            let activation = 0.0; // TODO: Get from creature.activate()
            let value = None; // TODO: Get from creature state
            let errors = vec![]; // TODO: Get from creature.record()

            let record = DiscoverRecord::new(
                obs_index as u32,
                neuron.uuid.clone(),
                value,
                activation,
                errors,
            );

            all_records.push(record);
        }
    }

    // Write all records to Parquet file
    let parquet_file = temp_dir.join("discovery_data.parquet");
    let parquet_path = parquet_file
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("Invalid file path"))?;

    write_records_to_parquet(parquet_path, &all_records)
        .context("Failed to write records to Parquet")?;

    Ok(RecordResult {
        temp_dir: input.temp_dir.clone(),
        file: "discovery_data.parquet".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_input() -> RecordDiscoveryInput {
        RecordDiscoveryInput {
            creature: crate::CreatureJson {
                neurons: vec![
                    crate::NeuronJson {
                        uuid: "hidden-1".to_string(),
                        neuron_type: "hidden".to_string(),
                        squash: "TANH".to_string(),
                        bias: 0.0,
                    },
                    crate::NeuronJson {
                        uuid: "output-0".to_string(),
                        neuron_type: "output".to_string(),
                        squash: "IDENTITY".to_string(),
                        bias: 0.0,
                    },
                ],
                synapses: vec![],
                input: 2,
                output: 1,
            },
            training_data: vec![
                crate::TrainingRecord {
                    input: vec![0.1, 0.2],
                    output: vec![0.5],
                },
                crate::TrainingRecord {
                    input: vec![0.3, 0.4],
                    output: vec![0.6],
                },
            ],
            temp_dir: ".discovery/test".to_string(),
            binary_file_path: None,
            record_indices: None,
            timeout_seconds: None,
        }
    }

    #[test]
    fn test_record_discovery_data_creates_file() {
        let temp_dir = TempDir::new().unwrap();
        let mut input = create_test_input();
        input.temp_dir = temp_dir.path().to_str().unwrap().to_string();

        let result = record_discovery_data(&input).unwrap();

        let parquet_file = Path::new(&result.temp_dir).join(&result.file);
        assert!(parquet_file.exists());
    }

    #[test]
    fn test_record_discovery_data_empty_training_data() {
        let temp_dir = TempDir::new().unwrap();
        let mut input = create_test_input();
        input.temp_dir = temp_dir.path().to_str().unwrap().to_string();
        input.training_data = vec![];

        // Empty training data should result in no records, which causes an error
        let result = record_discovery_data(&input);
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        // Error should mention "No records" or "Failed to write"
        assert!(error_msg.contains("No records") || error_msg.contains("Failed to write"));
    }
}
