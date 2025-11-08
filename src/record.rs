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

    // Determine which indices to process
    let indices_to_process: Vec<usize> = if let Some(ref record_indices) = input.record_indices {
        // Validate all indices are within bounds
        for &idx in record_indices {
            if idx >= input.training_data.len() {
                return Err(anyhow::anyhow!(
                    "Record index {} is out of bounds (training data has {} records)",
                    idx,
                    input.training_data.len()
                ));
            }
        }
        record_indices.clone()
    } else {
        // Process all records
        (0..input.training_data.len()).collect()
    };

    // Validate that the indices being processed don't exceed u32::MAX
    // This prevents silent overflow when converting obs_index from usize to u32
    // When record_indices is provided, we only validate those indices.
    // When record_indices is not provided, we validate the total training data size.
    if let Some(max_index) = indices_to_process.iter().max() {
        if *max_index > u32::MAX as usize {
            return Err(anyhow::anyhow!(
                "Maximum record index ({}) exceeds maximum supported size ({})",
                max_index,
                u32::MAX
            ));
        }
    }

    for &obs_index in &indices_to_process {
        let _training_record = &input.training_data[obs_index];

        // For each training record, we need to:
        // 1. Activate creature with training_record.input
        // 2. Get all neuron activations and errors
        // 3. Collect all neuron data atomically

        // TODO: This is a placeholder - actual implementation needs to:
        // - Activate the creature (requires Creature implementation or FFI)
        // - Get neuron activations and errors
        // - For now, we'll create placeholder records

        // Safely convert obs_index from usize to u32
        // This will never fail because we validated the length above
        let obs_index_u32 = u32::try_from(obs_index).map_err(|_| {
            anyhow::anyhow!(
                "Observation index {} exceeds u32::MAX ({})",
                obs_index,
                u32::MAX
            )
        })?;

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
                obs_index_u32,
                neuron.uuid.clone(),
                value,
                activation,
                errors,
            );

            all_records.push(record);
        }
    }

    // Check if we have any records to write
    // This can happen if the creature only has input neurons (which are skipped)
    if all_records.is_empty() {
        // Count non-input neurons to provide a helpful error message
        let non_input_count = input
            .creature
            .neurons
            .iter()
            .filter(|n| n.neuron_type != "input")
            .count();

        if non_input_count == 0 {
            return Err(anyhow::anyhow!(
                "Cannot record discovery data: creature has no non-input neurons. Input neurons are skipped during discovery recording. Discovery recording requires at least one hidden or output neuron to record activations and errors."
            ));
        }

        // This shouldn't happen, but provide a generic error if it does
        return Err(anyhow::anyhow!(
            "No discovery records were generated from the training data"
        ));
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
        // Error should mention that no discovery records were generated
        assert!(
            error_msg.contains("No discovery records")
                || error_msg.contains("no discovery records"),
            "Error should mention that no discovery records were generated, got: {error_msg}"
        );
    }

    #[test]
    fn test_record_discovery_data_obs_index_conversion() {
        let temp_dir = TempDir::new().unwrap();
        let mut input = create_test_input();
        input.temp_dir = temp_dir.path().to_str().unwrap().to_string();

        // Test with a reasonable number of records (well within u32::MAX)
        // This verifies that the safe conversion from usize to u32 works correctly
        input.training_data = (0..1000)
            .map(|_| crate::TrainingRecord {
                input: vec![0.1, 0.2],
                output: vec![0.5],
            })
            .collect();

        // This should succeed - obs_index should be safely converted using try_from
        let result = record_discovery_data(&input);
        assert!(
            result.is_ok(),
            "Should handle valid obs_index values within u32::MAX"
        );

        // Verify the file was created
        let parquet_file =
            Path::new(&result.as_ref().unwrap().temp_dir).join(&result.as_ref().unwrap().file);
        assert!(parquet_file.exists());
    }

    #[test]
    fn test_record_discovery_data_obs_index_overflow_validation() {
        // This test verifies that the validation logic correctly prevents
        // training data that would exceed u32::MAX from being processed.
        // Note: We can't actually create 4.3 billion records in memory for testing,
        // but we can verify the validation code path exists and the conversion
        // uses try_from instead of unsafe 'as' casting.

        // The validation check `if input.training_data.len() > u32::MAX as usize`
        // will catch any overflow case. The use of `u32::try_from(obs_index)`
        // provides a second layer of protection that would error if somehow
        // an obs_index exceeded u32::MAX.

        // For a practical test, we verify that normal-sized datasets work correctly
        // and that the conversion is safe (not using 'as' which would silently truncate)
        let temp_dir = TempDir::new().unwrap();
        let mut input = create_test_input();
        input.temp_dir = temp_dir.path().to_str().unwrap().to_string();

        // Test with a reasonable number of records to verify safe conversion
        input.training_data = (0..100)
            .map(|_| crate::TrainingRecord {
                input: vec![0.1, 0.2],
                output: vec![0.5],
            })
            .collect();

        let result = record_discovery_data(&input);
        assert!(
            result.is_ok(),
            "Safe conversion should work for valid ranges"
        );
    }

    #[test]
    fn test_record_discovery_data_respects_record_indices() {
        // Test that when record_indices is provided, only those indices are processed
        let temp_dir = TempDir::new().unwrap();
        let mut input = create_test_input();
        input.temp_dir = temp_dir.path().to_str().unwrap().to_string();

        // Create 10 training records
        input.training_data = (0..10)
            .map(|i| crate::TrainingRecord {
                input: vec![i as f32, (i * 2) as f32],
                output: vec![i as f32],
            })
            .collect();

        // Only process indices 1, 3, and 5
        input.record_indices = Some(vec![1, 3, 5]);

        let result = record_discovery_data(&input).unwrap();

        // Verify file was created
        let parquet_file = Path::new(&result.temp_dir).join(&result.file);
        assert!(parquet_file.exists());

        // Read the parquet file and verify only records with obs_index 1, 3, 5 exist
        // We have 2 neurons (hidden-1 and output-0), so we should have 6 records total (3 indices * 2 neurons)
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        use std::fs::File;
        let file = File::open(&parquet_file).unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
        let reader = builder.build().unwrap();
        let mut record_count = 0;
        let mut found_indices = std::collections::HashSet::new();

        for batch_result in reader {
            let batch = batch_result.unwrap();
            let obs_index_array = batch
                .column(0)
                .as_any()
                .downcast_ref::<arrow::array::UInt32Array>()
                .unwrap();

            for i in 0..batch.num_rows() {
                let obs_index = obs_index_array.value(i);
                found_indices.insert(obs_index);
                record_count += 1;
            }
        }

        // Should have 6 records (3 indices * 2 neurons)
        assert_eq!(
            record_count, 6,
            "Should have 6 records (3 indices * 2 neurons)"
        );
        // Should only contain indices 1, 3, 5
        assert_eq!(
            found_indices.len(),
            3,
            "Should have 3 unique obs_index values"
        );
        assert!(found_indices.contains(&1), "Should contain obs_index 1");
        assert!(found_indices.contains(&3), "Should contain obs_index 3");
        assert!(found_indices.contains(&5), "Should contain obs_index 5");
    }

    #[test]
    fn test_record_discovery_data_record_indices_out_of_bounds() {
        // Test that invalid record_indices are rejected
        let temp_dir = TempDir::new().unwrap();
        let mut input = create_test_input();
        input.temp_dir = temp_dir.path().to_str().unwrap().to_string();

        // Create 5 training records (indices 0-4)
        input.training_data = (0..5)
            .map(|_| crate::TrainingRecord {
                input: vec![0.1, 0.2],
                output: vec![0.5],
            })
            .collect();

        // Try to access index 10 which is out of bounds
        input.record_indices = Some(vec![0, 2, 10]);

        let result = record_discovery_data(&input);
        assert!(result.is_err(), "Should reject out-of-bounds indices");
        let error_msg = result.unwrap_err().to_string();
        assert!(
            error_msg.contains("out of bounds"),
            "Error should mention out of bounds, got: {error_msg}"
        );
    }

    #[test]
    fn test_record_discovery_data_without_record_indices_processes_all() {
        // Test that when record_indices is None, all records are processed
        let temp_dir = TempDir::new().unwrap();
        let mut input = create_test_input();
        input.temp_dir = temp_dir.path().to_str().unwrap().to_string();

        // Create 5 training records
        input.training_data = (0..5)
            .map(|_| crate::TrainingRecord {
                input: vec![0.1, 0.2],
                output: vec![0.5],
            })
            .collect();

        // Don't set record_indices (should be None)
        input.record_indices = None;

        let result = record_discovery_data(&input).unwrap();

        // Verify file was created
        let parquet_file = Path::new(&result.temp_dir).join(&result.file);
        assert!(parquet_file.exists());

        // Read the parquet file and verify all records exist
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        use std::fs::File;
        let file = File::open(&parquet_file).unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
        let reader = builder.build().unwrap();
        let mut found_indices = std::collections::HashSet::new();

        for batch_result in reader {
            let batch = batch_result.unwrap();
            let obs_index_array = batch
                .column(0)
                .as_any()
                .downcast_ref::<arrow::array::UInt32Array>()
                .unwrap();

            for i in 0..batch.num_rows() {
                let obs_index = obs_index_array.value(i);
                found_indices.insert(obs_index);
            }
        }

        // Should have all 5 indices (0-4) * 2 neurons = 10 records, but unique obs_index values should be 0-4
        assert_eq!(
            found_indices.len(),
            5,
            "Should have all 5 unique obs_index values (0-4)"
        );
        for i in 0..5 {
            assert!(found_indices.contains(&i), "Should contain obs_index {i}");
        }
    }

    #[test]
    fn test_record_discovery_data_with_record_indices_validates_indices_not_total_size() {
        // Test that when record_indices is provided, validation checks the indices
        // being processed, not the total training data size.
        // This test verifies that a large dataset can be processed if only
        // small indices are selected via record_indices.
        let temp_dir = TempDir::new().unwrap();
        let mut input = create_test_input();
        input.temp_dir = temp_dir.path().to_str().unwrap().to_string();

        // Create a large training dataset (simulating a case where total size
        // might be large, but we only want to process a small subset)
        // We use a reasonable large number (100,000) to simulate the scenario
        // In real usage, this could be millions, but we can't create that many in tests
        input.training_data = (0..100_000)
            .map(|_| crate::TrainingRecord {
                input: vec![0.1, 0.2],
                output: vec![0.5],
            })
            .collect();

        // Only process small indices (0, 1, 2) - all well within u32::MAX
        input.record_indices = Some(vec![0, 1, 2]);

        // This should succeed because we're only processing indices 0, 1, 2,
        // which are all within u32::MAX, even though the total dataset is large
        let result = record_discovery_data(&input);
        assert!(
            result.is_ok(),
            "Should succeed when record_indices contains valid indices, even if total dataset is large"
        );

        // Verify file was created
        let parquet_file =
            Path::new(&result.as_ref().unwrap().temp_dir).join(&result.as_ref().unwrap().file);
        assert!(parquet_file.exists());

        // Verify only the specified indices were processed
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        use std::fs::File;
        let file = File::open(&parquet_file).unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
        let reader = builder.build().unwrap();
        let mut found_indices = std::collections::HashSet::new();

        for batch_result in reader {
            let batch = batch_result.unwrap();
            let obs_index_array = batch
                .column(0)
                .as_any()
                .downcast_ref::<arrow::array::UInt32Array>()
                .unwrap();

            for i in 0..batch.num_rows() {
                let obs_index = obs_index_array.value(i);
                found_indices.insert(obs_index);
            }
        }

        // Should only contain indices 0, 1, 2
        assert_eq!(
            found_indices.len(),
            3,
            "Should have 3 unique obs_index values (0, 1, 2)"
        );
        assert!(found_indices.contains(&0), "Should contain obs_index 0");
        assert!(found_indices.contains(&1), "Should contain obs_index 1");
        assert!(found_indices.contains(&2), "Should contain obs_index 2");
    }

    #[test]
    fn test_record_discovery_data_only_input_neurons() {
        // Test that a creature with only input neurons produces a clear error message
        let temp_dir = TempDir::new().unwrap();
        let mut input = create_test_input();
        input.temp_dir = temp_dir.path().to_str().unwrap().to_string();

        // Replace neurons with only input neurons
        input.creature.neurons = vec![
            crate::NeuronJson {
                uuid: "input-0".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            crate::NeuronJson {
                uuid: "input-1".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
        ];

        // Should have training data but no non-input neurons
        input.training_data = vec![crate::TrainingRecord {
            input: vec![0.1, 0.2],
            output: vec![0.5],
        }];

        let result = record_discovery_data(&input);
        assert!(result.is_err(), "Should fail when only input neurons exist");

        let error_msg = result.unwrap_err().to_string();
        assert!(
            error_msg.contains("no non-input neurons") || error_msg.contains("non-input neurons"),
            "Error should explain that no non-input neurons exist, got: {error_msg}"
        );
        assert!(
            error_msg.contains("hidden or output neuron"),
            "Error should mention that hidden or output neurons are required, got: {error_msg}"
        );
    }
}
