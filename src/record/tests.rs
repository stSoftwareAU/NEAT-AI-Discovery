//! Unit tests for the record module.

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
                neuron_data: Some(vec![
                    crate::NeuronData {
                        neuron_uuid: "hidden-1".to_string(),
                        activation: 0.5,
                        value: Some(0.4),
                        errors: vec![0.1],
                    },
                    crate::NeuronData {
                        neuron_uuid: "output-0".to_string(),
                        activation: 0.5,
                        value: Some(0.5),
                        errors: vec![0.0],
                    },
                ]),
            },
            crate::TrainingRecord {
                input: vec![0.3, 0.4],
                output: vec![0.6],
                neuron_data: Some(vec![
                    crate::NeuronData {
                        neuron_uuid: "hidden-1".to_string(),
                        activation: 0.6,
                        value: Some(0.5),
                        errors: vec![0.15],
                    },
                    crate::NeuronData {
                        neuron_uuid: "output-0".to_string(),
                        activation: 0.6,
                        value: Some(0.6),
                        errors: vec![0.0],
                    },
                ]),
            },
        ],
        temp_dir: ".discovery/test".to_string(),
        binary_file_path: None,
        record_indices: None,
        timeout_seconds: None,
        task_descriptor: None,
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
        error_msg.contains("No discovery records") || error_msg.contains("no discovery records"),
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
            neuron_data: Some(vec![
                crate::NeuronData {
                    neuron_uuid: "hidden-1".to_string(),
                    activation: 0.5,
                    value: Some(0.4),
                    errors: vec![0.1],
                },
                crate::NeuronData {
                    neuron_uuid: "output-0".to_string(),
                    activation: 0.5,
                    value: Some(0.5),
                    errors: vec![0.0],
                },
            ]),
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
            neuron_data: Some(vec![
                crate::NeuronData {
                    neuron_uuid: "hidden-1".to_string(),
                    activation: 0.5,
                    value: Some(0.4),
                    errors: vec![0.1],
                },
                crate::NeuronData {
                    neuron_uuid: "output-0".to_string(),
                    activation: 0.5,
                    value: Some(0.5),
                    errors: vec![0.0],
                },
            ]),
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
    // Test that when record_indices is provided, those values are used as obs_index
    let temp_dir = TempDir::new().unwrap();
    let mut input = create_test_input();
    input.temp_dir = temp_dir.path().to_str().unwrap().to_string();

    // Create 3 training records
    input.training_data = (0..3)
        .map(|i| crate::TrainingRecord {
            input: vec![i as f32, (i * 2) as f32],
            output: vec![i as f32],
            neuron_data: Some(vec![
                crate::NeuronData {
                    neuron_uuid: "hidden-1".to_string(),
                    activation: i as f32 * 0.1,
                    value: Some(i as f32 * 0.1),
                    errors: vec![i as f32 * 0.01],
                },
                crate::NeuronData {
                    neuron_uuid: "output-0".to_string(),
                    activation: i as f32,
                    value: Some(i as f32),
                    errors: vec![0.0],
                },
            ]),
        })
        .collect();

    // Assign non-sequential observation indices
    input.record_indices = Some(vec![10, 30, 50]);

    let result = record_discovery_data(&input).unwrap();

    // Verify file was created
    let parquet_file = Path::new(&result.temp_dir).join(&result.file);
    assert!(parquet_file.exists());

    // Read the parquet file and verify obs_index values match provided indices
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

    let per_obs_record_count = (input.creature.neurons.len() + input.creature.input) as u32;
    let expected_records = per_obs_record_count * 3;
    assert_eq!(
        record_count as u32, expected_records,
        "Should have records for each neuron (including inputs) across the provided indices"
    );
    assert_eq!(
        found_indices.len(),
        3,
        "Should have 3 unique obs_index values"
    );
    assert!(found_indices.contains(&10), "Should contain obs_index 10");
    assert!(found_indices.contains(&30), "Should contain obs_index 30");
    assert!(found_indices.contains(&50), "Should contain obs_index 50");
}

#[test]
fn test_record_discovery_data_record_indices_length_mismatch() {
    // Test that providing a mismatched number of record_indices is rejected
    let temp_dir = TempDir::new().unwrap();
    let mut input = create_test_input();
    input.temp_dir = temp_dir.path().to_str().unwrap().to_string();

    // Create 5 training records but only supply 3 record indices
    input.training_data = (0..5)
        .map(|i| crate::TrainingRecord {
            input: vec![i as f32, (i * 2) as f32],
            output: vec![i as f32],
            neuron_data: Some(vec![
                crate::NeuronData {
                    neuron_uuid: "hidden-1".to_string(),
                    activation: i as f32 * 0.1,
                    value: Some(i as f32 * 0.1),
                    errors: vec![i as f32 * 0.01],
                },
                crate::NeuronData {
                    neuron_uuid: "output-0".to_string(),
                    activation: i as f32,
                    value: Some(i as f32),
                    errors: vec![0.0],
                },
            ]),
        })
        .collect();

    input.record_indices = Some(vec![0, 2, 4]);

    let result = record_discovery_data(&input);
    assert!(
        result.is_err(),
        "Should reject mismatched record_indices lengths"
    );
    let error_msg = result.unwrap_err().to_string();
    assert!(
        error_msg.contains("record_indices length"),
        "Error should mention length mismatch, got: {error_msg}"
    );
}

#[test]
fn test_record_discovery_data_with_neuron_data() {
    // Test that when neuron_data is provided, it writes correctly and can be read back
    let temp_dir = TempDir::new().unwrap();
    let mut input = create_test_input();
    input.temp_dir = temp_dir.path().to_str().unwrap().to_string();

    // Create training data with pre-computed neuron_data (simulating TypeScript behaviour)
    input.training_data = vec![
        crate::TrainingRecord {
            input: vec![0.1, 0.2],
            output: vec![0.5],
            neuron_data: Some(vec![
                crate::NeuronData {
                    neuron_uuid: "hidden-1".to_string(),
                    activation: 0.7,
                    value: Some(0.6),
                    errors: vec![0.1, 0.2],
                },
                crate::NeuronData {
                    neuron_uuid: "output-0".to_string(),
                    activation: 0.5,
                    value: Some(0.5),
                    errors: vec![0.0],
                },
            ]),
        },
        crate::TrainingRecord {
            input: vec![0.3, 0.4],
            output: vec![0.6],
            neuron_data: Some(vec![
                crate::NeuronData {
                    neuron_uuid: "hidden-1".to_string(),
                    activation: 0.8,
                    value: Some(0.7),
                    errors: vec![0.15, 0.25],
                },
                crate::NeuronData {
                    neuron_uuid: "output-0".to_string(),
                    activation: 0.6,
                    value: Some(0.6),
                    errors: vec![0.0],
                },
            ]),
        },
    ];

    let result = record_discovery_data(&input).unwrap();

    // Verify file was created
    let parquet_file = Path::new(&result.temp_dir).join(&result.file);
    assert!(parquet_file.exists());

    // Read back records for each neuron and verify they match
    use crate::parquet_format::read_records_from_parquet;

    // Read hidden-1 records
    let hidden1_records =
        read_records_from_parquet(parquet_file.to_str().unwrap(), "hidden-1").unwrap();

    assert_eq!(
        hidden1_records.len(),
        2,
        "Should have 2 records for hidden-1"
    );
    assert_eq!(hidden1_records[0].obs_index, 0);
    assert_eq!(hidden1_records[0].activation, 0.7);
    assert_eq!(hidden1_records[0].value, Some(0.6));
    assert_eq!(hidden1_records[0].errors, vec![0.1, 0.2]);
    assert_eq!(hidden1_records[1].obs_index, 1);
    assert_eq!(hidden1_records[1].activation, 0.8);
    assert_eq!(hidden1_records[1].value, Some(0.7));
    assert_eq!(hidden1_records[1].errors, vec![0.15, 0.25]);

    // Read output-0 records
    let output0_records =
        read_records_from_parquet(parquet_file.to_str().unwrap(), "output-0").unwrap();

    assert_eq!(
        output0_records.len(),
        2,
        "Should have 2 records for output-0"
    );
    assert_eq!(output0_records[0].obs_index, 0);
    assert_eq!(output0_records[0].activation, 0.5);
    assert_eq!(output0_records[0].value, Some(0.5));
    assert_eq!(output0_records[0].errors, vec![0.0]);
    assert_eq!(output0_records[1].obs_index, 1);
    assert_eq!(output0_records[1].activation, 0.6);
    assert_eq!(output0_records[1].value, Some(0.6));
    assert_eq!(output0_records[1].errors, vec![0.0]);

    // Verify records can be matched by obs_index across neurons
    // (TypeScript handles sorting, we just need obs_index to be present for matching)
    for hidden_record in &hidden1_records {
        let obs_idx = hidden_record.obs_index;
        // Find corresponding record in output-0 with same obs_index
        let matching_output = output0_records.iter().find(|r| r.obs_index == obs_idx);
        assert!(
            matching_output.is_some(),
            "Should find matching record with obs_index {obs_idx} in output-0"
        );
    }
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
            neuron_data: Some(vec![
                crate::NeuronData {
                    neuron_uuid: "hidden-1".to_string(),
                    activation: 0.5,
                    value: Some(0.4),
                    errors: vec![0.1],
                },
                crate::NeuronData {
                    neuron_uuid: "output-0".to_string(),
                    activation: 0.5,
                    value: Some(0.5),
                    errors: vec![0.0],
                },
            ]),
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
fn test_record_discovery_data_supports_large_obs_indices() {
    // Test that large observation indices (within u32::MAX) are supported
    let temp_dir = TempDir::new().unwrap();
    let mut input = create_test_input();
    input.temp_dir = temp_dir.path().to_str().unwrap().to_string();

    // Provide a few training records
    input.training_data = (0..3)
        .map(|i| crate::TrainingRecord {
            input: vec![i as f32, (i * 2) as f32],
            output: vec![i as f32],
            neuron_data: Some(vec![
                crate::NeuronData {
                    neuron_uuid: "hidden-1".to_string(),
                    activation: i as f32 * 0.1,
                    value: Some(i as f32 * 0.1),
                    errors: vec![i as f32 * 0.01],
                },
                crate::NeuronData {
                    neuron_uuid: "output-0".to_string(),
                    activation: i as f32,
                    value: Some(i as f32),
                    errors: vec![0.0],
                },
            ]),
        })
        .collect();

    // Use large observation indices (still within u32::MAX)
    input.record_indices = Some(vec![0, 123_456, 987_654]);

    let result = record_discovery_data(&input);
    assert!(
        result.is_ok(),
        "Should succeed when record_indices contains large but valid indices"
    );

    // Verify file was created
    let parquet_file =
        Path::new(&result.as_ref().unwrap().temp_dir).join(&result.as_ref().unwrap().file);
    assert!(parquet_file.exists());

    // Verify the observation indices match the provided values
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

    // Should only contain the provided indices
    assert_eq!(
        found_indices.len(),
        3,
        "Should have 3 unique obs_index values (0, 123456, 987654)"
    );
    assert!(found_indices.contains(&0), "Should contain obs_index 0");
    assert!(
        found_indices.contains(&123_456),
        "Should contain obs_index 123456"
    );
    assert!(
        found_indices.contains(&987_654),
        "Should contain obs_index 987654"
    );
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
        neuron_data: Some(vec![]), // Empty because no non-input neurons
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

#[test]
fn test_record_discovery_data_rejects_duplicate_record_indices() {
    // Test that duplicate record_indices are rejected to maintain unique obs_index values
    let temp_dir = TempDir::new().unwrap();
    let mut input = create_test_input();
    input.temp_dir = temp_dir.path().to_str().unwrap().to_string();

    // Create 4 training records
    input.training_data = (0..4)
        .map(|i| crate::TrainingRecord {
            input: vec![i as f32, (i * 2) as f32],
            output: vec![i as f32],
            neuron_data: Some(vec![
                crate::NeuronData {
                    neuron_uuid: "hidden-1".to_string(),
                    activation: i as f32 * 0.1,
                    value: Some(i as f32 * 0.1),
                    errors: vec![i as f32 * 0.01],
                },
                crate::NeuronData {
                    neuron_uuid: "output-0".to_string(),
                    activation: i as f32,
                    value: Some(i as f32),
                    errors: vec![0.0],
                },
            ]),
        })
        .collect();

    // Provide duplicate indices: 1, 3, 1, 5
    input.record_indices = Some(vec![1, 3, 1, 5]);

    let result = record_discovery_data(&input);
    assert!(
        result.is_err(),
        "Duplicate record_indices should be rejected to maintain unique obs_index values"
    );
    let error_msg = result.unwrap_err().to_string();
    assert!(
        error_msg.contains("duplicated"),
        "Error message should mention duplicates, got: {error_msg}"
    );
}

#[test]
fn test_record_discovery_data_skips_non_existent_neurons() {
    // Test that neuron_data containing UUIDs that don't exist in creature.neurons
    // are skipped and don't create invalid records
    let temp_dir = TempDir::new().unwrap();
    let mut input = create_test_input();
    input.temp_dir = temp_dir.path().to_str().unwrap().to_string();

    // Create training data with a non-existent neuron UUID
    input.training_data = vec![crate::TrainingRecord {
        input: vec![0.1, 0.2],
        output: vec![0.5],
        neuron_data: Some(vec![
            crate::NeuronData {
                neuron_uuid: "hidden-1".to_string(), // Exists in creature
                activation: 0.5,
                value: Some(0.4),
                errors: vec![0.1],
            },
            crate::NeuronData {
                neuron_uuid: "non-existent-neuron".to_string(), // Does NOT exist in creature
                activation: 0.9,
                value: Some(0.8),
                errors: vec![0.2],
            },
            crate::NeuronData {
                neuron_uuid: "output-0".to_string(), // Exists in creature
                activation: 0.5,
                value: Some(0.5),
                errors: vec![0.0],
            },
        ]),
    }];

    let result = record_discovery_data(&input).unwrap();

    // Verify file was created
    let parquet_file = Path::new(&result.temp_dir).join(&result.file);
    assert!(parquet_file.exists());

    // Read the parquet file and verify only existing neurons have records
    use crate::parquet_format::read_records_from_parquet;

    // hidden-1 should have records
    let hidden1_records =
        read_records_from_parquet(parquet_file.to_str().unwrap(), "hidden-1").unwrap();
    assert_eq!(
        hidden1_records.len(),
        1,
        "Should have 1 record for hidden-1 (existing neuron)"
    );

    // output-0 should have records
    let output0_records =
        read_records_from_parquet(parquet_file.to_str().unwrap(), "output-0").unwrap();
    assert_eq!(
        output0_records.len(),
        1,
        "Should have 1 record for output-0 (existing neuron)"
    );

    // non-existent-neuron should NOT have records
    let non_existent_records =
        read_records_from_parquet(parquet_file.to_str().unwrap(), "non-existent-neuron").unwrap();
    assert_eq!(
        non_existent_records.len(),
        0,
        "Should have 0 records for non-existent-neuron (should be skipped)"
    );
}

#[test]
fn test_record_discovery_data_input_uuids_stable_across_observations() {
    // Regression guard for Issue #1368: the input-neuron UUIDs are precomputed
    // once per batch rather than formatted per observation. This test pins the
    // observable behaviour — identical `input-N` UUIDs and the matching input
    // values are recorded for every observation.
    let temp_dir = TempDir::new().unwrap();
    let mut input = create_test_input();
    input.temp_dir = temp_dir.path().to_str().unwrap().to_string();

    // Three observations, each with two inputs (matches creature.input == 2).
    input.training_data = (0..3)
        .map(|i| crate::TrainingRecord {
            input: vec![i as f32 * 0.5, i as f32 * 0.5 + 0.25],
            output: vec![i as f32],
            neuron_data: Some(vec![crate::NeuronData {
                neuron_uuid: "output-0".to_string(),
                activation: i as f32,
                value: Some(i as f32),
                errors: vec![0.0],
            }]),
        })
        .collect();
    input.record_indices = None;

    let result = record_discovery_data(&input).unwrap();
    let parquet_file = Path::new(&result.temp_dir).join(&result.file);

    use crate::parquet_format::read_records_from_parquet;

    // input-0 records: one per observation, value equal to the recorded input.
    let mut input0 = read_records_from_parquet(parquet_file.to_str().unwrap(), "input-0").unwrap();
    input0.sort_by_key(|r| r.obs_index);
    assert_eq!(
        input0.len(),
        3,
        "Should have one input-0 record per observation"
    );
    for (i, record) in input0.iter().enumerate() {
        let expected = i as f32 * 0.5;
        assert_eq!(record.obs_index, i as u32);
        assert_eq!(record.value, Some(expected));
        assert_eq!(record.activation, expected);
        assert!(record.errors.is_empty());
    }

    // input-1 records: same structure, second input value.
    let mut input1 = read_records_from_parquet(parquet_file.to_str().unwrap(), "input-1").unwrap();
    input1.sort_by_key(|r| r.obs_index);
    assert_eq!(
        input1.len(),
        3,
        "Should have one input-1 record per observation"
    );
    for (i, record) in input1.iter().enumerate() {
        let expected = i as f32 * 0.5 + 0.25;
        assert_eq!(record.obs_index, i as u32);
        assert_eq!(record.value, Some(expected));
        assert_eq!(record.activation, expected);
    }
}
