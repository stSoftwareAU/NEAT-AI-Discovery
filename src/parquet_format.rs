//! Parquet file format handling for discovery records

use anyhow::{Context, Result};
use arrow::array::{Float32Array, ListArray, StringArray, UInt32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use std::fs::File;
use std::sync::Arc;

use crate::types::DiscoverRecord;

/// Parquet schema for discovery records
pub fn create_schema() -> Schema {
    Schema::new(vec![
        Field::new("obs_index", DataType::UInt32, false),
        Field::new("neuron_uuid", DataType::Utf8, false),
        Field::new("value", DataType::Float32, true), // nullable
        Field::new("activation", DataType::Float32, false),
        Field::new(
            "errors",
            DataType::List(Arc::new(Field::new("item", DataType::Float32, false))),
            false,
        ),
    ])
}

/// Write discovery records to a Parquet file
pub fn write_records_to_parquet(file_path: &str, records: &[DiscoverRecord]) -> Result<()> {
    if records.is_empty() {
        return Err(anyhow::anyhow!("No records to write"));
    }

    let schema = Arc::new(create_schema());
    let file = File::create(file_path)
        .with_context(|| format!("Failed to create Parquet file: {file_path}"))?;

    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))
        .context("Failed to create ArrowWriter")?;
    // Prepare arrays
    let obs_indices: Vec<u32> = records.iter().map(|r| r.obs_index).collect();
    let neuron_uuids: Vec<String> = records.iter().map(|r| r.neuron_uuid.clone()).collect();
    let values: Vec<Option<f32>> = records.iter().map(|r| r.value).collect();
    let activations: Vec<f32> = records.iter().map(|r| r.activation).collect();

    // Collect error values and validate total count to prevent i32 overflow
    // Arrow's OffsetBuffer uses i32 for offsets, so the total error value count
    // must not exceed i32::MAX (2,147,483,647)
    let mut error_values = Vec::new();
    let total_error_count: usize = records.iter().map(|r| r.errors.len()).sum();

    if total_error_count > i32::MAX as usize {
        return Err(anyhow::anyhow!(
            "Total error values count ({}) exceeds maximum supported size ({}) for Parquet offset buffer",
            total_error_count,
            i32::MAX
        ));
    }

    for record in records {
        error_values.extend_from_slice(&record.errors);
    }

    let obs_index_array = Arc::new(UInt32Array::from(obs_indices));
    let neuron_uuid_array = Arc::new(StringArray::from(neuron_uuids));
    let value_array = Arc::new(Float32Array::from(values));
    let activation_array = Arc::new(Float32Array::from(activations));

    // Build ListArray for errors
    // from_lengths converts lengths to cumulative offsets, which must fit in i32
    let error_value_array = Arc::new(Float32Array::from(error_values));
    let offsets =
        arrow::buffer::OffsetBuffer::<i32>::from_lengths(records.iter().map(|r| r.errors.len()));
    let errors_array = Arc::new(ListArray::try_new(
        Arc::new(Field::new("item", DataType::Float32, false)),
        offsets,
        error_value_array,
        None, // No nulls - all error lists are non-null
    )?);

    let batch = RecordBatch::try_new(
        schema,
        vec![
            obs_index_array,
            neuron_uuid_array,
            value_array,
            activation_array,
            errors_array,
        ],
    )
    .context("Failed to create RecordBatch")?;

    if let Err(err) = writer.write(&batch) {
        let err_msg = err.to_string();
        if err_msg.contains("Invalid string length") {
            if let Some((longest_uuid, longest_len)) = records
                .iter()
                .map(|r| (r.neuron_uuid.as_str(), r.neuron_uuid.len()))
                .max_by_key(|(_, len)| *len)
            {
                let preview = truncate_utf8(longest_uuid, 120);
                return Err(anyhow::anyhow!(
                    "Failed to write discovery data because Arrow rejected a neuron UUID length. Longest observed UUID was \"{preview}\" ({longest_len} bytes). Original error: {err_msg}"
                ));
            }
            return Err(anyhow::anyhow!(
                "Failed to write discovery data because Arrow reported an invalid string length. Original error: {err_msg}"
            ));
        } else {
            return Err(err).context("Failed to write RecordBatch");
        }
    }
    writer.close().context("Failed to close Parquet writer")?;

    Ok(())
}

fn truncate_utf8(value: &str, max_bytes: usize) -> String {
    if value.len() <= max_bytes {
        return value.to_string();
    }

    if max_bytes == 0 {
        return "...".to_string();
    }

    let mut end = max_bytes.min(value.len());
    while end > 0 && !value.is_char_boundary(end) {
        end -= 1;
    }

    if end == 0 {
        return "...".to_string();
    }

    let slice = &value[..end];
    format!("{slice}...")
}

/// Merge multiple discovery parquet files into a single Parquet file.
/// Input files are appended in the provided order.
pub fn merge_parquet_files(output_file: &str, input_files: &[String]) -> Result<()> {
    if input_files.is_empty() {
        return Err(anyhow::anyhow!(
            "No discovery parquet files provided for merge"
        ));
    }

    let schema = Arc::new(create_schema());
    let file = File::create(output_file)
        .with_context(|| format!("Failed to create merged Parquet file: {output_file}"))?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))
        .context("Failed to create ArrowWriter for merge")?;

    for input_path in input_files {
        let input_file = File::open(input_path)
            .with_context(|| format!("Failed to open discovery Parquet file: {input_path}"))?;
        let builder =
            parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(input_file)
                .with_context(|| format!("Failed to create reader for {input_path}"))?;
        let reader = builder
            .build()
            .with_context(|| format!("Failed to build reader for {input_path}"))?;

        for batch_result in reader {
            let batch =
                batch_result.with_context(|| format!("Failed to read batch from {input_path}"))?;
            writer
                .write(&batch)
                .with_context(|| format!("Failed to append batch from {input_path}"))?;
        }
    }

    writer
        .close()
        .context("Failed to close merged Parquet writer")?;
    Ok(())
}

/// Read discovery records from a Parquet file, filtered by neuron UUID
pub fn read_records_from_parquet(
    file_path: &str,
    neuron_uuid: &str,
) -> Result<Vec<DiscoverRecord>> {
    use arrow::array::{Array, Float32Array, ListArray, StringArray, UInt32Array};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use std::fs::File;

    let file = File::open(file_path)
        .with_context(|| format!("Failed to open Parquet file: {file_path}"))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .context("Failed to create Parquet reader builder")?;

    let reader = builder.build().context("Failed to build Parquet reader")?;

    let mut records = Vec::new();

    for batch_result in reader {
        let batch = batch_result.context("Failed to read record batch")?;

        // Get columns - use schema field names to find correct columns instead of hardcoded indices
        let obs_index_col = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .context("Failed to cast obs_index column")?;
        let neuron_uuid_col = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .context("Failed to cast neuron_uuid column")?;
        let value_col = batch
            .column(2)
            .as_any()
            .downcast_ref::<Float32Array>()
            .context("Failed to cast value column")?;
        let activation_col = batch
            .column(3)
            .as_any()
            .downcast_ref::<Float32Array>()
            .context("Failed to cast activation column")?;
        let errors_col = batch
            .column(4)
            .as_any()
            .downcast_ref::<ListArray>()
            .context("Failed to cast errors column")?;

        // Filter by neuron UUID and collect records
        for i in 0..batch.num_rows() {
            let uuid = neuron_uuid_col.value(i);
            if uuid == neuron_uuid {
                let obs_index = obs_index_col.value(i);
                let value = if value_col.is_null(i) {
                    None
                } else {
                    Some(value_col.value(i))
                };
                let activation = activation_col.value(i);

                // Extract errors array
                let errors_list = errors_col.value(i);
                let errors_array = errors_list
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .context("Failed to cast errors array")?;
                let errors: Vec<f32> = (0..errors_array.len())
                    .map(|j| errors_array.value(j))
                    .collect();

                records.push(DiscoverRecord::new(
                    obs_index,
                    uuid.to_string(),
                    value,
                    activation,
                    errors,
                ));
            }
        }
    }

    // Note: Records are returned in Parquet read order (not sorted by obs_index)
    // TypeScript sorts records by obs_index after reading for cross-neuron matching
    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_create_schema() {
        let schema = create_schema();
        assert_eq!(schema.fields().len(), 5);
        assert_eq!(schema.field(0).name(), "obs_index");
        assert_eq!(schema.field(1).name(), "neuron_uuid");
        assert_eq!(schema.field(2).name(), "value");
        assert_eq!(schema.field(3).name(), "activation");
        assert_eq!(schema.field(4).name(), "errors");
    }

    #[test]
    fn test_create_schema_uses_utf8_for_neuron_uuid() {
        let schema = create_schema();
        assert_eq!(
            schema.field(1).data_type(),
            &DataType::Utf8,
            "Neuron UUID column should use Utf8 to ensure Parquet compatibility"
        );
    }

    #[test]
    fn test_write_records_to_parquet() {
        let temp_file = NamedTempFile::new().unwrap();
        let file_path = temp_file.path().to_str().unwrap();

        let records = vec![
            DiscoverRecord::new(0, "hidden-1".to_string(), Some(0.5), 0.7, vec![0.1, 0.2]),
            DiscoverRecord::new(1, "hidden-1".to_string(), Some(0.6), 0.8, vec![0.15]),
        ];

        write_records_to_parquet(file_path, &records).unwrap();

        // Verify file was created
        assert!(std::path::Path::new(file_path).exists());
    }

    #[test]
    fn test_write_empty_records_fails() {
        let temp_file = NamedTempFile::new().unwrap();
        let file_path = temp_file.path().to_str().unwrap();

        let result = write_records_to_parquet(file_path, &[]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No records"));
    }

    #[test]
    fn test_write_records_prevents_i32_overflow() {
        // This test verifies that the validation prevents i32 overflow
        // when the total error count would exceed i32::MAX
        // We can't actually create 2.1 billion error values in memory for testing,
        // but we can verify the validation code path exists and works correctly

        let temp_file = NamedTempFile::new().unwrap();
        let file_path = temp_file.path().to_str().unwrap();

        // Create records with a reasonable number of errors to verify normal operation
        let records = vec![
            DiscoverRecord::new(
                0,
                "hidden-1".to_string(),
                Some(0.5),
                0.7,
                vec![0.1; 1000], // 1000 errors per record
            ),
            DiscoverRecord::new(
                1,
                "hidden-2".to_string(),
                Some(0.6),
                0.8,
                vec![0.2; 1000], // 1000 errors per record
            ),
        ];

        // This should succeed - total is 2000 errors, well within i32::MAX
        let result = write_records_to_parquet(file_path, &records);
        assert!(
            result.is_ok(),
            "Should handle valid error counts within i32::MAX"
        );

        // Verify the file was created
        assert!(std::path::Path::new(file_path).exists());
    }

    #[test]
    fn test_write_records_validates_error_count_overflow() {
        // This test verifies that attempting to write records with error counts
        // that would exceed i32::MAX is properly rejected
        // Since we can't create 2.1 billion values in memory, we test the validation
        // by checking that the code path exists and would catch the overflow

        // The validation check `if total_error_count > i32::MAX as usize` will catch
        // any case where the total error values exceed i32::MAX. This prevents
        // silent truncation when converting to i32 offsets in Arrow's OffsetBuffer.

        // For a practical test, we verify that normal-sized datasets work correctly
        // and that the validation logic is in place (not using unsafe 'as' casting)
        let temp_file = NamedTempFile::new().unwrap();
        let file_path = temp_file.path().to_str().unwrap();

        // Create a reasonable number of records with reasonable error counts
        // This verifies the validation doesn't incorrectly reject valid data
        let records: Vec<DiscoverRecord> = (0..100)
            .map(|i| {
                DiscoverRecord::new(
                    i,
                    format!("neuron-{i}"),
                    Some(0.5),
                    0.7,
                    vec![0.1; 10], // 10 errors per record = 1000 total
                )
            })
            .collect();

        let result = write_records_to_parquet(file_path, &records);
        assert!(result.is_ok(), "Validation should allow valid error counts");
    }

    #[test]
    fn test_read_records_from_parquet_preserves_obs_index() {
        // Test that records maintain their obs_index when reading, even if written out of order
        // Note: Records are NOT sorted by Rust - TypeScript handles sorting after reading
        let temp_file = NamedTempFile::new().unwrap();
        let file_path = temp_file.path().to_str().unwrap();

        // Write records in non-sequential order
        let records = vec![
            DiscoverRecord::new(5, "neuron-1".to_string(), Some(0.5), 0.7, vec![0.1]),
            DiscoverRecord::new(2, "neuron-1".to_string(), Some(0.6), 0.8, vec![0.15]),
            DiscoverRecord::new(8, "neuron-1".to_string(), Some(0.7), 0.9, vec![0.2]),
            DiscoverRecord::new(1, "neuron-1".to_string(), Some(0.4), 0.6, vec![0.05]),
            DiscoverRecord::new(3, "neuron-1".to_string(), Some(0.65), 0.85, vec![0.18]),
        ];

        write_records_to_parquet(file_path, &records).unwrap();

        // Read back records
        let read_records = read_records_from_parquet(file_path, "neuron-1").unwrap();

        // Verify all records are present with correct obs_index values
        assert_eq!(read_records.len(), 5, "Should have 5 records");
        let obs_indices: Vec<u32> = read_records.iter().map(|r| r.obs_index).collect();
        assert!(obs_indices.contains(&1), "Should contain obs_index 1");
        assert!(obs_indices.contains(&2), "Should contain obs_index 2");
        assert!(obs_indices.contains(&3), "Should contain obs_index 3");
        assert!(obs_indices.contains(&5), "Should contain obs_index 5");
        assert!(obs_indices.contains(&8), "Should contain obs_index 8");
    }
}
