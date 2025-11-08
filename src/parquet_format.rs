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
    let num_records = records.len();
    let obs_indices: Vec<u32> = records.iter().map(|r| r.obs_index).collect();
    let neuron_uuids: Vec<String> = records.iter().map(|r| r.neuron_uuid.clone()).collect();
    let values: Vec<Option<f32>> = records.iter().map(|r| r.value).collect();
    let activations: Vec<f32> = records.iter().map(|r| r.activation).collect();

    // Build errors as ListArray
    let mut error_offsets = Vec::with_capacity(num_records + 1);
    error_offsets.push(0i32);
    let mut error_values = Vec::new();

    for record in records {
        error_values.extend_from_slice(&record.errors);
        error_offsets.push(error_values.len() as i32);
    }

    let obs_index_array = Arc::new(UInt32Array::from(obs_indices));
    let neuron_uuid_array = Arc::new(StringArray::from(neuron_uuids));
    let value_array = Arc::new(Float32Array::from(values));
    let activation_array = Arc::new(Float32Array::from(activations));

    // Build ListArray for errors
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

    writer
        .write(&batch)
        .context("Failed to write RecordBatch")?;
    writer.close().context("Failed to close Parquet writer")?;

    Ok(())
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
}
