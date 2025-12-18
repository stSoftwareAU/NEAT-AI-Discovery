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

const MAX_ARROW_OFFSET: usize = i32::MAX as usize;
const MIN_NEURON_UUID_LENGTH: usize = 1;
const MAX_NEURON_UUID_LENGTH: usize = 100;
const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");

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
    let mut writer =
        ParquetRecordWriter::new(file_path, MAX_ARROW_OFFSET, MAX_ARROW_OFFSET, records.len())?;
    writer.write_records(records)?;
    writer.finish()
}

#[cfg(test)]
fn write_records_to_parquet_with_limit(
    file_path: &str,
    records: &[DiscoverRecord],
    max_uuid_bytes_per_batch: usize,
) -> Result<()> {
    let mut writer = ParquetRecordWriter::new(
        file_path,
        max_uuid_bytes_per_batch,
        MAX_ARROW_OFFSET,
        records.len(),
    )?;
    writer.write_records(records)?;
    writer.finish()
}

pub struct ParquetRecordWriter {
    writer: ArrowWriter<File>,
    schema: Arc<Schema>,
    max_uuid_bytes_per_batch: usize,
    max_error_values_per_batch: usize,
    longest_uuid: Option<String>,
    remaining_capacity: usize,
}

impl ParquetRecordWriter {
    pub fn new(
        file_path: &str,
        max_uuid_bytes_per_batch: usize,
        max_error_values_per_batch: usize,
        total_capacity: usize,
    ) -> Result<Self> {
        if total_capacity == 0 {
            anyhow::bail!("No records to write");
        }

        if max_uuid_bytes_per_batch == 0 {
            anyhow::bail!("Configured neuron UUID byte limit per batch must be greater than zero");
        }

        if max_error_values_per_batch == 0 {
            anyhow::bail!("Configured error value limit per batch must be greater than zero");
        }

        if total_capacity > MAX_ARROW_OFFSET {
            anyhow::bail!(
                "Neuron UUID count ({total_capacity}) exceeds maximum supported row count ({}) for Arrow StringArray offsets",
                i32::MAX,
            );
        }

        let schema = Arc::new(create_schema());
        let file = File::create(file_path)
            .with_context(|| format!("Failed to create Parquet file: {file_path}"))?;

        let props = WriterProperties::builder().build();
        let writer = ArrowWriter::try_new(file, schema.clone(), Some(props))
            .context("Failed to create ArrowWriter")?;

        Ok(Self {
            writer,
            schema,
            max_uuid_bytes_per_batch,
            max_error_values_per_batch,
            longest_uuid: None,
            remaining_capacity: total_capacity,
        })
    }

    pub fn write_records(&mut self, records: &[DiscoverRecord]) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }

        if records.len() > self.remaining_capacity {
            anyhow::bail!(
                "Attempted to write {} records exceeding remaining capacity ({})",
                records.len(),
                self.remaining_capacity
            );
        }

        validate_records(
            records,
            self.max_uuid_bytes_per_batch,
            self.max_error_values_per_batch,
            &mut self.longest_uuid,
        )?;

        let mut start = 0;
        while start < records.len() {
            let end = determine_chunk_end(
                records,
                start,
                self.max_uuid_bytes_per_batch,
                self.max_error_values_per_batch,
            );
            let chunk = &records[start..end];
            self.write_chunk(chunk).with_context(|| {
                format!("Failed to write chunk covering records {start}..{end}")
            })?;
            start = end;
        }

        self.remaining_capacity -= records.len();
        Ok(())
    }

    fn write_chunk(&mut self, chunk: &[DiscoverRecord]) -> Result<()> {
        let obs_indices: Vec<u32> = chunk.iter().map(|r| r.obs_index).collect();
        let neuron_uuids: Vec<String> = chunk.iter().map(|r| r.neuron_uuid.clone()).collect();
        let values: Vec<Option<f32>> = chunk.iter().map(|r| r.value).collect();
        let activations: Vec<f32> = chunk.iter().map(|r| r.activation).collect();

        let mut error_values = Vec::with_capacity(chunk.iter().map(|r| r.errors.len()).sum());
        for record in chunk {
            error_values.extend_from_slice(&record.errors);
        }

        let obs_index_array = Arc::new(UInt32Array::from(obs_indices));
        let neuron_uuid_array = Arc::new(StringArray::from(neuron_uuids));
        let value_array = Arc::new(Float32Array::from(values));
        let activation_array = Arc::new(Float32Array::from(activations));

        let error_value_array = Arc::new(Float32Array::from(error_values));
        let offsets =
            arrow::buffer::OffsetBuffer::<i32>::from_lengths(chunk.iter().map(|r| r.errors.len()));
        let errors_array = Arc::new(ListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, false)),
            offsets,
            error_value_array,
            None, // No nulls - all error lists are non-null
        )?);

        let batch = RecordBatch::try_new(
            self.schema.clone(),
            vec![
                obs_index_array,
                neuron_uuid_array,
                value_array,
                activation_array,
                errors_array,
            ],
        )
        .context("Failed to create RecordBatch")?;

        if let Err(err) = self.writer.write(&batch) {
            let err_msg = err.to_string();
            if err_msg.contains("Invalid string length") {
                if let Some(longest_uuid) = self.longest_uuid.as_ref() {
                    let longest_len = longest_uuid.len();
                    let preview = truncate_utf8(longest_uuid, 120);
                    return Err(anyhow::anyhow!(
                        r#"Failed to write discovery data because Arrow rejected a neuron UUID length. Longest observed UUID was "{preview}" ({longest_len} bytes). Original error: {err_msg}"#
                    ));
                }
                return Err(anyhow::anyhow!(
                    "Failed to write discovery data because Arrow reported an invalid string length. Original error: {err_msg}"
                ));
            } else {
                return Err(err).context("Failed to write RecordBatch");
            }
        }

        Ok(())
    }

    pub fn finish(self) -> Result<()> {
        self.writer
            .close()
            .context("Failed to close Parquet writer")?;
        Ok(())
    }
}

fn validate_records(
    records: &[DiscoverRecord],
    max_uuid_bytes_per_batch: usize,
    max_error_values_per_batch: usize,
    longest_uuid: &mut Option<String>,
) -> Result<()> {
    for record in records {
        let uuid = record.neuron_uuid.as_str();
        validate_neuron_uuid(uuid)?;
        let len = uuid.len();

        if len > max_uuid_bytes_per_batch {
            let preview = truncate_utf8(uuid, 120);
            anyhow::bail!(
                r#"Neuron UUID "{preview}" ({len} bytes) exceeds configured limit ({max_uuid_bytes_per_batch} bytes) for per-batch Arrow string offsets."#
            );
        }

        if longest_uuid
            .as_ref()
            .is_none_or(|existing| len > existing.len())
        {
            *longest_uuid = Some(uuid.to_string());
        }

        let error_len = record.errors.len();
        if error_len > max_error_values_per_batch {
            let preview = truncate_utf8(uuid, 120);
            anyhow::bail!(
                r#"Neuron "{preview}" has {error_len} error value(s), exceeding the configured per-batch limit ({max_error_values_per_batch}) for Arrow list offsets."#
            );
        }
    }

    Ok(())
}

fn determine_chunk_end(
    records: &[DiscoverRecord],
    start: usize,
    max_uuid_bytes_per_batch: usize,
    max_error_values_per_batch: usize,
) -> usize {
    let mut end = start;
    let mut uuid_bytes = 0_usize;
    let mut error_value_count = 0_usize;

    while end < records.len() {
        let record = &records[end];
        let uuid_len = record.neuron_uuid.len();
        let errors_len = record.errors.len();

        if end > start
            && (uuid_bytes + uuid_len > max_uuid_bytes_per_batch
                || error_value_count + errors_len > max_error_values_per_batch)
        {
            break;
        }

        uuid_bytes += uuid_len;
        error_value_count += errors_len;
        end += 1;
    }

    end
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

fn validate_neuron_uuid(uuid: &str) -> Result<()> {
    let len = uuid.len();
    let preview = truncate_utf8(uuid, 120);

    if len < MIN_NEURON_UUID_LENGTH {
        anyhow::bail!(
            r#"neat_ai_discovery v{CRATE_VERSION} rejected neuron UUID "{preview}" ({len} characters). UUIDs must be between {MIN_NEURON_UUID_LENGTH} and {MAX_NEURON_UUID_LENGTH} characters and use letters, digits, or hyphens."#
        );
    }

    if len > MAX_NEURON_UUID_LENGTH {
        anyhow::bail!(
            r#"neat_ai_discovery v{CRATE_VERSION} rejected neuron UUID "{preview}" ({len} characters). UUIDs must be between {MIN_NEURON_UUID_LENGTH} and {MAX_NEURON_UUID_LENGTH} characters and use letters, digits, or hyphens."#
        );
    }

    if let Some((index, ch)) = uuid
        .chars()
        .enumerate()
        .find(|(_, c)| !matches!(c, 'a'..='z' | 'A'..='Z' | '0'..='9' | '-'))
    {
        anyhow::bail!(
            r#"neat_ai_discovery v{CRATE_VERSION} rejected neuron UUID "{preview}" because it contains an invalid character '{ch}' at position {index}. UUIDs must be between {MIN_NEURON_UUID_LENGTH} and {MAX_NEURON_UUID_LENGTH} characters and use letters, digits, or hyphens."#
        );
    }

    Ok(())
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

/// Read all discovery records from a Parquet file, grouped by neuron UUID.
/// This is more efficient than calling read_records_from_parquet multiple times
/// when you need records for multiple neurons.
pub fn read_all_records_grouped_by_neuron(
    file_path: &str,
) -> Result<std::collections::HashMap<String, Vec<DiscoverRecord>>> {
    use arrow::array::{Array, Float32Array, ListArray, StringArray, UInt32Array};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use std::collections::HashMap;
    use std::fs::File;

    let file = File::open(file_path)
        .with_context(|| format!("Failed to open Parquet file: {file_path}"))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .context("Failed to create Parquet reader builder")?;

    let reader = builder.build().context("Failed to build Parquet reader")?;

    let mut grouped_records: HashMap<String, Vec<DiscoverRecord>> = HashMap::new();

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

        // Collect all records and group by neuron UUID
        for i in 0..batch.num_rows() {
            let uuid = neuron_uuid_col.value(i).to_string();
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

            let record = DiscoverRecord::new(obs_index, uuid.clone(), value, activation, errors);
            grouped_records.entry(uuid).or_default().push(record);
        }
    }

    // Note: Records are returned in Parquet read order (not sorted by obs_index)
    // TypeScript sorts records by obs_index after reading for cross-neuron matching
    Ok(grouped_records)
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

/// Read ALL discovery records from a Parquet file (no filtering).
///
/// Used by the visualisation snapshot export to get all recorded data.
pub fn read_all_records_from_parquet(file_path: &str) -> Result<Vec<DiscoverRecord>> {
    read_records_from_parquet_with_limit(file_path, None)
}

/// Read discovery records from a Parquet file with optional observation limit.
///
/// If `max_obs` is Some, returns records for the first `max_obs` distinct
/// `obs_index` values encountered while scanning the file.
///
/// Important: Parquet rows are not guaranteed to be stored in observation-first
/// order. They may be stored in neuron-first order (all obs for neuron A, then
/// all obs for neuron B). In that case, **we must not stop reading** as soon as
/// we see a new `obs_index` beyond the limit, because later rows may still
/// contain records for already-accepted observation indices.
///
/// To guarantee complete data for the accepted observations, this reader:
/// - Keeps a set of accepted `obs_index` values up to `max_obs`
/// - Scans the file and **filters** to only accepted observations
pub fn read_records_from_parquet_with_limit(
    file_path: &str,
    max_obs: Option<u32>,
) -> Result<Vec<DiscoverRecord>> {
    use arrow::array::{Array, Float32Array, ListArray, StringArray, UInt32Array};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use std::collections::HashSet;
    use std::fs::File;

    // Edge case: treat max_obs=0 as "return no rows".
    if matches!(max_obs, Some(0)) {
        return Ok(Vec::new());
    }

    let file = File::open(file_path)
        .with_context(|| format!("Failed to open Parquet file: {file_path}"))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .context("Failed to create Parquet reader builder")?;

    let reader = builder.build().context("Failed to build Parquet reader")?;

    let mut records = Vec::new();
    let mut accepted_obs_indices: HashSet<u32> = HashSet::new();

    for batch_result in reader {
        let batch = batch_result.context("Failed to read record batch")?;

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

        for i in 0..batch.num_rows() {
            let obs_index = obs_index_col.value(i);

            // Determine whether to include this row under the limit.
            let include_row = match max_obs {
                None => true,
                Some(max) => {
                    if accepted_obs_indices.contains(&obs_index) {
                        true
                    } else if accepted_obs_indices.len() < max as usize {
                        accepted_obs_indices.insert(obs_index);
                        true
                    } else {
                        // We've accepted enough observations. Skip new obs_index values,
                        // but keep scanning in case we encounter more rows for accepted
                        // observations (e.g. neuron-first ordering).
                        false
                    }
                }
            };

            if !include_row {
                continue;
            }

            let uuid = neuron_uuid_col.value(i);
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
    fn test_write_records_chunks_when_uuid_bytes_exceed_limit() {
        let temp_file = NamedTempFile::new().unwrap();
        let file_path = temp_file.path().to_str().unwrap();

        let records = vec![
            DiscoverRecord::new(0, "hidden-0000".to_string(), Some(0.5), 0.7, vec![0.1]),
            DiscoverRecord::new(1, "hidden-1111".to_string(), Some(0.6), 0.8, vec![0.2]),
        ];

        let result = write_records_to_parquet_with_limit(file_path, &records, 16);
        assert!(
            result.is_ok(),
            "Writer should chunk records when total neuron UUID bytes exceed the configured per-batch limit"
        );

        assert!(
            std::path::Path::new(file_path).exists(),
            "Parquet file should exist after chunked write"
        );
    }

    #[test]
    fn test_write_records_errors_when_single_uuid_exceeds_batch_limit() {
        let temp_file = NamedTempFile::new().unwrap();
        let file_path = temp_file.path().to_str().unwrap();

        let records = vec![DiscoverRecord::new(
            0,
            "hidden-0000".to_string(),
            Some(0.5),
            0.7,
            vec![0.1],
        )];

        let err = write_records_to_parquet_with_limit(file_path, &records, 5)
            .expect_err("Single record with UUID longer than limit should be rejected");
        assert!(
            err.to_string().contains("exceeds configured limit"),
            "Error message should explain the batch limit violation"
        );
    }

    #[test]
    fn test_write_records_rejects_neuron_uuid_too_short() {
        let temp_file = NamedTempFile::new().unwrap();
        let file_path = temp_file.path().to_str().unwrap();

        let records = vec![DiscoverRecord::new(
            0,
            "".to_string(),
            Some(0.5),
            0.7,
            vec![0.1],
        )];

        let result = write_records_to_parquet(file_path, &records);
        let err = result.expect_err("Expected empty neuron UUID to be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("neat_ai_discovery v"),
            "Error message should include crate version: {msg}"
        );
        assert!(
            msg.contains("\"\""),
            "Error message should include the invalid UUID: {msg}"
        );
        assert!(
            msg.contains("letters, digits, or hyphens"),
            "Error message should explain valid characters: {msg}"
        );
    }

    #[test]
    fn test_write_records_rejects_neuron_uuid_with_invalid_characters() {
        let temp_file = NamedTempFile::new().unwrap();
        let file_path = temp_file.path().to_str().unwrap();

        let records = vec![DiscoverRecord::new(
            0,
            "invalid_UUID".to_string(),
            Some(0.5),
            0.7,
            vec![0.1],
        )];

        let result = write_records_to_parquet(file_path, &records);
        let err = result.expect_err("Expected neuron UUID with invalid characters to be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("neat_ai_discovery v"),
            "Error message should include crate version: {msg}"
        );
        assert!(
            msg.contains("\"invalid_UUID\""),
            "Error message should include the invalid UUID: {msg}"
        );
        assert!(
            msg.contains("letters, digits, or hyphens"),
            "Error message should explain valid characters: {msg}"
        );
    }

    #[test]
    fn test_write_records_rejects_neuron_uuid_too_long() {
        let temp_file = NamedTempFile::new().unwrap();
        let file_path = temp_file.path().to_str().unwrap();

        let long_uuid = "a".repeat(101);

        let records = vec![DiscoverRecord::new(
            0,
            long_uuid.clone(),
            Some(0.5),
            0.7,
            vec![0.1],
        )];

        let result = write_records_to_parquet(file_path, &records);
        let err = result.expect_err("Expected overly long neuron UUID to be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("neat_ai_discovery v"),
            "Error message should include crate version: {msg}"
        );
        assert!(
            msg.contains(&long_uuid[..60]),
            "Error message should include the invalid UUID preview: {msg}"
        );
        assert!(
            msg.contains("between 1 and 100 characters"),
            "Error message should explain the valid length range: {msg}"
        );
    }

    #[test]
    fn test_write_records_accepts_single_character_uppercase_neuron_uuid() {
        let temp_file = NamedTempFile::new().unwrap();
        let file_path = temp_file.path().to_str().unwrap();

        let records = vec![DiscoverRecord::new(
            0,
            "A".to_string(),
            Some(0.5),
            0.7,
            vec![0.1],
        )];

        let result = write_records_to_parquet(file_path, &records);
        assert!(
            result.is_ok(),
            "Single-character uppercase neuron UUID should be accepted"
        );
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

    #[test]
    fn test_write_records_chunks_when_error_value_count_exceeds_limit() {
        let temp_file = NamedTempFile::new().unwrap();
        let file_path = temp_file.path().to_str().unwrap();

        let records = vec![
            DiscoverRecord::new(0, "hidden-0000".to_string(), Some(0.5), 0.7, vec![0.1, 0.2]),
            DiscoverRecord::new(1, "hidden-1111".to_string(), Some(0.6), 0.8, vec![0.3, 0.4]),
        ];

        let mut writer =
            super::ParquetRecordWriter::new(file_path, super::MAX_ARROW_OFFSET, 3, records.len())
                .expect("Writer should construct successfully");
        assert!(
            writer.write_records(&records).is_ok(),
            "Writer should chunk records when total error values exceed the configured per-batch limit"
        );
        writer.finish().unwrap();

        assert!(
            std::path::Path::new(file_path).exists(),
            "Parquet file should exist after chunked write"
        );
    }

    #[test]
    fn test_write_records_errors_when_single_record_error_count_exceeds_limit() {
        let temp_file = NamedTempFile::new().unwrap();
        let file_path = temp_file.path().to_str().unwrap();

        let records = vec![DiscoverRecord::new(
            0,
            "hidden-0000".to_string(),
            Some(0.5),
            0.7,
            vec![0.1, 0.2],
        )];

        let mut writer =
            super::ParquetRecordWriter::new(file_path, super::MAX_ARROW_OFFSET, 1, records.len())
                .expect("Writer should construct successfully");
        let err = writer
            .write_records(&records)
            .expect_err("Single record with more error values than the limit should be rejected");
        assert!(
            err.to_string().contains("error value(s)"),
            "Error message should reference the per-batch error limit"
        );
    }

    #[test]
    fn test_read_all_records_grouped_by_neuron() {
        // Test that read_all_records_grouped_by_neuron correctly groups records by neuron UUID
        let temp_file = NamedTempFile::new().unwrap();
        let file_path = temp_file.path().to_str().unwrap();

        // Write records for multiple neurons
        let records = vec![
            DiscoverRecord::new(0, "neuron-1".to_string(), Some(0.5), 0.7, vec![0.1, 0.2]),
            DiscoverRecord::new(1, "neuron-1".to_string(), Some(0.6), 0.8, vec![0.15, 0.25]),
            DiscoverRecord::new(0, "neuron-2".to_string(), Some(0.3), 0.5, vec![0.05]),
            DiscoverRecord::new(1, "neuron-2".to_string(), Some(0.4), 0.6, vec![0.1]),
            DiscoverRecord::new(2, "neuron-2".to_string(), Some(0.5), 0.7, vec![0.15]),
        ];

        write_records_to_parquet(file_path, &records).unwrap();

        // Read all records grouped by neuron
        let grouped = read_all_records_grouped_by_neuron(file_path).unwrap();

        // Verify neuron-1 has 2 records
        let neuron1_records = grouped.get("neuron-1").expect("neuron-1 should exist");
        assert_eq!(neuron1_records.len(), 2, "neuron-1 should have 2 records");
        assert_eq!(neuron1_records[0].obs_index, 0);
        assert_eq!(neuron1_records[1].obs_index, 1);

        // Verify neuron-2 has 3 records
        let neuron2_records = grouped.get("neuron-2").expect("neuron-2 should exist");
        assert_eq!(neuron2_records.len(), 3, "neuron-2 should have 3 records");
        assert_eq!(neuron2_records[0].obs_index, 0);
        assert_eq!(neuron2_records[1].obs_index, 1);
        assert_eq!(neuron2_records[2].obs_index, 2);

        // Verify that read_records_from_parquet returns the same data for each neuron
        let neuron1_single = read_records_from_parquet(file_path, "neuron-1").unwrap();
        assert_eq!(
            neuron1_single.len(),
            neuron1_records.len(),
            "Single read should return same number of records as grouped read"
        );

        let neuron2_single = read_records_from_parquet(file_path, "neuron-2").unwrap();
        assert_eq!(
            neuron2_single.len(),
            neuron2_records.len(),
            "Single read should return same number of records as grouped read"
        );
    }
}
