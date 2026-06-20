//! Parquet writing and serialisation for discovery records

use anyhow::{Context, Result};
use arrow::array::{Float32Array, ListArray, StringArray, UInt32Array};
use arrow::datatypes::{DataType, Field};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use std::fs::File;
use std::sync::Arc;

use crate::parquet_format::schema::{
    MAX_ARROW_OFFSET, create_schema, truncate_utf8, validate_neuron_uuid,
};
use crate::types::DiscoverRecord;

/// Write discovery records to a Parquet file
pub fn write_records_to_parquet(file_path: &str, records: &[DiscoverRecord]) -> Result<()> {
    let mut writer =
        ParquetRecordWriter::new(file_path, MAX_ARROW_OFFSET, MAX_ARROW_OFFSET, records.len())?;
    writer.write_records(records)?;
    writer.finish()
}

#[cfg(test)]
pub(crate) fn write_records_to_parquet_with_limit(
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

/// Streaming writer for discovery [`DiscoverRecord`]s into a Parquet file.
///
/// Buffers records into batches bounded by configured per-batch UUID byte and
/// error-value limits so a single record batch never exceeds Arrow's offset
/// constraints. Drive it by constructing one via [`Self::new`], pushing
/// records with [`Self::write_records`], and finalising the file with
/// [`Self::finish`].
pub struct ParquetRecordWriter {
    writer: ArrowWriter<File>,
    schema: Arc<arrow::datatypes::Schema>,
    max_uuid_bytes_per_batch: usize,
    max_error_values_per_batch: usize,
    longest_uuid: Option<String>,
    remaining_capacity: usize,
}

impl ParquetRecordWriter {
    /// Create a new writer targeting `file_path`.
    ///
    /// Validates the byte/value batch limits and `total_capacity` (must be
    /// non-zero and within Arrow's `i32::MAX` offset budget) before creating
    /// the underlying Parquet file. Returns an error if the file cannot be
    /// created or the limits are invalid.
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

    /// Validates and writes a batch of `records` to the underlying Parquet file.
    ///
    /// An empty slice is a no-op. Records are split into Arrow-offset-safe
    /// batches before being appended.
    ///
    /// # Errors
    ///
    /// Returns an error if `records` exceeds the writer's remaining capacity,
    /// if record validation fails (e.g. an invalid neuron UUID), or if the
    /// underlying Arrow/Parquet write fails.
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

    /// Finalises the file, flushing buffered data and closing the writer.
    ///
    /// Must be called after the final [`Self::write_records`] to produce a valid
    /// Parquet file; consumes the writer.
    ///
    /// # Errors
    ///
    /// Returns an error if closing the underlying Parquet writer fails (e.g. an
    /// I/O error while flushing the file footer).
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
