//! Parquet writing and serialisation for discovery records

use anyhow::{Context, Result};
use arrow::array::{Float32Array, ListArray, StringArray, UInt32Array};
use arrow::datatypes::{DataType, Field};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::parquet_format::reader::{ColumnProfile, validate_parquet_schema};
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

/// Resolve `path` for aliasing comparison (Issue #1900).
///
/// Canonicalises the path when it exists; otherwise canonicalises its parent
/// directory and re-joins the file name so a not-yet-created destination still
/// compares correctly against existing inputs.
fn resolve_for_alias_check(path: &str) -> PathBuf {
    let candidate = Path::new(path);
    if let Ok(canonical) = candidate.canonicalize() {
        return canonical;
    }

    match (candidate.parent(), candidate.file_name()) {
        (Some(parent), Some(name)) => {
            let parent = if parent.as_os_str().is_empty() {
                Path::new(".")
            } else {
                parent
            };
            parent
                .canonicalize()
                .map_or_else(|_| candidate.to_path_buf(), |dir| dir.join(name))
        }
        _ => candidate.to_path_buf(),
    }
}

/// Remove a temporary merge file, warning loudly if the cleanup itself fails.
fn remove_merge_temp_file(tmp_path: &str) {
    if let Err(err) = fs::remove_file(tmp_path)
        && err.kind() != std::io::ErrorKind::NotFound
    {
        tracing::warn!(
            tmp_path,
            error = %err,
            "Failed to remove temporary merged Parquet file"
        );
    }
}

/// Read every input file and write the merged result into `tmp_path`.
///
/// Each input's schema is validated before any batch is appended, so a foreign
/// schema is rejected rather than smuggled into the merged output.
fn merge_into_temp_file(tmp_path: &str, input_files: &[String]) -> Result<()> {
    let schema = Arc::new(create_schema());
    let file = File::create(tmp_path)
        .with_context(|| format!("Failed to create temporary merged Parquet file: {tmp_path}"))?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))
        .context("Failed to create ArrowWriter for merge")?;

    for input_path in input_files {
        let input_file = File::open(input_path)
            .with_context(|| format!("Failed to open discovery Parquet file: {input_path}"))?;
        let builder =
            parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(input_file)
                .with_context(|| format!("Failed to create reader for {input_path}"))?;
        validate_parquet_schema(&builder, input_path, ColumnProfile::Full)?;
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

/// Merge multiple discovery parquet files into a single Parquet file.
/// Input files are appended in the provided order.
///
/// The merge is non-destructive (Issue #1900): every input is read and
/// schema-validated into a sibling `{output_file}.tmp`, which is renamed onto
/// `output_file` only once the merged writer has closed cleanly. Any failure
/// removes the temporary file and leaves an existing destination untouched.
///
/// # Errors
///
/// Returns an error if `input_files` is empty, if any entry of `input_files`
/// resolves to `output_file` (which would otherwise truncate an input), if any
/// input is missing, unreadable, or carries a non-discovery schema, or if the
/// temporary file cannot be written or renamed.
pub fn merge_parquet_files(output_file: &str, input_files: &[String]) -> Result<()> {
    if input_files.is_empty() {
        return Err(anyhow::anyhow!(
            "No discovery parquet files provided for merge"
        ));
    }

    let output_resolved = resolve_for_alias_check(output_file);
    for input_path in input_files {
        if resolve_for_alias_check(input_path) == output_resolved {
            anyhow::bail!(
                "Merge output file '{output_file}' also appears in the input list as \
                 '{input_path}'; refusing to merge a file onto itself"
            );
        }
    }

    let tmp_path = format!("{output_file}.tmp");
    if let Err(err) = merge_into_temp_file(&tmp_path, input_files) {
        remove_merge_temp_file(&tmp_path);
        return Err(err);
    }

    if let Err(err) = fs::rename(&tmp_path, output_file) {
        remove_merge_temp_file(&tmp_path);
        return Err(anyhow::Error::new(err).context(format!(
            "Failed to rename temporary file {tmp_path} to {output_file}"
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int32Array;
    use arrow::datatypes::Schema;
    use tempfile::TempDir;

    fn sample_records(obs_index: u32) -> Vec<DiscoverRecord> {
        vec![
            DiscoverRecord {
                obs_index,
                neuron_uuid: "neuron-a".to_string(),
                value: Some(0.25),
                activation: 0.5,
                errors: vec![0.1, 0.2],
            },
            DiscoverRecord {
                obs_index,
                neuron_uuid: "neuron-b".to_string(),
                value: None,
                activation: -0.5,
                errors: vec![],
            },
        ]
    }

    /// Write a Parquet file whose schema is not the discovery schema.
    fn write_foreign_schema_parquet(path: &str) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("alpha", DataType::Int32, false),
            Field::new("beta", DataType::Int32, false),
            Field::new("gamma", DataType::Int32, false),
            Field::new("delta", DataType::Int32, false),
            Field::new("epsilon", DataType::Int32, false),
        ]));
        let file = File::create(path).expect("create foreign parquet");
        let mut writer = ArrowWriter::try_new(file, schema.clone(), None).expect("arrow writer");
        let column: Arc<dyn arrow::array::Array> = Arc::new(Int32Array::from(vec![1, 2]));
        let batch = RecordBatch::try_new(schema, vec![column; 5]).expect("foreign batch");
        writer.write(&batch).expect("write foreign batch");
        writer.close().expect("close foreign writer");
    }

    fn path_in(dir: &TempDir, name: &str) -> String {
        dir.path().join(name).to_string_lossy().into_owned()
    }

    #[test]
    fn merge_preserves_destination_when_input_missing() {
        let dir = TempDir::new().expect("temp dir");
        let destination = path_in(&dir, "merged.parquet");
        let valid_input = path_in(&dir, "input_a.parquet");

        write_records_to_parquet(&destination, &sample_records(0)).expect("seed destination");
        write_records_to_parquet(&valid_input, &sample_records(1)).expect("write input");
        let original = std::fs::read(&destination).expect("read destination");

        let missing_input = path_in(&dir, "does_not_exist.parquet");
        let err = merge_parquet_files(&destination, &[valid_input, missing_input])
            .expect_err("merge must fail when an input is missing");
        assert!(
            err.to_string().contains("does_not_exist.parquet"),
            "unexpected error: {err}"
        );

        assert_eq!(
            std::fs::read(&destination).expect("re-read destination"),
            original,
            "destination must be byte-identical after a failed merge"
        );
        assert!(
            !Path::new(&format!("{destination}.tmp")).exists(),
            "temporary file must not be left behind"
        );
    }

    #[test]
    fn merge_rejects_foreign_schema_and_leaves_destination_untouched() {
        let dir = TempDir::new().expect("temp dir");
        let destination = path_in(&dir, "merged.parquet");
        let foreign_input = path_in(&dir, "foreign.parquet");

        write_records_to_parquet(&destination, &sample_records(0)).expect("seed destination");
        write_foreign_schema_parquet(&foreign_input);
        let original = std::fs::read(&destination).expect("read destination");

        let err = merge_parquet_files(&destination, &[foreign_input])
            .expect_err("merge must reject a non-discovery schema");
        assert!(
            err.to_string().contains("schema mismatch"),
            "unexpected error: {err}"
        );

        assert_eq!(
            std::fs::read(&destination).expect("re-read destination"),
            original,
            "destination must be unchanged after a schema rejection"
        );
        assert!(
            !Path::new(&format!("{destination}.tmp")).exists(),
            "temporary file must not be left behind"
        );
    }

    #[test]
    fn merge_rejects_output_aliasing_input() {
        let dir = TempDir::new().expect("temp dir");
        let destination = path_in(&dir, "merged.parquet");
        let other_input = path_in(&dir, "input_a.parquet");

        write_records_to_parquet(&destination, &sample_records(0)).expect("seed destination");
        write_records_to_parquet(&other_input, &sample_records(1)).expect("write input");
        let original = std::fs::read(&destination).expect("read destination");

        let err = merge_parquet_files(&destination, &[other_input, destination.clone()])
            .expect_err("merge must reject an output that aliases an input");
        assert!(
            err.to_string().contains("input list"),
            "unexpected error: {err}"
        );

        assert_eq!(
            std::fs::read(&destination).expect("re-read destination"),
            original,
            "aliased destination must not be truncated"
        );
        assert!(
            !Path::new(&format!("{destination}.tmp")).exists(),
            "temporary file must not be left behind"
        );
    }

    #[test]
    fn merge_writes_all_inputs_and_leaves_no_temp_file() {
        let dir = TempDir::new().expect("temp dir");
        let destination = path_in(&dir, "merged.parquet");
        let first = path_in(&dir, "input_a.parquet");
        let second = path_in(&dir, "input_b.parquet");

        write_records_to_parquet(&first, &sample_records(0)).expect("write first");
        write_records_to_parquet(&second, &sample_records(1)).expect("write second");

        merge_parquet_files(&destination, &[first, second]).expect("merge succeeds");

        let merged = crate::parquet_format::read_all_records_from_parquet(&destination)
            .expect("read merged file");
        assert_eq!(merged.len(), 4, "merged file must contain every input row");
        assert!(
            !Path::new(&format!("{destination}.tmp")).exists(),
            "temporary file must not be left behind"
        );
    }
}
