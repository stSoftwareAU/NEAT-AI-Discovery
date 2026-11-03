//! Writing discovery records into Parquet, and the atomicity boundary callers
//! must respect.
//!
//! [`ParquetRecordWriter`] streams into its destination file as records arrive:
//! each batch is validated against
//! [`create_schema`](crate::parquet_format::create_schema) and split so no
//! batch exceeds Arrow's offset budget, then appended immediately. **The file
//! is not a readable Parquet file until [`ParquetRecordWriter::finish`] writes
//! the footer**, so a concurrent reader must never be pointed at a file still
//! being written — write to a `.parquet.tmp` sibling and rename it into place
//! once `finish` returns (what `src/streaming.rs` does, and what the reader's
//! `.parquet.tmp` rejection backstops). [`merge_parquet_files`] is the one
//! function here that publishes atomically on the caller's behalf, via the same
//! temp-then-rename.
//!
//! Atomicity *within* a record is the caller's obligation, not this module's:
//! per AGENTS.md "Atomic Record Writes", every field of a
//! [`DiscoverRecord`](crate::types::DiscoverRecord) must come from one training
//! record. The writer stores each record's fields exactly as supplied and never
//! recombines them across rows, so it cannot detect a mixed record.

use anyhow::{Context, Result};
use arrow::array::{Float32Array, ListArray, StringArray, UInt32Array};
use arrow::datatypes::{DataType, Field};
use arrow::record_batch::RecordBatch;
use parquet::arrow::{ArrowWriter, ProjectionMask};
use parquet::file::properties::WriterProperties;
use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

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
fn remove_merge_temp_file(tmp_path: &Path) {
    if let Err(err) = fs::remove_file(tmp_path)
        && err.kind() != std::io::ErrorKind::NotFound
    {
        tracing::warn!(
            tmp_path = %tmp_path.display(),
            error = %err,
            "Failed to remove temporary merged Parquet file"
        );
    }
}

fn create_temp_file_exclusive(path: &Path) -> std::io::Result<File> {
    OpenOptions::new().write(true).create_new(true).open(path)
}

/// Allocate an exclusive sibling path for a merge, retrying name collisions.
fn create_merge_temp_file_with_ids(
    output_file: &str,
    mut next_id: impl FnMut() -> u64,
) -> Result<(PathBuf, File)> {
    const MAX_TEMP_ATTEMPTS: u32 = 128;

    let output_path = Path::new(output_file);
    let parent = output_path.parent().unwrap_or_else(|| Path::new("."));
    let file_name = output_path
        .file_name()
        .ok_or_else(|| anyhow::anyhow!("Merge output path has no file name: {output_file}"))?;

    for _ in 0..MAX_TEMP_ATTEMPTS {
        let id = next_id();
        let tmp_name = format!(
            "{}.tmp.{}.{}",
            file_name.to_string_lossy(),
            std::process::id(),
            id
        );
        let tmp_path = parent.join(tmp_name);
        match create_temp_file_exclusive(&tmp_path) {
            Ok(file) => return Ok((tmp_path, file)),
            Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(err) => {
                return Err(err).with_context(|| {
                    format!(
                        "Failed to create temporary merged Parquet file: {}",
                        tmp_path.display()
                    )
                });
            }
        }
    }

    anyhow::bail!(
        "Failed to allocate a unique temporary file for merge output {output_file} after {MAX_TEMP_ATTEMPTS} attempts"
    )
}

/// Decoded bytes buffered at which a merge closes the current output row group.
const MERGE_OUTPUT_ROW_GROUP_FLUSH_BYTES: usize = 2 * 1024 * 1024;

/// Read every input file and write the merged result into the open temp file.
///
/// Each input's schema is validated before any batch is appended, so a foreign
/// schema is rejected rather than smuggled into the merged output.
fn merge_into_temp_file(file: File, input_files: &[String]) -> Result<()> {
    let schema = Arc::new(create_schema());
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))
        .context("Failed to create ArrowWriter for merge")?;

    let mut buffered_bytes = 0_usize;
    for input_path in input_files {
        let input_file = File::open(input_path)
            .with_context(|| format!("Failed to open discovery Parquet file: {input_path}"))?;
        let builder = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(
            input_file.try_clone()?,
        )
        .with_context(|| format!("Failed to create reader for {input_path}"))?;
        validate_parquet_schema(&builder, input_path, ColumnProfile::Full)?;
        crate::parquet_format::decode_budget::DecodeBudget::resolve(None).preflight_row_groups(
            builder.metadata(),
            &ColumnProfile::Full.root_names(),
            input_path,
        )?;
        let mask =
            ProjectionMask::roots(builder.parquet_schema(), ColumnProfile::Full.root_indices());
        let reader = crate::parquet_format::decode_budget::row_group_batches(
            input_file,
            builder.metadata().clone(),
            mask,
        )?;

        for batch_result in reader {
            let batch =
                batch_result.with_context(|| format!("Failed to read batch from {input_path}"))?;
            writer
                .write(&batch)
                .with_context(|| format!("Failed to append batch from {input_path}"))?;
            // Close the output row group once enough decoded data has been
            // buffered, so the merged file keeps few, large row groups while
            // the encoder stays bounded by this limit plus one input batch.
            // The encoder's own size estimate undercounts highly compressible
            // data, so count the decoded Arrow bytes instead.
            buffered_bytes = buffered_bytes.saturating_add(batch.get_array_memory_size());
            if buffered_bytes >= MERGE_OUTPUT_ROW_GROUP_FLUSH_BYTES {
                writer
                    .flush()
                    .context("Failed to flush merged Parquet row group")?;
                buffered_bytes = 0;
            }
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
/// schema-validated into a uniquely named sibling temporary file, which is
/// created exclusively and renamed onto `output_file` only once the merged
/// writer has closed cleanly. Any failure removes only the temporary file
/// created by this invocation and leaves an existing destination untouched.
///
/// # Errors
///
/// Returns an error if `input_files` is empty, if any entry of `input_files`
/// resolves to `output_file` or the reserved `{output_file}.tmp` path (which
/// would otherwise collide with a merge temporary path), if any input is
/// missing, unreadable, or carries a non-discovery schema, or if a unique
/// temporary file cannot be written or renamed.
pub fn merge_parquet_files(output_file: &str, input_files: &[String]) -> Result<()> {
    static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);
    merge_parquet_files_with_ids(output_file, input_files, || {
        NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed)
    })
}

fn merge_parquet_files_with_ids(
    output_file: &str,
    input_files: &[String],
    next_id: impl FnMut() -> u64,
) -> Result<()> {
    if input_files.is_empty() {
        return Err(anyhow::anyhow!(
            "No discovery parquet files provided for merge"
        ));
    }

    let output_resolved = resolve_for_alias_check(output_file);
    let legacy_tmp_path = format!("{output_file}.tmp");
    let legacy_tmp_resolved = resolve_for_alias_check(&legacy_tmp_path);
    for input_path in input_files {
        if resolve_for_alias_check(input_path) == output_resolved {
            anyhow::bail!(
                "Merge output file '{output_file}' also appears in the input list as \
                 '{input_path}'; refusing to merge a file onto itself"
            );
        }
        if resolve_for_alias_check(input_path) == legacy_tmp_resolved {
            anyhow::bail!(
                "Merge input '{input_path}' aliases the reserved temporary path '{legacy_tmp_path}'; refusing to merge it"
            );
        }
    }

    let (tmp_path, file) = create_merge_temp_file_with_ids(output_file, next_id)?;
    let tmp_path_string = tmp_path.to_string_lossy().into_owned();
    let resolved_tmp_path = resolve_for_alias_check(&tmp_path_string);
    if let Some(input_path) = input_files
        .iter()
        .find(|input_path| resolve_for_alias_check(input_path) == resolved_tmp_path)
    {
        drop(file);
        remove_merge_temp_file(&tmp_path);
        anyhow::bail!(
            "Merge input '{input_path}' aliases the temporary path '{tmp_path_string}'; refusing to merge it"
        );
    }
    if let Err(err) = merge_into_temp_file(file, input_files) {
        remove_merge_temp_file(&tmp_path);
        return Err(err);
    }

    if let Err(err) = fs::rename(&tmp_path, output_file) {
        remove_merge_temp_file(&tmp_path);
        return Err(anyhow::Error::new(err).context(format!(
            "Failed to rename temporary file {tmp_path_string} to {output_file}"
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

    fn assert_no_merge_temp_files(dir: &TempDir, destination: &str) {
        let prefix = format!(
            "{}.tmp.",
            Path::new(destination)
                .file_name()
                .unwrap()
                .to_string_lossy()
        );
        let leftovers = fs::read_dir(dir.path())
            .expect("read directory")
            .map(|entry| entry.expect("directory entry").file_name())
            .filter(|name| name.to_string_lossy().starts_with(&prefix))
            .collect::<Vec<_>>();
        assert!(
            leftovers.is_empty(),
            "merge temp files remain: {leftovers:?}"
        );
    }

    #[test]
    fn merge_temp_creation_does_not_open_an_existing_path() {
        let dir = TempDir::new().expect("temp dir");
        let path = dir.path().join("occupied.tmp");
        fs::write(&path, b"preserve these bytes").expect("write existing file");

        let err = create_temp_file_exclusive(&path).expect_err("existing path must be refused");

        assert_eq!(err.kind(), std::io::ErrorKind::AlreadyExists);
        assert_eq!(
            fs::read(path).expect("read existing file"),
            b"preserve these bytes"
        );
    }

    #[test]
    fn merge_temp_allocation_uses_distinct_sibling_paths() {
        let dir = TempDir::new().expect("temp dir");
        let output = path_in(&dir, "merged.parquet");

        let mut ids = [0, 1].into_iter();
        let (first_path, first_file) =
            create_merge_temp_file_with_ids(&output, || ids.next().expect("first candidate"))
                .expect("first temp file");
        let (second_path, second_file) =
            create_merge_temp_file_with_ids(&output, || ids.next().expect("second candidate"))
                .expect("second temp file");

        assert_ne!(first_path, second_path);
        assert_eq!(first_path.parent(), Path::new(&output).parent());
        assert_eq!(second_path.parent(), Path::new(&output).parent());
        drop((first_file, second_file));
        fs::remove_file(first_path).expect("remove first temp file");
        fs::remove_file(second_path).expect("remove second temp file");
    }

    #[test]
    fn merge_temp_allocation_retries_collision_without_changing_occupied_file() {
        let dir = TempDir::new().expect("temp dir");
        let output = path_in(&dir, "merged.parquet");
        let occupied = dir
            .path()
            .join(format!("merged.parquet.tmp.{}.17", std::process::id()));
        fs::write(&occupied, b"preserve collision").expect("write occupied candidate");
        let original = fs::read(&occupied).expect("read occupied candidate");
        let mut ids = [17, 18].into_iter();

        let (allocated, file) =
            create_merge_temp_file_with_ids(&output, || ids.next().expect("next candidate"))
                .expect("retry with the next candidate");

        assert_eq!(
            allocated.file_name().unwrap().to_string_lossy(),
            format!("merged.parquet.tmp.{}.18", std::process::id())
        );
        assert_eq!(
            fs::read(&occupied).expect("re-read occupied candidate"),
            original
        );
        drop(file);
        fs::remove_file(allocated).expect("remove allocated temp file");
    }

    #[test]
    fn merge_temp_allocation_fails_after_collision_limit() {
        let dir = TempDir::new().expect("temp dir");
        let output = path_in(&dir, "merged.parquet");
        let occupied = dir
            .path()
            .join(format!("merged.parquet.tmp.{}.19", std::process::id()));
        fs::write(&occupied, b"still occupied").expect("write occupied candidate");
        let original = fs::read(&occupied).expect("read occupied candidate");

        let error = create_merge_temp_file_with_ids(&output, || 19)
            .expect_err("exhausted collisions must fail");

        assert!(format!("{error:#}").contains("after 128 attempts"));
        assert_eq!(
            fs::read(occupied).expect("re-read occupied candidate"),
            original
        );
    }

    #[test]
    fn merge_temp_allocation_reports_non_collision_errors() {
        let dir = TempDir::new().expect("temp dir");
        let output = dir
            .path()
            .join("missing-parent")
            .join("merged.parquet")
            .to_string_lossy()
            .into_owned();

        let error = create_merge_temp_file_with_ids(&output, || 0)
            .expect_err("missing parent must fail allocation");

        assert!(format!("{error:#}").contains("Failed to create temporary merged Parquet file"));
    }

    #[cfg(unix)]
    #[test]
    fn merge_rejects_input_alias_to_new_temp_path() {
        use std::os::unix::fs::symlink;

        let dir = TempDir::new().expect("temp dir");
        let output = path_in(&dir, "merged.parquet");
        let temp_target = dir
            .path()
            .join(format!("merged.parquet.tmp.{}.42", std::process::id()));
        let input_alias = dir.path().join("input-alias.parquet");
        symlink(&temp_target, &input_alias).expect("create dangling input symlink");

        let error = merge_parquet_files_with_ids(
            &output,
            &[input_alias.to_string_lossy().into_owned()],
            || 42,
        )
        .expect_err("an input resolving to the allocated temp must be refused");

        assert!(format!("{error:#}").contains("aliases the temporary path"));
        assert!(input_alias.is_symlink());
        assert!(!temp_target.exists(), "allocated temp must be cleaned up");
    }

    #[test]
    fn merge_rename_failure_preserves_destination_and_cleans_created_temp() {
        let dir = TempDir::new().expect("temp dir");
        let destination = dir.path().join("destination");
        fs::create_dir(&destination).expect("create destination directory");
        let input = path_in(&dir, "input.parquet");
        write_records_to_parquet(&input, &sample_records(0)).expect("write input");

        let error = merge_parquet_files(&destination.to_string_lossy(), &[input])
            .expect_err("renaming a file over a directory must fail");

        assert!(format!("{error:#}").contains("Failed to rename temporary file"));
        assert!(destination.is_dir(), "destination directory must remain");
        let leftovers = fs::read_dir(dir.path())
            .expect("read directory")
            .map(|entry| entry.expect("directory entry").file_name())
            .filter(|name| name.to_string_lossy().starts_with("destination.tmp."))
            .collect::<Vec<_>>();
        assert!(
            leftovers.is_empty(),
            "merge temp files remain: {leftovers:?}"
        );
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
        assert_no_merge_temp_files(&dir, &destination);
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
        assert_no_merge_temp_files(&dir, &destination);
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
        assert_no_merge_temp_files(&dir, &destination);
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
        assert_no_merge_temp_files(&dir, &destination);
    }

    /// Build a record whose UUID is `uuid_len` valid bytes and which carries
    /// `error_count` error values.
    fn shaped_record(uuid_len: usize, error_count: usize) -> DiscoverRecord {
        DiscoverRecord {
            obs_index: 0,
            neuron_uuid: "a".repeat(uuid_len),
            value: None,
            activation: 0.0,
            errors: vec![0.0; error_count],
        }
    }

    /// Build records from `(uuid_len, error_count)` pairs.
    fn shaped_records(shape: &[(usize, usize)]) -> Vec<DiscoverRecord> {
        shape
            .iter()
            .map(|&(uuid_len, error_count)| shaped_record(uuid_len, error_count))
            .collect()
    }

    /// Mirror of the chunking loop in `write_records`, returning each chunk's
    /// end index. Panics instead of spinning if a chunk makes no progress
    /// (Issue #2453).
    fn chunk_ends(records: &[DiscoverRecord], uuid_limit: usize, err_limit: usize) -> Vec<usize> {
        let mut ends = Vec::new();
        let mut start = 0;
        while start < records.len() {
            let end = determine_chunk_end(records, start, uuid_limit, err_limit);
            assert!(
                end > start,
                "Issue #2453: chunk starting at record {start} made no progress"
            );
            ends.push(end);
            start = end;
        }
        ends
    }

    /// Issue #2453: pins the chunk boundaries of `determine_chunk_end` when only
    /// the UUID byte budget binds. Five 8-byte UUIDs, unlimited error budget.
    /// A chunk may hold a total of exactly `limit` UUID bytes.
    #[test]
    fn determine_chunk_end_uuid_limit_binds_only() {
        let records = shaped_records(&[(8, 0); 5]);
        assert_eq!(
            chunk_ends(&records, 20, usize::MAX),
            vec![2, 4, 5],
            "limit 20: 8+8=16 fits, third UUID would reach 24"
        );
        assert_eq!(
            chunk_ends(&records, 16, usize::MAX),
            vec![2, 4, 5],
            "limit 16: exact fit (8+8=16) must stay in the chunk"
        );
        assert_eq!(
            chunk_ends(&records, 15, usize::MAX),
            vec![1, 2, 3, 4, 5],
            "limit 15: second UUID would reach 16 and must split"
        );
        assert_eq!(
            chunk_ends(&records, 8, usize::MAX),
            vec![1, 2, 3, 4, 5],
            "limit 8: exactly one record per chunk"
        );
        assert_eq!(
            chunk_ends(&records, 40, usize::MAX),
            vec![5],
            "limit 40: all five UUIDs (40 bytes) fit exactly"
        );
    }

    /// Issue #2453: pins the chunk boundaries of `determine_chunk_end` when only
    /// the error-value budget binds. Five records with 2 error values each,
    /// unlimited UUID budget. A chunk may hold exactly `limit` error values.
    #[test]
    fn determine_chunk_end_error_limit_binds_only() {
        let records = shaped_records(&[(1, 2); 5]);
        assert_eq!(
            chunk_ends(&records, usize::MAX, 5),
            vec![2, 4, 5],
            "limit 5: 2+2=4 fits, third record would reach 6"
        );
        assert_eq!(
            chunk_ends(&records, usize::MAX, 4),
            vec![2, 4, 5],
            "limit 4: exact fit (2+2=4) must stay in the chunk"
        );
        assert_eq!(
            chunk_ends(&records, usize::MAX, 3),
            vec![1, 2, 3, 4, 5],
            "limit 3: second record would reach 4 and must split"
        );
    }

    /// Issue #2453: pins the exact chunk vector for mixed error-value counts
    /// under the error budget only. Chunks are [2,3], [1,4], [5,0] and [2].
    #[test]
    fn determine_chunk_end_mixed_error_counts_exact_vector() {
        let records = shaped_records(&[(1, 2), (1, 3), (1, 1), (1, 4), (1, 5), (1, 0), (1, 2)]);
        assert_eq!(
            chunk_ends(&records, usize::MAX, 5),
            vec![2, 4, 6, 7],
            "error budget 5 over counts [2,3,1,4,5,0,2]"
        );
    }

    /// Issue #2453: pins chunks where the two budgets bind at different
    /// records. The first chunk ends on the UUID byte budget (4+4 fits in 10,
    /// a further 4 would reach 12), the second on the error budget (1+2 fits
    /// in 3 error values, a further 2 would reach 5), and the last two on the
    /// error budget again.
    #[test]
    fn determine_chunk_end_both_limits_bind_at_different_records() {
        let records = shaped_records(&[(4, 1), (4, 1), (4, 1), (1, 2), (1, 2), (1, 2)]);
        assert_eq!(
            chunk_ends(&records, 10, 3),
            vec![2, 4, 5, 6],
            "uuid limit 10 and error limit 3 with mixed binding operands"
        );
    }

    /// Issue #2453: progress guarantee. A first record whose UUID alone exceeds
    /// the per-batch UUID limit must still yield `start + 1`, so the loop in
    /// `write_records` advances. `validate_records` rejects such records on the
    /// real path, so this pins the helper directly. Also checked with
    /// `start > 0`.
    #[test]
    fn determine_chunk_end_oversized_first_uuid_still_makes_progress() {
        let records = shaped_records(&[(10, 0), (1, 0), (1, 0)]);
        assert_eq!(
            determine_chunk_end(&records, 0, 5, usize::MAX),
            1,
            "oversized first UUID (10 > 5) must yield end == start + 1"
        );
        assert_eq!(
            chunk_ends(&records, 5, usize::MAX),
            vec![1, 3],
            "oversized record isolated, the two 1-byte followers share a chunk (2 <= 5)"
        );

        let offset_records = shaped_records(&[(1, 0), (10, 0), (1, 0)]);
        assert_eq!(
            determine_chunk_end(&offset_records, 1, 5, usize::MAX),
            2,
            "oversized UUID at start 1 must yield end == 2"
        );
        assert_eq!(
            determine_chunk_end(&offset_records, 2, 5, usize::MAX),
            3,
            "trailing record at start 2 must yield end == 3"
        );
    }

    /// Issue #2453: progress guarantee for the error budget. A first record
    /// whose error count alone exceeds the limit must yield `start + 1`. Also
    /// checked with `start > 0`.
    #[test]
    fn determine_chunk_end_oversized_first_error_count_still_makes_progress() {
        let records = shaped_records(&[(1, 5), (1, 0)]);
        assert_eq!(
            determine_chunk_end(&records, 0, usize::MAX, 3),
            1,
            "oversized first error count (5 > 3) must yield end == start + 1"
        );
        assert_eq!(
            chunk_ends(&records, usize::MAX, 3),
            vec![1, 2],
            "oversized record isolated, follower in its own chunk"
        );

        let offset_records = shaped_records(&[(1, 0), (1, 5), (1, 0)]);
        assert_eq!(
            determine_chunk_end(&offset_records, 1, usize::MAX, 3),
            2,
            "oversized error count at start 1 must yield end == 2"
        );
    }

    /// Issue #2453: pins that `determine_chunk_end` honours a non-zero `start`:
    /// budgets are counted from `start`, not from record 0, and the returned
    /// index is always greater than `start`.
    #[test]
    fn determine_chunk_end_respects_nonzero_start() {
        let uuid_records = shaped_records(&[(8, 0); 5]);
        assert_eq!(
            determine_chunk_end(&uuid_records, 1, 20, usize::MAX),
            3,
            "start 1: records 1 and 2 fit (16 bytes), record 3 would reach 24"
        );
        assert_eq!(
            determine_chunk_end(&uuid_records, 2, 20, usize::MAX),
            4,
            "start 2: records 2 and 3 fit, record 4 would reach 24"
        );
        assert_eq!(
            determine_chunk_end(&uuid_records, 4, 20, usize::MAX),
            5,
            "start 4: single trailing record"
        );

        let error_records = shaped_records(&[(1, 2); 5]);
        assert_eq!(
            determine_chunk_end(&error_records, 1, usize::MAX, 5),
            3,
            "start 1: records 1 and 2 fit (4 error values), record 3 would reach 6"
        );
    }

    /// Issue #2453: pins that unlimited budgets (`usize::MAX`) produce a single
    /// chunk covering every record.
    #[test]
    fn determine_chunk_end_unlimited_limits_yield_single_chunk() {
        let records = shaped_records(&[(8, 2); 5]);
        assert_eq!(
            chunk_ends(&records, usize::MAX, usize::MAX),
            vec![5],
            "unlimited budgets must cover all records in one chunk"
        );
        assert_eq!(
            determine_chunk_end(&records, 0, usize::MAX, usize::MAX),
            5,
            "unlimited budgets must return records.len() from start 0"
        );
    }
}
