//! Parquet footer statistics used to project a decode's real cost (Issue #1869).
//!
//! The footer carries the **decompressed** shape of the file — the exact row
//! count and the uncompressed byte size of every row group — so a projection
//! built from it does not inherit the compression ratio of the on-disk bytes.
//! Reading it is metadata-only: no column data is decoded.

use anyhow::{Context, Result};
use parquet::arrow::arrow_reader::{DEFAULT_BATCH_SIZE, ParquetRecordBatchReaderBuilder};
use parquet::basic::Encoding;
use parquet::file::metadata::{ColumnChunkMetaData, RowGroupMetaData};

use crate::types::DiscoverRecord;

/// Bytes charged for the fixed part of a materialised [`DiscoverRecord`]
/// (the struct itself: `obs_index`, the `String` and `Vec` headers, `value`
/// and `activation`).
pub const RECORD_STRUCT_BYTES: u64 = size_of::<DiscoverRecord>() as u64;

/// Per-row heap floor for a neuron UUID when the footer carries no size
/// statistic for its column. An RFC 4122 UUID is 36 characters. The footer's
/// `unencoded_byte_array_data_bytes` statistic, when present, supersedes this
/// figure, so it is not a bound on the decoded identifier length.
pub const ASSUMED_UUID_HEAP_BYTES: u64 = 36;

/// Root column name of the per-record error list.
const ERRORS_COLUMN: &str = "errors";

/// Root column name of the neuron identifier.
const NEURON_UUID_COLUMN: &str = "neuron_uuid";

/// Bytes each decoded `f32` error occupies in the record's `Vec<f32>`.
const ERROR_VALUE_BYTES: u64 = 4;

fn sum_row_group_rows(mut row_counts: impl Iterator<Item = i64>) -> Result<u64> {
    row_counts.try_fold(0u64, |sum, row_count| {
        let row_count = u64::try_from(row_count)
            .with_context(|| format!("Invalid Parquet footer row-group num_rows: {row_count}"))?;
        sum.checked_add(row_count)
            .context("Overflow while summing Parquet footer row-group num_rows")
    })
}

/// Decompressed shape of a Parquet file, read from its footer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParquetFooterStats {
    /// Total rows across all row groups (exact, not estimated).
    pub num_rows: u64,
    /// Sum of the row groups' **uncompressed** byte sizes. Still an *encoded*
    /// figure — dictionary and RLE pages stay compact even uncompressed — so it
    /// is a floor, not the decoded cost.
    pub uncompressed_bytes: u64,
    /// Total decoded `f32` values in the `errors` list column across all rows
    /// (exact, from the leaf column chunks' value counts).
    pub error_values: u64,
    /// Decoded `neuron_uuid` string bytes across all row groups, from the
    /// footer's size statistic or the dictionary-aware bound described on
    /// [`neuron_uuid_heap_bytes`]. Excludes the per-row Arrow offset word.
    pub uuid_heap_bytes: u64,
}

impl ParquetFooterStats {
    /// Project the in-memory bytes needed to materialise every row as a
    /// [`DiscoverRecord`].
    ///
    /// Each row costs its struct, each decoded error costs four bytes in the
    /// record's `Vec<f32>`, and the UUID strings cost `uuid_heap_bytes`. All
    /// three come from the footer and are independent of how well the file
    /// compressed. The uncompressed row-group size acts as a floor for any
    /// payload the three counts miss.
    #[must_use]
    pub const fn projected_in_memory_bytes(&self) -> u64 {
        let records = self.num_rows.saturating_mul(RECORD_STRUCT_BYTES);
        let errors = self.error_values.saturating_mul(ERROR_VALUE_BYTES);
        let projected = records
            .saturating_add(self.uuid_heap_bytes)
            .saturating_add(errors);
        if projected > self.uncompressed_bytes {
            projected
        } else {
            self.uncompressed_bytes
        }
    }
}

/// Estimate the selected decoded columns from an already-open reader's footer.
/// `record_overhead` is the per-row allocation outside the selected Arrow
/// columns, and `include_errors` adds the decoded list value count. The
/// selected uncompressed size is a floor, as in the preload planner's estimate.
pub(crate) fn projected_bytes_for_columns(
    metadata: &parquet::file::metadata::ParquetMetaData,
    selected_roots: &[&str],
    record_overhead: u64,
    include_errors: bool,
) -> u64 {
    projected_bytes_for_row_groups(
        metadata.row_groups(),
        selected_roots,
        record_overhead,
        include_errors,
    )
}

/// Estimate the selected columns of the supplied decode units.
///
/// When `neuron_uuid` is selected, its decoded string heap from
/// [`neuron_uuid_heap_bytes`] is added to the per-row overhead term. Callers
/// therefore pass only the per-row overhead of the other selected columns.
pub(crate) fn projected_bytes_for_row_groups(
    row_groups: &[RowGroupMetaData],
    selected_roots: &[&str],
    record_overhead: u64,
    include_errors: bool,
) -> u64 {
    let rows = row_groups
        .iter()
        .map(|group| u64::try_from(group.num_rows()).unwrap_or(0))
        .fold(0_u64, u64::saturating_add);
    let selected_uncompressed = row_groups
        .iter()
        .flat_map(RowGroupMetaData::columns)
        .filter(|column| {
            column
                .column_path()
                .parts()
                .first()
                .is_some_and(|root| selected_roots.contains(&root.as_str()))
        })
        .map(|column| u64::try_from(column.uncompressed_size()).unwrap_or(0))
        .fold(0_u64, u64::saturating_add);
    let selected_errors = if include_errors {
        row_groups
            .iter()
            .flat_map(RowGroupMetaData::columns)
            .filter(|column| {
                column
                    .column_path()
                    .parts()
                    .first()
                    .is_some_and(|root| root == ERRORS_COLUMN)
            })
            .map(|column| u64::try_from(column.num_values()).unwrap_or(0))
            .fold(0_u64, u64::saturating_add)
            .saturating_mul(ERROR_VALUE_BYTES)
    } else {
        0
    };
    let selected_uuid_heap = if selected_roots.contains(&NEURON_UUID_COLUMN) {
        neuron_uuid_heap_bytes(row_groups)
    } else {
        0
    };

    rows.saturating_mul(record_overhead)
        .saturating_add(selected_errors)
        .saturating_add(selected_uuid_heap)
        .max(selected_uncompressed)
}

/// Decoded `neuron_uuid` string bytes across the supplied row groups (Issue #2507).
///
/// Arrow copies every value into its batch, so a dictionary-encoded column
/// costs its decoded length once per row even though the file stores each
/// distinct string once. A footer can therefore be tiny while the decode is
/// large. The per-chunk value is chosen as follows:
///
/// 1. The `unencoded_byte_array_data_bytes` size statistic is the exact total
///    of the decoded value bytes, so it is used when present and non-negative.
///    A negative statistic is malformed and is treated as absent.
/// 2. Otherwise, a dictionary-encoded chunk is bounded by
///    `min(rows, DEFAULT_BATCH_SIZE) * uncompressed`: one batch holds at most
///    that many copies, and the longest value cannot exceed the chunk's
///    uncompressed size.
/// 3. Otherwise, a plain chunk stores every value, so its uncompressed size is
///    already the decoded total.
///
/// Fallbacks 2 and 3 are raised to `rows * ASSUMED_UUID_HEAP_BYTES`, the
/// per-row floor. Arithmetic saturates.
pub(crate) fn neuron_uuid_heap_bytes(row_groups: &[RowGroupMetaData]) -> u64 {
    row_groups
        .iter()
        .map(|group| {
            let rows = u64::try_from(group.num_rows()).unwrap_or(0);
            group
                .columns()
                .iter()
                .filter(|column| {
                    column
                        .column_path()
                        .parts()
                        .first()
                        .is_some_and(|root| root == NEURON_UUID_COLUMN)
                })
                .map(|column| column_uuid_heap_bytes(column, rows))
                .fold(0_u64, u64::saturating_add)
        })
        .fold(0_u64, u64::saturating_add)
}

/// Decoded `neuron_uuid` string bytes for one column chunk of `rows` rows.
/// See [`neuron_uuid_heap_bytes`] for the rules.
fn column_uuid_heap_bytes(column: &ColumnChunkMetaData, rows: u64) -> u64 {
    if let Some(bytes) = column
        .unencoded_byte_array_data_bytes()
        .and_then(|bytes| u64::try_from(bytes).ok())
    {
        return bytes;
    }

    let floor = rows.saturating_mul(ASSUMED_UUID_HEAP_BYTES);
    let uncompressed = u64::try_from(column.uncompressed_size()).unwrap_or(0);
    let dictionary = column.encodings().any(|encoding| {
        matches!(
            encoding,
            Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY
        )
    });
    let decoded = if dictionary {
        rows.min(DEFAULT_BATCH_SIZE as u64)
            .saturating_mul(uncompressed)
    } else {
        uncompressed
    };
    decoded.max(floor)
}

/// Read the row count and uncompressed row-group sizes from `file_path`'s
/// Parquet footer.
///
/// Returns an error when the file cannot be opened, its footer cannot be
/// parsed, or its signed counts and sizes are negative or inconsistent —
/// callers decide whether an unknown projection is fatal.
pub fn read_footer_stats(file_path: &str) -> Result<ParquetFooterStats> {
    let file = std::fs::File::open(file_path)
        .with_context(|| format!("Failed to open Parquet file for footer read: {file_path}"))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("Failed to read Parquet footer: {file_path}"))?;
    let metadata = builder.metadata();

    let file_num_rows = metadata.file_metadata().num_rows();
    let num_rows = u64::try_from(file_num_rows)
        .with_context(|| format!("Invalid Parquet footer num_rows: {file_num_rows}"))?;

    let mut uncompressed_bytes = 0u64;
    let mut error_values = 0u64;
    let row_group_rows = sum_row_group_rows(
        metadata
            .row_groups()
            .iter()
            .map(parquet::file::metadata::RowGroupMetaData::num_rows),
    )?;
    for row_group in metadata.row_groups() {
        let byte_size = row_group.total_byte_size();
        let byte_size = u64::try_from(byte_size)
            .with_context(|| format!("Invalid Parquet footer total_byte_size: {byte_size}"))?;
        uncompressed_bytes = uncompressed_bytes.saturating_add(byte_size);

        // Leaf value count of the `errors` list column: the exact number of
        // f32s the decode will materialise, however tightly they were encoded.
        for column in row_group.columns() {
            let value_count = column.num_values();
            let value_count = u64::try_from(value_count)
                .with_context(|| format!("Invalid Parquet footer num_values: {value_count}"))?;
            if column
                .column_path()
                .parts()
                .first()
                .is_some_and(|part| part == ERRORS_COLUMN)
            {
                error_values = error_values.saturating_add(value_count);
            }
        }
    }

    anyhow::ensure!(
        num_rows == row_group_rows,
        "Parquet footer num_rows ({num_rows}) does not match row-group num_rows sum ({row_group_rows})"
    );

    let uuid_heap_bytes = neuron_uuid_heap_bytes(metadata.row_groups());

    Ok(ParquetFooterStats {
        num_rows,
        uncompressed_bytes,
        error_values,
        uuid_heap_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        ASSUMED_UUID_HEAP_BYTES, neuron_uuid_heap_bytes, read_footer_stats, sum_row_group_rows,
    };
    use parquet::basic::{Encoding, Repetition, Type as PhysicalType};
    use parquet::file::metadata::{ColumnChunkMetaData, RowGroupMetaData};
    use parquet::schema::types::{SchemaDescriptor, Type};
    use std::sync::Arc;

    /// One row group holding a single `neuron_uuid` `BYTE_ARRAY` column chunk.
    fn neuron_uuid_row_group(
        rows: i64,
        encodings: Vec<Encoding>,
        uncompressed: i64,
        size_statistic: Option<i64>,
    ) -> RowGroupMetaData {
        let field = Type::primitive_type_builder("neuron_uuid", PhysicalType::BYTE_ARRAY)
            .with_repetition(Repetition::REQUIRED)
            .build()
            .expect("neuron_uuid field");
        let schema = Type::group_type_builder("schema")
            .with_fields(vec![Arc::new(field)])
            .build()
            .expect("schema");
        let descr = Arc::new(SchemaDescriptor::new(Arc::new(schema)));
        let column = ColumnChunkMetaData::builder(descr.column(0))
            .set_encodings(encodings)
            .set_num_values(rows)
            .set_total_uncompressed_size(uncompressed)
            .set_unencoded_byte_array_data_bytes(size_statistic)
            .build()
            .expect("neuron_uuid column chunk");
        RowGroupMetaData::builder(Arc::clone(&descr))
            .set_num_rows(rows)
            .set_column_metadata(vec![column])
            .build()
            .expect("neuron_uuid row group")
    }

    fn heap_bytes(group: RowGroupMetaData) -> u64 {
        neuron_uuid_heap_bytes(&[group])
    }

    #[test]
    fn size_statistic_is_trusted_exactly_when_present() {
        // Distinct from rows * 36 (72_000) and from the uncompressed size (5_000),
        // and the dictionary fallback would give 1_024 * 5_000 instead.
        let group = neuron_uuid_row_group(
            2_000,
            vec![Encoding::PLAIN, Encoding::RLE_DICTIONARY],
            5_000,
            Some(123_457),
        );
        assert_eq!(heap_bytes(group), 123_457);
    }

    #[test]
    fn absent_statistic_on_dictionary_chunk_bounds_one_batch_of_copies() {
        for encoding in [Encoding::RLE_DICTIONARY, Encoding::PLAIN_DICTIONARY] {
            // Fewer rows than one batch: every row is copied.
            let small = neuron_uuid_row_group(300, vec![encoding], 10_000, None);
            assert_eq!(heap_bytes(small), 300 * 10_000);

            // More rows than one batch: only one batch of copies is bounded.
            let large = neuron_uuid_row_group(3_000, vec![encoding], 10_000, None);
            assert_eq!(heap_bytes(large), 1_024 * 10_000);
        }
    }

    #[test]
    fn absent_statistic_on_plain_chunk_uses_uncompressed_size_with_row_floor() {
        // Plain pages store each value once, so the uncompressed size is the
        // decoded total, raised to the per-row floor when that is larger.
        let floored = neuron_uuid_row_group(3_000, vec![Encoding::PLAIN], 50_000, None);
        assert_eq!(heap_bytes(floored), 3_000 * ASSUMED_UUID_HEAP_BYTES);

        let uncompressed = neuron_uuid_row_group(10, vec![Encoding::PLAIN], 1_000_000, None);
        assert_eq!(heap_bytes(uncompressed), 1_000_000);
    }

    #[test]
    fn negative_statistic_is_not_trusted() {
        // A negative size statistic is malformed. It falls back to the same
        // bounds as an absent one rather than being read as zero or wrapped.
        let plain = neuron_uuid_row_group(10, vec![Encoding::PLAIN], 1_000, Some(-1));
        assert_eq!(heap_bytes(plain), 1_000);

        let dictionary =
            neuron_uuid_row_group(2_000, vec![Encoding::RLE_DICTIONARY], 10_000, Some(-5));
        assert_eq!(heap_bytes(dictionary), 1_024 * 10_000);
    }

    #[test]
    fn row_group_row_sum_rejects_overflow() {
        let error = sum_row_group_rows([i64::MAX, i64::MAX, i64::MAX].into_iter())
            .expect_err("row-group row sum must not wrap");
        let message = format!("{error:#}");
        assert!(
            message.contains("num_rows"),
            "error names num_rows: {message}"
        );
        assert!(
            message.contains("Overflow"),
            "error names overflow: {message}"
        );
    }

    #[test]
    fn empty_footer_stats_remain_valid() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("empty.parquet");
        let writer = parquet::arrow::ArrowWriter::try_new(
            std::fs::File::create(&path).expect("create empty Parquet file"),
            std::sync::Arc::new(crate::parquet_format::create_schema()),
            None,
        )
        .expect("create Parquet writer");
        writer.close().expect("write empty footer");

        let stats =
            read_footer_stats(path.to_str().expect("UTF-8 path")).expect("valid empty footer");
        assert_eq!(stats.num_rows, 0);
        assert_eq!(stats.uncompressed_bytes, 0);
        assert_eq!(stats.error_values, 0);
        assert_eq!(stats.uuid_heap_bytes, 0);
    }
}
