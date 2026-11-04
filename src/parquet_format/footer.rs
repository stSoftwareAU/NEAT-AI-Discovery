//! Parquet footer statistics used to project a decode's real cost (Issue #1869).
//!
//! The footer carries the **decompressed** shape of the file — the exact row
//! count and the uncompressed byte size of every row group — so a projection
//! built from it does not inherit the compression ratio of the on-disk bytes.
//! Reading it is metadata-only: no column data is decoded.

use anyhow::{Context, Result};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::types::DiscoverRecord;

/// Bytes charged for the fixed part of a materialised [`DiscoverRecord`]
/// (the struct itself: `obs_index`, the `String` and `Vec` headers, `value`
/// and `activation`).
pub const RECORD_STRUCT_BYTES: u64 = size_of::<DiscoverRecord>() as u64;

/// Heap bytes assumed for a neuron UUID when only the footer is available.
/// An RFC 4122 UUID is 36 characters; shorter identifiers (`input-N`) cost
/// less, so this is a representative rather than a worst case.
pub const ASSUMED_UUID_HEAP_BYTES: u64 = 36;

/// Root column name of the per-record error list.
const ERRORS_COLUMN: &str = "errors";

/// Bytes each decoded `f32` error occupies in the record's `Vec<f32>`.
const ERROR_VALUE_BYTES: u64 = 4;

/// Convert a signed footer field to `u64`, naming the field when it is negative.
///
/// The parquet crate accepts negative counts and sizes at parse time (Issue
/// #2447), so every footer reader converts through this one check rather than
/// letting a negative value count as zero (Issue #2509).
fn footer_u64(value: i64, field: &str) -> Result<u64> {
    u64::try_from(value).with_context(|| format!("Invalid Parquet footer {field}: {value}"))
}

fn sum_row_group_rows(mut row_counts: impl Iterator<Item = i64>) -> Result<u64> {
    row_counts.try_fold(0u64, |sum, row_count| {
        let row_count = footer_u64(row_count, "row-group num_rows")?;
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
}

impl ParquetFooterStats {
    /// Project the in-memory bytes needed to materialise every row as a
    /// [`DiscoverRecord`].
    ///
    /// Each row costs its struct plus a heap `String` for the UUID, and each
    /// decoded error costs four bytes in the record's `Vec<f32>`. Both counts
    /// come from the footer and are independent of how well the file
    /// compressed. The uncompressed row-group size acts as a floor for any
    /// payload the two counts miss.
    #[must_use]
    pub const fn projected_in_memory_bytes(&self) -> u64 {
        let records = self
            .num_rows
            .saturating_mul(RECORD_STRUCT_BYTES + ASSUMED_UUID_HEAP_BYTES);
        let errors = self.error_values.saturating_mul(ERROR_VALUE_BYTES);
        let projected = records.saturating_add(errors);
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
///
/// Fails when a row-group `num_rows`, a selected column's `uncompressed_size`
/// or an `errors` column's `num_values` is negative (Issue #2509).
pub(crate) fn projected_bytes_for_columns(
    metadata: &parquet::file::metadata::ParquetMetaData,
    selected_roots: &[&str],
    record_overhead: u64,
    include_errors: bool,
) -> Result<u64> {
    projected_bytes_for_row_groups(
        metadata.row_groups(),
        selected_roots,
        record_overhead,
        include_errors,
    )
}

/// Estimate the selected columns of the supplied decode units.
///
/// Fails, naming the field, when a row-group `num_rows`, a selected column's
/// `uncompressed_size` or an `errors` column's `num_values` is negative
/// (Issue #2509).
pub(crate) fn projected_bytes_for_row_groups(
    row_groups: &[parquet::file::metadata::RowGroupMetaData],
    selected_roots: &[&str],
    record_overhead: u64,
    include_errors: bool,
) -> Result<u64> {
    let mut rows = 0_u64;
    let mut selected_uncompressed = 0_u64;
    let mut error_values = 0_u64;
    for group in row_groups {
        rows = rows.saturating_add(footer_u64(group.num_rows(), "row-group num_rows")?);
        for column in group.columns() {
            let Some(root) = column.column_path().parts().first() else {
                continue;
            };
            if selected_roots.contains(&root.as_str()) {
                selected_uncompressed = selected_uncompressed
                    .saturating_add(footer_u64(column.uncompressed_size(), "uncompressed_size")?);
            }
            if include_errors && root == ERRORS_COLUMN {
                error_values =
                    error_values.saturating_add(footer_u64(column.num_values(), "num_values")?);
            }
        }
    }

    Ok(rows
        .saturating_mul(record_overhead)
        .saturating_add(error_values.saturating_mul(ERROR_VALUE_BYTES))
        .max(selected_uncompressed))
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
    let num_rows = footer_u64(file_num_rows, "num_rows")?;

    let mut uncompressed_bytes = 0u64;
    let mut error_values = 0u64;
    let row_group_rows = sum_row_group_rows(
        metadata
            .row_groups()
            .iter()
            .map(parquet::file::metadata::RowGroupMetaData::num_rows),
    )?;
    for row_group in metadata.row_groups() {
        let byte_size = footer_u64(row_group.total_byte_size(), "total_byte_size")?;
        uncompressed_bytes = uncompressed_bytes.saturating_add(byte_size);

        // Leaf value count of the `errors` list column: the exact number of
        // f32s the decode will materialise, however tightly they were encoded.
        for column in row_group.columns() {
            let value_count = footer_u64(column.num_values(), "num_values")?;
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

    Ok(ParquetFooterStats {
        num_rows,
        uncompressed_bytes,
        error_values,
    })
}

#[cfg(test)]
mod tests {
    use super::{read_footer_stats, sum_row_group_rows};

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
    }
}
