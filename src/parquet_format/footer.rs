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
