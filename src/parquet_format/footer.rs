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
/// Returns an error when the file cannot be opened or its footer cannot be
/// parsed — callers decide whether an unknown projection is fatal.
pub fn read_footer_stats(file_path: &str) -> Result<ParquetFooterStats> {
    let file = std::fs::File::open(file_path)
        .with_context(|| format!("Failed to open Parquet file for footer read: {file_path}"))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("Failed to read Parquet footer: {file_path}"))?;
    let metadata = builder.metadata();

    let num_rows = u64::try_from(metadata.file_metadata().num_rows()).unwrap_or(0);
    let uncompressed_bytes = metadata
        .row_groups()
        .iter()
        .map(|rg| u64::try_from(rg.total_byte_size()).unwrap_or(0))
        .fold(0u64, u64::saturating_add);

    // Leaf value count of the `errors` list column: the exact number of f32s
    // the decode will materialise, however tightly they were encoded.
    let error_values = metadata
        .row_groups()
        .iter()
        .flat_map(parquet::file::metadata::RowGroupMetaData::columns)
        .filter(|col| {
            col.column_path()
                .parts()
                .first()
                .is_some_and(|p| p == ERRORS_COLUMN)
        })
        .map(|col| u64::try_from(col.num_values()).unwrap_or(0))
        .fold(0u64, u64::saturating_add);

    Ok(ParquetFooterStats {
        num_rows,
        uncompressed_bytes,
        error_values,
    })
}
