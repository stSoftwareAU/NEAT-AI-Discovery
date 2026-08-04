//! The one place that knows how a discovery Parquet batch maps onto a
//! [`DiscoverRecord`] (Issue #2005).
//!
//! The schema is declared once in [`crate::parquet_format::create_schema`], but
//! the rule for *reading* it back — resolve `obs_index` / `neuron_uuid` /
//! `value` / `activation` / `errors` by name, downcast each to its Arrow type,
//! then per row read the nullable `value` and decode the `errors` list — used to
//! be copy-pasted into four readers, and the copies had diverged (different
//! error wording, and only three of the four charged the decode budget).
//!
//! Callers keep their own row filtering, grouping and budget policy; this
//! decoder owns only column resolution and per-row decoding. `include_errors`
//! is the single axis the callers vary, mirroring
//! [`crate::parquet_format::ColumnProfile`]'s errors projection (Issue #1073).
//!
//! Everything that can be hoisted out of the per-row path is (Issue #2007): the
//! `errors` list is flattened to its offsets and its child values slice once per
//! batch, so a row copies a sub-slice instead of allocating a fresh Arrow array
//! and reading it back one element at a time.

use anyhow::{Context, Result};
use arrow::array::{Array, Float32Array, ListArray, RecordBatch, StringArray, UInt32Array};

use crate::types::DiscoverRecord;

/// The `errors` list column, flattened once per batch (Issue #2007).
///
/// `ListArray::value(row)` builds a fresh `ArrayRef` for every row — a heap
/// allocation and an atomic refcount just to read a handful of floats, which is
/// then re-read element by element. Holding the offsets and the child values
/// slice instead makes a row's errors a plain sub-slice.
#[derive(Debug)]
struct ErrorLists<'a> {
    /// Row `i`'s values span `offsets[i]..offsets[i + 1]` of `values`.
    offsets: &'a [i32],
    values: &'a [f32],
}

impl<'a> ErrorLists<'a> {
    /// Flatten the list column into its offsets and child values.
    fn resolve(errors: &'a ListArray) -> Result<Self> {
        let values = errors
            .values()
            .as_any()
            .downcast_ref::<Float32Array>()
            .context("Failed to cast errors array")?;
        Ok(Self {
            offsets: errors.value_offsets(),
            values: values.values(),
        })
    }

    /// The error values at `row`.
    ///
    /// A malformed file whose offsets run past the child array is a recoverable
    /// error rather than a slice panic — the reader must fail loud, not abort
    /// the process.
    fn row(&self, row: usize) -> Result<&'a [f32]> {
        let start = self.offset_at(row)?;
        let end = self.offset_at(row + 1)?;
        self.values.get(start..end).with_context(|| {
            format!(
                "errors list at row {row} spans {start}..{end}, past the {} decoded error values",
                self.values.len()
            )
        })
    }

    /// One offset, rejected if it is absent or negative.
    fn offset_at(&self, index: usize) -> Result<usize> {
        let offset = *self
            .offsets
            .get(index)
            .with_context(|| format!("errors list column has no offset for row {index}"))?;
        usize::try_from(offset)
            .with_context(|| format!("errors list offset {offset} at row {index} is negative"))
    }
}

/// Resolve one column by name and downcast it to its Arrow array type.
fn typed_column<'a, T: 'static>(batch: &'a RecordBatch, name: &str) -> Result<&'a T> {
    let index = batch
        .schema_ref()
        .index_of(name)
        .with_context(|| format!("Parquet schema mismatch: missing '{name}' column"))?;
    batch
        .column(index)
        .as_any()
        .downcast_ref::<T>()
        .with_context(|| format!("Failed to cast {name} column"))
}

/// Typed references to a discovery batch's columns, resolved by name once per
/// batch rather than once per row.
#[derive(Debug)]
pub(crate) struct DiscoveryBatchColumns<'a> {
    obs_index: &'a UInt32Array,
    neuron_uuid: &'a StringArray,
    value: &'a Float32Array,
    activation: &'a Float32Array,
    /// `None` when the caller projected the `errors` column away — decoded rows
    /// then carry an empty `errors` vec.
    errors: Option<ErrorLists<'a>>,
}

impl<'a> DiscoveryBatchColumns<'a> {
    /// Resolve every discovery column in `batch` by name.
    ///
    /// Columns are resolved by name rather than by position because projection
    /// shifts indices, and because `RecordBatch::column(index)` panics on an
    /// out-of-bounds index — a short or mismatched schema must return a
    /// recoverable error instead of crashing the process (Issue #1482).
    pub(crate) fn resolve(batch: &'a RecordBatch, include_errors: bool) -> Result<Self> {
        Ok(Self {
            obs_index: typed_column(batch, "obs_index")?,
            neuron_uuid: typed_column(batch, "neuron_uuid")?,
            value: typed_column(batch, "value")?,
            activation: typed_column(batch, "activation")?,
            errors: if include_errors {
                Some(ErrorLists::resolve(typed_column::<ListArray>(
                    batch, "errors",
                )?)?)
            } else {
                None
            },
        })
    }

    /// Number of rows in the batch.
    pub(crate) fn num_rows(&self) -> usize {
        self.obs_index.len()
    }

    /// The observation index at `row`, for callers that filter before decoding.
    pub(crate) fn obs_index(&self, row: usize) -> u32 {
        self.obs_index.value(row)
    }

    /// The neuron UUID at `row`, for callers that filter before decoding.
    pub(crate) fn neuron_uuid(&self, row: usize) -> &str {
        self.neuron_uuid.value(row)
    }

    /// Decode one row into a [`DiscoverRecord`].
    pub(crate) fn decode_row(&self, row: usize) -> Result<DiscoverRecord> {
        let value = if self.value.is_null(row) {
            None
        } else {
            Some(self.value.value(row))
        };

        let errors = match &self.errors {
            Some(lists) => lists.row(row)?.to_vec(),
            None => Vec::new(),
        };

        Ok(DiscoverRecord::new(
            self.obs_index.value(row),
            self.neuron_uuid.value(row).to_string(),
            value,
            self.activation.value(row),
            errors,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::DiscoveryBatchColumns;
    use crate::parquet_format::create_schema;
    use arrow::array::{
        ArrayRef, Float32Array, Float32Builder, ListBuilder, RecordBatch, StringArray, UInt32Array,
    };
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    /// Three rows: a full row, a row with a null `value` and an empty `errors`
    /// list, and a row with a longer list.
    fn sample_batch() -> RecordBatch {
        let obs_index = UInt32Array::from(vec![0u32, 1, 2]);
        let neuron_uuid = StringArray::from(vec!["neuron-a", "neuron-b", "neuron-a"]);
        let value = Float32Array::from(vec![Some(0.5f32), None, Some(-1.5)]);
        let activation = Float32Array::from(vec![0.75f32, -0.25, 0.0]);

        // The discovery schema declares non-null list items, and
        // `RecordBatch::try_new` checks that exactly.
        let mut errors = ListBuilder::new(Float32Builder::new()).with_field(Arc::new(Field::new(
            "item",
            DataType::Float32,
            false,
        )));
        errors.values().append_slice(&[0.1, 0.2]);
        errors.append(true);
        errors.append(true); // empty list
        errors.values().append_slice(&[0.3, 0.4, 0.5]);
        errors.append(true);

        RecordBatch::try_new(
            Arc::new(create_schema()),
            vec![
                Arc::new(obs_index) as ArrayRef,
                Arc::new(neuron_uuid) as ArrayRef,
                Arc::new(value) as ArrayRef,
                Arc::new(activation) as ArrayRef,
                Arc::new(errors.finish()) as ArrayRef,
            ],
        )
        .expect("batch matches the discovery schema")
    }

    #[test]
    fn decodes_every_column_including_nullable_value_and_error_lists() {
        let batch = sample_batch();
        let columns = DiscoveryBatchColumns::resolve(&batch, true).expect("columns resolve");

        assert_eq!(columns.num_rows(), 3);
        assert_eq!(columns.obs_index(1), 1);
        assert_eq!(columns.neuron_uuid(1), "neuron-b");

        let first = columns.decode_row(0).expect("row 0 decodes");
        assert_eq!(first.obs_index, 0);
        assert_eq!(first.neuron_uuid, "neuron-a");
        assert_eq!(first.value, Some(0.5));
        assert!((first.activation - 0.75).abs() < f32::EPSILON);
        assert_eq!(first.errors, vec![0.1, 0.2]);

        let second = columns.decode_row(1).expect("row 1 decodes");
        assert_eq!(second.value, None, "a null value decodes as None");
        assert!(second.errors.is_empty(), "an empty list decodes as empty");

        let third = columns.decode_row(2).expect("row 2 decodes");
        assert_eq!(third.errors, vec![0.3, 0.4, 0.5]);
    }

    #[test]
    fn skipping_errors_yields_empty_error_lists() {
        let batch = sample_batch();
        let columns =
            DiscoveryBatchColumns::resolve(&batch, false).expect("columns resolve without errors");

        for row in 0..columns.num_rows() {
            let record = columns.decode_row(row).expect("row decodes");
            assert!(
                record.errors.is_empty(),
                "a projected-away errors column decodes as an empty vec"
            );
        }
        assert_eq!(
            columns.decode_row(1).expect("row 1 decodes").value,
            None,
            "nullability is independent of the errors projection"
        );
    }

    /// Issue #2007: rows are decoded from the list column's offsets rather than
    /// from a freshly allocated per-row array, so a *sliced* batch — whose
    /// offsets no longer start at zero — must still read each row's own values.
    #[test]
    fn a_sliced_batch_decodes_each_row_from_its_own_offsets() {
        let batch = sample_batch().slice(1, 2);
        let columns = DiscoveryBatchColumns::resolve(&batch, true).expect("columns resolve");

        assert_eq!(columns.num_rows(), 2, "the slice covers rows 1 and 2");

        let first = columns.decode_row(0).expect("row decodes");
        assert_eq!(first.neuron_uuid, "neuron-b");
        assert!(first.errors.is_empty(), "row 1's list is empty");

        let second = columns.decode_row(1).expect("row decodes");
        assert_eq!(second.neuron_uuid, "neuron-a");
        assert_eq!(second.errors, vec![0.3, 0.4, 0.5], "row 2's own values");
    }

    #[test]
    fn a_missing_column_is_a_recoverable_schema_mismatch() {
        let schema = Schema::new(vec![
            Field::new("neuron_uuid", DataType::Utf8, false),
            Field::new("activation", DataType::Float32, false),
        ]);
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(StringArray::from(vec!["neuron-a"])) as ArrayRef,
                Arc::new(Float32Array::from(vec![0.5f32])) as ArrayRef,
            ],
        )
        .expect("batch builds");

        let err = DiscoveryBatchColumns::resolve(&batch, true)
            .expect_err("obs_index is absent from this batch");
        assert!(
            err.to_string()
                .contains("Parquet schema mismatch: missing 'obs_index' column"),
            "every reader must report the same wording: {err}"
        );
    }

    #[test]
    fn a_wrongly_typed_column_is_a_recoverable_cast_failure() {
        let schema = Schema::new(vec![
            Field::new("obs_index", DataType::Utf8, false),
            Field::new("neuron_uuid", DataType::Utf8, false),
            Field::new("value", DataType::Float32, true),
            Field::new("activation", DataType::Float32, false),
        ]);
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(StringArray::from(vec!["not-a-u32"])) as ArrayRef,
                Arc::new(StringArray::from(vec!["neuron-a"])) as ArrayRef,
                Arc::new(Float32Array::from(vec![Some(0.5f32)])) as ArrayRef,
                Arc::new(Float32Array::from(vec![0.5f32])) as ArrayRef,
            ],
        )
        .expect("batch builds");

        let err = DiscoveryBatchColumns::resolve(&batch, false)
            .expect_err("obs_index is not a UInt32Array");
        assert!(
            err.to_string().contains("Failed to cast obs_index column"),
            "a downcast failure names the column: {err}"
        );
    }
}
