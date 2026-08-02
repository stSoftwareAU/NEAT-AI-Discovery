//! Schema validation must check column types and nullability, not just names
//! (Issue #1901).
//!
//! A discovery Parquet whose `neuron_uuid`, `obs_index` or `activation` column
//! is nullable passes a name-only check, and Arrow's `value()` accessors then
//! return the physical buffer contents for a null slot — `""` for a UUID, `0.0`
//! for an activation. Both flow into analysis indistinguishable from genuine
//! data, so the whole file must be rejected up front.

use arrow::array::{Float32Array, Float64Array, ListArray, RecordBatch, StringArray, UInt32Array};
use arrow::buffer::OffsetBuffer;
use arrow::datatypes::{DataType, Field, Schema};
use neat_ai_discovery::parquet_format::{
    ColumnProfile, read_all_records_from_parquet, read_all_records_grouped_by_neuron,
    read_all_records_grouped_by_neuron_with_profile, read_records_from_parquet,
    write_records_to_parquet,
};
use neat_ai_discovery::types::DiscoverRecord;
use parquet::arrow::ArrowWriter;
use std::sync::Arc;
use tempfile::TempDir;

/// Non-null `errors` column of two empty lists, matching the discovery schema.
fn empty_errors_column(rows: usize) -> Arc<ListArray> {
    let offsets = OffsetBuffer::<i32>::from_lengths(std::iter::repeat_n(0_usize, rows));
    Arc::new(
        ListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, false)),
            offsets,
            Arc::new(Float32Array::from(Vec::<f32>::new())),
            None,
        )
        .expect("errors list array"),
    )
}

/// Write a Parquet file with the discovery column *names* but a caller-supplied
/// schema and columns, mimicking a foreign producer.
fn write_custom_parquet(
    path: &str,
    fields: Vec<Field>,
    columns: Vec<Arc<dyn arrow::array::Array>>,
) {
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema.clone(), columns).expect("record batch");
    let file = std::fs::File::create(path).expect("create parquet");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("arrow writer");
    writer.write(&batch).expect("write batch");
    writer.close().expect("close writer");
}

fn path_in(dir: &TempDir, name: &str) -> String {
    dir.path().join(name).to_string_lossy().into_owned()
}

/// Discovery schema fields, with one field swapped for a hostile variant.
fn fields_with(replacement: Field) -> Vec<Field> {
    let mut fields = vec![
        Field::new("obs_index", DataType::UInt32, false),
        Field::new("neuron_uuid", DataType::Utf8, false),
        Field::new("value", DataType::Float32, true),
        Field::new("activation", DataType::Float32, false),
        Field::new(
            "errors",
            DataType::List(Arc::new(Field::new("item", DataType::Float32, false))),
            false,
        ),
    ];
    let index = fields
        .iter()
        .position(|f| f.name() == replacement.name())
        .expect("replacement must name a discovery column");
    fields[index] = replacement;
    fields
}

fn assert_validator_rejected(err: &anyhow::Error, column: &str) {
    let msg = err.to_string();
    assert!(
        msg.contains("schema mismatch"),
        "error should come from schema validation: {msg}"
    );
    assert!(
        msg.contains(column),
        "error should name the offending column '{column}': {msg}"
    );
    assert!(
        !msg.contains("Failed to cast"),
        "rejection must happen in validation, not in the later downcast: {msg}"
    );
}

#[test]
fn nullable_activation_with_null_row_is_rejected() {
    let dir = TempDir::new().expect("temp dir");
    let path = path_in(&dir, "nullable_activation.parquet");

    write_custom_parquet(
        &path,
        fields_with(Field::new("activation", DataType::Float32, true)),
        vec![
            Arc::new(UInt32Array::from(vec![0, 1])),
            Arc::new(StringArray::from(vec!["neuron-a", "neuron-a"])),
            Arc::new(Float32Array::from(vec![Some(0.5), Some(0.6)])),
            Arc::new(Float32Array::from(vec![Some(0.7), None])),
            empty_errors_column(2),
        ],
    );

    let err = read_all_records_grouped_by_neuron(&path)
        .expect_err("a nullable activation column must be rejected");
    assert_validator_rejected(&err, "activation");
}

#[test]
fn nullable_neuron_uuid_with_null_row_is_rejected() {
    let dir = TempDir::new().expect("temp dir");
    let path = path_in(&dir, "nullable_uuid.parquet");

    write_custom_parquet(
        &path,
        fields_with(Field::new("neuron_uuid", DataType::Utf8, true)),
        vec![
            Arc::new(UInt32Array::from(vec![0, 1])),
            Arc::new(StringArray::from(vec![Some("neuron-a"), None])),
            Arc::new(Float32Array::from(vec![Some(0.5), Some(0.6)])),
            Arc::new(Float32Array::from(vec![0.7, 0.8])),
            empty_errors_column(2),
        ],
    );

    let err = read_all_records_grouped_by_neuron(&path)
        .expect_err("a nullable neuron_uuid column must be rejected");
    assert_validator_rejected(&err, "neuron_uuid");

    // The filtered read walks the same validator.
    let err = read_records_from_parquet(&path, "neuron-a")
        .expect_err("filtered read must reject the same file");
    assert_validator_rejected(&err, "neuron_uuid");
}

#[test]
fn nullable_obs_index_is_rejected() {
    let dir = TempDir::new().expect("temp dir");
    let path = path_in(&dir, "nullable_obs_index.parquet");

    write_custom_parquet(
        &path,
        fields_with(Field::new("obs_index", DataType::UInt32, true)),
        vec![
            Arc::new(UInt32Array::from(vec![Some(0), None])),
            Arc::new(StringArray::from(vec!["neuron-a", "neuron-a"])),
            Arc::new(Float32Array::from(vec![Some(0.5), Some(0.6)])),
            Arc::new(Float32Array::from(vec![0.7, 0.8])),
            empty_errors_column(2),
        ],
    );

    let err = read_all_records_from_parquet(&path)
        .expect_err("a nullable obs_index column must be rejected");
    assert_validator_rejected(&err, "obs_index");
}

#[test]
fn wrong_data_type_is_rejected_by_validation_not_the_downcast() {
    let dir = TempDir::new().expect("temp dir");
    let path = path_in(&dir, "float64_activation.parquet");

    write_custom_parquet(
        &path,
        fields_with(Field::new("activation", DataType::Float64, false)),
        vec![
            Arc::new(UInt32Array::from(vec![0, 1])),
            Arc::new(StringArray::from(vec!["neuron-a", "neuron-a"])),
            Arc::new(Float32Array::from(vec![Some(0.5), Some(0.6)])),
            Arc::new(Float64Array::from(vec![0.7_f64, 0.8_f64])),
            empty_errors_column(2),
        ],
    );

    let err = read_all_records_grouped_by_neuron(&path)
        .expect_err("a Float64 activation column must be rejected");
    assert_validator_rejected(&err, "activation");
}

#[test]
fn nullable_column_is_rejected_under_the_without_errors_profile() {
    let dir = TempDir::new().expect("temp dir");
    let path = path_in(&dir, "nullable_activation_projected.parquet");

    write_custom_parquet(
        &path,
        fields_with(Field::new("activation", DataType::Float32, true)),
        vec![
            Arc::new(UInt32Array::from(vec![0])),
            Arc::new(StringArray::from(vec!["neuron-a"])),
            Arc::new(Float32Array::from(vec![Some(0.5)])),
            Arc::new(Float32Array::from(vec![None::<f32>])),
            empty_errors_column(1),
        ],
    );

    let err = read_all_records_grouped_by_neuron_with_profile(&path, ColumnProfile::WithoutErrors)
        .expect_err("projection must not skip nullability validation");
    assert_validator_rejected(&err, "activation");
}

#[test]
fn nullable_errors_elements_are_rejected() {
    let dir = TempDir::new().expect("temp dir");
    let path = path_in(&dir, "nullable_error_elements.parquet");

    let offsets = OffsetBuffer::<i32>::from_lengths([2_usize]);
    let errors = Arc::new(
        ListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            offsets,
            Arc::new(Float32Array::from(vec![Some(0.1), None])),
            None,
        )
        .expect("nullable errors list"),
    );

    write_custom_parquet(
        &path,
        fields_with(Field::new(
            "errors",
            DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
            false,
        )),
        vec![
            Arc::new(UInt32Array::from(vec![0])),
            Arc::new(StringArray::from(vec!["neuron-a"])),
            Arc::new(Float32Array::from(vec![Some(0.5)])),
            Arc::new(Float32Array::from(vec![0.7])),
            errors,
        ],
    );

    let err = read_all_records_grouped_by_neuron(&path)
        .expect_err("null error values must not decode as 0.0");
    assert_validator_rejected(&err, "errors");
}

#[test]
fn writer_output_still_reads_back_unchanged() {
    let dir = TempDir::new().expect("temp dir");
    let path = path_in(&dir, "valid.parquet");

    let records = vec![
        DiscoverRecord::new(0, "neuron-a".to_string(), Some(0.5), 0.7, vec![0.1, 0.2]),
        // A null `value` is legitimate — the column is declared nullable.
        DiscoverRecord::new(1, "neuron-a".to_string(), None, 0.8, vec![]),
        DiscoverRecord::new(0, "neuron-b".to_string(), Some(0.3), -0.5, vec![0.4]),
    ];
    write_records_to_parquet(&path, &records).expect("write valid parquet");

    let grouped = read_all_records_grouped_by_neuron(&path).expect("valid file must still read");
    assert_eq!(grouped.len(), 2, "both neurons must be grouped");
    assert_eq!(grouped["neuron-a"].len(), 2);
    assert_eq!(
        grouped["neuron-a"][1].value, None,
        "null value is preserved"
    );
    assert_eq!(grouped["neuron-b"][0].activation, -0.5);

    let projected =
        read_all_records_grouped_by_neuron_with_profile(&path, ColumnProfile::WithoutErrors)
            .expect("projected read must still succeed");
    assert_eq!(projected.len(), 2);

    let filtered = read_records_from_parquet(&path, "neuron-a").expect("filtered read");
    assert_eq!(filtered.len(), 2);

    let all = read_all_records_from_parquet(&path).expect("unfiltered read");
    assert_eq!(all.len(), 3);
}
