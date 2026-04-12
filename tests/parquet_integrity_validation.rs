//! Integration tests for Parquet file integrity validation (Issue #1085).
//!
//! Verifies that corrupted, truncated, and schema-mismatched Parquet files
//! produce clear errors rather than panics, and that `.parquet.tmp` files
//! are detected and warned about.

use neat_ai_discovery::parquet_format::{
    read_all_records_from_parquet, read_all_records_grouped_by_neuron, read_records_from_parquet,
    write_records_to_parquet,
};
use neat_ai_discovery::types::DiscoverRecord;
use std::io::Write;
use tempfile::{NamedTempFile, TempDir};

/// Helper: create a minimal set of valid test records.
fn create_test_records() -> Vec<DiscoverRecord> {
    vec![
        DiscoverRecord::new(0, "neuron-a".to_string(), Some(0.5), 0.7, vec![0.1, 0.2]),
        DiscoverRecord::new(1, "neuron-a".to_string(), Some(0.6), 0.8, vec![0.15]),
    ]
}

/// Helper: write test records to a temp parquet file and return the path guard + path string.
fn write_temp_parquet(records: &[DiscoverRecord]) -> (NamedTempFile, String) {
    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    let path = temp_file.path().to_str().unwrap().to_string();
    write_records_to_parquet(&path, records).expect("Failed to write parquet");
    (temp_file, path)
}

// ============================================================================
// Truncated / corrupted file tests
// ============================================================================

#[test]
fn test_truncated_parquet_file_returns_error_not_panic() {
    // Write a valid parquet file first, then truncate it
    let records = create_test_records();
    let (_tmp, path) = write_temp_parquet(&records);

    // Read the valid file to get its bytes, then write a truncated version
    let full_bytes = std::fs::read(&path).expect("Failed to read parquet file");
    assert!(
        full_bytes.len() > 20,
        "Valid parquet file should be larger than 20 bytes"
    );

    // Truncate to half the original size — this destroys the footer
    let truncated = &full_bytes[..full_bytes.len() / 2];
    std::fs::write(&path, truncated).expect("Failed to write truncated file");

    // Reading the truncated file should return an error, not panic
    let result = read_all_records_grouped_by_neuron(&path);
    assert!(
        result.is_err(),
        "Truncated parquet file should produce an error"
    );
    let err_msg = result.unwrap_err().to_string().to_lowercase();
    assert!(
        err_msg.contains("parquet") || err_msg.contains("corrupt") || err_msg.contains("footer"),
        "Error should mention parquet/corruption/footer: {err_msg}"
    );
}

#[test]
fn test_empty_file_returns_error_not_panic() {
    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    let path = temp_file.path().to_str().unwrap();

    // Write an empty file (0 bytes)
    std::fs::write(path, b"").expect("Failed to write empty file");

    let result = read_all_records_grouped_by_neuron(path);
    assert!(result.is_err(), "Empty file should produce an error");
}

#[test]
fn test_random_bytes_file_returns_error_not_panic() {
    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    let path = temp_file.path().to_str().unwrap();

    // Write random (non-parquet) bytes
    let garbage: Vec<u8> = (0..=255u8)
        .map(|i| i.wrapping_mul(37).wrapping_add(13))
        .collect();
    std::fs::write(path, &garbage).expect("Failed to write garbage file");

    let result = read_records_from_parquet(path, "neuron-a");
    assert!(
        result.is_err(),
        "Random bytes file should produce an error, not panic"
    );
}

#[test]
fn test_truncated_file_with_read_all_returns_error() {
    let records = create_test_records();
    let (_tmp, path) = write_temp_parquet(&records);

    // Truncate to just 8 bytes — smaller than the Parquet magic number
    std::fs::write(&path, [0u8; 8]).expect("Failed to write truncated file");

    let result = read_all_records_from_parquet(&path);
    assert!(
        result.is_err(),
        "Truncated file should produce an error via read_all_records_from_parquet"
    );
}

// ============================================================================
// Schema mismatch tests
// ============================================================================

#[test]
fn test_schema_mismatch_returns_clear_error() {
    // Create a Parquet file with a different schema (not the discovery schema)
    use arrow::array::{Int64Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    let path = temp_file.path().to_str().unwrap();

    // Write a parquet file with a completely different schema
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]));

    let ids = Int64Array::from(vec![1, 2, 3]);
    let names = arrow::array::StringArray::from(vec!["a", "b", "c"]);
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(names)]).unwrap();

    let file = std::fs::File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    // Attempt to read this as a discovery Parquet file — should get a schema error
    let result = read_all_records_grouped_by_neuron(path);
    assert!(
        result.is_err(),
        "Schema-mismatched file should produce an error"
    );
    let err_msg = result.unwrap_err().to_string().to_lowercase();
    assert!(
        err_msg.contains("schema") || err_msg.contains("missing") || err_msg.contains("column"),
        "Error should mention schema mismatch: {err_msg}"
    );
}

#[test]
fn test_schema_mismatch_with_filtered_read_returns_clear_error() {
    use arrow::array::{Float64Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    let path = temp_file.path().to_str().unwrap();

    // Write a parquet file that has obs_index but with wrong type (Float64 instead of UInt32)
    let schema = Arc::new(Schema::new(vec![
        Field::new("obs_index", DataType::Float64, false),
        Field::new("neuron_uuid", DataType::Utf8, false),
        Field::new("value", DataType::Float64, false),
        Field::new("activation", DataType::Float64, false),
    ]));

    let obs = Float64Array::from(vec![1.0, 2.0]);
    let uuids = arrow::array::StringArray::from(vec!["a", "b"]);
    let values = Float64Array::from(vec![0.5, 0.6]);
    let activations = Float64Array::from(vec![0.7, 0.8]);
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(obs),
            Arc::new(uuids),
            Arc::new(values),
            Arc::new(activations),
        ],
    )
    .unwrap();

    let file = std::fs::File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    // Attempt to read — should fail because obs_index is Float64, not UInt32
    let result = read_records_from_parquet(path, "a");
    assert!(
        result.is_err(),
        "Wrong column types should produce an error"
    );
}

// ============================================================================
// .parquet.tmp file detection tests
// ============================================================================

#[test]
fn test_tmp_parquet_file_is_rejected_with_warning() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");

    // Create a .parquet.tmp file (simulates incomplete write)
    let tmp_path = temp_dir.path().join("discovery_data.parquet.tmp");
    let mut file = std::fs::File::create(&tmp_path).expect("Failed to create tmp file");
    file.write_all(b"incomplete parquet data")
        .expect("Failed to write");

    let path_str = tmp_path.to_str().unwrap();

    // Attempting to read a .parquet.tmp file should produce a clear error
    let result = read_all_records_grouped_by_neuron(path_str);
    assert!(result.is_err(), ".parquet.tmp file should produce an error");
    let err_msg = result.unwrap_err().to_string().to_lowercase();
    assert!(
        err_msg.contains("tmp") || err_msg.contains("incomplete"),
        "Error should mention tmp/incomplete file: {err_msg}"
    );
}

#[test]
fn test_tmp_parquet_file_detected_by_all_read_functions() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");

    let tmp_path = temp_dir.path().join("data.parquet.tmp");
    std::fs::write(&tmp_path, b"garbage").expect("Failed to write");

    let path_str = tmp_path.to_str().unwrap();

    // All read functions should reject .parquet.tmp files
    assert!(
        read_all_records_from_parquet(path_str).is_err(),
        "read_all_records_from_parquet should reject .parquet.tmp"
    );
    assert!(
        read_records_from_parquet(path_str, "neuron-a").is_err(),
        "read_records_from_parquet should reject .parquet.tmp"
    );
    assert!(
        read_all_records_grouped_by_neuron(path_str).is_err(),
        "read_all_records_grouped_by_neuron should reject .parquet.tmp"
    );
}

// ============================================================================
// Valid file still works (no regression)
// ============================================================================

#[test]
fn test_valid_parquet_file_still_reads_correctly() {
    let records = create_test_records();
    let (_tmp, path) = write_temp_parquet(&records);

    // Should still read successfully
    let result = read_all_records_grouped_by_neuron(&path);
    assert!(
        result.is_ok(),
        "Valid parquet file should read successfully"
    );
    let grouped = result.unwrap();
    assert_eq!(grouped.len(), 1, "Should have 1 neuron group");
    assert_eq!(
        grouped.get("neuron-a").unwrap().len(),
        2,
        "Should have 2 records for neuron-a"
    );
}
