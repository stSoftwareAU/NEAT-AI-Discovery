//! Integration tests for Parquet column pruning (Issue #1073).
//!
//! Verifies that column projection profiles correctly skip the errors
//! column while preserving all other record fields.

use neat_ai_discovery::parquet_format::{
    ColumnProfile, read_all_records_grouped_by_neuron_with_profile,
    read_records_from_parquet_with_limit_and_profile, read_records_from_parquet_with_profile,
    write_records_to_parquet,
};
use neat_ai_discovery::types::DiscoverRecord;
use tempfile::NamedTempFile;

/// Helper: create test records with known errors values.
fn create_test_records() -> Vec<DiscoverRecord> {
    vec![
        DiscoverRecord::new(0, "neuron-a".to_string(), Some(0.5), 0.7, vec![0.1, 0.2]),
        DiscoverRecord::new(1, "neuron-a".to_string(), Some(0.6), 0.8, vec![0.15, 0.25]),
        DiscoverRecord::new(0, "neuron-b".to_string(), Some(0.3), 0.5, vec![0.05]),
        DiscoverRecord::new(1, "neuron-b".to_string(), None, 0.6, vec![0.1]),
    ]
}

/// Helper: write test records to a temp parquet file and return the path guard + path string.
fn write_temp_parquet(records: &[DiscoverRecord]) -> (NamedTempFile, String) {
    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    let path = temp_file.path().to_str().unwrap().to_string();
    write_records_to_parquet(&path, records).expect("Failed to write parquet");
    (temp_file, path)
}

#[test]
fn test_full_profile_returns_errors() {
    let records = create_test_records();
    let (_tmp, path) = write_temp_parquet(&records);

    let result =
        read_records_from_parquet_with_profile(&path, "neuron-a", ColumnProfile::Full).unwrap();

    assert_eq!(result.len(), 2);
    assert!(
        !result[0].errors.is_empty(),
        "Full profile should include errors"
    );
    assert_eq!(result[0].errors.len(), 2);
}

#[test]
fn test_without_errors_profile_returns_empty_errors() {
    let records = create_test_records();
    let (_tmp, path) = write_temp_parquet(&records);

    let result =
        read_records_from_parquet_with_profile(&path, "neuron-a", ColumnProfile::WithoutErrors)
            .unwrap();

    assert_eq!(result.len(), 2);
    assert!(
        result[0].errors.is_empty(),
        "WithoutErrors profile should return empty errors vec"
    );
    assert!(result[1].errors.is_empty());
}

#[test]
fn test_without_errors_preserves_other_fields() {
    let records = create_test_records();
    let (_tmp, path) = write_temp_parquet(&records);

    let result =
        read_records_from_parquet_with_profile(&path, "neuron-a", ColumnProfile::WithoutErrors)
            .unwrap();

    assert_eq!(result.len(), 2);

    // Verify obs_index, value, and activation are preserved
    let r0 = &result[0];
    assert_eq!(r0.obs_index, 0);
    assert_eq!(r0.value, Some(0.5));
    assert!((r0.activation - 0.7).abs() < f32::EPSILON);

    let r1 = &result[1];
    assert_eq!(r1.obs_index, 1);
    assert_eq!(r1.value, Some(0.6));
    assert!((r1.activation - 0.8).abs() < f32::EPSILON);
}

#[test]
fn test_without_errors_preserves_nullable_value() {
    let records = create_test_records();
    let (_tmp, path) = write_temp_parquet(&records);

    let result =
        read_records_from_parquet_with_profile(&path, "neuron-b", ColumnProfile::WithoutErrors)
            .unwrap();

    // neuron-b record at obs_index=1 has value=None
    let null_value_record = result.iter().find(|r| r.obs_index == 1).unwrap();
    assert_eq!(null_value_record.value, None);
}

#[test]
fn test_grouped_read_without_errors() {
    let records = create_test_records();
    let (_tmp, path) = write_temp_parquet(&records);

    let grouped =
        read_all_records_grouped_by_neuron_with_profile(&path, ColumnProfile::WithoutErrors)
            .unwrap();

    assert_eq!(grouped.len(), 2);

    let neuron_a = grouped.get("neuron-a").unwrap();
    assert_eq!(neuron_a.len(), 2);
    assert!(neuron_a[0].errors.is_empty());

    let neuron_b = grouped.get("neuron-b").unwrap();
    assert_eq!(neuron_b.len(), 2);
    assert!(neuron_b[0].errors.is_empty());
}

#[test]
fn test_grouped_read_full_profile_matches_original() {
    let records = create_test_records();
    let (_tmp, path) = write_temp_parquet(&records);

    let grouped =
        read_all_records_grouped_by_neuron_with_profile(&path, ColumnProfile::Full).unwrap();

    let neuron_a = grouped.get("neuron-a").unwrap();
    assert_eq!(neuron_a.len(), 2);
    assert_eq!(neuron_a[0].errors.len(), 2);
}

#[test]
fn test_limit_read_without_errors() {
    let records = create_test_records();
    let (_tmp, path) = write_temp_parquet(&records);

    let result = read_records_from_parquet_with_limit_and_profile(
        &path,
        Some(1),
        ColumnProfile::WithoutErrors,
    )
    .unwrap();

    // With max_obs=1, we get records for only one distinct obs_index
    assert!(!result.is_empty());
    let obs_indices: std::collections::HashSet<u32> = result.iter().map(|r| r.obs_index).collect();
    assert_eq!(obs_indices.len(), 1);
    assert!(result.iter().all(|r| r.errors.is_empty()));
}

#[test]
fn test_column_profile_debug_display() {
    // Verify ColumnProfile derives Debug and the variants are distinct
    let full = format!("{:?}", ColumnProfile::Full);
    let without = format!("{:?}", ColumnProfile::WithoutErrors);
    assert_ne!(full, without);
    assert!(full.contains("Full"));
    assert!(without.contains("WithoutErrors"));
}
