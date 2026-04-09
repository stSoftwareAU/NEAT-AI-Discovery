//! Tests for cache behaviour when a parquet file is deleted after cache construction (Issue #1049).
//!
//! Verifies that lazy cache reads handle missing parquet files gracefully,
//! returning clear errors rather than panicking, and that bulk load methods
//! return partial results when the file disappears mid-analysis.

use neat_ai_discovery::analysis::cache::{CompressedLruRecordCache, LruRecordCache};
use neat_ai_discovery::parquet_format::{read_records_from_parquet, write_records_to_parquet};
use neat_ai_discovery::types::DiscoverRecord;
use std::sync::Arc;
use tempfile::NamedTempFile;

/// Helper: create a temporary parquet file with test records and return the path.
fn create_test_parquet() -> (NamedTempFile, String) {
    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    let file_path = temp_file.path().to_str().unwrap().to_string();

    let records = vec![
        DiscoverRecord::new(0, "neuron-a".to_string(), Some(0.5), 0.7, vec![0.1, 0.2]),
        DiscoverRecord::new(1, "neuron-a".to_string(), Some(0.6), 0.8, vec![0.15, 0.25]),
        DiscoverRecord::new(0, "neuron-b".to_string(), Some(0.3), 0.5, vec![0.05]),
        DiscoverRecord::new(1, "neuron-b".to_string(), Some(0.4), 0.6, vec![0.1]),
    ];

    write_records_to_parquet(&file_path, &records).expect("Failed to write test parquet");
    (temp_file, file_path)
}

#[test]
fn lru_cache_handles_missing_parquet_without_panic() {
    let (temp_file, file_path) = create_test_parquet();

    // Create cache while file exists
    let cache =
        LruRecordCache::new(&file_path, 10 * 1024 * 1024).expect("Cache creation should succeed");

    // Verify cache works while file exists
    let records = cache.get("neuron-a").expect("Should load neuron-a");
    assert_eq!(records.len(), 2);

    // Delete the file
    drop(temp_file);
    std::fs::remove_file(&file_path).ok(); // Ensure path is gone

    // Cache miss for a neuron not yet cached should return an error, not panic
    let result = cache.get("neuron-b");
    assert!(
        result.is_err(),
        "Should return error when parquet file is missing"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("parquet file removed") || err_msg.contains("Parquet file removed"),
        "Error should indicate file was removed, got: {err_msg}"
    );
}

#[test]
fn compressed_cache_handles_missing_parquet_without_panic() {
    let (temp_file, file_path) = create_test_parquet();

    // Create cache while file exists
    let cache = CompressedLruRecordCache::new(&file_path, 10 * 1024 * 1024)
        .expect("Cache creation should succeed");

    // Verify cache works while file exists
    let records = cache.get("neuron-a").expect("Should load neuron-a");
    assert_eq!(records.len(), 2);

    // Delete the file
    drop(temp_file);
    std::fs::remove_file(&file_path).ok();

    // Cache miss for uncached neuron should return an error, not panic
    let result = cache.get("neuron-b");
    assert!(
        result.is_err(),
        "Should return error when parquet file is missing"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("parquet file removed") || err_msg.contains("Parquet file removed"),
        "Error should indicate file was removed, got: {err_msg}"
    );
}

#[test]
fn record_cache_lazy_handles_missing_parquet_without_panic() {
    use neat_ai_discovery::analysis::cache::RecordCache;

    let (temp_file, file_path) = create_test_parquet();

    // Create a lazy RecordCache using the custom loader path
    // Use with_loader to simulate lazy loading behaviour
    let cache = RecordCache::with_loader(
        &file_path,
        Arc::new(move |file: &str, neuron_uuid: &str| read_records_from_parquet(file, neuron_uuid)),
    );

    // Load neuron-a while file exists
    let records = cache.get("neuron-a").expect("Should load neuron-a");
    assert_eq!(records.len(), 2);

    // Delete the file
    drop(temp_file);
    std::fs::remove_file(&file_path).ok();

    // Cache miss for uncached neuron should return an error, not panic
    let result = cache.get("neuron-b");
    assert!(
        result.is_err(),
        "Should return error when parquet file is missing"
    );
}

#[test]
fn read_records_from_parquet_returns_file_removed_error() {
    let (temp_file, file_path) = create_test_parquet();

    // Verify reads work while file exists
    let records = read_records_from_parquet(&file_path, "neuron-a").expect("Should read neuron-a");
    assert_eq!(records.len(), 2);

    // Delete the file
    drop(temp_file);
    std::fs::remove_file(&file_path).ok();

    // Reading should return a clear error, not panic
    let result = read_records_from_parquet(&file_path, "neuron-a");
    assert!(result.is_err(), "Should return error for missing file");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("parquet file removed")
            || err_msg.contains("Parquet file removed")
            || err_msg.contains("no longer exists"),
        "Error should indicate file was removed, got: {err_msg}"
    );
}

#[test]
fn lru_cache_returns_cached_data_after_file_deletion() {
    let (temp_file, file_path) = create_test_parquet();

    let cache =
        LruRecordCache::new(&file_path, 10 * 1024 * 1024).expect("Cache creation should succeed");

    // Load both neurons into cache while file exists
    let records_a = cache.get("neuron-a").expect("Should load neuron-a");
    let records_b = cache.get("neuron-b").expect("Should load neuron-b");
    assert_eq!(records_a.len(), 2);
    assert_eq!(records_b.len(), 2);

    // Delete the file
    drop(temp_file);
    std::fs::remove_file(&file_path).ok();

    // Cached data should still be accessible (cache hits don't need the file)
    let records_a_again = cache
        .get("neuron-a")
        .expect("Cached neuron-a should still work");
    assert_eq!(records_a_again.len(), 2);
    let records_b_again = cache
        .get("neuron-b")
        .expect("Cached neuron-b should still work");
    assert_eq!(records_b_again.len(), 2);
}
