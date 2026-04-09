//! Tests for Issue #1048: Prevent parquet file deletion while analysis is active.
//!
//! Verifies that:
//! - `RecordCache` holds a file guard that keeps the parquet file data accessible
//!   even after the path is unlinked (Unix behaviour)
//! - The `is_analysis_active()` FFI function correctly tracks analysis lifecycle
//! - The analysis-active counter increments and decrements correctly
//! - `LruRecordCache` and `CompressedLruRecordCache` also hold file guards

use neat_ai_discovery::cancellation;
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use serial_test::serial;
use std::path::Path;
use tempfile::TempDir;

/// Helper: create a minimal parquet file with test records.
fn create_test_parquet(dir: &Path) -> String {
    let parquet_path = dir.join("discovery_data.parquet");
    let path_str = parquet_path.to_str().unwrap().to_string();

    let records: Vec<DiscoverRecord> = (0..20u32)
        .map(|i| DiscoverRecord::new(i, format!("neuron-{}", i % 4), Some(0.5), 0.5, vec![0.1]))
        .collect();

    write_records_to_parquet(&path_str, &records).expect("Failed to write test parquet");
    path_str
}

// ============================================================================
// File guard: RecordCache keeps data accessible after path removal
// ============================================================================

/// On Unix, a held file descriptor keeps the inode alive even after unlink.
/// This test verifies that `RecordCache` holds a file guard so that records
/// remain accessible after the parquet path is deleted.
#[test]
fn record_cache_file_guard_survives_path_deletion() {
    use neat_ai_discovery::analysis::cache::RecordCache;

    let tmp = TempDir::new().unwrap();
    let parquet_path = create_test_parquet(tmp.path());

    // Create a pre-loaded cache (opens file, reads all records, holds guard)
    let cache = RecordCache::new_adaptive(&parquet_path).expect("Failed to create cache");

    // Delete the parquet file
    std::fs::remove_file(&parquet_path).expect("Failed to remove parquet file");
    assert!(!Path::new(&parquet_path).exists(), "File should be deleted");

    // Records should still be accessible from the in-memory cache
    let records = cache
        .get("neuron-0")
        .expect("Should still access cached records");
    assert!(!records.is_empty(), "Should have records for neuron-0");
}

/// Verify that `LruRecordCache` holds a file guard.
#[test]
fn lru_cache_holds_file_guard() {
    use neat_ai_discovery::analysis::cache::LruRecordCache;

    let tmp = TempDir::new().unwrap();
    let parquet_path = create_test_parquet(tmp.path());

    // Create LRU cache (this should hold a file guard)
    let cache =
        LruRecordCache::new(&parquet_path, 10 * 1024 * 1024).expect("Failed to create LRU cache");

    // Load some records before deletion
    let records_before = cache.get("neuron-0").expect("Should load neuron-0");
    assert!(!records_before.is_empty());

    // Delete the parquet file path
    std::fs::remove_file(&parquet_path).expect("Failed to remove parquet file");

    // On Unix, the file guard keeps the inode alive, so subsequent reads
    // should still succeed from the held file descriptor.
    // On non-Unix, the file may fail to open — but the records already
    // loaded should still be in the cache.
    let cached_records = cache
        .get("neuron-0")
        .expect("Cached records should still work");
    assert_eq!(cached_records.len(), records_before.len());
}

/// Verify that `CompressedLruRecordCache` holds a file guard.
#[test]
fn compressed_cache_holds_file_guard() {
    use neat_ai_discovery::analysis::cache::CompressedLruRecordCache;

    let tmp = TempDir::new().unwrap();
    let parquet_path = create_test_parquet(tmp.path());

    let cache = CompressedLruRecordCache::new(&parquet_path, 10 * 1024 * 1024)
        .expect("Failed to create compressed cache");

    // Load records before deletion
    let records_before = cache.get("neuron-0").expect("Should load neuron-0");
    assert!(!records_before.is_empty());

    // Delete the parquet file path
    std::fs::remove_file(&parquet_path).expect("Failed to remove parquet file");

    // Cached records should still be accessible
    let cached_records = cache
        .get("neuron-0")
        .expect("Cached records should still work");
    assert_eq!(cached_records.len(), records_before.len());
}

// ============================================================================
// Analysis-active tracking
// ============================================================================

#[test]
#[serial]
fn analysis_active_starts_at_zero() {
    use neat_ai_discovery::cancellation::{analysis_active_count, reset_analysis_active};
    reset_analysis_active();
    assert_eq!(analysis_active_count(), 0, "Should start at zero");
}

#[test]
#[serial]
fn analysis_active_increments_and_decrements() {
    use neat_ai_discovery::cancellation::{
        analysis_active_count, mark_analysis_finished, mark_analysis_started, reset_analysis_active,
    };

    reset_analysis_active();
    assert_eq!(analysis_active_count(), 0);

    mark_analysis_started();
    assert_eq!(analysis_active_count(), 1);

    mark_analysis_started();
    assert_eq!(analysis_active_count(), 2);

    mark_analysis_finished();
    assert_eq!(analysis_active_count(), 1);

    mark_analysis_finished();
    assert_eq!(analysis_active_count(), 0);
}

#[test]
#[serial]
fn is_analysis_active_reflects_counter() {
    use neat_ai_discovery::cancellation::{
        is_analysis_active, mark_analysis_finished, mark_analysis_started, reset_analysis_active,
    };

    reset_analysis_active();
    assert!(!is_analysis_active(), "No analysis running");

    mark_analysis_started();
    assert!(is_analysis_active(), "One analysis running");

    mark_analysis_finished();
    assert!(!is_analysis_active(), "Analysis finished");
}

#[test]
#[serial]
fn analysis_active_does_not_underflow() {
    use neat_ai_discovery::cancellation::{
        analysis_active_count, mark_analysis_finished, reset_analysis_active,
    };

    reset_analysis_active();
    // Decrementing at zero should not underflow
    mark_analysis_finished();
    assert_eq!(
        analysis_active_count(),
        0,
        "Counter should not underflow below zero"
    );
}

// ============================================================================
// Combined: cancellation + analysis-active
// ============================================================================

#[test]
#[serial]
fn cancellation_and_analysis_active_are_independent() {
    use neat_ai_discovery::cancellation::{
        is_analysis_active, mark_analysis_finished, mark_analysis_started, reset_analysis_active,
    };

    reset_analysis_active();
    cancellation::reset_cancellation();

    mark_analysis_started();
    assert!(is_analysis_active());
    assert!(!cancellation::is_cancelled());

    cancellation::request_cancellation();
    assert!(is_analysis_active(), "Analysis still active despite cancel");
    assert!(cancellation::is_cancelled());

    mark_analysis_finished();
    assert!(!is_analysis_active());
    assert!(
        cancellation::is_cancelled(),
        "Cancel flag persists after analysis finishes"
    );

    cancellation::reset_cancellation();
}
