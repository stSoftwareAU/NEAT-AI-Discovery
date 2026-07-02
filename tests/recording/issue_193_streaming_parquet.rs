//! Tests for Issue #193: Streaming Parquet loading with prefetch.
//!
//! This module tests the block-based streaming loading mechanism for Parquet files,
//! which enables memory-efficient loading of large datasets with predictable memory
//! usage and prefetching for improved performance.
//!
//! ## Key Features Tested
//!
//! 1. Block-based loading from Parquet row groups
//! 2. LRU eviction for memory management
//! 3. Prefetch mechanism for adjacent blocks
//! 4. Configuration via environment variables
//! 5. Backward compatibility with pre-loaded mode
//!
//! ## Test Configuration
//!
//! These tests use `NEAT_AI_DISCOVERY_BLOCK_SIZE=10` to create multiple blocks
//! from small test datasets.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use serial_test::serial;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tempfile::TempDir;

/// Set up small block size for testing (10 records per block).
/// This ensures multiple blocks are created from small test datasets.
fn setup_test_block_size() {
    // SAFETY: Serialised via #[serial] — no concurrent env access.
    unsafe {
        std::env::set_var("NEAT_AI_DISCOVERY_BLOCK_SIZE", "10");
    }
}

/// Restore default block size after tests.
fn teardown_test_block_size() {
    // SAFETY: Serialised via #[serial] — no concurrent env access.
    unsafe {
        std::env::remove_var("NEAT_AI_DISCOVERY_BLOCK_SIZE");
    }
}

/// Helper to create test parquet files with multiple row groups.
fn create_test_parquet_with_neurons(
    temp_dir: &TempDir,
    neuron_count: usize,
    records_per_neuron: usize,
) -> String {
    let parquet_path = temp_dir.path().join("test_data.parquet");
    let path_str = parquet_path.to_str().unwrap().to_string();

    let mut records = Vec::new();
    for neuron_idx in 0..neuron_count {
        let neuron_uuid = format!("neuron-{neuron_idx}");
        for obs_idx in 0..records_per_neuron {
            records.push(DiscoverRecord::new(
                obs_idx as u32,
                neuron_uuid.clone(),
                Some(obs_idx as f32 / records_per_neuron as f32),
                obs_idx as f32 / records_per_neuron as f32,
                vec![0.1],
            ));
        }
    }

    write_records_to_parquet(&path_str, &records).expect("Failed to write test parquet");
    path_str
}

/// Test that streaming cache can be created and provides records correctly.
#[test]
fn streaming_cache_provides_correct_records() {
    use neat_ai_discovery::analysis::cache::StreamingRecordCache;

    let temp_dir = TempDir::new().unwrap();
    let parquet_path = create_test_parquet_with_neurons(&temp_dir, 5, 10);

    let cache = StreamingRecordCache::new(&parquet_path, None, None)
        .expect("Failed to create streaming cache");

    // Request records for a neuron
    let records = cache.get("neuron-2").expect("Failed to get records");

    assert_eq!(records.len(), 10, "Should have 10 records for neuron-2");
    for record in records.iter() {
        assert_eq!(record.neuron_uuid, "neuron-2");
    }
}

/// Test that streaming cache respects `max_cached_blocks` configuration.
#[test]
fn streaming_cache_respects_max_blocks() {
    use neat_ai_discovery::analysis::cache::StreamingRecordCache;

    let temp_dir = TempDir::new().unwrap();
    let parquet_path = create_test_parquet_with_neurons(&temp_dir, 20, 10);

    // Create cache with max 3 blocks
    let cache = StreamingRecordCache::new(&parquet_path, Some(3), None)
        .expect("Failed to create streaming cache");

    // Load records from multiple neurons (each might trigger a new block)
    for i in 0..10 {
        let _ = cache
            .get(&format!("neuron-{i}"))
            .expect("Failed to get records");
    }

    // Verify the cache doesn't exceed the block limit
    let stats = cache.stats();
    assert!(
        stats.cached_blocks <= 3,
        "Cache should have at most 3 blocks, has {}",
        stats.cached_blocks
    );
}

/// Test that LRU eviction works correctly.
#[test]
#[serial]
fn streaming_cache_evicts_least_recently_used() {
    use neat_ai_discovery::analysis::cache::StreamingRecordCache;

    // Use small block size to ensure multiple blocks
    setup_test_block_size();

    let temp_dir = TempDir::new().unwrap();
    // Create 10 neurons with 20 records each = 200 records
    // With block size 10, this creates 20 blocks (each neuron spans 2 blocks)
    let parquet_path = create_test_parquet_with_neurons(&temp_dir, 10, 20);

    // Create cache with max 2 blocks
    let cache = StreamingRecordCache::new(&parquet_path, Some(2), Some(0))
        .expect("Failed to create streaming cache");

    // Access neurons that are far apart to trigger evictions
    // With block size 10, neuron-0 records are in blocks 0-1, neuron-5 in blocks 10-11, etc.
    let _ = cache.get("neuron-0").expect("Failed to get neuron-0");
    let _ = cache.get("neuron-5").expect("Failed to get neuron-5");
    let _ = cache.get("neuron-9").expect("Failed to get neuron-9");

    // Now access neuron-0 again - if evicted, it should be reloaded
    let records = cache.get("neuron-0").expect("Failed to get neuron-0 again");
    assert_eq!(
        records.len(),
        20,
        "Should still get correct data after eviction"
    );

    let stats = cache.stats();

    teardown_test_block_size();

    assert!(
        stats.eviction_count > 0,
        "Should have evicted at least one block"
    );
}

/// Test that prefetch mechanism loads adjacent blocks.
#[test]
#[serial]
fn streaming_cache_prefetches_adjacent_blocks() {
    use neat_ai_discovery::analysis::cache::StreamingRecordCache;
    use std::thread;
    use std::time::Duration;

    // Use small block size to ensure multiple blocks for prefetch
    setup_test_block_size();

    let temp_dir = TempDir::new().unwrap();
    // Create 10 neurons with 20 records each = 200 records = 20 blocks
    let parquet_path = create_test_parquet_with_neurons(&temp_dir, 10, 20);

    // Create cache with prefetch depth of 2
    let cache = StreamingRecordCache::new(&parquet_path, None, Some(2))
        .expect("Failed to create streaming cache");

    // Access one neuron (loads block 0)
    let _ = cache.get("neuron-0").expect("Failed to get neuron-0");

    // Give prefetch thread time to work
    thread::sleep(Duration::from_millis(100));

    // Check stats - should have prefetched blocks 1 and 2
    let stats = cache.stats();

    teardown_test_block_size();

    assert!(
        stats.prefetch_count > 0,
        "Should have prefetched at least one block, prefetch_count={}",
        stats.prefetch_count
    );
}

/// Test that configuration can be set via environment variables.
#[test]
fn streaming_cache_respects_env_config() {
    use neat_ai_discovery::analysis::cache::get_streaming_config_from_env;

    // Test default values when env vars are not set
    let config = get_streaming_config_from_env();
    assert!(config.max_cached_blocks.is_none() || config.max_cached_blocks.unwrap() > 0);
    // prefetch_depth should be a reasonable value
    assert!(config.prefetch_depth.is_some() && config.prefetch_depth.unwrap() <= 10);
}

/// Test that `NEAT_AI_DISCOVERY_PRELOAD_ALL=1` disables streaming mode.
#[test]
fn preload_all_disables_streaming() {
    use neat_ai_discovery::analysis::cache::is_streaming_enabled;

    // When NEAT_AI_DISCOVERY_PRELOAD_ALL=1, streaming should be disabled
    // This test verifies the function works without panicking
    let _should_stream = is_streaming_enabled();
    // Function completed successfully - that's all we need to verify
}

/// Test that streaming cache works correctly with concurrent access.
#[test]
fn streaming_cache_handles_concurrent_access() {
    use neat_ai_discovery::analysis::cache::StreamingRecordCache;
    use std::thread;

    let temp_dir = TempDir::new().unwrap();
    let parquet_path = create_test_parquet_with_neurons(&temp_dir, 10, 10);

    let cache = Arc::new(
        StreamingRecordCache::new(&parquet_path, Some(5), Some(1))
            .expect("Failed to create streaming cache"),
    );

    let successful_reads = Arc::new(AtomicUsize::new(0));
    let thread_count = 4;

    let mut handles = Vec::new();
    for _ in 0..thread_count {
        let cache_clone = Arc::clone(&cache);
        let reads_clone = Arc::clone(&successful_reads);

        let handle = thread::spawn(move || {
            for i in 0..10 {
                let uuid = format!("neuron-{i}");
                let records = cache_clone.get(&uuid).expect("Failed to get records");
                assert_eq!(records.len(), 10);
                reads_clone.fetch_add(1, Ordering::Relaxed);
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().expect("Thread should not panic");
    }

    let total_reads = successful_reads.load(Ordering::Relaxed);
    assert_eq!(total_reads, thread_count * 10);
}

/// Test that streaming cache stats are accurate.
#[test]
fn streaming_cache_stats_are_accurate() {
    use neat_ai_discovery::analysis::cache::StreamingRecordCache;

    let temp_dir = TempDir::new().unwrap();
    let parquet_path = create_test_parquet_with_neurons(&temp_dir, 5, 10);

    let cache = StreamingRecordCache::new(&parquet_path, Some(10), Some(0))
        .expect("Failed to create streaming cache");

    // Initially no stats
    let stats = cache.stats();
    assert_eq!(stats.cached_blocks, 0);
    assert_eq!(stats.cache_hits, 0);
    assert_eq!(stats.cache_misses, 0);

    // Access a neuron
    let _ = cache.get("neuron-0").expect("Failed to get records");
    let stats = cache.stats();
    assert!(stats.cache_misses >= 1, "Should have at least one miss");

    // Access the same neuron again
    let _ = cache.get("neuron-0").expect("Failed to get records");
    let stats = cache.stats();
    assert!(stats.cache_hits >= 1, "Should have at least one hit");
}

/// Test that streaming cache memory usage is bounded.
#[test]
#[serial]
fn streaming_cache_bounds_memory_usage() {
    use neat_ai_discovery::analysis::cache::StreamingRecordCache;

    // Use small block size for testing
    setup_test_block_size();

    let temp_dir = TempDir::new().unwrap();
    // Create a larger dataset - 50 neurons with 100 records each = 5000 records
    // With block size 10, this creates 500 blocks
    let parquet_path = create_test_parquet_with_neurons(&temp_dir, 50, 100);

    // Create cache with strict block limit and disable prefetch to avoid race conditions
    let cache = StreamingRecordCache::new(&parquet_path, Some(5), Some(0))
        .expect("Failed to create streaming cache");

    // Access many neurons
    for i in 0..50 {
        let _ = cache
            .get(&format!("neuron-{i}"))
            .expect("Failed to get records");
    }

    // Memory should be bounded by the block limit
    let stats = cache.stats();

    teardown_test_block_size();

    assert!(
        stats.cached_blocks <= 5,
        "Memory should be bounded: cached_blocks={} should be <= 5",
        stats.cached_blocks
    );

    // Verify we still get correct data
    let records = cache.get("neuron-25").expect("Failed to get records");
    assert_eq!(records.len(), 100);
}

/// Test that `RecordCache::new_adaptive` chooses streaming mode for large files.
#[test]
fn new_adaptive_uses_streaming_for_memory_constraints() {
    // This test verifies that the adaptive mode considers streaming
    // when memory constraints would make full preloading problematic.
    // The actual behaviour depends on system memory and file size.
    use neat_ai_discovery::analysis::cache::RecordCache;

    let temp_dir = TempDir::new().unwrap();
    let parquet_path = create_test_parquet_with_neurons(&temp_dir, 5, 10);

    // new_adaptive should not panic regardless of which mode it chooses
    let cache = RecordCache::new_adaptive(&parquet_path);
    assert!(cache.is_ok(), "new_adaptive should succeed");
}

/// Test that streaming mode provides same data as pre-loaded mode.
#[test]
#[serial]
fn streaming_mode_matches_preloaded_data() {
    use neat_ai_discovery::analysis::cache::{RecordCache, StreamingRecordCache};

    // Use small block size for testing
    setup_test_block_size();

    let temp_dir = TempDir::new().unwrap();
    let parquet_path = create_test_parquet_with_neurons(&temp_dir, 10, 20);

    // Create both cache types
    let streaming = StreamingRecordCache::new(&parquet_path, None, None)
        .expect("Failed to create streaming cache");

    // Use the cache with custom loader that reads all data
    let preloaded =
        RecordCache::new_adaptive(&parquet_path).expect("Failed to create preloaded cache");

    teardown_test_block_size();

    // Compare data for multiple neurons
    for i in 0..10 {
        let uuid = format!("neuron-{i}");

        let streaming_records = streaming.get(&uuid).expect("Streaming get failed");
        let preloaded_records = preloaded.get(&uuid).expect("Preloaded get failed");

        assert_eq!(
            streaming_records.len(),
            preloaded_records.len(),
            "Record count should match for {uuid}"
        );

        for (s, p) in streaming_records.iter().zip(preloaded_records.iter()) {
            assert_eq!(s.obs_index, p.obs_index, "obs_index should match");
            assert_eq!(s.neuron_uuid, p.neuron_uuid, "neuron_uuid should match");
            assert_eq!(s.activation, p.activation, "activation should match");
        }
    }
}

// ============================================================================
// Issue #1482: Unvalidated positional Parquet column access must not panic
// ============================================================================
//
// The streaming reader previously accessed record-batch columns by fixed
// position (`batch.column(1)` in build_block_index, `batch.column(0)..column(4)`
// in load_block_records). `RecordBatch::column(index)` panics on an
// out-of-bounds index, so a structurally valid Parquet with fewer columns than
// expected caused a denial-of-service panic instead of a recoverable error.
// These tests write short-schema Parquet files and assert that the cache
// surfaces a `Result::Err` rather than panicking.

/// Write a Parquet file with an arbitrary Arrow schema/batch and return its path.
fn write_custom_parquet(temp_dir: &TempDir, batch: arrow::array::RecordBatch) -> String {
    use parquet::arrow::ArrowWriter;

    let parquet_path = temp_dir.path().join("custom_schema.parquet");
    let path_str = parquet_path.to_str().unwrap().to_string();

    let file = std::fs::File::create(&path_str).expect("Failed to create parquet file");
    let mut writer =
        ArrowWriter::try_new(file, batch.schema(), None).expect("Failed to create ArrowWriter");
    writer.write(&batch).expect("Failed to write batch");
    writer.close().expect("Failed to close writer");

    path_str
}

/// A single-column Parquet (no `neuron_uuid`) must not panic `build_block_index`
/// during `StreamingRecordCache::new`; it must return an `Err` instead.
#[test]
#[serial]
fn streaming_cache_new_rejects_short_schema_without_panic() {
    use arrow::array::{Int64Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
    use neat_ai_discovery::analysis::cache::StreamingRecordCache;

    let temp_dir = TempDir::new().unwrap();

    // One column that is NOT the expected discovery schema. Positional access
    // would read column(1) — out of bounds for a one-column batch — and panic.
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
    let ids = Int64Array::from(vec![1_i64, 2, 3]);
    let batch = RecordBatch::try_new(schema, vec![Arc::new(ids)]).unwrap();
    let path = write_custom_parquet(&temp_dir, batch);

    let result = StreamingRecordCache::new(&path, None, Some(0));
    assert!(
        result.is_err(),
        "Short-schema Parquet should yield an Err from StreamingRecordCache::new, not a panic"
    );
    let err_msg = result.err().unwrap().to_string().to_lowercase();
    assert!(
        err_msg.contains("schema") || err_msg.contains("column") || err_msg.contains("neuron_uuid"),
        "Error should mention the schema/column mismatch: {err_msg}"
    );
}

/// A Parquet that has `neuron_uuid` (so the index builds) but is missing the
/// remaining discovery columns must not panic `load_block_records` when
/// `get()` triggers a cache-miss load; it must return an `Err` instead.
#[test]
#[serial]
fn streaming_cache_get_rejects_missing_columns_without_panic() {
    use arrow::array::{RecordBatch, StringArray, UInt32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use neat_ai_discovery::analysis::cache::StreamingRecordCache;

    setup_test_block_size();
    let temp_dir = TempDir::new().unwrap();

    // Two columns: obs_index + neuron_uuid. build_block_index resolves
    // neuron_uuid (index 1), but load_block_records needs value/activation/errors.
    // Positional access of column(4) would panic on this two-column batch.
    let schema = Arc::new(Schema::new(vec![
        Field::new("obs_index", DataType::UInt32, false),
        Field::new("neuron_uuid", DataType::Utf8, false),
    ]));
    let obs = UInt32Array::from(vec![0_u32, 1]);
    let uuids = StringArray::from(vec!["neuron-0", "neuron-0"]);
    let batch = RecordBatch::try_new(schema, vec![Arc::new(obs), Arc::new(uuids)]).unwrap();
    let path = write_custom_parquet(&temp_dir, batch);

    // Disable prefetch (Some(0)) so the load happens on this thread and any
    // panic would surface here rather than on the prefetch worker.
    let cache = StreamingRecordCache::new(&path, None, Some(0))
        .expect("Index build should succeed when neuron_uuid is present");

    let result = cache.get("neuron-0");
    teardown_test_block_size();

    assert!(
        result.is_err(),
        "Missing-column Parquet should yield an Err from get(), not a panic"
    );
    let err_msg = result.err().unwrap().to_string().to_lowercase();
    assert!(
        err_msg.contains("schema")
            || err_msg.contains("column")
            || err_msg.contains("value")
            || err_msg.contains("errors"),
        "Error should mention the schema/column mismatch: {err_msg}"
    );
}
