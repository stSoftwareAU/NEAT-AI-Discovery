//! Tests for Issue #215: Tiered/streaming Parquet loader for large files.
//!
//! This module tests the tiered loading strategy that automatically selects
//! between different caching modes based on file size and available memory:
//!
//! 1. **PreloadAll**: Load everything into memory (current behaviour for small files)
//! 2. **LruCache**: Keep frequently-accessed neurons in memory, evict others (medium files)
//! 3. **Streaming**: Load on-demand via block-based loading (very large files)
//!
//! ## Key Features Tested
//!
//! - Strategy selection based on file size and memory
//! - LRU cache eviction correctness for neuron-level caching
//! - Memory usage stays within bounds
//! - Correct data returned regardless of strategy
//!
//! ## TDD Approach
//!
//! These tests are written first (failing) and then implementation follows.

use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use std::sync::Arc;
use tempfile::TempDir;

/// Create test parquet file with specified number of neurons and records per neuron.
fn create_test_parquet(
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
                vec![0.1, 0.2],
            ));
        }
    }

    write_records_to_parquet(&path_str, &records).expect("Failed to write test parquet");
    path_str
}

// =============================================================================
// Strategy Selection Tests
// =============================================================================

/// Test that LoadingStrategy can be determined based on file size and available memory.
#[test]
fn loading_strategy_selection_preload_all_for_small_files() {
    use neat_ai_discovery::analysis::cache::{LoadingStrategy, select_loading_strategy};

    // Small file: 10MB file, 8GB available memory
    // Estimated expanded = 10MB * 3 = 30MB, which is < 2GB (available_memory / 4)
    let file_size = 10 * 1024 * 1024; // 10 MB
    let available_memory = 8 * 1024 * 1024 * 1024_u64; // 8 GB

    let strategy = select_loading_strategy(file_size, available_memory);

    assert!(
        matches!(strategy, LoadingStrategy::PreloadAll),
        "Small files should use PreloadAll strategy, got {strategy:?}"
    );
}

/// Test that LruCache strategy is selected for medium-sized files.
#[test]
fn loading_strategy_selection_lru_cache_for_medium_files() {
    use neat_ai_discovery::analysis::cache::{LoadingStrategy, select_loading_strategy};

    // Medium file: 1GB file, 8GB available memory
    // Estimated expanded = 1GB * 3 = 3GB
    // 3GB < 8GB but > 2GB (available_memory / 4), so use LRU cache
    let file_size = 1024 * 1024 * 1024_u64; // 1 GB
    let available_memory = 8 * 1024 * 1024 * 1024_u64; // 8 GB

    let strategy = select_loading_strategy(file_size, available_memory);

    assert!(
        matches!(strategy, LoadingStrategy::LruCache { .. }),
        "Medium files should use LruCache strategy, got {strategy:?}"
    );
}

/// Test that Streaming strategy is selected for very large files.
#[test]
fn loading_strategy_selection_streaming_for_large_files() {
    use neat_ai_discovery::analysis::cache::{LoadingStrategy, select_loading_strategy};

    // Large file: 4GB file, 8GB available memory
    // Estimated expanded = 4GB * 3 = 12GB, which is > 8GB available
    let file_size = 4 * 1024 * 1024 * 1024_u64; // 4 GB
    let available_memory = 8 * 1024 * 1024 * 1024_u64; // 8 GB

    let strategy = select_loading_strategy(file_size, available_memory);

    assert!(
        matches!(strategy, LoadingStrategy::Streaming),
        "Large files should use Streaming strategy, got {strategy:?}"
    );
}

/// Test that LruCache capacity is set appropriately based on available memory.
#[test]
fn lru_cache_capacity_scales_with_memory() {
    use neat_ai_discovery::analysis::cache::{LoadingStrategy, select_loading_strategy};

    // 2GB file with 16GB available memory
    // Estimated expanded = 2GB * 3 = 6GB
    // Small threshold = 16GB / 4 = 4GB
    // 6GB > 4GB but < 16GB, so LruCache is selected
    let file_size = 2 * 1024 * 1024 * 1024_u64; // 2 GB
    let available_memory = 16 * 1024 * 1024 * 1024_u64; // 16 GB

    let strategy = select_loading_strategy(file_size, available_memory);

    if let LoadingStrategy::LruCache { capacity_bytes } = strategy {
        // Capacity should be approximately half of available memory
        let expected_capacity = available_memory as usize / 2;
        let tolerance = expected_capacity / 10; // 10% tolerance
        assert!(
            (capacity_bytes as i64 - expected_capacity as i64).unsigned_abs() < tolerance as u64,
            "LruCache capacity {capacity_bytes} should be approximately {expected_capacity} (±{tolerance})"
        );
    } else {
        panic!("Expected LruCache strategy, got {strategy:?}");
    }
}

// =============================================================================
// LRU Cache Eviction Tests
// =============================================================================

/// Test that LruRecordCache correctly evicts least-recently-used neurons.
#[test]
fn lru_record_cache_evicts_least_recently_used() {
    use neat_ai_discovery::analysis::cache::LruRecordCache;

    let temp_dir = TempDir::new().unwrap();
    // Create 10 neurons with 100 records each
    let parquet_path = create_test_parquet(&temp_dir, 10, 100);

    // Create LRU cache with capacity for approximately 3 neurons worth of data
    // Each neuron has ~100 records, each record is roughly 100 bytes
    let capacity = 3 * 100 * 100; // ~30KB capacity

    let cache = LruRecordCache::new(&parquet_path, capacity).expect("Failed to create LRU cache");

    // Access neurons 0, 1, 2 (fills cache)
    let _ = cache.get("neuron-0").expect("Failed to get neuron-0");
    let _ = cache.get("neuron-1").expect("Failed to get neuron-1");
    let _ = cache.get("neuron-2").expect("Failed to get neuron-2");

    // Access neuron-0 again to make it most recently used
    let _ = cache.get("neuron-0").expect("Failed to get neuron-0");

    // Access neuron-3, which should evict neuron-1 (least recently used)
    let _ = cache.get("neuron-3").expect("Failed to get neuron-3");

    // Check stats
    let stats = cache.stats();
    assert!(
        stats.eviction_count > 0,
        "Should have evicted at least one neuron"
    );

    // neuron-1 was evicted, so accessing it again should be a cache miss
    let before_misses = stats.cache_misses;
    let _ = cache.get("neuron-1").expect("Failed to get neuron-1 again");
    let after_stats = cache.stats();
    assert!(
        after_stats.cache_misses > before_misses,
        "Accessing evicted neuron-1 should be a cache miss"
    );
}

/// Test that LruRecordCache returns correct data after eviction and reload.
#[test]
fn lru_record_cache_returns_correct_data_after_eviction() {
    use neat_ai_discovery::analysis::cache::LruRecordCache;

    let temp_dir = TempDir::new().unwrap();
    let parquet_path = create_test_parquet(&temp_dir, 5, 50);

    // Very small capacity to force evictions
    let capacity = 100 * 50; // ~5KB, enough for 1 neuron

    let cache = LruRecordCache::new(&parquet_path, capacity).expect("Failed to create LRU cache");

    // Access multiple neurons to trigger evictions
    for i in 0..5 {
        let records = cache
            .get(&format!("neuron-{i}"))
            .expect("Failed to get neuron");
        // Verify correct number of records
        assert_eq!(
            records.len(),
            50,
            "Neuron-{i} should have 50 records, got {}",
            records.len()
        );
        // Verify all records belong to the correct neuron
        for record in records.iter() {
            assert_eq!(
                record.neuron_uuid,
                format!("neuron-{i}"),
                "Record should belong to neuron-{i}"
            );
        }
    }

    // Access neuron-0 again after it was likely evicted
    let records = cache.get("neuron-0").expect("Failed to get neuron-0");
    assert_eq!(
        records.len(),
        50,
        "Reloaded neuron-0 should have 50 records"
    );
}

/// Test that LruRecordCache respects capacity limits.
#[test]
fn lru_record_cache_respects_capacity_limits() {
    use neat_ai_discovery::analysis::cache::LruRecordCache;

    let temp_dir = TempDir::new().unwrap();
    let parquet_path = create_test_parquet(&temp_dir, 20, 100);

    // Capacity for approximately 5 neurons
    let capacity = 5 * 100 * 100; // ~50KB

    let cache = LruRecordCache::new(&parquet_path, capacity).expect("Failed to create LRU cache");

    // Access all 20 neurons
    for i in 0..20 {
        let _ = cache
            .get(&format!("neuron-{i}"))
            .expect("Failed to get neuron");
    }

    let stats = cache.stats();
    // Current bytes should not exceed capacity (with some tolerance for estimation)
    let tolerance = capacity / 2; // 50% tolerance for size estimation inaccuracies
    assert!(
        stats.current_bytes <= capacity + tolerance,
        "Current bytes ({}) should be <= capacity ({}) + tolerance ({})",
        stats.current_bytes,
        capacity,
        tolerance
    );
}

// =============================================================================
// Memory Usage Bounds Tests
// =============================================================================

/// Test that tiered cache respects memory bounds regardless of access pattern.
#[test]
fn tiered_cache_bounds_memory_usage() {
    use neat_ai_discovery::analysis::cache::TieredRecordCache;

    let temp_dir = TempDir::new().unwrap();
    // Create a moderately sized dataset
    let parquet_path = create_test_parquet(&temp_dir, 50, 100);

    // Force LRU mode with specific capacity
    let max_memory = 10 * 100 * 100; // Capacity for ~10 neurons

    let cache = TieredRecordCache::new_with_memory_limit(&parquet_path, max_memory)
        .expect("Failed to create tiered cache");

    // Access all neurons in random order
    let access_order = [25, 0, 49, 10, 30, 5, 45, 15, 35, 20, 40, 1, 48, 9, 29];
    for &i in &access_order {
        let _ = cache
            .get(&format!("neuron-{i}"))
            .expect("Failed to get neuron");
    }

    let stats = cache.stats();
    // Memory should stay bounded
    let tolerance = max_memory / 2;
    assert!(
        stats.current_bytes <= max_memory + tolerance,
        "Memory usage ({}) should be bounded by {} (+ {} tolerance)",
        stats.current_bytes,
        max_memory,
        tolerance
    );
}

// =============================================================================
// Data Correctness Tests
// =============================================================================

/// Test that all loading strategies return identical data.
#[test]
fn all_strategies_return_identical_data() {
    use neat_ai_discovery::analysis::cache::{LruRecordCache, RecordCache};

    let temp_dir = TempDir::new().unwrap();
    let parquet_path = create_test_parquet(&temp_dir, 10, 50);

    // Create preloaded cache (baseline)
    let preloaded =
        RecordCache::new_adaptive(&parquet_path).expect("Failed to create preloaded cache");

    // Create LRU cache with enough capacity to hold all data
    let lru_cache =
        LruRecordCache::new(&parquet_path, 100 * 1024 * 1024).expect("Failed to create LRU cache");

    // Compare data for all neurons
    for i in 0..10 {
        let uuid = format!("neuron-{i}");

        let preloaded_records = preloaded.get(&uuid).expect("Preloaded get failed");
        let lru_records = lru_cache.get(&uuid).expect("LRU get failed");

        assert_eq!(
            preloaded_records.len(),
            lru_records.len(),
            "Record count should match for {uuid}"
        );

        for (p, l) in preloaded_records.iter().zip(lru_records.iter()) {
            assert_eq!(p.obs_index, l.obs_index, "obs_index should match");
            assert_eq!(p.neuron_uuid, l.neuron_uuid, "neuron_uuid should match");
            assert_eq!(p.activation, l.activation, "activation should match");
            assert_eq!(p.value, l.value, "value should match");
            assert_eq!(p.errors, l.errors, "errors should match");
        }
    }
}

// =============================================================================
// Integration with RecordCache Tests
// =============================================================================

/// Test that RecordCache::new_tiered automatically selects appropriate strategy.
#[test]
fn new_tiered_selects_appropriate_strategy() {
    use neat_ai_discovery::analysis::cache::RecordCache;

    let temp_dir = TempDir::new().unwrap();
    // Small file - should use PreloadAll
    let parquet_path = create_test_parquet(&temp_dir, 5, 10);

    // new_tiered should work without panicking and provide correct data
    let cache = RecordCache::new_tiered(&parquet_path).expect("Failed to create tiered cache");

    // Verify data access works
    let records = cache.get("neuron-0").expect("Failed to get neuron-0");
    assert_eq!(records.len(), 10, "Should have correct number of records");
}

// =============================================================================
// Concurrent Access Tests
// =============================================================================

/// Test that LruRecordCache handles concurrent access correctly.
#[test]
fn lru_record_cache_concurrent_access() {
    use neat_ai_discovery::analysis::cache::LruRecordCache;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;

    let temp_dir = TempDir::new().unwrap();
    let parquet_path = create_test_parquet(&temp_dir, 10, 100);

    // Capacity for ~5 neurons
    let capacity = 5 * 100 * 100;
    let cache =
        Arc::new(LruRecordCache::new(&parquet_path, capacity).expect("Failed to create LRU cache"));

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
                assert_eq!(records.len(), 100);
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

// =============================================================================
// Statistics Tests
// =============================================================================

/// Test that LruRecordCache provides accurate statistics.
#[test]
fn lru_record_cache_statistics_are_accurate() {
    use neat_ai_discovery::analysis::cache::LruRecordCache;

    let temp_dir = TempDir::new().unwrap();
    let parquet_path = create_test_parquet(&temp_dir, 5, 50);

    // Large capacity to avoid evictions initially
    let capacity = 100 * 1024 * 1024;
    let cache = LruRecordCache::new(&parquet_path, capacity).expect("Failed to create LRU cache");

    // Initial stats
    let stats = cache.stats();
    assert_eq!(stats.cache_hits, 0);
    assert_eq!(stats.cache_misses, 0);
    assert_eq!(stats.cached_neurons, 0);

    // First access - should be a miss
    let _ = cache.get("neuron-0").expect("Failed to get neuron-0");
    let stats = cache.stats();
    assert_eq!(stats.cache_misses, 1, "First access should be a miss");
    assert_eq!(stats.cached_neurons, 1, "Should have 1 cached neuron");

    // Second access to same neuron - should be a hit
    let _ = cache.get("neuron-0").expect("Failed to get neuron-0");
    let stats = cache.stats();
    assert_eq!(stats.cache_hits, 1, "Second access should be a hit");
    assert_eq!(stats.cache_misses, 1, "Misses should not increase");
}
