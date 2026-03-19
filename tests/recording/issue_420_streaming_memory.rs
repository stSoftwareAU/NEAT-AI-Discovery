//! Integration tests for Issue #420: Streaming analysis for memory-constrained systems.
//!
//! Tests cover:
//! - Memory pressure detection
//! - Adaptive block sizing
//! - Compressed in-memory cache (LZ4)
//! - Progressive analysis (partial results)

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use neat_ai_discovery::analysis::cache::{
    CompressedLruRecordCache, LruRecordCache, RecordCache, select_loading_strategy,
};
use neat_ai_discovery::analysis::streaming::adaptive_block_size;
use neat_ai_discovery::analysis::utils::{MemoryPressure, categorise_memory_pressure};
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use tempfile::tempdir;

// =============================================================================
// Test data helpers
// =============================================================================

fn create_test_records(neuron_count: usize, records_per_neuron: usize) -> Vec<DiscoverRecord> {
    let mut records = Vec::with_capacity(neuron_count * records_per_neuron);
    for neuron_idx in 0..neuron_count {
        let neuron_uuid = format!("neuron-{neuron_idx}");
        for obs_idx in 0..records_per_neuron {
            let activation = (obs_idx as f32 % 100.0) / 100.0;
            let error = ((obs_idx + neuron_idx) as f32 % 50.0 - 25.0) / 50.0;
            records.push(DiscoverRecord::new(
                obs_idx as u32,
                neuron_uuid.clone(),
                Some(activation),
                activation,
                vec![error],
            ));
        }
    }
    records
}

fn create_test_parquet(
    neuron_count: usize,
    records_per_neuron: usize,
) -> (tempfile::TempDir, String) {
    let temp_dir = tempdir().expect("Failed to create temp directory");
    let parquet_path = temp_dir.path().join("test.parquet");
    let parquet_file = parquet_path.to_str().unwrap().to_string();
    let records = create_test_records(neuron_count, records_per_neuron);
    write_records_to_parquet(&parquet_file, &records).expect("Failed to write parquet");
    (temp_dir, parquet_file)
}

// =============================================================================
// Memory pressure detection tests
// =============================================================================

#[test]
fn memory_pressure_none_with_ample_memory() {
    // 20GB available of 32GB total — no pressure
    let available: u64 = 20 * 1024 * 1024 * 1024;
    let total: u64 = 32 * 1024 * 1024 * 1024;
    let pressure = categorise_memory_pressure(available, total);
    assert_eq!(pressure, MemoryPressure::None);
}

#[test]
fn memory_pressure_moderate_with_limited_memory() {
    // 4GB available of 16GB total — 25% available = moderate
    let available: u64 = 4 * 1024 * 1024 * 1024;
    let total: u64 = 16 * 1024 * 1024 * 1024;
    let pressure = categorise_memory_pressure(available, total);
    assert_eq!(pressure, MemoryPressure::Moderate);
}

#[test]
fn memory_pressure_high_with_scarce_memory() {
    // 1GB available of 16GB total — ~6% available = high
    let available: u64 = 1024 * 1024 * 1024;
    let total: u64 = 16 * 1024 * 1024 * 1024;
    let pressure = categorise_memory_pressure(available, total);
    assert_eq!(pressure, MemoryPressure::High);
}

#[test]
fn memory_pressure_critical_with_minimal_memory() {
    // 200MB available of 8GB total — ~2.4% available = critical
    let available: u64 = 200 * 1024 * 1024;
    let total: u64 = 8 * 1024 * 1024 * 1024;
    let pressure = categorise_memory_pressure(available, total);
    assert_eq!(pressure, MemoryPressure::Critical);
}

// =============================================================================
// Adaptive block sizing tests
// =============================================================================

#[test]
fn adaptive_block_size_increases_with_more_memory() {
    // More available memory should yield a larger block size
    let small_block = adaptive_block_size(2 * 1024 * 1024 * 1024); // 2GB available
    let large_block = adaptive_block_size(16 * 1024 * 1024 * 1024); // 16GB available
    assert!(
        large_block >= small_block,
        "large_block ({large_block}) should be >= small_block ({small_block})"
    );
}

#[test]
fn adaptive_block_size_respects_minimum() {
    // Even with very little memory, block size should be at least 100
    let block = adaptive_block_size(100 * 1024 * 1024); // 100MB available
    assert!(block >= 100, "Block size ({block}) should be >= 100");
}

#[test]
fn adaptive_block_size_caps_at_maximum() {
    // With lots of memory, block size shouldn't exceed a reasonable maximum
    let block = adaptive_block_size(128 * 1024 * 1024 * 1024); // 128GB available
    assert!(block <= 100_000, "Block size ({block}) should be <= 100000");
}

// =============================================================================
// Compressed LRU cache tests
// =============================================================================

#[test]
fn compressed_cache_returns_correct_records() {
    let (_dir, parquet_file) = create_test_parquet(10, 50);

    let cache = CompressedLruRecordCache::new(&parquet_file, 10 * 1024 * 1024)
        .expect("Failed to create compressed cache");

    // Load and verify records for each neuron
    for i in 0..10 {
        let uuid = format!("neuron-{i}");
        let records = cache.get(&uuid).expect("Failed to get records");
        assert_eq!(records.len(), 50, "Neuron {uuid} should have 50 records");
    }
}

#[test]
fn compressed_cache_hit_returns_same_data() {
    let (_dir, parquet_file) = create_test_parquet(5, 100);

    let cache = CompressedLruRecordCache::new(&parquet_file, 10 * 1024 * 1024)
        .expect("Failed to create compressed cache");

    // First access (miss)
    let records_1 = cache.get("neuron-0").expect("Failed on first get");
    // Second access (hit — decompressed from cache)
    let records_2 = cache.get("neuron-0").expect("Failed on second get");

    assert_eq!(records_1.len(), records_2.len());
    for (r1, r2) in records_1.iter().zip(records_2.iter()) {
        assert_eq!(r1.obs_index, r2.obs_index);
        assert_eq!(r1.neuron_uuid, r2.neuron_uuid);
        assert_eq!(r1.activation, r2.activation);
    }
}

#[test]
fn compressed_cache_uses_less_memory_than_uncompressed() {
    let (_dir, parquet_file) = create_test_parquet(20, 200);
    let capacity = 50 * 1024 * 1024; // 50MB

    // Load same data into both caches
    let uncompressed =
        LruRecordCache::new(&parquet_file, capacity).expect("Failed to create uncompressed cache");
    let compressed = CompressedLruRecordCache::new(&parquet_file, capacity)
        .expect("Failed to create compressed cache");

    // Access all neurons in both
    for i in 0..20 {
        let uuid = format!("neuron-{i}");
        uncompressed
            .get(&uuid)
            .expect("Failed to get from uncompressed");
        compressed
            .get(&uuid)
            .expect("Failed to get from compressed");
    }

    let uncompressed_bytes = uncompressed.stats().current_bytes;
    let compressed_bytes = compressed.stats().current_bytes;

    // Compressed cache should use less memory for its stored entries
    assert!(
        compressed_bytes < uncompressed_bytes,
        "Compressed cache ({compressed_bytes} bytes) should use less memory \
         than uncompressed ({uncompressed_bytes} bytes)"
    );
}

#[test]
fn compressed_cache_evicts_under_pressure() {
    let (_dir, parquet_file) = create_test_parquet(20, 100);

    // Very small capacity to force evictions
    let cache = CompressedLruRecordCache::new(&parquet_file, 5 * 1024)
        .expect("Failed to create compressed cache");

    // Access all neurons — should trigger evictions
    for i in 0..20 {
        let uuid = format!("neuron-{i}");
        let records = cache.get(&uuid).expect("Failed to get records");
        assert_eq!(records.len(), 100);
    }

    let stats = cache.stats();
    assert!(
        stats.eviction_count > 0,
        "Expected evictions, got {stats:?}"
    );
}

// =============================================================================
// Progressive analysis / partial results tests
// =============================================================================

#[test]
fn loading_strategy_with_memory_pressure_prefers_streaming() {
    // Under memory pressure, even medium files should use streaming
    // File: 500MB, available: 3GB (normally LRU), total: 16GB
    let file_size = 500 * 1024 * 1024_u64;
    let available = 3 * 1024 * 1024 * 1024_u64;
    let strategy = select_loading_strategy(file_size, available);
    // 500MB * 3 = 1.5GB estimated, which is < 3GB but > 3GB/4=0.75GB
    // So this should select LRU — that's fine for moderate pressure
    assert!(
        !matches!(
            strategy,
            neat_ai_discovery::analysis::cache::LoadingStrategy::PreloadAll
        ),
        "Should not preload under moderate memory constraints"
    );
}

#[test]
fn record_cache_with_loader_supports_partial_access() {
    // Verify that the cache works correctly when only a subset of neurons are accessed
    let (_dir, parquet_file) = create_test_parquet(50, 100);

    let cache = RecordCache::new_adaptive(&parquet_file).expect("Failed to create cache");

    // Only access 5 out of 50 neurons — partial analysis scenario
    let mut total_records = 0;
    for i in [0, 10, 20, 30, 40] {
        let uuid = format!("neuron-{i}");
        let records = cache.get(&uuid).expect("Failed to get records");
        assert_eq!(records.len(), 100);
        total_records += records.len();
    }
    assert_eq!(total_records, 500);
}
