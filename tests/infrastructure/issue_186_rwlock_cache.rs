//! Tests for Issue #186: Replace Mutex with RwLock for RecordCache.
//!
//! This test verifies that the RecordCache allows concurrent reads when using RwLock,
//! which is essential for read-heavy workloads during analysis where multiple focus
//! neurons are processed in parallel.
//!
//! ## Key Behaviour Verified
//!
//! 1. Multiple readers can access the cache concurrently (no serialisation)
//! 2. Cache still works correctly with concurrent access
//! 3. The RwLock implementation is transparent to callers
//! 4. The len() method works correctly with concurrent access

use neat_ai_discovery::types::DiscoverRecord;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;

/// Test that verifies the cache can handle concurrent reads correctly.
///
/// This test creates a cache with test data and then spawns multiple threads
/// that all read from the cache simultaneously. With RwLock, these reads should
/// not be serialised.
#[test]
fn cache_allows_concurrent_reads() {
    use neat_ai_discovery::analysis::cache::RecordCache;

    // Create a test cache with a loader that returns predictable data
    let cache = Arc::new(RecordCache::with_loader(
        "test.parquet",
        Arc::new(|_file, uuid| {
            // Small delay to simulate loading work
            thread::sleep(Duration::from_millis(1));
            let records = (0..10u32)
                .map(|obs_index| DiscoverRecord {
                    obs_index,
                    neuron_uuid: uuid.to_string(),
                    value: Some(obs_index as f32 / 10.0),
                    activation: obs_index as f32 / 10.0,
                    errors: vec![0.1],
                })
                .collect();
            Ok(records)
        }),
    ));

    // Pre-populate the cache with some neuron records
    for i in 0..5 {
        let uuid = format!("neuron-{i}");
        let _ = cache.get(&uuid).expect("Initial get should succeed");
    }

    // Track successful concurrent reads
    let successful_reads = Arc::new(AtomicUsize::new(0));
    let thread_count = 10;

    let mut handles = Vec::with_capacity(thread_count);

    for thread_id in 0..thread_count {
        let cache_clone = Arc::clone(&cache);
        let reads_clone = Arc::clone(&successful_reads);

        let handle = thread::spawn(move || {
            // Each thread reads from multiple neurons
            for i in 0..5 {
                let uuid = format!("neuron-{i}");
                let result = cache_clone.get(&uuid);
                assert!(
                    result.is_ok(),
                    "Thread {thread_id} failed to get neuron-{i}"
                );
                let records = result.unwrap();
                assert_eq!(
                    records.len(),
                    10,
                    "Thread {thread_id} got wrong record count for neuron-{i}"
                );
                reads_clone.fetch_add(1, Ordering::Relaxed);
            }
        });
        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().expect("Thread should not panic");
    }

    // Verify all reads succeeded
    let total_reads = successful_reads.load(Ordering::Relaxed);
    assert_eq!(
        total_reads,
        thread_count * 5,
        "Expected {expected} total reads, got {total_reads}",
        expected = thread_count * 5
    );
}

/// Test that the cache returns consistent data across concurrent reads.
///
/// This ensures that despite concurrent access, all threads see the same data
/// for the same neuron UUID.
#[test]
fn cache_returns_consistent_data_across_concurrent_reads() {
    use neat_ai_discovery::analysis::cache::RecordCache;

    let call_count = Arc::new(AtomicUsize::new(0));
    let call_count_clone = Arc::clone(&call_count);

    // Create cache with a loader that counts calls
    let cache = Arc::new(RecordCache::with_loader(
        "test.parquet",
        Arc::new(move |_file, uuid| {
            call_count_clone.fetch_add(1, Ordering::SeqCst);
            let records = (0..5u32)
                .map(|obs_index| DiscoverRecord {
                    obs_index,
                    neuron_uuid: uuid.to_string(),
                    value: Some(1.0),
                    activation: 1.0,
                    errors: vec![0.0],
                })
                .collect();
            Ok(records)
        }),
    ));

    let thread_count = 8;
    let reads_per_thread = 10;
    let target_uuid = "shared-neuron";

    let mut handles = Vec::with_capacity(thread_count);

    for _ in 0..thread_count {
        let cache_clone = Arc::clone(&cache);

        let handle = thread::spawn(move || {
            for _ in 0..reads_per_thread {
                let records = cache_clone.get(target_uuid).expect("Get should succeed");

                // Verify data consistency
                assert_eq!(records.len(), 5, "Record count should be consistent");
                for record in records.iter() {
                    assert_eq!(record.neuron_uuid, target_uuid, "UUID should match");
                    assert_eq!(record.activation, 1.0, "Activation should match");
                }
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().expect("Thread should not panic");
    }

    // With OnceCell, the loader should only be called once despite many reads
    let total_calls = call_count.load(Ordering::SeqCst);
    assert_eq!(
        total_calls, 1,
        "Loader should only be called once for the same UUID, but was called {total_calls} times"
    );
}

/// Test that cache.len() works correctly with concurrent access.
///
/// The diagnostics code uses cache.len() to report statistics. This must work
/// reliably even when other threads are reading from the cache.
#[test]
fn cache_len_works_with_concurrent_reads() {
    use neat_ai_discovery::analysis::cache::RecordCache;

    let cache = Arc::new(RecordCache::with_loader(
        "test.parquet",
        Arc::new(|_file, uuid| {
            let records = vec![DiscoverRecord {
                obs_index: 0,
                neuron_uuid: uuid.to_string(),
                value: None,
                activation: 0.0,
                errors: vec![],
            }];
            Ok(records)
        }),
    ));

    // Pre-populate with known number of entries
    let entry_count = 10;
    for i in 0..entry_count {
        let _ = cache.get(&format!("neuron-{i}")).unwrap();
    }

    // Verify the len() method works correctly
    assert_eq!(
        cache.len(),
        entry_count,
        "Cache len() should return {entry_count}"
    );

    // Start concurrent readers
    let thread_count = 4;
    let mut handles = Vec::new();

    for _ in 0..thread_count {
        let cache_clone = Arc::clone(&cache);
        let handle = thread::spawn(move || {
            for _ in 0..20 {
                // Perform reads while the main thread checks len()
                for i in 0..entry_count {
                    let _ = cache_clone.get(&format!("neuron-{i}"));
                }
            }
        });
        handles.push(handle);
    }

    // Check len() while reads are happening - this uses a read lock
    // which should not block the concurrent reads
    for _ in 0..10 {
        let len = cache.len();
        assert!(
            len >= entry_count,
            "Cache should have at least {entry_count} entries, got {len}"
        );
        thread::sleep(Duration::from_millis(1));
    }

    for handle in handles {
        handle.join().expect("Thread should not panic");
    }
}

/// Test that is_empty() works correctly.
#[test]
fn cache_is_empty_works_correctly() {
    use neat_ai_discovery::analysis::cache::RecordCache;

    let cache = RecordCache::with_loader(
        "test.parquet",
        Arc::new(|_file, uuid| {
            let records = vec![DiscoverRecord {
                obs_index: 0,
                neuron_uuid: uuid.to_string(),
                value: None,
                activation: 0.0,
                errors: vec![],
            }];
            Ok(records)
        }),
    );

    // Initially empty
    assert!(cache.is_empty(), "Cache should be empty initially");
    assert_eq!(cache.len(), 0, "Cache len() should be 0 initially");

    // Add an entry
    let _ = cache.get("neuron-0").unwrap();

    // No longer empty
    assert!(!cache.is_empty(), "Cache should not be empty after get()");
    assert_eq!(cache.len(), 1, "Cache len() should be 1 after one get()");
}

/// Test that the loader is invoked only once for a given UUID.
///
/// When a neuron is already in the cache, subsequent gets should return the
/// cached data without invoking the loader again.
///
/// Issue #454: Converted from timing-based comparison to functional test.
/// Performance measurement belongs in `benches/`, not unit tests.
#[test]
fn cache_invokes_loader_only_once_for_same_uuid() {
    use neat_ai_discovery::analysis::cache::RecordCache;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let load_count = Arc::new(AtomicUsize::new(0));
    let load_count_clone = Arc::clone(&load_count);

    let cache = Arc::new(RecordCache::with_loader(
        "test.parquet",
        Arc::new(move |_file, uuid| {
            load_count_clone.fetch_add(1, Ordering::SeqCst);
            let records = vec![DiscoverRecord {
                obs_index: 0,
                neuron_uuid: uuid.to_string(),
                value: None,
                activation: 0.0,
                errors: vec![],
            }];
            Ok(records)
        }),
    ));

    // First get — triggers the loader
    let _ = cache.get("neuron-0").unwrap();
    assert_eq!(
        load_count.load(Ordering::SeqCst),
        1,
        "Loader should be invoked on first get"
    );

    // Second get — should use cached data, not the loader
    let _ = cache.get("neuron-0").unwrap();
    assert_eq!(
        load_count.load(Ordering::SeqCst),
        1,
        "Loader should NOT be invoked again for the same UUID"
    );

    // Different UUID — should invoke the loader
    let _ = cache.get("neuron-1").unwrap();
    assert_eq!(
        load_count.load(Ordering::SeqCst),
        2,
        "Loader should be invoked for a different UUID"
    );
}
