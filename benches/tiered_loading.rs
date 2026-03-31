//! Benchmark for Issue #215: Tiered/streaming Parquet loader for large files.
//!
//! This benchmark compares memory usage and access performance across different
//! loading strategies (`PreloadAll`, `LruCache`, Streaming) to validate the tiered
//! loading implementation.
//!
//! ## Key Metrics
//!
//! 1. **Memory usage**: Compares peak memory for each strategy
//! 2. **Cache hit rates**: Measures LRU cache effectiveness
//! 3. **Access time**: Compares time to access neurons across strategies
//! 4. **Eviction overhead**: Measures cost of LRU eviction

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use neat_ai_discovery::analysis::cache::{LruRecordCache, RecordCache, TieredRecordCache};
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use std::hint::black_box;
use std::sync::Arc;
use tempfile::tempdir;

/// Create test records for benchmarking.
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

/// Benchmark comparing access patterns across loading strategies.
fn benchmark_tiered_loading_access(c: &mut Criterion) {
    let neuron_count = 100;
    let records_per_neuron = 1000;

    let temp_dir = tempdir().expect("Failed to create temp directory");
    let parquet_path = temp_dir.path().join("benchmark.parquet");
    let parquet_file = parquet_path.to_str().unwrap().to_string();

    // Create test data
    let records = create_test_records(neuron_count, records_per_neuron);
    write_records_to_parquet(&parquet_file, &records).expect("Failed to write parquet");

    let mut group = c.benchmark_group("tiered_loading_access");

    // Benchmark preloaded cache
    group.bench_function("preloaded_sequential", |b| {
        let cache = RecordCache::new_adaptive(&parquet_file).expect("Failed to create cache");
        b.iter(|| {
            for i in 0..neuron_count {
                let uuid = format!("neuron-{i}");
                let records = cache.get(&uuid).expect("Failed to get records");
                black_box(records.len());
            }
        });
    });

    // Benchmark LRU cache with enough capacity
    group.bench_function("lru_cache_sequential", |b| {
        let cache = LruRecordCache::new(&parquet_file, 100 * 1024 * 1024)
            .expect("Failed to create LRU cache");
        b.iter(|| {
            for i in 0..neuron_count {
                let uuid = format!("neuron-{i}");
                let records = cache.get(&uuid).expect("Failed to get records");
                black_box(records.len());
            }
        });
    });

    // Benchmark LRU cache with limited capacity (forces evictions)
    group.bench_function("lru_cache_evicting", |b| {
        // Capacity for about 10 neurons
        let cache = LruRecordCache::new(&parquet_file, 10 * 1000 * 100)
            .expect("Failed to create LRU cache");
        b.iter(|| {
            for i in 0..neuron_count {
                let uuid = format!("neuron-{i}");
                let records = cache.get(&uuid).expect("Failed to get records");
                black_box(records.len());
            }
        });
    });

    // Benchmark tiered cache (auto-selection)
    group.bench_function("tiered_auto", |b| {
        let cache = TieredRecordCache::new(&parquet_file).expect("Failed to create tiered cache");
        b.iter(|| {
            for i in 0..neuron_count {
                let uuid = format!("neuron-{i}");
                let records = cache.get(&uuid).expect("Failed to get records");
                black_box(records.len());
            }
        });
    });

    group.finish();
}

/// Benchmark LRU cache hit rate with different access patterns.
fn benchmark_lru_cache_patterns(c: &mut Criterion) {
    let neuron_count = 50;
    let records_per_neuron = 500;

    let temp_dir = tempdir().expect("Failed to create temp directory");
    let parquet_path = temp_dir.path().join("lru_benchmark.parquet");
    let parquet_file = parquet_path.to_str().unwrap().to_string();

    let records = create_test_records(neuron_count, records_per_neuron);
    write_records_to_parquet(&parquet_file, &records).expect("Failed to write parquet");

    let mut group = c.benchmark_group("lru_cache_patterns");

    // Sequential access (good for LRU)
    group.bench_function("sequential", |b| {
        let cache =
            LruRecordCache::new(&parquet_file, 20 * 500 * 100).expect("Failed to create LRU cache");
        b.iter(|| {
            for _ in 0..3 {
                for i in 0..neuron_count {
                    let uuid = format!("neuron-{i}");
                    let records = cache.get(&uuid).expect("Failed to get records");
                    black_box(records.len());
                }
            }
        });
    });

    // Random access (poor for LRU with limited capacity)
    group.bench_function("random", |b| {
        let cache =
            LruRecordCache::new(&parquet_file, 20 * 500 * 100).expect("Failed to create LRU cache");
        // Pseudo-random access pattern
        let access_pattern: Vec<usize> = (0..neuron_count * 3)
            .map(|i| (i * 17 + 7) % neuron_count)
            .collect();
        b.iter(|| {
            for &i in &access_pattern {
                let uuid = format!("neuron-{i}");
                let records = cache.get(&uuid).expect("Failed to get records");
                black_box(records.len());
            }
        });
    });

    // Hot-spot access (few neurons accessed frequently)
    group.bench_function("hotspot", |b| {
        let cache =
            LruRecordCache::new(&parquet_file, 20 * 500 * 100).expect("Failed to create LRU cache");
        // 80% of accesses to 20% of neurons
        let hot_neurons: Vec<usize> = (0..10).collect();
        let cold_neurons: Vec<usize> = (10..neuron_count).collect();
        b.iter(|| {
            // Access hot neurons 4 times each
            for _ in 0..4 {
                for &i in &hot_neurons {
                    let uuid = format!("neuron-{i}");
                    let records = cache.get(&uuid).expect("Failed to get records");
                    black_box(records.len());
                }
            }
            // Access cold neurons once
            for &i in &cold_neurons {
                let uuid = format!("neuron-{i}");
                let records = cache.get(&uuid).expect("Failed to get records");
                black_box(records.len());
            }
        });
    });

    group.finish();
}

/// Benchmark concurrent access to LRU cache.
fn benchmark_concurrent_access(c: &mut Criterion) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;

    let neuron_count = 50;
    let records_per_neuron = 200;

    let temp_dir = tempdir().expect("Failed to create temp directory");
    let parquet_path = temp_dir.path().join("concurrent_benchmark.parquet");
    let parquet_file = parquet_path.to_str().unwrap().to_string();

    let records = create_test_records(neuron_count, records_per_neuron);
    write_records_to_parquet(&parquet_file, &records).expect("Failed to write parquet");

    let mut group = c.benchmark_group("concurrent_access");

    for thread_count in [1, 2, 4, 8] {
        group.bench_with_input(
            BenchmarkId::from_parameter(thread_count),
            &thread_count,
            |b, &thread_count| {
                let cache = Arc::new(
                    LruRecordCache::new(&parquet_file, 30 * 200 * 100)
                        .expect("Failed to create LRU cache"),
                );

                b.iter(|| {
                    let access_count = Arc::new(AtomicUsize::new(0));
                    let mut handles = Vec::new();

                    for t in 0..thread_count {
                        let cache_clone = Arc::clone(&cache);
                        let access_clone = Arc::clone(&access_count);

                        let handle = thread::spawn(move || {
                            // Each thread accesses neurons in a different order
                            for j in 0..neuron_count {
                                let i = (j + t * 7) % neuron_count;
                                let uuid = format!("neuron-{i}");
                                let records =
                                    cache_clone.get(&uuid).expect("Failed to get records");
                                black_box(records.len());
                                access_clone.fetch_add(1, Ordering::Relaxed);
                            }
                        });
                        handles.push(handle);
                    }

                    for handle in handles {
                        handle.join().expect("Thread panicked");
                    }

                    access_count.load(Ordering::Relaxed)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_tiered_loading_access,
    benchmark_lru_cache_patterns,
    benchmark_concurrent_access
);
criterion_main!(benches);
