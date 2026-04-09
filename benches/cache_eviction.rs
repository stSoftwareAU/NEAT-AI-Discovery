//! Benchmark for Issue #1040: Cache eviction patterns under memory pressure.
//!
//! `lru_cache.rs` and `compressed_cache.rs` handle memory-constrained scenarios,
//! but no benchmark exercises hit/miss patterns under pressure. This benchmark
//! measures LRU and compressed cache throughput under random-access and
//! sequential-access patterns with varying memory limits.
//!
//! ## Key Metrics
//!
//! 1. **Sequential access**: Good locality — few evictions expected
//! 2. **Random access**: Poor locality — frequent evictions under memory pressure
//! 3. **Hot-spot access**: Skewed workload — 80/20 access pattern
//! 4. **LRU vs Compressed**: Compare eviction overhead with and without LZ4

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)] // Intentional numeric casts for benchmark data generation (Issue #873)

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use neat_ai_discovery::analysis::cache::{CompressedLruRecordCache, LruRecordCache};
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use std::hint::black_box;
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
                vec![error, error * 0.5],
            ));
        }
    }
    records
}

/// Write test records to a temporary parquet file.
fn setup_parquet(neuron_count: usize, records_per_neuron: usize) -> (tempfile::TempDir, String) {
    let temp_dir = tempdir().expect("Failed to create temp directory");
    let parquet_path = temp_dir.path().join("eviction_bench.parquet");
    let parquet_file = parquet_path.to_str().unwrap().to_string();
    let records = create_test_records(neuron_count, records_per_neuron);
    write_records_to_parquet(&parquet_file, &records).expect("Failed to write parquet");
    (temp_dir, parquet_file)
}

/// Benchmark LRU cache under sequential access with varying memory limits.
fn bench_lru_sequential_pressure(c: &mut Criterion) {
    let neuron_count = 80;
    let records_per_neuron = 500;
    let (_temp_dir, parquet_file) = setup_parquet(neuron_count, records_per_neuron);

    let mut group = c.benchmark_group("cache_eviction_lru_sequential");

    // Capacity for roughly N neurons: each neuron ~500 records × ~40 bytes ≈ 20 KB
    for capacity_neurons in [10, 40, 80] {
        let capacity_bytes = capacity_neurons * records_per_neuron * 40;
        let cache =
            LruRecordCache::new(&parquet_file, capacity_bytes).expect("Failed to create LRU cache");

        let id = format!("{capacity_neurons}_of_{neuron_count}_neurons");
        group.bench_function(BenchmarkId::new("sequential", &id), |b| {
            b.iter(|| {
                for i in 0..neuron_count {
                    let uuid = format!("neuron-{i}");
                    let records = cache.get(&uuid).expect("Failed to get records");
                    black_box(records.len());
                }
            });
        });
    }

    group.finish();
}

/// Benchmark LRU cache under random access with varying memory limits.
fn bench_lru_random_pressure(c: &mut Criterion) {
    let neuron_count = 80;
    let records_per_neuron = 500;
    let (_temp_dir, parquet_file) = setup_parquet(neuron_count, records_per_neuron);

    let mut group = c.benchmark_group("cache_eviction_lru_random");

    // Pseudo-random access pattern (deterministic for reproducibility)
    let access_pattern: Vec<usize> = (0..neuron_count * 2)
        .map(|i| (i * 37 + 13) % neuron_count)
        .collect();

    for capacity_neurons in [10, 40, 80] {
        let capacity_bytes = capacity_neurons * records_per_neuron * 40;
        let cache =
            LruRecordCache::new(&parquet_file, capacity_bytes).expect("Failed to create LRU cache");

        let id = format!("{capacity_neurons}_of_{neuron_count}_neurons");
        group.bench_function(BenchmarkId::new("random", &id), |b| {
            b.iter(|| {
                for &i in &access_pattern {
                    let uuid = format!("neuron-{i}");
                    let records = cache.get(&uuid).expect("Failed to get records");
                    black_box(records.len());
                }
            });
        });
    }

    group.finish();
}

/// Benchmark LRU cache under hot-spot (80/20) access pattern.
fn bench_lru_hotspot_pressure(c: &mut Criterion) {
    let neuron_count = 80;
    let records_per_neuron = 500;
    let (_temp_dir, parquet_file) = setup_parquet(neuron_count, records_per_neuron);

    let mut group = c.benchmark_group("cache_eviction_lru_hotspot");

    // 20% of neurons are "hot" — accessed 4× more frequently
    let hot_neurons: Vec<usize> = (0..neuron_count / 5).collect();
    let cold_neurons: Vec<usize> = (neuron_count / 5..neuron_count).collect();

    // Build access pattern: hot neurons 4 times, cold neurons once
    let mut access_pattern = Vec::new();
    for _ in 0..4 {
        for &i in &hot_neurons {
            access_pattern.push(i);
        }
    }
    for &i in &cold_neurons {
        access_pattern.push(i);
    }

    for capacity_neurons in [10, 40, 80] {
        let capacity_bytes = capacity_neurons * records_per_neuron * 40;
        let cache =
            LruRecordCache::new(&parquet_file, capacity_bytes).expect("Failed to create LRU cache");

        let id = format!("{capacity_neurons}_of_{neuron_count}_neurons");
        group.bench_function(BenchmarkId::new("hotspot", &id), |b| {
            b.iter(|| {
                for &i in &access_pattern {
                    let uuid = format!("neuron-{i}");
                    let records = cache.get(&uuid).expect("Failed to get records");
                    black_box(records.len());
                }
            });
        });
    }

    group.finish();
}

/// Benchmark compressed LRU cache vs uncompressed under memory pressure.
fn bench_compressed_vs_uncompressed(c: &mut Criterion) {
    let neuron_count = 80;
    let records_per_neuron = 500;
    let (_temp_dir, parquet_file) = setup_parquet(neuron_count, records_per_neuron);

    let mut group = c.benchmark_group("cache_eviction_compressed_vs_lru");

    // Moderate pressure: capacity for about half the neurons
    let capacity_bytes = 40 * records_per_neuron * 40;

    // Sequential access pattern through all neurons
    let lru_cache =
        LruRecordCache::new(&parquet_file, capacity_bytes).expect("Failed to create LRU cache");
    group.bench_function("lru_sequential", |b| {
        b.iter(|| {
            for i in 0..neuron_count {
                let uuid = format!("neuron-{i}");
                let records = lru_cache.get(&uuid).expect("Failed to get records");
                black_box(records.len());
            }
        });
    });

    let compressed_cache = CompressedLruRecordCache::new(&parquet_file, capacity_bytes)
        .expect("Failed to create compressed cache");
    group.bench_function("compressed_sequential", |b| {
        b.iter(|| {
            for i in 0..neuron_count {
                let uuid = format!("neuron-{i}");
                let records = compressed_cache.get(&uuid).expect("Failed to get records");
                black_box(records.len());
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_lru_sequential_pressure,
    bench_lru_random_pressure,
    bench_lru_hotspot_pressure,
    bench_compressed_vs_uncompressed
);
criterion_main!(benches);
