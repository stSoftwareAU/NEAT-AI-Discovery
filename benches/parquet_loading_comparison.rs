//! Benchmark for Issue #1040: Parquet loading strategy comparison.
//!
//! The record cache supports pre-loaded, LRU, and streaming tiers, but no
//! benchmark compares them on the same dataset. This benchmark exercises all
//! three strategies on synthetic datasets of varying sizes and measures access
//! throughput.
//!
//! ## Key Metrics
//!
//! 1. **Pre-loaded**: Fastest reads, highest memory usage
//! 2. **LRU cache**: Balanced — bounded memory with per-neuron eviction
//! 3. **Streaming**: Lowest memory usage, block-based loading with prefetch
//! 4. **Tiered (auto)**: Automatic selection — validates the strategy picker

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)] // Intentional numeric casts for benchmark data generation (Issue #873)

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use neat_ai_discovery::analysis::cache::{
    LruRecordCache, RecordCache, StreamingRecordCache, TieredRecordCache,
};
use neat_ai_discovery::parquet_format::{
    ColumnProfile, read_all_records_grouped_by_neuron_with_profile, write_records_to_parquet,
};
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
                vec![error],
            ));
        }
    }
    records
}

/// Write test records to a temporary parquet file and return the dir guard + path.
fn setup_parquet(neuron_count: usize, records_per_neuron: usize) -> (tempfile::TempDir, String) {
    let temp_dir = tempdir().expect("Failed to create temp directory");
    let parquet_path = temp_dir.path().join("loading_bench.parquet");
    let parquet_file = parquet_path.to_str().unwrap().to_string();
    let records = create_test_records(neuron_count, records_per_neuron);
    write_records_to_parquet(&parquet_file, &records).expect("Failed to write parquet");
    (temp_dir, parquet_file)
}

/// Benchmark all loading strategies on a small dataset (50 neurons × 200 records).
fn bench_small_dataset(c: &mut Criterion) {
    let neuron_count = 50;
    let records_per_neuron = 200;
    let (_temp_dir, parquet_file) = setup_parquet(neuron_count, records_per_neuron);

    let mut group = c.benchmark_group("parquet_loading_small");

    // Pre-loaded (adaptive defaults to preload for small files)
    group.bench_function("preloaded", |b| {
        let cache = RecordCache::new_adaptive(&parquet_file).expect("Failed to create cache");
        b.iter(|| {
            for i in 0..neuron_count {
                let uuid = format!("neuron-{i}");
                let records = cache.get(&uuid).expect("Failed to get records");
                black_box(records.len());
            }
        });
    });

    // LRU cache with generous capacity
    group.bench_function("lru_generous", |b| {
        let cache = LruRecordCache::new(&parquet_file, 50 * 1024 * 1024)
            .expect("Failed to create LRU cache");
        b.iter(|| {
            for i in 0..neuron_count {
                let uuid = format!("neuron-{i}");
                let records = cache.get(&uuid).expect("Failed to get records");
                black_box(records.len());
            }
        });
    });

    // Streaming cache
    group.bench_function("streaming", |b| {
        let cache = StreamingRecordCache::new(&parquet_file, Some(20), Some(2))
            .expect("Failed to create streaming cache");
        b.iter(|| {
            for i in 0..neuron_count {
                let uuid = format!("neuron-{i}");
                let records = cache.get(&uuid).expect("Failed to get records");
                black_box(records.len());
            }
        });
    });

    // Tiered (auto-selection)
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

/// Benchmark all loading strategies on a medium dataset (200 neurons × 500 records).
fn bench_medium_dataset(c: &mut Criterion) {
    let neuron_count = 200;
    let records_per_neuron = 500;
    let (_temp_dir, parquet_file) = setup_parquet(neuron_count, records_per_neuron);

    let mut group = c.benchmark_group("parquet_loading_medium");

    // Pre-loaded
    group.bench_function("preloaded", |b| {
        let cache = RecordCache::new_adaptive(&parquet_file).expect("Failed to create cache");
        b.iter(|| {
            for i in 0..neuron_count {
                let uuid = format!("neuron-{i}");
                let records = cache.get(&uuid).expect("Failed to get records");
                black_box(records.len());
            }
        });
    });

    // LRU with bounded capacity (capacity for ~100 neurons)
    group.bench_function("lru_bounded", |b| {
        let cache = LruRecordCache::new(&parquet_file, 100 * records_per_neuron * 40)
            .expect("Failed to create LRU cache");
        b.iter(|| {
            for i in 0..neuron_count {
                let uuid = format!("neuron-{i}");
                let records = cache.get(&uuid).expect("Failed to get records");
                black_box(records.len());
            }
        });
    });

    // Streaming cache
    group.bench_function("streaming", |b| {
        let cache = StreamingRecordCache::new(&parquet_file, Some(30), Some(2))
            .expect("Failed to create streaming cache");
        b.iter(|| {
            for i in 0..neuron_count {
                let uuid = format!("neuron-{i}");
                let records = cache.get(&uuid).expect("Failed to get records");
                black_box(records.len());
            }
        });
    });

    // Tiered (auto-selection)
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

/// Benchmark loading strategies with varying dataset sizes for scaling analysis.
fn bench_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("parquet_loading_scaling");

    for (neuron_count, records_per_neuron) in [(25, 100), (100, 500), (400, 1000)] {
        let (_temp_dir, parquet_file) = setup_parquet(neuron_count, records_per_neuron);
        let total_records = neuron_count * records_per_neuron;
        let id = format!("{neuron_count}n_{records_per_neuron}r_{total_records}total");

        // Pre-loaded access
        let cache = RecordCache::new_adaptive(&parquet_file).expect("Failed to create cache");
        group.bench_with_input(
            BenchmarkId::new("preloaded", &id),
            &neuron_count,
            |b, &nc| {
                b.iter(|| {
                    for i in 0..nc {
                        let uuid = format!("neuron-{i}");
                        let records = cache.get(&uuid).expect("Failed to get records");
                        black_box(records.len());
                    }
                });
            },
        );

        // LRU access (capacity for half the neurons)
        let lru_capacity = (neuron_count / 2) * records_per_neuron * 40;
        let lru_cache =
            LruRecordCache::new(&parquet_file, lru_capacity).expect("Failed to create LRU cache");
        group.bench_with_input(
            BenchmarkId::new("lru_half_capacity", &id),
            &neuron_count,
            |b, &nc| {
                b.iter(|| {
                    for i in 0..nc {
                        let uuid = format!("neuron-{i}");
                        let records = lru_cache.get(&uuid).expect("Failed to get records");
                        black_box(records.len());
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark column pruning: full read vs without-errors read (Issue #1073).
///
/// The `errors` column is a `ListArray` requiring full materialisation.
/// Skipping it when not needed should reduce I/O and deserialisation time.
fn bench_column_pruning(c: &mut Criterion) {
    let mut group = c.benchmark_group("parquet_column_pruning");

    // Use larger error arrays to amplify the ListArray deserialisation cost
    for (neuron_count, records_per_neuron, errors_per_record) in
        [(50, 200, 5), (100, 500, 10), (200, 500, 20)]
    {
        let temp_dir = tempdir().expect("Failed to create temp directory");
        let parquet_path = temp_dir.path().join("pruning_bench.parquet");
        let parquet_file = parquet_path.to_str().unwrap().to_string();

        let mut records = Vec::with_capacity(neuron_count * records_per_neuron);
        for neuron_idx in 0..neuron_count {
            let neuron_uuid = format!("neuron-{neuron_idx}");
            for obs_idx in 0..records_per_neuron {
                let activation = (obs_idx as f32 % 100.0) / 100.0;
                let errors: Vec<f32> = (0..errors_per_record)
                    .map(|e| ((obs_idx + e) as f32 % 50.0 - 25.0) / 50.0)
                    .collect();
                records.push(DiscoverRecord::new(
                    obs_idx as u32,
                    neuron_uuid.clone(),
                    Some(activation),
                    activation,
                    errors,
                ));
            }
        }
        write_records_to_parquet(&parquet_file, &records).expect("Failed to write parquet");

        let total = neuron_count * records_per_neuron;
        let id = format!("{neuron_count}n_{records_per_neuron}r_{errors_per_record}e_{total}total");

        group.bench_with_input(
            BenchmarkId::new("full_read", &id),
            &parquet_file,
            |b, pf| {
                b.iter(|| {
                    let grouped =
                        read_all_records_grouped_by_neuron_with_profile(pf, ColumnProfile::Full)
                            .expect("Failed to read");
                    black_box(grouped.len());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("without_errors", &id),
            &parquet_file,
            |b, pf| {
                b.iter(|| {
                    let grouped = read_all_records_grouped_by_neuron_with_profile(
                        pf,
                        ColumnProfile::WithoutErrors,
                    )
                    .expect("Failed to read");
                    black_box(grouped.len());
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_small_dataset,
    bench_medium_dataset,
    bench_scaling,
    bench_column_pruning
);
criterion_main!(benches);
