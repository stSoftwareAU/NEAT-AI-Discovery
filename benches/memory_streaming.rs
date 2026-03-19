//! Benchmark for Issue #420: Memory-constrained streaming analysis.
//!
//! Compares memory efficiency between compressed and uncompressed LRU caches
//! to demonstrate that LZ4 compression reduces memory footprint while maintaining
//! acceptable access performance.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use neat_ai_discovery::analysis::cache::{CompressedLruRecordCache, LruRecordCache};
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
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

/// Benchmark comparing compressed vs uncompressed LRU cache access.
fn benchmark_compressed_vs_uncompressed(c: &mut Criterion) {
    let neuron_count = 50;
    let records_per_neuron = 500;
    let capacity = 50 * 1024 * 1024; // 50MB

    let temp_dir = tempdir().expect("Failed to create temp directory");
    let parquet_path = temp_dir.path().join("benchmark.parquet");
    let parquet_file = parquet_path.to_str().unwrap().to_string();

    let records = create_test_records(neuron_count, records_per_neuron);
    write_records_to_parquet(&parquet_file, &records).expect("Failed to write parquet");

    let mut group = c.benchmark_group("compressed_vs_uncompressed");

    // Benchmark uncompressed LRU cache
    group.bench_function("uncompressed_sequential", |b| {
        let cache = LruRecordCache::new(&parquet_file, capacity)
            .expect("Failed to create uncompressed cache");
        b.iter(|| {
            for i in 0..neuron_count {
                let uuid = format!("neuron-{i}");
                let records = cache.get(&uuid).expect("Failed to get records");
                black_box(records.len());
            }
        });
    });

    // Benchmark compressed LRU cache
    group.bench_function("compressed_sequential", |b| {
        let cache = CompressedLruRecordCache::new(&parquet_file, capacity)
            .expect("Failed to create compressed cache");
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

/// Benchmark comparing memory usage with eviction pressure.
fn benchmark_memory_pressure(c: &mut Criterion) {
    let neuron_count = 100;
    let records_per_neuron = 200;

    let temp_dir = tempdir().expect("Failed to create temp directory");
    let parquet_path = temp_dir.path().join("pressure_benchmark.parquet");
    let parquet_file = parquet_path.to_str().unwrap().to_string();

    let records = create_test_records(neuron_count, records_per_neuron);
    write_records_to_parquet(&parquet_file, &records).expect("Failed to write parquet");

    let mut group = c.benchmark_group("memory_pressure");

    // Very constrained capacity — forces frequent evictions
    for capacity_kb in [100, 500, 2000] {
        let capacity = capacity_kb * 1024;

        group.bench_with_input(
            BenchmarkId::new("uncompressed", format!("{capacity_kb}KB")),
            &capacity,
            |b, &cap| {
                let cache =
                    LruRecordCache::new(&parquet_file, cap).expect("Failed to create cache");
                b.iter(|| {
                    for i in 0..neuron_count {
                        let uuid = format!("neuron-{i}");
                        let records = cache.get(&uuid).expect("Failed to get records");
                        black_box(records.len());
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("compressed", format!("{capacity_kb}KB")),
            &capacity,
            |b, &cap| {
                let cache = CompressedLruRecordCache::new(&parquet_file, cap)
                    .expect("Failed to create cache");
                b.iter(|| {
                    for i in 0..neuron_count {
                        let uuid = format!("neuron-{i}");
                        let records = cache.get(&uuid).expect("Failed to get records");
                        black_box(records.len());
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_compressed_vs_uncompressed,
    benchmark_memory_pressure
);
criterion_main!(benches);
