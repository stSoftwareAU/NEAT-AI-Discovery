//! Benchmark for Issue #196: Cache locality with varying input counts.
//!
//! This benchmark measures analysis performance across different input counts
//! to verify that the implementation scales efficiently.
//!
//! ## Key Findings
//!
//! 1. **Pre-loading**: The `RecordCache` pre-loads ALL parquet records into memory at
//!    startup via `read_all_records_grouped_by_neuron()`. This means no disk I/O occurs
//!    during the analysis loop.
//!
//! 2. **`HashMap` access**: After pre-loading, cache access is O(1) `HashMap` lookups
//!    (`cache.get(source_uuid)`). The order of access doesn't affect performance because
//!    we're not doing sequential disk reads.
//!
//! 3. **Sub-linear scaling**: Benchmarks show scaling factor ~0.19 (sub-linear), meaning
//!    time per input *decreases* as input count increases. This is because fixed overhead
//!    (GPU init, parquet loading) is amortized across more inputs.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use neat_ai_discovery::analysis::{
    analyze_synapses_with_cache_and_gpu_queue,
    cache::RecordCache,
    gpu::{GpuAnalyzer, GpuWorkQueue},
};
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{AnalyzeSynapsesInput, CreatureJson, NeuronJson};
use std::sync::Arc;
use tempfile::tempdir;

/// Create a test creature with the specified number of inputs and a single output.
fn create_test_creature(input_count: usize) -> CreatureJson {
    CreatureJson {
        input: input_count,
        output: 1,
        neurons: vec![NeuronJson {
            uuid: "output-0".to_string(),
            neuron_type: "output".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        }],
        synapses: Vec::new(),
    }
}

/// Create test records for a creature with the specified number of inputs.
/// Records are created with correlated error patterns to ensure candidates are found.
fn create_test_records(input_count: usize, record_count: usize) -> Vec<DiscoverRecord> {
    let mut records = Vec::with_capacity((input_count + 1) * record_count);

    for obs_index in 0..record_count as u32 {
        // Create input neurons with varying activations
        for input_idx in 0..input_count {
            let activation = ((obs_index as f32 + input_idx as f32) % 20.0 - 10.0) / 10.0;
            records.push(DiscoverRecord::new(
                obs_index,
                format!("input-{input_idx}"),
                None,
                activation,
                Vec::new(),
            ));
        }

        // Output neuron with error correlated to some inputs
        let error = if obs_index % 2 == 0 { 0.3 } else { -0.3 };
        records.push(DiscoverRecord::new(
            obs_index,
            "output-0".to_string(),
            Some(0.5),
            0.5,
            vec![error],
        ));
    }

    records
}

fn benchmark_cache_locality(c: &mut Criterion) {
    // Skip benchmark if no GPU available
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping benchmark: no GPU available");
        return;
    }

    let input_counts = [100, 500, 1000, 2000];
    let record_count = 50; // Enough records for meaningful analysis

    let mut group = c.benchmark_group("cache_locality");

    for &input_count in &input_counts {
        // Set deadline: 10 minutes per iteration for large input counts, 5 minutes for smaller
        let deadline_ms = if input_count > 1000 {
            Some(10 * 60 * 1000) // 10 minutes
        } else {
            Some(5 * 60 * 1000) // 5 minutes
        };

        let temp_dir = tempdir().expect("Failed to create temp directory");
        let parquet_path = temp_dir.path().join("records.parquet");
        let parquet_file = parquet_path.to_str().unwrap().to_string();

        // Create and write test data
        let records = create_test_records(input_count, record_count);
        neat_ai_discovery::parquet_format::write_records_to_parquet(&parquet_file, &records)
            .expect("Failed to write parquet");

        let creature = create_test_creature(input_count);

        // Create cache and GPU queue once to reuse across all iterations
        // This eliminates ~100ms GPU initialization overhead per iteration
        let cache =
            Arc::new(RecordCache::new_adaptive(&parquet_file).expect("Failed to create cache"));
        let gpu_queue = Arc::new(GpuWorkQueue::new().expect("Failed to create GPU queue"));

        group.bench_with_input(
            BenchmarkId::from_parameter(input_count),
            &input_count,
            |b, &_input_count| {
                b.iter(|| {
                    let input = AnalyzeSynapsesInput {
                        parquet_file: parquet_file.clone(),
                        creature: creature.clone(),
                        focus_neurons: vec!["output-0".to_string()],
                        max_candidates: Some(10), // Limit candidates to reduce GPU time variance
                        analysis_deadline_ms: deadline_ms,
                        random_seed: Some(42),
                    };

                    let result = analyze_synapses_with_cache_and_gpu_queue(
                        &input,
                        Arc::clone(&cache),
                        Arc::clone(&gpu_queue),
                    )
                    .expect("Analysis should succeed");
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, benchmark_cache_locality);
criterion_main!(benches);
