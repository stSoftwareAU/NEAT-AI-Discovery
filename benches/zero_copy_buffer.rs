//! Benchmark for Issue #228: Zero-copy buffer sharing performance.
//!
//! This benchmark compares the performance of zero-copy vs traditional copying
//! on unified memory architectures (Apple Silicon).

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use criterion::{Criterion, criterion_group, criterion_main};
use neat_ai_discovery::analysis::{GpuAnalyzer, analyze_synapses, supports_unified_memory};
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{AnalyzeSynapsesInput, CreatureJson, NeuronJson};
use std::hint::black_box;
use tempfile::tempdir;

fn benchmark_zero_copy_vs_copying(c: &mut Criterion) {
    // Skip benchmark if no GPU available
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping benchmark: no GPU available");
        return;
    }

    let temp_dir = tempdir().expect("Failed to create temp directory");
    let parquet_path = temp_dir.path().join("records.parquet");
    let parquet_file = parquet_path.to_str().unwrap().to_string();

    // Create a large dataset for meaningful benchmarking
    let sample_count = 100_000;
    let mut records = Vec::with_capacity(sample_count * 2);

    for obs_index in 0..sample_count as u32 {
        let input_activation = ((obs_index as f32) % 100.0 - 50.0) / 50.0;

        records.push(DiscoverRecord::new(
            obs_index,
            "input-0".to_string(),
            None,
            input_activation,
            Vec::new(),
        ));

        let error = input_activation * 0.2;
        records.push(DiscoverRecord::new(
            obs_index,
            "output-0".to_string(),
            Some(0.5),
            0.5,
            vec![error],
        ));
    }

    write_records_to_parquet(&parquet_file, &records).expect("Failed to write parquet");

    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![NeuronJson {
            uuid: "output-0".to_string(),
            neuron_type: "output".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        }],
        synapses: Vec::new(),
    };

    let mut group = c.benchmark_group("zero_copy_buffer");

    // Benchmark with zero-copy disabled (traditional copying)
    group.bench_function("with_copying", |b| {
        // SAFETY: Benchmarks run single-threaded, no concurrent env access.
        unsafe { std::env::set_var("NEAT_AI_DISCOVERY_ZERO_COPY", "0") };
        b.iter(|| {
            let input = AnalyzeSynapsesInput {
                parquet_file: parquet_file.clone(),
                creature: creature.clone(),
                focus_neurons: vec!["output-0".to_string()],
                max_candidates: Some(10),
                analysis_deadline_ms: None,
                random_seed: Some(42),
                temperature: 1.0,
                failure_cache: None,
            };
            let result = analyze_synapses(&input).expect("Analysis should succeed");
            black_box(result);
        });
        unsafe { std::env::remove_var("NEAT_AI_DISCOVERY_ZERO_COPY") };
    });

    // Benchmark with zero-copy enabled
    if supports_unified_memory() {
        group.bench_function("with_zero_copy", |b| {
            // SAFETY: Benchmarks run single-threaded, no concurrent env access.
            unsafe { std::env::set_var("NEAT_AI_DISCOVERY_ZERO_COPY", "1") };
            b.iter(|| {
                let input = AnalyzeSynapsesInput {
                    parquet_file: parquet_file.clone(),
                    creature: creature.clone(),
                    focus_neurons: vec!["output-0".to_string()],
                    max_candidates: Some(10),
                    analysis_deadline_ms: None,
                    random_seed: Some(42),
                    temperature: 1.0,
                    failure_cache: None,
                };
                let result = analyze_synapses(&input).expect("Analysis should succeed");
                black_box(result);
            });
            unsafe { std::env::remove_var("NEAT_AI_DISCOVERY_ZERO_COPY") };
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark_zero_copy_vs_copying);
criterion_main!(benches);
