//! Benchmark for Issue #221: Sample locality batching performance.
//!
//! This benchmark measures the performance improvement from batching sources
//! that share the same observation indices.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use neat_ai_discovery::analysis::{GpuAnalyzer, analyze_synapses};
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{AnalyzeSynapsesInput, CreatureJson, NeuronJson};
use std::hint::black_box;
use tempfile::tempdir;

/// Create a test creature with multiple inputs and a single output.
fn create_correlated_input_creature(input_count: usize) -> CreatureJson {
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

/// Create test records where all inputs share the same `obs_indices`.
fn create_fully_correlated_records(input_count: usize, record_count: usize) -> Vec<DiscoverRecord> {
    let mut records = Vec::with_capacity((input_count + 1) * record_count);

    for obs_index in 0..record_count as u32 {
        // All inputs share the same obs_indices
        for input_idx in 0..input_count {
            let activation = ((obs_index as f32 + input_idx as f32) % 10.0 - 5.0) / 5.0;
            records.push(DiscoverRecord::new(
                obs_index,
                format!("input-{input_idx}"),
                None,
                activation,
                Vec::new(),
            ));
        }

        // Output neuron with correlated error
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

/// Create test records where inputs have partial overlap in `obs_indices`.
fn create_partially_correlated_records(
    input_count: usize,
    record_count: usize,
    overlap_fraction: f32,
) -> Vec<DiscoverRecord> {
    let mut records = Vec::new();
    let overlap_count = (record_count as f32 * overlap_fraction) as usize;

    for obs_index in 0..record_count as u32 {
        // Output neuron records - all obs_indices
        let error = if obs_index % 2 == 0 { 0.3 } else { -0.3 };
        records.push(DiscoverRecord::new(
            obs_index,
            "output-0".to_string(),
            Some(0.5),
            0.5,
            vec![error],
        ));
    }

    // Create input records with partial overlap
    for input_idx in 0..input_count {
        // First half of inputs share obs_indices 0..overlap_count
        // Second half share obs_indices overlap_count..record_count
        let start_obs = if input_idx < input_count / 2 {
            0
        } else {
            overlap_count
        };
        let end_obs = if input_idx < input_count / 2 {
            overlap_count
        } else {
            record_count
        };

        for obs_index in start_obs..end_obs {
            let activation = ((obs_index as f32 + input_idx as f32) % 10.0 - 5.0) / 5.0;
            records.push(DiscoverRecord::new(
                obs_index as u32,
                format!("input-{input_idx}"),
                None,
                activation,
                Vec::new(),
            ));
        }
    }

    records
}

fn benchmark_sample_locality(c: &mut Criterion) {
    // Skip benchmark if no GPU available
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping benchmark: no GPU available");
        return;
    }

    let input_count = 100;
    let record_count = 100;

    let mut group = c.benchmark_group("sample_locality");

    // Benchmark with fully correlated records (all inputs share same obs_indices)
    {
        let temp_dir = tempdir().expect("Failed to create temp directory");
        let parquet_path = temp_dir.path().join("records.parquet");
        let parquet_file = parquet_path.to_str().unwrap().to_string();

        let records = create_fully_correlated_records(input_count, record_count);
        neat_ai_discovery::parquet_format::write_records_to_parquet(&parquet_file, &records)
            .expect("Failed to write parquet");

        let creature = create_correlated_input_creature(input_count);

        group.bench_function("fully_correlated", |b| {
            b.iter(|| {
                let input = AnalyzeSynapsesInput {
                    parquet_file: parquet_file.clone(),
                    creature: creature.clone(),
                    focus_neurons: vec!["output-0".to_string()],
                    max_candidates: Some(10),
                    analysis_deadline_ms: None,
                    random_seed: Some(42),
                };
                let result = analyze_synapses(&input).expect("Analysis should succeed");
                black_box(result);
            });
        });
    }

    // Benchmark with 90% overlap
    {
        let temp_dir = tempdir().expect("Failed to create temp directory");
        let parquet_path = temp_dir.path().join("records.parquet");
        let parquet_file = parquet_path.to_str().unwrap().to_string();

        let records = create_partially_correlated_records(input_count, record_count, 0.9);
        neat_ai_discovery::parquet_format::write_records_to_parquet(&parquet_file, &records)
            .expect("Failed to write parquet");

        let creature = create_correlated_input_creature(input_count);

        group.bench_with_input(BenchmarkId::new("partial_overlap", "90%"), &0.9, |b, _| {
            b.iter(|| {
                let input = AnalyzeSynapsesInput {
                    parquet_file: parquet_file.clone(),
                    creature: creature.clone(),
                    focus_neurons: vec!["output-0".to_string()],
                    max_candidates: Some(10),
                    analysis_deadline_ms: None,
                    random_seed: Some(42),
                };
                let result = analyze_synapses(&input).expect("Analysis should succeed");
                black_box(result);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark_sample_locality);
criterion_main!(benches);
