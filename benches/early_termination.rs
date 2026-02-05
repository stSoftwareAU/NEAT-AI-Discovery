//! Benchmark for Issue #429: Early termination improvements for low-value candidates.
//!
//! This benchmark measures the performance improvements from:
//! 1. Hierarchical candidate filtering (quick pre-filter)
//! 2. Incremental confidence early termination
//! 3. Cross-module deduplication
//!
//! ## Purpose
//!
//! Measure before/after performance for candidate analysis on large creatures to verify
//! that the early termination improvements achieve the target 30-50% reduction in
//! analysis time.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use neat_ai_discovery::analysis::{
    analyze_synapses_with_cache_and_gpu_queue,
    cache::RecordCache,
    gpu::{GpuAnalyzer, GpuWorkQueue},
};
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{AnalyzeSynapsesInput, CreatureJson, NeuronJson, SynapseJson};
use std::sync::Arc;
use tempfile::tempdir;

/// Configuration for benchmark scenarios.
struct BenchmarkConfig {
    /// Number of input neurons
    input_count: usize,
    /// Number of hidden neurons
    hidden_count: usize,
    /// Number of output neurons
    output_count: usize,
    /// Number of synapses to create
    synapse_count: usize,
    /// Number of records to generate
    record_count: usize,
    /// Name for the benchmark
    name: &'static str,
}

impl BenchmarkConfig {
    /// Small creature: Baseline performance test
    fn small() -> Self {
        Self {
            input_count: 50,
            hidden_count: 10,
            output_count: 1,
            synapse_count: 100,
            record_count: 100,
            name: "small",
        }
    }

    /// Medium creature: Typical real-world scenario
    fn medium() -> Self {
        Self {
            input_count: 200,
            hidden_count: 50,
            output_count: 5,
            synapse_count: 500,
            record_count: 200,
            name: "medium",
        }
    }

    /// Large creature: Target for early termination benefits
    fn large() -> Self {
        Self {
            input_count: 500,
            hidden_count: 100,
            output_count: 10,
            synapse_count: 1500,
            record_count: 300,
            name: "large",
        }
    }
}

/// Create a test creature with specified configuration.
fn create_test_creature(config: &BenchmarkConfig) -> CreatureJson {
    let mut neurons = Vec::new();
    let mut synapses = Vec::new();

    // Create hidden neurons
    for i in 0..config.hidden_count {
        neurons.push(NeuronJson {
            uuid: format!("hidden-{i}"),
            neuron_type: "hidden".to_string(),
            squash: "TANH".to_string(),
            bias: 0.0,
        });
    }

    // Create output neurons
    for i in 0..config.output_count {
        neurons.push(NeuronJson {
            uuid: format!("output-{i}"),
            neuron_type: "output".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        });
    }

    // Create synapses connecting inputs to hidden and hidden to outputs
    let mut synapse_idx = 0;

    // Input -> Hidden connections
    for hidden_idx in 0..config.hidden_count {
        let inputs_per_hidden = (config.synapse_count / 2) / config.hidden_count.max(1);
        for i in 0..inputs_per_hidden {
            if synapse_idx >= config.synapse_count {
                break;
            }
            let input_idx = (hidden_idx * inputs_per_hidden + i) % config.input_count;
            synapses.push(SynapseJson {
                from_uuid: format!("input-{input_idx}"),
                to_uuid: format!("hidden-{hidden_idx}"),
                weight: ((synapse_idx as f32 * 0.1) % 1.0) - 0.5,
                synapse_type: None,
            });
            synapse_idx += 1;
        }
    }

    // Hidden -> Output connections
    for output_idx in 0..config.output_count {
        let hidden_per_output = (config.synapse_count / 2) / config.output_count.max(1);
        for i in 0..hidden_per_output {
            if synapse_idx >= config.synapse_count {
                break;
            }
            let hidden_idx = (output_idx * hidden_per_output + i) % config.hidden_count;
            synapses.push(SynapseJson {
                from_uuid: format!("hidden-{hidden_idx}"),
                to_uuid: format!("output-{output_idx}"),
                weight: ((synapse_idx as f32 * 0.1) % 1.0) - 0.5,
                synapse_type: None,
            });
            synapse_idx += 1;
        }
    }

    CreatureJson {
        input: config.input_count,
        output: config.output_count,
        neurons,
        synapses,
    }
}

/// Create test records with varying quality candidates.
///
/// This creates records where:
/// - Some inputs strongly correlate with output error (high-value candidates)
/// - Some inputs weakly correlate (medium-value candidates)
/// - Some inputs are near-random (low-value candidates - early termination targets)
fn create_test_records(config: &BenchmarkConfig) -> Vec<DiscoverRecord> {
    let capacity =
        (config.input_count + config.hidden_count + config.output_count) * config.record_count;
    let mut records = Vec::with_capacity(capacity);

    for obs_index in 0..config.record_count as u32 {
        // Create input neuron records with varying correlation patterns
        for input_idx in 0..config.input_count {
            // Divide inputs into three categories:
            // - First third: Strong correlation with error (high-value)
            // - Second third: Weak correlation (medium-value)
            // - Last third: Random/noise (low-value - early termination targets)
            let activation = if input_idx < config.input_count / 3 {
                // Strong correlation: follows error pattern
                if obs_index % 2 == 0 {
                    0.8
                } else {
                    -0.8
                }
            } else if input_idx < 2 * config.input_count / 3 {
                // Weak correlation: noisy pattern
                let base = if obs_index % 2 == 0 { 0.3 } else { -0.3 };
                let noise = ((obs_index as f32 + input_idx as f32) % 0.4) - 0.2;
                base + noise
            } else {
                // Random: no correlation (should be filtered early)
                ((obs_index as f32 * 0.123 + input_idx as f32 * 0.456) % 2.0) - 1.0
            };

            records.push(DiscoverRecord::new(
                obs_index,
                format!("input-{input_idx}"),
                None,
                activation,
                Vec::new(),
            ));
        }

        // Create hidden neuron records
        for hidden_idx in 0..config.hidden_count {
            let activation = ((obs_index as f32 + hidden_idx as f32) % 2.0) - 1.0;
            records.push(DiscoverRecord::new(
                obs_index,
                format!("hidden-{hidden_idx}"),
                Some(activation),
                activation.tanh(),
                vec![0.1 * ((hidden_idx as f32 % 3.0) - 1.0)],
            ));
        }

        // Create output neuron records with error correlated to "good" inputs
        for output_idx in 0..config.output_count {
            let error = if obs_index % 2 == 0 { 0.3 } else { -0.3 };
            records.push(DiscoverRecord::new(
                obs_index,
                format!("output-{output_idx}"),
                Some(0.5),
                0.5,
                vec![error],
            ));
        }
    }

    records
}

/// Benchmark synapse analysis with early termination improvements.
fn benchmark_early_termination(c: &mut Criterion) {
    // Skip benchmark if no GPU available
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping benchmark: no GPU available");
        return;
    }

    let configs = [
        BenchmarkConfig::small(),
        BenchmarkConfig::medium(),
        BenchmarkConfig::large(),
    ];

    let mut group = c.benchmark_group("early_termination");
    // Set measurement time for more accurate results
    group.measurement_time(std::time::Duration::from_secs(30));
    group.sample_size(10);

    for config in &configs {
        let temp_dir = tempdir().expect("Failed to create temp directory");
        let parquet_path = temp_dir.path().join("records.parquet");
        let parquet_file = parquet_path.to_str().unwrap().to_string();

        // Create and write test data
        let creature = create_test_creature(config);
        let records = create_test_records(config);
        neat_ai_discovery::parquet_format::write_records_to_parquet(&parquet_file, &records)
            .expect("Failed to write parquet");

        // Create cache and GPU queue once to reuse across iterations
        let cache =
            Arc::new(RecordCache::new_adaptive(&parquet_file).expect("Failed to create cache"));
        let gpu_queue = Arc::new(GpuWorkQueue::new().expect("Failed to create GPU queue"));

        // Get focus neurons (outputs for synapse analysis)
        let focus_neurons: Vec<String> = (0..config.output_count)
            .map(|i| format!("output-{i}"))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("analysis", config.name),
            &config.name,
            |b, _| {
                b.iter(|| {
                    let input = AnalyzeSynapsesInput {
                        parquet_file: parquet_file.clone(),
                        creature: creature.clone(),
                        focus_neurons: focus_neurons.clone(),
                        max_candidates: Some(50),
                        analysis_deadline_ms: Some(60_000), // 1 minute deadline
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

/// Benchmark specifically measuring the effect of early termination on low-value candidates.
///
/// This creates a scenario with many low-value candidates that should be filtered early,
/// allowing us to measure the effectiveness of the early termination improvements.
fn benchmark_low_value_filtering(c: &mut Criterion) {
    // Skip benchmark if no GPU available
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping benchmark: no GPU available");
        return;
    }

    let mut group = c.benchmark_group("low_value_filtering");
    group.measurement_time(std::time::Duration::from_secs(30));
    group.sample_size(10);

    // Create a scenario with mostly low-value candidates
    let config = BenchmarkConfig {
        input_count: 300,
        hidden_count: 20,
        output_count: 5,
        synapse_count: 100,
        record_count: 200,
        name: "mostly_low_value",
    };

    let temp_dir = tempdir().expect("Failed to create temp directory");
    let parquet_path = temp_dir.path().join("records.parquet");
    let parquet_file = parquet_path.to_str().unwrap().to_string();

    let creature = create_test_creature(&config);

    // Create records where most inputs are low-value (random correlation)
    let mut records = Vec::new();
    for obs_index in 0..config.record_count as u32 {
        // 90% of inputs are random/noise
        for input_idx in 0..config.input_count {
            let activation = if input_idx < config.input_count / 10 {
                // Only 10% are high-value
                if obs_index % 2 == 0 {
                    0.8
                } else {
                    -0.8
                }
            } else {
                // 90% are random (low-value)
                ((obs_index as f32 * 0.789 + input_idx as f32 * 0.234) % 2.0) - 1.0
            };

            records.push(DiscoverRecord::new(
                obs_index,
                format!("input-{input_idx}"),
                None,
                activation,
                Vec::new(),
            ));
        }

        // Hidden neurons
        for hidden_idx in 0..config.hidden_count {
            records.push(DiscoverRecord::new(
                obs_index,
                format!("hidden-{hidden_idx}"),
                Some(0.5),
                0.5,
                vec![0.05],
            ));
        }

        // Output neurons with error
        for output_idx in 0..config.output_count {
            let error = if obs_index % 2 == 0 { 0.3 } else { -0.3 };
            records.push(DiscoverRecord::new(
                obs_index,
                format!("output-{output_idx}"),
                Some(0.5),
                0.5,
                vec![error],
            ));
        }
    }

    neat_ai_discovery::parquet_format::write_records_to_parquet(&parquet_file, &records)
        .expect("Failed to write parquet");

    let cache = Arc::new(RecordCache::new_adaptive(&parquet_file).expect("Failed to create cache"));
    let gpu_queue = Arc::new(GpuWorkQueue::new().expect("Failed to create GPU queue"));

    let focus_neurons: Vec<String> = (0..config.output_count)
        .map(|i| format!("output-{i}"))
        .collect();

    group.bench_function("analysis_with_early_termination", |b| {
        b.iter(|| {
            let input = AnalyzeSynapsesInput {
                parquet_file: parquet_file.clone(),
                creature: creature.clone(),
                focus_neurons: focus_neurons.clone(),
                max_candidates: Some(20),
                analysis_deadline_ms: Some(60_000),
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
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_early_termination,
    benchmark_low_value_filtering
);
criterion_main!(benches);
