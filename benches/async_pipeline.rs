//! Benchmark for Issue #568: Async pipeline CPU/GPU overlap.
//!
//! Measures the end-to-end analysis pipeline throughput to determine whether
//! overlapping CPU analysis with GPU computation provides measurable improvement.
//!
//! Run with: `cargo bench --bench async_pipeline`

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use neat_ai_discovery::analysis::analyze_all;
use neat_ai_discovery::analysis::gpu::GpuAnalyzer;
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{AnalyzeAllInput, CreatureJson, NeuronJson, SynapseJson};
use std::hint::black_box;
use tempfile::tempdir;

/// Create a test creature with the specified number of hidden neurons.
///
/// Structure: 3 inputs → N hidden neurons → 2 outputs, fully connected.
/// Larger than `parallel_discovery` benchmark to increase GPU/CPU work ratio.
fn create_benchmark_creature(num_hidden: usize) -> CreatureJson {
    let mut neurons = Vec::new();

    // Input neurons
    for i in 0..3 {
        neurons.push(NeuronJson {
            uuid: format!("input-{i}"),
            neuron_type: "input".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        });
    }

    // Hidden neurons with varying activation functions
    let squash_fns = ["RELU", "TANH", "IDENTITY", "LOGISTIC", "HARD_TANH"];
    for i in 0..num_hidden {
        neurons.push(NeuronJson {
            uuid: format!("hidden-{i}"),
            neuron_type: "hidden".to_string(),
            squash: squash_fns[i % squash_fns.len()].to_string(),
            bias: if i % 4 == 0 { 3.0 } else { 0.1 },
        });
    }

    // Output neurons
    for i in 0..2 {
        neurons.push(NeuronJson {
            uuid: format!("output-{i}"),
            neuron_type: "output".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        });
    }

    // Build synapses: input → hidden, hidden → output
    let mut synapses = Vec::new();
    for i in 0..num_hidden {
        for j in 0..3 {
            synapses.push(SynapseJson {
                from_uuid: format!("input-{j}"),
                to_uuid: format!("hidden-{i}"),
                weight: 0.3 / (i as f32 + 1.0),
                synapse_type: None,
            });
        }
        for j in 0..2 {
            synapses.push(SynapseJson {
                from_uuid: format!("hidden-{i}"),
                to_uuid: format!("output-{j}"),
                weight: 0.5 / (i as f32 + 1.0),
                synapse_type: None,
            });
        }
    }

    CreatureJson {
        neurons,
        synapses,
        input: 3,
        output: 2,
    }
}

/// Create parquet records with patterns that trigger multiple detection modules.
fn create_benchmark_records(
    creature: &CreatureJson,
    records_per_neuron: usize,
) -> Vec<DiscoverRecord> {
    let mut records = Vec::new();

    for obs in 0..records_per_neuron as u32 {
        let t = obs as f32 / records_per_neuron as f32;

        for neuron in &creature.neurons {
            let (activation, value, errors) = match neuron.neuron_type.as_str() {
                "input" => ((t * std::f32::consts::TAU).sin(), None, Vec::new()),
                "hidden" => {
                    let act = if neuron.bias > 1.0 {
                        0.999 // saturated
                    } else if neuron.uuid.ends_with("-0") {
                        0.0 // dead
                    } else {
                        0.3 + 0.4 * (t * std::f32::consts::TAU).sin()
                    };
                    (act, Some(act), vec![0.01 * (t * 5.0).cos()])
                }
                "output" => {
                    let error = 0.1 * (t * std::f32::consts::PI).cos();
                    (0.5 + 0.2 * t, Some(0.5 + 0.2 * t), vec![error])
                }
                _ => continue,
            };

            records.push(DiscoverRecord {
                obs_index: obs,
                neuron_uuid: neuron.uuid.clone(),
                value,
                activation,
                errors,
            });
        }
    }

    records
}

fn bench_async_pipeline(c: &mut Criterion) {
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping benchmark: no GPU available");
        return;
    }

    let mut group = c.benchmark_group("async_pipeline");
    // Use fewer iterations for a large benchmark — wall-clock time is the focus.
    group.sample_size(10);

    // Test with different creature sizes to see where CPU/GPU overlap matters.
    let sizes: Vec<(usize, usize, &str)> = vec![(10, 200, "10h_200r"), (30, 300, "30h_300r")];

    for (num_hidden, records_per_neuron, label) in sizes {
        let creature = create_benchmark_creature(num_hidden);
        let records = create_benchmark_records(&creature, records_per_neuron);

        let temp_dir = tempdir().expect("Failed to create temp directory");
        let parquet_path = temp_dir.path().join("bench.parquet");
        let parquet_file = parquet_path.to_str().unwrap().to_string();

        write_records_to_parquet(&parquet_file, &records).expect("Failed to write parquet");

        // Focus on output + some hidden neurons to exercise more GPU work
        let focus_neurons: Vec<String> = creature
            .neurons
            .iter()
            .filter(|n| n.neuron_type == "output" || n.neuron_type == "hidden")
            .map(|n| n.uuid.clone())
            .collect();

        group.bench_with_input(
            BenchmarkId::new("analyze_all_synapse", label),
            &parquet_file,
            |b, pf| {
                b.iter(|| {
                    let input = AnalyzeAllInput {
                        parquet_file: pf.clone(),
                        creature: creature.clone(),
                        focus_neurons: focus_neurons.clone(),
                        max_synapse_candidates: None,
                        max_neuron_candidates: None,
                        analysis_deadline_ms: None,
                        include_synapse_analysis: Some(true),
                        include_neuron_analysis: Some(false),
                        random_seed: Some(42),
                        previous_neuron_fingerprints: None,
                        module_outcome_tracker: None,
                        max_analysis_memory_mb: None,
                        max_discovery_wall_clock_minutes: None,
                        temperature: 1.0,
                        failure_cache: None,
                        discovery_outcome_log: None,
                        cost_name: None,
                    };
                    black_box(analyze_all(&input).expect("analyze_all failed"))
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_async_pipeline);
criterion_main!(benches);
