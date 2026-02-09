//! Benchmark for Issue #419: Parallel discovery module execution.
//!
//! Measures the wall-clock time for the complete `analyze_all()` pipeline
//! with synapse analysis enabled (which includes all 25 discovery modules).
//! The benchmark uses creatures with hidden neurons to activate most
//! detection modules.
//!
//! Run with: `cargo bench --bench parallel_discovery`

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use neat_ai_discovery::analysis::analyze_all;
use neat_ai_discovery::analysis::gpu::GpuAnalyzer;
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{AnalyzeAllInput, CreatureJson, NeuronJson, SynapseJson};
use tempfile::tempdir;

/// Create a test creature with the specified number of hidden neurons.
///
/// Structure: 2 inputs → N hidden neurons → 1 output, fully connected.
fn create_benchmark_creature(num_hidden: usize) -> CreatureJson {
    let mut neurons = Vec::new();

    // Input neurons (listed in neurons for discovery)
    neurons.push(NeuronJson {
        uuid: "input-0".to_string(),
        neuron_type: "input".to_string(),
        squash: "IDENTITY".to_string(),
        bias: 0.0,
    });
    neurons.push(NeuronJson {
        uuid: "input-1".to_string(),
        neuron_type: "input".to_string(),
        squash: "IDENTITY".to_string(),
        bias: 0.0,
    });

    // Hidden neurons with varying activation functions
    let squash_fns = ["RELU", "TANH", "IDENTITY", "LOGISTIC"];
    for i in 0..num_hidden {
        neurons.push(NeuronJson {
            uuid: format!("hidden-{i}"),
            neuron_type: "hidden".to_string(),
            squash: squash_fns[i % squash_fns.len()].to_string(),
            bias: if i % 3 == 0 { 5.0 } else { 0.0 }, // some with high bias for saturation
        });
    }

    // Output neuron
    neurons.push(NeuronJson {
        uuid: "output-0".to_string(),
        neuron_type: "output".to_string(),
        squash: "IDENTITY".to_string(),
        bias: 0.0,
    });

    // Build synapses: input → hidden, hidden → output
    let mut synapses = Vec::new();
    for i in 0..num_hidden {
        synapses.push(SynapseJson {
            from_uuid: "input-0".to_string(),
            to_uuid: format!("hidden-{i}"),
            weight: 0.5 / (i as f32 + 1.0),
            synapse_type: None,
        });
        synapses.push(SynapseJson {
            from_uuid: "input-1".to_string(),
            to_uuid: format!("hidden-{i}"),
            weight: 0.3 / (i as f32 + 1.0),
            synapse_type: None,
        });
        synapses.push(SynapseJson {
            from_uuid: format!("hidden-{i}"),
            to_uuid: "output-0".to_string(),
            weight: 0.8 / (i as f32 + 1.0),
            synapse_type: None,
        });
    }

    CreatureJson {
        neurons,
        synapses,
        input: 2,
        output: 1,
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
                    // Some dead, some saturated, some active
                    let act = if neuron.bias > 1.0 {
                        0.999 // saturated
                    } else if neuron.uuid.ends_with("-0") {
                        0.0 // dead
                    } else {
                        0.3 + 0.4 * (t * std::f32::consts::TAU).sin()
                    };
                    (act, Some(act), vec![0.01])
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

fn bench_parallel_discovery(c: &mut Criterion) {
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping benchmark: no GPU available");
        return;
    }

    let mut group = c.benchmark_group("parallel_discovery");

    // Test with different creature sizes
    let sizes: Vec<(usize, usize, &str)> = vec![
        (5, 100, "5h_100r"),
        (20, 200, "20h_200r"),
        (50, 200, "50h_200r"),
    ];

    for (num_hidden, records_per_neuron, label) in sizes {
        let creature = create_benchmark_creature(num_hidden);
        let records = create_benchmark_records(&creature, records_per_neuron);

        let temp_dir = tempdir().expect("Failed to create temp directory");
        let parquet_path = temp_dir.path().join("bench.parquet");
        let parquet_file = parquet_path.to_str().unwrap().to_string();

        write_records_to_parquet(&parquet_file, &records).expect("Failed to write parquet");

        let focus_neurons: Vec<String> = creature
            .neurons
            .iter()
            .filter(|n| n.neuron_type == "output")
            .map(|n| n.uuid.clone())
            .collect();

        group.bench_with_input(
            BenchmarkId::new("analyze_all", label),
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
                    };
                    black_box(analyze_all(&input).expect("analyze_all failed"))
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_parallel_discovery);
criterion_main!(benches);
