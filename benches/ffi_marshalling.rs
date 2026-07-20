//! Benchmark for Issue #1040: FFI JSON marshalling overhead.
//!
//! The FFI boundary serialises/deserialises large JSON payloads for every
//! analysis call. This benchmark measures that overhead for varying result
//! set sizes (10, 100, 1000 candidates).
//!
//! ## Key Metrics
//!
//! 1. **Serialisation time**: `AnalyzeParallelOutput` → JSON string
//! 2. **Deserialisation time**: JSON string → `AnalyzeParallelInput`
//! 3. **Round-trip**: Serialise output + deserialise input for a full FFI call

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)] // Intentional numeric casts for benchmark data generation (Issue #873)

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use neat_ai_discovery::{
    AnalyzeParallelOutput, CandidateNeuronJson, CandidateSynapseJson,
    CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson, NeuronJson,
    NeuronStatsJson, SynapseJson, SynapseWeightUpdateCandidateJson,
};
use std::hint::black_box;

/// Create a synthetic creature with the given number of hidden neurons.
fn create_test_creature(num_hidden: usize) -> CreatureJson {
    let mut neurons = Vec::with_capacity(num_hidden + 1);
    for i in 0..num_hidden {
        neurons.push(NeuronJson {
            uuid: format!("hidden-{i}"),
            neuron_type: "hidden".to_string(),
            squash: "TANH".to_string(),
            bias: (i as f32) * 0.01,
        });
    }
    neurons.push(NeuronJson {
        uuid: "output-0".to_string(),
        neuron_type: "output".to_string(),
        squash: "IDENTITY".to_string(),
        bias: 0.0,
    });

    let mut synapses = Vec::with_capacity(num_hidden * 2);
    for i in 0..num_hidden {
        synapses.push(SynapseJson {
            from_uuid: format!("input-{}", i % 3),
            to_uuid: format!("hidden-{i}"),
            weight: 0.5 + (i as f32) * 0.01,
            ..Default::default()
        });
        synapses.push(SynapseJson {
            from_uuid: format!("hidden-{i}"),
            to_uuid: "output-0".to_string(),
            weight: 0.3 - (i as f32) * 0.005,
            ..Default::default()
        });
    }

    CreatureJson {
        neurons,
        synapses,
        input: 3,
        output: 1,
    }
}

/// Generate synthetic synapse candidates.
fn create_synapse_candidates(count: usize) -> Vec<CandidateSynapseJson> {
    (0..count)
        .map(|i| CandidateSynapseJson {
            from_neuron_uuid: format!("input-{}", i % 3),
            to_neuron_uuid: format!("hidden-{}", i % 50),
            from_neuron_index: Some(i % 3),
            to_neuron_index: Some(3 + i % 50),
            weight: 0.1 + (i as f32) * 0.001,
            target_neuron_impact: 0.8 - (i as f32) * 0.0005,
            expected_creature_error_reduction: 0.05 - (i as f32) * 0.00003,
            expected_creature_score_gain: 0.05 - (i as f32) * 0.00003,
            improved_count: (100 - i as u32 % 50),
            total_count: 200,
            improvement_magnitude_ratio: None,
            target_neuron_stats: Some(NeuronStatsJson {
                mean_error: 0.1,
                error_variance: 0.02,
                mean_activation: 0.5,
                activation_variance: 0.1,
                error_spike_count: 3,
                activation_spike_count: 1,
                activation_min: -0.9,
                activation_max: 0.95,
            }),
            outlier_reduction_info: None,
            prediction_confidence: 0.85,
            expected_score_gain_confidence_interval: [0.01, 0.09],
            comment: Some(format!("candidate-{i}")),
            variant_key: None,
        })
        .collect()
}

/// Generate synthetic neuron candidates.
fn create_neuron_candidates(count: usize) -> Vec<CandidateNeuronJson> {
    (0..count)
        .map(|i| CandidateNeuronJson {
            source_neuron_uuid: format!("input-{}", i % 3),
            target_neuron_uuid: format!("hidden-{}", i % 50),
            source_neuron_index: Some(i % 3),
            target_neuron_index: Some(3 + i % 50),
            incoming_weight: 0.5 + (i as f32) * 0.001,
            outgoing_weight: 0.3 - (i as f32) * 0.001,
            squash: "TANH".to_string(),
            bias: 0.0,
            comment: Some(format!("neuron-candidate-{i}")),
            target_neuron_impact: 0.7,
            expected_creature_error_reduction: 0.04,
            expected_creature_score_gain: 0.04,
            improved_count: 80,
            total_count: 200,
            improvement_magnitude_ratio: None,
            target_neuron_stats: None,
            prediction_confidence: 0.75,
            expected_score_gain_confidence_interval: [0.01, 0.07],
            target_saturation_factor: None,
            variant_key: None,
        })
        .collect()
}

/// Generate synthetic weight update candidates.
fn create_weight_update_candidates(count: usize) -> Vec<SynapseWeightUpdateCandidateJson> {
    (0..count)
        .map(|i| SynapseWeightUpdateCandidateJson {
            from_neuron_uuid: format!("hidden-{}", i % 50),
            to_neuron_uuid: "output-0".to_string(),
            from_neuron_index: Some(3 + i % 50),
            to_neuron_index: Some(53),
            old_weight: 0.5,
            new_weight: 0.5 + (i as f32) * 0.001,
            delta_weight: (i as f32) * 0.001,
            target_neuron_impact: 1.0,
            expected_creature_error_reduction: 0.02,
            expected_creature_score_gain: 0.02,
            improved_count: 90,
            total_count: 200,
            target_neuron_stats: None,
        })
        .collect()
}

/// Generate synthetic coordinated structural candidates.
fn create_coordinated_candidates(count: usize) -> Vec<CoordinatedStructuralCandidateJson> {
    (0..count)
        .map(|i| CoordinatedStructuralCandidateJson {
            remove_neuron_compensation: None,
            constant_neuron_bias_fold: None,
            operations: vec![
                CoordinatedStructuralOpJson::AddNeuron {
                    neuron_uuid: format!("new-neuron-{i}"),
                    neuron_type: "hidden".to_string(),
                    squash: "TANH".to_string(),
                    bias: 0.0,
                    insert_before_neuron_uuid: Some("output-0".to_string()),
                },
                CoordinatedStructuralOpJson::AddSynapse {
                    from_neuron_uuid: format!("input-{}", i % 3),
                    to_neuron_uuid: format!("new-neuron-{i}"),
                    weight: 0.5,
                },
                CoordinatedStructuralOpJson::AddSynapse {
                    from_neuron_uuid: format!("new-neuron-{i}"),
                    to_neuron_uuid: "output-0".to_string(),
                    weight: 0.3,
                },
            ],
            expected_creature_score_gain: 0.06 - (i as f32) * 0.0001,
            comment: Some(format!("coordinated-{i}")),
        })
        .collect()
}

/// Build a synthetic `AnalyzeParallelOutput` with the given candidate counts.
fn create_analysis_output(candidate_count: usize) -> AnalyzeParallelOutput {
    let synapse_count = candidate_count;
    let neuron_count = candidate_count / 5;
    let weight_update_count = candidate_count / 3;
    let coordinated_count = candidate_count / 10;

    AnalyzeParallelOutput {
        success: true,
        schema_version: "2".to_string(),
        helpful_synapses: Some(create_synapse_candidates(synapse_count)),
        harmful_synapses: Some(create_synapse_candidates(synapse_count / 2)),
        synapse_diagnostics: None,
        synapse_gpu_used: Some(true),
        synapse_metadata: None,
        helpful_neurons: Some(create_neuron_candidates(neuron_count)),
        synapse_weight_updates: Some(create_weight_update_candidates(weight_update_count)),
        coordinated_structural_candidates: Some(create_coordinated_candidates(coordinated_count)),
        candidate_clusters: None,
        neuron_diagnostics: None,
        neuron_gpu_used: Some(true),
        neuron_metadata: None,
        neuron_fingerprints: None,
        fingerprint_cache_hits: Some(5),
        fingerprint_cache_misses: Some(10),
        module_outcome_tracker: None,
        memory_budget_exceeded: Some(false),
        cancelled: Some(false),
        memory_pressure_cancelled: None,
        environmentally_disabled: None,
        zero_candidate_summary: None,
        error: None,
        error_kind: None,
        retryable: None,
    }
}

/// Build a JSON string representing `AnalyzeParallelInput` for deserialisation benchmarks.
fn create_analysis_input_json(neuron_count: usize) -> String {
    let creature = create_test_creature(neuron_count);
    let focus_neurons: Vec<String> = (0..neuron_count).map(|i| format!("hidden-{i}")).collect();

    let creature_json = serde_json::to_string(&creature).expect("serialise creature");
    let focus_json = serde_json::to_string(&focus_neurons).expect("serialise focus");

    format!(
        r#"{{"parquetFile":"/tmp/test.parquet","creature":{creature_json},"focusNeurons":{focus_json},"maxSynapseCandidates":100,"maxNeuronCandidates":20,"temperature":1.0}}"#
    )
}

/// Benchmark JSON serialisation of analysis output at varying candidate counts.
fn bench_serialisation(c: &mut Criterion) {
    let mut group = c.benchmark_group("ffi_marshalling_serialise");

    for candidate_count in [10, 100, 1000] {
        let output = create_analysis_output(candidate_count);
        group.bench_with_input(
            BenchmarkId::new("analyze_output", candidate_count),
            &output,
            |b, output| {
                b.iter(|| {
                    let json = serde_json::to_string(black_box(output)).expect("serialise");
                    black_box(json.len())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark JSON deserialisation of analysis input at varying creature sizes.
fn bench_deserialisation(c: &mut Criterion) {
    let mut group = c.benchmark_group("ffi_marshalling_deserialise");

    for neuron_count in [10, 100, 1000] {
        let json = create_analysis_input_json(neuron_count);
        group.bench_with_input(
            BenchmarkId::new("analyze_input", neuron_count),
            &json,
            |b, json| {
                b.iter(|| {
                    let input: neat_ai_discovery::AnalyzeParallelInput =
                        serde_json::from_str(black_box(json)).expect("deserialise");
                    black_box(input.creature.neurons.len())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark full round-trip: serialise output then deserialise input.
fn bench_round_trip(c: &mut Criterion) {
    let mut group = c.benchmark_group("ffi_marshalling_round_trip");

    for candidate_count in [10, 100, 1000] {
        let output = create_analysis_output(candidate_count);
        let input_json = create_analysis_input_json(candidate_count.min(100));

        let id = format!("{candidate_count}_candidates");
        group.bench_function(BenchmarkId::new("serialize_then_deserialize", &id), |b| {
            b.iter(|| {
                // Serialise output (Rust → JSON for FFI return)
                let output_json =
                    serde_json::to_string(black_box(&output)).expect("serialise output");
                black_box(output_json.len());

                // Deserialise input (JSON from FFI → Rust)
                let input: neat_ai_discovery::AnalyzeParallelInput =
                    serde_json::from_str(black_box(&input_json)).expect("deserialise input");
                black_box(input.creature.neurons.len())
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_serialisation,
    bench_deserialisation,
    bench_round_trip
);
criterion_main!(benches);
