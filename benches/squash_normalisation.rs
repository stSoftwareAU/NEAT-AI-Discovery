//! Benchmark for Issue #753: Squash string normalisation at load time.
//!
//! Measures the cost of repeated `.to_ascii_uppercase()` calls in detection
//! modules versus pre-normalised squash strings. Exercises saturation,
//! unbounded capping, and activation mismatch detection on a creature with
//! 200+ hidden neurons using mixed-case activation function names.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use neat_ai_discovery::analysis::detection::saturation::detect_saturated_neurons;
use neat_ai_discovery::analysis::detection::unbounded_capping::detect_unbounded_capping_candidates;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

/// Mixed-case activation names as they might arrive from TypeScript.
const MIXED_CASE_SQUASHES: &[&str] = &[
    "Tanh",
    "logistic",
    "Hard_Tanh",
    "Relu",
    "identity",
    "Softsign",
    "LeakyRelu",
    "Elu",
    "Selu",
    "Softplus",
    "relu6",
    "Swish",
    "Mish",
    "Gelu",
    "arctan",
];

/// Benchmark test data: hidden neurons, their records, and the creature.
struct BenchData {
    hidden_neurons: Vec<(String, String, f32)>,
    neuron_records: Vec<(String, Vec<DiscoverRecord>)>,
    _creature: CreatureJson,
}

/// Create a creature with `neuron_count` hidden neurons using mixed-case
/// squash names and 100 discovery records per neuron.
fn create_mixed_case_creature(neuron_count: usize) -> BenchData {
    let mut neurons_json = Vec::new();
    let mut synapses = Vec::new();
    let mut hidden_neurons = Vec::new();
    let mut neuron_records = Vec::new();

    // Input neuron
    neurons_json.push(NeuronJson {
        uuid: "input-0".to_string(),
        neuron_type: "input".to_string(),
        squash: "IDENTITY".to_string(),
        bias: 0.0,
    });

    for i in 0..neuron_count {
        let uuid = format!("hidden-{i}");
        let squash = MIXED_CASE_SQUASHES[i % MIXED_CASE_SQUASHES.len()].to_string();
        let bias = 0.1 * (i as f32 % 5.0);

        neurons_json.push(NeuronJson {
            uuid: uuid.clone(),
            neuron_type: "hidden".to_string(),
            squash: squash.clone(),
            bias,
        });

        synapses.push(SynapseJson {
            from_uuid: "input-0".to_string(),
            to_uuid: uuid.clone(),
            weight: 0.5,
            synapse_type: None,
        });

        hidden_neurons.push((uuid.clone(), squash, bias));

        // Generate 100 discovery records per neuron
        let records: Vec<DiscoverRecord> = (0..100)
            .map(|obs| {
                let value = (obs as f32 * 0.02) - 1.0;
                let activation = value.tanh(); // approximate bounded activation
                DiscoverRecord::new(
                    obs,
                    uuid.clone(),
                    Some(value),
                    activation,
                    vec![activation * 0.01],
                )
            })
            .collect();

        neuron_records.push((uuid, records));
    }

    // Output neuron
    neurons_json.push(NeuronJson {
        uuid: "output-0".to_string(),
        neuron_type: "output".to_string(),
        squash: "IDENTITY".to_string(),
        bias: 0.0,
    });

    for i in 0..neuron_count {
        synapses.push(SynapseJson {
            from_uuid: format!("hidden-{i}"),
            to_uuid: "output-0".to_string(),
            weight: 0.3,
            synapse_type: None,
        });
    }

    let creature = CreatureJson {
        neurons: neurons_json,
        synapses,
        input: 1,
        output: 1,
    };

    BenchData {
        hidden_neurons,
        neuron_records,
        _creature: creature,
    }
}

fn bench_saturation_detection(c: &mut Criterion) {
    let data = create_mixed_case_creature(250);

    c.bench_function("saturation_detect_250_neurons", |b| {
        b.iter(|| {
            detect_saturated_neurons(
                black_box(&data.hidden_neurons),
                black_box(&data.neuron_records),
            )
        });
    });
}

fn bench_unbounded_capping_detection(c: &mut Criterion) {
    let data = create_mixed_case_creature(250);

    c.bench_function("unbounded_capping_250_neurons", |b| {
        b.iter(|| {
            detect_unbounded_capping_candidates(
                black_box(&data.hidden_neurons),
                black_box(&data.neuron_records),
            )
        });
    });
}

fn bench_combined_detection(c: &mut Criterion) {
    let data = create_mixed_case_creature(250);

    c.bench_function("combined_detection_250_neurons", |b| {
        b.iter(|| {
            let sat = detect_saturated_neurons(
                black_box(&data.hidden_neurons),
                black_box(&data.neuron_records),
            );
            let cap = detect_unbounded_capping_candidates(
                black_box(&data.hidden_neurons),
                black_box(&data.neuron_records),
            );
            (sat, cap)
        });
    });
}

criterion_group!(
    benches,
    bench_saturation_detection,
    bench_unbounded_capping_detection,
    bench_combined_detection,
);
criterion_main!(benches);
