//! Criterion benchmark for Issue #2350's fan-in squash lookup conversion.

#![allow(clippy::cast_precision_loss)]

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use neat_ai_discovery::analysis::detection::fanin_polarity_conflict::{
    FaninPolarityConflictCandidate, fanin_polarity_conflicts_to_coordinated_candidates,
};
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
use std::hint::black_box;

fn fixture(count: usize) -> (CreatureJson, Vec<FaninPolarityConflictCandidate>) {
    let mut neurons = vec![
        neuron("positive", "input", "IDENTITY"),
        neuron("negative", "input", "IDENTITY"),
    ];
    let mut synapses = Vec::with_capacity(count * 3);
    let mut candidates = Vec::with_capacity(count);
    for index in 0..count {
        let uuid = format!("hidden-{index}");
        neurons.push(neuron(&uuid, "hidden", "TANH"));
        synapses.push(synapse("positive", &uuid, 2.0));
        synapses.push(synapse("negative", &uuid, -2.0));
        candidates.push(FaninPolarityConflictCandidate {
            neuron_uuid: uuid,
            positive_weight_sum: 2.0,
            negative_weight_sum: 2.0,
            positive_count: 1,
            negative_count: 1,
            conflict_score: 1.0,
            sample_count: 30,
            estimated_improvement: 0.01,
        });
    }
    neurons.push(neuron("output", "output", "IDENTITY"));
    for index in 0..count {
        synapses.push(synapse(&format!("hidden-{index}"), "output", 1.0));
    }
    (
        CreatureJson {
            neurons,
            synapses,
            input: 2,
            output: 1,
        },
        candidates,
    )
}

fn neuron(uuid: &str, neuron_type: &str, squash: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
        squash: squash.to_string(),
        bias: 0.0,
    }
}

fn synapse(from: &str, to: &str, weight: f32) -> SynapseJson {
    SynapseJson {
        from_uuid: from.to_string(),
        to_uuid: to.to_string(),
        weight,
        synapse_type: None,
    }
}

fn bench_squash_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("fanin_polarity_squash_lookup");
    group.sample_size(10);
    for count in [500, 2_000, 8_000] {
        let (creature, candidates) = fixture(count);
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, _| {
            b.iter(|| {
                fanin_polarity_conflicts_to_coordinated_candidates(
                    black_box(&candidates),
                    black_box(&creature),
                )
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_squash_conversion);
criterion_main!(benches);
