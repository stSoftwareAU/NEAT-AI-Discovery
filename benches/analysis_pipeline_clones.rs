//! Benchmark for Issue #983: Clone reduction in analysis detection and recommendation pipeline.
//!
//! Measures performance of modules with avoidable `clone()` calls:
//! - Topology diversification candidate conversion (`HashSet` of UUID pairs)
//! - Batch-successful candidate grouping (`HashSet` dedup of target UUIDs)
//! - `SynapseCounts` construction (`HashMap` entry UUID cloning)

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use criterion::{Criterion, criterion_group, criterion_main};
use neat_ai_discovery::analysis::detection::topology_diversification::{
    TopologyDiversificationCandidate, topology_diversification_to_coordinated_candidates,
};
use neat_ai_discovery::analysis::recommendation::batch_successful::{
    IndividualCandidate, group_into_batches,
};
use neat_ai_discovery::focus::SynapseCounts;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
use std::hint::black_box;

/// Create a creature with many synapses for topology diversification benchmarking.
fn create_topo_div_creature(synapse_count: usize) -> CreatureJson {
    let neuron_count = synapse_count / 3;
    let mut neurons = Vec::new();

    for i in 0..neuron_count {
        neurons.push(NeuronJson {
            uuid: format!("hidden-{i}"),
            neuron_type: "hidden".to_string(),
            squash: "TANH".to_string(),
            bias: 0.0,
        });
    }

    for o in 0..3 {
        neurons.push(NeuronJson {
            uuid: format!("output-{o}"),
            neuron_type: "output".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        });
    }

    let mut synapses = Vec::new();
    for i in 0..synapse_count {
        let from_idx = i % neuron_count;
        let to_idx = (i * 7 + 3) % neuron_count;
        synapses.push(SynapseJson {
            from_uuid: format!("hidden-{from_idx}"),
            to_uuid: format!("hidden-{to_idx}"),
            weight: 0.5,
            synapse_type: None,
        });
    }

    CreatureJson {
        neurons,
        synapses,
        input: neuron_count,
        output: 3,
    }
}

/// Create topology diversification candidates for benchmarking.
fn create_topo_div_candidates(count: usize) -> Vec<TopologyDiversificationCandidate> {
    (0..count)
        .map(|i| TopologyDiversificationCandidate {
            output_neuron_uuid: format!("output-{}", i % 3),
            source_input_uuid: format!("input-{i}"),
            max_hidden_depth: 0,
            direct_path_count: 2,
            mean_output_error: 0.15 + 0.01 * i as f32,
            estimated_improvement: 0.05 - 0.001 * i as f32,
        })
        .collect()
}

/// Create individual candidates for batch grouping benchmarking.
fn create_individual_candidates(count: usize) -> Vec<IndividualCandidate> {
    (0..count)
        .map(|i| IndividualCandidate {
            source_uuid: format!("source-{i}"),
            target_uuid: format!("target-{}", i % 10),
            weight: 0.3 + 0.01 * i as f32,
            improvement: 0.1 - 0.001 * i as f32,
            sample_count: 50,
        })
        .collect()
}

fn bench_topology_diversification_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("topology_div_clone_reduction");

    for synapse_count in [100, 500, 2000] {
        let creature = create_topo_div_creature(synapse_count);
        let candidates = create_topo_div_candidates(20);

        group.bench_function(format!("convert_{synapse_count}_synapses"), |b| {
            b.iter(|| {
                topology_diversification_to_coordinated_candidates(
                    black_box(&candidates),
                    black_box(&creature),
                )
            });
        });
    }

    group.finish();
}

fn bench_batch_grouping(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_grouping_clone_reduction");

    for count in [20, 50, 100] {
        let candidates = create_individual_candidates(count);

        group.bench_function(format!("group_{count}_candidates"), |b| {
            b.iter(|| group_into_batches(black_box(&candidates)));
        });
    }

    group.finish();
}

fn bench_synapse_counts_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("synapse_counts_clone_reduction");

    for synapse_count in [500, 5000, 20_000] {
        let neuron_count = synapse_count / 5;
        let mut neurons = Vec::new();
        let mut synapses = Vec::new();

        for n in 0..neuron_count {
            neurons.push(NeuronJson {
                uuid: format!("neuron-{n}"),
                neuron_type: "hidden".to_string(),
                squash: "TANH".to_string(),
                bias: 0.0,
            });
        }

        for i in 0..synapse_count {
            synapses.push(SynapseJson {
                from_uuid: format!("neuron-{}", i % neuron_count),
                to_uuid: format!("neuron-{}", (i * 7 + 3) % neuron_count),
                weight: 0.5,
                synapse_type: None,
            });
        }

        let creature = CreatureJson {
            neurons,
            synapses,
            input: neuron_count,
            output: 1,
        };

        group.bench_function(format!("new_{synapse_count}_synapses"), |b| {
            b.iter(|| SynapseCounts::new(black_box(&creature)));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_topology_diversification_conversion,
    bench_batch_grouping,
    bench_synapse_counts_construction,
);
criterion_main!(benches);
