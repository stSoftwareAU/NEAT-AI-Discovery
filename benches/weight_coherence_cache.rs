//! Benchmark for Issue #770: Pass `CreatureTopologyCache` to `weight_coherence`.
//!
//! Measures the performance of `detect_incoherent_weight_ratios` with and
//! without a pre-computed `CreatureTopologyCache`, varying creature size.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use neat_ai_discovery::analysis::detection::topology_cache::CreatureTopologyCache;
use neat_ai_discovery::analysis::detection::weight_coherence::{
    WeightCoherenceConfig, detect_incoherent_weight_ratios,
};
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
use std::hint::black_box;

/// Create a creature with the given number of hidden neurons and proportional synapses.
fn create_test_creature(hidden_count: usize) -> CreatureJson {
    let input_count = (hidden_count / 5).max(2);
    let output_count = (hidden_count / 10).max(1);

    let mut neurons = Vec::with_capacity(input_count + hidden_count + output_count);
    let mut synapses = Vec::new();

    for i in 0..input_count {
        neurons.push(NeuronJson {
            uuid: format!("inp-{i}"),
            neuron_type: "input".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        });
    }

    for i in 0..hidden_count {
        neurons.push(NeuronJson {
            uuid: format!("hid-{i}"),
            neuron_type: "hidden".to_string(),
            squash: "TANH".to_string(),
            bias: 0.1 * (i % 5) as f32,
        });
    }

    for i in 0..output_count {
        neurons.push(NeuronJson {
            uuid: format!("out-{i}"),
            neuron_type: "output".to_string(),
            squash: "LOGISTIC".to_string(),
            bias: 0.0,
        });
    }

    for i in 0..hidden_count {
        let inp_idx = i % input_count;
        synapses.push(SynapseJson {
            from_uuid: format!("inp-{inp_idx}"),
            to_uuid: format!("hid-{i}"),
            weight: 50.0 - 0.1 * (i % 10) as f32,
            ..Default::default()
        });

        if i + 1 < hidden_count {
            synapses.push(SynapseJson {
                from_uuid: format!("hid-{i}"),
                to_uuid: format!("hid-{}", i + 1),
                weight: 0.001,
                ..Default::default()
            });
        }

        if i % 3 == 0 {
            let out_idx = i % output_count;
            synapses.push(SynapseJson {
                from_uuid: format!("hid-{i}"),
                to_uuid: format!("out-{out_idx}"),
                weight: 0.002,
                ..Default::default()
            });
        }
    }

    CreatureJson {
        neurons,
        synapses,
        input: input_count,
        output: output_count,
    }
}

/// Create activation records for all hidden neurons.
fn create_records(hidden_count: usize, samples: usize) -> Vec<(String, Vec<DiscoverRecord>)> {
    (0..hidden_count)
        .map(|h| {
            let uuid = format!("hid-{h}");
            let records: Vec<DiscoverRecord> = (0..samples)
                .map(|i| {
                    let activation = ((i as f32 * 0.1).sin()).tanh();
                    DiscoverRecord {
                        obs_index: i as u32,
                        neuron_uuid: uuid.clone(),
                        value: Some(activation),
                        activation,
                        errors: vec![0.1],
                    }
                })
                .collect();
            (uuid, records)
        })
        .collect()
}

fn bench_weight_coherence_cache(c: &mut Criterion) {
    let sizes: &[usize] = &[50, 100, 200, 500];

    let mut group = c.benchmark_group("weight_coherence_cache");

    for &size in sizes {
        let creature = create_test_creature(size);
        let records = create_records(size, 50);
        let config = WeightCoherenceConfig::default();
        let id = format!("{size}_neurons");

        // Baseline: no cache (builds topology locally)
        group.bench_with_input(
            BenchmarkId::new("without_cache", &id),
            &(&creature, &records, &config),
            |b, &(creature, records, config)| {
                b.iter(|| {
                    black_box(detect_incoherent_weight_ratios(
                        black_box(creature),
                        black_box(records),
                        black_box(config),
                        None,
                    ));
                });
            },
        );

        // With pre-computed cache
        let topo = CreatureTopologyCache::new(&creature);
        group.bench_with_input(
            BenchmarkId::new("with_cache", &id),
            &(&creature, &records, &config, &topo),
            |b, &(creature, records, config, topo)| {
                b.iter(|| {
                    black_box(detect_incoherent_weight_ratios(
                        black_box(creature),
                        black_box(records),
                        black_box(config),
                        Some(black_box(topo)),
                    ));
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_weight_coherence_cache);
criterion_main!(benches);
