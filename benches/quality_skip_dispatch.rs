//! Benchmark for Issue #1074: Quality-based module skipping during dispatch merge.
//!
//! Measures the wall-clock time for the `merge_discovery_module_results` phase
//! with and without quality-based skipping. Uses synthetic detection results
//! to isolate the merge overhead from actual detection work.
//!
//! Run with: `cargo bench --bench quality_skip_dispatch`

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for benchmarking

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use neat_ai_discovery::analysis::discovery_dispatch::{
    DiscoveryDetectionResult, DiscoveryModuleDetectionEntry, DiscoveryModuleDetectionResults,
    merge_discovery_module_results,
};
use neat_ai_discovery::analysis::module_weights::ModuleOutcomeTracker;
use neat_ai_discovery::analysis::shared;
use neat_ai_discovery::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson};
use std::hint::black_box;

fn empty_synapse_result() -> shared::AnalyzeSynapsesResult {
    shared::AnalyzeSynapsesResult {
        helpful_synapses: Vec::new(),
        harmful_synapses: Vec::new(),
        synapse_weight_updates: Vec::new(),
        coordinated_structural_candidates: Vec::new(),
        candidate_clusters: Vec::new(),
        gpu_used: false,
        no_candidate_reasons: Vec::new(),
        metadata: shared::SynapseAnalysisMetadata {
            candidates_found: 0,
            candidates_returned: 0,
            ..Default::default()
        },
    }
}

fn make_candidate(gain: f32) -> CoordinatedStructuralCandidateJson {
    CoordinatedStructuralCandidateJson {
        remove_neuron_compensation: None,
        constant_neuron_bias_fold: None,
        operations: vec![CoordinatedStructuralOpJson::RemoveSynapse {
            from_neuron_uuid: "a".to_string(),
            to_neuron_uuid: "b".to_string(),
        }],
        expected_creature_score_gain: gain,
        comment: Some(format!("bench gain={gain}")),
    }
}

fn make_detection_results(
    module_count: usize,
    candidates_per_module: usize,
) -> DiscoveryModuleDetectionResults {
    let entries: Vec<DiscoveryModuleDetectionEntry> = (0..module_count)
        .map(|i| {
            let candidates: Vec<_> = (0..candidates_per_module)
                .map(|j| {
                    let gain = 0.1 + 0.01 * (i as f32) + 0.001 * (j as f32);
                    make_candidate(gain)
                })
                .collect();
            DiscoveryModuleDetectionEntry {
                module_name: format!("module_{i}"),
                phase_name: "bench_phase",
                max_candidates: 0,
                result: Some(DiscoveryDetectionResult {
                    detected_count: candidates.len(),
                    candidates,
                }),
                // Issue #1925: every module in this benchmark ran.
                skip_reason: None,
            }
        })
        .collect();
    DiscoveryModuleDetectionResults { entries }
}

fn bench_merge_with_quality_skip(c: &mut Criterion) {
    let mut group = c.benchmark_group("quality_skip_merge");

    // Vary module count and candidates per module.
    let configs: Vec<(usize, usize, &str)> = vec![
        (10, 5, "10m_5c"),
        (30, 5, "30m_5c"),
        (47, 5, "47m_5c"),
        (47, 10, "47m_10c"),
    ];

    for (modules, candidates, label) in configs {
        group.bench_with_input(
            BenchmarkId::new("merge_discovery_results", label),
            &(modules, candidates),
            |b, &(m, c)| {
                b.iter(|| {
                    let mut syn = empty_synapse_result();
                    let results = make_detection_results(m, c);
                    let mut tracker = ModuleOutcomeTracker::new();
                    merge_discovery_module_results(&mut syn, results, None, false, &mut tracker);
                    black_box(&syn);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_merge_with_quality_skip);
criterion_main!(benches);
