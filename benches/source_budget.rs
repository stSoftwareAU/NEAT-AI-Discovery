//! Benchmark for Issue #1542: two-stage per-target source budget.
//!
//! Measures end-to-end `analyze_synapses` wall-clock for a single focus target
//! with a large fan-in of eligible upstream sources, comparing the pre-#1542
//! unlimited enumeration (baseline) against capping stage-2 sample-building +
//! GPU evaluation to the top-K priority-ordered sources via
//! `NEAT_AI_DISCOVERY_MAX_SOURCES_PER_TARGET`.
//!
//! The A/B is driven purely by the env var: baseline leaves it unset (unlimited,
//! the shipped default), and each K run sets it. Because the ordering is
//! identical, the only difference is how many sources reach the GPU.
//!
//! Run: `cargo bench --bench source_budget`

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use criterion::{Criterion, criterion_group, criterion_main};
use neat_ai_discovery::analysis::{GpuAnalyzer, analyze_synapses};
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{AnalyzeSynapsesInput, CreatureJson, NeuronJson};
use std::hint::black_box;
use tempfile::tempdir;

const ENV: &str = "NEAT_AI_DISCOVERY_MAX_SOURCES_PER_TARGET";

/// Creature: `input_count` inputs feeding a single IDENTITY output target.
fn create_fan_in_creature(input_count: usize) -> CreatureJson {
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

/// Records where every input has activations correlated with the output error,
/// so each input is a genuine add-synapse candidate that must be evaluated.
fn create_records(input_count: usize, record_count: usize) -> Vec<DiscoverRecord> {
    let mut records = Vec::with_capacity((input_count + 1) * record_count);
    for obs_index in 0..record_count as u32 {
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

/// Set (or clear) the per-target source budget for the following measurement.
/// Benches run single-threaded per group, so there is no concurrent env access.
fn set_budget(budget: Option<usize>) {
    match budget {
        Some(k) => {
            // SAFETY: benches run single-threaded per group; no concurrent env access.
            unsafe { std::env::set_var(ENV, k.to_string()) };
        }
        None => {
            // SAFETY: benches run single-threaded per group; no concurrent env access.
            unsafe { std::env::remove_var(ENV) };
        }
    }
}

fn benchmark_source_budget(c: &mut Criterion) {
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping benchmark: no GPU available");
        return;
    }

    let input_count = 800;
    let record_count = 150;

    let temp_dir = tempdir().expect("Failed to create temp directory");
    let parquet_path = temp_dir.path().join("records.parquet");
    let parquet_file = parquet_path.to_str().unwrap().to_string();
    let records = create_records(input_count, record_count);
    neat_ai_discovery::parquet_format::write_records_to_parquet(&parquet_file, &records)
        .expect("Failed to write parquet");
    let creature = create_fan_in_creature(input_count);

    let run = |parquet_file: &str, creature: &CreatureJson| {
        let input = AnalyzeSynapsesInput {
            parquet_file: parquet_file.to_string(),
            creature: creature.clone(),
            focus_neurons: vec!["output-0".to_string()],
            max_candidates: Some(10),
            analysis_deadline_ms: None,
            random_seed: Some(42),
            temperature: 1.0,
            failure_cache: None,
            discovery_outcome_log: None,
        };
        let result = analyze_synapses(&input).expect("Analysis should succeed");
        black_box(result);
    };

    let mut group = c.benchmark_group("source_budget");

    // Baseline: unlimited (pre-#1542 behaviour, the shipped default).
    set_budget(None);
    group.bench_function("unlimited", |b| {
        b.iter(|| run(&parquet_file, &creature));
    });

    // Capped stage-2 work at K ∈ {64, 128, 256}.
    for k in [64usize, 128, 256] {
        set_budget(Some(k));
        group.bench_function(format!("k_{k}"), |b| {
            b.iter(|| run(&parquet_file, &creature));
        });
    }

    set_budget(None);
    group.finish();
}

criterion_group!(benches, benchmark_source_budget);
criterion_main!(benches);
