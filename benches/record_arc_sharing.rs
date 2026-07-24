//! Benchmark for Issue #1543: Arc-share `DiscoverRecord` across detection modules.
//!
//! The discovery dispatch builds ~48 module specs per `analyze_all` post-processing
//! pass. Each spec's `detect_fn` loads the records it needs from the shared
//! `RecordCache` via one of the `load_records_for_*` bulk loaders. Historically
//! those loaders deep-cloned the inner `Vec<DiscoverRecord>` for every module,
//! materialising tens of GB of transient record copies on production-scale
//! creatures (~1662 hidden neurons).
//!
//! This benchmark reproduces that dispatch loader pattern against a preloaded
//! cache and reports both:
//! - **Peak transient bytes** materialised by the loaders, measured with the
//!   library's own tracking allocator via `discovery_memory_usage_bytes()` — the
//!   peak-RSS proxy named in the issue's benchmark plan.
//! - **Wall-clock** of the loader phase.
//!
//! With `Arc`-sharing the loaders hand out `Arc::clone`s of the cache's existing
//! allocation, so the transient bytes collapse to near zero. The benchmark source
//! is identical before and after the migration — only the measured numbers move —
//! so it doubles as a regression guard: re-introducing a deep clone shows up as a
//! step change here.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use neat_ai_discovery::analysis::cache::RecordCache;
use neat_ai_discovery::ffi::discovery_memory_usage_bytes;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

/// Production-scaled-down fixture: enough neurons/records that a deep clone is clearly
/// visible, small enough to run quickly in CI.
const NEURON_COUNT: usize = 300;
const RECORDS_PER_NEURON: usize = 400;
const ERRORS_PER_RECORD: usize = 4;

fn build_records(neuron_count: usize) -> HashMap<String, Vec<DiscoverRecord>> {
    let mut map = HashMap::with_capacity(neuron_count);
    for n in 0..neuron_count {
        let uuid = format!("hidden-{n}");
        let mut records = Vec::with_capacity(RECORDS_PER_NEURON);
        for r in 0..RECORDS_PER_NEURON {
            records.push(DiscoverRecord {
                obs_index: r as u32,
                neuron_uuid: uuid.clone(),
                value: Some(0.5 + 0.001 * r as f32),
                activation: 0.25 * r as f32,
                errors: (0..ERRORS_PER_RECORD).map(|e| 0.01 * e as f32).collect(),
            });
        }
        map.insert(uuid, records);
    }
    map
}

fn build_creature(neuron_count: usize) -> CreatureJson {
    let mut neurons = Vec::with_capacity(neuron_count + 3);
    for n in 0..neuron_count {
        neurons.push(NeuronJson {
            uuid: format!("hidden-{n}"),
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
    let synapses = (0..neuron_count)
        .map(|i| SynapseJson {
            from_uuid: format!("hidden-{i}"),
            to_uuid: format!("output-{}", i % 3),
            weight: 0.5,
            synapse_type: None,
        })
        .collect();
    CreatureJson {
        neurons,
        synapses,
        input: 0,
        output: 3,
    }
}

fn preloaded_cache(data: HashMap<String, Vec<DiscoverRecord>>) -> RecordCache {
    let data = Arc::new(data);
    RecordCache::with_loader(
        "bench.parquet",
        Arc::new(move |_file: &str, neuron_uuid: &str| {
            Ok(data.get(neuron_uuid).cloned().unwrap_or_default())
        }),
    )
}

/// Warm every neuron's cache entry so the loaders exercise the hot path (cache
/// hits) rather than the one-off lazy load.
fn warm(cache: &RecordCache, creature: &CreatureJson) {
    for n in &creature.neurons {
        let _ = cache.get(&n.uuid);
    }
}

fn hidden_tuples(creature: &CreatureJson) -> Vec<(String, String, f32)> {
    creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "hidden")
        .map(|n| (n.uuid.clone(), n.squash.clone(), n.bias))
        .collect()
}

/// One representative `analyze_all` post-processing loader pass. The call counts
/// mirror the real dispatch (`src/analysis/module_dispatch_specs/`): ~20 hidden,
/// ~17 all-neuron, ~10 neuron-type, 1 synapse-source loads.
///
/// The results are *retained* in the returned vector so nothing is freed mid-pass
/// — this mirrors the real dispatch, where rayon runs many modules concurrently
/// and their loader outputs are alive at the same time. The `len()` sum defeats
/// dead-code elimination.
#[allow(clippy::type_complexity)]
fn dispatch_pass_retained(cache: &RecordCache, creature: &CreatureJson) -> Vec<usize> {
    let hidden = hidden_tuples(creature);
    let mut retained: Vec<usize> = Vec::new();
    // Keep the loader outputs alive by summing their lengths into a Vec whose
    // entries hold references indirectly through the closure captures below.
    let mut sink: Vec<Box<dyn std::any::Any>> = Vec::new();

    for _ in 0..20 {
        let loaded = cache.load_records_for_hidden(&hidden);
        retained.push(loaded.iter().map(|(_, r)| r.len()).sum());
        sink.push(Box::new(loaded));
    }
    for _ in 0..17 {
        let loaded = cache.load_records_for_all_neurons(creature);
        retained.push(loaded.iter().map(|(_, r)| r.len()).sum());
        sink.push(Box::new(loaded));
    }
    for _ in 0..10 {
        let loaded = cache.load_records_for_neuron_types(creature, &["hidden", "output"]);
        retained.push(loaded.iter().map(|(_, r)| r.len()).sum());
        sink.push(Box::new(loaded));
    }
    {
        let loaded = cache.load_records_for_synapse_sources(creature);
        retained.push(loaded.iter().map(|(_, r)| r.len()).sum());
        sink.push(Box::new(loaded));
    }

    // Measure peak while everything is still alive, then drop.
    let peak = discovery_memory_usage_bytes();
    retained.push(peak as usize);
    drop(sink);
    retained
}

fn main() {
    let creature = build_creature(NEURON_COUNT);
    let cache = preloaded_cache(build_records(NEURON_COUNT));
    warm(&cache, &creature);

    // Warm-up pass (not measured) to stabilise allocator/heap state.
    let _ = dispatch_pass_retained(&cache, &creature);

    // Baseline live bytes with nothing retained.
    let baseline = discovery_memory_usage_bytes();

    // Peak-bytes measurement: run one pass holding every loader output alive.
    let hidden = hidden_tuples(&creature);
    let mut sink: Vec<Vec<(String, _)>> = Vec::new();
    for _ in 0..20 {
        sink.push(cache.load_records_for_hidden(&hidden));
    }
    for _ in 0..17 {
        sink.push(cache.load_records_for_all_neurons(&creature));
    }
    for _ in 0..10 {
        sink.push(cache.load_records_for_neuron_types(&creature, &["hidden", "output"]));
    }
    sink.push(cache.load_records_for_synapse_sources(&creature));
    let peak = discovery_memory_usage_bytes();
    // Defeat DCE and keep sink alive until after the peak read.
    let checksum: usize = sink
        .iter()
        .map(|v| v.iter().map(|(_, r)| r.len()).sum::<usize>())
        .sum();
    drop(sink);
    let transient_bytes = peak.saturating_sub(baseline);

    // Wall-clock measurement: repeat the pass and time it.
    const PASSES: usize = 30;
    let start = Instant::now();
    let mut wc_checksum = 0usize;
    for _ in 0..PASSES {
        let r = dispatch_pass_retained(&cache, &creature);
        wc_checksum += r.iter().sum::<usize>();
    }
    let elapsed = start.elapsed();
    let per_pass_ms = elapsed.as_secs_f64() * 1000.0 / PASSES as f64;

    println!("record_arc_sharing benchmark (Issue #1543)");
    println!("  neurons={NEURON_COUNT} records/neuron={RECORDS_PER_NEURON}");
    println!("  checksum={checksum} wc_checksum={wc_checksum}");
    println!(
        "  peak transient bytes (one concurrent dispatch pass): {transient_bytes} ({:.2} MiB)",
        transient_bytes as f64 / (1024.0 * 1024.0)
    );
    println!("  loader-phase wall-clock per pass: {per_pass_ms:.3} ms");
}
