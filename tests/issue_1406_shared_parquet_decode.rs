//! Issue #1406: Eliminate the cross-phase parquet reload.
//!
//! Focus selection (`rank_focus_neurons`) and the analysis record load
//! (`RecordCache::new_adaptive`, built fresh inside `analyze_all`) are two
//! separate FFI calls that each used to scan the full parquet from scratch. The
//! process-shared single-decode cache lets the second phase reuse the first
//! phase's grouped decode, so a discovery cycle reads the file at most once.
//!
//! These tests live in their own integration binary (process) and are
//! `#[serial]`, so the process-global shared cache has a single user at a time —
//! the per-file decode counter is therefore deterministic. The analysis GPU
//! work is not exercised here: the reload this issue removes happens during
//! record loading, before any GPU dispatch.

#![allow(clippy::cast_precision_loss)]

use neat_ai_discovery::analysis::cache::RecordCache;
use neat_ai_discovery::focus::{FocusLoadingMode, rank_focus_neurons};
use neat_ai_discovery::parquet_format::shared_records::{
    decodes_for_path, invalidate, load_grouped_records_shared,
};
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
use serial_test::serial;
use std::sync::Arc;
use tempfile::NamedTempFile;

/// Small forward-only creature: a handful of hidden neurons feeding one output.
fn make_creature(hidden: usize) -> CreatureJson {
    let mut neurons = Vec::new();
    let mut synapses = Vec::new();

    for i in 0..hidden {
        let uuid = format!("h{i}");
        neurons.push(NeuronJson {
            uuid: uuid.clone(),
            neuron_type: "hidden".to_string(),
            squash: "TANH".to_string(),
            bias: 0.0,
        });
        synapses.push(SynapseJson {
            from_uuid: "in-0".to_string(),
            to_uuid: uuid.clone(),
            weight: 1.0,
            synapse_type: None,
        });
        synapses.push(SynapseJson {
            from_uuid: uuid,
            to_uuid: "out".to_string(),
            weight: 1.0,
            synapse_type: None,
        });
    }

    neurons.push(NeuronJson {
        uuid: "out".to_string(),
        neuron_type: "output".to_string(),
        squash: "LOGISTIC".to_string(),
        bias: 0.0,
    });

    CreatureJson {
        neurons,
        synapses,
        input: 1,
        output: 1,
    }
}

fn make_records(uuid: &str, seed: usize, count: u32) -> Vec<DiscoverRecord> {
    (0..count)
        .map(|i| DiscoverRecord {
            obs_index: i,
            neuron_uuid: uuid.to_string(),
            value: Some(0.1 + 0.01 * (seed as f32 + i as f32)),
            activation: 0.2 + 0.03 * seed as f32,
            errors: vec![0.05 + 0.02 * seed as f32],
        })
        .collect()
}

fn write_temp_parquet(records: &[DiscoverRecord]) -> NamedTempFile {
    let tmp = NamedTempFile::new().expect("create temp file");
    write_records_to_parquet(tmp.path().to_str().unwrap(), records).expect("write parquet");
    tmp
}

/// A second load of the same file is a cache hit: no second decode, identical
/// shared data handed back.
#[test]
#[serial]
fn second_load_is_a_cache_hit() {
    let records = vec![
        DiscoverRecord::new(1, "neuron-1".to_string(), Some(0.5), 0.7, vec![0.1]),
        DiscoverRecord::new(0, "neuron-1".to_string(), Some(0.6), 0.8, vec![0.2]),
        DiscoverRecord::new(0, "neuron-2".to_string(), Some(0.3), 0.5, vec![0.05]),
    ];
    let tmp = write_temp_parquet(&records);
    let path = tmp.path().to_str().unwrap();
    invalidate();

    let first = load_grouped_records_shared(path, None).unwrap();
    assert_eq!(decodes_for_path(path), 1, "first load should decode once");

    let second = load_grouped_records_shared(path, None).unwrap();
    assert_eq!(
        decodes_for_path(path),
        1,
        "second load on the same file must not decode again",
    );

    assert!(
        Arc::ptr_eq(&first, &second),
        "the second load must hand back the identical shared decode",
    );
    assert_eq!(first.get("neuron-1").unwrap().len(), 2);
    assert_eq!(first.get("neuron-2").unwrap().len(), 1);

    invalidate();
    assert_eq!(decodes_for_path(path), 0, "invalidate clears the slot");
}

/// A changed file (different length / mtime) invalidates the cache and decodes
/// fresh, so stale records are never served.
#[test]
#[serial]
fn changed_file_invalidates_cache() {
    let records = vec![
        DiscoverRecord::new(0, "neuron-1".to_string(), Some(0.5), 0.7, vec![0.1]),
        DiscoverRecord::new(0, "neuron-2".to_string(), Some(0.3), 0.5, vec![0.05]),
    ];
    let tmp = write_temp_parquet(&records);
    let path = tmp.path().to_str().unwrap();
    invalidate();

    let _ = load_grouped_records_shared(path, None).unwrap();
    assert_eq!(decodes_for_path(path), 1);

    // Rewrite with substantially more records so the byte length differs — a
    // robust key change even when the filesystem mtime resolution is coarse
    // enough that the rewrite shares the original timestamp.
    let bigger: Vec<DiscoverRecord> = (0..64)
        .map(|i| DiscoverRecord::new(i, "neuron-3".to_string(), Some(0.1), 0.2, vec![0.9, 0.8]))
        .collect();
    write_records_to_parquet(path, &bigger).unwrap();

    let reloaded = load_grouped_records_shared(path, None).unwrap();
    assert_eq!(
        decodes_for_path(path),
        1,
        "a changed file decodes fresh and resets the per-file counter",
    );
    assert!(reloaded.contains_key("neuron-3"));
    assert!(!reloaded.contains_key("neuron-1"));

    invalidate();
}

/// The acceptance criterion: focus selection followed by the analysis record
/// load on the same parquet file performs exactly one decode.
#[test]
#[serial]
fn focus_then_analysis_decodes_parquet_once() {
    const HIDDEN: usize = 6;
    const PER_NEURON: u32 = 32;

    let creature = make_creature(HIDDEN);
    let mut records = Vec::new();
    for i in 0..HIDDEN {
        records.extend(make_records(&format!("h{i}"), i + 1, PER_NEURON));
    }
    records.extend(make_records("out", HIDDEN + 1, PER_NEURON));

    let tmp = write_temp_parquet(&records);
    let path = tmp.path().to_str().unwrap();

    // Fresh cycle: ensure no stale slot from an earlier test.
    invalidate();

    // Phase 1 — focus selection. A small file stays on the preload path, which
    // populates the shared decode.
    let stats = rank_focus_neurons(path, &creature, None, None).expect("focus ranking succeeds");
    assert_eq!(
        stats.loading_mode,
        FocusLoadingMode::Preload,
        "small fixture should use the preload path that shares its decode",
    );
    assert_eq!(
        decodes_for_path(path),
        1,
        "focus selection should decode the parquet exactly once",
    );

    // Phase 2 — the analysis record load. This is the cache `analyze_all` builds
    // per invocation; it must reuse the focus decode rather than scan the file
    // again.
    let cache = RecordCache::new_adaptive(path).expect("analysis cache builds");
    assert!(!cache.is_empty(), "analysis cache should hold the records");

    assert_eq!(
        decodes_for_path(path),
        1,
        "analysis must reuse the focus decode — no second full parquet scan",
    );

    invalidate();
}
