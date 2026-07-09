//! Tests for Issue #1543: Arc-share `DiscoverRecord` across detection modules.
//!
//! The bulk `load_records_for_*` loaders previously deep-cloned each neuron's
//! inner `Vec<DiscoverRecord>` for every one of the ~48 discovery modules
//! dispatched per `analyze_all` pass. They now return [`SharedRecords`], a cheap
//! `Arc::clone` of the cache's existing allocation.
//!
//! These tests are the deterministic regression guard the issue asks for: they
//! assert the loader output shares the *same allocation* as the cached entry via
//! `Arc::ptr_eq`. A future revert to deep-cloning would fail here immediately,
//! rather than only showing up as a slow benchmark / RSS regression.

use std::collections::HashMap;
use std::sync::Arc;

use neat_ai_discovery::analysis::cache::RecordCache;
use neat_ai_discovery::types::DiscoverRecord;

use crate::common::{hidden, make_creature, output, record, synapse};

/// Build a preloaded-style test cache backed by in-memory data. The custom
/// loader mirrors the preloaded path: the first `get()` for a UUID materialises
/// the records behind an `Arc`, and every later access hands out `Arc::clone`s of
/// that same allocation.
fn test_cache_with_data(data: Vec<(String, Vec<DiscoverRecord>)>) -> RecordCache {
    let data_map: HashMap<String, Vec<DiscoverRecord>> = data.into_iter().collect();
    let data_map = Arc::new(data_map);

    RecordCache::with_loader(
        "test.parquet",
        Arc::new(move |_file: &str, neuron_uuid: &str| {
            Ok(data_map.get(neuron_uuid).cloned().unwrap_or_default())
        }),
    )
}

/// `load_records_for_uuids` hands out the cache's own `Arc` allocation, not a
/// deep copy.
#[test]
fn load_records_for_uuids_shares_cache_allocation() {
    let cache = test_cache_with_data(vec![
        ("h1".to_string(), vec![record("h1", 0, 0.5, None)]),
        ("h2".to_string(), vec![record("h2", 0, 0.6, None)]),
    ]);

    let uuids = vec!["h1".to_string(), "h2".to_string()];
    let loaded = cache.load_records_for_uuids(&uuids);
    assert_eq!(loaded.len(), 2);

    for (uuid, shared) in &loaded {
        let cached = cache.get(uuid).expect("cached entry exists");
        assert!(
            Arc::ptr_eq(shared.arc(), &cached),
            "loader for {uuid} must share the cache allocation, not deep-clone it"
        );
    }
}

/// `load_records_for_hidden` shares the cache allocation for every hidden neuron.
#[test]
fn load_records_for_hidden_shares_cache_allocation() {
    let cache = test_cache_with_data(vec![
        ("h1".to_string(), vec![record("h1", 0, 0.5, None)]),
        ("h2".to_string(), vec![record("h2", 0, 0.6, None)]),
    ]);

    let hidden_tuples = vec![
        ("h1".to_string(), "TANH".to_string(), 0.0_f32),
        ("h2".to_string(), "TANH".to_string(), 0.0_f32),
    ];
    let loaded = cache.load_records_for_hidden(&hidden_tuples);
    assert_eq!(loaded.len(), 2);

    for (uuid, shared) in &loaded {
        let cached = cache.get(uuid).expect("cached entry exists");
        assert!(
            Arc::ptr_eq(shared.arc(), &cached),
            "hidden loader for {uuid} must share the cache allocation"
        );
    }
}

/// `load_records_for_all_neurons` shares the cache allocation and preserves the
/// records themselves (behaviour unchanged — only the ownership is shared).
#[test]
fn load_records_for_all_neurons_shares_allocation_and_preserves_data() {
    let creature = make_creature(
        vec![hidden("h1", "TANH"), output("o1", "IDENTITY")],
        vec![synapse("h1", "o1", 0.8)],
    );

    let cache = test_cache_with_data(vec![
        (
            "h1".to_string(),
            vec![record("h1", 0, 0.5, None), record("h1", 1, 0.7, None)],
        ),
        ("o1".to_string(), vec![record("o1", 0, 0.9, None)]),
    ]);

    let loaded = cache.load_records_for_all_neurons(&creature);
    assert_eq!(loaded.len(), 2);

    for (uuid, shared) in &loaded {
        let cached = cache.get(uuid).expect("cached entry exists");
        // Same allocation …
        assert!(
            Arc::ptr_eq(shared.arc(), &cached),
            "all-neurons loader for {uuid} must share the cache allocation"
        );
        // … and the same record contents as the cache holds.
        assert_eq!(shared.arc().as_slice(), cached.as_slice());
    }

    // Records survive the Arc-sharing intact (no behavioural change).
    let h1 = loaded
        .iter()
        .find(|(u, _)| u == "h1")
        .expect("h1 present in load");
    assert_eq!(h1.1.arc().len(), 2);
}

/// Two independent loader calls for the same neuron return the *same* shared
/// allocation — proving no per-call deep copy is made.
#[test]
fn repeated_loads_return_same_allocation() {
    let cache = test_cache_with_data(vec![("h1".to_string(), vec![record("h1", 0, 0.5, None)])]);

    let uuids = vec!["h1".to_string()];
    let first = cache.load_records_for_uuids(&uuids);
    let second = cache.load_records_for_uuids(&uuids);

    assert!(
        Arc::ptr_eq(first[0].1.arc(), second[0].1.arc()),
        "repeated loads must share one allocation, not clone per call"
    );
}
