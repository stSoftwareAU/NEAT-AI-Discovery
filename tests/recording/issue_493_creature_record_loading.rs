//! Tests for Issue #493: Creature-aware record-loading helpers on `RecordCache`.
//!
//! Validates that the new convenience methods correctly combine UUID extraction
//! from `CreatureJson` with record loading, eliminating the repeated
//! UUID-collection boilerplate in `analyze_all()`.

use crate::common::{hidden, make_creature, neuron, output, record, synapse};
use neat_ai_discovery::analysis::cache::RecordCache;
use neat_ai_discovery::types::DiscoverRecord;
use std::collections::HashMap;
use std::sync::Arc;

/// Helper to create a test cache backed by in-memory data.
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

// =============================================================================
// load_records_for_all_neurons
// =============================================================================

/// Verify that `load_records_for_all_neurons` returns records for every neuron in the creature.
#[test]
fn test_load_all_neurons_returns_records_for_every_neuron() {
    let creature = make_creature(
        vec![
            neuron("i1", "input", "IDENTITY"),
            neuron("i2", "input", "IDENTITY"),
            hidden("h1", "LOGISTIC"),
            output("o1", "LOGISTIC"),
        ],
        vec![synapse("i1", "h1", 0.5), synapse("h1", "o1", 0.8)],
    );

    let cache = test_cache_with_data(vec![
        ("i1".to_string(), vec![record("i1", 0, 0.1, Some(1.0))]),
        ("i2".to_string(), vec![record("i2", 0, 0.2, Some(2.0))]),
        ("h1".to_string(), vec![record("h1", 0, 0.5, None)]),
        ("o1".to_string(), vec![record("o1", 0, 0.9, None)]),
    ]);

    let records = cache.load_records_for_all_neurons(&creature);

    // Should return records for all 4 neurons
    assert_eq!(records.len(), 4);
    let uuids: Vec<&String> = records.iter().map(|(u, _)| u).collect();
    assert!(uuids.contains(&&"i1".to_string()));
    assert!(uuids.contains(&&"i2".to_string()));
    assert!(uuids.contains(&&"h1".to_string()));
    assert!(uuids.contains(&&"o1".to_string()));
}

/// Verify that `load_records_for_all_neurons` handles a creature with no neurons.
#[test]
fn test_load_all_neurons_empty_creature() {
    let creature = make_creature(vec![], vec![]);
    let cache = test_cache_with_data(vec![("n1".to_string(), vec![record("n1", 0, 0.5, None)])]);

    let records = cache.load_records_for_all_neurons(&creature);
    assert!(records.is_empty());
}

// =============================================================================
// load_records_for_neuron_types
// =============================================================================

/// Verify that `load_records_for_neuron_types` returns only the requested types.
#[test]
fn test_load_by_type_filters_correctly() {
    let creature = make_creature(
        vec![
            neuron("i1", "input", "IDENTITY"),
            neuron("i2", "input", "IDENTITY"),
            hidden("h1", "LOGISTIC"),
            output("o1", "LOGISTIC"),
        ],
        vec![synapse("i1", "h1", 0.5), synapse("h1", "o1", 0.8)],
    );

    let cache = test_cache_with_data(vec![
        ("i1".to_string(), vec![record("i1", 0, 0.1, Some(1.0))]),
        ("i2".to_string(), vec![record("i2", 0, 0.2, Some(2.0))]),
        ("h1".to_string(), vec![record("h1", 0, 0.5, None)]),
        ("o1".to_string(), vec![record("o1", 0, 0.9, None)]),
    ]);

    // Request only output neurons
    let records = cache.load_records_for_neuron_types(&creature, &["output"]);
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].0, "o1");

    // Request only input neurons
    let records = cache.load_records_for_neuron_types(&creature, &["input"]);
    assert_eq!(records.len(), 2);
    let uuids: Vec<&String> = records.iter().map(|(u, _)| u).collect();
    assert!(uuids.contains(&&"i1".to_string()));
    assert!(uuids.contains(&&"i2".to_string()));
}

/// Verify that `load_records_for_neuron_types` supports multiple type filters.
#[test]
fn test_load_by_type_multiple_types() {
    let creature = make_creature(
        vec![
            neuron("i1", "input", "IDENTITY"),
            hidden("h1", "LOGISTIC"),
            output("o1", "LOGISTIC"),
        ],
        vec![synapse("i1", "h1", 0.5), synapse("h1", "o1", 0.8)],
    );

    let cache = test_cache_with_data(vec![
        ("i1".to_string(), vec![record("i1", 0, 0.1, Some(1.0))]),
        ("h1".to_string(), vec![record("h1", 0, 0.5, None)]),
        ("o1".to_string(), vec![record("o1", 0, 0.9, None)]),
    ]);

    // Request input + output
    let records = cache.load_records_for_neuron_types(&creature, &["input", "output"]);
    assert_eq!(records.len(), 2);
    let uuids: Vec<&String> = records.iter().map(|(u, _)| u).collect();
    assert!(uuids.contains(&&"i1".to_string()));
    assert!(uuids.contains(&&"o1".to_string()));

    // Request input + hidden
    let records = cache.load_records_for_neuron_types(&creature, &["input", "hidden"]);
    assert_eq!(records.len(), 2);
    let uuids: Vec<&String> = records.iter().map(|(u, _)| u).collect();
    assert!(uuids.contains(&&"i1".to_string()));
    assert!(uuids.contains(&&"h1".to_string()));
}

/// Verify that `load_records_for_neuron_types` with no matching types returns empty.
#[test]
fn test_load_by_type_no_matching_types() {
    let creature = make_creature(
        vec![neuron("i1", "input", "IDENTITY"), output("o1", "LOGISTIC")],
        vec![],
    );

    let cache = test_cache_with_data(vec![
        ("i1".to_string(), vec![record("i1", 0, 0.1, Some(1.0))]),
        ("o1".to_string(), vec![record("o1", 0, 0.9, None)]),
    ]);

    let records = cache.load_records_for_neuron_types(&creature, &["hidden"]);
    assert!(records.is_empty());
}

// =============================================================================
// load_records_for_synapse_sources
// =============================================================================

/// Verify that `load_records_for_synapse_sources` returns unique source UUIDs.
#[test]
fn test_load_synapse_sources_returns_unique_sources() {
    let creature = make_creature(
        vec![
            neuron("i1", "input", "IDENTITY"),
            neuron("i2", "input", "IDENTITY"),
            hidden("h1", "LOGISTIC"),
            output("o1", "LOGISTIC"),
        ],
        vec![
            synapse("i1", "h1", 0.5),
            synapse("i1", "o1", 0.3), // i1 appears twice as source
            synapse("i2", "h1", 0.7),
            synapse("h1", "o1", 0.8),
        ],
    );

    let cache = test_cache_with_data(vec![
        ("i1".to_string(), vec![record("i1", 0, 0.1, Some(1.0))]),
        ("i2".to_string(), vec![record("i2", 0, 0.2, Some(2.0))]),
        ("h1".to_string(), vec![record("h1", 0, 0.5, None)]),
    ]);

    let records = cache.load_records_for_synapse_sources(&creature);

    // Should return records for 3 unique source UUIDs: i1, i2, h1
    assert_eq!(records.len(), 3);
    let uuids: Vec<&String> = records.iter().map(|(u, _)| u).collect();
    assert!(uuids.contains(&&"i1".to_string()));
    assert!(uuids.contains(&&"i2".to_string()));
    assert!(uuids.contains(&&"h1".to_string()));
}

/// Verify that `load_records_for_synapse_sources` handles empty synapses.
#[test]
fn test_load_synapse_sources_empty_synapses() {
    let creature = make_creature(vec![neuron("i1", "input", "IDENTITY")], vec![]);
    let cache = test_cache_with_data(vec![(
        "i1".to_string(),
        vec![record("i1", 0, 0.1, Some(1.0))],
    )]);

    let records = cache.load_records_for_synapse_sources(&creature);
    assert!(records.is_empty());
}

// =============================================================================
// load_records_for_hidden consistency
// =============================================================================

/// Verify that `load_records_for_hidden` returns records matching the hidden neuron tuples.
#[test]
fn test_load_hidden_returns_correct_records() {
    let cache = test_cache_with_data(vec![
        ("h1".to_string(), vec![record("h1", 0, 0.5, None)]),
        ("h2".to_string(), vec![record("h2", 0, 0.7, None)]),
        ("o1".to_string(), vec![record("o1", 0, 0.9, None)]),
    ]);

    let hidden_neurons = vec![
        ("h1".to_string(), "LOGISTIC".to_string(), 0.0_f32),
        ("h2".to_string(), "TANH".to_string(), 0.1_f32),
    ];

    let records = cache.load_records_for_hidden(&hidden_neurons);

    assert_eq!(records.len(), 2);
    let uuids: Vec<&String> = records.iter().map(|(u, _)| u).collect();
    assert!(uuids.contains(&&"h1".to_string()));
    assert!(uuids.contains(&&"h2".to_string()));
}
