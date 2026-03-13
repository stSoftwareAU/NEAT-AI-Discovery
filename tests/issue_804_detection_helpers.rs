//! Tests for Issue #804: Shared detection module helpers.
//!
//! Verifies that the `build_record_map` helper correctly constructs a lookup
//! map from neuron records, matching the behaviour previously inlined in every
//! detection module.
//!
//! ## TDD Plan
//! 1. Verify build_record_map returns correct entries for typical input
//! 2. Verify empty input produces an empty map
//! 3. Verify multiple neurons each get their own entry
//! 4. Verify the map is usable for lookups by UUID string

use neat_ai_discovery::analysis::detection::helpers::build_record_map;
use neat_ai_discovery::types::DiscoverRecord;

/// Helper: create a DiscoverRecord with minimal fields.
fn make_record(neuron_uuid: &str, obs_index: u32, activation: f32) -> DiscoverRecord {
    DiscoverRecord {
        obs_index,
        neuron_uuid: neuron_uuid.to_string(),
        value: Some(activation),
        activation,
        errors: vec![0.01],
    }
}

/// Test 1: build_record_map returns correct entries for typical input.
#[test]
fn test_build_record_map_typical_input() {
    let neuron_records = vec![
        (
            "neuron-a".to_string(),
            vec![
                make_record("neuron-a", 0, 0.5),
                make_record("neuron-a", 1, 0.6),
            ],
        ),
        (
            "neuron-b".to_string(),
            vec![make_record("neuron-b", 0, -0.3)],
        ),
    ];

    let map = build_record_map(&neuron_records);

    assert_eq!(map.len(), 2);
    assert_eq!(map["neuron-a"].len(), 2);
    assert_eq!(map["neuron-b"].len(), 1);
}

/// Test 2: Empty input produces an empty map.
#[test]
fn test_build_record_map_empty_input() {
    let neuron_records: Vec<(String, Vec<DiscoverRecord>)> = vec![];
    let map = build_record_map(&neuron_records);
    assert!(map.is_empty());
}

/// Test 3: Multiple neurons each get their own entry with correct records.
#[test]
fn test_build_record_map_multiple_neurons() {
    let neuron_records = vec![
        (
            "uuid-1".to_string(),
            vec![
                make_record("uuid-1", 0, 1.0),
                make_record("uuid-1", 1, 2.0),
                make_record("uuid-1", 2, 3.0),
            ],
        ),
        ("uuid-2".to_string(), vec![make_record("uuid-2", 0, -1.0)]),
        (
            "uuid-3".to_string(),
            vec![make_record("uuid-3", 0, 0.0), make_record("uuid-3", 1, 0.1)],
        ),
    ];

    let map = build_record_map(&neuron_records);

    assert_eq!(map.len(), 3);
    assert_eq!(map["uuid-1"].len(), 3);
    assert_eq!(map["uuid-2"].len(), 1);
    assert_eq!(map["uuid-3"].len(), 2);

    // Verify actual record data is accessible
    assert!((map["uuid-1"][0].activation - 1.0).abs() < f32::EPSILON);
    assert!((map["uuid-2"][0].activation - (-1.0)).abs() < f32::EPSILON);
}

/// Test 4: The map is usable for lookups by UUID string (the primary use case).
#[test]
fn test_build_record_map_string_lookup() {
    let neuron_records = vec![(
        "target-uuid".to_string(),
        vec![make_record("target-uuid", 0, 0.42)],
    )];

    let map = build_record_map(&neuron_records);

    // Look up with a &str — the common usage pattern in detection modules
    let lookup_key: &str = "target-uuid";
    let records = map.get(lookup_key);
    assert!(records.is_some());
    assert_eq!(records.unwrap().len(), 1);
    assert!((records.unwrap()[0].activation - 0.42).abs() < f32::EPSILON);

    // Non-existent key returns None
    assert!(!map.contains_key("missing-uuid"));
}
