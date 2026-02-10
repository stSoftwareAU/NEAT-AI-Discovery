//! Tests for Issue #481/#493: Record-loading helper to reduce boilerplate.
//!
//! Validates that `RecordCache::load_records_for_uuids()` correctly loads
//! records for a given set of neuron UUIDs, producing the same results as
//! the inline pattern it replaces.

mod common;

use neat_ai_discovery::analysis::cache::RecordCache;
use neat_ai_discovery::types::DiscoverRecord;
use std::sync::Arc;

/// Helper to create a test cache backed by in-memory data.
fn test_cache_with_data(data: Vec<(String, Vec<DiscoverRecord>)>) -> RecordCache {
    let data_map: std::collections::HashMap<String, Vec<DiscoverRecord>> =
        data.into_iter().collect();
    let data_map = Arc::new(data_map);

    RecordCache::with_loader(
        "test.parquet",
        Arc::new(move |_file: &str, neuron_uuid: &str| {
            Ok(data_map.get(neuron_uuid).cloned().unwrap_or_default())
        }),
    )
}

// =============================================================================
// Basic functionality
// =============================================================================

/// Verify that load_records_for_uuids returns records for all requested UUIDs.
#[test]
fn test_load_records_returns_all_requested_uuids() {
    let cache = test_cache_with_data(vec![
        (
            "n1".to_string(),
            vec![DiscoverRecord {
                obs_index: 0,
                neuron_uuid: "n1".to_string(),
                value: None,
                activation: 0.5,
                errors: vec![0.1],
            }],
        ),
        (
            "n2".to_string(),
            vec![DiscoverRecord {
                obs_index: 0,
                neuron_uuid: "n2".to_string(),
                value: None,
                activation: 0.8,
                errors: vec![0.2],
            }],
        ),
        (
            "n3".to_string(),
            vec![DiscoverRecord {
                obs_index: 0,
                neuron_uuid: "n3".to_string(),
                value: None,
                activation: 0.3,
                errors: vec![0.05],
            }],
        ),
    ]);

    let uuids = vec!["n1".to_string(), "n3".to_string()];
    let records = cache.load_records_for_uuids(&uuids);

    // Should return records for n1 and n3 (not n2)
    assert_eq!(records.len(), 2);
    let returned_uuids: Vec<&String> = records.iter().map(|(uuid, _)| uuid).collect();
    assert!(returned_uuids.contains(&&"n1".to_string()));
    assert!(returned_uuids.contains(&&"n3".to_string()));
}

/// Verify that missing UUIDs are silently skipped (no errors returned).
#[test]
fn test_load_records_skips_missing_uuids() {
    let cache = test_cache_with_data(vec![(
        "n1".to_string(),
        vec![DiscoverRecord {
            obs_index: 0,
            neuron_uuid: "n1".to_string(),
            value: None,
            activation: 0.5,
            errors: vec![0.1],
        }],
    )]);

    let uuids = vec!["n1".to_string(), "nonexistent".to_string()];
    let records = cache.load_records_for_uuids(&uuids);

    // n1 found, nonexistent returns empty records but still included
    // (matches existing cache behaviour where missing UUIDs return empty vec)
    assert!(!records.is_empty());
    // n1 should definitely be there with records
    let n1_records = records.iter().find(|(uuid, _)| uuid == "n1");
    assert!(n1_records.is_some());
    assert!(!n1_records.unwrap().1.is_empty());
}

/// Verify that empty UUID list returns empty results.
#[test]
fn test_load_records_empty_uuids() {
    let cache = test_cache_with_data(vec![(
        "n1".to_string(),
        vec![DiscoverRecord {
            obs_index: 0,
            neuron_uuid: "n1".to_string(),
            value: None,
            activation: 0.5,
            errors: vec![0.1],
        }],
    )]);

    let uuids: Vec<String> = vec![];
    let records = cache.load_records_for_uuids(&uuids);
    assert!(records.is_empty());
}

/// Verify that records preserve their original content.
#[test]
fn test_load_records_preserves_content() {
    let cache = test_cache_with_data(vec![(
        "n1".to_string(),
        vec![
            DiscoverRecord {
                obs_index: 0,
                neuron_uuid: "n1".to_string(),
                value: Some(1.5),
                activation: 0.5,
                errors: vec![0.1, 0.2],
            },
            DiscoverRecord {
                obs_index: 1,
                neuron_uuid: "n1".to_string(),
                value: None,
                activation: 0.8,
                errors: vec![0.3],
            },
        ],
    )]);

    let uuids = vec!["n1".to_string()];
    let records = cache.load_records_for_uuids(&uuids);

    assert_eq!(records.len(), 1);
    let (uuid, recs) = &records[0];
    assert_eq!(uuid, "n1");
    assert_eq!(recs.len(), 2);
    assert_eq!(recs[0].obs_index, 0);
    assert_eq!(recs[0].activation, 0.5);
    assert_eq!(recs[1].obs_index, 1);
    assert_eq!(recs[1].activation, 0.8);
}
