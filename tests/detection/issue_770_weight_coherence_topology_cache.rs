//! Tests for Issue #770: Pass `CreatureTopologyCache` to `weight_coherence` detection functions.
//!
//! Verifies that the weight coherence detection functions produce identical results
//! when given a pre-computed topology cache vs building locally (None).

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use crate::common::{make_creature, neuron, synapse};
use neat_ai_discovery::analysis::detection::topology_cache::CreatureTopologyCache;
use neat_ai_discovery::analysis::detection::weight_coherence::{
    WeightCoherenceConfig, detect_incoherent_weight_ratios, detect_near_constant_paths,
    detect_symmetric_cancellation,
};
use neat_ai_discovery::types::DiscoverRecord;

// =============================================================================
// Helper: build records for weight coherence tests
// =============================================================================

fn make_hidden_records(uuid: &str, count: u32) -> (String, Vec<DiscoverRecord>) {
    let records: Vec<DiscoverRecord> = (0..count)
        .map(|i| {
            let activation = ((i as f32 * 0.1).sin()).tanh();
            DiscoverRecord {
                obs_index: i,
                neuron_uuid: uuid.to_string(),
                value: Some(activation),
                activation,
                errors: vec![0.1],
            }
        })
        .collect();
    (uuid.to_string(), records)
}

fn make_constant_records(uuid: &str, count: u32) -> (String, Vec<DiscoverRecord>) {
    let records: Vec<DiscoverRecord> = (0..count)
        .map(|i| {
            let activation = 0.999 + 0.0001 * ((i as f32 * 0.1).sin());
            DiscoverRecord {
                obs_index: i,
                neuron_uuid: uuid.to_string(),
                value: Some(10.0),
                activation,
                errors: vec![0.05],
            }
        })
        .collect();
    (uuid.to_string(), records)
}

fn make_correlated_records(uuid: &str, count: u32) -> (String, Vec<DiscoverRecord>) {
    let records: Vec<DiscoverRecord> = (0..count)
        .map(|i| {
            let activation = 0.5 + 0.4 * ((i as f32 * 0.1).sin());
            DiscoverRecord {
                obs_index: i,
                neuron_uuid: uuid.to_string(),
                value: Some(activation),
                activation,
                errors: vec![0.01],
            }
        })
        .collect();
    (uuid.to_string(), records)
}

// =============================================================================
// Test 1: detect_incoherent_weight_ratios produces same results with cache
// =============================================================================

#[test]
fn test_incoherent_weight_ratios_with_topology_cache() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("hidden-imbalanced", "hidden", "TANH"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "hidden-imbalanced", 50.0),
            synapse("hidden-imbalanced", "output-1", 0.001),
        ],
    );

    let records = vec![make_hidden_records("hidden-imbalanced", 100)];
    let config = WeightCoherenceConfig::default();
    let topo = CreatureTopologyCache::new(&creature);

    // Without cache (backward compatibility)
    let without_cache = detect_incoherent_weight_ratios(&creature, &records, &config, None);

    // With cache
    let with_cache = detect_incoherent_weight_ratios(&creature, &records, &config, Some(&topo));

    assert_eq!(
        without_cache.len(),
        with_cache.len(),
        "Should detect same number of candidates with and without cache"
    );
    assert!(!with_cache.is_empty(), "Should detect the incoherent ratio");

    // Verify same neuron detected
    assert_eq!(
        without_cache[0].neuron_uuid, with_cache[0].neuron_uuid,
        "Same neuron should be detected"
    );
    assert_eq!(
        without_cache[0].incoming_outgoing_ratio, with_cache[0].incoming_outgoing_ratio,
        "Same ratio should be computed"
    );
}

// =============================================================================
// Test 2: detect_near_constant_paths produces same results with cache
// =============================================================================

#[test]
fn test_near_constant_paths_with_topology_cache() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("hidden-constant", "hidden", "TANH"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "hidden-constant", 100.0),
            synapse("hidden-constant", "output-1", 0.1),
        ],
    );

    let records = vec![make_constant_records("hidden-constant", 100)];
    let config = WeightCoherenceConfig::default();
    let topo = CreatureTopologyCache::new(&creature);

    let without_cache = detect_near_constant_paths(&creature, &records, &config, None);
    let with_cache = detect_near_constant_paths(&creature, &records, &config, Some(&topo));

    assert_eq!(
        without_cache.len(),
        with_cache.len(),
        "Should detect same number of candidates with and without cache"
    );
    assert!(
        !with_cache.is_empty(),
        "Should detect the near-constant path"
    );
    assert_eq!(
        without_cache[0].neuron_uuid, with_cache[0].neuron_uuid,
        "Same neuron should be detected"
    );
    assert_eq!(
        without_cache[0].activation_variance, with_cache[0].activation_variance,
        "Same variance should be computed"
    );
}

// =============================================================================
// Test 3: detect_symmetric_cancellation produces same results with cache
// =============================================================================

#[test]
fn test_symmetric_cancellation_with_topology_cache() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("input-2", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "output-1", 5.0),
            synapse("input-2", "output-1", -5.0),
        ],
    );

    let records = vec![
        make_correlated_records("input-1", 100),
        make_correlated_records("input-2", 100),
        make_correlated_records("output-1", 100),
    ];

    let config = WeightCoherenceConfig::default();
    let topo = CreatureTopologyCache::new(&creature);

    let without_cache = detect_symmetric_cancellation(&creature, &records, &config, None);
    let with_cache = detect_symmetric_cancellation(&creature, &records, &config, Some(&topo));

    assert_eq!(
        without_cache.len(),
        with_cache.len(),
        "Should detect same number of candidates with and without cache"
    );
    assert!(
        !with_cache.is_empty(),
        "Should detect symmetric cancellation"
    );
    assert_eq!(
        without_cache[0].target_neuron_uuid, with_cache[0].target_neuron_uuid,
        "Same target neuron should be detected"
    );
}

// =============================================================================
// Test 4: Multiple hidden neurons with cache
// =============================================================================

#[test]
fn test_multiple_hidden_neurons_with_cache() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("hidden-1", "hidden", "TANH"),
            neuron("hidden-2", "hidden", "RELU"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "hidden-1", 50.0),
            synapse("input-1", "hidden-2", 30.0),
            synapse("hidden-1", "output-1", 0.001),
            synapse("hidden-2", "output-1", 0.002),
        ],
    );

    let records = vec![
        make_hidden_records("hidden-1", 100),
        make_hidden_records("hidden-2", 100),
    ];
    let config = WeightCoherenceConfig::default();
    let topo = CreatureTopologyCache::new(&creature);

    let without_cache = detect_incoherent_weight_ratios(&creature, &records, &config, None);
    let with_cache = detect_incoherent_weight_ratios(&creature, &records, &config, Some(&topo));

    assert_eq!(
        without_cache.len(),
        with_cache.len(),
        "Same count with multiple hidden neurons"
    );

    // Both should detect at least one incoherent ratio
    assert!(
        !with_cache.is_empty(),
        "Should detect incoherent ratios in multi-hidden network"
    );
}

// =============================================================================
// Test 5: No hidden neurons with cache returns empty
// =============================================================================

#[test]
fn test_no_hidden_neurons_with_cache_returns_empty() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![synapse("input-1", "output-1", 50.0)],
    );

    let records = vec![make_hidden_records("input-1", 100)];
    let config = WeightCoherenceConfig::default();
    let topo = CreatureTopologyCache::new(&creature);

    let with_cache = detect_incoherent_weight_ratios(&creature, &records, &config, Some(&topo));
    assert!(
        with_cache.is_empty(),
        "No hidden neurons means no candidates"
    );
}
