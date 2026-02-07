//! Tests for Issue #424: Consolidate discovery constants into central module.
//!
//! Verifies that the central `constants` module provides a single source of truth
//! for discovery thresholds, and that all analysis modules use these constants
//! consistently.
//!
//! ## TDD Plan
//! 1. Test that constants are accessible from the central module
//! 2. Test that constant values match expected defaults
//! 3. Test that detection modules still function correctly using centralised constants
//! 4. Test that sentinel-related constants are consistent across observation, bounded-range,
//!    and sentinel-gating modules

mod common;

use neat_ai_discovery::analysis::constants;

/// Verify MIN_NEURON_SAMPLE_COUNT is accessible from the central constants module.
#[test]
fn test_min_neuron_sample_count_accessible() {
    assert_eq!(constants::MIN_NEURON_SAMPLE_COUNT, 10);
}

/// Verify MIN_DISCOVERY_SAMPLE_COUNT is accessible from the central constants module.
#[test]
fn test_min_discovery_sample_count_accessible() {
    assert_eq!(constants::MIN_DISCOVERY_SAMPLE_COUNT, 20);
}

/// Verify sentinel detection constants are accessible from the central module.
#[test]
fn test_sentinel_constants_accessible() {
    assert_eq!(constants::CANDIDATE_SENTINELS, [-1.0_f32, 0.0, 1.0]);
    assert!((constants::MIN_SENTINEL_FRACTION - 0.15).abs() < f32::EPSILON);
    assert!((constants::SENTINEL_TOLERANCE - 0.02).abs() < f32::EPSILON);
    assert!((constants::MIN_SENTINEL_GAP - 0.05).abs() < f32::EPSILON);
}

/// Verify MIN_SOURCE_STD_DEV is accessible from the central module.
#[test]
fn test_min_source_std_dev_accessible() {
    assert!((constants::MIN_SOURCE_STD_DEV - 0.05).abs() < f32::EPSILON);
}

/// Verify DIVERSIFY_TOP_K is accessible from the central module.
#[test]
fn test_diversify_top_k_accessible() {
    assert_eq!(constants::DIVERSIFY_TOP_K, 64);
}

/// Verify that saturated neuron detection still works (uses MIN_NEURON_SAMPLE_COUNT
/// indirectly via sample count filtering).
#[test]
fn test_saturation_detection_uses_centralised_constants() {
    use neat_ai_discovery::analysis::saturation::detect_saturated_neurons;
    use neat_ai_discovery::types::DiscoverRecord;

    let neurons = vec![("h1".to_string(), "LOGISTIC".to_string(), 0.0_f32)];

    // Fewer than MIN_NEURON_SAMPLE_COUNT (10) samples — should return no detections
    let too_few_records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "h1".to_string(),
        (0..5)
            .map(|i| DiscoverRecord {
                obs_index: i,
                neuron_uuid: "h1".to_string(),
                value: None,
                activation: 0.999,
                errors: vec![0.1],
            })
            .collect(),
    )];

    let detected = detect_saturated_neurons(&neurons, &too_few_records);
    assert!(
        detected.is_empty(),
        "Should not detect saturation with fewer than MIN_NEURON_SAMPLE_COUNT samples"
    );
}

/// Verify that dead neuron detection still works with centralised constants.
#[test]
fn test_dead_neuron_detection_uses_centralised_constants() {
    use neat_ai_discovery::analysis::dead_neuron::detect_dead_neurons;
    use neat_ai_discovery::types::DiscoverRecord;
    use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

    let creature = CreatureJson {
        neurons: vec![
            NeuronJson {
                uuid: "input-0".into(),
                neuron_type: "input".into(),
                squash: "IDENTITY".into(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "h1".into(),
                neuron_type: "hidden".into(),
                squash: "LOGISTIC".into(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "output-0".into(),
                neuron_type: "output".into(),
                squash: "LOGISTIC".into(),
                bias: 0.0,
            },
        ],
        synapses: vec![
            SynapseJson {
                from_uuid: "input-0".into(),
                to_uuid: "h1".into(),
                weight: 1.0,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "h1".into(),
                to_uuid: "output-0".into(),
                weight: 1.0,
                synapse_type: None,
            },
        ],
        input: 1,
        output: 1,
    };

    // Fewer than MIN_DISCOVERY_SAMPLE_COUNT (20) samples — should return no detections
    let too_few: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "h1".to_string(),
        (0..5)
            .map(|i| DiscoverRecord {
                obs_index: i,
                neuron_uuid: "h1".to_string(),
                value: None,
                activation: 0.0,
                errors: vec![0.0],
            })
            .collect(),
    )];

    let detected = detect_dead_neurons(&creature, &too_few);
    assert!(
        detected.is_empty(),
        "Should not detect dead neurons with fewer than MIN_DISCOVERY_SAMPLE_COUNT samples"
    );
}
