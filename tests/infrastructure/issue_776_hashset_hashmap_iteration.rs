//! Tests for Issue #776: Replace HashSet/HashMap used only for iteration.
//!
//! Verifies that weight coherence detection functions produce correct results
//! after replacing HashMap/HashSet with Vec where collections are only iterated.

use crate::common::{make_creature, neuron, synapse};
use neat_ai_discovery::analysis::detection::weight_coherence::{
    WeightCoherenceConfig, detect_near_constant_paths, detect_symmetric_cancellation,
};
use neat_ai_discovery::types::DiscoverRecord;

/// Helper to build discovery records for a neuron with specified activations.
fn make_records(uuid: &str, activations: &[f32]) -> (String, Vec<DiscoverRecord>) {
    let records = activations
        .iter()
        .enumerate()
        .map(|(i, &activation)| DiscoverRecord {
            obs_index: i as u32,
            neuron_uuid: uuid.to_string(),
            value: Some(activation),
            activation,
            errors: vec![0.01],
        })
        .collect();
    (uuid.to_string(), records)
}

// =============================================================================
// Test: detect_near_constant_paths produces correct results
// =============================================================================

/// Verify near-constant path detection works correctly with hidden neurons
/// that have very low activation variance (saturated TANH).
#[test]
fn test_near_constant_paths_detects_saturated_hidden_neuron() {
    let creature = make_creature(
        vec![
            neuron("in-1", "input", "IDENTITY"),
            neuron("h-saturated", "hidden", "TANH"),
            neuron("out-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("in-1", "h-saturated", 100.0), // Large weight causes saturation
            synapse("h-saturated", "out-1", 0.5),
        ],
    );

    // Near-constant activations (saturated at ~1.0)
    let activations: Vec<f32> = (0..50).map(|_| 0.9999).collect();
    let records = vec![make_records("h-saturated", &activations)];

    let config = WeightCoherenceConfig::default();
    let candidates = detect_near_constant_paths(&creature, &records, &config, None);

    assert!(
        !candidates.is_empty(),
        "Should detect near-constant path for saturated neuron"
    );
    assert_eq!(candidates[0].neuron_uuid, "h-saturated");
    assert!(
        candidates[0].activation_variance < config.min_activation_variance,
        "Variance should be below threshold"
    );
    // Saturating squash (TANH) with high mean → should recommend setBias
    assert_eq!(candidates[0].recommended_action, "setBias");
}

/// Verify that neurons with sufficient variance are NOT flagged.
#[test]
fn test_near_constant_paths_skips_variable_neurons() {
    let creature = make_creature(
        vec![
            neuron("in-1", "input", "IDENTITY"),
            neuron("h-active", "hidden", "TANH"),
            neuron("out-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("in-1", "h-active", 1.0),
            synapse("h-active", "out-1", 0.5),
        ],
    );

    // Varied activations — should NOT be flagged
    let activations: Vec<f32> = (0..50).map(|i| (i as f32 * 0.1).sin()).collect();
    let records = vec![make_records("h-active", &activations)];

    let config = WeightCoherenceConfig::default();
    let candidates = detect_near_constant_paths(&creature, &records, &config, None);

    assert!(
        candidates.is_empty(),
        "Should not flag neurons with sufficient activation variance"
    );
}

// =============================================================================
// Test: detect_symmetric_cancellation produces correct results
// =============================================================================

/// Verify symmetric cancellation detection finds opposite-sign, correlated inputs.
#[test]
fn test_symmetric_cancellation_detects_opposite_correlated_inputs() {
    let creature = make_creature(
        vec![
            neuron("in-1", "input", "IDENTITY"),
            neuron("in-2", "input", "IDENTITY"),
            neuron("h-target", "hidden", "TANH"),
            neuron("out-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("in-1", "h-target", 2.0),  // Positive weight
            synapse("in-2", "h-target", -2.0), // Negative weight (opposite sign, same magnitude)
            synapse("h-target", "out-1", 1.0),
        ],
    );

    // Highly correlated activations for both inputs
    let act1: Vec<f32> = (0..50).map(|i| i as f32 * 0.1).collect();
    let act2: Vec<f32> = (0..50).map(|i| i as f32 * 0.1 + 0.01).collect(); // Nearly identical

    let records = vec![make_records("in-1", &act1), make_records("in-2", &act2)];

    let config = WeightCoherenceConfig::default();
    let candidates = detect_symmetric_cancellation(&creature, &records, &config, None);

    assert!(
        !candidates.is_empty(),
        "Should detect symmetric cancellation between correlated opposite-weight inputs"
    );
    assert_eq!(candidates[0].target_neuron_uuid, "h-target");
    assert!(
        candidates[0].correlation.abs() >= config.min_correlation_for_cancellation,
        "Correlation should exceed threshold: {}",
        candidates[0].correlation
    );
}

/// Verify that uncorrelated inputs are NOT flagged even with opposite weights.
#[test]
fn test_symmetric_cancellation_skips_uncorrelated_inputs() {
    let creature = make_creature(
        vec![
            neuron("in-1", "input", "IDENTITY"),
            neuron("in-2", "input", "IDENTITY"),
            neuron("h-target", "hidden", "TANH"),
            neuron("out-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("in-1", "h-target", 2.0),
            synapse("in-2", "h-target", -2.0),
            synapse("h-target", "out-1", 1.0),
        ],
    );

    // Uncorrelated activations
    let act1: Vec<f32> = (0..50).map(|i| (i as f32 * 0.7).sin()).collect();
    let act2: Vec<f32> = (0..50).map(|i| (i as f32 * 1.3).cos()).collect();

    let records = vec![make_records("in-1", &act1), make_records("in-2", &act2)];

    let config = WeightCoherenceConfig::default();
    let candidates = detect_symmetric_cancellation(&creature, &records, &config, None);

    assert!(
        candidates.is_empty(),
        "Should not flag uncorrelated inputs even with opposite weights"
    );
}
