//! Tests for Issue #437: Weight coherence validation.
//!
//! Part of the "Brilliant but Brittle" initiative (Issue #432). This module validates
//! that proposed weight configurations are coherent with network structure and won't
//! create brittleness.
//!
//! ## TDD Plan
//! 1. Test detection of large incoming weights with tiny outgoing weights
//! 2. Test detection of weights that create near-constant output paths
//! 3. Test detection of symmetric weights that cancel meaningful signal
//! 4. Test detection of bypassed neuron computation (pass-through paths)
//! 5. Test candidate generation for setWeight adjustments
//! 6. Test configurable threshold via WeightCoherenceConfig
//! 7. Test edge cases: single synapse, no hidden neurons
//! 8. Test coherence metric normalisation for comparison

use crate::common::{make_creature, neuron, synapse};
use neat_ai_discovery::analysis::detection::weight_coherence::{
    IncoherentWeightRatioCandidate, NearConstantPathCandidate, SymmetricCancellationCandidate,
    WeightCoherenceConfig, detect_incoherent_weight_ratios, detect_near_constant_paths,
    detect_symmetric_cancellation, incoherent_ratios_to_coordinated_candidates,
    near_constant_paths_to_coordinated_candidates,
    symmetric_cancellation_to_coordinated_candidates,
};
use neat_ai_discovery::types::DiscoverRecord;

// =============================================================================
// Test 1: Detect large incoming weights with tiny outgoing weights
// =============================================================================

/// A neuron receiving a very large incoming weight but outputting via a tiny weight
/// creates an inefficient amplification/attenuation pattern that is brittle.
#[test]
fn test_detects_large_incoming_tiny_outgoing() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("hidden-imbalanced", "hidden", "TANH"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "hidden-imbalanced", 50.0), // Very large incoming
            synapse("hidden-imbalanced", "output-1", 0.001), // Tiny outgoing
        ],
    );

    // Create records showing the hidden neuron is active
    let hidden_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = ((i as f32 * 0.1).sin()).tanh();
            DiscoverRecord {
                obs_index: i,
                neuron_uuid: "hidden-imbalanced".to_string(),
                value: Some(activation),
                activation,
                errors: vec![0.1],
            }
        })
        .collect();

    let config = WeightCoherenceConfig::default();
    let candidates = detect_incoherent_weight_ratios(
        &creature,
        &[("hidden-imbalanced".to_string(), hidden_records)],
        &config,
        None,
    );

    assert!(
        !candidates.is_empty(),
        "Should detect incoherent weight ratio"
    );
    assert_eq!(candidates[0].neuron_uuid, "hidden-imbalanced");
    assert!(
        candidates[0].incoming_outgoing_ratio > 100.0,
        "Ratio should be very high: {}",
        candidates[0].incoming_outgoing_ratio
    );
}

// =============================================================================
// Test 2: Balanced weight ratio is NOT flagged
// =============================================================================

/// A neuron with balanced incoming/outgoing weights should not be flagged.
#[test]
fn test_balanced_weight_ratio_not_flagged() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("hidden-balanced", "hidden", "TANH"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "hidden-balanced", 1.0),
            synapse("hidden-balanced", "output-1", 0.5),
        ],
    );

    let hidden_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = ((i as f32 * 0.1).sin()).tanh();
            DiscoverRecord {
                obs_index: i,
                neuron_uuid: "hidden-balanced".to_string(),
                value: Some(activation),
                activation,
                errors: vec![0.1],
            }
        })
        .collect();

    let config = WeightCoherenceConfig::default();
    let candidates = detect_incoherent_weight_ratios(
        &creature,
        &[("hidden-balanced".to_string(), hidden_records)],
        &config,
        None,
    );

    assert!(
        candidates.is_empty(),
        "Balanced weight ratio should not be flagged"
    );
}

// =============================================================================
// Test 3: Detect near-constant output paths
// =============================================================================

/// When weights cause a neuron to produce near-constant output regardless of inputs,
/// this creates a brittle "constant offset" that doesn't respond to data patterns.
#[test]
fn test_detects_near_constant_output_path() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("hidden-constant", "hidden", "TANH"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "hidden-constant", 100.0), // Huge weight saturates TANH
            synapse("hidden-constant", "output-1", 0.1),
        ],
    );

    // Create records where the neuron is always saturated at +1 (extremely low variance)
    // This simulates a case where the huge incoming weight forces saturation
    let hidden_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            // Tiny variation around 0.999 (saturated TANH output)
            let activation = 0.999 + 0.0001 * ((i as f32 * 0.1).sin());
            DiscoverRecord {
                obs_index: i,
                neuron_uuid: "hidden-constant".to_string(),
                value: Some(10.0), // High pre-activation value
                activation,
                errors: vec![0.05],
            }
        })
        .collect();

    let config = WeightCoherenceConfig::default();
    let candidates = detect_near_constant_paths(
        &creature,
        &[("hidden-constant".to_string(), hidden_records)],
        &config,
        None,
    );

    assert!(
        !candidates.is_empty(),
        "Should detect near-constant output path"
    );
    assert_eq!(candidates[0].neuron_uuid, "hidden-constant");
}

// =============================================================================
// Test 4: Variable output path is NOT flagged as constant
// =============================================================================

/// A neuron with variable output across samples should not be flagged as constant.
#[test]
fn test_variable_output_not_flagged_as_constant() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("hidden-variable", "hidden", "TANH"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "hidden-variable", 1.0),
            synapse("hidden-variable", "output-1", 0.5),
        ],
    );

    // TANH(1.0 * x) varies nicely for moderate input range
    let hidden_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let input_val = (i as f32 - 50.0) / 25.0; // -2 to 2
            let activation = input_val.tanh();
            DiscoverRecord {
                obs_index: i,
                neuron_uuid: "hidden-variable".to_string(),
                value: Some(input_val),
                activation,
                errors: vec![0.1],
            }
        })
        .collect();

    let config = WeightCoherenceConfig::default();
    let candidates = detect_near_constant_paths(
        &creature,
        &[("hidden-variable".to_string(), hidden_records)],
        &config,
        None,
    );

    assert!(
        candidates.is_empty(),
        "Variable output should not be flagged as constant"
    );
}

// =============================================================================
// Test 5: Detect symmetric weights that cancel meaningful signal
// =============================================================================

/// When two synapses from inputs to a target have opposite weights of similar magnitude,
/// they can cancel out the meaningful signal contribution.
#[test]
fn test_detects_symmetric_weight_cancellation() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("input-2", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "output-1", 5.0),  // Positive
            synapse("input-2", "output-1", -5.0), // Negative - cancels if inputs correlated
        ],
    );

    // Inputs are highly correlated (often move together)
    let input1_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = 0.5 + 0.4 * ((i as f32 * 0.1).sin());
            DiscoverRecord {
                obs_index: i,
                neuron_uuid: "input-1".to_string(),
                value: Some(activation),
                activation,
                errors: vec![0.01],
            }
        })
        .collect();

    let input2_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            // Same pattern as input-1 (correlated)
            let activation = 0.5 + 0.4 * ((i as f32 * 0.1).sin());
            DiscoverRecord {
                obs_index: i,
                neuron_uuid: "input-2".to_string(),
                value: Some(activation),
                activation,
                errors: vec![0.01],
            }
        })
        .collect();

    let output_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            // Output error is high because signals cancel
            DiscoverRecord {
                obs_index: i,
                neuron_uuid: "output-1".to_string(),
                value: Some(0.0),
                activation: 0.0,
                errors: vec![0.3],
            }
        })
        .collect();

    let config = WeightCoherenceConfig::default();
    let candidates = detect_symmetric_cancellation(
        &creature,
        &[
            ("input-1".to_string(), input1_records),
            ("input-2".to_string(), input2_records),
            ("output-1".to_string(), output_records),
        ],
        &config,
        None,
    );

    assert!(
        !candidates.is_empty(),
        "Should detect symmetric weight cancellation"
    );
    assert_eq!(candidates[0].target_neuron_uuid, "output-1");
}

// =============================================================================
// Test 6: Uncorrelated inputs with opposite weights are NOT flagged
// =============================================================================

/// Opposite weights from uncorrelated inputs don't cancel and shouldn't be flagged.
#[test]
fn test_uncorrelated_opposite_weights_not_flagged() {
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

    // Inputs are uncorrelated (independent patterns)
    let input1_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = 0.5 + 0.4 * ((i as f32 * 0.1).sin());
            DiscoverRecord {
                obs_index: i,
                neuron_uuid: "input-1".to_string(),
                value: Some(activation),
                activation,
                errors: vec![0.01],
            }
        })
        .collect();

    let input2_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            // Different pattern (uncorrelated)
            let activation = 0.3 + 0.3 * ((i as f32 * 0.23).cos());
            DiscoverRecord {
                obs_index: i,
                neuron_uuid: "input-2".to_string(),
                value: Some(activation),
                activation,
                errors: vec![0.01],
            }
        })
        .collect();

    let output_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| DiscoverRecord {
            obs_index: i,
            neuron_uuid: "output-1".to_string(),
            value: Some(0.5),
            activation: 0.5,
            errors: vec![0.1],
        })
        .collect();

    let config = WeightCoherenceConfig::default();
    let candidates = detect_symmetric_cancellation(
        &creature,
        &[
            ("input-1".to_string(), input1_records),
            ("input-2".to_string(), input2_records),
            ("output-1".to_string(), output_records),
        ],
        &config,
        None,
    );

    assert!(
        candidates.is_empty(),
        "Uncorrelated inputs with opposite weights should not be flagged"
    );
}

// =============================================================================
// Test 7: Minimum samples required
// =============================================================================

/// Detection requires minimum samples for statistical reliability.
#[test]
fn test_weight_coherence_minimum_samples_required() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("hidden-1", "hidden", "TANH"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "hidden-1", 50.0),
            synapse("hidden-1", "output-1", 0.001),
        ],
    );

    // Only 5 samples - insufficient
    let hidden_records: Vec<DiscoverRecord> = (0..5)
        .map(|i| DiscoverRecord {
            obs_index: i,
            neuron_uuid: "hidden-1".to_string(),
            value: Some(0.9),
            activation: 0.9,
            errors: vec![0.1],
        })
        .collect();

    let config = WeightCoherenceConfig::default();
    let candidates = detect_incoherent_weight_ratios(
        &creature,
        &[("hidden-1".to_string(), hidden_records)],
        &config,
        None,
    );

    assert!(
        candidates.is_empty(),
        "Should require minimum samples for detection"
    );
}

// =============================================================================
// Test 8: Incoherent ratio produces setWeight candidate
// =============================================================================

/// Incoherent weight ratios should produce setWeight candidates to balance weights.
#[test]
fn test_incoherent_ratio_produces_setweight_candidate() {
    let candidate = IncoherentWeightRatioCandidate {
        neuron_uuid: "hidden-imbalanced".to_string(),
        incoming_weight_sum: 50.0,
        outgoing_weight_sum: 0.001,
        incoming_outgoing_ratio: 50000.0,
        recommended_outgoing_weight: 0.05,
        sample_count: 100,
        estimated_improvement: 0.01,
    };

    let coordinated = incoherent_ratios_to_coordinated_candidates(&[candidate]);

    assert_eq!(coordinated.len(), 1);
    let ops_json = serde_json::to_string(&coordinated[0].operations).unwrap();
    assert!(
        ops_json.contains("setWeight"),
        "Should produce setWeight candidate"
    );
    assert!(
        coordinated[0].expected_creature_score_gain > 0.0,
        "Expected improvement should be positive"
    );
}

// =============================================================================
// Test 9: Near-constant path produces setBias candidate
// =============================================================================

/// Near-constant paths may recommend setBias to shift the operating point.
#[test]
fn test_near_constant_path_produces_setbias_candidate() {
    let candidate = NearConstantPathCandidate {
        neuron_uuid: "hidden-constant".to_string(),
        activation_variance: 0.001,
        mean_activation: 0.99,
        causing_weight: 100.0,
        recommended_action: "setBias".to_string(),
        sample_count: 100,
        estimated_improvement: 0.008,
    };

    let coordinated = near_constant_paths_to_coordinated_candidates(&[candidate]);

    assert_eq!(coordinated.len(), 1);
    let ops_json = serde_json::to_string(&coordinated[0].operations).unwrap();
    assert!(
        ops_json.contains("setBias"),
        "Should produce setBias candidate"
    );
}

// =============================================================================
// Test 10: Symmetric cancellation produces coordinated candidate
// =============================================================================

/// Symmetric cancellation should produce a coordinated candidate to fix the issue.
#[test]
fn test_symmetric_cancellation_produces_coordinated_candidate() {
    let candidate = SymmetricCancellationCandidate {
        source1_neuron_uuid: "input-1".to_string(),
        source2_neuron_uuid: "input-2".to_string(),
        target_neuron_uuid: "output-1".to_string(),
        weight1: 5.0,
        weight2: -5.0,
        correlation: 0.95,
        cancellation_ratio: 0.98,
        recommended_action: "setWeight".to_string(),
        sample_count: 100,
        estimated_improvement: 0.015,
    };

    let coordinated = symmetric_cancellation_to_coordinated_candidates(&[candidate]);

    assert_eq!(coordinated.len(), 1);
    assert!(
        coordinated[0].expected_creature_score_gain > 0.0,
        "Expected improvement should be positive"
    );
}

// =============================================================================
// Test 11: Configurable threshold
// =============================================================================

/// The ratio threshold can be configured via WeightCoherenceConfig.
#[test]
fn test_configurable_ratio_threshold() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("hidden-1", "hidden", "TANH"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "hidden-1", 10.0),
            synapse("hidden-1", "output-1", 0.1),
        ],
    );

    let hidden_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = ((i as f32 * 0.1).sin()).tanh();
            DiscoverRecord {
                obs_index: i,
                neuron_uuid: "hidden-1".to_string(),
                value: Some(activation),
                activation,
                errors: vec![0.1],
            }
        })
        .collect();

    // Low threshold - more sensitive detection (ratio = 100)
    let low_config = WeightCoherenceConfig {
        max_weight_ratio: 50.0,
        ..Default::default()
    };

    // High threshold - less sensitive detection
    let high_config = WeightCoherenceConfig {
        max_weight_ratio: 500.0,
        ..Default::default()
    };

    let candidates_low = detect_incoherent_weight_ratios(
        &creature,
        &[("hidden-1".to_string(), hidden_records.clone())],
        &low_config,
        None,
    );

    let candidates_high = detect_incoherent_weight_ratios(
        &creature,
        &[("hidden-1".to_string(), hidden_records)],
        &high_config,
        None,
    );

    // Lower threshold should detect more (or equal) candidates
    assert!(
        candidates_low.len() >= candidates_high.len(),
        "Lower threshold should detect at least as many candidates"
    );
}

// =============================================================================
// Test 12: Candidate includes diagnostic comment
// =============================================================================

/// Candidates should include diagnostic comments explaining the detection.
#[test]
fn test_weight_coherence_candidate_includes_diagnostic_comment() {
    let candidate = IncoherentWeightRatioCandidate {
        neuron_uuid: "hidden-imbalanced".to_string(),
        incoming_weight_sum: 50.0,
        outgoing_weight_sum: 0.001,
        incoming_outgoing_ratio: 50000.0,
        recommended_outgoing_weight: 0.05,
        sample_count: 100,
        estimated_improvement: 0.012,
    };

    let coordinated = incoherent_ratios_to_coordinated_candidates(&[candidate]);

    assert!(
        coordinated[0].comment.is_some(),
        "Should have diagnostic comment"
    );
    let comment = coordinated[0].comment.as_ref().unwrap();
    assert!(
        comment.contains("ratio") || comment.contains("coherence") || comment.contains("weight"),
        "Comment should explain weight coherence issue"
    );
}

// =============================================================================
// Test 13: Input and output neurons excluded from ratio check
// =============================================================================

/// Only hidden neurons should be checked for incoherent weight ratios.
#[test]
fn test_input_output_excluded_from_ratio_check() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "output-1", 50.0), // Large weight on direct connection
        ],
    );

    let input_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| DiscoverRecord {
            obs_index: i,
            neuron_uuid: "input-1".to_string(),
            value: Some(0.5),
            activation: 0.5,
            errors: vec![0.01],
        })
        .collect();

    let config = WeightCoherenceConfig::default();
    let candidates = detect_incoherent_weight_ratios(
        &creature,
        &[("input-1".to_string(), input_records)],
        &config,
        None,
    );

    assert!(
        candidates.is_empty(),
        "Input neurons should not be flagged for ratio issues"
    );
}

// =============================================================================
// Test 14: Empty records handled gracefully
// =============================================================================

/// Empty records should be handled without panicking.
#[test]
fn test_weight_coherence_empty_records_handled() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("hidden-1", "hidden", "TANH"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "hidden-1", 50.0),
            synapse("hidden-1", "output-1", 0.001),
        ],
    );

    let config = WeightCoherenceConfig::default();
    let candidates = detect_incoherent_weight_ratios(&creature, &[], &config, None);

    assert!(
        candidates.is_empty(),
        "Empty records should produce no candidates"
    );
}

// =============================================================================
// Test 15: Multiple incoherent neurons sorted by improvement
// =============================================================================

/// Multiple incoherent neurons should be sorted by estimated improvement.
#[test]
fn test_multiple_incoherent_sorted_by_improvement() {
    let candidate1 = IncoherentWeightRatioCandidate {
        neuron_uuid: "hidden-1".to_string(),
        incoming_weight_sum: 10.0,
        outgoing_weight_sum: 0.01,
        incoming_outgoing_ratio: 1000.0,
        recommended_outgoing_weight: 0.05,
        sample_count: 100,
        estimated_improvement: 0.005,
    };

    let candidate2 = IncoherentWeightRatioCandidate {
        neuron_uuid: "hidden-2".to_string(),
        incoming_weight_sum: 50.0,
        outgoing_weight_sum: 0.001,
        incoming_outgoing_ratio: 50000.0,
        recommended_outgoing_weight: 0.05,
        sample_count: 100,
        estimated_improvement: 0.015,
    };

    let coordinated = incoherent_ratios_to_coordinated_candidates(&[candidate1, candidate2]);

    assert_eq!(coordinated.len(), 2);
    assert!(
        coordinated[0].expected_creature_score_gain >= coordinated[1].expected_creature_score_gain,
        "Candidates should be sorted by estimated improvement (best first)"
    );
}

// =============================================================================
// Test 16: Tiny outgoing with large incoming suggests amplification brittleness
// =============================================================================

/// When a neuron has large incoming but tiny outgoing, the neuron is doing work
/// that barely affects the output, which is inefficient and brittle.
#[test]
fn test_detects_amplification_attenuation_pattern() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("input-2", "input", "IDENTITY"),
            neuron("hidden-amplify", "hidden", "RELU"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "hidden-amplify", 20.0),
            synapse("input-2", "hidden-amplify", 30.0), // Total incoming ~50
            synapse("hidden-amplify", "output-1", 0.002), // Tiny outgoing
        ],
    );

    let hidden_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = (i as f32 * 0.1).max(0.0); // ReLU output
            DiscoverRecord {
                obs_index: i,
                neuron_uuid: "hidden-amplify".to_string(),
                value: Some(activation),
                activation,
                errors: vec![0.1],
            }
        })
        .collect();

    let config = WeightCoherenceConfig::default();
    let candidates = detect_incoherent_weight_ratios(
        &creature,
        &[("hidden-amplify".to_string(), hidden_records)],
        &config,
        None,
    );

    assert!(
        !candidates.is_empty(),
        "Should detect amplification/attenuation pattern"
    );
    assert!(
        candidates[0].incoming_weight_sum > 40.0,
        "Should sum incoming weights: {}",
        candidates[0].incoming_weight_sum
    );
}
