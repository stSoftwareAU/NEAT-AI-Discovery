//! Tests for Issue #435: Input sensitivity analysis module.
//!
//! Part of the "Brilliant but Brittle" initiative (Issue #432). This module analyses
//! how sensitive predictions are to small changes in input observations, identifying
//! inputs with excessive leverage that cause brittle predictions.
//!
//! ## TDD Plan
//! 1. Test detection of inputs with disproportionate prediction impact
//! 2. Test detection of single inputs that dominate predictions
//! 3. Test detection of threshold effects (small changes flip predictions)
//! 4. Test candidate generation (setWeight, addNeuron, setBias)
//! 5. Test normalised sensitivity metrics for comparison
//! 6. Test configurable sensitivity thresholds via environment variable
//! 7. Test edge cases: zero variance, constant inputs

mod common;

use common::{make_creature, neuron, synapse};
use neat_ai_discovery::analysis::detection::input_sensitivity::{
    DominantInputCandidate, InputSensitivityConfig, ThresholdEffectCandidate,
    detect_dominant_inputs, detect_threshold_effects, dominant_inputs_to_coordinated_candidates,
    threshold_effects_to_coordinated_candidates,
};
use neat_ai_discovery::types::DiscoverRecord;

// =============================================================================
// Test 1: Detect input with disproportionate impact on predictions
// =============================================================================

/// An input neuron that contributes disproportionately to output variance
/// should be detected as having excessive leverage.
#[test]
fn test_detects_dominant_input_with_high_leverage() {
    let creature = make_creature(
        vec![
            neuron("input-dominant", "input", "IDENTITY"),
            neuron("input-normal", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-dominant", "output-1", 10.0), // High weight = high leverage
            synapse("input-normal", "output-1", 0.1),    // Normal weight
        ],
    );

    // Create records showing dominant input has high variance
    let dominant_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = 0.5 + 0.4 * ((i as f32 * 0.1).sin()); // High variance input
            DiscoverRecord {
                obs_index: i,
                neuron_uuid: "input-dominant".to_string(),
                value: Some(activation),
                activation,
                errors: vec![0.1], // Input neurons typically have low error
            }
        })
        .collect();

    let normal_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = 0.5 + 0.1 * ((i as f32 * 0.1).sin()); // Lower variance
            DiscoverRecord {
                obs_index: i,
                neuron_uuid: "input-normal".to_string(),
                value: Some(activation),
                activation,
                errors: vec![0.1],
            }
        })
        .collect();

    let output_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            // Output error correlates strongly with dominant input
            let error = 0.3 * ((i as f32 * 0.1).sin());
            DiscoverRecord {
                obs_index: i,
                neuron_uuid: "output-1".to_string(),
                value: Some(0.5),
                activation: 0.5,
                errors: vec![error],
            }
        })
        .collect();

    let config = InputSensitivityConfig::default();
    let candidates = detect_dominant_inputs(
        &creature,
        &[
            ("input-dominant".to_string(), dominant_records),
            ("input-normal".to_string(), normal_records),
            ("output-1".to_string(), output_records),
        ],
        &config,
    );

    assert!(
        !candidates.is_empty(),
        "Should detect dominant input with high leverage"
    );
    assert_eq!(candidates[0].input_neuron_uuid, "input-dominant");
}

// =============================================================================
// Test 2: Normal inputs are not flagged as dominant
// =============================================================================

/// Inputs with balanced contribution to predictions should not be flagged.
#[test]
fn test_does_not_flag_balanced_inputs() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("input-2", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "output-1", 1.0),
            synapse("input-2", "output-1", 1.0),
        ],
    );

    // Both inputs have similar variance and contribution
    let records1: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = 0.5 + 0.2 * ((i as f32 * 0.1).sin());
            DiscoverRecord {
                obs_index: i,
                neuron_uuid: "input-1".to_string(),
                value: Some(activation),
                activation,
                errors: vec![0.1],
            }
        })
        .collect();

    let records2: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = 0.5 + 0.2 * ((i as f32 * 0.15).sin());
            DiscoverRecord {
                obs_index: i,
                neuron_uuid: "input-2".to_string(),
                value: Some(activation),
                activation,
                errors: vec![0.1],
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

    let config = InputSensitivityConfig::default();
    let candidates = detect_dominant_inputs(
        &creature,
        &[
            ("input-1".to_string(), records1),
            ("input-2".to_string(), records2),
            ("output-1".to_string(), output_records),
        ],
        &config,
    );

    assert!(
        candidates.is_empty(),
        "Balanced inputs should not be flagged"
    );
}

// =============================================================================
// Test 3: Detect threshold effects where small input changes flip predictions
// =============================================================================

/// When small changes in input cause large changes in output (threshold effect),
/// this indicates brittleness.
#[test]
fn test_detects_threshold_effect() {
    let creature = make_creature(
        vec![
            neuron("input-threshold", "input", "IDENTITY"),
            neuron("hidden-steep", "hidden", "TANH"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-threshold", "hidden-steep", 20.0), // Large weight creates steep response
            synapse("hidden-steep", "output-1", 1.0),
        ],
    );

    // Input hovers around the threshold region
    let input_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            // Small variation around 0 (TANH threshold region)
            let activation = 0.01 * ((i as f32 * 0.05).sin());
            DiscoverRecord {
                obs_index: i,
                neuron_uuid: "input-threshold".to_string(),
                value: Some(activation),
                activation,
                errors: vec![0.01],
            }
        })
        .collect();

    let hidden_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            // TANH(20 * small_value) switches between -1 and +1 abruptly
            let input_val = 0.01 * ((i as f32 * 0.05).sin());
            let activation = (20.0 * input_val).tanh();
            DiscoverRecord {
                obs_index: i,
                neuron_uuid: "hidden-steep".to_string(),
                value: Some(20.0 * input_val),
                activation,
                errors: vec![0.1],
            }
        })
        .collect();

    let output_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let input_val = 0.01 * ((i as f32 * 0.05).sin());
            let hidden_activation = (20.0 * input_val).tanh();
            DiscoverRecord {
                obs_index: i,
                neuron_uuid: "output-1".to_string(),
                value: Some(hidden_activation),
                activation: hidden_activation,
                errors: vec![0.2 * hidden_activation.abs()],
            }
        })
        .collect();

    let config = InputSensitivityConfig::default();
    let candidates = detect_threshold_effects(
        &creature,
        &[
            ("input-threshold".to_string(), input_records),
            ("hidden-steep".to_string(), hidden_records),
            ("output-1".to_string(), output_records),
        ],
        &config,
    );

    assert!(
        !candidates.is_empty(),
        "Should detect threshold effect from steep weight"
    );
}

// =============================================================================
// Test 4: Zero variance input is not flagged
// =============================================================================

/// Constant inputs (zero variance) should be handled gracefully and not flagged.
#[test]
fn test_zero_variance_input_handled() {
    let creature = make_creature(
        vec![
            neuron("input-constant", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![synapse("input-constant", "output-1", 5.0)],
    );

    // Constant input - no variance
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| DiscoverRecord {
            obs_index: i,
            neuron_uuid: "input-constant".to_string(),
            value: Some(1.0), // Always 1.0
            activation: 1.0,
            errors: vec![0.1],
        })
        .collect();

    let output_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| DiscoverRecord {
            obs_index: i,
            neuron_uuid: "output-1".to_string(),
            value: Some(5.0),
            activation: 5.0,
            errors: vec![0.1],
        })
        .collect();

    let config = InputSensitivityConfig::default();
    let candidates = detect_dominant_inputs(
        &creature,
        &[
            ("input-constant".to_string(), records),
            ("output-1".to_string(), output_records),
        ],
        &config,
    );

    // Should not panic and should not flag constant inputs
    assert!(
        candidates.is_empty(),
        "Constant inputs should not be flagged as dominant"
    );
}

// =============================================================================
// Test 5: Minimum sample requirement
// =============================================================================

/// Detection requires minimum samples for statistical reliability.
#[test]
fn test_minimum_samples_required() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![synapse("input-1", "output-1", 10.0)],
    );

    // Only 5 samples - insufficient
    let records: Vec<DiscoverRecord> = (0..5)
        .map(|i| DiscoverRecord {
            obs_index: i,
            neuron_uuid: "input-1".to_string(),
            value: Some(0.5),
            activation: 0.5,
            errors: vec![0.1],
        })
        .collect();

    let config = InputSensitivityConfig::default();
    let candidates =
        detect_dominant_inputs(&creature, &[("input-1".to_string(), records)], &config);

    assert!(
        candidates.is_empty(),
        "Should require minimum samples for detection"
    );
}

// =============================================================================
// Test 6: Dominant input produces setWeight candidate
// =============================================================================

/// Dominant inputs should produce setWeight candidates to reduce sensitivity.
#[test]
fn test_dominant_input_produces_setweight_candidate() {
    let candidate = DominantInputCandidate {
        input_neuron_uuid: "input-dominant".to_string(),
        target_neuron_uuid: "output-1".to_string(),
        sensitivity_score: 5.0,
        leverage_ratio: 3.5,
        current_weight: 10.0,
        recommended_weight: 3.0,
        sample_count: 100,
        estimated_improvement: 0.01,
    };

    let coordinated = dominant_inputs_to_coordinated_candidates(&[candidate]);

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
// Test 7: Threshold effect produces addNeuron candidate for dampening
// =============================================================================

/// Threshold effects may recommend addNeuron to provide smoothing/dampening.
#[test]
fn test_threshold_effect_produces_addneuron_candidate() {
    let candidate = ThresholdEffectCandidate {
        input_neuron_uuid: "input-threshold".to_string(),
        intermediate_neuron_uuid: Some("hidden-steep".to_string()),
        target_neuron_uuid: "output-1".to_string(),
        gradient_magnitude: 50.0,
        threshold_proximity: 0.02,
        recommended_action: "addNeuron".to_string(),
        sample_count: 100,
        estimated_improvement: 0.015,
    };

    let coordinated = threshold_effects_to_coordinated_candidates(&[candidate]);

    assert_eq!(coordinated.len(), 1);
    let ops_json = serde_json::to_string(&coordinated[0].operations).unwrap();
    assert!(
        ops_json.contains("addNeuron"),
        "Should produce addNeuron candidate for dampening"
    );
}

// =============================================================================
// Test 8: Threshold effect may produce setBias candidate
// =============================================================================

/// Threshold effects near saturation may recommend setBias to shift operating point.
#[test]
fn test_threshold_effect_produces_setbias_candidate() {
    let candidate = ThresholdEffectCandidate {
        input_neuron_uuid: "input-threshold".to_string(),
        intermediate_neuron_uuid: Some("hidden-threshold".to_string()),
        target_neuron_uuid: "output-1".to_string(),
        gradient_magnitude: 30.0,
        threshold_proximity: 0.01,
        recommended_action: "setBias".to_string(),
        sample_count: 100,
        estimated_improvement: 0.008,
    };

    let coordinated = threshold_effects_to_coordinated_candidates(&[candidate]);

    assert_eq!(coordinated.len(), 1);
    let ops_json = serde_json::to_string(&coordinated[0].operations).unwrap();
    assert!(
        ops_json.contains("setBias"),
        "Should produce setBias candidate"
    );
}

// =============================================================================
// Test 9: Normalised sensitivity metrics for comparison
// =============================================================================

/// Sensitivity scores should be normalised to allow comparison across different inputs.
#[test]
fn test_sensitivity_scores_are_normalised() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("input-2", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "output-1", 5.0),
            synapse("input-2", "output-1", 5.0),
        ],
    );

    // Input 1: Large absolute values but moderate variance
    let records1: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = 100.0 + 10.0 * ((i as f32 * 0.1).sin());
            DiscoverRecord {
                obs_index: i,
                neuron_uuid: "input-1".to_string(),
                value: Some(activation),
                activation,
                errors: vec![0.1],
            }
        })
        .collect();

    // Input 2: Small absolute values but high relative variance
    let records2: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = 0.5 + 0.4 * ((i as f32 * 0.1).sin());
            DiscoverRecord {
                obs_index: i,
                neuron_uuid: "input-2".to_string(),
                value: Some(activation),
                activation,
                errors: vec![0.1],
            }
        })
        .collect();

    let output_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| DiscoverRecord {
            obs_index: i,
            neuron_uuid: "output-1".to_string(),
            value: Some(0.5),
            activation: 0.5,
            errors: vec![0.5],
        })
        .collect();

    let config = InputSensitivityConfig::default();
    let candidates = detect_dominant_inputs(
        &creature,
        &[
            ("input-1".to_string(), records1),
            ("input-2".to_string(), records2),
            ("output-1".to_string(), output_records),
        ],
        &config,
    );

    // Sensitivity scores should be comparable (normalised)
    for candidate in &candidates {
        assert!(
            candidate.sensitivity_score >= 0.0,
            "Sensitivity score should be non-negative"
        );
    }
}

// =============================================================================
// Test 10: Configurable sensitivity threshold
// =============================================================================

/// The sensitivity threshold can be configured via InputSensitivityConfig.
#[test]
fn test_configurable_sensitivity_threshold() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![synapse("input-1", "output-1", 3.0)],
    );

    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = 0.5 + 0.3 * ((i as f32 * 0.1).sin());
            DiscoverRecord {
                obs_index: i,
                neuron_uuid: "input-1".to_string(),
                value: Some(activation),
                activation,
                errors: vec![0.1],
            }
        })
        .collect();

    let output_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| DiscoverRecord {
            obs_index: i,
            neuron_uuid: "output-1".to_string(),
            value: Some(0.5),
            activation: 0.5,
            errors: vec![0.3],
        })
        .collect();

    // Low threshold - more sensitive detection
    let low_config = InputSensitivityConfig {
        dominance_threshold: 1.0,
        ..Default::default()
    };

    // High threshold - less sensitive detection
    let high_config = InputSensitivityConfig {
        dominance_threshold: 10.0,
        ..Default::default()
    };

    let candidates_low = detect_dominant_inputs(
        &creature,
        &[
            ("input-1".to_string(), records.clone()),
            ("output-1".to_string(), output_records.clone()),
        ],
        &low_config,
    );

    let candidates_high = detect_dominant_inputs(
        &creature,
        &[
            ("input-1".to_string(), records),
            ("output-1".to_string(), output_records),
        ],
        &high_config,
    );

    // Lower threshold should detect more (or equal) candidates
    assert!(
        candidates_low.len() >= candidates_high.len(),
        "Lower threshold should detect at least as many candidates"
    );
}

// =============================================================================
// Test 11: Candidates include diagnostic comment
// =============================================================================

/// Candidates should include diagnostic comments explaining the detection.
#[test]
fn test_candidate_includes_diagnostic_comment() {
    let candidate = DominantInputCandidate {
        input_neuron_uuid: "input-dominant".to_string(),
        target_neuron_uuid: "output-1".to_string(),
        sensitivity_score: 4.2,
        leverage_ratio: 2.8,
        current_weight: 8.0,
        recommended_weight: 2.5,
        sample_count: 150,
        estimated_improvement: 0.012,
    };

    let coordinated = dominant_inputs_to_coordinated_candidates(&[candidate]);

    assert!(
        coordinated[0].comment.is_some(),
        "Should have diagnostic comment"
    );
    let comment = coordinated[0].comment.as_ref().unwrap();
    assert!(
        comment.contains("sensitivity")
            || comment.contains("leverage")
            || comment.contains("dominant"),
        "Comment should explain sensitivity issue"
    );
}

// =============================================================================
// Test 12: Multiple dominant inputs sorted by improvement
// =============================================================================

/// Multiple dominant inputs should be sorted by estimated improvement.
#[test]
fn test_multiple_dominant_inputs_sorted() {
    let candidate1 = DominantInputCandidate {
        input_neuron_uuid: "input-1".to_string(),
        target_neuron_uuid: "output-1".to_string(),
        sensitivity_score: 3.0,
        leverage_ratio: 2.0,
        current_weight: 5.0,
        recommended_weight: 2.0,
        sample_count: 100,
        estimated_improvement: 0.005,
    };

    let candidate2 = DominantInputCandidate {
        input_neuron_uuid: "input-2".to_string(),
        target_neuron_uuid: "output-1".to_string(),
        sensitivity_score: 6.0,
        leverage_ratio: 4.0,
        current_weight: 10.0,
        recommended_weight: 2.5,
        sample_count: 100,
        estimated_improvement: 0.015,
    };

    let coordinated = dominant_inputs_to_coordinated_candidates(&[candidate1, candidate2]);

    assert_eq!(coordinated.len(), 2);
    assert!(
        coordinated[0].expected_creature_score_gain >= coordinated[1].expected_creature_score_gain,
        "Candidates should be sorted by estimated improvement (best first)"
    );
}

// =============================================================================
// Test 13: Hidden neurons excluded from dominant input detection
// =============================================================================

/// Only input neurons should be considered for dominant input detection.
#[test]
fn test_hidden_neurons_excluded_from_dominant_detection() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("hidden-1", "hidden", "RELU"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "hidden-1", 1.0),
            synapse("hidden-1", "output-1", 10.0),
        ],
    );

    let input_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| DiscoverRecord {
            obs_index: i,
            neuron_uuid: "input-1".to_string(),
            value: Some(0.5),
            activation: 0.5,
            errors: vec![0.1],
        })
        .collect();

    let hidden_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = 0.5 + 0.4 * ((i as f32 * 0.1).sin());
            DiscoverRecord {
                obs_index: i,
                neuron_uuid: "hidden-1".to_string(),
                value: Some(activation),
                activation,
                errors: vec![0.3],
            }
        })
        .collect();

    let output_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| DiscoverRecord {
            obs_index: i,
            neuron_uuid: "output-1".to_string(),
            value: Some(0.5),
            activation: 0.5,
            errors: vec![0.2],
        })
        .collect();

    let config = InputSensitivityConfig::default();
    let candidates = detect_dominant_inputs(
        &creature,
        &[
            ("input-1".to_string(), input_records),
            ("hidden-1".to_string(), hidden_records),
            ("output-1".to_string(), output_records),
        ],
        &config,
    );

    // Only input neurons should be in candidates
    for candidate in &candidates {
        assert_ne!(
            candidate.input_neuron_uuid, "hidden-1",
            "Hidden neurons should not be flagged as dominant inputs"
        );
    }
}

// =============================================================================
// Test 14: Empty records handled gracefully
// =============================================================================

/// Empty records should be handled without panicking.
#[test]
fn test_empty_records_handled() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![synapse("input-1", "output-1", 5.0)],
    );

    let config = InputSensitivityConfig::default();
    let candidates = detect_dominant_inputs(&creature, &[], &config);

    assert!(
        candidates.is_empty(),
        "Empty records should produce no candidates"
    );
}
