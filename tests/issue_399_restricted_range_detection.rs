//! Tests for Issue #399: Bounded range neuron detection — identify neurons with
//! restricted activation ranges.
//!
//! Hidden neurons may operate in a restricted sub-range of their activation
//! function's output domain. For example, a TANH neuron consistently outputting
//! values in [0.1, 0.3] is only using 10% of its [-1, 1] range. This module
//! detects such neurons and proposes corrective candidates.
//!
//! ## TDD Plan
//! 1. Detect TANH neuron using only 10% of its range
//! 2. Verify neurons using full range are NOT flagged
//! 3. Verify unbounded activations (IDENTITY, RELU) are excluded
//! 4. Verify output neurons are excluded
//! 5. Verify insufficient samples are skipped
//! 6. Verify dead neurons (near-zero activation) are excluded
//! 7. Test candidate generation: changeSquash, setBias, setWeight
//! 8. Test configurable utilisation threshold
//! 9. Test multiple neurons — only restricted ones detected

mod common;

use common::{hidden, hidden_with_bias, make_creature, neuron, output, synapse};
use neat_ai_discovery::analysis::restricted_range::{
    detect_restricted_range_neurons, restricted_range_to_coordinated_candidates,
    RestrictedRangeConfig,
};
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::CoordinatedStructuralOpJson;

/// Helper to create records for a neuron with activations in a specific range.
fn make_records_in_range(
    uuid: &str,
    count: usize,
    min_val: f32,
    max_val: f32,
) -> Vec<DiscoverRecord> {
    (0..count)
        .map(|i| {
            let t = i as f32 / (count - 1).max(1) as f32;
            let activation = min_val + t * (max_val - min_val);
            common::record(uuid, i as u32, activation, Some(activation))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Test 1: TANH neuron using only 10% of its [-1, 1] range is detected.
// ---------------------------------------------------------------------------
#[test]
fn test_detects_tanh_neuron_with_restricted_range() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            hidden("h1", "TANH"),
            output("output-1", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "h1", 0.5),
            synapse("h1", "output-1", 0.3),
        ],
    );

    // TANH range is [-1, 1] (theoretical range = 2.0)
    // Activations in [0.1, 0.3] → observed range = 0.2, utilisation = 0.2/2.0 = 10%
    let records = make_records_in_range("h1", 100, 0.1, 0.3);

    let config = RestrictedRangeConfig::default();
    let detected =
        detect_restricted_range_neurons(&creature, &[("h1".to_string(), records)], &config);

    assert_eq!(
        detected.len(),
        1,
        "Should detect one restricted-range neuron"
    );
    let n = &detected[0];
    assert_eq!(n.neuron_uuid, "h1");
    assert!(
        n.range_utilisation < 0.20,
        "Range utilisation should be < 20%, got {:.2}%",
        n.range_utilisation * 100.0
    );
    assert_eq!(n.squash, "TANH");
}

// ---------------------------------------------------------------------------
// Test 2: TANH neuron using 80% of its range is NOT flagged.
// ---------------------------------------------------------------------------
#[test]
fn test_does_not_flag_neuron_using_full_range() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            hidden("h1", "TANH"),
            output("output-1", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "h1", 0.5),
            synapse("h1", "output-1", 0.3),
        ],
    );

    // Activations in [-0.8, 0.8] → observed range = 1.6, utilisation = 1.6/2.0 = 80%
    let records = make_records_in_range("h1", 100, -0.8, 0.8);

    let config = RestrictedRangeConfig::default();
    let detected =
        detect_restricted_range_neurons(&creature, &[("h1".to_string(), records)], &config);

    assert!(
        detected.is_empty(),
        "Neuron using 80% of range should not be flagged"
    );
}

// ---------------------------------------------------------------------------
// Test 3: Unbounded activations (IDENTITY, RELU) are excluded.
// ---------------------------------------------------------------------------
#[test]
fn test_unbounded_activations_excluded() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            hidden("h-identity", "IDENTITY"),
            hidden("h-relu", "RELU"),
            output("output-1", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "h-identity", 0.5),
            synapse("input-1", "h-relu", 0.5),
            synapse("h-identity", "output-1", 0.3),
            synapse("h-relu", "output-1", 0.3),
        ],
    );

    // Even with narrow ranges, unbounded activations should not be flagged
    let identity_records = make_records_in_range("h-identity", 100, 0.1, 0.2);
    let relu_records = make_records_in_range("h-relu", 100, 0.1, 0.2);

    let config = RestrictedRangeConfig::default();
    let detected = detect_restricted_range_neurons(
        &creature,
        &[
            ("h-identity".to_string(), identity_records),
            ("h-relu".to_string(), relu_records),
        ],
        &config,
    );

    assert!(
        detected.is_empty(),
        "Unbounded activations should not be flagged"
    );
}

// ---------------------------------------------------------------------------
// Test 4: Output neurons are excluded.
// ---------------------------------------------------------------------------
#[test]
fn test_output_neurons_excluded() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            output("output-1", "TANH"),
        ],
        vec![synapse("input-1", "output-1", 0.5)],
    );

    // Output neuron with restricted range — should NOT be flagged
    let records = make_records_in_range("output-1", 100, 0.1, 0.2);

    let config = RestrictedRangeConfig::default();
    let detected =
        detect_restricted_range_neurons(&creature, &[("output-1".to_string(), records)], &config);

    assert!(
        detected.is_empty(),
        "Output neurons should be excluded from detection"
    );
}

// ---------------------------------------------------------------------------
// Test 5: Insufficient samples are skipped.
// ---------------------------------------------------------------------------
#[test]
fn test_insufficient_samples_skipped() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            hidden("h1", "TANH"),
            output("output-1", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "h1", 0.5),
            synapse("h1", "output-1", 0.3),
        ],
    );

    // Only 5 samples — below minimum threshold
    let records = make_records_in_range("h1", 5, 0.1, 0.2);

    let config = RestrictedRangeConfig::default();
    let detected =
        detect_restricted_range_neurons(&creature, &[("h1".to_string(), records)], &config);

    assert!(
        detected.is_empty(),
        "Should not detect with insufficient samples"
    );
}

// ---------------------------------------------------------------------------
// Test 6: Dead neurons (near-zero activation) are excluded.
// ---------------------------------------------------------------------------
#[test]
fn test_dead_neurons_excluded() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            hidden("h1", "TANH"),
            output("output-1", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "h1", 0.5),
            synapse("h1", "output-1", 0.3),
        ],
    );

    // Near-zero activations — this is a dead neuron, not a restricted range issue
    let records = make_records_in_range("h1", 100, -0.001, 0.001);

    let config = RestrictedRangeConfig::default();
    let detected =
        detect_restricted_range_neurons(&creature, &[("h1".to_string(), records)], &config);

    assert!(
        detected.is_empty(),
        "Dead neurons should be excluded (near-zero activation)"
    );
}

// ---------------------------------------------------------------------------
// Test 7: Candidate generation includes changeSquash, setBias, setWeight.
// ---------------------------------------------------------------------------
#[test]
fn test_candidate_generation_includes_expected_operations() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            hidden_with_bias("h1", "TANH", 2.0),
            output("output-1", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "h1", 0.1),
            synapse("h1", "output-1", 0.3),
        ],
    );

    // TANH neuron stuck in narrow band [0.3, 0.5] due to bias — clearly within bounds
    let records = make_records_in_range("h1", 100, 0.3, 0.5);

    let config = RestrictedRangeConfig::default();
    let detected =
        detect_restricted_range_neurons(&creature, &[("h1".to_string(), records)], &config);

    assert!(
        !detected.is_empty(),
        "Should detect restricted-range neuron"
    );

    let coordinated = restricted_range_to_coordinated_candidates(&detected, &creature);

    assert!(
        !coordinated.is_empty(),
        "Should produce coordinated candidates"
    );

    // Collect all operation types across all candidates
    let mut has_change_squash = false;
    let mut has_set_bias = false;
    let mut has_set_weight = false;

    for candidate in &coordinated {
        assert!(
            candidate.expected_creature_score_gain > 0.0,
            "Expected improvement should be positive"
        );
        assert!(
            candidate.comment.is_some(),
            "Should include a descriptive comment"
        );

        for op in &candidate.operations {
            match op {
                CoordinatedStructuralOpJson::ChangeSquash { .. } => has_change_squash = true,
                CoordinatedStructuralOpJson::SetBias { .. } => has_set_bias = true,
                CoordinatedStructuralOpJson::SetWeight { .. } => has_set_weight = true,
                _ => {}
            }
        }
    }

    assert!(
        has_change_squash || has_set_bias || has_set_weight,
        "Should produce at least one of changeSquash, setBias, or setWeight"
    );
}

// ---------------------------------------------------------------------------
// Test 8: Configurable utilisation threshold.
// ---------------------------------------------------------------------------
#[test]
fn test_configurable_utilisation_threshold() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            hidden("h1", "TANH"),
            output("output-1", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "h1", 0.5),
            synapse("h1", "output-1", 0.3),
        ],
    );

    // TANH with range [-0.4, 0.4] → utilisation = 0.8/2.0 = 40%
    let records = make_records_in_range("h1", 100, -0.4, 0.4);

    // Default threshold (20%) — should NOT flag at 40% utilisation
    let config_default = RestrictedRangeConfig::default();
    let detected = detect_restricted_range_neurons(
        &creature,
        &[("h1".to_string(), records.clone())],
        &config_default,
    );
    assert!(
        detected.is_empty(),
        "40% utilisation should not be flagged with default 20% threshold"
    );

    // Higher threshold (50%) — should flag at 40% utilisation
    let config_strict = RestrictedRangeConfig {
        utilisation_threshold: 0.50,
        ..RestrictedRangeConfig::default()
    };
    let detected =
        detect_restricted_range_neurons(&creature, &[("h1".to_string(), records)], &config_strict);
    assert_eq!(
        detected.len(),
        1,
        "40% utilisation should be flagged with 50% threshold"
    );
}

// ---------------------------------------------------------------------------
// Test 9: Multiple neurons — only restricted ones detected.
// ---------------------------------------------------------------------------
#[test]
fn test_multiple_neurons_only_restricted_detected() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            hidden("h-good", "TANH"),
            hidden("h-bad", "LOGISTIC"),
            output("output-1", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "h-good", 0.5),
            synapse("input-1", "h-bad", 0.5),
            synapse("h-good", "output-1", 0.3),
            synapse("h-bad", "output-1", 0.3),
        ],
    );

    // h-good: TANH using 70% of range → not flagged
    let good_records = make_records_in_range("h-good", 100, -0.7, 0.7);

    // h-bad: LOGISTIC using 10% of [0, 1] range → flagged
    let bad_records = make_records_in_range("h-bad", 100, 0.45, 0.55);

    let config = RestrictedRangeConfig::default();
    let detected = detect_restricted_range_neurons(
        &creature,
        &[
            ("h-good".to_string(), good_records),
            ("h-bad".to_string(), bad_records),
        ],
        &config,
    );

    assert_eq!(
        detected.len(),
        1,
        "Only the restricted-range neuron should be detected"
    );
    assert_eq!(detected[0].neuron_uuid, "h-bad");
}

// ---------------------------------------------------------------------------
// Test 10: LOGISTIC neuron with restricted range is detected.
// ---------------------------------------------------------------------------
#[test]
fn test_detects_logistic_neuron_with_restricted_range() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            hidden("h1", "LOGISTIC"),
            output("output-1", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "h1", 0.5),
            synapse("h1", "output-1", 0.3),
        ],
    );

    // LOGISTIC range is [0, 1] (theoretical range = 1.0)
    // Activations in [0.48, 0.52] → utilisation = 0.04/1.0 = 4%
    let records = make_records_in_range("h1", 100, 0.48, 0.52);

    let config = RestrictedRangeConfig::default();
    let detected =
        detect_restricted_range_neurons(&creature, &[("h1".to_string(), records)], &config);

    assert_eq!(
        detected.len(),
        1,
        "Should detect restricted LOGISTIC neuron"
    );
    let n = &detected[0];
    assert!(
        n.range_utilisation < 0.10,
        "Range utilisation should be < 10%, got {:.2}%",
        n.range_utilisation * 100.0
    );
}

// ---------------------------------------------------------------------------
// Test 11: Saturated neurons are NOT flagged (saturation is a different issue).
// ---------------------------------------------------------------------------
#[test]
fn test_saturated_neurons_not_flagged_as_restricted() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            hidden("h1", "TANH"),
            output("output-1", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "h1", 0.5),
            synapse("h1", "output-1", 0.3),
        ],
    );

    // TANH neuron stuck near +1 (saturated) — should NOT be flagged here
    // Range [0.95, 1.0] → small range, but it's at the boundary (saturation, not restricted range)
    let records = make_records_in_range("h1", 100, 0.95, 1.0);

    let config = RestrictedRangeConfig::default();
    let detected =
        detect_restricted_range_neurons(&creature, &[("h1".to_string(), records)], &config);

    // This is saturation, not a restricted range issue — should be excluded
    assert!(
        detected.is_empty(),
        "Saturated neurons (at activation bounds) should not be flagged as restricted range"
    );
}
