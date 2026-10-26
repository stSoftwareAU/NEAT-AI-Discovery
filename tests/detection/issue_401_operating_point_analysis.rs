//! Tests for hidden neuron operating-point analysis (Issue #401).
//!
//! Detects neurons working outside their effective zone — e.g., a LOGISTIC
//! neuron with a very negative bias that always outputs ~0, or a TANH neuron
//! only using the linear region around zero.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use crate::common::{hidden, hidden_with_bias, make_creature, neuron, output, synapse};
use neat_ai_discovery::analysis::detection::operating_point::{
    OperatingPointConfig, detect_operating_point_issues, operating_point_to_coordinated_candidates,
};
use neat_ai_discovery::types::DiscoverRecord;

/// Helper to build a record with a specific pre-activation value.
fn record_with_value(uuid: &str, obs_index: u32, activation: f32, value: f32) -> DiscoverRecord {
    DiscoverRecord {
        obs_index,
        neuron_uuid: uuid.to_string(),
        value: Some(value),
        activation,
        errors: vec![0.01],
    }
}

// ============================================================================
// Detection tests
// ============================================================================

/// A LOGISTIC neuron with very negative pre-activation values (always outputting ~0)
/// should be flagged — the S-curve is wasted.
#[test]
fn test_detects_logistic_neuron_with_negative_operating_point() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            hidden_with_bias("h1", "LOGISTIC", -10.0),
            output("out", "IDENTITY"),
        ],
        vec![synapse("input-1", "h1", 0.5), synapse("h1", "out", 1.0)],
    );

    // Pre-activation always very negative → activation always near 0
    let mut records = Vec::new();
    for i in 0..100 {
        let value = -8.0 + (i as f32) * 0.02; // range [-8.0, -6.0]
        let activation = 1.0 / (1.0 + (-value).exp()); // logistic → near 0
        records.push(record_with_value("h1", i, activation, value));
    }

    let config = OperatingPointConfig::default();
    let detected =
        detect_operating_point_issues(&creature, &[("h1".to_string(), records)], &config);

    assert_eq!(
        detected.len(),
        1,
        "Should detect the poorly-placed LOGISTIC neuron"
    );
    assert_eq!(detected[0].neuron_uuid, "h1");
    assert!(
        detected[0].dynamic_range_utilisation < 0.20,
        "Utilisation should be very low: got {:.3}",
        detected[0].dynamic_range_utilisation
    );
}

/// A TANH neuron where pre-activation is always near zero (only using linear region)
/// should be flagged.
#[test]
fn test_detects_tanh_neuron_using_only_linear_region() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            hidden("h1", "TANH"),
            output("out", "IDENTITY"),
        ],
        vec![synapse("input-1", "h1", 0.01), synapse("h1", "out", 1.0)],
    );

    // Pre-activation in [-0.1, 0.1] → TANH outputs in ~[-0.1, 0.1] (linear region)
    let mut records = Vec::new();
    for i in 0..100 {
        let value = -0.1 + (i as f32) * 0.002;
        let activation = value.tanh();
        records.push(record_with_value("h1", i, activation, value));
    }

    let config = OperatingPointConfig::default();
    let detected =
        detect_operating_point_issues(&creature, &[("h1".to_string(), records)], &config);

    assert_eq!(
        detected.len(),
        1,
        "Should detect TANH neuron in linear region only"
    );
    assert!(detected[0].dynamic_range_utilisation < 0.15);
}

/// A TANH neuron well-placed in its active zone should NOT be flagged.
#[test]
fn test_well_placed_tanh_not_flagged() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            hidden("h1", "TANH"),
            output("out", "IDENTITY"),
        ],
        vec![synapse("input-1", "h1", 1.0), synapse("h1", "out", 1.0)],
    );

    // Pre-activation spread across [-2, 2] → TANH outputs span most of [-0.96, 0.96]
    let mut records = Vec::new();
    for i in 0..100 {
        let value = -2.0 + (i as f32) * 0.04;
        let activation = value.tanh();
        records.push(record_with_value("h1", i, activation, value));
    }

    let config = OperatingPointConfig::default();
    let detected =
        detect_operating_point_issues(&creature, &[("h1".to_string(), records)], &config);

    assert!(
        detected.is_empty(),
        "Well-placed neuron should not be flagged"
    );
}

/// Output neurons should be excluded from operating-point analysis.
#[test]
fn test_operating_point_output_neurons_excluded() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("out", "output", "LOGISTIC"),
        ],
        vec![synapse("input-1", "out", 0.01)],
    );

    let mut records = Vec::new();
    for i in 0..100 {
        let value = -0.05 + (i as f32) * 0.001;
        let activation = 1.0 / (1.0 + (-value).exp());
        records.push(record_with_value("out", i, activation, value));
    }

    let config = OperatingPointConfig::default();
    let detected =
        detect_operating_point_issues(&creature, &[("out".to_string(), records)], &config);

    assert!(detected.is_empty(), "Output neurons should be excluded");
}

/// Input neurons should be excluded from operating-point analysis.
#[test]
fn test_operating_point_input_neurons_excluded() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "LOGISTIC"),
            output("out", "IDENTITY"),
        ],
        vec![synapse("input-1", "out", 1.0)],
    );

    let mut records = Vec::new();
    for i in 0..100 {
        let value = -0.05 + (i as f32) * 0.001;
        let activation = 1.0 / (1.0 + (-value).exp());
        records.push(record_with_value("input-1", i, activation, value));
    }

    let config = OperatingPointConfig::default();
    let detected =
        detect_operating_point_issues(&creature, &[("input-1".to_string(), records)], &config);

    assert!(detected.is_empty(), "Input neurons should be excluded");
}

/// Neurons with fewer than `min_samples` should not be flagged.
#[test]
fn test_operating_point_insufficient_samples_skipped() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            hidden_with_bias("h1", "LOGISTIC", -10.0),
            output("out", "IDENTITY"),
        ],
        vec![synapse("input-1", "h1", 0.5), synapse("h1", "out", 1.0)],
    );

    // Only 5 samples (below the default threshold of 20)
    let mut records = Vec::new();
    for i in 0..5 {
        let value = -8.0 + (i as f32) * 0.5;
        let activation = 1.0 / (1.0 + (-value).exp());
        records.push(record_with_value("h1", i, activation, value));
    }

    let config = OperatingPointConfig::default();
    let detected =
        detect_operating_point_issues(&creature, &[("h1".to_string(), records)], &config);

    assert!(
        detected.is_empty(),
        "Too few samples should not trigger detection"
    );
}

/// Records without pre-activation values (value=None) should be skipped.
#[test]
fn test_records_without_value_skipped() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            hidden_with_bias("h1", "LOGISTIC", -10.0),
            output("out", "IDENTITY"),
        ],
        vec![synapse("input-1", "h1", 0.5), synapse("h1", "out", 1.0)],
    );

    // Records without pre-activation value
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| DiscoverRecord {
            obs_index: i,
            neuron_uuid: "h1".to_string(),
            value: None,
            activation: 0.01,
            errors: vec![0.01],
        })
        .collect();

    let config = OperatingPointConfig::default();
    let detected =
        detect_operating_point_issues(&creature, &[("h1".to_string(), records)], &config);

    assert!(
        detected.is_empty(),
        "Records without value should not trigger detection"
    );
}

/// Unbounded activations (IDENTITY, RELU) have no fixed active zone,
/// so they should not be flagged by this module.
#[test]
fn test_operating_point_unbounded_activations_excluded() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            hidden("h1", "IDENTITY"),
            hidden("h2", "RELU"),
            output("out", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "h1", 0.01),
            synapse("input-1", "h2", 0.01),
            synapse("h1", "out", 1.0),
            synapse("h2", "out", 1.0),
        ],
    );

    let mut records = Vec::new();
    for i in 0..100 {
        let value = -0.05 + (i as f32) * 0.001;
        records.push(record_with_value("h1", i, value, value)); // IDENTITY: activation==value
        records.push(record_with_value("h2", i, value.max(0.0), value));
    }

    let config = OperatingPointConfig::default();
    let h1_records: Vec<_> = records
        .iter()
        .filter(|r| r.neuron_uuid == "h1")
        .cloned()
        .collect();
    let h2_records: Vec<_> = records
        .iter()
        .filter(|r| r.neuron_uuid == "h2")
        .cloned()
        .collect();
    let detected = detect_operating_point_issues(
        &creature,
        &[
            ("h1".to_string(), h1_records),
            ("h2".to_string(), h2_records),
        ],
        &config,
    );

    assert!(
        detected.is_empty(),
        "Unbounded activations should be excluded"
    );
}

/// Multiple neurons — only the poorly-placed one should be detected.
#[test]
fn test_multiple_neurons_only_poorly_placed_detected() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            hidden_with_bias("good", "TANH", 0.0),
            hidden_with_bias("bad", "LOGISTIC", -10.0),
            output("out", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "good", 1.0),
            synapse("input-1", "bad", 0.5),
            synapse("good", "out", 1.0),
            synapse("bad", "out", 1.0),
        ],
    );

    // "good" neuron: well-placed
    let good_records: Vec<_> = (0..100)
        .map(|i| {
            let value = -2.0 + (i as f32) * 0.04;
            record_with_value("good", i, value.tanh(), value)
        })
        .collect();

    // "bad" neuron: stuck near 0
    let bad_records: Vec<_> = (0..100)
        .map(|i| {
            let value = -8.0 + (i as f32) * 0.02;
            let activation = 1.0 / (1.0 + (-value).exp());
            record_with_value("bad", i, activation, value)
        })
        .collect();

    let config = OperatingPointConfig::default();
    let detected = detect_operating_point_issues(
        &creature,
        &[
            ("good".to_string(), good_records),
            ("bad".to_string(), bad_records),
        ],
        &config,
    );

    assert_eq!(
        detected.len(),
        1,
        "Only the poorly-placed neuron should be detected"
    );
    assert_eq!(detected[0].neuron_uuid, "bad");
}

// ============================================================================
// Candidate generation tests
// ============================================================================

/// Candidate generation should produce setBias, changeSquash, and setWeight candidates.
#[test]
fn test_operating_point_candidate_generation_includes_expected_operations() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            hidden_with_bias("h1", "LOGISTIC", -10.0),
            output("out", "IDENTITY"),
        ],
        vec![synapse("input-1", "h1", 0.5), synapse("h1", "out", 1.0)],
    );

    let records: Vec<_> = (0..100)
        .map(|i| {
            let value = -8.0 + (i as f32) * 0.02;
            let activation = 1.0 / (1.0 + (-value).exp());
            record_with_value("h1", i, activation, value)
        })
        .collect();

    let config = OperatingPointConfig::default();
    let detected =
        detect_operating_point_issues(&creature, &[("h1".to_string(), records)], &config);

    assert!(!detected.is_empty());

    let candidates = operating_point_to_coordinated_candidates(&detected, &creature);
    assert!(!candidates.is_empty(), "Should produce candidates");

    // Check we have the three candidate types
    let comments: Vec<String> = candidates
        .iter()
        .filter_map(|c| c.comment.as_ref().cloned())
        .collect();

    let has_set_bias = comments
        .iter()
        .any(|c| c.contains("setBias") || c.contains("bias"));
    let has_change_squash = comments
        .iter()
        .any(|c| c.contains("changeSquash") || c.contains("squash"));
    let has_set_weight = comments
        .iter()
        .any(|c| c.contains("setWeight") || c.contains("weight"));

    assert!(
        has_set_bias,
        "Should have a setBias candidate. Comments: {comments:?}"
    );
    assert!(
        has_change_squash,
        "Should have a changeSquash candidate. Comments: {comments:?}"
    );
    assert!(
        has_set_weight,
        "Should have a setWeight candidate. Comments: {comments:?}"
    );

    // All candidates should have positive expected improvement
    for c in &candidates {
        assert!(
            c.expected_creature_score_gain > 0.0,
            "Expected positive improvement, got {}",
            c.expected_creature_score_gain
        );
    }
}

/// Configurable utilisation threshold should work.
#[test]
fn test_operating_point_configurable_utilisation_threshold() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            hidden("h1", "TANH"),
            output("out", "IDENTITY"),
        ],
        vec![synapse("input-1", "h1", 0.5), synapse("h1", "out", 1.0)],
    );

    // Pre-activation range [-1, 1] → TANH outputs ~[-0.76, 0.76] = ~76% utilisation
    let records: Vec<_> = (0..100)
        .map(|i| {
            let value = -1.0 + (i as f32) * 0.02;
            record_with_value("h1", i, value.tanh(), value)
        })
        .collect();

    // Default threshold (20%) — should NOT flag this neuron
    let default_config = OperatingPointConfig::default();
    let detected = detect_operating_point_issues(
        &creature,
        &[("h1".to_string(), records.clone())],
        &default_config,
    );
    assert!(
        detected.is_empty(),
        "76% utilisation should not be flagged at default threshold"
    );

    // High threshold (80%) — SHOULD flag this neuron
    let strict_config = OperatingPointConfig {
        utilisation_threshold: 0.80,
        ..Default::default()
    };
    let detected =
        detect_operating_point_issues(&creature, &[("h1".to_string(), records)], &strict_config);
    assert_eq!(
        detected.len(),
        1,
        "76% utilisation should be flagged at 80% threshold"
    );
}
