//! Unit tests for saturated neuron detection (Issue #376).

use super::*;

/// Helper: create a DiscoverRecord with the given activation and optional value.
fn make_record(
    neuron_uuid: &str,
    obs_index: u32,
    activation: f32,
    value: Option<f32>,
) -> DiscoverRecord {
    DiscoverRecord {
        obs_index,
        neuron_uuid: neuron_uuid.to_string(),
        value,
        activation,
        errors: vec![0.01],
    }
}

/// Helper: generate `count` records with the same activation for a neuron.
fn constant_activation_records(uuid: &str, activation: f32, count: usize) -> Vec<DiscoverRecord> {
    (0..count)
        .map(|i| make_record(uuid, i as u32, activation, Some(activation * 2.0)))
        .collect()
}

// ── Detection criteria ──────────────────────────────────────────────────────

#[test]
fn tanh_neuron_saturated_near_positive_bound_is_detected() {
    let neurons = vec![("n1".to_string(), "TANH".to_string(), 0.5_f32)];
    let records = vec![(
        "n1".to_string(),
        constant_activation_records("n1", 0.98, 30),
    )];

    let detected = detect_saturated_neurons(&neurons, &records);

    assert!(
        !detected.is_empty(),
        "expected saturated TANH neuron to be detected"
    );
    assert_eq!(detected[0].neuron_uuid, "n1");
    assert_eq!(detected[0].recommended_squash, Some("IDENTITY".to_string()));
}

#[test]
fn logistic_neuron_saturated_near_zero_is_detected() {
    let neurons = vec![("n1".to_string(), "LOGISTIC".to_string(), -2.0_f32)];
    let records = vec![(
        "n1".to_string(),
        constant_activation_records("n1", 0.02, 30),
    )];

    let detected = detect_saturated_neurons(&neurons, &records);

    assert!(
        !detected.is_empty(),
        "expected saturated LOGISTIC neuron near 0 to be detected"
    );
    assert_eq!(detected[0].neuron_uuid, "n1");
}

#[test]
fn relu_dead_zone_neuron_is_detected() {
    let neurons = vec![("n1".to_string(), "RELU".to_string(), 0.0_f32)];
    let records = vec![("n1".to_string(), constant_activation_records("n1", 0.0, 30))];

    let detected = detect_saturated_neurons(&neurons, &records);

    assert!(
        !detected.is_empty(),
        "expected dead-zone RELU neuron to be detected"
    );
    assert_eq!(detected[0].recommended_squash, Some("IDENTITY".to_string()));
    assert!(
        detected[0].recommended_bias_delta.is_some(),
        "expected a positive bias recommendation to revive the neuron"
    );
}

// ── Exclusion criteria ──────────────────────────────────────────────────────

#[test]
fn tanh_neuron_in_active_region_is_not_detected() {
    let neurons = vec![("n1".to_string(), "TANH".to_string(), 0.0_f32)];
    // Mean activation ~0.3 — well within the active region
    let records = vec![("n1".to_string(), constant_activation_records("n1", 0.3, 30))];

    let detected = detect_saturated_neurons(&neurons, &records);

    assert!(
        detected.is_empty(),
        "neuron with moderate activation should not be detected"
    );
}

#[test]
fn identity_neuron_is_never_detected() {
    let neurons = vec![("n1".to_string(), "IDENTITY".to_string(), 0.0_f32)];
    // Even extreme activations should not trigger IDENTITY (unbounded)
    let records = vec![(
        "n1".to_string(),
        constant_activation_records("n1", 100.0, 30),
    )];

    let detected = detect_saturated_neurons(&neurons, &records);

    assert!(
        detected.is_empty(),
        "IDENTITY (unbounded) should never be flagged as saturated"
    );
}

#[test]
fn high_activation_variance_prevents_detection() {
    let neurons = vec![("n1".to_string(), "TANH".to_string(), 0.0_f32)];
    // Activations oscillate between 0.98 and 0.5 — high std dev
    let records: Vec<DiscoverRecord> = (0..30)
        .map(|i| {
            let activation = if i % 2 == 0 { 0.98 } else { 0.5 };
            make_record("n1", i, activation, None)
        })
        .collect();
    let neuron_records = vec![("n1".to_string(), records)];

    let detected = detect_saturated_neurons(&neurons, &neuron_records);

    assert!(
        detected.is_empty(),
        "high activation variance should prevent saturation detection"
    );
}

// ── Edge cases ──────────────────────────────────────────────────────────────

#[test]
fn insufficient_samples_prevents_detection() {
    let neurons = vec![("n1".to_string(), "TANH".to_string(), 0.5_f32)];
    // Only 5 samples — below MIN_SAMPLES_FOR_SATURATION (20)
    let records = vec![("n1".to_string(), constant_activation_records("n1", 0.99, 5))];

    let detected = detect_saturated_neurons(&neurons, &records);

    assert!(
        detected.is_empty(),
        "fewer than 20 samples should prevent detection"
    );
}

#[test]
fn empty_neuron_list_returns_empty() {
    let neurons: Vec<(String, String, f32)> = vec![];
    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![];

    let detected = detect_saturated_neurons(&neurons, &records);

    assert!(detected.is_empty());
}

#[test]
fn neuron_with_no_matching_records_is_skipped() {
    let neurons = vec![("n1".to_string(), "TANH".to_string(), 0.0_f32)];
    // Records for a different neuron
    let records = vec![(
        "n2".to_string(),
        constant_activation_records("n2", 0.99, 30),
    )];

    let detected = detect_saturated_neurons(&neurons, &records);

    assert!(
        detected.is_empty(),
        "neuron without matching records should be skipped"
    );
}

// ── Conversion ──────────────────────────────────────────────────────────────

#[test]
fn conversion_produces_change_squash_and_set_bias_operations() {
    let candidates = vec![SaturatedNeuronCandidate {
        neuron_uuid: "n1".to_string(),
        current_squash: "TANH".to_string(),
        mean_activation: 0.98,
        activation_std_dev: 0.01,
        input_std_dev: 0.5,
        recommended_squash: Some("IDENTITY".to_string()),
        recommended_bias_delta: Some(-1.0),
        estimated_improvement: 0.005,
    }];

    let coordinated = saturated_neurons_to_coordinated_candidates(&candidates);

    // Should produce both a squash+bias candidate and a bias-only candidate
    assert!(
        coordinated.len() >= 2,
        "expected at least 2 coordinated candidates (squash+bias and bias-only)"
    );

    // First candidate: ChangeSquash + SetBias combined
    let first = &coordinated[0];
    assert!(first.expected_creature_score_gain > 0.0);
    let has_change_squash = first
        .operations
        .iter()
        .any(|op| matches!(op, CoordinatedStructuralOpJson::ChangeSquash { .. }));
    assert!(
        has_change_squash,
        "primary candidate should include ChangeSquash"
    );

    // Verify comment mentions the neuron
    assert!(first.comment.as_ref().unwrap().contains("n1"));
}

#[test]
fn conversion_empty_candidates_returns_empty() {
    let candidates: Vec<SaturatedNeuronCandidate> = vec![];
    let coordinated = saturated_neurons_to_coordinated_candidates(&candidates);
    assert!(coordinated.is_empty());
}
