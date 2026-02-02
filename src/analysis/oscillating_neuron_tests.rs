//! Unit tests for oscillating neuron detection (Issue #376).

use super::*;

/// Helper: create a DiscoverRecord.
fn make_record(neuron_uuid: &str, obs_index: u32, activation: f32) -> DiscoverRecord {
    DiscoverRecord {
        obs_index,
        neuron_uuid: neuron_uuid.to_string(),
        value: None,
        activation,
        errors: vec![0.01],
    }
}

/// Helper: generate oscillating records that alternate between positive and negative.
fn oscillating_records(uuid: &str, count: usize, magnitude: f32) -> Vec<DiscoverRecord> {
    (0..count)
        .map(|i| {
            let activation = if i % 2 == 0 { magnitude } else { -magnitude };
            make_record(uuid, i as u32, activation)
        })
        .collect()
}

// ── Detection criteria ──────────────────────────────────────────────────────

#[test]
fn neuron_with_frequent_sign_changes_is_detected() {
    let neurons = vec![("n1".to_string(), "TANH".to_string(), 0.0_f32)];
    let records = vec![("n1".to_string(), oscillating_records("n1", 30, 0.5))];

    let detected = detect_oscillating_neurons(&neurons, &records);

    assert!(
        !detected.is_empty(),
        "expected oscillating neuron to be detected"
    );
    assert_eq!(detected[0].neuron_uuid, "n1");
    assert!(
        detected[0].sign_change_fraction >= 0.3,
        "sign change fraction should be above threshold"
    );
    assert_eq!(detected[0].recommended_squash, "ABSOLUTE");
}

#[test]
fn oscillating_logistic_neuron_recommends_relu() {
    let neurons = vec![("n1".to_string(), "LOGISTIC".to_string(), 0.0_f32)];
    let records = vec![("n1".to_string(), oscillating_records("n1", 30, 0.5))];

    let detected = detect_oscillating_neurons(&neurons, &records);

    assert!(!detected.is_empty());
    assert_eq!(detected[0].recommended_squash, "RELU");
}

// ── Exclusion criteria ──────────────────────────────────────────────────────

#[test]
fn neuron_with_consistent_sign_is_not_detected() {
    let neurons = vec![("n1".to_string(), "TANH".to_string(), 0.0_f32)];
    // All positive activations — no oscillation
    let records: Vec<DiscoverRecord> = (0..30)
        .map(|i| make_record("n1", i, 0.5 + (i as f32) * 0.001))
        .collect();
    let neuron_records = vec![("n1".to_string(), records)];

    let detected = detect_oscillating_neurons(&neurons, &neuron_records);

    assert!(
        detected.is_empty(),
        "neuron with consistent positive activations should not be detected"
    );
}

#[test]
fn near_dead_neuron_is_not_detected_as_oscillating() {
    let neurons = vec![("n1".to_string(), "TANH".to_string(), 0.0_f32)];
    // Oscillates but with tiny magnitude < MIN_MEAN_ABS_ACTIVATION (0.01)
    let records = vec![("n1".to_string(), oscillating_records("n1", 30, 0.001))];

    let detected = detect_oscillating_neurons(&neurons, &records);

    assert!(
        detected.is_empty(),
        "near-dead neuron should not be flagged as oscillating"
    );
}

#[test]
fn heavily_biased_sign_is_not_detected() {
    let neurons = vec![("n1".to_string(), "TANH".to_string(), 0.0_f32)];
    // 95% positive, 5% negative — minority fraction < MIN_MINORITY_SIGN_FRACTION (0.2)
    let records: Vec<DiscoverRecord> = (0..40)
        .map(|i| {
            let activation = if i < 38 { 0.5 } else { -0.5 };
            make_record("n1", i, activation)
        })
        .collect();
    let neuron_records = vec![("n1".to_string(), records)];

    let detected = detect_oscillating_neurons(&neurons, &neuron_records);

    assert!(
        detected.is_empty(),
        "heavily biased sign distribution should not be flagged"
    );
}

// ── Edge cases ──────────────────────────────────────────────────────────────

#[test]
fn insufficient_samples_prevents_oscillation_detection() {
    let neurons = vec![("n1".to_string(), "TANH".to_string(), 0.0_f32)];
    let records = vec![("n1".to_string(), oscillating_records("n1", 5, 0.5))];

    let detected = detect_oscillating_neurons(&neurons, &records);

    assert!(
        detected.is_empty(),
        "fewer than 20 samples should prevent detection"
    );
}

#[test]
fn bias_recommendation_shifts_toward_minority_sign() {
    let neurons = vec![("n1".to_string(), "TANH".to_string(), 0.0_f32)];
    // ~65% positive, ~35% negative with frequent sign changes (interleaved pattern)
    // Pattern: +, +, -, +, +, -, ... to get ~65% positive with frequent oscillation
    let records: Vec<DiscoverRecord> = (0..30)
        .map(|i| {
            let activation = if i % 3 == 2 { -0.5 } else { 0.5 };
            make_record("n1", i, activation)
        })
        .collect();
    let neuron_records = vec![("n1".to_string(), records)];

    let detected = detect_oscillating_neurons(&neurons, &neuron_records);

    assert!(
        !detected.is_empty(),
        "expected oscillating neuron with biased sign to be detected"
    );
    // positive_fraction ≈ 0.67 > 0.6 → recommended_bias_delta should be Some(-0.1)
    assert_eq!(detected[0].recommended_bias_delta, Some(-0.1));
}

// ── Conversion ──────────────────────────────────────────────────────────────

#[test]
fn conversion_produces_change_squash_operation() {
    let candidates = vec![OscillatingNeuronCandidate {
        neuron_uuid: "n1".to_string(),
        current_squash: "TANH".to_string(),
        sign_change_fraction: 0.5,
        positive_fraction: 0.5,
        mean_abs_activation: 0.5,
        sample_count: 30,
        recommended_squash: "ABSOLUTE".to_string(),
        recommended_bias_delta: None,
        estimated_improvement: 0.0025,
    }];

    let coordinated = oscillating_neurons_to_coordinated_candidates(&candidates);

    assert_eq!(coordinated.len(), 1);
    let first = &coordinated[0];
    let has_change_squash = first.operations.iter().any(|op| {
        matches!(op, CoordinatedStructuralOpJson::ChangeSquash { squash, .. } if squash == "ABSOLUTE")
    });
    assert!(has_change_squash, "expected ChangeSquash to ABSOLUTE");
    assert!(first.expected_creature_score_gain > 0.0);
}
