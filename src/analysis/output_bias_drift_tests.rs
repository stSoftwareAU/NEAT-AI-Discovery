//! Unit tests for output bias drift detection (Issue #376).

use super::*;
use crate::{CreatureJson, NeuronJson, SynapseJson};

/// Helper: build a NeuronJson with specified bias.
fn neuron_with_bias(uuid: &str, neuron_type: &str, squash: &str, bias: f32) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
        squash: squash.to_string(),
        bias,
    }
}

/// Helper: build a SynapseJson.
fn synapse(from: &str, to: &str, weight: f32) -> SynapseJson {
    SynapseJson {
        from_uuid: from.to_string(),
        to_uuid: to.to_string(),
        weight,
        synapse_type: None,
    }
}

/// Helper: create records with consistent positive errors.
fn positive_error_records(uuid: &str, count: usize, error: f32) -> Vec<DiscoverRecord> {
    (0..count)
        .map(|i| DiscoverRecord {
            obs_index: i as u32,
            neuron_uuid: uuid.to_string(),
            value: None,
            activation: 0.5,
            errors: vec![error],
        })
        .collect()
}

/// Helper: create records with mixed sign errors.
fn mixed_error_records(uuid: &str, count: usize) -> Vec<DiscoverRecord> {
    (0..count)
        .map(|i| DiscoverRecord {
            obs_index: i as u32,
            neuron_uuid: uuid.to_string(),
            value: None,
            activation: 0.5,
            errors: vec![if i % 2 == 0 { 0.1 } else { -0.1 }],
        })
        .collect()
}

fn make_output_creature(bias: f32) -> CreatureJson {
    CreatureJson {
        neurons: vec![
            neuron_with_bias("i0", "input", "IDENTITY", 0.0),
            neuron_with_bias("o1", "output", "IDENTITY", bias),
        ],
        synapses: vec![synapse("i0", "o1", 1.0)],
        input: 1,
        output: 1,
    }
}

// ── Detection criteria ──────────────────────────────────────────────────────

#[test]
fn output_with_consistently_positive_errors_is_detected() {
    let creature = make_output_creature(0.0);
    let records = vec![("o1".to_string(), positive_error_records("o1", 30, 0.5))];

    let detected = detect_output_bias_drift(&creature, &records);

    assert!(
        !detected.is_empty(),
        "expected output bias drift to be detected"
    );
    assert_eq!(detected[0].neuron_uuid, "o1");
    assert!(detected[0].mean_error > 0.0);
    // Recommended bias delta should be negative (to counteract positive errors)
    assert!(
        detected[0].recommended_bias_delta < 0.0,
        "recommended delta should counteract the mean error"
    );
}

#[test]
fn output_with_consistently_negative_errors_is_detected() {
    let creature = make_output_creature(0.0);
    let records = vec![("o1".to_string(), positive_error_records("o1", 30, -0.5))];

    let detected = detect_output_bias_drift(&creature, &records);

    assert!(
        !detected.is_empty(),
        "expected negative bias drift to be detected"
    );
    assert!(detected[0].recommended_bias_delta > 0.0);
}

// ── Exclusion criteria ──────────────────────────────────────────────────────

#[test]
fn output_with_balanced_errors_is_not_detected() {
    let creature = make_output_creature(0.0);
    // 50/50 positive/negative errors — no sign dominance
    let records = vec![("o1".to_string(), mixed_error_records("o1", 30))];

    let detected = detect_output_bias_drift(&creature, &records);

    assert!(
        detected.is_empty(),
        "balanced positive/negative errors should not be detected as drift"
    );
}

#[test]
fn hidden_neuron_is_not_detected() {
    let creature = CreatureJson {
        neurons: vec![
            neuron_with_bias("i0", "input", "IDENTITY", 0.0),
            neuron_with_bias("h1", "hidden", "TANH", 0.0),
            neuron_with_bias("o1", "output", "IDENTITY", 0.0),
        ],
        synapses: vec![synapse("i0", "h1", 1.0), synapse("h1", "o1", 1.0)],
        input: 1,
        output: 1,
    };

    // Provide records only for the hidden neuron with biased errors
    let records = vec![("h1".to_string(), positive_error_records("h1", 30, 0.5))];

    let detected = detect_output_bias_drift(&creature, &records);

    assert!(detected.is_empty(), "hidden neurons should not be detected");
}

#[test]
fn small_error_magnitude_is_not_detected() {
    let creature = make_output_creature(0.0);
    // Errors all positive but very small — below MIN_MEAN_ERROR_MAGNITUDE (0.01)
    let records = vec![("o1".to_string(), positive_error_records("o1", 30, 0.005))];

    let detected = detect_output_bias_drift(&creature, &records);

    assert!(
        detected.is_empty(),
        "error magnitude below threshold should not trigger detection"
    );
}

// ── Edge cases ──────────────────────────────────────────────────────────────

#[test]
fn insufficient_samples_prevents_bias_drift_detection() {
    let creature = make_output_creature(0.0);
    // Only 5 samples — below MIN_SAMPLES_FOR_BIAS_DRIFT (20)
    let records = vec![("o1".to_string(), positive_error_records("o1", 5, 0.5))];

    let detected = detect_output_bias_drift(&creature, &records);

    assert!(
        detected.is_empty(),
        "fewer than 20 samples should prevent detection"
    );
}

#[test]
fn records_without_errors_are_skipped() {
    let creature = make_output_creature(0.0);
    // Records with empty error vectors
    let records: Vec<DiscoverRecord> = (0..30)
        .map(|i| DiscoverRecord {
            obs_index: i,
            neuron_uuid: "o1".to_string(),
            value: None,
            activation: 0.5,
            errors: vec![],
        })
        .collect();

    let neuron_records = vec![("o1".to_string(), records)];

    let detected = detect_output_bias_drift(&creature, &neuron_records);

    assert!(
        detected.is_empty(),
        "records without errors should be effectively skipped"
    );
}

// ── Conversion ──────────────────────────────────────────────────────────────

#[test]
fn conversion_produces_set_bias_operation_with_adjusted_value() {
    let candidates = vec![OutputBiasDriftCandidate {
        neuron_uuid: "o1".to_string(),
        current_bias: 0.5,
        mean_error: 0.3,
        positive_error_fraction: 0.9,
        sample_count: 30,
        recommended_bias_delta: -0.3,
        estimated_improvement: 0.012,
    }];

    let coordinated = output_bias_drift_to_coordinated_candidates(&candidates);

    assert_eq!(coordinated.len(), 1);
    let first = &coordinated[0];

    let has_set_bias = first.operations.iter().any(|op| {
        matches!(op, CoordinatedStructuralOpJson::SetBias { neuron_uuid, bias }
            if neuron_uuid == "o1" && (*bias - 0.2).abs() < 1e-6) // 0.5 + (-0.3) = 0.2
    });
    assert!(
        has_set_bias,
        "expected SetBias with adjusted value (current + delta)"
    );
    assert!(first.expected_creature_score_gain > 0.0);
}

#[test]
fn conversion_empty_candidates_returns_empty() {
    let candidates: Vec<OutputBiasDriftCandidate> = vec![];
    let coordinated = output_bias_drift_to_coordinated_candidates(&candidates);
    assert!(coordinated.is_empty());
}
