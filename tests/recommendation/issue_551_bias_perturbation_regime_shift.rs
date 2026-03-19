//! Tests for Issue #551: Local minimum escape — bias perturbation for activation
//! regime shifts.
//!
//! ## TDD Plan
//! 1. Test detection identifies neurons in saturated tail regimes
//! 2. Test detection identifies neurons in flat/linear-only regimes
//! 3. Test setBias candidates target specific regime shifts (e.g., tail → centre)
//! 4. Test no false positives for neurons in healthy operating regimes
//! 5. Test coordinated candidates include compensating weight adjustments
//! 6. Test multiple suboptimal neurons each produce candidates
//! 7. Test insufficient samples returns empty

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use neat_ai_discovery::analysis::detection::bias_perturbation::{
    bias_perturbation_to_coordinated_candidates, detect_bias_perturbation_candidates,
};
use neat_ai_discovery::types::DiscoverRecord;

// =============================================================================
// Helpers
// =============================================================================

fn make_record(uuid: &str, idx: u32, value: f32, activation: f32, error: f32) -> DiscoverRecord {
    DiscoverRecord {
        obs_index: idx,
        neuron_uuid: uuid.to_string(),
        value: Some(value),
        activation,
        errors: vec![error],
    }
}

fn hidden_neuron(uuid: &str, squash: &str, bias: f32) -> (String, String, f32) {
    (uuid.to_string(), squash.to_string(), bias)
}

// =============================================================================
// 1. Detects neurons operating in saturated tail regime
// =============================================================================

#[test]
fn test_detects_neuron_in_saturated_tail_regime() {
    // TANH neuron with bias=5.0 — pre-activation values are all far positive,
    // so the neuron is stuck in the saturated upper tail (~1.0 output).
    let hidden_neurons = vec![hidden_neuron("h1", "TANH", 5.0)];

    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "h1".to_string(),
        (0..60)
            .map(|i| {
                // Pre-activation values all in [4.5, 5.5] — deep in TANH saturation
                let value = 4.5 + (i as f32) / 60.0;
                let activation = value.tanh();
                let error = 0.3; // Consistent error — stuck
                make_record("h1", i, value, activation, error)
            })
            .collect(),
    )];

    let candidates = detect_bias_perturbation_candidates(&hidden_neurons, &records);
    assert!(
        !candidates.is_empty(),
        "Should detect neuron in saturated tail regime"
    );

    let coordinated = bias_perturbation_to_coordinated_candidates(&candidates);
    assert!(
        !coordinated.is_empty(),
        "Should produce coordinated candidates"
    );

    // Must include a setBias operation targeting the active zone
    let ops_json = serde_json::to_string(&coordinated[0].operations).unwrap();
    assert!(
        ops_json.contains("setBias"),
        "Must include setBias for regime shift, got: {ops_json}"
    );
}

// =============================================================================
// 2. Detects neurons in flat/linear-only regime
// =============================================================================

#[test]
fn test_detects_neuron_in_saturated_negative_tail() {
    // LOGISTIC neuron with large negative bias — stuck near output 0.0
    // in the saturated lower tail.
    let hidden_neurons = vec![hidden_neuron("h1", "LOGISTIC", -6.0)];

    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "h1".to_string(),
        (0..60)
            .map(|i| {
                // Pre-activation in [-6.5, -5.5] — deep in logistic saturation
                let value = -6.5 + (i as f32) / 60.0;
                let activation = 1.0 / (1.0 + (-value).exp());
                let error = 0.25;
                make_record("h1", i, value, activation, error)
            })
            .collect(),
    )];

    let candidates = detect_bias_perturbation_candidates(&hidden_neurons, &records);
    assert!(
        !candidates.is_empty(),
        "Should detect neuron stuck in saturated negative tail of LOGISTIC"
    );
}

// =============================================================================
// 3. setBias candidates target specific regime shifts
// =============================================================================

#[test]
fn test_set_bias_targets_active_zone_centre() {
    // TANH neuron stuck in the saturated positive tail (bias=6.0).
    // The new bias should shift the operating point towards the centre
    // of the active zone (~0.0 for TANH).
    let hidden_neurons = vec![hidden_neuron("h1", "TANH", 6.0)];

    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "h1".to_string(),
        (0..60)
            .map(|i| {
                let value = 5.5 + (i as f32) / 100.0;
                let activation = value.tanh();
                let error = 0.2;
                make_record("h1", i, value, activation, error)
            })
            .collect(),
    )];

    let candidates = detect_bias_perturbation_candidates(&hidden_neurons, &records);
    assert!(!candidates.is_empty());

    // The recommended bias should bring the neuron significantly closer to the
    // active zone centre
    let candidate = &candidates[0];
    assert!(
        candidate.recommended_bias.abs() < candidate.current_bias.abs(),
        "Recommended bias ({}) should be closer to active zone centre than current ({})",
        candidate.recommended_bias,
        candidate.current_bias
    );
}

// =============================================================================
// 4. No false positives for healthy operating regimes
// =============================================================================

#[test]
fn test_no_false_positives_for_healthy_regimes() {
    // TANH neuron operating well within the active zone [-2, 2] with low error
    let hidden_neurons = vec![hidden_neuron("h1", "TANH", 0.0)];

    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "h1".to_string(),
        (0..60)
            .map(|i| {
                // Pre-activation spans the active zone well
                let value = -1.5 + (i as f32) * 3.0 / 60.0;
                let activation = value.tanh();
                let error = 0.01; // Very low error = healthy
                make_record("h1", i, value, activation, error)
            })
            .collect(),
    )];

    let candidates = detect_bias_perturbation_candidates(&hidden_neurons, &records);
    assert!(
        candidates.is_empty(),
        "Healthy operating regime should produce no candidates"
    );
}

// =============================================================================
// 5. Coordinated candidates include compensating weight adjustments
// =============================================================================

#[test]
fn test_coordinated_candidates_include_set_bias_with_positive_gain() {
    let hidden_neurons = vec![hidden_neuron("h1", "TANH", 5.0)];

    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "h1".to_string(),
        (0..60)
            .map(|i| {
                let value = 4.5 + (i as f32) / 60.0;
                let activation = value.tanh();
                let error = 0.3;
                make_record("h1", i, value, activation, error)
            })
            .collect(),
    )];

    let candidates = detect_bias_perturbation_candidates(&hidden_neurons, &records);
    assert!(!candidates.is_empty());

    let coordinated = bias_perturbation_to_coordinated_candidates(&candidates);
    assert!(!coordinated.is_empty());

    // Positive expected gain
    assert!(
        coordinated[0].expected_creature_score_gain > 0.0,
        "Expected gain should be positive for regime shift"
    );

    // Comment should reference issue #551
    assert!(
        coordinated[0].comment.as_ref().unwrap().contains("551"),
        "Comment should reference issue #551, got: {:?}",
        coordinated[0].comment
    );
}

// =============================================================================
// 6. Multiple suboptimal neurons each produce candidates
// =============================================================================

#[test]
fn test_multiple_suboptimal_neurons_produce_candidates() {
    let hidden_neurons = vec![
        hidden_neuron("h1", "TANH", 5.0),
        hidden_neuron("h2", "LOGISTIC", -6.0),
    ];

    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        (
            "h1".to_string(),
            (0..60)
                .map(|i| {
                    let value = 4.5 + (i as f32) / 60.0;
                    let activation = value.tanh();
                    make_record("h1", i, value, activation, 0.25)
                })
                .collect(),
        ),
        (
            "h2".to_string(),
            (0..60)
                .map(|i| {
                    // Far negative pre-activation — stuck near 0 on logistic
                    let value = -6.5 + (i as f32) / 60.0;
                    let activation = 1.0 / (1.0 + (-value).exp());
                    make_record("h2", i, value, activation, 0.20)
                })
                .collect(),
        ),
    ];

    let candidates = detect_bias_perturbation_candidates(&hidden_neurons, &records);
    assert!(
        candidates.len() >= 2,
        "Should detect both suboptimal neurons, got: {}",
        candidates.len()
    );

    let uuids: Vec<&str> = candidates.iter().map(|c| c.neuron_uuid.as_str()).collect();
    assert!(uuids.contains(&"h1"), "Should detect h1");
    assert!(uuids.contains(&"h2"), "Should detect h2");
}

// =============================================================================
// 7. Insufficient samples returns empty
// =============================================================================

#[test]
fn test_bias_perturbation_insufficient_samples_returns_empty() {
    let hidden_neurons = vec![hidden_neuron("h1", "TANH", 5.0)];

    // Only 5 samples — below MIN_DISCOVERY_SAMPLE_COUNT
    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "h1".to_string(),
        (0..5)
            .map(|i| {
                let value = 4.5 + (i as f32) / 10.0;
                let activation = value.tanh();
                make_record("h1", i, value, activation, 0.3)
            })
            .collect(),
    )];

    let candidates = detect_bias_perturbation_candidates(&hidden_neurons, &records);
    assert!(
        candidates.is_empty(),
        "Insufficient samples should produce no candidates"
    );
}
