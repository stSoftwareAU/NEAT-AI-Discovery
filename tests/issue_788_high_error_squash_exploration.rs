//! Issue #788: Integration tests for high-error squash exploration detection.
//!
//! Tests the `high_error_squash_exploration` detection module which identifies
//! hidden neurons with high prediction error and proactively suggests alternative
//! activation functions that may reduce the error.
//!
//! This module increases change-squash candidate volume by triggering on error
//! magnitude rather than waiting for structural problems (saturation, mismatch).

use neat_ai_discovery::analysis::detection::high_error_squash_exploration::{
    HighErrorSquashCandidate, detect_high_error_squash_candidates,
    high_error_squash_to_coordinated_candidates,
};
use neat_ai_discovery::types::DiscoverRecord;

// =============================================================================
// Helpers
// =============================================================================

fn make_record(
    uuid: &str,
    idx: u32,
    value: Option<f32>,
    activation: f32,
    error: f32,
) -> DiscoverRecord {
    DiscoverRecord {
        obs_index: idx,
        neuron_uuid: uuid.to_string(),
        value,
        activation,
        errors: vec![error],
    }
}

/// Build hidden neuron tuples in the format expected by detection modules.
fn hidden_neuron(uuid: &str, squash: &str, bias: f32) -> (String, String, f32) {
    (uuid.to_string(), squash.to_string(), bias)
}

// =============================================================================
// Neuron with high error and pre-activation data triggers exploration
// =============================================================================

#[test]
fn high_error_neuron_with_pre_activation_data_triggers_exploration() {
    // A TANH neuron where the true target is linear (pre_act itself), meaning
    // TANH compresses the signal and causes large errors at the extremes.
    // IDENTITY should be recommended because it matches the target exactly.
    let neurons = vec![hidden_neuron("high-err", "TANH", 0.0)];

    // Pre-activation values span a wide range. TANH compresses the extremes,
    // so the error = target - activation = pre_act - tanh(pre_act) is large
    // at the tails, producing high MAE. IDENTITY would produce exact match.
    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "high-err".to_string(),
        (0..50)
            .map(|i| {
                let pre_act = (i as f32 - 25.0) / 8.0; // range -3.1 to 3.0
                let activation = pre_act.tanh(); // compressed by TANH
                // Error reflects that the target is pre_act (linear output needed)
                let error = pre_act - activation;
                make_record("high-err", i, Some(pre_act), activation, error)
            })
            .collect(),
    )];

    let candidates = detect_high_error_squash_candidates(&neurons, &records);

    assert!(
        !candidates.is_empty(),
        "Neuron with high error should trigger squash exploration"
    );

    let c = &candidates[0];
    assert_eq!(c.neuron_uuid, "high-err");
    assert_eq!(c.current_squash, "TANH");
    assert!(
        c.recommended_squash.is_some(),
        "Should recommend an alternative activation"
    );
    assert!(
        c.estimated_improvement > 0.0,
        "Should have positive estimated improvement"
    );
    assert!(
        c.mean_absolute_error > 0.0,
        "Should report the mean absolute error"
    );
}

// =============================================================================
// Neuron with low error does NOT trigger exploration
// =============================================================================

#[test]
fn low_error_neuron_not_detected() {
    let neurons = vec![hidden_neuron("low-err", "TANH", 0.0)];

    // Low error — no reason to change activation
    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "low-err".to_string(),
        (0..50)
            .map(|i| {
                let pre_act = (i as f32 - 25.0) / 50.0;
                let activation = pre_act.tanh();
                let error = 0.001; // very low error
                make_record("low-err", i, Some(pre_act), activation, error)
            })
            .collect(),
    )];

    let candidates = detect_high_error_squash_candidates(&neurons, &records);

    assert!(
        candidates.is_empty(),
        "Neuron with low error should NOT trigger squash exploration"
    );
}

// =============================================================================
// Neuron without pre-activation data is skipped
// =============================================================================

#[test]
fn neuron_without_pre_activation_skipped() {
    let neurons = vec![hidden_neuron("no-pre", "TANH", 0.0)];

    // No pre-activation values (value = None)
    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "no-pre".to_string(),
        (0..50)
            .map(|i| {
                let activation = (i as f32 - 25.0) / 27.0;
                make_record("no-pre", i, None, activation, 0.5)
            })
            .collect(),
    )];

    let candidates = detect_high_error_squash_candidates(&neurons, &records);

    assert!(
        candidates.is_empty(),
        "Neuron without pre-activation data should be skipped"
    );
}

// =============================================================================
// Insufficient samples returns empty
// =============================================================================

#[test]
fn insufficient_samples_returns_empty() {
    let neurons = vec![hidden_neuron("few-samp", "TANH", 0.0)];

    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "few-samp".to_string(),
        (0..5)
            .map(|i| make_record("few-samp", i, Some(0.5), 0.46, 0.5))
            .collect(),
    )];

    let candidates = detect_high_error_squash_candidates(&neurons, &records);

    assert!(
        candidates.is_empty(),
        "Too few samples should return empty results"
    );
}

// =============================================================================
// IDENTITY neurons are skipped (already unbounded/linear)
// =============================================================================

#[test]
fn identity_neurons_skipped() {
    let neurons = vec![hidden_neuron("ident", "IDENTITY", 0.0)];

    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "ident".to_string(),
        (0..50)
            .map(|i| {
                let val = (i as f32 - 25.0) / 10.0;
                make_record("ident", i, Some(val), val, 0.5)
            })
            .collect(),
    )];

    let candidates = detect_high_error_squash_candidates(&neurons, &records);

    assert!(
        candidates.is_empty(),
        "IDENTITY neurons should be skipped (already linear)"
    );
}

// =============================================================================
// Does not recommend the same activation function
// =============================================================================

#[test]
fn does_not_recommend_same_squash() {
    let neurons = vec![hidden_neuron("same-sq", "TANH", 0.0)];

    // Even with high error, if no alternative is better, no recommendation
    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "same-sq".to_string(),
        (0..50)
            .map(|i| {
                // Values in the ideal TANH operating range
                let pre_act = (i as f32 - 25.0) / 25.0; // range -1 to 1
                let activation = pre_act.tanh();
                // Error is high but evenly distributed — TANH is still the best fit
                let error = 0.2;
                make_record("same-sq", i, Some(pre_act), activation, error)
            })
            .collect(),
    )];

    let candidates = detect_high_error_squash_candidates(&neurons, &records);

    // Any candidate found must recommend a DIFFERENT squash
    for c in &candidates {
        if let Some(ref rec) = c.recommended_squash {
            assert_ne!(
                rec, &c.current_squash,
                "Must not recommend the same activation function"
            );
        }
    }
}

// =============================================================================
// Coordinated candidate conversion
// =============================================================================

#[test]
fn candidates_convert_to_coordinated_candidates() {
    let candidates = vec![HighErrorSquashCandidate {
        neuron_uuid: "h1".to_string(),
        current_squash: "TANH".to_string(),
        recommended_squash: Some("IDENTITY".to_string()),
        mean_absolute_error: 0.3,
        error_reduction_fraction: 0.25,
        estimated_improvement: 0.005,
        reason: "High error neuron — IDENTITY reduces error by 25%".to_string(),
    }];

    let coordinated = high_error_squash_to_coordinated_candidates(&candidates);

    assert_eq!(coordinated.len(), 1);
    assert_eq!(coordinated[0].operations.len(), 1);
    assert!(coordinated[0].expected_creature_score_gain > 0.0);
    assert!(coordinated[0].comment.as_ref().unwrap().contains("788"));
}

// =============================================================================
// Multiple neurons can produce multiple candidates
// =============================================================================

#[test]
fn multiple_high_error_neurons_detected() {
    let neurons = vec![
        hidden_neuron("h-a", "TANH", 0.0),
        hidden_neuron("h-b", "LOGISTIC", 0.0),
    ];

    // Both neurons have errors indicating their target is linear (pre_act),
    // so both should trigger recommendations for IDENTITY or similar.
    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        (
            "h-a".to_string(),
            (0..50)
                .map(|i| {
                    let pre_act = (i as f32 - 25.0) / 8.0;
                    let activation = pre_act.tanh();
                    let error = pre_act - activation;
                    make_record("h-a", i, Some(pre_act), activation, error)
                })
                .collect(),
        ),
        (
            "h-b".to_string(),
            (0..50)
                .map(|i| {
                    let pre_act = (i as f32 - 25.0) / 8.0;
                    let logistic = 1.0 / (1.0 + (-pre_act).exp());
                    let error = pre_act - logistic;
                    make_record("h-b", i, Some(pre_act), logistic, error)
                })
                .collect(),
        ),
    ];

    let candidates = detect_high_error_squash_candidates(&neurons, &records);

    // Both should trigger (both have high error with linear target pattern)
    assert!(
        candidates.len() >= 2,
        "Both high-error neurons should trigger exploration, got {}",
        candidates.len()
    );
}

// =============================================================================
// Neuron not in records returns empty
// =============================================================================

#[test]
fn neuron_without_records_returns_empty() {
    let neurons = vec![hidden_neuron("missing", "RELU", 0.0)];
    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![];

    let candidates = detect_high_error_squash_candidates(&neurons, &records);

    assert!(
        candidates.is_empty(),
        "Neuron without records should return empty"
    );
}
