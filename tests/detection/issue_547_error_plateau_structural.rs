//! Tests for Issue #547: Error stagnation plateau detection — coordinated
//! `changeSquash` + `setBias` structural escape candidates.
//!
//! ## TDD Plan
//! 1. Test coordinated candidates include both changeSquash and setBias operations
//! 2. Test recommended bias is computed from mean error to centre the output
//! 3. Test no setBias generated when bias adjustment is negligible
//! 4. Test multiple plateau neurons each get their own coordinated pair
//! 5. Test that healthy networks produce no candidates
//! 6. Test that the estimated improvement accounts for both squash and bias changes

use neat_ai_discovery::analysis::detection::error_plateau::{
    detect_error_plateaus, error_plateaus_to_coordinated_candidates,
};
use neat_ai_discovery::types::DiscoverRecord;

// =============================================================================
// Helpers
// =============================================================================

fn make_record(uuid: &str, idx: u32, activation: f32, error: f32) -> DiscoverRecord {
    DiscoverRecord {
        obs_index: idx,
        neuron_uuid: uuid.to_string(),
        value: Some(activation),
        activation,
        errors: vec![error],
    }
}

fn output_neuron(uuid: &str, squash: &str) -> (String, String, f32) {
    (uuid.to_string(), squash.to_string(), 0.0)
}

fn output_neuron_with_bias(uuid: &str, squash: &str, bias: f32) -> (String, String, f32) {
    (uuid.to_string(), squash.to_string(), bias)
}

// =============================================================================
// 1. Coordinated candidates include both changeSquash and setBias
// =============================================================================

#[test]
fn test_coordinated_candidate_has_squash_and_bias_operations() {
    let outputs = vec![output_neuron("output-1", "HARD_TANH")];

    // Plateau: consistently high error (~0.3), tightly clustered
    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "output-1".to_string(),
        (0..60)
            .map(|i| {
                let activation = (i as f32 - 30.0) / 40.0;
                let error = 0.30 + (i as f32 * 0.001).sin() * 0.01;
                make_record("output-1", i, activation, error)
            })
            .collect(),
    )];

    let candidates = detect_error_plateaus(&outputs, &records);
    assert!(!candidates.is_empty(), "Should detect error plateau");

    let coordinated = error_plateaus_to_coordinated_candidates(&candidates);
    assert!(
        !coordinated.is_empty(),
        "Should produce coordinated candidates"
    );

    let ops_json = serde_json::to_string(&coordinated[0].operations).unwrap();

    // Must contain both changeSquash and setBias for a structural jump
    assert!(
        ops_json.contains("changeSquash"),
        "Coordinated candidate must include changeSquash, got: {ops_json}"
    );
    assert!(
        ops_json.contains("setBias"),
        "Coordinated candidate must include setBias for structural escape, got: {ops_json}"
    );
}

// =============================================================================
// 2. Recommended bias adjusts towards reducing mean error
// =============================================================================

#[test]
fn test_recommended_bias_adjusts_for_plateau() {
    // Neuron with existing bias of 0.0 and consistent positive error (~0.4)
    // means the output is consistently too high → bias should be adjusted
    let outputs = vec![output_neuron_with_bias("output-1", "HARD_TANH", 0.0)];

    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "output-1".to_string(),
        (0..60)
            .map(|i| {
                let activation = 0.5;
                // Consistent positive error means activation is above target
                let error = 0.40 + (i as f32 * 0.001).sin() * 0.01;
                make_record("output-1", i, activation, error)
            })
            .collect(),
    )];

    let candidates = detect_error_plateaus(&outputs, &records);
    assert!(
        !candidates.is_empty(),
        "Should detect error plateau with consistent positive error"
    );

    // The recommended bias should be different from the current bias
    let candidate = &candidates[0];
    assert!(
        (candidate.recommended_bias - 0.0).abs() > 0.01,
        "Recommended bias should differ from current bias (0.0), got: {}",
        candidate.recommended_bias
    );
}

// =============================================================================
// 3. No setBias when bias adjustment is negligible
// =============================================================================

#[test]
fn test_no_set_bias_when_adjustment_negligible() {
    // Errors are symmetric around zero → mean signed error ≈ 0 → no bias shift needed
    let outputs = vec![output_neuron("output-1", "HARD_TANH")];

    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "output-1".to_string(),
        (0..60)
            .map(|i| {
                let activation = (i as f32 - 30.0) / 40.0;
                // Alternating sign errors with same magnitude → mean signed error ≈ 0
                // but abs error is consistently ~0.3 (plateau)
                let error = if i % 2 == 0 { 0.30 } else { -0.30 };
                make_record("output-1", i, activation, error)
            })
            .collect(),
    )];

    let candidates = detect_error_plateaus(&outputs, &records);
    assert!(
        !candidates.is_empty(),
        "Should detect error plateau even with symmetric errors"
    );

    let coordinated = error_plateaus_to_coordinated_candidates(&candidates);
    assert!(
        !coordinated.is_empty(),
        "Should produce coordinated candidates"
    );

    // When bias adjustment is negligible, should only have changeSquash (no setBias)
    let ops_json = serde_json::to_string(&coordinated[0].operations).unwrap();
    assert!(
        ops_json.contains("changeSquash"),
        "Must include changeSquash"
    );
    assert!(
        !ops_json.contains("setBias"),
        "Should NOT include setBias when signed mean error is near zero, got: {ops_json}"
    );
}

// =============================================================================
// 4. Multiple plateau neurons each get coordinated pairs
// =============================================================================

#[test]
fn test_multiple_plateau_neurons_get_coordinated_pairs() {
    let outputs = vec![
        output_neuron("output-a", "HARD_TANH"),
        output_neuron("output-b", "LOGISTIC"),
    ];

    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        (
            "output-a".to_string(),
            (0..60)
                .map(|i| {
                    let activation = (i as f32 - 30.0) / 40.0;
                    let error = 0.35 + (i as f32 * 0.002).sin() * 0.005;
                    make_record("output-a", i, activation, error)
                })
                .collect(),
        ),
        (
            "output-b".to_string(),
            (0..60)
                .map(|i| {
                    let activation = 0.5 + (i as f32 - 30.0) / 100.0;
                    let error = 0.25 + (i as f32 * 0.001).sin() * 0.008;
                    make_record("output-b", i, activation, error)
                })
                .collect(),
        ),
    ];

    let candidates = detect_error_plateaus(&outputs, &records);
    assert!(
        candidates.len() >= 2,
        "Should detect plateaus for both neurons, got: {}",
        candidates.len()
    );

    let coordinated = error_plateaus_to_coordinated_candidates(&candidates);
    assert!(
        coordinated.len() >= 2,
        "Should produce coordinated candidates for each plateau neuron"
    );

    // Each coordinated candidate should target a different neuron
    let uuids: Vec<String> = candidates.iter().map(|c| c.neuron_uuid.clone()).collect();
    assert!(uuids.contains(&"output-a".to_string()));
    assert!(uuids.contains(&"output-b".to_string()));
}

// =============================================================================
// 5. Healthy networks produce no candidates
// =============================================================================

#[test]
fn test_healthy_network_no_candidates() {
    let outputs = vec![output_neuron("output-1", "TANH")];

    // Very low errors = healthy, converged network
    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "output-1".to_string(),
        (0..60)
            .map(|i| {
                let activation = (i as f32 - 30.0) / 40.0;
                let error = 0.001 + (i as f32 * 0.001).sin() * 0.0005;
                make_record("output-1", i, activation, error)
            })
            .collect(),
    )];

    let candidates = detect_error_plateaus(&outputs, &records);
    assert!(
        candidates.is_empty(),
        "Healthy network should produce no plateau candidates"
    );
}

// =============================================================================
// 6. Estimated improvement accounts for both squash and bias changes
// =============================================================================

#[test]
fn test_error_plateau_structural_estimated_improvement_positive() {
    let outputs = vec![output_neuron("output-1", "HARD_TANH")];

    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "output-1".to_string(),
        (0..60)
            .map(|i| {
                let activation = (i as f32 - 30.0) / 40.0;
                let error = 0.30 + (i as f32 * 0.001).sin() * 0.01;
                make_record("output-1", i, activation, error)
            })
            .collect(),
    )];

    let candidates = detect_error_plateaus(&outputs, &records);
    assert!(!candidates.is_empty());

    let coordinated = error_plateaus_to_coordinated_candidates(&candidates);
    assert!(!coordinated.is_empty());

    assert!(
        coordinated[0].expected_creature_score_gain > 0.0,
        "Expected score gain should be positive for structural escape"
    );
}
