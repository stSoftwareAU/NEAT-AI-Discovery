//! Tests for Issue #545 / #547: Error stagnation plateau detection.
//!
//! Detects when output neuron error distributions show stagnation patterns —
//! consistently high error with low variance — indicating the network is stuck
//! in a local minimum and needs structural changes to escape.
//!
//! ## TDD Plan
//! 1. Test plateau detection with tightly clustered non-zero errors
//! 2. Test no detection for healthy decreasing error patterns
//! 3. Test no detection for low error (already converged)
//! 4. Test insufficient samples returns empty
//! 5. Test coordinated candidate conversion produces valid candidates
//! 6. Test multiple neurons with different plateau characteristics
//! 7. Test high-variance errors are not flagged (noisy, not stagnant)

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

// =============================================================================
// 1. Plateau detection: tightly clustered non-zero errors
// =============================================================================

#[test]
fn test_stagnant_error_plateau_detected() {
    // Output neuron with errors that are consistently around 0.3
    // (high error, very low variance = plateau)
    let outputs = vec![output_neuron("output-1", "HARD_TANH")];

    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "output-1".to_string(),
        (0..60)
            .map(|i| {
                let activation = (i as f32 - 30.0) / 40.0; // mild variation
                // Error hovers around 0.3 with very small noise
                let error = 0.30 + (i as f32 * 0.001).sin() * 0.01;
                make_record("output-1", i, activation, error)
            })
            .collect(),
    )];

    let candidates = detect_error_plateaus(&outputs, &records);

    assert!(
        !candidates.is_empty(),
        "Should detect error plateau: consistently high error with low variance"
    );
    assert_eq!(candidates[0].neuron_uuid, "output-1");
    assert!(
        candidates[0].mean_error > 0.1,
        "Mean error should be non-trivial, got: {}",
        candidates[0].mean_error
    );
    assert!(
        candidates[0].error_coefficient_of_variation < 0.5,
        "Coefficient of variation should be low (tight clustering), got: {}",
        candidates[0].error_coefficient_of_variation
    );
}

// =============================================================================
// 2. No detection for healthy/improving patterns
// =============================================================================

#[test]
fn test_no_detection_for_low_error() {
    // Output neuron with very low errors (already converged)
    let outputs = vec![output_neuron("output-1", "TANH")];

    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "output-1".to_string(),
        (0..60)
            .map(|i| {
                let activation = (i as f32 - 30.0) / 40.0;
                let error = 0.005 + (i as f32 * 0.001).sin() * 0.002;
                make_record("output-1", i, activation, error)
            })
            .collect(),
    )];

    let candidates = detect_error_plateaus(&outputs, &records);

    assert!(
        candidates.is_empty(),
        "Should NOT detect plateau when error is already very low"
    );
}

// =============================================================================
// 3. Insufficient samples returns empty
// =============================================================================

#[test]
fn test_error_plateau_insufficient_samples_returns_empty() {
    let outputs = vec![output_neuron("output-1", "HARD_TANH")];

    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "output-1".to_string(),
        (0..5)
            .map(|i| make_record("output-1", i, 0.5, 0.3))
            .collect(),
    )];

    let candidates = detect_error_plateaus(&outputs, &records);

    assert!(
        candidates.is_empty(),
        "Should return empty when sample count is insufficient"
    );
}

// =============================================================================
// 4. Coordinated candidate conversion
// =============================================================================

#[test]
fn test_error_plateau_coordinated_candidate_conversion() {
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
    assert!(
        !candidates.is_empty(),
        "Precondition: should have candidates"
    );

    let coordinated = error_plateaus_to_coordinated_candidates(&candidates);

    assert!(
        !coordinated.is_empty(),
        "Should produce coordinated candidates"
    );
    assert!(
        coordinated[0].expected_creature_score_gain > 0.0,
        "Expected score gain should be positive"
    );
    assert!(
        coordinated[0].comment.as_ref().unwrap().contains("547"),
        "Comment should reference issue #547"
    );

    let ops_json = serde_json::to_string(&coordinated[0].operations).unwrap();
    assert!(
        ops_json.contains("changeSquash") || ops_json.contains("setBias"),
        "Should include changeSquash or setBias operation, got: {ops_json}"
    );
}

// =============================================================================
// 5. Multiple neurons with different plateau characteristics
// =============================================================================

#[test]
fn test_multiple_neurons_mixed_results() {
    let outputs = vec![
        output_neuron("output-plateau", "HARD_TANH"),
        output_neuron("output-healthy", "TANH"),
    ];

    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        (
            "output-plateau".to_string(),
            (0..60)
                .map(|i| {
                    let activation = (i as f32 - 30.0) / 40.0;
                    let error = 0.35 + (i as f32 * 0.002).sin() * 0.005;
                    make_record("output-plateau", i, activation, error)
                })
                .collect(),
        ),
        (
            "output-healthy".to_string(),
            (0..60)
                .map(|i| {
                    let activation = (i as f32 - 30.0) / 40.0;
                    let error = 0.002; // very low error
                    make_record("output-healthy", i, activation, error)
                })
                .collect(),
        ),
    ];

    let candidates = detect_error_plateaus(&outputs, &records);

    // Should detect the plateau neuron but not the healthy one
    assert!(!candidates.is_empty(), "Should detect the plateau neuron");
    assert!(
        candidates.iter().all(|c| c.neuron_uuid == "output-plateau"),
        "Should only flag the plateau neuron, not the healthy one"
    );
}

// =============================================================================
// 6. High-variance errors are not flagged (noisy, not stagnant)
// =============================================================================

#[test]
fn test_high_variance_errors_not_detected() {
    // High error but high variance = the error is changing (not a plateau)
    let outputs = vec![output_neuron("output-1", "HARD_TANH")];

    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "output-1".to_string(),
        (0..60)
            .map(|i| {
                let activation = (i as f32 - 30.0) / 40.0;
                // Wildly varying errors (not a plateau pattern)
                let error = if i % 2 == 0 { 0.8 } else { 0.02 };
                make_record("output-1", i, activation, error)
            })
            .collect(),
    )];

    let candidates = detect_error_plateaus(&outputs, &records);

    assert!(
        candidates.is_empty(),
        "Should NOT detect plateau when error variance is high (noisy, not stagnant)"
    );
}

// =============================================================================
// 7. Empty inputs
// =============================================================================

#[test]
fn test_empty_outputs_returns_empty() {
    let outputs: Vec<(String, String, f32)> = vec![];
    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![];

    let candidates = detect_error_plateaus(&outputs, &records);
    assert!(
        candidates.is_empty(),
        "Empty inputs should produce no candidates"
    );
}
