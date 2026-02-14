//! Tests for Issue #545 / #546: Output squash mismatch detection.
//!
//! Detects when output neurons use an activation function that doesn't match
//! the target data range, trapping the network in a local minimum.
//!
//! ## TDD Plan
//! 1. Test HARD_TANH output with TANH-shaped targets detects mismatch
//! 2. Test correctly matched output squash produces no detection
//! 3. Test insufficient samples returns empty
//! 4. Test only output neurons are analysed (hidden neurons ignored)
//! 5. Test coordinated candidate conversion produces valid changeSquash
//! 6. Test multiple output neurons with different mismatches
//! 7. Test edge cases (constant output, no pre-activation data)

mod common;

use neat_ai_discovery::analysis::detection::output_squash_mismatch::{
    detect_output_squash_mismatches, output_squash_mismatch_to_coordinated_candidates,
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

fn output_neuron(uuid: &str, squash: &str) -> (String, String, f32) {
    (uuid.to_string(), squash.to_string(), 0.0)
}

// =============================================================================
// 1. Positive detection: HARD_TANH output with TANH-shaped targets
// =============================================================================

#[test]
fn test_hard_tanh_output_with_smooth_tanh_targets_detected() {
    // Simulate an output neuron using HARD_TANH where the target data follows
    // a smooth TANH curve. The pre-activation values span a wide range, but
    // HARD_TANH clips them to [-1, 1] with flat gradients at the extremes.
    // A true TANH would give smooth, continuous output.
    let outputs = vec![output_neuron("output-1", "HARD_TANH")];

    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "output-1".to_string(),
        (0..60)
            .map(|i| {
                let x = (i as f32 - 30.0) / 10.0; // pre-activation: -3.0 to 2.9
                let hard_tanh_out = x.clamp(-1.0, 1.0);
                // Error represents difference from smooth TANH target
                let tanh_target = x.tanh();
                let error = (hard_tanh_out - tanh_target).abs();
                make_record("output-1", i, Some(x), hard_tanh_out, error)
            })
            .collect(),
    )];

    let candidates = detect_output_squash_mismatches(&outputs, &records);

    assert!(
        !candidates.is_empty(),
        "Should detect HARD_TANH mismatch when target data is TANH-shaped"
    );
    assert_eq!(candidates[0].neuron_uuid, "output-1");
    assert_eq!(candidates[0].current_squash, "HARD_TANH");
    assert!(
        candidates[0].recommended_squash == "TANH",
        "Should recommend TANH, got: {}",
        candidates[0].recommended_squash
    );
    assert!(candidates[0].confidence > 0.0);
}

// =============================================================================
// 2. Negative detection: correctly matched squash
// =============================================================================

#[test]
fn test_tanh_output_with_tanh_targets_no_detection() {
    // When the output already uses TANH and the data fits naturally,
    // no mismatch should be detected.
    let outputs = vec![output_neuron("output-1", "TANH")];

    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "output-1".to_string(),
        (0..60)
            .map(|i| {
                let x = (i as f32 - 30.0) / 15.0;
                let activation = x.tanh();
                make_record("output-1", i, Some(x), activation, 0.01)
            })
            .collect(),
    )];

    let candidates = detect_output_squash_mismatches(&outputs, &records);

    assert!(
        candidates.is_empty(),
        "Should NOT detect mismatch when output squash already matches target range"
    );
}

// =============================================================================
// 3. Insufficient samples returns empty
// =============================================================================

#[test]
fn test_insufficient_samples_returns_empty() {
    let outputs = vec![output_neuron("output-1", "HARD_TANH")];

    // Only 5 records — below MIN_DISCOVERY_SAMPLE_COUNT (20)
    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "output-1".to_string(),
        (0..5)
            .map(|i| {
                let x = (i as f32 - 2.5) / 1.0;
                make_record("output-1", i, Some(x), x.clamp(-1.0, 1.0), 0.3)
            })
            .collect(),
    )];

    let candidates = detect_output_squash_mismatches(&outputs, &records);

    assert!(
        candidates.is_empty(),
        "Should return empty when sample count is insufficient"
    );
}

// =============================================================================
// 4. Only output neurons are analysed
// =============================================================================

#[test]
fn test_hidden_neurons_are_ignored() {
    // Even if a hidden neuron has mismatched squash, this module should not flag it
    // (hidden neuron mismatch is handled by activation_mismatch module).
    // We pass output neurons only, so this test verifies the function signature.
    let outputs: Vec<(String, String, f32)> = vec![];

    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![];

    let candidates = detect_output_squash_mismatches(&outputs, &records);
    assert!(
        candidates.is_empty(),
        "Empty output neurons should produce no candidates"
    );
}

// =============================================================================
// 5. Coordinated candidate conversion
// =============================================================================

#[test]
fn test_coordinated_candidate_conversion() {
    let outputs = vec![output_neuron("output-1", "HARD_TANH")];

    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "output-1".to_string(),
        (0..60)
            .map(|i| {
                let x = (i as f32 - 30.0) / 10.0;
                let hard_tanh_out = x.clamp(-1.0, 1.0);
                let tanh_target = x.tanh();
                let error = (hard_tanh_out - tanh_target).abs();
                make_record("output-1", i, Some(x), hard_tanh_out, error)
            })
            .collect(),
    )];

    let candidates = detect_output_squash_mismatches(&outputs, &records);
    assert!(
        !candidates.is_empty(),
        "Precondition: should have candidates"
    );

    let coordinated = output_squash_mismatch_to_coordinated_candidates(&candidates);

    assert!(
        !coordinated.is_empty(),
        "Should produce coordinated candidates"
    );
    assert!(
        coordinated[0].expected_creature_score_gain > 0.0,
        "Expected score gain should be positive"
    );
    assert!(
        coordinated[0].comment.as_ref().unwrap().contains("546"),
        "Comment should reference issue #546"
    );

    let ops_json = serde_json::to_string(&coordinated[0].operations).unwrap();
    assert!(
        ops_json.contains("changeSquash"),
        "Should include changeSquash operation"
    );
}

// =============================================================================
// 6. Multiple output neurons with different mismatches
// =============================================================================

#[test]
fn test_multiple_output_neurons_detected() {
    let outputs = vec![
        output_neuron("output-1", "HARD_TANH"),
        output_neuron("output-2", "IDENTITY"),
    ];

    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        (
            "output-1".to_string(),
            (0..60)
                .map(|i| {
                    let x = (i as f32 - 30.0) / 10.0;
                    let hard_tanh_out = x.clamp(-1.0, 1.0);
                    let error = (hard_tanh_out - x.tanh()).abs();
                    make_record("output-1", i, Some(x), hard_tanh_out, error)
                })
                .collect(),
        ),
        (
            "output-2".to_string(),
            (0..60)
                .map(|i| {
                    // IDENTITY output but targets are bounded [-1, 1] — suggests TANH
                    let x = (i as f32 - 30.0) / 10.0;
                    let activation = x; // IDENTITY: output = input
                    // High error when output exceeds [-1, 1] bounds
                    let error = if x.abs() > 1.0 {
                        (x.abs() - 1.0).min(1.0)
                    } else {
                        0.01
                    };
                    make_record("output-2", i, Some(x), activation, error)
                })
                .collect(),
        ),
    ];

    let candidates = detect_output_squash_mismatches(&outputs, &records);

    // At least one should be detected (HARD_TANH for sure)
    assert!(
        !candidates.is_empty(),
        "Should detect at least one output squash mismatch"
    );
}

// =============================================================================
// 7. Edge case: no pre-activation data
// =============================================================================

#[test]
fn test_no_preactivation_data_still_analyses() {
    // When value (pre-activation) is None, the module should still try to
    // detect mismatches from activation/error patterns alone.
    let outputs = vec![output_neuron("output-1", "HARD_TANH")];

    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "output-1".to_string(),
        (0..60)
            .map(|i| {
                let x = (i as f32 - 30.0) / 10.0;
                let hard_tanh_out = x.clamp(-1.0, 1.0);
                let error = (hard_tanh_out - x.tanh()).abs();
                // No pre-activation value
                make_record("output-1", i, None, hard_tanh_out, error)
            })
            .collect(),
    )];

    let candidates = detect_output_squash_mismatches(&outputs, &records);

    // With HARD_TANH, even without pre-activation values, the clipping pattern
    // in activation values and high errors at bounds should be detectable.
    assert!(
        !candidates.is_empty(),
        "Should detect mismatch even without pre-activation data"
    );
}

// =============================================================================
// 8. LOGISTIC output when targets are in [-1, 1] range
// =============================================================================

#[test]
fn test_logistic_output_with_symmetric_targets() {
    // LOGISTIC outputs [0, 1] but targets are in [-1, 1] → should recommend TANH
    let outputs = vec![output_neuron("output-1", "LOGISTIC")];

    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "output-1".to_string(),
        (0..60)
            .map(|i| {
                let x = (i as f32 - 30.0) / 10.0;
                let activation = 1.0 / (1.0 + (-x).exp()); // logistic
                // Targets are in [-1, 1], so negative targets cause huge errors
                let target = x.tanh();
                let error = (activation - target).abs();
                make_record("output-1", i, Some(x), activation, error)
            })
            .collect(),
    )];

    let candidates = detect_output_squash_mismatches(&outputs, &records);

    assert!(
        !candidates.is_empty(),
        "Should detect LOGISTIC mismatch when targets are symmetric [-1, 1]"
    );
}

// =============================================================================
// 9. Pre-activation comparison: SOFTSIGN output when TANH fits better
// =============================================================================

#[test]
fn test_preactivation_comparison_finds_better_squash() {
    // SOFTSIGN and TANH have the same range [-1, 1] but different curvatures.
    // When the pre-activation values span a moderate range, TANH is often a
    // better fit for smooth target data. The pre-activation comparison strategy
    // should detect this by simulating alternative squash functions.
    let outputs = vec![output_neuron("output-1", "SOFTSIGN")];

    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "output-1".to_string(),
        (0..60)
            .map(|i| {
                let x = (i as f32 - 30.0) / 10.0; // range: -3.0 to 2.9
                // SOFTSIGN: x / (1 + |x|)
                let softsign_out = x / (1.0 + x.abs());
                // Target follows TANH curve
                let tanh_target = x.tanh();
                let error = softsign_out - tanh_target;
                make_record("output-1", i, Some(x), softsign_out, error)
            })
            .collect(),
    )];

    let candidates = detect_output_squash_mismatches(&outputs, &records);

    assert!(
        !candidates.is_empty(),
        "Should detect mismatch via pre-activation comparison"
    );
    assert_eq!(candidates[0].neuron_uuid, "output-1");
    assert_eq!(candidates[0].current_squash, "SOFTSIGN");
    assert_eq!(
        candidates[0].recommended_squash, "TANH",
        "Should recommend TANH when it reduces error vs SOFTSIGN"
    );
}

// =============================================================================
// 10. Pre-activation comparison: no false positive when squash is optimal
// =============================================================================

#[test]
fn test_preactivation_comparison_no_false_positive() {
    // When TANH is already the correct squash and targets follow TANH curve,
    // the pre-activation comparison should not recommend any change.
    let outputs = vec![output_neuron("output-1", "TANH")];

    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "output-1".to_string(),
        (0..60)
            .map(|i| {
                let x = (i as f32 - 30.0) / 15.0;
                let activation = x.tanh();
                // Small residual error (network is converging)
                make_record("output-1", i, Some(x), activation, 0.005)
            })
            .collect(),
    )];

    let candidates = detect_output_squash_mismatches(&outputs, &records);

    assert!(
        candidates.is_empty(),
        "Should NOT detect mismatch when squash already fits target data"
    );
}
