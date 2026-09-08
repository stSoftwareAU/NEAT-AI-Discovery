//! Tests for Issue #2044: shared error coefficient-of-variation assessment.
//!
//! The mean → variance → standard-deviation → coefficient-of-variation chain
//! (and the `plateau_tightness` score derived from it) was copy-pasted across
//! three detectors, with a subtly different numeric guard in the third copy.
//! These tests pin the shared rule in
//! `analysis::detection::error_dispersion` and check the detectors that
//! consume it report exactly what the shared helper computes.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use neat_ai_discovery::analysis::detection::error_dispersion::{
    MIN_CV_DENOMINATOR, assess_error_plateau, error_dispersion,
};
use neat_ai_discovery::analysis::detection::error_plateau::detect_error_plateaus;
use neat_ai_discovery::analysis::detection::weight_magnitude_reset::detect_stuck_synapse_weight_resets;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

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

// =============================================================================
// 1. Business floor — the mean must reach the caller's threshold
// =============================================================================

#[test]
fn empty_errors_have_no_dispersion() {
    assert!(error_dispersion(&[], 0.0).is_none());
}

#[test]
fn mean_below_business_floor_is_rejected() {
    let errors = vec![0.02_f32; 20];
    assert!(
        error_dispersion(&errors, 0.05).is_none(),
        "mean 0.02 is below the 0.05 floor"
    );
}

#[test]
fn mean_on_the_business_floor_is_accepted() {
    let errors = vec![0.05_f32; 20];
    let dispersion = error_dispersion(&errors, 0.05).expect("mean 0.05 meets the 0.05 floor");
    assert!((dispersion.mean_error - 0.05).abs() < 1e-6);
}

// =============================================================================
// 2. The maths — mean, standard deviation, coefficient of variation
// =============================================================================

#[test]
fn dispersion_matches_hand_computed_statistics() {
    let errors = vec![0.2_f32, 0.4, 0.6, 0.8];
    let dispersion = error_dispersion(&errors, 0.0).expect("non-empty errors above a zero floor");

    // mean = 0.5, population variance = 0.05, std_dev = sqrt(0.05)
    assert!((dispersion.mean_error - 0.5).abs() < 1e-6);
    assert!((dispersion.std_dev - 0.05_f32.sqrt()).abs() < 1e-6);
    assert!((dispersion.cv - 0.05_f32.sqrt() / 0.5).abs() < 1e-6);
}

#[test]
fn constant_errors_have_zero_coefficient_of_variation() {
    let dispersion =
        error_dispersion(&[0.3_f32; 30], 0.1).expect("constant errors above the floor");
    assert!(
        dispersion.cv < 1e-6,
        "constant errors barely vary, got {}",
        dispersion.cv
    );
    assert!((dispersion.plateau_tightness(0.3) - 1.0).abs() < 1e-5);
}

// =============================================================================
// 3. Numerical floor — separate from the business floor
// =============================================================================

#[test]
fn near_zero_mean_yields_infinite_coefficient_of_variation() {
    // Passes a zero business floor, so only the numerical floor can guard the
    // division. A mean at or below MIN_CV_DENOMINATOR must not masquerade as a
    // tight cluster.
    let errors = vec![MIN_CV_DENOMINATOR / 2.0, MIN_CV_DENOMINATOR / 2.0];
    let dispersion = error_dispersion(&errors, 0.0).expect("non-empty errors above a zero floor");
    assert_eq!(
        dispersion.cv,
        f32::INFINITY,
        "a near-zero mean must not produce a finite CV"
    );
    assert_eq!(dispersion.plateau_tightness(0.3), 0.0);
    assert!(
        assess_error_plateau(&errors, 0.0, 0.3).is_none(),
        "an infinite CV can never be a plateau"
    );
}

// =============================================================================
// 4. Tightness ceiling
// =============================================================================

#[test]
fn coefficient_of_variation_on_the_ceiling_is_still_a_plateau() {
    // mean 0.5, std_dev sqrt(0.05) → cv ≈ 0.4472
    let errors = vec![0.2_f32, 0.4, 0.6, 0.8];
    let cv = error_dispersion(&errors, 0.0).expect("dispersion").cv;

    assert!(
        assess_error_plateau(&errors, 0.0, cv).is_some(),
        "cv == ceiling is inside the plateau"
    );
    assert!(
        assess_error_plateau(&errors, 0.0, cv * 0.99).is_none(),
        "cv above the ceiling is not a plateau"
    );
}

#[test]
fn plateau_tightness_falls_to_zero_at_the_ceiling() {
    let dispersion = error_dispersion(&[0.2_f32, 0.4, 0.6, 0.8], 0.0).expect("dispersion");
    let cv = dispersion.cv;

    assert!((dispersion.plateau_tightness(cv * 2.0) - 0.5).abs() < 1e-6);
    assert_eq!(dispersion.plateau_tightness(cv), 0.0);
    assert_eq!(dispersion.plateau_tightness(cv / 2.0), 0.0);
    assert_eq!(
        dispersion.plateau_tightness(0.0),
        0.0,
        "a non-positive ceiling must not divide by zero"
    );
}

// =============================================================================
// 5. The detectors report exactly what the shared helper computes
// =============================================================================

#[test]
fn error_plateau_detector_reports_shared_statistics() {
    let outputs = vec![("output-1".to_string(), "HARD_TANH".to_string(), 0.0_f32)];
    let errors: Vec<f32> = (0..60)
        .map(|i| 0.30 + (i as f32 * 0.001).sin() * 0.01)
        .collect();

    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "output-1".to_string(),
        errors
            .iter()
            .enumerate()
            .map(|(i, &e)| {
                let idx = u32::try_from(i).expect("index fits in u32");
                make_record("output-1", idx, (i as f32 - 30.0) / 40.0, e)
            })
            .collect(),
    )];

    let candidates = detect_error_plateaus(&outputs, &records);
    assert_eq!(
        candidates.len(),
        1,
        "tightly clustered high error is a plateau"
    );

    // MIN_PLATEAU_ERROR = 0.05, MAX_COEFFICIENT_OF_VARIATION = 0.3
    let shared = assess_error_plateau(&errors, 0.05, 0.3).expect("shared helper agrees");
    assert!((candidates[0].mean_error - shared.mean_error).abs() < 1e-6);
    assert!((candidates[0].error_std_dev - shared.std_dev).abs() < 1e-6);
    assert!((candidates[0].error_coefficient_of_variation - shared.cv).abs() < 1e-6);
}

#[test]
fn weight_magnitude_reset_detector_reports_shared_statistics() {
    let creature = CreatureJson {
        neurons: vec![
            NeuronJson {
                uuid: "hidden-1".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "TANH".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "output-1".to_string(),
                neuron_type: "output".to_string(),
                squash: "TANH".to_string(),
                bias: 0.0,
            },
        ],
        synapses: vec![SynapseJson {
            from_uuid: "hidden-1".to_string(),
            to_uuid: "output-1".to_string(),
            weight: 0.5,
            synapse_type: None,
        }],
        input: 1,
        output: 1,
    };

    let target_errors: Vec<f32> = (0..60)
        .map(|i| 0.40 + (i as f32 * 0.01).sin() * 0.02)
        .collect();
    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![
        (
            "hidden-1".to_string(),
            (0..60)
                .map(|i| make_record("hidden-1", i, 0.6, 0.1))
                .collect(),
        ),
        (
            "output-1".to_string(),
            target_errors
                .iter()
                .enumerate()
                .map(|(i, &e)| {
                    make_record(
                        "output-1",
                        u32::try_from(i).expect("index fits in u32"),
                        0.2,
                        e,
                    )
                })
                .collect(),
        ),
    ];

    let candidates = detect_stuck_synapse_weight_resets(&creature, &records);
    assert_eq!(
        candidates.len(),
        1,
        "a stuck synapse feeding a plateaued target"
    );

    // MIN_STUCK_ERROR = 0.1, MAX_ERROR_CV = 0.4
    let shared = assess_error_plateau(&target_errors, 0.1, 0.4).expect("shared helper agrees");
    assert!((candidates[0].target_mean_error - shared.mean_error).abs() < 1e-6);
    assert!((candidates[0].target_error_cv - shared.cv).abs() < 1e-6);
}
