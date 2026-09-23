//! Issue #2184 — activation recommendation ranking must be deterministic, and
//! the RELU6 gradient-flow penalty must actually apply.
//!
//! Two defects in `src/analysis/recommendation/activation_recommendation.rs`:
//!
//! 1. `recommend_activation_function` picked the best activation with
//!    `HashMap::iter().max_by(score)`. `HashMap` iteration order is randomised
//!    per map, so a tie on the maximum score resolved differently from call to
//!    call for byte-identical input.
//! 2. `apply_gradient_flow_penalty` looked the RELU6 score up as `"ReLU6"`.
//!    Squash names are normalised to uppercase at deserialisation (Issue #753),
//!    so that key can never exist and the penalty branch was dead.

#![allow(clippy::cast_precision_loss)] // Issue #873

use std::collections::BTreeSet;

use neat_ai_discovery::analysis::recommendation::activation_recommendation::{
    InputDistribution, InputDistributionClass, analyse_input_distribution,
    classify_activation_suitability, recommend_activation_function,
};
use neat_ai_discovery::types::DiscoverRecord;

/// The unpenalised sparse-distribution suitability scores the source assigns.
const SPARSE_RELU_SCORE: f32 = 0.9;
const SPARSE_RELU6_SCORE: f32 = 0.85;

fn record(obs_index: u32, activation: f32) -> DiscoverRecord {
    DiscoverRecord {
        obs_index,
        neuron_uuid: "neuron-2184".to_string(),
        value: Some(activation),
        activation,
        errors: vec![0.01],
    }
}

/// Two tight clusters at ±2.0 — bimodal, and far enough from `[-1.1, 1.1]`
/// that the bounded branch cannot claim it.
fn bimodal_records() -> Vec<DiscoverRecord> {
    (0..100)
        .map(|i| record(i, if i % 2 == 0 { 2.0 } else { -2.0 }))
        .collect()
}

#[test]
fn bimodal_recommendation_does_not_vary_between_calls() {
    let records = bimodal_records();

    // Preconditions: the tie this test exercises must genuinely be present,
    // otherwise the assertion below passes vacuously (Issue #1799).
    let distribution = analyse_input_distribution(&records);
    assert_eq!(
        distribution.class,
        InputDistributionClass::Bimodal,
        "precondition: the record set must classify as bimodal"
    );
    let suitability = classify_activation_suitability(&distribution);
    let tanh = suitability
        .get("TANH")
        .copied()
        .expect("precondition: TANH is scored for a bimodal distribution");
    let hard_tanh = suitability
        .get("HARD_TANH")
        .copied()
        .expect("precondition: HARD_TANH is scored for a bimodal distribution");
    assert!(
        (tanh - hard_tanh).abs() < f32::EPSILON,
        "precondition: TANH ({tanh}) and HARD_TANH ({hard_tanh}) must tie"
    );
    let best = suitability
        .values()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(
        (tanh - best).abs() < f32::EPSILON,
        "precondition: the tie must be on the maximum score ({best})"
    );

    // Each call builds a fresh `HashMap`, so each call gets a fresh iteration
    // order. Before the fix the winner flipped between TANH and HARD_TANH.
    let mut recommended: BTreeSet<String> = BTreeSet::new();
    for _ in 0..200 {
        let recommendation = recommend_activation_function(&records, "RELU")
            .expect("precondition: a recommendation is emitted for this record set");
        recommended.insert(recommendation.recommended_squash);
    }

    assert_eq!(
        recommended.len(),
        1,
        "recommended_squash must be identical across calls for identical input, saw {recommended:?}"
    );
    // Ties are broken on the activation name, so the winner is pinned.
    assert!(
        recommended.contains("TANH"),
        "the tie must resolve to the documented winner, saw {recommended:?}"
    );
}

#[test]
fn relu6_is_penalised_when_inputs_are_negative_heavy() {
    // Sparse, negative-heavy: half the observed range sits below zero, which is
    // exactly the case the RELU-family penalty exists to discourage.
    let distribution = InputDistribution {
        class: InputDistributionClass::Sparse,
        mean: -0.4,
        std_dev: 1.2,
        min: -3.0,
        max: 3.0,
        sparsity: 0.7,
        kurtosis: 4.0,
    };

    let scores = classify_activation_suitability(&distribution);

    let relu = scores
        .get("RELU")
        .copied()
        .expect("precondition: RELU is scored for a sparse distribution");
    let relu6 = scores
        .get("RELU6")
        .copied()
        .expect("precondition: RELU6 is scored for a sparse distribution");
    assert!(
        relu < SPARSE_RELU_SCORE,
        "precondition: RELU must be penalised for this distribution, got {relu}"
    );
    assert!(
        !scores.contains_key("ReLU6"),
        "precondition: suitability keys are uppercase (Issue #753)"
    );

    assert!(
        relu6 < SPARSE_RELU6_SCORE,
        "RELU6 must be discounted, not left at its unpenalised {SPARSE_RELU6_SCORE}, got {relu6}"
    );
    // Same negative fraction, so RELU and RELU6 take the same discount factor.
    let expected = SPARSE_RELU6_SCORE * (relu / SPARSE_RELU_SCORE);
    assert!(
        (relu6 - expected).abs() < 1e-5,
        "RELU6 must take the same gradient-flow discount as RELU: expected {expected}, got {relu6}"
    );
}
