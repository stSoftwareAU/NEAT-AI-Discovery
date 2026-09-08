//! Tests for Issue #2043: batch-successful detection must use the canonical
//! least-squares weight function.
//!
//! `batch_successful::detection::evaluate_individual` used to reimplement
//! `w = Σ(error × activation) / Σ(activation²)` inline, skipping the validity
//! checks that `calculate_optimal_outgoing_weight` applies (the `EPSILON`
//! activation-energy floor and the `MAX_OUTGOING_WEIGHT` clamp). These tests
//! pin the canonical behaviour on the detection path.

use neat_ai_discovery::analysis::recommendation::batch_successful::detect_individually_successful;
use neat_ai_discovery::analysis::scoring::weights::MAX_OUTGOING_WEIGHT;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

/// Sample count comfortably above `MIN_DISCOVERY_SAMPLE_COUNT` (20).
const SAMPLES: u32 = 40;

fn record(neuron_uuid: &str, obs_index: u32, activation: f32, errors: Vec<f32>) -> DiscoverRecord {
    DiscoverRecord {
        obs_index,
        neuron_uuid: neuron_uuid.to_string(),
        value: Some(activation),
        activation,
        errors,
    }
}

fn neuron(uuid: &str, neuron_type: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
        squash: "IDENTITY".to_string(),
        bias: 0.0,
    }
}

/// One input → one output, with `errors[i] = slope × activations[i]`, so the
/// unclamped least-squares weight is exactly `slope`.
fn creature_with_linear_error(
    activation_for: impl Fn(u32) -> f32,
    slope: f32,
) -> (CreatureJson, Vec<(String, Vec<DiscoverRecord>)>) {
    let creature = CreatureJson {
        neurons: vec![neuron("input-a", "input"), neuron("output-1", "output")],
        synapses: Vec::<SynapseJson>::new(),
        input: 1,
        output: 1,
    };

    let records = vec![
        (
            "input-a".to_string(),
            (0..SAMPLES)
                .map(|i| record("input-a", i, activation_for(i), vec![]))
                .collect(),
        ),
        (
            "output-1".to_string(),
            (0..SAMPLES)
                .map(|i| record("output-1", i, 0.5, vec![slope * activation_for(i)]))
                .collect(),
        ),
    ];

    (creature, records)
}

/// Activations in the ordinary range, alternating so the source is not constant.
fn ordinary_activation(i: u32) -> f32 {
    if i.is_multiple_of(2) { 0.9 } else { 0.1 }
}

/// Activations tiny enough that `Σ activation²` (≈ 4e-9) sits below the
/// canonical `EPSILON` floor (1e-8) but above the old inline `1e-10` floor.
fn degenerate_activation(i: u32) -> f32 {
    1e-5 * (1.0 + f32::from(u8::try_from(i % 5).unwrap_or(0)) * 0.1)
}

#[test]
fn emitted_weight_is_clamped_to_max_outgoing_weight() {
    // Unclamped least-squares weight is 0.5 — 50× the ceiling.
    let (creature, records) = creature_with_linear_error(ordinary_activation, 0.5);
    let candidates = detect_individually_successful(&creature, &records);

    assert!(
        !candidates.is_empty(),
        "A strongly correlated source should still be detected"
    );
    for candidate in &candidates {
        assert!(
            candidate.weight.abs() <= MAX_OUTGOING_WEIGHT,
            "Weight {} exceeds MAX_OUTGOING_WEIGHT {MAX_OUTGOING_WEIGHT}",
            candidate.weight
        );
    }
}

#[test]
fn negative_correlation_weight_is_clamped_to_negative_ceiling() {
    let (creature, records) = creature_with_linear_error(ordinary_activation, -0.5);
    let candidates = detect_individually_successful(&creature, &records);

    assert!(
        !candidates.is_empty(),
        "A strongly (negatively) correlated source should still be detected"
    );
    for candidate in &candidates {
        assert!(
            candidate.weight < 0.0 && candidate.weight.abs() <= MAX_OUTGOING_WEIGHT,
            "Weight {} should be negative and within MAX_OUTGOING_WEIGHT",
            candidate.weight
        );
    }
}

#[test]
fn degenerate_activation_energy_is_rejected_at_the_canonical_epsilon() {
    // Σ activation² ≈ 4e-9: rejected by the canonical EPSILON floor (1e-8),
    // accepted by the old inline 1e-10 floor despite a perfect fit.
    let (creature, records) = creature_with_linear_error(degenerate_activation, 0.5);
    let candidates = detect_individually_successful(&creature, &records);

    assert!(
        candidates.is_empty(),
        "Sources below the canonical activation-energy floor must be rejected, got {candidates:?}"
    );
}

#[test]
fn improvement_is_computed_from_the_emitted_weight() {
    let (creature, records) = creature_with_linear_error(ordinary_activation, 0.5);
    let candidates = detect_individually_successful(&creature, &records);
    let candidate = candidates
        .first()
        .expect("A strongly correlated source should be detected");

    let original_sse: f64 = (0..SAMPLES)
        .map(|i| {
            let e = f64::from(0.5 * ordinary_activation(i));
            e * e
        })
        .sum();
    let residual_sse: f64 = (0..SAMPLES)
        .map(|i| {
            let a = f64::from(ordinary_activation(i));
            let residual =
                f64::from(0.5 * ordinary_activation(i)) - f64::from(candidate.weight) * a;
            residual * residual
        })
        .sum();
    let expected = 1.0 - residual_sse / original_sse;

    assert!(
        (f64::from(candidate.improvement) - expected).abs() < 1e-6,
        "improvement {} should match the emitted weight's error reduction {expected}",
        candidate.improvement
    );
}
