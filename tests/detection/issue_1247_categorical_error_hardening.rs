//! Tests for Issue #1247: harden discovery detection for `CATEGORICAL_ERROR`'s
//! quantised `{0, 1}` output errors.
//!
//! Every module flagged by the #1245 audit as sensitive to the per-record
//! error distribution must:
//! - never return `NaN` or panic on the quantised `{0, 1}` regime,
//! - never divide by zero on a zero-variance batch (all `0` or all `1`),
//! - either skip cleanly with a diagnostic or produce well-formed output.
//!
//! These tests exercise the public API of each module directly with
//! quantised fixtures so the contract is enforced even if the internal
//! statistical helpers are refactored.

#![allow(clippy::cast_precision_loss)]

use neat_ai_discovery::analysis::detection::bimodal_neuron::detect_bimodal_neurons;
use neat_ai_discovery::analysis::detection::monotonicity::detect_non_monotonic_neurons;
use neat_ai_discovery::analysis::quantised_error::is_quantised_zero_one;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

fn neuron(uuid: &str, neuron_type: &str, squash: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
        squash: squash.to_string(),
        bias: 0.0,
    }
}

fn synapse(from: &str, to: &str, weight: f32) -> SynapseJson {
    SynapseJson {
        from_uuid: from.to_string(),
        to_uuid: to.to_string(),
        weight,
        synapse_type: None,
    }
}

fn make_creature(neurons: Vec<NeuronJson>, synapses: Vec<SynapseJson>) -> CreatureJson {
    let input_count = neurons.iter().filter(|n| n.neuron_type == "input").count();
    let output_count = neurons.iter().filter(|n| n.neuron_type == "output").count();
    CreatureJson {
        neurons,
        synapses,
        input: input_count,
        output: output_count,
    }
}

fn record(uuid: &str, obs_index: u32, activation: f32, errors: Vec<f32>) -> DiscoverRecord {
    DiscoverRecord {
        obs_index,
        neuron_uuid: uuid.to_string(),
        value: Some(activation),
        activation,
        errors,
    }
}

// ---------------------------------------------------------------------------
// Monotonicity detector: must skip the quantised regime cleanly.
// ---------------------------------------------------------------------------

/// Hidden neuron whose recorded error vector is the `CATEGORICAL_ERROR`
/// `{0, 1}` flag should be skipped by the monotonicity detector — rank
/// correlation on two tied groups is meaningless and would otherwise
/// mass-flag every `CATEGORICAL_ERROR`-driven hidden neuron.
#[test]
fn monotonicity_skips_quantised_zero_one_regime() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("hidden-1", "hidden", "LOGISTIC"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "hidden-1", 0.5),
            synapse("hidden-1", "output-1", 0.8),
        ],
    );

    // Continuous activation paired with a quantised `{0, 1}` error
    // series. About half the samples are misclassified.
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = (i as f32) / 100.0;
            let err = if i % 2 == 0 { 0.0 } else { 1.0 };
            record("hidden-1", i, activation, vec![err])
        })
        .collect();

    let candidates = detect_non_monotonic_neurons(&creature, &[("hidden-1".to_string(), records)]);

    assert!(
        candidates.is_empty(),
        "Quantised {{0,1}} errors must skip monotonicity detection, got {} candidate(s)",
        candidates.len(),
    );
}

/// Even when every recorded error is `0` (or every error is `1`), the
/// monotonicity detector must not panic or emit a NaN-bearing candidate.
/// All-zero batches sit below the quantised regime threshold (no mass
/// at both modes), so they fall through to the Spearman computation —
/// which is already guarded against zero variance and returns `0.0`,
/// causing the detector to *not* flag the neuron.
#[test]
fn monotonicity_handles_zero_variance_error_batch_without_nan() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("hidden-1", "hidden", "LOGISTIC"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "hidden-1", 0.5),
            synapse("hidden-1", "output-1", 0.8),
        ],
    );

    let records: Vec<DiscoverRecord> = (0..50)
        .map(|i| record("hidden-1", i, (i as f32) / 50.0, vec![0.0]))
        .collect();

    let candidates = detect_non_monotonic_neurons(&creature, &[("hidden-1".to_string(), records)]);

    for c in &candidates {
        assert!(
            c.monotonicity_score.is_finite(),
            "monotonicity_score must be finite, got {}",
            c.monotonicity_score,
        );
        assert!(
            c.estimated_improvement.is_finite(),
            "estimated_improvement must be finite, got {}",
            c.estimated_improvement,
        );
    }
}

/// Hidden neuron with a *continuous* error series must still be
/// processed by the monotonicity detector — the quantised-regime guard
/// must not regress the existing non-monotonic detection.
#[test]
fn monotonicity_still_flags_continuous_non_monotonic_neuron() {
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("hidden-1", "hidden", "LOGISTIC"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "hidden-1", 0.5),
            synapse("hidden-1", "output-1", 0.8),
        ],
    );

    // V-shaped error curve: error is high at activation = 0 and
    // activation = 1, low in the middle. Spearman rho ≈ 0.
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let act = (i as f32) / 100.0;
            let err = (act - 0.5).abs() * 2.0;
            record("hidden-1", i, act, vec![err])
        })
        .collect();

    let candidates = detect_non_monotonic_neurons(&creature, &[("hidden-1".to_string(), records)]);

    assert!(
        !candidates.is_empty(),
        "Continuous non-monotonic neuron should still be flagged (regression guard)",
    );
}

// ---------------------------------------------------------------------------
// Bimodal neuron detector: operates on `value`, not errors. Confirm that
// the pre-activation `{0, 1}` case is handled cleanly.
// ---------------------------------------------------------------------------

/// A neuron whose pre-activation `value` is itself quantised `{0, 1}`
/// is genuinely bimodal — the detector should either flag it cleanly
/// or skip it cleanly, but never NaN or panic.
#[test]
fn bimodal_neuron_handles_quantised_pre_activation_without_nan() {
    let neurons = vec![("bimodal-q".to_string(), "TANH".to_string(), 0.0)];

    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let v = if i % 2 == 0 { 0.0 } else { 1.0 };
            DiscoverRecord {
                obs_index: i,
                neuron_uuid: "bimodal-q".to_string(),
                value: Some(v),
                activation: v.tanh(),
                errors: vec![],
            }
        })
        .collect();

    let candidates = detect_bimodal_neurons(&neurons, &[("bimodal-q".to_string(), records)]);

    for c in &candidates {
        assert!(
            c.bimodality_score.is_finite(),
            "bimodality_score must be finite for quantised pre-activation, got {}",
            c.bimodality_score,
        );
        assert!(
            c.lower_mode_mean.is_finite(),
            "lower_mode_mean must be finite",
        );
        assert!(
            c.upper_mode_mean.is_finite(),
            "upper_mode_mean must be finite",
        );
        assert!(
            c.estimated_improvement.is_finite(),
            "estimated_improvement must be finite, got {}",
            c.estimated_improvement,
        );
    }
}

/// All pre-activations identical → no bimodality, no NaN, no panic.
#[test]
fn bimodal_neuron_handles_constant_value_without_nan() {
    let neurons = vec![("constant".to_string(), "TANH".to_string(), 0.0)];

    let records: Vec<DiscoverRecord> = (0..40)
        .map(|i| DiscoverRecord {
            obs_index: i,
            neuron_uuid: "constant".to_string(),
            value: Some(0.5),
            activation: 0.5_f32.tanh(),
            errors: vec![],
        })
        .collect();

    let candidates = detect_bimodal_neurons(&neurons, &[("constant".to_string(), records)]);

    // Should not flag, but if it does, the values must still be finite.
    for c in &candidates {
        assert!(c.bimodality_score.is_finite());
        assert!(c.estimated_improvement.is_finite());
    }
}

// ---------------------------------------------------------------------------
// quantised_error helper used to drive the gating.
// ---------------------------------------------------------------------------

#[test]
fn quantised_helper_recognises_balanced_categorical_error_batch() {
    let errors: Vec<f32> = (0..40)
        .map(|i| if i % 2 == 0 { 0.0 } else { 1.0 })
        .collect();
    assert!(is_quantised_zero_one(&errors));
}

#[test]
fn quantised_helper_rejects_continuous_residuals() {
    let errors: Vec<f32> = (0..40).map(|i| (i as f32) * 0.01).collect();
    assert!(!is_quantised_zero_one(&errors));
}
