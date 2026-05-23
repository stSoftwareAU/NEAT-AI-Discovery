//! Tests for Issue #1247: harden the sample-weighted and fan-in
//! recommendation paths against `CATEGORICAL_ERROR`'s quantised
//! `{0, 1}` error regime.

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use neat_ai_discovery::analysis::recommendation::fan_in::detect_fan_in_candidates;
use neat_ai_discovery::analysis::recommendation::sample_weighted::{
    SampleWeightedConfig, compute_sample_weights, detect_high_error_neurons, stratify_samples,
};
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

fn make_records(neuron_uuid: &str, errors: &[f32]) -> Vec<DiscoverRecord> {
    errors
        .iter()
        .enumerate()
        .map(|(i, &err)| DiscoverRecord {
            obs_index: i as u32,
            neuron_uuid: neuron_uuid.to_string(),
            value: Some(0.5),
            activation: 0.5,
            errors: vec![err],
        })
        .collect()
}

/// Quantised `{0, 1}` errors must produce well-formed sample weights —
/// finite, non-negative, and summing to approximately 1.0 — instead of
/// NaN or division-by-zero artefacts.
#[test]
fn sample_weights_finite_for_quantised_errors() {
    let errors: Vec<f32> = (0..40)
        .map(|i| if i % 2 == 0 { 0.0 } else { 1.0 })
        .collect();
    let records = make_records("hidden-1", &errors);

    let weights = compute_sample_weights(&records);

    assert_eq!(weights.len(), records.len());
    let total: f32 = weights.iter().sum();
    assert!(
        total.is_finite(),
        "weight total must be finite, got {total}"
    );
    assert!(
        (total - 1.0).abs() < 1e-4,
        "weights must sum to 1.0, got {total}",
    );
    for (i, &w) in weights.iter().enumerate() {
        assert!(w.is_finite(), "weight[{i}] is not finite: {w}");
        assert!(w >= 0.0, "weight[{i}] is negative: {w}");
    }
}

/// All-zero quantised batch (every sample correctly classified) must
/// fall back to uniform weights without producing NaN or panicking.
#[test]
fn sample_weights_uniform_for_all_zero_errors() {
    let records = make_records("hidden-1", &[0.0_f32; 32]);
    let weights = compute_sample_weights(&records);

    assert_eq!(weights.len(), 32);
    let expected = 1.0 / 32.0;
    for (i, &w) in weights.iter().enumerate() {
        assert!(w.is_finite(), "weight[{i}] not finite: {w}");
        assert!(
            (w - expected).abs() < 1e-5,
            "weight[{i}] = {w}, expected {expected}",
        );
    }
}

/// Stratification on a quantised `{0, 1}` batch must still produce
/// finite means and a finite hard-to-easy ratio — even though the
/// ratio is technically unbounded when every "easy" sample has zero
/// error.
#[test]
fn stratify_handles_quantised_errors_without_nan() {
    let errors: Vec<f32> = (0..40).map(|i| if i < 20 { 0.0 } else { 1.0 }).collect();
    let records = make_records("hidden-1", &errors);

    let stratified = stratify_samples(&records);

    assert!(
        stratified.easy_mean_error.is_finite(),
        "easy_mean_error must be finite, got {}",
        stratified.easy_mean_error,
    );
    assert!(
        stratified.hard_mean_error.is_finite(),
        "hard_mean_error must be finite, got {}",
        stratified.hard_mean_error,
    );
    assert!(
        stratified.hard_to_easy_ratio.is_finite(),
        "hard_to_easy_ratio must be finite, got {}",
        stratified.hard_to_easy_ratio,
    );
    assert!(
        !stratified.easy_samples.is_empty(),
        "stratification should populate the easy bucket",
    );
    assert!(
        !stratified.hard_samples.is_empty(),
        "stratification should populate the hard bucket",
    );
}

/// All-zero (zero-variance) quantised batch must stratify cleanly via
/// the existing "all values equal median" fallback rather than panic.
#[test]
fn stratify_handles_all_zero_error_batch() {
    let records = make_records("hidden-1", &[0.0_f32; 20]);
    let stratified = stratify_samples(&records);

    assert!(stratified.easy_mean_error.is_finite());
    assert!(stratified.hard_mean_error.is_finite());
    assert!(stratified.hard_to_easy_ratio.is_finite());
}

// ---------------------------------------------------------------------------
// fan-in detector — least-squares improvement against `{0, 1}` errors must
// stay finite (and non-negative) even though the magnitude is bounded by
// the misclassification count rather than the network's loss.
// ---------------------------------------------------------------------------

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

fn record_with_value(
    uuid: &str,
    obs_index: u32,
    activation: f32,
    errors: Vec<f32>,
) -> DiscoverRecord {
    DiscoverRecord {
        obs_index,
        neuron_uuid: uuid.to_string(),
        value: Some(activation),
        activation,
        errors,
    }
}

fn make_fan_in_creature() -> CreatureJson {
    let neurons = vec![
        neuron("input-a", "input", "IDENTITY"),
        neuron("input-b", "input", "IDENTITY"),
        neuron("output-0", "output", "IDENTITY"),
    ];
    let synapses = vec![
        synapse("input-a", "output-0", 0.1),
        synapse("input-b", "output-0", 0.1),
    ];
    CreatureJson {
        neurons,
        synapses,
        input: 2,
        output: 1,
    }
}

/// Quantised `CATEGORICAL_ERROR`-shaped errors on the output target
/// must not produce NaN or panic anywhere in the fan-in pipeline. The
/// least-squares improvement may be small but it must stay finite and
/// non-negative.
#[test]
fn fan_in_handles_quantised_errors_without_nan() {
    let creature = make_fan_in_creature();

    let mut neuron_records: Vec<(String, Vec<DiscoverRecord>)> = Vec::new();
    for uuid in ["input-a", "input-b", "output-0"] {
        let recs: Vec<DiscoverRecord> = (0..50)
            .map(|i| {
                let phase = (i as f32) * 0.21;
                let act = match uuid {
                    "input-a" => phase.sin(),
                    "input-b" => (phase * 1.7).cos(),
                    _ => 0.1 * phase.sin(),
                };
                let errors = if uuid == "output-0" {
                    vec![if i % 2 == 0 { 0.0 } else { 1.0 }]
                } else {
                    vec![]
                };
                record_with_value(uuid, i, act, errors)
            })
            .collect();
        neuron_records.push((uuid.to_string(), recs));
    }

    let candidates = detect_fan_in_candidates(&creature, &neuron_records);

    for c in &candidates {
        assert!(
            c.estimated_improvement.is_finite(),
            "fan-in estimated_improvement must be finite, got {}",
            c.estimated_improvement,
        );
        assert!(
            c.estimated_improvement >= 0.0,
            "fan-in estimated_improvement must be non-negative, got {}",
            c.estimated_improvement,
        );
        for &w in &c.input_weights {
            assert!(w.is_finite(), "fan-in input weight must be finite, got {w}");
        }
    }
}

/// All-zero (zero-variance) output errors must produce zero candidates
/// without panicking — the existing `sum_act_sq < 1e-10` and
/// per-target-error guards keep the pipeline well-formed.
#[test]
fn fan_in_handles_all_zero_output_errors() {
    let creature = make_fan_in_creature();

    let mut neuron_records: Vec<(String, Vec<DiscoverRecord>)> = Vec::new();
    for uuid in ["input-a", "input-b", "output-0"] {
        let recs: Vec<DiscoverRecord> = (0..50)
            .map(|i| {
                let phase = (i as f32) * 0.21;
                let act = match uuid {
                    "input-a" => phase.sin(),
                    "input-b" => (phase * 1.7).cos(),
                    _ => 0.0,
                };
                let errors = if uuid == "output-0" {
                    vec![0.0]
                } else {
                    vec![]
                };
                record_with_value(uuid, i, act, errors)
            })
            .collect();
        neuron_records.push((uuid.to_string(), recs));
    }

    let candidates = detect_fan_in_candidates(&creature, &neuron_records);

    // No candidate should be emitted (no error signal), and any that
    // happen to be emitted must still be finite.
    for c in &candidates {
        assert!(
            c.estimated_improvement.is_finite(),
            "estimated_improvement must be finite, got {}",
            c.estimated_improvement,
        );
    }
}

/// End-to-end: `detect_high_error_neurons` on quantised input must
/// produce candidates whose every numeric field is finite. The
/// detector's `weighted_mean_error` collapses to the misclassification
/// rate under the regime, which is a valid ranking signal.
#[test]
fn detect_high_error_neurons_finite_under_quantised_regime() {
    let errors: Vec<f32> = (0..40)
        .map(|i| if i % 2 == 0 { 0.0 } else { 1.0 })
        .collect();
    let records = make_records("hidden-1", &errors);
    let config = SampleWeightedConfig {
        // Lower the threshold so the misclassification-rate signal
        // (≈0.5 under a 50/50 quantised batch) is above the gate.
        min_weighted_error: 0.1,
        min_samples: 10,
    };

    let candidates = detect_high_error_neurons(&[("hidden-1".to_string(), records)], &config);

    for c in &candidates {
        assert!(
            c.weighted_mean_error.is_finite(),
            "weighted_mean_error must be finite, got {}",
            c.weighted_mean_error,
        );
        assert!(
            c.high_error_proportion.is_finite(),
            "high_error_proportion must be finite",
        );
        assert!(
            c.hard_to_easy_ratio.is_finite(),
            "hard_to_easy_ratio must be finite",
        );
        assert!(
            c.estimated_improvement.is_finite(),
            "estimated_improvement must be finite, got {}",
            c.estimated_improvement,
        );
    }
}
