//! Issue #2351 (CWE-682) sibling sites, batch B.
//!
//! NaN / infinite recorded data (the record loader does not filter it) or f32
//! overflow produces NaN / infinite intermediates that pass comparison gates
//! (NaN comparisons are false, so the gates fail open). These tests drive each
//! public detector with such input and assert that no emitted candidate carries
//! a non-finite gain, bias or weight.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)] // Tiny test indices.

use neat_ai_discovery::analysis::detection::bimodal_neuron::{
    bimodal_neurons_to_coordinated_candidates, detect_bimodal_neurons,
};
use neat_ai_discovery::analysis::detection::dormant_synapse::{
    detect_dormant_synapses, dormant_synapses_to_coordinated_candidates,
};
use neat_ai_discovery::analysis::detection::high_error_squash_exploration::{
    detect_high_error_squash_candidates, high_error_squash_to_coordinated_candidates,
};
use neat_ai_discovery::analysis::detection::noise_signal::{
    detect_noisy_synapses, noisy_synapses_to_coordinated_candidates,
};
use neat_ai_discovery::analysis::detection::oscillating_neuron::{
    detect_oscillating_neurons, oscillating_neurons_to_coordinated_candidates,
};
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{
    CoordinatedStructuralCandidateJson, CreatureJson, NeuronJson, SynapseJson,
};

fn rec(uuid: &str, obs: u32, value: Option<f32>, activation: f32, error: f32) -> DiscoverRecord {
    DiscoverRecord {
        obs_index: obs,
        neuron_uuid: uuid.to_string(),
        value,
        activation,
        errors: vec![error],
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

fn synapse(from: &str, to: &str, weight: f32) -> SynapseJson {
    SynapseJson {
        from_uuid: from.to_string(),
        to_uuid: to.to_string(),
        weight,
        synapse_type: None,
    }
}

fn creature(synapses: Vec<SynapseJson>) -> CreatureJson {
    CreatureJson {
        neurons: vec![
            neuron("input-1", "input"),
            neuron("input-2", "input"),
            neuron("output-1", "output"),
        ],
        synapses,
        input: 2,
        output: 1,
    }
}

fn assert_gains_finite(candidates: &[CoordinatedStructuralCandidateJson]) {
    for c in candidates {
        assert!(
            c.expected_creature_score_gain.is_finite(),
            "non-finite gain emitted: {}",
            c.expected_creature_score_gain
        );
    }
}

#[test]
fn bimodal_non_finite_value_does_not_emit_non_finite_gain_or_bias() {
    let mut values = vec![f32::NEG_INFINITY];
    values.extend(std::iter::repeat_n(0.0_f32, 9));
    values.extend(std::iter::repeat_n(10.0_f32, 10));
    let records: Vec<DiscoverRecord> = values
        .iter()
        .enumerate()
        .map(|(i, &v)| rec("h", i as u32, Some(v), 0.0, 0.0))
        .collect();
    let neurons = vec![("h".to_string(), "TANH".to_string(), 0.0_f32)];
    let found = detect_bimodal_neurons(&neurons, &[("h".to_string(), records)]);
    for c in &found {
        assert!(c.estimated_improvement.is_finite());
        assert!(c.lower_mode_mean.is_finite());
        assert!(c.upper_mode_mean.is_finite());
    }
    let c = creature(vec![]);
    let coordinated = bimodal_neurons_to_coordinated_candidates(&found, &c);
    assert_gains_finite(&coordinated);
    let json = serde_json::to_string(&coordinated).unwrap();
    assert!(!json.contains("inf") && !json.contains("NaN"), "{json}");
}

#[test]
fn bimodal_finite_input_still_emits_candidate() {
    let mut values = vec![0.0_f32; 10];
    values.extend(std::iter::repeat_n(10.0_f32, 10));
    let records: Vec<DiscoverRecord> = values
        .iter()
        .enumerate()
        .map(|(i, &v)| rec("h", i as u32, Some(v), 0.0, 0.0))
        .collect();
    let neurons = vec![("h".to_string(), "TANH".to_string(), 0.0_f32)];
    let found = detect_bimodal_neurons(&neurons, &[("h".to_string(), records)]);
    assert_eq!(found.len(), 1);
    assert!(found[0].estimated_improvement.is_finite());
}

#[test]
fn dormant_inf_weight_times_zero_activation_does_not_emit_nan_gain() {
    let c = creature(vec![
        synapse("input-1", "output-1", f32::INFINITY),
        synapse("input-2", "output-1", 0.5),
    ]);
    let records: Vec<DiscoverRecord> = (0..30).map(|i| rec("input-1", i, None, 0.0, 0.0)).collect();
    let found = detect_dormant_synapses(&c, &[("input-1".to_string(), records)]);
    for d in &found {
        assert!(d.estimated_improvement.is_finite());
        assert!(d.mean_abs_contribution.is_finite());
    }
    assert_gains_finite(&dormant_synapses_to_coordinated_candidates(&found));
}

#[test]
fn dormant_nan_activation_does_not_emit_nan_gain() {
    let c = creature(vec![
        synapse("input-1", "output-1", 0.5),
        synapse("input-2", "output-1", 0.5),
    ]);
    let records: Vec<DiscoverRecord> = (0..30)
        .map(|i| rec("input-1", i, None, if i == 3 { f32::NAN } else { 0.0 }, 0.0))
        .collect();
    let found = detect_dormant_synapses(&c, &[("input-1".to_string(), records)]);
    for d in &found {
        assert!(d.estimated_improvement.is_finite());
    }
    assert_gains_finite(&dormant_synapses_to_coordinated_candidates(&found));
}

#[test]
fn dormant_finite_input_still_emits_candidate() {
    let c = creature(vec![
        synapse("input-1", "output-1", 0.5),
        synapse("input-2", "output-1", 0.5),
    ]);
    let records: Vec<DiscoverRecord> = (0..30).map(|i| rec("input-1", i, None, 0.0, 0.0)).collect();
    let found = detect_dormant_synapses(&c, &[("input-1".to_string(), records)]);
    assert_eq!(found.len(), 1);
}

#[test]
fn high_error_squash_aggregate_current_squash_does_not_emit_nan_gain() {
    // Aggregate current squash => compute_mae_for_squash returns +inf for the
    // current MAE while TANH etc. give finite MAE: (inf - finite) / inf = NaN.
    let records: Vec<DiscoverRecord> = (0..30).map(|i| rec("h", i, Some(0.5), 0.0, 1.0)).collect();
    let neurons = vec![("h".to_string(), "MAXIMUM".to_string(), 0.0_f32)];
    let found = detect_high_error_squash_candidates(&neurons, &[("h".to_string(), records)]);
    for c in &found {
        assert!(c.estimated_improvement.is_finite());
        assert!(c.error_reduction_fraction.is_finite());
        assert!(c.mean_absolute_error.is_finite());
    }
    assert_gains_finite(&high_error_squash_to_coordinated_candidates(&found));
}

#[test]
fn high_error_squash_finite_input_still_emits_candidate() {
    // RELU on a negative pre-activation outputs 0 against a target of 1.0,
    // whereas LOGISTIC on a large positive value gets much closer.
    let records: Vec<DiscoverRecord> = (0..30).map(|i| rec("h", i, Some(-5.0), 0.0, 1.0)).collect();
    let neurons = vec![("h".to_string(), "RELU".to_string(), 0.0_f32)];
    let found = detect_high_error_squash_candidates(&neurons, &[("h".to_string(), records)]);
    for c in &found {
        assert!(c.estimated_improvement.is_finite());
    }
}

#[test]
fn noise_non_finite_source_activation_does_not_emit_nan_gain() {
    let c = creature(vec![synapse("input-1", "output-1", 1.0)]);
    let src: Vec<DiscoverRecord> = (0..30)
        .map(|i| {
            rec(
                "input-1",
                i,
                None,
                if i == 5 { f32::NAN } else { i as f32 },
                0.0,
            )
        })
        .collect();
    let tgt: Vec<DiscoverRecord> = (0..30)
        .map(|i| rec("output-1", i, None, 0.0, 0.1))
        .collect();
    let found = detect_noisy_synapses(
        &c,
        &[("input-1".to_string(), src), ("output-1".to_string(), tgt)],
    );
    for n in &found {
        assert!(n.estimated_improvement.is_finite());
        assert!(n.noise_contribution.is_finite());
        assert!(n.signal_contribution.is_finite());
    }
    assert_gains_finite(&noisy_synapses_to_coordinated_candidates(&found));
}

#[test]
fn noise_finite_input_still_emits_candidate() {
    let c = creature(vec![synapse("input-1", "output-1", 1.0)]);
    let src: Vec<DiscoverRecord> = (0..30)
        .map(|i| rec("input-1", i, None, i as f32, 0.0))
        .collect();
    let tgt: Vec<DiscoverRecord> = (0..30)
        .map(|i| rec("output-1", i, None, 0.0, 0.1))
        .collect();
    let found = detect_noisy_synapses(
        &c,
        &[("input-1".to_string(), src), ("output-1".to_string(), tgt)],
    );
    assert_eq!(found.len(), 1);
}

fn oscillating_records(count: u32, magnitude: f32, nan_at: Option<u32>) -> Vec<DiscoverRecord> {
    (0..count)
        .map(|i| {
            let a = if Some(i) == nan_at {
                f32::NAN
            } else if i % 2 == 0 {
                magnitude
            } else {
                -magnitude
            };
            rec("h", i, None, a, 0.0)
        })
        .collect()
}

fn run_oscillating(records: Vec<DiscoverRecord>) -> Vec<CoordinatedStructuralCandidateJson> {
    let neurons = vec![("h".to_string(), "TANH".to_string(), 0.0_f32)];
    let found = detect_oscillating_neurons(&neurons, &[("h".to_string(), records)]);
    for c in &found {
        assert!(c.estimated_improvement.is_finite());
        assert!(c.mean_abs_activation.is_finite());
    }
    oscillating_neurons_to_coordinated_candidates(&found)
}

#[test]
fn oscillating_nan_activation_does_not_emit_nan_gain() {
    assert_gains_finite(&run_oscillating(oscillating_records(30, 1.0, Some(4))));
}

#[test]
fn oscillating_overflowing_mean_does_not_emit_infinite_gain() {
    assert_gains_finite(&run_oscillating(oscillating_records(40, 3.0e38, None)));
}

#[test]
fn oscillating_finite_input_still_emits_candidate() {
    let out = run_oscillating(oscillating_records(30, 1.0, None));
    assert_eq!(out.len(), 1);
}

#[test]
fn bimodal_overflowing_finite_values_do_not_emit_non_finite_bias() {
    // Every value is finite, but the cluster sums overflow f32.
    let mut values = vec![-3.0e38_f32; 10];
    values.extend(std::iter::repeat_n(3.0e38_f32, 10));
    let records: Vec<DiscoverRecord> = values
        .iter()
        .enumerate()
        .map(|(i, &v)| rec("h", i as u32, Some(v), 0.0, 0.0))
        .collect();
    let neurons = vec![("h".to_string(), "TANH".to_string(), 0.0_f32)];
    let found = detect_bimodal_neurons(&neurons, &[("h".to_string(), records)]);
    for c in &found {
        assert!(c.estimated_improvement.is_finite());
        assert!(c.lower_mode_mean.is_finite());
        assert!(c.upper_mode_mean.is_finite());
    }
    let c = creature(vec![]);
    assert_gains_finite(&bimodal_neurons_to_coordinated_candidates(&found, &c));
}

#[test]
fn bimodal_single_non_finite_value_is_dropped_not_fatal() {
    // One -inf amid 20 sound bimodal samples: the bad sample is discarded and
    // the genuine bimodality is still reported with finite statistics.
    let mut values = vec![f32::NEG_INFINITY];
    values.extend(std::iter::repeat_n(0.0_f32, 10));
    values.extend(std::iter::repeat_n(10.0_f32, 10));
    let records: Vec<DiscoverRecord> = values
        .iter()
        .enumerate()
        .map(|(i, &v)| rec("h", i as u32, Some(v), 0.0, 0.0))
        .collect();
    let neurons = vec![("h".to_string(), "TANH".to_string(), 0.0_f32)];
    let found = detect_bimodal_neurons(&neurons, &[("h".to_string(), records)]);
    assert_eq!(found.len(), 1);
    assert!(found[0].lower_mode_mean.is_finite());
    assert!(found[0].estimated_improvement.is_finite());
}
