//! Tests for Issue #1249 — compound bias/weight degradation must not
//! emit `setWeight` corrections sized by the SSE-improvement formula
//! when the target's errors are quantised `{0, 1}` flags.
//!
//! Under `CATEGORICAL_ERROR` the `baseline_error_sq` / `corrected_error_sq`
//! difference no longer corresponds to the network's loss reduction, so
//! the weight-correction phase of `detect_compound_bias_weight_degradations`
//! must skip the affected synapse instead of producing a misleading
//! `setWeight` recommendation.

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use neat_ai_discovery::analysis::detection::compound_degradation::detect_compound_bias_weight_degradations;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

fn neuron(uuid: &str, neuron_type: &str, squash: &str, bias: f32) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
        squash: squash.to_string(),
        bias,
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

fn record(
    uuid: &str,
    obs_index: u32,
    activation: f32,
    value: Option<f32>,
    errors: Vec<f32>,
) -> DiscoverRecord {
    DiscoverRecord {
        obs_index,
        neuron_uuid: uuid.to_string(),
        value,
        activation,
        errors,
    }
}

fn make_compound_creature() -> CreatureJson {
    CreatureJson {
        neurons: vec![
            neuron("input-a", "input", "IDENTITY", 0.0),
            neuron("hidden-1", "hidden", "TANH", 0.0),
            neuron("output-0", "output", "IDENTITY", 0.0),
        ],
        synapses: vec![
            synapse("input-a", "hidden-1", 0.6),
            synapse("hidden-1", "output-0", 0.5),
        ],
        input: 1,
        output: 1,
    }
}

fn build_records<F>(make_output_err: F) -> Vec<(String, Vec<DiscoverRecord>)>
where
    F: Fn(usize) -> f32,
{
    let mut records: Vec<(String, Vec<DiscoverRecord>)> = Vec::new();
    for uuid in ["input-a", "hidden-1", "output-0"] {
        let recs: Vec<DiscoverRecord> = (0..60)
            .map(|i| {
                let phase = (i as f32) * 0.21;
                let act = match uuid {
                    "input-a" => phase.sin(),
                    "hidden-1" => (phase * 0.6).tanh(),
                    _ => 0.0,
                };
                let value = match uuid {
                    "hidden-1" => Some(phase * 0.6),
                    _ => Some(act),
                };
                let errors = if uuid == "hidden-1" {
                    // Drive the bias-drift signal so a BiasCorrection is
                    // produced; the weight-correction phase is what we
                    // are gating in this test.
                    vec![0.4 * phase.sin() + 0.5]
                } else if uuid == "output-0" {
                    vec![make_output_err(i)]
                } else {
                    vec![]
                };
                record(uuid, i as u32, act, value, errors)
            })
            .collect();
        records.push((uuid.to_string(), recs));
    }
    records
}

/// Sanity check: continuous output errors that correlate with the
/// hidden activation must still produce at least one compound
/// degradation candidate. Without this anchor, the "no candidate under
/// quantised" assertion below would be vacuous.
#[test]
fn compound_degradation_emits_for_continuous_errors() {
    let creature = make_compound_creature();
    let records = build_records(|i| {
        let phase = (i as f32) * 0.21;
        // Strongly correlated continuous residual on the output.
        0.8 * (phase * 0.6).tanh() + 0.4
    });

    let candidates = detect_compound_bias_weight_degradations(&creature, &records);
    assert!(
        !candidates.is_empty(),
        "compound degradation continuous baseline must emit at least one candidate (got 0); without this the gate test below is vacuous",
    );
}

/// Under the quantised `{0, 1}` regime the SSE-based weight-correction
/// improvement is bounded by the misclassification count rather than a
/// loss reduction (Issue #1249). The detector must therefore skip the
/// weight correction whose **target** neuron has quantised errors —
/// otherwise the emitted `setWeight` magnitude does not correspond to
/// any real loss reduction. Weight corrections targeting neurons with
/// continuous errors remain valid.
#[test]
fn compound_degradation_skips_quantised_target_errors() {
    let creature = make_compound_creature();
    let records = build_records(|i| if i % 2 == 0 { 0.0 } else { 1.0 });

    let candidates = detect_compound_bias_weight_degradations(&creature, &records);

    // Pre-fix the unfixed code emitted candidates whose `weight_to_uuid`
    // was the quantised output neuron, sized by the SSE-improvement
    // formula. Post-fix, no candidate may target a quantised-error
    // neuron — other corrections (e.g. on the continuous-error hidden
    // neuron) remain valid.
    let targeting_quantised = candidates
        .iter()
        .filter(|c| c.weight_to_uuid == "output-0")
        .count();
    assert_eq!(
        targeting_quantised, 0,
        "weight correction targeting quantised neuron must be gated under Issue #1249, got {targeting_quantised} candidate(s)",
    );
}
