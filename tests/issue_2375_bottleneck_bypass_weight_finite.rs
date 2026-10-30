//! Regression tests for Issue #2375: the bottleneck bypass converter computed
//! `upstream_weight * downstream_weight * 0.5` in f32 with no finiteness check.
//! Two finite weights whose product overflows f32 (e.g. 2e19 x 2e19) emitted an
//! `AddSynapse` carrying `±inf`, because the gain gates test only the gain and
//! never the emitted weight. The fix skips any bypass whose weight is not finite.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use neat_ai_discovery::analysis::detection::bottleneck::{
    BottleneckNeuronCandidate, bottleneck_neurons_to_coordinated_candidates,
    detect_bottleneck_neurons,
};
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{
    CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson, NeuronJson,
    SynapseJson,
};

fn neuron(uuid: &str, neuron_type: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
        squash: "TANH".to_string(),
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

/// 5 inputs -> hidden `h-bn` (fan-in 5) -> 2 outputs (fan-out 2, ratio 2.5).
///
/// `in-0` is the strongest upstream (largest |weight|), so it is the bypass source
/// the converter picks. The other inputs stay at 0.1 magnitude.
fn bypass_creature(strong_upstream: f32, out0_weight: f32, out1_weight: f32) -> CreatureJson {
    CreatureJson {
        neurons: vec![
            neuron("in-0", "input"),
            neuron("in-1", "input"),
            neuron("in-2", "input"),
            neuron("in-3", "input"),
            neuron("in-4", "input"),
            neuron("h-bn", "hidden"),
            neuron("out-0", "output"),
            neuron("out-1", "output"),
        ],
        synapses: vec![
            synapse("in-0", "h-bn", strong_upstream),
            synapse("in-1", "h-bn", 0.1),
            synapse("in-2", "h-bn", -0.1),
            synapse("in-3", "h-bn", 0.1),
            synapse("in-4", "h-bn", -0.1),
            synapse("h-bn", "out-0", out0_weight),
            synapse("h-bn", "out-1", out1_weight),
        ],
        input: 5,
        output: 2,
    }
}

/// 100 finite records for `h-bn`, enough to pass `MIN_SAMPLES_FOR_BOTTLENECK`.
fn bottleneck_records() -> Vec<(String, Vec<DiscoverRecord>)> {
    vec![(
        "h-bn".to_string(),
        (0..100u32)
            .map(|i| DiscoverRecord {
                obs_index: i,
                neuron_uuid: "h-bn".to_string(),
                value: Some(i as f32 * 0.05),
                activation: (i as f32 * 0.1).sin() * 0.5,
                errors: vec![0.1 * (1.0 + (i as f32 * 0.03).sin())],
            })
            .collect(),
    )]
}

/// Detect the bottleneck and assert the precondition that makes the bypass branch run,
/// so the tests below cannot pass vacuously.
fn detect_bypass_candidate(creature: &CreatureJson) -> Vec<BottleneckNeuronCandidate> {
    let records = bottleneck_records();
    let candidates = detect_bottleneck_neurons(creature, &records, None);
    assert!(
        !candidates.is_empty(),
        "precondition: h-bn must be flagged as a bottleneck"
    );
    assert!(
        candidates[0]
            .recommended_actions
            .iter()
            .any(|a| a == "addBypassSynapse"),
        "precondition: h-bn must recommend addBypassSynapse, got {:?}",
        candidates[0].recommended_actions
    );
    candidates
}

/// Every `AddSynapse` weight across all emitted candidates and ops.
fn all_add_synapse_weights(cands: &[CoordinatedStructuralCandidateJson]) -> Vec<f32> {
    cands
        .iter()
        .flat_map(|c| c.operations.iter())
        .filter_map(|op| match op {
            CoordinatedStructuralOpJson::AddSynapse { weight, .. } => Some(*weight),
            _ => None,
        })
        .collect()
}

/// Weights of every `AddSynapse` from `from` to `to`.
fn bypass_weights(cands: &[CoordinatedStructuralCandidateJson], from: &str, to: &str) -> Vec<f32> {
    cands
        .iter()
        .flat_map(|c| c.operations.iter())
        .filter_map(|op| match op {
            CoordinatedStructuralOpJson::AddSynapse {
                from_neuron_uuid,
                to_neuron_uuid,
                weight,
            } if from_neuron_uuid.as_str() == from && to_neuron_uuid.as_str() == to => {
                Some(*weight)
            }
            _ => None,
        })
        .collect()
}

fn assert_close(actual: f32, expected: f32) {
    let tol = expected.abs() * 1e-5;
    assert!(
        (actual - expected).abs() <= tol,
        "expected {expected}, got {actual}"
    );
}

/// The emitted bypass weight must always be finite, even when the f32 product overflows.
#[test]
fn bypass_weight_is_never_non_finite() {
    let creature = bypass_creature(2e19, 2e19, 0.8);
    let candidates = detect_bypass_candidate(&creature);
    let out = bottleneck_neurons_to_coordinated_candidates(&candidates, &creature, None);

    let weights = all_add_synapse_weights(&out);
    assert!(
        !weights.is_empty(),
        "converter should emit AddSynapse candidates"
    );
    for w in &weights {
        assert!(w.is_finite(), "AddSynapse weight must be finite, got {w}");
    }

    // Non-vacuity: the finite in-0 -> out-1 bypass must still be present.
    assert!(
        !bypass_weights(&out, "in-0", "out-1").is_empty(),
        "finite in-0 -> out-1 bypass should still be emitted"
    );
}

/// The overflowing in-0 -> out-0 bypass is dropped, but the finite in-0 -> out-1 bypass
/// (2e19 x 0.8 x 0.5 = 8e18) is still emitted with that weight.
#[test]
fn finite_bypass_is_still_emitted_for_non_overflowing_downstream() {
    let creature = bypass_creature(2e19, 2e19, 0.8);
    let candidates = detect_bypass_candidate(&creature);
    let out = bottleneck_neurons_to_coordinated_candidates(&candidates, &creature, None);

    let to_out1 = bypass_weights(&out, "in-0", "out-1");
    assert_eq!(
        to_out1.len(),
        1,
        "exactly one in-0 -> out-1 bypass expected"
    );
    assert_close(to_out1[0], 8e18);

    let to_out0 = bypass_weights(&out, "in-0", "out-0");
    assert!(
        to_out0.is_empty(),
        "overflowing in-0 -> out-0 bypass must be dropped, got {to_out0:?}"
    );
}

/// A negative overflow (-2e19 x 2e19 = -inf) must not emit a non-finite weight either.
#[test]
fn negative_weight_overflow_is_dropped() {
    let creature = bypass_creature(-2e19, 2e19, 0.8);
    let candidates = detect_bypass_candidate(&creature);
    let out = bottleneck_neurons_to_coordinated_candidates(&candidates, &creature, None);

    let weights = all_add_synapse_weights(&out);
    assert!(
        !weights.is_empty(),
        "converter should emit AddSynapse candidates"
    );
    for w in &weights {
        assert!(w.is_finite(), "AddSynapse weight must be finite, got {w}");
    }

    let to_out0 = bypass_weights(&out, "in-0", "out-0");
    assert!(
        to_out0.is_empty(),
        "overflowing in-0 -> out-0 bypass must be dropped, got {to_out0:?}"
    );
}

/// Ordinary weights (0.5 x 0.8 x 0.5 = 0.2) give the same bypass weight as before.
#[test]
fn ordinary_weights_unchanged() {
    let creature = bypass_creature(0.5, 0.8, 0.8);
    let candidates = detect_bypass_candidate(&creature);
    let out = bottleneck_neurons_to_coordinated_candidates(&candidates, &creature, None);

    let to_out0 = bypass_weights(&out, "in-0", "out-0");
    assert_eq!(
        to_out0.len(),
        1,
        "exactly one in-0 -> out-0 bypass expected"
    );
    assert_close(to_out0[0], 0.2);

    let to_out1 = bypass_weights(&out, "in-0", "out-1");
    assert_eq!(
        to_out1.len(),
        1,
        "exactly one in-0 -> out-1 bypass expected"
    );
    assert_close(to_out1[0], 0.2);
}
