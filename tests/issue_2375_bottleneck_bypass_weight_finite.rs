//! Regression tests for Issue #2375 (CWE-682): the bottleneck bypass converter
//! computed `bypass_weight = upstream_weight * downstream_weight * 0.5` in f32.
//! Two finite weights whose product overflows f32 (e.g. 2e19 x 2e19) produced an
//! `AddSynapse { weight: +inf }` candidate. Neither the gain gate nor the
//! post-processing gate inspects operation weights, so the non-finite weight
//! reached the caller under a finite gain.

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

/// Number of recorded samples per neuron; comfortably above `MIN_DISCOVERY_SAMPLE_COUNT` (20).
const SAMPLES: u32 = 50;

fn neuron(uuid: &str, kind: &str, squash: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: kind.to_string(),
        squash: squash.to_string(),
        bias: 0.0,
    }
}

fn syn(from: &str, to: &str, weight: f32) -> SynapseJson {
    SynapseJson {
        from_uuid: from.to_string(),
        to_uuid: to.to_string(),
        weight,
        synapse_type: None,
    }
}

fn records(uuid: &str, error: f32) -> Vec<DiscoverRecord> {
    (0..SAMPLES)
        .map(|i| DiscoverRecord {
            obs_index: i,
            neuron_uuid: uuid.to_string(),
            value: None,
            activation: i as f32 * 0.01,
            errors: vec![error],
        })
        .collect()
}

/// Five inputs feed hidden `h-bottleneck` (fan-in 5), which fans out to two outputs
/// (fan-out 2, ratio 2.5). Fan-in > 3 means `addBypassSynapse` is recommended.
/// `strongest_in` is the weight of `in-0 -> h-bottleneck` (the best bypass upstream);
/// `out_weight` is the weight of both `h-bottleneck -> out-*` synapses.
fn bottleneck_creature(strongest_in: f32, out_weight: f32) -> CreatureJson {
    CreatureJson {
        neurons: vec![
            neuron("in-0", "input", "IDENTITY"),
            neuron("in-1", "input", "IDENTITY"),
            neuron("in-2", "input", "IDENTITY"),
            neuron("in-3", "input", "IDENTITY"),
            neuron("in-4", "input", "IDENTITY"),
            neuron("h-bottleneck", "hidden", "TANH"),
            neuron("out-0", "output", "TANH"),
            neuron("out-1", "output", "TANH"),
        ],
        synapses: vec![
            syn("in-0", "h-bottleneck", strongest_in),
            syn("in-1", "h-bottleneck", 0.1),
            syn("in-2", "h-bottleneck", 0.1),
            syn("in-3", "h-bottleneck", 0.1),
            syn("in-4", "h-bottleneck", 0.1),
            syn("h-bottleneck", "out-0", out_weight),
            syn("h-bottleneck", "out-1", out_weight),
        ],
        input: 5,
        output: 2,
    }
}

/// Runs the two-stage pipeline the production caller uses: detect, then convert.
fn run_pipeline(
    creature: &CreatureJson,
) -> (
    Vec<BottleneckNeuronCandidate>,
    Vec<CoordinatedStructuralCandidateJson>,
) {
    let neuron_records = vec![("h-bottleneck".to_string(), records("h-bottleneck", 0.5))];
    let detected = detect_bottleneck_neurons(creature, &neuron_records, None);
    let coordinated = bottleneck_neurons_to_coordinated_candidates(&detected, creature, None);
    (detected, coordinated)
}

/// Returns the weights of every `AddSynapse` op in a candidate list.
fn add_synapse_weights(candidates: &[CoordinatedStructuralCandidateJson]) -> Vec<f32> {
    candidates
        .iter()
        .flat_map(|c| c.operations.iter())
        .filter_map(|op| match op {
            CoordinatedStructuralOpJson::AddSynapse { weight, .. } => Some(*weight),
            _ => None,
        })
        .collect()
}

/// Returns the bypass candidates (comment mentions "bypass").
fn bypass_candidates(
    candidates: &[CoordinatedStructuralCandidateJson],
) -> Vec<&CoordinatedStructuralCandidateJson> {
    candidates
        .iter()
        .filter(|c| c.comment.as_ref().is_some_and(|s| s.contains("bypass")))
        .collect()
}

/// Overflow case: 2e19 x 2e19 = 4e38 overflows f32 to +inf before the 0.5 factor
/// in the unfixed converter. Every emitted `AddSynapse` weight must be finite.
#[test]
fn bypass_weight_is_never_non_finite() {
    let creature = bottleneck_creature(2e19, 2e19);
    let (detected, coordinated) = run_pipeline(&creature);

    // The converter must actually have been reached.
    assert!(
        detected.iter().any(|c| c.neuron_uuid == "h-bottleneck"),
        "detector must return the h-bottleneck candidate, got {detected:?}"
    );
    assert!(
        detected.iter().any(|c| c
            .recommended_actions
            .iter()
            .any(|a| a == "addBypassSynapse")),
        "h-bottleneck must recommend addBypassSynapse"
    );

    // The parallel-neuron candidate is unaffected by the bypass fix and must still be emitted.
    assert!(
        coordinated
            .iter()
            .any(|c| c.comment.as_ref().is_some_and(|s| s.contains("parallel"))),
        "parallel-neuron candidate must still be emitted"
    );

    // Every AddSynapse weight, bypass or parallel, must be finite.
    let weights = add_synapse_weights(&coordinated);
    assert!(
        !weights.is_empty(),
        "expected AddSynapse operations to be emitted"
    );
    for w in &weights {
        assert!(
            w.is_finite(),
            "emitted AddSynapse weight must be finite, got {w}"
        );
    }

    // The overflowing bypass downstreams are skipped, so no bypass candidate is emitted.
    assert!(
        bypass_candidates(&coordinated).is_empty(),
        "overflowing bypass must be skipped, got {:?}",
        bypass_candidates(&coordinated)
            .iter()
            .map(|c| &c.operations)
            .collect::<Vec<_>>()
    );
}

/// Positive control: ordinary weights must still produce the bypass candidate with
/// the expected weight 0.8 * 0.6 * 0.5 = 0.24, so the overflow test cannot pass vacuously.
#[test]
fn bypass_weight_is_emitted_for_ordinary_weights() {
    let creature = bottleneck_creature(0.8, 0.6);
    let (_detected, coordinated) = run_pipeline(&creature);

    let bypasses = bypass_candidates(&coordinated);
    assert!(
        !bypasses.is_empty(),
        "ordinary weights must produce bypass candidates"
    );

    let expected = 0.8_f32 * 0.6_f32 * 0.5_f32;
    let found = bypasses.iter().any(|c| {
        c.operations.iter().any(|op| match op {
            CoordinatedStructuralOpJson::AddSynapse {
                from_neuron_uuid,
                to_neuron_uuid,
                weight,
            } => {
                from_neuron_uuid == "in-0"
                    && to_neuron_uuid == "out-0"
                    && (weight - expected).abs() < 1e-6
            }
            _ => false,
        })
    });
    assert!(
        found,
        "expected AddSynapse in-0 -> out-0 with weight {expected}, got {:?}",
        bypasses.iter().map(|c| &c.operations).collect::<Vec<_>>()
    );
}

/// Boundary: 1e19 x 1e19 x 0.5 = 5e37 is large but finite in f32, so the bypass
/// must still be emitted with a finite weight.
#[test]
fn bypass_weight_is_emitted_below_overflow_boundary() {
    let creature = bottleneck_creature(1e19, 1e19);
    let (_detected, coordinated) = run_pipeline(&creature);

    let bypasses = bypass_candidates(&coordinated);
    assert!(
        !bypasses.is_empty(),
        "finite-but-large bypass weight must still produce bypass candidates"
    );

    let weights = add_synapse_weights(&coordinated);
    for w in &weights {
        assert!(
            w.is_finite(),
            "emitted AddSynapse weight must be finite, got {w}"
        );
    }

    let expected = 5e37_f32;
    let found = bypasses.iter().any(|c| {
        c.operations.iter().any(|op| match op {
            CoordinatedStructuralOpJson::AddSynapse {
                from_neuron_uuid,
                to_neuron_uuid,
                weight,
            } => {
                from_neuron_uuid == "in-0"
                    && to_neuron_uuid == "out-0"
                    && weight.is_finite()
                    && (weight - expected).abs() / expected < 1e-3
            }
            _ => false,
        })
    });
    assert!(
        found,
        "expected finite AddSynapse in-0 -> out-0 with weight about {expected}, got {:?}",
        bypasses.iter().map(|c| &c.operations).collect::<Vec<_>>()
    );
}
