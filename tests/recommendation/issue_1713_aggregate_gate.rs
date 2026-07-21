//! Tests for Issue #1713: Contribution-propagation break — the
//! expected-error-reduction estimator ignores downstream aggregate selection.
//!
//! Follow-up from the #1704 extent report (gap G3). The squash + weight rescale
//! estimator (`detect_squash_weight_rescale_candidates`) simulates each
//! candidate neuron in isolation as `f(x)` and compares MAE against that
//! neuron's own target. When the branch feeds a downstream MAX/MIN/IF selection
//! aggregate, a change that flips the branch's activation sign/range changes
//! which branch the aggregate selects — an effect the local estimate cannot
//! see, so it systematically mispredicts the gain (SELU→ABSOLUTE predicted
//! `+4.2e-10` but measured `−8.7e-4`).
//!
//! ## TDD Plan
//! 1. A neuron whose output feeds an aggregate selection is gated out.
//! 2. Gating is narrow — an otherwise-identical neuron feeding a pure `f(x)`
//!    neuron still produces candidates.
//! 3. All selection/aggregate targets (MAX/MIN/IF/MEAN/HYPOT) trigger the gate.
//! 4. A neuron feeding an aggregate *and* a pure neuron is still gated out
//!    (the aggregate path poisons the estimate).

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use neat_ai_discovery::analysis::detection::squash_weight_rescale::detect_squash_weight_rescale_candidates;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

// =============================================================================
// Helpers
// =============================================================================

fn make_record(uuid: &str, idx: u32, value: f32, activation: f32, error: f32) -> DiscoverRecord {
    DiscoverRecord {
        obs_index: idx,
        neuron_uuid: uuid.to_string(),
        value: Some(value),
        activation,
        errors: vec![error],
    }
}

fn make_creature(neurons: Vec<NeuronJson>, synapses: Vec<SynapseJson>) -> CreatureJson {
    CreatureJson {
        neurons,
        synapses,
        input: 2,
        output: 1,
    }
}

fn neuron(uuid: &str, kind: &str, squash: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: kind.to_string(),
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

/// SELU-branch records: pre-activations span negative and positive, so a squash
/// change (e.g. to ABSOLUTE) looks locally attractive — high, one-sided error.
/// Mirrors the `v2_change-squash_selu-to-absolute` committed fixture shape.
fn selu_branch_records(uuid: &str) -> Vec<(String, Vec<DiscoverRecord>)> {
    vec![(
        uuid.to_string(),
        (0..40)
            .map(|i| {
                let value = (i as f32 - 20.0) / 10.0; // -2.0 to 1.9
                let activation = value.max(0.0); // clipped, one-sided
                let error = 0.15 + (i as f32 * 0.002).sin() * 0.02;
                make_record(uuid, i, value, activation, error)
            })
            .collect(),
    )]
}

// =============================================================================
// 1. Branch feeding an aggregate selection is gated out
// =============================================================================

#[test]
fn test_branch_feeding_aggregate_is_gated_out() {
    // hidden-selu → MAXIMUM aggregate → output. The local estimate would happily
    // recommend a squash change, but the aggregate decides selection, so gate it.
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("input-2", "input", "IDENTITY"),
            neuron("hidden-selu", "hidden", "SELU"),
            neuron("agg", "hidden", "MAXIMUM"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "hidden-selu", 1.5),
            synapse("input-2", "hidden-selu", -0.8),
            synapse("hidden-selu", "agg", 1.0),
            synapse("agg", "output-1", 1.0),
        ],
    );

    let hidden_neurons: Vec<(String, String, f32)> =
        vec![("hidden-selu".to_string(), "SELU".to_string(), 0.0)];

    let candidates = detect_squash_weight_rescale_candidates(
        &creature,
        &hidden_neurons,
        &selu_branch_records("hidden-selu"),
    );

    assert!(
        candidates.is_empty(),
        "A branch feeding an aggregate selection must be gated out (Issue #1713), got: {candidates:?}"
    );
}

// =============================================================================
// 2. Gating is narrow — pure f(x) downstream still produces candidates
// =============================================================================

#[test]
fn test_branch_feeding_pure_neuron_still_produces_candidates() {
    // Identical branch, but downstream is a pure IDENTITY neuron — the local
    // estimate is valid here, so the candidate must survive.
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("input-2", "input", "IDENTITY"),
            neuron("hidden-selu", "hidden", "SELU"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "hidden-selu", 1.5),
            synapse("input-2", "hidden-selu", -0.8),
            synapse("hidden-selu", "output-1", 1.0),
        ],
    );

    let hidden_neurons: Vec<(String, String, f32)> =
        vec![("hidden-selu".to_string(), "SELU".to_string(), 0.0)];

    let candidates = detect_squash_weight_rescale_candidates(
        &creature,
        &hidden_neurons,
        &selu_branch_records("hidden-selu"),
    );

    assert!(
        !candidates.is_empty(),
        "A branch feeding only pure f(x) neurons must still produce candidates (gate must be narrow)"
    );
}

// =============================================================================
// 3. Every aggregate/selection target triggers the gate
// =============================================================================

#[test]
fn test_all_aggregate_targets_trigger_gate() {
    for agg in ["MAXIMUM", "MINIMUM", "IF", "MEAN", "HYPOT", "HYPOTV2"] {
        let creature = make_creature(
            vec![
                neuron("input-1", "input", "IDENTITY"),
                neuron("input-2", "input", "IDENTITY"),
                neuron("hidden-selu", "hidden", "SELU"),
                neuron("agg", "hidden", agg),
                neuron("output-1", "output", "IDENTITY"),
            ],
            vec![
                synapse("input-1", "hidden-selu", 1.5),
                synapse("input-2", "hidden-selu", -0.8),
                synapse("hidden-selu", "agg", 1.0),
                synapse("agg", "output-1", 1.0),
            ],
        );

        let hidden_neurons: Vec<(String, String, f32)> =
            vec![("hidden-selu".to_string(), "SELU".to_string(), 0.0)];

        let candidates = detect_squash_weight_rescale_candidates(
            &creature,
            &hidden_neurons,
            &selu_branch_records("hidden-selu"),
        );

        assert!(
            candidates.is_empty(),
            "A branch feeding a {agg} aggregate must be gated out (Issue #1713)"
        );
    }
}

// =============================================================================
// 4. Mixed fan-out (aggregate + pure) is still gated out
// =============================================================================

#[test]
fn test_mixed_fanout_with_aggregate_is_gated_out() {
    // The branch feeds both a pure output neuron and a MAX aggregate. The
    // aggregate path poisons the whole-creature estimate, so gate it out.
    let creature = make_creature(
        vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("input-2", "input", "IDENTITY"),
            neuron("hidden-selu", "hidden", "SELU"),
            neuron("agg", "hidden", "MAXIMUM"),
            neuron("output-1", "output", "IDENTITY"),
        ],
        vec![
            synapse("input-1", "hidden-selu", 1.5),
            synapse("input-2", "hidden-selu", -0.8),
            synapse("hidden-selu", "output-1", 1.0),
            synapse("hidden-selu", "agg", 1.0),
            synapse("agg", "output-1", 1.0),
        ],
    );

    let hidden_neurons: Vec<(String, String, f32)> =
        vec![("hidden-selu".to_string(), "SELU".to_string(), 0.0)];

    let candidates = detect_squash_weight_rescale_candidates(
        &creature,
        &hidden_neurons,
        &selu_branch_records("hidden-selu"),
    );

    assert!(
        candidates.is_empty(),
        "A branch feeding an aggregate (even alongside a pure neuron) must be gated out (Issue #1713)"
    );
}
