//! Issue #1622 — promote zero-variance (functionally-constant) hidden neurons
//! past the #1518 honest gain ranking and #1448 drought demotion as priority
//! remove-neuron candidates.
//!
//! These tests exercise the observable outcome ("what"): a neuron the detector
//! sub-issue has flagged functionally-constant is surfaced as a *priority*
//! remove-neuron candidate even though its honest, propagation-aware gain is
//! ≈0 and even during a simulated search-exhaustion drought. An *unflagged*
//! ≈0-gain neuron must not be promoted (the over-promotion guard).
//!
//! The pipeline order mirrored here matches `analysis::orchestration`:
//!   1. #1530 honest-gain override (`apply_honest_remove_neuron_gain`)
//!   2. #1448 drought demotion (`deprioritise_remove_neuron_candidates`)
//!   3. #1622 constant-neuron promotion (`promote_constant_remove_neuron_candidates`)

use std::collections::HashSet;

use neat_ai_discovery::analysis::discovery_dispatch::apply_honest_remove_neuron_gain;
use neat_ai_discovery::analysis::remove_neuron_constant_promotion::{
    CONSTANT_NEURON_PRIORITY_GAIN, promote_constant_remove_neuron_candidates,
};
use neat_ai_discovery::analysis::remove_neuron_drought::{
    DroughtDeprioritisationInputs, deprioritise_remove_neuron_candidates,
    remove_neuron_deprioritisation_factor,
};
use neat_ai_discovery::{
    CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson,
};

/// A creature with two hidden neurons: `constant` carries no downstream
/// influence (weight 0 to the output — a zero-influence stand-in for a
/// functionally-constant neuron, honest gain ≈0), while `live` carries genuine
/// influence.
fn creature() -> CreatureJson {
    serde_json::from_str(
        r#"{
            "input": 1, "output": 1,
            "neurons": [
                {"uuid": "input-0", "type": "constant"},
                {"uuid": "constant", "type": "hidden", "squash": "IDENTITY"},
                {"uuid": "live", "type": "hidden", "squash": "IDENTITY"},
                {"uuid": "out-0", "type": "output", "squash": "IDENTITY"}
            ],
            "synapses": [
                {"fromUUID": "input-0", "toUUID": "constant", "weight": 1.0},
                {"fromUUID": "input-0", "toUUID": "live", "weight": 1.0},
                {"fromUUID": "constant", "toUUID": "out-0", "weight": 0.0},
                {"fromUUID": "live", "toUUID": "out-0", "weight": 5.0}
            ]
        }"#,
    )
    .expect("valid creature JSON")
}

fn remove_neuron_candidate(uuid: &str, gain: f32) -> CoordinatedStructuralCandidateJson {
    CoordinatedStructuralCandidateJson {
        remove_neuron_compensation: None,
        operations: vec![CoordinatedStructuralOpJson::RemoveNeuron {
            neuron_uuid: uuid.to_string(),
        }],
        expected_creature_score_gain: gain,
        comment: None,
    }
}

fn positive_gain_candidate(uuid: &str, gain: f32) -> CoordinatedStructuralCandidateJson {
    CoordinatedStructuralCandidateJson {
        remove_neuron_compensation: None,
        operations: vec![CoordinatedStructuralOpJson::SetBias {
            neuron_uuid: uuid.to_string(),
            bias: 0.1,
        }],
        expected_creature_score_gain: gain,
        comment: None,
    }
}

fn flagged(uuids: &[&str]) -> HashSet<String> {
    uuids.iter().map(|u| (*u).to_string()).collect()
}

/// Sort a candidate slice by expected gain descending — the deterministic,
/// NaN-safe order the downstream pipeline uses.
fn rank(candidates: &mut [CoordinatedStructuralCandidateJson]) {
    candidates.sort_by(|a, b| {
        b.expected_creature_score_gain
            .total_cmp(&a.expected_creature_score_gain)
    });
}

/// The op target UUID, used to identify a candidate after ranking.
fn target(candidate: &CoordinatedStructuralCandidateJson) -> &str {
    match candidate.operations.first().expect("at least one op") {
        CoordinatedStructuralOpJson::RemoveNeuron { neuron_uuid }
        | CoordinatedStructuralOpJson::SetBias { neuron_uuid, .. } => neuron_uuid,
        _ => "?",
    }
}

/// Regression mode 1: promotion bypasses the #1518 honest gain ranking.
///
/// After the honest-gain override drops the flagged constant neuron to ≈0 (far
/// below the positive-gain candidates), promotion lifts it to the priority gain
/// so it ranks first, ahead of every normal gain-ranked candidate.
#[test]
fn promotion_bypasses_honest_gain_ranking() {
    let creature = creature();
    let mut candidates = vec![
        remove_neuron_candidate("constant", 0.0),
        positive_gain_candidate("live", 0.002),
        positive_gain_candidate("out-0", 0.001),
    ];

    // #1530 honest-gain override: the constant neuron's gain becomes ≈0.
    apply_honest_remove_neuron_gain(&creature, &mut candidates);
    let honest = candidates
        .iter()
        .find(|c| target(c) == "constant")
        .expect("constant candidate present")
        .expected_creature_score_gain;
    assert!(
        honest.abs() < 1e-3,
        "honest gain for the zero-influence neuron should be ≈0, got {honest}"
    );

    // #1622 promotion.
    let promoted =
        promote_constant_remove_neuron_candidates(&mut candidates, &flagged(&["constant"]));
    assert_eq!(promoted, 1, "the flagged constant neuron is promoted");

    rank(&mut candidates);
    assert_eq!(
        target(&candidates[0]),
        "constant",
        "the promoted constant neuron must rank first, ahead of positive-gain candidates"
    );
    assert!(
        (candidates[0].expected_creature_score_gain - CONSTANT_NEURON_PRIORITY_GAIN).abs()
            < f32::EPSILON,
        "promoted candidate carries the priority gain"
    );
}

/// Regression mode 2: promotion survives the #1448 drought demotion.
///
/// Under an active search-exhaustion drought the demotion multiplies positive
/// remove-neuron gains by a factor < 1. Promotion (applied after) still lifts
/// the flagged constant neuron to priority, so it ranks first regardless.
#[test]
fn promotion_survives_drought_demotion() {
    let creature = creature();
    let mut candidates = vec![
        remove_neuron_candidate("constant", 0.0),
        positive_gain_candidate("live", 0.002),
    ];

    apply_honest_remove_neuron_gain(&creature, &mut candidates);

    // Simulated active search-exhaustion drought.
    let inputs = DroughtDeprioritisationInputs {
        consecutive_failures: 10,
        environmentally_disabled_passes: 0,
        drought_threshold: 5,
    };
    let factor = remove_neuron_deprioritisation_factor(&inputs, 0.1);
    assert!(factor < 1.0, "the drought must be active for this test");
    deprioritise_remove_neuron_candidates(&mut candidates, factor);

    let promoted =
        promote_constant_remove_neuron_candidates(&mut candidates, &flagged(&["constant"]));
    assert_eq!(
        promoted, 1,
        "the flagged constant neuron is promoted despite the drought"
    );

    rank(&mut candidates);
    assert_eq!(
        target(&candidates[0]),
        "constant",
        "the promoted constant neuron must rank first even during a drought"
    );
}

/// Regression mode 3: deterministic ordering.
///
/// Two runs over the same flagged-constant + gain-ranked candidate mix must
/// yield an identical interleaving order.
#[test]
fn promotion_ordering_is_deterministic() {
    let build = || {
        vec![
            remove_neuron_candidate("constant", 0.0),
            positive_gain_candidate("live", 0.002),
            positive_gain_candidate("out-0", 0.001),
            remove_neuron_candidate("live", 0.0),
        ]
    };

    let order = |mut candidates: Vec<CoordinatedStructuralCandidateJson>| {
        promote_constant_remove_neuron_candidates(&mut candidates, &flagged(&["constant"]));
        rank(&mut candidates);
        candidates
            .iter()
            .map(|c| target(c).to_string())
            .collect::<Vec<_>>()
    };

    assert_eq!(
        order(build()),
        order(build()),
        "the interleaving order must be identical across runs"
    );
}

/// Over-promotion guard: an *unflagged* ≈0-gain remove-neuron candidate must
/// NOT be promoted, so it cannot flood the queue and displace genuine
/// positive-gain removals.
#[test]
fn unflagged_zero_gain_neuron_is_not_promoted() {
    let creature = creature();
    let mut candidates = vec![
        remove_neuron_candidate("constant", 0.0),
        positive_gain_candidate("live", 0.002),
    ];

    apply_honest_remove_neuron_gain(&creature, &mut candidates);

    // Empty flag set — the detector flagged nothing this pass.
    let promoted = promote_constant_remove_neuron_candidates(&mut candidates, &flagged(&[]));
    assert_eq!(promoted, 0, "nothing is promoted when no neuron is flagged");

    rank(&mut candidates);
    assert_eq!(
        target(&candidates[0]),
        "live",
        "the genuine positive-gain candidate must still rank first"
    );
    let constant_gain = candidates
        .iter()
        .find(|c| target(c) == "constant")
        .expect("constant candidate present")
        .expected_creature_score_gain;
    assert!(
        constant_gain < CONSTANT_NEURON_PRIORITY_GAIN,
        "an unflagged neuron keeps its honest ≈0 gain, not the priority gain"
    );
}
