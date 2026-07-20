//! Issue #1448 — deprioritise destructive remove-neuron candidates during a
//! search-exhaustion drought.
//!
//! On a plateaued dense creature the remove-neuron path dominates the failure
//! cache with low-impact proposals that never pass scoring. These tests verify
//! the public contract of
//! [`neat_ai_discovery::analysis::remove_neuron_drought`]:
//! - the deprioritisation engages only during an active *search-exhaustion*
//!   drought (not below threshold, and not for an environmental drought);
//! - it demotes only single-op `RemoveNeuron` coordinated candidates, leaving
//!   other change types untouched; and
//! - the configured factor is resolved and clamped from the environment.

use neat_ai_discovery::analysis::remove_neuron_drought::{
    DEFAULT_REMOVE_NEURON_DROUGHT_FACTOR, DroughtDeprioritisationInputs,
    MIN_REMOVE_NEURON_DROUGHT_FACTOR, NEUTRAL_DEPRIORITISATION_FACTOR,
    deprioritise_remove_neuron_candidates, is_single_op_remove_neuron,
    remove_neuron_deprioritisation_factor,
};
use neat_ai_discovery::config::resolve_remove_neuron_drought_factor;
use neat_ai_discovery::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson};

fn inputs(failures: u32, disabled: u32, threshold: u32) -> DroughtDeprioritisationInputs {
    DroughtDeprioritisationInputs {
        consecutive_failures: failures,
        environmentally_disabled_passes: disabled,
        drought_threshold: threshold,
    }
}

fn remove_neuron(uuid: &str, gain: f32) -> CoordinatedStructuralCandidateJson {
    CoordinatedStructuralCandidateJson {
        remove_neuron_compensation: None,
        constant_neuron_bias_fold: None,
        operations: vec![CoordinatedStructuralOpJson::RemoveNeuron {
            neuron_uuid: uuid.to_string(),
        }],
        expected_creature_score_gain: gain,
        comment: None,
    }
}

fn change_squash(uuid: &str, gain: f32) -> CoordinatedStructuralCandidateJson {
    CoordinatedStructuralCandidateJson {
        remove_neuron_compensation: None,
        constant_neuron_bias_fold: None,
        operations: vec![CoordinatedStructuralOpJson::ChangeSquash {
            neuron_uuid: uuid.to_string(),
            squash: "TANH".to_string(),
        }],
        expected_creature_score_gain: gain,
        comment: None,
    }
}

#[test]
fn factor_neutral_below_drought_threshold() {
    // Four trailing failures, threshold five — no active drought yet.
    let factor = remove_neuron_deprioritisation_factor(&inputs(4, 0, 5), 0.1);
    assert_eq!(factor, NEUTRAL_DEPRIORITISATION_FACTOR);
}

#[test]
fn factor_engages_during_search_exhaustion_drought() {
    let factor = remove_neuron_deprioritisation_factor(&inputs(10, 0, 5), 0.1);
    assert!((factor - 0.1).abs() < f32::EPSILON);
}

#[test]
fn factor_neutral_for_environmental_drought() {
    // Host could not evaluate most passes -> environmental -> not the
    // creature's fault, so remove-neuron is left untouched.
    let factor = remove_neuron_deprioritisation_factor(&inputs(6, 20, 5), 0.1);
    assert_eq!(factor, NEUTRAL_DEPRIORITISATION_FACTOR);
}

#[test]
fn configured_factor_of_one_disables_deprioritisation() {
    let factor = remove_neuron_deprioritisation_factor(&inputs(10, 0, 5), 1.0);
    assert_eq!(factor, NEUTRAL_DEPRIORITISATION_FACTOR);
}

#[test]
fn only_single_op_remove_neuron_is_classified() {
    assert!(is_single_op_remove_neuron(&remove_neuron("h1", 0.2)));
    assert!(!is_single_op_remove_neuron(&change_squash("h1", 0.2)));

    // A multi-op group containing a RemoveNeuron is NOT a single-op remove.
    let multi = CoordinatedStructuralCandidateJson {
        remove_neuron_compensation: None,
        constant_neuron_bias_fold: None,
        operations: vec![
            CoordinatedStructuralOpJson::RemoveNeuron {
                neuron_uuid: "h1".to_string(),
            },
            CoordinatedStructuralOpJson::AddSynapse {
                from_neuron_uuid: "a".to_string(),
                to_neuron_uuid: "b".to_string(),
                weight: 0.5,
            },
        ],
        expected_creature_score_gain: 0.2,
        comment: None,
    };
    assert!(!is_single_op_remove_neuron(&multi));
}

#[test]
fn deprioritise_demotes_only_remove_neuron_candidates() {
    let mut candidates = vec![
        remove_neuron("h1", 0.2),
        change_squash("h2", 0.2),
        remove_neuron("h3", 0.05),
    ];

    let demoted = deprioritise_remove_neuron_candidates(&mut candidates, 0.1);
    assert_eq!(demoted, 2, "both remove-neuron candidates demoted");

    // Remove-neuron gains scaled by 0.1; the change-squash gain is untouched.
    assert!((candidates[0].expected_creature_score_gain - 0.02).abs() < 1e-6);
    assert!((candidates[1].expected_creature_score_gain - 0.2).abs() < 1e-6);
    assert!((candidates[2].expected_creature_score_gain - 0.005).abs() < 1e-6);
}

#[test]
fn deprioritise_is_noop_for_neutral_factor() {
    let mut candidates = vec![remove_neuron("h1", 0.2)];
    let demoted =
        deprioritise_remove_neuron_candidates(&mut candidates, NEUTRAL_DEPRIORITISATION_FACTOR);
    assert_eq!(demoted, 0);
    assert!((candidates[0].expected_creature_score_gain - 0.2).abs() < f32::EPSILON);
}

#[test]
fn deprioritise_skips_non_positive_gains() {
    // A zero / negative gain is left for the dedicated non-positive filter.
    let mut candidates = vec![remove_neuron("h1", 0.0), remove_neuron("h2", -0.3)];
    let demoted = deprioritise_remove_neuron_candidates(&mut candidates, 0.1);
    assert_eq!(demoted, 0);
    assert!((candidates[0].expected_creature_score_gain - 0.0).abs() < f32::EPSILON);
    assert!((candidates[1].expected_creature_score_gain + 0.3).abs() < f32::EPSILON);
}

#[test]
fn end_to_end_drought_deprioritisation_demotes_remove_neuron() {
    // The orchestrator's flow: resolve the factor from the drought signal, then
    // apply it. During a search-exhaustion drought the remove-neuron gain is
    // demoted while the constructive candidate keeps its gain.
    let factor = remove_neuron_deprioritisation_factor(&inputs(12, 0, 5), 0.1);
    let mut candidates = vec![remove_neuron("h1", 0.2), change_squash("h2", 0.15)];
    let demoted = deprioritise_remove_neuron_candidates(&mut candidates, factor);

    assert_eq!(demoted, 1);
    assert!(
        candidates[0].expected_creature_score_gain < candidates[1].expected_creature_score_gain,
        "remove-neuron now sorts below the constructive change type"
    );
}

#[test]
fn resolve_factor_defaults_and_clamps() {
    // Unset / unparsable -> default.
    assert!(
        (resolve_remove_neuron_drought_factor(None, DEFAULT_REMOVE_NEURON_DROUGHT_FACTOR)
            - DEFAULT_REMOVE_NEURON_DROUGHT_FACTOR)
            .abs()
            < f32::EPSILON
    );
    assert!(
        (resolve_remove_neuron_drought_factor(Some("abc"), DEFAULT_REMOVE_NEURON_DROUGHT_FACTOR)
            - DEFAULT_REMOVE_NEURON_DROUGHT_FACTOR)
            .abs()
            < f32::EPSILON
    );

    // Valid override is honoured.
    assert!(
        (resolve_remove_neuron_drought_factor(Some("0.25"), DEFAULT_REMOVE_NEURON_DROUGHT_FACTOR)
            - 0.25)
            .abs()
            < f32::EPSILON
    );

    // Below the floor clamps up; above 1.0 clamps down.
    assert!(
        (resolve_remove_neuron_drought_factor(Some("0.0"), DEFAULT_REMOVE_NEURON_DROUGHT_FACTOR)
            - MIN_REMOVE_NEURON_DROUGHT_FACTOR)
            .abs()
            < f32::EPSILON
    );
    assert!(
        (resolve_remove_neuron_drought_factor(Some("5.0"), DEFAULT_REMOVE_NEURON_DROUGHT_FACTOR)
            - NEUTRAL_DEPRIORITISATION_FACTOR)
            .abs()
            < f32::EPSILON
    );
}
