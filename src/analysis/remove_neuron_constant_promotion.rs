//! Priority promotion of functionally-constant hidden neurons as remove-neuron
//! candidates (Issue #1622).
//!
//! A zero-variance (functionally-constant) hidden neuron contributes nothing an
//! equivalent bias adjustment on its targets could not, yet two mechanisms keep
//! it alive indefinitely (the #1620 bug):
//!
//! 1. **Propagation-aware gain ranking (#1518,
//!    [`super::remove_neuron_gain`])** gives a zero-influence neuron a signed
//!    honest gain of ≈0, which never beats positive-gain candidates, so it is
//!    never selected.
//! 2. **Drought demotion (#1448, [`super::remove_neuron_drought`])** further
//!    deprioritises remove-neuron candidates during a search-exhaustion drought.
//!
//! This module promotes neurons the constant-neuron detector (a sibling
//! sub-issue of the #1620 milestone) has flagged: it overwrites the flagged
//! candidate's `expected_creature_score_gain` with
//! [`CONSTANT_NEURON_PRIORITY_GAIN`], so the candidate ranks ahead of the normal
//! gain-ranked stream instead of being ranked ≈0 and dropped.
//!
//! ## Where it hooks in
//!
//! The orchestrator runs this **after** the #1530 honest-gain override and the
//! #1448 drought demotion, so overwriting the flagged candidate's gain bypasses
//! *both* — whatever value those stages left is discarded for flagged neurons.
//! It runs **before** the final coordinated gain floor, so the priority gain
//! (well above every floor) guarantees the promoted candidate survives.
//!
//! ## Safety
//!
//! Promoting independent of the measured gain is safe: the score-preservation
//! guarantee is provided by the removal sub-issue's bias-fold + evaluate-before-
//! accept gate, not by a positive gain estimate. A constant neuron removed there
//! is folded into its targets' biases, so the network's behaviour is preserved.
//!
//! ## Determinism
//!
//! Promotion is a pure, order-preserving pass: it rewrites the gain of flagged
//! candidates in place and leaves every other candidate untouched, so the
//! downstream gain-descending sort (`total_cmp`, NaN-safe) yields an identical
//! interleaving for identical inputs.

use std::collections::HashSet;

use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson};

/// Priority expected-creature-score gain assigned to a flagged
/// functionally-constant remove-neuron candidate (Issue #1622).
///
/// `1.0` is the maximum meaningful score gain (score = 1 − error ∈ `[0, 1]`),
/// so it sorts a promoted candidate ahead of every realistic (~`1e-4`) and honest
/// (≈0 / negative) remove-neuron gain and comfortably clears the downstream
/// coordinated gain floors. It is a *ranking* marker, not a prediction: the
/// score-preservation guarantee comes from the removal sub-issue's evaluate-
/// before-accept gate, not from this value.
pub const CONSTANT_NEURON_PRIORITY_GAIN: f32 = 1.0;

/// If a candidate's sole operation is a `RemoveNeuron`, return the target neuron
/// UUID; otherwise `None`.
///
/// Only a lone `RemoveNeuron` op is a bare neuron removal. A multi-operation
/// coordinated candidate reflects the combined effect of the whole atomic group,
/// so promoting it would misrepresent that group — those are left untouched.
fn single_op_remove_neuron_uuid(candidate: &CoordinatedStructuralCandidateJson) -> Option<&str> {
    match candidate.operations.as_slice() {
        [CoordinatedStructuralOpJson::RemoveNeuron { neuron_uuid }] => Some(neuron_uuid.as_str()),
        _ => None,
    }
}

/// Promote every single-op `RemoveNeuron` candidate whose neuron is flagged
/// functionally-constant to the priority gain (Issue #1622).
///
/// For each candidate whose sole operation removes a neuron in
/// `flagged_constant_uuids`, the `expected_creature_score_gain` is overwritten
/// with [`CONSTANT_NEURON_PRIORITY_GAIN`], overriding whatever the #1518 honest
/// gain ranking and #1448 drought demotion left. Non-flagged candidates,
/// multi-operation candidates, and non-removal candidates are left exactly as
/// they are — so a genuine positive-gain removal is never displaced and an
/// unflagged ≈0-gain neuron is never promoted.
///
/// Returns the number of candidates promoted, for diagnostics.
pub fn promote_constant_remove_neuron_candidates(
    candidates: &mut [CoordinatedStructuralCandidateJson],
    flagged_constant_uuids: &HashSet<String>,
) -> u32 {
    if flagged_constant_uuids.is_empty() {
        return 0;
    }
    let mut promoted = 0_u32;
    for candidate in candidates.iter_mut() {
        let is_flagged = single_op_remove_neuron_uuid(candidate)
            .is_some_and(|uuid| flagged_constant_uuids.contains(uuid));
        if is_flagged {
            candidate.expected_creature_score_gain = CONSTANT_NEURON_PRIORITY_GAIN;
            promoted = promoted.saturating_add(1);
        }
    }
    promoted
}

/// Collect the UUIDs of hidden neurons flagged functionally-constant for this
/// analysis pass — the orchestrator's consumption seam for the flag set
/// (Issue #1622).
///
/// The functionally-constant detector (a sibling sub-issue of the #1620
/// milestone) owns flag production. Until it is wired in, no neuron is flagged
/// and the set is empty, so the downstream promotion promotes nothing. This is
/// a documented dependency gate, not a masked fault: an empty set means "no
/// constant neuron was detected this pass", which is the correct behaviour.
#[must_use]
pub fn functionally_constant_neuron_uuids(_creature: &CreatureJson) -> HashSet<String> {
    HashSet::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn remove_candidate(uuid: &str, gain: f32) -> CoordinatedStructuralCandidateJson {
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

    fn flagged(uuids: &[&str]) -> HashSet<String> {
        uuids.iter().map(|u| (*u).to_string()).collect()
    }

    #[test]
    fn promotes_flagged_constant_neuron() {
        let mut candidates = vec![remove_candidate("c", -0.0001)];
        let promoted = promote_constant_remove_neuron_candidates(&mut candidates, &flagged(&["c"]));
        assert_eq!(promoted, 1);
        assert!(
            (candidates[0].expected_creature_score_gain - CONSTANT_NEURON_PRIORITY_GAIN).abs()
                < f32::EPSILON
        );
    }

    #[test]
    fn does_not_promote_unflagged_neuron() {
        let mut candidates = vec![remove_candidate("c", -0.0001)];
        let promoted =
            promote_constant_remove_neuron_candidates(&mut candidates, &flagged(&["other"]));
        assert_eq!(promoted, 0);
        assert!((candidates[0].expected_creature_score_gain - -0.0001).abs() < f32::EPSILON);
    }

    #[test]
    fn empty_flag_set_is_a_no_op() {
        let mut candidates = vec![remove_candidate("c", 0.0)];
        let promoted = promote_constant_remove_neuron_candidates(&mut candidates, &flagged(&[]));
        assert_eq!(promoted, 0);
        assert_eq!(candidates[0].expected_creature_score_gain, 0.0);
    }

    #[test]
    fn multi_op_candidate_is_not_promoted() {
        // A flagged UUID but inside a multi-op group — not a bare removal.
        let mut candidates = vec![CoordinatedStructuralCandidateJson {
            remove_neuron_compensation: None,
            constant_neuron_bias_fold: None,
            operations: vec![
                CoordinatedStructuralOpJson::RemoveSynapse {
                    from_neuron_uuid: "input-0".to_string(),
                    to_neuron_uuid: "c".to_string(),
                },
                CoordinatedStructuralOpJson::RemoveNeuron {
                    neuron_uuid: "c".to_string(),
                },
            ],
            expected_creature_score_gain: 0.05,
            comment: None,
        }];
        let promoted = promote_constant_remove_neuron_candidates(&mut candidates, &flagged(&["c"]));
        assert_eq!(promoted, 0);
        assert!((candidates[0].expected_creature_score_gain - 0.05).abs() < f32::EPSILON);
    }

    #[test]
    fn non_removal_candidate_is_not_promoted() {
        let mut candidates = vec![CoordinatedStructuralCandidateJson {
            remove_neuron_compensation: None,
            constant_neuron_bias_fold: None,
            operations: vec![CoordinatedStructuralOpJson::SetBias {
                neuron_uuid: "c".to_string(),
                bias: 0.1,
            }],
            expected_creature_score_gain: 0.003,
            comment: None,
        }];
        let promoted = promote_constant_remove_neuron_candidates(&mut candidates, &flagged(&["c"]));
        assert_eq!(promoted, 0);
        assert!((candidates[0].expected_creature_score_gain - 0.003).abs() < f32::EPSILON);
    }

    #[test]
    fn only_flagged_candidate_in_a_mixed_batch_is_promoted() {
        let mut candidates = vec![remove_candidate("flag", 0.0), remove_candidate("keep", 0.0)];
        let promoted =
            promote_constant_remove_neuron_candidates(&mut candidates, &flagged(&["flag"]));
        assert_eq!(promoted, 1);
        assert!(
            (candidates[0].expected_creature_score_gain - CONSTANT_NEURON_PRIORITY_GAIN).abs()
                < f32::EPSILON
        );
        assert_eq!(candidates[1].expected_creature_score_gain, 0.0);
    }

    #[test]
    fn seam_returns_empty_until_detector_is_wired() {
        let creature: CreatureJson = serde_json::from_str(
            r#"{"input":1,"output":1,
                "neurons":[{"uuid":"input-0","type":"constant"},
                           {"uuid":"out-0","type":"output","squash":"IDENTITY"}],
                "synapses":[{"fromUUID":"input-0","toUUID":"out-0","weight":1.0}]}"#,
        )
        .expect("valid creature JSON");
        assert!(functionally_constant_neuron_uuids(&creature).is_empty());
    }
}
