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
//! This module promotes flagged neurons: it overwrites the flagged candidate's
//! `expected_creature_score_gain` with [`CONSTANT_NEURON_PRIORITY_GAIN`], so the
//! candidate ranks ahead of the normal gain-ranked stream instead of being ranked
//! ≈0 and dropped.
//!
//! Flags come from two live seams, unioned by the orchestrator:
//!
//! - **Structural** ([`functionally_constant_neuron_uuids`], wired by Issue
//!   #1813): hidden neurons whose output cannot vary given the topology alone —
//!   no incoming synapses, all incoming weights zero, or every source itself
//!   constant. This needs no recorded activations, so it reaches the case the
//!   measured seam structurally cannot.
//! - **Measured** ([`bias_folded_constant_neuron_uuids`], Issue #1779): the
//!   neurons whose removal candidate carries an **accepted** #1623 bias fold,
//!   verified against the recorded activations.
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

use std::collections::{HashMap, HashSet};

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

/// A neuron whose value is fixed by the topology alone — the seed set the
/// structural fixpoint grows from (Issue #1813).
///
/// Input neurons vary by definition; the NEAT-AI input-side `"constant"` class
/// does not, so it seeds the propagation without ever being flagged itself
/// (only hidden neurons are removal candidates).
fn is_declared_constant(neuron: &crate::NeuronJson) -> bool {
    neuron.neuron_type == "constant"
}

/// Collect the UUIDs of hidden neurons flagged **structurally**
/// functionally-constant for this analysis pass — the orchestrator's
/// consumption seam for the flag set (Issue #1622, wired by Issue #1813).
///
/// A hidden neuron is structurally constant when its output cannot vary given
/// the network topology alone, i.e. every incoming synapse either
///
/// - carries a zero weight (its contribution is `0 × anything = 0`), or
/// - originates at a neuron that is itself constant (a declared `"constant"`
///   neuron, or another structurally-constant hidden neuron);
///
/// a hidden neuron with **no** incoming synapses satisfies this vacuously — its
/// value is `squash(bias)` on every observation. Because the pre-activation is
/// fixed, the squash output is fixed too, for scalar and aggregate squashes
/// alike.
///
/// Constancy propagates to a fixpoint, so a chain of constant sources is flagged
/// end to end. Nothing else is: one varying source, or one source not present in
/// the creature (which cannot be proven constant), leaves the neuron unflagged.
///
/// This is deliberately a *structural* judgement and complements — rather than
/// replaces — the *measured* one made by [`bias_folded_constant_neuron_uuids`].
/// A neuron that merely *looks* constant over the recorded window is not flagged
/// here; that case belongs to the #1623 bias fold, which verifies it against the
/// activations.
#[must_use]
pub fn functionally_constant_neuron_uuids(creature: &CreatureJson) -> HashSet<String> {
    let mut constant: HashSet<&str> = creature
        .neurons
        .iter()
        .filter(|n| is_declared_constant(n))
        .map(|n| n.uuid.as_str())
        .collect();

    // Incoming synapses per target, so each fixpoint pass is a linear scan.
    let mut incoming: HashMap<&str, Vec<&crate::SynapseJson>> = HashMap::new();
    for synapse in &creature.synapses {
        incoming
            .entry(synapse.to_uuid.as_str())
            .or_default()
            .push(synapse);
    }

    let hidden: Vec<&str> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "hidden")
        .map(|n| n.uuid.as_str())
        .collect();

    // Monotone fixpoint: each pass can only add to `constant`, so at most one
    // pass per hidden neuron is ever needed.
    for _ in 0..=hidden.len() {
        let mut changed = false;
        for uuid in &hidden {
            if constant.contains(uuid) {
                continue;
            }
            let sources_fixed = incoming.get(uuid).is_none_or(|synapses| {
                synapses
                    .iter()
                    .all(|s| s.weight == 0.0 || constant.contains(s.from_uuid.as_str()))
            });
            if sources_fixed {
                constant.insert(uuid);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    hidden
        .into_iter()
        .filter(|uuid| constant.contains(uuid))
        .map(ToString::to_string)
        .collect()
}

/// Collect the UUIDs of neurons **measured** functionally constant this pass —
/// those whose sole-op `RemoveNeuron` candidate carries an accepted #1623 bias
/// fold (Issue #1779).
///
/// This is the verified flag source the #1622 promotion was waiting on. The fold
/// is only attached when the evaluate-before-accept gate passes against the
/// recorded activations, which is exactly the safety condition this module's
/// promotion relies on: the removal is behaviour-preserving because the neuron's
/// fixed contribution is folded into its targets' biases. A candidate with no
/// fold — a variance-carrying neuron, one that only *looks* constant, or one with
/// no recorded activations — is never flagged, so nothing is promoted on
/// assumption.
///
/// Complements (rather than replaces) the structural
/// [`functionally_constant_neuron_uuids`] seam: the orchestrator unions the two.
#[must_use]
pub fn bias_folded_constant_neuron_uuids(
    candidates: &[CoordinatedStructuralCandidateJson],
) -> HashSet<String> {
    candidates
        .iter()
        .filter(|c| c.constant_neuron_bias_fold.is_some())
        .filter_map(|c| single_op_remove_neuron_uuid(c).map(ToString::to_string))
        .collect()
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

    /// Only *hidden* neurons are removal candidates, so a creature with none
    /// flags nothing even though its input-side neuron is a declared constant.
    ///
    /// Renamed from `seam_returns_empty_until_detector_is_wired` by Issue #1813:
    /// the assertion is unchanged, but the seam is now wired, so what it pins is
    /// the hidden-only scope rather than an unconditional empty set.
    #[test]
    fn seam_flags_nothing_without_hidden_neurons() {
        let creature: CreatureJson = serde_json::from_str(
            r#"{"input":1,"output":1,
                "neurons":[{"uuid":"input-0","type":"constant"},
                           {"uuid":"out-0","type":"output","squash":"IDENTITY"}],
                "synapses":[{"fromUUID":"input-0","toUUID":"out-0","weight":1.0}]}"#,
        )
        .expect("valid creature JSON");
        assert!(functionally_constant_neuron_uuids(&creature).is_empty());
    }

    /// The structural detector flags a topology-constant hidden neuron and
    /// leaves a variance-carrying one alone (Issue #1813).
    #[test]
    fn structural_detector_flags_only_topology_constant_neurons() {
        let creature: CreatureJson = serde_json::from_str(
            r#"{"input":1,"output":1,
                "neurons":[{"uuid":"in-0","type":"input","squash":"IDENTITY"},
                           {"uuid":"h-const","type":"hidden","squash":"TANH","bias":0.25},
                           {"uuid":"h-live","type":"hidden","squash":"RELU"},
                           {"uuid":"out-0","type":"output","squash":"IDENTITY"}],
                "synapses":[{"fromUUID":"in-0","toUUID":"h-live","weight":0.7},
                            {"fromUUID":"h-const","toUUID":"out-0","weight":0.5},
                            {"fromUUID":"h-live","toUUID":"out-0","weight":0.5}]}"#,
        )
        .expect("valid creature JSON");
        let flagged = functionally_constant_neuron_uuids(&creature);
        assert_eq!(flagged.len(), 1, "one constant neuron, got {flagged:?}");
        assert!(flagged.contains("h-const"));
    }

    /// An empty creature is handled without panicking and flags nothing.
    #[test]
    fn empty_creature_flags_nothing() {
        let creature: CreatureJson =
            serde_json::from_str(r#"{"input":0,"output":0,"neurons":[],"synapses":[]}"#)
                .expect("valid creature JSON");
        assert!(functionally_constant_neuron_uuids(&creature).is_empty());
    }

    /// A candidate carrying an accepted bias fold is the measured flag source
    /// (Issue #1779).
    #[test]
    fn bias_folded_candidate_is_flagged() {
        let mut folded = remove_candidate("c", -0.75);
        folded.constant_neuron_bias_fold = Some(crate::ConstantNeuronBiasFoldJson {
            constant_activation: 0.02,
            activation_variance: 0.0,
            max_residual: 0.0,
            folded_targets: vec![crate::FoldedBiasDeltaJson {
                target_neuron_uuid: "out-0".to_string(),
                bias_delta: 0.06,
            }],
        });
        let mut candidates = vec![folded, remove_candidate("varying", -0.1)];

        let flags = bias_folded_constant_neuron_uuids(&candidates);
        assert_eq!(flags.len(), 1, "only the folded candidate is flagged");
        assert!(flags.contains("c"));

        // …and the flag is what lifts it past the gain floor.
        let promoted = promote_constant_remove_neuron_candidates(&mut candidates, &flags);
        assert_eq!(promoted, 1);
        assert!(
            (candidates[0].expected_creature_score_gain - CONSTANT_NEURON_PRIORITY_GAIN).abs()
                < f32::EPSILON
        );
        assert!((candidates[1].expected_creature_score_gain + 0.1).abs() < f32::EPSILON);
    }

    /// No fold ⇒ no flag: nothing is promoted on assumption.
    #[test]
    fn unfolded_candidates_flag_nothing() {
        let candidates = vec![remove_candidate("a", -0.2), remove_candidate("b", 0.0)];
        assert!(bias_folded_constant_neuron_uuids(&candidates).is_empty());
        assert!(bias_folded_constant_neuron_uuids(&[]).is_empty());
    }

    /// A multi-op candidate is not a bare removal, so its fold (if any) does not
    /// flag the neuron.
    #[test]
    fn multi_op_folded_candidate_is_not_flagged() {
        let mut candidate = CoordinatedStructuralCandidateJson {
            remove_neuron_compensation: None,
            constant_neuron_bias_fold: None,
            operations: vec![
                CoordinatedStructuralOpJson::SetBias {
                    neuron_uuid: "out-0".to_string(),
                    bias: 0.1,
                },
                CoordinatedStructuralOpJson::RemoveNeuron {
                    neuron_uuid: "c".to_string(),
                },
            ],
            expected_creature_score_gain: 0.01,
            comment: None,
        };
        candidate.constant_neuron_bias_fold = Some(crate::ConstantNeuronBiasFoldJson {
            constant_activation: 0.0,
            activation_variance: 0.0,
            max_residual: 0.0,
            folded_targets: Vec::new(),
        });
        assert!(bias_folded_constant_neuron_uuids(&[candidate]).is_empty());
    }
}
