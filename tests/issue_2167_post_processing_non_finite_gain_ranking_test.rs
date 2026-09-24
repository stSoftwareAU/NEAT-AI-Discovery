//! Regression tests for Issue #2167 — a non-finite expected gain must not rank
//! first in the synapse post-processing candidate sorts.
//!
//! `apply_post_processing` sorts its three candidate lists descending by
//! `expected_creature_score_gain` using `f32::total_cmp`. Under IEEE-754
//! totalOrder every finite value sits below `+∞`, which in turn sits below a
//! positive `NaN`, so a descending sort ranks a non-finite gain **first**. The
//! upstream filters do not stop it: `apply_min_expected_gain_floor_for_synapses`
//! retains `gain >= floor` and the coordinated retain keeps `gain > 0.0` — both
//! reject `NaN` but pass `+∞`. A single `+∞` gain therefore displaces every
//! genuine candidate from the head of the returned list.
//!
//! The fix applies the Issue #1367 non-finite gate to all three lists
//! immediately before the sorts, and — as in #1367 — *counts* the rejections
//! under `REJECTION_NON_FINITE_GAIN` rather than dropping them silently.

use neat_ai_discovery::CandidateSynapseJson;
use neat_ai_discovery::analysis::diagnostics::RejectionBreakdown;
use neat_ai_discovery::analysis::diagnostics::rejection_reasons::REJECTION_NON_FINITE_GAIN;
use neat_ai_discovery::analysis::synapse::post_processing::{
    reject_non_finite_and_rank_coordinated_candidates,
    reject_non_finite_and_rank_synapse_candidates, scale_by_error_fraction,
};
use neat_ai_discovery::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson};

/// Build a synapse candidate carrying `gain`, labelled with `label` so the
/// post-sort ordering can be asserted by identity rather than by value.
fn synapse_candidate(gain: f32, label: &str) -> CandidateSynapseJson {
    CandidateSynapseJson {
        from_neuron_uuid: "from".to_string(),
        to_neuron_uuid: "to".to_string(),
        from_neuron_index: None,
        to_neuron_index: None,
        weight: 0.1,
        target_neuron_impact: 1.0,
        expected_creature_error_reduction: gain,
        expected_creature_score_gain: gain,
        improved_count: 8,
        total_count: 10,
        improvement_magnitude_ratio: None,
        target_neuron_stats: None,
        prediction_confidence: 0.5,
        expected_score_gain_confidence_interval: [0.0, 0.0],
        comment: Some(label.to_string()),
        variant_key: None,
    }
}

fn coordinated_candidate(gain: f32, label: &str) -> CoordinatedStructuralCandidateJson {
    CoordinatedStructuralCandidateJson {
        remove_neuron_compensation: None,
        constant_neuron_bias_fold: None,
        operations: vec![CoordinatedStructuralOpJson::RemoveSynapse {
            from_neuron_uuid: "a".to_string(),
            to_neuron_uuid: "b".to_string(),
        }],
        expected_creature_score_gain: gain,
        comment: Some(label.to_string()),
    }
}

fn labels(candidates: &[CandidateSynapseJson]) -> Vec<&str> {
    candidates
        .iter()
        .map(|c| c.comment.as_deref().unwrap_or(""))
        .collect()
}

/// A crafted gain vector containing `+∞`, `NaN`, `-0.0` and several finite
/// gains must rank with no non-finite survivor at all, and the finite gains —
/// `-0.0` included, since it *is* finite — must come back in descending order.
///
/// Against the unfixed code this fails at the very first assertion: the sort
/// alone leaves `NaN` at position 0 and `+∞` at position 1.
#[test]
fn ranking_synapse_candidates_drops_non_finite_gains_and_keeps_finite_order() {
    let mut candidates = vec![
        synapse_candidate(0.25, "finite-0.25"),
        synapse_candidate(f32::INFINITY, "pos-inf"),
        synapse_candidate(-0.0, "negative-zero"),
        synapse_candidate(2.0, "finite-2.0"),
        synapse_candidate(f32::NAN, "nan"),
        synapse_candidate(0.5, "finite-0.5"),
        synapse_candidate(f32::NEG_INFINITY, "neg-inf"),
    ];

    let dropped = reject_non_finite_and_rank_synapse_candidates(&mut candidates);

    assert!(
        candidates
            .iter()
            .all(|c| c.expected_creature_score_gain.is_finite()),
        "no non-finite gain may survive ranking, got {:?}",
        labels(&candidates)
    );
    assert_eq!(dropped, 3, "+∞, NaN and -∞ are the three rejections");
    assert_eq!(
        labels(&candidates),
        vec!["finite-2.0", "finite-0.5", "finite-0.25", "negative-zero"],
        "finite gains must rank in descending order with the best first"
    );
}

/// The same gate and ordering contract applies to the coordinated-structural
/// list, which reaches the sort through a `> 0.0` retain that `+∞` passes.
#[test]
fn ranking_coordinated_candidates_drops_non_finite_gains_and_keeps_finite_order() {
    let mut candidates = vec![
        coordinated_candidate(0.25, "finite-0.25"),
        coordinated_candidate(f32::INFINITY, "pos-inf"),
        coordinated_candidate(1.5, "finite-1.5"),
        coordinated_candidate(f32::NAN, "nan"),
    ];

    let dropped = reject_non_finite_and_rank_coordinated_candidates(&mut candidates);

    assert_eq!(dropped, 2, "+∞ and NaN are the two rejections");
    let surviving: Vec<&str> = candidates
        .iter()
        .map(|c| c.comment.as_deref().unwrap_or(""))
        .collect();
    assert_eq!(
        surviving,
        vec!["finite-1.5", "finite-0.25"],
        "finite coordinated gains must rank in descending order"
    );
}

/// Every discarded non-finite gain must be *counted* under
/// `REJECTION_NON_FINITE_GAIN`, matching the Issue #1367 accounting, so the
/// drought diagnostic can distinguish a poisoned batch from a genuine drought.
#[test]
fn rejected_non_finite_gains_are_counted_under_the_non_finite_reason() {
    let mut helpful = vec![
        synapse_candidate(f32::INFINITY, "pos-inf"),
        synapse_candidate(0.5, "finite-0.5"),
    ];
    let mut harmful = vec![
        synapse_candidate(f32::NAN, "nan"),
        synapse_candidate(f32::NEG_INFINITY, "neg-inf"),
        synapse_candidate(0.125, "finite-0.125"),
    ];
    let mut coordinated = vec![
        coordinated_candidate(f32::NAN, "nan"),
        coordinated_candidate(0.75, "finite-0.75"),
    ];

    let dropped = reject_non_finite_and_rank_synapse_candidates(&mut helpful)
        + reject_non_finite_and_rank_synapse_candidates(&mut harmful)
        + reject_non_finite_and_rank_coordinated_candidates(&mut coordinated);

    let mut breakdown = RejectionBreakdown::new();
    breakdown.record_many_u32(REJECTION_NON_FINITE_GAIN, dropped);

    assert_eq!(
        breakdown.counts().get(REJECTION_NON_FINITE_GAIN),
        Some(&4),
        "every discarded non-finite gain must be counted, not silently dropped"
    );
}

/// A clean batch must record nothing and keep every candidate: the gate is a
/// filter on non-finite gains, not a general ranking tax.
#[test]
fn ranking_a_finite_batch_rejects_nothing() {
    let mut candidates = vec![
        synapse_candidate(0.1, "finite-0.1"),
        synapse_candidate(0.3, "finite-0.3"),
    ];

    let dropped = reject_non_finite_and_rank_synapse_candidates(&mut candidates);

    assert_eq!(dropped, 0, "a finite batch has nothing to reject");
    assert_eq!(labels(&candidates), vec!["finite-0.3", "finite-0.1"]);
}

/// `scale_by_error_fraction` must reject a non-finite `total_error_sq`
/// explicitly. `NaN <= EPSILON` is false, so the existing guard passes a `NaN`
/// total straight through to the division, and `f32::clamp` propagates the
/// `NaN` into the returned gain.
#[test]
fn scale_by_error_fraction_rejects_non_finite_inputs() {
    for (raw, target, total, case) in [
        (1.0_f32, 1.0_f32, f32::NAN, "NaN total"),
        (1.0, 1.0, f32::INFINITY, "infinite total"),
        (1.0, f32::NAN, 4.0, "NaN target"),
        (1.0, f32::INFINITY, 4.0, "infinite target"),
        (f32::NAN, 1.0, 4.0, "NaN raw prediction"),
        (f32::INFINITY, 1.0, 4.0, "infinite raw prediction"),
    ] {
        let scaled = scale_by_error_fraction(raw, target, total);
        assert_eq!(
            scaled, 0.0,
            "{case} must scale to a neutral 0.0, got {scaled}"
        );
    }
}

/// The finite path is untouched: a target contributing a quarter of the total
/// squared error keeps a quarter of the raw prediction.
#[test]
fn scale_by_error_fraction_still_scales_finite_inputs() {
    assert_eq!(scale_by_error_fraction(1.0, 1.0, 4.0), 0.25);
}
