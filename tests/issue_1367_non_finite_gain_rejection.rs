//! Regression tests for Issue #1367 — reject non-finite expected gains before
//! ranking and returning candidates.
//!
//! Candidate ranking sorts by `expected_creature_score_gain` using `total_cmp`,
//! which orders a positive `NaN` **above** `+∞`. A candidate whose gain is
//! `NaN` or `±∞` would therefore sort to the top and be returned as the *best*
//! candidate — wasting the controller's ablation-test budget on a meaningless
//! candidate. The library's contract is to only return candidates expected to
//! improve the creature's score; a non-finite gain is not a valid positive
//! improvement and must be dropped (and the drop recorded in the rejection
//! breakdown) before reranking / final selection.

use neat_ai_discovery::analysis::candidate_aggregation::{
    apply_final_coordinated_gain_floor, reject_non_finite_gains,
};
use neat_ai_discovery::analysis::diagnostics::rejection_reasons::REJECTION_NON_FINITE_GAIN;
use neat_ai_discovery::analysis::discovery_mode::DiscoveryMode;
use neat_ai_discovery::analysis::shared::{AnalyzeSynapsesResult, SynapseAnalysisMetadata};
use neat_ai_discovery::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson};

fn empty_synapse_result() -> AnalyzeSynapsesResult {
    AnalyzeSynapsesResult {
        helpful_synapses: Vec::new(),
        harmful_synapses: Vec::new(),
        synapse_weight_updates: Vec::new(),
        coordinated_structural_candidates: Vec::new(),
        candidate_clusters: Vec::new(),
        gpu_used: false,
        no_candidate_reasons: Vec::new(),
        metadata: SynapseAnalysisMetadata {
            candidates_found: 0,
            candidates_returned: 0,
            ..Default::default()
        },
    }
}

fn single_op_candidate(gain: f32, label: &str) -> CoordinatedStructuralCandidateJson {
    CoordinatedStructuralCandidateJson {
        operations: vec![CoordinatedStructuralOpJson::RemoveSynapse {
            from_neuron_uuid: "a".to_string(),
            to_neuron_uuid: "b".to_string(),
        }],
        expected_creature_score_gain: gain,
        comment: Some(label.to_string()),
    }
}

/// The pure filter removes `NaN` and `±∞` candidates, preserves the relative
/// order of the finite survivors, and returns the dropped count.
#[test]
fn reject_non_finite_gains_drops_non_finite_and_preserves_order() {
    let mut candidates = vec![
        single_op_candidate(1.0, "finite-1"),
        single_op_candidate(f32::NAN, "nan"),
        single_op_candidate(0.5, "finite-2"),
        single_op_candidate(f32::INFINITY, "pos-inf"),
        single_op_candidate(0.25, "finite-3"),
        single_op_candidate(f32::NEG_INFINITY, "neg-inf"),
    ];

    let dropped = reject_non_finite_gains(&mut candidates);

    assert_eq!(dropped, 3, "three non-finite candidates should be dropped");
    let labels: Vec<&str> = candidates
        .iter()
        .map(|c| c.comment.as_deref().unwrap_or(""))
        .collect();
    assert_eq!(
        labels,
        vec!["finite-1", "finite-2", "finite-3"],
        "finite candidates must survive in their original order"
    );
}

/// The final selection gate (`apply_final_coordinated_gain_floor`) must drop
/// `NaN` / `+∞` candidates and record the drop in the rejection breakdown so it
/// stays observable, while leaving the finite candidates ranked normally.
#[test]
fn final_gate_rejects_non_finite_and_records_breakdown() {
    let mut syn = empty_synapse_result();
    syn.coordinated_structural_candidates = vec![
        single_op_candidate(f32::NAN, "nan"),
        single_op_candidate(1.0, "finite-high"),
        single_op_candidate(f32::INFINITY, "pos-inf"),
        single_op_candidate(0.5, "finite-mid"),
        single_op_candidate(0.25, "finite-low"),
    ];

    apply_final_coordinated_gain_floor(&mut syn, DiscoveryMode::Normal, 1.0);

    // Neither non-finite candidate survives.
    let labels: Vec<&str> = syn
        .coordinated_structural_candidates
        .iter()
        .map(|c| c.comment.as_deref().unwrap_or(""))
        .collect();
    assert!(
        !labels.contains(&"nan") && !labels.contains(&"pos-inf"),
        "non-finite candidates must not be returned, got {labels:?}"
    );
    assert_eq!(
        labels.len(),
        3,
        "exactly the three finite candidates should remain, got {labels:?}"
    );
    assert!(
        labels.contains(&"finite-high")
            && labels.contains(&"finite-mid")
            && labels.contains(&"finite-low"),
        "all finite candidates should survive, got {labels:?}"
    );

    // The rejection breakdown reflects the dropped non-finite count.
    assert_eq!(
        syn.metadata
            .rejection_breakdown
            .counts()
            .get(REJECTION_NON_FINITE_GAIN),
        Some(&2),
        "breakdown should record two non-finite drops"
    );
}
