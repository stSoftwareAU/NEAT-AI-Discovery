//! Regression tests for Issue #2375 (sibling sites): a coordinated candidate
//! whose operation carries a non-finite weight or bias must be dropped even
//! when its `expected_creature_score_gain` is finite.
//!
//! Detectors such as `bias_perturbation`, `bounded_range`, `error_plateau`,
//! `observation_utilisation`, `operating_point`, `output_range_compression`,
//! `restricted_range`, `sentinel_gating` and `squash_weight_rescale` can overflow
//! an `f32` parameter to `±∞` under a finite gain. The shared gate
//! (`reject_non_finite_gains`) and the final selection gate
//! (`apply_final_coordinated_gain_floor`) every detector's candidates pass
//! through must drop them.

use neat_ai_discovery::analysis::candidate_aggregation::{
    apply_final_coordinated_gain_floor, reject_non_finite_gains,
};
use neat_ai_discovery::analysis::diagnostics::rejection_reasons::REJECTION_NON_FINITE_GAIN;
use neat_ai_discovery::analysis::discovery_mode::DiscoveryMode;
use neat_ai_discovery::analysis::shared::{AnalyzeSynapsesResult, SynapseAnalysisMetadata};
use neat_ai_discovery::analysis::synapse::post_processing::reject_non_finite_and_rank_coordinated_candidates;
use neat_ai_discovery::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson};

type OpBuilder = fn(f32) -> CoordinatedStructuralOpJson;

fn add_synapse(weight: f32) -> CoordinatedStructuralOpJson {
    CoordinatedStructuralOpJson::AddSynapse {
        from_neuron_uuid: "a".into(),
        to_neuron_uuid: "b".into(),
        weight,
    }
}

fn set_weight(weight: f32) -> CoordinatedStructuralOpJson {
    CoordinatedStructuralOpJson::SetWeight {
        from_neuron_uuid: "a".into(),
        to_neuron_uuid: "b".into(),
        weight,
    }
}

fn add_neuron(bias: f32) -> CoordinatedStructuralOpJson {
    CoordinatedStructuralOpJson::AddNeuron {
        neuron_uuid: "n".into(),
        neuron_type: "hidden".into(),
        squash: "TANH".into(),
        bias,
        insert_before_neuron_uuid: None,
    }
}

fn set_bias(bias: f32) -> CoordinatedStructuralOpJson {
    CoordinatedStructuralOpJson::SetBias {
        neuron_uuid: "n".into(),
        bias,
    }
}

const BUILDERS: [(&str, OpBuilder); 4] = [
    ("AddSynapse", add_synapse),
    ("SetWeight", set_weight),
    ("AddNeuron", add_neuron),
    ("SetBias", set_bias),
];

fn candidate(
    ops: Vec<CoordinatedStructuralOpJson>,
    label: &str,
) -> CoordinatedStructuralCandidateJson {
    CoordinatedStructuralCandidateJson {
        remove_neuron_compensation: None,
        constant_neuron_bias_fold: None,
        operations: ops,
        expected_creature_score_gain: 0.5,
        comment: Some(label.to_string()),
    }
}

fn labels(candidates: &[CoordinatedStructuralCandidateJson]) -> Vec<&str> {
    candidates
        .iter()
        .map(|c| c.comment.as_deref().unwrap_or(""))
        .collect()
}

/// Each operation kind with a numeric parameter is dropped for `±∞` and `NaN`,
/// while the same operation with a finite parameter (the positive control)
/// survives under the identical finite gain.
#[test]
fn every_parameterised_operation_kind_is_gated() {
    for (kind, build) in BUILDERS {
        let mut ok = vec![candidate(vec![build(1.5)], "finite")];
        assert_eq!(
            reject_non_finite_gains(&mut ok),
            0,
            "{kind}: finite control"
        );
        assert_eq!(labels(&ok), vec!["finite"], "{kind}: finite control kept");

        for bad in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
            let mut batch = vec![candidate(vec![build(bad)], "bad")];
            assert_eq!(
                reject_non_finite_gains(&mut batch),
                1,
                "{kind} with parameter {bad} must be dropped under a finite gain"
            );
            assert!(batch.is_empty(), "{kind} with {bad} survived");
        }
    }
}

/// A single non-finite parameter anywhere in a multi-operation candidate drops
/// the whole candidate; operations without a numeric parameter are ignored, and
/// finite siblings keep their order.
#[test]
fn bad_parameter_in_later_operation_drops_candidate_and_keeps_order() {
    let mut batch = vec![
        candidate(vec![add_neuron(0.1), add_synapse(0.2)], "keep-1"),
        candidate(vec![add_neuron(0.1), set_bias(f32::NEG_INFINITY)], "drop"),
        candidate(
            vec![
                CoordinatedStructuralOpJson::RemoveSynapse {
                    from_neuron_uuid: "a".into(),
                    to_neuron_uuid: "b".into(),
                },
                CoordinatedStructuralOpJson::ChangeSquash {
                    neuron_uuid: "n".into(),
                    squash: "RELU".into(),
                },
                CoordinatedStructuralOpJson::RemoveNeuron {
                    neuron_uuid: "n".into(),
                },
            ],
            "keep-2",
        ),
    ];
    assert_eq!(reject_non_finite_gains(&mut batch), 1);
    assert_eq!(labels(&batch), vec!["keep-1", "keep-2"]);
}

/// The synapse-stage gate delegates to the same check.
#[test]
fn post_processing_gate_drops_non_finite_operation_parameter() {
    let mut batch = vec![
        candidate(vec![set_weight(f32::INFINITY)], "drop"),
        candidate(vec![set_weight(0.3)], "keep"),
    ];
    assert_eq!(
        reject_non_finite_and_rank_coordinated_candidates(&mut batch),
        1
    );
    assert_eq!(labels(&batch), vec!["keep"]);
}

/// The final selection gate drops the candidate and records it in the
/// rejection breakdown rather than discarding it silently.
#[test]
fn final_gate_drops_and_records_non_finite_operation_parameter() {
    let mut syn = AnalyzeSynapsesResult {
        helpful_synapses: Vec::new(),
        harmful_synapses: Vec::new(),
        synapse_weight_updates: Vec::new(),
        coordinated_structural_candidates: vec![
            candidate(vec![add_neuron(f32::NEG_INFINITY)], "inf-bias"),
            candidate(vec![add_neuron(0.25)], "finite-bias"),
        ],
        candidate_clusters: Vec::new(),
        gpu_used: false,
        no_candidate_reasons: Vec::new(),
        metadata: SynapseAnalysisMetadata {
            candidates_found: 0,
            candidates_returned: 0,
            ..Default::default()
        },
    };

    apply_final_coordinated_gain_floor(&mut syn, DiscoveryMode::Normal, 1.0);

    assert_eq!(
        labels(&syn.coordinated_structural_candidates),
        vec!["finite-bias"]
    );
    assert_eq!(
        syn.metadata
            .rejection_breakdown
            .counts()
            .get(REJECTION_NON_FINITE_GAIN),
        Some(&1)
    );
}
