//! Overflowing operation parameters must not reach coordinated results.
use neat_ai_discovery::analysis::candidate_aggregation::reject_non_finite_operations;
use neat_ai_discovery::analysis::detection::{
    bias_perturbation, bounded_range, error_plateau, observation_utilisation, operating_point,
    output_range_compression, restricted_range, sentinel_gating,
};
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{
    CoordinatedStructuralCandidateJson, CreatureJson, NeuronJson, SynapseJson,
};

fn creature(weight: f32) -> CreatureJson {
    CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![
            NeuronJson {
                uuid: "input-0".into(),
                neuron_type: "input".into(),
                squash: "IDENTITY".into(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "output-0".into(),
                neuron_type: "output".into(),
                squash: "TANH".into(),
                bias: 0.0,
            },
        ],
        synapses: vec![SynapseJson {
            from_uuid: "input-0".into(),
            to_uuid: "output-0".into(),
            weight,
            synapse_type: None,
        }],
    }
}
fn filtered(mut candidates: Vec<CoordinatedStructuralCandidateJson>, expected: u32) {
    assert!(!candidates.is_empty(), "fixture must produce candidates");
    assert!(
        candidates
            .iter()
            .all(|c| c.expected_creature_score_gain.is_finite())
    );
    assert_eq!(reject_non_finite_operations(&mut candidates), expected);
    assert_eq!(reject_non_finite_operations(&mut candidates), 0);
}
#[test]
fn bounded_range_and_sentinel_gate_bias_overflow() {
    let c = creature(1.0);
    let b = |v| bounded_range::BoundedRangeCandidate {
        neuron_uuid: "input-0".into(),
        boundary_value: 0.0,
        boundary_fraction: 0.5,
        useful_range_min: v,
        useful_range_max: v,
        sample_count: 100,
        detection_confidence: 0.5,
        estimated_improvement: 0.005,
    };
    filtered(
        bounded_range::bounded_range_to_coordinated_candidates(&[b(3e38)], &c),
        1,
    );
    filtered(
        bounded_range::bounded_range_to_coordinated_candidates(&[b(1.0)], &c),
        0,
    );
    let s = |v| sentinel_gating::SentinelGatingCandidate {
        neuron_uuid: "input-0".into(),
        sentinel_value: v,
        sentinel_fraction: 0.5,
        useful_range_min: v,
        useful_range_max: v,
        sample_count: 100,
        estimated_improvement: 0.005,
    };
    filtered(
        sentinel_gating::sentinel_gating_to_coordinated_candidates(&[s(3e38)], &c),
        1,
    );
    filtered(
        sentinel_gating::sentinel_gating_to_coordinated_candidates(&[s(1.0)], &c),
        0,
    );
}
#[test]
fn observation_bias_overflow() {
    let c = creature(1.0);
    let o = |v| observation_utilisation::UnderutilisedObservation {
        neuron_uuid: "input-0".into(),
        sentinel_values: vec![0.0],
        utilisation_ratio: 0.1,
        effective_min: v,
        effective_max: v,
        sample_count: 100,
        estimated_improvement: 0.005,
        reason: String::new(),
    };
    filtered(
        observation_utilisation::observation_utilisation_to_coordinated_candidates(&[o(3e38)], &c),
        1,
    );
    filtered(
        observation_utilisation::observation_utilisation_to_coordinated_candidates(&[o(1.0)], &c),
        0,
    );
}
#[test]
fn operating_point_bias_and_weight_overflow() {
    let issue = |v| operating_point::OperatingPointIssue {
        neuron_uuid: "output-0".into(),
        squash: "LOGISTIC".into(),
        bias: 0.0,
        value_min: v,
        value_max: v,
        active_zone_min: -4.0,
        active_zone_max: 4.0,
        dynamic_range_utilisation: 0.0,
        sample_count: 100,
    };
    filtered(
        operating_point::operating_point_to_coordinated_candidates(&[issue(3e38)], &creature(1e36)),
        2,
    );
    filtered(
        operating_point::operating_point_to_coordinated_candidates(&[issue(1.0)], &creature(0.5)),
        0,
    );
}
#[test]
fn range_scaling_weight_and_bias_overflow() {
    let o = output_range_compression::OutputRangeCompressionNeuron {
        neuron_uuid: "output-0".into(),
        squash: "TANH".into(),
        bias: 0.0,
        activation_min: 0.0,
        activation_max: 0.01,
        theoretical_range: 2.0,
        range_utilisation: 0.01,
        sample_count: 100,
    };
    filtered(
        output_range_compression::output_range_compression_to_coordinated_candidates(
            std::slice::from_ref(&o),
            &creature(1e37),
        ),
        1,
    );
    filtered(
        output_range_compression::output_range_compression_to_coordinated_candidates(
            &[o],
            &creature(0.5),
        ),
        0,
    );
    let r = restricted_range::RestrictedRangeNeuron {
        neuron_uuid: "output-0".into(),
        squash: "TANH".into(),
        bias: 1e37,
        activation_min: 0.0,
        activation_max: 0.01,
        theoretical_range: 2.0,
        range_utilisation: 0.01,
        sample_count: 100,
    };
    filtered(
        restricted_range::restricted_range_to_coordinated_candidates(
            std::slice::from_ref(&r),
            &creature(1e37),
        ),
        1,
    );
    let mut control = r;
    control.bias = 0.0;
    filtered(
        restricted_range::restricted_range_to_coordinated_candidates(&[control], &creature(0.5)),
        0,
    );
}
#[test]
fn bias_perturbation_detector_overflow() {
    let neurons = vec![("output-0".into(), "TANH".into(), 0.0)];
    let records = |v| {
        vec![(
            "output-0".into(),
            (0..30)
                .map(|i| DiscoverRecord::new(i, "output-0".into(), Some(v), 1.0, vec![0.5]))
                .collect::<Vec<_>>(),
        )]
    };
    let detected = bias_perturbation::detect_bias_perturbation_candidates(&neurons, &records(3e38));
    filtered(
        bias_perturbation::bias_perturbation_to_coordinated_candidates(&detected),
        1,
    );
    let detected = bias_perturbation::detect_bias_perturbation_candidates(&neurons, &records(10.0));
    filtered(
        bias_perturbation::bias_perturbation_to_coordinated_candidates(&detected),
        0,
    );
}
#[test]
fn error_plateau_bias_overflow() {
    let candidate = error_plateau::ErrorPlateauCandidate {
        neuron_uuid: "output-0".into(),
        current_squash: "TANH".into(),
        recommended_squash: "IDENTITY".into(),
        current_bias: 3.4e38,
        recommended_bias: 3.4e38_f32 - (-1.7e37_f32),
        mean_error: 0.5,
        error_std_dev: 0.0,
        error_coefficient_of_variation: 0.0,
        confidence: 1.0,
        estimated_improvement: 0.005,
        reason: String::new(),
    };
    filtered(
        error_plateau::error_plateaus_to_coordinated_candidates(std::slice::from_ref(&candidate)),
        1,
    );
    let mut control = candidate;
    control.current_bias = 0.0;
    control.recommended_bias = 1.0;
    filtered(
        error_plateau::error_plateaus_to_coordinated_candidates(&[control]),
        0,
    );
}

#[test]
fn dispatch_rejects_overflow_before_capping_and_records_reason() {
    use neat_ai_discovery::analysis::diagnostics::rejection_reasons::REJECTION_NON_FINITE_OPERATION;
    use neat_ai_discovery::analysis::discovery_dispatch::{
        DiscoveryDetectionResult, run_discovery_module,
    };
    use neat_ai_discovery::analysis::shared::{AnalyzeSynapsesResult, SynapseAnalysisMetadata};
    let mut syn = AnalyzeSynapsesResult {
        helpful_synapses: vec![],
        harmful_synapses: vec![],
        synapse_weight_updates: vec![],
        coordinated_structural_candidates: vec![],
        candidate_clusters: vec![],
        gpu_used: false,
        no_candidate_reasons: vec![],
        metadata: SynapseAnalysisMetadata::default(),
    };
    let issue = |v| operating_point::OperatingPointIssue {
        neuron_uuid: "output-0".into(),
        squash: "LOGISTIC".into(),
        bias: 0.0,
        value_min: v,
        value_max: v,
        active_zone_min: -4.0,
        active_zone_max: 4.0,
        dynamic_range_utilisation: 0.0,
        sample_count: 100,
    };
    let mut candidates =
        operating_point::operating_point_to_coordinated_candidates(&[issue(3e38)], &creature(1e36));
    candidates.extend(operating_point::operating_point_to_coordinated_candidates(
        &[issue(1.0)],
        &creature(0.5),
    ));
    run_discovery_module(&mut syn, "test", "test", None, false, || {
        Some(DiscoveryDetectionResult {
            detected_count: 2,
            candidates,
        })
    });
    assert_eq!(
        syn.metadata
            .rejection_breakdown
            .counts()
            .get(REJECTION_NON_FINITE_OPERATION),
        Some(&2)
    );
    assert!(!syn.coordinated_structural_candidates.is_empty());
    assert_eq!(
        reject_non_finite_operations(&mut syn.coordinated_structural_candidates),
        0
    );
}

#[test]
fn parallel_merge_rejects_overflow_before_module_budget() {
    use neat_ai_discovery::analysis::diagnostics::rejection_reasons::REJECTION_NON_FINITE_OPERATION;
    use neat_ai_discovery::analysis::discovery_dispatch::{
        DiscoveryDetectionResult, DiscoveryModuleDetectionEntry, DiscoveryModuleDetectionResults,
        merge_discovery_module_results,
    };
    use neat_ai_discovery::analysis::module_weights::ModuleOutcomeTracker;
    use neat_ai_discovery::analysis::shared::{AnalyzeSynapsesResult, SynapseAnalysisMetadata};
    let mut syn = AnalyzeSynapsesResult {
        helpful_synapses: vec![],
        harmful_synapses: vec![],
        synapse_weight_updates: vec![],
        coordinated_structural_candidates: vec![],
        candidate_clusters: vec![],
        gpu_used: false,
        no_candidate_reasons: vec![],
        metadata: SynapseAnalysisMetadata::default(),
    };
    let c = creature(1.0);
    let b = |v| bounded_range::BoundedRangeCandidate {
        neuron_uuid: "input-0".into(),
        boundary_value: 0.0,
        boundary_fraction: 0.5,
        useful_range_min: v,
        useful_range_max: v,
        sample_count: 100,
        detection_confidence: 0.5,
        estimated_improvement: 0.005,
    };
    let candidates = bounded_range::bounded_range_to_coordinated_candidates(&[b(3e38), b(1.0)], &c);
    let results = DiscoveryModuleDetectionResults {
        entries: vec![DiscoveryModuleDetectionEntry {
            module_name: "test".into(),
            phase_name: "test",
            max_candidates: 1,
            result: Some(DiscoveryDetectionResult {
                detected_count: 2,
                candidates,
            }),
            skip_reason: None,
        }],
    };
    merge_discovery_module_results(
        &mut syn,
        results,
        None,
        false,
        &mut ModuleOutcomeTracker::new(),
    );
    assert_eq!(
        syn.metadata
            .rejection_breakdown
            .counts()
            .get(REJECTION_NON_FINITE_OPERATION),
        Some(&1)
    );
    assert_eq!(syn.coordinated_structural_candidates.len(), 1);
    assert_eq!(
        reject_non_finite_operations(&mut syn.coordinated_structural_candidates),
        0
    );
}

#[test]
fn final_guard_rejects_all_non_finite_operation_variants() {
    use neat_ai_discovery::CoordinatedStructuralOpJson as Op;
    use neat_ai_discovery::analysis::candidate_aggregation::apply_final_coordinated_gain_floor;
    use neat_ai_discovery::analysis::diagnostics::rejection_reasons::REJECTION_NON_FINITE_OPERATION;
    use neat_ai_discovery::analysis::discovery_mode::DiscoveryMode;
    use neat_ai_discovery::analysis::shared::{AnalyzeSynapsesResult, SynapseAnalysisMetadata};
    let ops = vec![
        Op::AddSynapse {
            from_neuron_uuid: "a".into(),
            to_neuron_uuid: "b".into(),
            weight: f32::NAN,
        },
        Op::SetWeight {
            from_neuron_uuid: "a".into(),
            to_neuron_uuid: "b".into(),
            weight: f32::INFINITY,
        },
        Op::AddNeuron {
            neuron_uuid: "n".into(),
            neuron_type: "hidden".into(),
            squash: "TANH".into(),
            bias: f32::NEG_INFINITY,
            insert_before_neuron_uuid: None,
        },
        Op::SetBias {
            neuron_uuid: "n".into(),
            bias: f32::NAN,
        },
        Op::SetBias {
            neuron_uuid: "n".into(),
            bias: 1.0,
        },
    ];
    let candidates = ops
        .into_iter()
        .map(|op| CoordinatedStructuralCandidateJson {
            operations: vec![op],
            expected_creature_score_gain: 0.1,
            comment: None,
            remove_neuron_compensation: None,
            constant_neuron_bias_fold: None,
        })
        .collect();
    let mut syn = AnalyzeSynapsesResult {
        helpful_synapses: vec![],
        harmful_synapses: vec![],
        synapse_weight_updates: vec![],
        coordinated_structural_candidates: candidates,
        candidate_clusters: vec![],
        gpu_used: false,
        no_candidate_reasons: vec![],
        metadata: SynapseAnalysisMetadata::default(),
    };
    apply_final_coordinated_gain_floor(&mut syn, DiscoveryMode::Normal, 1.0);
    assert_eq!(syn.coordinated_structural_candidates.len(), 1);
    assert_eq!(
        syn.metadata
            .rejection_breakdown
            .counts()
            .get(REJECTION_NON_FINITE_OPERATION),
        Some(&4)
    );
    assert_eq!(syn.metadata.candidates_returned, 1);
}

#[test]
fn error_plateau_detector_overflow_reaches_the_guard() {
    let records = vec![(
        "output-0".into(),
        (0..20)
            .map(|i| DiscoverRecord::new(i, "output-0".into(), Some(0.0), 0.0, vec![-1e37]))
            .collect::<Vec<_>>(),
    )];
    let neurons = vec![("output-0".into(), "TANH".into(), 3.4e38)];
    let detected = error_plateau::detect_error_plateaus(&neurons, &records);
    filtered(
        error_plateau::error_plateaus_to_coordinated_candidates(&detected),
        1,
    );
}
