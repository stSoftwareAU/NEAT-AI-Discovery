use super::*;
use crate::{
    CandidateNeuronJson, CandidateSynapseJson, CoordinatedStructuralCandidateJson,
    CoordinatedStructuralOpJson,
};

#[test]
fn watchdog_beats_do_not_claim_finished_when_analysis_is_skipped() {
    let _lock = crate::watchdog::lock_for_test_serialisation();
    // Ensure a watchdog is active so `beat()` is observable in tests.
    let wd = crate::watchdog::Watchdog::start(crate::watchdog::WatchdogConfig {
        stall_timeout: std::time::Duration::from_secs(60),
        abort_delay: std::time::Duration::from_secs(1),
    });

    let skipped = "analysis::analyze_all → neuron analysis skipped";
    let finished = "analysis::analyze_all → neuron analysis finished";

    // When disabled, we should record "skipped" and never execute the closure.
    let result: Option<()> = run_optional_analysis(
        false,
        "starting",
        finished,
        skipped,
        "test_phase",
        || -> Result<()> { unreachable!("disabled analysis closure must not run") },
    )
    .expect("should not error");
    assert!(result.is_none());
    assert_eq!(
        crate::watchdog::active_stage_for_test().as_deref(),
        Some(skipped)
    );

    // When enabled, we should end on "finished".
    let result: Option<()> =
        run_optional_analysis(true, "starting", finished, "skipped", "test_phase", || {
            Ok(())
        })
        .expect("should not error");
    assert!(result.is_some());
    assert_eq!(
        crate::watchdog::active_stage_for_test().as_deref(),
        Some(finished)
    );

    drop(wd);
}

#[test]
fn choose_deadline_order_is_deterministic_for_fixed_inputs() {
    let now_ms = 1_700_000_000_000u64;
    assert_eq!(
        choose_deadline_order_synapse_first(Some(123), now_ms),
        choose_deadline_order_synapse_first(Some(123), now_ms)
    );
}

#[test]
fn choose_deadline_order_varies_over_time() {
    // The chooser mixes in epoch-ms, so adjacent milliseconds should flip the result.
    let now_ms = 1_700_000_000_000u64;
    assert_ne!(
        choose_deadline_order_synapse_first(Some(0), now_ms),
        choose_deadline_order_synapse_first(Some(0), now_ms + 1)
    );
}

#[test]
fn choose_deadline_order_can_be_controlled_by_seed() {
    // With a fixed time, different seeds should be able to flip ordering.
    let now_ms = 1_700_000_000_000u64;
    assert_ne!(
        choose_deadline_order_synapse_first(Some(0), now_ms),
        choose_deadline_order_synapse_first(Some(1), now_ms)
    );
}

#[test]
fn postprocess_reapplies_max_synapse_candidates_after_coordinated_merge() {
    // Regression test (7-Jan-2026):
    // Neuron→coordinated conversion happens after synapse analysis truncation, so we must
    // re-apply `maxSynapseCandidates` (Issue #173 follow-up).

    let mut syn = shared::AnalyzeSynapsesResult {
        helpful_synapses: Vec::new(),
        harmful_synapses: Vec::new(),
        synapse_weight_updates: Vec::new(),
        coordinated_structural_candidates: Vec::new(),
        gpu_used: false,
        no_candidate_reasons: Vec::new(),
        metadata: shared::SynapseAnalysisMetadata {
            candidates_found: 0,
            candidates_returned: 0,
            ..Default::default()
        },
    };

    let replacements = vec![CoordinatedStructuralCandidateJson {
        operations: vec![CoordinatedStructuralOpJson::RemoveSynapse {
            from_neuron_uuid: "a".to_string(),
            to_neuron_uuid: "b".to_string(),
        }],
        expected_creature_score_gain: 1.0,
        comment: Some("test replacement".to_string()),
    }];

    merge_coordinated_structural_replacements(&mut syn, replacements, Some(0), false);

    assert!(
        syn.coordinated_structural_candidates.is_empty(),
        "expected coordinated candidates to be truncated to 0 when maxSynapseCandidates=0"
    );
    assert_eq!(syn.metadata.candidates_returned, 0);
}

#[test]
fn postprocess_truncates_across_all_synapse_buckets_not_just_coordinated() {
    // Ensure we keep the global cap semantics (helpful + harmful + coordinated <= max).

    let mut syn = shared::AnalyzeSynapsesResult {
        helpful_synapses: vec![CandidateSynapseJson {
            from_neuron_uuid: "x".to_string(),
            to_neuron_uuid: "y".to_string(),
            from_neuron_index: None,
            to_neuron_index: None,
            weight: 0.1,
            target_neuron_impact: 1.0,
            expected_creature_error_reduction: 0.1,
            expected_creature_score_gain: 0.1,
            improved_count: 1,
            total_count: 1,
            target_neuron_stats: None,
            outlier_reduction_info: None,
        }],
        harmful_synapses: Vec::new(),
        synapse_weight_updates: Vec::new(),
        coordinated_structural_candidates: Vec::new(),
        gpu_used: false,
        no_candidate_reasons: Vec::new(),
        metadata: shared::SynapseAnalysisMetadata {
            candidates_found: 1,
            candidates_returned: 1,
            ..Default::default()
        },
    };

    let replacements = vec![CoordinatedStructuralCandidateJson {
        operations: vec![CoordinatedStructuralOpJson::RemoveSynapse {
            from_neuron_uuid: "a".to_string(),
            to_neuron_uuid: "b".to_string(),
        }],
        expected_creature_score_gain: 2.0, // better than the helpful candidate
        comment: None,
    }];

    merge_coordinated_structural_replacements(&mut syn, replacements, Some(1), false);

    let total = syn.helpful_synapses.len()
        + syn.harmful_synapses.len()
        + syn.coordinated_structural_candidates.len();
    assert_eq!(total, 1);
    assert_eq!(syn.helpful_synapses.len(), 0);
    assert_eq!(syn.coordinated_structural_candidates.len(), 1);
    assert_eq!(syn.metadata.candidates_returned, 1);
}

#[test]
fn postprocess_updates_candidates_returned_after_merging_neuron_replacements() {
    // Regression test (7 Jan 2026):
    // When neuron candidates are converted to coordinated structural replacements during
    // post-processing, we must update `synapseMetadata.candidatesReturned` to include them.
    //
    // This would fail under the old behaviour where we used `extend()` without recomputing
    // the metadata count.

    let mut syn = shared::AnalyzeSynapsesResult {
        helpful_synapses: vec![CandidateSynapseJson {
            from_neuron_uuid: "x".to_string(),
            to_neuron_uuid: "y".to_string(),
            from_neuron_index: None,
            to_neuron_index: None,
            weight: 0.1,
            target_neuron_impact: 1.0,
            expected_creature_error_reduction: 0.1,
            expected_creature_score_gain: 0.1,
            improved_count: 1,
            total_count: 1,
            target_neuron_stats: None,
            outlier_reduction_info: None,
        }],
        harmful_synapses: Vec::new(),
        synapse_weight_updates: Vec::new(),
        coordinated_structural_candidates: Vec::new(),
        gpu_used: false,
        no_candidate_reasons: Vec::new(),
        metadata: shared::SynapseAnalysisMetadata {
            candidates_found: 1,
            candidates_returned: 1,
            ..Default::default()
        },
    };

    let replacements = vec![CoordinatedStructuralCandidateJson {
        operations: vec![CoordinatedStructuralOpJson::RemoveSynapse {
            from_neuron_uuid: "a".to_string(),
            to_neuron_uuid: "b".to_string(),
        }],
        expected_creature_score_gain: 1.0,
        comment: Some("test replacement".to_string()),
    }];

    merge_coordinated_structural_replacements(&mut syn, replacements, None, false);

    assert_eq!(syn.helpful_synapses.len(), 1);
    assert_eq!(syn.coordinated_structural_candidates.len(), 1);
    assert_eq!(
        syn.metadata.candidates_returned, 2,
        "expected candidates_returned to include coordinated candidates merged during post-processing"
    );
}

#[test]
fn postprocess_updates_neuron_candidates_returned_after_filtering_replacements() {
    // Regression test (7-Jan-2026):
    // When we convert some add-neuron candidates into coordinated structural replacements,
    // we filter them out of `helpful_neurons`. The metadata must be updated to reflect the
    // new returned count, otherwise JSON output becomes misleading.

    let cand_a = CandidateNeuronJson {
        source_neuron_uuid: "s1".to_string(),
        target_neuron_uuid: "t1".to_string(),
        source_neuron_index: None,
        target_neuron_index: None,
        incoming_weight: 0.1,
        outgoing_weight: 0.2,
        squash: "relu".to_string(),
        bias: 0.0,
        comment: None,
        target_neuron_impact: 1.0,
        expected_creature_error_reduction: 0.01,
        expected_creature_score_gain: 0.01,
        improved_count: 1,
        total_count: 1,
        target_neuron_stats: None,
    };

    let cand_b = CandidateNeuronJson {
        source_neuron_uuid: "s2".to_string(),
        target_neuron_uuid: "t2".to_string(),
        source_neuron_index: None,
        target_neuron_index: None,
        incoming_weight: 0.1,
        outgoing_weight: 0.2,
        squash: "relu".to_string(),
        bias: 0.0,
        comment: None,
        target_neuron_impact: 1.0,
        expected_creature_error_reduction: 0.01,
        expected_creature_score_gain: 0.01,
        improved_count: 1,
        total_count: 1,
        target_neuron_stats: None,
    };

    let mut neuron = shared::AnalyzeNeuronsResult {
        helpful_neurons: vec![cand_a.clone(), cand_b],
        gpu_used: false,
        no_candidate_reasons: Vec::new(),
        metadata: shared::NeuronAnalysisMetadata {
            candidates_found: 2,
            candidates_returned: 2,
            ..Default::default()
        },
    };

    apply_kept_neuron_candidates(&mut neuron, vec![cand_a]);

    assert_eq!(neuron.helpful_neurons.len(), 1);
    assert_eq!(neuron.metadata.candidates_returned, 1);
}
