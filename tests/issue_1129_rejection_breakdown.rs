//! Integration tests for Issue #1129 — structured rejection-reason breakdown
//! emitted on `synapseMetadata` / `neuronMetadata` even when zero candidates
//! survive.
//!
//! Two scenarios are covered:
//! 1. **Floor-dominated**: every coordinated candidate from a discovery module
//!    falls below `COORDINATED_MIN_EXPECTED_GAIN` so the floor filter consumes
//!    them all. `rejection_breakdown` records a `below_expected_gain_floor`
//!    count and `top_level_summary` names that floor.
//! 2. **Mixed reasons**: a combination of below-floor drops and budget-truncated
//!    drops produces separate counts in the breakdown.

use neat_ai_discovery::analysis::candidate_aggregation::apply_coordinated_gain_floor;
use neat_ai_discovery::analysis::diagnostics::rejection_reasons::{
    REJECTION_BELOW_EXPECTED_GAIN_FLOOR, REJECTION_BUDGET_TRUNCATED,
    REJECTION_INTERFERENCE_FILTERED, RejectionBreakdown, top_level_summary,
};
use neat_ai_discovery::analysis::discovery_dispatch::{
    DiscoveryDetectionResult, DiscoveryModuleSpec, run_discovery_modules_parallel,
};
use neat_ai_discovery::analysis::module_weights::ModuleOutcomeTracker;
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
        metadata: SynapseAnalysisMetadata::default(),
    }
}

fn single_op_candidate(gain: f32, module: &str) -> CoordinatedStructuralCandidateJson {
    CoordinatedStructuralCandidateJson {
        remove_neuron_compensation: None,
        operations: vec![CoordinatedStructuralOpJson::RemoveSynapse {
            from_neuron_uuid: "a".to_string(),
            to_neuron_uuid: "b".to_string(),
        }],
        expected_creature_score_gain: gain,
        comment: Some(module.to_string()),
    }
}

/// Floor-dominated case: every candidate from a discovery module sits well
/// below `COORDINATED_MIN_EXPECTED_GAIN`, so the pre-merge floor consumes
/// them all. The metadata must record a `below_expected_gain_floor` count
/// and `top_level_summary` must name the floor (so operators see the
/// single dominant rejection reason at a glance).
#[test]
fn floor_dominated_case_records_below_expected_gain_floor() {
    let mut syn = empty_synapse_result();
    let mut tracker = ModuleOutcomeTracker::new();

    // 32 candidates all at 5e-7 — well below the 1e-5 floor.
    let modules = vec![DiscoveryModuleSpec {
        module_name: "noise-module".to_string(),
        phase_name: "test_floor_dominated",
        max_candidates: 0,
        detect_fn: Box::new(|| {
            let candidates = (0..32)
                .map(|_| single_op_candidate(5e-7, "noise"))
                .collect::<Vec<_>>();
            Some(DiscoveryDetectionResult {
                detected_count: 32,
                candidates,
            })
        }),
    }];

    run_discovery_modules_parallel(&mut syn, modules, None, false, &mut tracker);

    // All below-floor candidates should have been dropped.
    assert!(
        syn.coordinated_structural_candidates.is_empty(),
        "floor-dominated batch should yield zero survivors"
    );

    // The breakdown must credit the below-expected-gain-floor reason.
    let count = syn
        .metadata
        .rejection_breakdown
        .counts()
        .get(REJECTION_BELOW_EXPECTED_GAIN_FLOOR)
        .copied()
        .unwrap_or(0);
    assert!(
        count >= 32,
        "breakdown should show ≥32 floor rejections, got {count}: \
         {:?}",
        syn.metadata.rejection_breakdown.counts()
    );

    // Dominant reason + synthetic top_level_summary must mention the floor.
    let (dominant, _) = syn
        .metadata
        .rejection_breakdown
        .dominant_reason()
        .expect("dominant reason present");
    assert_eq!(dominant, REJECTION_BELOW_EXPECTED_GAIN_FLOOR);

    let summary = top_level_summary(&syn.metadata.rejection_breakdown, None)
        .expect("summary populated when any rejection recorded");
    assert!(
        summary.contains("expected-gain floor"),
        "summary should mention the expected-gain floor, got: {summary}"
    );
}

/// Mixed-reason case: the breakdown surfaces multiple independent rejection
/// reasons so operators can see the whole story, not just the dominant one.
#[test]
fn mixed_reason_case_records_each_reason_independently() {
    let mut breakdown = RejectionBreakdown::new();
    breakdown.record_many_u32(REJECTION_BELOW_EXPECTED_GAIN_FLOOR, 12);
    breakdown.record_many_u32(REJECTION_INTERFERENCE_FILTERED, 5);
    breakdown.record_many_u32(REJECTION_BUDGET_TRUNCATED, 3);

    // All three reasons must be stored with exact counts.
    assert_eq!(
        breakdown
            .counts()
            .get(REJECTION_BELOW_EXPECTED_GAIN_FLOOR)
            .copied(),
        Some(12)
    );
    assert_eq!(
        breakdown
            .counts()
            .get(REJECTION_INTERFERENCE_FILTERED)
            .copied(),
        Some(5)
    );
    assert_eq!(
        breakdown.counts().get(REJECTION_BUDGET_TRUNCATED).copied(),
        Some(3)
    );
    assert_eq!(breakdown.total(), 20);

    // Top-level summary names the dominant reason (12 > 5 > 3).
    let summary = top_level_summary(&breakdown, Some(20)).expect("summary populated");
    assert!(
        summary.contains("12 of 20"),
        "summary should report 12 of 20 rejections, got: {summary}"
    );
    assert!(
        summary.contains("expected-gain floor"),
        "summary should name the dominant reason, got: {summary}"
    );
}

/// The pre-merge `apply_coordinated_gain_floor` helper returns the number of
/// candidates it dropped so the caller can feed it straight into the
/// structured breakdown.
#[test]
fn apply_coordinated_gain_floor_returns_drop_count() {
    // `apply_coordinated_gain_floor` uses the post-discount noise floor
    // (5e-7). Two of these are strictly below the floor.
    let mut cands = vec![
        single_op_candidate(1e-7, "below-1"),
        single_op_candidate(2e-7, "below-2"),
        single_op_candidate(1.0, "keep"),
    ];
    let dropped = apply_coordinated_gain_floor(&mut cands);
    assert_eq!(dropped, 2);
    assert_eq!(cands.len(), 1);
}
