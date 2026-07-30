//! Integration tests for Issue #1446 — surface `zeroCandidateSummary` on the
//! analysis FFI response when a discovery pass produces no candidates.
//!
//! When discovery finds nothing, operators previously saw an unhelpful "Built 0
//! candidates" block with no visible reason. The Rust side already populates
//! `rejection_breakdown`, `drought_diagnostic`, and `creature_drought_alarm` in
//! analysis metadata (#1129, #1202, #1424); this builds a single
//! `zeroCandidateSummary` object so the dominant rejection reason is identifiable
//! without opening `.discovery/` JSON sidecars.

use std::collections::HashMap;

use neat_ai_discovery::analysis::EnvironmentalDisableReason;
use neat_ai_discovery::analysis::diagnostics::RejectionBreakdown;
use neat_ai_discovery::analysis::diagnostics::rejection_reasons::{
    REJECTION_NO_SAMPLES, REJECTION_NO_TARGET_RECORDS,
};
use neat_ai_discovery::analysis::shared::{NeuronAnalysisMetadata, SynapseAnalysisMetadata};
use neat_ai_discovery::{EnvironmentalGatesJson, build_zero_candidate_summary};

fn gates() -> EnvironmentalGatesJson {
    EnvironmentalGatesJson {
        memory_budget_exceeded: false,
        memory_pressure_cancelled: false,
        cancelled: false,
        environmentally_disabled: None,
    }
}

/// Acceptance criterion: a fixture whose dominant synapse rejection is
/// `no_target_records` populates `dominantRejectionReason: "no_target_records"`.
#[test]
fn no_target_records_fixture_sets_dominant_reason() {
    let mut synapse = SynapseAnalysisMetadata::default();
    synapse
        .rejection_breakdown
        .record_many_u32(REJECTION_NO_TARGET_RECORDS, 4);
    synapse
        .rejection_breakdown
        .record_many_u32(REJECTION_NO_SAMPLES, 1);

    let summary =
        build_zero_candidate_summary(Some(&synapse), None, &RejectionBreakdown::new(), gates());

    assert_eq!(
        summary.dominant_rejection_reason.as_deref(),
        Some("no_target_records"),
        "dominant reason should be the most-frequent rejection"
    );
    assert_eq!(
        summary
            .rejection_breakdown
            .get("no_target_records")
            .copied(),
        Some(4)
    );
    assert_eq!(
        summary.rejection_breakdown.get("no_samples").copied(),
        Some(1)
    );
}

/// The breakdown merges synapse and neuron rejection counts so the dominant
/// reason reflects the whole pass, not just one analysis half.
#[test]
fn merges_synapse_and_neuron_breakdowns() {
    let mut synapse = SynapseAnalysisMetadata::default();
    synapse
        .rejection_breakdown
        .record_many_u32(REJECTION_NO_TARGET_RECORDS, 2);

    let mut neuron = NeuronAnalysisMetadata::default();
    neuron
        .rejection_breakdown
        .record_many_u32(REJECTION_NO_TARGET_RECORDS, 3);
    neuron
        .rejection_breakdown
        .record_many_u32(REJECTION_NO_SAMPLES, 1);

    let summary = build_zero_candidate_summary(
        Some(&synapse),
        Some(&neuron),
        &RejectionBreakdown::new(),
        gates(),
    );

    assert_eq!(
        summary
            .rejection_breakdown
            .get("no_target_records")
            .copied(),
        Some(5),
        "synapse (2) + neuron (3) no_target_records counts must merge"
    );
    assert_eq!(
        summary.dominant_rejection_reason.as_deref(),
        Some("no_target_records")
    );
}

/// Environmental gates flow through so operators can tell a host-gated pass
/// (memory / GPU / cancellation) apart from genuine search exhaustion.
#[test]
fn environmental_gates_are_surfaced() {
    let synapse = SynapseAnalysisMetadata::default();
    let env_gates = EnvironmentalGatesJson {
        memory_budget_exceeded: true,
        memory_pressure_cancelled: false,
        cancelled: false,
        environmentally_disabled: Some(EnvironmentalDisableReason::MemoryGated),
    };

    let summary =
        build_zero_candidate_summary(Some(&synapse), None, &RejectionBreakdown::new(), env_gates);

    assert!(summary.environmental_gates.memory_budget_exceeded);
    assert_eq!(
        summary.environmental_gates.environmentally_disabled,
        Some(EnvironmentalDisableReason::MemoryGated)
    );
    // No rejections recorded → no dominant reason, but the summary still
    // explains the zero-candidate outcome via the environmental gate.
    assert!(summary.dominant_rejection_reason.is_none());
}

/// The summary serialises with the camelCase field names documented in
/// `docs/FFI_API.md` so the TypeScript host can read them directly.
#[test]
fn serialises_with_camel_case_field_names() {
    let mut synapse = SynapseAnalysisMetadata::default();
    synapse
        .rejection_breakdown
        .record_many_u32(REJECTION_NO_TARGET_RECORDS, 4);

    let summary =
        build_zero_candidate_summary(Some(&synapse), None, &RejectionBreakdown::new(), gates());
    let value: serde_json::Value =
        serde_json::to_value(&summary).expect("summary serialises to JSON");

    assert_eq!(
        value
            .get("dominantRejectionReason")
            .and_then(serde_json::Value::as_str),
        Some("no_target_records")
    );
    assert!(
        value.get("rejectionBreakdown").is_some(),
        "rejectionBreakdown must be present"
    );
    let env = value
        .get("environmentalGates")
        .expect("environmentalGates present");
    assert_eq!(
        env.get("memoryBudgetExceeded")
            .and_then(serde_json::Value::as_bool),
        Some(false)
    );

    // Round-trip the breakdown to confirm counts survive serialisation.
    let breakdown: HashMap<String, u32> =
        serde_json::from_value(value.get("rejectionBreakdown").unwrap().clone()).unwrap();
    assert_eq!(breakdown.get("no_target_records").copied(), Some(4));
}
