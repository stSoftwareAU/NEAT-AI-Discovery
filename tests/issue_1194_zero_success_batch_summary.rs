//! Integration tests for Issue #1194 — structured zero-success batch summary
//! events with failure-cluster diagnostics.
//!
//! These tests exercise the public API surface only:
//! `ZeroSuccessBatchSummary::aggregate` over a synthetic failure cache and
//! the `emit_zero_success_batch_summary` helper. They verify the behaviour
//! called out in the acceptance criteria:
//!
//! 1. A synthetic batch with zero accepted candidates produces an event with
//!    aggregates matching the input failure cache (cluster detection).
//! 2. An empty failure cache (representing "no batch was evaluated") yields
//!    no event.
//! 3. JSON serialisation produces the documented camelCase field names.

use neat_ai_discovery::analysis::scoring::calibration_correction::FailureCacheEntry;
use neat_ai_discovery::observability::{
    ZeroSuccessBatchSummary, emit_zero_success_batch_summary, maybe_emit_zero_success_batch_summary,
};

/// Helper that builds a failure-cache entry with all fields populated.
fn fc_entry(
    change_type: &str,
    expected: f32,
    actual: f32,
    squash: Option<&str>,
    variant: Option<&str>,
    target_uuid: Option<&str>,
) -> FailureCacheEntry {
    FailureCacheEntry {
        change_type: change_type.to_string(),
        expected_error_reduction: expected,
        actual_error_reduction: actual,
        target_squash: squash.map(str::to_string),
        variant_key: variant.map(str::to_string),
        target_uuid: target_uuid.map(str::to_string),
        improved_count: None,
        total_count: None,
        age_epochs: None,
    }
}

/// Acceptance: the canonical failure cluster from Issue #1189 — three
/// failures targeting the same neuron with the same change type — is
/// captured in the summary's distinct counts and median ratio.
#[test]
fn zero_success_batch_summary_captures_failure_cluster() {
    let cache = vec![
        fc_entry(
            "add-neurons",
            0.001,
            -0.5,
            Some("SELU"),
            Some("gentle-nudge"),
            Some("neuron-A"),
        ),
        fc_entry(
            "add-neurons",
            0.002,
            -0.4,
            Some("SELU"),
            Some("gentle-nudge"),
            Some("neuron-A"),
        ),
        fc_entry(
            "add-neurons",
            0.003,
            -0.3,
            Some("SELU"),
            Some("gentle-nudge"),
            Some("neuron-A"),
        ),
    ];

    let summary = ZeroSuccessBatchSummary::aggregate(123, &cache);

    assert_eq!(summary.batch_id, 123);
    assert_eq!(summary.candidate_count, 3);
    assert_eq!(
        summary.distinct_target_uuids, 1,
        "all three target the same neuron"
    );
    assert_eq!(summary.distinct_change_types, 1);
    assert_eq!(summary.distinct_target_squashes, 1);
    assert_eq!(summary.distinct_variant_keys, 1);

    // Median expected = 0.002, median actual = -0.4 → ratio = -200.
    let ratio = summary
        .median_actual_over_expected
        .expect("ratio defined for finite non-zero median expected");
    assert!(
        (ratio - (-200.0)).abs() < 1e-3,
        "expected median ratio ≈ -200, got {ratio}"
    );
}

/// Acceptance: an entry without a `target_uuid` does not contribute to the
/// distinct-target count, but is still counted in `candidate_count`.
#[test]
fn zero_success_batch_summary_handles_missing_uuids() {
    let cache = vec![
        fc_entry("add-neurons", 1.0, 0.5, None, None, None),
        fc_entry("add-synapses", 2.0, 1.0, Some("SELU"), None, Some("n-1")),
    ];

    let summary = ZeroSuccessBatchSummary::aggregate(0, &cache);

    assert_eq!(summary.candidate_count, 2);
    assert_eq!(summary.distinct_change_types, 2);
    assert_eq!(summary.distinct_target_uuids, 1);
    assert_eq!(summary.distinct_target_squashes, 1);
    assert_eq!(summary.distinct_variant_keys, 0);
}

/// Acceptance: the helper returns `None` for an empty failure cache so that
/// no event is emitted when there is nothing to summarise.
#[test]
fn emit_helper_returns_none_for_empty_cache() {
    let result = emit_zero_success_batch_summary(&[]);
    assert!(result.is_none(), "no event when cache is empty");
}

/// Acceptance: the helper returns the summary for a non-empty cache, and the
/// returned summary has the expected aggregates so callers can assert on it.
#[test]
fn emit_helper_returns_summary_for_non_empty_cache() {
    let cache = vec![fc_entry(
        "add-neurons",
        0.001,
        -0.5,
        Some("SELU"),
        Some("gentle-nudge"),
        Some("neuron-A"),
    )];
    let summary =
        emit_zero_success_batch_summary(&cache).expect("non-empty cache yields a summary");
    assert_eq!(summary.candidate_count, 1);
    assert_eq!(summary.distinct_target_uuids, 1);
    assert_eq!(summary.distinct_change_types, 1);
}

/// Acceptance: when at least one candidate was accepted, no event is emitted
/// even if the failure cache is non-empty.
#[test]
fn no_event_when_at_least_one_candidate_accepted() {
    let cache = vec![fc_entry(
        "add-neurons",
        0.001,
        -0.5,
        Some("SELU"),
        Some("gentle-nudge"),
        Some("neuron-A"),
    )];
    let result = maybe_emit_zero_success_batch_summary(1, &cache);
    assert!(result.is_none(), "no event when accepted > 0");
}

/// Acceptance: when zero candidates were accepted and the failure cache is
/// non-empty, the event is emitted exactly once with the expected
/// aggregates.
#[test]
fn event_emitted_when_zero_accepted_and_cache_non_empty() {
    let cache = vec![
        fc_entry("add-neurons", 0.001, -0.5, None, None, Some("neuron-A")),
        fc_entry("add-neurons", 0.002, -0.4, None, None, Some("neuron-A")),
    ];
    let summary = maybe_emit_zero_success_batch_summary(0, &cache)
        .expect("zero accepted + non-empty cache emits");
    assert_eq!(summary.candidate_count, 2);
    assert_eq!(summary.distinct_target_uuids, 1);
}

/// Acceptance: serialising the summary to JSON produces camelCase field
/// names so downstream tooling can match on `event = "zero_success_batch"`
/// alongside the documented field schema.
#[test]
fn summary_serialises_with_camel_case_fields() {
    let cache = vec![fc_entry("add-neurons", 1.0, 0.5, None, None, None)];
    let summary = ZeroSuccessBatchSummary::aggregate(7, &cache);
    let json = serde_json::to_value(&summary).expect("serialise");

    assert_eq!(json["batchId"], 7);
    assert_eq!(json["candidateCount"], 1);
    assert!(json["medianExpected"].is_number());
    assert!(json["medianActual"].is_number());
    assert!(json["medianActualOverExpected"].is_number());
    assert!(json["distinctTargetUuids"].is_number());
    assert!(json["distinctTargetSquashes"].is_number());
    assert!(json["distinctChangeTypes"].is_number());
    assert!(json["distinctVariantKeys"].is_number());
}

/// Acceptance: `target_uuid` is parsed from `targetNeuronInfo.uuid` in the
/// failure-cache JSON shape NEAT-AI emits, ensuring the upstream payload
/// contributes to the distinct-target count.
#[test]
fn target_uuid_parsed_from_target_neuron_info_uuid() {
    let json = serde_json::json!({
        "changeType": "add-neurons",
        "expectedErrorReduction": 0.001,
        "actualErrorReduction": -0.5,
        "targetNeuronInfo": {
            "uuid": "neuron-A",
            "squash": "SELU"
        }
    });
    let entry: FailureCacheEntry = serde_json::from_value(json).expect("parse");
    assert_eq!(entry.target_uuid.as_deref(), Some("neuron-A"));
    assert_eq!(entry.target_squash.as_deref(), Some("SELU"));
}

/// Acceptance: top-level `targetUuid` is also accepted (alternate shape).
#[test]
fn target_uuid_parsed_from_top_level_field() {
    let json = serde_json::json!({
        "changeType": "add-neurons",
        "expectedErrorReduction": 0.001,
        "actualErrorReduction": -0.5,
        "targetUuid": "neuron-B"
    });
    let entry: FailureCacheEntry = serde_json::from_value(json).expect("parse");
    assert_eq!(entry.target_uuid.as_deref(), Some("neuron-B"));
}
