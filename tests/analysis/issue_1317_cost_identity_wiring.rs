//! Tests for Issue #1317: Wire the cost identity into the
//! target-reconstruction guard.
//!
//! The dispatch path threads a [`CostFunctionHint`] derived from the
//! incoming cost-function name (via [`TaskDescriptor`]) into the two
//! detectors that reconstruct an implied target from `activation ± error`:
//!
//! - `output_squash_mismatch::detect_output_squash_mismatches` (Strategy 4).
//! - `high_error_squash_exploration::detect_high_error_squash_candidates`.
//!
//! These tests verify the wiring at three layers:
//!
//! 1. [`TaskDescriptor::cost_function_hint`] maps each recognised cost name
//!    onto the right [`CostFunctionHint`] variant, with `OTHER` / unknown /
//!    neutral collapsing to a conservative skip (`NonLinearResidual`).
//! 2. The per-detector cost-hint contract from Issue #1250 — already
//!    exercised by `tests/detection/issue_1250_implied_target_cost_hint.rs`
//!    — keeps holding once the dispatch wiring is in place.
//! 3. The FFI request structs now accept an optional `cost_name` field and
//!    serde round-trips it.

#![allow(clippy::cast_precision_loss)]

use neat_ai_discovery::analysis::cost_function_hint::CostFunctionHint;
use neat_ai_discovery::analysis::detection::high_error_squash_exploration::detect_high_error_squash_candidates_with_cost_hint;
use neat_ai_discovery::analysis::task_descriptor::TaskDescriptor;
use neat_ai_discovery::types::DiscoverRecord;

// =============================================================================
// Helpers
// =============================================================================

fn make_record(
    uuid: &str,
    idx: u32,
    value: Option<f32>,
    activation: f32,
    error: f32,
) -> DiscoverRecord {
    DiscoverRecord {
        obs_index: idx,
        neuron_uuid: uuid.to_string(),
        value,
        activation,
        errors: vec![error],
    }
}

/// 50 high-error TANH samples where the recorded "target" is a linear ramp
/// (the MSE convention used by the existing Issue #1250 test fixtures).
fn high_error_records(uuid: &str) -> Vec<DiscoverRecord> {
    (0..50)
        .map(|i| {
            let pre_act = (i as f32 - 25.0) / 8.0;
            let output = pre_act.tanh();
            let target = pre_act;
            let error = output - target;
            make_record(uuid, i, Some(pre_act), output, error)
        })
        .collect()
}

// =============================================================================
// 1. TaskDescriptor → CostFunctionHint mapping
// =============================================================================

#[test]
fn task_descriptor_maps_linear_costs_to_linear_residual() {
    for name in ["MSE", "MAE", "BINARY_CROSS_ENTROPY", "CROSS_ENTROPY"] {
        let descriptor = TaskDescriptor::from_name(name, 3);
        assert_eq!(
            descriptor.cost_function_hint(),
            CostFunctionHint::LinearResidual,
            "{name} descriptor must map to LinearResidual",
        );
    }
}

#[test]
fn task_descriptor_maps_non_linear_costs_to_non_linear_residual() {
    for name in ["MAPE", "MSLE", "HINGE", "CATEGORICAL_ERROR"] {
        let descriptor = TaskDescriptor::from_name(name, 3);
        assert_eq!(
            descriptor.cost_function_hint(),
            CostFunctionHint::NonLinearResidual,
            "{name} descriptor must map to NonLinearResidual",
        );
    }
}

#[test]
fn task_descriptor_for_other_or_neutral_is_conservative_skip() {
    // Issue #1317 acceptance: OTHER / Unknown / absent ⇒ conservative skip.
    // The conservative-skip value is `NonLinearResidual` — it gates the
    // reconstruction-dependent detectors off so they cannot emit spurious
    // candidates against an unknown cost shape.
    let cases = ["OTHER", "EXOTIC_LOSS", ""];
    for name in cases {
        let descriptor = TaskDescriptor::from_name(name, 3);
        assert_eq!(
            descriptor.cost_function_hint(),
            CostFunctionHint::NonLinearResidual,
            "{name:?} must map to a conservative skip",
        );
    }
    assert_eq!(
        TaskDescriptor::neutral().cost_function_hint(),
        CostFunctionHint::NonLinearResidual,
        "neutral descriptor must map to a conservative skip",
    );
}

// =============================================================================
// 2. Detector gating under the wired-up hints
// =============================================================================

#[test]
fn linear_descriptor_runs_high_error_squash_detector() {
    // A LinearResidual descriptor (MSE) must keep the implied-target
    // reconstruction enabled.
    let records = high_error_records("h1");
    let neurons = vec![("h1".to_string(), "TANH".to_string(), 0.0)];
    let neuron_records = vec![("h1".to_string(), records)];

    let hint = TaskDescriptor::from_name("MSE", 1).cost_function_hint();
    let candidates =
        detect_high_error_squash_candidates_with_cost_hint(&neurons, &neuron_records, hint);

    assert!(
        !candidates.is_empty(),
        "Linear-residual descriptor must keep high-error squash exploration running",
    );
}

#[test]
fn non_linear_descriptor_skips_high_error_squash_detector() {
    let records = high_error_records("h1");
    let neurons = vec![("h1".to_string(), "TANH".to_string(), 0.0)];
    let neuron_records = vec![("h1".to_string(), records)];

    for name in ["MAPE", "MSLE", "HINGE", "CATEGORICAL_ERROR"] {
        let hint = TaskDescriptor::from_name(name, 1).cost_function_hint();
        let candidates =
            detect_high_error_squash_candidates_with_cost_hint(&neurons, &neuron_records, hint);
        assert!(
            candidates.is_empty(),
            "{name}: non-linear descriptor must gate high-error squash exploration off",
        );
    }
}

#[test]
fn neutral_descriptor_skips_high_error_squash_detector() {
    // Issue #1317 acceptance: absent / OTHER / Unknown ⇒ conservative skip.
    let records = high_error_records("h1");
    let neurons = vec![("h1".to_string(), "TANH".to_string(), 0.0)];
    let neuron_records = vec![("h1".to_string(), records)];

    let hint = TaskDescriptor::neutral().cost_function_hint();
    let candidates =
        detect_high_error_squash_candidates_with_cost_hint(&neurons, &neuron_records, hint);
    assert!(
        candidates.is_empty(),
        "Neutral descriptor must conservatively gate high-error squash exploration off",
    );
}

// Note: `output_squash_mismatch` Strategy 4 gating is covered by
// `tests/detection/issue_1250_implied_target_cost_hint.rs`. The dispatch
// wiring here uses the same `detect_output_squash_mismatches_with_cost_hint`
// entry point, so no per-strategy duplication is needed.

// =============================================================================
// 3. FFI request structs accept an optional `cost_name`
// =============================================================================

#[test]
fn analyze_parallel_input_round_trips_cost_name() {
    // Issue #1317: FFI ingest carries the cost-function name into analysis.
    // A payload that omits it must default to `None`; one that supplies it
    // must round-trip the value.
    use neat_ai_discovery::AnalyzeParallelInput;

    let payload_without = r#"{
        "parquetFile": "/tmp/x.parquet",
        "creature": {"neurons": [], "synapses": [], "input": 0, "output": 0},
        "focusNeurons": []
    }"#;
    let parsed: AnalyzeParallelInput = serde_json::from_str(payload_without).expect("parse");
    assert!(
        parsed.cost_name.is_none(),
        "Absent cost_name must deserialize to None",
    );

    let payload_with = r#"{
        "parquetFile": "/tmp/x.parquet",
        "creature": {"neurons": [], "synapses": [], "input": 0, "output": 0},
        "focusNeurons": [],
        "costName": "MAPE"
    }"#;
    let parsed: AnalyzeParallelInput = serde_json::from_str(payload_with).expect("parse");
    assert_eq!(
        parsed.cost_name.as_deref(),
        Some("MAPE"),
        "Supplied cost_name must round-trip verbatim",
    );
}
