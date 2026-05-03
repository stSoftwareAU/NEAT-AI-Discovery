//! Issue #1184 — Reject creatures carrying recurrent synapses at the FFI boundary.
//!
//! Mirrors the GRQ-10 sweep where `output-0 -> output-0` self-loops survived
//! upstream NEAT-AI `loadFrom` stripping (warn-and-continue) and entered the
//! discovery pipeline. The discovery library now refuses to consume a
//! corrupt creature: every FFI entry point that accepts a `CreatureJson`
//! validates the forward-only invariant before reaching business logic and
//! returns a structured `data_validation` error when violations are found.

use neat_ai_discovery::{CreatureJson, validate_forward_only_synapses};

/// A forward-only creature with valid synapses must pass validation.
#[test]
fn validate_accepts_forward_only_creature() {
    let json = r#"{
        "neurons": [
            {"uuid": "input-0", "type": "input", "squash": "IDENTITY"},
            {"uuid": "input-1", "type": "input", "squash": "IDENTITY"},
            {"uuid": "hidden-a", "type": "hidden", "squash": "RELU", "bias": 0.1},
            {"uuid": "output-0", "type": "output", "squash": "LOGISTIC", "bias": 0.0}
        ],
        "synapses": [
            {"fromUUID": "input-0", "toUUID": "hidden-a", "weight": 0.5},
            {"fromUUID": "input-1", "toUUID": "hidden-a", "weight": -0.3},
            {"fromUUID": "hidden-a", "toUUID": "output-0", "weight": 0.8},
            {"fromUUID": "input-0", "toUUID": "output-0", "weight": 0.2}
        ],
        "input": 2,
        "output": 1
    }"#;

    let creature: CreatureJson = serde_json::from_str(json).expect("valid JSON");
    validate_forward_only_synapses(&creature).expect("forward-only creature must pass validation");
}

/// An `output-0 -> output-0` self-loop is the exact corruption pattern from
/// Issue #1184. Validation must reject it.
#[test]
fn validate_rejects_output_self_loop() {
    let json = r#"{
        "neurons": [
            {"uuid": "input-0", "type": "input", "squash": "IDENTITY"},
            {"uuid": "output-0", "type": "output", "squash": "LOGISTIC", "bias": 0.0}
        ],
        "synapses": [
            {"fromUUID": "input-0", "toUUID": "output-0", "weight": 0.5},
            {"fromUUID": "output-0", "toUUID": "output-0", "weight": 0.9}
        ],
        "input": 1,
        "output": 1
    }"#;

    let creature: CreatureJson = serde_json::from_str(json).expect("valid JSON");
    let err =
        validate_forward_only_synapses(&creature).expect_err("output self-loop must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("self-loop"),
        "msg should describe self-loop: {msg}"
    );
    assert!(
        msg.contains("Issue #1184"),
        "msg should cite the tracking issue: {msg}"
    );
}

/// `record_discovery_internal` must surface a structured `data_validation`
/// error response when the supplied creature carries a recurrent synapse.
#[test]
fn ffi_record_discovery_rejects_recurrent_synapse() {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let temp_path = temp_dir.path().to_str().unwrap();

    let input_json = format!(
        r#"{{
            "creature": {{
                "neurons": [
                    {{"uuid": "input-0", "type": "input", "squash": "IDENTITY"}},
                    {{"uuid": "input-1", "type": "input", "squash": "IDENTITY"}},
                    {{"uuid": "output-0", "type": "output", "squash": "LOGISTIC", "bias": 0.0}}
                ],
                "synapses": [
                    {{"fromUUID": "input-0", "toUUID": "output-0", "weight": 0.5}},
                    {{"fromUUID": "output-0", "toUUID": "output-0", "weight": 0.9}}
                ],
                "input": 2,
                "output": 1
            }},
            "training_data": [
                {{
                    "input": [0.5, 0.3],
                    "output": [0.7],
                    "neuron_data": [
                        {{"neuron_uuid": "output-0", "activation": 0.7, "errors": [-0.05]}}
                    ]
                }}
            ],
            "temp_dir": "{temp_path}"
        }}"#
    );

    let result = neat_ai_discovery::record_discovery_internal(&input_json)
        .expect("internal call must not panic");
    let parsed: serde_json::Value =
        serde_json::from_str(&result).expect("response must be valid JSON");

    assert_eq!(
        parsed["success"], false,
        "recording with recurrent synapse must fail: {result}"
    );
    assert_eq!(
        parsed["errorKind"], "data_validation",
        "recurrent synapse must classify as data_validation: {result}"
    );
    assert_eq!(
        parsed["retryable"], false,
        "data validation errors must not be retryable: {result}"
    );
    let err_msg = parsed["error"]
        .as_str()
        .expect("error message must be present");
    assert!(
        err_msg.contains("forward-only"),
        "error must mention forward-only invariant: {err_msg}"
    );
}

/// `analyze_parallel_internal` must surface a structured `data_validation`
/// error response when the supplied creature carries a recurrent synapse.
/// This guards the analysis pipeline even on hosts without a GPU — the
/// validation runs before any GPU initialisation.
#[test]
fn ffi_analyze_parallel_rejects_recurrent_synapse() {
    let input_json = r#"{
        "parquetFile": "/tmp/does-not-exist.parquet",
        "creature": {
            "neurons": [
                {"uuid": "input-0", "type": "input", "squash": "IDENTITY"},
                {"uuid": "hidden-0", "type": "hidden", "squash": "RELU", "bias": 0.0},
                {"uuid": "output-0", "type": "output", "squash": "LOGISTIC", "bias": 0.0}
            ],
            "synapses": [
                {"fromUUID": "input-0", "toUUID": "hidden-0", "weight": 0.5},
                {"fromUUID": "hidden-0", "toUUID": "output-0", "weight": 0.8},
                {"fromUUID": "output-0", "toUUID": "hidden-0", "weight": 0.4}
            ],
            "input": 1,
            "output": 1
        },
        "focusNeurons": ["output-0"]
    }"#;

    let result = neat_ai_discovery::analyze_parallel_internal(input_json)
        .expect("internal call must not panic");
    let parsed: serde_json::Value =
        serde_json::from_str(&result).expect("response must be valid JSON");

    assert_eq!(
        parsed["success"], false,
        "analysis with back-edge must fail: {result}"
    );
    assert_eq!(
        parsed["errorKind"], "data_validation",
        "back-edge must classify as data_validation: {result}"
    );
    let err_msg = parsed["error"]
        .as_str()
        .expect("error message must be present");
    assert!(
        err_msg.contains("back-edge") || err_msg.contains("forward-only"),
        "error must describe the violation: {err_msg}"
    );
}

/// `rank_focus_neurons_internal` must reject a creature with a self-loop
/// before it begins ranking work.
#[test]
fn ffi_rank_focus_neurons_rejects_recurrent_synapse() {
    let input_json = r#"{
        "parquetFile": "/tmp/does-not-exist.parquet",
        "creature": {
            "neurons": [
                {"uuid": "input-0", "type": "input", "squash": "IDENTITY"},
                {"uuid": "output-0", "type": "output", "squash": "LOGISTIC", "bias": 0.0}
            ],
            "synapses": [
                {"fromUUID": "input-0", "toUUID": "output-0", "weight": 0.5},
                {"fromUUID": "output-0", "toUUID": "output-0", "weight": 0.9}
            ],
            "input": 1,
            "output": 1
        }
    }"#;

    let result = neat_ai_discovery::rank_focus_neurons_internal(input_json)
        .expect("internal call must not panic");
    let parsed: serde_json::Value =
        serde_json::from_str(&result).expect("response must be valid JSON");

    assert_eq!(
        parsed["success"], false,
        "ranking with self-loop must fail: {result}"
    );
    assert_eq!(
        parsed["errorKind"], "data_validation",
        "self-loop must classify as data_validation: {result}"
    );
}

/// Synapses referencing neuron UUIDs that are not in the creature are
/// tolerated — they are out of scope for Issue #1184 and the existing
/// analysis pipeline already silently filters them. The forward-only
/// gate must not regress that behaviour.
#[test]
fn validate_tolerates_unknown_neuron_uuid() {
    let json = r#"{
        "neurons": [
            {"uuid": "input-0", "type": "input", "squash": "IDENTITY"},
            {"uuid": "output-0", "type": "output", "squash": "LOGISTIC", "bias": 0.0}
        ],
        "synapses": [
            {"fromUUID": "ghost-neuron", "toUUID": "output-0", "weight": 0.5}
        ],
        "input": 1,
        "output": 1
    }"#;

    let creature: CreatureJson = serde_json::from_str(json).expect("valid JSON");
    validate_forward_only_synapses(&creature)
        .expect("dangling reference must not be rejected by the forward-only gate");
}
