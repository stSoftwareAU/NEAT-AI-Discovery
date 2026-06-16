//! Regression tests for Issue #1402: a malformed / wrong-cased
//! `taskDescriptor` must never fail the whole `analyze_parallel` (or any
//! discovery FFI) JSON parse. It must parse to the correct descriptor or
//! fall back to [`TaskDescriptor::neutral`].
//!
//! Background: Issue #1314 added optional `task_descriptor` plumbing with
//! `#[serde(default)]` for the **absent** case. When the field is **present
//! but invalid** — as it was when NEAT-AI #2785 forwarded the internal
//! TypeScript descriptor shape (`"outputSquashFamily": "unbounded"`,
//! lowercase) instead of the `PascalCase` wire shape — the entire FFI request
//! failed to deserialise with:
//!
//! ```text
//! Invalid input: Failed to parse input JSON: unknown variant `unbounded`,
//! expected one of `BoundedUnipolar`, `BoundedBipolar`, `Positive`,
//! `Unbounded`, `Any`
//! ```
//!
//! Discovery degraded: focus selection completed, but the Rust
//! synapse/neuron analysis was unavailable. These tests lock in the
//! defensive consumer-side hardening so a future producer mistake cannot
//! silently break analysis again.

use neat_ai_discovery::analysis::task_descriptor::TaskDescriptor;
use neat_ai_discovery::{
    AnalyzeParallelInput, RankFocusNeuronsInput, RecordDiscoveryInput, analyze_parallel_internal,
};

/// The exact `taskDescriptor` subtree NEAT-AI currently emits for MSE — the
/// internal TypeScript shape: lowercase enum strings and TS-side field names
/// (`topology` / `range`) that do not match the Rust wire contract.
const NEAT_AI_MSE_TASK_DESCRIPTOR: &str = r#"{
    "costName": "MSE",
    "topology": "independent",
    "range": "unbounded",
    "outputSquashFamily": "unbounded"
}"#;

// =============================================================================
// 1. Failing regression test (red on current Develop): the producer payload
//    must deserialise rather than fail the whole input JSON parse.
// =============================================================================

#[test]
fn analyze_parallel_input_with_lowercase_task_descriptor_falls_back_to_neutral() {
    let payload = format!(
        r#"{{
            "parquetFile": "/tmp/x.parquet",
            "creature": {{"neurons": [], "synapses": [], "input": 0, "output": 0}},
            "focusNeurons": [],
            "taskDescriptor": {NEAT_AI_MSE_TASK_DESCRIPTOR}
        }}"#
    );
    let parsed: AnalyzeParallelInput =
        serde_json::from_str(&payload).expect("malformed taskDescriptor must not fail input parse");
    // The malformed descriptor falls back to neutral rather than failing.
    assert_eq!(
        parsed.task_descriptor.unwrap_or_default(),
        TaskDescriptor::neutral(),
        "lowercase/unknown taskDescriptor must collapse to neutral()",
    );
}

#[test]
fn analyze_parallel_internal_does_not_return_input_json_parse_error() {
    // Drive the real FFI entry point. The malformed taskDescriptor must not
    // produce a "Failed to parse input JSON" error. A missing parquet file
    // is an acceptable downstream error — what must NOT happen is the
    // top-level input parse failing on the descriptor.
    let payload = format!(
        r#"{{
            "parquetFile": "/tmp/nonexistent-issue-1402.parquet",
            "creature": {{"neurons": [], "synapses": [], "input": 0, "output": 0}},
            "focusNeurons": [],
            "taskDescriptor": {NEAT_AI_MSE_TASK_DESCRIPTOR}
        }}"#
    );
    let result_json =
        analyze_parallel_internal(&payload).expect("internal call should return JSON");
    assert!(
        !result_json.contains("Failed to parse input JSON"),
        "malformed taskDescriptor must not cause an input JSON parse failure, got: {result_json}",
    );
}

// =============================================================================
// 2. Record-discovery and rank-focus inputs harden the same way.
// =============================================================================

#[test]
fn record_discovery_input_with_lowercase_task_descriptor_falls_back_to_neutral() {
    // RecordDiscoveryInput uses snake_case at the top level.
    let payload = format!(
        r#"{{
            "creature": {{"neurons": [], "synapses": [], "input": 0, "output": 0}},
            "training_data": [],
            "temp_dir": "/tmp/x",
            "task_descriptor": {NEAT_AI_MSE_TASK_DESCRIPTOR}
        }}"#
    );
    let parsed: RecordDiscoveryInput = serde_json::from_str(&payload)
        .expect("malformed task_descriptor must not fail input parse");
    assert_eq!(
        parsed.task_descriptor.unwrap_or_default(),
        TaskDescriptor::neutral(),
    );
}

#[test]
fn rank_focus_neurons_input_with_lowercase_task_descriptor_falls_back_to_neutral() {
    let payload = format!(
        r#"{{
            "parquetFile": "/tmp/x.parquet",
            "creature": {{"neurons": [], "synapses": [], "input": 0, "output": 0}},
            "taskDescriptor": {NEAT_AI_MSE_TASK_DESCRIPTOR}
        }}"#
    );
    let parsed: RankFocusNeuronsInput =
        serde_json::from_str(&payload).expect("malformed taskDescriptor must not fail input parse");
    assert_eq!(
        parsed.task_descriptor.unwrap_or_default(),
        TaskDescriptor::neutral(),
    );
}

// =============================================================================
// 3. Explicit regression guard — individual lowercase enum strings and
//    unknown fields must never cause a top-level FFI parse failure.
// =============================================================================

#[test]
fn lowercase_enum_strings_never_fail_top_level_parse() {
    // Each of these is an invalid enum string or an unknown field shape that
    // the strict #1314 derive would reject. All must collapse to neutral.
    let malformed_descriptors = [
        r#"{"targetTopology": "independent"}"#,
        r#"{"targetRange": "unbounded"}"#,
        r#"{"outputSquashFamily": "bounded_unipolar"}"#,
        r#"{"targetTopology": "one_hot", "targetRange": "unit"}"#,
        r#"{"completely": "unexpected", "shape": 42}"#,
        // A type mismatch on numOutputs must also be tolerated.
        r#"{"targetTopology": "Independent", "numOutputs": "five"}"#,
    ];
    for descriptor in malformed_descriptors {
        let payload = format!(
            r#"{{
                "parquetFile": "/tmp/x.parquet",
                "creature": {{"neurons": [], "synapses": [], "input": 0, "output": 0}},
                "focusNeurons": [],
                "taskDescriptor": {descriptor}
            }}"#
        );
        let parsed: AnalyzeParallelInput = serde_json::from_str(&payload)
            .unwrap_or_else(|e| panic!("descriptor {descriptor} must not fail parse: {e}"));
        assert_eq!(
            parsed.task_descriptor.unwrap_or_default(),
            TaskDescriptor::neutral(),
            "malformed descriptor {descriptor} must collapse to neutral()",
        );
    }
}

// =============================================================================
// 4. A `taskDescriptor` that is not even an object (e.g. a bare string)
//    must still not break the top-level parse.
// =============================================================================

#[test]
fn non_object_task_descriptor_falls_back_to_neutral() {
    let payload = r#"{
        "parquetFile": "/tmp/x.parquet",
        "creature": {"neurons": [], "synapses": [], "input": 0, "output": 0},
        "focusNeurons": [],
        "taskDescriptor": "unbounded"
    }"#;
    let parsed: AnalyzeParallelInput =
        serde_json::from_str(payload).expect("non-object taskDescriptor must not fail input parse");
    assert_eq!(
        parsed.task_descriptor.unwrap_or_default(),
        TaskDescriptor::neutral(),
    );
}
