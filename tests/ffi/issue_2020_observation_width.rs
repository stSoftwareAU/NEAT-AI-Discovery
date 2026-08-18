//! Issue #2020 — `input < 1` / `output < 1` is never accepted at the FFI
//! boundary, and an emitted creature always carries the source width.
//!
//! The creature's top-level `input` / `output` integers are the observation
//! width. They **cannot** be re-derived: `neurons` lists only non-input
//! neurons, so a creature with `"input": 0` is corrupt, not input-less.
//! Every FFI entry point that accepts a `CreatureJson` must return the
//! structured `data_validation` error — mirroring the TypeScript reference
//! wording `Must have at least one input neurons was: N` — and the crate
//! must refuse to serialise a creature whose width is below one.

use neat_ai_discovery::{CreatureJson, DiscoveryErrorKind, validate_creature_input_bounds};

/// A well-formed creature: 2 inputs (implied, not listed), one output.
fn creature_json(input: usize, output: usize) -> String {
    format!(
        r#"{{
            "neurons": [
                {{"uuid": "output-0", "type": "output", "squash": "LOGISTIC", "bias": 0.0}}
            ],
            "synapses": [
                {{"fromUUID": "input-0", "toUUID": "output-0", "weight": 0.5}}
            ],
            "input": {input},
            "output": {output}
        }}"#
    )
}

fn expected_message(field: &str, count: usize) -> String {
    format!("Must have at least one {field} neurons was: {count}")
}

// ---------------------------------------------------------------------------
// serde boundary
// ---------------------------------------------------------------------------

#[test]
fn deserialise_rejects_zero_input() {
    let err = serde_json::from_str::<CreatureJson>(&creature_json(0, 1))
        .expect_err("input: 0 must be rejected at deserialisation");
    let msg = err.to_string();
    assert!(
        msg.contains(&expected_message("input", 0)),
        "error must mirror the TS reference wording: {msg}"
    );
    assert!(
        msg.contains("Issue #2020"),
        "error must cite the issue: {msg}"
    );
}

#[test]
fn deserialise_rejects_zero_output() {
    let err = serde_json::from_str::<CreatureJson>(&creature_json(2, 0))
        .expect_err("output: 0 must be rejected at deserialisation");
    let msg = err.to_string();
    assert!(
        msg.contains(&expected_message("output", 0)),
        "error must mirror the TS reference wording: {msg}"
    );
}

#[test]
fn deserialise_still_requires_the_fields() {
    // No `#[serde(default)]` — a missing width is a missing field, not zero.
    let json = r#"{"neurons": [], "synapses": [], "output": 1}"#;
    let err = serde_json::from_str::<CreatureJson>(json).expect_err("missing input must fail");
    assert!(
        err.to_string().contains("missing field `input`"),
        "missing input must surface as a missing field: {err}"
    );
    let json = r#"{"neurons": [], "synapses": [], "input": 1}"#;
    let err = serde_json::from_str::<CreatureJson>(json).expect_err("missing output must fail");
    assert!(
        err.to_string().contains("missing field `output`"),
        "missing output must surface as a missing field: {err}"
    );
}

#[test]
fn deserialise_accepts_width_of_one() {
    let creature: CreatureJson =
        serde_json::from_str(&creature_json(1, 1)).expect("1/1 is the minimum valid width");
    assert_eq!(creature.input, 1);
    assert_eq!(creature.output, 1);
}

#[test]
fn serialise_refuses_zero_input() {
    let mut creature: CreatureJson =
        serde_json::from_str(&creature_json(2, 1)).expect("valid creature");
    creature.input = 0;
    let err = serde_json::to_string(&creature).expect_err("input: 0 must not be emitted");
    assert!(
        err.to_string().contains(&expected_message("input", 0)),
        "serialise error must name the width: {err}"
    );
}

#[test]
fn serialise_refuses_zero_output() {
    let mut creature: CreatureJson =
        serde_json::from_str(&creature_json(2, 1)).expect("valid creature");
    creature.output = 0;
    let err = serde_json::to_string(&creature).expect_err("output: 0 must not be emitted");
    assert!(
        err.to_string().contains(&expected_message("output", 0)),
        "serialise error must name the width: {err}"
    );
}

#[test]
fn serialise_round_trips_the_source_width() {
    let creature: CreatureJson =
        serde_json::from_str(&creature_json(20, 3)).expect("valid creature");
    let emitted = serde_json::to_value(&creature).expect("valid width serialises");
    assert_eq!(emitted["input"], 20, "emitted input must equal the source");
    assert_eq!(emitted["output"], 3, "emitted output must equal the source");
}

// ---------------------------------------------------------------------------
// Belt-and-braces gate for creatures constructed in Rust
// ---------------------------------------------------------------------------

#[test]
fn bounds_gate_rejects_zero_input_and_zero_output() {
    let mut creature: CreatureJson =
        serde_json::from_str(&creature_json(2, 1)).expect("valid creature");
    creature.input = 0;
    let err = validate_creature_input_bounds(&creature).expect_err("input: 0 must be rejected");
    assert_eq!(err.error_kind(), DiscoveryErrorKind::DataValidation);
    assert!(err.to_string().contains(&expected_message("input", 0)));

    creature.input = 2;
    creature.output = 0;
    let err = validate_creature_input_bounds(&creature).expect_err("output: 0 must be rejected");
    assert_eq!(err.error_kind(), DiscoveryErrorKind::DataValidation);
    assert!(err.to_string().contains(&expected_message("output", 0)));
}

// ---------------------------------------------------------------------------
// FFI surface: every entry point that accepts a `CreatureJson`
// ---------------------------------------------------------------------------

fn assert_width_error(response_json: &str, label: &str, field: &str) {
    let parsed: serde_json::Value = serde_json::from_str(response_json)
        .unwrap_or_else(|_| panic!("{label} response must be valid JSON: {response_json}"));
    assert_eq!(
        parsed["success"], false,
        "{label} must report success=false: {response_json}"
    );
    assert_eq!(
        parsed["errorKind"], "data_validation",
        "{label} must classify as data_validation: {response_json}"
    );
    assert_eq!(
        parsed["retryable"], false,
        "{label} must mark the error as not retryable: {response_json}"
    );
    let err_msg = parsed["error"]
        .as_str()
        .expect("error message must be present");
    assert!(
        err_msg.contains(&expected_message(field, 0)),
        "{label} error must name the zero {field} width: {err_msg}"
    );
}

fn record_input_with(creature_json: &str, temp_dir: &str) -> String {
    format!(
        r#"{{
            "creature": {creature_json},
            "training_data": [
                {{
                    "input": [0.5, 0.25],
                    "output": [0.7],
                    "neuron_data": [
                        {{"neuron_uuid": "output-0", "activation": 0.7, "errors": [-0.05]}}
                    ]
                }}
            ],
            "temp_dir": "{temp_dir}"
        }}"#
    )
}

fn analyze_input_with(creature_json: &str) -> String {
    format!(
        r#"{{
            "parquetFile": "/tmp/does-not-exist.parquet",
            "creature": {creature_json},
            "focusNeurons": ["output-0"]
        }}"#
    )
}

fn rank_input_with(creature_json: &str) -> String {
    format!(
        r#"{{
            "parquetFile": "/tmp/does-not-exist.parquet",
            "creature": {creature_json}
        }}"#
    )
}

fn export_input_with(creature_json: &str, out_path: &str) -> String {
    format!(
        r#"{{
            "parquetFile": "/tmp/does-not-exist.parquet",
            "creature": {creature_json},
            "outFile": "{out_path}",
            "includePerSynapseSeries": false,
            "includeReconstructionChecks": false
        }}"#
    )
}

fn start_session_input(creature_json: &str, temp_dir: &str) -> String {
    format!(
        r#"{{
            "creature": {creature_json},
            "tempDir": "{temp_dir}"
        }}"#
    )
}

fn call_start_discovery_session(input_json: &str) -> String {
    use std::ffi::{CStr, CString};
    let input_c = CString::new(input_json).expect("input must not contain null bytes");
    // SAFETY: we pass a valid, non-null, null-terminated UTF-8 C string,
    // and we free the returned pointer with `free_discovery_result`.
    let response_ptr = unsafe { neat_ai_discovery::ffi::start_discovery_session(input_c.as_ptr()) };
    assert!(
        !response_ptr.is_null(),
        "FFI must return a non-null response pointer"
    );
    // SAFETY: the pointer was just produced by the FFI call and is valid.
    let response = unsafe { CStr::from_ptr(response_ptr) }
        .to_string_lossy()
        .into_owned();
    // SAFETY: `response_ptr` was produced by `start_discovery_session`,
    // matching the contract for `free_discovery_result`.
    unsafe { neat_ai_discovery::ffi::free_discovery_result(response_ptr) };
    response
}

#[test]
fn record_discovery_rejects_zero_input() {
    let temp = tempfile::tempdir().expect("temp dir");
    let input = record_input_with(&creature_json(0, 1), temp.path().to_str().unwrap());
    let response =
        neat_ai_discovery::record_discovery_internal(&input).expect("internal call must not panic");
    assert_width_error(&response, "record_discovery (input: 0)", "input");
}

#[test]
fn record_discovery_rejects_zero_output() {
    let temp = tempfile::tempdir().expect("temp dir");
    let input = record_input_with(&creature_json(2, 0), temp.path().to_str().unwrap());
    let response =
        neat_ai_discovery::record_discovery_internal(&input).expect("internal call must not panic");
    assert_width_error(&response, "record_discovery (output: 0)", "output");
}

#[test]
fn analyze_parallel_rejects_zero_input() {
    let input = analyze_input_with(&creature_json(0, 1));
    let response =
        neat_ai_discovery::analyze_parallel_internal(&input).expect("internal call must not panic");
    assert_width_error(&response, "analyze_parallel (input: 0)", "input");
}

#[test]
fn analyze_parallel_rejects_zero_output() {
    let input = analyze_input_with(&creature_json(2, 0));
    let response =
        neat_ai_discovery::analyze_parallel_internal(&input).expect("internal call must not panic");
    assert_width_error(&response, "analyze_parallel (output: 0)", "output");
}

#[test]
fn rank_focus_neurons_rejects_zero_input() {
    let input = rank_input_with(&creature_json(0, 1));
    let response = neat_ai_discovery::rank_focus_neurons_internal(&input)
        .expect("internal call must not panic");
    assert_width_error(&response, "rank_focus_neurons (input: 0)", "input");
}

#[test]
fn rank_focus_neurons_rejects_zero_output() {
    let input = rank_input_with(&creature_json(2, 0));
    let response = neat_ai_discovery::rank_focus_neurons_internal(&input)
        .expect("internal call must not panic");
    assert_width_error(&response, "rank_focus_neurons (output: 0)", "output");
}

#[test]
fn export_visualisation_snapshot_rejects_zero_input() {
    let temp = tempfile::tempdir().expect("temp dir");
    let out = temp.path().join("snapshot.json");
    let input = export_input_with(&creature_json(0, 1), out.to_str().unwrap());
    let response = neat_ai_discovery::export_visualisation_snapshot_internal(&input)
        .expect("internal call must not panic");
    assert_width_error(
        &response,
        "export_visualisation_snapshot (input: 0)",
        "input",
    );
    assert!(
        !out.exists(),
        "no snapshot may be written for a width-less creature"
    );
}

#[test]
fn export_visualisation_snapshot_rejects_zero_output() {
    let temp = tempfile::tempdir().expect("temp dir");
    let out = temp.path().join("snapshot.json");
    let input = export_input_with(&creature_json(2, 0), out.to_str().unwrap());
    let response = neat_ai_discovery::export_visualisation_snapshot_internal(&input)
        .expect("internal call must not panic");
    assert_width_error(
        &response,
        "export_visualisation_snapshot (output: 0)",
        "output",
    );
}

#[test]
fn start_discovery_session_rejects_zero_input() {
    let temp = tempfile::tempdir().expect("temp dir");
    let input = start_session_input(&creature_json(0, 1), temp.path().to_str().unwrap());
    let response = call_start_discovery_session(&input);
    assert_width_error(&response, "start_discovery_session (input: 0)", "input");
}

#[test]
fn start_discovery_session_rejects_zero_output() {
    let temp = tempfile::tempdir().expect("temp dir");
    let input = start_session_input(&creature_json(2, 0), temp.path().to_str().unwrap());
    let response = call_start_discovery_session(&input);
    assert_width_error(&response, "start_discovery_session (output: 0)", "output");
}
