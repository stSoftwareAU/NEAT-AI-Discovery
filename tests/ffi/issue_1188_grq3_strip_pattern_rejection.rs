//! Issue #1188 — End-to-end regression for the GRQ-3-rocket.log strip patterns.
//!
//! The dylib's forward-only validator (Issue #1184) rejects creatures with
//! recurrent or self-looping synapses. Production logs from
//! `GRQ-3-rocket.log` continued to show `[loadFrom] Stripping recurrent
//! synapse ... source=fromJSON` warnings against `libneat_ai_discovery
//! v0.74.35`, which prompted an audit of the FFI surface. This test file
//! confirms that the three strip-depth patterns observed in the log are
//! rejected at every FFI entry point that accepts a `CreatureJson`:
//!
//! * **depth-0**: `output-0 -> output-0` self-loop (creature `10598e7e`).
//! * **depth-1**: a single-step back-edge (`hidden-1 -> hidden-0`,
//!   creature `bcc06579`).
//! * **depth-2**: a two-step back-edge spanning hidden layers
//!   (`output-0 -> hidden-0`, creature `751f7217`).
//!
//! Each test parses the strip pattern as `CreatureJson`, drives it through
//! the corresponding FFI internal entry point, and asserts the response
//! reports `success: false` and `errorKind: "data_validation"`.

use neat_ai_discovery::{CreatureJson, validate_forward_only_synapses};

/// GRQ-3 depth-0 pattern: an `output-0 -> output-0` self-loop. This is
/// the original Issue #1184 corruption signature and the most common
/// strip warning in `GRQ-3-rocket.log`.
fn depth0_creature_json() -> &'static str {
    r#"{
        "neurons": [
            {"uuid": "input-0", "type": "input", "squash": "IDENTITY"},
            {"uuid": "input-1", "type": "input", "squash": "IDENTITY"},
            {"uuid": "output-0", "type": "output", "squash": "LOGISTIC", "bias": 0.0}
        ],
        "synapses": [
            {"fromUUID": "input-0", "toUUID": "output-0", "weight": 0.5},
            {"fromUUID": "input-1", "toUUID": "output-0", "weight": -0.3},
            {"fromUUID": "output-0", "toUUID": "output-0", "weight": 0.9}
        ],
        "input": 2,
        "output": 1
    }"#
}

/// GRQ-3 depth-1 pattern: a single-step back-edge `hidden-1 -> hidden-0`.
/// `loadFrom` strips this with a "Stripping recurrent synapse" warning
/// (depth = 1) when the source neuron sits one position later in the
/// activation order than the target.
fn depth1_creature_json() -> &'static str {
    r#"{
        "neurons": [
            {"uuid": "input-0", "type": "input", "squash": "IDENTITY"},
            {"uuid": "hidden-0", "type": "hidden", "squash": "RELU", "bias": 0.0},
            {"uuid": "hidden-1", "type": "hidden", "squash": "RELU", "bias": 0.0},
            {"uuid": "output-0", "type": "output", "squash": "LOGISTIC", "bias": 0.0}
        ],
        "synapses": [
            {"fromUUID": "input-0", "toUUID": "hidden-0", "weight": 0.5},
            {"fromUUID": "hidden-0", "toUUID": "hidden-1", "weight": 0.4},
            {"fromUUID": "hidden-1", "toUUID": "output-0", "weight": 0.7},
            {"fromUUID": "hidden-1", "toUUID": "hidden-0", "weight": 0.6}
        ],
        "input": 1,
        "output": 1
    }"#
}

/// GRQ-3 depth-2 pattern: an output-to-hidden back-edge that crosses two
/// activation-order positions. `loadFrom` reports depth = 2 because the
/// source neuron is two positions later than the target.
fn depth2_creature_json() -> &'static str {
    r#"{
        "neurons": [
            {"uuid": "input-0", "type": "input", "squash": "IDENTITY"},
            {"uuid": "hidden-0", "type": "hidden", "squash": "RELU", "bias": 0.0},
            {"uuid": "hidden-1", "type": "hidden", "squash": "RELU", "bias": 0.0},
            {"uuid": "output-0", "type": "output", "squash": "LOGISTIC", "bias": 0.0}
        ],
        "synapses": [
            {"fromUUID": "input-0", "toUUID": "hidden-0", "weight": 0.5},
            {"fromUUID": "hidden-0", "toUUID": "hidden-1", "weight": 0.4},
            {"fromUUID": "hidden-1", "toUUID": "output-0", "weight": 0.7},
            {"fromUUID": "output-0", "toUUID": "hidden-0", "weight": 0.3}
        ],
        "input": 1,
        "output": 1
    }"#
}

// ---------------------------------------------------------------------------
// Validator-level checks: each strip pattern is rejected with a structured
// `DiscoveryError::InvalidInput` carrying the `data_validation` kind.
// ---------------------------------------------------------------------------

#[test]
fn validator_rejects_depth0_self_loop_pattern() {
    let creature: CreatureJson = serde_json::from_str(depth0_creature_json()).expect("valid JSON");
    let err = validate_forward_only_synapses(&creature)
        .expect_err("depth-0 GRQ-3 self-loop must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("self-loop"),
        "msg should describe self-loop: {msg}"
    );
    assert_eq!(
        err.error_kind(),
        neat_ai_discovery::DiscoveryErrorKind::DataValidation,
        "depth-0 must classify as data_validation"
    );
}

#[test]
fn validator_rejects_depth1_backedge_pattern() {
    let creature: CreatureJson = serde_json::from_str(depth1_creature_json()).expect("valid JSON");
    let err = validate_forward_only_synapses(&creature)
        .expect_err("depth-1 GRQ-3 back-edge must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("back-edge"),
        "msg should describe back-edge: {msg}"
    );
    assert_eq!(
        err.error_kind(),
        neat_ai_discovery::DiscoveryErrorKind::DataValidation
    );
}

#[test]
fn validator_rejects_depth2_backedge_pattern() {
    let creature: CreatureJson = serde_json::from_str(depth2_creature_json()).expect("valid JSON");
    let err = validate_forward_only_synapses(&creature)
        .expect_err("depth-2 GRQ-3 back-edge must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("back-edge"),
        "msg should describe back-edge: {msg}"
    );
    assert_eq!(
        err.error_kind(),
        neat_ai_discovery::DiscoveryErrorKind::DataValidation
    );
}

// ---------------------------------------------------------------------------
// FFI surface checks: each strip pattern is rejected at every entry point
// that accepts a `CreatureJson`.
// ---------------------------------------------------------------------------

fn assert_data_validation_error(response_json: &str, label: &str) {
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
        err_msg.contains("forward-only") || err_msg.contains("recurrent"),
        "{label} error must reference the forward-only invariant: {err_msg}"
    );
}

fn record_input_with(creature_json: &str, temp_dir: &str) -> String {
    format!(
        r#"{{
            "creature": {creature_json},
            "training_data": [
                {{
                    "input": [0.5],
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

#[test]
fn record_discovery_rejects_grq3_depth0_self_loop() {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let temp_path = temp_dir.path().to_str().unwrap();
    let input = record_input_with(depth0_creature_json(), temp_path);
    let response =
        neat_ai_discovery::record_discovery_internal(&input).expect("internal call must not panic");
    assert_data_validation_error(&response, "record_discovery (depth-0)");
}

#[test]
fn record_discovery_rejects_grq3_depth1_back_edge() {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let temp_path = temp_dir.path().to_str().unwrap();
    let input = record_input_with(depth1_creature_json(), temp_path);
    let response =
        neat_ai_discovery::record_discovery_internal(&input).expect("internal call must not panic");
    assert_data_validation_error(&response, "record_discovery (depth-1)");
}

#[test]
fn record_discovery_rejects_grq3_depth2_back_edge() {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let temp_path = temp_dir.path().to_str().unwrap();
    let input = record_input_with(depth2_creature_json(), temp_path);
    let response =
        neat_ai_discovery::record_discovery_internal(&input).expect("internal call must not panic");
    assert_data_validation_error(&response, "record_discovery (depth-2)");
}

#[test]
fn analyze_parallel_rejects_grq3_depth0_self_loop() {
    let input = analyze_input_with(depth0_creature_json());
    let response =
        neat_ai_discovery::analyze_parallel_internal(&input).expect("internal call must not panic");
    assert_data_validation_error(&response, "analyze_parallel (depth-0)");
}

#[test]
fn analyze_parallel_rejects_grq3_depth1_back_edge() {
    let input = analyze_input_with(depth1_creature_json());
    let response =
        neat_ai_discovery::analyze_parallel_internal(&input).expect("internal call must not panic");
    assert_data_validation_error(&response, "analyze_parallel (depth-1)");
}

#[test]
fn analyze_parallel_rejects_grq3_depth2_back_edge() {
    let input = analyze_input_with(depth2_creature_json());
    let response =
        neat_ai_discovery::analyze_parallel_internal(&input).expect("internal call must not panic");
    assert_data_validation_error(&response, "analyze_parallel (depth-2)");
}

#[test]
fn rank_focus_neurons_rejects_grq3_depth0_self_loop() {
    let input = rank_input_with(depth0_creature_json());
    let response = neat_ai_discovery::rank_focus_neurons_internal(&input)
        .expect("internal call must not panic");
    assert_data_validation_error(&response, "rank_focus_neurons (depth-0)");
}

#[test]
fn rank_focus_neurons_rejects_grq3_depth1_back_edge() {
    let input = rank_input_with(depth1_creature_json());
    let response = neat_ai_discovery::rank_focus_neurons_internal(&input)
        .expect("internal call must not panic");
    assert_data_validation_error(&response, "rank_focus_neurons (depth-1)");
}

#[test]
fn rank_focus_neurons_rejects_grq3_depth2_back_edge() {
    let input = rank_input_with(depth2_creature_json());
    let response = neat_ai_discovery::rank_focus_neurons_internal(&input)
        .expect("internal call must not panic");
    assert_data_validation_error(&response, "rank_focus_neurons (depth-2)");
}

#[test]
fn export_visualisation_snapshot_rejects_grq3_depth0_self_loop() {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let out_path = temp_dir.path().join("snapshot.json");
    let input = export_input_with(depth0_creature_json(), out_path.to_str().unwrap());
    let response = neat_ai_discovery::export_visualisation_snapshot_internal(&input)
        .expect("internal call must not panic");
    assert_data_validation_error(&response, "export_visualisation_snapshot (depth-0)");
}

#[test]
fn export_visualisation_snapshot_rejects_grq3_depth1_back_edge() {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let out_path = temp_dir.path().join("snapshot.json");
    let input = export_input_with(depth1_creature_json(), out_path.to_str().unwrap());
    let response = neat_ai_discovery::export_visualisation_snapshot_internal(&input)
        .expect("internal call must not panic");
    assert_data_validation_error(&response, "export_visualisation_snapshot (depth-1)");
}

#[test]
fn export_visualisation_snapshot_rejects_grq3_depth2_back_edge() {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let out_path = temp_dir.path().join("snapshot.json");
    let input = export_input_with(depth2_creature_json(), out_path.to_str().unwrap());
    let response = neat_ai_discovery::export_visualisation_snapshot_internal(&input)
        .expect("internal call must not panic");
    assert_data_validation_error(&response, "export_visualisation_snapshot (depth-2)");
}

// ---------------------------------------------------------------------------
// Streaming session: `start_discovery_session` is a separate FFI surface
// that previously bypassed the forward-only validator. Issue #1188 wires
// the validator into the FFI handler. We exercise it via the public C
// entry point so the test mirrors how NEAT-AI calls the dylib.
// ---------------------------------------------------------------------------

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

fn start_session_input(creature_json: &str, temp_dir: &str) -> String {
    format!(
        r#"{{
            "creature": {creature_json},
            "tempDir": "{temp_dir}"
        }}"#
    )
}

#[test]
fn start_discovery_session_rejects_grq3_depth0_self_loop() {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let input = start_session_input(depth0_creature_json(), temp_dir.path().to_str().unwrap());
    let response = call_start_discovery_session(&input);
    assert_data_validation_error(&response, "start_discovery_session (depth-0)");
}

#[test]
fn start_discovery_session_rejects_grq3_depth1_back_edge() {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let input = start_session_input(depth1_creature_json(), temp_dir.path().to_str().unwrap());
    let response = call_start_discovery_session(&input);
    assert_data_validation_error(&response, "start_discovery_session (depth-1)");
}

#[test]
fn start_discovery_session_rejects_grq3_depth2_back_edge() {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let input = start_session_input(depth2_creature_json(), temp_dir.path().to_str().unwrap());
    let response = call_start_discovery_session(&input);
    assert_data_validation_error(&response, "start_discovery_session (depth-2)");
}
