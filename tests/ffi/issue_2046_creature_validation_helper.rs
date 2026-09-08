//! Issue #2046 — One composed creature-validation gate.
//!
//! The forward-only check and the input-bounds check were chained by hand at
//! every FFI entry point that accepts a `CreatureJson`, so the documented
//! invariant (both checks, in that order, before any business logic) relied on
//! each new entry point re-typing the chain correctly. `validate_creature`
//! composes them once; these tests pin the composition and the observable
//! responses of all five call sites.

use neat_ai_discovery::{CreatureJson, DiscoveryErrorKind, validate_creature};

// ---------------------------------------------------------------------------
// The composed helper itself.
// ---------------------------------------------------------------------------

/// A forward-only creature within the input bounds passes both checks.
fn valid_creature() -> CreatureJson {
    serde_json::from_str(
        r#"{
            "neurons": [
                {"uuid": "hidden-0", "type": "hidden", "squash": "RELU", "bias": 0.0},
                {"uuid": "output-0", "type": "output", "squash": "LOGISTIC", "bias": 0.0}
            ],
            "synapses": [
                {"fromUUID": "hidden-0", "toUUID": "output-0", "weight": 0.5}
            ],
            "input": 2,
            "output": 1
        }"#,
    )
    .expect("valid creature JSON")
}

#[test]
fn valid_creature_passes_the_composed_gate() {
    validate_creature(&valid_creature()).expect("a forward-only, in-bounds creature must validate");
}

#[test]
fn composed_gate_rejects_a_forward_only_violation() {
    let mut creature = valid_creature();
    creature.synapses[0].from_uuid = "output-0".to_string();
    creature.synapses[0].to_uuid = "hidden-0".to_string();

    let err = validate_creature(&creature).expect_err("a back-edge must be rejected");
    assert_eq!(
        err.error_kind(),
        DiscoveryErrorKind::DataValidation,
        "forward-only violations classify as data_validation"
    );
    assert!(
        err.to_string().contains("forward-only"),
        "error must name the forward-only invariant: {err}"
    );
}

#[test]
fn composed_gate_rejects_an_input_bounds_violation() {
    let mut creature = valid_creature();
    creature.input = usize::MAX;

    let err =
        validate_creature(&creature).expect_err("an out-of-range input count must be rejected");
    assert_eq!(
        err.error_kind(),
        DiscoveryErrorKind::DataValidation,
        "bound violations classify as data_validation"
    );
    assert!(
        err.to_string().contains("Issue #1867"),
        "error must cite the input-bound issue: {err}"
    );
}

/// The lower observation-width bound (Issue #2020) also runs through the
/// composed gate. A zero width cannot arrive as JSON — `CreatureJson`'s serde
/// impl rejects it on read — so the creature is built directly.
#[test]
fn composed_gate_rejects_a_zero_observation_width() {
    let mut creature = valid_creature();
    creature.output = 0;

    let err = validate_creature(&creature).expect_err("a zero output width must be rejected");
    assert!(
        err.to_string().contains("Issue #2020"),
        "error must cite the observation-width issue: {err}"
    );
}

/// Order matters: the forward-only check runs first, so a creature violating
/// both invariants reports the topology fault, not the bound.
#[test]
fn composed_gate_reports_the_forward_only_fault_first() {
    let mut creature = valid_creature();
    creature.synapses[0].from_uuid = "output-0".to_string();
    creature.synapses[0].to_uuid = "hidden-0".to_string();
    creature.input = usize::MAX;

    let err = validate_creature(&creature).expect_err("both invariants are violated");
    assert!(
        err.to_string().contains("forward-only"),
        "the forward-only check must run first: {err}"
    );
}

// ---------------------------------------------------------------------------
// The five entry points that run the gate.
// ---------------------------------------------------------------------------

/// A creature carrying an `output-0 -> output-0` self-loop.
const BACK_EDGE_CREATURE: &str = r#"{
    "neurons": [
        {"uuid": "hidden-0", "type": "hidden", "squash": "RELU", "bias": 0.0},
        {"uuid": "output-0", "type": "output", "squash": "LOGISTIC", "bias": 0.0}
    ],
    "synapses": [
        {"fromUUID": "output-0", "toUUID": "output-0", "weight": 0.5}
    ],
    "input": 2,
    "output": 1
}"#;

/// A forward-only creature whose input count is far past the cap. The count
/// is the caller-supplied observation width, so it is only rejected by the
/// second half of the composed gate.
const OVERSIZED_INPUT_CREATURE: &str = r#"{
    "neurons": [
        {"uuid": "hidden-0", "type": "hidden", "squash": "RELU", "bias": 0.0},
        {"uuid": "output-0", "type": "output", "squash": "LOGISTIC", "bias": 0.0}
    ],
    "synapses": [
        {"fromUUID": "hidden-0", "toUUID": "output-0", "weight": 0.5}
    ],
    "input": 10000000000,
    "output": 1
}"#;

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
    // SAFETY: a valid, non-null, NUL-terminated UTF-8 C string is passed and
    // the returned pointer is freed with `free_discovery_result`.
    let response_ptr = unsafe { neat_ai_discovery::ffi::start_discovery_session(input_c.as_ptr()) };
    assert!(
        !response_ptr.is_null(),
        "FFI must return a non-null response pointer"
    );
    // SAFETY: the pointer was just produced by the FFI call and is valid.
    let response = unsafe { CStr::from_ptr(response_ptr) }
        .to_string_lossy()
        .into_owned();
    // SAFETY: `response_ptr` came from `start_discovery_session`, matching the
    // contract for `free_discovery_result`.
    unsafe { neat_ai_discovery::ffi::free_discovery_result(response_ptr) };
    response
}

/// Every entry point's response for a given creature, keyed by entry-point
/// name so a failure names the site that drifted.
fn responses_for(creature_json: &str) -> Vec<(&'static str, String)> {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let temp_path = temp_dir.path().to_str().expect("utf-8 temp path");
    let out_path = temp_dir.path().join("snapshot.json");

    vec![
        (
            "record_discovery",
            neat_ai_discovery::record_discovery_internal(&record_input_with(
                creature_json,
                temp_path,
            ))
            .expect("internal call must not panic"),
        ),
        (
            "analyze_parallel",
            neat_ai_discovery::analyze_parallel_internal(&analyze_input_with(creature_json))
                .expect("internal call must not panic"),
        ),
        (
            "rank_focus_neurons",
            neat_ai_discovery::rank_focus_neurons_internal(&rank_input_with(creature_json))
                .expect("internal call must not panic"),
        ),
        (
            "export_visualisation_snapshot",
            neat_ai_discovery::export_visualisation_snapshot_internal(&export_input_with(
                creature_json,
                out_path.to_str().expect("utf-8 out path"),
            ))
            .expect("internal call must not panic"),
        ),
        (
            "start_discovery_session",
            call_start_discovery_session(&start_session_input(creature_json, temp_path)),
        ),
    ]
}

#[test]
fn every_creature_entry_point_rejects_a_forward_only_violation() {
    for (name, response) in responses_for(BACK_EDGE_CREATURE) {
        assert_data_validation_error(&response, name);
        let parsed: serde_json::Value =
            serde_json::from_str(&response).expect("valid JSON response");
        let err_msg = parsed["error"].as_str().expect("error message present");
        assert!(
            err_msg.contains("forward-only") || err_msg.contains("recurrent"),
            "{name} must reject via the forward-only half of the gate: {err_msg}"
        );
    }
}

#[test]
fn every_creature_entry_point_rejects_an_input_bounds_violation() {
    for (name, response) in responses_for(OVERSIZED_INPUT_CREATURE) {
        assert_data_validation_error(&response, name);
        let parsed: serde_json::Value =
            serde_json::from_str(&response).expect("valid JSON response");
        let err_msg = parsed["error"].as_str().expect("error message present");
        assert!(
            err_msg.contains("Issue #1867"),
            "{name} must reject via the input-bounds half of the gate: {err_msg}"
        );
    }
}
