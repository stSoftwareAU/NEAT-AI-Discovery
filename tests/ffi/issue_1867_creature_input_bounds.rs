//! Issue #1867 — Bound `creature.input` at the FFI boundary.
//!
//! `CreatureJson.input` crosses the FFI boundary as a bare `usize` taken
//! straight from caller-supplied JSON. Unbounded, it drove a `Vec<String>`
//! of input-neuron UUIDs whose allocation failure routes through
//! `handle_alloc_error` — an **abort**, which `panic::catch_unwind` cannot
//! intercept, so the host process died instead of receiving an error
//! response. Values near `usize::MAX` additionally wrapped the
//! `non_input_neuron_count + creature.input` additions in release builds
//! (no `overflow-checks`), defeating the `checked_mul` guard below them.
//!
//! The gate bounds `input` to `MAX_CREATURE_INPUT_NEURONS` and the three
//! additions are now `checked_add`.

use neat_ai_discovery::{
    CreatureJson, DiscoveryErrorKind, MAX_CREATURE_INPUT_NEURONS, validate_creature_input_bounds,
};

/// The documented `record_discovery` example (`docs/FFI_API.md`) declares
/// `"input": 20` with a single entry in `neurons` — input neurons are not
/// listed in `creature.neurons`, they are implied by the count. A creature
/// with far more inputs than listed neurons is therefore ordinary, and the
/// bound must not reject it.
#[test]
fn wide_input_creature_with_few_listed_neurons_is_accepted() {
    let json = r#"{
        "neurons": [
            {"uuid": "hidden-1", "type": "hidden", "squash": "TANH", "bias": 0.0}
        ],
        "synapses": [
            {"fromUUID": "input-0", "toUUID": "hidden-1", "weight": 0.5}
        ],
        "input": 20,
        "output": 2
    }"#;

    let creature: CreatureJson = serde_json::from_str(json).expect("valid JSON");
    validate_creature_input_bounds(&creature)
        .expect("20 inputs with 1 listed neuron is the documented shape");
}

/// A creature sitting exactly on the limit is still accepted — the bound is
/// inclusive.
#[test]
fn creature_input_exactly_at_limit_is_accepted() {
    let creature = CreatureJson {
        neurons: Vec::new(),
        synapses: Vec::new(),
        input: MAX_CREATURE_INPUT_NEURONS,
        output: 1,
    };
    validate_creature_input_bounds(&creature).expect("the limit itself must be accepted");
}

/// One past the limit is rejected as a data-validation error.
#[test]
fn creature_input_above_limit_is_rejected() {
    let creature = CreatureJson {
        neurons: Vec::new(),
        synapses: Vec::new(),
        input: MAX_CREATURE_INPUT_NEURONS + 1,
        output: 1,
    };
    let err = validate_creature_input_bounds(&creature).expect_err("above the limit must fail");
    assert_eq!(
        err.error_kind(),
        DiscoveryErrorKind::DataValidation,
        "bound violations must classify as data_validation"
    );
    let msg = err.to_string();
    assert!(
        msg.contains("Issue #1867"),
        "error must cite the tracking issue: {msg}"
    );
}

/// A `usize::MAX` input — the silent-wrap trigger — is rejected too.
#[test]
fn creature_input_at_usize_max_is_rejected() {
    let creature = CreatureJson {
        neurons: Vec::new(),
        synapses: Vec::new(),
        input: usize::MAX,
        output: 1,
    };
    validate_creature_input_bounds(&creature).expect_err("usize::MAX input must fail");
}

/// Regression for the abort: `record_discovery` with a ten-billion input
/// count must return a structured `data_validation` error instead of
/// allocating ten billion `String`s and aborting the host process.
#[test]
fn record_discovery_rejects_oversized_input_without_aborting() {
    let input_json = r#"{
        "creature": {
            "neurons": [
                {"uuid": "hidden-1", "type": "hidden", "squash": "TANH", "bias": 0.0},
                {"uuid": "output-0", "type": "output", "squash": "LOGISTIC", "bias": 0.0}
            ],
            "synapses": [
                {"fromUUID": "hidden-1", "toUUID": "output-0", "weight": 0.5}
            ],
            "input": 10000000000,
            "output": 1
        },
        "training_data": [
            {"input": [0.1, 0.2], "output": [0.5], "neuron_data": [
                {"neuron_uuid": "hidden-1", "activation": 0.5, "value": 0.4, "errors": [0.1]},
                {"neuron_uuid": "output-0", "activation": 0.5, "value": 0.5, "errors": [0.0]}
            ]}
        ],
        "temp_dir": "/tmp/neat-ai-discovery-issue-1867"
    }"#;

    let result = neat_ai_discovery::record_discovery_internal(input_json)
        .expect("internal call must not panic");
    let parsed: serde_json::Value =
        serde_json::from_str(&result).expect("response must be valid JSON");

    assert_eq!(
        parsed["success"], false,
        "oversized creature.input must fail: {result}"
    );
    assert_eq!(
        parsed["errorKind"], "data_validation",
        "oversized creature.input must classify as data_validation: {result}"
    );
}

/// The analysis entry point shares the gate — it sizes per-input lookup maps
/// from the same field.
#[test]
fn analyze_parallel_rejects_oversized_input() {
    let input_json = r#"{
        "parquetFile": "/tmp/does-not-exist.parquet",
        "creature": {
            "neurons": [
                {"uuid": "hidden-0", "type": "hidden", "squash": "RELU", "bias": 0.0},
                {"uuid": "output-0", "type": "output", "squash": "LOGISTIC", "bias": 0.0}
            ],
            "synapses": [
                {"fromUUID": "hidden-0", "toUUID": "output-0", "weight": 0.8}
            ],
            "input": 10000000000,
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
        "oversized creature.input must fail: {result}"
    );
    assert_eq!(
        parsed["errorKind"], "data_validation",
        "oversized creature.input must classify as data_validation: {result}"
    );
}

/// The recording pipeline itself must not wrap when it is reached with a
/// `usize::MAX` input (a direct library call bypassing the FFI gate). The
/// `non_input_neuron_count + creature.input` additions are `checked_add`, so
/// this returns a typed error rather than a wrapped-around record estimate.
#[test]
fn record_discovery_data_reports_overflow_instead_of_wrapping() {
    let mut input = overflow_input();
    input.creature.input = usize::MAX;

    let err = neat_ai_discovery::record::record_discovery_data(&input)
        .expect_err("usize::MAX input must not wrap");
    let msg = err.to_string();
    assert!(
        msg.contains("overflow"),
        "error must name the overflow: {msg}"
    );
}

/// Minimal recording input used by the overflow regression test.
fn overflow_input() -> neat_ai_discovery::RecordDiscoveryInput {
    neat_ai_discovery::RecordDiscoveryInput {
        creature: CreatureJson {
            neurons: vec![neat_ai_discovery::NeuronJson {
                uuid: "output-0".to_string(),
                neuron_type: "output".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            }],
            synapses: Vec::new(),
            input: 1,
            output: 1,
        },
        training_data: vec![neat_ai_discovery::TrainingRecord {
            input: vec![0.1],
            output: vec![0.5],
            neuron_data: Some(vec![neat_ai_discovery::NeuronData {
                neuron_uuid: "output-0".to_string(),
                activation: 0.5,
                value: Some(0.5),
                errors: vec![0.0],
            }]),
        }],
        temp_dir: std::env::temp_dir()
            .join("neat-ai-discovery-issue-1867-overflow")
            .to_string_lossy()
            .to_string(),
        binary_file_path: None,
        record_indices: None,
        timeout_seconds: None,
        task_descriptor: None,
    }
}
