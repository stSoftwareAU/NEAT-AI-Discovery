//! Issue #2078 — Bound `creature.output` at the FFI boundary.
//!
//! `CreatureJson.output` crossed the FFI boundary as a bare `usize` taken
//! straight from caller-supplied JSON, bounded only below (`output >= 1`,
//! Issue #2020). `CreatureTopologyCache::new`
//! (`src/analysis/detection/topology_cache.rs`) sizes an output-UUID
//! `HashSet` from that count, so an attacker-chosen `output` drove an
//! allocation of arbitrary size. Allocation failure routes through
//! `handle_alloc_error` — an **abort**, which the `panic::catch_unwind`
//! wrapper around every FFI entry point cannot intercept — so the host
//! process died instead of receiving an error response.
//!
//! The upper bound mirrors the `input` cap from Issue #1867.

use neat_ai_discovery::{
    CreatureJson, DiscoveryErrorKind, MAX_CREATURE_OUTPUT_NEURONS, validate_creature_input_bounds,
};

/// Build a minimal creature with the given output count.
fn creature_with_output(output: usize) -> CreatureJson {
    CreatureJson {
        neurons: vec![neat_ai_discovery::NeuronJson {
            uuid: "output-0".to_string(),
            neuron_type: "output".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        }],
        synapses: Vec::new(),
        input: 2,
        output,
    }
}

/// An ordinary creature — far more listed neurons than outputs — still
/// validates. Output neurons *are* listed in `neurons`, but the count remains
/// authoritative, so the bound must be absolute rather than relative.
#[test]
fn ordinary_output_creature_is_accepted() {
    validate_creature_input_bounds(&creature_with_output(2))
        .expect("an ordinary two-output creature must validate");
}

/// A creature sitting exactly on the limit is accepted — the bound is
/// inclusive, matching `MAX_CREATURE_INPUT_NEURONS`.
#[test]
fn creature_output_exactly_at_limit_is_accepted() {
    validate_creature_input_bounds(&creature_with_output(MAX_CREATURE_OUTPUT_NEURONS))
        .expect("the limit itself must be accepted");
}

/// One past the limit is rejected as a data-validation error.
#[test]
fn creature_output_above_limit_is_rejected() {
    let err =
        validate_creature_input_bounds(&creature_with_output(MAX_CREATURE_OUTPUT_NEURONS + 1))
            .expect_err("above the limit must fail");
    assert_eq!(
        err.error_kind(),
        DiscoveryErrorKind::DataValidation,
        "bound violations must classify as data_validation"
    );
    let msg = err.to_string();
    assert!(
        msg.contains("Issue #2078"),
        "error must cite the tracking issue: {msg}"
    );
}

/// The `usize::MAX` trigger from the issue report is rejected.
#[test]
fn creature_output_at_usize_max_is_rejected() {
    validate_creature_input_bounds(&creature_with_output(usize::MAX))
        .expect_err("usize::MAX output must fail");
}

/// Regression for the abort: `analyze_parallel` with the issue's exact
/// payload must return a structured `data_validation` error instead of
/// reaching `CreatureTopologyCache::new` and aborting the host process on
/// `HashSet::with_capacity(usize::MAX)`.
#[test]
fn analyze_parallel_rejects_oversized_output_without_aborting() {
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
            "input": 1,
            "output": 18446744073709551615
        },
        "focusNeurons": ["output-0"]
    }"#;

    let result = neat_ai_discovery::analyze_parallel_internal(input_json)
        .expect("internal call must not panic");
    let parsed: serde_json::Value =
        serde_json::from_str(&result).expect("response must be valid JSON");

    assert_eq!(
        parsed["success"], false,
        "oversized creature.output must fail: {result}"
    );
    assert_eq!(
        parsed["errorKind"], "data_validation",
        "oversized creature.output must classify as data_validation: {result}"
    );
}

/// `rank_focus_neurons` shares the same gate and the same topology cache.
#[test]
fn rank_focus_neurons_rejects_oversized_output_without_aborting() {
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
            "input": 1,
            "output": 18446744073709551615
        }
    }"#;

    let result = neat_ai_discovery::rank_focus_neurons_internal(input_json)
        .expect("internal call must not panic");
    let parsed: serde_json::Value =
        serde_json::from_str(&result).expect("response must be valid JSON");

    assert_eq!(
        parsed["success"], false,
        "oversized creature.output must fail: {result}"
    );
    assert_eq!(
        parsed["errorKind"], "data_validation",
        "oversized creature.output must classify as data_validation: {result}"
    );
}
