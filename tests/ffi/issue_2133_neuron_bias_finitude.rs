//! Issue #2133 — a neuron bias that is not finite never crosses the FFI
//! boundary.
//!
//! The two doors, why each is open, and what a non-finite bias corrupts
//! downstream are documented once in
//! [`neat_ai_discovery::ffi_types::neuron_bias`](../../src/ffi_types/neuron_bias.rs);
//! this suite covers each layer from the outside: deserialisation, the
//! `validate_creature` gate, serialisation, and the shipped entry points.

use neat_ai_discovery::{
    CreatureJson, DiscoveryErrorKind, NeuronJson, rank_focus_neurons_internal, validate_creature,
};

/// A well-formed creature whose single hidden neuron carries `bias`.
fn creature_json(bias: &str) -> String {
    format!(
        r#"{{
            "neurons": [
                {{"uuid": "hidden-0", "type": "hidden", "squash": "IDENTITY", "bias": {bias}}},
                {{"uuid": "output-0", "type": "output", "squash": "LOGISTIC", "bias": 0.0}}
            ],
            "synapses": [
                {{"fromUUID": "input-0",  "toUUID": "hidden-0", "weight": 0.5}},
                {{"fromUUID": "hidden-0", "toUUID": "output-0", "weight": 0.5}}
            ],
            "input": 1,
            "output": 1
        }}"#
    )
}

/// A creature built in Rust — the path that never touches serde.
fn creature_with_bias(bias: f32) -> CreatureJson {
    CreatureJson {
        neurons: vec![
            NeuronJson {
                uuid: "hidden-0".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "IDENTITY".to_string(),
                bias,
            },
            NeuronJson {
                uuid: "output-0".to_string(),
                neuron_type: "output".to_string(),
                squash: "LOGISTIC".to_string(),
                bias: 0.0,
            },
        ],
        synapses: Vec::new(),
        input: 1,
        output: 1,
    }
}

// ---------------------------------------------------------------------------
// serde boundary — acceptance criteria 1-3.
// ---------------------------------------------------------------------------

#[test]
fn serde_json_itself_rejects_a_bias_overflowing_the_json_exponent() {
    // Acceptance criterion 3: `"bias": 1e400` must not deserialise. It never
    // reaches our validator — `1e400` overflows `f64`, so `serde_json`'s own
    // number parser refuses it first. This test pins that upstream behaviour,
    // which is why it asserts the parser's range error rather than the
    // Issue #2133 message the sibling `1e39` cases carry.
    let err = serde_json::from_str::<CreatureJson>(&creature_json("1e400"))
        .expect_err("bias: 1e400 must be rejected at deserialisation");
    let msg = err.to_string();
    assert!(
        msg.contains("number out of range"),
        "rejection must come from serde_json's number parser: {msg}"
    );
}

#[test]
fn serde_json_itself_rejects_a_negative_bias_overflowing_the_json_exponent() {
    let err = serde_json::from_str::<CreatureJson>(&creature_json("-1e400"))
        .expect_err("bias: -1e400 must be rejected at deserialisation");
    let msg = err.to_string();
    assert!(
        msg.contains("number out of range"),
        "rejection must come from serde_json's number parser: {msg}"
    );
}

#[test]
fn deserialise_rejects_bias_overflowing_f32() {
    // The genuine hole: finite as JSON and as `f64`, infinite once narrowed.
    let err = serde_json::from_str::<CreatureJson>(&creature_json("1e39"))
        .expect_err("bias: 1e39 overflows f32 and must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("Issue #2133"),
        "error must cite the issue: {msg}"
    );
    assert!(
        msg.contains("bias"),
        "error must name the offending field: {msg}"
    );
}

#[test]
fn deserialise_rejects_negative_bias_overflowing_f32() {
    let err = serde_json::from_str::<CreatureJson>(&creature_json("-1e39"))
        .expect_err("bias: -1e39 overflows f32 and must be rejected");
    assert!(
        err.to_string().contains("Issue #2133"),
        "error must cite the issue"
    );
}

#[test]
fn deserialise_accepts_finite_biases() {
    for bias in ["0.0", "0.5", "-2.25", "3.0e38", "-3.0e38"] {
        let creature = serde_json::from_str::<CreatureJson>(&creature_json(bias))
            .unwrap_or_else(|e| panic!("finite bias {bias} must deserialise: {e}"));
        assert!(
            creature.neurons[0].bias.is_finite(),
            "bias {bias} must survive as a finite value"
        );
    }
}

#[test]
fn missing_bias_still_defaults_to_zero() {
    // The validator must not cost the field its `#[serde(default)]`.
    let creature = serde_json::from_str::<CreatureJson>(
        r#"{
            "neurons": [{"uuid": "output-0", "type": "output", "squash": "LOGISTIC"}],
            "synapses": [],
            "input": 1,
            "output": 1
        }"#,
    )
    .expect("an omitted bias must still default");
    assert_eq!(creature.neurons[0].bias, 0.0);
}

// ---------------------------------------------------------------------------
// Rust-constructed creatures — acceptance criteria 2 and 4.
// ---------------------------------------------------------------------------

#[test]
fn validate_creature_rejects_a_nan_bias() {
    // JSON has no NaN literal, so this is the only door NaN can use.
    let err = validate_creature(&creature_with_bias(f32::NAN))
        .expect_err("a NaN bias must be rejected at the gate");
    assert_eq!(err.error_kind(), DiscoveryErrorKind::DataValidation);
    let msg = err.to_string();
    assert!(
        msg.contains("Issue #2133") && msg.contains("hidden-0"),
        "error must cite the issue and name the neuron: {msg}"
    );
}

#[test]
fn validate_creature_rejects_an_infinite_bias() {
    for bias in [f32::INFINITY, f32::NEG_INFINITY] {
        let err = validate_creature(&creature_with_bias(bias))
            .expect_err("an infinite bias must be rejected at the gate");
        assert_eq!(err.error_kind(), DiscoveryErrorKind::DataValidation);
    }
}

#[test]
fn validate_creature_accepts_finite_biases() {
    for bias in [0.0, 0.5, -2.25, f32::MAX, f32::MIN] {
        validate_creature(&creature_with_bias(bias))
            .unwrap_or_else(|e| panic!("finite bias {bias} must pass the gate: {e}"));
    }
}

// ---------------------------------------------------------------------------
// Acceptance criterion 4 — the shipped entry point, which fronts the
// arithmetic / conversion / hashing consumption sites.
// ---------------------------------------------------------------------------

#[test]
fn entry_point_rejects_a_bias_that_overflows_f32() {
    let input = format!(
        r#"{{
            "parquetFile": "/nonexistent/issue-2133/discovery.parquet",
            "costOfGrowth": 1e-4,
            "creature": {}
        }}"#,
        creature_json("1e39")
    );
    let raw = rank_focus_neurons_internal(&input)
        .expect("rank_focus_neurons_internal must return a response, not Err");
    let parsed: serde_json::Value =
        serde_json::from_str(&raw).expect("response must be valid JSON");
    assert_eq!(
        parsed["success"], false,
        "an overflowing bias must never reach the consumption sites: {raw}"
    );
}
