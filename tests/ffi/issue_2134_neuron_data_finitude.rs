//! `NeuronData` float finitude at the FFI boundary (Issue #2134).
//!
//! JSON has no `Infinity` literal, and there are two distinct routes to an
//! infinite activation, value or error:
//!
//! - `1e400` overflows **f64** itself, so `serde_json` rejects it with its own
//!   "number out of range" error before any field validator runs.
//! - `1e39` is a perfectly ordinary finite f64, but exceeds `f32::MAX` and so
//!   saturates to `f32::INFINITY` the moment serde casts it down. Nothing in
//!   serde objects — this is the reachable hole.
//!
//! An infinite activation then flows unchecked into every analysis site: means,
//! variances and covariances over the records become Infinity or NaN, target
//! values (`activation + error`) become Infinity, and the caller is handed
//! confidently wrong metrics.
//!
//! The three float fields are therefore validated once, at deserialisation, so
//! every consumption site downstream of [`neat_ai_discovery::NeuronData`]
//! receives finite values by construction rather than each re-checking them.

use neat_ai_discovery::NeuronData;

/// A standalone neuron-data payload carrying the given field literals.
fn neuron_data_json(activation: &str, value: &str, errors: &str) -> String {
    format!(
        r#"{{"neuron_uuid": "output-0", "activation": {activation}, "value": {value}, "errors": {errors}}}"#
    )
}

/// A payload whose only non-finite candidate is the activation.
fn activation_json(activation: &str) -> String {
    neuron_data_json(activation, "0.25", "[-0.05]")
}

/// A payload whose only non-finite candidate is the optional value.
fn value_json(value: &str) -> String {
    neuron_data_json("0.7", value, "[-0.05]")
}

/// A payload whose only non-finite candidate lives inside the errors vector.
fn errors_json(errors: &str) -> String {
    neuron_data_json("0.7", "0.25", errors)
}

/// Asserts the payload is refused, whoever refuses it.
fn assert_rejected(json: &str) -> String {
    serde_json::from_str::<NeuronData>(json)
        .expect_err("a non-finite float must be rejected at deserialisation")
        .to_string()
}

/// Asserts the payload is refused *by this crate's finitude check*, which names
/// the field, the requirement and the issue.
fn assert_rejected_as_non_finite(json: &str, field: &str) {
    let msg = assert_rejected(json);
    assert!(
        msg.contains("finite"),
        "error for {json} must name the finitude requirement: {msg}"
    );
    assert!(
        msg.contains(field),
        "error for {json} must name the offending field {field}: {msg}"
    );
    assert!(
        msg.contains("Issue #2134"),
        "error for {json} must cite the issue: {msg}"
    );
}

// ---------------------------------------------------------------------------
// Deserialisation rejects non-finite floats
// ---------------------------------------------------------------------------

#[test]
fn deserialise_rejects_f64_overflowing_literals() {
    // `1e400` exceeds f64 range, so serde_json refuses it during number parsing
    // — before any field validator is reached. The payload is still rejected,
    // which is what the boundary owes the caller; the message is serde's own.
    for json in [
        activation_json("1e400"),
        value_json("-1e400"),
        errors_json("[1e400]"),
    ] {
        assert_rejected(&json);
    }
}

#[test]
fn deserialise_rejects_f32_saturating_activation() {
    // `1e39` is a perfectly finite f64, but saturates to `f32::INFINITY` once
    // serde casts it down — serde raises nothing, so this is the reachable
    // hole. Validation must inspect the stored f32, never the literal text.
    assert_rejected_as_non_finite(&activation_json("1e39"), "activation");
    assert_rejected_as_non_finite(&activation_json("-3.5e38"), "activation");
}

#[test]
fn deserialise_rejects_f32_saturating_value() {
    assert_rejected_as_non_finite(&value_json("1e39"), "value");
    assert_rejected_as_non_finite(&value_json("-3.5e38"), "value");
}

#[test]
fn deserialise_rejects_f32_saturating_error_element() {
    // Every element is checked, not merely the first — a single poisoned entry
    // anywhere in the vector is enough to corrupt the error statistics.
    assert_rejected_as_non_finite(&errors_json("[1e39]"), "errors");
    assert_rejected_as_non_finite(&errors_json("[-0.05, 1e39]"), "errors");
    assert_rejected_as_non_finite(&errors_json("[-0.05, 0.1, -3.5e38]"), "errors");
}

#[test]
fn deserialise_rejects_infinite_activation_nested_in_training_record() {
    let record = format!(
        r#"{{
            "input": [0.5, 0.25],
            "output": [0.7],
            "neuron_data": [{}]
        }}"#,
        activation_json("1e39")
    );
    let err = serde_json::from_str::<neat_ai_discovery::TrainingRecord>(&record)
        .expect_err("an infinite activation must sink the whole training record");
    assert!(
        err.to_string().contains("Issue #2134"),
        "record-level error must cite the issue: {err}"
    );
}

// ---------------------------------------------------------------------------
// Finite floats are unaffected
// ---------------------------------------------------------------------------

#[test]
fn deserialise_accepts_finite_floats() {
    for literal in ["0.7", "-0.75", "0", "1e38", "3.4e38", "-3.4e38"] {
        let parsed: NeuronData =
            serde_json::from_str(&neuron_data_json(literal, literal, &format!("[{literal}]")))
                .unwrap_or_else(|e| panic!("finite literal {literal} must deserialise: {e}"));
        assert!(
            parsed.activation.is_finite(),
            "finite activation {literal} must survive deserialisation"
        );
        assert!(
            parsed.value.expect("value must be present").is_finite(),
            "finite value {literal} must survive deserialisation"
        );
        assert!(
            parsed.errors.iter().all(|e| e.is_finite()),
            "finite error {literal} must survive deserialisation"
        );
    }

    let parsed: NeuronData = serde_json::from_str(&neuron_data_json("0.7", "0.25", "[-0.05, 0.1]"))
        .expect("a finite payload must deserialise");
    assert_eq!(parsed.activation, 0.7, "the activation must be preserved");
    assert_eq!(parsed.value, Some(0.25), "the value must be preserved");
    assert_eq!(
        parsed.errors,
        vec![-0.05, 0.1],
        "the errors must be preserved in order"
    );
}

#[test]
fn deserialise_defaults_missing_value_to_none() {
    let parsed: NeuronData =
        serde_json::from_str(r#"{"neuron_uuid": "output-0", "activation": 0.7, "errors": []}"#)
            .expect("an omitted value must still default");
    assert_eq!(
        parsed.value, None,
        "an omitted value must keep defaulting to None"
    );
    assert!(
        parsed.errors.is_empty(),
        "an empty errors vector must remain acceptable"
    );
}

#[test]
fn deserialise_accepts_explicit_null_value() {
    let parsed: NeuronData = serde_json::from_str(
        r#"{"neuron_uuid": "output-0", "activation": 0.7, "value": null, "errors": [-0.05]}"#,
    )
    .expect("an explicit null value must still deserialise");
    assert_eq!(parsed.value, None, "an explicit null must map to None");
}

// ---------------------------------------------------------------------------
// The shipped FFI entry point rejects it too (Issue #1806 convention)
// ---------------------------------------------------------------------------

#[test]
fn record_discovery_rejects_infinite_activation() {
    let temp = tempfile::tempdir().expect("temp dir must be creatable");
    let input = format!(
        r#"{{
            "creature": {{
                "neurons": [
                    {{"uuid": "output-0", "type": "output", "squash": "LOGISTIC", "bias": 0.0}}
                ],
                "synapses": [
                    {{"fromUUID": "input-0", "toUUID": "output-0", "weight": 0.5}}
                ],
                "input": 2,
                "output": 1
            }},
            "training_data": [
                {{
                    "input": [0.5, 0.25],
                    "output": [0.7],
                    "neuron_data": [{}]
                }}
            ],
            "temp_dir": "{}"
        }}"#,
        activation_json("1e39"),
        temp.path().display()
    );

    let response =
        neat_ai_discovery::record_discovery_internal(&input).expect("internal call must not panic");
    let parsed: serde_json::Value =
        serde_json::from_str(&response).expect("response must be valid JSON");

    assert_eq!(
        parsed["success"], false,
        "an infinite activation must fail the entry point: {response}"
    );
    assert_eq!(
        parsed["errorKind"], "data_validation",
        "the rejection must classify as data_validation: {response}"
    );
    let err_msg = parsed["error"]
        .as_str()
        .expect("error message must be present");
    assert!(
        err_msg.contains("Issue #2134"),
        "entry-point error must cite the issue: {err_msg}"
    );
}
