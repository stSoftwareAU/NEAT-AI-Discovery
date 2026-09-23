//! `TrainingRecord` input/output vector finitude at the FFI boundary
//! (Issue #2135).
//!
//! JSON has no `Infinity` literal, and there are two distinct routes to an
//! infinite feature or target element:
//!
//! - `1e400` overflows **f64** itself, so `serde_json` rejects it with its own
//!   "number out of range" error before any field validator runs.
//! - `1e39` is a perfectly ordinary finite f64, but exceeds `f32::MAX` and so
//!   saturates to `f32::INFINITY` the moment serde casts it down. Nothing in
//!   serde objects — this is the reachable hole.
//!
//! An infinite element then flows unchecked into the statistics built over the
//! training records: covariance and correlation matrices become Infinity or
//! NaN, and the caller is handed confidently wrong training metrics.
//!
//! Both vectors are therefore validated once, at deserialisation, so every
//! consumption site downstream of [`neat_ai_discovery::TrainingRecord`]
//! receives finite vectors by construction rather than each re-checking them.

use neat_ai_discovery::TrainingRecord;

/// A standalone training record carrying the given vector literals.
fn record_json(input: &str, output: &str) -> String {
    format!(r#"{{"input": {input}, "output": {output}}}"#)
}

/// A record whose only non-finite candidate lives in the input vector.
fn input_json(input: &str) -> String {
    record_json(input, "[0.7]")
}

/// A record whose only non-finite candidate lives in the output vector.
fn output_json(output: &str) -> String {
    record_json("[0.5, 0.25]", output)
}

/// Asserts the payload is refused, whoever refuses it.
fn assert_rejected(json: &str) -> String {
    serde_json::from_str::<TrainingRecord>(json)
        .expect_err("a non-finite vector element must be rejected at deserialisation")
        .to_string()
}

/// Asserts the payload is refused *by this crate's finitude check*, which names
/// the field, the offending index, the requirement and the issue.
fn assert_rejected_as_non_finite(json: &str, field: &str, index: usize) {
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
        msg.contains(&format!("{field}[{index}]")),
        "error for {json} must name the offending index {index}: {msg}"
    );
    assert!(
        msg.contains("Issue #2135"),
        "error for {json} must cite the issue: {msg}"
    );
}

// ---------------------------------------------------------------------------
// Deserialisation rejects non-finite vector elements
// ---------------------------------------------------------------------------

#[test]
fn deserialise_rejects_f64_overflowing_literals() {
    // `1e400` exceeds f64 range, so serde_json refuses it during number parsing
    // — before any field validator is reached. The payload is still rejected,
    // which is what the boundary owes the caller; the message is serde's own.
    for json in [
        input_json("[1e400, 1e400]"),
        output_json("[-1e400]"),
        input_json("[0.5, 1e400]"),
    ] {
        assert_rejected(&json);
    }
}

#[test]
fn deserialise_rejects_f32_saturating_input_element() {
    // `1e39` is a perfectly finite f64, but saturates to `f32::INFINITY` once
    // serde casts it down — serde raises nothing, so this is the reachable
    // hole. Validation must inspect the stored f32, never the literal text.
    //
    // Every element is checked, not merely the first — a single poisoned entry
    // anywhere in the vector is enough to corrupt the covariance matrix.
    assert_rejected_as_non_finite(&input_json("[1e39]"), "input", 0);
    assert_rejected_as_non_finite(&input_json("[0.5, 1e39, 0.25]"), "input", 1);
    assert_rejected_as_non_finite(&input_json("[0.5, 0.25, -3.5e38]"), "input", 2);
}

#[test]
fn deserialise_rejects_f32_saturating_output_element() {
    assert_rejected_as_non_finite(&output_json("[1e39]"), "output", 0);
    assert_rejected_as_non_finite(&output_json("[0.7, 1e39, 0.1]"), "output", 1);
    assert_rejected_as_non_finite(&output_json("[0.7, 0.1, -3.5e38]"), "output", 2);
}

// ---------------------------------------------------------------------------
// Finite vectors are unaffected
// ---------------------------------------------------------------------------

#[test]
fn deserialise_accepts_finite_vectors() {
    for literal in ["0.7", "-0.75", "0", "1e38", "3.4e38", "-3.4e38"] {
        let json = record_json(&format!("[{literal}, 0.5]"), &format!("[{literal}]"));
        let parsed: TrainingRecord = serde_json::from_str(&json)
            .unwrap_or_else(|e| panic!("finite literal {literal} must deserialise: {e}"));
        assert!(
            parsed.input.iter().all(|v| v.is_finite()),
            "finite input {literal} must survive deserialisation"
        );
        assert!(
            parsed.output.iter().all(|v| v.is_finite()),
            "finite output {literal} must survive deserialisation"
        );
    }

    let parsed: TrainingRecord = serde_json::from_str(&record_json("[0.5, 0.25]", "[0.7, -0.1]"))
        .expect("a finite record must deserialise");
    assert_eq!(
        parsed.input,
        vec![0.5, 0.25],
        "the input vector must be preserved in order"
    );
    assert_eq!(
        parsed.output,
        vec![0.7, -0.1],
        "the output vector must be preserved in order"
    );
    assert!(
        parsed.neuron_data.is_none(),
        "an omitted neuron_data must keep defaulting to None"
    );
}

#[test]
fn deserialise_accepts_empty_vectors() {
    let parsed: TrainingRecord =
        serde_json::from_str(&record_json("[]", "[]")).expect("empty vectors must deserialise");
    assert!(
        parsed.input.is_empty() && parsed.output.is_empty(),
        "empty vectors must remain acceptable"
    );
}

// ---------------------------------------------------------------------------
// The shipped FFI entry point rejects it too (Issue #1806 convention)
// ---------------------------------------------------------------------------

/// A record carrying valid neuron data, so the vector under test is the only
/// thing the entry point can possibly object to.
fn recorded_json(input: &str, output: &str) -> String {
    format!(
        r#"{{"input": {input}, "output": {output}, "neuron_data": [{{"neuron_uuid": "output-0", "activation": 0.7, "value": 0.25, "errors": [-0.05]}}]}}"#
    )
}

/// Drives the shipped entry point with `training_data` carrying `record`, and
/// returns its parsed JSON response.
fn record_discovery_with(record: &str) -> serde_json::Value {
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
            "training_data": [{record}],
            "temp_dir": "{}"
        }}"#,
        temp.path().display()
    );

    let response =
        neat_ai_discovery::record_discovery_internal(&input).expect("internal call must not panic");
    serde_json::from_str(&response).expect("response must be valid JSON")
}

#[test]
fn record_discovery_rejects_infinite_input_vector() {
    let parsed = record_discovery_with(&recorded_json("[0.5, 1e39]", "[0.7]"));
    assert_eq!(
        parsed["success"], false,
        "an infinite input element must fail the entry point: {parsed}"
    );
    assert_eq!(
        parsed["errorKind"], "data_validation",
        "the rejection must classify as data_validation: {parsed}"
    );
    let err_msg = parsed["error"]
        .as_str()
        .expect("error message must be present");
    assert!(
        err_msg.contains("Issue #2135"),
        "entry-point error must cite the issue: {err_msg}"
    );
}

#[test]
fn record_discovery_rejects_infinite_output_vector() {
    let parsed = record_discovery_with(&recorded_json("[0.5, 0.25]", "[1e39]"));
    assert_eq!(
        parsed["success"], false,
        "an infinite output element must fail the entry point: {parsed}"
    );
    let err_msg = parsed["error"]
        .as_str()
        .expect("error message must be present");
    assert!(
        err_msg.contains("Issue #2135") && err_msg.contains("output"),
        "entry-point error must name the field and cite the issue: {err_msg}"
    );
}
