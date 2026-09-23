//! Synapse weight finitude at the FFI boundary (Issue #2132).
//!
//! JSON has no `Infinity` literal, and there are two distinct routes to an
//! infinite weight:
//!
//! - `1e400` overflows **f64** itself, so `serde_json` rejects it with its own
//!   "number out of range" error before any field validator runs.
//! - `1e39` is a perfectly ordinary finite f64, but exceeds `f32::MAX` and so
//!   saturates to `f32::INFINITY` the moment serde casts it down. Nothing in
//!   serde objects — this is the reachable hole.
//!
//! An infinite weight then flows unchecked into every analysis site
//! (`synapse.weight * activation`), where it beats every contribution threshold
//! and silently bypasses dormancy, polarity-flip and noise detection, so the
//! caller is handed confidently wrong results.
//!
//! The weight is therefore validated once, at deserialisation, so all 23
//! consumption sites receive a finite value by construction rather than each
//! re-checking it.

use neat_ai_discovery::{CreatureJson, SynapseJson};

/// A standalone synapse payload carrying the given weight literal.
fn synapse_json(weight: &str) -> String {
    format!(r#"{{"fromUUID": "input-0", "toUUID": "output-0", "weight": {weight}}}"#)
}

/// A whole creature payload whose single synapse carries the given weight.
fn creature_json(weight: &str) -> String {
    format!(
        r#"{{
            "neurons": [
                {{"uuid": "output-0", "type": "output", "squash": "LOGISTIC", "bias": 0.0}}
            ],
            "synapses": [
                {{"fromUUID": "input-0", "toUUID": "output-0", "weight": {weight}}}
            ],
            "input": 2,
            "output": 1
        }}"#
    )
}

/// Asserts the weight literal is refused, whoever refuses it.
fn assert_weight_rejected(weight: &str) -> String {
    serde_json::from_str::<SynapseJson>(&synapse_json(weight))
        .expect_err("a non-finite weight must be rejected at deserialisation")
        .to_string()
}

/// Asserts the weight literal is refused *by this crate's finitude check*,
/// which names the requirement and cites the issue.
fn assert_weight_rejected_as_non_finite(weight: &str) {
    let msg = assert_weight_rejected(weight);
    assert!(
        msg.contains("finite"),
        "error for weight {weight} must name the finitude requirement: {msg}"
    );
    assert!(
        msg.contains("Issue #2132"),
        "error for weight {weight} must cite the issue: {msg}"
    );
}

// ---------------------------------------------------------------------------
// Deserialisation rejects non-finite weights
// ---------------------------------------------------------------------------

#[test]
fn deserialise_rejects_f64_overflowing_literals() {
    // `1e400` exceeds f64 range, so serde_json refuses it during number parsing
    // — before any field validator is reached. The payload is still rejected,
    // which is what the boundary owes the caller; the message is serde's own.
    for weight in ["1e400", "-1e400"] {
        assert_weight_rejected(weight);
    }
}

#[test]
fn deserialise_rejects_f32_saturating_magnitude() {
    // `1e39` is a perfectly finite f64, but saturates to `f32::INFINITY` once
    // serde casts it down — serde raises nothing, so this is the reachable
    // hole. Validation must inspect the stored f32, never the literal text.
    assert_weight_rejected_as_non_finite("1e39");
    assert_weight_rejected_as_non_finite("-3.5e38");
}

#[test]
fn deserialise_rejects_infinite_weight_nested_in_creature() {
    let err = serde_json::from_str::<CreatureJson>(&creature_json("1e39"))
        .expect_err("an infinite synapse weight must sink the whole creature payload");
    assert!(
        err.to_string().contains("Issue #2132"),
        "creature-level error must cite the issue: {err}"
    );
}

// ---------------------------------------------------------------------------
// Finite weights are unaffected
// ---------------------------------------------------------------------------

#[test]
fn deserialise_accepts_finite_weights() {
    for weight in ["0.5", "-0.75", "0", "3.4e38", "-3.4e38"] {
        let synapse: SynapseJson = serde_json::from_str(&synapse_json(weight))
            .unwrap_or_else(|e| panic!("finite weight {weight} must deserialise: {e}"));
        assert!(
            synapse.weight.is_finite(),
            "finite weight {weight} must survive deserialisation"
        );
    }

    let synapse: SynapseJson =
        serde_json::from_str(&synapse_json("0.5")).expect("a finite weight must deserialise");
    assert_eq!(synapse.weight, 0.5, "the weight value must be preserved");

    let creature: CreatureJson = serde_json::from_str(&creature_json("0.5"))
        .expect("a finite creature weight must deserialise");
    assert_eq!(creature.synapses[0].weight, 0.5);
}

#[test]
fn deserialise_defaults_missing_weight_to_zero() {
    let synapse: SynapseJson =
        serde_json::from_str(r#"{"fromUUID": "input-0", "toUUID": "output-0"}"#)
            .expect("an omitted weight must still default");
    assert_eq!(
        synapse.weight, 0.0,
        "an omitted weight must keep defaulting to 0.0"
    );
}

// ---------------------------------------------------------------------------
// The shipped FFI entry point rejects it too (Issue #1806 convention)
// ---------------------------------------------------------------------------

#[test]
fn record_discovery_rejects_infinite_synapse_weight() {
    let temp = tempfile::tempdir().expect("temp dir must be creatable");
    let input = format!(
        r#"{{
            "creature": {},
            "training_data": [
                {{
                    "input": [0.5, 0.25],
                    "output": [0.7],
                    "neuron_data": [
                        {{"neuron_uuid": "output-0", "activation": 0.7, "errors": [-0.05]}}
                    ]
                }}
            ],
            "temp_dir": "{}"
        }}"#,
        creature_json("1e39"),
        temp.path().display()
    );

    let response =
        neat_ai_discovery::record_discovery_internal(&input).expect("internal call must not panic");
    let parsed: serde_json::Value =
        serde_json::from_str(&response).expect("response must be valid JSON");

    assert_eq!(
        parsed["success"], false,
        "an infinite synapse weight must fail the entry point: {response}"
    );
    assert_eq!(
        parsed["errorKind"], "data_validation",
        "the rejection must classify as data_validation: {response}"
    );
    let err_msg = parsed["error"]
        .as_str()
        .expect("error message must be present");
    assert!(
        err_msg.contains("Issue #2132"),
        "entry-point error must cite the issue: {err_msg}"
    );
}
