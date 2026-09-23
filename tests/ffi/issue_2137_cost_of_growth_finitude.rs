//! `costOfGrowth` finitude at the FFI boundary (Issue #2137, SEC-2090-06).
//!
//! `RankFocusNeuronsInput::cost_of_growth` had no finitude validation of its
//! own. A non-finite value was only ever mitigated *downstream*, by
//! `effective_cost_of_growth` (Issue #1807), which replaced it with
//! [`DEFAULT_COST_OF_GROWTH`] and logged a WARN. That is inconsistent with the
//! other float fields on this surface (Issues #2132, #2133, #2134, #2136),
//! which reject Infinity and NaN at the boundary instead of letting a nonsense
//! request through as a successful response.
//!
//! Two distinct rejection mechanisms are pinned here, because JSON has no
//! `Infinity` or `NaN` literal:
//!
//! - `1e400` overflows **f64**, so `serde_json`'s own number parser refuses it
//!   before any custom validator runs. That is the literal named in the issue,
//!   and the test below pins that it is genuinely refused.
//! - `1e39` / `-1e39` / `1e300` are finite as f64 but saturate to
//!   `±f32::INFINITY` when narrowed to `f32`. That is the reachable hole, and
//!   only this crate's own boundary check refuses it — the error names the
//!   field, the finitude requirement and Issue #2137.
//!
//! The boundary check is **finitude only**. Zero, negative and underflowing
//! (`1e-60`) costs are still accepted here and still handled downstream by the
//! Issue #1807 fallback, so that contract is unchanged — the tests below pin
//! that split explicitly.

use neat_ai_discovery::{
    CreatureJson, NeuronJson, RankFocusNeuronsInput, SynapseJson, rank_focus_neurons_internal,
};
use serde_json::{Value, json};

// ============================================================================
// Helpers
// ============================================================================

/// A discovery parquet that cannot be opened — the focus path is structure-only
/// (Issue #1766), so a successful response also proves no decode was attempted.
const MISSING_PARQUET: &str = "/nonexistent/issue-2137/discovery.parquet";

/// A cost of growth large enough to clear the noise floor on [`make_creature`],
/// so the acceptance assertions below are not vacuous on an empty list.
const CLEARS_NOISE_FLOOR: &str = "1e-4";

fn neuron(uuid: &str, neuron_type: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
        squash: "IDENTITY".to_string(),
        bias: 0.0,
    }
}

fn synapse(from: &str, to: &str, weight: f32) -> SynapseJson {
    SynapseJson {
        from_uuid: from.to_string(),
        to_uuid: to.to_string(),
        weight,
        synapse_type: None,
    }
}

/// One input feeding a dominant hidden neuron and three negligible ones, ordered
/// inputs → hidden → output so the forward-only FFI gate (Issue #1184) passes.
fn make_creature() -> CreatureJson {
    CreatureJson {
        neurons: vec![
            neuron("in-0", "input"),
            neuron("h-high", "hidden"),
            neuron("h-low-a", "hidden"),
            neuron("h-low-b", "hidden"),
            neuron("h-low-c", "hidden"),
            neuron("out", "output"),
        ],
        synapses: vec![
            synapse("in-0", "h-high", 0.5),
            synapse("in-0", "h-low-a", 0.5),
            synapse("in-0", "h-low-b", 0.5),
            synapse("in-0", "h-low-c", 0.5),
            synapse("h-high", "out", 1.0),
            synapse("h-low-a", "out", 1e-8),
            synapse("h-low-b", "out", 1e-8),
            synapse("h-low-c", "out", 1e-8),
        ],
        input: 1,
        output: 1,
    }
}

/// A focus-ranking request carrying `costOfGrowth` as a raw JSON literal, so
/// payloads `serde_json::Value` cannot represent (`NaN`, `1e400`) are testable.
fn request_json(cost_literal: &str) -> String {
    let creature =
        serde_json::to_string(&make_creature()).expect("the test creature must serialise");
    format!(
        r#"{{
            "parquetFile": "{MISSING_PARQUET}",
            "creature": {creature},
            "maxResults": 64,
            "focusSetSize": 4,
            "focusSelectionCursor": 0,
            "costOfGrowth": {cost_literal}
        }}"#
    )
}

/// The same request with `costOfGrowth` omitted entirely.
fn request_json_without_cost_of_growth() -> String {
    let creature =
        serde_json::to_string(&make_creature()).expect("the test creature must serialise");
    format!(
        r#"{{
            "parquetFile": "{MISSING_PARQUET}",
            "creature": {creature},
            "maxResults": 64
        }}"#
    )
}

/// Asserts the literal is refused by *something* — `serde_json` itself or this
/// crate's validator — and returns the error message for further inspection.
fn assert_rejected(cost_literal: &str) -> String {
    let json = request_json(cost_literal);
    serde_json::from_str::<RankFocusNeuronsInput>(&json)
        .expect_err(&format!("costOfGrowth {cost_literal} must be rejected"))
        .to_string()
}

/// Asserts the literal is refused *by this crate's finitude check*, which names
/// the field, the requirement and the issue.
fn assert_rejected_as_non_finite(cost_literal: &str) {
    let msg = assert_rejected(cost_literal);
    assert!(
        msg.contains("finite"),
        "error for {cost_literal} must name the finitude requirement: {msg}"
    );
    assert!(
        msg.contains("costOfGrowth"),
        "error for {cost_literal} must name the offending field: {msg}"
    );
    assert!(
        msg.contains("Issue #2137"),
        "error for {cost_literal} must cite the issue: {msg}"
    );
}

/// Reads `cost_of_growth` back out after a successful parse.
fn parsed_cost_of_growth(json: &str) -> Option<f32> {
    serde_json::from_str::<RankFocusNeuronsInput>(json)
        .expect("a finite costOfGrowth must parse")
        .cost_of_growth
}

/// Drives the shipped FFI focus path and returns the decoded response.
fn ffi_focus_response(cost_literal: &str) -> Value {
    let raw = rank_focus_neurons_internal(&request_json(cost_literal)).expect("FFI focus path");
    serde_json::from_str(&raw).expect("FFI response JSON")
}

// ============================================================================
// Rejection — the literal named in the issue
// ============================================================================

/// `1e400` overflows f64, so `serde_json` itself refuses it. The custom
/// validator never sees the value; this test pins that the payload named in the
/// issue is genuinely rejected rather than silently becoming Infinity.
#[test]
fn deserialise_rejects_f64_overflowing_cost_of_growth() {
    for literal in ["1e400", "-1e400"] {
        let msg = assert_rejected(literal);
        assert!(
            !msg.is_empty(),
            "serde_json must report why {literal} was rejected"
        );
    }
}

/// `Infinity` and `NaN` are not JSON literals, so the parser refuses the tokens
/// outright. Pinned so the contract stays explicit.
#[test]
fn deserialise_rejects_bare_infinity_and_nan_tokens() {
    for literal in ["Infinity", "-Infinity", "NaN"] {
        assert_rejected(literal);
    }
}

// ============================================================================
// Rejection — the reachable f32 narrowing hole
// ============================================================================

/// Finite as f64, `±Infinity` once narrowed to f32. Only this crate's boundary
/// check can refuse these, and its message must be diagnosable.
#[test]
fn deserialise_rejects_f32_saturating_cost_of_growth() {
    for literal in ["1e39", "-1e39", "1e300", "-3.5e38"] {
        assert_rejected_as_non_finite(literal);
    }
}

// ============================================================================
// Acceptance — finite costs are unaffected by the new boundary check
// ============================================================================

#[test]
fn deserialise_accepts_finite_costs_of_growth() {
    for literal in ["1e-7", "1e-4", "0.01", "1.0", "3.4e38"] {
        let expected: f32 = literal.parse().expect("test literal must parse as f32");
        let parsed = parsed_cost_of_growth(&request_json(literal))
            .expect("a present costOfGrowth must survive as Some");
        assert!(
            (parsed - expected).abs() <= expected.abs() * f32::EPSILON,
            "costOfGrowth {literal} must survive deserialisation, got {parsed}"
        );
    }
}

/// The boundary check is finitude-only. Zero, negative and underflowing costs
/// are finite, so they still reach the Issue #1807 downstream fallback rather
/// than being rejected here — that division of responsibility is the contract.
#[test]
fn deserialise_accepts_finite_but_unusable_costs_of_growth() {
    for literal in ["0.0", "-1.0", "-1e-4", "1e-60"] {
        let parsed = parsed_cost_of_growth(&request_json(literal))
            .expect("a finite costOfGrowth must reach the downstream guard, not be rejected here");
        assert!(
            parsed.is_finite(),
            "{literal} is finite and must be passed through, got {parsed}"
        );
    }
}

#[test]
fn deserialise_defaults_absent_and_null_cost_of_growth() {
    assert_eq!(
        parsed_cost_of_growth(&request_json_without_cost_of_growth()),
        None,
        "an absent costOfGrowth must stay None so the default applies"
    );
    assert_eq!(
        parsed_cost_of_growth(&request_json("null")),
        None,
        "an explicitly null costOfGrowth must stay None so the default applies"
    );
}

// ============================================================================
// Acceptance criterion 2 — the real FFI entry point
// ============================================================================

/// `rank_focus_neurons_internal` must refuse a non-finite cost with the
/// structured failure envelope rather than ranking against Infinity.
#[test]
fn rank_focus_neurons_rejects_non_finite_cost_of_growth() {
    for literal in ["1e39", "-1e39"] {
        let response = ffi_focus_response(literal);

        assert_eq!(
            response["success"], false,
            "a non-finite costOfGrowth must not succeed: {response:?}"
        );
        assert_eq!(
            response["errorKind"], "data_validation",
            "a non-finite costOfGrowth is invalid input: {response:?}"
        );
        let error = response["error"]
            .as_str()
            .expect("a failed response must carry an error message");
        assert!(
            error.contains("Issue #2137"),
            "the FFI error must cite the issue: {error}"
        );
        assert!(
            error.contains("costOfGrowth"),
            "the FFI error must name the offending field: {error}"
        );
    }
}

// ============================================================================
// Acceptance criterion 4 — ranking receives valid finite cost values
// ============================================================================

/// The keys of `candidate` whose value is a JSON number.
///
/// `serde_json` serialises a non-finite `f32` as `null`, never as a number, so
/// "this key is still a number" is exactly the assertion that proves no
/// Infinity or NaN reached the ranking arithmetic.
fn numeric_keys(candidate: &Value) -> Vec<String> {
    candidate
        .as_object()
        .expect("a removal candidate must be a JSON object")
        .iter()
        .filter(|(_, v)| v.is_number())
        .map(|(k, _)| k.clone())
        .collect()
}

/// Every finite cost that clears the boundary must produce ranking output whose
/// numeric fields are all present and finite — no `null` standing in for an
/// Infinity that survived the narrowing.
#[test]
fn finite_cost_of_growth_yields_finite_ranking_values() {
    let baseline = ffi_focus_response(CLEARS_NOISE_FLOOR);
    let baseline_candidates = baseline["removalCandidates"]
        .as_array()
        .expect("removalCandidates must be an array")
        .clone();
    assert!(
        !baseline_candidates.is_empty(),
        "a cost clearing the noise floor must yield candidates, or this test proves nothing: {baseline:?}"
    );
    let expected_numeric = numeric_keys(&baseline_candidates[0]);
    assert!(
        !expected_numeric.is_empty(),
        "a removal candidate must carry numeric ranking fields: {:?}",
        baseline_candidates[0]
    );

    for literal in ["1e-7", CLEARS_NOISE_FLOOR, "0.01", "1.0", "3.4e38"] {
        let response = ffi_focus_response(literal);
        assert_eq!(
            response["success"], true,
            "a finite costOfGrowth must be accepted: {response:?}"
        );

        for candidate in response["removalCandidates"]
            .as_array()
            .expect("removalCandidates must be an array")
        {
            for key in &expected_numeric {
                let value = candidate
                    .get(key)
                    .unwrap_or_else(|| panic!("candidate must keep field {key}: {candidate:?}"));
                let number = value.as_f64().unwrap_or_else(|| {
                    panic!(
                        "field {key} must stay a number for costOfGrowth {literal} \
                         — null means a non-finite float was serialised: {candidate:?}"
                    )
                });
                assert!(
                    number.is_finite(),
                    "field {key} must be finite for costOfGrowth {literal}, got {number}"
                );
            }
        }
    }
}
