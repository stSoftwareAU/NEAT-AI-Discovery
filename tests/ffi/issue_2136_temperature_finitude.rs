//! Temperature finitude at the FFI boundary (Issue #2136, SEC-2090-05).
//!
//! `temperature` is deserialised into every analysis request struct. Before
//! this issue the field had no finitude validation of its own — a non-finite
//! value was only ever mitigated *downstream*, by the range clamps in
//! `analysis::constants::temperature`. That is inconsistent with the other
//! float fields on the FFI surface (Issues #2132, #2133, #2134), which reject
//! Infinity and NaN at the boundary.
//!
//! Two distinct rejection mechanisms are pinned here, because JSON has no
//! `Infinity` or `NaN` literal:
//!
//! - `1e400` overflows **f64**, so `serde_json`'s own number parser refuses it
//!   before any custom validator runs. That is the literal named in the issue,
//!   and the test below pins that it is genuinely refused.
//! - `1e39` / `-3.5e38` are finite as f64 but saturate to `±f32::INFINITY`
//!   when narrowed to `f32`. That is the reachable hole, and only this crate's
//!   own boundary check refuses it — the error names the field, the finitude
//!   requirement and Issue #2136.

use neat_ai_discovery::analysis::constants::{
    DEFAULT_TEMPERATURE, MAX_TEMPERATURE, MIN_TEMPERATURE, scale_mh_temperature,
    scale_ratio_by_temperature, scale_threshold_by_temperature,
};
use neat_ai_discovery::{
    AnalyzeAllInput, AnalyzeNeuronsInput, AnalyzeParallelInput, AnalyzeSynapsesInput,
    analyze_parallel_internal,
};

// ============================================================================
// Helpers
// ============================================================================

/// A minimal analysis request carrying `temperature` as a raw JSON literal.
///
/// The three mandatory fields are shared by all four request structs, so one
/// builder feeds every variant.
fn request_json(temperature_literal: &str) -> String {
    format!(
        r#"{{
            "parquetFile": "/tmp/issue-2136.parquet",
            "creature": {{ "neurons": [], "synapses": [], "input": 1, "output": 1 }},
            "focusNeurons": [],
            "temperature": {temperature_literal}
        }}"#
    )
}

/// The same request with `temperature` omitted entirely.
fn request_json_without_temperature() -> String {
    r#"{
        "parquetFile": "/tmp/issue-2136.parquet",
        "creature": { "neurons": [], "synapses": [], "input": 1, "output": 1 },
        "focusNeurons": []
    }"#
    .to_string()
}

/// Asserts every request variant refuses the literal, whoever refuses it, and
/// returns the four error messages for further inspection.
fn assert_rejected_by_every_variant(temperature_literal: &str) -> Vec<String> {
    let json = request_json(temperature_literal);
    let expectation = format!("temperature {temperature_literal} must be rejected: {json}");

    vec![
        serde_json::from_str::<AnalyzeParallelInput>(&json)
            .expect_err(&expectation)
            .to_string(),
        serde_json::from_str::<AnalyzeSynapsesInput>(&json)
            .expect_err(&expectation)
            .to_string(),
        serde_json::from_str::<AnalyzeNeuronsInput>(&json)
            .expect_err(&expectation)
            .to_string(),
        serde_json::from_str::<AnalyzeAllInput>(&json)
            .expect_err(&expectation)
            .to_string(),
    ]
}

/// Asserts the literal is refused *by this crate's finitude check*, which names
/// the field, the requirement and the issue.
fn assert_rejected_as_non_finite(temperature_literal: &str) {
    for msg in assert_rejected_by_every_variant(temperature_literal) {
        assert!(
            msg.contains("finite"),
            "error for {temperature_literal} must name the finitude requirement: {msg}"
        );
        assert!(
            msg.contains("temperature"),
            "error for {temperature_literal} must name the offending field: {msg}"
        );
        assert!(
            msg.contains("Issue #2136"),
            "error for {temperature_literal} must cite the issue: {msg}"
        );
    }
}

/// Reads `temperature` back out of each variant after a successful parse.
fn parsed_temperatures(json: &str) -> Vec<f32> {
    vec![
        serde_json::from_str::<AnalyzeParallelInput>(json)
            .expect("parallel input must parse")
            .temperature,
        serde_json::from_str::<AnalyzeSynapsesInput>(json)
            .expect("synapses input must parse")
            .temperature,
        serde_json::from_str::<AnalyzeNeuronsInput>(json)
            .expect("neurons input must parse")
            .temperature,
        serde_json::from_str::<AnalyzeAllInput>(json)
            .expect("all input must parse")
            .temperature,
    ]
}

// ============================================================================
// Rejection — the literal named in the issue
// ============================================================================

/// `1e400` overflows f64, so `serde_json` itself refuses it. The custom
/// validator never sees the value; this test pins that the payload named in
/// the issue is genuinely rejected rather than silently becoming Infinity.
#[test]
fn deserialise_rejects_f64_overflowing_temperature() {
    for msg in assert_rejected_by_every_variant("1e400") {
        assert!(
            !msg.is_empty(),
            "serde_json must report why 1e400 was rejected"
        );
    }
    for msg in assert_rejected_by_every_variant("-1e400") {
        assert!(
            !msg.is_empty(),
            "serde_json must report why -1e400 was rejected"
        );
    }
}

/// `Infinity` and `NaN` are not JSON literals, so the parser refuses the
/// tokens outright. Pinned so the contract stays explicit.
#[test]
fn deserialise_rejects_bare_infinity_and_nan_tokens() {
    for literal in ["Infinity", "-Infinity", "NaN"] {
        assert_rejected_by_every_variant(literal);
    }
}

// ============================================================================
// Rejection — the reachable f32 narrowing hole
// ============================================================================

/// Finite as f64, `±Infinity` once narrowed to f32. Only this crate's boundary
/// check can refuse these, and it must do so for every request variant.
#[test]
fn deserialise_rejects_f32_saturating_temperature() {
    for literal in ["1e39", "-3.5e38", "1e300"] {
        assert_rejected_as_non_finite(literal);
    }
}

// ============================================================================
// Acceptance — valid finite temperatures are unaffected
// ============================================================================

#[test]
fn deserialise_accepts_finite_temperatures() {
    for literal in ["1.0", "0.5", "2.0", "1.25", "0.01", "5.0", "3.4e38"] {
        let json = request_json(literal);
        let expected: f32 = literal.parse().expect("test literal must parse as f32");
        for parsed in parsed_temperatures(&json) {
            assert!(
                (parsed - expected).abs() < f32::EPSILON,
                "temperature {literal} must survive deserialisation, got {parsed}"
            );
        }
    }
}

#[test]
fn deserialise_defaults_missing_temperature() {
    let json = request_json_without_temperature();
    for parsed in parsed_temperatures(&json) {
        assert!(
            (parsed - DEFAULT_TEMPERATURE).abs() < f32::EPSILON,
            "an absent temperature must still default to {DEFAULT_TEMPERATURE}, got {parsed}"
        );
    }
}

// ============================================================================
// Acceptance criterion 5 — analysis still behaves for finite temperatures
// ============================================================================

/// A temperature that survives the boundary must still drive the scaling
/// functions exactly as before: identity at the default, and finite,
/// in-range behaviour either side of it.
#[test]
fn finite_temperatures_still_scale_analysis_thresholds() {
    let base_threshold = 0.05;
    let base_ratio = 0.6;
    let base_mh = 0.01;

    let identity = scale_threshold_by_temperature(base_threshold, DEFAULT_TEMPERATURE);
    assert!(
        (identity - base_threshold).abs() < f32::EPSILON,
        "default temperature must leave the threshold unchanged, got {identity}"
    );

    let hot = scale_threshold_by_temperature(base_threshold, 2.0);
    let cold = scale_threshold_by_temperature(base_threshold, 0.5);
    assert!(hot < base_threshold, "high temperature must lower threshold");
    assert!(cold > base_threshold, "low temperature must raise threshold");

    for temperature in [MIN_TEMPERATURE, 0.5, DEFAULT_TEMPERATURE, 2.0, MAX_TEMPERATURE] {
        assert!(
            scale_threshold_by_temperature(base_threshold, temperature).is_finite(),
            "threshold scaling must stay finite at {temperature}"
        );
        let ratio = scale_ratio_by_temperature(base_ratio, temperature);
        assert!(
            ratio.is_finite() && (0.0..=1.0).contains(&ratio),
            "ratio scaling must stay finite and in range at {temperature}, got {ratio}"
        );
        assert!(
            scale_mh_temperature(base_mh, temperature) > 0.0,
            "MH temperature must stay positive at {temperature}"
        );
    }
}

// ============================================================================
// Acceptance criteria 2 and 5 — the real FFI entry point
// ============================================================================

/// `analyze_parallel_internal` must refuse a non-finite temperature with the
/// structured failure envelope rather than analysing with Infinity.
#[test]
fn analyze_parallel_rejects_non_finite_temperature() {
    let input = request_json("1e39");
    let response = analyze_parallel_internal(&input).expect("FFI call must return a response");
    let parsed: serde_json::Value =
        serde_json::from_str(&response).expect("response must be valid JSON");

    assert_eq!(
        parsed["success"], false,
        "a non-finite temperature must not succeed: {response}"
    );
    let error = parsed["error"]
        .as_str()
        .expect("a failed response must carry an error message");
    assert!(
        error.contains("Issue #2136"),
        "the FFI error must cite the issue: {error}"
    );
    assert!(
        error.contains("temperature"),
        "the FFI error must name the offending field: {error}"
    );
}
