//! Issue #975 — Verify FFI boundary never panics on malformed input.
//!
//! The FFI boundary must return structured JSON error responses instead of
//! panicking when given malformed, truncated, or invalid input. Panics across
//! FFI boundaries are undefined behaviour in Rust.
//!
//! This test file complements `issue_574_ffi_json_fuzz_edge_cases.rs` by
//! ensuring full coverage of all internal entry points that accept external
//! JSON input, including `analyze_parallel_internal` and
//! `get_calibration_summary_internal` which were previously untested for
//! malformed input in the systematic malformed-input test.

// ============================================================================
// Helper
// ============================================================================

/// Verify that a JSON string from an internal function is well-formed and
/// contains `"success": false`.
fn assert_error_json(json: &str, context: &str) {
    let parsed: serde_json::Value = serde_json::from_str(json).unwrap_or_else(|e| {
        panic!("{context}: response should be valid JSON, got error: {e}, raw: {json}")
    });
    assert_eq!(
        parsed["success"], false,
        "{context}: invalid input should yield success=false, got: {json}"
    );
}

// ============================================================================
// analyze_parallel_internal — malformed JSON produces error responses
// ============================================================================

#[test]
fn analyze_parallel_internal_returns_error_json_for_malformed_input() {
    let malformed_inputs: Vec<&str> = vec![
        "",
        "null",
        "42",
        "{",
        "{}",
        r#"{"creature": null}"#,
        r#"{"parquetFile": 123}"#,
        r#"{"parquetFile": "/tmp/test.parquet"}"#,
        r#"not json at all"#,
    ];

    for input in &malformed_inputs {
        let result = neat_ai_discovery::analyze_parallel_internal(input);
        assert!(
            result.is_ok(),
            "analyze_parallel_internal should not return Err for input: {input}"
        );
        assert_error_json(
            &result.unwrap(),
            &format!("analyze_parallel_internal({input})"),
        );
    }
}

// ============================================================================
// get_calibration_summary_internal — malformed JSON produces error responses
// ============================================================================

#[test]
fn get_calibration_summary_internal_returns_error_json_for_malformed_input() {
    let malformed_inputs: Vec<&str> = vec![
        "",
        "null",
        "42",
        "{",
        "{}",
        r#"{"discoveryHistory": null}"#,
        r#"not json at all"#,
        r#"{"discoveryHistory": 12345}"#,
    ];

    for input in &malformed_inputs {
        let result = neat_ai_discovery::get_calibration_summary_internal(input);
        assert!(
            result.is_ok(),
            "get_calibration_summary_internal should not return Err for input: {input}"
        );
        assert_error_json(
            &result.unwrap(),
            &format!("get_calibration_summary_internal({input})"),
        );
    }
}

// ============================================================================
// All internal entry points — structured error response validation
// ============================================================================

/// Verify that every internal FFI entry point that accepts JSON input returns
/// a structured error response with `error` and `errorKind` fields when given
/// completely invalid JSON.
#[test]
fn all_internal_entry_points_return_structured_error_for_garbage_input() {
    let garbage = "this is not json {{{[[[";

    // record_discovery_internal
    let result = neat_ai_discovery::record_discovery_internal(garbage);
    assert!(result.is_ok());
    let json = result.unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed["success"], false);
    assert!(
        parsed["error"].is_string(),
        "record_discovery_internal should include an error message"
    );

    // merge_discovery_parquet_internal
    let result = neat_ai_discovery::merge_discovery_parquet_internal(garbage);
    assert!(result.is_ok());
    let json = result.unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed["success"], false);
    assert!(
        parsed["error"].is_string(),
        "merge_discovery_parquet_internal should include an error message"
    );

    // rank_focus_neurons_internal
    let result = neat_ai_discovery::rank_focus_neurons_internal(garbage);
    assert!(result.is_ok());
    let json = result.unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed["success"], false);
    assert!(
        parsed["error"].is_string(),
        "rank_focus_neurons_internal should include an error message"
    );

    // analyze_parallel_internal
    let result = neat_ai_discovery::analyze_parallel_internal(garbage);
    assert!(result.is_ok());
    let json = result.unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed["success"], false);
    assert!(
        parsed["error"].is_string(),
        "analyze_parallel_internal should include an error message"
    );

    // export_visualisation_snapshot_internal
    let result = neat_ai_discovery::export_visualisation_snapshot_internal(garbage);
    assert!(result.is_ok());
    let json = result.unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed["success"], false);
    assert!(
        parsed["error"].is_string(),
        "export_visualisation_snapshot_internal should include an error message"
    );

    // read_discovery_records
    let result = neat_ai_discovery::read_discovery_records(garbage);
    assert!(result.is_ok());
    let json = result.unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed["success"], false);
    assert!(
        parsed["error"].is_string(),
        "read_discovery_records should include an error message"
    );

    // get_calibration_summary_internal
    let result = neat_ai_discovery::get_calibration_summary_internal(garbage);
    assert!(result.is_ok());
    let json = result.unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed["success"], false);
    assert!(
        parsed["error"].is_string(),
        "get_calibration_summary_internal should include an error message"
    );
}

/// Verify that error responses include `errorKind` for diagnostic classification.
#[test]
fn error_responses_include_error_kind_field() {
    let garbage = "<<<not json>>>";

    let check_error_kind = |name: &str, json_str: &str| {
        let parsed: serde_json::Value = serde_json::from_str(json_str)
            .unwrap_or_else(|e| panic!("{name}: response should be valid JSON: {e}"));
        assert_eq!(parsed["success"], false, "{name}: should fail");
        // The field may be serialised as camelCase or snake_case depending on
        // the output type's serde configuration.
        let has_error_kind = parsed["errorKind"].is_string() || parsed["error_kind"].is_string();
        assert!(
            has_error_kind,
            "{name}: error response should include errorKind or error_kind field, got: {json_str}"
        );
    };

    let result = neat_ai_discovery::record_discovery_internal(garbage).unwrap();
    check_error_kind("record_discovery_internal", &result);

    let result = neat_ai_discovery::analyze_parallel_internal(garbage).unwrap();
    check_error_kind("analyze_parallel_internal", &result);

    let result = neat_ai_discovery::rank_focus_neurons_internal(garbage).unwrap();
    check_error_kind("rank_focus_neurons_internal", &result);

    let result = neat_ai_discovery::get_calibration_summary_internal(garbage).unwrap();
    check_error_kind("get_calibration_summary_internal", &result);

    let result = neat_ai_discovery::merge_discovery_parquet_internal(garbage).unwrap();
    check_error_kind("merge_discovery_parquet_internal", &result);

    let result = neat_ai_discovery::read_discovery_records(garbage).unwrap();
    check_error_kind("read_discovery_records", &result);

    let result = neat_ai_discovery::export_visualisation_snapshot_internal(garbage).unwrap();
    check_error_kind("export_visualisation_snapshot_internal", &result);
}
