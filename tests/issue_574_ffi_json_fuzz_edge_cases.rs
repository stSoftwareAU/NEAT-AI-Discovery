//! Issue #574 — FFI JSON boundary edge-case tests.
//!
//! These tests exercise the FFI JSON deserialisation and internal entry points
//! with malformed, truncated, and adversarial JSON inputs. They verify that no
//! combination of inputs causes a panic — all invalid inputs must produce a
//! well-formed JSON error response with `success: false`.
//!
//! These complement the cargo-fuzz targets in `fuzz/` by running deterministic
//! edge cases as part of the standard `cargo test` suite.

use neat_ai_discovery::{
    AnalyzeParallelInput, AppendRecordsInput, CancelSessionInput, CreatureJson,
    ExportVisualisationSnapshotInput, FinishSessionInput, MergeParquetInput, RankFocusNeuronsInput,
    ReadDiscoveryInput, RecordDiscoveryInput, StartSessionInput,
};

// ============================================================================
// Helper: verify error JSON response structure
// ============================================================================

/// Verify that a JSON string from an internal function is well-formed and
/// contains `"success": false`.
fn assert_error_json(json: &str) {
    let parsed: serde_json::Value =
        serde_json::from_str(json).expect("response should be valid JSON");
    assert_eq!(
        parsed["success"], false,
        "invalid input should yield success=false, got: {json}"
    );
}

// ============================================================================
// Deserialisation edge cases — malformed JSON
// ============================================================================

/// Inputs that are syntactically invalid JSON. Deserialisation must return Err
/// for all FFI input types.
#[test]
fn deserialise_malformed_json_never_panics() {
    let deeply_nested_braces = "{".repeat(100);
    let deeply_nested_brackets = "[".repeat(100);
    let long_string = "a".repeat(10_000);

    let malformed_inputs: Vec<&str> = vec![
        "",
        " ",
        "null",
        "true",
        "42",
        "\"hello\"",
        "[",
        "{",
        "}",
        "}{",
        "{{}",
        "{\"",
        "{\"a",
        "{\"a\"",
        "{\"a\":}",
        "{\"a\":\"b\",}",
        "[1,2,",
        r#"{"neurons": null}"#,
        r#"{"creature": null}"#,
        r#"{"creature": "not an object"}"#,
        r#"{"creature": {"neurons": "not an array"}}"#,
        // Deeply nested
        &deeply_nested_braces,
        &deeply_nested_brackets,
        // Very long string
        &long_string,
        // Null bytes
        "\0",
        "{\0}",
        "{\"a\":\"\0\"}",
        // Unicode edge cases
        "\u{FEFF}", // BOM
        "\u{200B}", // Zero-width space
    ];

    for input in &malformed_inputs {
        // Each of these must return Err, not panic.
        assert!(serde_json::from_str::<RecordDiscoveryInput>(input).is_err());
        assert!(serde_json::from_str::<StartSessionInput>(input).is_err());
        assert!(serde_json::from_str::<AppendRecordsInput>(input).is_err());
        assert!(serde_json::from_str::<FinishSessionInput>(input).is_err());
        assert!(serde_json::from_str::<CancelSessionInput>(input).is_err());
        assert!(serde_json::from_str::<MergeParquetInput>(input).is_err());
        assert!(serde_json::from_str::<RankFocusNeuronsInput>(input).is_err());
        assert!(serde_json::from_str::<AnalyzeParallelInput>(input).is_err());
        assert!(serde_json::from_str::<ReadDiscoveryInput>(input).is_err());
        assert!(serde_json::from_str::<ExportVisualisationSnapshotInput>(input).is_err());
    }
}

// ============================================================================
// Internal entry points — malformed JSON produces error responses
// ============================================================================

/// Internal FFI functions must return `Ok(json)` with `success: false` for
/// malformed inputs, never panic or return `Err`.
#[test]
fn internal_entry_points_return_error_json_for_malformed_input() {
    let long_string = "a".repeat(10_000);
    let malformed_inputs: Vec<&str> = vec![
        "",
        "null",
        "42",
        "{",
        "{}",
        r#"{"creature": null}"#,
        &long_string,
    ];

    for input in &malformed_inputs {
        // record_discovery_internal
        let result = neat_ai_discovery::record_discovery_internal(input);
        assert!(result.is_ok(), "record_discovery_internal should not Err");
        assert_error_json(&result.unwrap());

        // merge_discovery_parquet_internal
        let result = neat_ai_discovery::merge_discovery_parquet_internal(input);
        assert!(
            result.is_ok(),
            "merge_discovery_parquet_internal should not Err"
        );
        assert_error_json(&result.unwrap());

        // rank_focus_neurons_internal
        let result = neat_ai_discovery::rank_focus_neurons_internal(input);
        assert!(result.is_ok(), "rank_focus_neurons_internal should not Err");
        assert_error_json(&result.unwrap());

        // export_visualisation_snapshot_internal
        let result = neat_ai_discovery::export_visualisation_snapshot_internal(input);
        assert!(
            result.is_ok(),
            "export_visualisation_snapshot_internal should not Err"
        );
        assert_error_json(&result.unwrap());

        // read_discovery_records
        let result = neat_ai_discovery::read_discovery_records(input);
        assert!(result.is_ok(), "read_discovery_records should not Err");
        assert_error_json(&result.unwrap());
    }
}

// ============================================================================
// Valid-structure but extreme-value JSON
// ============================================================================

/// JSON that parses successfully but contains extreme values. The internal
/// functions should handle these gracefully (return error JSON, not panic).
#[test]
fn internal_entry_points_handle_extreme_values_gracefully() {
    // Empty arrays where data is expected
    let empty_creature = serde_json::json!({
        "creature": {
            "neurons": [],
            "synapses": [],
            "input": 0,
            "output": 0
        },
        "training_data": [],
        "temp_dir": "/nonexistent/path"
    })
    .to_string();

    let result = neat_ai_discovery::record_discovery_internal(&empty_creature);
    assert!(result.is_ok());
    // May succeed or fail, but must not panic

    // Enormous input/output counts
    let huge_counts = serde_json::json!({
        "creature": {
            "neurons": [],
            "synapses": [],
            "input": 999999999,
            "output": 999999999
        },
        "training_data": [],
        "temp_dir": "/nonexistent/path"
    })
    .to_string();

    let result = neat_ai_discovery::record_discovery_internal(&huge_counts);
    assert!(result.is_ok());

    // NaN and Infinity in numeric fields (serde_json rejects these by default,
    // but we test that the rejection is clean)
    let nan_json = r#"{"creature":{"neurons":[],"synapses":[],"input":1,"output":1},"training_data":[],"temp_dir":"/tmp","timeout_seconds":NaN}"#;
    let result = neat_ai_discovery::record_discovery_internal(nan_json);
    assert!(result.is_ok());
    assert_error_json(&result.unwrap());

    // Negative values where unsigned expected
    let negative = r#"{"creature":{"neurons":[],"synapses":[],"input":-1,"output":-1},"training_data":[],"temp_dir":"/tmp"}"#;
    let result = neat_ai_discovery::record_discovery_internal(negative);
    assert!(result.is_ok());
    assert_error_json(&result.unwrap());
}

/// Truncated JSON (simulates incomplete network transmission).
#[test]
fn truncated_json_never_panics() {
    let full_json = serde_json::json!({
        "creature": {
            "neurons": [
                {"uuid": "output-0", "type": "output", "squash": "IDENTITY", "bias": 0.0}
            ],
            "synapses": [
                {"from_uuid": "input-0", "to_uuid": "output-0", "weight": 0.5}
            ],
            "input": 1,
            "output": 1
        },
        "training_data": [
            {"input": [0.5], "output": [0.7]}
        ],
        "temp_dir": "/tmp/test"
    })
    .to_string();

    // Truncate at every byte position — none should panic
    for i in 0..full_json.len() {
        let truncated = &full_json[..i];
        // Must return Ok with error JSON, or Ok with success — never panic
        let _ = neat_ai_discovery::record_discovery_internal(truncated);
    }
}

/// JSON with extremely long string values.
#[test]
fn long_string_values_do_not_panic() {
    let long_uuid = "x".repeat(100_000);
    let json = serde_json::json!({
        "parquet_file": long_uuid,
        "neuron_uuid": long_uuid
    })
    .to_string();

    let result = neat_ai_discovery::read_discovery_records(&json);
    assert!(result.is_ok());
    // Will fail with file-not-found, but must not panic
}

/// JSON with extreme float values in neuron data.
#[test]
fn extreme_float_values_do_not_panic() {
    let json = serde_json::json!({
        "creature": {
            "neurons": [
                {"uuid": "output-0", "type": "output", "squash": "IDENTITY", "bias": 1e38}
            ],
            "synapses": [
                {"from_uuid": "input-0", "to_uuid": "output-0", "weight": -1e38}
            ],
            "input": 1,
            "output": 1
        },
        "training_data": [
            {
                "input": [1e38],
                "output": [-1e38],
                "neuron_data": [
                    {
                        "neuron_uuid": "output-0",
                        "activation": 1e38,
                        "value": -1e38,
                        "errors": [1e38, -1e38, 0.0]
                    }
                ]
            }
        ],
        "temp_dir": "/nonexistent/path"
    })
    .to_string();

    let result = neat_ai_discovery::record_discovery_internal(&json);
    assert!(result.is_ok());
}

/// Merge with empty and nonexistent file paths.
#[test]
fn merge_parquet_edge_cases_do_not_panic() {
    // Empty input files list
    let json = serde_json::json!({
        "outputFile": "/tmp/out.parquet",
        "inputFiles": []
    })
    .to_string();

    let result = neat_ai_discovery::merge_discovery_parquet_internal(&json);
    assert!(result.is_ok());
    assert_error_json(&result.unwrap());

    // Nonexistent files
    let json = serde_json::json!({
        "outputFile": "/nonexistent/out.parquet",
        "inputFiles": ["/nonexistent/a.parquet", "/nonexistent/b.parquet"]
    })
    .to_string();

    let result = neat_ai_discovery::merge_discovery_parquet_internal(&json);
    assert!(result.is_ok());
    // Should produce error JSON (file not found)
}

/// Rank focus neurons with nonexistent parquet file.
#[test]
fn rank_focus_neurons_nonexistent_file_does_not_panic() {
    let json = serde_json::json!({
        "parquetFile": "/nonexistent/records.parquet",
        "creature": {
            "neurons": [
                {"uuid": "output-0", "type": "output", "squash": "IDENTITY", "bias": 0.0}
            ],
            "synapses": [],
            "input": 1,
            "output": 1
        }
    })
    .to_string();

    let result = neat_ai_discovery::rank_focus_neurons_internal(&json);
    assert!(result.is_ok());
    assert_error_json(&result.unwrap());
}

/// Analyze parallel with nonexistent parquet file.
#[test]
fn analyze_parallel_nonexistent_file_does_not_panic() {
    let json = serde_json::json!({
        "parquetFile": "/nonexistent/records.parquet",
        "creature": {
            "neurons": [
                {"uuid": "output-0", "type": "output", "squash": "IDENTITY", "bias": 0.0}
            ],
            "synapses": [],
            "input": 1,
            "output": 1
        },
        "focusNeurons": ["output-0"]
    })
    .to_string();

    let result = neat_ai_discovery::analyze_parallel_internal(&json);
    assert!(result.is_ok());
    assert_error_json(&result.unwrap());
}

/// Export visualisation snapshot with nonexistent files.
#[test]
fn export_visualisation_nonexistent_file_does_not_panic() {
    let json = serde_json::json!({
        "parquetFile": "/nonexistent/records.parquet",
        "creature": {
            "neurons": [],
            "synapses": [],
            "input": 1,
            "output": 1
        },
        "outFile": "/nonexistent/snapshot.json"
    })
    .to_string();

    let result = neat_ai_discovery::export_visualisation_snapshot_internal(&json);
    assert!(result.is_ok());
    assert_error_json(&result.unwrap());
}

// ============================================================================
// Deserialisation with valid structure — type coercion edge cases
// ============================================================================

/// Verify that CreatureJson deserialises with default values for optional fields.
#[test]
fn creature_json_defaults_applied_correctly() {
    // Minimal neuron — squash defaults to IDENTITY, bias defaults to 0.0
    let json = r#"{"uuid":"n1","type":"hidden"}"#;
    let neuron: neat_ai_discovery::NeuronJson = serde_json::from_str(json).unwrap();
    assert_eq!(neuron.squash, "IDENTITY");
    assert_eq!(neuron.bias, 0.0);

    // Minimal synapse — all fields default to empty/0.0
    let json = r#"{}"#;
    let synapse: neat_ai_discovery::SynapseJson = serde_json::from_str(json).unwrap();
    assert_eq!(synapse.from_uuid, "");
    assert_eq!(synapse.to_uuid, "");
    assert_eq!(synapse.weight, 0.0);
    assert!(synapse.synapse_type.is_none());
}

/// Verify that CreatureJson round-trips via serde correctly.
#[test]
fn creature_json_round_trip() {
    let json = serde_json::json!({
        "neurons": [
            {"uuid": "output-0", "type": "output", "squash": "LOGISTIC", "bias": 0.5}
        ],
        "synapses": [
            {"from_uuid": "input-0", "to_uuid": "output-0", "weight": 0.3}
        ],
        "input": 2,
        "output": 1
    });

    let creature: CreatureJson = serde_json::from_value(json).unwrap();
    assert_eq!(creature.neurons.len(), 1);
    assert_eq!(creature.synapses.len(), 1);
    assert_eq!(creature.input, 2);
    assert_eq!(creature.output, 1);

    // Serialise back and check key fields
    let reserialized = serde_json::to_value(&creature).unwrap();
    assert_eq!(reserialized["neurons"][0]["uuid"], "output-0");
    // f32 precision: 0.3 round-trips as 0.30000001192092896
    let weight = reserialized["synapses"][0]["weight"].as_f64().unwrap();
    assert!((weight - 0.3).abs() < 1e-6, "weight should be ~0.3");
}
