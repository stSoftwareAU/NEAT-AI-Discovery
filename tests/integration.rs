//! Integration tests for discovery recording

mod common;

use neat_ai_discovery::record_discovery;
use tempfile::TempDir;

#[test]
fn test_record_discovery_integration() {
    let temp_dir = TempDir::new().unwrap();
    let temp_path = temp_dir.path().to_str().unwrap();

    let input = format!(
        r#"{{
            "creature": {{
                "neurons": [
                    {{
                        "uuid": "hidden-1",
                        "type": "hidden",
                        "squash": "TANH",
                        "bias": 0.0
                    }},
                    {{
                        "uuid": "output-0",
                        "type": "output",
                        "squash": "IDENTITY",
                        "bias": 0.0
                    }}
                ],
                "synapses": [],
                "input": 2,
                "output": 1
            }},
            "training_data": [
                {{"input": [0.1, 0.2], "output": [0.5]}},
                {{"input": [0.3, 0.4], "output": [0.6]}}
            ],
            "temp_dir": "{temp_path}"
        }}"#
    );

    let result = record_discovery(&input).unwrap();
    let output: serde_json::Value = serde_json::from_str(&result).unwrap();

    assert_eq!(output["success"], true);
    assert!(output["temp_dir"].as_str().is_some());
    assert_eq!(output["file"], "discovery_data.parquet");

    // Verify Parquet file was created
    let parquet_file = std::path::Path::new(output["temp_dir"].as_str().unwrap())
        .join(output["file"].as_str().unwrap());
    assert!(parquet_file.exists());
}

#[test]
fn test_record_discovery_invalid_json() {
    // Invalid JSON should return JSON error response, not a Rust error
    let result = record_discovery("invalid json");
    assert!(
        result.is_ok(),
        "record_discovery should return Ok(String) even on invalid JSON"
    );

    let output_str = result.unwrap();
    let output: serde_json::Value =
        serde_json::from_str(&output_str).expect("Output should be valid JSON");

    assert_eq!(output["success"], false);
    assert!(output["error"].is_string());
    assert!(
        output["error"]
            .as_str()
            .unwrap()
            .contains("Failed to parse input JSON"),
        "Error should mention JSON parsing failure"
    );
}

#[test]
fn test_record_discovery_missing_fields() {
    // Missing fields should return JSON error response, not a Rust error
    let input = r#"{"creature": {}}"#;
    let result = record_discovery(input);
    assert!(
        result.is_ok(),
        "record_discovery should return Ok(String) even on missing fields"
    );

    let output_str = result.unwrap();
    let output: serde_json::Value =
        serde_json::from_str(&output_str).expect("Output should be valid JSON");

    assert_eq!(output["success"], false);
    assert!(output["error"].is_string());
}

#[test]
fn test_record_discovery_returns_json_error_on_failure() {
    // Test that when record_discovery_data fails, it returns JSON with success=false
    // instead of a Rust error
    let temp_dir = TempDir::new().unwrap();
    let temp_path = temp_dir.path().to_str().unwrap();

    // Create input with only input neurons (which will cause record_discovery_data to fail)
    let input = format!(
        r#"{{
            "creature": {{
                "neurons": [
                    {{
                        "uuid": "input-0",
                        "type": "input",
                        "squash": "IDENTITY",
                        "bias": 0.0
                    }},
                    {{
                        "uuid": "input-1",
                        "type": "input",
                        "squash": "IDENTITY",
                        "bias": 0.0
                    }}
                ],
                "synapses": [],
                "input": 2,
                "output": 1
            }},
            "training_data": [
                {{"input": [0.1, 0.2], "output": [0.5]}}
            ],
            "temp_dir": "{temp_path}"
        }}"#
    );

    // Should return JSON string, not a Rust error
    let result = record_discovery(&input);
    assert!(
        result.is_ok(),
        "record_discovery should return Ok(String) even on failure"
    );

    let output_str = result.unwrap();
    let output: serde_json::Value =
        serde_json::from_str(&output_str).expect("Output should be valid JSON");

    // Should have success=false and an error message
    assert_eq!(output["success"], false);
    assert!(output["error"].is_string());
    assert!(
        output["error"]
            .as_str()
            .unwrap()
            .contains("non-input neurons"),
        "Error message should explain the issue"
    );
}
