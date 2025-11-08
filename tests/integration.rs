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
    let result = record_discovery("invalid json");
    assert!(result.is_err());
}

#[test]
fn test_record_discovery_missing_fields() {
    let input = r#"{"creature": {}}"#;
    let result = record_discovery(input);
    assert!(result.is_err());
}
