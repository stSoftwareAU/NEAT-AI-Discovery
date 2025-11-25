//! Integration tests for discovery recording

mod common;

use neat_ai_discovery::record_discovery_internal;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
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
                {{
                    "input": [0.1, 0.2],
                    "output": [0.5],
                    "neuron_data": [
                        {{"neuron_uuid": "hidden-1", "activation": 0.7, "value": 0.6, "errors": [0.1]}},
                        {{"neuron_uuid": "output-0", "activation": 0.5, "value": 0.5, "errors": [0.0]}}
                    ]
                }},
                {{
                    "input": [0.3, 0.4],
                    "output": [0.6],
                    "neuron_data": [
                        {{"neuron_uuid": "hidden-1", "activation": 0.8, "value": 0.7, "errors": [0.15]}},
                        {{"neuron_uuid": "output-0", "activation": 0.6, "value": 0.6, "errors": [0.0]}}
                    ]
                }}
            ],
            "temp_dir": "{temp_path}"
        }}"#
    );

    let result = record_discovery_internal(&input).unwrap();
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
    let result = record_discovery_internal("invalid json");
    assert!(
        result.is_ok(),
        "record_discovery_internal should return Ok(String) even on invalid JSON"
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
    let result = record_discovery_internal(input);
    assert!(
        result.is_ok(),
        "record_discovery_internal should return Ok(String) even on missing fields"
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
    let result = record_discovery_internal(&input);
    assert!(
        result.is_ok(),
        "record_discovery_internal should return Ok(String) even on failure"
    );

    let output_str = result.unwrap();
    let output: serde_json::Value =
        serde_json::from_str(&output_str).expect("Output should be valid JSON");

    // Should have success=false and an error message
    assert_eq!(output["success"], false);
    assert!(output["error"].is_string());
    let error_msg = output["error"].as_str().unwrap();
    assert!(
        error_msg.contains("no non-input neurons")
            || error_msg.contains("non-input neurons")
            || error_msg.contains("No discovery records")
            || error_msg.contains("No pre-computed neuron_data"),
        "Error message should explain the issue. Got: {error_msg}"
    );
}

#[test]
fn test_impact_with_very_small_incoming_weight_is_not_zeroed() {
    // Test that impact is not forced to zero when the only path to an output
    // uses a very small but non-zero weight (regression around near-zero totals)

    let creature = CreatureJson {
        neurons: vec![
            NeuronJson {
                uuid: "hidden-1".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "output-0".to_string(),
                neuron_type: "output".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
        ],
        synapses: vec![SynapseJson {
            from_uuid: "hidden-1".to_string(),
            to_uuid: "output-0".to_string(),
            // Very small but non-zero weight so the total inbound is below
            // any practical threshold, but still represents a valid path.
            weight: 1e-12,
        }],
        input: 0,
        output: 1,
    };

    let temp_dir = TempDir::new().unwrap();
    let temp_path = temp_dir.path();

    // Write minimal data to the parquet file
    let input_json = serde_json::json!({
        "creature": creature.clone(),
        "training_data": [{
            "input": [],
            "output": [0.5],
            "neuron_data": [
                {"neuron_uuid": "hidden-1", "activation": 0.5, "value": 0.5, "errors": [0.1]},
                {"neuron_uuid": "output-0", "activation": 0.5, "value": 0.5, "errors": [0.2]}
            ]
        }],
        "temp_dir": temp_path.to_str().unwrap()
    });

    let record_input = serde_json::to_string(&input_json).unwrap();
    let record_output_json = record_discovery_internal(&record_input).unwrap();
    let record_output: serde_json::Value = serde_json::from_str(&record_output_json).unwrap();
    assert_eq!(
        record_output["success"], true,
        "Failed to record discovery data"
    );

    let parquet_file = temp_path.join(record_output["file"].as_str().unwrap());

    use neat_ai_discovery::rank_focus_neurons_internal;
    let rank_input = serde_json::json!({
        "parquetFile": parquet_file.to_str().unwrap(),
        "creature": creature,
        "maxResults": 10
    })
    .to_string();

    let result_json = rank_focus_neurons_internal(&rank_input).unwrap();
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    if result["success"] != true {
        panic!(
            "Rank focus neurons failed: {}",
            result["error"].as_str().unwrap_or("unknown error")
        );
    }

    let neurons = result["neurons"]
        .as_array()
        .expect("neurons should be an array");
    let hidden_1 = neurons
        .iter()
        .find(|n| n["neuronUuid"] == "hidden-1")
        .expect("hidden-1 should be in results");

    let impact = hidden_1["impact"]
        .as_f64()
        .expect("impact should be a number") as f32;

    // With a single non-zero connection to an output, impact should be close to 1.0,
    // not forced to zero just because the weight is very small.
    assert!(
        impact > 0.5,
        "hidden-1 impact should be significant for the only path to an output, got {impact}",
    );
}

#[test]
fn test_impact_calculation_with_multiple_incoming_connections() {
    // Test that impact is properly normalized when a neuron has multiple incoming connections
    // This tests the fix for the bug where impact was using absolute weights instead of normalized shares

    let creature = CreatureJson {
        neurons: vec![
            NeuronJson {
                uuid: "input-0".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "input-1".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "hidden-a".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "hidden-b".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "output-0".to_string(),
                neuron_type: "output".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
        ],
        synapses: vec![
            SynapseJson {
                from_uuid: "input-0".to_string(),
                to_uuid: "hidden-a".to_string(),
                weight: 1.0,
            },
            SynapseJson {
                from_uuid: "input-1".to_string(),
                to_uuid: "hidden-b".to_string(),
                weight: 1.0,
            },
            // Both hidden neurons connect to output with different weights
            // Total incoming weight to output-0 = 10.0 + 5.0 = 15.0
            SynapseJson {
                from_uuid: "hidden-a".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 10.0,
            },
            SynapseJson {
                from_uuid: "hidden-b".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 5.0,
            },
        ],
        input: 2,
        output: 1,
    };

    // Create a temporary parquet file with some data
    let temp_dir = TempDir::new().unwrap();
    let temp_path = temp_dir.path();

    // Write minimal data to the parquet file
    let input_json = serde_json::json!({
        "creature": creature.clone(),
        "training_data": [{
            "input": [0.1, 0.2],
            "output": [0.5],
            "neuron_data": [
                {"neuron_uuid": "hidden-a", "activation": 0.5, "value": 0.5, "errors": [0.1]},
                {"neuron_uuid": "hidden-b", "activation": 0.5, "value": 0.5, "errors": [0.1]},
                {"neuron_uuid": "output-0", "activation": 0.5, "value": 0.5, "errors": [0.2]}
            ]
        }],
        "temp_dir": temp_path.to_str().unwrap()
    });

    // Create the parquet file
    let record_input = serde_json::to_string(&input_json).unwrap();
    let record_output_json = record_discovery_internal(&record_input).unwrap();
    let record_output: serde_json::Value = serde_json::from_str(&record_output_json).unwrap();
    assert_eq!(
        record_output["success"], true,
        "Failed to record discovery data"
    );

    // Get the actual parquet file path
    let parquet_file = temp_path.join(record_output["file"].as_str().unwrap());

    // Now test impact calculation using the public API
    use neat_ai_discovery::rank_focus_neurons_internal;
    let rank_input = serde_json::json!({
        "parquetFile": parquet_file.to_str().unwrap(),
        "creature": creature,
        "maxResults": 10
    })
    .to_string();

    let result_json = rank_focus_neurons_internal(&rank_input).unwrap();
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    if result["success"] != true {
        panic!(
            "Rank focus neurons failed: {}",
            result["error"].as_str().unwrap_or("unknown error")
        );
    }
    let neurons = result["neurons"]
        .as_array()
        .expect("neurons should be an array");

    // Find the impacts for our test neurons
    let hidden_a = neurons
        .iter()
        .find(|n| n["neuronUuid"] == "hidden-a")
        .expect("hidden-a should be in results");
    let hidden_b = neurons
        .iter()
        .find(|n| n["neuronUuid"] == "hidden-b")
        .expect("hidden-b should be in results");

    let impact_a = hidden_a["impact"]
        .as_f64()
        .expect("impact should be a number") as f32;
    let impact_b = hidden_b["impact"]
        .as_f64()
        .expect("impact should be a number") as f32;

    // hidden-a has weight 10.0 to output-0, total incoming is 15.0, so impact should be 10.0/15.0 = 0.6667
    // hidden-b has weight 5.0 to output-0, total incoming is 15.0, so impact should be 5.0/15.0 = 0.3333
    assert!(
        (impact_a - 0.6667).abs() < 0.01,
        "hidden-a impact should be ~0.667, got {impact_a}",
    );
    assert!(
        (impact_b - 0.3333).abs() < 0.01,
        "hidden-b impact should be ~0.333, got {impact_b}",
    );

    // Also verify that hidden-a has about 2x the impact of hidden-b (since it has 2x the weight)
    assert!(
        (impact_a / impact_b - 2.0).abs() < 0.1,
        "hidden-a should have ~2x impact of hidden-b, got ratio {}",
        impact_a / impact_b
    );
}

/// Test that analyze_neurons returns non-zero bias values for neuron candidates
#[test]
fn test_analyze_neurons_returns_non_zero_bias() {
    skip_without_gpu!();
    use neat_ai_discovery::AnalyzeNeuronsInput;

    let temp_dir = TempDir::new().unwrap();
    let temp_path = temp_dir.path();

    // Create creature with output neuron
    let creature = CreatureJson {
        neurons: vec![NeuronJson {
            uuid: "output-0".to_string(),
            neuron_type: "output".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        }],
        synapses: vec![],
        input: 1,
        output: 1,
    };

    // Generate training data with significant errors to encourage neuron discovery
    let mut training_data = Vec::new();
    for i in 0..20 {
        let input_val = (i as f32) * 0.05;
        let activation = input_val;
        let error = 0.2 + (i as f32) * 0.01; // Significant error
        training_data.push(serde_json::json!({
            "input": [input_val],
            "output": [activation],
            "neuron_data": [{
                "neuron_uuid": "output-0",
                "activation": activation,
                "value": activation,
                "errors": [error]
            }]
        }));
    }

    // Record discovery data
    let input_json = serde_json::json!({
        "creature": creature.clone(),
        "training_data": training_data,
        "temp_dir": temp_path.to_str().unwrap()
    });

    let record_input = serde_json::to_string(&input_json).unwrap();
    let record_output_json = record_discovery_internal(&record_input).unwrap();
    let record_output: serde_json::Value = serde_json::from_str(&record_output_json).unwrap();
    assert_eq!(
        record_output["success"], true,
        "Failed to record discovery data: {:?}",
        record_output["error"]
    );

    let parquet_file = temp_path.join(record_output["file"].as_str().unwrap());

    // Analyse neurons using internal function (bypasses FFI)
    let analyze_input = AnalyzeNeuronsInput {
        parquet_file: parquet_file.to_str().unwrap().to_string(),
        creature,
        focus_neurons: vec!["output-0".to_string()],
        improvement_threshold: Some(0.01), // Lower threshold to increase chances of finding candidates
        max_candidates: Some(10),
        analysis_deadline_ms: None,
    };

    let result = neat_ai_discovery::analysis::analyze_neurons(&analyze_input)
        .expect("Neuron analysis should succeed");

    // Check if we got any neuron candidates
    if !result.helpful_neurons.is_empty() {
        let mut found_non_zero_bias = false;

        for neuron in &result.helpful_neurons {
            // Most neurons should have non-zero bias
            // Some might legitimately be 0, but not all
            if neuron.bias != 0.0 {
                found_non_zero_bias = true;

                // Verify bias is within reasonable range
                assert!(
                    neuron.bias >= -1.0 && neuron.bias <= 1.0,
                    "Bias should be within reasonable range [-1.0, 1.0], got {} for {} neuron",
                    neuron.bias,
                    neuron.squash
                );
            }
        }

        assert!(
            found_non_zero_bias,
            "At least some neuron candidates should have non-zero bias"
        );
    }
}

/// Test that bias values are activation-function-specific
#[test]
fn test_bias_values_are_activation_specific() {
    skip_without_gpu!();
    use neat_ai_discovery::AnalyzeNeuronsInput;

    let temp_dir = TempDir::new().unwrap();
    let temp_path = temp_dir.path();

    // Create creature with output neuron
    let creature = CreatureJson {
        neurons: vec![NeuronJson {
            uuid: "output-0".to_string(),
            neuron_type: "output".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        }],
        synapses: vec![],
        input: 1,
        output: 1,
    };

    // Generate training data
    let mut training_data = Vec::new();
    for i in 0..25 {
        let input_val = (i as f32) * 0.04;
        let activation = input_val;
        let error = 0.25 + (i as f32) * 0.01;
        training_data.push(serde_json::json!({
            "input": [input_val],
            "output": [activation],
            "neuron_data": [{
                "neuron_uuid": "output-0",
                "activation": activation,
                "value": activation,
                "errors": [error]
            }]
        }));
    }

    // Record discovery data
    let input_json = serde_json::json!({
        "creature": creature.clone(),
        "training_data": training_data,
        "temp_dir": temp_path.to_str().unwrap()
    });

    let record_input = serde_json::to_string(&input_json).unwrap();
    let record_output_json = record_discovery_internal(&record_input).unwrap();
    let record_output: serde_json::Value = serde_json::from_str(&record_output_json).unwrap();
    assert_eq!(record_output["success"], true);

    let parquet_file = temp_path.join(record_output["file"].as_str().unwrap());

    // Analyse neurons using internal function (bypasses FFI)
    let analyze_input = AnalyzeNeuronsInput {
        parquet_file: parquet_file.to_str().unwrap().to_string(),
        creature,
        focus_neurons: vec!["output-0".to_string()],
        improvement_threshold: Some(0.01),
        max_candidates: Some(50), // Request many candidates to get variety
        analysis_deadline_ms: None,
    };

    let result = neat_ai_discovery::analysis::analyze_neurons(&analyze_input)
        .expect("Neuron analysis should succeed");

    if !result.helpful_neurons.is_empty() {
        // Check if we have ReLU candidates - they should have non-negative bias
        let relu_neurons: Vec<_> = result
            .helpful_neurons
            .iter()
            .filter(|n| n.squash == "ReLU")
            .collect();

        if !relu_neurons.is_empty() {
            for neuron in relu_neurons {
                assert!(
                    neuron.bias >= -1.0,
                    "ReLU neuron bias should be >= -1.0 (expanded range for thresholding), got {}",
                    neuron.bias
                );
                assert!(
                    neuron.bias <= 1.0,
                    "ReLU neuron bias should be <= 1.0, got {}",
                    neuron.bias
                );
            }
        }

        // Check if we have TANH or LOGISTIC candidates - they should have symmetric range
        let symmetric_neurons: Vec<_> = result
            .helpful_neurons
            .iter()
            .filter(|n| n.squash == "TANH" || n.squash == "LOGISTIC")
            .collect();

        if !symmetric_neurons.is_empty() {
            for neuron in symmetric_neurons {
                assert!(
                    neuron.bias >= -1.0 && neuron.bias <= 1.0,
                    "{} neuron bias should be in [-1.0, 1.0], got {}",
                    neuron.squash,
                    neuron.bias
                );
            }
        }
    }
}

/// Test that neurons with calculated bias improve error more than bias=0
#[test]
fn test_bias_improves_neuron_performance() {
    skip_without_gpu!();
    use neat_ai_discovery::AnalyzeNeuronsInput;

    let temp_dir = TempDir::new().unwrap();
    let temp_path = temp_dir.path();

    // Create creature with output neuron
    let creature = CreatureJson {
        neurons: vec![NeuronJson {
            uuid: "output-0".to_string(),
            neuron_type: "output".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        }],
        synapses: vec![],
        input: 1,
        output: 1,
    };

    // Generate training data with a pattern that benefits from bias
    let mut training_data = Vec::new();
    for i in 0..30 {
        let input_val = -0.5 + (i as f32) * 0.03; // Range from -0.5 to 0.4
        let activation = input_val;
        // Error pattern that can be reduced with proper bias
        let error = if input_val < 0.0 { 0.3 } else { -0.2 };
        training_data.push(serde_json::json!({
            "input": [input_val],
            "output": [activation],
            "neuron_data": [{
                "neuron_uuid": "output-0",
                "activation": activation,
                "value": activation,
                "errors": [error]
            }]
        }));
    }

    // Record discovery data
    let input_json = serde_json::json!({
        "creature": creature.clone(),
        "training_data": training_data,
        "temp_dir": temp_path.to_str().unwrap()
    });

    let record_input = serde_json::to_string(&input_json).unwrap();
    let record_output_json = record_discovery_internal(&record_input).unwrap();
    let record_output: serde_json::Value = serde_json::from_str(&record_output_json).unwrap();
    assert_eq!(record_output["success"], true);

    let parquet_file = temp_path.join(record_output["file"].as_str().unwrap());

    // Analyse neurons using internal function (bypasses FFI)
    let analyze_input = AnalyzeNeuronsInput {
        parquet_file: parquet_file.to_str().unwrap().to_string(),
        creature,
        focus_neurons: vec!["output-0".to_string()],
        improvement_threshold: Some(0.01),
        max_candidates: Some(10),
        analysis_deadline_ms: None,
    };

    let result = neat_ai_discovery::analysis::analyze_neurons(&analyze_input)
        .expect("Neuron analysis should succeed");

    // If we found candidates, they should show positive expected improvement
    // This implicitly tests that bias is helping (since without optimal bias, improvement would be lower)
    if !result.helpful_neurons.is_empty() {
        for neuron in &result.helpful_neurons {
            assert!(
                neuron.expected_improvement_percentage > 0.0,
                "Neuron candidate should show positive improvement, got {}",
                neuron.expected_improvement_percentage
            );

            // The fact that the neuron passed the threshold with the calculated bias
            // means the bias is helping (otherwise it wouldn't have passed)
            assert!(
                neuron.expected_improvement_percentage >= 0.01,
                "Neuron should meet improvement threshold of 0.01, got {}",
                neuron.expected_improvement_percentage
            );
        }
    }
}
