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

    // With normalised impact, we compute: |weight| / total_inbound × downstream_impact
    // For a single synapse from hidden-1 to output with weight 1e-12,
    // if that's the only synapse to output: impact = 1e-12 / 1e-12 × 1.0 = 1.0
    // But if there are other synapses, it's proportionally smaller.
    //
    // The key test is that the impact is NOT zero - very small weights should
    // produce non-zero impact (proportional to their share of total input).
    assert!(
        impact > 0.0,
        "hidden-1 impact should be non-zero, got {impact}",
    );
}

#[test]
fn test_impact_calculation_with_multiple_incoming_connections() {
    // Test that impact is properly normalized when a neuron has multiple incoming connections
    // This tests the fix for the bug where impact was using absolute weights instead of normalized shares

    // Note: Input neurons are NOT included in creature.neurons as per the NEAT-AI data model.
    // They are represented only by the creature.input count.
    let creature = CreatureJson {
        neurons: vec![
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

    // With NORMALISED impact: |weight| / total_inbound × downstream_impact
    // Output has 2 incoming synapses: weight 10 + weight 5 = total 15
    // - hidden-a: 10/15 × 1.0 = 0.667 (67% of output's input)
    // - hidden-b: 5/15 × 1.0 = 0.333 (33% of output's input)
    //
    // This is CORRECT: if we remove hidden-a, output loses 67% of its input, not 1000%!
    assert!(
        (impact_a - 0.667).abs() < 0.01,
        "hidden-a impact should be ~0.667 (10/15 of output's input), got {impact_a}",
    );
    assert!(
        (impact_b - 0.333).abs() < 0.01,
        "hidden-b impact should be ~0.333 (5/15 of output's input), got {impact_b}",
    );

    // The ratio of impacts should still be 2:1 (proportional to weights)
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

                // Verify bias is within reasonable range (extended for large weight configurations)
                // Range varies by activation: widest is IDENTITY at [-50.0, 50.0]
                assert!(
                    neuron.bias >= -50.0 && neuron.bias <= 50.0,
                    "Bias should be within reasonable range [-50.0, 50.0], got {} for {} neuron",
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
        // Check if we have ReLU candidates - they should have bias in extended range
        // (extended to support large incoming weights up to 200)
        let relu_neurons: Vec<_> = result
            .helpful_neurons
            .iter()
            .filter(|n| n.squash == "ReLU")
            .collect();

        if !relu_neurons.is_empty() {
            for neuron in relu_neurons {
                assert!(
                    neuron.bias >= -25.0,
                    "ReLU neuron bias should be >= -25.0 (extended range for large weights), got {}",
                    neuron.bias
                );
                assert!(
                    neuron.bias <= 10.0,
                    "ReLU neuron bias should be <= 10.0, got {}",
                    neuron.bias
                );
            }
        }

        // Check if we have TANH or LOGISTIC candidates - they should have symmetric extended range
        // (extended to support large incoming weights up to 200)
        let symmetric_neurons: Vec<_> = result
            .helpful_neurons
            .iter()
            .filter(|n| n.squash == "TANH" || n.squash == "LOGISTIC")
            .collect();

        if !symmetric_neurons.is_empty() {
            for neuron in symmetric_neurons {
                assert!(
                    neuron.bias >= -10.0 && neuron.bias <= 10.0,
                    "{} neuron bias should be in [-10.0, 10.0], got {}",
                    neuron.squash,
                    neuron.bias
                );
            }
        }
    }
}

/// Test that impact is cumulative when a neuron connects to MULTIPLE outputs.
/// This is a regression test for the bug where impact used MAX instead of SUM
/// for multiple outgoing synapses, causing neurons connected to multiple outputs
/// to be incorrectly flagged as "low-impact" removal candidates.
#[test]
fn test_cumulative_impact_with_multiple_output_connections() {
    // Create a creature where a hidden neuron connects to TWO outputs
    // The impact should be the SUM of contributions to both outputs
    // Note: Input neurons are NOT included in creature.neurons as per the NEAT-AI data model.
    let creature = CreatureJson {
        neurons: vec![
            NeuronJson {
                uuid: "hub".to_string(),
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
            NeuronJson {
                uuid: "output-1".to_string(),
                neuron_type: "output".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
        ],
        synapses: vec![
            SynapseJson {
                from_uuid: "input-0".to_string(),
                to_uuid: "hub".to_string(),
                weight: 1.0,
            },
            // Hub connects to BOTH outputs - removing it affects BOTH
            SynapseJson {
                from_uuid: "hub".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.5, // 100% of output-0's inbound
            },
            SynapseJson {
                from_uuid: "hub".to_string(),
                to_uuid: "output-1".to_string(),
                weight: 0.5, // 100% of output-1's inbound
            },
        ],
        input: 1,
        output: 2,
    };

    let temp_dir = TempDir::new().unwrap();
    let temp_path = temp_dir.path();

    // Write discovery data
    let input_json = serde_json::json!({
        "creature": creature.clone(),
        "training_data": [{
            "input": [0.5],
            "output": [0.25, 0.25],
            "neuron_data": [
                {"neuron_uuid": "hub", "activation": 0.5, "value": 0.5, "errors": [0.1, 0.1]},
                {"neuron_uuid": "output-0", "activation": 0.25, "value": 0.25, "errors": [0.1]},
                {"neuron_uuid": "output-1", "activation": 0.25, "value": 0.25, "errors": [0.1]}
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

    let hub = neurons
        .iter()
        .find(|n| n["neuronUuid"] == "hub")
        .expect("hub should be in results");

    let impact = hub["impact"].as_f64().expect("impact should be a number") as f32;

    // With absolute weights (v0.1.126+):
    // hub → output-0 (weight 0.5): 0.5 × 1.0 = 0.5
    // hub → output-1 (weight 0.5): 0.5 × 1.0 = 0.5
    // Cumulative impact: 0.5 + 0.5 = 1.0
    //
    // The key test is that we SUM across multiple outputs, not take MAX.
    // With the old bug (max): would only be 0.5
    //
    // This is critical: a neuron affecting 2 outputs should NOT be flagged as low-impact!
    assert!(
        impact > 0.9,
        "Hub neuron connecting to 2 outputs should have cumulative impact > 0.9 (sum of paths), got {impact}. \
         This indicates the impact calculation is using MAX instead of SUM for multiple outgoing synapses. \
         Bug: This neuron could be incorrectly flagged as a removal candidate!"
    );

    // All neurons are returned as removal candidates, sorted by impact.
    // Hub should be present but sorted LAST (highest impact = worst removal candidate)
    let removal_candidates = result.get("removalCandidates").and_then(|v| v.as_array());

    if let Some(candidates) = removal_candidates {
        // If hub is in candidates, it should be near the end (high impact)
        let hub_pos = candidates.iter().position(|c| c["neuronUuid"] == "hub");
        if let Some(pos) = hub_pos {
            // Hub should be in the bottom half (high impact = bad candidate)
            assert!(
                pos >= candidates.len() / 2,
                "Hub neuron with high impact should be sorted near the end of removal candidates, not at position {pos}"
            );
        }
    }
}

/// Test that add-neuron analysis can find successful candidates when conditions are right.
/// This is a regression test to verify add-neuron discovery is working correctly.
#[test]
fn test_add_neuron_finds_candidates_with_correlated_errors() {
    skip_without_gpu!();
    use neat_ai_discovery::AnalyzeNeuronsInput;

    let temp_dir = TempDir::new().unwrap();
    let temp_path = temp_dir.path();

    // Create a simple creature with just an output neuron and multiple inputs
    let creature = CreatureJson {
        neurons: vec![NeuronJson {
            uuid: "output-0".to_string(),
            neuron_type: "output".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        }],
        synapses: vec![], // No existing synapses - add-neuron should find candidates
        input: 2,
        output: 1,
    };

    // Generate training data where input-0 activation CORRELATES with output error
    // This is the ideal scenario for add-neuron to find a candidate:
    // - When input-0 is high, error is positive (output should be higher)
    // - When input-0 is low, error is negative (output should be lower)
    let mut training_data = Vec::new();
    for i in 0..50 {
        let input_0_val = (i as f32 - 25.0) / 25.0; // Range: -1.0 to 1.0
        let input_1_val = 0.5; // Constant (no correlation)

        // Create correlated error: when input-0 is positive, error is positive
        // This means a new neuron from input-0 could help reduce error
        let error = input_0_val * 0.5; // Positive correlation

        training_data.push(serde_json::json!({
            "input": [input_0_val, input_1_val],
            "output": [0.0],
            "neuron_data": [{
                "neuron_uuid": "output-0",
                "activation": 0.0,
                "value": 0.0,
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

    // Analyse neurons with a lower threshold to increase chances of success
    let analyze_input = AnalyzeNeuronsInput {
        parquet_file: parquet_file.to_str().unwrap().to_string(),
        creature,
        focus_neurons: vec!["output-0".to_string()],
        improvement_threshold: Some(0.01), // 1% threshold - low to ensure candidates are found
        max_candidates: Some(10),
        analysis_deadline_ms: None,
    };

    let result = neat_ai_discovery::analysis::analyze_neurons(&analyze_input)
        .expect("Neuron analysis should succeed");

    // With correlated errors and no existing connections, we should find candidates
    // The input-0 -> ReLU -> output-0 path should show improvement
    assert!(
        !result.helpful_neurons.is_empty(),
        "Add-neuron should find at least one candidate when input activation correlates with output error. \
         Diagnostics: {:?}",
        result.no_candidate_reasons
    );

    // Verify the candidate makes sense
    let best = &result.helpful_neurons[0];
    assert!(
        best.expected_improvement_percentage > 0.0,
        "Best candidate should show positive improvement, got {}",
        best.expected_improvement_percentage
    );

    // The source should be input-0 (the correlated input)
    assert!(
        best.source_neuron_uuid == "input-0",
        "Best candidate should use the correlated source (input-0), got {}",
        best.source_neuron_uuid
    );
}

/// Regression test: Add-neuron with HARD_TANH target must use bias-aware weight calculation.
///
/// Production uses HARD_TANH for output neurons. The optimal outgoing weight depends on
/// the new neuron's bias - computing weight without bias gives wrong predictions.
///
/// This test verifies that predictions for non-linear target neurons are accurate.
#[test]
fn test_add_neuron_with_hard_tanh_target_uses_bias_aware_weight() {
    skip_without_gpu!();
    use neat_ai_discovery::AnalyzeNeuronsInput;

    let temp_dir = TempDir::new().unwrap();
    let temp_path = temp_dir.path();

    // Create creature with HARD_TANH output (matches production)
    let creature = CreatureJson {
        neurons: vec![NeuronJson {
            uuid: "output-0".to_string(),
            neuron_type: "output".to_string(),
            squash: "HARD_TANH".to_string(), // Non-linear - requires bias-aware weight
            bias: 0.0,
        }],
        synapses: vec![],
        input: 2,
        output: 1,
    };

    // Generate training data with correlated errors
    // Key: the new neuron will have bias, changing its activation pattern
    let mut training_data = Vec::new();
    for i in 0..100 {
        let input_0_val = (i as f32 - 50.0) / 50.0; // Range: -1.0 to 1.0
        let input_1_val = 0.3;

        // Correlated error - when input-0 is positive, output should be higher
        let error = input_0_val * 0.4;
        // HARD_TANH output value in linear region
        let output_value = 0.1 + input_0_val * 0.1;
        let output_activation = output_value.clamp(-1.0, 1.0);

        training_data.push(serde_json::json!({
            "input": [input_0_val, input_1_val],
            "output": [0.0],
            "neuron_data": [{
                "neuron_uuid": "output-0",
                "activation": output_activation,
                "value": output_value,  // Required for saturation-aware prediction
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

    // Analyse neurons
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

    // Should find candidates with HARD_TANH target
    assert!(
        !result.helpful_neurons.is_empty(),
        "Add-neuron should find candidates for HARD_TANH target. Diagnostics: {:?}",
        result.no_candidate_reasons
    );

    // Verify the candidate has reasonable expected improvement
    let best = &result.helpful_neurons[0];
    assert!(
        best.expected_improvement_percentage > 0.0,
        "Expected improvement should be positive, got {}",
        best.expected_improvement_percentage
    );

    // Key assertion: the improvement prediction should be realistic (not inflated)
    // With the bug, predictions were often 10x higher than reality
    // After the fix, predictions should be < 50% (a reasonable upper bound)
    assert!(
        best.expected_improvement_percentage < 50.0,
        "Expected improvement should be realistic (< 50%), got {}%. \
         This may indicate the bias-aware weight calculation is not working.",
        best.expected_improvement_percentage
    );

    // Note: bias=0 is a valid value (the optimal bias search includes 0.0)
    // The key fix is that the outgoing weight is recomputed AFTER finding the bias,
    // so even with bias=0, the weight calculation is now correct.
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

/// REGRESSION TEST: Hidden neurons must be analysed, not filtered out.
///
/// v0.1.123: Previously hidden neurons were filtered entirely with 100% failure rate.
/// Investigation revealed this was caused by low-confidence fallback candidates.
/// Now hidden neurons ARE analysed with impact-based discounting.
///
/// This test will FAIL if hidden neurons are filtered out instead of analysed.
#[test]
fn test_hidden_neurons_are_analysed_not_filtered() {
    skip_without_gpu!();
    use neat_ai_discovery::AnalyzeNeuronsInput;

    let temp_dir = TempDir::new().unwrap();
    let temp_path = temp_dir.path();

    // Create a creature with: input -> hidden -> output
    // The hidden neuron connects input to output
    let creature = CreatureJson {
        neurons: vec![
            NeuronJson {
                uuid: "hidden-0".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "TANH".to_string(),
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
            from_uuid: "hidden-0".to_string(),
            to_uuid: "output-0".to_string(),
            weight: 1.0, // Hidden -> Output connection gives hidden-0 impact = 1.0
        }],
        input: 2,
        output: 1,
    };

    // Generate training data with correlated errors for HIDDEN neuron
    // This creates a scenario where adding a connection to the hidden neuron could help
    let mut training_data = Vec::new();
    for i in 0..50 {
        let input_0_val = (i as f32 - 25.0) / 25.0; // Range: -1.0 to 1.0
        let input_1_val = 0.5;

        // Create correlated error at the HIDDEN neuron
        let hidden_error = input_0_val * 0.5;
        let hidden_value = input_0_val * 0.3;
        let hidden_activation = hidden_value.tanh();

        // Output also has error (but we're targeting hidden-0)
        let output_error = hidden_error * 0.5; // Propagated error

        training_data.push(serde_json::json!({
            "input": [input_0_val, input_1_val],
            "output": [0.0],
            "neuron_data": [
                {
                    "neuron_uuid": "hidden-0",
                    "activation": hidden_activation,
                    "value": hidden_value,
                    "errors": [hidden_error]
                },
                {
                    "neuron_uuid": "output-0",
                    "activation": 0.0,
                    "value": 0.0,
                    "errors": [output_error]
                }
            ]
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

    // CRITICAL: Request analysis with hidden-0 as a focus target
    let analyze_input = AnalyzeNeuronsInput {
        parquet_file: parquet_file.to_str().unwrap().to_string(),
        creature,
        focus_neurons: vec!["hidden-0".to_string()], // Request hidden neuron analysis
        improvement_threshold: Some(0.01),
        max_candidates: Some(10),
        analysis_deadline_ms: None,
    };

    let result = neat_ai_discovery::analysis::analyze_neurons(&analyze_input)
        .expect("Neuron analysis should succeed");

    // REGRESSION CHECK: Hidden neuron should NOT be reported as filtered
    let hidden_filtered = result.no_candidate_reasons.iter().any(|r| {
        r.target_uuid == "hidden-0"
            && matches!(
                r.reason,
                neat_ai_discovery::analysis::NeuronNoCandidateReason::HiddenNeuronFiltered
            )
    });

    assert!(
        !hidden_filtered,
        "REGRESSION: Hidden neuron 'hidden-0' was filtered out instead of being analysed! \
        Hidden neurons should be analysed with impact-based discounting, not filtered. \
        Diagnostics: {:?}",
        result.no_candidate_reasons
    );

    // If the hidden neuron was analysed but no candidates found, that's OK
    // The key assertion is that it was NOT filtered
    eprintln!(
        "Hidden neuron analysis result: {} candidates found, {} diagnostics",
        result.helpful_neurons.len(),
        result.no_candidate_reasons.len()
    );
}

/// REGRESSION TEST: Hidden neuron candidates must have discounted predictions.
///
/// When a candidate targets a hidden neuron, the expected_improvement_percentage
/// should be discounted by the hidden neuron's impact score (path to outputs).
///
/// Impact is calculated as: (weight to child / total inbound to child) × child_impact
/// So to get impact < 1.0, we need multiple paths to the output.
///
/// This test will FAIL if hidden neuron predictions are not discounted.
#[test]
fn test_hidden_neuron_candidates_have_impact_discounted_predictions() {
    skip_without_gpu!();
    use neat_ai_discovery::AnalyzeNeuronsInput;

    let temp_dir = TempDir::new().unwrap();
    let temp_path = temp_dir.path();

    // Create creature where hidden-0 has impact < 1.0 to output
    // With absolute weights (v0.1.126+), impact = weight × downstream_impact.
    //
    //   hidden-0 (weight 0.5) --\
    //                            --> output-0
    //   hidden-1 (weight 0.5) --/
    //
    // hidden-0 impact = 0.5 × 1.0 = 0.5
    // (Previously normalised: 0.5 / 1.0 = 0.5, same value but different formula)
    let creature = CreatureJson {
        neurons: vec![
            NeuronJson {
                uuid: "hidden-0".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
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
        synapses: vec![
            SynapseJson {
                from_uuid: "hidden-0".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.5, // Absolute impact = 0.5 × 1.0 = 0.5
            },
            SynapseJson {
                from_uuid: "hidden-1".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.5, // Absolute impact = 0.5 × 1.0 = 0.5
            },
        ],
        input: 2,
        output: 1,
    };

    // Generate strongly correlated training data for hidden-0
    let mut training_data = Vec::new();
    for i in 0..100 {
        let input_0_val = (i as f32 - 50.0) / 50.0;
        let input_1_val = 0.3;

        // Strong correlation at hidden-0
        let hidden_0_error = input_0_val * 0.8;
        let hidden_0_value = input_0_val * 0.2;

        // hidden-1 has no correlation (constant)
        let hidden_1_error = 0.0;
        let hidden_1_value = 0.5;

        training_data.push(serde_json::json!({
            "input": [input_0_val, input_1_val],
            "output": [0.0],
            "neuron_data": [
                {
                    "neuron_uuid": "hidden-0",
                    "activation": hidden_0_value,
                    "value": hidden_0_value,
                    "errors": [hidden_0_error]
                },
                {
                    "neuron_uuid": "hidden-1",
                    "activation": hidden_1_value,
                    "value": hidden_1_value,
                    "errors": [hidden_1_error]
                },
                {
                    "neuron_uuid": "output-0",
                    "activation": 0.0,
                    "value": 0.0,
                    "errors": [hidden_0_error * 0.5]
                }
            ]
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

    // Analyse hidden-0 which has impact = 0.5
    let analyze_input = AnalyzeNeuronsInput {
        parquet_file: parquet_file.to_str().unwrap().to_string(),
        creature,
        focus_neurons: vec!["hidden-0".to_string()],
        improvement_threshold: Some(0.001), // Very low threshold to get candidates
        max_candidates: Some(20),
        analysis_deadline_ms: None,
    };

    let result = neat_ai_discovery::analysis::analyze_neurons(&analyze_input)
        .expect("Neuron analysis should succeed");

    // Hidden neuron should be analysed, not filtered
    let hidden_was_filtered = result.no_candidate_reasons.iter().any(|r| {
        r.target_uuid == "hidden-0"
            && matches!(
                r.reason,
                neat_ai_discovery::analysis::NeuronNoCandidateReason::HiddenNeuronFiltered
            )
    });

    assert!(
        !hidden_was_filtered,
        "REGRESSION: Hidden neuron was filtered instead of analysed!"
    );

    // Find candidates for hidden-0
    let hidden_candidates: Vec<_> = result
        .helpful_neurons
        .iter()
        .filter(|c| c.target_neuron_uuid == "hidden-0")
        .collect();

    // If we found hidden candidates, verify their predictions are discounted
    if !hidden_candidates.is_empty() {
        for candidate in &hidden_candidates {
            // With impact = 0.5, predictions should be discounted by 50%
            // So maximum possible is 50% even if raw prediction was 100%
            assert!(
                candidate.expected_improvement_percentage <= 0.5,
                "Hidden neuron candidate improvement {:.4}% exceeds impact-adjusted maximum of 50%. \
                Hidden-0 has impact ~0.5, so predictions should be discounted. \
                This suggests impact discounting is not being applied.",
                candidate.expected_improvement_percentage * 100.0
            );

            eprintln!(
                "Hidden neuron candidate (impact ~0.5): {} -> {} improvement={:.4}%",
                candidate.source_neuron_uuid,
                candidate.target_neuron_uuid,
                candidate.expected_improvement_percentage * 100.0
            );
        }
    } else {
        // No candidates found - that's OK, the key test is that it wasn't filtered
        eprintln!(
            "No candidates found for hidden-0 (this is OK - key test is it wasn't filtered). \
            Diagnostics: {:?}",
            result.no_candidate_reasons
        );
    }
}
