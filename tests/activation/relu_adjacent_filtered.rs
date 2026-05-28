//! Regression test ensuring ReLU-adjacent squashes are not suggested as add-neuron candidates.
//!
//! Rationale (26-Dec-2025, Issue #148):
//! - NEAT-AI-Discovery uses value-domain errors and correlation-style evaluation rather than
//!   derivative backpropagation.
//! - In a time-bounded discovery run, proposing ReLU-adjacent squashes (eg Swish/SELU/LeakyReLU)
//!   reduces the budget available to scan more (source,target) structural possibilities.
//! - Existing creatures can still *use* any squash; this is only about what we propose as NEW
//!   add-neuron candidates.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use crate::skip_without_gpu;
use neat_ai_discovery::record_discovery_internal;
use neat_ai_discovery::{AnalyzeNeuronsInput, CreatureJson, NeuronJson};
use tempfile::TempDir;

#[test]
fn add_neuron_candidates_do_not_include_relu_adjacent_squashes() {
    skip_without_gpu!();

    let temp_dir = TempDir::new().unwrap();
    let temp_path = temp_dir.path();

    // Simple creature: single output neuron.
    let creature = CreatureJson {
        neurons: vec![NeuronJson {
            uuid: "output-0".to_string(),
            neuron_type: "output".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        }],
        synapses: vec![],
        input: 2,
        output: 1,
    };

    // Training data with a strong positive correlation between input-0 and output error.
    //
    // We keep input-0 strictly positive so ReLU is always active. This makes many smooth
    // squashes (including the ReLU-adjacent ones) look beneficial unless filtered.
    let mut training_data = Vec::new();
    for i in 0..128 {
        let input_0_val = (i as f32 + 1.0) / 128.0; // (0, 1]
        let input_1_val = 0.5;
        let error = input_0_val * 0.5;
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

    let input_json = serde_json::json!({
        "creature": creature,
        "training_data": training_data,
        "temp_dir": temp_path.to_str().unwrap()
    });

    let record_input = serde_json::to_string(&input_json).unwrap();
    let record_output_json = record_discovery_internal(&record_input).unwrap();
    let record_output: serde_json::Value = serde_json::from_str(&record_output_json).unwrap();
    assert_eq!(record_output["success"], true);

    let parquet_file = temp_path.join(record_output["file"].as_str().unwrap());

    let analyze_input = AnalyzeNeuronsInput {
        parquet_file: parquet_file.to_str().unwrap().to_string(),
        creature,
        focus_neurons: vec!["output-0".to_string()],
        max_candidates: Some(512),
        analysis_deadline_ms: None,
        random_seed: None,
        temperature: 1.0,
        failure_cache: None,
        discovery_outcome_log: None,
        task_descriptor: None,
    };

    let result = neat_ai_discovery::analysis::analyze_neurons(&analyze_input)
        .expect("Analysis should succeed");

    assert!(
        !result.helpful_neurons.is_empty(),
        "Expected at least one add-neuron candidate for the correlated-error scenario"
    );

    let forbidden = ["LeakyReLU", "Swish", "SELU"];
    let found: Vec<&str> = result
        .helpful_neurons
        .iter()
        .map(|c| c.squash.as_str())
        .filter(|s| forbidden.contains(s))
        .collect();

    assert!(
        found.is_empty(),
        "ReLU-adjacent squashes should not be suggested as add-neuron candidates. Found: {found:?}"
    );
}
