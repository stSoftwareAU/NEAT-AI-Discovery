//! Regression test ensuring LeakyReLU is not suggested as an add-neuron candidate.
//!
//! Rationale:
//! - In our current discovery workflow LeakyReLU offers little practical advantage over ReLU,
//!   but it generates noisy/low-quality candidates in production runs.
//! - We still support LeakyReLU for existing creatures (targets/sources may use it), we just
//!   avoid proposing NEW LeakyReLU neurons.

mod common;

use neat_ai_discovery::record_discovery_internal;
use neat_ai_discovery::{AnalyzeNeuronsInput, CreatureJson, NeuronJson};
use tempfile::TempDir;

#[test]
fn add_neuron_candidates_do_not_include_leaky_relu() {
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

    // Training data with strong correlation between input-0 and output error.
    // This should yield multiple viable add-neuron candidates across the activation set.
    let mut training_data = Vec::new();
    for i in 0..64 {
        let input_0_val = (i as f32 - 32.0) / 32.0; // [-1, 1]
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
        "creature": creature.clone(),
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
        improvement_threshold: Some(0.0),
        max_candidates: Some(64),
        analysis_deadline_ms: None,
    };

    let result = neat_ai_discovery::analysis::analyze_neurons(&analyze_input)
        .expect("Analysis should succeed");

    assert!(
        !result.helpful_neurons.is_empty(),
        "Expected at least one add-neuron candidate for the correlated-error scenario"
    );

    assert!(
        result
            .helpful_neurons
            .iter()
            .all(|c| c.squash != "LeakyReLU"),
        "LeakyReLU should not be suggested as an add-neuron activation (existing creatures still supported)"
    );
}
