//! Regression tests for candidate synapse index fields.
//!
//! The JSON API includes `fromNeuronIndex` / `toNeuronIndex` on `CandidateSynapseJson` to
//! aid debugging and analysis (eg candidate clustering near the end of the evaluation order).
//!
//! This test ensures those fields are populated consistently with `CandidateNeuronJson`.

#![allow(clippy::cast_possible_truncation)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use crate::skip_without_gpu;
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{AnalyzeSynapsesInput, CreatureJson, NeuronJson, SynapseJson};
use tempfile::NamedTempFile;

#[test]
fn synapse_candidates_populate_from_and_to_indices() {
    skip_without_gpu!();

    // Simple creature:
    // - inputs: input-0, input-1
    // - output: output-0
    // - existing synapse input-0 → output-0
    // Eligible synapse candidate should include input-1 → output-0.
    let creature = CreatureJson {
        neurons: vec![NeuronJson {
            uuid: "output-0".to_string(),
            neuron_type: "output".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        }],
        synapses: vec![SynapseJson {
            from_uuid: "input-0".to_string(),
            to_uuid: "output-0".to_string(),
            weight: 0.4,
            synapse_type: None,
        }],
        input: 2,
        output: 1,
    };

    // Records designed so input-1 correlates positively with output error:
    // - When error is positive (output should be higher), input-1 is high.
    // - When error is negative (output should be lower), input-1 is low.
    // Issue #789: Use enough positive-error samples so the improved ratio exceeds
    // the raised MIN_IMPROVED_RATIO threshold of 0.6.
    let mut records = Vec::new();
    let patterns: &[(f32, f32)] = &[
        (1.0, 1.0),
        (1.0, 0.9),
        (-1.0, 0.0),
        (-1.0, 0.1),
        (0.8, 0.95),
        (0.6, 0.85),
        (-0.7, 0.05),
        (-0.9, 0.0),
        (0.5, 0.7),
        (0.4, 0.65),
        (0.9, 0.92),
        (0.7, 0.88),
    ];

    for (obs_index, (error, input_1_activation)) in patterns.iter().copied().enumerate() {
        let obs_index = obs_index as u32;

        // input-0 exists but is uninformative for this test.
        records.push(DiscoverRecord::new(
            obs_index,
            "input-0".to_string(),
            Some(0.0),
            0.0,
            Vec::new(),
        ));

        records.push(DiscoverRecord::new(
            obs_index,
            "input-1".to_string(),
            Some(input_1_activation),
            input_1_activation,
            Vec::new(),
        ));

        // output-0: constant activation, varying error.
        records.push(DiscoverRecord::new(
            obs_index,
            "output-0".to_string(),
            Some(0.5),
            0.5,
            vec![error],
        ));
    }

    let temp_file = NamedTempFile::new().expect("Failed to create temp parquet file");
    let file_path = temp_file
        .path()
        .to_str()
        .expect("temp path should be valid UTF-8")
        .to_string();

    write_records_to_parquet(&file_path, &records).expect("Failed to write parquet test data");

    let input = AnalyzeSynapsesInput {
        parquet_file: file_path,
        creature,
        focus_neurons: vec!["output-0".to_string()],
        max_candidates: Some(10),
        analysis_deadline_ms: Some(30000),
        random_seed: None,
        temperature: 1.0,
    };

    let result = neat_ai_discovery::analysis::analyze_synapses(&input)
        .expect("Synapse analysis should succeed");

    let candidate = result
        .helpful_synapses
        .iter()
        .find(|c| c.from_neuron_uuid == "input-1" && c.to_neuron_uuid == "output-0")
        .expect("Expected to find candidate input-1 → output-0");

    // Index mapping is: input-0 = 0, input-1 = 1, output-0 = 2.
    assert_eq!(candidate.from_neuron_index, Some(1));
    assert_eq!(candidate.to_neuron_index, Some(2));
}
