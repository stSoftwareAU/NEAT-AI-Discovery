//! Issue #1444: Fail-fast analysis when the record phase times out with
//! insufficient Parquet coverage.
//!
//! When the recording phase times out, the selected focus neurons can have zero
//! rows in the Parquet file. Synapse/neuron analysis then correctly returns
//! nothing — but only after spending the full analysis budget. These tests
//! verify the fail-fast gate: when every selected focus neuron has zero Parquet
//! rows, analysis is skipped and `insufficient_recording` is surfaced as the
//! dominant rejection reason in the performance-summary metadata.
//!
//! Reuses the partial-record fixture pattern from
//! `issue_1101_no_target_records_diagnostic.rs`.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for test data generation

use crate::common::{neuron, output, synapse};
use neat_ai_discovery::CreatureJson;
use neat_ai_discovery::analysis::GpuAnalyzer;
use neat_ai_discovery::analyze_parallel_internal;
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;

/// Skip test if no GPU available (the gate runs after the GPU availability
/// check, so the pipeline must reach it).
macro_rules! skip_without_gpu {
    () => {
        if !GpuAnalyzer::gpu_is_available() {
            eprintln!("Skipping test: no GPU available");
            return;
        }
    };
}

/// Build a creature with `inputs` input neurons feeding a single TANH output.
fn creature_with_inputs(inputs: usize) -> CreatureJson {
    let mut neurons = Vec::new();
    let mut synapses = Vec::new();
    for i in 0..inputs {
        neurons.push(neuron(&format!("input-{i}"), "input", "IDENTITY"));
        synapses.push(synapse(
            &format!("input-{i}"),
            "output-0",
            0.3 + i as f32 * 0.1,
        ));
    }
    neurons.push(output("output-0", "TANH"));
    CreatureJson {
        neurons,
        synapses,
        input: inputs,
        output: 1,
    }
}

/// Write records only for the input neurons, deliberately omitting `output-0`,
/// simulating a record-phase timeout where the focus neuron has zero rows.
fn write_inputs_only_parquet(path: &str, inputs: usize, rows: u32) {
    let mut records = Vec::new();
    for obs_index in 0..rows {
        for i in 0..inputs {
            records.push(DiscoverRecord::new(
                obs_index,
                format!("input-{i}"),
                Some(0.1 + obs_index as f32 * 0.01),
                0.1 + obs_index as f32 * 0.01,
                vec![],
            ));
        }
    }
    write_records_to_parquet(path, &records).expect("write parquet");
}

#[test]
fn partial_record_phase_fails_fast_with_insufficient_recording() {
    skip_without_gpu!();

    let creature = creature_with_inputs(2);
    let temp_file = tempfile::NamedTempFile::new().expect("temp parquet");
    let file_path = temp_file.path().to_str().expect("temp path").to_string();

    // output-0 (the only focus neuron) has zero rows.
    write_inputs_only_parquet(&file_path, 2, 30);

    let input_json = serde_json::json!({
        "parquetFile": file_path,
        "creature": creature,
        "focusNeurons": ["output-0"],
        "maxSynapseCandidates": 10,
        "maxNeuronCandidates": 10,
        "randomSeed": 42
    })
    .to_string();

    let output_json = analyze_parallel_internal(&input_json).expect("analysis should return JSON");
    let parsed: serde_json::Value =
        serde_json::from_str(&output_json).expect("output should be valid JSON");

    assert_eq!(
        parsed["success"],
        true,
        "Analysis should succeed (skip path is not an error): {}",
        parsed.get("error").unwrap_or(&serde_json::Value::Null)
    );

    // No candidates were produced.
    assert_eq!(
        parsed["helpfulSynapses"].as_array().map_or(0, Vec::len),
        0,
        "no helpful synapses expected on a skipped pass"
    );
    assert_eq!(
        parsed["helpfulNeurons"].as_array().map_or(0, Vec::len),
        0,
        "no helpful neurons expected on a skipped pass"
    );

    // Synapse metadata names insufficient_recording as the dominant rejection.
    let syn_meta = &parsed["synapseMetadata"];
    assert_eq!(
        syn_meta["rejectionBreakdown"]["insufficient_recording"], 1,
        "synapse rejection breakdown should record one insufficient_recording, got {syn_meta}"
    );
    let syn_summary = syn_meta["topLevelSummary"]
        .as_str()
        .expect("top-level summary present");
    assert!(
        syn_summary.contains("insufficient Parquet recording"),
        "summary should name the dominant reason, got: {syn_summary}"
    );

    // Structured diagnostic carries the coverage counts.
    let diag = &syn_meta["insufficientRecording"];
    assert_eq!(diag["focusNeuronsTotal"], 1, "diag: {diag}");
    assert_eq!(diag["focusNeuronsWithZeroRows"], 1, "diag: {diag}");
    assert_eq!(
        diag["focusNeuronRecordsTotal"], 0,
        "focus neuron had zero rows: {diag}"
    );

    // Neuron metadata carries the same diagnostic.
    let neu_meta = &parsed["neuronMetadata"];
    assert_eq!(
        neu_meta["rejectionBreakdown"]["insufficient_recording"], 1,
        "neuron rejection breakdown should record insufficient_recording, got {neu_meta}"
    );
    assert_eq!(
        neu_meta["insufficientRecording"]["focusNeuronsWithZeroRows"], 1,
        "neuron diagnostic present: {neu_meta}"
    );
}

#[test]
fn full_recording_does_not_trigger_fail_fast_gate() {
    skip_without_gpu!();

    let creature = creature_with_inputs(2);
    let temp_file = tempfile::NamedTempFile::new().expect("temp parquet");
    let file_path = temp_file.path().to_str().expect("temp path").to_string();

    // Write records for BOTH inputs and the output focus neuron.
    let mut records = Vec::new();
    for obs_index in 0..30u32 {
        records.push(DiscoverRecord::new(
            obs_index,
            "input-0".to_string(),
            Some(0.1 + obs_index as f32 * 0.01),
            0.1 + obs_index as f32 * 0.01,
            vec![],
        ));
        records.push(DiscoverRecord::new(
            obs_index,
            "input-1".to_string(),
            Some(0.2 + obs_index as f32 * 0.005),
            0.2 + obs_index as f32 * 0.005,
            vec![],
        ));
        records.push(DiscoverRecord::new(
            obs_index,
            "output-0".to_string(),
            Some(0.05 * obs_index as f32),
            0.05 * obs_index as f32,
            vec![0.3],
        ));
    }
    write_records_to_parquet(&file_path, &records).expect("write parquet");

    let input_json = serde_json::json!({
        "parquetFile": file_path,
        "creature": creature,
        "focusNeurons": ["output-0"],
        "maxSynapseCandidates": 10,
        "maxNeuronCandidates": 10,
        "randomSeed": 42
    })
    .to_string();

    let output_json = analyze_parallel_internal(&input_json).expect("analysis should return JSON");
    let parsed: serde_json::Value =
        serde_json::from_str(&output_json).expect("output should be valid JSON");

    assert_eq!(
        parsed["success"],
        true,
        "Analysis should succeed: {}",
        parsed.get("error").unwrap_or(&serde_json::Value::Null)
    );

    // The focus neuron has records, so the gate must NOT fire — no diagnostic
    // and no insufficient_recording rejection.
    let syn_meta = &parsed["synapseMetadata"];
    assert!(
        syn_meta["insufficientRecording"].is_null(),
        "gate must not fire when focus neurons have records: {syn_meta}"
    );
    assert!(
        syn_meta["rejectionBreakdown"]
            .get("insufficient_recording")
            .is_none(),
        "no insufficient_recording rejection expected: {syn_meta}"
    );
}
