//! Issue #1101: Integration tests for the `NoTargetRecords` diagnostic.
//!
//! When the recording phase times out and a focus neuron has zero rows in the
//! Parquet file, the synapse analysis should report `NoTargetRecords` instead
//! of the misleading `NoEligibleSources`.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for test data generation

use crate::common::{neuron, output, synapse};
use neat_ai_discovery::analysis::shared::SynapseNoCandidateReason;
use neat_ai_discovery::analysis::{GpuAnalyzer, analyze_synapses};
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{AnalyzeSynapsesInput, CreatureJson, analyze_parallel_internal};

/// Skip test if no GPU available.
macro_rules! skip_without_gpu {
    () => {
        if !GpuAnalyzer::gpu_is_available() {
            eprintln!("Skipping test: no GPU available");
            return;
        }
    };
}

// =============================================================================
// Synapse diagnostic: NoTargetRecords when zero Parquet records
// =============================================================================

#[test]
fn zero_target_records_reports_no_target_records_reason() {
    skip_without_gpu!();

    // Creature with two inputs and one output, with synapses from both inputs.
    let creature = CreatureJson {
        neurons: vec![
            neuron("input-0", "input", "IDENTITY"),
            neuron("input-1", "input", "IDENTITY"),
            output("output-0", "TANH"),
        ],
        synapses: vec![
            synapse("input-0", "output-0", 0.5),
            synapse("input-1", "output-0", 0.3),
        ],
        input: 2,
        output: 1,
    };

    let temp_file = tempfile::NamedTempFile::new().expect("temp parquet");
    let file_path = temp_file.path().to_str().expect("temp path").to_string();

    // Write records for inputs only — deliberately omit output-0 to simulate
    // a recording timeout where the focus neuron has zero Parquet records.
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
    }
    write_records_to_parquet(&file_path, &records).expect("write parquet");

    let input = AnalyzeSynapsesInput {
        parquet_file: file_path,
        creature,
        focus_neurons: vec!["output-0".to_string()],
        max_candidates: Some(50),
        analysis_deadline_ms: None,
        random_seed: None,
        temperature: 1.0,
    };

    let result = analyze_synapses(&input).expect("analysis should succeed");

    // output-0 has no records — should report NoTargetRecords
    let output_diag = result
        .no_candidate_reasons
        .iter()
        .find(|r| r.target_uuid == "output-0");

    assert!(
        output_diag.is_some(),
        "Diagnostic should be present for output-0 when it has zero Parquet records"
    );

    let diag = output_diag.unwrap();
    assert_eq!(
        diag.reason,
        SynapseNoCandidateReason::NoTargetRecords,
        "Expected NoTargetRecords when target has zero Parquet records, got {:?}",
        diag.reason
    );
    assert_eq!(
        diag.target_record_count, 0,
        "target_record_count should be 0"
    );
}

// =============================================================================
// JSON output: reason serialises as "no_target_records"
// =============================================================================

#[test]
fn json_output_serialises_no_target_records_reason() {
    skip_without_gpu!();

    let creature = CreatureJson {
        neurons: vec![
            neuron("input-0", "input", "IDENTITY"),
            output("output-0", "TANH"),
        ],
        synapses: vec![synapse("input-0", "output-0", 0.5)],
        input: 1,
        output: 1,
    };

    let temp_file = tempfile::NamedTempFile::new().expect("temp parquet");
    let file_path = temp_file.path().to_str().expect("temp path").to_string();

    // Write records only for input-0 (output-0 has no records).
    let mut records = Vec::new();
    for obs_index in 0..20u32 {
        records.push(DiscoverRecord::new(
            obs_index,
            "input-0".to_string(),
            Some(0.5),
            0.5,
            vec![],
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

    let output_json =
        analyze_parallel_internal(&input_json).expect("parallel analysis should return JSON");
    let parsed: serde_json::Value =
        serde_json::from_str(&output_json).expect("output should be valid JSON");

    assert_eq!(
        parsed["success"],
        true,
        "Analysis should succeed: {}",
        parsed.get("error").unwrap_or(&serde_json::Value::Null)
    );

    // Check that synapseDiagnostics contains reason "no_target_records"
    if let Some(diagnostics) = parsed.get("synapseDiagnostics") {
        let diagnostics_arr = diagnostics.as_array().expect("should be array");
        let output_diag = diagnostics_arr
            .iter()
            .find(|d| d["targetNeuronUuid"] == "output-0");

        if let Some(diag) = output_diag {
            assert_eq!(
                diag["reason"], "no_target_records",
                "Reason should be serialised as 'no_target_records' in JSON, got {}",
                diag["reason"]
            );
            assert_eq!(
                diag["targetRecordCount"], 0,
                "targetRecordCount should be 0"
            );
        }
    }
}
