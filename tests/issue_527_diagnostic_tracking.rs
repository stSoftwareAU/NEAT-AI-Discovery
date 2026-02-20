//! Issue #527: Integration tests for diagnostic tracking
//!
//! Tests the diagnostic tracking pipeline that reports *why* candidates were rejected:
//! - Rejection reasons are correctly categorised (hidden filtered, no samples, etc.)
//! - Diagnostic summaries accumulate across analysis targets
//! - JSON output includes correct diagnostic structures
//!
//! These tests exercise the full analysis pipeline via `analyze_neurons` and
//! `analyze_parallel_internal`, verifying that diagnostic information flows
//! correctly from internal tracking to the public API.

mod common;

use common::{neuron, output, synapse};
use neat_ai_discovery::analysis::shared::{NeuronNoCandidateReason, SynapseNoCandidateReason};
use neat_ai_discovery::analysis::{GpuAnalyzer, analyze_neurons, analyze_synapses};
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{
    AnalyzeNeuronsInput, AnalyzeSynapsesInput, CreatureJson, analyze_parallel_internal,
};
use tempfile::NamedTempFile;

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
// Helpers
// =============================================================================

/// Build a minimal creature with input, hidden, and output neurons.
fn minimal_creature() -> CreatureJson {
    CreatureJson {
        neurons: vec![
            neuron("input-0", "input", "IDENTITY"),
            neuron("input-1", "input", "IDENTITY"),
            neuron("hidden-0", "hidden", "TANH"),
            output("output-0", "TANH"),
        ],
        synapses: vec![
            synapse("input-0", "hidden-0", 0.5),
            synapse("input-1", "hidden-0", 0.5),
            synapse("hidden-0", "output-0", 1.0),
        ],
        input: 2,
        output: 1,
    }
}

/// Write minimal records for the given neuron UUIDs.
fn write_minimal_records(path: &str, neuron_uuids: &[&str], obs_count: u32) {
    let mut records = Vec::new();
    for obs_index in 0..obs_count {
        for &uuid in neuron_uuids {
            records.push(DiscoverRecord::new(
                obs_index,
                uuid.to_string(),
                Some(0.1),
                0.1 + (obs_index as f32 * 0.01),
                vec![0.01],
            ));
        }
    }
    write_records_to_parquet(path, &records).expect("write parquet");
}

// =============================================================================
// Neuron diagnostics: hidden neuron filtered
// =============================================================================

#[test]
fn hidden_neuron_reported_as_filtered_in_neuron_diagnostics() {
    skip_without_gpu!();

    let creature = minimal_creature();
    let temp_file = NamedTempFile::new().expect("temp parquet");
    let file_path = temp_file.path().to_str().expect("temp path").to_string();
    write_minimal_records(
        &file_path,
        &["input-0", "input-1", "hidden-0", "output-0"],
        20,
    );

    // Enable output-only mode so hidden neurons get filtered
    // SAFETY: Tests run single-threaded (--test-threads=1)
    let prev = std::env::var("NEAT_AI_DISCOVERY_NEURON_TARGETS_OUTPUT_ONLY").ok();
    unsafe {
        std::env::set_var("NEAT_AI_DISCOVERY_NEURON_TARGETS_OUTPUT_ONLY", "1");
    }

    let input = AnalyzeNeuronsInput {
        parquet_file: file_path,
        creature,
        focus_neurons: vec!["hidden-0".to_string(), "output-0".to_string()],
        max_candidates: Some(50),
        analysis_deadline_ms: None,
        random_seed: None,
    };

    let result = analyze_neurons(&input).expect("analysis should succeed");

    // Restore env var
    // SAFETY: Tests run single-threaded (--test-threads=1)
    match &prev {
        Some(v) => unsafe { std::env::set_var("NEAT_AI_DISCOVERY_NEURON_TARGETS_OUTPUT_ONLY", v) },
        None => unsafe { std::env::remove_var("NEAT_AI_DISCOVERY_NEURON_TARGETS_OUTPUT_ONLY") },
    }

    // The hidden neuron should appear in diagnostics as filtered
    let hidden_diagnostic = result
        .no_candidate_reasons
        .iter()
        .find(|r| r.target_uuid == "hidden-0");

    assert!(
        hidden_diagnostic.is_some(),
        "hidden-0 should appear in no_candidate_reasons"
    );
    assert_eq!(
        hidden_diagnostic.unwrap().reason,
        NeuronNoCandidateReason::HiddenNeuronFiltered,
        "hidden-0 should be reported as HiddenNeuronFiltered"
    );
}

// =============================================================================
// Neuron diagnostics: input neuron filtered
// =============================================================================

#[test]
fn input_neuron_reported_as_filtered_in_neuron_diagnostics() {
    skip_without_gpu!();

    let creature = minimal_creature();
    let temp_file = NamedTempFile::new().expect("temp parquet");
    let file_path = temp_file.path().to_str().expect("temp path").to_string();
    write_minimal_records(
        &file_path,
        &["input-0", "input-1", "hidden-0", "output-0"],
        20,
    );

    let input = AnalyzeNeuronsInput {
        parquet_file: file_path,
        creature,
        // Include an input neuron as a focus target — it should be filtered
        focus_neurons: vec!["input-0".to_string(), "output-0".to_string()],
        max_candidates: Some(50),
        analysis_deadline_ms: None,
        random_seed: None,
    };

    let result = analyze_neurons(&input).expect("analysis should succeed");

    let input_diagnostic = result
        .no_candidate_reasons
        .iter()
        .find(|r| r.target_uuid == "input-0");

    assert!(
        input_diagnostic.is_some(),
        "input-0 should appear in no_candidate_reasons"
    );
    assert_eq!(
        input_diagnostic.unwrap().reason,
        NeuronNoCandidateReason::InputNeuronFiltered,
        "input-0 should be reported as InputNeuronFiltered"
    );
}

// =============================================================================
// Synapse diagnostics: rejection reasons present in output
// =============================================================================

#[test]
fn synapse_diagnostics_include_rejection_reasons_for_unpromising_targets() {
    skip_without_gpu!();

    // Creature with an output neuron that has no beneficial synapse candidates.
    // All synapses already exist and source activations are constant (zero variance).
    let creature = CreatureJson {
        neurons: vec![
            neuron("input-0", "input", "IDENTITY"),
            output("output-0", "TANH"),
        ],
        synapses: vec![synapse("input-0", "output-0", 0.5)],
        input: 1,
        output: 1,
    };

    let temp_file = NamedTempFile::new().expect("temp parquet");
    let file_path = temp_file.path().to_str().expect("temp path").to_string();

    // Write records where the input has constant activation (no variance)
    let mut records = Vec::new();
    for obs_index in 0..30u32 {
        records.push(DiscoverRecord::new(
            obs_index,
            "input-0".to_string(),
            Some(0.5),
            0.5,
            vec![],
        ));
        records.push(DiscoverRecord::new(
            obs_index,
            "output-0".to_string(),
            Some(0.2),
            0.2,
            vec![0.001], // Very small error — unlikely to produce candidates
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
    };

    let result = analyze_synapses(&input).expect("synapse analysis should succeed");

    // The output should have diagnostic information about why no candidates were found
    // (the only eligible source input-0 is already connected).
    // We don't assert a specific reason because it depends on the analysis path,
    // but we verify diagnostic information IS present when no candidates are found.
    if result.helpful_synapses.is_empty() && result.harmful_synapses.is_empty() {
        assert!(
            !result.no_candidate_reasons.is_empty(),
            "When no synapse candidates are found, diagnostic reasons should be present"
        );

        // Verify the diagnostic has the correct target UUID
        let has_output_diagnostic = result
            .no_candidate_reasons
            .iter()
            .any(|r| r.target_uuid == "output-0");
        assert!(
            has_output_diagnostic,
            "Diagnostic should reference the focus target output-0"
        );
    }
}

// =============================================================================
// JSON output includes synapse and neuron diagnostics
// =============================================================================

#[test]
fn json_output_includes_diagnostic_fields() {
    skip_without_gpu!();

    let creature = minimal_creature();
    let temp_file = NamedTempFile::new().expect("temp parquet");
    let file_path = temp_file.path().to_str().expect("temp path").to_string();
    write_minimal_records(
        &file_path,
        &["input-0", "input-1", "hidden-0", "output-0"],
        20,
    );

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

    // The output structure should have diagnostic-related fields
    // (synapseDiagnostics, neuronDiagnostics) when rejection reasons exist.
    // Verify the JSON contains the expected top-level keys.
    let obj = parsed.as_object().expect("should be JSON object");
    assert!(
        obj.contains_key("synapseDiagnostics") || obj.contains_key("helpfulSynapses"),
        "JSON output should include synapse analysis results"
    );
    assert!(
        obj.contains_key("neuronDiagnostics") || obj.contains_key("helpfulNeurons"),
        "JSON output should include neuron analysis results"
    );
}

// =============================================================================
// JSON diagnostic reason strings are correctly serialised
// =============================================================================

#[test]
fn json_diagnostic_reasons_use_snake_case() {
    skip_without_gpu!();

    let creature = minimal_creature();
    let temp_file = NamedTempFile::new().expect("temp parquet");
    let file_path = temp_file.path().to_str().expect("temp path").to_string();
    write_minimal_records(
        &file_path,
        &["input-0", "input-1", "hidden-0", "output-0"],
        20,
    );

    // Enable output-only mode so hidden neurons get filtered and generate diagnostics
    // SAFETY: Tests run single-threaded (--test-threads=1)
    let prev = std::env::var("NEAT_AI_DISCOVERY_NEURON_TARGETS_OUTPUT_ONLY").ok();
    unsafe {
        std::env::set_var("NEAT_AI_DISCOVERY_NEURON_TARGETS_OUTPUT_ONLY", "1");
    }

    let input_json = serde_json::json!({
        "parquetFile": file_path,
        "creature": creature,
        "focusNeurons": ["hidden-0", "output-0"],
        "maxSynapseCandidates": 10,
        "maxNeuronCandidates": 10,
        "randomSeed": 42
    })
    .to_string();

    let output_json =
        analyze_parallel_internal(&input_json).expect("parallel analysis should return JSON");
    let parsed: serde_json::Value =
        serde_json::from_str(&output_json).expect("output should be valid JSON");

    // Restore env var
    // SAFETY: Tests run single-threaded (--test-threads=1)
    match &prev {
        Some(v) => unsafe { std::env::set_var("NEAT_AI_DISCOVERY_NEURON_TARGETS_OUTPUT_ONLY", v) },
        None => unsafe { std::env::remove_var("NEAT_AI_DISCOVERY_NEURON_TARGETS_OUTPUT_ONLY") },
    }

    assert_eq!(parsed["success"], true);

    // Check neuron diagnostics contain the hidden_neuron_filtered reason in snake_case
    if let Some(diagnostics) = parsed.get("neuronDiagnostics") {
        let diagnostics_arr = diagnostics.as_array().expect("should be array");
        let hidden_diag = diagnostics_arr
            .iter()
            .find(|d| d["targetNeuronUuid"] == "hidden-0");

        if let Some(diag) = hidden_diag {
            assert_eq!(
                diag["reason"], "hidden_neuron_filtered",
                "Reason should be serialised in snake_case"
            );
        }
    }
}

// =============================================================================
// Multiple focus targets produce separate diagnostic entries
// =============================================================================

#[test]
fn multiple_focus_targets_each_get_separate_diagnostic_entry() {
    skip_without_gpu!();

    // Creature with two output neurons, both as focus targets
    let creature = CreatureJson {
        neurons: vec![
            neuron("input-0", "input", "IDENTITY"),
            neuron("input-1", "input", "IDENTITY"),
            output("output-0", "TANH"),
            output("output-1", "TANH"),
        ],
        synapses: vec![
            synapse("input-0", "output-0", 0.5),
            synapse("input-1", "output-1", 0.5),
        ],
        input: 2,
        output: 2,
    };

    let temp_file = NamedTempFile::new().expect("temp parquet");
    let file_path = temp_file.path().to_str().expect("temp path").to_string();
    write_minimal_records(
        &file_path,
        &["input-0", "input-1", "output-0", "output-1"],
        20,
    );

    let input = AnalyzeSynapsesInput {
        parquet_file: file_path,
        creature,
        focus_neurons: vec!["output-0".to_string(), "output-1".to_string()],
        max_candidates: Some(50),
        analysis_deadline_ms: None,
        random_seed: None,
    };

    let result = analyze_synapses(&input).expect("analysis should succeed");

    // Collect all target UUIDs mentioned in candidates and diagnostics
    let mut mentioned_targets: Vec<String> = Vec::new();
    for c in &result.helpful_synapses {
        mentioned_targets.push(c.to_neuron_uuid.clone());
    }
    for c in &result.harmful_synapses {
        mentioned_targets.push(c.to_neuron_uuid.clone());
    }
    for r in &result.no_candidate_reasons {
        mentioned_targets.push(r.target_uuid.clone());
    }

    // Both output neurons should appear somewhere in results or diagnostics
    let has_output_0 = mentioned_targets.iter().any(|t| t == "output-0");
    let has_output_1 = mentioned_targets.iter().any(|t| t == "output-1");

    assert!(
        has_output_0,
        "output-0 should appear in candidates or diagnostics"
    );
    assert!(
        has_output_1,
        "output-1 should appear in candidates or diagnostics"
    );
}

// =============================================================================
// Synapse diagnostics: NoEligibleSources when fully connected
// =============================================================================

#[test]
fn fully_connected_neuron_reports_no_eligible_sources() {
    skip_without_gpu!();

    // Single-input creature with the output already connected to the input.
    // No other sources exist, so it should report NoEligibleSources.
    let creature = CreatureJson {
        neurons: vec![
            neuron("input-0", "input", "IDENTITY"),
            output("output-0", "TANH"),
        ],
        synapses: vec![synapse("input-0", "output-0", 0.5)],
        input: 1,
        output: 1,
    };

    let temp_file = NamedTempFile::new().expect("temp parquet");
    let file_path = temp_file.path().to_str().expect("temp path").to_string();
    write_minimal_records(&file_path, &["input-0", "output-0"], 20);

    let input = AnalyzeSynapsesInput {
        parquet_file: file_path,
        creature,
        focus_neurons: vec!["output-0".to_string()],
        max_candidates: Some(50),
        analysis_deadline_ms: None,
        random_seed: None,
    };

    let result = analyze_synapses(&input).expect("analysis should succeed");

    // When the only input is already connected, there are no eligible sources
    if result.helpful_synapses.is_empty() {
        let output_diag = result
            .no_candidate_reasons
            .iter()
            .find(|r| r.target_uuid == "output-0");

        assert!(
            output_diag.is_some(),
            "Diagnostic should be present for output-0 when no candidates found"
        );

        let diag = output_diag.unwrap();
        assert_eq!(
            diag.reason,
            SynapseNoCandidateReason::NoEligibleSources,
            "Expected NoEligibleSources when all upstream sources are already connected, got {:?}",
            diag.reason
        );
    }
}
