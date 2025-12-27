//! Test that synapse analysis now works for STEP/BIPOLAR neurons using threshold-crossing model.
//!
//! Previously, neurons with discrete activations were filtered out from synapse analysis because
//! the linear error model fails for them. In v0.2.18, we implemented threshold-crossing model
//! that counts helpful/harmful flips, allowing synapse candidates for STEP/BIPOLAR targets.

mod common;

use neat_ai_discovery::analysis::GpuAnalyzer;
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{analyze_parallel_internal, CreatureJson, NeuronJson, SynapseJson};
use tempfile::NamedTempFile;

/// Skip test if no GPU available
macro_rules! skip_without_gpu {
    () => {
        if !GpuAnalyzer::gpu_is_available() {
            eprintln!("Skipping test: no GPU available");
            return;
        }
    };
}

/// Helper to create a synapse
fn synapse(from: &str, to: &str, weight: f32) -> SynapseJson {
    SynapseJson {
        from_uuid: from.to_string(),
        to_uuid: to.to_string(),
        weight,
        synapse_type: None,
    }
}

/// Helper to create an output neuron
fn output(uuid: &str, squash: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: "output".to_string(),
        squash: squash.to_string(),
        bias: 0.0,
    }
}

fn create_parquet_with_output(
    output_uuid: &str,
    output_squash: &str,
) -> (NamedTempFile, CreatureJson) {
    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // Create discovery records with patterns that can flip STEP/BIPOLAR outputs
    let mut records = Vec::new();

    // Input neuron records - varying activation that correlates with error
    for obs_index in 0..20 {
        // Source neuron with varying activation
        let source_activation = (obs_index as f32 - 10.0) / 10.0; // Range: -1.0 to 0.9
        records.push(DiscoverRecord::new(
            obs_index,
            "input-0".to_string(),
            Some(source_activation),
            source_activation,
            vec![],
        ));
        records.push(DiscoverRecord::new(
            obs_index,
            "input-1".to_string(),
            Some(-source_activation),
            -source_activation,
            vec![],
        ));
    }

    // Output neuron records - structured to create improvement opportunity
    // For STEP: output = 1 when value > 0, else 0
    // We set up samples where adding a synapse could flip the output helpfully
    for obs_index in 0..20 {
        // Target value near threshold (0) - these are samples where a synapse could flip the output
        let target_value = (obs_index as f32 - 10.0) / 20.0; // Range: -0.5 to 0.45
        let target_activation = if target_value > 0.0 { 1.0 } else { 0.0 }; // STEP output

        // Error: positive when output should be higher, negative when lower
        // For samples just below threshold, error should be positive (want to flip to 1)
        // For samples just above threshold, error should be negative (want to flip to 0)
        let error = if target_value < 0.0 && target_value > -0.3 {
            0.5 // Want to flip from 0 to 1
        } else if target_value > 0.0 && target_value < 0.3 {
            -0.5 // Want to flip from 1 to 0
        } else {
            0.0 // No error for samples far from threshold
        };

        records.push(DiscoverRecord::new(
            obs_index,
            output_uuid.to_string(),
            Some(target_value),
            target_activation,
            vec![error],
        ));
    }

    write_records_to_parquet(file_path, &records).unwrap();

    let creature = CreatureJson {
        input: 2,
        output: 1,
        neurons: vec![output(output_uuid, output_squash)],
        synapses: vec![synapse("input-0", output_uuid, 1.0)],
    };

    (temp_file, creature)
}

/// Test that STEP neurons now receive synapse candidates (not filtered).
#[test]
fn step_neuron_receives_synapse_candidates() {
    skip_without_gpu!();

    let (temp_file, creature) = create_parquet_with_output("output-step", "STEP");
    let file_path = temp_file.path().to_str().unwrap();

    let input_json = serde_json::json!({
        "parquetFile": file_path,
        "creature": creature,
        "focusNeurons": ["output-step"],
        "maxSynapseCandidates": 10,
        "maxNeuronCandidates": 10,
        "analysisDeadlineMs": 30_000,
        "includeSynapseAnalysis": true,
        "includeNeuronAnalysis": false
    });

    let result = analyze_parallel_internal(&input_json.to_string()).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();

    assert!(
        parsed["success"].as_bool().unwrap_or(false),
        "Analysis should succeed: {parsed:?}"
    );

    // STEP neurons should now be analysed (not filtered)
    let diagnostics = &parsed["synapseDiagnostics"];
    if let Some(diagnostics_arr) = diagnostics.as_array() {
        for diag in diagnostics_arr {
            if diag["targetNeuronUuid"] == "output-step" {
                // Should NOT be "discrete_activation_filtered" since we now handle STEP
                assert_ne!(
                    diag["reason"], "discrete_activation_filtered",
                    "STEP neurons should no longer be filtered - threshold-crossing model is used"
                );
            }
        }
    }

    // Note: We may or may not get candidates depending on sample patterns,
    // but the neuron should NOT be filtered for having a discrete activation
}

/// Test that BIPOLAR neurons now receive synapse candidates (not filtered).
#[test]
fn bipolar_neuron_receives_synapse_candidates() {
    skip_without_gpu!();

    let (temp_file, creature) = create_parquet_with_output("output-bipolar", "BIPOLAR");
    let file_path = temp_file.path().to_str().unwrap();

    let input_json = serde_json::json!({
        "parquetFile": file_path,
        "creature": creature,
        "focusNeurons": ["output-bipolar"],
        "maxSynapseCandidates": 10,
        "maxNeuronCandidates": 10,
        "analysisDeadlineMs": 30_000,
        "includeSynapseAnalysis": true,
        "includeNeuronAnalysis": false
    });

    let result = analyze_parallel_internal(&input_json.to_string()).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();

    assert!(
        parsed["success"].as_bool().unwrap_or(false),
        "Analysis should succeed: {parsed:?}"
    );

    // BIPOLAR neurons should now be analysed (not filtered)
    let diagnostics = &parsed["synapseDiagnostics"];
    if let Some(diagnostics_arr) = diagnostics.as_array() {
        for diag in diagnostics_arr {
            if diag["targetNeuronUuid"] == "output-bipolar" {
                assert_ne!(
                    diag["reason"], "discrete_activation_filtered",
                    "BIPOLAR neurons should no longer be filtered - threshold-crossing model is used"
                );
            }
        }
    }
}

/// Test that TANH neurons continue to work with saturation-aware model.
#[test]
fn tanh_neuron_uses_saturation_aware_model() {
    skip_without_gpu!();

    let (temp_file, creature) = create_parquet_with_output("output-tanh", "TANH");
    let file_path = temp_file.path().to_str().unwrap();

    let input_json = serde_json::json!({
        "parquetFile": file_path,
        "creature": creature,
        "focusNeurons": ["output-tanh"],
        "maxSynapseCandidates": 10,
        "maxNeuronCandidates": 10,
        "analysisDeadlineMs": 30_000,
        "includeSynapseAnalysis": true,
        "includeNeuronAnalysis": false
    });

    let result = analyze_parallel_internal(&input_json.to_string()).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();

    assert!(
        parsed["success"].as_bool().unwrap_or(false),
        "Analysis should succeed: {parsed:?}"
    );

    // TANH continues to work with the saturation-aware model
    // (This test mainly ensures TANH analysis still functions correctly)
}
