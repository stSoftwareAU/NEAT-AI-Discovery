//! Tests for creature-level metrics (Issue #128)
//!
//! ## Problem (Issue #128)
//!
//! The `expectedImprovementPercentage` field was measuring the TARGET NEURON's error
//! reduction, not the CREATURE's error reduction. This is meaningless because:
//!
//! - A focus neuron with error 0.0003 where we find a candidate reducing error to 0.0002
//!   shows "33% improvement"
//! - BUT if that neuron has impact 0.0001 on the creature's error, the actual creature
//!   improvement is ~0.000033, which may be less than cost of growth!
//!
//! ## Solution
//!
//! Replace `expectedImprovementPercentage` with creature-level metrics:
//!
//! 1. `targetNeuronImpact` - The target neuron's impact on the creature (0.0 to 1.0)
//! 2. `expectedCreatureErrorReduction` - Expected reduction in creature's error
//! 3. `expectedCreatureScoreGain` - Expected improvement in creature's score
//!
//! All candidates should be sorted by `expectedCreatureScoreGain` (highest first).

use neat_ai_discovery::analysis::GpuAnalyzer;
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson, analyze_parallel_internal};
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

/// Helper to create a hidden neuron
fn hidden(uuid: &str, squash: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: "hidden".to_string(),
        squash: squash.to_string(),
        bias: 0.0,
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

/// Build a minimal set of records that produces synapse candidates without triggering
/// Issue #178 constant-source folding (7-Jan-2026).
///
/// We intentionally make the input activation vary slightly so synapse analysis returns an
/// `addSynapse` candidate (with creature-level fields) rather than converting it into a
/// coordinated `setBias` operation.
fn records_single_output_with_varying_input(error: f32) -> Vec<DiscoverRecord> {
    let mut records = Vec::new();
    for obs_index in 0..10u32 {
        let input_activation = if (obs_index % 2) == 0 { 0.4 } else { 0.6 };
        records.push(DiscoverRecord::new(
            obs_index,
            "input-0".to_string(),
            Some(input_activation),
            input_activation,
            vec![],
        ));
        records.push(DiscoverRecord::new(
            obs_index,
            "output-0".to_string(),
            Some(0.0),
            0.0,
            vec![error],
        ));
    }
    records
}

// =============================================================================
// Test: expectedImprovementPercentage field should NOT exist (Issue #128)
// =============================================================================

/// Issue #128: The `expectedImprovementPercentage` field should be removed from candidates.
///
/// This test verifies that the old meaningless field is no longer present in the JSON output.
#[test]
fn test_expected_improvement_percentage_field_removed() {
    skip_without_gpu!();

    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![output("output-0", "IDENTITY")],
        synapses: vec![],
    };

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // Create records with error so we get candidates
    let records = records_single_output_with_varying_input(0.5);
    write_records_to_parquet(file_path, &records).unwrap();

    let input_json = serde_json::json!({
        "parquetFile": file_path,
        "creature": creature,
        "focusNeurons": ["output-0"],
    })
    .to_string();

    let output_json = analyze_parallel_internal(&input_json).expect("analysis should succeed");

    // Verify the old field is NOT present in the JSON
    assert!(
        !output_json.contains("expectedImprovementPercentage"),
        "expectedImprovementPercentage should be removed from JSON output. Found in: {output_json}"
    );
}

// =============================================================================
// Test: New creature-level fields should exist
// =============================================================================

/// Issue #128: Candidates should include `targetNeuronImpact` field.
///
/// This field shows the impact of the target neuron on the creature's output (0.0 to 1.0).
/// Output neurons have impact = 1.0, hidden neurons have impact < 1.0.
#[test]
fn test_target_neuron_impact_field_exists() {
    skip_without_gpu!();

    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![output("output-0", "IDENTITY")],
        synapses: vec![],
    };

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    let records = records_single_output_with_varying_input(0.5);
    write_records_to_parquet(file_path, &records).unwrap();

    let input_json = serde_json::json!({
        "parquetFile": file_path,
        "creature": creature,
        "focusNeurons": ["output-0"],
    })
    .to_string();

    let output_json = analyze_parallel_internal(&input_json).expect("analysis should succeed");

    // Verify the new field IS present in the JSON
    assert!(
        output_json.contains("targetNeuronImpact"),
        "targetNeuronImpact field should be present in JSON output. Got: {output_json}"
    );
}

/// Issue #128: Candidates should include `expectedCreatureErrorReduction` field.
///
/// This field shows the expected reduction in the creature's overall error.
/// Formula: neuron_error_reduction × target_neuron_impact
#[test]
fn test_expected_creature_error_reduction_field_exists() {
    skip_without_gpu!();

    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![output("output-0", "IDENTITY")],
        synapses: vec![],
    };

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    let records = records_single_output_with_varying_input(0.5);
    write_records_to_parquet(file_path, &records).unwrap();

    let input_json = serde_json::json!({
        "parquetFile": file_path,
        "creature": creature,
        "focusNeurons": ["output-0"],
    })
    .to_string();

    let output_json = analyze_parallel_internal(&input_json).expect("analysis should succeed");

    assert!(
        output_json.contains("expectedCreatureErrorReduction"),
        "expectedCreatureErrorReduction field should be present in JSON output. Got: {output_json}"
    );
}

/// Issue #128: Candidates should include `expectedCreatureScoreGain` field.
///
/// This field shows the expected improvement in the creature's score.
/// Since score = 1 - error, score_gain ≈ error_reduction.
#[test]
fn test_expected_creature_score_gain_field_exists() {
    skip_without_gpu!();

    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![output("output-0", "IDENTITY")],
        synapses: vec![],
    };

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    let records = records_single_output_with_varying_input(0.5);
    write_records_to_parquet(file_path, &records).unwrap();

    let input_json = serde_json::json!({
        "parquetFile": file_path,
        "creature": creature,
        "focusNeurons": ["output-0"],
    })
    .to_string();

    let output_json = analyze_parallel_internal(&input_json).expect("analysis should succeed");

    assert!(
        output_json.contains("expectedCreatureScoreGain"),
        "expectedCreatureScoreGain field should be present in JSON output. Got: {output_json}"
    );
}

// =============================================================================
// Test: Output neuron has impact = 1.0
// =============================================================================

/// Issue #128: Output neurons should have targetNeuronImpact = 1.0
///
/// Output neurons directly contribute to the creature's score, so their impact is 1.0.
#[test]
fn test_output_neuron_has_impact_one() {
    skip_without_gpu!();

    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![output("output-0", "IDENTITY")],
        synapses: vec![],
    };

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    let records = records_single_output_with_varying_input(0.5);
    write_records_to_parquet(file_path, &records).unwrap();

    let input_json = serde_json::json!({
        "parquetFile": file_path,
        "creature": creature,
        "focusNeurons": ["output-0"],
    })
    .to_string();

    let output_json = analyze_parallel_internal(&input_json).expect("analysis should succeed");
    let output: serde_json::Value =
        serde_json::from_str(&output_json).expect("output should be valid JSON");

    // Check synapse candidates (if any)
    if let Some(synapses) = output["helpfulSynapses"].as_array() {
        for synapse in synapses {
            let impact = synapse["targetNeuronImpact"]
                .as_f64()
                .expect("targetNeuronImpact should be a number");
            assert!(
                (impact - 1.0).abs() < 0.01,
                "Output neuron should have impact = 1.0, got {impact}"
            );
        }
    }

    // Check neuron candidates (if any)
    if let Some(neurons) = output["helpfulNeurons"].as_array() {
        for neuron in neurons {
            let impact = neuron["targetNeuronImpact"]
                .as_f64()
                .expect("targetNeuronImpact should be a number");
            assert!(
                (impact - 1.0).abs() < 0.01,
                "Output neuron should have impact = 1.0, got {impact}"
            );
        }
    }
}

// =============================================================================
// Test: Hidden neuron has discounted impact
// =============================================================================

/// Issue #128: Hidden neurons should have targetNeuronImpact < 1.0
///
/// Hidden neurons don't directly contribute to the creature's score - their impact
/// is based on their weighted paths to output neurons.
#[test]
fn test_hidden_neuron_has_discounted_impact() {
    skip_without_gpu!();

    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![hidden("hidden-0", "TANH"), output("output-0", "IDENTITY")],
        synapses: vec![
            synapse("input-0", "hidden-0", 1.0),
            synapse("hidden-0", "output-0", 0.5), // 50% weight to output
        ],
    };

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // Create records for all neurons
    let mut records = Vec::new();
    for i in 0..10u32 {
        records.push(DiscoverRecord::new(
            i,
            "input-0".to_string(),
            Some(0.5),
            0.5,
            vec![],
        ));
        records.push(DiscoverRecord::new(
            i,
            "hidden-0".to_string(),
            Some(0.46),
            0.46, // tanh(0.5) ≈ 0.46
            vec![0.3],
        ));
        records.push(DiscoverRecord::new(
            i,
            "output-0".to_string(),
            Some(0.23),
            0.23,
            vec![0.2],
        ));
    }
    write_records_to_parquet(file_path, &records).unwrap();

    let input_json = serde_json::json!({
        "parquetFile": file_path,
        "creature": creature,
        "focusNeurons": ["hidden-0"],
    })
    .to_string();

    let output_json = analyze_parallel_internal(&input_json).expect("analysis should succeed");
    let output: serde_json::Value =
        serde_json::from_str(&output_json).expect("output should be valid JSON");

    // Check synapse candidates targeting hidden-0
    if let Some(synapses) = output["helpfulSynapses"].as_array() {
        for synapse in synapses {
            if synapse["toNeuronUuid"].as_str() == Some("hidden-0") {
                let impact = synapse["targetNeuronImpact"]
                    .as_f64()
                    .expect("targetNeuronImpact should be a number");
                assert!(
                    impact < 1.0,
                    "Hidden neuron should have impact < 1.0, got {impact}"
                );
                assert!(
                    impact > 0.0,
                    "Hidden neuron should have impact > 0.0, got {impact}"
                );
            }
        }
    }
}

// =============================================================================
// Test: Candidates sorted by expectedCreatureScoreGain
// =============================================================================

/// Issue #128: Candidates should be sorted by expectedCreatureScoreGain (highest first).
///
/// The goal is to improve the CREATURE's SCORE. Candidates should be sorted by their
/// expected contribution to that goal.
#[test]
fn test_candidates_sorted_by_expected_creature_score_gain() {
    skip_without_gpu!();

    let creature = CreatureJson {
        input: 2,
        output: 1,
        neurons: vec![output("output-0", "IDENTITY")],
        synapses: vec![],
    };

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // Create records with two inputs having different correlations with error
    let mut records = Vec::new();
    for i in 0..20u32 {
        // input-0: high positive correlation with error
        records.push(DiscoverRecord::new(
            i,
            "input-0".to_string(),
            Some(0.8),
            0.8,
            vec![],
        ));
        // input-1: low correlation with error
        records.push(DiscoverRecord::new(
            i,
            "input-1".to_string(),
            Some(0.1),
            0.1,
            vec![],
        ));
        // Output has consistent error
        records.push(DiscoverRecord::new(
            i,
            "output-0".to_string(),
            Some(0.0),
            0.0,
            vec![0.5],
        ));
    }
    write_records_to_parquet(file_path, &records).unwrap();

    let input_json = serde_json::json!({
        "parquetFile": file_path,
        "creature": creature,
        "focusNeurons": ["output-0"],
        "maxSynapseCandidates": 10,
    })
    .to_string();

    let output_json = analyze_parallel_internal(&input_json).expect("analysis should succeed");
    let output: serde_json::Value =
        serde_json::from_str(&output_json).expect("output should be valid JSON");

    // Check that synapse candidates are sorted by expectedCreatureScoreGain descending
    if let Some(synapses) = output["helpfulSynapses"].as_array()
        && synapses.len() > 1
    {
        let mut prev_score_gain = f64::INFINITY;
        for synapse in synapses {
            let score_gain = synapse["expectedCreatureScoreGain"]
                .as_f64()
                .expect("expectedCreatureScoreGain should be a number");
            assert!(
                score_gain <= prev_score_gain,
                "Candidates should be sorted by expectedCreatureScoreGain descending. \
                 Got {score_gain} after {prev_score_gain}"
            );
            prev_score_gain = score_gain;
        }
    }
}
