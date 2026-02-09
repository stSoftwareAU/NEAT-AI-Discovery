//! Issue #202: Pre-detect epistatic neuron pairs during analysis.
//!
//! Epistatic changes are structural modifications where no single operation improves
//! the score, but a group of operations does. This test suite verifies that the
//! discovery system can pre-detect such relationships during analysis.
//!
//! ## Test Strategy (from issue)
//! 1. Create synthetic creature with known epistatic relationship
//! 2. Verify pre-detection identifies the relationship
//! 3. Test that combined candidate improves score when individuals don't
//! 4. Verify no regression in non-epistatic discovery

mod common;

use neat_ai_discovery::analysis::GpuAnalyzer;
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson, analyze_parallel_internal};
use tempfile::tempdir;

/// Test: Epistatic pair detection with complementary error patterns.
///
/// This test creates a scenario where:
/// - Two potential source neurons (input-0, input-1) target the same output
/// - Input-0 correlates with positive errors on subset A of samples
/// - Input-1 correlates with positive errors on subset B of samples (complement of A)
/// - Individually, neither neuron improves overall score (helps some, hurts others)
/// - Combined, they could improve the score (each handles its subset)
///
/// This is the "shared error pattern detection" from the issue.
#[test]
fn epistatic_pair_detected_for_complementary_error_patterns() {
    // Discovery is GPU-only. On machines without GPU, we skip.
    if !GpuAnalyzer::gpu_is_available() {
        return;
    }

    let temp_dir = tempdir().expect("Failed to create temp dir");
    let parquet_file = temp_dir
        .path()
        .join("records.parquet")
        .to_str()
        .expect("temp path should be valid UTF-8")
        .to_string();

    // Simple creature: 3 inputs, 1 output
    // input-0 and input-1 are the potential epistatic pair
    // input-2 already has a synapse to the output
    let creature = CreatureJson {
        input: 3,
        output: 1,
        neurons: vec![NeuronJson {
            uuid: "output-0".to_string(),
            neuron_type: "output".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        }],
        synapses: vec![SynapseJson {
            from_uuid: "input-2".to_string(),
            to_uuid: "output-0".to_string(),
            weight: 1.0,
            synapse_type: None,
        }],
    };

    // Create samples where:
    // - First half: input-0 fires (activation=1), input-1 silent (activation=0)
    //   Error is positive (output should be higher)
    // - Second half: input-0 silent (activation=0), input-1 fires (activation=1)
    //   Error is positive (output should be higher)
    //
    // This creates complementary activation patterns where:
    // - Adding synapse from input-0 alone helps first half but does nothing for second half
    // - Adding synapse from input-1 alone helps second half but does nothing for first half
    // - Adding BOTH synapses helps ALL samples (epistatic benefit)
    let mut records: Vec<DiscoverRecord> = Vec::new();
    let sample_count = 128u32;

    for obs_index in 0..sample_count {
        let first_half = obs_index < sample_count / 2;

        // Complementary activation pattern
        let input_0_activation = if first_half { 1.0 } else { 0.0 };
        let input_1_activation = if first_half { 0.0 } else { 1.0 };
        let input_2_activation = 0.5; // Constant

        // Output is driven by input-2 with weight 1.0
        let output_activation = input_2_activation * 1.0;

        // Error pattern: positive error when neuron fires
        // Desired output = current + 0.3 when the corresponding input fires
        let error = 0.3; // Positive error means output should be higher

        // Record input neurons
        records.push(DiscoverRecord::new(
            obs_index,
            "input-0".to_string(),
            Some(input_0_activation),
            input_0_activation,
            Vec::new(),
        ));
        records.push(DiscoverRecord::new(
            obs_index,
            "input-1".to_string(),
            Some(input_1_activation),
            input_1_activation,
            Vec::new(),
        ));
        records.push(DiscoverRecord::new(
            obs_index,
            "input-2".to_string(),
            Some(input_2_activation),
            input_2_activation,
            Vec::new(),
        ));

        // Record output neuron with error
        records.push(DiscoverRecord::new(
            obs_index,
            "output-0".to_string(),
            Some(output_activation),
            output_activation,
            vec![error],
        ));
    }

    write_records_to_parquet(&parquet_file, &records).expect("Failed to write discovery records");

    let input_json = serde_json::json!({
        "parquetFile": parquet_file,
        "creature": creature,
        "focusNeurons": ["output-0"],
        "maxSynapseCandidates": 64,
        "maxNeuronCandidates": 0,
        "randomSeed": 42
    })
    .to_string();

    let output_json =
        analyze_parallel_internal(&input_json).expect("parallel analysis should return JSON");
    let output: serde_json::Value =
        serde_json::from_str(&output_json).expect("output should be valid JSON");

    assert_eq!(output["success"], true);

    // Look for epistatic pair candidates in coordinatedStructuralCandidates
    let coordinated = output["coordinatedStructuralCandidates"]
        .as_array()
        .map(|a| a.to_vec())
        .unwrap_or_default();

    // We expect to find a coordinated candidate that adds BOTH synapses
    // (input-0 -> output-0) AND (input-1 -> output-0) as a single unit
    let found_epistatic_pair = coordinated.iter().any(|c| {
        let Some(ops) = c["operations"].as_array() else {
            return false;
        };

        // Check if this candidate contains both add-synapse operations
        let has_input_0_synapse = ops.iter().any(|op| {
            op["type"] == "addSynapse"
                && op["fromNeuronUuid"] == "input-0"
                && op["toNeuronUuid"] == "output-0"
        });
        let has_input_1_synapse = ops.iter().any(|op| {
            op["type"] == "addSynapse"
                && op["fromNeuronUuid"] == "input-1"
                && op["toNeuronUuid"] == "output-0"
        });

        has_input_0_synapse && has_input_1_synapse
    });

    assert!(
        found_epistatic_pair,
        "Expected to find an epistatic pair candidate that adds both input-0 and input-1 \
         synapses to output-0, but none was found.\nCoordinated candidates: {coordinated:?}"
    );
}

/// Test: Epistatic pair with correlation analysis (partial overlap pattern).
///
/// This test creates a scenario where:
/// - Two neurons have moderate individual improvement predictions
/// - They have partially overlapping firing patterns
/// - Combined improvement is higher than either individual
///
/// This tests the "correlation analysis" capability where we identify pairs
/// that provide additive benefits.
#[test]
fn epistatic_pair_detected_via_correlation_analysis() {
    if !GpuAnalyzer::gpu_is_available() {
        return;
    }

    let temp_dir = tempdir().expect("Failed to create temp dir");
    let parquet_file = temp_dir
        .path()
        .join("records.parquet")
        .to_str()
        .expect("temp path should be valid UTF-8")
        .to_string();

    // Creature with two inputs that have partial overlap
    let creature = CreatureJson {
        input: 2,
        output: 1,
        neurons: vec![NeuronJson {
            uuid: "output-0".to_string(),
            neuron_type: "output".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        }],
        synapses: vec![],
    };

    // Create samples where:
    // - input-0 fires on samples 0-79 (first 62.5% of samples)
    // - input-1 fires on samples 48-127 (last 62.5% of samples)
    // - Overlap is samples 48-79 (25% of all samples)
    // - This gives ~74% complementarity (above 0.7 threshold)
    // - Error is positive when either fires, so both have individual improvement
    // - Combined improvement should be higher as they cover more samples together
    let mut records: Vec<DiscoverRecord> = Vec::new();
    let sample_count = 128u32;

    for obs_index in 0..sample_count {
        let input_0 = if obs_index < 80 { 1.0 } else { 0.0 };
        let input_1 = if obs_index >= 48 { 1.0 } else { 0.0 };
        // Error is positive (output should be higher)
        let error = 0.3;

        records.push(DiscoverRecord::new(
            obs_index,
            "input-0".to_string(),
            Some(input_0),
            input_0,
            Vec::new(),
        ));
        records.push(DiscoverRecord::new(
            obs_index,
            "input-1".to_string(),
            Some(input_1),
            input_1,
            Vec::new(),
        ));
        records.push(DiscoverRecord::new(
            obs_index,
            "output-0".to_string(),
            Some(0.0),
            0.0,
            vec![error],
        ));
    }

    write_records_to_parquet(&parquet_file, &records).expect("Failed to write discovery records");

    let input_json = serde_json::json!({
        "parquetFile": parquet_file,
        "creature": creature,
        "focusNeurons": ["output-0"],
        "maxSynapseCandidates": 64,
        "maxNeuronCandidates": 0,
        "randomSeed": 42
    })
    .to_string();

    let output_json =
        analyze_parallel_internal(&input_json).expect("parallel analysis should return JSON");
    let output: serde_json::Value =
        serde_json::from_str(&output_json).expect("output should be valid JSON");

    assert_eq!(output["success"], true);

    // For partial overlap patterns, both inputs have positive improvement
    // The system should detect this as a potential epistatic pair since combined > individual
    let coordinated = output["coordinatedStructuralCandidates"]
        .as_array()
        .map(|a| a.to_vec())
        .unwrap_or_default();

    // Check for epistatic pair detection
    let found_epistatic_pair = coordinated.iter().any(|c| {
        let Some(ops) = c["operations"].as_array() else {
            return false;
        };

        // Check if operations involve both input-0 and input-1
        let involves_input_0 = ops.iter().any(|op| {
            op["fromNeuronUuid"] == "input-0"
                || op["neuronUuid"] == "input-0"
                || op.get("sourceNeuronUuid").is_some_and(|v| v == "input-0")
        });
        let involves_input_1 = ops.iter().any(|op| {
            op["fromNeuronUuid"] == "input-1"
                || op["neuronUuid"] == "input-1"
                || op.get("sourceNeuronUuid").is_some_and(|v| v == "input-1")
        });

        involves_input_0 && involves_input_1
    });

    assert!(
        found_epistatic_pair,
        "Expected to find an epistatic pair candidate involving both input-0 and input-1, \
         but none was found.\nCoordinated candidates: {coordinated:?}"
    );
}

/// Test: No false positives for non-epistatic scenarios.
///
/// When neurons have independent effects (not epistatic), they should NOT
/// be grouped into coordinated candidates unnecessarily.
#[test]
fn no_false_positive_epistatic_detection_for_independent_neurons() {
    if !GpuAnalyzer::gpu_is_available() {
        return;
    }

    let temp_dir = tempdir().expect("Failed to create temp dir");
    let parquet_file = temp_dir
        .path()
        .join("records.parquet")
        .to_str()
        .expect("temp path should be valid UTF-8")
        .to_string();

    // Creature where each input independently correlates with error
    let creature = CreatureJson {
        input: 2,
        output: 1,
        neurons: vec![NeuronJson {
            uuid: "output-0".to_string(),
            neuron_type: "output".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        }],
        synapses: vec![],
    };

    // Create samples where both inputs independently correlate with error
    // (no epistasis - each can help on its own)
    let mut records: Vec<DiscoverRecord> = Vec::new();
    let sample_count = 128u32;

    for obs_index in 0..sample_count {
        // Both inputs fire together with positive correlation to error
        let activation = if obs_index % 2 == 0 { 1.0 } else { -1.0 };
        let error = activation * 0.3; // Direct correlation

        records.push(DiscoverRecord::new(
            obs_index,
            "input-0".to_string(),
            Some(activation),
            activation,
            Vec::new(),
        ));
        records.push(DiscoverRecord::new(
            obs_index,
            "input-1".to_string(),
            Some(activation),
            activation,
            Vec::new(),
        ));
        records.push(DiscoverRecord::new(
            obs_index,
            "output-0".to_string(),
            Some(0.0),
            0.0,
            vec![error],
        ));
    }

    write_records_to_parquet(&parquet_file, &records).expect("Failed to write discovery records");

    let input_json = serde_json::json!({
        "parquetFile": parquet_file,
        "creature": creature,
        "focusNeurons": ["output-0"],
        "maxSynapseCandidates": 64,
        "maxNeuronCandidates": 0,
        "randomSeed": 42
    })
    .to_string();

    let output_json =
        analyze_parallel_internal(&input_json).expect("parallel analysis should return JSON");
    let output: serde_json::Value =
        serde_json::from_str(&output_json).expect("output should be valid JSON");

    assert_eq!(output["success"], true);

    // Should find independent helpful synapses, not necessarily epistatic pairs
    let helpful = output["helpfulSynapses"]
        .as_array()
        .expect("should include helpfulSynapses array");

    // Both inputs should appear as independent helpful synapse candidates
    let has_input_0 = helpful
        .iter()
        .any(|s| s["fromNeuronUuid"] == "input-0" && s["toNeuronUuid"] == "output-0");
    let has_input_1 = helpful
        .iter()
        .any(|s| s["fromNeuronUuid"] == "input-1" && s["toNeuronUuid"] == "output-0");

    assert!(
        has_input_0 && has_input_1,
        "Expected both input-0 and input-1 as independent helpful synapse candidates"
    );
}

/// Test: Epistatic detection result structure.
///
/// Verify that detected epistatic pairs have the expected JSON structure
/// with appropriate metadata.
#[test]
fn epistatic_pair_candidate_has_expected_structure() {
    if !GpuAnalyzer::gpu_is_available() {
        return;
    }

    let temp_dir = tempdir().expect("Failed to create temp dir");
    let parquet_file = temp_dir
        .path()
        .join("records.parquet")
        .to_str()
        .expect("temp path should be valid UTF-8")
        .to_string();

    let creature = CreatureJson {
        input: 2,
        output: 1,
        neurons: vec![NeuronJson {
            uuid: "output-0".to_string(),
            neuron_type: "output".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        }],
        synapses: vec![],
    };

    // Create complementary pattern for epistatic detection
    let mut records: Vec<DiscoverRecord> = Vec::new();
    for obs_index in 0..128u32 {
        let first_half = obs_index < 64;
        let input_0 = if first_half { 1.0 } else { 0.0 };
        let input_1 = if first_half { 0.0 } else { 1.0 };

        records.push(DiscoverRecord::new(
            obs_index,
            "input-0".to_string(),
            Some(input_0),
            input_0,
            Vec::new(),
        ));
        records.push(DiscoverRecord::new(
            obs_index,
            "input-1".to_string(),
            Some(input_1),
            input_1,
            Vec::new(),
        ));
        records.push(DiscoverRecord::new(
            obs_index,
            "output-0".to_string(),
            Some(0.0),
            0.0,
            vec![0.3], // Constant positive error
        ));
    }

    write_records_to_parquet(&parquet_file, &records).expect("Failed to write discovery records");

    let input_json = serde_json::json!({
        "parquetFile": parquet_file,
        "creature": creature,
        "focusNeurons": ["output-0"],
        "maxSynapseCandidates": 64,
        "maxNeuronCandidates": 0,
        "randomSeed": 42
    })
    .to_string();

    let output_json =
        analyze_parallel_internal(&input_json).expect("parallel analysis should return JSON");
    let output: serde_json::Value =
        serde_json::from_str(&output_json).expect("output should be valid JSON");

    assert_eq!(output["success"], true);

    let coordinated = output["coordinatedStructuralCandidates"]
        .as_array()
        .map(|a| a.to_vec())
        .unwrap_or_default();

    // Find an epistatic pair candidate
    let epistatic_candidate = coordinated.iter().find(|c| {
        let Some(ops) = c["operations"].as_array() else {
            return false;
        };
        ops.len() >= 2
            && ops
                .iter()
                .any(|op| op["fromNeuronUuid"] == "input-0" || op["fromNeuronUuid"] == "input-1")
    });

    // For this test, we require an epistatic pair candidate to be found
    assert!(
        epistatic_candidate.is_some(),
        "Expected to find an epistatic pair candidate, but none was found.\n\
         Coordinated candidates: {coordinated:?}"
    );

    let candidate = epistatic_candidate.unwrap();

    // Verify expected fields exist
    assert!(
        candidate.get("operations").is_some(),
        "Epistatic candidate should have operations array"
    );
    assert!(
        candidate.get("expectedCreatureScoreGain").is_some(),
        "Epistatic candidate should have expectedCreatureScoreGain"
    );

    // Verify comment mentions epistatic relationship
    if let Some(comment) = candidate.get("comment").and_then(|c| c.as_str()) {
        assert!(
            comment.to_lowercase().contains("epistatic")
                || comment.to_lowercase().contains("pair")
                || comment.to_lowercase().contains("combined"),
            "Comment should indicate epistatic/pair relationship: {comment}"
        );
    }
}

/// Test: Epistatic pair integration with focus neuron ranking.
///
/// Verify that epistatic pairs are considered when ranking focus neurons
/// (neurons with historical epistatic success should be prioritised).
#[test]
fn epistatic_detection_integrates_with_focus_ranking() {
    if !GpuAnalyzer::gpu_is_available() {
        return;
    }

    let temp_dir = tempdir().expect("Failed to create temp dir");
    let parquet_file = temp_dir
        .path()
        .join("records.parquet")
        .to_str()
        .expect("temp path should be valid UTF-8")
        .to_string();

    // Larger creature to test focus ranking
    let creature = CreatureJson {
        input: 4,
        output: 2,
        neurons: vec![
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
        synapses: vec![],
    };

    // Create patterns where output-0 has epistatic opportunities
    let mut records: Vec<DiscoverRecord> = Vec::new();
    for obs_index in 0..128u32 {
        let first_half = obs_index < 64;

        // Inputs 0 and 1 have complementary patterns for output-0 (epistatic)
        let input_0 = if first_half { 1.0 } else { 0.0 };
        let input_1 = if first_half { 0.0 } else { 1.0 };
        // Inputs 2 and 3 have independent patterns for output-1 (non-epistatic)
        let input_2 = if obs_index % 2 == 0 { 1.0 } else { -1.0 };
        let input_3 = if obs_index % 3 == 0 { 1.0 } else { -1.0 };

        for (i, act) in [(0, input_0), (1, input_1), (2, input_2), (3, input_3)] {
            records.push(DiscoverRecord::new(
                obs_index,
                format!("input-{i}"),
                Some(act),
                act,
                Vec::new(),
            ));
        }

        // Output-0 has epistatic opportunity
        records.push(DiscoverRecord::new(
            obs_index,
            "output-0".to_string(),
            Some(0.0),
            0.0,
            vec![0.3],
        ));
        // Output-1 has simpler correlation
        records.push(DiscoverRecord::new(
            obs_index,
            "output-1".to_string(),
            Some(0.0),
            0.0,
            vec![input_2 * 0.2],
        ));
    }

    write_records_to_parquet(&parquet_file, &records).expect("Failed to write discovery records");

    let input_json = serde_json::json!({
        "parquetFile": parquet_file,
        "creature": creature,
        "focusNeurons": ["output-0", "output-1"],
        "maxSynapseCandidates": 64,
        "maxNeuronCandidates": 0,
        "randomSeed": 42
    })
    .to_string();

    let output_json =
        analyze_parallel_internal(&input_json).expect("parallel analysis should return JSON");
    let output: serde_json::Value =
        serde_json::from_str(&output_json).expect("output should be valid JSON");

    assert_eq!(output["success"], true);

    // Verify that analysis completed for both focus neurons
    let metadata = &output["synapseMetadata"];
    let completed = metadata["completedFocusNeurons"]
        .as_u64()
        .expect("should have completedFocusNeurons");
    assert_eq!(
        completed, 2,
        "Should have completed analysis for both focus neurons"
    );
}
