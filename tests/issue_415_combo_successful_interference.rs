//! Issue #415: Fix combo-successful discovery - 0% success rate.
//!
//! This test suite verifies that the discovery system can detect and filter out
//! **interfering** candidate pairs that would hurt each other when combined.
//!
//! ## Problem
//! The combo-successful discovery type has a 0% success rate because individual
//! successful changes interfere when combined. This happens when:
//!
//! 1. **Shared target interference**: Two candidates both target the same neuron
//!    and their combined effect is worse than either alone
//! 2. **Activation saturation**: Combined contributions push a neuron into saturation
//! 3. **Redundant paths**: Two candidates add paths that compute the same thing
//!
//! ## Solution
//! Add interference detection to filter out incompatible candidate pairs before
//! returning coordinated structural candidates.

mod common;

use neat_ai_discovery::analysis::GpuAnalyzer;
use neat_ai_discovery::analysis::recommendation::epistatic::{
    InterferenceType, detect_interfering_pairs,
};
use neat_ai_discovery::analysis::samples::HelpfulSample;
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, analyze_parallel_internal};
use tempfile::tempdir;

/// Test: Detect interference when two candidates target the same neuron with conflicting weights.
///
/// Scenario:
/// - Candidate A: Add synapse from input-0 to output-0 with weight +0.5
/// - Candidate B: Add synapse from input-0 to output-0 with weight -0.5
/// - Combined: These cancel each other out, making the combo useless
///
/// Expected: The system should detect this as a conflicting pair.
#[test]
fn detect_conflicting_weights_to_same_target() {
    // Create two candidates targeting the same synapse with opposite weights
    let sample_count = 64;
    let mut samples_a = Vec::new();
    let mut samples_b = Vec::new();

    for i in 0..sample_count {
        let activation = if i % 2 == 0 { 1.0 } else { -1.0 };
        let error = activation * 0.3;
        samples_a.push(HelpfulSample {
            activation,
            avg_error: error,
            target_value: None,
            target_activation: None,
        });
        // B has the same activation but would use opposite weight
        samples_b.push(HelpfulSample {
            activation,
            avg_error: error,
            target_value: None,
            target_activation: None,
        });
    }

    // Both candidates target the same neuron from the same source
    let interference = detect_interfering_pairs(
        "output-0",
        &[
            ("input-0", &samples_a, 0.5),  // Candidate A: positive weight
            ("input-0", &samples_b, -0.5), // Candidate B: negative weight (conflict!)
        ],
    );

    assert!(
        !interference.is_empty(),
        "Expected to detect interference between conflicting weight candidates"
    );

    let first = &interference[0];
    assert_eq!(
        first.interference_type,
        InterferenceType::ConflictingWeights,
        "Expected ConflictingWeights interference type"
    );
}

/// Test: Detect interference when combined contributions cause saturation.
///
/// Scenario:
/// - Target neuron has TANH activation (saturates at ±1)
/// - Both candidates have high activations with high weights
/// - Combined contributions exceed the saturation risk threshold
///
/// Expected: The system should detect this as saturation interference.
#[test]
fn detect_saturation_interference() {
    // Create samples where both candidates push in the same direction
    // causing potential saturation when combined
    let sample_count = 64;
    let mut samples_a = Vec::new();
    let mut samples_b = Vec::new();

    for _i in 0..sample_count {
        // Both sources have high activations
        // With weight 1.0 each, combined contribution = 1.0 + 0.9 = 1.9 > 1.5 threshold
        samples_a.push(HelpfulSample {
            activation: 1.0,
            avg_error: 0.3,          // Positive error means we want to increase output
            target_value: Some(0.5), // Target is already moderately activated
            target_activation: Some(0.8),
        });
        samples_b.push(HelpfulSample {
            activation: 0.9,
            avg_error: 0.3,
            target_value: Some(0.5),
            target_activation: Some(0.8),
        });
    }

    let interference = detect_interfering_pairs(
        "output-0",
        &[
            ("input-0", &samples_a, 1.0), // High weight
            ("input-1", &samples_b, 1.0), // High weight
        ],
    );

    // This scenario creates saturation risk
    let has_saturation_warning = interference
        .iter()
        .any(|r| r.interference_type == InterferenceType::SaturationRisk);

    assert!(
        has_saturation_warning,
        "Expected to detect saturation risk when combined contributions are high. Interference: {interference:?}"
    );
}

/// Test: Detect interference when candidates are redundant (compute the same thing).
///
/// Scenario:
/// - Candidate A: Add synapse from input-0 to output-0
/// - Candidate B: Add synapse from input-1 to output-0
/// - But input-0 and input-1 have 100% correlated activations
/// - Combined: Adding both is redundant - one would suffice
///
/// Expected: The system should detect this as redundancy.
#[test]
fn detect_redundant_candidates() {
    let sample_count = 64;
    let mut samples_a = Vec::new();
    let mut samples_b = Vec::new();

    for i in 0..sample_count {
        let activation = (i as f32 / sample_count as f32) * 2.0 - 1.0;
        let error = activation * 0.3;

        // Both sources have identical activation patterns (100% correlation)
        samples_a.push(HelpfulSample {
            activation,
            avg_error: error,
            target_value: None,
            target_activation: None,
        });
        samples_b.push(HelpfulSample {
            activation, // Same activation = redundant
            avg_error: error,
            target_value: None,
            target_activation: None,
        });
    }

    let interference = detect_interfering_pairs(
        "output-0",
        &[("input-0", &samples_a, 0.5), ("input-1", &samples_b, 0.5)],
    );

    let has_redundancy = interference
        .iter()
        .any(|r| r.interference_type == InterferenceType::RedundantContribution);

    assert!(
        has_redundancy,
        "Expected to detect redundancy when candidates have identical activation patterns. Interference: {interference:?}"
    );
}

/// Test: No interference detected for truly compatible candidates.
///
/// Scenario:
/// - Candidate A: Helps on first half of samples
/// - Candidate B: Helps on second half of samples
/// - Combined: They are complementary, not interfering
///
/// Expected: No interference should be detected.
#[test]
fn no_interference_for_complementary_candidates() {
    let sample_count = 64;
    let mut samples_a = Vec::new();
    let mut samples_b = Vec::new();

    for i in 0..sample_count {
        let first_half = i < sample_count / 2;

        // A fires on first half, B fires on second half (complementary)
        let activation_a = if first_half { 1.0 } else { 0.0 };
        let activation_b = if first_half { 0.0 } else { 1.0 };

        samples_a.push(HelpfulSample {
            activation: activation_a,
            avg_error: 0.3,
            target_value: None,
            target_activation: None,
        });
        samples_b.push(HelpfulSample {
            activation: activation_b,
            avg_error: 0.3,
            target_value: None,
            target_activation: None,
        });
    }

    let interference = detect_interfering_pairs(
        "output-0",
        &[("input-0", &samples_a, 0.5), ("input-1", &samples_b, 0.5)],
    );

    assert!(
        interference.is_empty(),
        "Expected no interference for complementary candidates. Found: {interference:?}"
    );
}

/// Integration test: Verify that combo-successful filters out interfering candidates.
///
/// This test creates a scenario where:
/// 1. Individual candidates would be successful alone
/// 2. But they interfere when combined
/// 3. The system should NOT propose them as a coordinated candidate
#[test]
fn combo_successful_filters_interfering_pairs() {
    // Discovery is GPU-only
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

    // Create a creature where two inputs have highly correlated activations
    // (redundant - combo would fail)
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

    // Create samples where both inputs have identical patterns
    // This means they're redundant - adding both is no better than one
    let mut records: Vec<DiscoverRecord> = Vec::new();
    let sample_count = 128u32;

    for obs_index in 0..sample_count {
        // Both inputs have the same activation (100% correlated)
        let activation = ((obs_index as f32 / sample_count as f32) * 2.0 - 1.0) * 0.8;
        let error = activation * 0.5; // Error correlates with activation

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
            Some(activation), // Same as input-0 = redundant
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

    // Both inputs should appear as individual helpful synapses
    let helpful = output["helpfulSynapses"]
        .as_array()
        .expect("should have helpfulSynapses");

    let has_input_0 = helpful.iter().any(|s| s["fromNeuronUuid"] == "input-0");
    let has_input_1 = helpful.iter().any(|s| s["fromNeuronUuid"] == "input-1");

    assert!(
        has_input_0 || has_input_1,
        "At least one input should be helpful"
    );

    // BUT they should NOT appear together in a synergistic/epistatic candidate
    // because they're redundant (would cause combo-successful failure)
    let coordinated = output["coordinatedStructuralCandidates"]
        .as_array()
        .cloned()
        .unwrap_or_default();

    let has_combined_candidate = coordinated.iter().any(|c| {
        let Some(ops) = c["operations"].as_array() else {
            return false;
        };
        let has_input_0 = ops.iter().any(|op| op["fromNeuronUuid"] == "input-0");
        let has_input_1 = ops.iter().any(|op| op["fromNeuronUuid"] == "input-1");
        has_input_0 && has_input_1
    });

    assert!(
        !has_combined_candidate,
        "Should NOT find a combined candidate for redundant inputs (would cause combo failure).\n\
         Coordinated candidates: {coordinated:?}"
    );
}

/// Integration test: Verify that truly compatible candidates are still proposed.
///
/// This is a regression test to ensure that filtering doesn't block valid combos.
#[test]
fn combo_successful_allows_compatible_pairs() {
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

    // Create complementary pattern - each input helps different samples
    let mut records: Vec<DiscoverRecord> = Vec::new();
    let sample_count = 128u32;

    for obs_index in 0..sample_count {
        let first_half = obs_index < sample_count / 2;

        // Complementary activation pattern
        let input_0_activation = if first_half { 1.0 } else { 0.0 };
        let input_1_activation = if first_half { 0.0 } else { 1.0 };

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
            "output-0".to_string(),
            Some(0.0),
            0.0,
            vec![0.5], // Constant error - both inputs needed
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

    // Complementary inputs should appear as a combined candidate
    let coordinated = output["coordinatedStructuralCandidates"]
        .as_array()
        .cloned()
        .unwrap_or_default();

    let has_combined_candidate = coordinated.iter().any(|c| {
        let Some(ops) = c["operations"].as_array() else {
            return false;
        };
        let has_input_0 = ops.iter().any(|op| op["fromNeuronUuid"] == "input-0");
        let has_input_1 = ops.iter().any(|op| op["fromNeuronUuid"] == "input-1");
        has_input_0 && has_input_1
    });

    assert!(
        has_combined_candidate,
        "Should find a combined candidate for complementary inputs.\n\
         Coordinated candidates: {coordinated:?}"
    );
}
