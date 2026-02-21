//! Issue #189: Cross-neuron interaction detection for synergistic discoveries.
//!
//! This test suite verifies that the discovery system can find synergistic improvements
//! where two sources together reduce error better than either alone (XOR-like patterns).
//!
//! ## Test Strategy (from issue)
//! 1. Create synthetic creature with known XOR pattern
//! 2. Verify current discovery misses the pattern with individual analysis
//! 3. Verify new residual-based approach finds synergistic candidates
//! 4. Measure that discovery still finds non-synergistic patterns correctly

mod common;

use neat_ai_discovery::analysis::GpuAnalyzer;
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, analyze_parallel_internal};
use tempfile::tempdir;

/// Test: XOR-like pattern detection where neither input alone predicts the output.
///
/// This is the canonical synergistic pattern:
/// - Input A = 0, Input B = 0 → Output should be 0 (low)
/// - Input A = 0, Input B = 1 → Output should be 1 (high)
/// - Input A = 1, Input B = 0 → Output should be 1 (high)
/// - Input A = 1, Input B = 1 → Output should be 0 (low)
///
/// Neither input alone correlates with the output, but together they do (A XOR B).
/// The residual analysis approach should detect this:
/// 1. Find best single source (weak correlation ~0%)
/// 2. Compute residual error
/// 3. Find second source that reduces residual
/// 4. Detect that combined improvement > individual improvements
#[test]
fn synergistic_discovery_detects_xor_pattern() {
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

    // Simple creature: 2 XOR inputs, 1 output
    // No existing synapses - discovery should find that both inputs are needed
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

    // Create XOR pattern samples
    // XOR truth table:
    // A=0, B=0 → desired=0, error=0 (since output=0)
    // A=0, B=1 → desired=1, error=1 (output should be higher)
    // A=1, B=0 → desired=1, error=1 (output should be higher)
    // A=1, B=1 → desired=0, error=0 (since output=0)
    let mut records: Vec<DiscoverRecord> = Vec::new();
    let samples_per_pattern = 64u32;

    // Pattern 00: Both inputs 0, output should be 0
    for i in 0..samples_per_pattern {
        let obs_index = i;
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
            Some(0.0),
            0.0,
            Vec::new(),
        ));
        records.push(DiscoverRecord::new(
            obs_index,
            "output-0".to_string(),
            Some(0.0),
            0.0,
            vec![0.0], // No error - output is correct
        ));
    }

    // Pattern 01: Input-0=0, Input-1=1, output should be 1
    for i in 0..samples_per_pattern {
        let obs_index = samples_per_pattern + i;
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
            Some(1.0),
            1.0,
            Vec::new(),
        ));
        records.push(DiscoverRecord::new(
            obs_index,
            "output-0".to_string(),
            Some(0.0),
            0.0,
            vec![1.0], // Error = 1 (output should be higher)
        ));
    }

    // Pattern 10: Input-0=1, Input-1=0, output should be 1
    for i in 0..samples_per_pattern {
        let obs_index = 2 * samples_per_pattern + i;
        records.push(DiscoverRecord::new(
            obs_index,
            "input-0".to_string(),
            Some(1.0),
            1.0,
            Vec::new(),
        ));
        records.push(DiscoverRecord::new(
            obs_index,
            "input-1".to_string(),
            Some(0.0),
            0.0,
            Vec::new(),
        ));
        records.push(DiscoverRecord::new(
            obs_index,
            "output-0".to_string(),
            Some(0.0),
            0.0,
            vec![1.0], // Error = 1 (output should be higher)
        ));
    }

    // Pattern 11: Both inputs 1, output should be 0
    for i in 0..samples_per_pattern {
        let obs_index = 3 * samples_per_pattern + i;
        records.push(DiscoverRecord::new(
            obs_index,
            "input-0".to_string(),
            Some(1.0),
            1.0,
            Vec::new(),
        ));
        records.push(DiscoverRecord::new(
            obs_index,
            "input-1".to_string(),
            Some(1.0),
            1.0,
            Vec::new(),
        ));
        records.push(DiscoverRecord::new(
            obs_index,
            "output-0".to_string(),
            Some(0.0),
            0.0,
            vec![0.0], // No error - output is correct
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

    // Look for synergistic candidates in coordinatedStructuralCandidates
    let coordinated = output["coordinatedStructuralCandidates"]
        .as_array()
        .cloned()
        .unwrap_or_default();

    // We expect to find a synergistic candidate that combines both inputs
    // The comment should mention "synergistic" or "residual"
    let found_synergistic = coordinated.iter().any(|c| {
        let Some(ops) = c["operations"].as_array() else {
            return false;
        };

        // Check if this candidate involves both inputs
        let has_input_0 = ops
            .iter()
            .any(|op| op["type"] == "addSynapse" && op["fromNeuronUuid"] == "input-0");
        let has_input_1 = ops
            .iter()
            .any(|op| op["type"] == "addSynapse" && op["fromNeuronUuid"] == "input-1");

        // Check comment mentions synergistic relationship
        let is_synergistic = c.get("comment").and_then(|v| v.as_str()).is_some_and(|s| {
            s.to_lowercase().contains("synergistic")
                || s.to_lowercase().contains("residual")
                || s.to_lowercase().contains("combined")
        });

        has_input_0 && has_input_1 && is_synergistic
    });

    assert!(
        found_synergistic,
        "Expected to find a synergistic candidate for XOR pattern, but none found.\n\
         Individual inputs have weak/zero correlation, but together they explain the error.\n\
         Coordinated candidates: {coordinated:?}"
    );
}

/// Test: Residual analysis finds second-best source after applying first.
///
/// This tests the core residual analysis algorithm:
/// 1. Source A partially explains the error
/// 2. After applying A, residual error remains
/// 3. Source B explains the residual
/// 4. Combined (A + B) is better than A alone
#[test]
fn residual_analysis_finds_complementary_sources() {
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

    // Create samples where:
    // - Error = input-0 * 0.6 + input-1 * 0.4 (linear combination)
    // - Input-0 alone explains 60% of variance
    // - Input-1 alone explains 40% of variance
    // - Together they explain 100%
    let mut records: Vec<DiscoverRecord> = Vec::new();
    let sample_count = 256u32;

    for obs_index in 0..sample_count {
        // Create varied inputs
        let input_0 = ((obs_index as f32 / sample_count as f32) * 2.0 - 1.0) * 0.8;
        let input_1 =
            ((obs_index * 7 % sample_count) as f32 / sample_count as f32 * 2.0 - 1.0) * 0.8;

        // Error is a combination of both inputs
        let error = input_0 * 0.6 + input_1 * 0.4;

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

    // Both individual synapses should be helpful
    let helpful = output["helpfulSynapses"]
        .as_array()
        .expect("should have helpfulSynapses");

    let has_input_0 = helpful.iter().any(|s| s["fromNeuronUuid"] == "input-0");
    let has_input_1 = helpful.iter().any(|s| s["fromNeuronUuid"] == "input-1");

    assert!(has_input_0, "Input-0 should be a helpful synapse candidate");
    assert!(has_input_1, "Input-1 should be a helpful synapse candidate");

    // The residual analysis should also find a synergistic candidate
    // where combined improvement > max(individual improvements)
    let coordinated = output["coordinatedStructuralCandidates"]
        .as_array()
        .cloned()
        .unwrap_or_default();

    let found_synergistic = coordinated.iter().any(|c| {
        let Some(ops) = c["operations"].as_array() else {
            return false;
        };

        let has_input_0 = ops.iter().any(|op| op["fromNeuronUuid"] == "input-0");
        let has_input_1 = ops.iter().any(|op| op["fromNeuronUuid"] == "input-1");

        has_input_0 && has_input_1
    });

    // For this linear combination case, we should detect synergy
    assert!(
        found_synergistic,
        "Expected synergistic candidate combining both inputs for residual error reduction.\n\
         Coordinated candidates: {coordinated:?}"
    );
}

/// Test: Interference cancellation pattern detection.
///
/// Two noisy inputs that cancel each other's noise when combined.
/// - Input A has noise pattern P
/// - Input B has noise pattern -P (anti-correlated with A's noise)
/// - A + B cancels the noise, leaving clean signal
#[test]
fn synergistic_discovery_detects_interference_cancellation() {
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

    // Create samples where inputs have anti-correlated noise
    // Combined: error = (input-0 + input-1) / 2 reduces variance
    let mut records: Vec<DiscoverRecord> = Vec::new();
    let sample_count = 256u32;

    for obs_index in 0..sample_count {
        // Generate base signal
        let signal = ((obs_index as f32 / sample_count as f32) * 2.0 - 1.0) * 0.5;

        // Add anti-correlated noise
        let noise = if obs_index % 2 == 0 { 0.3 } else { -0.3 };
        let input_0 = signal + noise;
        let input_1 = signal - noise; // Anti-correlated noise

        // Error needs both inputs to cancel noise
        // If we only add one input, noise remains
        // If we add both with equal weights, noise cancels
        let error = signal; // The true error is just the signal (noise should cancel)

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

    // Check for synergistic candidate (interference cancellation)
    let coordinated = output["coordinatedStructuralCandidates"]
        .as_array()
        .cloned()
        .unwrap_or_default();

    let found_interference_cancellation = coordinated.iter().any(|c| {
        let Some(ops) = c["operations"].as_array() else {
            return false;
        };

        let has_input_0 = ops.iter().any(|op| op["fromNeuronUuid"] == "input-0");
        let has_input_1 = ops.iter().any(|op| op["fromNeuronUuid"] == "input-1");

        has_input_0 && has_input_1
    });

    assert!(
        found_interference_cancellation,
        "Expected synergistic candidate for interference cancellation pattern.\n\
         Coordinated candidates: {coordinated:?}"
    );
}

/// Test: Synergistic candidate structure contains expected fields.
#[test]
fn synergistic_candidate_has_expected_structure() {
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

    // Create complementary pattern
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

    let coordinated = output["coordinatedStructuralCandidates"]
        .as_array()
        .cloned()
        .unwrap_or_default();

    // Find a candidate that involves both inputs
    let synergistic_candidate = coordinated.iter().find(|c| {
        let Some(ops) = c["operations"].as_array() else {
            return false;
        };
        let has_input_0 = ops.iter().any(|op| op["fromNeuronUuid"] == "input-0");
        let has_input_1 = ops.iter().any(|op| op["fromNeuronUuid"] == "input-1");
        has_input_0 && has_input_1
    });

    assert!(
        synergistic_candidate.is_some(),
        "Expected to find a synergistic candidate"
    );

    let candidate = synergistic_candidate.unwrap();

    // Verify required fields
    assert!(
        candidate.get("operations").is_some(),
        "Should have operations array"
    );
    assert!(
        candidate.get("expectedCreatureScoreGain").is_some(),
        "Should have expectedCreatureScoreGain"
    );

    // Verify combined improvement is positive
    let gain = candidate["expectedCreatureScoreGain"]
        .as_f64()
        .expect("expectedCreatureScoreGain should be a number");
    assert!(
        gain > 0.0,
        "Expected positive score gain for synergistic candidate, got {gain}"
    );
}

/// Test: No false positives when inputs are truly independent.
///
/// When inputs have completely independent effects (no synergy),
/// they should NOT be grouped into synergistic candidates.
#[test]
fn no_false_positives_for_independent_inputs() {
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

    // Create samples where inputs are truly independent
    // Each input independently correlates with error, no synergy
    let mut records: Vec<DiscoverRecord> = Vec::new();
    let sample_count = 128u32;

    for obs_index in 0..sample_count {
        // Both inputs fire together (complete overlap)
        let activation = if obs_index % 2 == 0 { 1.0 } else { -1.0 };
        let error = activation * 0.5; // Direct correlation

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

    // Both should be helpful individually
    let helpful = output["helpfulSynapses"]
        .as_array()
        .expect("should have helpfulSynapses");

    assert!(
        !helpful.is_empty(),
        "At least one helpful synapse should be found"
    );

    // Check coordinated candidates - should NOT have synergistic pairs
    // for completely overlapping inputs (no synergy benefit)
    let coordinated = output["coordinatedStructuralCandidates"]
        .as_array()
        .cloned()
        .unwrap_or_default();

    // Filter for true synergistic candidates (not other types like epistatic)
    let synergistic_count = coordinated
        .iter()
        .filter(|c| {
            c.get("comment")
                .and_then(|v| v.as_str())
                .is_some_and(|s| s.to_lowercase().contains("synergistic"))
        })
        .count();

    // With complete overlap, synergistic detection should not find benefit
    // (adding both is no better than adding one with double weight)
    assert_eq!(
        synergistic_count, 0,
        "Should not find synergistic candidates for completely overlapping inputs"
    );
}
