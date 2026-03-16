//! Issue #164: Redundant path pruning with renormalisation.
//!
//! Detects that two subnetworks feeding the same output compute the same thing,
//! and proposes pruning one while scaling the survivor's weight to compensate.
//!
//! ## Test Strategy
//! 1. Create a creature with two redundant paths (highly correlated activations) to the same output
//! 2. Verify the discovery system detects the redundancy
//! 3. Verify the candidate prunes the weaker path and renormalises the survivor
//! 4. Verify no false positives for independent (uncorrelated) paths
//!
//! ## Discovery type
//! `COORDINATED_PRUNE_AND_REWEIGHT` – a coordinated structural candidate with:
//! - `removeSynapse` for the pruned path
//! - `setWeight` for the survivor (renormalised weight)

use neat_ai_discovery::analysis::GpuAnalyzer;
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson, analyze_parallel_internal};
use tempfile::tempdir;

/// Test: Two sources with identical activation patterns feeding the same output
/// should be detected as redundant paths.
///
/// Creature topology:
/// ```text
///   input-0 ──(w=0.5)──→ output-0
///   input-1 ──(w=0.3)──→ output-0
/// ```
///
/// Both inputs produce identical activations (perfectly correlated).
/// Expected: prune the weaker path (input-1, w=0.3) and set input-0's weight to 0.8.
#[test]
fn redundant_identical_paths_detected() {
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

    // Creature with two synapses feeding the same output
    let creature = CreatureJson {
        input: 2,
        output: 1,
        neurons: vec![NeuronJson {
            uuid: "output-0".to_string(),
            neuron_type: "output".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        }],
        synapses: vec![
            SynapseJson {
                from_uuid: "input-0".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-1".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.3,
                synapse_type: None,
            },
        ],
    };

    // Create samples where both inputs have identical activation patterns.
    // The output error is moderate so the system has something to optimise.
    let mut records: Vec<DiscoverRecord> = Vec::new();
    let sample_count = 256u32;

    for obs_index in 0..sample_count {
        // Both inputs produce the same activation value
        let activation = (obs_index as f32 / sample_count as f32) * 2.0 - 1.0;

        // The combined contribution: 0.5 * act + 0.3 * act = 0.8 * act
        // After renormalisation (one path with w=0.8), the same output is achieved.
        let combined = 0.8 * activation;
        let target_error = 0.5 - combined; // Some residual error

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
            Some(combined),
            combined,
            vec![target_error],
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

    // Look for redundant path pruning candidates in coordinatedStructuralCandidates
    let coordinated = output["coordinatedStructuralCandidates"]
        .as_array()
        .cloned()
        .unwrap_or_default();

    let found_redundant = coordinated.iter().any(|c| {
        let comment = c
            .get("comment")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_lowercase();

        // Check comment mentions redundant path pruning
        let is_redundant =
            comment.contains("redundant") || comment.contains("164") || comment.contains("prune");

        if !is_redundant {
            return false;
        }

        let Some(ops) = c["operations"].as_array() else {
            return false;
        };

        // Should have a removeSynapse and a setWeight operation
        let has_remove = ops.iter().any(|op| op["type"] == "removeSynapse");
        let has_set_weight = ops.iter().any(|op| op["type"] == "setWeight");

        has_remove && has_set_weight
    });

    assert!(
        found_redundant,
        "Expected to find a redundant path pruning candidate for identical input paths.\n\
         Coordinated candidates: {coordinated:?}"
    );
}

/// Test: Two sources with independent activation patterns should NOT be detected as redundant.
///
/// Creature topology:
/// ```text
///   input-0 ──(w=0.5)──→ output-0   (activation varies linearly)
///   input-1 ──(w=0.3)──→ output-0   (activation varies independently)
/// ```
#[test]
fn independent_paths_not_detected_as_redundant() {
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
        synapses: vec![
            SynapseJson {
                from_uuid: "input-0".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-1".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.3,
                synapse_type: None,
            },
        ],
    };

    // Create samples where inputs have independent (uncorrelated) activation patterns
    let mut records: Vec<DiscoverRecord> = Vec::new();
    let sample_count = 256u32;

    for obs_index in 0..sample_count {
        // Input-0: linear ramp
        let activation_0 = (obs_index as f32 / sample_count as f32) * 2.0 - 1.0;
        // Input-1: scrambled/independent pattern
        let activation_1 =
            ((obs_index * 97 + 13) % sample_count) as f32 / sample_count as f32 * 2.0 - 1.0;

        let combined = 0.5 * activation_0 + 0.3 * activation_1;
        let target_error = 0.5 - combined;

        records.push(DiscoverRecord::new(
            obs_index,
            "input-0".to_string(),
            Some(activation_0),
            activation_0,
            Vec::new(),
        ));
        records.push(DiscoverRecord::new(
            obs_index,
            "input-1".to_string(),
            Some(activation_1),
            activation_1,
            Vec::new(),
        ));
        records.push(DiscoverRecord::new(
            obs_index,
            "output-0".to_string(),
            Some(combined),
            combined,
            vec![target_error],
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

    // Check coordinated candidates - should NOT have redundant path pruning
    let coordinated = output["coordinatedStructuralCandidates"]
        .as_array()
        .cloned()
        .unwrap_or_default();

    let redundant_count = coordinated
        .iter()
        .filter(|c| {
            c.get("comment").and_then(|v| v.as_str()).is_some_and(|s| {
                let lower = s.to_lowercase();
                lower.contains("redundant") || lower.contains("164")
            })
        })
        .count();

    assert_eq!(
        redundant_count, 0,
        "Should not find redundant path candidates for independent input paths.\n\
         Coordinated candidates: {coordinated:?}"
    );
}

/// Test: Redundant path candidate structure contains expected fields.
///
/// The COORDINATED_PRUNE_AND_REWEIGHT candidate should have:
/// - operations: [removeSynapse, setWeight]
/// - expectedCreatureScoreGain > 0
/// - comment referencing redundant path / Issue #164
#[test]
fn redundant_path_candidate_has_expected_structure() {
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
        synapses: vec![
            SynapseJson {
                from_uuid: "input-0".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.6,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-1".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.4,
                synapse_type: None,
            },
        ],
    };

    // Identical activations → redundant paths
    let mut records: Vec<DiscoverRecord> = Vec::new();
    let sample_count = 256u32;

    for obs_index in 0..sample_count {
        let activation = (obs_index as f32 / sample_count as f32) * 2.0 - 1.0;
        let combined = 1.0 * activation; // 0.6 + 0.4
        let target_error = 0.3 - combined;

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
            Some(combined),
            combined,
            vec![target_error],
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

    // Find the redundant path candidate
    let redundant_candidate = coordinated.iter().find(|c| {
        c.get("comment").and_then(|v| v.as_str()).is_some_and(|s| {
            let lower = s.to_lowercase();
            lower.contains("redundant") || lower.contains("164")
        })
    });

    assert!(
        redundant_candidate.is_some(),
        "Expected to find a redundant path pruning candidate.\n\
         Coordinated candidates: {coordinated:?}"
    );

    let candidate = redundant_candidate.unwrap();

    // Verify required fields
    assert!(
        candidate.get("operations").is_some(),
        "Should have operations array"
    );
    assert!(
        candidate.get("expectedCreatureScoreGain").is_some(),
        "Should have expectedCreatureScoreGain"
    );

    let ops = candidate["operations"]
        .as_array()
        .expect("operations should be an array");
    assert_eq!(ops.len(), 2, "Should have exactly 2 operations");

    // First op: removeSynapse (prune the weaker path)
    assert_eq!(
        ops[0]["type"], "removeSynapse",
        "First operation should be removeSynapse"
    );

    // Second op: setWeight (renormalise the survivor)
    assert_eq!(
        ops[1]["type"], "setWeight",
        "Second operation should be setWeight"
    );

    // Verify positive score gain
    let gain = candidate["expectedCreatureScoreGain"]
        .as_f64()
        .expect("expectedCreatureScoreGain should be a number");
    assert!(
        gain > 0.0,
        "Expected positive score gain for redundant path pruning, got {gain}"
    );
}
