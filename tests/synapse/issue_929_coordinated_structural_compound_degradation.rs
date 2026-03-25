//! Scenario test: coordinated-structural discovery for compound degradations (Issue #929).
//!
//! Topology (whole creature):
//! ```text
//! input-0 ──(1.0)──▶ hidden-A (TANH, bias 0.3) ──(1.0)──▶ hidden-C (RELU)
//! input-1 ──(1.0)──▶ hidden-B (RELU, bias -0.1) ──(0.8)──▶ hidden-C
//! hidden-C ──(1.0)──▶ output-0 (IDENTITY)
//! ```
//!
//! Crippled creature (coordinated degradation):
//! - hidden-A bias changed: 0.3 → 0.0
//! - hidden-B → hidden-C weight changed: 0.8 → 0.05 (nearly disabled)
//!
//! Same topology, degraded parameters.
//!
//! Discovery should produce a coordinated-structural candidate with both
//! `setBias` and `setWeight` operations restoring the degraded parameters.

#![allow(clippy::cast_precision_loss)]

use neat_ai_discovery::analysis::GpuAnalyzer;
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson, analyze_parallel_internal};

/// Build the crippled creature with both degradations applied.
fn crippled_creature() -> CreatureJson {
    CreatureJson {
        input: 2,
        output: 1,
        neurons: vec![
            NeuronJson {
                uuid: "hidden-A".into(),
                neuron_type: "hidden".into(),
                squash: "TANH".into(),
                bias: 0.0, // Degraded: should be 0.3
            },
            NeuronJson {
                uuid: "hidden-B".into(),
                neuron_type: "hidden".into(),
                squash: "ReLU".into(),
                bias: -0.1,
            },
            NeuronJson {
                uuid: "hidden-C".into(),
                neuron_type: "hidden".into(),
                squash: "ReLU".into(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "output-0".into(),
                neuron_type: "output".into(),
                squash: "IDENTITY".into(),
                bias: 0.0,
            },
        ],
        synapses: vec![
            SynapseJson {
                from_uuid: "input-0".into(),
                to_uuid: "hidden-A".into(),
                weight: 1.0,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-1".into(),
                to_uuid: "hidden-B".into(),
                weight: 1.0,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "hidden-A".into(),
                to_uuid: "hidden-C".into(),
                weight: 1.0,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "hidden-B".into(),
                to_uuid: "hidden-C".into(),
                weight: 0.05, // Degraded: should be 0.8
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "hidden-C".into(),
                to_uuid: "output-0".into(),
                weight: 1.0,
                synapse_type: None,
            },
        ],
    }
}

/// Generate discovery records from the crippled creature with error signals
/// derived from the whole (undegraded) creature's expected output.
fn generate_records(num_observations: u32) -> Vec<DiscoverRecord> {
    let mut records = Vec::new();

    for obs in 0..num_observations {
        let t = obs as f32 / num_observations as f32;
        let x0 = t * 2.0 - 1.0; // [-1, 1]
        let x1 = (t * 3.7).sin() * 0.8; // Varied pattern in [-0.8, 0.8]

        // --- hidden-A: TANH(input-0 * 1.0 + bias) ---
        let ha_pre_whole = x0 * 1.0 + 0.3; // Whole: bias = 0.3
        let ha_act_whole = ha_pre_whole.tanh();
        let ha_pre_crippled = x0 * 1.0 + 0.0; // Crippled: bias = 0.0
        let ha_act_crippled = ha_pre_crippled.tanh();

        // --- hidden-B: ReLU(input-1 * 1.0 + bias) ---
        let hb_pre = x1 * 1.0 + (-0.1); // Same in both
        let hb_act = hb_pre.max(0.0);

        // --- hidden-C: ReLU(hidden-A * 1.0 + hidden-B * weight) ---
        let hc_pre_whole = ha_act_whole * 1.0 + hb_act * 0.8; // Whole: weight = 0.8
        let hc_act_whole = hc_pre_whole.max(0.0);
        let hc_pre_crippled = ha_act_crippled * 1.0 + hb_act * 0.05; // Crippled: weight = 0.05
        let hc_act_crippled = hc_pre_crippled.max(0.0);

        // --- output-0: IDENTITY(hidden-C * 1.0) ---
        let out_whole = hc_act_whole * 1.0;
        let out_crippled = hc_act_crippled * 1.0;

        // Error = whole - crippled (what's missing due to degradation)
        let out_err = out_whole - out_crippled;
        let hc_err = hc_act_whole - hc_act_crippled;
        let ha_err = ha_act_whole - ha_act_crippled;
        // hidden-B itself is correct; small propagated error
        let hb_err = out_err * 0.05;

        records.push(DiscoverRecord::new(
            obs,
            "hidden-A".into(),
            Some(ha_pre_crippled),
            ha_act_crippled,
            vec![ha_err],
        ));
        records.push(DiscoverRecord::new(
            obs,
            "hidden-B".into(),
            Some(hb_pre),
            hb_act,
            vec![hb_err],
        ));
        records.push(DiscoverRecord::new(
            obs,
            "hidden-C".into(),
            Some(hc_pre_crippled),
            hc_act_crippled,
            vec![hc_err],
        ));
        records.push(DiscoverRecord::new(
            obs,
            "output-0".into(),
            Some(out_crippled),
            out_crippled,
            vec![out_err],
        ));
    }
    records
}

/// Run the full discovery pipeline and return parsed JSON output.
fn run_analysis(creature: &CreatureJson, records: &[DiscoverRecord]) -> serde_json::Value {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let parquet_file = temp_dir
        .path()
        .join("records.parquet")
        .to_str()
        .expect("valid UTF-8 path")
        .to_string();

    write_records_to_parquet(&parquet_file, records).expect("write parquet");

    let input_json = serde_json::json!({
        "parquetFile": parquet_file,
        "creature": creature,
        "focusNeurons": ["output-0", "hidden-C", "hidden-A"],
        "maxSynapseCandidates": 64,
        "maxNeuronCandidates": 32,
        "randomSeed": 42
    })
    .to_string();

    let output_json = analyze_parallel_internal(&input_json).expect("analysis should return JSON");
    let output: serde_json::Value =
        serde_json::from_str(&output_json).expect("output should be valid JSON");

    assert_eq!(
        output["success"],
        true,
        "Analysis failed: {}",
        output["error"].as_str().unwrap_or("unknown")
    );

    output
}

/// Check whether any coordinated structural candidate contains both a setBias
/// operation for `neuron_uuid` and a setWeight operation for (from, to).
fn find_compound_candidate(
    coordinated: &[serde_json::Value],
    bias_neuron: &str,
    weight_from: &str,
    weight_to: &str,
) -> Option<serde_json::Value> {
    for group in coordinated {
        let Some(ops) = group["operations"].as_array() else {
            continue;
        };

        let has_set_bias = ops.iter().any(|op| {
            op["type"].as_str() == Some("setBias") && op["neuronUuid"].as_str() == Some(bias_neuron)
        });

        let has_set_weight = ops.iter().any(|op| {
            op["type"].as_str() == Some("setWeight")
                && op["fromNeuronUuid"].as_str() == Some(weight_from)
                && op["toNeuronUuid"].as_str() == Some(weight_to)
        });

        if has_set_bias && has_set_weight {
            return Some(group.clone());
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Test: Discovery produces a coordinated-structural candidate with setBias + setWeight
// ---------------------------------------------------------------------------

/// The discovery engine should identify both degraded parameters (hidden-A bias
/// and hidden-B→hidden-C weight) and produce a coordinated-structural candidate
/// containing both setBias and setWeight operations.
#[test]
fn issue_929_discovers_compound_bias_weight_degradation() {
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping: no GPU available");
        return;
    }

    let creature = crippled_creature();
    let records = generate_records(128);
    let output = run_analysis(&creature, &records);

    let empty = vec![];
    let coordinated = output["coordinatedStructuralCandidates"]
        .as_array()
        .unwrap_or(&empty);

    let compound = find_compound_candidate(coordinated, "hidden-A", "hidden-B", "hidden-C");

    assert!(
        compound.is_some(),
        "Expected a coordinated-structural candidate with both setBias(hidden-A) \
         and setWeight(hidden-B → hidden-C). \
         coordinatedStructuralCandidates count: {}, \
         candidates: {:#?}",
        coordinated.len(),
        coordinated
    );

    let group = compound.unwrap();
    let ops = group["operations"].as_array().expect("operations array");

    // Verify the candidate has exactly 2 operations
    assert_eq!(
        ops.len(),
        2,
        "Expected exactly 2 operations (setBias + setWeight), got {}: {ops:#?}",
        ops.len()
    );

    // Verify positive expected score gain
    let score_gain = group["expectedCreatureScoreGain"]
        .as_f64()
        .expect("expectedCreatureScoreGain should be a number");
    assert!(
        score_gain > 0.0,
        "Expected positive expectedCreatureScoreGain, got {score_gain}"
    );

    eprintln!(
        "Issue #929 PASS: compound candidate found with {ops_count} operations, \
         scoreGain={score_gain:.6}",
        ops_count = ops.len()
    );
}

// ---------------------------------------------------------------------------
// Test: The setBias operation targets the correct neuron with reasonable bias
// ---------------------------------------------------------------------------

/// Verify that the setBias operation in the compound candidate correctly
/// identifies hidden-A and recommends a bias approximately restoring 0.3.
#[test]
fn issue_929_set_bias_targets_correct_neuron() {
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping: no GPU available");
        return;
    }

    let creature = crippled_creature();
    let records = generate_records(128);
    let output = run_analysis(&creature, &records);

    let empty = vec![];
    let coordinated = output["coordinatedStructuralCandidates"]
        .as_array()
        .unwrap_or(&empty);

    let compound = find_compound_candidate(coordinated, "hidden-A", "hidden-B", "hidden-C");

    assert!(
        compound.is_some(),
        "Expected compound candidate with setBias(hidden-A) + setWeight(hidden-B → hidden-C)"
    );

    let group = compound.unwrap();
    let ops = group["operations"].as_array().unwrap();

    let bias_op = ops
        .iter()
        .find(|op| {
            op["type"].as_str() == Some("setBias") && op["neuronUuid"].as_str() == Some("hidden-A")
        })
        .expect("setBias operation for hidden-A");

    let bias = bias_op["bias"].as_f64().expect("bias should be a number");

    // The recommended bias should be positive (moving toward 0.3 from 0.0)
    assert!(
        bias > 0.0,
        "Expected positive bias correction for hidden-A (toward 0.3), got {bias}"
    );

    eprintln!("Issue #929: setBias(hidden-A) bias={bias:.4} (target ≈ 0.3)");
}

// ---------------------------------------------------------------------------
// Test: The setWeight operation targets the correct synapse with reasonable weight
// ---------------------------------------------------------------------------

/// Verify that the setWeight operation in the compound candidate correctly
/// identifies the hidden-B → hidden-C synapse and recommends a weight
/// significantly higher than the degraded 0.05.
#[test]
fn issue_929_set_weight_targets_correct_synapse() {
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping: no GPU available");
        return;
    }

    let creature = crippled_creature();
    let records = generate_records(128);
    let output = run_analysis(&creature, &records);

    let empty = vec![];
    let coordinated = output["coordinatedStructuralCandidates"]
        .as_array()
        .unwrap_or(&empty);

    let compound = find_compound_candidate(coordinated, "hidden-A", "hidden-B", "hidden-C");

    assert!(
        compound.is_some(),
        "Expected compound candidate with setBias(hidden-A) + setWeight(hidden-B → hidden-C)"
    );

    let group = compound.unwrap();
    let ops = group["operations"].as_array().unwrap();

    let weight_op = ops
        .iter()
        .find(|op| {
            op["type"].as_str() == Some("setWeight")
                && op["fromNeuronUuid"].as_str() == Some("hidden-B")
                && op["toNeuronUuid"].as_str() == Some("hidden-C")
        })
        .expect("setWeight operation for hidden-B → hidden-C");

    let weight = weight_op["weight"]
        .as_f64()
        .expect("weight should be a number");

    // The recommended weight should be significantly higher than 0.05
    // (the degraded value), moving toward the original 0.8.
    assert!(
        weight > 0.1,
        "Expected weight significantly above degraded 0.05 (toward 0.8), got {weight}"
    );

    eprintln!("Issue #929: setWeight(hidden-B → hidden-C) weight={weight:.4} (target ≈ 0.8)");
}

// ---------------------------------------------------------------------------
// Test: All returned candidates have positive expected improvement
// ---------------------------------------------------------------------------

/// Verify that all candidates returned by the pipeline have positive expected
/// score gain.
#[test]
fn issue_929_all_candidates_have_positive_improvement() {
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping: no GPU available");
        return;
    }

    let creature = crippled_creature();
    let records = generate_records(128);
    let output = run_analysis(&creature, &records);

    let empty = vec![];

    // Check coordinated structural candidates
    let coordinated = output["coordinatedStructuralCandidates"]
        .as_array()
        .unwrap_or(&empty);
    for (i, group) in coordinated.iter().enumerate() {
        if let Some(gain) = group["expectedCreatureScoreGain"].as_f64() {
            assert!(
                gain > 0.0,
                "coordinatedStructural[{i}] expected positive gain, got {gain}"
            );
        }
    }

    // Check helpful synapses
    if let Some(helpful) = output["helpfulSynapses"].as_array() {
        for (i, c) in helpful.iter().enumerate() {
            if let Some(gain) = c["expectedCreatureScoreGain"].as_f64() {
                assert!(
                    gain > 0.0,
                    "helpfulSynapses[{i}] expected positive score gain, got {gain}"
                );
            }
        }
    }

    eprintln!(
        "Issue #929: all candidates have positive improvement \
         (coordinated={}, helpful={})",
        coordinated.len(),
        output["helpfulSynapses"]
            .as_array()
            .map_or(0, std::vec::Vec::len),
    );
}
