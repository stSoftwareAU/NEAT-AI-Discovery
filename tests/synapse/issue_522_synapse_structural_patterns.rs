//! Issue #522: Targeted tests for synapse `structural_patterns` sub-module
//!
//! Tests the coordinated structural discovery functions in
//! `src/analysis/synapse/structural_patterns.rs`:
//! - `detect_noisy_vs_trusted` — noisy input folding with varied topologies
//! - `detect_collapsible_hidden_neurons` — 1-in/1-out chain collapse
//!
//! These tests exercise the full analysis pipeline with crafted creatures and
//! verify correct structural candidate generation.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use crate::common::CoordinatedNoiseFloorRelaxGuard;
use neat_ai_discovery::analysis::GpuAnalyzer;
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson, analyze_parallel_internal};
use serial_test::serial;

/// Helper: run analysis and return parsed JSON output.
fn run_analysis(
    creature: &CreatureJson,
    parquet_path: &str,
    focus: &[&str],
    max_synapse: usize,
) -> serde_json::Value {
    let focus_vec: Vec<String> = focus.iter().map(std::string::ToString::to_string).collect();
    let input_json = serde_json::json!({
        "parquetFile": parquet_path,
        "creature": creature,
        "focusNeurons": focus_vec,
        "maxSynapseCandidates": max_synapse,
        "maxNeuronCandidates": 0,
        "randomSeed": 42
    })
    .to_string();

    let output_json =
        analyze_parallel_internal(&input_json).expect("parallel analysis should return JSON");
    serde_json::from_str(&output_json).expect("output should be valid JSON")
}

/// Helper: extract coordinated structural candidates from JSON output.
fn get_coordinated_candidates(output: &serde_json::Value) -> Vec<serde_json::Value> {
    output["coordinatedStructuralCandidates"]
        .as_array()
        .cloned()
        .unwrap_or_default()
}

/// Helper: check if a JSON candidate group contains an operation matching the predicate.
fn has_operation(
    group: &serde_json::Value,
    predicate: impl Fn(&serde_json::Value) -> bool,
) -> bool {
    group["operations"]
        .as_array()
        .is_some_and(|ops| ops.iter().any(&predicate))
}

// =============================================================================
// Noisy vs Trusted — requires identical weights, identical means, variance ratio >= 10
// =============================================================================

/// When two inputs have identical weights and means but one has much higher
/// variance, the pipeline should generate a coordinated candidate that removes
/// both existing synapses and re-adds only the trusted input.
#[test]
fn noisy_vs_trusted_generates_coordinated_prune_candidate() {
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping test: no GPU available");
        return;
    }

    let temp_dir = tempfile::tempdir().unwrap();
    let parquet_path = temp_dir
        .path()
        .join("records.parquet")
        .to_str()
        .unwrap()
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
                from_uuid: "input-0".to_string(), // trusted (low variance)
                to_uuid: "output-0".to_string(),
                weight: 0.3,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-1".to_string(), // noisy (high variance)
                to_uuid: "output-0".to_string(),
                weight: 0.3,
                synapse_type: None,
            },
        ],
    };

    let mut records = Vec::new();
    for obs in 0..200u32 {
        // Trusted: low amplitude oscillation (variance ~0.01)
        let trusted = if obs % 2 == 0 { 0.05 } else { -0.05 };
        // Noisy: high amplitude oscillation (variance ~1.0, ratio > 10)
        let noisy = if obs % 2 == 0 { 1.0 } else { -1.0 };

        let current = 0.3 * trusted + 0.3 * noisy;
        let error = -current; // desired output is 0

        records.push(DiscoverRecord::new(
            obs,
            "input-0".to_string(),
            Some(trusted),
            trusted,
            Vec::new(),
        ));
        records.push(DiscoverRecord::new(
            obs,
            "input-1".to_string(),
            Some(noisy),
            noisy,
            Vec::new(),
        ));
        records.push(DiscoverRecord::new(
            obs,
            "output-0".to_string(),
            Some(current),
            current,
            vec![error],
        ));
    }
    write_records_to_parquet(&parquet_path, &records).unwrap();

    let output = run_analysis(&creature, &parquet_path, &["output-0"], 64);
    assert_eq!(output["success"], true);

    let groups = get_coordinated_candidates(&output);

    // Find a group that removes input-1 (noisy) and re-adds input-0 (trusted)
    let found = groups.iter().any(|g| {
        let removes_noisy = has_operation(g, |op| {
            op["type"] == "removeSynapse"
                && op["fromNeuronUuid"] == "input-1"
                && op["toNeuronUuid"] == "output-0"
        });
        let adds_trusted = has_operation(g, |op| {
            op["type"] == "addSynapse"
                && op["fromNeuronUuid"] == "input-0"
                && op["toNeuronUuid"] == "output-0"
        });
        removes_noisy && adds_trusted
    });

    assert!(
        found,
        "Expected a coordinated candidate pruning noisy input-1 and re-adding trusted input-0. \
        Found candidates: {groups:?}"
    );
}

/// When inputs have different weights, the noisy-vs-trusted pattern should
/// NOT fire (strict weight matching requirement).
#[test]
fn noisy_vs_trusted_skipped_when_weights_differ() {
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping test: no GPU available");
        return;
    }

    let temp_dir = tempfile::tempdir().unwrap();
    let parquet_path = temp_dir
        .path()
        .join("records.parquet")
        .to_str()
        .unwrap()
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
                weight: 0.3, // Different weight
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-1".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.7, // Different weight — should prevent matching
                synapse_type: None,
            },
        ],
    };

    let mut records = Vec::new();
    for obs in 0..200u32 {
        let trusted = if obs % 2 == 0 { 0.05 } else { -0.05 };
        let noisy = if obs % 2 == 0 { 1.0 } else { -1.0 };

        let current = 0.3 * trusted + 0.7 * noisy;
        let error = -current;

        records.push(DiscoverRecord::new(
            obs,
            "input-0".to_string(),
            Some(trusted),
            trusted,
            Vec::new(),
        ));
        records.push(DiscoverRecord::new(
            obs,
            "input-1".to_string(),
            Some(noisy),
            noisy,
            Vec::new(),
        ));
        records.push(DiscoverRecord::new(
            obs,
            "output-0".to_string(),
            Some(current),
            current,
            vec![error],
        ));
    }
    write_records_to_parquet(&parquet_path, &records).unwrap();

    let output = run_analysis(&creature, &parquet_path, &["output-0"], 64);
    assert_eq!(output["success"], true);

    let groups = get_coordinated_candidates(&output);

    // The noisy-vs-trusted pattern specifically requires identical weights.
    // We should NOT find a candidate that removes both input-0 and input-1
    // and re-adds only input-0 (which is the noisy-vs-trusted pattern).
    let noisy_trusted_found = groups.iter().any(|g| {
        let ops = g["operations"].as_array().cloned().unwrap_or_default();
        let removes_input0 = ops.iter().any(|op| {
            op["type"] == "removeSynapse"
                && op["fromNeuronUuid"] == "input-0"
                && op["toNeuronUuid"] == "output-0"
        });
        let removes_input1 = ops.iter().any(|op| {
            op["type"] == "removeSynapse"
                && op["fromNeuronUuid"] == "input-1"
                && op["toNeuronUuid"] == "output-0"
        });
        // Noisy-vs-trusted pattern: remove both + re-add one
        removes_input0 && removes_input1 && ops.len() == 3
    });

    assert!(
        !noisy_trusted_found,
        "Should NOT find noisy-vs-trusted pattern when weights differ"
    );
}

// =============================================================================
// Collapse Hidden Neuron — 1-in/1-out chain to direct synapse
// =============================================================================

/// A hidden neuron with exactly 1 incoming and 1 outgoing synapse should be
/// proposed for collapse into a direct synapse.
#[test]
#[serial]
fn collapse_hidden_neuron_with_identity_squash() {
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping test: no GPU available");
        return;
    }

    // Issue #1272: the collapse candidate is a 4-op coordinated group whose
    // post-calibration gain (~9.95e-7) sits just below the new 4+-op noise
    // floor (5e-6). Relax the floor for this contract test so the collapse
    // proposal is observable; production callers leave the env var unset.
    let _noise_floor_guard = CoordinatedNoiseFloorRelaxGuard::new();

    let temp_dir = tempfile::tempdir().unwrap();
    let parquet_path = temp_dir
        .path()
        .join("records.parquet")
        .to_str()
        .unwrap()
        .to_string();

    // Chain: input-0 → hidden-0 → output-0
    // Hidden neuron is a pure passthrough (IDENTITY, bias=0) with under-weight
    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![
            NeuronJson {
                uuid: "hidden-0".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "output-0".to_string(),
                neuron_type: "output".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
        ],
        synapses: vec![
            SynapseJson {
                from_uuid: "input-0".to_string(),
                to_uuid: "hidden-0".to_string(),
                weight: 1.0,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "hidden-0".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.5,
                synapse_type: None,
            },
        ],
    };

    let mut records = Vec::new();
    for obs in 0..100u32 {
        let x = (obs as f32 / 50.0) - 1.0; // [-1, 1]
        let hidden_act = x; // IDENTITY passthrough
        let current_output = 0.5 * x;
        let desired = x;
        let error = desired - current_output;

        records.push(DiscoverRecord::new(
            obs,
            "input-0".to_string(),
            Some(x),
            x,
            Vec::new(),
        ));
        records.push(DiscoverRecord::new(
            obs,
            "hidden-0".to_string(),
            Some(hidden_act),
            hidden_act,
            Vec::new(),
        ));
        records.push(DiscoverRecord::new(
            obs,
            "output-0".to_string(),
            Some(current_output),
            current_output,
            vec![error],
        ));
    }
    write_records_to_parquet(&parquet_path, &records).unwrap();

    let output = run_analysis(&creature, &parquet_path, &["output-0"], 64);
    assert_eq!(output["success"], true);

    let groups = get_coordinated_candidates(&output);

    // Find a group that removes hidden-0 and adds input-0 → output-0
    let found = groups.iter().any(|g| {
        let removes_neuron = has_operation(g, |op| {
            op["type"] == "removeNeuron" && op["neuronUuid"] == "hidden-0"
        });
        let adds_bypass = has_operation(g, |op| {
            op["type"] == "addSynapse"
                && op["fromNeuronUuid"] == "input-0"
                && op["toNeuronUuid"] == "output-0"
        });
        removes_neuron && adds_bypass
    });

    assert!(
        found,
        "Expected collapse candidate removing hidden-0 and adding bypass synapse. \
        Found candidates: {groups:?}"
    );
}

/// A hidden neuron with 2 incoming synapses should NOT be proposed for collapse.
#[test]
fn no_collapse_when_hidden_has_multiple_incoming() {
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping test: no GPU available");
        return;
    }

    let temp_dir = tempfile::tempdir().unwrap();
    let parquet_path = temp_dir
        .path()
        .join("records.parquet")
        .to_str()
        .unwrap()
        .to_string();

    // Two inputs feed hidden-0, which feeds output-0.
    // hidden-0 has 2 incoming synapses → NOT collapsible.
    let creature = CreatureJson {
        input: 2,
        output: 1,
        neurons: vec![
            NeuronJson {
                uuid: "hidden-0".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "output-0".to_string(),
                neuron_type: "output".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
        ],
        synapses: vec![
            SynapseJson {
                from_uuid: "input-0".to_string(),
                to_uuid: "hidden-0".to_string(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-1".to_string(),
                to_uuid: "hidden-0".to_string(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "hidden-0".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.5,
                synapse_type: None,
            },
        ],
    };

    let mut records = Vec::new();
    for obs in 0..100u32 {
        let x0 = (obs as f32 / 50.0) - 1.0;
        let x1 = ((obs as f32) * 1.7).sin();
        let hidden_act = 0.5 * x0 + 0.5 * x1;
        let current_output = 0.5 * hidden_act;
        let error = hidden_act - current_output; // desired = hidden_act

        records.push(DiscoverRecord::new(
            obs,
            "input-0".to_string(),
            Some(x0),
            x0,
            Vec::new(),
        ));
        records.push(DiscoverRecord::new(
            obs,
            "input-1".to_string(),
            Some(x1),
            x1,
            Vec::new(),
        ));
        records.push(DiscoverRecord::new(
            obs,
            "hidden-0".to_string(),
            Some(hidden_act),
            hidden_act,
            Vec::new(),
        ));
        records.push(DiscoverRecord::new(
            obs,
            "output-0".to_string(),
            Some(current_output),
            current_output,
            vec![error],
        ));
    }
    write_records_to_parquet(&parquet_path, &records).unwrap();

    let output = run_analysis(&creature, &parquet_path, &["output-0"], 64);
    assert_eq!(output["success"], true);

    let groups = get_coordinated_candidates(&output);

    // No collapse candidate should exist for hidden-0 (2 incoming synapses)
    let collapse_found = groups.iter().any(|g| {
        has_operation(g, |op| {
            op["type"] == "removeNeuron" && op["neuronUuid"] == "hidden-0"
        })
    });

    assert!(
        !collapse_found,
        "Should NOT propose collapse for hidden neuron with multiple incoming synapses"
    );
}

/// A hidden neuron should not be collapsed if a direct synapse already exists
/// between the source and target.
#[test]
fn no_collapse_when_bypass_synapse_already_exists() {
    if !GpuAnalyzer::gpu_is_available() {
        eprintln!("Skipping test: no GPU available");
        return;
    }

    let temp_dir = tempfile::tempdir().unwrap();
    let parquet_path = temp_dir
        .path()
        .join("records.parquet")
        .to_str()
        .unwrap()
        .to_string();

    // Chain: input-0 → hidden-0 → output-0
    // But also: input-0 → output-0 (direct synapse already exists)
    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![
            NeuronJson {
                uuid: "hidden-0".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "output-0".to_string(),
                neuron_type: "output".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
        ],
        synapses: vec![
            SynapseJson {
                from_uuid: "input-0".to_string(),
                to_uuid: "hidden-0".to_string(),
                weight: 1.0,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "hidden-0".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-0".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.3, // Bypass already exists
                synapse_type: None,
            },
        ],
    };

    let mut records = Vec::new();
    for obs in 0..100u32 {
        let x = (obs as f32 / 50.0) - 1.0;
        let hidden_act = x;
        let current_output = 0.5 * x + 0.3 * x;
        let error = x - current_output;

        records.push(DiscoverRecord::new(
            obs,
            "input-0".to_string(),
            Some(x),
            x,
            Vec::new(),
        ));
        records.push(DiscoverRecord::new(
            obs,
            "hidden-0".to_string(),
            Some(hidden_act),
            hidden_act,
            Vec::new(),
        ));
        records.push(DiscoverRecord::new(
            obs,
            "output-0".to_string(),
            Some(current_output),
            current_output,
            vec![error],
        ));
    }
    write_records_to_parquet(&parquet_path, &records).unwrap();

    let output = run_analysis(&creature, &parquet_path, &["output-0"], 64);
    assert_eq!(output["success"], true);

    let groups = get_coordinated_candidates(&output);

    // No collapse candidate should include removeNeuron for hidden-0
    // because a direct synapse input-0 → output-0 already exists
    let collapse_found = groups.iter().any(|g| {
        has_operation(g, |op| {
            op["type"] == "removeNeuron" && op["neuronUuid"] == "hidden-0"
        })
    });

    assert!(
        !collapse_found,
        "Should NOT propose collapse when bypass synapse already exists"
    );
}
