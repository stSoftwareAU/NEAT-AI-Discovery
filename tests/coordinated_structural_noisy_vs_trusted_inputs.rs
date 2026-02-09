use neat_ai_discovery::analysis::GpuAnalyzer;
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson, analyze_parallel_internal};

/// Regression/integration test for Issue #165 (3-Jan-2026): Coordinated Structural Discovery.
///
/// This verifies the "noisy vs trusted" pattern described in the issue:
/// - Two input signals feed the same target with the same starting weight.
/// - One input is much noisier (higher activation variance).
///
/// The expected coordinated fix is a grouped candidate that:
/// - removes the noisy synapse
/// - removes the trusted synapse
/// - adds the trusted synapse back with a higher weight
#[test]
fn coordinated_structural_can_prune_noisy_input_and_reweight_trusted_input() {
    // Discovery is GPU-only. On machines without GPU, we skip.
    if !GpuAnalyzer::gpu_is_available() {
        return;
    }

    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let parquet_file = temp_dir
        .path()
        .join("records.parquet")
        .to_str()
        .expect("temp path should be valid UTF-8")
        .to_string();

    // Two input synapses into one output with equal starting weights.
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
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-1".to_string(), // noisy (high variance)
                to_uuid: "output-0".to_string(),
                weight: 0.5,
                synapse_type: None,
            },
        ],
    };

    // Deterministic dataset:
    // - trusted signal has a small variance (low amplitude), mean ~0
    // - noisy signal has a much larger variance (high amplitude), mean ~0
    //
    // This is the strict "simple case" the coordinated-noise logic targets:
    // - same synapse weights
    // - same activation means
    // - large variance ratio (noisy/trusted >= 10×)
    //
    // Desired output is ~0.0 (mean-preserving). Current output is polluted by the noisy input,
    // so removing noisy and moving its weight onto the trusted input reduces error magnitude.
    let mut records: Vec<DiscoverRecord> = Vec::new();
    for obs_index in 0..128u32 {
        let trusted = if (obs_index % 2) == 0 { 0.1 } else { -0.1 };
        let noisy = if (obs_index % 2) == 0 { 1.0 } else { -1.0 };

        let current = 0.5 * trusted + 0.5 * noisy;
        let desired = 0.0;
        let error = desired - current;

        records.push(DiscoverRecord::new(
            obs_index,
            "input-0".to_string(),
            Some(trusted),
            trusted,
            vec![0.0],
        ));
        records.push(DiscoverRecord::new(
            obs_index,
            "input-1".to_string(),
            Some(noisy),
            noisy,
            vec![0.0],
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
        // Coordinated structural candidates are part of the synapse candidate budget.
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

    let groups = output["coordinatedStructuralCandidates"]
        .as_array()
        .expect("should include coordinatedStructuralCandidates array");
    assert!(
        !groups.is_empty(),
        "expected at least one coordinated structural candidate"
    );

    // We expect a group with:
    // - removeSynapse(input-1 -> output-0) (noisy)
    // - removeSynapse(input-0 -> output-0) (trusted)
    // - addSynapse(input-0 -> output-0, weight ~= 1.0)
    let found = groups.iter().any(|g| {
        let Some(ops) = g["operations"].as_array() else {
            return false;
        };

        let remove_noisy = ops.iter().any(|op| {
            op["type"] == "removeSynapse"
                && op["fromNeuronUuid"] == "input-1"
                && op["toNeuronUuid"] == "output-0"
        });
        let remove_trusted = ops.iter().any(|op| {
            op["type"] == "removeSynapse"
                && op["fromNeuronUuid"] == "input-0"
                && op["toNeuronUuid"] == "output-0"
        });
        let add_trusted_higher = ops.iter().any(|op| {
            op["type"] == "addSynapse"
                && op["fromNeuronUuid"] == "input-0"
                && op["toNeuronUuid"] == "output-0"
                && (op["weight"].as_f64().unwrap_or(0.0) - 1.0).abs() < 1e-6
        });

        remove_noisy && remove_trusted && add_trusted_higher
    });

    assert!(
        found,
        "expected a coordinated candidate that prunes the noisy input and reweights the trusted input"
    );
}
