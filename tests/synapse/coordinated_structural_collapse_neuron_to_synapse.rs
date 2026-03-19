#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use neat_ai_discovery::analysis::GpuAnalyzer;
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson, analyze_parallel_internal};

/// Regression/integration test for coordinated structural collapse (7-Jan-2026).
///
/// Verifies we can discover that a simple 1-in/1-out hidden neuron chain should be
/// replaced by a single synapse, emitted as an atomic coordinated-structural group:
/// - removeSynapse(a -> h)
/// - removeSynapse(h -> b)
/// - removeNeuron(h)
/// - addSynapse(a -> b)
#[test]
fn coordinated_structural_can_collapse_hidden_neuron_to_single_synapse() {
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

    // Creature with a simple chain: input-0 -> hidden-0 -> output-0.
    //
    // The chain under-weights the input by 0.5, so a direct synapse with weight=1.0 is better.
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

    // Deterministic dataset:
    // - hidden activation = x
    // - current output    = 0.5 * x
    // - desired output    = x
    // - error             = desired - current = 0.5 * x
    let mut records: Vec<DiscoverRecord> = Vec::new();
    for obs_index in 0..64u32 {
        let x = (obs_index % 7) as f32 - 3.0; // [-3..3] repeating
        let desired = x;
        let current = 0.5 * x;
        let error = desired - current;

        records.push(DiscoverRecord::new(
            obs_index,
            "input-0".to_string(),
            Some(x),
            x,
            vec![0.0],
        ));
        records.push(DiscoverRecord::new(
            obs_index,
            "hidden-0".to_string(),
            Some(x),
            x,
            vec![0.0],
        ));
        records.push(DiscoverRecord::new(
            obs_index,
            "output-0".to_string(),
            Some(current),
            current,
            vec![error],
        ));
    }
    write_records_to_parquet(&parquet_file, &records).expect("Failed to write discovery records");

    let input_json = serde_json::json!({
        "parquetFile": parquet_file,
        "creature": creature,
        "focusNeurons": ["output-0"],
        "maxSynapseCandidates": 32,
        "maxNeuronCandidates": 0,
        "randomSeed": 1
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

    // Find a group that removes hidden-0 and adds a direct input-0 -> output-0 synapse.
    let found = groups.iter().any(|g| {
        let Some(ops) = g["operations"].as_array() else {
            return false;
        };
        let has_remove_neuron = ops
            .iter()
            .any(|op| op["type"] == "removeNeuron" && op["neuronUuid"] == "hidden-0");
        let has_add_bypass = ops.iter().any(|op| {
            op["type"] == "addSynapse"
                && op["fromNeuronUuid"] == "input-0"
                && op["toNeuronUuid"] == "output-0"
        });
        has_remove_neuron && has_add_bypass
    });
    assert!(
        found,
        "expected a coordinated candidate that removes hidden-0 and adds a bypass synapse input-0 -> output-0"
    );
}
