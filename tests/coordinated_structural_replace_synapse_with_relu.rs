use neat_ai_discovery::analysis::GpuAnalyzer;
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson, analyze_parallel_internal};

/// Regression/integration test for Issue #173 (7-Jan-2026).
///
/// This verifies we can discover that an existing direct synapse should be replaced
/// by inserting a hidden ReLU neuron (as a single coordinated-structural group):
/// - removeSynapse(source -> target)
/// - addNeuron(hidden)
/// - addSynapse(source -> hidden)
/// - addSynapse(hidden -> target)
#[test]
fn coordinated_structural_can_replace_synapse_with_hidden_relu_neuron() {
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

    // Crippled creature: direct synapse only.
    //
    // The direct weight is intentionally small so add-neurons discovery prefers to
    // add a ReLU path even before we convert it into a coordinated replacement.
    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![NeuronJson {
            uuid: "output-0".to_string(),
            neuron_type: "output".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        }],
        synapses: vec![SynapseJson {
            from_uuid: "input-0".to_string(),
            to_uuid: "output-0".to_string(),
            weight: 0.01,
            synapse_type: None,
        }],
    };

    // Deterministic dataset and expected behaviour:
    // - expected = ReLU(x) (the uncrippled creature behaviour)
    // - current  = 0.01 * x (crippled direct synapse)
    //
    // The best coordinated fix is to remove the weak synapse and insert a hidden ReLU.
    let mut records: Vec<DiscoverRecord> = Vec::new();
    for obs_index in 0..64u32 {
        let x = (obs_index % 5) as f32 - 1.0; // [-1, 0, 1, 2, 3] repeating
        let expected = x.max(0.0);
        let current = 0.01 * x;
        let error = expected - current;

        records.push(DiscoverRecord::new(
            obs_index,
            "input-0".to_string(),
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
        // Coordinated structural candidates are part of the synapse candidate budget.
        // Issue #507: Micro-nudge variant adds a 4th variant per extreme candidate,
        // so we need a larger neuron budget to ensure the ReLU candidate is not truncated.
        "maxSynapseCandidates": 32,
        "maxNeuronCandidates": 48,
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

    // Find a group that contains an addNeuron op with ReLU squash.
    let found = groups.iter().any(|g| {
        let Some(ops) = g["operations"].as_array() else {
            return false;
        };
        ops.iter()
            .any(|op| op["type"] == "addNeuron" && op["squash"] == "ReLU")
            && ops.iter().any(|op| op["type"] == "removeSynapse")
            && ops.iter().filter(|op| op["type"] == "addSynapse").count() >= 2
    });
    assert!(
        found,
        "expected a coordinated candidate that removes the synapse and inserts a hidden ReLU"
    );
}
